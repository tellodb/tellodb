use crate::api::auth::{
    principal_namespace_prefix, principal_user_id, record_usage_for_principal, scope_entity_id,
    RequestPrincipal,
};
use crate::api::plan::*;
use crate::api::types::RankedItem;
use crate::api::types::{
    AnalyticsQueryPayload, AnalyticsQueryResult, BucketedResult, EvidenceCard, GraphExportPayload,
    GraphQueryPayload, GraphWalkPayload, ProofCheck, ProofPacket, ProofTurn, QueryPayload,
    QueryResult,
};
use crate::api::utils::{
    apply_decay_with_policy, clip_profile_to_budget, cosine_similarity_from_distance,
    elapsed_ms_and_us, extract_named_phrases, insert_f32_header, insert_stage_timing_headers,
    insert_u64_header, parse_temporal_window, scoped_semantic_min_hits, scoped_semantic_start,
    scoped_semantic_step, scoped_semantic_top, should_apply_neural_rerank,
    temporal_recency_scoring_enabled, SEMANTIC_TOP_DEFAULT,
};
use crate::api::{EngineState, PlatformWriteOp};
use crate::features::{self, Feature};
use crate::metrics;
use crate::ml::cosine_similarity;
use crate::retrieval::lanes::{self, Lane};
use crate::retrieval::{rrf_fuse, ScoringWeights};
use crate::storage::{
    AgentObservation, MemoryCard, MemoryCardSearchInput, MemoryKind, TenantStore,
};
use axum::{
    extract::{Extension, Json, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
};
use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Default)]
pub struct QueryDiagnostics {
    route_ms: u64,
    route_us: u64,
    embed_ms: u64,
    embed_us: u64,
    ann_ms: u64,
    ann_us: u64,
    scoped_ann_top: u64,
    scoped_ann_attempts: u64,
    scoped_primary_hits: u64,
    rerank_ms: u64,
    rerank_us: u64,
    fts_ms: u64,
    fts_us: u64,
    fuse_ms: u64,
    fuse_us: u64,
    hydrate_ms: u64,
    hydrate_us: u64,
    preference_ms: u64,
    preference_us: u64,
    graph_ms: u64,
    graph_us: u64,
    /// Graph sub-stages (µs), and memories whose edge neighbours were read.
    graph_links_us: u64,
    graph_edges_us: u64,
    graph_seeds_wall_us: u64,
    graph_entities_us: u64,
    graph_lookup_us: u64,
    graph_expanded: u64,
    session_ms: u64,
    session_us: u64,
    /// Sub-stages of scoring and response building (µs), so a slow query can
    /// be attributed without guessing.
    score_loop_us: u64,
    factver_us: u64,
    build_cards_us: u64,
    proof_us: u64,
    confidence_us: u64,
    card_ms: u64,
    card_us: u64,
    planning_ms: u64,
    planning_us: u64,
    route_session_ms: u64,
    route_window_ms: u64,
    route_pivot_ms: u64,
    route_ann_ms: u64,
    route_profile_ms: u64,
    hydrate_obs_ms: u64,
    hydrate_obs_us: u64,
    fetch_obs_ms: u64,
    fetch_obs_us: u64,
    fetch_cards_ms: u64,
    fetch_cards_us: u64,
    fetch_vectors_ms: u64,
    fetch_vectors_us: u64,
    fetch_neg_ms: u64,
    fetch_neg_us: u64,
    fetch_invalid_ms: u64,
    fetch_invalid_us: u64,
    scoring_loop_ms: u64,
    scoring_loop_us: u64,
    trace_ms: u64,
    trace_us: u64,
    total_ms: u64,
    total_us: u64,
    rerank_applied: bool,
    rerank_reason: RerankDecision,
    routed_sessions: u64,
    memory_card_hits: u64,
    evidence_confidence_bp: u64,
    abstain_recommended: bool,
}

/// Core profile with only facts that are valid at `point_in_time_ms` (or now):
/// superseded fact versions and facts recorded after the as-of time are
/// dropped, newest first. The stored profile is an append-only log of recent
/// facts, so without this the context block contradicted the fact history.
fn current_core_profile(
    tenant: &TenantStore,
    entity_id: &str,
    point_in_time_ms: Option<u64>,
) -> anyhow::Result<Option<String>> {
    let Some(raw) = tenant.get_core_profile(entity_id)? else {
        return Ok(None);
    };
    let Ok(mut profile) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return Ok(Some(raw));
    };
    if let Some(facts) = profile.get_mut("facts").and_then(|f| f.as_array_mut()) {
        let ids: Vec<String> = facts
            .iter()
            .filter_map(|f| f.get("memory_id").and_then(|m| m.as_str()).map(str::to_string))
            .collect();
        let invalid = match point_in_time_ms {
            Some(pit) => tenant.invalidated_set_at_time(pit, &ids)?,
            None => tenant.invalidated_set(&ids)?,
        };
        let timestamp = |f: &serde_json::Value| f.get("timestamp_ms").and_then(|t| t.as_u64());
        facts.retain(|f| {
            let stale =
                f.get("memory_id").and_then(|m| m.as_str()).is_some_and(|m| invalid.contains(m));
            let future =
                matches!((point_in_time_ms, timestamp(f)), (Some(pit), Some(ts)) if ts > pit);
            !stale && !future
        });
        facts.sort_by_key(|f| std::cmp::Reverse(timestamp(f).unwrap_or(0)));
    }
    Ok(Some(serde_json::to_string(&profile)?))
}

fn build_entity_observation_block(
    tenant: &TenantStore,
    entity_id: &str,
    query_text: &str,
    results: &[QueryResult],
    point_in_time_ms: Option<u64>,
) -> Result<String, StatusCode> {
    let internal = |err: anyhow::Error| {
        tracing::error!(error = ?err, "observation block read failed");
        StatusCode::INTERNAL_SERVER_ERROR
    };
    let profile = current_core_profile(tenant, entity_id, point_in_time_ms)
        .map_err(internal)?
        .map(|p| clip_profile_to_budget(&p, 8));

    let mut scenes = Vec::new();
    for entity in extract_named_phrases(&[query_text.to_string()]) {
        let lines =
            tenant.graph_edge_summaries_for_label(entity_id, &entity, 5).map_err(internal)?;
        if !lines.is_empty() {
            scenes.push(lines.join("; "));
        }
    }
    let top_chunk_texts: Vec<String> =
        results.iter().take(5).map(|r| r.textual_content.clone()).collect();
    Ok(build_observation_block(profile.as_deref(), &scenes, &top_chunk_texts))
}

pub async fn query_handler(
    State(state): State<EngineState>,
    Extension(principal): Extension<RequestPrincipal>,
    Json(payload): Json<QueryPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let ns_prefix = principal_namespace_prefix(&principal);
    let tenant_id = principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|e| {
        tracing::warn!("Failed to get tenant store: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    let profile_query_text = payload.textual_query.clone();

    let mut payload = payload;
    if let Some(ref p) = ns_prefix {
        payload.entity_id = Some(match payload.entity_id {
            Some(eid) => scope_entity_id(&eid, Some(p.as_str())),
            None => p.trim_end_matches(':').to_string(),
        });
    }
    let limit = payload.limit.max(1);
    let enable_neural_rerank = payload.enable_neural_rerank.unwrap_or(false);

    let entity_id_for_core_profile = payload.entity_id.clone();
    let point_in_time_ms = payload.point_in_time_ms;
    let block_query_text = profile_query_text.clone();

    // Pipeline and observation block both read SQLite, so both run on the
    // blocking pool rather than on an async worker thread.
    let (mut results, diagnostics, obs_block) = {
        let state_for_query = state.clone();
        let tenant = tenant.clone();
        tokio::task::spawn_blocking(move || {
            let (results, diagnostics) = execute_query_pipeline(
                payload,
                state_for_query,
                tenant.clone(),
                limit,
                enable_neural_rerank,
            )?;
            let obs_block = match entity_id_for_core_profile.as_deref() {
                Some(eid) => Some((
                    eid.to_string(),
                    build_entity_observation_block(
                        &tenant,
                        eid,
                        &block_query_text,
                        &results,
                        point_in_time_ms,
                    )?,
                )),
                None => None,
            };
            Ok::<_, StatusCode>((results, diagnostics, obs_block))
        })
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??
    };

    if let Some((eid, text)) = obs_block {
        results.insert(
            0,
            QueryResult {
                memory_id: "observation_block".to_string(),
                entity_id: eid,
                session_id: "system".to_string(),
                turn_index: 0,
                similarity: 1.0,
                created_at_ms: 0,
                textual_content: text,
                evidence: None,
                inference_notes: None,
                fact_key: None,
                conflict_flag: None,
                superseded_by: None,
                why_stale: None,
                stability_score: None,
            },
        );
    }

    let mut h = HeaderMap::new();
    insert_stage_timing_headers(&mut h, "x-tm-route", diagnostics.route_ms, diagnostics.route_us);
    insert_stage_timing_headers(
        &mut h,
        "x-tm-planning",
        diagnostics.planning_ms,
        diagnostics.planning_us,
    );
    insert_stage_timing_headers(&mut h, "x-tm-embed", diagnostics.embed_ms, diagnostics.embed_us);
    insert_stage_timing_headers(&mut h, "x-tm-ann", diagnostics.ann_ms, diagnostics.ann_us);
    insert_stage_timing_headers(&mut h, "x-tm-route-session", diagnostics.route_session_ms, 0);
    insert_stage_timing_headers(&mut h, "x-tm-route-window", diagnostics.route_window_ms, 0);
    insert_stage_timing_headers(&mut h, "x-tm-route-pivot", diagnostics.route_pivot_ms, 0);
    insert_stage_timing_headers(&mut h, "x-tm-route-profile", diagnostics.route_profile_ms, 0);
    insert_stage_timing_headers(&mut h, "x-tm-route-ann", diagnostics.route_ann_ms, 0);
    insert_stage_timing_headers(
        &mut h,
        "x-tm-rerank",
        diagnostics.rerank_ms,
        diagnostics.rerank_us,
    );
    insert_stage_timing_headers(&mut h, "x-tm-fts", diagnostics.fts_ms, diagnostics.fts_us);
    insert_stage_timing_headers(&mut h, "x-tm-card", diagnostics.card_ms, diagnostics.card_us);
    insert_stage_timing_headers(&mut h, "x-tm-fuse", diagnostics.fuse_ms, diagnostics.fuse_us);
    insert_stage_timing_headers(
        &mut h,
        "x-tm-hydrate",
        diagnostics.hydrate_ms,
        diagnostics.hydrate_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-hydrate-obs",
        diagnostics.hydrate_obs_ms,
        diagnostics.hydrate_obs_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-fetch-obs",
        diagnostics.fetch_obs_ms,
        diagnostics.fetch_obs_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-fetch-cards",
        diagnostics.fetch_cards_ms,
        diagnostics.fetch_cards_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-fetch-vectors",
        diagnostics.fetch_vectors_ms,
        diagnostics.fetch_vectors_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-fetch-neg",
        diagnostics.fetch_neg_ms,
        diagnostics.fetch_neg_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-fetch-invalid",
        diagnostics.fetch_invalid_ms,
        diagnostics.fetch_invalid_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-scoring-loop",
        diagnostics.scoring_loop_ms,
        diagnostics.scoring_loop_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-preference",
        diagnostics.preference_ms,
        diagnostics.preference_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-graph-bridge",
        diagnostics.graph_ms,
        diagnostics.graph_us,
    );
    insert_u64_header(&mut h, "x-tm-score-loop-us", diagnostics.score_loop_us);
    insert_u64_header(&mut h, "x-tm-factver-us", diagnostics.factver_us);
    insert_u64_header(&mut h, "x-tm-build-cards-us", diagnostics.build_cards_us);
    insert_u64_header(&mut h, "x-tm-proof-us", diagnostics.proof_us);
    insert_u64_header(&mut h, "x-tm-confidence-us", diagnostics.confidence_us);
    insert_u64_header(&mut h, "x-tm-graph-links-us", diagnostics.graph_links_us);
    insert_u64_header(&mut h, "x-tm-graph-edges-us", diagnostics.graph_edges_us);
    insert_u64_header(&mut h, "x-tm-graph-seeds-wall-us", diagnostics.graph_seeds_wall_us);
    insert_u64_header(&mut h, "x-tm-graph-entities-us", diagnostics.graph_entities_us);
    insert_u64_header(&mut h, "x-tm-graph-lookup-us", diagnostics.graph_lookup_us);
    insert_u64_header(&mut h, "x-tm-graph-expanded", diagnostics.graph_expanded);
    insert_stage_timing_headers(
        &mut h,
        "x-tm-session",
        diagnostics.session_ms,
        diagnostics.session_us,
    );
    insert_stage_timing_headers(&mut h, "x-tm-total", diagnostics.total_ms, diagnostics.total_us);
    insert_stage_timing_headers(&mut h, "x-tm-trace", diagnostics.trace_ms, diagnostics.trace_us);
    insert_u64_header(&mut h, "x-tm-scoped-ann-top", diagnostics.scoped_ann_top);
    insert_u64_header(&mut h, "x-tm-scoped-ann-attempts", diagnostics.scoped_ann_attempts);
    insert_u64_header(&mut h, "x-tm-scoped-primary-hits", diagnostics.scoped_primary_hits);
    insert_u64_header(&mut h, "x-tm-routed-sessions", diagnostics.routed_sessions);
    insert_u64_header(&mut h, "x-tm-memory-card-hits", diagnostics.memory_card_hits);
    insert_f32_header(
        &mut h,
        "x-tm-evidence-confidence",
        diagnostics.evidence_confidence_bp as f32 / 10_000.0,
    );
    h.insert(
        "x-tm-abstain-recommended",
        HeaderValue::from_static(if diagnostics.abstain_recommended { "1" } else { "0" }),
    );
    h.insert(
        "x-tm-rerank-applied",
        HeaderValue::from_static(if diagnostics.rerank_applied { "1" } else { "0" }),
    );
    insert_u64_header(&mut h, "x-tm-rerank-reason", diagnostics.rerank_reason as u64);

    if let Some(uid) = principal_user_id(&principal) {
        if let Err(e) = state.platform_write_tx.try_send(PlatformWriteOp::Profile {
            user_id: uid.to_string(),
            text: profile_query_text,
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            source: "query".to_string(),
        }) {
            tracing::warn!("platform writer channel full (profile query): {:?}", e);
        }
    }
    record_usage_for_principal(&state, &principal, "query");
    metrics::increment_query();
    if diagnostics.total_ms > 0 {
        metrics::observe_query_duration(diagnostics.total_ms as f64 / 1000.0);
    }
    Ok((StatusCode::OK, h, Json(results)))
}

fn is_synthetic_query_memory(memory_id: &str) -> bool {
    memory_id.split("::").nth(3) == Some("sq")
}

fn deterministic_subqueries(query: &str) -> Vec<String> {
    let normalized = query
        .replace(" and ", " | ")
        .replace(" or ", " | ")
        .replace(" but ", " | ")
        .replace(" while ", " | ");
    let mut out = Vec::new();
    for part in normalized.split(['|', '?', ';']) {
        let part = part.trim();
        if part.len() >= 8 && part.len() + 4 < query.len() {
            out.push(part.to_string());
        }
    }
    out.truncate(4);
    out
}

fn promote_query_variant(queries: &mut Vec<String>, candidate: String) {
    let candidate = candidate.trim().to_string();
    if candidate.is_empty() {
        return;
    }
    let lower = candidate.to_ascii_lowercase();
    if let Some(pos) = queries.iter().position(|query| query.to_ascii_lowercase() == lower) {
        if pos > 1 {
            let existing = queries.remove(pos);
            queries.insert(1, existing);
        }
    } else if queries.is_empty() {
        queries.push(candidate);
    } else {
        queries.insert(1, candidate);
    }
}

fn lifecycle_rank_adjustment(
    lifecycle: &crate::lifecycle::LifecycleMetadata,
    kind: MemoryKind,
    now_ms: u64,
) -> Option<f32> {
    if matches!(
        lifecycle.lifecycle_state,
        crate::lifecycle::LifecycleState::Expired
            | crate::lifecycle::LifecycleState::Invalidated
            | crate::lifecycle::LifecycleState::Tombstoned
    ) {
        return None;
    }
    if lifecycle.expires_at_ms.map(|expires_at| expires_at <= now_ms).unwrap_or(false) {
        return None;
    }

    let mut adjustment = (lifecycle.admission_score - 0.50) * 0.06
        + (lifecycle.utility_score - 0.50) * 0.035
        + (lifecycle.confidence_score - 0.50) * 0.035
        + (lifecycle.specificity_score - 0.45) * 0.025;

    if lifecycle.promote_to_profile
        || matches!(kind, MemoryKind::Fact | MemoryKind::Preference | MemoryKind::Decision)
    {
        adjustment += 0.025;
    }
    adjustment += match lifecycle.retention_class {
        crate::lifecycle::RetentionClass::LongTerm => 0.025,
        crate::lifecycle::RetentionClass::Archive => 0.012,
        crate::lifecycle::RetentionClass::Episodic => 0.006,
        crate::lifecycle::RetentionClass::Working => -0.006,
        crate::lifecycle::RetentionClass::Ephemeral => -0.035,
        // Sensitivity is keyword-based ("token", "bank", "health") and says
        // nothing about relevance, so it no longer affects ranking.
        crate::lifecycle::RetentionClass::ComplianceSensitive => 0.0,
    };
    if lifecycle.is_inference {
        adjustment -= 0.025;
    }
    Some(adjustment.clamp(-0.08, 0.10))
}

fn attractor_negative_penalty(
    scorable: &ScorableObservation<'_>,
    plan: &QueryPlan,
    query_text: &str,
    entity_hits: usize,
    lexical_hits: usize,
    temporal_hits: usize,
    facet_mask: u64,
) -> f32 {
    let query_lower = query_text.to_ascii_lowercase();
    let attractors = [
        "dog", "dogs", "pet", "pets", "animal", "nature", "trail", "hike", "book", "game", "music",
        "festival", "car", "cars", "health", "yoga", "travel", "trip", "family", "friend",
        "friends", "work", "job", "project",
    ];
    let shared_attractors = attractors
        .iter()
        .filter(|term| query_lower.contains(**term) && scorable.lower.contains(**term))
        .count();
    if shared_attractors == 0 {
        return 0.0;
    }

    let required_entities = plan.subject_entities.len();
    let entity_deficit = required_entities.saturating_sub(entity_hits).min(3) as f32;
    let facet_deficit = if plan.coverage_mode && facet_mask.count_ones() == 0 { 1.0 } else { 0.0 };
    let temporal_deficit =
        if !plan.temporal_terms.is_empty() && temporal_hits == 0 { 1.0 } else { 0.0 };
    let weak_specificity = entity_deficit > 0.0
        || facet_deficit > 0.0
        || temporal_deficit > 0.0
        || (plan.needs_decomposition && lexical_hits < 2);
    if !weak_specificity {
        return 0.0;
    }

    let mut penalty = (shared_attractors as f32).min(3.0) * 0.018;
    penalty += entity_deficit * if plan.cross_entity { 0.055 } else { 0.030 };
    penalty += facet_deficit * 0.045;
    penalty += temporal_deficit * 0.035;
    if plan.needs_decomposition && lexical_hits == 0 {
        penalty += 0.035;
    }
    penalty.min(0.18)
}

fn build_proof_packet(
    tenant: &TenantStore,
    query_text: &str,
    plan: &QueryPlan,
    card: &EvidenceCard,
    proof_mode: &str,
    verify_evidence: bool,
    evidence_radius: u32,
) -> ProofPacket {
    let mut source_turns = Vec::new();
    if evidence_radius > 0 && !card.source_session_id.is_empty() {
        let center = card.source_turn_index as u32;
        if let Ok(turns) = tenant.get_turn_window(
            &card.entity_id,
            &card.source_session_id,
            center,
            evidence_radius,
        ) {
            source_turns = turns
                .into_iter()
                .map(|turn| ProofTurn {
                    turn_id: turn.turn_id,
                    session_id: turn.session_id,
                    turn_index: turn.turn_index,
                    speaker: turn.speaker,
                    text: turn.raw_text,
                })
                .collect();
        }
    }
    if source_turns.is_empty() {
        let turn_ids = vec![card.source_memory_id.clone()];
        if let Ok(turns) = tenant.get_ledger_turns_batch(&turn_ids) {
            source_turns = turns
                .into_values()
                .map(|turn| ProofTurn {
                    turn_id: turn.turn_id,
                    session_id: turn.session_id,
                    turn_index: turn.turn_index,
                    speaker: turn.speaker,
                    text: turn.raw_text,
                })
                .collect();
            source_turns.sort_by_key(|turn| turn.turn_index);
        }
    }
    if source_turns.is_empty() {
        source_turns.push(ProofTurn {
            turn_id: card.source_memory_id.clone(),
            session_id: card.source_session_id.clone(),
            turn_index: card.source_turn_index as u32,
            speaker: None,
            text: card.claim_text.clone(),
        });
    }

    let missing_facets = plan
        .coverage_facets
        .iter()
        .enumerate()
        .filter_map(|(idx, facet)| {
            if idx < 64 && (card.facet_mask & (1u64 << idx)) == 0 {
                Some(facet.text.clone())
            } else {
                None
            }
        })
        .take(8)
        .collect::<Vec<_>>();

    let entity_required = !plan.subject_entities.is_empty();
    let lexical_required = !plan.lexical_terms.is_empty();
    let temporal_required = !plan.temporal_terms.is_empty();
    let entity_ok = !entity_required || card.entity_hits > 0;
    let lexical_ok = !lexical_required || card.lexical_hits > 0;
    let temporal_ok = !temporal_required || card.temporal_hits > 0;
    let source_ok = !card.source_memory_id.is_empty() && !card.source_session_id.is_empty();
    let facet_ok = missing_facets.is_empty() || !plan.coverage_mode;
    let query_terms = crate::fts::tokenize_for_similarity(query_text);
    let proof_text = source_turns
        .iter()
        .map(|turn| turn.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    let lexical_overlap = query_terms
        .iter()
        .filter(|term| term.len() > 3 && proof_text.contains(term.as_str()))
        .count();
    let lexical_trace_ok = query_terms.is_empty() || lexical_overlap > 0;

    let mut checks = vec![
        ProofCheck {
            name: "source_backed".to_string(),
            passed: source_ok,
            detail: card.source_memory_id.clone(),
        },
        ProofCheck {
            name: "entity_support".to_string(),
            passed: entity_ok,
            detail: format!("{} entity hit(s)", card.entity_hits),
        },
        ProofCheck {
            name: "lexical_support".to_string(),
            passed: lexical_ok && lexical_trace_ok,
            detail: format!(
                "{} lexical hit(s), {} proof overlap(s)",
                card.lexical_hits, lexical_overlap
            ),
        },
        ProofCheck {
            name: "temporal_support".to_string(),
            passed: temporal_ok,
            detail: format!("{} temporal hit(s)", card.temporal_hits),
        },
        ProofCheck {
            name: "facet_coverage".to_string(),
            passed: facet_ok,
            detail: format!("{} missing facet(s)", missing_facets.len()),
        },
    ];

    let verified = if verify_evidence {
        checks.iter().all(|check| check.passed)
    } else {
        checks.iter().filter(|check| check.name != "facet_coverage").all(|check| check.passed)
    };
    if !verify_evidence {
        checks.push(ProofCheck {
            name: "verification_mode".to_string(),
            passed: true,
            detail: "lightweight proof pack only".to_string(),
        });
    }

    let support_score = (card.entity_hits.min(3) as f32 * 0.10)
        + (card.lexical_hits.min(5) as f32 * 0.055)
        + (card.temporal_hits.min(2) as f32 * 0.075)
        + (card.facet_mask.count_ones().min(5) as f32 * 0.045)
        + if source_ok { 0.20 } else { 0.0 }
        + if verified { 0.15 } else { 0.0 };
    let confidence = support_score.clamp(0.05, 0.99);

    ProofPacket {
        proof_mode: proof_mode.to_string(),
        verified,
        confidence,
        source_memory_id: card.source_memory_id.clone(),
        source_session_id: card.source_session_id.clone(),
        source_turn_index: card.source_turn_index,
        supporting_card_ids: card.card_id.clone().into_iter().collect(),
        supporting_event_ids: Vec::new(),
        entities_hit: card.entity_hits,
        lexical_hits: card.lexical_hits,
        temporal_hits: card.temporal_hits,
        missing_facets,
        checks,
        source_turns,
    }
}

fn query_allows_stale_cards(query: &str, plan: &QueryPlan) -> bool {
    let lower = query.to_ascii_lowercase();
    matches!(plan.intent, QueryIntent::TemporalAggregation)
        || (matches!(plan.intent, QueryIntent::Inference) && plan.needs_decomposition)
        || plan.coverage_mode
        || plan.cross_entity
        || lower.contains("previous")
        || lower.contains("before")
        || lower.contains("used to")
        || lower.contains("formerly")
        || lower.contains("history")
        || lower.contains("past")
        || lower.contains("old ")
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RetrievalBudget {
    semantic_top: usize,
    fts_top: usize,
    semantic_query_limit: usize,
    fts_query_limit: usize,
    session_router_limit: usize,
    route_probe_query_limit: usize,
    route_probe_hit_limit: usize,
    route_take_simple: usize,
    route_take_hard: usize,
    card_limit: usize,
}

#[repr(usize)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RetrievalProfile {
    Fast = 0,
    Balanced = 1,
    Research = 2,
}

fn retrieval_profile() -> RetrievalProfile {
    static CACHED: std::sync::OnceLock<RetrievalProfile> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("TEMPORAL_MEMORY_RETRIEVAL_PROFILE")
            .unwrap_or_else(|_| "fast".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "research" | "full" | "v2" => RetrievalProfile::Research,
            "balanced" | "default" => RetrievalProfile::Balanced,
            _ => RetrievalProfile::Fast,
        }
    })
}

fn auto_rerank_enabled(profile: RetrievalProfile) -> bool {
    static OVERRIDE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let configured = *OVERRIDE.get_or_init(|| {
        std::env::var("TEMPORAL_MEMORY_AUTO_RERANK").ok().and_then(|value| {
            match value.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Some(true),
                "0" | "false" | "no" | "off" => Some(false),
                _ => None,
            }
        })
    });
    configured.unwrap_or(matches!(profile, RetrievalProfile::Research))
}

#[repr(usize)]
#[derive(Clone, Copy)]
enum QueryShape {
    Hard = 0,
    Temporal = 1,
    Numeric = 2,
    Simple = 3,
}

fn query_shape(plan: &QueryPlan) -> QueryShape {
    let hard = plan.needs_decomposition
        || plan.cross_entity
        || plan.coverage_mode
        || matches!(
            plan.intent,
            QueryIntent::Inference | QueryIntent::Recommendation | QueryIntent::PeripheralMention
        );
    let temporal = matches!(plan.intent, QueryIntent::TemporalAggregation)
        || !plan.temporal_terms.is_empty()
        || plan.ordinal_rank.is_some();
    let numeric = matches!(plan.intent, QueryIntent::NumericAggregation);

    if hard {
        QueryShape::Hard
    } else if temporal {
        QueryShape::Temporal
    } else if numeric {
        QueryShape::Numeric
    } else {
        QueryShape::Simple
    }
}

const BUDGETS: [[RetrievalBudget; 4]; 3] = [
    [
        RetrievalBudget {
            semantic_top: 320,
            fts_top: 80,
            semantic_query_limit: 3,
            fts_query_limit: 3,
            session_router_limit: 12,
            route_probe_query_limit: 1,
            route_probe_hit_limit: 10,
            route_take_simple: 6,
            route_take_hard: 10,
            card_limit: 84,
        },
        RetrievalBudget {
            semantic_top: 280,
            fts_top: 72,
            semantic_query_limit: 1,
            fts_query_limit: 1,
            session_router_limit: 8,
            route_probe_query_limit: 1,
            route_probe_hit_limit: 8,
            route_take_simple: 6,
            route_take_hard: 8,
            card_limit: 64,
        },
        RetrievalBudget {
            semantic_top: 260,
            fts_top: 64,
            semantic_query_limit: 1,
            fts_query_limit: 1,
            session_router_limit: 8,
            route_probe_query_limit: 0,
            route_probe_hit_limit: 0,
            route_take_simple: 6,
            route_take_hard: 8,
            card_limit: 56,
        },
        RetrievalBudget {
            semantic_top: 240,
            fts_top: 64,
            semantic_query_limit: 1,
            fts_query_limit: 1,
            session_router_limit: 6,
            route_probe_query_limit: 0,
            route_probe_hit_limit: 0,
            route_take_simple: 5,
            route_take_hard: 7,
            card_limit: 48,
        },
    ],
    [
        RetrievalBudget {
            semantic_top: 420,
            fts_top: 96,
            semantic_query_limit: 3,
            fts_query_limit: 3,
            session_router_limit: 14,
            route_probe_query_limit: 1,
            route_probe_hit_limit: 12,
            route_take_simple: 8,
            route_take_hard: 12,
            card_limit: 120,
        },
        RetrievalBudget {
            semantic_top: 320,
            fts_top: 80,
            semantic_query_limit: 2,
            fts_query_limit: 2,
            session_router_limit: 10,
            route_probe_query_limit: 1,
            route_probe_hit_limit: 10,
            route_take_simple: 8,
            route_take_hard: 10,
            card_limit: 88,
        },
        RetrievalBudget {
            semantic_top: 300,
            fts_top: 72,
            semantic_query_limit: 2,
            fts_query_limit: 2,
            session_router_limit: 10,
            route_probe_query_limit: 1,
            route_probe_hit_limit: 10,
            route_take_simple: 7,
            route_take_hard: 9,
            card_limit: 72,
        },
        RetrievalBudget {
            semantic_top: 240,
            fts_top: 64,
            semantic_query_limit: 1,
            fts_query_limit: 1,
            session_router_limit: 8,
            route_probe_query_limit: 0,
            route_probe_hit_limit: 0,
            route_take_simple: 6,
            route_take_hard: 8,
            card_limit: 56,
        },
    ],
    [
        RetrievalBudget {
            semantic_top: 640,
            fts_top: 160,
            semantic_query_limit: 5,
            fts_query_limit: 5,
            session_router_limit: 24,
            route_probe_query_limit: 4,
            route_probe_hit_limit: 20,
            route_take_simple: 10,
            route_take_hard: 16,
            card_limit: 180,
        },
        RetrievalBudget {
            semantic_top: 420,
            fts_top: 100,
            semantic_query_limit: 3,
            fts_query_limit: 3,
            session_router_limit: 16,
            route_probe_query_limit: 2,
            route_probe_hit_limit: 16,
            route_take_simple: 10,
            route_take_hard: 14,
            card_limit: 120,
        },
        RetrievalBudget {
            semantic_top: 360,
            fts_top: 84,
            semantic_query_limit: 3,
            fts_query_limit: 3,
            session_router_limit: 14,
            route_probe_query_limit: 2,
            route_probe_hit_limit: 14,
            route_take_simple: 9,
            route_take_hard: 12,
            card_limit: 96,
        },
        RetrievalBudget {
            semantic_top: 240,
            fts_top: 64,
            semantic_query_limit: 2,
            fts_query_limit: 2,
            session_router_limit: 10,
            route_probe_query_limit: 1,
            route_probe_hit_limit: 12,
            route_take_simple: 8,
            route_take_hard: 10,
            card_limit: 72,
        },
    ],
];

fn retrieval_budget_for_plan(plan: &QueryPlan, profile: RetrievalProfile) -> RetrievalBudget {
    BUDGETS[profile as usize][query_shape(plan) as usize]
}

/// Edge-graph scores for each seed memory: a breadth-first walk of up to
/// `max_depth` hops, each hop's weight decayed by 0.6. Intent-aligned edges
/// count 1.5x (Inference/Recommendation: caused_by, leads_to, prefers;
/// PeripheralMention: updates, supports, contradicts).
///
/// All seeds advance one level at a time and a memory's neighbours are read
/// once per query, in batches, so the SQL cost is a few queries per level
/// rather than one per expanded memory per seed. Returns one map per seed
/// and the number of memories whose neighbours were read.
fn collect_edge_cluster_scores_for_seeds(
    tenant: &TenantStore,
    seeds: &[String],
    max_depth: usize,
    edge_type_filter: Option<&str>,
    intent: Option<crate::api::plan::types::QueryIntent>,
) -> (Vec<HashMap<String, f32>>, u64) {
    const NEIGHBORS_PER_NODE: usize = 50;
    const DEPTH_DECAY: f32 = 0.6;
    let max_node_degree = graph_max_node_degree();

    let mut neighbours: HashMap<String, Vec<(String, f32, String)>> = HashMap::new();
    let mut accumulated: Vec<HashMap<String, f32>> = vec![HashMap::new(); seeds.len()];
    let mut frontiers: Vec<Vec<String>> = seeds.iter().map(|seed| vec![seed.clone()]).collect();
    let mut expanded = 0u64;
    let mut path_weight = 1.0f32;
    for _depth in 0..max_depth {
        let mut missing: Vec<String> = frontiers
            .iter()
            .flatten()
            .filter(|node| !neighbours.contains_key(*node))
            .cloned()
            .collect();
        missing.sort();
        missing.dedup();
        if !missing.is_empty() {
            expanded += missing.len() as u64;
            match tenant.get_edge_cluster_neighbors_batch(
                &missing,
                edge_type_filter,
                NEIGHBORS_PER_NODE,
                max_node_degree,
            ) {
                Ok(found) => neighbours.extend(found),
                Err(err) => {
                    tracing::warn!(error = ?err, "edge cluster expansion failed");
                    break;
                }
            }
        }

        let mut any_next = false;
        for (frontier, scores) in frontiers.iter_mut().zip(accumulated.iter_mut()) {
            let mut next = Vec::new();
            let mut seen_next = HashSet::new();
            for node in frontier.iter() {
                for (linked_mid, weight, edge_type) in neighbours.get(node).into_iter().flatten() {
                    let intent_mult = intent_weight_for_edge(edge_type.as_str(), intent);
                    *scores.entry(linked_mid.clone()).or_insert(0.0) +=
                        path_weight * weight * intent_mult;
                    if seen_next.insert(linked_mid.as_str()) {
                        next.push(linked_mid.clone());
                    }
                }
            }
            any_next |= !next.is_empty();
            *frontier = next;
        }
        if !any_next {
            break;
        }
        path_weight *= DEPTH_DECAY;
    }

    (accumulated, expanded)
}

/// Top fused candidates the graph stage expands from (`TELLODB_GRAPH_SEEDS`,
/// default 24).
fn graph_seed_count() -> usize {
    static SEEDS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SEEDS.get_or_init(|| {
        std::env::var("TELLODB_GRAPH_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(24)
    })
}

/// Hops walked from each seed (`TELLODB_GRAPH_MAX_DEPTH`, default 2).
fn graph_max_depth() -> usize {
    static DEPTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("TELLODB_GRAPH_MAX_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
    })
}

/// Graph nodes with more edges than this are treated as hubs and not
/// traversed (`TELLODB_GRAPH_MAX_NODE_DEGREE`, default 128).
fn graph_max_node_degree() -> usize {
    static DEGREE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEGREE.get_or_init(|| {
        std::env::var("TELLODB_GRAPH_MAX_NODE_DEGREE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v: &usize| v > 0)
            .unwrap_or(128)
    })
}

/// Returns 1.5 for edges that align with the query intent, 1.0 otherwise.
fn intent_weight_for_edge(
    edge_type: &str,
    intent: Option<crate::api::plan::types::QueryIntent>,
) -> f32 {
    use crate::api::plan::types::QueryIntent;
    let et = edge_type.to_ascii_lowercase();
    match intent {
        Some(QueryIntent::Inference) | Some(QueryIntent::Recommendation) => {
            if et == "caused_by" || et == "leads_to" || et == "prefers" {
                1.5
            } else {
                1.0
            }
        }
        Some(QueryIntent::PeripheralMention) => {
            if et == "updates" || et == "supports" || et == "contradicts" {
                1.5
            } else {
                1.0
            }
        }
        _ => 1.0,
    }
}

fn parse_graph_direction_str(direction: Option<&str>) -> &'static str {
    match direction.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("in" | "inbound") => "Inbound",
        Some("both") => "Both",
        _ => "Outbound",
    }
}

fn scoped_graph_node_id(
    principal: &crate::api::auth::RequestPrincipal,
    requested: Option<String>,
) -> Result<String, StatusCode> {
    let ns_prefix = principal_namespace_prefix(principal);
    let node_id = match requested {
        Some(id) if !id.trim().is_empty() => scope_entity_id(id.trim(), ns_prefix.as_deref()),
        None => ns_prefix
            .as_deref()
            .map(|prefix| prefix.trim_end_matches(':').to_string())
            .ok_or(StatusCode::BAD_REQUEST)?,
        _ => return Err(StatusCode::BAD_REQUEST),
    };
    Ok(node_id)
}

pub async fn graph_query_handler(
    State(state): State<EngineState>,
    Extension(principal): Extension<RequestPrincipal>,
    Json(payload): Json<GraphQueryPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    // If subject is provided, scope it. If not, fallback to scoping the requested user_id.
    let subject = if !payload.subject.trim().is_empty() {
        let ns_prefix = principal_namespace_prefix(&principal);
        scope_entity_id(payload.subject.trim(), ns_prefix.as_deref())
    } else {
        scoped_graph_node_id(&principal, payload.user_id)?
    };

    let tenant_id = crate::api::auth::principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let tenant_clone = tenant.clone();
    let results = tokio::task::spawn_blocking(move || {
        tenant_clone.graph_query_edges(
            &subject,
            payload.edge_type.as_deref(),
            parse_graph_direction_str(payload.direction.as_deref()),
            payload.limit.unwrap_or(50).min(500),
        )
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    record_usage_for_principal(&state, &principal, "query");
    Ok((StatusCode::OK, Json(results)))
}

pub async fn graph_walk_handler(
    State(state): State<EngineState>,
    Extension(principal): Extension<RequestPrincipal>,
    Json(payload): Json<GraphWalkPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let node = if !payload.node.trim().is_empty() {
        let ns_prefix = principal_namespace_prefix(&principal);
        scope_entity_id(payload.node.trim(), ns_prefix.as_deref())
    } else {
        scoped_graph_node_id(&principal, payload.user_id)?
    };

    let tenant_id = crate::api::auth::principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let tenant_clone = tenant.clone();
    let results = tokio::task::spawn_blocking(move || {
        // TODO: actually implement depth/breadth walk. For now, pass first edge_type.
        let edge_type = payload.edge_types.as_ref().and_then(|et| et.first().map(|s| s.as_str()));
        tenant_clone.graph_query_edges(
            &node,
            edge_type,
            parse_graph_direction_str(payload.direction.as_deref()),
            payload.limit.unwrap_or(250).min(2_000),
        )
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    record_usage_for_principal(&state, &principal, "query");
    Ok((StatusCode::OK, Json(results)))
}

pub async fn graph_export_handler(
    State(state): State<EngineState>,
    Extension(principal): Extension<RequestPrincipal>,
    Json(payload): Json<GraphExportPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let seed = if !payload.seed.trim().is_empty() {
        let ns_prefix = principal_namespace_prefix(&principal);
        scope_entity_id(payload.seed.trim(), ns_prefix.as_deref())
    } else {
        scoped_graph_node_id(&principal, payload.user_id)?
    };

    let tenant_id = crate::api::auth::principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let tenant_clone = tenant.clone();
    let results = tokio::task::spawn_blocking(move || {
        // TODO: implement export walk using breadth/depth. For now fallback to query edges.
        let edge_type = payload.edge_types.as_ref().and_then(|et| et.first().map(|s| s.as_str()));
        tenant_clone.graph_query_edges(
            &seed,
            edge_type,
            parse_graph_direction_str(payload.direction.as_deref()),
            payload.max_nodes.unwrap_or(500).min(5_000),
        )
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    record_usage_for_principal(&state, &principal, "query");
    Ok((StatusCode::OK, Json(results)))
}

pub async fn analytics_query_handler(
    State(state): State<EngineState>,
    Extension(principal): Extension<RequestPrincipal>,
    Json(payload): Json<AnalyticsQueryPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let ns_prefix = principal_namespace_prefix(&principal);
    let user_id = crate::api::auth::principal_user_id(&principal).unwrap_or("default");
    let mut payload = payload;
    payload.entity_id = scope_entity_id(&payload.entity_id, ns_prefix.as_deref());
    let s = payload.start_timestamp_ms.unwrap_or(0);
    let e = payload.end_timestamp_ms.unwrap_or(u64::MAX);

    let agg = state
        .analytics
        .aggregate_range(user_id, &payload.entity_id, &payload.label, s, e)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let buckets = if let Some(bucket_str) = &payload.bucket {
        let bucket = match bucket_str.to_lowercase().as_str() {
            "hour" => crate::analytics::TemporalBucket::Hour,
            "day" => crate::analytics::TemporalBucket::Day,
            "week" => crate::analytics::TemporalBucket::Week,
            "month" => crate::analytics::TemporalBucket::Month,
            "year" => crate::analytics::TemporalBucket::Year,
            _ => return Err(StatusCode::BAD_REQUEST),
        };
        let bucketed = state
            .analytics
            .aggregate_bucketed(user_id, &payload.entity_id, &payload.label, s, e, bucket)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        Some(
            bucketed
                .into_iter()
                .map(|b| BucketedResult {
                    bucket_start_ms: b.bucket_start_ms,
                    bucket_end_ms: b.bucket_end_ms,
                    sum: b.result.sum,
                    count: b.result.count,
                    avg: b.result.avg,
                    min: b.result.min,
                    max: b.result.max,
                    stddev: b.result.stddev,
                })
                .collect(),
        )
    } else {
        None
    };

    record_usage_for_principal(&state, &principal, "query_analytics");
    Ok((
        StatusCode::OK,
        Json(AnalyticsQueryResult {
            entity_id: payload.entity_id,
            label: payload.label,
            sum: agg.sum,
            count: agg.count,
            avg: agg.avg,
            min: agg.min,
            max: agg.max,
            stddev: agg.stddev,
            buckets,
        }),
    ))
}

const NEURAL_TOP: usize = 25;
const NEURAL_BATCH: usize = 32;
// Minimum similarity for an HNSW hit to be retained post-ANN.
// Hits below this threshold (similarity = 1.0 - distance) are dropped to
// prevent zero-or-near-zero-similarity noise from contaminating RRF.
const MIN_HIT_SIMILARITY: f32 = 0.30;

#[derive(Default)]
struct QueryPipelineData {
    raw_query_text: String,
    query_text: String,
    include_evidence: bool,
    verify_evidence: bool,
    proof_mode: String,
    evidence_radius: u32,
    plan: QueryPlan,
    primary_qembed: Vec<f32>,
    budget: RetrievalBudget,
    semantic_top: usize,
    fts_top: usize,
    adaptive_profile: QueryAdaptiveProfile,
    diag: QueryDiagnostics,

    session_route_scores: HashMap<String, f32>,
    routed_memory_ids: HashMap<String, f32>,

    primary_hnsw_raw: Vec<(u64, f32)>,
    semantic_ranked_lists: Vec<(f32, Vec<RankedItem>)>,
    fts_ranked_lists: Vec<(f32, Vec<RankedItem>)>,
    card_ranked_items: Vec<RankedItem>,

    neural_scores: HashMap<String, f32>,

    fused: Vec<(String, u64, f32)>,
    graph_scores: HashMap<String, f32>,

    observations: HashMap<String, AgentObservation>,
    memory_cards: HashMap<String, MemoryCard>,
    invalidated_facts: HashSet<String>,
}

struct QueryPipelineState {
    payload: QueryPayload,
    state: EngineState,
    tenant: std::sync::Arc<TenantStore>,
    limit: usize,
    enable_neural_rerank: bool,
    weights: ScoringWeights,
    total_start: Instant,
    route_start: Instant,
    now_ms: u64,
    data: QueryPipelineData,
}

impl Deref for QueryPipelineState {
    type Target = QueryPipelineData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for QueryPipelineState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl QueryPipelineState {
    fn new(
        payload: QueryPayload,
        state: EngineState,
        tenant: std::sync::Arc<TenantStore>,
        limit: usize,
        enable_neural_rerank: bool,
    ) -> Self {
        let ambiguity_threshold = state
            .ranking_config
            .ambiguity_delta_threshold
            .unwrap_or_else(|| ScoringWeights::default().ambiguity_delta_threshold);
        let now_ms = payload.reference_time_ms.unwrap_or_else(|| {
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
        });
        Self {
            payload,
            state,
            tenant,
            limit,
            enable_neural_rerank,
            weights: ScoringWeights {
                ambiguity_delta_threshold: ambiguity_threshold,
                ..Default::default()
            },
            total_start: Instant::now(),
            route_start: Instant::now(),
            now_ms,
            data: QueryPipelineData::default(),
        }
    }
}

fn plan_phase(s: &mut QueryPipelineState) {
    s.raw_query_text = s.payload.textual_query.clone();
    s.query_text = rewrite_query_for_retrieval(&s.raw_query_text);
    s.include_evidence =
        s.payload.include_evidence.unwrap_or(false) || s.payload.verify_evidence.unwrap_or(false);
    s.verify_evidence = s.payload.verify_evidence.unwrap_or(false);
    s.proof_mode = s
        .payload
        .proof_mode
        .clone()
        .unwrap_or_else(|| if s.verify_evidence { "light" } else { "off" }.to_string())
        .to_ascii_lowercase();
    s.evidence_radius = s.payload.max_evidence_turns_per_session.unwrap_or(0).min(3) as u32;

    let planning_start = Instant::now();
    s.plan = build_query_plan(&s.query_text, s.state.intent_classifier.as_deref());

    if let Some(hyde_query) = build_hyde_query(&s.query_text, &s.plan) {
        promote_query_variant(&mut s.plan.semantic_queries, hyde_query.clone());
        promote_query_variant(&mut s.plan.fts_queries, hyde_query);
    }

    if s.plan.needs_decomposition {
        for sq in deterministic_subqueries(&s.query_text) {
            if !s.plan.semantic_queries.contains(&sq) {
                s.plan.semantic_queries.push(sq.clone());
            }
            if !s.plan.fts_queries.contains(&sq) {
                s.plan.fts_queries.push(sq);
            }
        }
    }

    let retrieval_profile = retrieval_profile();
    s.budget = retrieval_budget_for_plan(&s.plan, retrieval_profile);
    s.fts_top = match s.plan.intent {
        QueryIntent::Inference | QueryIntent::PeripheralMention => 180,
        QueryIntent::TemporalAggregation => 120,
        QueryIntent::NumericAggregation => 90,
        QueryIntent::Recommendation | QueryIntent::General => 72,
    }
    .min(s.budget.fts_top);
    s.semantic_top =
        if s.payload.entity_id.is_some() { scoped_semantic_top() } else { SEMANTIC_TOP_DEFAULT };
    s.semantic_top = if s.plan.cross_entity || s.plan.ordinal_rank.is_some() {
        s.semantic_top.saturating_mul(2).min(1200)
    } else if matches!(s.plan.intent, QueryIntent::Inference) {
        s.semantic_top.saturating_mul(2).min(1000)
    } else if matches!(s.plan.intent, QueryIntent::PeripheralMention) {
        s.semantic_top.saturating_mul(2).min(900)
    } else {
        s.semantic_top
    }
    .min(s.budget.semantic_top);
    s.diag = QueryDiagnostics::default();
    s.adaptive_profile = build_query_adaptive_profile(&s.query_text, &s.plan);

    // On failure keep the embedding empty: retrieval_ann skips ANN for a
    // wrong-dimension vector, and searching with a zero vector would return
    // arbitrary neighbours.
    s.primary_qembed =
        s.state.semantic.generate_query_embedding(&s.query_text).unwrap_or_else(|err| {
            tracing::warn!(query = %s.raw_query_text, error = ?err, "query embedding failed; skipping ANN");
            Vec::new()
        });

    s.routed_memory_ids = HashMap::new();
    s.session_route_scores = HashMap::new();
    if let Some(ref entity_id) = s.payload.entity_id {
        let entity_for_routes = entity_id.clone();
        let lexical_for_routes = s.plan.lexical_terms.clone();
        let temporal_for_routes = s.plan.temporal_terms.clone();
        let subject_for_routes = s.plan.subject_entities.clone();
        let query_text_for_routes = s.query_text.clone();
        let tenant_for_routes = s.tenant.clone();
        let session_router_limit = s.budget.session_router_limit;
        let router_on = features::enabled(Feature::SessionRouter);

        let (sr_hits, win_hits, pivot_hits) = std::thread::scope(|sc| {
            let h_sr = {
                let tenant = tenant_for_routes.clone();
                let eid = entity_for_routes.clone();
                let q = query_text_for_routes.clone();
                let l = lexical_for_routes.clone();
                let t = temporal_for_routes.clone();
                let s_e = subject_for_routes.clone();
                sc.spawn(move || {
                    if !router_on {
                        return Vec::new();
                    }
                    tenant
                        .search_session_router(&eid, &q, &l, &t, &s_e, session_router_limit)
                        .unwrap_or_default()
                })
            };
            let h_win = {
                let tenant = tenant_for_routes.clone();
                let eid = entity_for_routes.clone();
                let q = query_text_for_routes.clone();
                let ref_time = s.now_ms;
                sc.spawn(move || {
                    if !router_on {
                        Vec::new()
                    } else if let Some((st, en)) = parse_temporal_window(&q, Some(ref_time)) {
                        tenant.sessions_in_time_window(&eid, st, en).unwrap_or_default()
                    } else {
                        Vec::new()
                    }
                })
            };
            let h_pivot = {
                let tenant = tenant_for_routes.clone();
                let eid = entity_for_routes.clone();
                let s_e = subject_for_routes.clone();
                sc.spawn(move || {
                    if router_on && !s_e.is_empty() {
                        tenant.entity_pivot_sessions(&eid, &s_e).unwrap_or_default()
                    } else {
                        Vec::new()
                    }
                })
            };

            (
                h_sr.join().unwrap_or_default(),
                h_win.join().unwrap_or_default(),
                h_pivot.join().unwrap_or_default(),
            )
        });

        let start_proc = Instant::now();
        for hit in sr_hits {
            let coverage_bonus = hit.lexical_hits as f32 * s.state.ranking_config.lexical_weight
                + hit.temporal_hits as f32 * s.state.ranking_config.temporal_weight
                + hit.entity_hits as f32 * s.state.ranking_config.entity_weight;
            *s.session_route_scores.entry(hit.session_id).or_insert(0.0) +=
                hit.score + coverage_bonus;
        }
        s.diag.route_session_ms = start_proc.elapsed().as_millis() as u64;

        let start_proc = Instant::now();
        for hit in win_hits {
            *s.session_route_scores.entry(hit.session_id).or_insert(0.0) +=
                s.weights.time_window_bonus;
        }
        s.diag.route_window_ms = start_proc.elapsed().as_millis() as u64;

        let start_proc = Instant::now();
        if !pivot_hits.is_empty() {
            let total_sessions = pivot_hits.len().max(1);
            let multi_entity = subject_for_routes.len() >= 2;
            for hit in &pivot_hits {
                if multi_entity && hit.entity_hits >= 2 {
                    *s.session_route_scores.entry(hit.session_id.clone()).or_insert(0.0) += 0.25;
                } else if hit.entity_hits >= 1 && total_sessions <= 12 {
                    *s.session_route_scores.entry(hit.session_id.clone()).or_insert(0.0) += 0.08;
                }
            }
        }
        s.diag.route_pivot_ms = start_proc.elapsed().as_millis() as u64;
    }
    (s.diag.planning_ms, s.diag.planning_us) = elapsed_ms_and_us(planning_start);
}

fn route_phase(s: &mut QueryPipelineState) {
    if !features::enabled(Feature::SessionRouter) || !lanes::enabled(Lane::Route) {
        return;
    }
    let route_probe_queries = if s.budget.route_probe_query_limit == 0 {
        Vec::new()
    } else if s.plan.coverage_facets.is_empty() {
        vec![s.plan.fts_queries.first().cloned().unwrap_or_else(|| s.query_text.clone())]
    } else {
        s.plan
            .coverage_facets
            .iter()
            .take(s.budget.route_probe_query_limit)
            .map(|facet| facet.text.clone())
            .collect::<Vec<_>>()
    };
    let mut route_probe_results = Vec::new();
    let tenant_for_probes = s.tenant.clone();
    let eid_for_probes = s.payload.entity_id.clone();
    let hit_limit = s.budget.route_probe_hit_limit;
    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for (probe_idx, probe_query) in route_probe_queries.iter().enumerate() {
            let tenant = tenant_for_probes.clone();
            let eid = eid_for_probes.clone();
            handles.push(sc.spawn(move || {
                let hits = tenant
                    .fts_search(probe_query.as_str(), hit_limit, eid.as_deref())
                    .unwrap_or_else(|err| {
                        // An empty lane and a failed lane look identical
                        // downstream, and the difference is a recall bug.
                        tracing::warn!(error = %err, probe = %probe_query, "route probe FTS failed");
                        Vec::new()
                    });
                (probe_idx, hits)
            }));
        }
        for handle in handles {
            if let Ok(res) = handle.join() {
                route_probe_results.push(res);
            }
        }
    });

    // Lanes for RRF fusion. Each lane is independently ranked.
    let mut lanes: Vec<Vec<(String, f32)>> = Vec::with_capacity(5);

    // Lane 1: entity-anchored pivot (resolved subjects → sessions). Strongest
    // signal for multi-hop and cross-entity questions.
    if !s.plan.subject_entities.is_empty() {
        let tenant = s.tenant.clone();
        let eid = s.payload.entity_id.clone();
        let subjects = s.plan.subject_entities.clone();
        if let Ok(hits) = tenant.entity_pivot_sessions(eid.as_deref().unwrap_or(""), &subjects) {
            lanes.push(hits.into_iter().map(|h| (h.session_id, h.score)).collect());
        }
    }

    // Lane 2: FTS session router (router_text OR of query terms).
    let s_router_lane: Vec<(String, f32)> =
        s.session_route_scores.iter().map(|(sid, score)| (sid.clone(), *score)).collect();
    if !s_router_lane.is_empty() {
        lanes.push(s_router_lane);
    }

    // Lane 3: time-window hits.
    // Already merged into s.session_route_scores via the plan_phase time-window
    // pass; we re-extract the contribution by snapshot before any further
    // mutations. For now this lane is implicit in lane 2.

    // Lane 4: FTS probe votes.
    let mut probe_lane: Vec<(String, f32)> = Vec::new();
    let probe_ids: Vec<String> = route_probe_results
        .iter()
        .flat_map(|(_, hits)| {
            hits.iter().take(s.budget.route_probe_hit_limit).map(|(m, _)| m.clone())
        })
        .collect();
    let probe_identity = s.tenant.memory_identity_batch(&probe_ids).unwrap_or_else(|err| {
        tracing::warn!(error = ?err, "route probe identity lookup failed");
        HashMap::new()
    });
    for (probe_idx, route_probe_hits) in route_probe_results {
        let vote_weight: f32 = match probe_idx {
            0 => 3.0,
            1 => 2.0,
            _ => 1.0,
        };
        let mut seen_in_probe: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (memory_id, _) in route_probe_hits.iter().take(s.budget.route_probe_hit_limit) {
            let session_id = probe_identity
                .get(memory_id)
                .map(|(session, _)| session.clone())
                .filter(|session| !session.is_empty())
                .or_else(|| routed_session_from_memory_id(memory_id));
            if let Some(session_id) = session_id {
                if seen_in_probe.insert(session_id.clone()) {
                    probe_lane.push((session_id, vote_weight));
                }
            }
        }
    }
    if !probe_lane.is_empty() {
        lanes.push(probe_lane);
    }

    // Lane 5: if we have no entity-anchored hits, fall back to raw FTS probe on
    // the plain question text. Cheap insurance.
    if lanes.is_empty() {
        let tenant = s.tenant.clone();
        let q = s.query_text.clone();
        let eid = s.payload.entity_id.clone();
        if let Ok(hits) = tenant.search_session_router(
            eid.as_deref().unwrap_or(""),
            &q,
            &[],
            &[],
            &s.plan.subject_entities,
            24,
        ) {
            let lane: Vec<(String, f32)> =
                hits.into_iter().map(|h| (h.session_id, h.score)).collect();
            if !lane.is_empty() {
                lanes.push(lane);
            }
        }
    }

    // Fuse with RRF (c=60, standard constant).
    let fused = rrf_fuse(&lanes, 60.0);

    // Also seed the legacy aggregate for downstream diagnostics.
    s.session_route_scores = fused.iter().cloned().collect();

    let route_take = if s.plan.needs_decomposition || s.plan.cross_entity {
        s.budget.route_take_hard
    } else {
        s.budget.route_take_simple
    };
    s.adaptive_profile.route_sessions =
        fused.into_iter().take(route_take).map(|(session, _)| session).collect();
    s.adaptive_profile.route_strength = if s.adaptive_profile.route_sessions.is_empty() {
        0.0
    } else if s.plan.needs_decomposition || s.plan.cross_entity {
        1.0
    } else {
        0.65
    };
    s.diag.routed_sessions = s.adaptive_profile.route_sessions.len() as u64;
    (s.diag.route_ms, s.diag.route_us) = elapsed_ms_and_us(s.route_start);
}

fn retrieval_phase(s: &mut QueryPipelineState) {
    retrieval_ann(s);
    retrieval_fts(s);
    retrieval_cards(s);
}

fn retrieval_ann(s: &mut QueryPipelineState) {
    if !lanes::enabled(Lane::Vector) {
        return;
    }
    let stage_start = Instant::now();
    let embed_dim = s.state.semantic.embedding_dim();
    let semantic_queries = s
        .plan
        .semantic_queries
        .iter()
        .take(s.budget.semantic_query_limit)
        .cloned()
        .collect::<Vec<_>>();
    let primary_qembed_clone = s.primary_qembed.clone();
    // Guard: only proceed if the primary embedding has the right dimension.
    // An empty or wrong-dimension vector causes usearch to exhibit undefined
    // behaviour (the debug assert is stripped in release builds) and the
    // search can spin/hang indefinitely → 30-second timeouts.
    if primary_qembed_clone.len() != embed_dim {
        tracing::warn!(
            got = primary_qembed_clone.len(),
            expected = embed_dim,
            "primary query embedding has wrong dimension; skipping ANN retrieval"
        );
        (s.diag.ann_ms, s.diag.ann_us) = elapsed_ms_and_us(stage_start);
        return;
    }
    let mut embeddings = vec![primary_qembed_clone];
    if semantic_queries.len() > 1 {
        let query_refs: Vec<&str> = semantic_queries.iter().skip(1).map(|q| q.as_str()).collect();
        match s.state.semantic.embed_queries(&query_refs) {
            Ok(batch_results) => embeddings.extend(batch_results),
            Err(err) => {
                tracing::warn!(error = ?err, "query variant embedding failed; using primary only")
            }
        }
    }

    let scoped_entity_id = s.payload.entity_id.clone();
    let stage_start = Instant::now();
    /// (variant index, rerank seeds, hits, raw neighbours, search attempts, final top-k)
    type AnnWorkerResult = (usize, Vec<String>, Vec<RankedItem>, Vec<(u64, f32)>, u64, u64);
    let ann_results: Vec<AnnWorkerResult> = {
        let tenant_clone = s.tenant.clone();
        let eid = scoped_entity_id.clone();
        let profile = &s.adaptive_profile;
        let query_limit = s.limit;
        let semantic_top = s.semantic_top;
        let dedup_threshold = s.weights.dedup_similarity_threshold;
        std::thread::scope(|sc| {
            let mut handles = Vec::with_capacity(embeddings.len());
            for (idx, embedding) in embeddings.iter().enumerate() {
                let tenant = tenant_clone.clone();
                let embedding = embedding.clone();
                let eid = eid.clone();
                let handle = sc.spawn(move || {
                    let mut hnsw_hits = Vec::new();
                    let mut variant_rerank_seed_ids = Vec::new();
                    let mut local_cache: HashMap<u64, Option<(u64, String)>> = HashMap::new();
                    let mut search_attempts = 1u64;
                    let mut search_top = semantic_top as u64;
                    let hnsw_raw = if let Some(entity_id) = eid.as_deref() {
                        let scoped_max_top = semantic_top;
                        let scoped_start = scoped_semantic_start(scoped_max_top);
                        let scoped_step = scoped_semantic_step();
                        let scoped_min_hits = scoped_semantic_min_hits(query_limit, scoped_max_top);
                        let mut current_top = scoped_start;
                        let mut attempts = 0usize;
                        let mut prev_hit_count: Option<usize> = None;
                        let mut prev_top_similarity: Option<f32> = None;
                        let mut seen_vector_ids: HashSet<u64> = HashSet::new();
                        let mut cumulative_hnsw_hits: Vec<RankedItem> = Vec::new();
                        let hnsw_raw = loop {
                            attempts += 1;
                            let current_raw = match tenant
                                .vectors()
                                .and_then(|v| v.search(Some(entity_id), &embedding, current_top))
                            {
                                Ok(raw) => raw,
                                Err(err) => {
                                    tracing::warn!(error = ?err, "scoped ANN search failed");
                                    Vec::new()
                                }
                            };
                            let scoped_last_raw = current_raw.clone();
                            let mut scoped_top_similarity = None;
                            let mut new_scoped_hits = 0usize;

                            let unresolved_ids: Vec<u64> = current_raw
                                .iter()
                                .map(|(vid, _)| *vid)
                                .filter(|vid| !local_cache.contains_key(vid))
                                .collect();
                            if !unresolved_ids.is_empty() {
                                if let Ok(looked) =
                                    tenant.lookup_by_vector_ids_batch(&unresolved_ids)
                                {
                                    for (vid, hit) in unresolved_ids.into_iter().zip(looked) {
                                        local_cache.insert(vid, hit);
                                    }
                                }
                            }

                            for (vid, dist) in current_raw.iter() {
                                if !seen_vector_ids.insert(*vid) {
                                    continue;
                                }
                                let lookup = local_cache.get(vid).cloned().unwrap_or(None);
                                let Some((ts, mem_id)) = lookup else {
                                    continue;
                                };
                                // Post-ANN similarity floor: drop zero/near-zero hits
                                // before they pollute cumulative_hnsw_hits and the
                                // downstream RRF lane.
                                let similarity = cosine_similarity_from_distance(*dist);
                                if similarity < MIN_HIT_SIMILARITY {
                                    continue;
                                }
                                new_scoped_hits += 1;
                                if cumulative_hnsw_hits.len() < NEURAL_TOP {
                                    variant_rerank_seed_ids.push(mem_id.clone());
                                }
                                cumulative_hnsw_hits
                                    .push(RankedItem { memory_id: mem_id, timestamp: ts });
                                if scoped_top_similarity.is_none() {
                                    scoped_top_similarity =
                                        Some(cosine_similarity_from_distance(*dist));
                                }
                            }

                            let scoped_state = crate::api::utils::ScopedAnnState {
                                attempt: attempts,
                                current_top,
                                max_top: scoped_max_top,
                                hit_count: cumulative_hnsw_hits.len(),
                                min_hits: scoped_min_hits,
                                top_similarity: scoped_top_similarity,
                                prev_hit_count,
                                prev_top_similarity,
                            };
                            if crate::api::utils::should_stop_scoped_ann(&scoped_state)
                                || (attempts >= 2
                                    && cumulative_hnsw_hits.len() >= scoped_min_hits
                                    && new_scoped_hits == 0)
                            {
                                break scoped_last_raw;
                            }
                            let next_top =
                                current_top.saturating_add(scoped_step).min(scoped_max_top);
                            if next_top == current_top {
                                break scoped_last_raw;
                            }
                            prev_hit_count = Some(cumulative_hnsw_hits.len());
                            prev_top_similarity = scoped_top_similarity;
                            current_top = next_top;
                        };
                        hnsw_hits = cumulative_hnsw_hits;
                        search_attempts = attempts as u64;
                        search_top = current_top as u64;
                        hnsw_raw
                    } else {
                        let hnsw_raw = match tenant
                            .vectors()
                            .and_then(|v| v.search(None, &embedding, semantic_top))
                        {
                            Ok(raw) => raw,
                            Err(err) => {
                                tracing::warn!(error = ?err, "ANN search failed");
                                Vec::new()
                            }
                        };
                        let vids: Vec<u64> = hnsw_raw.iter().map(|(vid, _)| *vid).collect();
                        if let Ok(looked) = tenant.lookup_by_vector_ids_batch(&vids) {
                            let looked_ids: Vec<String> =
                                looked.iter().flatten().map(|(_, m)| m.clone()).collect();
                            let identity =
                                tenant.memory_identity_batch(&looked_ids).unwrap_or_default();
                            for (rank, (_vid, dist)) in hnsw_raw.iter().enumerate() {
                                if let Some((ts, mem_id)) = looked[rank].clone() {
                                    // Post-ANN similarity floor: drop zero/near-zero hits
                                    // before they enter hnsw_hits and the downstream RRF lane.
                                    let raw_similarity = cosine_similarity_from_distance(*dist);
                                    if raw_similarity < MIN_HIT_SIMILARITY {
                                        continue;
                                    }
                                    let routed_match = identity
                                        .get(&mem_id)
                                        .map(|(session, _)| {
                                            profile.route_sessions.contains(session)
                                        })
                                        .unwrap_or(false);
                                    if profile.route_strength > 0.0
                                        && !routed_match
                                        && hnsw_hits.len() < (query_limit.saturating_div(2).max(1))
                                        && raw_similarity < dedup_threshold
                                    {
                                        continue;
                                    }
                                    if rank < NEURAL_TOP {
                                        variant_rerank_seed_ids.push(mem_id.clone());
                                    }
                                    hnsw_hits.push(RankedItem { memory_id: mem_id, timestamp: ts });
                                }
                            }
                        }
                        hnsw_raw
                    };
                    (idx, variant_rerank_seed_ids, hnsw_hits, hnsw_raw, search_attempts, search_top)
                });
                handles.push(handle);
            }
            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|e| {
                        tracing::error!("ANN search thread panicked: {:?}", e);
                        (usize::MAX, Vec::new(), Vec::new(), Vec::new(), 0, 0)
                    })
                })
                .collect::<Vec<_>>()
        })
    };

    let mut primary_hnsw_raw = Vec::new();
    let mut semantic_ranked_lists = Vec::new();
    for (idx, _seed_ids, hnsw_hits, hnsw_raw, attempts, top) in ann_results {
        if idx == usize::MAX {
            continue;
        }
        if idx == 0 {
            primary_hnsw_raw = hnsw_raw;
            s.diag.scoped_ann_attempts = attempts;
            s.diag.scoped_ann_top = top;
            s.diag.scoped_primary_hits = hnsw_hits.len() as u64;
        }
        semantic_ranked_lists.push((
            query_variant_weight(idx, QueryModality::Semantic, s.plan.intent)
                * s.adaptive_profile.semantic_scale,
            hnsw_hits,
        ));
    }
    s.primary_hnsw_raw = primary_hnsw_raw;
    s.semantic_ranked_lists = semantic_ranked_lists;
    (s.diag.ann_ms, s.diag.ann_us) = elapsed_ms_and_us(stage_start);
}

fn retrieval_fts(s: &mut QueryPipelineState) {
    if !lanes::enabled(Lane::Fts) {
        return;
    }
    let stage_start = Instant::now();
    let mut fts_ranked_lists = Vec::new();
    let mut fts_memory_ids_to_lookup = Vec::new();
    let mut fts_results_per_query = Vec::new();

    let fts_queries_to_run: Vec<&String> =
        s.plan.fts_queries.iter().take(s.budget.fts_query_limit).collect();
    {
        let tenant_clone = s.tenant.clone();
        let eid = s.payload.entity_id.clone();
        let fts_top = s.fts_top;
        std::thread::scope(|sc| {
            let mut handles = Vec::new();
            for (idx, fts_query) in fts_queries_to_run.into_iter().enumerate() {
                let tenant = tenant_clone.clone();
                let eid = eid.clone();
                handles.push(sc.spawn(move || {
                    let hits = tenant
                        .fts_search(fts_query.as_str(), fts_top, eid.as_deref())
                        .unwrap_or_else(|err| {
                            tracing::warn!(error = %err, query = %fts_query, "FTS lane failed");
                            Vec::new()
                        });
                    (idx, hits)
                }));
            }
            let mut temp_results = Vec::new();
            for handle in handles {
                if let Ok(res) = handle.join() {
                    temp_results.push(res);
                }
            }
            temp_results.sort_by_key(|(idx, _)| *idx);
            for (_, hits) in temp_results {
                fts_results_per_query.push(hits);
            }
        });
    }

    for hits in &fts_results_per_query {
        for (mid, _) in hits {
            fts_memory_ids_to_lookup.push(mid.clone());
        }
    }
    let fts_lookup =
        s.tenant.lookup_by_memory_ids_batch(&fts_memory_ids_to_lookup).unwrap_or_default();
    for (idx, hits) in fts_results_per_query.into_iter().enumerate() {
        let mut ranked_hits = Vec::new();
        for (mid, _) in hits {
            if let Some((ts, _)) = fts_lookup.get(&mid).copied() {
                ranked_hits.push(RankedItem { memory_id: mid, timestamp: ts });
            }
        }
        fts_ranked_lists.push((
            query_variant_weight(idx, QueryModality::Lexical, s.plan.intent)
                * s.adaptive_profile.lexical_scale,
            ranked_hits,
        ));
    }
    s.fts_ranked_lists = fts_ranked_lists;
    (s.diag.fts_ms, s.diag.fts_us) = elapsed_ms_and_us(stage_start);
}

fn retrieval_cards(s: &mut QueryPipelineState) {
    let stage_start = Instant::now();
    let mut card_ranked_items = Vec::new();
    let entity_scope = s
        .payload
        .entity_id
        .as_ref()
        .filter(|_| features::enabled(Feature::MemoryCards) && lanes::enabled(Lane::Cards));
    if let Some(entity_id) = entity_scope {
        let include_stale_cards = query_allows_stale_cards(&s.query_text, &s.plan);
        let card_hits = s
            .tenant
            .search_memory_cards(&MemoryCardSearchInput {
                entity_id,
                lexical_terms: &s.plan.lexical_terms,
                temporal_terms: &s.plan.temporal_terms,
                entities: &s.plan.subject_entities,
                route_sessions: &s.adaptive_profile.route_sessions,
                include_stale: include_stale_cards,
                limit: s.budget.card_limit,
            })
            .unwrap_or_default();
        s.diag.memory_card_hits = card_hits.len() as u64;
        for hit in card_hits {
            card_ranked_items.push(RankedItem { memory_id: hit.card_id, timestamp: hit.timestamp });
            if !hit.source_session_id.is_empty()
                && hit.lexical_hits + hit.temporal_hits + hit.entity_hits >= 2
            {
                *s.session_route_scores.entry(hit.source_session_id).or_insert(0.0) += hit.score
                    * s.state.ranking_config.session_boost
                    * s.state.ranking_config.session_boost_routed;
            }
        }
    }
    s.card_ranked_items = card_ranked_items;
    (s.diag.card_ms, s.diag.card_us) = elapsed_ms_and_us(stage_start);
}

/// Why the reranker did or did not run; reported as `x-tm-rerank-reason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub(crate) enum RerankDecision {
    #[default]
    Disabled = 0,
    TooFewCandidates = 1,
    HeuristicApplied = 2,
    HeuristicSkipped = 3,
    Always = 4,
    GateUncertain = 5,
    GateConfident = 6,
    Requested = 7,
}

impl RerankDecision {
    fn applies(self) -> bool {
        matches!(
            self,
            RerankDecision::HeuristicApplied
                | RerankDecision::Always
                | RerankDecision::GateUncertain
                | RerankDecision::Requested
        )
    }
}

/// `TELLODB_RERANK_POLICY`: `heuristic` (query keywords and intent, the
/// original behaviour), `always`, or `gate` (rerank only when the top ANN
/// similarities are too close to call).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RerankPolicy {
    Heuristic,
    Always,
    Gate,
}

pub(crate) fn rerank_policy_name() -> &'static str {
    match rerank_policy() {
        RerankPolicy::Heuristic => "heuristic",
        RerankPolicy::Always => "always",
        RerankPolicy::Gate => "gate",
    }
}

/// Default `gate`: rerank only when stage-1 retrieval is actually uncertain.
///
/// The previous default, `heuristic`, decided from the query string — it
/// fires on " and ", "would", "might", "why ", or merely a long question — so
/// it reranked 93% of LongMemEval dev. A gate that opens for almost every
/// query is not a gate, and the cross-encoder is the largest query stage
/// (169 ms p50) for a recall difference that is not distinguishable from zero
/// (-0.7, CI -2.0…+0.0). `gate` uses the retrieval scores it is supposed to.
fn rerank_policy() -> RerankPolicy {
    static POLICY: std::sync::OnceLock<RerankPolicy> = std::sync::OnceLock::new();
    *POLICY.get_or_init(|| {
        match std::env::var("TELLODB_RERANK_POLICY").unwrap_or_default().trim() {
            "always" => RerankPolicy::Always,
            "heuristic" => RerankPolicy::Heuristic,
            _ => RerankPolicy::Gate,
        }
    })
}

/// Relative similarity gap between the top-1 and top-5 ANN hits below which
/// the gate reranks (`TELLODB_RERANK_MARGIN`, default 0.05).
pub(crate) fn rerank_margin() -> f32 {
    static MARGIN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *MARGIN.get_or_init(|| {
        std::env::var("TELLODB_RERANK_MARGIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &f32| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.05)
    })
}

/// Candidates sent to the cross-encoder (`TELLODB_RERANK_TOP`, default 25).
pub(crate) fn rerank_top() -> usize {
    static TOP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TOP.get_or_init(|| {
        std::env::var("TELLODB_RERANK_TOP")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v| (2..=500).contains(v))
            .unwrap_or(NEURAL_TOP)
    })
}

/// True when stage-1 retrieval is uncertain: the top-1 similarity is within
/// `margin` (relative) of the fifth (or last) hit. `hnsw_raw` holds
/// `(vector_id, cosine distance)` sorted by distance.
pub(crate) fn rerank_gate_uncertain(hnsw_raw: &[(u64, f32)], margin: f32) -> bool {
    let kth_index = hnsw_raw.len().min(5).checked_sub(1);
    let (Some(first), Some(kth)) = (hnsw_raw.first(), kth_index.and_then(|i| hnsw_raw.get(i)))
    else {
        return false;
    };
    let top = 1.0 - first.1;
    let other = 1.0 - kth.1;
    if top <= f32::EPSILON {
        return true;
    }
    (top - other) / top < margin
}

fn rerank_phase(s: &mut QueryPipelineState) {
    s.neural_scores = HashMap::new();
    if !lanes::enabled(Lane::Rerank) {
        s.diag.rerank_reason = RerankDecision::Disabled;
        return;
    }
    if !s.state.semantic.is_rerank_enabled() {
        s.diag.rerank_reason = RerankDecision::Disabled;
        return;
    }
    let retrieval_profile = retrieval_profile();
    let auto_rerank = auto_rerank_enabled(retrieval_profile)
        && (s.plan.needs_decomposition
            || s.plan.cross_entity
            || s.plan.ordinal_rank.is_some()
            || matches!(
                s.plan.intent,
                QueryIntent::Inference
                    | QueryIntent::TemporalAggregation
                    | QueryIntent::NumericAggregation
                    | QueryIntent::PeripheralMention
            ));
    let decision = if s.primary_hnsw_raw.len() < 2 {
        RerankDecision::TooFewCandidates
    } else if s.enable_neural_rerank {
        RerankDecision::Requested
    } else {
        match rerank_policy() {
            RerankPolicy::Always => RerankDecision::Always,
            RerankPolicy::Gate if rerank_gate_uncertain(&s.primary_hnsw_raw, rerank_margin()) => {
                RerankDecision::GateUncertain
            }
            RerankPolicy::Gate => RerankDecision::GateConfident,
            RerankPolicy::Heuristic
                if should_apply_neural_rerank(&s.query_text, &s.primary_hnsw_raw, auto_rerank) =>
            {
                RerankDecision::HeuristicApplied
            }
            RerankPolicy::Heuristic => RerankDecision::HeuristicSkipped,
        }
    };
    s.diag.rerank_reason = decision;
    if decision.applies() {
        let stage_start = Instant::now();
        let neural_top = rerank_top();
        s.diag.rerank_applied = true;
        let mut rerank_seed_ids = Vec::new();
        for (_weight, list) in &s.semantic_ranked_lists {
            for item in list.iter().take(neural_top / 2) {
                rerank_seed_ids.push(item.memory_id.clone());
            }
        }
        for (_weight, list) in &s.fts_ranked_lists {
            for item in list.iter().take(neural_top / 2) {
                rerank_seed_ids.push(item.memory_id.clone());
            }
        }
        let mut rerank_seen = HashSet::new();
        let active_seeds: Vec<String> = rerank_seed_ids
            .into_iter()
            .filter(|mid: &String| rerank_seen.insert(mid.clone()))
            .collect();
        let lookup = s.tenant.lookup_by_memory_ids_batch(&active_seeds).unwrap_or_default();
        let obs_keys: Vec<(u64, String)> = active_seeds
            .iter()
            .filter_map(|mid: &String| lookup.get(mid).map(|(ts, _)| (*ts, mid.clone())))
            .take(neural_top)
            .collect();
        let observations = s.tenant.get_observations_batch(&obs_keys).unwrap_or_default();
        let mut rerank_items = Vec::new();
        let mut rerank_texts = Vec::new();
        for mid in active_seeds {
            if let Some(obs) = observations.get(&mid) {
                rerank_items.push(mid);
                rerank_texts.push(obs.textual_content.clone());
            }
        }
        for (item_chunk, text_chunk) in
            rerank_items.chunks(NEURAL_BATCH).zip(rerank_texts.chunks(NEURAL_BATCH))
        {
            let scores = s
                .state
                .semantic
                .predict_scores_batch(&s.query_text, text_chunk)
                .unwrap_or_else(|_| vec![0.0; item_chunk.len()]);
            for (mid, score) in item_chunk.iter().cloned().zip(scores.into_iter()) {
                s.neural_scores.insert(mid, score);
            }
        }
        (s.diag.rerank_ms, s.diag.rerank_us) = elapsed_ms_and_us(stage_start);
    }
}

fn fusion_phase(s: &mut QueryPipelineState) {
    let stage_start = Instant::now();
    let mut ranked_sources = s.semantic_ranked_lists.clone();
    ranked_sources.extend(s.fts_ranked_lists.clone());
    if !s.card_ranked_items.is_empty() {
        let card_weight = if s.plan.needs_decomposition || s.plan.cross_entity {
            s.state.ranking_config.card_boost * s.state.ranking_config.card_boost_strong
        } else if matches!(s.plan.intent, QueryIntent::TemporalAggregation | QueryIntent::Inference)
        {
            s.state.ranking_config.card_boost * s.state.ranking_config.card_boost_medium
        } else {
            s.state.ranking_config.card_boost
        };
        ranked_sources.push((card_weight, s.card_ranked_items.clone()));
    }
    if !s.neural_scores.is_empty() {
        let mut scored_items: Vec<_> = s.neural_scores.clone().into_iter().collect();
        scored_items.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
        });
        let mut neural_items = Vec::new();
        let mids: Vec<String> = scored_items.iter().map(|(mid, _)| mid.clone()).collect();
        let lookup = s.tenant.lookup_by_memory_ids_batch(&mids).unwrap_or_default();
        for (mid, _) in scored_items {
            if let Some(&(ts, _)) = lookup.get(&mid) {
                neural_items.push(RankedItem { memory_id: mid, timestamp: ts });
            }
        }
        ranked_sources.push((1.5, neural_items));
    }
    let fused = weighted_reciprocal_rank_fusion(ranked_sources, adaptive_rrf_k(&s.plan));
    (s.diag.fuse_ms, s.diag.fuse_us) = elapsed_ms_and_us(stage_start);

    if s.payload.reference_time_ms.is_none() {
        s.now_ms =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
    }
    let mut fused_map: HashMap<String, (u64, f32)> = HashMap::new();
    for (mid, ts, score) in fused {
        let entry = fused_map.entry(mid).or_insert((ts, 0.0));
        entry.0 = ts;
        entry.1 = entry.1.max(score);
    }

    for (mid, boost) in &s.routed_memory_ids {
        if let Some((ts, _)) = s.tenant.lookup_by_memory_id(mid).unwrap_or(None) {
            let entry = fused_map.entry(mid.clone()).or_insert((ts, 0.0));
            entry.0 = ts;
            entry.1 += *boost;
        }
    }

    let stage_start = Instant::now();
    if s.plan.intent == QueryIntent::Inference
        && !s.primary_qembed.is_empty()
        && features::enabled(Feature::Preferences)
    {
        if let Some(ref entity_id) = s.payload.entity_id {
            let preference_memories =
                s.tenant.get_preference_memories(entity_id, 96).unwrap_or_default();
            if !preference_memories.is_empty() {
                let option_embeddings = [s.primary_qembed.clone()];
                let memory_ids: Vec<String> =
                    preference_memories.iter().map(|(memory_id, _)| memory_id.clone()).collect();
                let lookup = s.tenant.lookup_by_memory_ids_batch(&memory_ids).unwrap_or_default();
                let observation_keys: Vec<(u64, String)> = memory_ids
                    .iter()
                    .filter_map(|memory_id| {
                        lookup.get(memory_id).map(|(ts, _)| (*ts, memory_id.clone()))
                    })
                    .collect();
                let observations =
                    s.tenant.get_observations_batch(&observation_keys).unwrap_or_default();
                for (memory_id, strength) in preference_memories {
                    let Some(obs) = observations.get(&memory_id) else {
                        continue;
                    };
                    if obs.embedding.len() != s.primary_qembed.len() || obs.embedding.is_empty() {
                        continue;
                    }
                    let best_similarity = option_embeddings
                        .iter()
                        .filter(|candidate| candidate.len() == obs.embedding.len())
                        .map(|candidate| cosine_similarity(candidate, &obs.embedding))
                        .fold(-1.0f32, f32::max);
                    if best_similarity >= 0.35 {
                        if let Some((ts, _)) = lookup.get(&memory_id).copied() {
                            let entry = fused_map.entry(memory_id.clone()).or_insert((ts, 0.0));
                            entry.0 = ts;
                            entry.1 += best_similarity.max(0.0) * 0.12 + strength * 0.04;
                        }
                    }
                }
            }
        }
    }
    (s.diag.preference_ms, s.diag.preference_us) = elapsed_ms_and_us(stage_start);

    let stage_start = Instant::now();
    let mut link_seed_ids: Vec<(String, f32)> =
        fused_map.iter().map(|(mid, (_, score))| (mid.clone(), *score)).collect();
    link_seed_ids.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
    });

    let mut graph_scores: HashMap<String, f32> = HashMap::new();
    let seed_top: Vec<String> =
        link_seed_ids.into_iter().take(graph_seed_count()).map(|(mid, _)| mid).collect();

    use rayon::prelude::*;
    let intent_for_graph = s.plan.intent;
    let graph_lane = lanes::enabled(Lane::Graph);
    let links_on = graph_lane
        && (features::enabled(Feature::DerivedLinks)
            || features::enabled(Feature::RetrospectiveLinks));
    let edges_on = graph_lane && features::enabled(Feature::GraphEdges);
    let seeds_start = Instant::now();
    let link_start = Instant::now();
    let link_scores: Vec<HashMap<String, f32>> = if links_on {
        seed_top
            .par_iter()
            .map(|seed_mid| {
                s.tenant.get_link_cluster_scores(seed_mid, graph_max_depth()).unwrap_or_default()
            })
            .collect()
    } else {
        Vec::new()
    };
    s.diag.graph_links_us = link_start.elapsed().as_micros() as u64;
    let edge_start = Instant::now();
    let (edge_scores, expanded) = if edges_on {
        collect_edge_cluster_scores_for_seeds(
            &s.tenant,
            &seed_top,
            graph_max_depth(),
            None,
            Some(intent_for_graph),
        )
    } else {
        (Vec::new(), 0)
    };
    s.diag.graph_edges_us = edge_start.elapsed().as_micros() as u64;
    s.diag.graph_expanded = expanded;
    s.diag.graph_seeds_wall_us = seeds_start.elapsed().as_micros() as u64;

    let mut all_linked: Vec<String> = Vec::with_capacity(seed_top.len() * 8);
    for link in link_scores {
        for (linked_mid, boost) in link {
            *graph_scores.entry(linked_mid.clone()).or_insert(0.0) += boost;
            all_linked.push(linked_mid);
        }
    }
    for edge in edge_scores {
        for (linked_mid, boost) in edge {
            *graph_scores.entry(linked_mid.clone()).or_insert(0.0) += boost;
            all_linked.push(linked_mid);
        }
    }
    all_linked.sort();
    all_linked.dedup();
    let lookup_start = Instant::now();
    if !all_linked.is_empty() {
        if let Ok(lookup) = s.tenant.lookup_by_memory_ids_batch(&all_linked) {
            for (linked_mid, (ts, _)) in lookup {
                let entry = fused_map.entry(linked_mid).or_insert((ts, 0.0));
                entry.0 = ts;
            }
        }
    }

    s.diag.graph_lookup_us += lookup_start.elapsed().as_micros() as u64;
    let entities_start = Instant::now();

    // Entity-graph retrieval lane: resolve query entities against registry,
    // then traverse the edges table to find connected memories.
    if let Some(entity_scope) = s.payload.entity_id.as_ref().filter(|_| edges_on) {
        if !entity_scope.is_empty() {
            let mut entity_seeds: Vec<String> = s.plan.subject_entities.clone();
            let query_lines: Vec<String> =
                std::iter::once(s.payload.textual_query.clone()).collect();
            for phrase in extract_named_phrases(&query_lines) {
                if phrase.len() >= 3
                    && !entity_seeds.iter().any(|e| e.eq_ignore_ascii_case(&phrase))
                {
                    entity_seeds.push(phrase);
                }
            }
            entity_seeds.truncate(8);

            let mut batched_edges = Vec::new();
            for seed in &entity_seeds {
                if let Ok(edges) = s.tenant.graph_query_edges(seed, None, "Both", 50) {
                    batched_edges.extend(edges.into_iter().filter(|e| !e.memory_id.is_empty()));
                }
            }
            if !batched_edges.is_empty() {
                let mut edge_mids =
                    batched_edges.iter().map(|e| e.memory_id.clone()).collect::<Vec<_>>();
                edge_mids.sort();
                edge_mids.dedup();

                if let Ok(lookup) = s.tenant.lookup_by_memory_ids_batch(&edge_mids) {
                    for edge in batched_edges {
                        if let Some(&(ts, _)) = lookup.get(&edge.memory_id) {
                            let boost = edge.weight * 0.5;
                            *graph_scores.entry(edge.memory_id.clone()).or_insert(0.0) += boost;
                            let entry =
                                fused_map.entry(edge.memory_id.clone()).or_insert((ts, 0.0));
                            entry.0 = ts;
                        }
                    }
                }
            }
        }
    }

    s.diag.graph_entities_us = entities_start.elapsed().as_micros() as u64;

    // Graph scores are sums over every traversed edge, so memories with many
    // derived records reached values in the hundreds and outranked
    // semantically relevant results regardless of the query. Scale to [0, 1]
    // so graph evidence is one bounded signal among the others.
    let max_graph = graph_scores.values().copied().fold(0.0f32, f32::max);
    if max_graph > 0.0 {
        for score in graph_scores.values_mut() {
            *score /= max_graph;
        }
    }
    s.graph_scores = graph_scores;
    (s.diag.graph_ms, s.diag.graph_us) = elapsed_ms_and_us(stage_start);

    let mut fused_vec: Vec<(String, u64, f32)> =
        fused_map.into_iter().map(|(mid, (ts, score))| (mid, ts, score)).collect();
    fused_vec.sort_by(|a, b| {
        b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
    });
    s.fused = fused_vec;
}

fn score_phase(s: &mut QueryPipelineState) -> Result<Vec<QueryResult>, StatusCode> {
    let hydrate_start = Instant::now();
    score_hydrate(s)?;
    (s.diag.hydrate_ms, s.diag.hydrate_us) = elapsed_ms_and_us(hydrate_start);

    let loop_start = Instant::now();
    let evidence_cards = score_loop(s);
    s.diag.score_loop_us = loop_start.elapsed().as_micros() as u64;

    score_build_response(s, evidence_cards)
}

/// Loads everything scoring needs for the fused candidates. The five reads are
/// independent, so they run on scoped threads (this already executes on the
/// blocking pool; nesting `spawn_blocking` + `block_on` here tied up extra
/// pool threads). Any read failure fails the query: scoring with missing
/// observations or stale-fact data would silently return wrong results.
fn score_hydrate(s: &mut QueryPipelineState) -> Result<(), StatusCode> {
    let observation_keys: Vec<(u64, String)> =
        s.fused.iter().map(|(mid, ts, _)| (*ts, mid.clone())).collect();
    let observation_memory_ids: Vec<String> =
        observation_keys.iter().map(|(_, mid)| mid.clone()).collect();
    let pit = s.payload.point_in_time_ms;
    let tenant = s.tenant.as_ref();

    fn timed<T>(f: impl FnOnce() -> anyhow::Result<T>) -> (anyhow::Result<T>, Duration) {
        let start = Instant::now();
        let out = f();
        (out, start.elapsed())
    }

    let (obs, cards, invalid) = std::thread::scope(|scope| {
        let obs = scope.spawn(|| timed(|| tenant.get_observations_batch(&observation_keys)));
        let cards =
            scope.spawn(|| timed(|| tenant.get_memory_cards_batch(&observation_memory_ids)));
        let invalid = scope.spawn(|| {
            timed(|| match pit {
                Some(pit_ms) => tenant.invalidated_set_at_time(pit_ms, &observation_memory_ids),
                None => tenant.invalidated_set(&observation_memory_ids),
            })
        });
        fn join<T>(name: &'static str, r: std::thread::Result<T>) -> Result<T, StatusCode> {
            r.map_err(|_| {
                tracing::error!(stage = name, "hydrate thread panicked");
                StatusCode::INTERNAL_SERVER_ERROR
            })
        }
        Ok::<_, StatusCode>((
            join("observations", obs.join())?,
            join("cards", cards.join())?,
            join("invalidated", invalid.join())?,
        ))
    })?;

    let fail = |name: &'static str| {
        move |err: anyhow::Error| {
            tracing::error!(stage = name, error = ?err, "hydrate read failed");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    };
    let us = |d: Duration| (d.as_millis() as u64, d.as_micros() as u64);
    s.observations = obs.0.map_err(fail("observations"))?;
    (s.diag.fetch_obs_ms, s.diag.fetch_obs_us) = us(obs.1);
    s.memory_cards = cards.0.map_err(fail("cards"))?;
    (s.diag.fetch_cards_ms, s.diag.fetch_cards_us) = us(cards.1);
    s.invalidated_facts = invalid.0.map_err(fail("invalidated"))?;
    (s.diag.fetch_invalid_ms, s.diag.fetch_invalid_us) = us(invalid.1);

    // Graph, link and edge lanes traverse tenant-wide structures, so they can
    // surface another entity's memories. Enforce the requested scope once,
    // here, where every candidate's owner is known.
    if let Some(scope) = s.payload.entity_id.clone() {
        let (observations, cards) = (&s.data.observations, &s.data.memory_cards);
        let in_scope = |mid: &String| {
            observations.get(mid).map_or(true, |o| o.entity_id == scope)
                && cards.get(mid).map_or(true, |c| c.entity_id == scope)
        };
        s.data.fused.retain(|(mid, _, _)| in_scope(mid));
        let kept: HashSet<String> = s.data.fused.iter().map(|(mid, _, _)| mid.clone()).collect();
        s.data.observations.retain(|mid, _| kept.contains(mid));
    }
    Ok(())
}

fn score_loop(s: &mut QueryPipelineState) -> Vec<EvidenceCard> {
    let loop_start = Instant::now();
    let mut scored = Vec::new();
    let primary_qembed = &s.primary_qembed;
    let graph_scores = &s.graph_scores;
    let memory_cards = &s.memory_cards;
    let observations = &s.observations;
    let invalidated_facts = &s.invalidated_facts;
    let plan = &s.plan;
    let plan_intent = s.plan.intent;
    let query_text = &s.query_text;
    let now_ms = s.now_ms;
    let adaptive_profile = &s.adaptive_profile;
    let session_route_scores = &s.session_route_scores;

    for (mid, ts, rrf_score) in &s.fused {
        if is_synthetic_query_memory(mid) {
            continue;
        }
        let Some(obs) = observations.get(mid) else {
            continue;
        };
        let is_stale_fact = (obs.kind == MemoryKind::Fact
            || obs.kind == MemoryKind::Preference
            || obs.kind == MemoryKind::Decision)
            && invalidated_facts.contains(mid);
        let created_at_ms = if obs.created_at_ms > 0 { obs.created_at_ms } else { *ts };
        if let Some(pit) = s.payload.point_in_time_ms {
            if created_at_ms > pit {
                continue;
            }
        }
        let scorable = ScorableObservation::new(&obs.textual_content);
        let entity_hits = entity_hit_count(&scorable, plan);
        let lexical_hits = lexical_hit_count(&scorable, plan);
        let temporal_hits = temporal_hit_count(&scorable, plan);
        let facet_mask = facet_match_mask(&scorable, plan);
        // Cross-encoder scores are unbounded logits; they enter through their
        // own lane in the rank fusion above. Using them directly here put
        // reranked candidates on a different scale from everything else.
        let mut base_score = *rrf_score;
        if base_score <= 0.001
            && !primary_qembed.is_empty()
            && obs.embedding.len() == primary_qembed.len()
        {
            base_score = cosine_similarity(primary_qembed, &obs.embedding).max(0.0);
        }
        let lifecycle =
            memory_cards.get(mid).and_then(|card| card.lifecycle.as_ref()).cloned().unwrap_or_else(
                || {
                    let mut lifecycle = crate::lifecycle::evaluate_lifecycle(
                        &obs.textual_content,
                        obs.kind,
                        created_at_ms,
                        None,
                        false,
                    );
                    // No stored lifecycle means no known storage time; never
                    // expire on the event timestamp alone.
                    lifecycle.expires_at_ms = None;
                    lifecycle
                },
            );
        let Some(lifecycle_adjustment) = lifecycle_rank_adjustment(&lifecycle, obs.kind, now_ms)
        else {
            continue;
        };
        let mut fs = apply_decay_with_policy(base_score, created_at_ms, obs.kind, now_ms);
        fs += lifecycle_adjustment;
        let superseded_card = memory_cards.get(mid).is_some_and(|card| !card.is_latest);
        if is_stale_fact || (plan.prefers_latest && superseded_card) {
            fs *= s.weights.stale_fact_decay;
        }
        fs -= attractor_negative_penalty(
            &scorable,
            plan,
            query_text,
            entity_hits,
            lexical_hits,
            temporal_hits,
            facet_mask,
        );
        fs += kind_query_bonus(obs.kind, plan, &scorable);
        fs += lexical_overlap_bonus(&scorable, plan);
        fs += entity_coverage_bonus(&scorable, plan);
        fs += numeric_signal_bonus(&obs.textual_content, &scorable.lower, plan_intent);
        fs += ordinal_signal_bonus(obs.kind, &scorable, plan);

        let graph_score = graph_scores.get(mid).copied().unwrap_or(0.0);
        let temporal_adjust = if temporal_recency_scoring_enabled() {
            temporal_consistency_adjustment(obs.kind, created_at_ms, now_ms, plan_intent)
        } else {
            0.0
        };
        let confidence_signal: f32 = if is_stale_fact {
            s.weights.rerank_stale_penalty
        } else if lifecycle.stability_score > 0.7 {
            s.weights.rerank_confidence_stable
        } else if lifecycle.confidence_score > 0.7 {
            s.weights.rerank_confidence_high
        } else {
            0.0
        };

        let weights = FourSignalWeights::for_intent(plan_intent);
        let semantic_signal = base_score.max(0.0);
        let temporal_signal = temporal_adjust.max(0.0);
        let reweighted = fuse_four_signals(
            semantic_signal,
            temporal_signal,
            confidence_signal.max(0.0),
            graph_score.max(0.0),
            &weights,
        );
        fs = fs * (1.0 - s.weights.four_signal_temporal_weight)
            + reweighted * s.weights.four_signal_temporal_weight;
        if adaptive_profile.route_strength > 0.0 {
            let routed_sid = memory_cards
                .get(mid)
                .map(|card| card.source_session_id.clone())
                .or_else(|| Some(obs.session_id.clone()).filter(|s| !s.is_empty()));
            if let Some(sid) = routed_sid {
                if adaptive_profile.route_sessions.contains(&sid) {
                    let route_score = session_route_scores.get(&sid).copied().unwrap_or(0.0);
                    fs += if plan.needs_decomposition || plan.cross_entity {
                        s.weights.route_boost_hard
                    } else {
                        s.weights.route_boost_simple
                    };
                    fs += route_score.min(0.35) * 0.18;
                } else if !(plan.needs_decomposition || plan.cross_entity) {
                    fs += s.weights.route_penalty;
                }
            }
        }
        let (source_memory_id, source_session_id) = if let Some(card) = memory_cards.get(mid) {
            if card.source_memory_id != *mid {
                (card.source_memory_id.clone(), card.source_session_id.clone())
            } else {
                (mid.clone(), obs.session_id.clone())
            }
        } else {
            (mid.clone(), obs.session_id.clone())
        };

        scored.push(EvidenceCard {
            claim_text: obs.textual_content.clone(),
            source_memory_id,
            source_session_id,
            card_id: if memory_cards.contains_key(mid) { Some(mid.clone()) } else { None },
            semantic_rank: None,
            semantic_score: base_score,
            bm25_rank: None,
            bm25_score: 0.0,
            session_router_rank: None,
            session_router_score: 0.0,
            card_score: 0.0,
            reranker_score: base_score,
            entity_hits,
            lexical_hits,
            temporal_hits,
            facet_mask,
            graph_score,
            child_score: 0.0,
            is_latest: false,
            card_type: format!("{:?}", obs.kind),
            final_score: fs,
            inference_notes: None,
            internal_kind: obs.kind,
            created_at_ms,
            entity_id: obs.entity_id.clone(),
            source_turn_index: obs.turn_index as usize,
        });
    }
    if plan.prefers_latest {
        apply_latest_preference(&mut scored);
    }
    (s.diag.scoring_loop_ms, s.diag.scoring_loop_us) = elapsed_ms_and_us(loop_start);
    scored
}

/// For current-value questions, adds a bonus that grows with how recent a
/// candidate is relative to the other candidates, scaled by the score spread
/// and by the candidate's own (squared) relative score, so newer versions of
/// relevant facts win without promoting recent unrelated memories
/// (`TELLODB_LATEST_RECENCY_WEIGHT`, default 0.35).
fn apply_latest_preference(scored: &mut [EvidenceCard]) {
    static WEIGHT: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    let weight = *WEIGHT.get_or_init(|| {
        std::env::var("TELLODB_LATEST_RECENCY_WEIGHT")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|w| w.is_finite() && *w >= 0.0)
            .unwrap_or(0.35)
    });
    let (Some(oldest), Some(newest)) = (
        scored.iter().map(|c| c.created_at_ms).min(),
        scored.iter().map(|c| c.created_at_ms).max(),
    ) else {
        return;
    };
    if newest == oldest {
        return;
    }
    let (lo, hi) = scored
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), c| (lo.min(c.final_score), hi.max(c.final_score)));
    let spread = (hi - lo).max(f32::EPSILON);
    for card in scored.iter_mut() {
        let recency = (card.created_at_ms - oldest) as f32 / (newest - oldest) as f32;
        // Gate by relevance: recency decides between relevant versions of a
        // fact; it must not lift recent but unrelated memories above them.
        let relevance = ((card.final_score - lo) / spread).clamp(0.0, 1.0);
        card.final_score += weight * spread * recency * relevance * relevance;
    }
}

/// Explains why a retrieved fact is no longer current: what replaced it,
/// when, and which memories state each value. `None` while it is current.
fn describe_stale_fact(
    version: &crate::storage::FactVersionRow,
) -> Option<crate::api::types::WhyStale> {
    if version.is_current {
        return None;
    }
    let superseded_by = version.superseded_by.clone()?;
    Some(crate::api::types::WhyStale {
        fact_key: version.fact_key.clone(),
        stale_value: version.object.clone(),
        current_value: version.current_object.clone(),
        superseded_by,
        superseded_at_ms: version.superseded_at_ms.or(version.valid_to_ms),
        valid_from_ms: version.valid_from_ms,
        valid_to_ms: version.valid_to_ms,
        evidence: version.evidence.clone(),
    })
}

fn score_build_response(
    s: &mut QueryPipelineState,
    mut evidence_cards: Vec<EvidenceCard>,
) -> Result<Vec<QueryResult>, StatusCode> {
    let stage_start = Instant::now();

    // Pre-synthesized Phase 1: direct fact lookup.
    // When the planner inferred a `fact_key` (e.g., "relationship_status",
    // "purchase", "favorite_team") and we have an entity scope, attempt a
    // deterministic lookup against the fact_versions table and inject the
    // answer as a high-priority synthetic EvidenceCard so the reader LLM
    // receives the fact verbatim at the top of its context.
    if let (Some(ref fact_key), Some(ref entity_id)) =
        (s.plan.fact_key.as_ref(), s.payload.entity_id.as_ref())
    {
        let fact_value = if features::enabled(Feature::Facts) {
            s.tenant.get_current_fact_value(entity_id, fact_key)
        } else {
            Ok(None)
        };
        if let Ok(Some(fact_value)) = fact_value {
            let synthetic_score = 1.0e9_f32;
            let now_ms = s.now_ms;
            let synthetic_id = format!("__pre_synth_fact::{}::{}", entity_id, fact_key);
            evidence_cards.push(EvidenceCard {
                claim_text: format!("{}: {}", fact_key.replace('_', " "), fact_value),
                source_memory_id: synthetic_id.clone(),
                source_session_id: String::new(),
                card_id: Some(synthetic_id),
                semantic_rank: None,
                semantic_score: synthetic_score,
                bm25_rank: None,
                bm25_score: 0.0,
                session_router_rank: None,
                session_router_score: 0.0,
                card_score: 0.0,
                reranker_score: synthetic_score,
                entity_hits: 0,
                lexical_hits: 0,
                temporal_hits: 0,
                facet_mask: 0,
                graph_score: 0.0,
                child_score: 0.0,
                is_latest: true,
                card_type: "PreSynthesizedFact".to_string(),
                final_score: synthetic_score,
                inference_notes: None,
                internal_kind: MemoryKind::Fact,
                created_at_ms: now_ms,
                entity_id: entity_id.to_string(),
                source_turn_index: 0,
            });
            tracing::debug!(
                entity_id = %entity_id,
                fact_key = %fact_key,
                "pre-synthesized fact lookup injected"
            );
        }
    }

    evidence_cards.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.source_memory_id.cmp(&b.source_memory_id))
    });

    // Ambiguity packet: when the top two candidates are nearly tied (e.g.
    // "James" vs "John"), surface the close runner-up to the reader LLM via
    // the top card's `inference_notes` so the model can disambiguate or
    // ask the user. The threshold is loaded from `ranking_config.json` and
    // propagated into `s.weights.ambiguity_delta_threshold` at construction.
    if evidence_cards.len() >= 2 {
        let top_score = evidence_cards[0].final_score;
        let second_score = evidence_cards[1].final_score;
        let delta = (top_score - second_score).abs();
        if delta < s.weights.ambiguity_delta_threshold
            && !evidence_cards[0].source_memory_id.starts_with("__pre_synth_")
        {
            let note = format!(
                "AmbiguityPacket: top-2 candidates are within {:.3} of each other ({} vs {}); consider asking the user to disambiguate.",
                delta,
                evidence_cards[0].source_memory_id,
                evidence_cards[1].source_memory_id
            );
            if let Some(card) = evidence_cards.get_mut(0) {
                if let Some(notes) = card.inference_notes.as_mut() {
                    notes.push(note);
                } else {
                    card.inference_notes = Some(vec![note]);
                }
            }
        }
    }

    // Pre-synthesized fact cards are extra context, not retrieved memories,
    // so they must not take slots from the requested `limit`.
    let synthetic_cards =
        evidence_cards.iter().filter(|c| c.source_memory_id.starts_with("__pre_synth_")).count();
    let selected = select_candidates_with_session_head(
        evidence_cards,
        s.limit + synthetic_cards,
        &s.plan,
        s.plan.prefer_distilled,
        s.plan.prefer_episodic,
    );

    let mut source_keys = Vec::new();
    for card in &selected {
        source_keys.push((card.created_at_ms, card.source_memory_id.clone()));
    }
    let hydrate_obs_start = Instant::now();
    let read_failed = |stage: &'static str| {
        move |err: anyhow::Error| {
            tracing::error!(stage, error = ?err, "response hydration failed");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    };
    let source_observations =
        s.tenant.get_observations_batch(&source_keys).map_err(read_failed("observations"))?;

    let mut fact_memory_ids = Vec::new();
    let mut card_ids = Vec::new();
    for card in &selected {
        if card.internal_kind == crate::storage::MemoryKind::Fact {
            fact_memory_ids.push(card.source_memory_id.clone());
        }
        if let Some(ref cid) = card.card_id {
            card_ids.push(cid.clone());
        }
    }

    let factver_start = Instant::now();
    let fact_versions = s
        .tenant
        .fact_versions_for_memories(&fact_memory_ids)
        .map_err(read_failed("fact_versions"))?;
    s.diag.factver_us = factver_start.elapsed().as_micros() as u64;
    let cards_start = Instant::now();
    let memory_cards =
        s.tenant.get_memory_cards_batch(&card_ids).map_err(read_failed("memory_cards"))?;
    s.diag.build_cards_us = cards_start.elapsed().as_micros() as u64;

    (s.diag.hydrate_obs_ms, s.diag.hydrate_obs_us) = elapsed_ms_and_us(hydrate_obs_start);

    let proof_us = std::sync::atomic::AtomicU64::new(0);
    let mut queries: Vec<QueryResult> = selected
        .into_iter()
        .map(|card| {
            let text = if let Some(source_obs) = source_observations.get(&card.source_memory_id) {
                source_obs.textual_content.clone()
            } else {
                card.claim_text.clone()
            };
            let evidence = if s.include_evidence && s.proof_mode != "off" {
                let proof_start = Instant::now();
                let packet = Some(build_proof_packet(
                    &s.tenant,
                    &s.query_text,
                    &s.plan,
                    &card,
                    &s.proof_mode,
                    s.verify_evidence,
                    s.evidence_radius,
                ));
                proof_us.fetch_add(
                    proof_start.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                packet
            } else {
                None
            };
            let mut fact_key = None;
            let mut superseded_by = None;
            let mut why_stale = None;
            // Facts are registered against derived records, so a turn is
            // matched through them (see `fact_versions_for_memories`).
            if let Some(version) = fact_versions.get(&card.source_memory_id) {
                fact_key = Some(version.fact_key.clone());
                superseded_by = version.superseded_by.clone();
                why_stale = describe_stale_fact(version);
            }

            let mut stability_score = None;
            if let Some(ref cid) = card.card_id {
                if let Some(mc) = memory_cards.get(cid) {
                    if let Some(ref lc) = mc.lifecycle {
                        stability_score = Some(lc.stability_score);
                    }
                }
            }

            QueryResult {
                memory_id: card.source_memory_id.clone(),
                entity_id: card.entity_id,
                session_id: card.source_session_id,
                turn_index: card.source_turn_index,
                created_at_ms: card.created_at_ms,
                similarity: card.final_score,
                textual_content: text,
                evidence,
                inference_notes: None,
                fact_key,
                conflict_flag: Some(!card.is_latest),
                superseded_by,
                why_stale,
                stability_score,
            }
        })
        .collect();
    s.diag.proof_us = proof_us.load(std::sync::atomic::Ordering::Relaxed);
    let confidence_start = Instant::now();
    let evidence_conf =
        compute_evidence_confidence(&queries, &s.query_text, s.state.intent_classifier.as_deref());
    s.diag.confidence_us = confidence_start.elapsed().as_micros() as u64;
    s.diag.evidence_confidence_bp = (evidence_conf * 10_000.0) as u64;
    s.diag.abstain_recommended = evidence_conf < 0.24 && !queries.is_empty();
    (s.diag.session_ms, s.diag.session_us) = elapsed_ms_and_us(stage_start);
    (s.diag.total_ms, s.diag.total_us) = elapsed_ms_and_us(s.total_start);

    // Pre-synthesized Phase 2: memory card as answer.
    // If the top-ranked result is backed by a latest, high-confidence memory
    // card, surface its claim_text as a synthetic answer row at position 0
    // so the reader LLM receives the distilled claim verbatim.
    if let Some(top) = queries.first() {
        if !top.memory_id.starts_with("__pre_synth_") {
            if let Ok(Some(card)) = s.tenant.get_memory_card_by_source(&top.memory_id) {
                if card.is_latest && card.confidence >= 0.70 {
                    // Dated like the memory it restates, so recency ordering holds.
                    let source_created_at_ms = top.created_at_ms;
                    let synthetic = QueryResult {
                        memory_id: format!("__pre_synth_card::{}", card.card_id),
                        entity_id: card.entity_id.clone(),
                        session_id: card.source_session_id.clone(),
                        turn_index: top.turn_index,
                        created_at_ms: source_created_at_ms,
                        similarity: 1.0,
                        textual_content: format!("{}: {}", card.subject, card.object),
                        evidence: None,
                        inference_notes: Some(vec![format!(
                            "Pre-synthesized from memory card {} (confidence {:.2})",
                            card.card_id, card.confidence
                        )]),
                        fact_key: None,
                        conflict_flag: Some(false),
                        superseded_by: None,
                        why_stale: None,
                        stability_score: None,
                    };
                    queries.insert(0, synthetic);
                }
            }
        }
    }

    Ok(queries)
}

pub fn execute_query_pipeline(
    payload: QueryPayload,
    state: EngineState,
    tenant: std::sync::Arc<TenantStore>,
    limit: usize,
    enable_neural_rerank: bool,
) -> Result<(Vec<QueryResult>, QueryDiagnostics), StatusCode> {
    let mut s = QueryPipelineState::new(payload, state, tenant, limit, enable_neural_rerank);
    plan_phase(&mut s);
    route_phase(&mut s);
    retrieval_phase(&mut s);
    rerank_phase(&mut s);
    fusion_phase(&mut s);
    let results = score_phase(&mut s)?;
    Ok((results, std::mem::take(&mut s.data.diag)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rerank_policy_is_the_confidence_gate() {
        // The string heuristic reranked 93% of LongMemEval dev, because it
        // fires on " and ", "would", "might" or a long question. If this ever
        // reverts to `heuristic`, the cross-encoder silently becomes an
        // always-on 169 ms stage again.
        assert_eq!(rerank_policy_name(), "gate");
    }

    #[test]
    fn rerank_gate_flags_close_top_hits() {
        // Similarities 0.80 vs 0.79 at rank 5: 1.25% relative gap.
        let close = [(1, 0.20), (2, 0.205), (3, 0.207), (4, 0.208), (5, 0.21)];
        assert!(rerank_gate_uncertain(&close, 0.05));
        // 0.90 vs 0.60: clear winner.
        let clear = [(1, 0.10), (2, 0.30), (3, 0.35), (4, 0.38), (5, 0.40)];
        assert!(!rerank_gate_uncertain(&clear, 0.05));
        // Fewer than five hits compare against the last one.
        assert!(!rerank_gate_uncertain(&[(1, 0.1), (2, 0.5)], 0.05));
        assert!(!rerank_gate_uncertain(&[], 0.05));
        assert!(
            RerankDecision::GateUncertain.applies() && !RerankDecision::GateConfident.applies()
        );
    }

    #[test]
    fn retrieval_budget_table_matches_existing_values() {
        const EXPECTED: [[RetrievalBudget; 4]; 3] = [
            [
                RetrievalBudget {
                    semantic_top: 320,
                    fts_top: 80,
                    semantic_query_limit: 3,
                    fts_query_limit: 3,
                    session_router_limit: 12,
                    route_probe_query_limit: 1,
                    route_probe_hit_limit: 10,
                    route_take_simple: 6,
                    route_take_hard: 10,
                    card_limit: 84,
                },
                RetrievalBudget {
                    semantic_top: 280,
                    fts_top: 72,
                    semantic_query_limit: 1,
                    fts_query_limit: 1,
                    session_router_limit: 8,
                    route_probe_query_limit: 1,
                    route_probe_hit_limit: 8,
                    route_take_simple: 6,
                    route_take_hard: 8,
                    card_limit: 64,
                },
                RetrievalBudget {
                    semantic_top: 260,
                    fts_top: 64,
                    semantic_query_limit: 1,
                    fts_query_limit: 1,
                    session_router_limit: 8,
                    route_probe_query_limit: 0,
                    route_probe_hit_limit: 0,
                    route_take_simple: 6,
                    route_take_hard: 8,
                    card_limit: 56,
                },
                RetrievalBudget {
                    semantic_top: 240,
                    fts_top: 64,
                    semantic_query_limit: 1,
                    fts_query_limit: 1,
                    session_router_limit: 6,
                    route_probe_query_limit: 0,
                    route_probe_hit_limit: 0,
                    route_take_simple: 5,
                    route_take_hard: 7,
                    card_limit: 48,
                },
            ],
            [
                RetrievalBudget {
                    semantic_top: 420,
                    fts_top: 96,
                    semantic_query_limit: 3,
                    fts_query_limit: 3,
                    session_router_limit: 14,
                    route_probe_query_limit: 1,
                    route_probe_hit_limit: 12,
                    route_take_simple: 8,
                    route_take_hard: 12,
                    card_limit: 120,
                },
                RetrievalBudget {
                    semantic_top: 320,
                    fts_top: 80,
                    semantic_query_limit: 2,
                    fts_query_limit: 2,
                    session_router_limit: 10,
                    route_probe_query_limit: 1,
                    route_probe_hit_limit: 10,
                    route_take_simple: 8,
                    route_take_hard: 10,
                    card_limit: 88,
                },
                RetrievalBudget {
                    semantic_top: 300,
                    fts_top: 72,
                    semantic_query_limit: 2,
                    fts_query_limit: 2,
                    session_router_limit: 10,
                    route_probe_query_limit: 1,
                    route_probe_hit_limit: 10,
                    route_take_simple: 7,
                    route_take_hard: 9,
                    card_limit: 72,
                },
                RetrievalBudget {
                    semantic_top: 240,
                    fts_top: 64,
                    semantic_query_limit: 1,
                    fts_query_limit: 1,
                    session_router_limit: 8,
                    route_probe_query_limit: 0,
                    route_probe_hit_limit: 0,
                    route_take_simple: 6,
                    route_take_hard: 8,
                    card_limit: 56,
                },
            ],
            [
                RetrievalBudget {
                    semantic_top: 640,
                    fts_top: 160,
                    semantic_query_limit: 5,
                    fts_query_limit: 5,
                    session_router_limit: 24,
                    route_probe_query_limit: 4,
                    route_probe_hit_limit: 20,
                    route_take_simple: 10,
                    route_take_hard: 16,
                    card_limit: 180,
                },
                RetrievalBudget {
                    semantic_top: 420,
                    fts_top: 100,
                    semantic_query_limit: 3,
                    fts_query_limit: 3,
                    session_router_limit: 16,
                    route_probe_query_limit: 2,
                    route_probe_hit_limit: 16,
                    route_take_simple: 10,
                    route_take_hard: 14,
                    card_limit: 120,
                },
                RetrievalBudget {
                    semantic_top: 360,
                    fts_top: 84,
                    semantic_query_limit: 3,
                    fts_query_limit: 3,
                    session_router_limit: 14,
                    route_probe_query_limit: 2,
                    route_probe_hit_limit: 14,
                    route_take_simple: 9,
                    route_take_hard: 12,
                    card_limit: 96,
                },
                RetrievalBudget {
                    semantic_top: 240,
                    fts_top: 64,
                    semantic_query_limit: 2,
                    fts_query_limit: 2,
                    session_router_limit: 10,
                    route_probe_query_limit: 1,
                    route_probe_hit_limit: 12,
                    route_take_simple: 8,
                    route_take_hard: 10,
                    card_limit: 72,
                },
            ],
        ];
        let profiles =
            [RetrievalProfile::Fast, RetrievalProfile::Balanced, RetrievalProfile::Research];
        let shapes =
            [QueryShape::Hard, QueryShape::Temporal, QueryShape::Numeric, QueryShape::Simple];

        for (profile_index, profile) in profiles.into_iter().enumerate() {
            for (shape_index, shape) in shapes.into_iter().enumerate() {
                assert_eq!(
                    retrieval_budget_for_plan(&plan_for_shape(shape), profile),
                    EXPECTED[profile_index][shape_index],
                    "profile {profile_index}, shape {shape_index}"
                );
            }
        }
    }

    fn plan_for_shape(shape: QueryShape) -> QueryPlan {
        let (intent, needs_decomposition) = match shape {
            QueryShape::Hard => (QueryIntent::General, true),
            QueryShape::Temporal => (QueryIntent::TemporalAggregation, false),
            QueryShape::Numeric => (QueryIntent::NumericAggregation, false),
            QueryShape::Simple => (QueryIntent::General, false),
        };
        QueryPlan {
            semantic_queries: Vec::new(),
            fts_queries: Vec::new(),
            coverage_facets: Vec::new(),
            requirements: Vec::new(),
            prefer_distilled: false,
            prefer_episodic: false,
            temporal_terms: Vec::new(),
            lexical_terms: Vec::new(),
            intent,
            subject_entities: Vec::new(),
            cross_entity: false,
            needs_decomposition,
            coverage_mode: false,
            ordinal_rank: None,
            fact_key: None,
            prefers_latest: false,
        }
    }

    #[test]
    fn core_profile_hides_superseded_and_future_facts() {
        let temp = tempfile::tempdir().unwrap();
        let tenant = TenantStore::new(&temp.path().join("tenant.db")).unwrap();
        let profile = serde_json::json!({
            "schema": "heuristic_core_profile_v1",
            "entity_id": "u",
            "facts": [
                {"memory_id": "m-austin", "timestamp_ms": 100, "text": "I live in Austin"},
                {"memory_id": "m-seattle", "timestamp_ms": 200, "text": "I moved to Seattle"},
                {"memory_id": "m-dog", "timestamp_ms": 150, "text": "My dog is Biscuit"}
            ]
        });
        tenant.set_core_profile("u", &profile.to_string()).unwrap();
        tenant
            .register_fact_versions_batch(
                "u",
                &[
                    ("residence", 100, "m-austin", "u", "lives_in", "Austin"),
                    ("residence", 200, "m-seattle", "u", "lives_in", "Seattle"),
                ],
            )
            .unwrap();

        let texts = |pit: Option<u64>| -> Vec<String> {
            let raw = current_core_profile(&tenant, "u", pit).unwrap().unwrap();
            let value: serde_json::Value = serde_json::from_str(&raw).unwrap();
            value["facts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| f["text"].as_str().unwrap().to_string())
                .collect()
        };
        // Now: Austin was superseded; newest first.
        assert_eq!(texts(None), vec!["I moved to Seattle", "My dog is Biscuit"]);
        // As of t=120: Austin is valid, later facts are not yet known.
        assert_eq!(texts(Some(120)), vec!["I live in Austin"]);
    }
}
