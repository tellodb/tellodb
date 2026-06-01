use axum::{
    extract::{Json, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::api::auth::{
    authorize_request, principal_namespace_prefix, principal_user_id, record_usage_for_principal,
    scope_entity_id,
};
use crate::api::planner::*;
use crate::api::types::RankedItem;
use crate::api::types::{
    AnalyticsQueryPayload, AnalyticsQueryResult, BucketedResult, EvidenceCard, GraphExportPayload,
    GraphQueryPayload, GraphWalkPayload, ProofCheck, ProofPacket, ProofTurn, QueryPayload,
    QueryResult,
};
use crate::api::utils::{
    apply_decay_with_policy, clip_profile_to_budget, cosine_similarity, elapsed_ms_and_us,
    env_bool, extract_named_phrases, insert_f32_header, insert_stage_timing_headers,
    insert_u64_header, parse_temporal_window, scoped_semantic_min_hits, scoped_semantic_start,
    scoped_semantic_step, scoped_semantic_top, session_id_from_memory_id,
    should_apply_neural_rerank, temporal_recency_scoring_enabled, turn_index_from_memory_id,
    SEMANTIC_TOP_DEFAULT,
};
use crate::api::{EngineState, PlatformWriteOp};
use crate::retrieval::ScoringWeights;
use crate::metrics;
use crate::storage::{AgentObservation, MemoryCard, MemoryCardSearchInput, MemoryKind, QueryTrace, SessionCandidateTrace, TenantStore};

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
    session_ms: u64,
    session_us: u64,
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
    routed_sessions: u64,
    memory_card_hits: u64,
    temporal_event_hits: u64,
    shadow_question_hits: u64,
    facet_posting_hits: u64,
    mem_scene_hits: u64,
    evidence_confidence_bp: u64,
    abstain_recommended: bool,
}

pub async fn query_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(payload): Json<QueryPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
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

    let (mut results, diagnostics) = {
        let state_for_query = state.clone();
        let tenant = tenant.clone();
        tokio::task::spawn_blocking(move || {
            execute_query_pipeline(
                payload,
                state_for_query,
                tenant,
                limit,
                enable_neural_rerank,
            )
        })
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??
    };

    let obs_block = if let Some(eid) = entity_id_for_core_profile.as_deref() {
        let _now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let profile = tenant
            .get_core_profile(eid)
            .ok()
            .flatten()
            .map(|p| clip_profile_to_budget(&p, 8));

        let mut scenes = Vec::new();
        let profile_query_lines: Vec<String> = vec![profile_query_text.clone()];
        for entity in extract_named_phrases(&profile_query_lines) {
            if let Ok(lines) = tenant.graph_edge_summaries_for_label(eid, &entity, 5) {
                if lines.is_empty() {
                    continue;
                }
                let mut sorted = lines.clone();
                sorted.sort();
                let scene = lines.join("; ");
                scenes.push(scene);
            }
        }
        let top_chunk_texts: Vec<String> = results
            .iter()
            .take(5)
            .map(|r| r.textual_content.clone())
            .collect();
        Some((
            eid.to_string(),
            build_observation_block(profile.as_deref(), &scenes, &top_chunk_texts),
        ))
    } else {
        None
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
            },
        );
    }

    let mut h = HeaderMap::new();
    insert_stage_timing_headers(
        &mut h,
        "x-tm-route",
        diagnostics.route_ms,
        diagnostics.route_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-planning",
        diagnostics.planning_ms,
        diagnostics.planning_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-embed",
        diagnostics.embed_ms,
        diagnostics.embed_us,
    );
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
    insert_stage_timing_headers(
        &mut h,
        "x-tm-card",
        diagnostics.card_ms,
        diagnostics.card_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-fuse",
        diagnostics.fuse_ms,
        diagnostics.fuse_us,
    );
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
    insert_stage_timing_headers(
        &mut h,
        "x-tm-session",
        diagnostics.session_ms,
        diagnostics.session_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-total",
        diagnostics.total_ms,
        diagnostics.total_us,
    );
    insert_stage_timing_headers(
        &mut h,
        "x-tm-trace",
        diagnostics.trace_ms,
        diagnostics.trace_us,
    );
    insert_u64_header(&mut h, "x-tm-scoped-ann-top", diagnostics.scoped_ann_top);
    insert_u64_header(
        &mut h,
        "x-tm-scoped-ann-attempts",
        diagnostics.scoped_ann_attempts,
    );
    insert_u64_header(
        &mut h,
        "x-tm-scoped-primary-hits",
        diagnostics.scoped_primary_hits,
    );
    insert_u64_header(&mut h, "x-tm-routed-sessions", diagnostics.routed_sessions);
    insert_u64_header(
        &mut h,
        "x-tm-memory-card-hits",
        diagnostics.memory_card_hits,
    );
    insert_u64_header(
        &mut h,
        "x-tm-temporal-event-hits",
        diagnostics.temporal_event_hits,
    );
    insert_u64_header(
        &mut h,
        "x-tm-shadow-question-hits",
        diagnostics.shadow_question_hits,
    );
    insert_u64_header(
        &mut h,
        "x-tm-facet-posting-hits",
        diagnostics.facet_posting_hits,
    );
    insert_u64_header(&mut h, "x-tm-mem-scene-hits", diagnostics.mem_scene_hits);
    insert_f32_header(
        &mut h,
        "x-tm-evidence-confidence",
        diagnostics.evidence_confidence_bp as f32 / 10_000.0,
    );
    h.insert(
        "x-tm-abstain-recommended",
        HeaderValue::from_static(if diagnostics.abstain_recommended {
            "1"
        } else {
            "0"
        }),
    );

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
    turn_index_from_memory_id(memory_id) >= 3_000_000
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
    if let Some(pos) = queries
        .iter()
        .position(|query| query.to_ascii_lowercase() == lower)
    {
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
    if lifecycle
        .expires_at_ms
        .map(|expires_at| expires_at <= now_ms)
        .unwrap_or(false)
    {
        return None;
    }

    let mut adjustment = (lifecycle.admission_score - 0.50) * 0.06
        + (lifecycle.utility_score - 0.50) * 0.035
        + (lifecycle.confidence_score - 0.50) * 0.035
        + (lifecycle.specificity_score - 0.45) * 0.025;

    if lifecycle.promote_to_profile
        || matches!(
            kind,
            MemoryKind::Fact | MemoryKind::Preference | MemoryKind::Decision
        )
    {
        adjustment += 0.025;
    }
    adjustment += match lifecycle.retention_class {
        crate::lifecycle::RetentionClass::LongTerm => 0.025,
        crate::lifecycle::RetentionClass::Archive => 0.012,
        crate::lifecycle::RetentionClass::Episodic => 0.006,
        crate::lifecycle::RetentionClass::Working => -0.006,
        crate::lifecycle::RetentionClass::Ephemeral => -0.035,
        crate::lifecycle::RetentionClass::ComplianceSensitive => -0.025,
    };
    adjustment += match lifecycle.sensitivity_class {
        crate::lifecycle::SensitivityClass::Public => 0.0,
        crate::lifecycle::SensitivityClass::Personal => -0.004,
        crate::lifecycle::SensitivityClass::Sensitive => -0.014,
        crate::lifecycle::SensitivityClass::Restricted => -0.035,
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
    let facet_deficit = if plan.coverage_mode && facet_mask.count_ones() == 0 {
        1.0
    } else {
        0.0
    };
    let temporal_deficit = if !plan.temporal_terms.is_empty() && temporal_hits == 0 {
        1.0
    } else {
        0.0
    };
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
        let center = turn_index_from_memory_id(&card.source_memory_id) as u32;
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
            turn_index: turn_index_from_memory_id(&card.source_memory_id) as u32,
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
        checks
            .iter()
            .filter(|check| check.name != "facet_coverage")
            .all(|check| check.passed)
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
        source_turn_index: turn_index_from_memory_id(&card.source_memory_id),
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

#[derive(Clone, Copy)]
struct RetrievalBudget {
    semantic_top: usize,
    fts_top: usize,
    semantic_query_limit: usize,
    fts_query_limit: usize,
    session_router_limit: usize,
    event_limit: usize,
    shadow_limit: usize,
    facet_limit: usize,
    scene_limit: usize,
    session_ann_limit: usize,
    event_vector_limit: usize,
    shadow_vector_limit: usize,
    route_probe_query_limit: usize,
    route_probe_hit_limit: usize,
    route_take_simple: usize,
    route_take_hard: usize,
    card_limit: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RetrievalProfile {
    Fast,
    Balanced,
    Research,
}

fn retrieval_profile() -> RetrievalProfile {
    match std::env::var("TEMPORAL_MEMORY_RETRIEVAL_PROFILE")
        .unwrap_or_else(|_| "fast".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "research" | "full" | "v2" => RetrievalProfile::Research,
        "balanced" | "default" => RetrievalProfile::Balanced,
        _ => RetrievalProfile::Fast,
    }
}

fn auto_rerank_enabled(profile: RetrievalProfile) -> bool {
    env_bool(
        "TEMPORAL_MEMORY_AUTO_RERANK",
        matches!(profile, RetrievalProfile::Research),
    )
}

fn retrieval_budget_for_plan(plan: &QueryPlan, profile: RetrievalProfile) -> RetrievalBudget {
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

    if matches!(profile, RetrievalProfile::Fast) {
        return if hard {
            RetrievalBudget {
                semantic_top: 320,
                fts_top: 80,
                semantic_query_limit: 3,
                fts_query_limit: 3,
                session_router_limit: 12,
                event_limit: if temporal { 48 } else { 8 },
                shadow_limit: 8,
                facet_limit: 24,
                scene_limit: 0,
                session_ann_limit: 0,
                event_vector_limit: 0,
                shadow_vector_limit: 0,
                route_probe_query_limit: 1,
                route_probe_hit_limit: 10,
                route_take_simple: 6,
                route_take_hard: 10,
                card_limit: 84,
            }
        } else if temporal {
            RetrievalBudget {
                semantic_top: 280,
                fts_top: 72,
                semantic_query_limit: 1,
                fts_query_limit: 1,
                session_router_limit: 8,
                event_limit: 48,
                shadow_limit: 0,
                facet_limit: 12,
                scene_limit: 0,
                session_ann_limit: 0,
                event_vector_limit: 0,
                shadow_vector_limit: 0,
                route_probe_query_limit: 1,
                route_probe_hit_limit: 8,
                route_take_simple: 6,
                route_take_hard: 8,
                card_limit: 64,
            }
        } else if numeric {
            RetrievalBudget {
                semantic_top: 260,
                fts_top: 64,
                semantic_query_limit: 1,
                fts_query_limit: 1,
                session_router_limit: 8,
                event_limit: 0,
                shadow_limit: 0,
                facet_limit: 0,
                scene_limit: 0,
                session_ann_limit: 0,
                event_vector_limit: 0,
                shadow_vector_limit: 0,
                route_probe_query_limit: 0,
                route_probe_hit_limit: 0,
                route_take_simple: 6,
                route_take_hard: 8,
                card_limit: 56,
            }
        } else {
            RetrievalBudget {
                semantic_top: 240,
                fts_top: 64,
                semantic_query_limit: 1,
                fts_query_limit: 1,
                session_router_limit: 6,
                event_limit: 0,
                shadow_limit: 0,
                facet_limit: 0,
                scene_limit: 0,
                session_ann_limit: 0,
                event_vector_limit: 0,
                shadow_vector_limit: 0,
                route_probe_query_limit: 0,
                route_probe_hit_limit: 0,
                route_take_simple: 5,
                route_take_hard: 7,
                card_limit: 48,
            }
        };
    }

    if matches!(profile, RetrievalProfile::Balanced) {
        return if hard {
            RetrievalBudget {
                semantic_top: 420,
                fts_top: 96,
                semantic_query_limit: 3,
                fts_query_limit: 3,
                session_router_limit: 14,
                event_limit: if temporal { 72 } else { 24 },
                shadow_limit: 32,
                facet_limit: 48,
                scene_limit: 16,
                session_ann_limit: 16,
                event_vector_limit: if temporal { 16 } else { 8 },
                shadow_vector_limit: 8,
                route_probe_query_limit: 1,
                route_probe_hit_limit: 12,
                route_take_simple: 8,
                route_take_hard: 12,
                card_limit: 120,
            }
        } else if temporal {
            RetrievalBudget {
                semantic_top: 320,
                fts_top: 80,
                semantic_query_limit: 2,
                fts_query_limit: 2,
                session_router_limit: 10,
                event_limit: 72,
                shadow_limit: 16,
                facet_limit: 24,
                scene_limit: 8,
                session_ann_limit: 8,
                event_vector_limit: 16,
                shadow_vector_limit: 0,
                route_probe_query_limit: 1,
                route_probe_hit_limit: 10,
                route_take_simple: 8,
                route_take_hard: 10,
                card_limit: 88,
            }
        } else if numeric {
            RetrievalBudget {
                semantic_top: 300,
                fts_top: 72,
                semantic_query_limit: 2,
                fts_query_limit: 2,
                session_router_limit: 10,
                event_limit: 16,
                shadow_limit: 16,
                facet_limit: 24,
                scene_limit: 0,
                session_ann_limit: 8,
                event_vector_limit: 0,
                shadow_vector_limit: 0,
                route_probe_query_limit: 1,
                route_probe_hit_limit: 10,
                route_take_simple: 7,
                route_take_hard: 9,
                card_limit: 72,
            }
        } else {
            RetrievalBudget {
                semantic_top: 240,
                fts_top: 64,
                semantic_query_limit: 1,
                fts_query_limit: 1,
                session_router_limit: 8,
                event_limit: 0,
                shadow_limit: 8,
                facet_limit: 16,
                scene_limit: 0,
                session_ann_limit: 0,
                event_vector_limit: 0,
                shadow_vector_limit: 0,
                route_probe_query_limit: 0,
                route_probe_hit_limit: 0,
                route_take_simple: 6,
                route_take_hard: 8,
                card_limit: 56,
            }
        };
    }

    if hard {
        RetrievalBudget {
            semantic_top: 640,
            fts_top: 160,
            semantic_query_limit: 5,
            fts_query_limit: 5,
            session_router_limit: 24,
            event_limit: if temporal { 96 } else { 48 },
            shadow_limit: 72,
            facet_limit: 96,
            scene_limit: 32,
            session_ann_limit: 32,
            event_vector_limit: if temporal { 32 } else { 16 },
            shadow_vector_limit: 32,
            route_probe_query_limit: 4,
            route_probe_hit_limit: 20,
            route_take_simple: 10,
            route_take_hard: 16,
            card_limit: 180,
        }
    } else if temporal {
        RetrievalBudget {
            semantic_top: 420,
            fts_top: 100,
            semantic_query_limit: 3,
            fts_query_limit: 3,
            session_router_limit: 16,
            event_limit: 96,
            shadow_limit: 32,
            facet_limit: 48,
            scene_limit: 16,
            session_ann_limit: 24,
            event_vector_limit: 32,
            shadow_vector_limit: 8,
            route_probe_query_limit: 2,
            route_probe_hit_limit: 16,
            route_take_simple: 10,
            route_take_hard: 14,
            card_limit: 120,
        }
    } else if numeric {
        RetrievalBudget {
            semantic_top: 360,
            fts_top: 84,
            semantic_query_limit: 3,
            fts_query_limit: 3,
            session_router_limit: 14,
            event_limit: 24,
            shadow_limit: 24,
            facet_limit: 40,
            scene_limit: 0,
            session_ann_limit: 16,
            event_vector_limit: 8,
            shadow_vector_limit: 8,
            route_probe_query_limit: 2,
            route_probe_hit_limit: 14,
            route_take_simple: 9,
            route_take_hard: 12,
            card_limit: 96,
        }
    } else {
        RetrievalBudget {
            semantic_top: 240,
            fts_top: 64,
            semantic_query_limit: 2,
            fts_query_limit: 2,
            session_router_limit: 10,
            event_limit: 0,
            shadow_limit: 16,
            facet_limit: 32,
            scene_limit: 0,
            session_ann_limit: 0,
            event_vector_limit: 0,
            shadow_vector_limit: 0,
            route_probe_query_limit: 1,
            route_probe_hit_limit: 12,
            route_take_simple: 8,
            route_take_hard: 10,
            card_limit: 72,
        }
    }
}

fn collect_link_cluster_scores(
    tenant: &TenantStore,
    seed_memory_id: &str,
    max_depth: usize,
) -> HashMap<String, f32> {
    let mut accumulated = HashMap::new();
    let mut visited = HashSet::new();
    let mut frontier = vec![(seed_memory_id.to_string(), 0usize, 1.0f32)];

    while let Some((current_mid, depth, path_weight)) = frontier.pop() {
        if depth >= max_depth {
            continue;
        }
        let visit_key = format!("{current_mid}:{depth}");
        if !visited.insert(visit_key) {
            continue;
        }
        let Ok(linked_memories) = tenant.get_linked_memories(&current_mid) else {
            continue;
        };
        for linked_mid in linked_memories {
            *accumulated.entry(linked_mid.clone()).or_insert(0.0) += path_weight;
            frontier.push((linked_mid, depth + 1, path_weight * 0.6));
        }
    }

    accumulated
}

fn parse_graph_direction_str(direction: Option<&str>) -> &'static str {
    match direction.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("in" | "inbound") => "Inbound",
        Some("both") => "Both",
        _ => "Outbound",
    }
}

fn scoped_graph_user_id(
    principal: &crate::api::auth::RequestPrincipal,
    requested: Option<String>,
) -> Result<String, StatusCode> {
    let ns_prefix = principal_namespace_prefix(principal);
    let user_id = match requested {
        Some(user_id) if !user_id.trim().is_empty() => {
            scope_entity_id(user_id.trim(), ns_prefix.as_deref())
        }
        None => ns_prefix
            .as_deref()
            .map(|prefix| prefix.trim_end_matches(':').to_string())
            .ok_or(StatusCode::BAD_REQUEST)?,
        _ => return Err(StatusCode::BAD_REQUEST),
    };
    Ok(user_id)
}

pub async fn graph_query_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(payload): Json<GraphQueryPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let user_id = scoped_graph_user_id(&principal, payload.user_id)?;
    let tenant_id = crate::api::auth::principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let tenant_clone = tenant.clone();
    let results = tokio::task::spawn_blocking(move || {
        tenant_clone.graph_query_edges(
            &user_id,
            payload.edge_type.as_deref(),
            "Both",
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
    headers: HeaderMap,
    Json(payload): Json<GraphWalkPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let user_id = scoped_graph_user_id(&principal, payload.user_id)?;
    let tenant_id = crate::api::auth::principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let tenant_clone = tenant.clone();
    let results = tokio::task::spawn_blocking(move || {
        tenant_clone.graph_query_edges(
            &user_id,
            None,
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
    headers: HeaderMap,
    Json(payload): Json<GraphExportPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let user_id = scoped_graph_user_id(&principal, payload.user_id)?;
    let tenant_id = crate::api::auth::principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let tenant_clone = tenant.clone();
    let results = tokio::task::spawn_blocking(move || {
        tenant_clone.graph_query_edges(
            &user_id,
            None,
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
    headers: HeaderMap,
    Json(payload): Json<AnalyticsQueryPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
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

pub async fn temporal_query_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let tenant_id = crate::api::auth::principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let _tenant_clone = tenant.clone();
    let results: Vec<(String, String)> = Vec::new();
    #[derive(Serialize)]
    struct TempObs {
        entity_id: String,
        textual_content: String,
    }
    let mapped: Vec<_> = results
        .into_iter()
        .map(|(entity_id, textual_content)| TempObs {
            entity_id,
            textual_content,
        })
        .collect();
    record_usage_for_principal(&state, &principal, "temporal_query");
    Ok((StatusCode::OK, Json(mapped)))
}
const NEURAL_TOP: usize = 25;
const NEURAL_BATCH: usize = 8;

struct QueryPipelineState {
    payload: QueryPayload,
    state: EngineState,
    tenant: std::sync::Arc<TenantStore>,
    limit: usize,
    enable_neural_rerank: bool,
    weights: ScoringWeights,

    total_start: Instant,
    route_start: Instant,

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

    now_ms: u64,
    fused: Vec<(String, u64, f32)>,
    graph_scores: HashMap<String, f32>,

    observations: HashMap<String, AgentObservation>,
    memory_cards: HashMap<String, MemoryCard>,
    disambiguation_vectors: Vec<(String, Vec<f32>)>,
    negative_centroids: Vec<(String, Vec<f32>)>,
    invalidated_facts: HashSet<String>,
}

impl QueryPipelineState {
    fn new(
        payload: QueryPayload,
        state: EngineState,
        tenant: std::sync::Arc<TenantStore>,
        limit: usize,
        enable_neural_rerank: bool,
    ) -> Self {
        Self {
            payload,
            state,
            tenant,
            limit,
            enable_neural_rerank,
            weights: ScoringWeights::default(),
            total_start: Instant::now(),
            route_start: Instant::now(),
            raw_query_text: String::new(),
            query_text: String::new(),
            include_evidence: false,
            verify_evidence: false,
            proof_mode: String::new(),
            evidence_radius: 0,
            plan: QueryPlan {
                semantic_queries: Vec::new(),
                fts_queries: Vec::new(),
                coverage_facets: Vec::new(),
                requirements: Vec::new(),
                prefer_distilled: false,
                prefer_episodic: false,
                temporal_terms: Vec::new(),
                lexical_terms: Vec::new(),
                intent: QueryIntent::General,
                subject_entities: Vec::new(),
                cross_entity: false,
                needs_decomposition: false,
                coverage_mode: false,
                ordinal_rank: None,
            },
            primary_qembed: Vec::new(),
            budget: RetrievalBudget {
                semantic_top: 0,
                fts_top: 0,
                semantic_query_limit: 0,
                fts_query_limit: 0,
                session_router_limit: 0,
                event_limit: 0,
                shadow_limit: 0,
                facet_limit: 0,
                scene_limit: 0,
                session_ann_limit: 0,
                event_vector_limit: 0,
                shadow_vector_limit: 0,
                route_probe_query_limit: 0,
                route_probe_hit_limit: 0,
                route_take_simple: 0,
                route_take_hard: 0,
                card_limit: 0,
            },
            semantic_top: 0,
            fts_top: 0,
            adaptive_profile: QueryAdaptiveProfile {
                semantic_scale: 0.0,
                lexical_scale: 0.0,
                route_sessions: HashSet::new(),
                route_strength: 0.0,
            },
            diag: QueryDiagnostics::default(),
            session_route_scores: HashMap::new(),
            routed_memory_ids: HashMap::new(),
            primary_hnsw_raw: Vec::new(),
            semantic_ranked_lists: Vec::new(),
            fts_ranked_lists: Vec::new(),
            card_ranked_items: Vec::new(),
            neural_scores: HashMap::new(),
            now_ms: 0,
            fused: Vec::new(),
            graph_scores: HashMap::new(),
            observations: HashMap::new(),
            memory_cards: HashMap::new(),
            disambiguation_vectors: Vec::new(),
            negative_centroids: Vec::new(),
            invalidated_facts: HashSet::new(),
        }
    }
}

fn plan_phase(s: &mut QueryPipelineState) {
    s.raw_query_text = s.payload.textual_query.clone();
    s.query_text = rewrite_query_for_retrieval(&s.raw_query_text);
    s.include_evidence = s.payload.include_evidence.unwrap_or(false)
        || s.payload.verify_evidence.unwrap_or(false);
    s.verify_evidence = s.payload.verify_evidence.unwrap_or(false);
    s.proof_mode = s.payload
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
    s.semantic_top = if s.payload.entity_id.is_some() {
        scoped_semantic_top()
    } else {
        SEMANTIC_TOP_DEFAULT
    };
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

    s.primary_qembed = s.state
        .semantic
        .generate_query_embedding(&s.query_text)
        .unwrap_or_default();

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

        let (sr_hits, win_hits, pivot_hits) = std::thread::scope(|sc| {
            let h_sr = {
                let tenant = tenant_for_routes.clone();
                let eid = entity_for_routes.clone();
                let q = query_text_for_routes.clone();
                let l = lexical_for_routes.clone();
                let t = temporal_for_routes.clone();
                let s_e = subject_for_routes.clone();
                sc.spawn(move || {
                    tenant.search_session_router(&eid, &q, &l, &t, &s_e, session_router_limit).unwrap_or_default()
                })
            };
            let h_win = {
                let tenant = tenant_for_routes.clone();
                let eid = entity_for_routes.clone();
                let q = query_text_for_routes.clone();
                sc.spawn(move || {
                    if let Some((st, en)) = parse_temporal_window(&q) {
                        tenant.sessions_in_time_window(&eid, st, en).unwrap_or_default()
                    } else { Vec::new() }
                })
            };
            let h_pivot = {
                let tenant = tenant_for_routes.clone();
                let eid = entity_for_routes.clone();
                let s_e = subject_for_routes.clone();
                sc.spawn(move || {
                    if !s_e.is_empty() {
                        tenant.entity_pivot_sessions(&eid, &s_e).unwrap_or_default()
                    } else { Vec::new() }
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
            *s.session_route_scores.entry(hit.session_id).or_insert(0.0) += hit.score + coverage_bonus;
        }
        s.diag.route_session_ms = start_proc.elapsed().as_millis() as u64;

        let start_proc = Instant::now();
        for hit in win_hits {
            *s.session_route_scores.entry(hit.session_id).or_insert(0.0) += s.weights.time_window_bonus;
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
    let mut session_votes: HashMap<String, usize> = HashMap::new();
    let route_probe_queries = if s.budget.route_probe_query_limit == 0 {
        Vec::new()
    } else if s.plan.coverage_facets.is_empty() {
        vec![s.plan
            .fts_queries
            .first()
            .cloned()
            .unwrap_or_else(|| s.query_text.clone())]
    } else {
        s.plan.coverage_facets
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
                let hits = tenant.fts_search(
                    probe_query.as_str(),
                    hit_limit,
                    eid.as_deref(),
                ).unwrap_or_default();
                (probe_idx, hits)
            }));
        }
        for handle in handles {
            if let Ok(res) = handle.join() {
                route_probe_results.push(res);
            }
        }
    });

    for (probe_idx, route_probe_hits) in route_probe_results {
        let vote_weight = match probe_idx {
            0 => 3,
            1 => 2,
            _ => 1,
        };
        for (memory_id, _) in route_probe_hits.iter().take(s.budget.route_probe_hit_limit) {
            if let Some(session_id) = routed_session_from_memory_id(memory_id) {
                *session_votes.entry(session_id).or_insert(0) += vote_weight;
            }
        }
    }
    for (session_id, votes) in session_votes {
        *s.session_route_scores.entry(session_id).or_insert(0.0) +=
            votes as f32 * s.state.ranking_config.session_boost;
    }
    let mut top_sessions = s.session_route_scores
        .iter()
        .map(|(sid, score)| (sid.clone(), *score))
        .collect::<Vec<_>>();
    top_sessions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let route_take = if s.plan.needs_decomposition || s.plan.cross_entity {
        s.budget.route_take_hard
    } else {
        s.budget.route_take_simple
    };
    s.adaptive_profile.route_sessions = top_sessions
        .into_iter()
        .take(route_take)
        .map(|(session, _)| session)
        .collect();
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
    let stage_start = Instant::now();
    let semantic_queries = s.plan
        .semantic_queries
        .iter()
        .take(s.budget.semantic_query_limit)
        .cloned()
        .collect::<Vec<_>>();
    let primary_qembed_clone = s.primary_qembed.clone();
    let mut embeddings = vec![primary_qembed_clone];
    for q in semantic_queries.iter().skip(1) {
        if let Ok(emb) = s.state.semantic.generate_query_embedding(q) {
            embeddings.push(emb);
        }
    }
    (s.diag.embed_ms, s.diag.embed_us) = elapsed_ms_and_us(stage_start);

    let scoped_entity_id = s.payload.entity_id.clone();
    let scope_prefix: Option<String> =
        scoped_entity_id.as_ref().map(|id| format!("{}::", id));
    let stage_start = Instant::now();
    let ann_results: Vec<(
        usize,
        Vec<String>,
        Vec<RankedItem>,
        Vec<(u64, f32)>,
    )> = {
        let state_clone = s.state.clone();
        let tenant_clone = s.tenant.clone();
        let eid = scoped_entity_id.clone();
        let prefix = scope_prefix.clone();
        let profile = &s.adaptive_profile;
        let query_limit = s.limit;
        let semantic_top = s.semantic_top;
        let dedup_threshold = s.weights.dedup_similarity_threshold;
        std::thread::scope(|sc| {
            let mut handles = Vec::with_capacity(embeddings.len());
            for (idx, embedding) in embeddings.iter().enumerate() {
                let state = state_clone.clone();
                let tenant = tenant_clone.clone();
                let embedding = embedding.clone();
                let eid = eid.clone();
                let prefix = prefix.clone();
                let handle = sc.spawn(move || {
                    let mut hnsw_hits = Vec::new();
                    let mut variant_rerank_seed_ids = Vec::new();
                    let mut local_cache: HashMap<u64, Option<(u64, String)>> = HashMap::new();
                    let hnsw_raw = if let (Some(entity_id), Some(prefix)) =
                        (eid.as_deref(), prefix.as_ref())
                    {
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
                            let current_raw = state
                                .vector_index
                                .search(Some(entity_id), &embedding, current_top)
                                .unwrap_or_default();
                            let scoped_last_raw = current_raw.clone();
                            let mut scoped_top_similarity = None;
                            let mut new_scoped_hits = 0usize;

                            let unresolved_ids: Vec<u64> = current_raw
                                .iter()
                                .map(|(vid, _)| *vid)
                                .filter(|vid| !local_cache.contains_key(vid))
                                .collect();
                            if !unresolved_ids.is_empty() {
                                if let Ok(looked) = tenant.lookup_by_vector_ids_batch(&unresolved_ids)
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
                                if !mem_id.starts_with(prefix.as_str()) {
                                    continue;
                                }
                                new_scoped_hits += 1;
                                if cumulative_hnsw_hits.len() < NEURAL_TOP {
                                    variant_rerank_seed_ids.push(mem_id.clone());
                                }
                                cumulative_hnsw_hits.push(RankedItem {
                                    memory_id: mem_id,
                                    timestamp: ts,
                                });
                                if scoped_top_similarity.is_none() {
                                    scoped_top_similarity = Some((1.0 - *dist).clamp(-1.0, 1.0));
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
                            let next_top = current_top.saturating_add(scoped_step).min(scoped_max_top);
                            if next_top == current_top {
                                break scoped_last_raw;
                            }
                            prev_hit_count = Some(cumulative_hnsw_hits.len());
                            prev_top_similarity = scoped_top_similarity;
                            current_top = next_top;
                        };
                        hnsw_hits = cumulative_hnsw_hits;
                        hnsw_raw
                    } else {
                        let hnsw_raw = state
                            .vector_index
                            .search(None, &embedding, semantic_top)
                            .unwrap_or_default();
                        let vids: Vec<u64> = hnsw_raw.iter().map(|(vid, _)| *vid).collect();
                        if let Ok(looked) = tenant.lookup_by_vector_ids_batch(&vids)
                        {
                            for (rank, (_vid, dist)) in hnsw_raw.iter().enumerate() {
                                if let Some((ts, mem_id)) = looked[rank].clone() {
                                    let routed_match = routed_session_from_memory_id(&mem_id)
                                        .map(|s| profile.route_sessions.contains(&s))
                                        .unwrap_or(false);
                                    if profile.route_strength > 0.0
                                        && !routed_match
                                        && hnsw_hits.len() < (query_limit.saturating_div(2).max(1))
                                    {
                                        let similarity = (1.0f32 - *dist * 0.5f32).max(0.0f32);
                                        if similarity < dedup_threshold {
                                            continue;
                                        }
                                    }
                                    if rank < NEURAL_TOP {
                                        variant_rerank_seed_ids.push(mem_id.clone());
                                    }
                                    hnsw_hits.push(RankedItem {
                                        memory_id: mem_id,
                                        timestamp: ts,
                                    });
                                }
                            }
                        }
                        hnsw_raw
                    };
                    (idx, variant_rerank_seed_ids, hnsw_hits, hnsw_raw)
                });
                handles.push(handle);
            }
            handles.into_iter().map(|h| h.join().expect("ANN search thread panicked")).collect::<Vec<_>>()
        })
    };

    let mut primary_hnsw_raw = Vec::new();
    let mut semantic_ranked_lists = Vec::new();
    for (idx, _seed_ids, hnsw_hits, hnsw_raw) in ann_results {
        if idx == 0 {
            primary_hnsw_raw = hnsw_raw;
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
    let stage_start = Instant::now();
    let mut fts_ranked_lists = Vec::new();
    let mut fts_memory_ids_to_lookup = Vec::new();
    let mut fts_results_per_query = Vec::new();

    let fts_queries_to_run: Vec<&String> = s.plan.fts_queries.iter().take(s.budget.fts_query_limit).collect();
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
                        .unwrap_or_default();
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
    let fts_lookup = s.tenant
        .lookup_by_memory_ids_batch(&fts_memory_ids_to_lookup)
        .unwrap_or_default();
    for (idx, hits) in fts_results_per_query.into_iter().enumerate() {
        let mut ranked_hits = Vec::new();
        for (mid, _) in hits {
            if let Some((ts, _)) = fts_lookup.get(&mid).copied() {
                ranked_hits.push(RankedItem {
                    memory_id: mid,
                    timestamp: ts,
                });
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
    if let Some(ref entity_id) = s.payload.entity_id {
        let include_stale_cards = query_allows_stale_cards(&s.query_text, &s.plan);
        let card_hits = s.tenant
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
            card_ranked_items.push(RankedItem {
                memory_id: hit.card_id,
                timestamp: hit.timestamp,
            });
            if !hit.source_session_id.is_empty()
                && hit.lexical_hits + hit.temporal_hits + hit.entity_hits >= 2
            {
                *s.session_route_scores
                    .entry(hit.source_session_id)
                    .or_insert(0.0) +=
                    hit.score * s.state.ranking_config.session_boost * 1.5;
            }
        }
    }
    s.card_ranked_items = card_ranked_items;
    (s.diag.card_ms, s.diag.card_us) = elapsed_ms_and_us(stage_start);
}

fn rerank_phase(s: &mut QueryPipelineState) {
    s.neural_scores = HashMap::new();
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
    if should_apply_neural_rerank(
        &s.query_text,
        &s.primary_hnsw_raw,
        s.enable_neural_rerank || auto_rerank,
    ) {
        let stage_start = Instant::now();
        s.diag.rerank_applied = true;
        let mut rerank_seed_ids = Vec::new();
        for (_weight, list) in &s.semantic_ranked_lists {
            for item in list.iter().take(NEURAL_TOP / 2) {
                rerank_seed_ids.push(item.memory_id.clone());
            }
        }
        for (_weight, list) in &s.fts_ranked_lists {
            for item in list.iter().take(NEURAL_TOP / 2) {
                rerank_seed_ids.push(item.memory_id.clone());
            }
        }
        let mut rerank_seen = HashSet::new();
        let active_seeds: Vec<String> = rerank_seed_ids
            .into_iter()
            .filter(|mid: &String| rerank_seen.insert(mid.clone()))
            .collect();
        let lookup = s.tenant
            .lookup_by_memory_ids_batch(&active_seeds)
            .unwrap_or_default();
        let obs_keys: Vec<(u64, String)> = active_seeds
            .iter()
            .filter_map(|mid: &String| lookup.get(mid).map(|(ts, _)| (*ts, mid.clone())))
            .take(NEURAL_TOP)
            .collect();
        let observations = s.tenant
            .get_observations_batch(&obs_keys)
            .unwrap_or_default();
        let mut rerank_items = Vec::new();
        let mut rerank_texts = Vec::new();
        for mid in active_seeds {
            if let Some(obs) = observations.get(&mid) {
                rerank_items.push(mid);
                rerank_texts.push(obs.textual_content.clone());
            }
        }
        for (item_chunk, text_chunk) in rerank_items
            .chunks(NEURAL_BATCH)
            .zip(rerank_texts.chunks(NEURAL_BATCH))
        {
            let scores = s.state
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
            s.state.ranking_config.card_boost * 1.3
        } else if matches!(
            s.plan.intent,
            QueryIntent::TemporalAggregation | QueryIntent::Inference
        ) {
            s.state.ranking_config.card_boost * 1.15
        } else {
            s.state.ranking_config.card_boost
        };
        ranked_sources.push((card_weight, s.card_ranked_items.clone()));
    }
    if !s.neural_scores.is_empty() {
        let mut scored_items: Vec<_> = s.neural_scores.clone().into_iter().collect();
        scored_items
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut neural_items = Vec::new();
        for (mid, _) in scored_items {
            if let Some((ts, _)) = s.tenant
                .lookup_by_memory_id(&mid)
                .unwrap_or(None)
            {
                neural_items.push(RankedItem {
                    memory_id: mid,
                    timestamp: ts,
                });
            }
        }
        ranked_sources.push((1.5, neural_items));
    }
    let fused = weighted_reciprocal_rank_fusion(ranked_sources, adaptive_rrf_k(&s.plan));
    (s.diag.fuse_ms, s.diag.fuse_us) = elapsed_ms_and_us(stage_start);

    s.now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let mut fused_map: HashMap<String, (u64, f32)> = HashMap::new();
    for (mid, ts, score) in fused {
        let entry = fused_map.entry(mid).or_insert((ts, 0.0));
        entry.0 = ts;
        entry.1 = entry.1.max(score);
    }

    for (mid, boost) in &s.routed_memory_ids {
        if let Some((ts, _)) = s.tenant
            .lookup_by_memory_id(mid)
            .unwrap_or(None)
        {
            let entry = fused_map.entry(mid.clone()).or_insert((ts, 0.0));
            entry.0 = ts;
            entry.1 += *boost;
        }
    }

    let stage_start = Instant::now();
    if s.plan.intent == QueryIntent::Inference && !s.primary_qembed.is_empty() {
        if let Some(ref entity_id) = s.payload.entity_id {
            let preference_memories = s.tenant
                .get_preference_memories(entity_id, 96)
                .unwrap_or_default();
            if !preference_memories.is_empty() {
                let option_embeddings = [s.primary_qembed.clone()];
                let memory_ids: Vec<String> = preference_memories
                    .iter()
                    .map(|(memory_id, _)| memory_id.clone())
                    .collect();
                let lookup = s.tenant
                    .lookup_by_memory_ids_batch(&memory_ids)
                    .unwrap_or_default();
                let observation_keys: Vec<(u64, String)> = memory_ids
                    .iter()
                    .filter_map(|memory_id| {
                        lookup
                            .get(memory_id)
                            .map(|(ts, _)| (*ts, memory_id.clone()))
                    })
                    .collect();
                let observations = s.tenant
                    .get_observations_batch(&observation_keys)
                    .unwrap_or_default();
                for (memory_id, strength) in preference_memories {
                    let Some(obs) = observations.get(&memory_id) else {
                        continue;
                    };
                    if obs.embedding.len() != s.primary_qembed.len()
                        || obs.embedding.is_empty()
                    {
                        continue;
                    }
                    let best_similarity = option_embeddings
                        .iter()
                        .filter(|candidate| candidate.len() == obs.embedding.len())
                        .map(|candidate| cosine_similarity(candidate, &obs.embedding))
                        .fold(-1.0f32, f32::max);
                    if best_similarity >= 0.35 {
                        if let Some((ts, _)) = lookup.get(&memory_id).copied() {
                            let entry =
                                fused_map.entry(memory_id.clone()).or_insert((ts, 0.0));
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
    let mut link_seed_ids: Vec<(String, f32)> = fused_map
        .iter()
        .map(|(mid, (_, score))| (mid.clone(), *score))
        .collect();
    link_seed_ids
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut graph_scores: HashMap<String, f32> = HashMap::new();
    for (seed_mid, _) in link_seed_ids.into_iter().take(24) {
        let cluster_scores = collect_link_cluster_scores(&s.tenant, &seed_mid, 2);
        for (linked_mid, boost) in cluster_scores {
            if let Some((ts, _)) = s.tenant
                .lookup_by_memory_id(&linked_mid)
                .unwrap_or(None)
            {
                *graph_scores.entry(linked_mid.clone()).or_insert(0.0) += boost;
                let entry = fused_map.entry(linked_mid).or_insert((ts, 0.0));
                entry.0 = ts;
            }
        }
    }
    if s.plan.needs_decomposition
        || s.plan.cross_entity
        || matches!(
            s.plan.intent,
            QueryIntent::Inference | QueryIntent::TemporalAggregation
        )
    {
        let mut graph_seed_nodes = s.plan.subject_entities.clone();
        let query_phrase_lines = [s.payload.textual_query.clone()];
        for phrase in extract_named_phrases(&query_phrase_lines) {
            if phrase.len() >= 3
                && !graph_seed_nodes
                    .iter()
                    .any(|existing| existing.eq_ignore_ascii_case(&phrase))
            {
                graph_seed_nodes.push(phrase);
            }
        }
        graph_seed_nodes.truncate(8);

    }
    s.graph_scores = graph_scores;
    (s.diag.graph_ms, s.diag.graph_us) = elapsed_ms_and_us(stage_start);

    let mut fused_vec: Vec<(String, u64, f32)> = fused_map
        .into_iter()
        .map(|(mid, (ts, score))| (mid, ts, score))
        .collect();
    fused_vec.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    s.fused = fused_vec;
}

fn score_phase(s: &mut QueryPipelineState) -> Vec<QueryResult> {
    score_hydrate(s);
    let evidence_cards = score_loop(s);
    score_build_response(s, evidence_cards)
}

fn score_hydrate(s: &mut QueryPipelineState) {
    let observation_keys: Vec<(u64, String)> = s.fused
        .iter()
        .map(|(mid, ts, _)| (*ts, mid.clone()))
        .collect();
    let obs_start = Instant::now();
    s.observations = s.tenant
        .get_observations_batch(&observation_keys)
        .unwrap_or_default();
    (s.diag.fetch_obs_ms, s.diag.fetch_obs_us) = elapsed_ms_and_us(obs_start);

    let observation_memory_ids: Vec<String> =
        s.fused.iter().map(|(mid, _, _)| mid.clone()).collect();

    let cards_start = Instant::now();
    s.memory_cards = s.tenant
        .get_memory_cards_batch(&observation_memory_ids)
        .unwrap_or_default();
    (s.diag.fetch_cards_ms, s.diag.fetch_cards_us) = elapsed_ms_and_us(cards_start);

    let vectors_start = Instant::now();
    let entity_for_vec = s.payload.entity_id.as_deref().unwrap_or("default");
    s.disambiguation_vectors = s.tenant
        .get_disambiguation_vectors_batch(entity_for_vec)
        .unwrap_or_default();
    (s.diag.fetch_vectors_ms, s.diag.fetch_vectors_us) = elapsed_ms_and_us(vectors_start);

    let neg_start = Instant::now();
    s.negative_centroids = s.tenant
        .get_negative_centroids_batch(entity_for_vec)
        .unwrap_or_default();
    (s.diag.fetch_neg_ms, s.diag.fetch_neg_us) = elapsed_ms_and_us(neg_start);

    let _fact_ids: Vec<String> = s.observations
        .iter()
        .filter_map(|(mid, obs)| (obs.kind == MemoryKind::Fact).then_some(mid.clone()))
        .collect();

    let invalid_start = Instant::now();
    s.invalidated_facts = s.tenant
        .invalidated_set()
        .unwrap_or_default();
    (s.diag.fetch_invalid_ms, s.diag.fetch_invalid_us) = elapsed_ms_and_us(invalid_start);
}

fn score_loop(s: &mut QueryPipelineState) -> Vec<EvidenceCard> {
    let loop_start = Instant::now();
    let mut scored = Vec::new();
    let primary_qembed = &s.primary_qembed;
    let graph_scores = &s.graph_scores;
    let neural_scores = &s.neural_scores;
    let memory_cards = &s.memory_cards;
    let observations = &s.observations;
    let invalidated_facts = &s.invalidated_facts;
    let disambiguation_vectors = &s.disambiguation_vectors;
    let negative_centroids = &s.negative_centroids;
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
        let is_stale_fact = obs.kind == MemoryKind::Fact && invalidated_facts.contains(mid);
        let created_at_ms = if obs.created_at_ms > 0 {
            obs.created_at_ms
        } else {
            *ts
        };
        let scorable = ScorableObservation::new(&obs.textual_content);
        let entity_hits = entity_hit_count(&scorable, plan);
        let lexical_hits = lexical_hit_count(&scorable, plan);
        let temporal_hits = temporal_hit_count(&scorable, plan);
        let facet_mask = facet_match_mask(&scorable, plan);
        let base_score = neural_scores.get(mid).copied().unwrap_or(*rrf_score);
        let lifecycle = memory_cards
            .get(mid)
            .and_then(|card| card.lifecycle.as_ref())
            .cloned()
            .unwrap_or_else(|| {
                crate::lifecycle::evaluate_lifecycle(
                    &obs.textual_content,
                    obs.kind,
                    created_at_ms,
                    None,
                    false,
                )
            });
        let Some(lifecycle_adjustment) =
            lifecycle_rank_adjustment(&lifecycle, obs.kind, now_ms)
        else {
            continue;
        };
        if let Some(mut fs) =
            apply_decay_with_policy(base_score, created_at_ms, obs.kind, now_ms)
        {
            fs += lifecycle_adjustment;
            if is_stale_fact {
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
            let disambiguation_map: std::collections::HashMap<&str, &[f32]> = disambiguation_vectors.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
            let negative_map: std::collections::HashMap<&str, &[f32]> = negative_centroids.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
            if let Some(disambiguation) = disambiguation_map.get(mid.as_str()) {
                if disambiguation.len() == primary_qembed.len()
                    && !primary_qembed.is_empty()
                {
                    fs +=
                        cosine_similarity(primary_qembed, disambiguation) * 0.05;
                }
            }
            if let Some(negative_centroid) = negative_map.get(mid.as_str()) {
                if negative_centroid.len() == primary_qembed.len()
                    && obs.embedding.len() == primary_qembed.len()
                    && !primary_qembed.is_empty()
                {
                    let margin =
                        cosine_similarity(primary_qembed, &obs.embedding)
                            - cosine_similarity(
                                primary_qembed,
                                negative_centroid,
                            );
                    fs += margin * 0.08;
                }
            }

            let graph_score = graph_scores.get(mid).copied().unwrap_or(0.0);
            let temporal_adjust = if temporal_recency_scoring_enabled() {
                temporal_consistency_adjustment(obs.kind, created_at_ms, now_ms, plan_intent)
            } else {
                0.0
            };
            let confidence_signal: f32 = if is_stale_fact { s.weights.rerank_stale_penalty }
                else if lifecycle.stability_score > 0.7 { s.weights.rerank_confidence_stable }
                else if lifecycle.confidence_score > 0.7 { s.weights.rerank_confidence_high }
                else { 0.0 };

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
            fs = fs * (1.0 - s.weights.four_signal_temporal_weight) + reweighted * s.weights.four_signal_temporal_weight;
            if adaptive_profile.route_strength > 0.0 {
                let routed_sid = memory_cards
                    .get(mid)
                    .map(|card| card.source_session_id.clone())
                    .or_else(|| routed_session_from_memory_id(mid));
                if let Some(sid) = routed_sid {
                    if adaptive_profile.route_sessions.contains(&sid) {
                        let route_score =
                            session_route_scores.get(&sid).copied().unwrap_or(0.0);
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
            let (source_memory_id, source_session_id) =
                if let Some(card) = memory_cards.get(mid) {
                    if card.source_memory_id != *mid {
                        (
                            card.source_memory_id.clone(),
                            card.source_session_id.clone(),
                        )
                    } else {
                        (
                            mid.clone(),
                            session_id_from_memory_id(mid).unwrap_or_default(),
                        )
                    }
                } else {
                    (
                        mid.clone(),
                        session_id_from_memory_id(mid).unwrap_or_default(),
                    )
                };

            scored.push(EvidenceCard {
                claim_text: obs.textual_content.clone(),
                source_memory_id,
                source_session_id,
                card_id: if memory_cards.contains_key(mid) {
                    Some(mid.clone())
                } else {
                    None
                },
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
                internal_kind: obs.kind,
                created_at_ms,
                entity_id: obs.entity_id.clone(),
            });
        }
    }
    (s.diag.scoring_loop_ms, s.diag.scoring_loop_us) = elapsed_ms_and_us(loop_start);
    scored
}

fn score_build_response(s: &mut QueryPipelineState, mut evidence_cards: Vec<EvidenceCard>) -> Vec<QueryResult> {
    (s.diag.hydrate_ms, s.diag.hydrate_us) = elapsed_ms_and_us(Instant::now());

    let stage_start = Instant::now();
    evidence_cards.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let selected = select_candidates_with_session_head(
        evidence_cards,
        s.limit,
        &s.plan,
        s.plan.prefer_distilled,
        s.plan.prefer_episodic,
    );

    let mut source_keys = Vec::new();
    for card in &selected {
        source_keys.push((card.created_at_ms, card.source_memory_id.clone()));
    }
    let hydrate_obs_start = Instant::now();
    let source_observations = s.tenant
        .get_observations_batch(&source_keys)
        .unwrap_or_default();
    (s.diag.hydrate_obs_ms, s.diag.hydrate_obs_us) = elapsed_ms_and_us(hydrate_obs_start);

    let selected_sessions = selected
        .iter()
        .map(|card| card.source_session_id.clone())
        .filter(|sid| !sid.is_empty())
        .collect::<Vec<_>>();
    let candidate_sessions = selected
        .iter()
        .enumerate()
        .map(|(rank, card)| {
            let mut features = HashMap::new();
            features.insert("rank".to_string(), rank as f32 + 1.0);
            features.insert("final_score".to_string(), card.final_score);
            features.insert("semantic_score".to_string(), card.semantic_score);
            features.insert("bm25_score".to_string(), card.bm25_score);
            features.insert(
                "session_router_score".to_string(),
                card.session_router_score,
            );
            features.insert("card_score".to_string(), card.card_score);
            features.insert("reranker_score".to_string(), card.reranker_score);
            features.insert("entity_hits".to_string(), card.entity_hits as f32);
            features.insert("lexical_hits".to_string(), card.lexical_hits as f32);
            features.insert("temporal_hits".to_string(), card.temporal_hits as f32);
            features.insert(
                "facet_hits".to_string(),
                card.facet_mask.count_ones() as f32,
            );
            features.insert("graph_score".to_string(), card.graph_score);
            features.insert("child_score".to_string(), card.child_score);
            SessionCandidateTrace {
                session_id: card.source_session_id.clone(),
                final_score: card.final_score,
                features,
                source_memory_ids: vec![card.source_memory_id.clone()],
                source_card_ids: card.card_id.clone().into_iter().collect(),
                source_event_ids: Vec::new(),
                is_gold: None,
            }
        })
        .collect::<Vec<_>>();
    let trace_id = format!(
        "query_trace::{}::{}",
        s.payload
            .entity_id
            .clone()
            .unwrap_or_else(|| "global".to_string()),
        s.now_ms
    );
    let query_trace = QueryTrace {
        query_trace_id: trace_id,
        entity_id: s.payload.entity_id.clone(),
        question: s.raw_query_text.clone(),
        query_plan: format!("{:?}", s.plan.intent),
        candidate_sessions,
        selected_sessions,
        returned_memory_ids: selected
            .iter()
            .map(|card| card.source_memory_id.clone())
            .collect(),
        latency_ms: s.total_start.elapsed().as_millis() as u64,
        gold_sessions: Vec::new(),
        created_at_ms: s.now_ms,
    };
    let trace_start = Instant::now();
    let _ = s.tenant.append_query_trace(&query_trace);
    (s.diag.trace_ms, s.diag.trace_us) = elapsed_ms_and_us(trace_start);

    let queries: Vec<QueryResult> = selected
        .into_iter()
        .map(|card| {
            let text =
                if let Some(source_obs) = source_observations.get(&card.source_memory_id) {
                    source_obs.textual_content.clone()
                } else {
                    card.claim_text.clone()
                };
            let evidence = if s.include_evidence && s.proof_mode != "off" {
                Some(build_proof_packet(
                    &s.tenant,
                    &s.query_text,
                    &s.plan,
                    &card,
                    &s.proof_mode,
                    s.verify_evidence,
                    s.evidence_radius,
                ))
            } else {
                None
            };
            QueryResult {
                memory_id: card.source_memory_id.clone(),
                entity_id: card.entity_id,
                session_id: card.source_session_id,
                turn_index: turn_index_from_memory_id(&card.source_memory_id),
                created_at_ms: card.created_at_ms,
                similarity: card.final_score,
                textual_content: text,
                evidence,
            }
        })
        .collect();
    let evidence_conf = compute_evidence_confidence(&queries, &s.query_text, s.state.intent_classifier.as_deref());
    s.diag.evidence_confidence_bp = (evidence_conf * 10_000.0) as u64;
    s.diag.abstain_recommended = evidence_conf < 0.24 && !queries.is_empty();
    (s.diag.session_ms, s.diag.session_us) = elapsed_ms_and_us(stage_start);
    (s.diag.total_ms, s.diag.total_us) = elapsed_ms_and_us(s.total_start);
    queries
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
    let results = score_phase(&mut s);
    Ok((results, s.diag))
}
