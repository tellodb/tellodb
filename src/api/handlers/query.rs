use crate::api::auth::{principal_user_id, record_usage_for_principal, RequestPrincipal};
use crate::api::plan::build_observation_block;
use crate::api::types::{
    AnalyticsQueryPayload, AnalyticsQueryResult, BucketedResult, GraphExportPayload,
    GraphQueryPayload, GraphWalkPayload, QueryPayload, QueryResult, ResultOrigin,
};
use crate::api::utils::{
    clip_profile_to_budget, extract_named_phrases, insert_f32_header, insert_stage_timing_headers,
    insert_u64_header,
};
use crate::api::{EngineState, PlatformWriteOp};
use crate::error::{EngineError, EngineResult};
use crate::metrics;
use crate::query::execute_query_pipeline;
use crate::storage::repo::traits::QueryRepo;
use axum::{
    extract::{Extension, Json, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
};
use std::time::{SystemTime, UNIX_EPOCH};

fn current_core_profile(
    tenant: &dyn QueryRepo,
    entity_id: &str,
    point_in_time_ms: Option<u64>,
) -> EngineResult<Option<String>> {
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
    Ok(Some(serde_json::to_string(&profile).map_err(anyhow::Error::from)?))
}

fn build_entity_observation_block(
    tenant: &dyn QueryRepo,
    entity_id: &str,
    query_text: &str,
    results: &[QueryResult],
    point_in_time_ms: Option<u64>,
) -> EngineResult<String> {
    let profile = current_core_profile(tenant, entity_id, point_in_time_ms)?
        .map(|p| clip_profile_to_budget(&p, 8));

    let mut scenes = Vec::new();
    for entity in extract_named_phrases(&[query_text.to_string()]) {
        let lines = tenant.graph_edge_summaries_for_label(entity_id, &entity, 5)?;
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
) -> Result<impl IntoResponse, EngineError> {
    let tenant_id = principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id)?;
    let profile_query_text = payload.textual_query.clone();

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
                        tenant.query_repo(),
                        eid,
                        &block_query_text,
                        &results,
                        point_in_time_ms,
                    )?,
                )),
                None => None,
            };
            Ok::<_, EngineError>((results, diagnostics, obs_block))
        })
        .await
        .map_err(|err| EngineError::internal(format!("query task failed: {err}")))??
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
                origin: ResultOrigin::Stored,
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

fn parse_graph_direction_str(direction: Option<&str>) -> &'static str {
    match direction.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("in" | "inbound") => "Inbound",
        Some("both") => "Both",
        _ => "Outbound",
    }
}

fn scoped_graph_node_id(requested: Option<String>) -> EngineResult<String> {
    let node_id = match requested {
        Some(id) if !id.trim().is_empty() => id.trim().to_string(),
        None => return Err(EngineError::bad_request("graph node is required")),
        _ => return Err(EngineError::bad_request("graph node is required")),
    };
    Ok(node_id)
}

pub async fn graph_query_handler(
    State(state): State<EngineState>,
    Extension(principal): Extension<RequestPrincipal>,
    Json(payload): Json<GraphQueryPayload>,
) -> Result<impl IntoResponse, EngineError> {
    // If subject is provided, use it. If not, use the requested user_id.
    let subject = if !payload.subject.trim().is_empty() {
        payload.subject.trim().to_string()
    } else {
        scoped_graph_node_id(payload.user_id)?
    };

    let tenant_id = crate::api::auth::principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id)?;
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
    .map_err(|err| EngineError::internal(format!("graph query task failed: {err}")))??;
    record_usage_for_principal(&state, &principal, "query");
    Ok((StatusCode::OK, Json(results)))
}

pub async fn graph_walk_handler(
    State(state): State<EngineState>,
    Extension(principal): Extension<RequestPrincipal>,
    Json(payload): Json<GraphWalkPayload>,
) -> Result<impl IntoResponse, EngineError> {
    let node = if !payload.node.trim().is_empty() {
        payload.node.trim().to_string()
    } else {
        scoped_graph_node_id(payload.user_id)?
    };

    let tenant_id = crate::api::auth::principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id)?;
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
    .map_err(|err| EngineError::internal(format!("graph walk task failed: {err}")))??;
    record_usage_for_principal(&state, &principal, "query");
    Ok((StatusCode::OK, Json(results)))
}

pub async fn graph_export_handler(
    State(state): State<EngineState>,
    Extension(principal): Extension<RequestPrincipal>,
    Json(payload): Json<GraphExportPayload>,
) -> Result<impl IntoResponse, EngineError> {
    let seed = if !payload.seed.trim().is_empty() {
        payload.seed.trim().to_string()
    } else {
        scoped_graph_node_id(payload.user_id)?
    };

    let tenant_id = crate::api::auth::principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id)?;
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
    .map_err(|err| EngineError::internal(format!("graph export task failed: {err}")))??;
    record_usage_for_principal(&state, &principal, "query");
    Ok((StatusCode::OK, Json(results)))
}

pub async fn analytics_query_handler(
    State(state): State<EngineState>,
    Extension(principal): Extension<RequestPrincipal>,
    Json(payload): Json<AnalyticsQueryPayload>,
) -> Result<impl IntoResponse, EngineError> {
    let user_id = crate::api::auth::principal_user_id(&principal).unwrap_or("default");
    let s = payload.start_timestamp_ms.unwrap_or(0);
    let e = payload.end_timestamp_ms.unwrap_or(u64::MAX);

    let agg = state
        .analytics
        .aggregate_range(user_id, &payload.entity_id, &payload.label, s, e)
        .map_err(|err| EngineError::internal(format!("analytics query failed: {err}")))?;

    let buckets = if let Some(bucket_str) = &payload.bucket {
        let bucket = match bucket_str.to_lowercase().as_str() {
            "hour" => crate::analytics::TemporalBucket::Hour,
            "day" => crate::analytics::TemporalBucket::Day,
            "week" => crate::analytics::TemporalBucket::Week,
            "month" => crate::analytics::TemporalBucket::Month,
            "year" => crate::analytics::TemporalBucket::Year,
            _ => return Err(EngineError::bad_request("unknown analytics bucket")),
        };
        let bucketed = state
            .analytics
            .aggregate_bucketed(user_id, &payload.entity_id, &payload.label, s, e, bucket)
            .map_err(|err| {
                EngineError::internal(format!("analytics bucket query failed: {err}"))
            })?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::plan::{QueryIntent, QueryPlan};
    use crate::config::RetrievalProfile;
    use crate::query::plan::{retrieval_budget_for_plan, QueryShape, RetrievalBudget};
    use crate::query::rerank::{rerank_gate_uncertain, rerank_policy_name};
    use crate::query::RerankDecision;
    use crate::storage::TenantStore;

    #[test]
    fn default_rerank_policy_is_the_confidence_gate() {
        // The string heuristic reranked 93% of LongMemEval dev, because it
        // fires on " and ", "would", "might" or a long question. If this ever
        // reverts to `heuristic`, the cross-encoder silently becomes an
        // always-on 169 ms stage again.
        assert_eq!(rerank_policy_name(&crate::config::Config::default()), "gate");
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
