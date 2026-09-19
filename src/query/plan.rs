#![allow(dead_code)]

use super::*;

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

pub(crate) fn lifecycle_rank_adjustment(
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

pub(crate) fn attractor_negative_penalty(
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

pub(crate) fn query_allows_stale_cards(query: &str, plan: &QueryPlan) -> bool {
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
pub(crate) struct RetrievalBudget {
    pub(crate) semantic_top: usize,
    pub(crate) fts_top: usize,
    pub(crate) semantic_query_limit: usize,
    pub(crate) fts_query_limit: usize,
    pub(crate) session_router_limit: usize,
    pub(crate) route_probe_query_limit: usize,
    pub(crate) route_probe_hit_limit: usize,
    pub(crate) route_take_simple: usize,
    pub(crate) route_take_hard: usize,
    pub(crate) card_limit: usize,
}

pub(crate) fn retrieval_profile(config: &crate::config::Config) -> RetrievalProfile {
    config.retrieval.profile
}

pub(crate) fn auto_rerank_enabled(
    config: &crate::config::Config,
    profile: RetrievalProfile,
) -> bool {
    config.retrieval.auto_rerank.unwrap_or(matches!(profile, RetrievalProfile::Research))
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

pub(crate) fn plan_phase(s: &mut QueryPipelineState) {
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
    s.plan = build_query_plan_with_profile(
        &s.query_text,
        s.state.intent_classifier.as_deref(),
        s.state.config.heuristics,
    );

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

    let retrieval_profile = retrieval_profile(&s.state.config);
    s.budget = retrieval_budget_for_plan(&s.plan, retrieval_profile);
    s.fts_top = match s.plan.intent {
        QueryIntent::Inference | QueryIntent::PeripheralMention => 180,
        QueryIntent::TemporalAggregation => 120,
        QueryIntent::NumericAggregation => 90,
        QueryIntent::Recommendation | QueryIntent::General => 72,
    }
    .min(s.budget.fts_top);
    s.semantic_top = if s.payload.entity_id.is_some() {
        s.state.config.retrieval.scoped_semantic_top
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

    // On failure keep the embedding empty: retrieval_ann skips ANN for a
    // wrong-dimension vector, and searching with a zero vector would return
    // arbitrary neighbours.
    s.primary_qembed =
        s.state.semantic.generate_query_embedding(&s.query_text).unwrap_or_else(|err| {
            tracing::warn!(query = %s.raw_query_text, error = ?err, "query embedding failed; skipping ANN");
            Vec::new()
        });

    s.route.memory_scores = HashMap::new();
    s.route.session_scores = HashMap::new();
    if let Some(ref entity_id) = s.payload.entity_id {
        let entity_for_routes = entity_id.clone();
        let lexical_for_routes = s.plan.lexical_terms.clone();
        let temporal_for_routes = s.plan.temporal_terms.clone();
        let subject_for_routes = s.plan.subject_entities.clone();
        let query_text_for_routes = s.query_text.clone();
        let tenant_for_routes = s.tenant.clone();
        let session_router_limit = s.budget.session_router_limit;
        let router_on = s.state.config.features.enabled(Feature::SessionRouter);

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
            *s.route.session_scores.entry(hit.session_id).or_insert(0.0) +=
                hit.score + coverage_bonus;
        }
        s.diag.route_session_ms = start_proc.elapsed().as_millis() as u64;

        let start_proc = Instant::now();
        for hit in win_hits {
            *s.route.session_scores.entry(hit.session_id).or_insert(0.0) +=
                s.weights.time_window_bonus;
        }
        s.diag.route_window_ms = start_proc.elapsed().as_millis() as u64;

        let start_proc = Instant::now();
        if !pivot_hits.is_empty() {
            let total_sessions = pivot_hits.len().max(1);
            let multi_entity = subject_for_routes.len() >= 2;
            for hit in &pivot_hits {
                if multi_entity && hit.entity_hits >= 2 {
                    *s.route.session_scores.entry(hit.session_id.clone()).or_insert(0.0) += 0.25;
                } else if hit.entity_hits >= 1 && total_sessions <= 12 {
                    *s.route.session_scores.entry(hit.session_id.clone()).or_insert(0.0) += 0.08;
                }
            }
        }
        s.diag.route_pivot_ms = start_proc.elapsed().as_millis() as u64;
    }
    (s.diag.planning_ms, s.diag.planning_us) = elapsed_ms_and_us(planning_start);
}
