#![allow(dead_code)]

pub use crate::api::plan::*;
pub use crate::api::types::{
    EvidenceCard, ProofCheck, ProofPacket, ProofTurn, QueryPayload, QueryResult, RankedItem,
};
pub use crate::api::utils::*;
pub use crate::api::EngineState;
pub use crate::config::{RerankPolicy, RetrievalProfile};
pub use crate::core::memory_id::{MemoryId, Tag};
pub use crate::error::{EngineError, EngineResult};
pub use crate::features::Feature;
pub use crate::metrics;
pub use crate::ml::cosine_similarity;
pub use crate::retrieval::lanes::Lane;
pub use crate::retrieval::{rrf_fuse, ScoringWeights};
pub use crate::storage::repo::traits::QueryRepo;
pub use crate::storage::{
    AgentObservation, MemoryCard, MemoryCardSearchInput, MemoryKind, TenantStore,
};
pub use std::collections::{HashMap, HashSet};
pub use std::ops::{Deref, DerefMut};
pub use std::sync::Arc;
pub use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub(crate) mod fuse;
pub(crate) mod plan;
pub(crate) mod rerank;
pub(crate) mod retrieve;
pub(crate) mod route;

pub(crate) use plan::query_allows_stale_cards;
pub(crate) use plan::{auto_rerank_enabled, retrieval_profile};
pub(crate) use route::collect_edge_cluster_scores_for_seeds;

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

#[derive(Default)]
pub struct QueryDiagnostics {
    pub(crate) route_ms: u64,
    pub(crate) route_us: u64,
    pub(crate) embed_ms: u64,
    pub(crate) embed_us: u64,
    pub(crate) ann_ms: u64,
    pub(crate) ann_us: u64,
    pub(crate) scoped_ann_top: u64,
    pub(crate) scoped_ann_attempts: u64,
    pub(crate) scoped_primary_hits: u64,
    pub(crate) rerank_ms: u64,
    pub(crate) rerank_us: u64,
    pub(crate) fts_ms: u64,
    pub(crate) fts_us: u64,
    pub(crate) fuse_ms: u64,
    pub(crate) fuse_us: u64,
    pub(crate) hydrate_ms: u64,
    pub(crate) hydrate_us: u64,
    pub(crate) preference_ms: u64,
    pub(crate) preference_us: u64,
    pub(crate) graph_ms: u64,
    pub(crate) graph_us: u64,
    pub(crate) graph_links_us: u64,
    pub(crate) graph_edges_us: u64,
    pub(crate) graph_seeds_wall_us: u64,
    pub(crate) graph_entities_us: u64,
    pub(crate) graph_lookup_us: u64,
    pub(crate) graph_expanded: u64,
    pub(crate) session_ms: u64,
    pub(crate) session_us: u64,
    pub(crate) score_loop_us: u64,
    pub(crate) factver_us: u64,
    pub(crate) build_cards_us: u64,
    pub(crate) proof_us: u64,
    pub(crate) confidence_us: u64,
    pub(crate) card_ms: u64,
    pub(crate) card_us: u64,
    pub(crate) planning_ms: u64,
    pub(crate) planning_us: u64,
    pub(crate) route_session_ms: u64,
    pub(crate) route_window_ms: u64,
    pub(crate) route_pivot_ms: u64,
    pub(crate) route_ann_ms: u64,
    pub(crate) route_profile_ms: u64,
    pub(crate) hydrate_obs_ms: u64,
    pub(crate) hydrate_obs_us: u64,
    pub(crate) fetch_obs_ms: u64,
    pub(crate) fetch_obs_us: u64,
    pub(crate) fetch_cards_ms: u64,
    pub(crate) fetch_cards_us: u64,
    pub(crate) fetch_vectors_ms: u64,
    pub(crate) fetch_vectors_us: u64,
    pub(crate) fetch_neg_ms: u64,
    pub(crate) fetch_neg_us: u64,
    pub(crate) fetch_invalid_ms: u64,
    pub(crate) fetch_invalid_us: u64,
    pub(crate) scoring_loop_ms: u64,
    pub(crate) scoring_loop_us: u64,
    pub(crate) trace_ms: u64,
    pub(crate) trace_us: u64,
    pub(crate) total_ms: u64,
    pub(crate) total_us: u64,
    pub(crate) rerank_applied: bool,
    pub(crate) rerank_reason: RerankDecision,
    pub(crate) routed_sessions: u64,
    pub(crate) memory_card_hits: u64,
    pub(crate) evidence_confidence_bp: u64,
    pub(crate) abstain_recommended: bool,
}

#[derive(Default)]
pub(crate) struct RouteResult {
    pub(crate) session_scores: HashMap<String, f32>,
    pub(crate) memory_scores: HashMap<String, f32>,
}

#[derive(Default)]
pub(crate) struct Candidates {
    pub(crate) primary_hnsw_raw: Vec<(u64, f32)>,
    pub(crate) semantic_ranked_lists: Vec<(f32, Vec<RankedItem>)>,
    pub(crate) fts_ranked_lists: Vec<(f32, Vec<RankedItem>)>,
    pub(crate) card_ranked_items: Vec<RankedItem>,
    pub(crate) neural_scores: HashMap<String, f32>,
}

#[derive(Default)]
pub(crate) struct Fused {
    pub(crate) items: Vec<(String, u64, f32)>,
}

#[derive(Default)]
pub(crate) struct ScoringContext {
    pub(crate) graph_scores: HashMap<String, f32>,
    pub(crate) observations: HashMap<String, AgentObservation>,
    pub(crate) memory_cards: HashMap<String, MemoryCard>,
    pub(crate) invalidated_facts: HashSet<String>,
}

#[derive(Default)]
pub(crate) struct QueryPipelineData {
    pub(crate) raw_query_text: String,
    pub(crate) query_text: String,
    pub(crate) include_evidence: bool,
    pub(crate) verify_evidence: bool,
    pub(crate) proof_mode: String,
    pub(crate) evidence_radius: u32,
    pub(crate) plan: QueryPlan,
    pub(crate) primary_qembed: Vec<f32>,
    pub(crate) budget: plan::RetrievalBudget,
    pub(crate) semantic_top: usize,
    pub(crate) fts_top: usize,
    pub(crate) adaptive_profile: QueryAdaptiveProfile,
    pub(crate) diag: QueryDiagnostics,
    pub(crate) route: RouteResult,
    pub(crate) candidates: Candidates,
    pub(crate) fused: Fused,
    pub(crate) scoring: ScoringContext,
}

pub(crate) struct QueryPipelineState {
    pub(crate) payload: QueryPayload,
    pub(crate) state: EngineState,
    pub(crate) tenant: Arc<TenantStore>,
    pub(crate) limit: usize,
    pub(crate) enable_neural_rerank: bool,
    pub(crate) weights: ScoringWeights,
    pub(crate) total_start: Instant,
    pub(crate) route_start: Instant,
    pub(crate) now_ms: u64,
    pub(crate) data: QueryPipelineData,
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
    pub(crate) fn new(
        payload: QueryPayload,
        state: EngineState,
        tenant: Arc<TenantStore>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_result_starts_empty() {
        let result = RouteResult::default();
        assert!(result.session_scores.is_empty());
        assert!(result.memory_scores.is_empty());
    }

    #[test]
    fn candidates_keep_stage_outputs_separate() {
        let candidates = Candidates { primary_hnsw_raw: vec![(4, 0.2)], ..Default::default() };
        assert_eq!(candidates.primary_hnsw_raw, vec![(4, 0.2)]);
        assert!(candidates.semantic_ranked_lists.is_empty());
    }

    #[test]
    fn fused_has_explicit_output_storage() {
        let fused = Fused { items: vec![(String::from("m"), 3, 0.8)] };
        assert_eq!(fused.items[0].0, "m");
        assert_eq!(fused.items[0].2, 0.8);
    }

    #[test]
    fn scoring_context_starts_without_hydrated_data() {
        let context = ScoringContext::default();
        assert!(context.graph_scores.is_empty());
        assert!(context.observations.is_empty());
        assert!(context.memory_cards.is_empty());
        assert!(context.invalidated_facts.is_empty());
    }
}
