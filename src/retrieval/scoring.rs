//! Retrieval scoring weights — all tunable scoring parameters live here so
//! they can be reasoned about, benchmarked, and versioned independently of
//! business logic.

/// Scoring weights and thresholds for the Aletheia retrieval pipeline.
///
/// These defaults are the result of empirical tuning on the LoCoMo benchmark.
/// Values between ±10% produce similar results; changes beyond that should be
/// benchmarked against the evaluation suite before merging.
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Bonus applied per session when the memory falls within the query's time window.
    /// Determined empirically; 0.55 balances precision and recall on LoCoMo.
    pub time_window_bonus: f32,

    /// Minimum cosine similarity for two memories to be considered duplicates.
    /// Below this threshold, both are retained; above it, the older is deduped.
    pub dedup_similarity_threshold: f32,

    /// Multiplier applied to stale (superseded) fact scores.
    /// 0.70 means a stale fact retains 70% of its score.
    pub stale_fact_decay: f32,

    /// Rerank confidence boost when stability score > 0.7.
    pub rerank_confidence_stable: f32,

    /// Rerank confidence boost when confidence score > 0.7.
    pub rerank_confidence_high: f32,

    /// Rerank penalty for stale facts.
    pub rerank_stale_penalty: f32,

    /// Route boost for decomposition/cross-entity queries.
    pub route_boost_hard: f32,

    /// Route boost for simple (non-decomposition) queries.
    pub route_boost_simple: f32,

    /// Route penalty for memories outside routed sessions.
    pub route_penalty: f32,

    /// Four-signal blend: weight of the temporal signal (0.0-1.0).
    /// Remaining weight goes to the legacy composite.
    pub four_signal_temporal_weight: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            time_window_bonus: 0.55,
            dedup_similarity_threshold: 0.50,
            stale_fact_decay: 0.70,
            rerank_confidence_stable: 0.05,
            rerank_confidence_high: 0.03,
            rerank_stale_penalty: -0.15,
            route_boost_hard: 0.075,
            route_boost_simple: 0.035,
            route_penalty: -0.008,
            four_signal_temporal_weight: 0.7,
        }
    }
}
