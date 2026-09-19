use std::collections::HashSet;

use crate::api::types::EvidenceCard;

#[derive(Debug, Clone, Default)]
pub struct QueryPlan {
    pub semantic_queries: Vec<String>,
    pub fts_queries: Vec<String>,
    pub coverage_facets: Vec<CoverageFacet>,
    pub requirements: Vec<QueryRequirement>,
    pub prefer_distilled: bool,
    pub prefer_episodic: bool,
    pub temporal_terms: Vec<String>,
    pub lexical_terms: Vec<String>,
    pub intent: QueryIntent,
    pub subject_entities: Vec<String>,
    pub cross_entity: bool,
    pub needs_decomposition: bool,
    pub coverage_mode: bool,
    pub ordinal_rank: Option<usize>,
    /// Inferred fact_key for direct fact lookup pre-synthesized paths.
    /// Populated from `infer_query_fact_key(query)` in the planner.
    pub fact_key: Option<String>,
    /// The question asks for the present value ("where do I live now?"), so
    /// newer evidence and non-superseded facts should outrank older ones.
    pub prefers_latest: bool,
}

#[derive(Debug, Clone)]
pub struct CoverageFacet {
    pub text: String,
    pub lexical_terms: Vec<String>,
    pub temporal_terms: Vec<String>,
    pub entities: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct QueryRequirement {
    pub text: String,
    pub lexical_terms: Vec<String>,
    pub temporal_terms: Vec<String>,
    pub entities: Vec<String>,
    pub require_all_entities: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueryIntent {
    NumericAggregation,
    TemporalAggregation,
    Recommendation,
    Inference,
    PeripheralMention,
    #[default]
    General,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryModality {
    Semantic,
    Lexical,
}

#[derive(Debug, Clone, Default)]
pub struct QueryAdaptiveProfile {
    pub semantic_scale: f32,
    pub lexical_scale: f32,
    pub route_sessions: HashSet<String>,
    pub route_strength: f32,
}

pub struct FourSignalWeights {
    pub semantic: f32,
    pub temporal: f32,
    pub confidence: f32,
    pub graph: f32,
}

impl FourSignalWeights {
    pub fn for_intent(intent: QueryIntent) -> Self {
        match intent {
            QueryIntent::NumericAggregation => {
                Self { semantic: 0.35, temporal: 0.30, confidence: 0.20, graph: 0.15 }
            }
            QueryIntent::TemporalAggregation => {
                Self { semantic: 0.20, temporal: 0.50, confidence: 0.15, graph: 0.15 }
            }
            QueryIntent::Recommendation => {
                Self { semantic: 0.40, temporal: 0.10, confidence: 0.30, graph: 0.20 }
            }
            QueryIntent::Inference => {
                Self { semantic: 0.30, temporal: 0.15, confidence: 0.30, graph: 0.25 }
            }
            QueryIntent::PeripheralMention => {
                Self { semantic: 0.25, temporal: 0.20, confidence: 0.25, graph: 0.30 }
            }
            QueryIntent::General => {
                Self { semantic: 0.50, temporal: 0.20, confidence: 0.15, graph: 0.15 }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScorableObservation {
    pub lower: String,
    pub tokens: Vec<String>,
    pub temporal_terms: Vec<String>,
    pub entities: Vec<String>,
    pub numeric_tokens: Vec<f32>,
}

impl ScorableObservation {
    pub fn new(text: &str) -> Self {
        use crate::core::calendar::extract_temporal_terms;
        use crate::core::text::{extract_named_phrases, normalize_alpha_tokens};
        let lower = text.to_ascii_lowercase();
        let tokens = normalize_alpha_tokens(text);
        let temporal_terms = extract_temporal_terms(text);
        let entities = extract_named_phrases(&[text.to_string()]);
        let numeric_tokens = super::scoring::extract_numeric_tokens(text);
        Self { lower, tokens, temporal_terms, entities, numeric_tokens }
    }
}

#[derive(Clone)]
pub struct SessionBucket {
    pub score: f32,
    pub items: Vec<EvidenceCard>,
    pub requirement_mask: u64,
    pub facet_mask: u64,
    pub max_entity_hits: usize,
    pub max_lexical_hits: usize,
    pub max_temporal_hits: usize,
}
