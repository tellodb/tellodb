use anyhow::{bail, Result};
use std::env;
use std::path::PathBuf;

use crate::api::types::RankingConfig;
use crate::features::Features;
use crate::heuristics::Profile;
use crate::retrieval::lanes::Lanes;
use crate::retrieval::scoring::ScoringWeights;
use crate::vector_index::VectorConfig;

const DEFAULT_EMBEDDING_MODEL: &str = "BAAI/bge-small-en-v1.5";
const DEFAULT_RERANK_MODEL: &str = "bge-reranker-base";
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 3000;
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;
const DEFAULT_CONTEXT_WINDOW: u32 = 1;
const DEFAULT_PREDICATE_CANON_TAU: f32 = 0.86;
const DEFAULT_EXTRACTOR_THRESHOLD: f32 = 0.5;

#[derive(Debug, Clone)]
pub struct Config {
    pub features: Features,
    pub heuristics: Profile,
    pub lanes: Lanes,
    pub ranking: RankingConfig,
    pub scoring: ScoringWeights,
    pub retrieval: RetrievalConfig,
    pub vector: VectorConfig,
    pub embedding: EmbeddingConfig,
    pub rerank: RerankConfig,
    pub server: ServerConfig,
    pub extractor: ExtractorConfig,
    pub ingest: IngestConfig,
    pub temporal: TemporalConfig,
}

#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    pub profile: RetrievalProfile,
    pub auto_rerank: Option<bool>,
    pub scoped_semantic_top: usize,
    pub scoped_semantic_start: usize,
    pub scoped_semantic_step: usize,
    pub scoped_min_hits: Option<usize>,
    pub scoped_stop_max_attempts: usize,
    pub scoped_stop_min_similarity: f32,
    pub scoped_stop_max_hit_gain: usize,
    pub scoped_stop_min_similarity_gain: f32,
    pub graph_seed_count: usize,
    pub graph_max_depth: usize,
    pub graph_max_node_degree: usize,
    pub latest_recency_weight: f32,
}

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalProfile {
    Fast,
    Balanced,
    Research,
}

impl RetrievalProfile {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "research" | "full" | "v2" => Self::Research,
            "balanced" | "default" => Self::Balanced,
            _ => Self::Fast,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub model_id: String,
    pub model_dir: Option<PathBuf>,
    pub cache_path: Option<PathBuf>,
    pub device: String,
    pub threads: usize,
    pub executors: usize,
    pub max_tokens: usize,
    pub batch: usize,
    pub dimension: Option<usize>,
    pub query_instruction: Option<String>,
    pub cache_enabled: bool,
    pub text: EmbedTextConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedTextMode {
    Legacy,
    Turn,
    Context,
}

impl EmbedTextMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Turn => "turn",
            Self::Context => "context",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedTextConfig {
    pub mode: EmbedTextMode,
    pub window: u32,
}

#[derive(Debug, Clone)]
pub struct RerankConfig {
    pub model: String,
    pub enabled: bool,
    pub executors: usize,
    pub cache_size: usize,
    pub policy: RerankPolicy,
    pub margin: f32,
    pub top: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankPolicy {
    Heuristic,
    Always,
    Gate,
}

impl RerankPolicy {
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "heuristic" => Self::Heuristic,
            "always" => Self::Always,
            _ => Self::Gate,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Heuristic => "heuristic",
            Self::Always => "always",
            Self::Gate => "gate",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub request_timeout_secs: u64,
    pub trust_proxy: bool,
    pub cors_origins: Vec<String>,
    pub api_key: Option<String>,
    pub ml_intent: bool,
}

#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    pub kind: String,
    pub model_dir: Option<PathBuf>,
    pub labels: Vec<String>,
    pub threshold: f32,
}

#[derive(Debug, Clone)]
pub struct IngestConfig {
    pub predicate_canon_tau: f32,
}

#[derive(Debug, Clone)]
pub struct TemporalConfig {
    pub recency_scoring: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            features: Features::default(),
            heuristics: Profile::Generic,
            lanes: Lanes::default(),
            ranking: RankingConfig::default(),
            scoring: ScoringWeights::default(),
            retrieval: RetrievalConfig::default(),
            vector: VectorConfig::new(384),
            embedding: EmbeddingConfig::default(),
            rerank: RerankConfig::default(),
            server: ServerConfig::default(),
            extractor: ExtractorConfig::default(),
            ingest: IngestConfig::default(),
            temporal: TemporalConfig::default(),
        }
    }
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let defaults = Self::default();
        let features = Features::parse(&value("TELLODB_DISABLE", None).unwrap_or_default())?;
        let heuristics = Profile::parse(&value("TELLODB_HEURISTICS", None).unwrap_or_default())?;
        let lanes = Lanes::parse(&value("TELLODB_LANES", None).unwrap_or_default())?;
        let dimensions = usize_value("TELLODB_EMBEDDING_DIM", None).filter(|value| *value > 0);
        let embedding = EmbeddingConfig::from_env(&defaults.embedding, dimensions);
        let vector = VectorConfig::from_values(
            dimensions.unwrap_or(defaults.vector.dimensions),
            value("TELLODB_VECTOR_QUANT", None).as_deref(),
            usize_value("TELLODB_FLAT_THRESHOLD", None),
            usize_value("TELLODB_RESCORE_FACTOR", None),
            usize_value("TELLODB_HNSW_CONNECTIVITY", None),
            usize_value("TELLODB_HNSW_EF_ADD", None),
            usize_value("TELLODB_HNSW_EF_SEARCH", None),
        )?;

        Ok(Self {
            features,
            heuristics,
            lanes,
            ranking: RankingConfig::default(),
            scoring: ScoringWeights::default(),
            retrieval: RetrievalConfig::from_env(&defaults.retrieval),
            vector,
            embedding,
            rerank: RerankConfig::from_env(&defaults.rerank),
            server: ServerConfig::from_env(&defaults.server),
            extractor: ExtractorConfig::from_env(&defaults.extractor)?,
            ingest: IngestConfig::from_env(&defaults.ingest),
            temporal: TemporalConfig::from_env(&defaults.temporal),
        })
    }
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            profile: RetrievalProfile::Fast,
            auto_rerank: None,
            scoped_semantic_top: 3000,
            scoped_semantic_start: 256,
            scoped_semantic_step: 256,
            scoped_min_hits: None,
            scoped_stop_max_attempts: 3,
            scoped_stop_min_similarity: 0.70,
            scoped_stop_max_hit_gain: 2,
            scoped_stop_min_similarity_gain: 0.01,
            graph_seed_count: 24,
            graph_max_depth: 2,
            graph_max_node_degree: 128,
            latest_recency_weight: 0.35,
        }
    }
}

impl RetrievalConfig {
    fn from_env(defaults: &Self) -> Self {
        Self {
            profile: RetrievalProfile::parse(
                &value("TELLODB_RETRIEVAL_PROFILE", Some("TEMPORAL_MEMORY_RETRIEVAL_PROFILE"))
                    .unwrap_or_default(),
            ),
            auto_rerank: bool_value("TELLODB_AUTO_RERANK", Some("TEMPORAL_MEMORY_AUTO_RERANK")),
            scoped_semantic_top: usize_value(
                "TELLODB_SCOPED_SEMANTIC_TOP",
                Some("TEMPORAL_MEMORY_SCOPED_SEMANTIC_TOP"),
            )
            .filter(|value| *value >= 100)
            .unwrap_or(defaults.scoped_semantic_top),
            scoped_semantic_start: usize_value(
                "TELLODB_SCOPED_SEMANTIC_START",
                Some("TEMPORAL_MEMORY_SCOPED_SEMANTIC_START"),
            )
            .filter(|value| *value >= 100)
            .unwrap_or(defaults.scoped_semantic_start),
            scoped_semantic_step: usize_value(
                "TELLODB_SCOPED_SEMANTIC_STEP",
                Some("TEMPORAL_MEMORY_SCOPED_SEMANTIC_STEP"),
            )
            .filter(|value| *value > 0)
            .unwrap_or(defaults.scoped_semantic_step),
            scoped_min_hits: usize_value(
                "TELLODB_SCOPED_MIN_HITS",
                Some("TEMPORAL_MEMORY_SCOPED_MIN_HITS"),
            )
            .filter(|value| *value > 0),
            scoped_stop_max_attempts: usize_value(
                "TELLODB_SCOPED_STOP_MAX_ATTEMPTS",
                Some("TEMPORAL_MEMORY_SCOPED_STOP_MAX_ATTEMPTS"),
            )
            .filter(|value| *value > 0)
            .unwrap_or(defaults.scoped_stop_max_attempts),
            scoped_stop_min_similarity: f32_value(
                "TELLODB_SCOPED_STOP_MIN_SIM",
                Some("TEMPORAL_MEMORY_SCOPED_STOP_MIN_SIM"),
            )
            .filter(|value| (-1.0..=1.0).contains(value))
            .unwrap_or(defaults.scoped_stop_min_similarity),
            scoped_stop_max_hit_gain: usize_value(
                "TELLODB_SCOPED_STOP_MAX_HIT_GAIN",
                Some("TEMPORAL_MEMORY_SCOPED_STOP_MAX_HIT_GAIN"),
            )
            .unwrap_or(defaults.scoped_stop_max_hit_gain),
            scoped_stop_min_similarity_gain: f32_value(
                "TELLODB_SCOPED_STOP_MIN_SIM_GAIN",
                Some("TEMPORAL_MEMORY_SCOPED_STOP_MIN_SIM_GAIN"),
            )
            .filter(|value| *value >= 0.0)
            .unwrap_or(defaults.scoped_stop_min_similarity_gain),
            graph_seed_count: usize_value("TELLODB_GRAPH_SEEDS", None)
                .unwrap_or(defaults.graph_seed_count),
            graph_max_depth: usize_value("TELLODB_GRAPH_MAX_DEPTH", None)
                .unwrap_or(defaults.graph_max_depth),
            graph_max_node_degree: usize_value("TELLODB_GRAPH_MAX_NODE_DEGREE", None)
                .filter(|value| *value > 0)
                .unwrap_or(defaults.graph_max_node_degree),
            latest_recency_weight: f32_value("TELLODB_LATEST_RECENCY_WEIGHT", None)
                .filter(|value| *value >= 0.0)
                .unwrap_or(defaults.latest_recency_weight),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: DEFAULT_EMBEDDING_MODEL.to_string(),
            model_dir: None,
            cache_path: None,
            device: String::new(),
            threads: num_cpus::get().max(1),
            executors: 1,
            max_tokens: 512,
            batch: 32,
            dimension: None,
            query_instruction: None,
            cache_enabled: true,
            text: EmbedTextConfig { mode: EmbedTextMode::Context, window: DEFAULT_CONTEXT_WINDOW },
        }
    }
}

impl EmbeddingConfig {
    fn from_env(defaults: &Self, dimension: Option<usize>) -> Self {
        let model_id = value("TELLODB_EMBEDDING_MODEL", Some("TEMPORAL_MEMORY_EMBEDDING_MODEL"))
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| defaults.model_id.clone());
        let mode = match value("TELLODB_EMBED_TEXT", None)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "legacy" => EmbedTextMode::Legacy,
            "turn" => EmbedTextMode::Turn,
            _ => EmbedTextMode::Context,
        };
        Self {
            model_id,
            model_dir: path_value("TELLODB_MODEL_DIR", None),
            cache_path: path_value("TELLODB_EMBEDDING_CACHE_PATH", None),
            device: value("TELLODB_DEVICE", Some("TEMPORAL_MEMORY_DEVICE")).unwrap_or_default(),
            threads: usize_value("TELLODB_THREADS", None)
                .filter(|value| *value > 0)
                .unwrap_or(defaults.threads),
            executors: usize_value(
                "TELLODB_EMBED_EXECUTORS",
                Some("TEMPORAL_MEMORY_EMBED_EXECUTORS"),
            )
            .filter(|value| *value > 0)
            .unwrap_or(defaults.executors),
            max_tokens: usize_value("TELLODB_EMBED_MAX_TOKENS", None)
                .filter(|value| *value > 0)
                .unwrap_or(defaults.max_tokens),
            batch: usize_value("TELLODB_EMBED_BATCH", None)
                .filter(|value| *value > 0)
                .unwrap_or(defaults.batch),
            dimension,
            query_instruction: value("TELLODB_QUERY_INSTRUCTION", None)
                .filter(|value| !value.trim().is_empty()),
            cache_enabled: !bool_value("TELLODB_EMBED_CACHE", None).is_some_and(|value| !value),
            text: EmbedTextConfig {
                mode,
                window: usize_value("TELLODB_CONTEXT_WINDOW", None)
                    .map(|value| (value as u32).min(4))
                    .unwrap_or(defaults.text.window),
            },
        }
    }
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_RERANK_MODEL.to_string(),
            enabled: true,
            executors: 1,
            cache_size: 4096,
            policy: RerankPolicy::Gate,
            margin: 0.05,
            top: 25,
        }
    }
}

impl RerankConfig {
    fn from_env(defaults: &Self) -> Self {
        let model = value("TELLODB_RERANK_MODEL", None)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| defaults.model.clone());
        let enabled = !bool_value("TELLODB_RERANK", None).is_some_and(|value| !value);
        Self {
            model,
            enabled,
            executors: usize_value(
                "TELLODB_RERANK_EXECUTORS",
                Some("TEMPORAL_MEMORY_RERANK_EXECUTORS"),
            )
            .filter(|value| *value > 0)
            .unwrap_or(defaults.executors),
            cache_size: usize_value(
                "TELLODB_RERANK_CACHE_SIZE",
                Some("TEMPORAL_MEMORY_RERANK_CACHE_SIZE"),
            )
            .unwrap_or(defaults.cache_size),
            policy: RerankPolicy::parse(&value("TELLODB_RERANK_POLICY", None).unwrap_or_default()),
            margin: f32_value("TELLODB_RERANK_MARGIN", None)
                .filter(|value| *value >= 0.0)
                .unwrap_or(defaults.margin),
            top: usize_value("TELLODB_RERANK_TOP", None)
                .filter(|value| (2..=500).contains(value))
                .unwrap_or(defaults.top),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            trust_proxy: false,
            cors_origins: vec!["https://tellodb.com".to_string()],
            api_key: None,
            ml_intent: false,
        }
    }
}

impl ServerConfig {
    fn from_env(defaults: &Self) -> Self {
        let api_key = value("TELLODB_API_KEY", Some("TEMPORAL_MEMORY_API_KEY"))
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let cors_origins =
            value("TELLODB_CORS_ALLOW_ORIGINS", Some("TEMPORAL_MEMORY_CORS_ALLOW_ORIGINS"))
                .map(|value| {
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|origin| !origin.is_empty())
                        .map(str::to_string)
                        .collect()
                })
                .filter(|origins: &Vec<String>| !origins.is_empty())
                .unwrap_or_else(|| defaults.cors_origins.clone());
        Self {
            host: value("TELLODB_HOST", Some("TEMPORAL_MEMORY_HOST"))
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| defaults.host.clone()),
            port: value("TELLODB_PORT", Some("TEMPORAL_MEMORY_PORT"))
                .or_else(|| value("PORT", None))
                .and_then(|value| value.parse().ok())
                .unwrap_or(defaults.port),
            request_timeout_secs: usize_value("TELLODB_REQUEST_TIMEOUT_SECS", None)
                .map(|value| value as u64)
                .unwrap_or(defaults.request_timeout_secs),
            trust_proxy: bool_value("TELLODB_TRUST_PROXY", None).unwrap_or(defaults.trust_proxy),
            cors_origins,
            api_key,
            ml_intent: bool_value("TELLODB_ML_INTENT", Some("TEMPORAL_MEMORY_ML_INTENT"))
                .unwrap_or(defaults.ml_intent),
        }
    }
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            kind: "rules".to_string(),
            model_dir: None,
            labels: Vec::new(),
            threshold: DEFAULT_EXTRACTOR_THRESHOLD,
        }
    }
}

impl ExtractorConfig {
    fn from_env(defaults: &Self) -> Result<Self> {
        let kind = value("TELLODB_EXTRACTOR", None).unwrap_or_else(|| defaults.kind.clone());
        if !matches!(kind.trim(), "rules" | "encoder") {
            bail!("unknown TELLODB_EXTRACTOR '{}' (rules, encoder)", kind.trim());
        }
        let labels = value("TELLODB_EXTRACTOR_LABELS", None)
            .map(|value| {
                value
                    .split(',')
                    .map(|label| label.split_whitespace().collect::<Vec<_>>().join(" "))
                    .filter(|label| !label.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            kind,
            model_dir: path_value("TELLODB_EXTRACTOR_MODEL_DIR", None),
            labels,
            threshold: f32_value("TELLODB_EXTRACTOR_THRESHOLD", None)
                .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
                .unwrap_or(defaults.threshold),
        })
    }
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self { predicate_canon_tau: DEFAULT_PREDICATE_CANON_TAU }
    }
}

impl IngestConfig {
    fn from_env(defaults: &Self) -> Self {
        Self {
            predicate_canon_tau: f32_value("TELLODB_PREDICATE_TAU", None)
                .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
                .unwrap_or(defaults.predicate_canon_tau),
        }
    }
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self { recency_scoring: true }
    }
}

impl TemporalConfig {
    fn from_env(defaults: &Self) -> Self {
        Self {
            recency_scoring: bool_value(
                "TELLODB_ENABLE_TEMPORAL_RECENCY_SCORING",
                Some("TEMPORAL_MEMORY_ENABLE_TEMPORAL_RECENCY_SCORING"),
            )
            .unwrap_or(defaults.recency_scoring),
        }
    }
}

fn value(name: &str, legacy: Option<&str>) -> Option<String> {
    match env::var(name).ok().filter(|value| !value.trim().is_empty()) {
        Some(value) => Some(value),
        None => legacy.and_then(|legacy_name| {
            env::var(legacy_name).ok().filter(|value| !value.trim().is_empty()).inspect(|_| {
                tracing::warn!(
                    canonical = name,
                    legacy = legacy_name,
                    "using legacy configuration variable"
                )
            })
        }),
    }
}

fn usize_value(name: &str, legacy: Option<&str>) -> Option<usize> {
    value(name, legacy).and_then(|value| value.trim().parse().ok())
}

fn f32_value(name: &str, legacy: Option<&str>) -> Option<f32> {
    value(name, legacy).and_then(|value| value.trim().parse().ok())
}

fn bool_value(name: &str, legacy: Option<&str>) -> Option<bool> {
    value(name, legacy).and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    })
}

fn path_value(name: &str, legacy: Option<&str>) -> Option<PathBuf> {
    value(name, legacy).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_documented_runtime_defaults() {
        let config = Config::default();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 3000);
        assert_eq!(config.retrieval.scoped_semantic_top, 3000);
        assert_eq!(config.embedding.max_tokens, 512);
        assert_eq!(config.rerank.top, 25);
    }
}
