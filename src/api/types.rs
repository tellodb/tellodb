use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct RankedItem {
    pub memory_id: String,
    pub timestamp: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(default)]
#[non_exhaustive]
pub struct RankingConfig {
    /// Weight for lexical (FTS/bm25) similarity score in final ranking. Default: 0.05
    pub lexical_weight: f32,
    /// Weight for entity match coverage in final ranking. Default: 0.05
    pub entity_weight: f32,
    /// Weight for temporal recency/relevance in final ranking. Default: 0.05
    pub temporal_weight: f32,
    /// Multiplicative boost applied when a memory card is present as an evidence card. Default: 1.15
    pub card_boost: f32,
    /// Additive boost for results whose source session matches a routed session. Default: 0.05
    pub session_boost: f32,
    /// Weight for the session-router ANN similarity score. Default: 0.12
    pub session_ann_weight: f32,
    /// Weight for temporal event search hit scores. Default: 0.18
    pub event_weight: f32,
    /// Weight for shadow question search hit scores. Default: 0.12
    pub shadow_weight: f32,
    /// Weight for facet posting match coverage. Default: 0.08
    pub facet_weight: f32,
    /// Weight for profile fact match coverage. Default: 0.10
    pub profile_weight: f32,
    /// Weight for graph traversal proximity score. Default: 1.0
    pub graph_weight: f32,
    /// Weight for evidence density (how many distinct supporting sources). Default: 0.03
    pub evidence_density_weight: f32,
    /// Penalty subtracted from score when a card is stale (not latest). Default: -0.08
    pub stale_penalty: f32,
    /// Penalty subtracted when a result contradicts an existing fact. Default: -0.10
    pub contradiction_penalty: f32,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            lexical_weight: 0.05,
            entity_weight: 0.05,
            temporal_weight: 0.05,
            card_boost: 1.15,
            session_boost: 0.05,
            session_ann_weight: 0.12,
            event_weight: 0.18,
            shadow_weight: 0.12,
            facet_weight: 0.08,
            profile_weight: 0.10,
            graph_weight: 1.0,
            evidence_density_weight: 0.03,
            stale_penalty: -0.08,
            contradiction_penalty: -0.10,
        }
    }
}

/// A single piece of evidence cited in a query response, with provenance and scores.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EvidenceCard {
    pub claim_text: String,
    pub source_memory_id: String,
    pub source_session_id: String,
    pub card_id: Option<String>,
    pub semantic_rank: Option<usize>,
    pub semantic_score: f32,
    pub bm25_rank: Option<usize>,
    pub bm25_score: f32,
    pub session_router_rank: Option<usize>,
    pub session_router_score: f32,
    pub card_score: f32,
    pub reranker_score: f32,
    pub entity_hits: usize,
    pub lexical_hits: usize,
    pub temporal_hits: usize,
    pub facet_mask: u64,
    pub graph_score: f32,
    pub child_score: f32,
    pub is_latest: bool,
    pub card_type: String,
    pub final_score: f32,
    // Internal fields for hydration
    #[serde(skip)]
    pub internal_kind: crate::storage::MemoryKind,
    #[serde(skip)]
    pub created_at_ms: u64,
    #[serde(skip)]
    pub entity_id: String,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub auth_required: bool,
    pub device: &'static str,
    pub data_root: String,
}

#[derive(Serialize)]
pub struct EngineStatus {
    pub device: String,
    pub data_root: String,
    pub cache_capacity: usize,
    pub cache_usage: usize,
}

#[derive(Serialize)]
pub struct VersionResponse {
    pub engine_version: &'static str,
    pub api_version: &'static str,
    pub auth_required: bool,
}

#[derive(Serialize)]
pub struct ProbeResponse {
    pub status: &'static str,
}

#[derive(Serialize)]
pub struct WarmupResponse {
    pub status: &'static str,
    pub device: &'static str,
    pub duration_ms: u128,
}

#[derive(Deserialize, Clone, Default)]
pub struct IngestPayload {
    pub entity_id: String,
    pub memory_id: String,
    pub timestamp: u64,
    pub textual_content: String,
    pub relations: Vec<(String, String, String)>,
    /// Optional memory type. Defaults to Conversational.
    pub kind: Option<String>,
    /// Optional internal fact key used for supersession tracking.
    pub fact_key: Option<String>,
    /// Optional internal provenance pointer to the source episodic memory.
    pub source_memory_id: Option<String>,
    /// Optional internal flag. When false, record is stored and FTS-indexed but not embedded/vector-indexed.
    pub index_semantic: Option<bool>,
    /// Optional benchmark override. Defaults to true in production.
    pub enable_semantic_dedup: Option<bool>,
    /// Optional benchmark override. Defaults to true in production.
    pub enable_consolidation: Option<bool>,
    /// Optional content type hint for ingest chunking.
    pub content_type: Option<String>,
    /// Optional fact semantics metadata.
    pub fact_operation: Option<String>,
    pub fact_confidence: Option<f32>,
    pub fact_subject: Option<String>,
    pub fact_predicate: Option<String>,
    pub fact_object: Option<String>,
    /// Optional multi-modal metadata (e.g. blip_caption)
    #[serde(alias = "blip_caption")]
    pub visual_description: Option<String>,
    /// Optional multi-modal metadata (e.g. image query)
    #[serde(alias = "query")]
    pub visual_query: Option<String>,
}

#[derive(Deserialize)]
pub struct BatchIngestPayload {
    pub items: Vec<IngestPayload>,
}

#[derive(Deserialize)]
pub struct AnalyticsQueryPayload {
    pub entity_id: String,
    pub label: String,
    pub start_timestamp_ms: Option<u64>,
    pub end_timestamp_ms: Option<u64>,
    pub bucket: Option<String>,
}

#[derive(Serialize)]
pub struct AnalyticsQueryResult {
    pub entity_id: String,
    pub label: String,
    pub sum: f64,
    pub count: usize,
    pub avg: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buckets: Option<Vec<BucketedResult>>,
}

#[derive(Serialize)]
pub struct BucketedResult {
    pub bucket_start_ms: u64,
    pub bucket_end_ms: u64,
    pub sum: f64,
    pub count: usize,
    pub avg: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
}

#[derive(Serialize, Clone, Debug)]
pub struct QueryResult {
    pub memory_id: String,
    pub entity_id: String,
    pub session_id: String,
    pub turn_index: usize,
    pub created_at_ms: u64,
    pub similarity: f32,
    pub textual_content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<ProofPacket>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_notes: Option<Vec<String>>,
}

/// Proof packet for evidence-verified query responses.
#[derive(Serialize, Clone, Debug)]
pub struct ProofPacket {
    pub proof_mode: String,
    pub verified: bool,
    pub confidence: f32,
    pub source_memory_id: String,
    pub source_session_id: String,
    pub source_turn_index: usize,
    pub supporting_card_ids: Vec<String>,
    pub supporting_event_ids: Vec<String>,
    pub entities_hit: usize,
    pub lexical_hits: usize,
    pub temporal_hits: usize,
    pub missing_facets: Vec<String>,
    pub checks: Vec<ProofCheck>,
    pub source_turns: Vec<ProofTurn>,
}

#[derive(Serialize, Clone, Debug)]
pub struct ProofCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Serialize, Clone, Debug)]
pub struct ProofTurn {
    pub turn_id: String,
    pub session_id: String,
    pub turn_index: u32,
    pub speaker: Option<String>,
    pub text: String,
}

#[derive(Deserialize)]
pub struct QueryPayload {
    pub textual_query: String,
    pub limit: usize,
    /// Optional scope — only return results whose memory_id starts with this entity_id
    pub entity_id: Option<String>,
    /// Optional benchmark override. Defaults to true in production.
    pub enable_neural_rerank: Option<bool>,
    /// Include structured source proof metadata in each result.
    pub include_evidence: Option<bool>,
    /// Run deterministic evidence checks before returning proof metadata.
    pub verify_evidence: Option<bool>,
    /// Proof mode: off, light, or strict.
    pub proof_mode: Option<String>,
    /// Nearby turns to attach per evidence result when proof mode is enabled.
    pub max_evidence_turns_per_session: Option<usize>,
    /// Optional point in time to run the query as-of (milliseconds timestamp).
    pub point_in_time_ms: Option<u64>,
}

#[derive(Deserialize)]
pub struct GraphQueryPayload {
    pub user_id: Option<String>,
    pub subject: String,
    pub edge_type: Option<String>,
    pub direction: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct GraphWalkPayload {
    pub user_id: Option<String>,
    pub node: String,
    pub direction: Option<String>,
    pub edge_types: Option<Vec<String>>,
    pub depth: Option<usize>,
    pub breadth: Option<usize>,
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct GraphExportPayload {
    pub user_id: Option<String>,
    pub seed: String,
    pub direction: Option<String>,
    pub edge_types: Option<Vec<String>>,
    pub depth: Option<usize>,
    pub breadth: Option<usize>,
    pub max_nodes: Option<usize>,
}

#[derive(Deserialize, Default)]
pub struct ResetPayload {
    pub confirm: Option<String>,
    pub clear_embedding_cache: Option<bool>,
}

#[derive(Deserialize)]
pub struct MemoryInspectPayload {
    pub memory_id: String,
    pub include_turn_window: Option<bool>,
    pub turn_window_radius: Option<u32>,
}

#[derive(Deserialize)]
pub struct MemoryDeletePayload {
    pub memory_id: String,
    pub reason: Option<String>,
}

#[derive(Serialize)]
pub struct MemoryAuditObservation {
    pub entity_id: String,
    pub kind: String,
    pub created_at_ms: u64,
    pub textual_content: String,
}

#[derive(Serialize)]
pub struct MemoryInspectResponse {
    pub memory_id: String,
    pub timestamp: Option<u64>,
    pub observation: Option<MemoryAuditObservation>,
    pub card: Option<crate::storage::MemoryCard>,
    pub ledger_turn: Option<crate::storage::LedgerTurn>,
    pub lifecycle: Option<crate::lifecycle::LifecycleMetadata>,
    pub artifacts: Vec<crate::storage::MemoryArtifact>,
    pub artifact_versions: Vec<crate::lifecycle::ArtifactVersionRecord>,
    pub deletion_tombstones: Vec<crate::lifecycle::DeletionTombstone>,
    pub turn_window: Vec<ProofTurn>,
}

#[derive(Serialize)]
pub struct MemoryDeleteResponse {
    pub deleted: bool,
    pub memory_id: String,
    pub timestamp: Option<u64>,
    pub vector_id: Option<u64>,
    pub tombstone: Option<crate::lifecycle::DeletionTombstone>,
    pub fts_removed: usize,
    pub graph_edges_removed: usize,
}

#[derive(Deserialize)]
pub struct PlatformSignupPayload {
    pub username: String,
    pub password: String,
}

#[derive(Deserialize)]
pub struct PlatformLoginPayload {
    pub username: String,
    pub password: String,
}

#[derive(Deserialize)]
pub struct PlatformCreateApiKeyPayload {
    pub name: String,
}

#[derive(Serialize)]
pub struct PlatformAuthResponse {
    pub token: String,
    pub user: crate::platform::PublicUser,
}

#[derive(Serialize)]
pub struct PlatformApiKeyCreateResponse {
    pub api_key: crate::platform::PublicApiKey,
    pub key: String,
}

#[derive(Serialize)]
pub struct PlatformApiKeyListResponse {
    pub api_keys: Vec<crate::platform::PublicApiKey>,
}

#[derive(Serialize)]
pub struct PlatformStatsResponse {
    pub usage: crate::platform::UsageStats,
}

#[derive(Serialize)]
pub struct PlatformProfileResponse {
    pub profile: crate::platform::UserProfile,
}

#[derive(Deserialize)]
pub struct AdminInjectApiKeyPayload {
    pub key_id: String,
    pub user_id: String,
    pub name: String,
    pub token: String,
    pub cluster_id: Option<String>,
}

#[derive(Serialize, Debug, Clone)]
pub struct HardwareStatsResponse {
    pub cpu_usage_percent: f32,
    pub ram_total_mb: u64,
    pub ram_used_mb: u64,
    pub storage_total_gb: u64,
    pub storage_used_gb: u64,
    pub gpu_usage_percent: Option<f32>,
    pub gpu_ram_total_mb: Option<u64>,
    pub gpu_ram_used_mb: Option<u64>,
}

#[derive(Serialize, Debug, Clone)]
pub struct StorageStatsResponse {
    pub memory_card_count: usize,
    pub edge_count: usize,
    pub memory_count: usize,
    pub metric_count: usize,
    pub ledger_turn_count: usize,
    pub memory_artifact_count: usize,
    pub temporal_event_count: usize,
    pub shadow_question_count: usize,
    pub facet_posting_count: usize,
    pub mem_cell_count: usize,
    pub mem_scene_count: usize,
    pub profile_fact_count: usize,
    pub session_router_count: usize,
    pub fact_version_count: usize,
    pub card_relation_count: usize,
    pub memory_link_count: usize,
    pub alias_count: usize,
    pub preference_count: usize,
    pub core_profile_count: usize,
    pub deletion_tombstone_count: usize,
    pub storage_bytes: usize,
}
