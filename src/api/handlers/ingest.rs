use axum::http::{HeaderMap, HeaderValue};
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::api::auth::{
    authorize_request, principal_namespace_prefix, principal_user_id, record_usage_for_principal,
    scope_entity_id,
};
use crate::api::ingest_utils::*;
use crate::api::types::{BatchIngestPayload, IngestPayload};
use crate::api::utils::*;
use crate::api::{EngineState, PlatformWriteOp};
use crate::graph::EdgeType;
use crate::ml::cosine_similarity;
use std::sync::Arc;

/// Compute a deterministic content hash for dedup.
/// Uses the text content, entity_id, and kind so that re-ingesting
/// the same factual statement is naturally idempotent.
fn content_hash(text: &str, entity_id: &str, kind: &str) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(text.as_bytes());
    hasher.update(b"::");
    hasher.update(entity_id.as_bytes());
    hasher.update(b"::");
    hasher.update(kind.as_bytes());
    let result = hasher.finalize();
    result.iter().map(|b| format!("{:02x}", b)).collect::<String>()
}
type RetrospectiveCandidate = (String, String, String, u64, String, String);
use crate::features::Feature;
use crate::lifecycle::{evaluate_lifecycle, LifecycleMetadata};
use crate::metrics;
use crate::storage::{
    build_session_router_text, AgentObservation, FactVersionStatus, GraphEdgeEntry, MemoryCard,
    MemoryKind, SessionRouterRecord, TenantStore,
};

#[derive(Default)]
pub(crate) struct IngestDiagnostics {
    input_count: usize,
    expanded_count: usize,
    embedded_count: usize,
    /// Records built per derived structure (see `crate::features`).
    structure_counts: std::collections::BTreeMap<&'static str, usize>,

    expand_ms: u64,
    enrich_ms: u64,
    embed_spec_prep_ms: u64,
    embed_ms: u64,
    embed_us: u64,
    dedup_build_ms: u64,
    storage_ms: u64,
    storage_us: u64,
    analytics_ms: u64,
    analytics_us: u64,
    artifact_build_ms: u64,
    derived_embed_ms: u64,
    derived_embed_us: u64,
    memory_cards_ms: u64,
    session_router_ms: u64,
    fts_ms: u64,
    fts_us: u64,
    vector_ms: u64,
    vector_us: u64,
    graph_ms: u64,
    graph_us: u64,
    preferences_ms: u64,
    retrospective_ms: u64,
    memory_links_ms: u64,
    fact_ms: u64,
    predicate_canon_ms: u64,
    fact_us: u64,
    card_latest_ms: u64,
    total_ms: u64,
    total_us: u64,
}

impl IngestDiagnostics {
    pub(crate) fn expanded_count(&self) -> usize {
        self.expanded_count
    }

    pub(crate) fn embedded_count(&self) -> usize {
        self.embedded_count
    }

    pub(crate) fn total_ms(&self) -> u64 {
        self.total_ms
    }

    fn count(&mut self, structure: &'static str, n: usize) {
        if n > 0 {
            *self.structure_counts.entry(structure).or_default() += n;
        }
    }

    /// `inputs=4,expanded=12,embedded=10,gist=1,...` for `x-tm-ingest-counts`.
    pub(crate) fn counts_header(&self) -> String {
        let mut parts = vec![
            format!("inputs={}", self.input_count),
            format!("expanded={}", self.expanded_count),
            format!("embedded={}", self.embedded_count),
        ];
        parts.extend(self.structure_counts.iter().map(|(k, v)| format!("{k}={v}")));
        parts.join(",")
    }

    fn log_table(&self) {
        let sum = self.expand_ms
            + self.enrich_ms
            + self.embed_spec_prep_ms
            + self.embed_ms
            + self.dedup_build_ms
            + self.storage_ms
            + self.analytics_ms
            + self.artifact_build_ms
            + self.memory_cards_ms
            + self.session_router_ms
            + self.fts_ms
            + self.vector_ms
            + self.graph_ms
            + self.preferences_ms
            + self.retrospective_ms
            + self.memory_links_ms
            + self.fact_ms
            + self.predicate_canon_ms
            + self.card_latest_ms;
        let indent_us = self.total_us % 1000;

        let rows: Vec<(&str, u64)> = vec![
            ("expand + companions", self.expand_ms),
            ("context enrichment", self.enrich_ms),
            ("embed spec preparation", self.embed_spec_prep_ms),
            ("main embedding", self.embed_ms),
            ("dedup + observation build", self.dedup_build_ms),
            ("observation insert (sqlite)", self.storage_ms),
            ("analytics processing", self.analytics_ms),
            ("artifact building (per-rec)", self.artifact_build_ms),
            ("memory card upserts", self.memory_cards_ms),
            ("session router + embed", self.session_router_ms),
            ("FTS indexing", self.fts_ms),
            ("vector index inserts", self.vector_ms),
            ("graph upsert + aliases", self.graph_ms),
            ("preference storage", self.preferences_ms),
            ("retrospective links", self.retrospective_ms),
            ("memory link storage", self.memory_links_ms),
            ("predicate grouping", self.predicate_canon_ms),
            ("fact supersession", self.fact_ms),
            ("card latest updates", self.card_latest_ms),
        ];

        tracing::debug!(
            "\n═══════════════════ Ingest Profile ═══════════════════\n\
             inputs: {} → expanded: {}\n\
             ───────────────────────────────────────────────\n\
             {:<38} {:>10}\n\
             ───────────────────────────────────────────────{}",
            self.input_count,
            self.expanded_count,
            "step",
            "time (ms)",
            rows.iter()
                .map(|(label, ms)| format!("\n{:<38} {:>10}", label, *ms))
                .collect::<Vec<_>>()
                .concat()
        );

        tracing::debug!(
            "───────────────────────────────────────────────\n\
             {:<38} {:>10} ms\n\
             {:<38} {:>8}.{:03} ms\n\
             ═════════════════════════════════════════════════",
            "TOTAL (sum)",
            sum,
            "TOTAL (wall clock)",
            self.total_ms,
            indent_us,
        );
    }
}

#[derive(Clone)]
pub struct ConsolidationTask {
    pub entity_id: String,
    pub memory_id: String,
    pub timestamp: u64,
    pub textual_content: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum EmbeddingMode {
    Standard,
}

#[derive(Clone)]
struct FactRegistration {
    entity_id: String,
    fact_key: String,
    timestamp: u64,
    memory_id: String,
    subject: String,
    predicate: String,
    object: String,
}

struct PreparedRecord {
    payload: IngestPayload,
    obs: AgentObservation,
    embedding: Vec<f32>,
    lifecycle: LifecycleMetadata,
    enable_consolidation: bool,
}

#[derive(Default)]
struct ArtifactBatches {
    fts_batch: Vec<(String, String, String)>,
    vector_batch: std::collections::HashMap<String, Vec<(u64, Vec<f32>)>>,
    memory_links_batch: Vec<(String, String, String)>,
    memory_card_batch: Vec<MemoryCard>,
    memory_card_latest_updates: Vec<(String, bool, u64)>,
    session_router_updates: Vec<SessionRouterRecord>,
    preference_batch: std::collections::HashMap<String, Vec<(String, f32)>>,
    retrospective_candidates: Vec<RetrospectiveCandidate>,
    fact_batch: Vec<FactRegistration>,
    consolidation_tasks: Vec<ConsolidationTask>,
}

pub async fn ingest_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(payload): Json<IngestPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let ns_prefix = principal_namespace_prefix(&principal);
    let profile_text = payload.textual_content.clone();
    let profile_ts = payload.timestamp;

    let tenant_id = principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|e| {
        tracing::warn!("Failed to get tenant store: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let mut payload = payload;
    payload.entity_id = scope_entity_id(&payload.entity_id, ns_prefix.as_deref());
    if !payload.memory_id.starts_with(ns_prefix.as_deref().unwrap_or("")) {
        if let Some(ref p) = ns_prefix {
            payload.memory_id = format!("{}{}", p, payload.memory_id);
        }
    }

    let (tasks, diag) = process_ingest_batch(&state, &tenant, vec![payload]).await?;

    spawn_consolidation_tasks(tenant.clone(), tasks);

    let mut headers = HeaderMap::new();
    insert_stage_timing_headers(&mut headers, "x-tm-embed", diag.embed_ms, diag.embed_us);
    insert_stage_timing_headers(
        &mut headers,
        "x-tm-derived-embed",
        diag.derived_embed_ms,
        diag.derived_embed_us,
    );
    insert_stage_timing_headers(&mut headers, "x-tm-storage", diag.storage_ms, diag.storage_us);
    insert_stage_timing_headers(&mut headers, "x-tm-fts", diag.fts_ms, diag.fts_us);
    insert_stage_timing_headers(&mut headers, "x-tm-vector", diag.vector_ms, diag.vector_us);
    insert_stage_timing_headers(&mut headers, "x-tm-graph", diag.graph_ms, diag.graph_us);
    insert_stage_timing_headers(&mut headers, "x-tm-fact", diag.fact_ms, diag.fact_us);
    insert_stage_timing_headers(
        &mut headers,
        "x-tm-analytics",
        diag.analytics_ms,
        diag.analytics_us,
    );
    insert_stage_timing_headers(&mut headers, "x-tm-total", diag.total_ms, diag.total_us);
    if let Ok(value) = HeaderValue::from_str(&diag.counts_header()) {
        headers.insert("x-tm-ingest-counts", value);
    }

    if let Some(user_id) = principal_user_id(&principal) {
        std::mem::drop(state.platform_write_tx.send(PlatformWriteOp::Profile {
            user_id: user_id.to_string(),
            text: profile_text,
            timestamp_ms: profile_ts,
            source: "ingest".to_string(),
        }));
    }
    record_usage_for_principal(&state, &principal, "ingest");
    metrics::increment_ingest();
    Ok((StatusCode::CREATED, headers))
}

pub async fn batch_ingest_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(payload): Json<BatchIngestPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let ns_prefix = principal_namespace_prefix(&principal);
    let profile_items = payload
        .items
        .iter()
        .map(|item| (item.textual_content.clone(), item.timestamp))
        .collect::<Vec<_>>();

    let tenant_id = principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|e| {
        tracing::warn!("Failed to get tenant store: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let mut payload = payload;
    for item in payload.items.iter_mut() {
        item.entity_id = scope_entity_id(&item.entity_id, ns_prefix.as_deref());
        if !item.memory_id.starts_with(ns_prefix.as_deref().unwrap_or("")) {
            if let Some(ref p) = ns_prefix {
                item.memory_id = format!("{}{}", p, item.memory_id);
            }
        }
    }

    let (tasks, diag) = process_ingest_batch(&state, &tenant, payload.items).await?;

    spawn_consolidation_tasks(tenant.clone(), tasks);

    let mut headers = HeaderMap::new();
    insert_stage_timing_headers(&mut headers, "x-tm-embed", diag.embed_ms, diag.embed_us);
    insert_stage_timing_headers(
        &mut headers,
        "x-tm-derived-embed",
        diag.derived_embed_ms,
        diag.derived_embed_us,
    );
    insert_stage_timing_headers(&mut headers, "x-tm-storage", diag.storage_ms, diag.storage_us);
    insert_stage_timing_headers(&mut headers, "x-tm-fts", diag.fts_ms, diag.fts_us);
    insert_stage_timing_headers(&mut headers, "x-tm-vector", diag.vector_ms, diag.vector_us);
    insert_stage_timing_headers(&mut headers, "x-tm-graph", diag.graph_ms, diag.graph_us);
    insert_stage_timing_headers(&mut headers, "x-tm-fact", diag.fact_ms, diag.fact_us);
    insert_stage_timing_headers(
        &mut headers,
        "x-tm-analytics",
        diag.analytics_ms,
        diag.analytics_us,
    );
    insert_stage_timing_headers(&mut headers, "x-tm-total", diag.total_ms, diag.total_us);
    if let Ok(value) = HeaderValue::from_str(&diag.counts_header()) {
        headers.insert("x-tm-ingest-counts", value);
    }

    if let Some(user_id) = principal_user_id(&principal) {
        for (text, timestamp_ms) in profile_items {
            std::mem::drop(state.platform_write_tx.send(PlatformWriteOp::Profile {
                user_id: user_id.to_string(),
                text,
                timestamp_ms,
                source: "ingest".to_string(),
            }));
        }
    }
    record_usage_for_principal(&state, &principal, "ingest");
    metrics::increment_ingest();
    Ok((StatusCode::CREATED, headers))
}

pub(crate) async fn process_ingest_batch(
    state: &EngineState,
    tenant: &std::sync::Arc<TenantStore>,
    payloads: Vec<IngestPayload>,
) -> Result<(Vec<ConsolidationTask>, IngestDiagnostics), StatusCode> {
    execute_ingest_pipeline(state, tenant, payloads).await
}

/// Cosine similarity above which two predicate wordings are treated as the
/// same predicate (`TELLODB_PREDICATE_TAU`, default 0.86).
fn predicate_canon_tau() -> f32 {
    static TAU: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *TAU.get_or_init(|| {
        std::env::var("TELLODB_PREDICATE_TAU")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|v: &f32| v.is_finite() && (0.0..=1.0).contains(v))
            .unwrap_or(0.86)
    })
}

/// Session of a (normalized) payload, if any.
fn payload_session(payload: &IngestPayload) -> Option<String> {
    payload.session_id.clone().filter(|s| !s.is_empty())
}

// ── Phase 1: Payload preparation ──
fn expand_and_enrich_payloads(
    payloads: Vec<IngestPayload>,
    diag: &mut IngestDiagnostics,
    total_start: Instant,
) -> (Vec<IngestPayload>, Vec<String>) {
    let stage_start = Instant::now();

    let features = crate::features::features();
    let mut expanded_payloads = Vec::new();
    for mut payload in payloads {
        normalize_payload_identity(&mut payload);
        let mut prefix = String::new();
        if let Some(ref desc) = payload.visual_description {
            if !desc.is_empty() {
                prefix.push_str(&format!("[Visual: {}] ", desc));
            }
        }
        if let Some(ref q) = payload.visual_query {
            if !q.is_empty() && payload.visual_description.as_ref() != Some(q) {
                prefix.push_str(&format!("[Context: {}] ", q));
            }
        }
        if !prefix.is_empty() {
            payload.textual_content = format!("{}{}", prefix, payload.textual_content);
        }

        let units = if features.enabled(Feature::Chunks) {
            expand_payload_for_content_type(&payload)
        } else {
            vec![payload.clone()]
        };
        if units.len() > 1 {
            diag.count("chunks", units.len());
        }
        for chunked_payload in units {
            expanded_payloads.push(chunked_payload.clone());
            if payload.enable_mining.unwrap_or(true) {
                let tag_prefix = format!("{}::", chunked_payload.memory_id);
                for mut companion in build_companion_payloads(&chunked_payload) {
                    let feature = companion
                        .memory_id
                        .strip_prefix(&tag_prefix)
                        .and_then(Feature::for_companion_tag);
                    if let Some(feature) = feature {
                        if !features.enabled(feature) {
                            continue;
                        }
                        diag.count(feature.name(), 1);
                    }
                    // Derived records belong to the same session and turn as
                    // their source.
                    companion.session_id = payload.session_id.clone();
                    companion.turn_index = payload.turn_index;
                    companion.role = payload.role.clone();
                    expanded_payloads.push(companion);
                }
            }
        }
    }

    diag.expand_ms = stage_start.elapsed().as_millis() as u64;
    diag.expanded_count = expanded_payloads.len();
    tracing::debug!("[CP] expand_done: μs={}", total_start.elapsed().as_micros());

    // Context window enrichment
    let mut session_groups: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (idx, payload) in expanded_payloads.iter().enumerate() {
        if let Some(sid) = payload_session(payload) {
            session_groups.entry(sid).or_default().push(idx);
        }
    }

    let stage_start = Instant::now();
    let mut enriched_texts = Vec::with_capacity(expanded_payloads.len());
    let legacy_headers = crate::api::ingest::embed_text::embed_text_config().mode
        == crate::api::ingest::embed_text::EmbedTextMode::Legacy;

    for (idx, payload) in expanded_payloads.iter().enumerate() {
        if !legacy_headers || payload.kind.as_deref() == Some("synthetic_query") {
            enriched_texts.push(payload.textual_content.clone());
            continue;
        }

        let mut final_text = payload.textual_content.clone();

        if let Some(sid) = payload_session(payload) {
            if let Some(session_indices) = session_groups.get(&sid) {
                let my_pos = session_indices.iter().position(|&i| i == idx);
                if let Some(pos) = my_pos {
                    let prev_text = if pos > 0 {
                        Some(expanded_payloads[session_indices[pos - 1]].textual_content.as_str())
                    } else {
                        None
                    };
                    let next_text = if pos + 1 < session_indices.len() {
                        Some(expanded_payloads[session_indices[pos + 1]].textual_content.as_str())
                    } else {
                        None
                    };
                    let context_header = build_context_header(prev_text, next_text);
                    if !context_header.is_empty() {
                        final_text = format!("{}{}", context_header, final_text);
                    }
                }
            }
        }

        enriched_texts.push(final_text);
    }
    diag.enrich_ms = stage_start.elapsed().as_millis() as u64;

    (expanded_payloads, enriched_texts)
}

// ── Phase 2: Embedding generation ──
async fn generate_embeddings(
    state: &EngineState,
    expanded_payloads: &[IngestPayload],
    enriched_texts: &[String],
    diag: &mut IngestDiagnostics,
) -> Result<Vec<Vec<f32>>, StatusCode> {
    let spec_prep_start = Instant::now();
    let semantic_embed_specs: Vec<(EmbeddingMode, String)> = expanded_payloads
        .iter()
        .enumerate()
        .filter(|(_, payload)| payload.index_semantic.unwrap_or(true))
        .map(|(idx, _)| (EmbeddingMode::Standard, enriched_texts[idx].clone()))
        .collect();

    let mut unique_specs = Vec::new();
    let mut spec_to_idx = std::collections::HashMap::new();
    for spec in &semantic_embed_specs {
        if !spec_to_idx.contains_key(spec) {
            spec_to_idx.insert(spec.clone(), unique_specs.len());
            unique_specs.push(spec.clone());
        }
    }
    diag.embed_spec_prep_ms = spec_prep_start.elapsed().as_millis() as u64;

    let unique_embeddings = if unique_specs.is_empty() {
        Vec::new()
    } else {
        let stage_start = Instant::now();
        let texts: Vec<String> = unique_specs.iter().map(|(_, t)| t.clone()).collect();
        diag.embedded_count += texts.len();
        // A failed embedding must fail the ingest: storing placeholder vectors
        // would make those memories silently unsearchable.
        let embeddings = state.semantic.embed_texts_async(texts).await.map_err(|err| {
            tracing::error!(error = ?err, "embedding failed; rejecting ingest batch");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        (diag.embed_ms, diag.embed_us) = elapsed_ms_and_us(stage_start);
        embeddings
    };

    let mut semantic_embeddings = Vec::with_capacity(semantic_embed_specs.len());
    for spec in semantic_embed_specs {
        let idx = *spec_to_idx.get(&spec).expect("embedding spec not found in index");
        semantic_embeddings.push(unique_embeddings[idx].clone());
    }

    Ok(semantic_embeddings)
}

// ── Phase 3: Dedup + observation building ──
fn build_observations(
    tenant: &TenantStore,
    expanded_payloads: Vec<IngestPayload>,
    semantic_embeddings: Vec<Vec<f32>>,
    diag: &mut IngestDiagnostics,
) -> Result<Vec<PreparedRecord>, StatusCode> {
    let dedup_build_start = Instant::now();

    let mut prepared = Vec::new();
    let mut semantic_seen: Vec<(String, Vec<f32>)> = Vec::new();
    let mut semantic_embedding_iter = semantic_embeddings.into_iter();

    for payload in expanded_payloads.into_iter() {
        let mut payload = payload;
        let kind = parse_kind(payload.kind.as_deref());
        if (kind == MemoryKind::Preference
            || kind == MemoryKind::Decision
            || kind == MemoryKind::Fact)
            && payload.fact_key.as_ref().map_or(true, |k| k.trim().is_empty())
        {
            if let Some(inferred_key) =
                crate::api::plan::infer_query_fact_key(&payload.textual_content)
            {
                payload.fact_key = Some(inferred_key);
            }
        }
        let index_semantic = payload.index_semantic.unwrap_or(true);
        let embedding = if index_semantic {
            semantic_embedding_iter.next().unwrap_or_default()
        } else {
            Vec::new()
        };
        let enable_semantic_dedup = payload.enable_semantic_dedup.unwrap_or(true);
        let enable_consolidation = payload.enable_consolidation.unwrap_or(true);
        let is_inference = payload
            .fact_operation
            .as_deref()
            .map(|op| {
                let op = op.to_ascii_lowercase();
                op == "derive" || op == "infer"
            })
            .unwrap_or(matches!(kind, MemoryKind::Lesson));
        // Retention runs from when the memory is stored, not from the event
        // time it describes: importing a two-year-old conversation must not
        // make it expire immediately.
        let recorded_at_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(payload.timestamp)
            .max(payload.timestamp);
        let lifecycle = evaluate_lifecycle(
            &payload.textual_content,
            kind,
            recorded_at_ms,
            payload.fact_confidence,
            is_inference,
        );
        let embedding = if lifecycle.index_vector { embedding } else { Vec::new() };

        if index_semantic
            && lifecycle.index_vector
            && enable_semantic_dedup
            && crate::features::enabled(Feature::SemanticDedup)
            && (kind == MemoryKind::Fact || kind == MemoryKind::Decision)
        {
            let vectors = tenant.vectors().map_err(|err| {
                tracing::error!(error = ?err, "tenant vector index unavailable");
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
            let is_dup = is_semantic_duplicate(vectors, &payload.entity_id, &embedding, 0.94)?;
            let in_batch = semantic_seen.iter().any(|(entity_id, prior_embedding)| {
                entity_id == &payload.entity_id
                    && cosine_similarity(prior_embedding, &embedding) >= 0.94
            });
            if is_dup || in_batch {
                continue;
            }
            semantic_seen.push((payload.entity_id.clone(), embedding.clone()));
        }

        let hash =
            content_hash(&payload.textual_content, &payload.entity_id, &format!("{:?}", kind));
        let obs = AgentObservation {
            entity_id: payload.entity_id.clone(),
            textual_content: payload.textual_content.clone(),
            embedding: embedding.clone(),
            kind,
            content_hash: hash,
            created_at_ms: payload.timestamp,
            session_id: payload.session_id.clone().unwrap_or_default(),
            turn_index: payload.turn_index.unwrap_or(0),
            role: payload.role.clone().unwrap_or_default(),
            parent_memory_id: payload.source_memory_id.clone(),
        };

        prepared.push(PreparedRecord { payload, obs, embedding, lifecycle, enable_consolidation });
    }

    diag.dedup_build_ms = dedup_build_start.elapsed().as_millis() as u64;
    Ok(prepared)
}

// ── Phase 4: Artifact building ──
fn build_artifacts(
    prepared: Vec<PreparedRecord>,
    inserted_flags: Vec<Option<u64>>,
    diag: &mut IngestDiagnostics,
) -> ArtifactBatches {
    let artifact_build_start = Instant::now();
    let features = crate::features::features();
    let mut batches = ArtifactBatches::default();

    for (record, vector_id) in prepared.into_iter().zip(inserted_flags.into_iter()) {
        let is_synthetic_query = record.payload.kind.as_deref() == Some("synthetic_query");
        if !is_synthetic_query {
            batches.fts_batch.push((
                record.payload.memory_id.clone(),
                record.payload.entity_id.clone(),
                record.payload.textual_content.clone(),
            ));
            if features.enabled(Feature::MemoryCards) {
                if let Some(card) = build_memory_card_from_payload(
                    &record.payload,
                    record.obs.kind,
                    &record.lifecycle,
                ) {
                    batches.memory_card_batch.push(card);
                }
            }
            if features.enabled(Feature::SessionRouter) {
                if let Some(router_update) =
                    build_session_router_update_from_payload(&record.payload, record.obs.kind)
                {
                    batches.session_router_updates.push(router_update);
                }
            }
        }
        if !record.embedding.is_empty() {
            if let Some(vid) = vector_id {
                batches
                    .vector_batch
                    .entry(record.payload.entity_id.clone())
                    .or_default()
                    .push((vid, record.embedding.clone()));
            }
        }

        if !is_synthetic_query && features.enabled(Feature::Preferences) {
            if let Some(strength) = preference_signal_strength(
                &record.payload.textual_content,
                &record.payload.relations,
            ) {
                batches
                    .preference_batch
                    .entry(record.payload.entity_id.clone())
                    .or_default()
                    .push((record.payload.memory_id.clone(), strength));
            }
        }
        if !is_synthetic_query && features.enabled(Feature::RetrospectiveLinks) {
            if let Some(reference_query) =
                extract_retrospective_reference_query(&record.payload.textual_content)
            {
                batches.retrospective_candidates.push((
                    record.payload.entity_id.clone(),
                    record
                        .payload
                        .source_memory_id
                        .clone()
                        .unwrap_or_else(|| record.payload.memory_id.clone()),
                    record.payload.memory_id.clone(),
                    record.payload.timestamp,
                    reference_query,
                    record.payload.textual_content.clone(),
                ));
            }
        }

        let derived_source = record
            .payload
            .source_memory_id
            .as_deref()
            .filter(|_| features.enabled(Feature::DerivedLinks));
        if let Some(source_memory_id) = derived_source {
            batches.memory_links_batch.push((
                record.payload.memory_id.clone(),
                source_memory_id.to_string(),
                EdgeType::DerivedFrom.as_str().to_string(),
            ));
            batches.memory_links_batch.push((
                source_memory_id.to_string(),
                record.payload.memory_id.clone(),
                EdgeType::DerivedVariant.as_str().to_string(),
            ));
        }

        if features.enabled(Feature::Facts)
            && matches!(
                record.obs.kind,
                MemoryKind::Fact | MemoryKind::Preference | MemoryKind::Decision
            )
        {
            if let Some(fact_key) = record.payload.fact_key.as_deref() {
                batches.fact_batch.push(FactRegistration {
                    entity_id: record.payload.entity_id.clone(),
                    fact_key: fact_key.to_string(),
                    timestamp: record.payload.timestamp,
                    memory_id: record.payload.memory_id.clone(),
                    subject: record
                        .payload
                        .fact_subject
                        .clone()
                        .unwrap_or_else(|| record.payload.entity_id.clone()),
                    predicate: record
                        .payload
                        .fact_predicate
                        .clone()
                        .unwrap_or_else(|| fact_key.to_string()),
                    object: record
                        .payload
                        .fact_object
                        .clone()
                        .unwrap_or_else(|| normalize_fact_text(&record.payload.textual_content)),
                });
            }
        }

        if record.enable_consolidation && features.enabled(Feature::Consolidation) {
            batches.consolidation_tasks.push(ConsolidationTask {
                entity_id: record.payload.entity_id.clone(),
                memory_id: record.payload.memory_id.clone(),
                timestamp: record.payload.timestamp,
                textual_content: record.payload.textual_content.clone(),
            });
        }
    }

    diag.count(Feature::MemoryCards.name(), batches.memory_card_batch.len());
    diag.count(Feature::SessionRouter.name(), batches.session_router_updates.len());
    diag.count(Feature::Preferences.name(), batches.preference_batch.values().map(Vec::len).sum());
    diag.count(Feature::DerivedLinks.name(), batches.memory_links_batch.len());
    diag.count(Feature::Facts.name(), batches.fact_batch.len());
    diag.artifact_build_ms = artifact_build_start.elapsed().as_millis() as u64;
    tracing::debug!("[CP] artifact_build_done: μs={}", artifact_build_start.elapsed().as_micros());
    batches
}

// ── Phase 5: Storage commit ──
async fn commit_batches(
    tenant: &std::sync::Arc<TenantStore>,
    state: &EngineState,
    batches: &mut ArtifactBatches,
    diag: &mut IngestDiagnostics,
) -> Result<(), StatusCode> {
    // Memory cards
    if !batches.memory_card_batch.is_empty() {
        let stage_start = Instant::now();
        let (tenant, cards) = (tenant.clone(), batches.memory_card_batch.clone());
        tokio::task::spawn_blocking(move || tenant.ingest_cards(&cards))
            .await
            .map_err(anyhow::Error::from)
            .and_then(|r| r)
            .map_err(|err| {
                tracing::error!(error = ?err, "memory card upsert failed");
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        diag.memory_cards_ms = stage_start.elapsed().as_millis() as u64;
    }

    tracing::debug!("[CP] upserts_done_before_session_router");

    // Session router merge
    if !batches.session_router_updates.is_empty() {
        let sr_start = Instant::now();
        let router_records = {
            let tenant = tenant.clone();
            let updates = batches.session_router_updates.clone();
            tokio::task::spawn_blocking(move || tenant.merge_session_router_records_batch(&updates))
                .await
                .map_err(|e| {
                    tracing::error!("session_router spawn panic: {:?}", e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?
                .map_err(|e| {
                    tracing::error!("session_router merge failed: {:?}", e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?
        };

        for record in &router_records {
            if !record.router_text.is_empty() {
                batches.fts_batch.push((
                    format!("{}::{}::0::router", record.entity_id, record.session_id),
                    record.entity_id.clone(),
                    record.router_text.clone(),
                ));
            }
        }

        diag.session_router_ms = sr_start.elapsed().as_millis() as u64;
    }

    // FTS + vector indexing (parallel). Failures fail the request: the rows
    // are already committed, and a 200 here would hide memories that can
    // never be retrieved.
    let (res_fts, res_vix) = tokio::join!(
        tokio::task::spawn_blocking({
            let tenant = tenant.clone();
            let batch = batches.fts_batch.clone();
            move || -> anyhow::Result<Duration> {
                let start = Instant::now();
                if !batch.is_empty() {
                    tenant.fts_index_batch(&batch)?;
                }
                Ok(start.elapsed())
            }
        }),
        tokio::task::spawn_blocking({
            let tenant = tenant.clone();
            let batch = batches.vector_batch.clone();
            move || -> anyhow::Result<Duration> {
                let start = Instant::now();
                if !batch.is_empty() {
                    let vectors = tenant.vectors()?;
                    for (entity_id, items) in batch {
                        vectors.insert_batch(&entity_id, &items)?;
                    }
                }
                Ok(start.elapsed())
            }
        })
    );
    let join_stage =
        |stage: &'static str, res: Result<anyhow::Result<Duration>, tokio::task::JoinError>| {
            res.map_err(anyhow::Error::from).and_then(|r| r).map_err(|err| {
                tracing::error!(stage, error = ?err, "ingest indexing stage failed");
                StatusCode::INTERNAL_SERVER_ERROR
            })
        };
    let res_fts = join_stage("fts", res_fts)?;
    let res_vix = join_stage("vector", res_vix)?;

    diag.fts_ms = res_fts.as_millis() as u64;
    diag.fts_us = res_fts.as_micros() as u64;
    diag.vector_ms = res_vix.as_millis() as u64;
    diag.vector_us = res_vix.as_micros() as u64;

    // Preferences
    if !batches.preference_batch.is_empty() {
        let stage_start = Instant::now();
        let tenant_pref = tenant.clone();
        let pref_batch = batches.preference_batch.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            for (entity_id, items) in pref_batch {
                tenant_pref.set_preference_memories_batch(&entity_id, &items)?;
            }
            Ok(())
        })
        .await
        .map_err(|e| {
            tracing::error!("preference spawn panic: {:?}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .map_err(|e| {
            tracing::error!("preference write failed: {:?}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        diag.preferences_ms = stage_start.elapsed().as_millis() as u64;
    }

    // Retrospective links
    if !batches.retrospective_candidates.is_empty() {
        let stage_start = Instant::now();
        let retrospective_links =
            build_retrospective_links(state, tenant, &batches.retrospective_candidates)?;
        diag.count(Feature::RetrospectiveLinks.name(), retrospective_links.len());
        batches.memory_links_batch.extend(retrospective_links);
        diag.retrospective_ms = stage_start.elapsed().as_millis() as u64;
    }

    // Memory links
    if !batches.memory_links_batch.is_empty() {
        let stage_start = Instant::now();
        let tenant_links = tenant.clone();
        let links = batches.memory_links_batch.clone();
        tokio::task::spawn_blocking(move || tenant_links.set_memory_links_batch(&links))
            .await
            .map_err(|e| {
                tracing::error!("memory_links spawn panic: {:?}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .map_err(|e| {
                tracing::error!("memory_links write failed: {:?}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        diag.memory_links_ms = stage_start.elapsed().as_millis() as u64;
    }

    // Predicate grouping: rule-derived keys vary in wording ("job title" vs
    // "job_title"), which would otherwise keep two chains for one fact.
    if crate::features::enabled(Feature::PredicateCanon) && !batches.fact_batch.is_empty() {
        let stage_start = Instant::now();
        let mut keys: Vec<(String, String)> =
            batches.fact_batch.iter().map(|f| (f.entity_id.clone(), f.fact_key.clone())).collect();
        keys.sort();
        keys.dedup();
        let texts: Vec<String> = keys.iter().map(|(_, key)| key.replace('_', " ")).collect();
        let embeddings = state.semantic.embed_texts_async(texts).await.map_err(|err| {
            tracing::error!(error = ?err, "predicate embedding failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        let mut by_entity: std::collections::HashMap<String, Vec<(String, Vec<f32>)>> =
            std::collections::HashMap::new();
        for ((entity_id, key), embedding) in keys.into_iter().zip(embeddings) {
            by_entity.entry(entity_id).or_default().push((key, embedding));
        }
        let tenant_canon = tenant.clone();
        let canonical = tokio::task::spawn_blocking(move || {
            let tau = predicate_canon_tau();
            let mut canonical = std::collections::HashMap::new();
            for (entity_id, predicates) in by_entity {
                let assigned =
                    tenant_canon.canonicalize_predicates(&entity_id, &predicates, tau)?;
                for (predicate, group) in assigned {
                    canonical.insert((entity_id.clone(), predicate), group);
                }
            }
            Ok::<_, anyhow::Error>(canonical)
        })
        .await
        .map_err(anyhow::Error::from)
        .and_then(|r| r)
        .map_err(|err| {
            tracing::error!(error = ?err, "predicate canonicalization failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        let mut regrouped = 0usize;
        for fact in batches.fact_batch.iter_mut() {
            if let Some(group) = canonical
                .get(&(fact.entity_id.clone(), fact.fact_key.clone()))
                .filter(|g| **g != fact.fact_key)
            {
                fact.predicate = fact.fact_key.replace('_', " ");
                fact.fact_key = group.clone();
                regrouped += 1;
            }
        }
        diag.count(Feature::PredicateCanon.name(), regrouped);
        diag.predicate_canon_ms = stage_start.elapsed().as_millis() as u64;
    }

    // Fact registration + supersession
    // Clone fact data for typed graph edges before drain consumes it.
    let facts_for_edges: Vec<FactRegistration> = batches.fact_batch.clone();
    if !batches.fact_batch.is_empty() {
        let stage_start = Instant::now();
        let mut by_entity: std::collections::HashMap<String, Vec<FactRegistration>> =
            std::collections::HashMap::new();
        for item in batches.fact_batch.drain(..) {
            by_entity.entry(item.entity_id.clone()).or_default().push(item);
        }

        #[derive(Default)]
        struct FactSideEffects {
            card_updates: Vec<(String, bool, u64)>,
        }

        let tenant_fact = tenant.clone();
        let fact_side_effects: Vec<Result<FactSideEffects, StatusCode>> =
            tokio::task::spawn_blocking(move || {
                by_entity
                    .into_iter()
                    .map(|(entity_id, registrations)| {
                        let mut se = FactSideEffects::default();
                        let items: Vec<(&str, u64, &str, &str, &str, &str)> = registrations
                            .iter()
                            .map(|r| {
                                (
                                    r.fact_key.as_str(),
                                    r.timestamp,
                                    r.memory_id.as_str(),
                                    r.subject.as_str(),
                                    r.predicate.as_str(),
                                    r.object.as_str(),
                                )
                            })
                            .collect();
                        let statuses = tenant_fact
                            .register_fact_versions_batch(&entity_id, &items)
                            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

                        let mut graph_status_batch = Vec::new();

                        for (status, reg) in statuses.iter().zip(registrations.iter()) {
                            match status {
                                FactVersionStatus::Current { superseded: Some((_, old_id)) } => {
                                    se.card_updates.push((old_id.clone(), false, reg.timestamp));
                                    se.card_updates.push((
                                        reg.memory_id.clone(),
                                        true,
                                        reg.timestamp,
                                    ));
                                    graph_status_batch.push(GraphEdgeEntry {
                                        memory_id: reg.memory_id.as_str(),
                                        subject: reg.subject.as_str(),
                                        predicate: reg.predicate.as_str(),
                                        object: reg.object.as_str(),
                                        status: "current",
                                        ref_info: Some((
                                            EdgeType::Supersedes.as_str(),
                                            old_id.as_str(),
                                        )),
                                        timestamp: reg.timestamp,
                                    });
                                    graph_status_batch.push(GraphEdgeEntry {
                                        memory_id: old_id.as_str(),
                                        subject: reg.subject.as_str(),
                                        predicate: reg.predicate.as_str(),
                                        object: reg.object.as_str(),
                                        status: "stale",
                                        ref_info: Some((
                                            EdgeType::SupersededBy.as_str(),
                                            reg.memory_id.as_str(),
                                        )),
                                        timestamp: reg.timestamp,
                                    });
                                }
                                FactVersionStatus::Stale { current: (_, cur_id) } => {
                                    se.card_updates.push((
                                        reg.memory_id.clone(),
                                        false,
                                        reg.timestamp,
                                    ));
                                    graph_status_batch.push(GraphEdgeEntry {
                                        memory_id: reg.memory_id.as_str(),
                                        subject: reg.subject.as_str(),
                                        predicate: reg.predicate.as_str(),
                                        object: reg.object.as_str(),
                                        status: "stale",
                                        ref_info: Some((
                                            EdgeType::SupersededBy.as_str(),
                                            cur_id.as_str(),
                                        )),
                                        timestamp: reg.timestamp,
                                    });
                                }
                                // A restatement confirms the version it
                                // matches; nothing about the chain changed.
                                FactVersionStatus::Confirmed { .. } => {}
                                FactVersionStatus::Current { superseded: None } => {
                                    se.card_updates.push((
                                        reg.memory_id.clone(),
                                        true,
                                        reg.timestamp,
                                    ));
                                    graph_status_batch.push(GraphEdgeEntry {
                                        memory_id: reg.memory_id.as_str(),
                                        subject: reg.subject.as_str(),
                                        predicate: reg.predicate.as_str(),
                                        object: reg.object.as_str(),
                                        status: "current",
                                        ref_info: None,
                                        timestamp: reg.timestamp,
                                    });
                                }
                            }
                        }

                        if !graph_status_batch.is_empty() {
                            tenant_fact
                                .graph_upsert_fact_status_batch(&entity_id, &graph_status_batch)
                                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                        }

                        Ok(se)
                    })
                    .collect()
            })
            .await
            .map_err(|e| {
                tracing::error!("fact supersession spawn panic: {:?}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        for result in fact_side_effects {
            match result {
                Ok(se) => {
                    batches.memory_card_latest_updates.extend(se.card_updates);
                }
                Err(e) => return Err(e),
            }
        }

        (diag.fact_ms, diag.fact_us) = elapsed_ms_and_us(stage_start);
    }

    // Typed graph edges from memory cards and fact registrations; they feed
    // the entity-graph retrieval lane.
    if crate::features::enabled(Feature::GraphEdges) {
        let stage_start = Instant::now();
        let tenant = tenant.clone();
        let cards = batches.memory_card_batch.clone();
        let written = tokio::task::spawn_blocking(move || {
            let edges: Vec<(&str, &str, &str, &str, u64)> = cards
                .iter()
                .map(|c| {
                    (
                        c.source_memory_id.as_str(),
                        c.subject.as_str(),
                        c.predicate.as_str(),
                        c.object.as_str(),
                        c.created_at_ms,
                    )
                })
                .chain(facts_for_edges.iter().map(|f| {
                    (
                        f.memory_id.as_str(),
                        f.subject.as_str(),
                        f.predicate.as_str(),
                        f.object.as_str(),
                        f.timestamp,
                    )
                }))
                .collect();
            tenant.graph_insert_edges_batch(&edges)
        })
        .await
        .map_err(anyhow::Error::from)
        .and_then(|r| r)
        .map_err(|err| {
            tracing::error!(error = ?err, "typed graph edge insert failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        diag.count(Feature::GraphEdges.name(), written);
        (diag.graph_ms, diag.graph_us) = elapsed_ms_and_us(stage_start);
    }

    // Card latest updates
    if !batches.memory_card_latest_updates.is_empty() {
        let stage_start = Instant::now();
        let tenant_cl = tenant.clone();
        let updates = batches.memory_card_latest_updates.clone();
        tokio::task::spawn_blocking(move || tenant_cl.set_memory_card_latest_batch(&updates))
            .await
            .map_err(|e| {
                tracing::error!("card_latest spawn panic: {:?}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .map_err(|e| {
                tracing::error!("card_latest write failed: {:?}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        diag.card_latest_ms = stage_start.elapsed().as_millis() as u64;
    }

    Ok(())
}

fn build_memory_card_from_payload(
    payload: &IngestPayload,
    kind: MemoryKind,
    lifecycle: &LifecycleMetadata,
) -> Option<MemoryCard> {
    let source_memory_id =
        payload.source_memory_id.clone().unwrap_or_else(|| payload.memory_id.clone());
    let source_session_id = payload
        .session_id
        .clone()
        .filter(|s| !s.is_empty())
        .or_else(|| session_id_from_memory_id(&source_memory_id))
        .unwrap_or_default();
    let source_turn_index = payload
        .turn_index
        .map(|t| t as usize)
        .unwrap_or_else(|| turn_index_from_memory_id(&source_memory_id));
    let document_time = extract_document_time_ms(&payload.textual_content, payload.timestamp);
    let event_time = extract_event_time_ms(&payload.textual_content, document_time);
    let memory_text = normalize_fact_text(&payload.textual_content);
    if memory_text.is_empty() {
        return None;
    }

    let subject = payload
        .fact_subject
        .clone()
        .or_else(|| {
            extract_named_phrases(std::slice::from_ref(&payload.textual_content)).into_iter().next()
        })
        .unwrap_or_else(|| payload.entity_id.clone());
    let predicate = payload
        .fact_predicate
        .clone()
        .or_else(|| payload.fact_key.as_ref().map(|key| key.replace('_', " ")))
        .unwrap_or_else(|| match kind {
            MemoryKind::Preference => "prefers".to_string(),
            MemoryKind::Decision => "decided".to_string(),
            MemoryKind::SessionSummary => "summarizes".to_string(),
            MemoryKind::Lesson => "learned".to_string(),
            MemoryKind::Fact => "states".to_string(),
            MemoryKind::Conversational => "mentions".to_string(),
        });
    let object =
        payload.fact_object.clone().unwrap_or_else(|| truncate_router_value(&memory_text, 320));
    let operation = payload.fact_operation.as_deref().unwrap_or_default().to_ascii_lowercase();
    let confidence = payload
        .fact_confidence
        .unwrap_or(match kind {
            MemoryKind::Conversational => 0.78,
            MemoryKind::SessionSummary => 0.84,
            MemoryKind::Fact | MemoryKind::Preference | MemoryKind::Decision => 0.92,
            MemoryKind::Lesson => 0.86,
        })
        .clamp(0.0, 1.0);
    let card_type = card_type_for_kind(kind, &memory_text);
    let is_static = is_static_profile_card(&predicate, &memory_text, kind);

    Some(MemoryCard {
        card_id: payload.memory_id.clone(),
        entity_id: payload.entity_id.clone(),
        user_id: payload.entity_id.clone(),
        source_memory_id,
        source_session_id,
        source_turn_index,
        document_time,
        conversation_time: document_time,
        event_time,
        subject,
        predicate,
        object,
        memory_text,
        card_type,
        confidence,
        is_latest: true,
        is_static,
        is_inference: operation == "derive" || operation == "infer",
        expires_at: None,
        root_card_id: payload.source_memory_id.clone(),
        parent_card_id: payload.source_memory_id.clone(),
        lifecycle: Some(lifecycle.clone()),
        created_at_ms: payload.timestamp,
        updated_at_ms: payload.timestamp,
    })
}

fn card_type_for_kind(kind: MemoryKind, text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    match kind {
        MemoryKind::Preference => "preference",
        MemoryKind::Decision => "decision",
        MemoryKind::Fact => "fact",
        MemoryKind::Lesson => "inference",
        MemoryKind::SessionSummary => {
            if lower.contains("canonical event memory") || !extract_temporal_terms(text).is_empty()
            {
                "event"
            } else {
                "episode"
            }
        }
        MemoryKind::Conversational => "episode",
    }
    .to_string()
}

fn is_static_profile_card(predicate: &str, text: &str, kind: MemoryKind) -> bool {
    if matches!(kind, MemoryKind::Preference | MemoryKind::Decision) {
        return true;
    }
    let lower = format!("{} {}", predicate, text).to_ascii_lowercase();
    [
        "identity",
        "occupation",
        "job",
        "works at",
        "family",
        "married",
        "spouse",
        "children",
        "lives in",
        "health",
        "allergy",
        "prefers",
        "favorite",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn build_session_router_update_from_payload(
    payload: &IngestPayload,
    kind: MemoryKind,
) -> Option<SessionRouterRecord> {
    let session_id = payload_session(payload)?;
    let document_time_ms = extract_document_time_ms(&payload.textual_content, payload.timestamp);
    let session_date = extract_bracketed_header_value(&payload.textual_content, "Session Date")
        .unwrap_or_else(|| "unknown".to_string());
    let session_focus = extract_bracketed_header_value(&payload.textual_content, "Session Focus")
        .unwrap_or_default();
    let dialogue = extract_dialogue_messages(&payload.textual_content);
    let dialogue_texts = if dialogue.is_empty() {
        vec![payload.textual_content.clone()]
    } else {
        dialogue.iter().map(|(speaker, line)| format!("{speaker}: {line}")).collect::<Vec<_>>()
    };
    let speakers = dedupe_preserve_order(
        dialogue
            .iter()
            .map(|(speaker, _)| speaker.clone())
            .filter(|speaker| !speaker.is_empty())
            .collect(),
    );
    let persons = extract_named_phrases(&dialogue_texts);
    let salient = extract_salient_terms(&payload.textual_content, 18);
    let lower = payload.textual_content.to_ascii_lowercase();
    let compact_text = truncate_router_value(&normalize_fact_text(&payload.textual_content), 360);

    let mut canonical_facts = Vec::new();
    let mut events = Vec::new();
    let mut preference_signals = Vec::new();
    if matches!(kind, MemoryKind::Fact | MemoryKind::Decision | MemoryKind::Lesson) {
        canonical_facts.push(compact_text.clone());
    }
    if matches!(kind, MemoryKind::Preference)
        || preference_signal_strength(&payload.textual_content, &payload.relations).is_some()
    {
        preference_signals.push(compact_text.clone());
    }
    if extract_event_time_ms(&payload.textual_content, document_time_ms).is_some()
        || !extract_temporal_terms(&payload.textual_content).is_empty()
        || ["went", "visited", "watched", "joined", "started", "finished", "won", "bought"]
            .iter()
            .any(|needle| lower.contains(needle))
    {
        events.push(compact_text.clone());
    }

    let mut record = SessionRouterRecord {
        session_id,
        entity_id: payload.entity_id.clone(),
        session_date,
        document_time_ms,
        speakers,
        persons,
        session_focus,
        canonical_facts,
        events,
        objects: Vec::new(),
        places: Vec::new(),
        activities: Vec::new(),
        preference_signals,
        salient_terms: salient,
        source_memory_ids: vec![payload.memory_id.clone()],
        router_text: String::new(),
        created_at_ms: payload.timestamp,
        updated_at_ms: payload.timestamp,
    };
    record.router_text = build_session_router_text(&record);
    Some(record)
}

fn truncate_router_value(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        text.chars().take(max_chars).collect::<String>().trim().to_string()
    }
}

fn build_retrospective_links(
    state: &EngineState,
    tenant: &TenantStore,
    candidates: &[RetrospectiveCandidate],
) -> Result<Vec<(String, String, String)>, StatusCode> {
    let mut links = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for (
        entity_id,
        source_memory_id,
        current_memory_id,
        current_timestamp,
        reference_query,
        current_text,
    ) in candidates
    {
        let mut score_by_memory: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        let mut timestamp_by_memory: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();

        let fts_hits = tenant.fts_search(reference_query, 8, Some(entity_id)).unwrap_or_default();
        for (rank, (memory_id, lexical_score)) in fts_hits.into_iter().enumerate() {
            if memory_id == *current_memory_id || memory_id == *source_memory_id {
                continue;
            }
            if let Some((ts, _)) = tenant.lookup_by_memory_id(&memory_id).unwrap_or(None) {
                timestamp_by_memory.insert(memory_id.clone(), ts);
                *score_by_memory.entry(memory_id).or_insert(0.0) +=
                    0.45 + lexical_score.min(1.0) * 0.10 - rank as f32 * 0.02;
            }
        }

        let query_embedding = ok_or_500(state.semantic.generate_query_embedding(reference_query))?;
        let ann_hits = ok_or_500(
            tenant.vectors().and_then(|v| v.search(Some(entity_id), &query_embedding, 10)),
        )?;
        let ann_ids: Vec<u64> = ann_hits.iter().map(|(vid, _)| *vid).collect();
        let ann_lookup = ok_or_500(tenant.lookup_by_vector_ids_batch(&ann_ids))?;
        for (rank, ((_, dist), maybe_lookup)) in
            ann_hits.iter().zip(ann_lookup.into_iter()).enumerate()
        {
            let Some((ts, memory_id)) = maybe_lookup else {
                continue;
            };
            if memory_id == *current_memory_id || memory_id == *source_memory_id {
                continue;
            }
            timestamp_by_memory.insert(memory_id.clone(), ts);
            *score_by_memory.entry(memory_id).or_insert(0.0) +=
                (1.0 - *dist).clamp(0.0, 1.0) * 0.70 - rank as f32 * 0.02;
        }

        let mut ranked = score_by_memory.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0))
        });

        if let Some((target_memory_id, score)) = ranked.into_iter().find(|(memory_id, _)| {
            timestamp_by_memory
                .get(memory_id)
                .copied()
                .map(|ts| ts < *current_timestamp)
                .unwrap_or(false)
        }) {
            if score >= 0.55 {
                let target_text = tenant
                    .lookup_by_memory_id(&target_memory_id)
                    .ok()
                    .flatten()
                    .and_then(|(ts, _)| {
                        tenant
                            .get_observation(ts, &target_memory_id)
                            .ok()
                            .flatten()
                            .map(|obs| obs.textual_content)
                    })
                    .unwrap_or_default();
                let (forward_type, reverse_type) = classify_retrospective_link(
                    current_text,
                    &target_text,
                    *current_timestamp,
                    timestamp_by_memory.get(&target_memory_id).copied().unwrap_or(0),
                );
                let forward =
                    (source_memory_id.clone(), target_memory_id.clone(), forward_type.to_string());
                let reverse =
                    (target_memory_id.clone(), source_memory_id.clone(), reverse_type.to_string());
                if seen.insert(forward.clone()) {
                    links.push(forward);
                }
                if seen.insert(reverse.clone()) {
                    links.push(reverse);
                }
            }
        }
    }

    Ok(links)
}

fn classify_retrospective_link(
    current_text: &str,
    target_text: &str,
    current_timestamp: u64,
    target_timestamp: u64,
) -> (&'static str, &'static str) {
    let current_lower = current_text.to_ascii_lowercase();
    let target_lower = target_text.to_ascii_lowercase();
    let contradiction_markers = ["actually", "turns out", "not ", "never ", "instead", "wrong"];
    if contradiction_markers
        .iter()
        .any(|marker| current_lower.contains(marker) && !target_lower.contains(marker))
    {
        return ("contradicts", "contradicted_by");
    }

    let current_temporal = extract_temporal_terms(current_text).len();
    let target_temporal = extract_temporal_terms(target_text).len();
    let current_numbers = current_text.chars().filter(|c| c.is_ascii_digit()).count();
    let target_numbers = target_text.chars().filter(|c| c.is_ascii_digit()).count();
    if current_temporal + current_numbers > target_temporal + target_numbers {
        return ("clarifies", "clarified_by");
    }

    let current_entities = extract_named_phrases(&[current_text.to_string()]);
    let target_entities = extract_named_phrases(&[target_text.to_string()]);
    let current_only =
        current_entities.iter().filter(|entity| !target_entities.contains(entity)).count();
    if current_only >= 1 || current_timestamp > target_timestamp {
        return ("extends", "extended_by");
    }

    ("recalls", "recalled_by")
}

pub(crate) fn spawn_consolidation_tasks(tenant: Arc<TenantStore>, tasks: Vec<ConsolidationTask>) {
    if tasks.is_empty() {
        return;
    }
    // One blocking task per batch; each profile update is its own transaction.
    tokio::task::spawn_blocking(move || {
        for task in &tasks {
            if let Err(err) = update_core_profile_heuristic(&tenant, task) {
                tracing::warn!(entity_id = %task.entity_id, error = ?err, "core profile update failed");
            }
        }
    });
}

/// Keeps the most recent facts per entity, ordered by fact time. Replaced
/// facts are filtered when the profile is read (`current_core_profile`),
/// because supersession may be recorded after this update runs.
fn update_core_profile_heuristic(
    tenant: &TenantStore,
    task: &ConsolidationTask,
) -> anyhow::Result<()> {
    const MAX_PROFILE_FACTS: usize = 24;
    let excerpt = truncate_router_value(&normalize_fact_text(&task.textual_content), 280);
    if excerpt.is_empty() {
        return Ok(());
    }
    tenant.update_core_profile(&task.entity_id, |current| {
        let mut profile = current
            .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
            .unwrap_or_else(|| {
                serde_json::json!({
                    "schema": "heuristic_core_profile_v1",
                    "entity_id": task.entity_id,
                    "facts": [],
                    "updated_at_ms": task.timestamp
                })
            });
        let facts = profile.get_mut("facts")?.as_array_mut()?;
        let already = facts
            .iter()
            .any(|item| item.get("memory_id").and_then(|v| v.as_str()) == Some(&task.memory_id));
        if already {
            return None;
        }
        facts.push(serde_json::json!({
            "memory_id": task.memory_id,
            "timestamp_ms": task.timestamp,
            "text": excerpt,
            "terms": extract_salient_terms(&task.textual_content, 6)
        }));
        // Keep the newest facts by fact time, not by arrival order.
        facts.sort_by_key(|f| std::cmp::Reverse(f.get("timestamp_ms").and_then(|t| t.as_u64())));
        facts.truncate(MAX_PROFILE_FACTS);
        let latest = facts.first().and_then(|f| f.get("timestamp_ms")).cloned();
        if let Some(latest) = latest {
            profile["updated_at_ms"] = latest;
        }
        serde_json::to_string(&profile).ok()
    })
}

async fn execute_ingest_pipeline(
    state: &EngineState,
    tenant: &std::sync::Arc<TenantStore>,
    payloads: Vec<IngestPayload>,
) -> Result<(Vec<ConsolidationTask>, IngestDiagnostics), StatusCode> {
    let mut diag = IngestDiagnostics::default();
    let total_start = Instant::now();

    if payloads.is_empty() {
        return Ok((Vec::new(), diag));
    }

    diag.input_count = payloads.len();

    // Numeric memory is extracted from the memories as sent, not from chunks
    // or derived companions (which restate the same numbers).
    let metric_sources: Vec<(String, String, u64, String)> = payloads
        .iter()
        .filter(|p| p.kind.as_deref() != Some("synthetic_query"))
        .filter(|_| crate::features::enabled(Feature::Metrics))
        .map(|p| (p.entity_id.clone(), p.memory_id.clone(), p.timestamp, p.textual_content.clone()))
        .collect();

    // Phase 1: Payload preparation
    let (mut expanded_payloads, mut enriched_texts) =
        expand_and_enrich_payloads(payloads, &mut diag, total_start);

    // Embedding text: `legacy` keeps the batch-neighbour header built above;
    // `turn` / `context` build text from the turn itself (and stored
    // neighbouring turns), independent of how requests are batched.
    let embed_config = crate::api::ingest::embed_text::embed_text_config();
    let mut neighbour_updates = Vec::new();
    if embed_config.mode != crate::api::ingest::embed_text::EmbedTextMode::Legacy {
        let (tenant_for_text, payloads_for_text) = (tenant.clone(), expanded_payloads.clone());
        let built = tokio::task::spawn_blocking(move || {
            crate::api::ingest::embed_text::build_embed_texts(
                &tenant_for_text,
                &payloads_for_text,
                embed_config,
            )
        })
        .await
        .map_err(anyhow::Error::from)
        .and_then(|r| r)
        .map_err(|err| {
            tracing::error!(error = ?err, "building embedding texts failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        enriched_texts = built.texts;
        neighbour_updates = built.neighbour_updates;
    }

    // Skip re-sends: a memory id already stored with identical content. (The
    // previous check matched content alone, dropping distinct memories that
    // repeat earlier text, e.g. the same statement on another day.)
    {
        let stage_start = Instant::now();
        let ids: Vec<String> = expanded_payloads.iter().map(|p| p.memory_id.clone()).collect();
        let stored = tenant.stored_content_hashes(&ids).map_err(|err| {
            tracing::error!(error = ?err, "content hash lookup failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        let keep: Vec<bool> = expanded_payloads
            .iter()
            .map(|p| {
                let hash = content_hash(
                    &p.textual_content,
                    &p.entity_id,
                    &format!("{:?}", parse_kind(p.kind.as_deref())),
                );
                stored.get(&p.memory_id) != Some(&hash)
            })
            .collect();
        let mut kept = keep.iter();
        expanded_payloads.retain(|_| *kept.next().expect("aligned"));
        let mut kept = keep.iter();
        enriched_texts.retain(|_| *kept.next().expect("aligned"));
        diag.dedup_build_ms += stage_start.elapsed().as_millis() as u64;
    }

    // Phase 2: Embedding
    let semantic_embeddings =
        generate_embeddings(state, &expanded_payloads, &enriched_texts, &mut diag).await?;

    tracing::debug!("[CP] embed_done: μs={}", total_start.elapsed().as_micros());

    // Phase 3: Dedup + observation building
    let prepared = build_observations(tenant, expanded_payloads, semantic_embeddings, &mut diag)?;

    let batch_items: Vec<(u64, String, AgentObservation)> = prepared
        .iter()
        .map(|record| {
            (record.payload.timestamp, record.payload.memory_id.clone(), record.obs.clone())
        })
        .collect();

    tracing::debug!("[CP] dedup_done: μs={}", total_start.elapsed().as_micros());

    let stage_start = Instant::now();
    let inserted_flags = tenant.insert_observations_batch(&batch_items).map_err(|e| {
        tracing::warn!("Batch Storage Error: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    (diag.storage_ms, diag.storage_us) = elapsed_ms_and_us(stage_start);

    // Phase 4: Artifact building
    let mut batches = build_artifacts(prepared, inserted_flags, &mut diag);

    // Phase 5: Storage commit
    commit_batches(tenant, state, &mut batches, &mut diag).await?;

    // Stored neighbours whose context window changed get re-embedded.
    if !neighbour_updates.is_empty() {
        let (ids, texts): (Vec<String>, Vec<String>) = neighbour_updates.into_iter().unzip();
        diag.count("context_reembeds", texts.len());
        diag.embedded_count += texts.len();
        let embeddings = state.semantic.embed_texts_async(texts).await.map_err(|err| {
            tracing::error!(error = ?err, "neighbour re-embedding failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        let tenant = tenant.clone();
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let applied =
                tenant.update_embeddings(&ids.into_iter().zip(embeddings).collect::<Vec<_>>())?;
            let vectors = tenant.vectors()?;
            let mut by_entity: std::collections::HashMap<String, Vec<(u64, Vec<f32>)>> =
                std::collections::HashMap::new();
            for (vector_id, entity_id, embedding) in applied {
                by_entity.entry(entity_id).or_default().push((vector_id, embedding));
            }
            for (entity_id, items) in by_entity {
                vectors.insert_batch(&entity_id, &items)?;
            }
            Ok(())
        })
        .await
        .map_err(anyhow::Error::from)
        .and_then(|r| r)
        .map_err(|err| {
            tracing::error!(error = ?err, "neighbour vector update failed");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    }

    // Phase 6: Numeric memory
    let stage_start = Instant::now();
    tokio::task::spawn_blocking({
        let (analytics, tenant) = (state.analytics.clone(), tenant.clone());
        move || -> anyhow::Result<()> {
            for (entity_id, memory_id, timestamp, text) in &metric_sources {
                analytics.record_memory(&tenant, entity_id, memory_id, *timestamp, text)?;
            }
            Ok(())
        }
    })
    .await
    .map_err(anyhow::Error::from)
    .and_then(|r| r)
    .map_err(|err| {
        tracing::error!(error = ?err, "metric extraction failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    (diag.analytics_ms, diag.analytics_us) = elapsed_ms_and_us(stage_start);

    (diag.total_ms, diag.total_us) = elapsed_ms_and_us(total_start);
    tracing::debug!("[CP] final: μs={}", total_start.elapsed().as_micros());
    if tracing::enabled!(tracing::Level::DEBUG) {
        diag.log_table();
    }
    Ok((batches.consolidation_tasks, diag))
}
