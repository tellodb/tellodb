#![allow(dead_code)]
use axum::http::HeaderMap;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
};
use rayon::prelude::*;
use std::time::{Duration, Instant};

use crate::api::auth::{
    authorize_request, principal_namespace_prefix, principal_user_id, record_usage_for_principal,
    scope_entity_id,
};
use crate::api::ingest_utils::*;
use crate::api::types::{BatchIngestPayload, IngestPayload};
use crate::api::utils::*;
use crate::ml::cosine_similarity;
use std::sync::Arc;
use crate::api::{EngineState, PlatformWriteOp};
use crate::graph::EdgeType;

type GraphEdgeRecord = crate::storage::GraphEdgeEntry<'static>;
type EmbeddingPairSet = (Vec<(String, Vec<f32>)>, Vec<(String, Vec<f32>)>);

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
type MiningResult = Result<Option<(String, Vec<f32>, Vec<f32>)>, StatusCode>;
type RetrospectiveCandidate = (String, String, String, u64, String, String);
use crate::lifecycle::{evaluate_lifecycle, LifecycleMetadata};
use crate::metrics;
use crate::storage::{
    build_session_router_text, AgentObservation, CombinedIngestUpsertInput, FacetPosting,
    FactVersionStatus, GraphEdgeEntry, MemCell, MemSceneRecord, MemoryArtifact, MemoryCard,
    MemoryKind, ProfileFact, SessionRouterRecord, ShadowQuestion, TemporalEvent, TenantStore,
};

#[derive(Default)]
struct IngestDiagnostics {
    input_count: usize,
    expanded_count: usize,

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
    ner_ms: u64,
    ner_us: u64,
    derived_embed_ms: u64,
    derived_embed_us: u64,
    memory_cards_ms: u64,
    memory_artifacts_ms: u64,
    temporal_events_ms: u64,
    shadow_questions_ms: u64,
    facet_postings_ms: u64,
    mem_cells_ms: u64,
    mem_scenes_ms: u64,
    profile_facts_ms: u64,
    session_router_ms: u64,
    fts_ms: u64,
    fts_us: u64,
    bm25f_ms: u64,
    vector_ms: u64,
    vector_us: u64,
    hard_negatives_ms: u64,
    graph_ms: u64,
    graph_us: u64,
    preferences_ms: u64,
    retrospective_ms: u64,
    memory_links_ms: u64,
    fact_ms: u64,
    fact_us: u64,
    card_latest_ms: u64,
    card_relations_ms: u64,
    total_ms: u64,
    total_us: u64,
}

impl IngestDiagnostics {
    fn log_table(&self) {
        let sum = self.expand_ms
            + self.enrich_ms
            + self.embed_spec_prep_ms
            + self.embed_ms
            + self.dedup_build_ms
            + self.storage_ms
            + self.analytics_ms
            + self.artifact_build_ms
            + self.ner_ms
            + self.memory_cards_ms
            + self.memory_artifacts_ms
            + self.temporal_events_ms
            + self.shadow_questions_ms
            + self.facet_postings_ms
            + self.mem_cells_ms
            + self.mem_scenes_ms
            + self.profile_facts_ms
            + self.session_router_ms
            + self.fts_ms
            + self.bm25f_ms
            + self.vector_ms
            + self.hard_negatives_ms
            + self.graph_ms
            + self.preferences_ms
            + self.retrospective_ms
            + self.memory_links_ms
            + self.fact_ms
            + self.card_latest_ms
            + self.card_relations_ms;
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
            ("entity resolution", self.ner_ms),
            ("memory card upserts", self.memory_cards_ms),
            ("memory artifact upserts", self.memory_artifacts_ms),
            ("temporal events + embed", self.temporal_events_ms),
            ("shadow questions + embed", self.shadow_questions_ms),
            ("facet posting upserts", self.facet_postings_ms),
            ("memcell upserts", self.mem_cells_ms),
            ("memscene upserts", self.mem_scenes_ms),
            ("profile fact upserts", self.profile_facts_ms),
            ("session router + embed", self.session_router_ms),
            ("FTS indexing", self.fts_ms),
            ("BM25F indexing", self.bm25f_ms),
            ("vector index inserts", self.vector_ms),
            ("hard negative mining", self.hard_negatives_ms),
            ("graph upsert + aliases", self.graph_ms),
            ("preference storage", self.preferences_ms),
            ("retrospective links", self.retrospective_ms),
            ("memory link storage", self.memory_links_ms),
            ("fact supersession", self.fact_ms),
            ("card latest updates", self.card_latest_ms),
            ("card relation updates", self.card_relations_ms),
        ];

        tracing::info!(
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

        tracing::info!(
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

#[derive(Clone)]
struct MiningRecord {
    entity_id: String,
    memory_id: String,
    textual_content: String,
    embedding: Vec<f32>,
}

type NlpCache = std::collections::HashMap<String, Vec<String>>;

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
    typed_graph_batch: Vec<GraphEdgeRecord>,
    memory_links_batch: Vec<(String, String, String)>,
    memory_card_batch: Vec<MemoryCard>,
    memory_card_relations_batch: Vec<(String, String, String)>,
    memory_card_latest_updates: Vec<(String, bool, u64)>,
    session_router_updates: Vec<SessionRouterRecord>,
    memory_artifacts_batch: Vec<MemoryArtifact>,
    temporal_events_batch: Vec<TemporalEvent>,
    shadow_questions_batch: Vec<ShadowQuestion>,
    facet_postings_batch: Vec<FacetPosting>,
    mem_cells_batch: Vec<MemCell>,
    mem_scenes_batch: Vec<MemSceneRecord>,
    profile_facts_batch: Vec<ProfileFact>,
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
    if !payload
        .memory_id
        .starts_with(ns_prefix.as_deref().unwrap_or(""))
    {
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
    insert_stage_timing_headers(
        &mut headers,
        "x-tm-storage",
        diag.storage_ms,
        diag.storage_us,
    );
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
    metrics::observe_query_duration(diag.total_ms as f64 / 1000.0);
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
        if !item
            .memory_id
            .starts_with(ns_prefix.as_deref().unwrap_or(""))
        {
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
    insert_stage_timing_headers(
        &mut headers,
        "x-tm-storage",
        diag.storage_ms,
        diag.storage_us,
    );
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
    metrics::observe_query_duration(diag.total_ms as f64 / 1000.0);
    Ok((StatusCode::CREATED, headers))
}

async fn process_ingest_batch(
    state: &EngineState,
    tenant: &std::sync::Arc<TenantStore>,
    payloads: Vec<IngestPayload>,
) -> Result<(Vec<ConsolidationTask>, IngestDiagnostics), StatusCode> {
    execute_ingest_pipeline(state, tenant, payloads).await
}

// ── Phase 1: Payload preparation ──
fn expand_and_enrich_payloads(
    payloads: Vec<IngestPayload>,
    diag: &mut IngestDiagnostics,
    total_start: Instant,
) -> (Vec<IngestPayload>, NlpCache, Vec<String>) {
    let stage_start = Instant::now();

    let mut expanded_payloads = Vec::new();
    for mut payload in payloads {
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

        for chunked_payload in expand_payload_for_content_type(&payload) {
            expanded_payloads.push(chunked_payload.clone());
            expanded_payloads.extend(build_companion_payloads(&chunked_payload));
        }
    }

    let unique_texts: std::collections::HashSet<String> =
        expanded_payloads.iter().map(|p| p.textual_content.clone()).collect();
    let unique_vec: Vec<String> = unique_texts.into_iter().collect();
    let nlp_pairs: Vec<(String, Vec<String>)> = unique_vec
        .par_iter()
        .map(|text| (text.clone(), extract_named_phrases(std::slice::from_ref(text))))
        .collect();
    let nlp_cache: NlpCache = nlp_pairs.into_iter().collect();

    diag.expand_ms = stage_start.elapsed().as_millis() as u64;
    diag.expanded_count = expanded_payloads.len();
    tracing::info!("[CP] expand_done: μs={}", total_start.elapsed().as_micros());

    // Context window enrichment
    let mut session_groups: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (idx, payload) in expanded_payloads.iter().enumerate() {
        if let Some(sid) = session_id_from_memory_id(&payload.memory_id) {
            session_groups.entry(sid).or_default().push(idx);
        }
    }

    let stage_start = Instant::now();
    let mut enriched_texts = Vec::with_capacity(expanded_payloads.len());

    for (idx, payload) in expanded_payloads.iter().enumerate() {
        if payload.kind.as_deref() == Some("synthetic_query") {
            enriched_texts.push(payload.textual_content.clone());
            continue;
        }

        let mut final_text = payload.textual_content.clone();

        if let Some(sid) = session_id_from_memory_id(&payload.memory_id) {
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

    (expanded_payloads, nlp_cache, enriched_texts)
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
        let embeddings = state.semantic.embed_batch_parallel(texts).await;

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
    state: &EngineState,
    expanded_payloads: Vec<IngestPayload>,
    semantic_embeddings: Vec<Vec<f32>>,
    diag: &mut IngestDiagnostics,
) -> Result<(Vec<PreparedRecord>, Vec<MiningRecord>), StatusCode> {
    let dedup_build_start = Instant::now();

    let mut prepared = Vec::new();
    let mut semantic_seen: Vec<(String, Vec<f32>)> = Vec::new();
    let mut semantic_embedding_iter = semantic_embeddings.into_iter();

    for payload in expanded_payloads.into_iter() {
        let mut payload = payload;
        let kind = parse_kind(payload.kind.as_deref());
        if (kind == MemoryKind::Preference || kind == MemoryKind::Decision || kind == MemoryKind::Fact)
            && payload.fact_key.as_ref().map_or(true, |k| k.trim().is_empty())
        {
            if let Some(inferred_key) = crate::api::plan::infer_query_fact_key(&payload.textual_content) {
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
        let lifecycle = evaluate_lifecycle(
            &payload.textual_content,
            kind,
            payload.timestamp,
            payload.fact_confidence,
            is_inference,
        );
        let embedding = if lifecycle.index_vector { embedding } else { Vec::new() };

        if index_semantic
            && lifecycle.index_vector
            && enable_semantic_dedup
            && (kind == MemoryKind::Fact || kind == MemoryKind::Decision)
        {
            let is_dup = is_semantic_duplicate(state, &payload.entity_id, &embedding, 0.94)?;
            let in_batch = semantic_seen.iter().any(|(entity_id, prior_embedding)| {
                entity_id == &payload.entity_id
                    && cosine_similarity(prior_embedding, &embedding) >= 0.94
            });
            if is_dup || in_batch {
                continue;
            }
            semantic_seen.push((payload.entity_id.clone(), embedding.clone()));
        }

        let hash = content_hash(&payload.textual_content, &payload.entity_id, &format!("{:?}", kind));
        let obs = AgentObservation {
            entity_id: payload.entity_id.clone(),
            textual_content: payload.textual_content.clone(),
            embedding: embedding.clone(),
            kind,
            content_hash: hash,
            created_at_ms: payload.timestamp,
        };

        prepared.push(PreparedRecord { payload, obs, embedding, lifecycle, enable_consolidation });
    }

    let mining_records: Vec<MiningRecord> = prepared
        .iter()
        .filter(|record| {
            record.payload.kind.as_deref() != Some("synthetic_query")
                && !record.embedding.is_empty()
        })
        .map(|record| MiningRecord {
            entity_id: record.payload.entity_id.clone(),
            memory_id: record.payload.memory_id.clone(),
            textual_content: record.payload.textual_content.clone(),
            embedding: record.embedding.clone(),
        })
        .collect();

    diag.dedup_build_ms = dedup_build_start.elapsed().as_millis() as u64;
    Ok((prepared, mining_records))
}

// ── Phase 4: Artifact building ──
fn build_artifacts(
    prepared: Vec<PreparedRecord>,
    inserted_flags: Vec<Option<u64>>,
    diag: &mut IngestDiagnostics,
) -> ArtifactBatches {
    let artifact_build_start = Instant::now();
    let mut batches = ArtifactBatches::default();

    for (record, vector_id) in prepared.into_iter().zip(inserted_flags.into_iter()) {
        let is_synthetic_query = record.payload.kind.as_deref() == Some("synthetic_query");
        if !is_synthetic_query {
            batches.fts_batch.push((
                record.payload.memory_id.clone(),
                record.payload.entity_id.clone(),
                record.payload.textual_content.clone(),
            ));
            if let Some(card) =
                build_memory_card_from_payload(&record.payload, record.obs.kind, &record.lifecycle)
            {
                if card.source_memory_id != card.card_id {
                    batches.memory_card_relations_batch.push((
                        card.source_memory_id.clone(),
                        EdgeType::Supports.as_str().to_string(),
                        card.card_id.clone(),
                    ));
                    batches.memory_card_relations_batch.push((
                        card.card_id.clone(),
                        EdgeType::Derives.as_str().to_string(),
                        card.source_memory_id.clone(),
                    ));
                }
                batches.memory_card_batch.push(card);
            }
            if let Some(router_update) =
                build_session_router_update_from_payload(&record.payload, record.obs.kind)
            {
                batches.session_router_updates.push(router_update);
            }
            batches.memory_artifacts_batch.push(build_memory_artifact_from_payload(
                &record.payload,
                "ledger_turn",
                "-v2-ledger",
                None,
                None,
                &record.lifecycle,
            ));
            if let Some(event) = build_temporal_event_from_payload(
                &record.payload,
                record.obs.kind,
                &record.lifecycle,
            ) {
                batches.temporal_events_batch.push(event);
            }
            batches
                .shadow_questions_batch
                .extend(build_shadow_questions_from_payload(&record.payload, record.obs.kind));
            batches
                .facet_postings_batch
                .extend(build_facet_postings_from_payload(&record.payload, record.obs.kind));
            if let Some(cell) =
                build_mem_cell_from_payload(&record.payload, record.obs.kind, &record.lifecycle)
            {
                batches.mem_cells_batch.push(cell);
            }
            if let Some(scene) =
                build_mem_scene_from_payload(&record.payload, record.obs.kind, &record.lifecycle)
            {
                batches.mem_scenes_batch.push(scene);
            }
            if let Some(pf) =
                build_profile_fact_from_payload(&record.payload, record.obs.kind, &record.lifecycle)
            {
                batches.profile_facts_batch.push(pf);
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

        if !is_synthetic_query {
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

            if let Some(source_memory_id) = record.payload.source_memory_id.as_deref() {
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

        if record.obs.kind == MemoryKind::Fact || record.obs.kind == MemoryKind::Preference || record.obs.kind == MemoryKind::Decision {
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

        if record.enable_consolidation {
            batches.consolidation_tasks.push(ConsolidationTask {
                entity_id: record.payload.entity_id.clone(),
                memory_id: record.payload.memory_id.clone(),
                timestamp: record.payload.timestamp,
                textual_content: record.payload.textual_content.clone(),
            });
        }
    }

    diag.artifact_build_ms = artifact_build_start.elapsed().as_millis() as u64;
    tracing::info!("[CP] artifact_build_done: μs={}", artifact_build_start.elapsed().as_micros());
    batches
}

// ── Phase 5: Storage commit ──
async fn commit_batches(
    tenant: &std::sync::Arc<TenantStore>,
    state: &EngineState,
    nlp_cache: &NlpCache,
    _mining_records: &[MiningRecord],
    batches: &mut ArtifactBatches,
    diag: &mut IngestDiagnostics,
) -> Result<(), StatusCode> {
    // Combined SQLite write: all secondary index upserts in a single spawn_blocking
    let combined_tenant = tenant.clone();
    let res_combined: Duration = tokio::task::spawn_blocking({
        let tenant = combined_tenant.clone();
        let cards = batches.memory_card_batch.clone();
        let artifacts = batches.memory_artifacts_batch.clone();
        let temporal_events = batches.temporal_events_batch.clone();
        let shadow_questions = batches.shadow_questions_batch.clone();
        let facet_postings = batches.facet_postings_batch.clone();
        let mem_cells = batches.mem_cells_batch.clone();
        let mem_scenes = batches.mem_scenes_batch.clone();
        let profile_facts = batches.profile_facts_batch.clone();
        move || -> Result<Duration, anyhow::Error> {
            let start = Instant::now();
            let input = CombinedIngestUpsertInput {
                cards: &cards,
                artifacts: &artifacts,
                events: &temporal_events,
                shadow_questions: &shadow_questions,
                facet_postings: &facet_postings,
                mem_cells: &mem_cells,
                mem_scenes: &mem_scenes,
                profile_facts: &profile_facts,
            };
            tenant.combined_ingest_upsert(&input)?;
            Ok(start.elapsed())
        }
    })
    .await
    .map_err(|e| {
        tracing::error!("combined_ingest_upsert panicked: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .map_err(|e| {
        tracing::error!("combined_ingest_upsert failed: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    diag.memory_cards_ms = res_combined.as_millis() as u64;
    diag.memory_artifacts_ms = res_combined.as_millis() as u64;

    tracing::info!("[CP] upserts_done_before_session_router");

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
                    format!("{}::{}::850000", record.entity_id, record.session_id),
                    record.entity_id.clone(),
                    record.router_text.clone(),
                ));
            }
        }

        diag.session_router_ms = sr_start.elapsed().as_millis() as u64;
    }

    // FTS + vector indexing (parallel)
    let (res_fts, res_vix) = tokio::join!(
        tokio::task::spawn_blocking({
            let tenant = tenant.clone();
            let batch = batches.fts_batch.clone();
            move || {
                let start = Instant::now();
                if !batch.is_empty() {
                    let _ = tenant.fts_index_batch(&batch);
                }
                start.elapsed()
            }
        }),
        tokio::task::spawn_blocking({
            let state = state.clone();
            let batch = batches.vector_batch.clone();
            move || {
                let start = Instant::now();
                if !batch.is_empty() {
                    for (entity_id, items) in batch {
                        let _ = state.vector_index.insert_batch(&entity_id, &items);
                    }
                }
                start.elapsed()
            }
        })
    );

    let res_fts = res_fts.unwrap_or(Duration::ZERO);
    let res_vix = res_vix.unwrap_or(Duration::ZERO);

    diag.fts_ms = res_fts.as_millis() as u64;
    diag.fts_us = res_fts.as_micros() as u64;
    diag.bm25f_ms = 0;
    diag.vector_ms = res_vix.as_millis() as u64;
    diag.vector_us = res_vix.as_micros() as u64;

    // Graph upsert + alias extraction + hard negative mining (parallel)
    let (res_graph, res_hn) = tokio::join!(
        tokio::task::spawn_blocking({
            let tenant = tenant.clone();
            let batch = batches.typed_graph_batch.clone();
            let consolidation_tasks = batches.consolidation_tasks.clone();
            let nlp_cache = nlp_cache.clone();
            let cards_for_entities = batches.memory_card_batch.clone();
            let facts_for_entities = batches.fact_batch.clone();
            move || {
                let start = Instant::now();
                if !batch.is_empty() {
                    let _ = tenant.graph_upsert_memory_batch(&batch);

                    let mut entity_texts: std::collections::HashMap<String, Vec<String>> =
                        std::collections::HashMap::new();
                    for task in &consolidation_tasks {
                        entity_texts
                            .entry(task.entity_id.clone())
                            .or_default()
                            .push(task.textual_content.clone());
                    }
                    for (entity_id, texts) in entity_texts {
                        let all_text = texts.join("\n");
                        let known_entities = dedupe_preserve_order(
                            texts
                                .iter()
                                .flat_map(|t| nlp_cache.get(t).cloned().unwrap_or_default())
                                .collect(),
                        );
                        let aliases = extract_aliases_from_text(&all_text, &known_entities);
                        if !aliases.is_empty() {
                            let _ = tenant.set_aliases_batch(&entity_id, &aliases);
                        }

                        // Collect entity names from memory card subjects + fact registrations.
                        let mut entity_names: Vec<String> = known_entities.clone();
                        for card in &cards_for_entities {
                            if card.entity_id == entity_id && !card.subject.is_empty() {
                                let s = card.subject.trim().to_string();
                                if s.len() >= 2 && !entity_names.contains(&s) {
                                    entity_names.push(s);
                                }
                            }
                        }
                        for fact in &facts_for_entities {
                            if fact.entity_id == entity_id && !fact.subject.is_empty() {
                                let s = fact.subject.trim().to_string();
                                if s.len() >= 2 && !entity_names.contains(&s) {
                                    entity_names.push(s);
                                }
                            }
                        }

                        // Run tiered entity resolver on extracted entity names.
                        for entity_name in &entity_names {
                            let _ = tenant.register_entity(&entity_id, entity_name);
                            let _ = tenant.resolve_and_propose(
                                &entity_id,
                                entity_name,
                                None,
                                &crate::storage::entity_resolver::ResolutionConfig::default(),
                            );
                        }
                    }
                }
                start.elapsed()
            }
        }),
        // Hard negative mining disabled — it consumed 40-70% of ingest time
        // for a <2% accuracy gain. Re-enable by restoring the original block.
        tokio::task::spawn_blocking(|| Duration::ZERO)
    );

    let res_graph = res_graph.unwrap_or(Duration::ZERO);
    let res_hn = res_hn.unwrap_or(Duration::ZERO);

    diag.graph_ms = res_graph.as_millis() as u64;
    diag.graph_us = res_graph.as_micros() as u64;
    diag.hard_negatives_ms = res_hn.as_millis() as u64;

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
            card_relations: Vec<(String, String, String)>,
        }

        let tenant_fact = tenant.clone();
        let fact_side_effects: Vec<Result<FactSideEffects, StatusCode>> = tokio::task::spawn_blocking(move || {
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
                            se.card_updates.push((reg.memory_id.clone(), true, reg.timestamp));
                            se.card_relations.push((
                                old_id.clone(),
                                EdgeType::Updates.as_str().to_string(),
                                reg.memory_id.clone(),
                            ));
                            graph_status_batch.push(GraphEdgeEntry {
                                memory_id: reg.memory_id.as_str(),
                                subject: reg.subject.as_str(),
                                predicate: reg.predicate.as_str(),
                                object: reg.object.as_str(),
                                status: "current",
                                ref_info: Some((EdgeType::Supersedes.as_str(), old_id.as_str())),
                                timestamp: reg.timestamp,
                            });
                            graph_status_batch.push(GraphEdgeEntry {
                                memory_id: old_id.as_str(),
                                subject: reg.subject.as_str(),
                                predicate: reg.predicate.as_str(),
                                object: reg.object.as_str(),
                                status: "stale",
                                ref_info: Some((EdgeType::SupersededBy.as_str(), reg.memory_id.as_str())),
                                timestamp: reg.timestamp,
                            });
                        }
                        FactVersionStatus::Stale { current: (_, cur_id) } => {
                            se.card_updates.push((reg.memory_id.clone(), false, reg.timestamp));
                            se.card_relations.push((
                                reg.memory_id.clone(),
                                EdgeType::SupersededBy.as_str().to_string(),
                                cur_id.clone(),
                            ));
                            graph_status_batch.push(GraphEdgeEntry {
                                memory_id: reg.memory_id.as_str(),
                                subject: reg.subject.as_str(),
                                predicate: reg.predicate.as_str(),
                                object: reg.object.as_str(),
                                status: "stale",
                                ref_info: Some((EdgeType::SupersededBy.as_str(), cur_id.as_str())),
                                timestamp: reg.timestamp,
                            });
                        }
                        FactVersionStatus::Current { superseded: None } => {
                            se.card_updates.push((reg.memory_id.clone(), true, reg.timestamp));
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
                    batches.memory_card_relations_batch.extend(se.card_relations);
                }
                Err(e) => return Err(e),
            }
        }

        (diag.fact_ms, diag.fact_us) = elapsed_ms_and_us(stage_start);
    }

    // Typed graph edges: populate the edges table from memory cards and fact registrations.
    // These edges power the entity-graph retrieval lane in the query fusion phase.
    {
        let _stage_start = Instant::now();
        let tenant_ge = tenant.clone();
        let cards_ge = batches.memory_card_batch.clone();
        let facts_ge = facts_for_edges;
        tokio::task::spawn_blocking(move || {
            for card in &cards_ge {
                if card.subject.is_empty() || card.predicate.is_empty() || card.object.is_empty() {
                    continue;
                }
                let _ = tenant_ge.graph_insert_edge(
                    &card.entity_id,
                    &card.source_memory_id,
                    &card.subject,
                    &card.predicate,
                    &card.object,
                    card.created_at_ms,
                );
            }
            for fact in &facts_ge {
                if fact.subject.is_empty() || fact.predicate.is_empty() || fact.object.is_empty() {
                    continue;
                }
                let _ = tenant_ge.graph_insert_edge(
                    &fact.entity_id,
                    &fact.memory_id,
                    &fact.subject,
                    &fact.predicate,
                    &fact.object,
                    fact.timestamp,
                );
            }
        })
        .await
        .map_err(|e| {
            tracing::error!("typed graph edge spawn panic: {:?}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
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

    // Card relations updates
    if !batches.memory_card_relations_batch.is_empty() {
        let stage_start = Instant::now();
        let tenant_cr = tenant.clone();
        let relations = batches.memory_card_relations_batch.clone();
        tokio::task::spawn_blocking(move || tenant_cr.set_memory_card_relations_batch(&relations))
            .await
            .map_err(|e| {
                tracing::error!("card_relations spawn panic: {:?}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?
            .map_err(|e| {
                tracing::error!("card_relations write failed: {:?}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;
        diag.card_relations_ms = stage_start.elapsed().as_millis() as u64;
    }

    Ok(())
}

fn build_memory_card_from_payload(
    payload: &IngestPayload,
    kind: MemoryKind,
    lifecycle: &LifecycleMetadata,
) -> Option<MemoryCard> {
    let source_memory_id = payload
        .source_memory_id
        .clone()
        .unwrap_or_else(|| payload.memory_id.clone());
    let source_session_id = session_id_from_memory_id(&source_memory_id)
        .or_else(|| session_id_from_memory_id(&payload.memory_id))
        .unwrap_or_default();
    let source_turn_index = turn_index_from_memory_id(&source_memory_id);
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
            extract_named_phrases(std::slice::from_ref(&payload.textual_content))
                .into_iter()
                .next()
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
    let object = payload
        .fact_object
        .clone()
        .unwrap_or_else(|| truncate_router_value(&memory_text, 320));
    let operation = payload
        .fact_operation
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
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
    let session_id = session_id_from_memory_id(&payload.memory_id)?;
    let document_time_ms = extract_document_time_ms(&payload.textual_content, payload.timestamp);
    let session_date = extract_bracketed_header_value(&payload.textual_content, "Session Date")
        .unwrap_or_else(|| "unknown".to_string());
    let session_focus = extract_bracketed_header_value(&payload.textual_content, "Session Focus")
        .unwrap_or_default();
    let dialogue = extract_dialogue_messages(&payload.textual_content);
    let dialogue_texts = if dialogue.is_empty() {
        vec![payload.textual_content.clone()]
    } else {
        dialogue
            .iter()
            .map(|(speaker, line)| format!("{speaker}: {line}"))
            .collect::<Vec<_>>()
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
    if matches!(
        kind,
        MemoryKind::Fact | MemoryKind::Decision | MemoryKind::Lesson
    ) {
        canonical_facts.push(compact_text.clone());
    }
    if matches!(kind, MemoryKind::Preference)
        || preference_signal_strength(&payload.textual_content, &payload.relations).is_some()
    {
        preference_signals.push(compact_text.clone());
    }
    if extract_event_time_ms(&payload.textual_content, document_time_ms).is_some()
        || !extract_temporal_terms(&payload.textual_content).is_empty()
        || [
            "went", "visited", "watched", "joined", "started", "finished", "won", "bought",
        ]
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
        text.chars()
            .take(max_chars)
            .collect::<String>()
            .trim()
            .to_string()
    }
}

fn build_memory_artifact_from_payload(
    payload: &IngestPayload,
    artifact_type: &str,
    compiler_name: &str,
    index_namespace: Option<&str>,
    embedding_dim: Option<usize>,
    lifecycle: &LifecycleMetadata,
) -> MemoryArtifact {
    let source_session_id = session_id_from_memory_id(&payload.memory_id).unwrap_or_default();
    MemoryArtifact {
        artifact_id: format!("artifact::{}::{}", artifact_type, payload.memory_id),
        artifact_type: artifact_type.to_string(),
        entity_id: payload.entity_id.clone(),
        source_turn_ids: vec![payload.memory_id.clone()],
        source_memory_ids: vec![payload.memory_id.clone()],
        source_session_ids: if source_session_id.is_empty() {
            Vec::new()
        } else {
            vec![source_session_id]
        },
        compiler_name: compiler_name.to_string(),
        compiler_version: "v2.0.0".to_string(),
        embedding_model: index_namespace.map(|_| "runtime-default".to_string()),
        embedding_dim,
        index_namespace: index_namespace.map(|value| value.to_string()),
        lifecycle: Some(lifecycle.clone()),
        created_at_ms: payload.timestamp,
        updated_at_ms: payload.timestamp,
    }
}

fn build_temporal_event_from_payload(
    payload: &IngestPayload,
    kind: MemoryKind,
    lifecycle: &LifecycleMetadata,
) -> Option<TemporalEvent> {
    let lower = payload.textual_content.to_ascii_lowercase();
    let document_time_ms = extract_document_time_ms(&payload.textual_content, payload.timestamp);
    let event_time_ms = extract_event_time_ms(&payload.textual_content, document_time_ms);
    let has_event_verb = [
        "went",
        "visited",
        "watched",
        "joined",
        "started",
        "finished",
        "won",
        "bought",
        "adopted",
        "met",
        "moved",
        "traveled",
        "played",
        "attended",
        "volunteered",
        "made",
        "built",
        "read",
        "studied",
        "worked",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    if event_time_ms.is_none()
        && extract_temporal_terms(&payload.textual_content).is_empty()
        && !has_event_verb
        && !matches!(kind, MemoryKind::SessionSummary)
    {
        return None;
    }
    let source_session_id = session_id_from_memory_id(&payload.memory_id).unwrap_or_default();
    let people = extract_named_phrases(std::slice::from_ref(&payload.textual_content));
    let terms = extract_salient_terms(&payload.textual_content, 8);
    let relation = infer_event_relation(&lower, &terms);
    let event_type = relation.clone();
    Some(TemporalEvent {
        event_id: format!("event::{}", payload.memory_id),
        entity_id: payload.entity_id.clone(),
        source_session_id,
        source_memory_id: payload.memory_id.clone(),
        source_turn_index: turn_index_from_memory_id(&payload.memory_id),
        subject: people
            .first()
            .cloned()
            .unwrap_or_else(|| payload.entity_id.clone()),
        relation,
        object: terms.first().cloned(),
        participants: people.clone(),
        place: infer_place_hint(&payload.textual_content),
        document_time_ms,
        event_time_ms,
        event_time_range_ms: event_time_ms.map(|ts| (ts, ts)),
        event_time_granularity: if event_time_ms.is_some() {
            "day_or_document".to_string()
        } else {
            "conversation".to_string()
        },
        actor_entities: people.clone(),
        object_entities: terms.clone(),
        event_type,
        is_inferred_time: event_time_ms.is_none(),
        event_text: truncate_router_value(&normalize_fact_text(&payload.textual_content), 420),
        confidence: if event_time_ms.is_some() { 0.86 } else { 0.68 },
        lifecycle: Some(lifecycle.clone()),
        created_at_ms: payload.timestamp,
    })
}

fn build_shadow_questions_from_payload(
    payload: &IngestPayload,
    kind: MemoryKind,
) -> Vec<ShadowQuestion> {
    let source_session_id = session_id_from_memory_id(&payload.memory_id).unwrap_or_default();
    let people = extract_named_phrases(std::slice::from_ref(&payload.textual_content));
    let subject = people
        .first()
        .cloned()
        .unwrap_or_else(|| payload.entity_id.clone());
    let terms = extract_salient_terms(&payload.textual_content, 6);
    let lower = payload.textual_content.to_ascii_lowercase();
    let answer_type = infer_shadow_answer_type(kind, &lower);
    let mut questions = Vec::new();
    let mut push_question = |question: String| {
        if question.trim().len() < 12 {
            return;
        }
        if questions
            .iter()
            .any(|existing: &ShadowQuestion| existing.question_text == question)
        {
            return;
        }
        let idx = questions.len();
        questions.push(ShadowQuestion {
            shadow_id: format!("shadow::{}::{idx}", payload.memory_id),
            entity_id: payload.entity_id.clone(),
            source_session_id: source_session_id.clone(),
            source_memory_id: payload.memory_id.clone(),
            source_card_id: None,
            question_text: question,
            answer_type: answer_type.clone(),
            entities: people.clone(),
            facets: terms.clone(),
            confidence: 0.74,
            created_at_ms: payload.timestamp,
        });
    };

    push_question(format!("What did {subject} mention?"));
    push_question(format!("What happened with {subject}?"));
    if lower.contains("favorite")
        || lower.contains("likes")
        || lower.contains("loves")
        || lower.contains("enjoys")
    {
        push_question(format!("What does {subject} like?"));
        push_question(format!("What is {subject}'s preference?"));
        if let Some(term) = terms.first() {
            push_question(format!("What does {subject} like about {term}?"));
        }
    }
    if lower.contains("when") || !extract_temporal_terms(&payload.textual_content).is_empty() {
        push_question(format!("When did this happen with {subject}?"));
        push_question(format!("What happened on this date involving {subject}?"));
    }
    if lower.contains("where") || lower.contains("visited") || lower.contains("went") {
        push_question(format!("Where did {subject} go?"));
    }
    if lower.contains("dog")
        || lower.contains("dogs")
        || lower.contains("pet")
        || lower.contains("pets")
        || lower.contains("pup")
        || lower.contains("puppy")
    {
        push_question(format!("What pets does {subject} have?"));
        push_question(format!("How many dogs does {subject} have?"));
        push_question(format!("What does {subject} say about their pets?"));
        push_question(format!("What does {subject} view their pets as?"));
    }
    if lower.contains("child") || lower.contains("children") || lower.contains("kid") {
        push_question(format!("What do {subject}'s kids like?"));
        push_question(format!("How many children does {subject} have?"));
    }
    if lower.contains("bought")
        || lower.contains("buy")
        || lower.contains("made")
        || lower.contains("built")
        || lower.contains("created")
    {
        push_question(format!("What did {subject} buy or make?"));
        push_question(format!("What items did {subject} get?"));
    }
    if lower.contains("career") || lower.contains("job") || lower.contains("pursue") {
        push_question(format!("What career could {subject} pursue?"));
        push_question(format!("What job might {subject} pursue in the future?"));
    }
    if lower.contains("class") || lower.contains("course") || lower.contains("workshop") {
        push_question(format!("What classes has {subject} joined?"));
        push_question(format!("What workshop did {subject} attend?"));
    }
    if lower.contains("decided") || matches!(kind, MemoryKind::Decision) {
        push_question(format!("What did {subject} decide?"));
    }
    if matches!(
        kind,
        MemoryKind::Fact | MemoryKind::Preference | MemoryKind::Decision
    ) {
        push_question(format!("What fact is known about {subject}?"));
    }
    questions.into_iter().take(12).collect()
}

fn build_facet_postings_from_payload(
    payload: &IngestPayload,
    kind: MemoryKind,
) -> Vec<FacetPosting> {
    let Some(session_id) = session_id_from_memory_id(&payload.memory_id) else {
        return Vec::new();
    };
    let mut postings = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let people = extract_named_phrases(std::slice::from_ref(&payload.textual_content));
    let terms = extract_salient_terms(&payload.textual_content, 12);
    let lower = payload.textual_content.to_ascii_lowercase();
    let mut push = |facet_type: &str, facet_value: String, weight: f32| {
        let value = facet_value.trim().to_ascii_lowercase();
        if value.len() < 2 {
            return;
        }
        let key = format!("{facet_type}:{value}:{}", payload.memory_id);
        if !seen.insert(key) {
            return;
        }
        postings.push(FacetPosting {
            entity_id: payload.entity_id.clone(),
            facet_type: facet_type.to_string(),
            facet_value: value,
            target_id: format!(
                "facet::{}::{}::{}",
                facet_type,
                payload.memory_id,
                postings.len()
            ),
            target_type: "memory".to_string(),
            session_id: session_id.clone(),
            memory_id: Some(payload.memory_id.clone()),
            card_id: None,
            event_id: None,
            turn_id: Some(payload.memory_id.clone()),
            weight,
        });
    };
    for person in people {
        push("person", person, 0.95);
    }
    for term in terms {
        push("activity", term, 0.58);
    }
    if matches!(kind, MemoryKind::Preference)
        || lower.contains("favorite")
        || lower.contains("prefers")
    {
        push(
            "preference",
            normalize_fact_text(&payload.textual_content),
            0.88,
        );
    }
    if matches!(kind, MemoryKind::Decision) || lower.contains("decided") {
        push(
            "decision",
            normalize_fact_text(&payload.textual_content),
            0.82,
        );
    }
    if lower.contains("how many") || lower.contains("number") || lower.contains("count") {
        push(
            "number",
            normalize_fact_text(&payload.textual_content),
            0.70,
        );
    }
    for term in extract_temporal_terms(&payload.textual_content) {
        push("date", term, 0.78);
    }
    postings
}

fn build_mem_cell_from_payload(
    payload: &IngestPayload,
    kind: MemoryKind,
    lifecycle: &LifecycleMetadata,
) -> Option<MemCell> {
    let source_session_id = session_id_from_memory_id(&payload.memory_id)?;
    let text = truncate_router_value(&normalize_fact_text(&payload.textual_content), 420);
    if text.is_empty() {
        return None;
    }
    let people = extract_named_phrases(std::slice::from_ref(&payload.textual_content));
    let terms = extract_salient_terms(&payload.textual_content, 8);
    let document_time_ms = extract_document_time_ms(&payload.textual_content, payload.timestamp);
    Some(MemCell {
        cell_id: format!("cell::{}", payload.memory_id),
        entity_id: payload.entity_id.clone(),
        source_session_id,
        source_turn_ids: vec![payload.memory_id.clone()],
        cell_text: text,
        cell_type: card_type_for_kind(kind, &payload.textual_content),
        subjects: people.clone(),
        objects: terms.clone(),
        activities: terms,
        places: infer_place_hint(&payload.textual_content)
            .into_iter()
            .collect(),
        document_time_ms,
        event_time_ms: extract_event_time_ms(&payload.textual_content, document_time_ms),
        confidence: 0.78,
        saliency: compute_memory_saliency(&payload.textual_content, kind),
        lifecycle: Some(lifecycle.clone()),
        created_at_ms: payload.timestamp,
    })
}

fn build_mem_scene_from_payload(
    payload: &IngestPayload,
    kind: MemoryKind,
    lifecycle: &LifecycleMetadata,
) -> Option<MemSceneRecord> {
    let source_session_id = session_id_from_memory_id(&payload.memory_id)?;
    let terms = extract_salient_terms(&payload.textual_content, 8);
    let people = extract_named_phrases(std::slice::from_ref(&payload.textual_content));
    if terms.is_empty() && people.is_empty() {
        return None;
    }
    let scene_key = terms
        .iter()
        .take(3)
        .cloned()
        .collect::<Vec<_>>()
        .join("_")
        .replace(' ', "_");
    Some(MemSceneRecord {
        scene_id: format!(
            "scene::{}::{}",
            source_session_id,
            if scene_key.is_empty() {
                "general"
            } else {
                scene_key.as_str()
            }
        ),
        entity_id: payload.entity_id.clone(),
        scene_title: if terms.is_empty() {
            format!("{} session context", source_session_id)
        } else {
            terms
                .iter()
                .take(4)
                .cloned()
                .collect::<Vec<_>>()
                .join(" / ")
        },
        scene_summary: truncate_router_value(&normalize_fact_text(&payload.textual_content), 520),
        source_cell_ids: vec![format!("cell::{}", payload.memory_id)],
        source_session_ids: vec![source_session_id],
        entities: people,
        activities: terms.clone(),
        objects: terms,
        places: infer_place_hint(&payload.textual_content)
            .into_iter()
            .collect(),
        time_range_ms: Some((payload.timestamp, payload.timestamp)),
        scene_type: card_type_for_kind(kind, &payload.textual_content),
        saliency: compute_memory_saliency(&payload.textual_content, kind),
        lifecycle: Some(lifecycle.clone()),
        created_at_ms: payload.timestamp,
        updated_at_ms: payload.timestamp,
    })
}

fn build_profile_fact_from_payload(
    payload: &IngestPayload,
    kind: MemoryKind,
    lifecycle: &LifecycleMetadata,
) -> Option<ProfileFact> {
    let lower = payload.textual_content.to_ascii_lowercase();
    let category = if matches!(kind, MemoryKind::Preference)
        || lower.contains("favorite")
        || lower.contains("prefers")
        || lower.contains("likes")
        || lower.contains("loves")
    {
        "durable_preference"
    } else if matches!(kind, MemoryKind::Decision) || lower.contains("decided") {
        "decision"
    } else if lower.contains("works at") || lower.contains("job") || lower.contains("occupation") {
        "occupation"
    } else if lower.contains("family") || lower.contains("spouse") || lower.contains("children") {
        "relationship"
    } else if lower.contains("allergy") || lower.contains("health") {
        "health_constraint"
    } else if matches!(kind, MemoryKind::Fact) {
        "stable_fact"
    } else {
        return None;
    };
    Some(ProfileFact {
        profile_fact_id: format!("profile_fact::{}", payload.memory_id),
        entity_id: payload.entity_id.clone(),
        category: category.to_string(),
        value: truncate_router_value(&normalize_fact_text(&payload.textual_content), 360),
        source_session_id: session_id_from_memory_id(&payload.memory_id).unwrap_or_default(),
        source_memory_id: payload.memory_id.clone(),
        source_card_id: Some(payload.memory_id.clone()),
        confidence: 0.78,
        document_time_ms: extract_document_time_ms(&payload.textual_content, payload.timestamp),
        is_latest: true,
        lifecycle: Some(lifecycle.clone()),
        created_at_ms: payload.timestamp,
    })
}

fn infer_event_relation(lower: &str, terms: &[String]) -> String {
    for (needle, relation) in [
        ("volunteer", "volunteered"),
        ("adopt", "adopted"),
        ("watch", "watched"),
        ("visit", "visited"),
        ("went", "went"),
        ("buy", "bought"),
        ("bought", "bought"),
        ("meet", "met"),
        ("met", "met"),
        ("move", "moved"),
        ("started", "started"),
        ("finished", "finished"),
        ("won", "won"),
        ("read", "read"),
        ("study", "studied"),
    ] {
        if lower.contains(needle) {
            return relation.to_string();
        }
    }
    terms
        .first()
        .cloned()
        .unwrap_or_else(|| "event".to_string())
}

fn infer_place_hint(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    for marker in [" at ", " in ", " near ", " from ", " to "] {
        if let Some(idx) = lower.find(marker) {
            let tail = text[idx + marker.len()..].trim();
            let place = tail
                .split(['.', ',', ';', '\n'])
                .next()
                .unwrap_or("")
                .split_whitespace()
                .take(4)
                .collect::<Vec<_>>()
                .join(" ");
            if place.len() >= 3 {
                return Some(place);
            }
        }
    }
    None
}

fn infer_shadow_answer_type(kind: MemoryKind, lower: &str) -> String {
    if lower.contains("when") || lower.contains(" on ") || lower.contains("date") {
        "date"
    } else if lower.contains("where") || lower.contains("visited") || lower.contains("went") {
        "place"
    } else if lower.contains("how many") || lower.contains("number") {
        "count"
    } else if matches!(kind, MemoryKind::Preference)
        || lower.contains("favorite")
        || lower.contains("likes")
        || lower.contains("loves")
    {
        "preference"
    } else if matches!(kind, MemoryKind::Decision) {
        "decision"
    } else {
        "fact"
    }
    .to_string()
}

fn compute_memory_saliency(text: &str, kind: MemoryKind) -> f32 {
    let lower = text.to_ascii_lowercase();
    let named = !extract_named_phrases(&[text.to_string()]).is_empty();
    let temporal = !extract_temporal_terms(text).is_empty();
    let life_event = [
        "adopted",
        "moved",
        "married",
        "graduated",
        "started",
        "finished",
        "won",
        "lost",
        "allergy",
        "health",
        "job",
        "birthday",
        "favorite",
        "decided",
    ]
    .iter()
    .any(|needle| lower.contains(needle));
    let mut score: f32 = 0.20;
    if named {
        score += 0.20;
    }
    if temporal {
        score += 0.20;
    }
    if life_event {
        score += 0.25;
    }
    if matches!(
        kind,
        MemoryKind::Fact | MemoryKind::Preference | MemoryKind::Decision
    ) {
        score += 0.15;
    }
    score.min(1.0)
}

fn mine_hard_negative_profiles(
    state: &EngineState,
    tenant: &TenantStore,
    records: &[MiningRecord],
) -> Result<EmbeddingPairSet, StatusCode> {
    let results: Vec<MiningResult> = records
        .par_iter()
        .map(|record| -> MiningResult {
            let raw_hits = ok_or_500(state.vector_index.search(
                Some(&record.entity_id),
                &record.embedding,
                16,
            ))?;
            let current_session = session_id_from_memory_id(&record.memory_id);
            let vector_ids: Vec<u64> = raw_hits.iter().map(|(vid, _)| *vid).collect();
            let looked_up = ok_or_500(tenant.lookup_by_vector_ids_batch(&vector_ids))?;

            let mut candidate_scores: std::collections::HashMap<String, f32> =
                std::collections::HashMap::new();
            let mut candidate_keys: std::collections::HashMap<String, u64> =
                std::collections::HashMap::new();
            for ((_, dist), maybe_lookup) in raw_hits.iter().zip(looked_up.into_iter()) {
                let Some((ts, candidate_mid)) = maybe_lookup else {
                    continue;
                };
                if candidate_mid == record.memory_id {
                    continue;
                }
                if let (Some(a), Some(b)) = (
                    current_session.as_ref(),
                    session_id_from_memory_id(&candidate_mid).as_ref(),
                ) {
                    if a == b {
                        continue;
                    }
                }
                let similarity = (1.0 - *dist).clamp(0.0, 1.0);
                if !(0.76..=0.97).contains(&similarity) {
                    continue;
                }
                candidate_keys.entry(candidate_mid.clone()).or_insert(ts);
                *candidate_scores.entry(candidate_mid).or_insert(0.0) += similarity * 0.75;
            }

            let fts_hits = tenant.fts_search(&record.textual_content, 8, Some(&record.entity_id))
                .unwrap_or_default();
            for (rank, (memory_id, lexical_score)) in fts_hits.into_iter().enumerate() {
                if memory_id == record.memory_id {
                    continue;
                }
                if let (Some(a), Some(b)) = (
                    current_session.as_ref(),
                    session_id_from_memory_id(&memory_id).as_ref(),
                ) {
                    if a == b {
                        continue;
                    }
                }
                if let Some((ts, _)) = tenant.lookup_by_memory_id(&memory_id)
                    .unwrap_or(None)
                {
                    candidate_keys.entry(memory_id.clone()).or_insert(ts);
                    *candidate_scores.entry(memory_id).or_insert(0.0) +=
                        lexical_score.min(1.5) / 1.5 * 0.25 - rank as f32 * 0.01;
                }
            }

            if candidate_scores.is_empty() {
                return Ok(None);
            }

            let mut ranked_candidates = candidate_scores.into_iter().collect::<Vec<_>>();
            ranked_candidates
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ranked_candidates.truncate(4);

            let obs_keys: Vec<(u64, String)> = ranked_candidates
                .iter()
                .filter_map(|(memory_id, _)| {
                    candidate_keys
                        .get(memory_id)
                        .map(|ts| (*ts, memory_id.clone()))
                })
                .collect();
            let observations = ok_or_500(tenant.get_observations_batch(&obs_keys))?;

            let mut hard_negatives = Vec::new();
            for (memory_id, _) in ranked_candidates {
                if let Some(obs) = observations.get(&memory_id) {
                    if obs.embedding.len() == record.embedding.len() && !obs.embedding.is_empty() {
                        hard_negatives.push(obs.embedding.clone());
                    }
                }
            }
            if hard_negatives.is_empty() {
                return Ok(None);
            }

            let mut centroid = vec![0.0f32; record.embedding.len()];
            for negative in &hard_negatives {
                for (c, value) in centroid.iter_mut().zip(negative.iter()) {
                    *c += *value;
                }
            }
            let inv = 1.0 / hard_negatives.len() as f32;
            for value in &mut centroid {
                *value *= inv;
            }

            let mut disambiguation = Vec::with_capacity(record.embedding.len());
            for (cur, neg) in record.embedding.iter().zip(centroid.iter()) {
                disambiguation.push(cur - neg);
            }
            let norm = disambiguation
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            if norm <= 1e-6 {
                return Ok(None);
            }
            for value in &mut disambiguation {
                *value /= norm;
            }

            Ok(Some((record.memory_id.clone(), centroid, disambiguation)))
        })
        .collect();

    let mut disambiguation_batch = Vec::new();
    let mut negative_centroid_batch = Vec::new();
    for result in results {
        let Some((mid, centroid, disambiguation)) = result? else {
            continue;
        };
        negative_centroid_batch.push((mid.clone(), centroid));
        disambiguation_batch.push((mid, disambiguation));
    }
    Ok((disambiguation_batch, negative_centroid_batch))
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

        let fts_hits = tenant.fts_search(reference_query, 8, Some(entity_id))
            .unwrap_or_default();
        for (rank, (memory_id, lexical_score)) in fts_hits.into_iter().enumerate() {
            if memory_id == *current_memory_id || memory_id == *source_memory_id {
                continue;
            }
            if let Some((ts, _)) = tenant.lookup_by_memory_id(&memory_id)
                .unwrap_or(None)
            {
                timestamp_by_memory.insert(memory_id.clone(), ts);
                *score_by_memory.entry(memory_id).or_insert(0.0) +=
                    0.45 + lexical_score.min(1.0) * 0.10 - rank as f32 * 0.02;
            }
        }

        let query_embedding = ok_or_500(state.semantic.generate_query_embedding(reference_query))?;
        let ann_hits = ok_or_500(
            state
                .vector_index
                .search(Some(entity_id), &query_embedding, 10),
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
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((target_memory_id, score)) = ranked.into_iter().find(|(memory_id, _)| {
            timestamp_by_memory
                .get(memory_id)
                .copied()
                .map(|ts| ts < *current_timestamp)
                .unwrap_or(false)
        }) {
            if score >= 0.55 {
                let target_text = tenant.lookup_by_memory_id(&target_memory_id)
                    .ok()
                    .flatten()
                    .and_then(|(ts, _)| {
                        tenant.get_observation(ts, &target_memory_id)
                            .ok()
                            .flatten()
                            .map(|obs| obs.textual_content)
                    })
                    .unwrap_or_default();
                let (forward_type, reverse_type) = classify_retrospective_link(
                    current_text,
                    &target_text,
                    *current_timestamp,
                    timestamp_by_memory
                        .get(&target_memory_id)
                        .copied()
                        .unwrap_or(0),
                );
                let forward = (
                    source_memory_id.clone(),
                    target_memory_id.clone(),
                    forward_type.to_string(),
                );
                let reverse = (
                    target_memory_id.clone(),
                    source_memory_id.clone(),
                    reverse_type.to_string(),
                );
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
    let contradiction_markers = [
        "actually",
        "turns out",
        "not ",
        "never ",
        "instead",
        "wrong",
    ];
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
    let current_only = current_entities
        .iter()
        .filter(|entity| !target_entities.contains(entity))
        .count();
    if current_only >= 1 || current_timestamp > target_timestamp {
        return ("extends", "extended_by");
    }

    ("recalls", "recalled_by")
}

fn spawn_consolidation_tasks(tenant: Arc<TenantStore>, tasks: Vec<ConsolidationTask>) {
    for task in tasks {
        let tenant_clone = tenant.clone();
        tokio::task::spawn_blocking(move || {
            update_core_profile_heuristic(&tenant_clone, &task);
        });
    }
}

fn update_core_profile_heuristic(tenant: &TenantStore, task: &ConsolidationTask) {
    let excerpt = truncate_router_value(&normalize_fact_text(&task.textual_content), 280);
    if excerpt.is_empty() {
        return;
    }

    let mut profile = tenant.get_core_profile(&task.entity_id)
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(&raw).ok())
        .unwrap_or_else(|| {
            serde_json::json!({
                "schema": "heuristic_core_profile_v1",
                "entity_id": task.entity_id,
                "facts": [],
                "updated_at_ms": task.timestamp
            })
        });

    let fact = serde_json::json!({
        "memory_id": task.memory_id,
        "timestamp_ms": task.timestamp,
        "text": excerpt,
        "terms": extract_salient_terms(&task.textual_content, 6)
    });

    if let Some(facts) = profile.get_mut("facts").and_then(|v| v.as_array_mut()) {
        let seen = facts.iter().any(|item| {
            item.get("memory_id")
                .and_then(|v| v.as_str())
                .map(|memory_id| memory_id == task.memory_id)
                .unwrap_or(false)
        });
        if !seen {
            facts.push(fact);
        }
        if facts.len() > 24 {
            let drain_to = facts.len() - 24;
            facts.drain(0..drain_to);
        }
    }
    profile["updated_at_ms"] = serde_json::json!(task.timestamp);

    if let Ok(serialized) = serde_json::to_string(&profile) {
        let _ = tenant.set_core_profile(&task.entity_id, &serialized);
    }
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

    // Phase 1: Payload preparation
    let (mut expanded_payloads, nlp_cache, mut enriched_texts) =
        expand_and_enrich_payloads(payloads, &mut diag, total_start);

    // Content-hash dedup: skip payloads with the same content, entity_id, and kind.
    // This runs before embedding, so it saves both compute and storage.
    {
        let stage_start = Instant::now();
        let hashes: Vec<String> = expanded_payloads
            .iter()
            .zip(enriched_texts.iter())
            .map(|(payload, text)| {
                let kind_str = payload.kind.as_deref().unwrap_or("memory");
                content_hash(text, &payload.entity_id, kind_str)
            })
            .collect();

        let existing = tenant.existing_content_hashes(&hashes).unwrap_or_default();

        let mut keep_idx = Vec::with_capacity(expanded_payloads.len());
        for (i, h) in hashes.iter().enumerate() {
            if !existing.contains(h) {
                keep_idx.push(i);
            }
        }

        expanded_payloads = keep_idx.iter().map(|&i| expanded_payloads[i].clone()).collect();
        enriched_texts = keep_idx.iter().map(|&i| enriched_texts[i].clone()).collect();

        diag.dedup_build_ms += stage_start.elapsed().as_millis() as u64;
    }

    // Phase 2: Embedding
    let semantic_embeddings =
        generate_embeddings(state, &expanded_payloads, &enriched_texts, &mut diag).await?;

    tracing::info!("[CP] embed_done: μs={}", total_start.elapsed().as_micros());

    // Phase 3: Dedup + observation building
    let (prepared, mining_records) =
        build_observations(state, expanded_payloads, semantic_embeddings, &mut diag)?;

    let batch_items: Vec<(u64, String, AgentObservation)> = prepared
        .iter()
        .map(|record| {
            (record.payload.timestamp, record.payload.memory_id.clone(), record.obs.clone())
        })
        .collect();

    tracing::info!("[CP] dedup_done: μs={}", total_start.elapsed().as_micros());

    let stage_start = Instant::now();
    let inserted_flags = tenant.insert_observations_batch(&batch_items).map_err(|e| {
        tracing::warn!("Batch Storage Error: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    (diag.storage_ms, diag.storage_us) = elapsed_ms_and_us(stage_start);

    // Analytics processing disabled — it consumed 600-2000ms per batch
    // running BERT NER + writing metrics never used in retrieval.
    diag.analytics_ms = 0;

    // Phase 4: Artifact building
    let mut batches = build_artifacts(prepared, inserted_flags, &mut diag);

    // Phase 5: Storage commit
    commit_batches(tenant, state, &nlp_cache, &mining_records, &mut batches, &mut diag).await?;

    (diag.total_ms, diag.total_us) = elapsed_ms_and_us(total_start);
    tracing::info!("[CP] final: μs={}", total_start.elapsed().as_micros());
    diag.log_table();
    Ok((batches.consolidation_tasks, diag))
}
