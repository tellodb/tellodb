use axum::http::HeaderMap;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
};
use std::time::Instant;

use crate::api::auth::{
    authorize_request, principal_namespace_prefix, principal_user_id, record_usage_for_principal,
    scope_entity_id,
};
use crate::api::types::*;
use crate::api::utils::RESET_CONFIRM_PHRASE;
use crate::api::EngineState;
use crate::metrics;
use anyhow::Context;

const CACHE_CAPACITY: usize = 10_000;
const BYTES_PER_MB: u64 = 1_048_576;
const BYTES_PER_GB: u64 = 1_073_741_824;
const ARTIFACT_HOURS: usize = 24;
const ARTIFACT_VERSION_HOURS: usize = 48;
const TURN_WINDOW_DEFAULT_RADIUS: u32 = 2;
const TURN_WINDOW_MAX_RADIUS: u32 = 8;
const DELETION_TOMBSTONE_HOURS: usize = 8;
const WARMUP_PROBE_TEXT: &str = "warmup probe";
const API_DELETE_REASON: &str = "api_delete";
const GRAPH_EDGE_LIMIT: usize = 1000;

pub async fn metrics_handler() -> impl IntoResponse {
    (StatusCode::OK, [("content-type", "text/plain; charset=utf-8")], metrics::render())
}

pub async fn status_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let tenant_id = principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let cache_usage = tenant.observation_cache_usage();
    let status = EngineStatus {
        device: state.semantic.device_label().to_string(),
        data_root: state.data_root.to_string(),
        cache_capacity: CACHE_CAPACITY,
        cache_usage: cache_usage as usize,
    };
    Ok((StatusCode::OK, Json(status)))
}

pub async fn health_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let device = crate::semantic::SemanticInference::device_label_static();
    let response = (
        StatusCode::OK,
        Json(HealthResponse {
            status: "ok",
            auth_required: state.auth.is_required(),
            device,
            data_root: state.data_root.to_string(),
        }),
    );
    record_usage_for_principal(&state, &principal, "health");
    Ok(response)
}

pub async fn healthz_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(ProbeResponse { status: "ok" }))
}

pub async fn version_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let response = (
        StatusCode::OK,
        Json(VersionResponse {
            engine_version: env!("CARGO_PKG_VERSION"),
            api_version: "v1",
            auth_required: state.auth.is_required(),
        }),
    );
    record_usage_for_principal(&state, &principal, "version");
    Ok(response)
}

pub async fn warmup_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let semantic = state.semantic.clone();
    let started = Instant::now();

    tokio::task::spawn_blocking(move || {
        semantic
            .generate_query_embedding(WARMUP_PROBE_TEXT)
            .context("warmup embedding generation failed")?;
        semantic
            .predict_score(WARMUP_PROBE_TEXT, WARMUP_PROBE_TEXT)
            .context("warmup scoring failed")?;
        Ok::<(), anyhow::Error>(())
    })
    .await
    .map_err(|err| {
        tracing::warn!("Warmup spawn blocking error: {:?}", err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .map_err(|err| {
        tracing::warn!("Warmup failed: {:?}", err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    record_usage_for_principal(&state, &principal, "warmup");
    Ok((
        StatusCode::OK,
        Json(WarmupResponse {
            status: "warmed",
            device: crate::semantic::SemanticInference::device_label_static(),
            duration_ms: started.elapsed().as_millis(),
        }),
    ))
}

pub async fn reset_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(payload): Json<ResetPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let tenant_id = principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|e| {
        tracing::error!("tenant_store lookup failed for tenant_id={}: {:?}", tenant_id, e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    let confirm = payload.confirm.as_deref();
    if confirm != Some(RESET_CONFIRM_PHRASE) && confirm != Some("RESET_DATA_DANGEROUS") {
        return Err(StatusCode::BAD_REQUEST);
    }

    tenant.clear_all().map_err(|e| {
        tracing::error!(error = ?e, "clear_all failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    tenant.fts_clear().map_err(|e| {
        tracing::error!(error = ?e, "fts_clear failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    tenant.graph_clear().map_err(|e| {
        tracing::error!(error = ?e, "graph_clear failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    state.vector_index.clear(None).map_err(|e| {
        tracing::error!(error = ?e, "vector_index.clear failed");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if payload.clear_embedding_cache.unwrap_or(false) {
        // embedding cache clearing not supported in current semantic module
    }

    record_usage_for_principal(&state, &principal, "admin_reset");
    Ok(StatusCode::OK)
}

pub async fn memory_inspect_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(payload): Json<MemoryInspectPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let ns_prefix = principal_namespace_prefix(&principal);
    let tenant_id = principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut payload = payload;
    payload.memory_id = scope_entity_id(&payload.memory_id, ns_prefix.as_deref());
    let response = tokio::task::spawn_blocking(move || {
        let tenant = tenant.clone();
        let timestamp = tenant
            .lookup_by_memory_id(&payload.memory_id)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .map(|(ts, _)| ts);

        let observation = if let Some(ts) = timestamp {
            tenant
                .get_observations_batch(&[(ts, payload.memory_id.clone())])
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
                .remove(&payload.memory_id)
                .map(|obs| MemoryAuditObservation {
                    entity_id: obs.entity_id,
                    kind: format!("{:?}", obs.kind),
                    created_at_ms: obs.created_at_ms,
                    textual_content: obs.textual_content,
                })
        } else {
            None
        };

        let card = tenant
            .get_memory_card(&payload.memory_id)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let ledger_turn = tenant
            .get_ledger_turns_batch(&[payload.memory_id.clone()])
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .remove(&payload.memory_id);
        let artifacts = tenant
            .get_memory_artifacts_for_source(&payload.memory_id, ARTIFACT_HOURS)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let artifact_ids =
            artifacts.iter().map(|artifact| artifact.artifact_id.clone()).collect::<Vec<_>>();
        let artifact_versions = tenant
            .get_artifact_versions_for_artifacts(&artifact_ids, ARTIFACT_VERSION_HOURS)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let deletion_tombstones = tenant
            .get_deletion_tombstones_for_target(&payload.memory_id, DELETION_TOMBSTONE_HOURS)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let lifecycle = card
            .as_ref()
            .and_then(|card| card.lifecycle.clone())
            .or_else(|| ledger_turn.as_ref().and_then(|turn| turn.lifecycle.clone()))
            .or_else(|| artifacts.iter().find_map(|artifact| artifact.lifecycle.clone()));

        let mut turn_window = Vec::new();
        if payload.include_turn_window.unwrap_or(false) {
            if let Some(turn) = ledger_turn.as_ref() {
                let radius = payload
                    .turn_window_radius
                    .unwrap_or(TURN_WINDOW_DEFAULT_RADIUS)
                    .min(TURN_WINDOW_MAX_RADIUS);
                turn_window = tenant
                    .get_turn_window(&turn.entity_id, &turn.session_id, turn.turn_index, radius)
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
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

        Ok::<MemoryInspectResponse, StatusCode>(MemoryInspectResponse {
            memory_id: payload.memory_id,
            timestamp,
            observation,
            card,
            ledger_turn,
            lifecycle,
            artifacts,
            artifact_versions,
            deletion_tombstones,
            turn_window,
        })
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    record_usage_for_principal(&state, &principal, "memory_inspect");
    Ok((StatusCode::OK, Json(response)))
}

pub async fn memory_delete_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(payload): Json<MemoryDeletePayload>,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;
    let ns_prefix = principal_namespace_prefix(&principal);
    let tenant_id = principal_user_id(&principal).unwrap_or("default");
    let tenant = state.tenant_store(tenant_id).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let mut payload = payload;
    payload.memory_id = scope_entity_id(&payload.memory_id, ns_prefix.as_deref());
    let reason = payload
        .reason
        .clone()
        .filter(|reason| !reason.trim().is_empty())
        .unwrap_or_else(|| API_DELETE_REASON.to_string());
    let state_for_delete = state.clone();
    let response = tokio::task::spawn_blocking(move || {
        let tenant = tenant.clone();
        let timestamp = tenant
            .lookup_by_memory_id(&payload.memory_id)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
            .map(|(ts, _)| ts);
        let observation = if let Some(ts) = timestamp {
            tenant
                .get_observations_batch(&[(ts, payload.memory_id.clone())])
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
                .remove(&payload.memory_id)
        } else {
            None
        };

        let Some(ts) = timestamp else {
            return Ok::<MemoryDeleteResponse, StatusCode>(MemoryDeleteResponse {
                deleted: false,
                memory_id: payload.memory_id,
                timestamp: None,
                vector_id: None,
                tombstone: None,
                fts_removed: 0,
                graph_edges_removed: 0,
            });
        };

        let deleted = tenant
            .delete_observation(ts, &payload.memory_id, &reason)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        let fts_removed = observation
            .as_ref()
            .and_then(|_obs| tenant.fts_remove_document(&payload.memory_id).ok().map(|_| 1))
            .unwrap_or(0);
        let graph_edges_removed = tenant.graph_remove_memory(&payload.memory_id).unwrap_or(0);
        if let Some(vector_id) = deleted.vector_id {
            let entity_id = if deleted.entity_id.is_empty() {
                observation.as_ref().map(|obs| obs.entity_id.as_str()).unwrap_or("")
            } else {
                deleted.entity_id.as_str()
            };
            if !entity_id.is_empty() {
                let _ = state_for_delete.vector_index.remove(entity_id, vector_id);
            }
        }

        Ok::<MemoryDeleteResponse, StatusCode>(MemoryDeleteResponse {
            deleted: true,
            memory_id: payload.memory_id,
            timestamp: Some(ts),
            vector_id: deleted.vector_id,
            tombstone: deleted.tombstone,
            fts_removed,
            graph_edges_removed,
        })
    })
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)??;

    record_usage_for_principal(&state, &principal, "memory_delete");
    Ok((StatusCode::OK, Json(response)))
}

fn get_system_metrics() -> (f32, u64, u64) {
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use sysinfo::System;
    static SYS: OnceLock<Mutex<System>> = OnceLock::new();
    let lock = SYS.get_or_init(|| {
        let mut sys = System::new_all();
        sys.refresh_all();
        Mutex::new(sys)
    });
    let mut sys = lock.lock().unwrap_or_else(|e| e.into_inner());
    sys.refresh_cpu_usage();
    sys.refresh_memory();

    let cpu = sys.global_cpu_usage();
    let total_ram = sys.total_memory() / BYTES_PER_MB;
    let used_ram = sys.used_memory() / BYTES_PER_MB;

    (cpu, total_ram, used_ram)
}

fn get_disk_metrics(data_root: &str) -> (u64, u64) {
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let mut storage_total_gb = 0;
    let mut storage_used_gb = 0;

    let data_path = std::path::Path::new(data_root);
    let data_path_abs = data_path.canonicalize().unwrap_or_else(|_| data_path.to_path_buf());
    let matched_disk = disks.iter().find(|d| data_path_abs.starts_with(d.mount_point()));

    if let Some(disk) = matched_disk {
        let total = disk.total_space();
        let avail = disk.available_space();
        storage_total_gb = total / BYTES_PER_GB;
        storage_used_gb = (total - avail) / BYTES_PER_GB;
    } else if let Some(disk) = disks.first() {
        let total = disk.total_space();
        let avail = disk.available_space();
        storage_total_gb = total / BYTES_PER_GB;
        storage_used_gb = (total - avail) / BYTES_PER_GB;
    }
    (storage_total_gb, storage_used_gb)
}

fn get_gpu_metrics() -> Option<(f32, u64, u64)> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let first_line = stdout.lines().next()?;
    let parts: Vec<&str> = first_line.split(',').map(|s| s.trim()).collect();
    if parts.len() >= 3 {
        let gpu_usage = parts[0].parse::<f32>().ok()?;
        let mem_used = parts[1].parse::<u64>().ok()?;
        let mem_total = parts[2].parse::<u64>().ok()?;
        Some((gpu_usage, mem_total, mem_used))
    } else {
        None
    }
}

pub async fn hardware_stats_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, StatusCode> {
    let principal = authorize_request(&headers, &state)?;

    let (cpu_usage_percent, ram_total_mb, ram_used_mb) = get_system_metrics();
    let (storage_total_gb, storage_used_gb) = get_disk_metrics(&state.data_root);

    let (gpu_usage_percent, gpu_ram_total_mb, gpu_ram_used_mb) = match get_gpu_metrics() {
        Some((usage, total, used)) => (Some(usage), Some(total), Some(used)),
        None => (None, None, None),
    };

    let response = HardwareStatsResponse {
        cpu_usage_percent,
        ram_total_mb,
        ram_used_mb,
        storage_total_gb,
        storage_used_gb,
        gpu_usage_percent,
        gpu_ram_total_mb,
        gpu_ram_used_mb,
    };

    record_usage_for_principal(&state, &principal, "hardware_stats");
    Ok((StatusCode::OK, Json(response)))
}

pub async fn cluster_stats_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Path(cluster_id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    crate::api::auth::authorize_global_api_key(&headers, &state.auth)?;

    let tenant = state
        .tenant_store(&cluster_id)
        .or_else(|_| state.tenant_store("default"))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut stats = tenant.db_stats().map_err(|err| {
        tracing::warn!("Failed to query db stats for cluster {}: {:?}", cluster_id, err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if let Ok(usage) = state.platform.total_usage_stats() {
        stats.request_count = usage.request_count as usize;
        stats.ingest_count = usage.ingest_count as usize;
        stats.query_count = usage.query_count as usize;
    }

    Ok((StatusCode::OK, Json(stats)))
}

pub async fn storage_stats_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Path(cluster_id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    crate::api::auth::authorize_global_api_key(&headers, &state.auth)?;
    let tenant = state
        .tenant_store(&cluster_id)
        .or_else(|_| state.tenant_store("default"))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let stats = tenant.detailed_db_stats().map_err(|err| {
        tracing::warn!("Failed to query storage stats for cluster {}: {:?}", cluster_id, err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok((StatusCode::OK, Json(stats)))
}

pub async fn cluster_graph_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Path(cluster_id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    crate::api::auth::authorize_global_api_key(&headers, &state.auth)?;
    let tenant = state
        .tenant_store(&cluster_id)
        .or_else(|_| state.tenant_store("default"))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let edges = tenant.get_all_edges(GRAPH_EDGE_LIMIT).map_err(|err| {
        tracing::warn!("Failed to query graph edges for cluster {}: {:?}", cluster_id, err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    Ok((StatusCode::OK, Json(edges)))
}

pub async fn admin_inject_api_key_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    Json(payload): Json<AdminInjectApiKeyPayload>,
) -> Result<impl IntoResponse, StatusCode> {
    crate::api::auth::authorize_global_api_key(&headers, &state.auth)?;

    state
        .platform
        .inject_api_key(
            &payload.key_id,
            &payload.user_id,
            &payload.name,
            &payload.token,
            payload.cluster_id.as_deref(),
        )
        .map_err(|err| {
            tracing::warn!("Failed to inject API key: {:?}", err);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(StatusCode::CREATED)
}

pub async fn admin_revoke_api_key_handler(
    State(state): State<EngineState>,
    headers: HeaderMap,
    axum::extract::Path(key_id): axum::extract::Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    crate::api::auth::authorize_global_api_key(&headers, &state.auth)?;

    state.platform.admin_revoke_api_key(&key_id).map_err(|err| {
        tracing::warn!("Failed to admin revoke API key: {:?}", err);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(StatusCode::NO_CONTENT)
}
