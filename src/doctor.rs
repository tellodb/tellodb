//! `tellodb doctor`: what the engine loaded and the state of each tenant.

use crate::api::EngineState;
use crate::runtime_paths::RuntimePaths;
use anyhow::Result;
use serde_json::{json, Value};

fn file_len(path: &std::path::Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Builds the report; `warnings` lists anything that needs attention.
pub fn report(state: &EngineState, paths: &RuntimePaths) -> Result<Value> {
    let semantic = &state.semantic;
    let embed = state.config.embedding.text;
    let features = state.config.features;
    let mut warnings: Vec<String> = Vec::new();

    let mut tenant_ids: Vec<String> = std::fs::read_dir(paths.root().join("tenants"))
        .map(|entries| {
            entries
                .filter_map(std::result::Result::ok)
                .filter(|e| e.path().is_dir())
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    tenant_ids.sort();

    let mut tenants = Vec::new();
    for tenant_id in &tenant_ids {
        let tenant = match state.tenant_store(tenant_id) {
            Ok(tenant) => tenant,
            Err(err) => {
                warnings.push(format!("tenant {tenant_id}: cannot open ({err})"));
                continue;
            }
        };
        let tenant_stats = tenant.detailed_db_stats()?;
        let (vectors, without_embedding) = tenant.stored_vector_counts()?;
        let db_path = paths.tenant_db(tenant_id);
        let mut wal_path = db_path.clone().into_os_string();
        wal_path.push("-wal");
        let wal_bytes = file_len(std::path::Path::new(&wal_path));
        if without_embedding > 0 {
            warnings.push(format!(
                "tenant {tenant_id}: {without_embedding} vectors have no stored embedding and are not searchable; re-ingest them"
            ));
        }
        if wal_bytes > 256 * 1024 * 1024 {
            warnings.push(format!(
                "tenant {tenant_id}: WAL is {wal_bytes} bytes; checkpoint may be blocked"
            ));
        }
        tenants.push(json!({
            "tenant": tenant_id,
            "memories": tenant_stats.memory_count,
            "vectors": vectors,
            "vectors_without_embedding": without_embedding,
            "fact_versions": tenant_stats.fact_version_count,
            "memory_cards": tenant_stats.memory_card_count,
            "edges": tenant_stats.edge_count,
            "db_bytes": file_len(&db_path),
            "db_used_bytes": tenant_stats.used_bytes,
            "wal_bytes": wal_bytes,
        }));
    }

    let mut vector_config = state.config.vector;
    vector_config.dimensions = semantic.embedding_dim();
    Ok(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "data_root": paths.root().display().to_string(),
        "device": semantic.device_label(),
        "embedding": {
            "model": semantic.embedding_model_id(),
            "dimensions": semantic.embedding_dim(),
            "max_tokens": semantic.embed_max_tokens(),
            "text_mode": embed.mode.as_str(),
            "context_window": embed.window,
            "query_instruction": semantic.query_instruction(),
            "cache_hits": semantic.embed_cache_hits(),
            "cache_misses": semantic.embed_cache_misses(),
        },
        "reranker": semantic.rerank_mode(),
        "vectors": {
            "quantization": vector_config.quantization.name(),
            "flat_threshold": vector_config.flat_threshold,
            "rescore_factor": vector_config.rescore_factor,
        },
        "disabled_structures": features.disabled_names(),
        "tenants": tenants,
        "warnings": warnings,
    }))
}
