//! Assembles an engine (models, tenant stores, analytics) for the HTTP
//! server, the embedded [`crate::db`] API and the CLI.

use crate::api::{self, EngineState};
use crate::runtime_paths::RuntimePaths;
use crate::{analytics, ml, platform, semantic, storage, vector_index};
use anyhow::Result;
use std::sync::Arc;
use tracing::{info, warn};

fn env_var_bool(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|v| {
        matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    })
}

/// Builds the engine rooted at `paths`. Must run inside a Tokio runtime
/// (background writers are spawned on it).
pub async fn build_state(paths: &RuntimePaths, auth: api::AuthConfig) -> Result<EngineState> {
    paths.ensure_dirs()?;
    paths.apply_process_env_defaults();
    info!(root = %paths.root().display(), "Runtime data root");

    let features = crate::features::init_from_env()?;
    if !features.disabled_names().is_empty() {
        info!(disabled = ?features.disabled_names(), "Ingest structures disabled");
    }

    let heuristics = crate::heuristics::init_from_env()?;
    info!(profile = heuristics.name(), "Heuristics profile");

    let lanes = crate::retrieval::lanes::init_from_env()?;
    if lanes != crate::retrieval::lanes::Lanes::default() {
        info!(enabled = ?lanes.enabled_names(), "Retrieval lanes restricted");
    }

    // Loading an encoder model is slow and a bad configuration should fail the
    // process rather than every ingest.
    let extractor = crate::extract::init_from_env()?;
    info!(extractor = extractor.name(), "Fact extractor");

    let cache_path = std::env::var("TELLODB_EMBEDDING_CACHE_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| paths.embedding_cache().to_path_buf());
    let semantic = Arc::new(semantic::SemanticInference::with_cache_path(Some(cache_path)).await?);
    info!(
        model_id = %semantic.embedding_model_id(),
        dims = %semantic.embedding_dim(),
        device = %semantic.device_label(),
        executors = %semantic.executor_count(),
        rerank = semantic.rerank_mode(),
        "Models loaded"
    );

    let intent_classifier = if env_var_bool("TEMPORAL_MEMORY_ML_INTENT") {
        match ml::QueryIntentClassifier::new(semantic.clone()) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                warn!("ML intent classifier failed to initialize: {e}; using heuristic rules");
                None
            }
        }
    } else {
        None
    };

    let vector_config = vector_index::VectorConfig::from_env(semantic.embedding_dim())?;
    info!(
        quantization = vector_config.quantization.name(),
        flat_threshold = vector_config.flat_threshold,
        rescore_factor = vector_config.rescore_factor,
        "Vector segments"
    );
    let tenant_manager =
        Arc::new(storage::TenantDatabaseManager::new(paths.clone(), vector_config));

    let platform =
        Arc::new(platform::PlatformStore::new(paths.platform_db().to_string_lossy().as_ref())?);
    let platform_write_tx = api::start_platform_writer(platform.clone());
    let analytics = Arc::new(analytics::MetricVault::new(tenant_manager.clone()));

    let mut ranking_config = api::types::RankingConfig::default();
    if let Ok(config_data) = std::fs::read_to_string(paths.root().join("ranking_config.json")) {
        match serde_json::from_str(&config_data) {
            Ok(parsed) => {
                info!("Loaded ranking config from ranking_config.json");
                ranking_config = parsed;
            }
            Err(err) => warn!(error = %err, "Failed to parse ranking_config.json; using defaults"),
        }
    }

    Ok(EngineState {
        tenant_manager,
        analytics,
        semantic,
        auth,
        platform,
        platform_write_tx,
        data_root: Arc::<str>::from(paths.root().display().to_string()),
        ranking_config: Arc::new(ranking_config),
        intent_classifier,
        rate_limiter: Arc::new(api::auth::RateLimiter::new()),
    })
}
