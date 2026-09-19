//! Assembles an engine (models, tenant stores, analytics) for the HTTP
//! server, the embedded [`crate::db`] API and the CLI.

use crate::api::{self, EngineState};
use crate::config::Config;
use crate::runtime_paths::RuntimePaths;
use crate::{analytics, ml, platform, semantic, storage};
use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::{info, warn};

/// Builds the engine rooted at `paths`. Must run inside a Tokio runtime
/// (background writers are spawned on it).
pub async fn build_state(
    paths: &RuntimePaths,
    auth: api::AuthConfig,
    mut config: Config,
) -> Result<EngineState> {
    paths.ensure_dirs()?;
    paths.apply_process_env_defaults();
    info!(root = %paths.root().display(), "Runtime data root");

    let ranking_path = paths.root().join("ranking_config.json");
    if ranking_path.exists() {
        let config_data = std::fs::read_to_string(&ranking_path)
            .with_context(|| format!("failed to read ranking config {}", ranking_path.display()))?;
        config.ranking = serde_json::from_str(&config_data).with_context(|| {
            format!("failed to parse ranking config {}", ranking_path.display())
        })?;
    }

    if !config.features.disabled_names().is_empty() {
        info!(disabled = ?config.features.disabled_names(), "Ingest structures disabled");
    }

    info!(profile = config.heuristics.name(), "Heuristics profile");

    if config.lanes != crate::retrieval::lanes::Lanes::default() {
        info!(enabled = ?config.lanes.enabled_names(), "Retrieval lanes restricted");
    }

    let cache_path = config
        .embedding
        .cache_path
        .clone()
        .unwrap_or_else(|| paths.embedding_cache().to_path_buf());
    config.embedding.cache_path = Some(cache_path.clone());
    let semantic = Arc::new(
        semantic::SemanticInference::with_config(
            Some(cache_path),
            &config.embedding,
            &config.rerank,
        )
        .await?,
    );
    config.embedding.dimension = Some(semantic.embedding_dim());
    let extractor = crate::extract::init(&config.extractor, config.heuristics)?;
    info!(extractor = extractor.name(), "Fact extractor");
    info!(
        model_id = %semantic.embedding_model_id(),
        dims = %semantic.embedding_dim(),
        device = %semantic.device_label(),
        executors = %semantic.executor_count(),
        rerank = semantic.rerank_mode(),
        "Models loaded"
    );

    let intent_classifier = if config.server.ml_intent {
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

    let mut vector_config = config.vector;
    vector_config.dimensions = semantic.embedding_dim();
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

    if ranking_path.exists() {
        info!("Loaded ranking config from ranking_config.json");
    }

    Ok(EngineState {
        tenant_manager,
        analytics,
        semantic,
        auth,
        platform,
        platform_write_tx,
        data_root: Arc::<str>::from(paths.root().display().to_string()),
        config: Arc::new(config.clone()),
        ranking_config: Arc::new(config.ranking.clone()),
        intent_classifier,
        rate_limiter: Arc::new(api::auth::RateLimiter::new()),
    })
}
