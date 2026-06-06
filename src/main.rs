use std::env;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::signal;
use tokio::time::{interval, Duration};
use tracing::{error, info, warn};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod analytics;
mod api;

mod fts;
mod graph;
mod graph_inference;
mod lifecycle;
mod metrics;
mod ml;
mod platform;
mod retrieval;
mod runtime_paths;
mod semantic;
mod storage;
mod vector_index;

pub fn init_tracing_subscriber() {
    use tracing_subscriber::{filter::EnvFilter, fmt, prelude::*, Registry};

    let env_filter =
        EnvFilter::builder().with_default_directive(tracing::Level::INFO.into()).from_env_lossy();

    let fmt_layer =
        fmt::layer().with_target(true).with_thread_ids(true).with_file(true).with_line_number(true);

    let subscriber = Registry::default().with(env_filter).with(fmt_layer);

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");
}

fn env_var_bool(name: &str) -> bool {
    std::env::var(name).ok().is_some_and(|v| {
        matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    })
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    init_tracing_subscriber();
    info!("Initializing Temporal Multi-Model Memory Engine...");

    let runtime_paths = runtime_paths::RuntimePaths::from_env()?;
    runtime_paths.ensure_dirs()?;
    runtime_paths.apply_process_env_defaults();

    info!(root = %runtime_paths.root().display(), "Runtime data root");

    info!(
        "Initializing Semantic Pipeline (embedder + MiniLM + BERT-NER)... [This may take a moment]"
    );
    let semantic = Arc::new(semantic::SemanticInference::new().await?);
    info!(
        model_id = %semantic.embedding_model_id(),
        dims = %semantic.embedding_dim(),
        "Semantic embedder initialized"
    );
    info!(
        device = %semantic.device_label(),
        executors = %semantic.executor_count(),
        "Semantic device selected"
    );

    let intent_classifier = if env_var_bool("TEMPORAL_MEMORY_ML_INTENT") {
        info!("Initializing ML Intent Classifier (embedding prototypes)...");
        match ml::QueryIntentClassifier::new(semantic.clone()) {
            Ok(c) => {
                info!(count = %c.prototype_count(), "ML Intent Classifier ready");
                Some(Arc::new(c))
            }
            Err(e) => {
                warn!("ML Intent Classifier failed to initialize: {e}. Falling back to heuristic rules.");
                None
            }
        }
    } else {
        None
    };

    let hnsw_max =
        env::var("TELLODB_HNSW_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
    let hnsw_conn =
        env::var("TELLODB_HNSW_CONNECTIVITY").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let hnsw_ef_add =
        env::var("TELLODB_HNSW_EF_ADD").ok().and_then(|v| v.parse().ok()).unwrap_or(128);
    let hnsw_ef_srch =
        env::var("TELLODB_HNSW_EF_SEARCH").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    info!("Initializing Vector Substrate (HNSW)...");
    let vector_index = vector_index::VectorIndex::new(
        semantic.embedding_dim(),
        hnsw_max,
        hnsw_conn,
        hnsw_ef_add,
        hnsw_ef_srch,
        Some(runtime_paths.vector_index().to_string_lossy().as_ref()),
    )?;

    info!("Initializing Tenant Database Manager (Sharded SQLite)...");
    let tenant_manager = Arc::new(storage::TenantDatabaseManager::new(runtime_paths.clone()));

    info!("Initializing Platform Substrate...");
    let platform = Arc::new(platform::PlatformStore::new(
        runtime_paths.platform_db().to_string_lossy().as_ref(),
    )?);
    let platform_write_tx = api::start_platform_writer(platform.clone());

    info!("Initializing Analytics Substrate (Numeric Vault)...");
    let analytics =
        Arc::new(analytics::MetricVault::new(tenant_manager.clone(), semantic.clone())?);

    let auth = api::AuthConfig::from_env();
    info!(
        test_key = %api::DEFAULT_TEST_API_KEY,
        "API key auth enabled on all routes. Override with TEMPORAL_MEMORY_API_KEY or TELLODB_API_KEY."
    );

    let host = env::var("TEMPORAL_MEMORY_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = env::var("PORT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| env::var("TEMPORAL_MEMORY_PORT").ok().filter(|value| !value.trim().is_empty()))
        .unwrap_or_else(|| "3000".to_string());
    let bind_address = format!("{}:{}", host, port);

    let vector_index_for_checkpoint = vector_index.clone();
    let tenant_manager_for_checkpoint = tenant_manager.clone();

    let mut ranking_config = api::types::RankingConfig::default();
    if let Ok(config_data) =
        std::fs::read_to_string(runtime_paths.root().join("ranking_config.json"))
    {
        if let Ok(parsed) = serde_json::from_str(&config_data) {
            info!("Loaded tuned ranking config from ranking_config.json");
            ranking_config = parsed;
        } else {
            warn!("Failed to parse ranking_config.json, using defaults.");
        }
    }

    let state = api::EngineState {
        vector_index,
        tenant_manager: tenant_manager.clone(),
        analytics,
        semantic,
        auth,
        platform,
        platform_write_tx,
        data_root: Arc::<str>::from(runtime_paths.root().display().to_string()),
        ranking_config: Arc::new(ranking_config),
        intent_classifier,
        rate_limiter: Arc::new(api::auth::RateLimiter::new()),
    };

    let app = api::build_api(state);

    {
        let checkpoint_every_secs = env::var("TEMPORAL_MEMORY_VECTOR_CHECKPOINT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(30);
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(checkpoint_every_secs));
            loop {
                ticker.tick().await;
                let indices = [&vector_index_for_checkpoint];
                let results: Vec<_> = indices.iter().map(|idx| idx.checkpoint_if_dirty()).collect();
                for (i, result) in results.into_iter().enumerate() {
                    if let Err(err) = result {
                        error!(index = %i, error = ?err, "Vector checkpoint error");
                    }
                }
            }
        });
    }

    // Periodic WAL checkpoint for tenant databases
    {
        let ckpt_tenants = tenant_manager_for_checkpoint;
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(300)); // every 5 min
            loop {
                ticker.tick().await;
                let tenants = ckpt_tenants.all_tenants();
                for tenant in tenants {
                    if let Err(e) = tenant.checkpoint() {
                        error!("WAL checkpoint failed: {:?}", e);
                    }
                }
            }
        });
    }

    // Periodic memory lifecycle expiration sweep
    {
        let sweep_tenants = tenant_manager.clone();
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(300)); // every 5 min
            loop {
                ticker.tick().await;
                let now_ms = match std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH) {
                    Ok(d) => d.as_millis() as u64,
                    Err(_) => continue,
                };
                let tenants = sweep_tenants.all_tenants();
                for tenant in tenants {
                    if let Err(e) = tenant.expire_records(now_ms) {
                        error!("Lifecycle expiration sweep failed: {:?}", e);
                    }
                }
            }
        });
    }

    info!(address = %bind_address, "Memory Engine live");
    let listener = TcpListener::bind(&bind_address).await?;
    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Shutdown signal received (Ctrl+C)");
        }
        _ = terminate => {
            info!("Shutdown signal received (SIGTERM)");
        }
    }

    tokio::time::sleep(Duration::from_millis(500)).await;
    info!("Shutting down...");
}
