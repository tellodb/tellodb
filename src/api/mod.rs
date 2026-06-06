pub mod auth;
pub mod handlers;
pub mod ingest;
pub mod ingest_utils;
pub mod plan;
pub mod planner;
pub mod types;
pub mod utils;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tokio::time::{interval, Duration};

pub use auth::{AuthConfig, DEFAULT_TEST_API_KEY};
pub use handlers::build_api;

use crate::analytics::MetricVault;
use crate::ml::QueryIntentClassifier;
use crate::platform::PlatformStore;
use crate::semantic::SemanticInference;
use crate::storage::{TenantDatabaseManager, TenantStore};
use crate::vector_index::VectorIndex;

#[derive(Clone)]
pub struct EngineState {
    pub vector_index: VectorIndex,
    pub tenant_manager: Arc<TenantDatabaseManager>,
    pub analytics: Arc<MetricVault>,
    pub semantic: Arc<SemanticInference>,
    pub auth: AuthConfig,
    pub platform: Arc<PlatformStore>,
    pub platform_write_tx: Sender<PlatformWriteOp>,
    pub data_root: Arc<str>,
    pub ranking_config: Arc<crate::api::types::RankingConfig>,
    pub intent_classifier: Option<Arc<QueryIntentClassifier>>,
    pub rate_limiter: Arc<crate::api::auth::RateLimiter>,
}

impl EngineState {
    pub fn tenant_store(&self, tenant_id: &str) -> Result<Arc<TenantStore>, anyhow::Error> {
        self.tenant_manager.get_tenant(tenant_id)
    }
}

#[derive(Debug, Clone)]
pub enum PlatformWriteOp {
    Usage { user_id: String, endpoint: String },
    Profile { user_id: String, text: String, timestamp_ms: u64, source: String },
}

pub fn start_platform_writer(platform: Arc<PlatformStore>) -> Sender<PlatformWriteOp> {
    let (tx, mut rx): (Sender<PlatformWriteOp>, Receiver<PlatformWriteOp>) = channel(4096);
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_millis(100));
        let mut pending_usage: HashMap<(String, String), u64> = HashMap::new();
        let mut pending_profiles: Vec<(String, String, u64, String)> = Vec::new();

        loop {
            tokio::select! {
                maybe_op = rx.recv() => {
                    let Some(op) = maybe_op else {
                        break;
                    };
                    match op {
                        PlatformWriteOp::Usage { user_id, endpoint } => {
                            *pending_usage.entry((user_id, endpoint)).or_insert(0) += 1;
                        }
                        PlatformWriteOp::Profile { user_id, text, timestamp_ms, source } => {
                            pending_profiles.push((user_id, text, timestamp_ms, source));
                        }
                    }
                    if pending_usage.len() + pending_profiles.len() >= 512 {
                        flush_platform_writes(platform.clone(), &mut pending_usage, &mut pending_profiles).await;
                    }
                }
                _ = ticker.tick() => {
                    if !pending_usage.is_empty() || !pending_profiles.is_empty() {
                        flush_platform_writes(platform.clone(), &mut pending_usage, &mut pending_profiles).await;
                    }
                }
            }
        }

        if !pending_usage.is_empty() || !pending_profiles.is_empty() {
            flush_platform_writes(platform.clone(), &mut pending_usage, &mut pending_profiles)
                .await;
        }
    });
    tx
}

async fn flush_platform_writes(
    platform: Arc<PlatformStore>,
    pending_usage: &mut HashMap<(String, String), u64>,
    pending_profiles: &mut Vec<(String, String, u64, String)>,
) {
    let usage = std::mem::take(pending_usage);
    let profiles = std::mem::take(pending_profiles);
    let _ = tokio::task::spawn_blocking(move || {
        for ((user_id, endpoint), count) in usage {
            if let Err(err) = platform.record_usage_n(user_id.as_str(), endpoint.as_str(), count) {
                tracing::warn!(
                    "failed buffered usage write user={} endpoint={} count={}: {:?}",
                    user_id,
                    endpoint,
                    count,
                    err
                );
            }
        }
        for (user_id, text, timestamp_ms, source) in profiles {
            if let Err(err) = platform.update_profile_from_text(
                user_id.as_str(),
                text.as_str(),
                timestamp_ms,
                source.as_str(),
            ) {
                tracing::warn!(
                    "failed buffered profile write user={} source={}: {:?}",
                    user_id,
                    source,
                    err
                );
            }
        }
    })
    .await;
}
