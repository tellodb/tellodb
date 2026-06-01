pub mod ingest;
pub mod mcp;
pub mod platform;
pub mod query;
pub mod system;

use axum::{
    routing::{delete, get, post},
    Router,
};
use tower_http::limit::RequestBodyLimitLayer;

use self::ingest::{batch_ingest_handler, ingest_handler};
use self::mcp::mcp_handler;
use self::platform::{
    platform_create_api_key_handler, platform_list_api_keys_handler, platform_login_handler,
    platform_me_handler, platform_profile_handler, platform_revoke_api_key_handler,
    platform_signup_handler, platform_stats_handler,
};
use self::query::{
    analytics_query_handler, graph_export_handler, graph_query_handler, graph_walk_handler,
    query_handler, temporal_query_handler,
};
use self::system::{
    admin_inject_api_key_handler, admin_revoke_api_key_handler, cluster_graph_handler,
    cluster_stats_handler, hardware_stats_handler, health_handler, healthz_handler,
    memory_delete_handler, memory_inspect_handler, metrics_handler, reset_handler, status_handler,
    storage_stats_handler, version_handler, warmup_handler,
};
use crate::api::{auth, EngineState};

pub fn build_api(state: EngineState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/healthz", get(healthz_handler))
        .route("/status", get(status_handler))
        .route("/version", get(version_handler))
        .route("/warmup", post(warmup_handler))
        .route("/reset", post(reset_handler))
        .route("/admin/reset", post(reset_handler))
        .route("/v1/admin/reset", post(reset_handler))
        .route("/admin/clusters/{cluster_id}/stats", get(cluster_stats_handler))
        .route("/admin/clusters/{cluster_id}/storage-stats", get(storage_stats_handler))
        .route("/admin/clusters/{cluster_id}/graph-edges", get(cluster_graph_handler))
        .route("/admin/stats/hardware", get(hardware_stats_handler))
        .route("/admin/api_keys", post(admin_inject_api_key_handler))
        .route("/admin/api_keys/{key_id}", delete(admin_revoke_api_key_handler))
        .route("/signup", post(platform_signup_handler))
        .route("/login", post(platform_login_handler))
        .route("/me", get(platform_me_handler))
        .route("/api-keys", post(platform_create_api_key_handler))
        .route("/api-keys", get(platform_list_api_keys_handler))
        .route("/api-keys/{prefix}", post(platform_revoke_api_key_handler))
        .route("/stats", get(platform_stats_handler))
        .route("/profile", get(platform_profile_handler))
        .route("/mcp", post(mcp_handler))
        .route("/ingest", post(ingest_handler))
        .route("/batch-ingest", post(batch_ingest_handler))
        .route("/ingest/batch", post(batch_ingest_handler))
        .route("/memory/inspect", post(memory_inspect_handler))
        .route("/memory/delete", post(memory_delete_handler))
        .route("/v1/memory/inspect", post(memory_inspect_handler))
        .route("/v1/memory/delete", post(memory_delete_handler))
        .route("/query", post(query_handler))
        .route("/query/semantic", post(query_handler))
        .route("/graph/query", post(graph_query_handler))
        .route("/graph/walk", post(graph_walk_handler))
        .route("/graph/export", post(graph_export_handler))
        .route("/analytics/query", post(analytics_query_handler))
        .route("/temporal/query", get(temporal_query_handler))
        .route("/metrics", get(metrics_handler))
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024)) // 10MB max body
        .layer(auth::build_cors_layer())
        .with_state(state)
}
