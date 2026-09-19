pub mod facts;
pub mod ingest;
pub mod mcp;
pub mod platform;
pub mod query;
pub mod system;

use axum::{
    extract::{Request, State},
    http::{header::HeaderValue, StatusCode},
    middleware::{self, Next},
    response::Response,
    routing::{delete, get, post},
    Router,
};
use tower_http::limit::RequestBodyLimitLayer;

use self::facts::{current_fact_handler, fact_history_handler};
use self::ingest::{batch_ingest_handler, ingest_handler};
use self::mcp::mcp_handler;
use self::platform::{
    platform_create_api_key_handler, platform_list_api_keys_handler, platform_login_handler,
    platform_logout_handler, platform_me_handler, platform_profile_handler,
    platform_revoke_api_key_handler, platform_signup_handler, platform_stats_handler,
};
use self::query::{
    analytics_query_handler, graph_export_handler, graph_query_handler, graph_walk_handler,
    query_handler,
};
use self::system::{
    admin_inject_api_key_handler, admin_revoke_api_key_handler, cluster_graph_handler,
    cluster_stats_handler, hardware_stats_handler, health_handler, healthz_handler,
    memory_delete_handler, memory_inspect_handler, metrics_handler, reset_handler, status_handler,
    storage_stats_handler, version_handler, warmup_handler,
};
use crate::api::{auth, EngineState};

const UNTIMED: &[&str] = &[
    "/health",
    "/healthz",
    "/version",
    "/metrics",
    "/ingest",
    "/ingest/batch",
    "/batch-ingest",
    "/reset",
    "/admin/reset",
    "/v1/admin/reset",
];

const DEPRECATED_ROUTE_PATHS: &[&str] = &[
    "/reset",
    "/admin/reset",
    "/batch-ingest",
    "/memory/inspect",
    "/memory/delete",
    "/query/semantic",
];

fn is_deprecated_route(path: &str) -> bool {
    DEPRECATED_ROUTE_PATHS.contains(&path)
}

fn mark_deprecated_response(path: &str, response: &mut axum::response::Response) {
    if is_deprecated_route(path) {
        response.headers_mut().insert("Deprecation", HeaderValue::from_static("true"));
    }
}

async fn deprecation_middleware(req: Request, next: Next) -> Response {
    let path = req.uri().path().to_string();
    let mut response = next.run(req).await;
    mark_deprecated_response(&path, &mut response);
    response
}

async fn rate_limit_middleware(
    State(state): State<EngineState>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let path = req.uri().path().to_string();
    if path == "/health" || path == "/healthz" || path == "/version" || path == "/metrics" {
        return Ok(next.run(req).await);
    }
    let peer = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|info| info.0);
    auth::check_rate_limit(&state, req.headers(), peer)?;
    Ok(next.run(req).await)
}

async fn auth_middleware(
    State(state): State<EngineState>,
    mut req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let principal = auth::authorize_request(req.headers(), &state)?;
    req.extensions_mut().insert(principal);
    Ok(next.run(req).await)
}

async fn request_timeout_middleware(
    State(state): State<EngineState>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let path = req.uri().path().to_string();
    if UNTIMED.contains(&path.as_str()) {
        return Ok(next.run(req).await);
    }
    let timeout_secs = state.config.server.request_timeout_secs;
    match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), next.run(req)).await {
        Ok(resp) => Ok(resp),
        Err(_) => {
            tracing::warn!(path = %path, "request exceeded timeout");
            Err(StatusCode::REQUEST_TIMEOUT)
        }
    }
}

pub fn build_api(state: EngineState) -> Router {
    let platform = Router::new()
        .route("/signup", post(platform_signup_handler))
        .route("/login", post(platform_login_handler))
        .route("/logout", post(platform_logout_handler))
        .route("/me", get(platform_me_handler))
        .route("/api-keys", post(platform_create_api_key_handler))
        .route("/api-keys", get(platform_list_api_keys_handler))
        .route("/api-keys/{prefix}", post(platform_revoke_api_key_handler))
        .route("/stats", get(platform_stats_handler))
        .route("/profile", get(platform_profile_handler))
        .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware));

    let protected = Router::new()
        .route("/health", get(health_handler))
        .route("/version", get(version_handler))
        .route("/status", get(status_handler))
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
        .route("/facts/current", get(current_fact_handler))
        .route("/facts/history", get(fact_history_handler))
        .layer(middleware::from_fn_with_state(state.clone(), auth_middleware))
        .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware));

    Router::new()
        .route("/healthz", get(healthz_handler))
        .route("/metrics", get(metrics_handler))
        .merge(platform)
        .merge(protected)
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024))
        .layer(middleware::from_fn(deprecation_middleware))
        .layer(middleware::from_fn_with_state(state.clone(), request_timeout_middleware))
        .layer(auth::build_cors_layer(&state.config.server.cors_origins))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;

    #[test]
    fn deprecated_aliases_are_marked_without_marking_canonical_routes() {
        for path in DEPRECATED_ROUTE_PATHS {
            let mut response = Response::new(Body::empty());
            mark_deprecated_response(path, &mut response);
            assert_eq!(
                response.headers().get("Deprecation").and_then(|value| value.to_str().ok()),
                Some("true")
            );
        }

        for path in [
            "/v1/admin/reset",
            "/ingest/batch",
            "/v1/memory/inspect",
            "/v1/memory/delete",
            "/query",
        ] {
            let mut response = Response::new(Body::empty());
            mark_deprecated_response(path, &mut response);
            assert!(response.headers().get("Deprecation").is_none());
        }
    }

    #[test]
    fn every_untimed_path_is_present_in_one_exemption_list() {
        for path in [
            "/health",
            "/healthz",
            "/version",
            "/metrics",
            "/ingest",
            "/ingest/batch",
            "/batch-ingest",
            "/reset",
            "/admin/reset",
            "/v1/admin/reset",
        ] {
            assert!(UNTIMED.contains(&path));
        }
    }
}
