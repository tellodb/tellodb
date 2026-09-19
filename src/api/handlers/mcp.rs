//! `POST /mcp`: the same Model Context Protocol server as `tellodb mcp`,
//! reached over HTTP with API-key auth and tenant scoping. The protocol and
//! the tools live in [`crate::mcp_stdio`]; this module handles the request.

use crate::api::auth::{
    principal_namespace_prefix, principal_user_id, record_usage_for_principal, RequestPrincipal,
};
use crate::api::EngineState;
use crate::db::Engine;
use crate::mcp_stdio::McpServer;
use axum::{
    extract::{Extension, State},
    Json,
};
use serde_json::{json, Value};

const JSONRPC_INTERNAL_ERROR: i32 = -32000;

fn rpc_error(id: Value, code: i32, message: impl Into<String>) -> Json<Value> {
    Json(json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message.into() }
    }))
}

pub async fn mcp_handler(
    State(state): State<EngineState>,
    Extension(principal): Extension<RequestPrincipal>,
    Json(request): Json<Value>,
) -> Json<Value> {
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    if request["jsonrpc"] != "2.0" {
        return rpc_error(id, -32600, "Invalid JSON-RPC version");
    }

    let tenant_id = principal_user_id(&principal).unwrap_or("default");
    let tenant = match state.tenant_store(tenant_id) {
        Ok(tenant) => tenant,
        Err(err) => {
            tracing::error!(error = ?err, tenant_id, "MCP tenant lookup failed");
            return rpc_error(id, JSONRPC_INTERNAL_ERROR, "Failed to open tenant store");
        }
    };
    // Memories of callers scoped to a namespace live under that prefix.
    let default_entity = principal_namespace_prefix(&principal)
        .map(|prefix| prefix.trim_end_matches(':').to_string())
        .unwrap_or_else(|| "user".to_string());

    let server = McpServer::new(Engine::from_parts(state.clone(), tenant), default_entity);
    let response = server.handle(request).await;
    record_usage_for_principal(&state, &principal, "mcp");
    // Notifications carry no id and get an empty 200 body.
    Json(response.unwrap_or(Value::Null))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_are_json_rpc_errors_with_the_request_id() {
        let Json(unauthorized) = rpc_error(json!("req-1"), -32001, "Unauthorized");
        assert_eq!(unauthorized["jsonrpc"], "2.0");
        assert_eq!(unauthorized["id"], json!("req-1"));
        assert_eq!(unauthorized["error"]["code"], -32001);
        assert!(unauthorized.get("result").is_none());

        let Json(bad_version) = rpc_error(json!(1), -32600, "Invalid JSON-RPC version");
        assert_eq!(bad_version["error"]["code"], -32600);
        assert_eq!(bad_version["error"]["message"], "Invalid JSON-RPC version");
    }
}
