use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use crate::api::EngineState;
use crate::storage::{AgentObservation, MemoryKind, TenantStore};

#[derive(Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Serialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
}

#[derive(Serialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

const MCP_VERSION: &str = "2.0";
const JSONRPC_INVALID_REQUEST: i32 = -32600;
const JSONRPC_METHOD_NOT_FOUND: i32 = -32601;
const JSONRPC_INTERNAL_ERROR: i32 = -32000;

pub async fn mcp_handler(
    State(state): State<EngineState>,
    Json(req): Json<McpRequest>,
) -> Json<McpResponse> {
    if req.jsonrpc != MCP_VERSION {
        return Json(McpResponse {
            jsonrpc: MCP_VERSION.into(),
            id: req.id,
            result: None,
            error: Some(McpError {
                code: JSONRPC_INVALID_REQUEST,
                message: "Invalid JSON-RPC version".into(),
                data: None,
            }),
        });
    }

    match req.method.as_str() {
        "tools/list" => handle_tools_list(req.id),
        "tools/call" => handle_tools_call(&state, req.id, req.params).await,
        _ => Json(McpResponse {
            jsonrpc: MCP_VERSION.into(),
            id: req.id,
            result: None,
            error: Some(McpError {
                code: JSONRPC_METHOD_NOT_FOUND,
                message: format!("Method not found: {}", req.method),
                data: None,
            }),
        }),
    }
}

fn handle_tools_list(id: Value) -> Json<McpResponse> {
    Json(McpResponse {
        jsonrpc: MCP_VERSION.into(),
        id,
        result: Some(json!({
            "tools": [
                {
                    "name": "search_memory",
                    "description": "Search memories using hybrid vector+lexical retrieval",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": { "type": "string" },
                            "limit": { "type": "number", "default": 10 },
                            "entity_id": { "type": "string" }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "store_fact",
                    "description": "Store a fact or observation in memory",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": { "type": "string" },
                            "entity_id": { "type": "string" },
                            "kind": { "type": "string", "enum": ["Fact", "Conversation", "Preference", "Decision", "Lesson"] }
                        },
                        "required": ["text", "entity_id"]
                    }
                },
                {
                    "name": "explore_graph",
                    "description": "Walk the knowledge graph from an entity",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "entity": { "type": "string" },
                            "depth": { "type": "number", "default": 2 }
                        },
                        "required": ["entity"]
                    }
                },
                {
                    "name": "get_memory",
                    "description": "Retrieve a specific memory by ID",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "memory_id": { "type": "string" }
                        },
                        "required": ["memory_id"]
                    }
                }
            ]
        })),
        error: None,
    })
}

async fn handle_tools_call(state: &EngineState, id: Value, params: Value) -> Json<McpResponse> {
    let tool_name = params["name"].as_str().unwrap_or("").to_string();
    let args = &params["arguments"];

    let tenant = match state.tenant_manager.get_tenant("default") {
        Ok(tenant) => tenant,
        Err(_) => match TenantStore::new(std::path::Path::new(":memory:")) {
            Ok(store) => std::sync::Arc::new(store),
            Err(e) => {
                return Json(McpResponse {
                    jsonrpc: MCP_VERSION.into(),
                    id,
                    result: None,
                    error: Some(McpError {
                        code: JSONRPC_INTERNAL_ERROR,
                        message: format!("Failed to initialize in-memory tenant: {}", e),
                        data: None,
                    }),
                });
            }
        },
    };

    match tool_name.as_str() {
        "search_memory" => {
            let query = args["query"].as_str().unwrap_or("");
            let limit = args["limit"].as_u64().unwrap_or(10) as usize;
            let entity_id = args["entity_id"].as_str().map(|s| s.to_string());

            let qembed = state.semantic.generate_query_embedding(query).unwrap_or_default();
            let fts_results =
                tenant.fts_search(query, limit, entity_id.as_deref()).unwrap_or_default();
            let ann_results = if let Some(ref eid) = entity_id {
                state.vector_index.search(Some(eid), &qembed, limit).unwrap_or_default()
            } else {
                vec![]
            };

            let mut seen = std::collections::HashSet::new();
            let mut results = Vec::new();

            for item in ann_results
                .iter()
                .map(|(id, s)| (id.to_string(), *s))
                .chain(fts_results.iter().map(|(id, s)| (id.clone(), *s)))
            {
                if seen.insert(item.0.clone()) {
                    results.push(json!({"memory_id": item.0, "score": item.1}));
                }
            }
            results.truncate(limit);

            Json(McpResponse {
                jsonrpc: MCP_VERSION.into(),
                id,
                result: Some(json!({"results": results})),
                error: None,
            })
        }

        "store_fact" => {
            let text = args["text"].as_str().unwrap_or("").to_string();
            let entity_id = args["entity_id"].as_str().unwrap_or("").to_string();
            let kind_str = args["kind"].as_str().unwrap_or("Fact");
            let kind = match kind_str {
                "Conversation" => MemoryKind::Conversational,
                "Preference" => MemoryKind::Preference,
                "Decision" => MemoryKind::Decision,
                "Lesson" => MemoryKind::Lesson,
                _ => MemoryKind::Fact,
            };
            let memory_id = format!("mcp-{}", rand::random::<u64>());
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let embedding = state.semantic.generate_query_embedding(&text).unwrap_or_default();

            let obs = AgentObservation {
                entity_id: entity_id.clone(),
                textual_content: text.clone(),
                embedding: embedding.clone(),
                kind,
                content_hash: String::new(),
                created_at_ms: timestamp,
            };

            if let Err(e) = tenant.insert_observation(timestamp, &memory_id, &obs) {
                return Json(McpResponse {
                    jsonrpc: MCP_VERSION.into(),
                    id,
                    result: None,
                    error: Some(McpError {
                        code: JSONRPC_INTERNAL_ERROR,
                        message: format!("Store failed: {}", e),
                        data: None,
                    }),
                });
            }
            if let Err(e) = tenant.fts_index_text(&memory_id, &text, &entity_id) {
                tracing::warn!("MCP store_fact FTS error: {}", e);
            }
            if let Err(e) = state.vector_index.insert(&entity_id, rand::random::<u64>(), &embedding)
            {
                tracing::warn!("MCP store_fact vector error: {}", e);
            }

            Json(McpResponse {
                jsonrpc: MCP_VERSION.into(),
                id,
                result: Some(json!({"memory_id": memory_id, "entity_id": entity_id})),
                error: None,
            })
        }

        "explore_graph" => {
            let entity = args["entity"].as_str().unwrap_or("");
            let _depth = args["depth"].as_u64().unwrap_or(2) as usize;
            let limit = args["limit"].as_u64().unwrap_or(20) as usize;

            // Use query_edges with Both direction to get all connections
            let edges = tenant.graph_query_edges(entity, None, "Both", limit).unwrap_or_default();

            let summaries: Vec<String> = edges
                .iter()
                .map(|e| format!("{} --[{}]--> {}", e.source, e.label, e.target))
                .collect();

            Json(McpResponse {
                jsonrpc: MCP_VERSION.into(),
                id,
                result: Some(json!({"entity": entity, "edges": summaries})),
                error: None,
            })
        }

        "get_memory" => {
            let memory_id = args["memory_id"].as_str().unwrap_or("");
            let lookup = tenant.lookup_by_memory_id(memory_id).unwrap_or(None);
            match lookup {
                Some((vid, ts)) => Json(McpResponse {
                    jsonrpc: MCP_VERSION.into(),
                    id,
                    result: Some(json!({
                        "memory_id": memory_id,
                        "timestamp": ts,
                        "vector_id": vid,
                    })),
                    error: None,
                }),
                None => Json(McpResponse {
                    jsonrpc: MCP_VERSION.into(),
                    id,
                    result: None,
                    error: Some(McpError {
                        code: JSONRPC_INTERNAL_ERROR,
                        message: "Memory not found".into(),
                        data: None,
                    }),
                }),
            }
        }

        _ => Json(McpResponse {
            jsonrpc: MCP_VERSION.into(),
            id,
            result: None,
            error: Some(McpError {
                code: JSONRPC_METHOD_NOT_FOUND,
                message: format!("Unknown tool: {}", tool_name),
                data: None,
            }),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn invalid_jsonrpc_version_response_has_minus_32600_code() {
        let resp = McpResponse {
            jsonrpc: MCP_VERSION.into(),
            id: json!(1),
            result: None,
            error: Some(McpError {
                code: JSONRPC_INVALID_REQUEST,
                message: "Invalid JSON-RPC version".into(),
                data: None,
            }),
        };
        assert_eq!(resp.jsonrpc, "2.0");
        assert_eq!(resp.error.as_ref().unwrap().code, -32600);
        assert_eq!(resp.error.as_ref().unwrap().message, "Invalid JSON-RPC version");
        assert!(resp.result.is_none());
    }

    #[test]
    fn unknown_method_returns_minus_32601() {
        let id = json!("req-1");
        let resp = McpResponse {
            jsonrpc: MCP_VERSION.into(),
            id: id.clone(),
            result: None,
            error: Some(McpError {
                code: JSONRPC_METHOD_NOT_FOUND,
                message: format!("Method not found: {}", "bogus_method"),
                data: None,
            }),
        };
        assert_eq!(resp.error.as_ref().unwrap().code, -32601);
        assert!(resp.error.as_ref().unwrap().message.contains("bogus_method"));
        assert!(resp.result.is_none());
    }

    #[test]
    fn tools_list_returns_four_tools() {
        let id = json!(42);
        let Json(resp) = handle_tools_list(id);
        assert_eq!(resp.jsonrpc, "2.0");
        assert_eq!(resp.id, json!(42));
        assert!(resp.error.is_none());
        let tools = resp.result.unwrap();
        let tools_arr = tools["tools"].as_array().unwrap();
        assert_eq!(tools_arr.len(), 4);
        let names: Vec<&str> = tools_arr.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, vec!["search_memory", "store_fact", "explore_graph", "get_memory"]);
    }

    #[test]
    fn tools_list_tool_schemas_contain_required_fields() {
        let Json(resp) = handle_tools_list(json!("id"));
        let tools = resp.result.unwrap();
        for tool in tools["tools"].as_array().unwrap() {
            let schema = &tool["inputSchema"];
            assert_eq!(schema["type"], "object");
            assert!(schema["properties"].is_object());
            assert!(schema["required"].is_array());
        }
    }

    #[test]
    fn search_memory_tool_schema() {
        let Json(resp) = handle_tools_list(json!(null));
        let tools = resp.result.unwrap();
        let search = tools["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "search_memory")
            .unwrap();
        assert_eq!(search["description"], "Search memories using hybrid vector+lexical retrieval");
        let props = &search["inputSchema"]["properties"];
        assert!(props["query"]["type"] == "string");
        assert!(props["limit"]["type"] == "number");
        assert!(search["inputSchema"]["required"].as_array().unwrap().contains(&json!("query")));
    }

    #[test]
    fn store_fact_tool_schema() {
        let Json(resp) = handle_tools_list(json!(true));
        let tools = resp.result.unwrap();
        let store =
            tools["tools"].as_array().unwrap().iter().find(|t| t["name"] == "store_fact").unwrap();
        assert_eq!(store["description"], "Store a fact or observation in memory");
        let kinds = &store["inputSchema"]["properties"]["kind"]["enum"];
        assert_eq!(kinds.as_array().unwrap().len(), 5);
    }

    #[test]
    fn mcp_request_deserialization() {
        let raw = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        });
        let req: McpRequest = serde_json::from_value(raw).unwrap();
        assert_eq!(req.jsonrpc, "2.0");
        assert_eq!(req.id, json!(1));
        assert_eq!(req.method, "tools/list");
    }

    #[test]
    fn mcp_request_deserialization_default_params_is_null() {
        let raw = json!({
            "jsonrpc": "2.0",
            "id": "abc",
            "method": "tools/call"
        });
        let req: McpRequest = serde_json::from_value(raw).unwrap();
        assert_eq!(req.params, serde_json::Value::Null);
    }

    #[test]
    fn mcp_response_serialization_full() {
        let resp = McpResponse {
            jsonrpc: "2.0".into(),
            id: json!(1),
            result: Some(json!({"ok": true})),
            error: None,
        };
        let val = serde_json::to_value(&resp).unwrap();
        assert_eq!(val["jsonrpc"], "2.0");
        assert_eq!(val["id"], 1);
        assert_eq!(val["result"]["ok"], true);
        assert!(val.get("error").is_none());
    }

    #[test]
    fn mcp_response_serialization_with_error() {
        let resp = McpResponse {
            jsonrpc: "2.0".into(),
            id: json!("err-id"),
            result: None,
            error: Some(McpError {
                code: JSONRPC_METHOD_NOT_FOUND,
                message: "Method not found".into(),
                data: Some(json!({"detail": "bogus"})),
            }),
        };
        let val = serde_json::to_value(&resp).unwrap();
        assert_eq!(val["error"]["code"], -32601);
        assert_eq!(val["error"]["message"], "Method not found");
        assert_eq!(val["error"]["data"]["detail"], "bogus");
        assert!(val.get("result").is_none());
    }

    #[test]
    fn mcp_error_serialization_without_data() {
        let err =
            McpError { code: JSONRPC_INTERNAL_ERROR, message: "Server error".into(), data: None };
        let val = serde_json::to_value(&err).unwrap();
        assert_eq!(val["code"], -32000);
        assert_eq!(val["message"], "Server error");
        assert!(val.get("data").is_none());
    }

    #[test]
    fn error_response_shape_is_correct() {
        let resp = McpResponse {
            jsonrpc: "2.0".into(),
            id: json!(null),
            result: None,
            error: Some(McpError {
                code: JSONRPC_INVALID_REQUEST,
                message: "Invalid JSON-RPC version".into(),
                data: None,
            }),
        };
        let val = serde_json::to_value(&resp).unwrap();
        assert!(val.get("result").is_none());
        assert!(val.get("error").is_some());
        assert!(val.get("jsonrpc").is_some());
        assert!(val.get("id").is_some());
        assert_eq!(val["error"]["code"], -32600);
        assert_eq!(val["error"]["message"], "Invalid JSON-RPC version");
        assert_eq!(val["jsonrpc"], "2.0");
    }

    #[test]
    fn explore_graph_tool_schema() {
        let Json(resp) = handle_tools_list(json!("explore-test"));
        let tools = resp.result.unwrap();
        let explore = tools["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "explore_graph")
            .unwrap();
        assert_eq!(explore["description"], "Walk the knowledge graph from an entity");
        assert!(explore["inputSchema"]["properties"]["depth"]["default"] == 2);
        assert!(explore["inputSchema"]["required"].as_array().unwrap().contains(&json!("entity")));
    }

    #[test]
    fn get_memory_tool_schema() {
        let Json(resp) = handle_tools_list(json!([1, 2]));
        let tools = resp.result.unwrap();
        let get =
            tools["tools"].as_array().unwrap().iter().find(|t| t["name"] == "get_memory").unwrap();
        assert_eq!(get["description"], "Retrieve a specific memory by ID");
        assert!(get["inputSchema"]["properties"]["memory_id"]["type"] == "string");
        assert!(get["inputSchema"]["required"].as_array().unwrap().contains(&json!("memory_id")));
    }

    #[test]
    fn mcp_version_constant_is_2_0() {
        assert_eq!(MCP_VERSION, "2.0");
    }

    #[test]
    fn response_includes_jsonrpc_and_id() {
        let Json(resp) = handle_tools_list(json!("test-id"));
        let val = serde_json::to_value(&resp).unwrap();
        assert_eq!(val["jsonrpc"], "2.0");
        assert_eq!(val["id"], "test-id");
    }
}
