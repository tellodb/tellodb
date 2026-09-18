//! Model Context Protocol server over stdio (`tellodb mcp`).
//!
//! Newline-delimited JSON-RPC 2.0 on stdin/stdout; logs go to stderr. Tools
//! run against an embedded [`Engine`], so no HTTP server or API key is
//! involved.

use crate::db::{Engine, Memory, Query};
use anyhow::Result;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

const PROTOCOL_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

pub struct McpServer {
    engine: Engine,
    default_entity: String,
}

fn tool_definitions() -> Value {
    json!([
        {
            "name": "remember",
            "description": "Store a memory (a fact, preference, decision or conversation turn) for later recall.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "What to remember." },
                    "entity_id": { "type": "string", "description": "Whose memory this is (defaults to the server's entity)." },
                    "session_id": { "type": "string" },
                    "turn_index": { "type": "integer", "minimum": 0 },
                    "role": { "type": "string", "description": "Speaker, e.g. user or assistant." },
                    "kind": { "type": "string", "enum": ["fact", "preference", "decision", "lesson", "conversational"] },
                    "timestamp_ms": { "type": "integer", "description": "When it happened (Unix ms); defaults to now." }
                },
                "required": ["text"]
            }
        },
        {
            "name": "recall",
            "description": "Search memories relevant to a question. Results include whether a fact has been superseded.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "entity_id": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10 },
                    "as_of_ms": { "type": "integer", "description": "Only consider memories that existed at this time." }
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_memory",
            "description": "Fetch one memory by its id.",
            "inputSchema": {
                "type": "object",
                "properties": { "memory_id": { "type": "string" } },
                "required": ["memory_id"]
            }
        },
        {
            "name": "explore_graph",
            "description": "List knowledge-graph edges touching an entity or subject.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node": { "type": "string", "description": "Entity or subject name." },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 500, "default": 20 }
                },
                "required": ["node"]
            }
        },
        {
            "name": "fact_history",
            "description": "How a fact changed over time: each value, when it held, and which memories stated it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "fact_key": { "type": "string" },
                    "entity_id": { "type": "string" },
                    "as_of_ms": { "type": "integer", "description": "Return only the value that held at this time." }
                },
                "required": ["fact_key"]
            }
        },
        {
            "name": "current_fact",
            "description": "The current value of a tracked fact such as residence, employer or job_title.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "fact_key": { "type": "string" },
                    "entity_id": { "type": "string" }
                },
                "required": ["fact_key"]
            }
        }
    ])
}

/// Echoes the client's protocol version when we speak it, else our newest.
fn negotiate_protocol_version(requested: &str) -> &'static str {
    PROTOCOL_VERSIONS.iter().find(|v| **v == requested).copied().unwrap_or(PROTOCOL_VERSIONS[0])
}

fn text_result(value: &Value, is_error: bool) -> Value {
    json!({
        "content": [{ "type": "text", "text": serde_json::to_string_pretty(value).unwrap_or_default() }],
        "isError": is_error,
    })
}

fn rpc_result(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn rpc_error(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message.into() } })
}

impl McpServer {
    pub fn new(engine: Engine, default_entity: impl Into<String>) -> Self {
        Self { engine, default_entity: default_entity.into() }
    }

    fn entity(&self, args: &Value) -> String {
        args["entity_id"]
            .as_str()
            .filter(|e| !e.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| self.default_entity.clone())
    }

    async fn call_tool(&self, name: &str, args: &Value) -> Result<Value, String> {
        match name {
            "remember" => {
                let text = args["text"].as_str().filter(|t| !t.trim().is_empty());
                let Some(text) = text else {
                    return Err("`text` is required".into());
                };
                let memory = Memory {
                    entity_id: self.entity(args),
                    text: text.to_string(),
                    memory_id: None,
                    session_id: args["session_id"].as_str().map(str::to_string),
                    turn_index: args["turn_index"].as_u64().map(|t| t as u32),
                    role: args["role"].as_str().map(str::to_string),
                    timestamp_ms: args["timestamp_ms"].as_u64(),
                    kind: args["kind"].as_str().map(str::to_string),
                };
                let report = self.engine.ingest(vec![memory]).await.map_err(|e| e.to_string())?;
                Ok(json!({ "stored": report.memories, "records": report.expanded }))
            }
            "recall" => {
                let query = args["query"].as_str().filter(|q| !q.trim().is_empty());
                let Some(query) = query else {
                    return Err("`query` is required".into());
                };
                let mut q = Query::new(query)
                    .entity(self.entity(args))
                    .limit(args["limit"].as_u64().unwrap_or(10).clamp(1, 100) as usize);
                q.as_of_ms = args["as_of_ms"].as_u64();
                let hits = self.engine.query(q).await.map_err(|e| e.to_string())?;
                Ok(json!({ "results": hits }))
            }
            "current_fact" => {
                let Some(key) = args["fact_key"].as_str().filter(|k| !k.is_empty()) else {
                    return Err("`fact_key` is required".into());
                };
                let entity = self.entity(args);
                let value = self.engine.current_fact(&entity, key).map_err(|e| e.to_string())?;
                Ok(json!({ "entity_id": entity, "fact_key": key, "value": value }))
            }
            "get_memory" => {
                let Some(memory_id) = args["memory_id"].as_str().filter(|m| !m.is_empty()) else {
                    return Err("`memory_id` is required".into());
                };
                match self.engine.get_memory(memory_id).map_err(|e| e.to_string())? {
                    Some(hit) => Ok(json!(hit)),
                    None => Err(format!("no memory with id `{memory_id}`")),
                }
            }
            "explore_graph" => {
                let node = args["node"].as_str().or_else(|| args["entity"].as_str());
                let Some(node) = node.filter(|n| !n.trim().is_empty()) else {
                    return Err("`node` is required".into());
                };
                let limit = args["limit"].as_u64().unwrap_or(20) as usize;
                let edges = self.engine.explore_graph(node, limit).map_err(|e| e.to_string())?;
                Ok(json!({ "node": node, "edges": edges }))
            }
            "fact_history" => {
                let Some(key) = args["fact_key"].as_str().filter(|k| !k.is_empty()) else {
                    return Err("`fact_key` is required".into());
                };
                let entity = self.entity(args);
                match args["as_of_ms"].as_u64() {
                    Some(as_of) => {
                        let version = self
                            .engine
                            .fact_as_of(&entity, key, as_of)
                            .map_err(|e| e.to_string())?;
                        Ok(json!({
                            "entity_id": entity,
                            "fact_key": key,
                            "as_of_ms": as_of,
                            "value": version.as_ref().map(|v| v.object.clone()),
                            "version": version,
                        }))
                    }
                    None => {
                        let history =
                            self.engine.fact_history(&entity, key).map_err(|e| e.to_string())?;
                        Ok(json!({ "entity_id": entity, "fact_key": key, "history": history }))
                    }
                }
            }
            other => Err(format!("unknown tool `{other}`")),
        }
    }

    /// Handles one JSON-RPC message; `None` for notifications.
    pub async fn handle(&self, message: Value) -> Option<Value> {
        let id = message.get("id").cloned();
        let method = message["method"].as_str().unwrap_or_default();
        let params = &message["params"];
        let Some(id) = id else {
            // Notifications (e.g. notifications/initialized) get no reply.
            return None;
        };
        Some(match method {
            "initialize" => {
                let version =
                    negotiate_protocol_version(params["protocolVersion"].as_str().unwrap_or(""));
                rpc_result(
                    id,
                    json!({
                        "protocolVersion": version,
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": "tellodb", "version": env!("CARGO_PKG_VERSION") }
                    }),
                )
            }
            "ping" => rpc_result(id, json!({})),
            "tools/list" => rpc_result(id, json!({ "tools": tool_definitions() })),
            "tools/call" => {
                let name = params["name"].as_str().unwrap_or_default();
                let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
                match self.call_tool(name, &args).await {
                    Ok(value) => rpc_result(id, text_result(&value, false)),
                    Err(message) => rpc_result(id, text_result(&json!({ "error": message }), true)),
                }
            }
            other => rpc_error(id, -32601, format!("method not found: {other}")),
        })
    }

    /// Serves until stdin closes.
    pub async fn run(&self) -> Result<()> {
        let mut lines = BufReader::new(tokio::io::stdin()).lines();
        let mut stdout = tokio::io::stdout();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            let reply = match serde_json::from_str::<Value>(&line) {
                Ok(message) => self.handle(message).await,
                Err(err) => Some(rpc_error(Value::Null, -32700, format!("parse error: {err}"))),
            };
            if let Some(reply) = reply {
                stdout.write_all(serde_json::to_string(&reply)?.as_bytes()).await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
        }
        self.engine.checkpoint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_results_use_text_content() {
        let result = text_result(&json!({ "a": 1 }), false);
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["isError"], false);
        assert_eq!(text_result(&json!({}), true)["isError"], true);
    }

    #[test]
    fn every_tool_declares_an_object_schema() {
        let tools = tool_definitions();
        let tools = tools.as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            ["remember", "recall", "get_memory", "explore_graph", "fact_history", "current_fact"]
        );
        for tool in tools {
            assert_eq!(tool["inputSchema"]["type"], "object", "{}", tool["name"]);
            assert!(tool["inputSchema"]["required"].is_array(), "{}", tool["name"]);
            assert!(tool["description"].as_str().is_some_and(|d| d.len() > 20));
        }
    }

    #[test]
    fn protocol_version_is_echoed_when_supported() {
        assert_eq!(negotiate_protocol_version("2024-11-05"), "2024-11-05");
        assert_eq!(negotiate_protocol_version("1999-01-01"), PROTOCOL_VERSIONS[0]);
        assert_eq!(negotiate_protocol_version(""), PROTOCOL_VERSIONS[0]);
    }

    #[test]
    fn rpc_shapes_follow_json_rpc() {
        let ok = rpc_result(json!(7), json!({ "x": 1 }));
        assert_eq!((&ok["jsonrpc"], &ok["id"]), (&json!("2.0"), &json!(7)));
        assert!(ok.get("error").is_none());
        let err = rpc_error(json!("a"), -32601, "method not found: bogus");
        assert_eq!(err["error"]["code"], -32601);
        assert!(err["error"]["message"].as_str().unwrap().contains("bogus"));
        assert!(err.get("result").is_none());
    }
}
