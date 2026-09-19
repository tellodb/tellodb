//! Embedded API: open a data directory and ingest or query in-process, with
//! no HTTP server.
//!
//! ```no_run
//! use tellodb::db::{Db, Memory, Query};
//!
//! let db = Db::open("./agent-memory")?;
//! db.ingest(vec![Memory::new("alice", "I just moved to Denver.").session("chat-1", 0)])?;
//! let hits = db.query(Query::new("where do I live?").entity("alice").limit(5))?;
//! # anyhow::Ok(())
//! ```
//!
//! [`Engine`] is the async form for callers that already run Tokio; [`Db`]
//! wraps it with its own runtime and blocking methods.

use crate::api::handlers::ingest::{process_ingest_batch, spawn_consolidation_tasks};
use crate::api::handlers::query::execute_query_pipeline;
use crate::api::types::{IngestPayload, QueryPayload};
use crate::api::EngineState;
use crate::runtime_paths::RuntimePaths;
use crate::storage::TenantStore;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A memory to store. Only `entity_id` and `text` are required.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Memory {
    pub entity_id: String,
    pub text: String,
    /// Stable id; generated when absent. Re-sending an id with new text
    /// replaces the memory.
    #[serde(default)]
    pub memory_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub turn_index: Option<u32>,
    /// Speaker, e.g. `user` or `assistant`.
    #[serde(default)]
    pub role: Option<String>,
    /// When it happened (event time); defaults to now.
    #[serde(default)]
    pub timestamp_ms: Option<u64>,
    /// `fact`, `preference`, `decision`, `lesson` or `conversational`.
    #[serde(default)]
    pub kind: Option<String>,
}

impl Memory {
    pub fn new(entity_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self { entity_id: entity_id.into(), text: text.into(), ..Default::default() }
    }

    pub fn session(mut self, session_id: impl Into<String>, turn_index: u32) -> Self {
        self.session_id = Some(session_id.into());
        self.turn_index = Some(turn_index);
        self
    }

    pub fn role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    pub fn at(mut self, timestamp_ms: u64) -> Self {
        self.timestamp_ms = Some(timestamp_ms);
        self
    }

    fn into_payload(self) -> IngestPayload {
        let timestamp = self.timestamp_ms.unwrap_or_else(now_ms);
        let session_id = self.session_id.filter(|s| !s.is_empty());
        let memory_id = self.memory_id.unwrap_or_else(|| {
            let session = session_id
                .clone()
                .unwrap_or_else(|| format!("mem-{timestamp}-{:032x}", rand::random::<u128>()));
            format!("{}::{}::{}", self.entity_id, session, self.turn_index.unwrap_or(0))
        });
        IngestPayload {
            entity_id: self.entity_id,
            memory_id,
            timestamp,
            textual_content: self.text,
            kind: self.kind,
            session_id,
            turn_index: self.turn_index,
            role: self.role,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub text: String,
    #[serde(default)]
    pub entity_id: Option<String>,
    #[serde(default = "Query::default_limit")]
    pub limit: usize,
    /// Only memories that existed at this time are considered.
    #[serde(default)]
    pub as_of_ms: Option<u64>,
    /// "Now" for relative dates in the query ("last week"); defaults to now.
    #[serde(default)]
    pub reference_time_ms: Option<u64>,
    /// Force the cross-encoder reranker for this query.
    #[serde(default)]
    pub rerank: bool,
}

impl Query {
    fn default_limit() -> usize {
        10
    }

    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            entity_id: None,
            limit: Self::default_limit(),
            as_of_ms: None,
            reference_time_ms: None,
            rerank: false,
        }
    }

    pub fn entity(mut self, entity_id: impl Into<String>) -> Self {
        self.entity_id = Some(entity_id.into());
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn as_of(mut self, as_of_ms: u64) -> Self {
        self.as_of_ms = Some(as_of_ms);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    pub memory_id: String,
    pub entity_id: String,
    pub session_id: String,
    pub text: String,
    pub score: f32,
    pub created_at_ms: u64,
    pub fact_key: Option<String>,
    /// The memory that replaced this fact, when it is no longer current.
    pub superseded_by: Option<String>,
    /// What replaced this fact and when, for stale results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why_stale: Option<crate::api::types::WhyStale>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestReport {
    pub memories: usize,
    /// Records built, including derived ones.
    pub expanded: usize,
    pub embedded: usize,
    pub total_ms: u64,
}

/// Async embedded engine for one tenant.
#[derive(Clone)]
pub struct Engine {
    state: EngineState,
    tenant: Arc<TenantStore>,
}

impl Engine {
    /// Opens (or creates) a data directory and loads the models.
    pub async fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_tenant(dir, "default").await
    }

    pub async fn open_tenant(dir: impl AsRef<Path>, tenant_id: &str) -> Result<Self> {
        Self::from_paths(&RuntimePaths::from_root(dir.as_ref().to_path_buf()), tenant_id).await
    }

    /// Wraps an engine state and tenant store that are already open (used by
    /// the HTTP MCP route, which authorizes the caller first).
    pub fn from_parts(state: EngineState, tenant: Arc<TenantStore>) -> Self {
        Self { state, tenant }
    }

    pub async fn from_paths(paths: &RuntimePaths, tenant_id: &str) -> Result<Self> {
        let config = crate::config::Config::from_env()?;
        let state =
            crate::engine::build_state(paths, crate::api::AuthConfig::embedded(), config).await?;
        let tenant = state.tenant_store(tenant_id)?;
        Ok(Self { state, tenant })
    }

    pub fn state(&self) -> &EngineState {
        &self.state
    }

    pub fn tenant(&self) -> &Arc<TenantStore> {
        &self.tenant
    }

    pub async fn ingest(&self, memories: Vec<Memory>) -> Result<IngestReport> {
        let count = memories.len();
        let payloads: Vec<IngestPayload> = memories.into_iter().map(Memory::into_payload).collect();
        let (tasks, diag) = process_ingest_batch(&self.state, &self.tenant, payloads)
            .await
            .map_err(|status| anyhow::anyhow!("ingest failed ({status})"))?;
        spawn_consolidation_tasks(self.tenant.clone(), tasks);
        Ok(IngestReport {
            memories: count,
            expanded: diag.expanded_count(),
            embedded: diag.embedded_count(),
            total_ms: diag.total_ms(),
        })
    }

    pub async fn query(&self, query: Query) -> Result<Vec<Hit>> {
        let limit = query.limit.clamp(1, 1_000);
        let payload = QueryPayload {
            textual_query: query.text,
            limit,
            entity_id: query.entity_id,
            enable_neural_rerank: Some(query.rerank),
            include_evidence: None,
            verify_evidence: None,
            proof_mode: None,
            max_evidence_turns_per_session: None,
            point_in_time_ms: query.as_of_ms,
            reference_time_ms: query.reference_time_ms,
        };
        let (state, tenant, rerank) = (self.state.clone(), self.tenant.clone(), query.rerank);
        let (results, _diag) = tokio::task::spawn_blocking(move || {
            execute_query_pipeline(payload, state, tenant, limit, rerank)
        })
        .await
        .context("query task panicked")?
        .map_err(|status| anyhow::anyhow!("query failed ({status})"))?;
        Ok(results
            .into_iter()
            .filter(|r| !r.memory_id.starts_with("__pre_synth_"))
            .map(|r| Hit {
                memory_id: r.memory_id,
                entity_id: r.entity_id,
                session_id: r.session_id,
                text: r.textual_content,
                score: r.similarity,
                created_at_ms: r.created_at_ms,
                fact_key: r.fact_key,
                superseded_by: r.superseded_by,
                why_stale: r.why_stale,
            })
            .collect())
    }

    /// One memory by id, if it exists.
    pub fn get_memory(&self, memory_id: &str) -> Result<Option<Hit>> {
        let Some((timestamp, _)) = self.tenant.lookup_by_memory_id(memory_id)? else {
            return Ok(None);
        };
        let mut observations =
            self.tenant.get_observations_batch(&[(timestamp, memory_id.to_string())])?;
        Ok(observations.remove(memory_id).map(|obs| Hit {
            memory_id: memory_id.to_string(),
            entity_id: obs.entity_id,
            session_id: obs.session_id,
            text: obs.textual_content,
            score: 1.0,
            created_at_ms: if obs.created_at_ms > 0 { obs.created_at_ms } else { timestamp },
            fact_key: None,
            superseded_by: None,
            why_stale: None,
        }))
    }

    /// Graph edges touching `node` (an entity or a subject), as
    /// `subject --[predicate]--> object` lines.
    pub fn explore_graph(&self, node: &str, limit: usize) -> Result<Vec<String>> {
        Ok(self
            .tenant
            .graph_query_edges(node, None, "Both", limit.clamp(1, 500))?
            .iter()
            .map(|e| format!("{} --[{}]--> {}", e.source, e.label, e.target))
            .collect())
    }

    /// The current value of a fact such as `residence`, if one is known.
    pub fn current_fact(&self, entity_id: &str, fact_key: &str) -> Result<Option<String>> {
        self.tenant.get_current_fact_value(entity_id, fact_key)
    }

    /// Every version of a fact, oldest first.
    pub fn fact_history(
        &self,
        entity_id: &str,
        fact_key: &str,
    ) -> Result<Vec<crate::storage::FactHistoryEntry>> {
        self.tenant.fact_history(entity_id, fact_key)
    }

    /// The value a fact held at `as_of_ms`.
    pub fn fact_as_of(
        &self,
        entity_id: &str,
        fact_key: &str,
        as_of_ms: u64,
    ) -> Result<Option<crate::storage::FactHistoryEntry>> {
        Ok(self.fact_history(entity_id, fact_key)?.into_iter().rev().find(|v| {
            v.valid_from_ms <= as_of_ms && v.valid_to_ms.map_or(true, |to| as_of_ms < to)
        }))
    }

    /// Flushes the write-ahead log into the database file.
    pub fn checkpoint(&self) -> Result<()> {
        self.tenant.checkpoint()
    }
}

/// Blocking embedded engine with its own Tokio runtime. Do not use from
/// inside an async runtime; use [`Engine`] there.
pub struct Db {
    engine: Engine,
    runtime: tokio::runtime::Runtime,
}

impl Db {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("tellodb")
            .build()
            .context("failed to start runtime")?;
        let engine = runtime.block_on(Engine::open(dir))?;
        Ok(Self { engine, runtime })
    }

    pub fn ingest(&self, memories: Vec<Memory>) -> Result<IngestReport> {
        self.runtime.block_on(self.engine.ingest(memories))
    }

    pub fn query(&self, query: Query) -> Result<Vec<Hit>> {
        self.runtime.block_on(self.engine.query(query))
    }

    pub fn current_fact(&self, entity_id: &str, fact_key: &str) -> Result<Option<String>> {
        self.engine.current_fact(entity_id, fact_key)
    }

    pub fn fact_history(
        &self,
        entity_id: &str,
        fact_key: &str,
    ) -> Result<Vec<crate::storage::FactHistoryEntry>> {
        self.engine.fact_history(entity_id, fact_key)
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }
}

impl Drop for Db {
    fn drop(&mut self) {
        if let Err(err) = self.engine.checkpoint() {
            tracing::warn!(error = ?err, "checkpoint on close failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_ids_follow_entity_session_turn() {
        let payload = Memory::new("alice", "hi").session("chat-1", 3).at(5).into_payload();
        assert_eq!(payload.memory_id, "alice::chat-1::3");
        assert_eq!((payload.timestamp, payload.turn_index), (5, Some(3)));

        let explicit = Memory { memory_id: Some("x".into()), ..Memory::new("alice", "hi") };
        assert_eq!(explicit.into_payload().memory_id, "x");

        let generated = Memory::new("alice", "hi").at(7).into_payload();
        assert!(generated.memory_id.starts_with("alice::mem-7-"));
        assert_eq!(generated.session_id, None);
    }

    #[test]
    fn generated_memory_ids_are_unique_at_same_timestamp() {
        let ids: std::collections::HashSet<_> = (0..10_000)
            .map(|_| Memory::new("alice", "hi").at(7).into_payload().memory_id)
            .collect();
        assert_eq!(ids.len(), 10_000);
    }
}
