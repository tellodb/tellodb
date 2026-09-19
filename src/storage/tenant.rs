use anyhow::{Context, Result};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use super::types::*;

type GraphEdgeBatch<'a> = [GraphEdgeEntry<'a>];

const PRAGMA_CACHE_SIZE: i64 = -262144;
const PRAGMA_MMAP_SIZE: i64 = 1073741824;
const PRAGMA_BUSY_TIMEOUT: i64 = 10000;
const PRAGMA_PAGE_SIZE: i64 = 8192;
const STATEMENT_CACHE_CAPACITY: usize = 512;

const METRICS_DDL: &str = "CREATE TABLE IF NOT EXISTS metrics (
    memory_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    entity_id TEXT NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    label TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    source_text TEXT,
    PRIMARY KEY(memory_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_metrics_entity_label ON metrics(entity_id, label, timestamp_ms);";

const SOURCE_DEPTH_DIVISOR: f32 = 8.0;

const SESSION_LEXICAL_WEIGHT: f32 = 0.48;
const SESSION_TEMPORAL_WEIGHT: f32 = 0.18;
const SESSION_ENTITY_WEIGHT: f32 = 0.24;
const SESSION_DEPTH_WEIGHT: f32 = 0.04;
const SESSION_FOCUS_BONUS: f32 = 0.06;

const CARD_LEXICAL_WEIGHT: f32 = 0.42;
const CARD_TEMPORAL_WEIGHT: f32 = 0.16;
const CARD_ENTITY_WEIGHT: f32 = 0.24;
const CARD_ROUTE_BOOST: f32 = 0.12;
const CARD_LATEST_BOOST: f32 = 0.04;
const CARD_STALE_PENALTY: f32 = -0.06;
const CARD_CONFIDENCE_WEIGHT: f32 = 0.08;

const FACT_TYPE_BOOST: f32 = 0.10;
const PREFERENCE_TYPE_BOOST: f32 = 0.09;
const EVENT_TYPE_BOOST: f32 = 0.07;
const DECISION_TYPE_BOOST: f32 = 0.06;
const INFERENCE_TYPE_BOOST: f32 = 0.04;
const OTHER_TYPE_BOOST: f32 = 0.03;

const FOCUS_MATCH_MIN_LEN: usize = 4;
const FTS_MIN_TERM_LEN: usize = 1;
const SEARCH_MIN_TERM_LEN: usize = 2;

fn unix_timestamp_ms() -> Result<i64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")
        .map(|d| d.as_millis() as i64)
}

pub struct TenantStore {
    pool: Pool<SqliteConnectionManager>,
    /// This tenant's vector index. Each tenant owns its own index because
    /// vector ids are per-tenant `memories.rowid`s and would collide in a
    /// shared index. Attached by `TenantDatabaseManager` after open.
    vectors: std::sync::OnceLock<crate::vector_index::VectorIndex>,
}

/// Stable FTS5 rowid for a document key. FTS tables have no unique key on
/// `memory_id`, so `INSERT OR REPLACE` only replaces when the rowid matches;
/// deriving it from the key makes re-ingest replace instead of duplicate and
/// makes deletes an O(log n) rowid lookup.
/// The single FTS token standing for an entity.
///
/// Hex-encoded so the result is one `unicode61` token whatever the entity id
/// contains, and prefixed so it cannot be mistaken for a content word. It
/// lives in its own indexed column, so `entity_tok:<tok>` restricts the scan
/// to one entity inside the index instead of filtering after the match.
pub(crate) fn fts_entity_tok(entity_id: &str) -> String {
    let mut out = String::with_capacity(1 + entity_id.len() * 2);
    out.push('e');
    for byte in entity_id.as_bytes() {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Wraps a term as an FTS5 string, doubling any embedded quote.
///
/// An unescaped `"` closes the phrase early and makes the whole MATCH
/// expression invalid, which the callers turn into an empty lane.
fn fts_quote(term: impl AsRef<str>) -> String {
    format!("\"{}\"", term.as_ref().replace('"', "\"\""))
}

pub(crate) fn fts_rowid(key: &str) -> i64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in key.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    // Positive and non-zero.
    ((hash >> 1) | 1) as i64
}

/// Schema version recorded in `PRAGMA user_version`.
const SCHEMA_VERSION: i64 = 3;

impl TenantStore {
    pub fn new(path: &Path) -> Result<Self> {
        let manager = SqliteConnectionManager::file(path).with_init(|conn| {
            // The store uses ~70 distinct cached statements plus IN-list
            // queries; with rusqlite's default capacity of 16 they were
            // evicted and re-parsed constantly.
            conn.set_prepared_statement_cache_capacity(STATEMENT_CACHE_CAPACITY);
            conn.execute_batch(&format!(
                "PRAGMA journal_mode = WAL;
                     PRAGMA synchronous = NORMAL;
                     PRAGMA foreign_keys = ON;
                     PRAGMA temp_store = MEMORY;
                     PRAGMA cache_size = {};
                     PRAGMA mmap_size = {};
                     PRAGMA busy_timeout = {};
                     PRAGMA page_size = {};",
                PRAGMA_CACHE_SIZE, PRAGMA_MMAP_SIZE, PRAGMA_BUSY_TIMEOUT, PRAGMA_PAGE_SIZE,
            ))
        });
        let max_size = (num_cpus::get().saturating_mul(2)).max(16) as u32;
        let pool = Pool::builder()
            .max_size(max_size)
            .build(manager)
            .context("failed to build connection pool")?;

        let conn = pool.get().context("failed to get initial connection from pool")?;
        Self::init_schema(&conn)?;
        info!(path = %path.display(), "Tenant database initialized");
        Ok(Self { pool, vectors: std::sync::OnceLock::new() })
    }

    pub fn attach_vectors(&self, index: crate::vector_index::VectorIndex) -> Result<()> {
        self.vectors.set(index).map_err(|_| anyhow::anyhow!("vector index already attached"))
    }

    /// This tenant's vector index.
    pub fn vectors(&self) -> Result<&crate::vector_index::VectorIndex> {
        self.vectors.get().context("tenant has no vector index attached")
    }

    /// A vector source reading this tenant's stored embeddings.
    pub fn vector_source(&self) -> std::sync::Arc<dyn crate::vector_index::VectorSource> {
        std::sync::Arc::new(SqliteVectorSource { pool: self.pool.clone() })
    }

    /// `(rows with a stored embedding, rows without one)` in `vector_lookup`.
    pub fn stored_vector_counts(&self) -> Result<(usize, usize)> {
        let conn = self.get_conn()?;
        Ok(conn.query_row(
            "SELECT COUNT(embedding), COUNT(*) - COUNT(embedding) FROM vector_lookup",
            [],
            |row| Ok((row.get::<_, i64>(0)? as usize, row.get::<_, i64>(1)? as usize)),
        )?)
    }

    fn init_schema(conn: &rusqlite::Connection) -> Result<()> {
        conn.execute_batch(
            "
            -- Core memories
            CREATE TABLE IF NOT EXISTS memories (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL UNIQUE,
                entity_id TEXT NOT NULL,
                content TEXT NOT NULL,
                kind TEXT NOT NULL,
                content_hash TEXT NOT NULL DEFAULT '',
                created_at_ms INTEGER NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                turn_index INTEGER NOT NULL DEFAULT 0,
                role TEXT NOT NULL DEFAULT '',
                parent_memory_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity_id);
            CREATE INDEX IF NOT EXISTS idx_memories_memory_id ON memories(memory_id);

            -- Memory cards
            CREATE TABLE IF NOT EXISTS memory_cards (
                card_id TEXT PRIMARY KEY,
                entity_id TEXT,
                user_id TEXT,
                source_memory_id TEXT,
                source_session_id TEXT,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                memory_text TEXT,
                card_type TEXT,
                confidence REAL,
                is_latest INTEGER NOT NULL DEFAULT 1,
                is_static INTEGER NOT NULL DEFAULT 0,
                is_inference INTEGER NOT NULL DEFAULT 0,
                expires_at INTEGER,
                root_card_id TEXT,
                parent_card_id TEXT,
                lifecycle TEXT,
                created_at_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_memory_cards_entity ON memory_cards(entity_id);
            CREATE INDEX IF NOT EXISTS idx_memory_cards_session ON memory_cards(source_session_id);
            CREATE INDEX IF NOT EXISTS idx_memory_cards_source ON memory_cards(source_memory_id);

            -- Graph edges
            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'default',
                label TEXT,
                weight REAL DEFAULT 1.0,
                status TEXT DEFAULT 'current',
                ref_source TEXT,
                ref_target TEXT,
                timestamp_ms INTEGER,
                memory_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
            CREATE INDEX IF NOT EXISTS idx_edges_memory ON edges(memory_id);
            CREATE INDEX IF NOT EXISTS idx_edges_source_type ON edges(source, edge_type);
            CREATE INDEX IF NOT EXISTS idx_edges_target_type ON edges(target, edge_type);
            CREATE INDEX IF NOT EXISTS idx_edges_label ON edges(label);

            -- Metrics
            CREATE TABLE IF NOT EXISTS metrics (
                memory_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                entity_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                label TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                source_text TEXT,
                PRIMARY KEY(memory_id, ordinal)
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_entity_label ON metrics(entity_id, label, timestamp_ms);

            -- FTS5
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
                memory_id UNINDEXED,
                entity_id UNINDEXED,
                entity_tok,
                content,
                tokenize='porter unicode61'
            );

            -- Vector lookup
            CREATE TABLE IF NOT EXISTS vector_lookup (
                vector_id INTEGER PRIMARY KEY,
                memory_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_vector_lookup_memory ON vector_lookup(memory_id);
            CREATE INDEX IF NOT EXISTS idx_vector_lookup_entity ON vector_lookup(entity_id);

            -- Session router
            CREATE TABLE IF NOT EXISTS session_router (
                session_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                router_text TEXT NOT NULL DEFAULT '',
                created_at_ms INTEGER,
                updated_at_ms INTEGER,
                PRIMARY KEY(session_id, entity_id)
            );
            CREATE INDEX IF NOT EXISTS idx_router_entity ON session_router(entity_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_session_router USING fts5(
                session_id UNINDEXED,
                entity_id UNINDEXED,
                router_text,
                tokenize='porter unicode61'
            );

            -- Aliases
            CREATE TABLE IF NOT EXISTS aliases (
                entity_id TEXT NOT NULL,
                alias TEXT NOT NULL,
                PRIMARY KEY(entity_id, alias)
            );
            CREATE INDEX IF NOT EXISTS idx_aliases_alias ON aliases(alias);

            -- Preferences
            CREATE TABLE IF NOT EXISTS preferences (
                entity_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                strength REAL NOT NULL DEFAULT 0.5,
                PRIMARY KEY(entity_id, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_prefs_entity ON preferences(entity_id);

            -- Memory links
            CREATE TABLE IF NOT EXISTS memory_links (
                source_memory_id TEXT NOT NULL,
                target_memory_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                PRIMARY KEY(source_memory_id, target_memory_id, link_type)
            );
            CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_memory_id);
            CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_memory_id);

            -- Fact versions
            CREATE TABLE IF NOT EXISTS fact_versions (
                fact_key TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                status TEXT NOT NULL DEFAULT 'current',
                timestamp_ms INTEGER NOT NULL,
                superseded_by TEXT,
                supersedes TEXT,
                valid_from_ms INTEGER,
                valid_to_ms INTEGER,
                PRIMARY KEY(fact_key, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_fact_entity ON fact_versions(entity_id);
            CREATE INDEX IF NOT EXISTS idx_fact_versions_lookup ON fact_versions(fact_key, entity_id, status);

            -- Memories that support a fact version (restatements merge into
            -- the version they confirm instead of creating a new one).
            CREATE TABLE IF NOT EXISTS fact_evidence (
                fact_key TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                version_memory_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                PRIMARY KEY(fact_key, entity_id, version_memory_id, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_fact_evidence_version
                ON fact_evidence(entity_id, fact_key, version_memory_id);

            -- Predicates grouped by meaning, so variants of one predicate
            -- supersede each other (see canonicalize_predicates).
            CREATE TABLE IF NOT EXISTS predicate_canon (
                entity_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                canonical TEXT NOT NULL,
                embedding BLOB,
                PRIMARY KEY(entity_id, predicate)
            );
            CREATE INDEX IF NOT EXISTS idx_predicate_canon_canonical
                ON predicate_canon(entity_id, canonical);

            -- Core profiles
            CREATE TABLE IF NOT EXISTS core_profiles (
                entity_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at_ms INTEGER
            );

            -- Entity embeddings
            CREATE TABLE IF NOT EXISTS entity_embeddings (
                entity_id TEXT NOT NULL,
                embedding_blob BLOB NOT NULL,
                updated_at_ms INTEGER,
                PRIMARY KEY(entity_id)
            );

            -- Deletion tombstones
            CREATE TABLE IF NOT EXISTS deletion_tombstones (
                tombstone_id TEXT PRIMARY KEY,
                target_memory_id TEXT NOT NULL,
                reason TEXT,
                timestamp_ms INTEGER,
                tombstone_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_tombstone_target ON deletion_tombstones(target_memory_id);

            -- Centroids and Disambiguations
            CREATE TABLE IF NOT EXISTS negative_centroids (
                entity_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                centroid_blob BLOB NOT NULL,
                PRIMARY KEY(entity_id, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_neg_centroids_entity ON negative_centroids(entity_id);

            CREATE TABLE IF NOT EXISTS disambiguation_vectors (
                entity_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                vector_blob BLOB NOT NULL,
                PRIMARY KEY(entity_id, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_disambiguation_entity ON disambiguation_vectors(entity_id);

            -- Entity registry for tiered resolution
            CREATE TABLE IF NOT EXISTS entity_registry (
                entity_id TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                soundex_key TEXT NOT NULL DEFAULT '',
                updated_at_ms INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(entity_id, canonical_name)
            );
            CREATE INDEX IF NOT EXISTS idx_registry_soundex ON entity_registry(entity_id, soundex_key);

            -- Name embeddings cache (for tier-4 of the resolver)
            CREATE TABLE IF NOT EXISTS name_embeddings (
                entity_id TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                embedding_blob BLOB NOT NULL,
                dim INTEGER NOT NULL DEFAULT 0,
                updated_at_ms INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(entity_id, canonical_name)
            );

            -- Merge proposals (same_as staging)
            CREATE TABLE IF NOT EXISTS merge_proposals (
                proposal_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                from_name TEXT NOT NULL,
                to_name TEXT NOT NULL,
                tier TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at_ms INTEGER NOT NULL DEFAULT 0,
                resolved_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_proposals_entity ON merge_proposals(entity_id);
            CREATE INDEX IF NOT EXISTS idx_proposals_status ON merge_proposals(status);
        ",
        )?;

        Self::migrate(conn)?;

        Ok(())
    }

    fn has_column(conn: &rusqlite::Connection, table: &str, column: &str) -> Result<bool> {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info(?1) WHERE name = ?2",
            params![table, column],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Upgrades databases created by older builds. Runs on every open and is
    /// idempotent; failures abort the open instead of leaving a half-migrated
    /// schema behind.
    fn migrate(conn: &rusqlite::Connection) -> Result<()> {
        if !Self::has_column(conn, "memories", "content_hash")? {
            conn.execute_batch(
                "ALTER TABLE memories ADD COLUMN content_hash TEXT NOT NULL DEFAULT '';",
            )?;
        }
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);",
        )?;
        for (column, ddl) in [
            ("session_id", "ALTER TABLE memories ADD COLUMN session_id TEXT NOT NULL DEFAULT ''"),
            ("turn_index", "ALTER TABLE memories ADD COLUMN turn_index INTEGER NOT NULL DEFAULT 0"),
            ("role", "ALTER TABLE memories ADD COLUMN role TEXT NOT NULL DEFAULT ''"),
            ("parent_memory_id", "ALTER TABLE memories ADD COLUMN parent_memory_id TEXT"),
        ] {
            if !Self::has_column(conn, "memories", column)? {
                conn.execute_batch(ddl)?;
            }
        }
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_memories_session_turn
                 ON memories(entity_id, session_id, turn_index);
             -- Derived records are looked up by their parent turn (evidence
             -- packets, fact versions). Without this the `OR parent_memory_id
             -- IN (...)` lookups degrade to a full scan of `memories`, which
             -- cost ~1.9 s per query on the LongMemEval dev split.
             CREATE INDEX IF NOT EXISTS idx_memories_parent
                 ON memories(parent_memory_id);",
        )?;
        if !Self::has_column(conn, "vector_lookup", "embedding")? {
            conn.execute_batch("ALTER TABLE vector_lookup ADD COLUMN embedding BLOB;")?;
        }
        if !Self::has_column(conn, "metrics", "memory_id")? {
            // The old table was keyed by (timestamp, entity, label), so two
            // amounts in one memory overwrote each other. Ingest never wrote
            // to it, so there is nothing to carry over.
            conn.execute_batch(&format!("DROP TABLE metrics; {}", METRICS_DDL))?;
        }
        if !Self::has_column(conn, "fact_versions", "recorded_at_ms")? {
            conn.execute_batch("ALTER TABLE fact_versions ADD COLUMN recorded_at_ms INTEGER;")?;
        }
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_fact_versions_memory ON fact_versions(memory_id);",
        )?;

        // Structures that were written on ingest but never read by retrieval.
        conn.execute_batch(
            "DROP TABLE IF EXISTS temporal_events;
             DROP TABLE IF EXISTS fts_temporal_events;
             DROP TABLE IF EXISTS shadow_questions;
             DROP TABLE IF EXISTS fts_shadow_questions;
             DROP TABLE IF EXISTS facet_postings;
             DROP TABLE IF EXISTS mem_cells;
             DROP TABLE IF EXISTS mem_scenes;
             DROP TABLE IF EXISTS profile_facts;
             DROP TABLE IF EXISTS card_relations;
             DROP TABLE IF EXISTS ledger_turns;
             DROP TABLE IF EXISTS memory_artifacts;
             DROP TABLE IF EXISTS artifact_versions;",
        )?;

        let version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if version < 2 {
            // FTS rows used to get random rowids, so re-ingest duplicated them.
            // Re-key every row by `fts_rowid`, keeping the newest copy. Each
            // table is rewritten in one transaction.
            for (table, key_col, cols) in
                [("fts_memories", "memory_id", "memory_id, entity_id, content")]
            {
                let tmp = format!("{table}_migrate");
                conn.execute_batch(&format!(
                    "BEGIN IMMEDIATE;
                     DROP TABLE IF EXISTS {tmp};
                     CREATE TEMP TABLE {tmp} AS
                        SELECT {cols} FROM {table} WHERE rowid IN
                            (SELECT MAX(rowid) FROM {table} GROUP BY {key_col});
                     DELETE FROM {table};"
                ))?;
                let rows: Vec<Vec<rusqlite::types::Value>> = {
                    let mut stmt = conn.prepare(&format!("SELECT {cols} FROM {tmp}"))?;
                    let width = stmt.column_count();
                    let mapped = stmt.query_map([], |row| {
                        (0..width).map(|i| row.get::<_, rusqlite::types::Value>(i)).collect()
                    })?;
                    mapped.collect::<rusqlite::Result<_>>()?
                };
                let placeholders = vec!["?"; cols.split(',').count() + 1].join(", ");
                {
                    let mut insert = conn.prepare(&format!(
                        "INSERT OR REPLACE INTO {table} (rowid, {cols}) VALUES ({placeholders})"
                    ))?;
                    for row in rows {
                        let key = match &row[0] {
                            rusqlite::types::Value::Text(key) => key.clone(),
                            _ => continue,
                        };
                        let mut values = vec![rusqlite::types::Value::Integer(fts_rowid(&key))];
                        values.extend(row);
                        insert.execute(rusqlite::params_from_iter(values))?;
                    }
                }
                conn.execute_batch(&format!("COMMIT; DROP TABLE {tmp};"))?;
            }
        }
        if version < 3 {
            // `entity_tok` (indexed) replaced the post-MATCH filter on the
            // UNINDEXED `entity_id`. Nothing rebuilds the contents: there is
            // no data worth preserving yet, and re-ingesting restores it.
            tracing::warn!("dropping FTS index for the entity-token schema; re-ingest to restore");
            conn.execute_batch("DROP TABLE IF EXISTS fts_memories;")?;
            conn.execute_batch(
                "CREATE VIRTUAL TABLE fts_memories USING fts5(
                     memory_id UNINDEXED,
                     entity_id UNINDEXED,
                     entity_tok,
                     content,
                     tokenize='porter unicode61'
                 );",
            )?;
        }
        if version < SCHEMA_VERSION {
            conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION};"))?;
        }
        Ok(())
    }

    pub fn get_conn(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.pool.get().context("Failed to get connection from pool")
    }

    pub fn checkpoint(&self) -> Result<()> {
        let conn = self.get_conn()?;
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }

    // ── Vector ID management ──

    fn allocate_vector_ids(
        &self,
        items: &[(u64, String, &AgentObservation)],
    ) -> Result<Vec<Option<u64>>> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut rowids = Vec::with_capacity(items.len());
        {
            let mut select_stmt =
                tx.prepare_cached("SELECT rowid FROM memories WHERE memory_id = ?1")?;
            let mut update_stmt = tx.prepare_cached(
                "UPDATE memories SET content = ?1, kind = ?2, created_at_ms = ?3, entity_id = ?4, content_hash = ?5,
                 session_id = ?7, turn_index = ?8, role = ?9, parent_memory_id = ?10 WHERE rowid = ?6",
            )?;
            let mut insert_stmt = tx.prepare_cached(
                "INSERT INTO memories (memory_id, entity_id, content, kind, content_hash, created_at_ms,
                                       session_id, turn_index, role, parent_memory_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )?;
            let mut vec_stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO vector_lookup (vector_id, memory_id, entity_id, timestamp_ms, embedding)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            let mut del_vec_stmt =
                tx.prepare_cached("DELETE FROM vector_lookup WHERE vector_id = ?1")?;
            for &(ts, ref mem_id, obs) in items {
                let existing_rid: Option<i64> =
                    select_stmt.query_row(params![mem_id], |row| row.get(0)).ok();
                let rid = if let Some(rid) = existing_rid {
                    update_stmt.execute(params![
                        obs.textual_content,
                        format!("{:?}", obs.kind),
                        ts,
                        obs.entity_id,
                        obs.content_hash,
                        rid,
                        obs.session_id,
                        obs.turn_index,
                        obs.role,
                        obs.parent_memory_id
                    ])?;
                    rid
                } else {
                    insert_stmt.execute(params![
                        mem_id,
                        obs.entity_id,
                        obs.textual_content,
                        format!("{:?}", obs.kind),
                        obs.content_hash,
                        ts,
                        obs.session_id,
                        obs.turn_index,
                        obs.role,
                        obs.parent_memory_id
                    ])?;
                    tx.last_insert_rowid()
                };

                if !obs.embedding.is_empty() {
                    vec_stmt.execute(params![
                        rid,
                        mem_id,
                        obs.entity_id,
                        ts,
                        vec_f32_to_bytes(&obs.embedding)
                    ])?;
                } else {
                    del_vec_stmt.execute(params![rid])?;
                }
                rowids.push(Some(rid as u64));
            }
        }
        tx.commit()?;
        Ok(rowids)
    }

    pub fn insert_observations_batch(
        &self,
        items: &[(u64, String, AgentObservation)],
    ) -> Result<Vec<Option<u64>>> {
        let mapped: Vec<(u64, String, &AgentObservation)> =
            items.iter().map(|(ts, mid, obs)| (*ts, mid.clone(), obs)).collect();
        self.allocate_vector_ids(&mapped)
    }

    pub fn insert_observation(
        &self,
        timestamp: u64,
        memory_id: &str,
        obs: &AgentObservation,
    ) -> Result<()> {
        self.insert_observations_batch(&[(timestamp, memory_id.to_string(), obs.clone())])?;
        Ok(())
    }

    /// Returns `(created_at_ms, vector_id)`; `vector_id` is `None` for
    /// memories stored without an embedding.
    pub fn lookup_by_memory_id(&self, memory_id: &str) -> Result<Option<(u64, Option<u64>)>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT m.created_at_ms, v.vector_id
             FROM memories m
             LEFT JOIN vector_lookup v ON v.memory_id = m.memory_id
             WHERE m.memory_id = ?1",
        )?;
        let res = stmt.query_row(params![memory_id], |row| {
            Ok((row.get::<_, i64>(0)? as u64, row.get::<_, Option<i64>>(1)?.map(|v| v as u64)))
        });
        match res {
            Ok(pair) => Ok(Some(pair)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn lookup_by_vector_ids_batch(
        &self,
        vector_ids: &[u64],
    ) -> Result<Vec<Option<(u64, String)>>> {
        if vector_ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.get_conn()?;
        let placeholders: Vec<String> = vector_ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT vector_id, memory_id, timestamp_ms FROM vector_lookup WHERE vector_id IN ({})",
            placeholders.join(",")
        );
        let mut stmt = conn.prepare_cached(&sql)?;
        let params: Vec<i64> = vector_ids.iter().map(|v| *v as i64).collect();
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p as &dyn rusqlite::types::ToSql).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, i64>(0)? as u64,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)? as u64,
            ))
        })?;

        let mut lookup: std::collections::HashMap<u64, (u64, String)> =
            std::collections::HashMap::with_capacity(vector_ids.len());
        for row in rows {
            let (vid, memory_id, ts) = row?;
            lookup.insert(vid, (ts, memory_id));
        }
        let results: Vec<Option<(u64, String)>> =
            vector_ids.iter().map(|vid| lookup.get(vid).cloned()).collect();
        Ok(results)
    }

    pub fn lookup_by_memory_ids_batch(
        &self,
        memory_ids: &[String],
    ) -> Result<std::collections::HashMap<String, (u64, u64)>> {
        // Values are `(created_at_ms, vector_id)`; every caller reads the first
        // element as the memory timestamp. (They used to be swapped, which fed
        // vector ids into recency scoring as timestamps.)
        if memory_ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let mut result = std::collections::HashMap::with_capacity(memory_ids.len());
        let conn = self.get_conn()?;
        let placeholders: Vec<String> = memory_ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT m.memory_id, v.vector_id, m.created_at_ms
             FROM memories m
             LEFT JOIN vector_lookup v ON v.memory_id = m.memory_id
             WHERE m.memory_id IN ({})",
            placeholders.join(",")
        );
        let mut stmt = conn.prepare_cached(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            memory_ids.iter().map(|s| s as &dyn rusqlite::types::ToSql).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<i64>>(1)?.map(|v| v as u64),
                row.get::<_, i64>(2)? as u64,
            ))
        })?;

        for row in rows {
            let (mid, vid_opt, ts) = row?;
            result.insert(mid, (ts, vid_opt.unwrap_or(0)));
        }
        Ok(result)
    }

    /// `memory_id -> (session_id, turn_index)` for stored memories.
    pub fn memory_identity_batch(
        &self,
        memory_ids: &[String],
    ) -> Result<HashMap<String, (String, u32)>> {
        let mut out = HashMap::with_capacity(memory_ids.len());
        if memory_ids.is_empty() {
            return Ok(out);
        }
        let conn = self.get_conn()?;
        for chunk in memory_ids.chunks(256) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT memory_id, session_id, turn_index FROM memories WHERE memory_id IN ({placeholders})"
            ))?;
            let rows = stmt.query_map(rusqlite::params_from_iter(chunk.iter()), |row| {
                Ok((row.get::<_, String>(0)?, (row.get::<_, String>(1)?, row.get::<_, u32>(2)?)))
            })?;
            for row in rows {
                let (id, identity) = row?;
                out.insert(id, identity);
            }
        }
        Ok(out)
    }

    pub fn get_observation(
        &self,
        _timestamp: u64,
        memory_id: &str,
    ) -> Result<Option<AgentObservation>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT entity_id, content, kind, created_at_ms, session_id, turn_index, role, parent_memory_id
             FROM memories WHERE memory_id = ?1",
        )?;
        let res = stmt.query_row(params![memory_id], |row| {
            Ok(AgentObservation {
                entity_id: row.get(0)?,
                textual_content: row.get(1)?,
                embedding: Vec::new(),
                kind: parse_kind_enum(row.get::<_, String>(2)?.as_str()),
                content_hash: String::new(),
                created_at_ms: row.get(3)?,
                session_id: row.get(4)?,
                turn_index: row.get(5)?,
                role: row.get(6)?,
                parent_memory_id: row.get(7)?,
            })
        });
        match res {
            Ok(obs) => Ok(Some(obs)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn get_observations_batch(
        &self,
        keys: &[(u64, String)],
    ) -> Result<std::collections::HashMap<String, AgentObservation>> {
        if keys.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let memory_ids: Vec<&str> = keys.iter().map(|(_, mid)| mid.as_str()).collect();
        let conn = self.get_conn()?;
        let placeholders: Vec<String> = memory_ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT memory_id, entity_id, content, kind, created_at_ms, session_id, turn_index, role, parent_memory_id
             FROM memories WHERE memory_id IN ({})",
            placeholders.join(",")
        );
        let mut stmt = conn.prepare_cached(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            memory_ids.iter().map(|s| s as &dyn rusqlite::types::ToSql).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0)?,
                AgentObservation {
                    entity_id: row.get::<_, String>(1)?,
                    textual_content: row.get::<_, String>(2)?,
                    embedding: Vec::new(),
                    kind: parse_kind_enum(row.get::<_, String>(3)?.as_str()),
                    content_hash: String::new(),
                    created_at_ms: row.get::<_, i64>(4)? as u64,
                    session_id: row.get(5)?,
                    turn_index: row.get(6)?,
                    role: row.get(7)?,
                    parent_memory_id: row.get(8)?,
                },
            ))
        })?;
        let mut result = std::collections::HashMap::new();
        for row in rows {
            let (memory_id, obs) = row?;
            result.insert(memory_id, obs);
        }
        Ok(result)
    }

    // ── Memory Cards ──

    pub fn ingest_cards(&self, cards: &[MemoryCard]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO memory_cards (
                    card_id, entity_id, user_id, source_memory_id, source_session_id,
                    subject, predicate, object, memory_text, card_type, confidence,
                    is_latest, is_static, is_inference, expires_at, root_card_id, parent_card_id,
                    lifecycle, created_at_ms, updated_at_ms
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20)",
            )?;
            for card in cards {
                stmt.execute(params![
                    card.card_id,
                    card.entity_id,
                    card.user_id,
                    card.source_memory_id,
                    card.source_session_id,
                    card.subject,
                    card.predicate,
                    card.object,
                    card.memory_text,
                    card.card_type,
                    card.confidence,
                    card.is_latest as i32,
                    card.is_static as i32,
                    card.is_inference as i32,
                    card.expires_at,
                    card.root_card_id,
                    card.parent_card_id,
                    card.lifecycle
                        .as_ref()
                        .map(serde_json::to_string)
                        .transpose()
                        .context("failed to serialize card lifecycle")?
                        .as_deref()
                        .unwrap_or(""),
                    card.created_at_ms,
                    card.updated_at_ms,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn get_memory_card(&self, card_id: &str) -> Result<Option<MemoryCard>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT card_id, entity_id, user_id, source_memory_id, source_session_id,
                    subject, predicate, object, memory_text, card_type, confidence,
                    is_latest, is_static, is_inference, expires_at, root_card_id, parent_card_id,
                    lifecycle, created_at_ms, updated_at_ms
             FROM memory_cards WHERE card_id = ?1",
        )?;
        let res = stmt.query_row(params![card_id], |row| {
            let lifecycle_str: Option<String> = row.get(17)?;
            Ok(MemoryCard {
                card_id: row.get(0)?,
                entity_id: row.get(1)?,
                user_id: row.get(2)?,
                source_memory_id: row.get(3)?,
                source_session_id: row.get(4)?,
                subject: row.get(5)?,
                predicate: row.get(6)?,
                object: row.get(7)?,
                memory_text: row.get(8)?,
                card_type: row.get(9)?,
                confidence: row.get(10)?,
                is_latest: row.get::<_, i32>(11)? != 0,
                is_static: row.get::<_, i32>(12)? != 0,
                is_inference: row.get::<_, i32>(13)? != 0,
                expires_at: row.get(14)?,
                root_card_id: row.get(15)?,
                parent_card_id: row.get(16)?,
                lifecycle: lifecycle_str.and_then(|s| serde_json::from_str(&s).ok()),
                source_turn_index: 0,
                document_time: 0,
                conversation_time: 0,
                event_time: None,
                created_at_ms: row.get(18)?,
                updated_at_ms: row.get(19)?,
            })
        });
        match res {
            Ok(card) => Ok(Some(card)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn set_memory_card_latest_batch(&self, updates: &[(String, bool, u64)]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "UPDATE memory_cards SET is_latest = ?1, updated_at_ms = ?2 WHERE card_id = ?3",
            )?;
            for (card_id, is_latest, ts) in updates {
                stmt.execute(params![*is_latest as i32, *ts as i64, card_id])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    // ── Aliases ──

    pub fn set_aliases_batch(&self, entity_id: &str, aliases: &[(String, String)]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR IGNORE INTO aliases (entity_id, alias) VALUES (?1, ?2)",
            )?;
            for (alias, _canonical) in aliases {
                stmt.execute(params![entity_id, alias])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    // ── Entity Registry (Tiered Resolver) ──

    /// Register a canonical entity name with its phonetic key.
    /// If the name already exists for this scope, it is a no-op.
    pub fn register_entity(&self, entity_id: &str, canonical_name: &str) -> Result<()> {
        let now =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        let sk = crate::storage::entity_resolver::phonetic_key(canonical_name);
        let conn = self.get_conn()?;
        conn.execute(
            "INSERT OR IGNORE INTO entity_registry (entity_id, canonical_name, soundex_key, updated_at_ms)
             VALUES (?1, ?2, ?3, ?4)",
            params![entity_id, canonical_name, sk, now as i64],
        )?;
        Ok(())
    }

    /// Load all entity candidates for a scope, with aliases, soundex keys, and embeddings.
    pub fn load_entity_candidates(
        &self,
        entity_id: &str,
    ) -> Result<Vec<crate::storage::entity_resolver::EntityCandidate>> {
        let conn = self.get_conn()?;

        let mut reg_stmt = conn.prepare_cached(
            "SELECT canonical_name, soundex_key FROM entity_registry WHERE entity_id = ?1",
        )?;
        let rows = reg_stmt.query_map(params![entity_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut candidates: Vec<crate::storage::entity_resolver::EntityCandidate> = Vec::new();
        let mut name_index: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for row in rows {
            let (name, sk) = row?;
            let idx = candidates.len();
            candidates.push(crate::storage::entity_resolver::EntityCandidate {
                name: name.clone(),
                aliases: Vec::new(),
                soundex_key: sk,
                embedding: None,
            });
            name_index.insert(name.to_ascii_lowercase(), idx);
        }

        let mut alias_stmt =
            conn.prepare_cached("SELECT alias FROM aliases WHERE entity_id = ?1")?;
        let alias_rows = alias_stmt.query_map(params![entity_id], |row| row.get::<_, String>(0))?;
        for alias in alias_rows {
            let alias = alias?;
            for (lower_name, idx) in &name_index {
                let candidate = &mut candidates[*idx];
                if alias.to_ascii_lowercase().contains(lower_name)
                    || lower_name.contains(&alias.to_ascii_lowercase())
                {
                    if !candidate.aliases.contains(&alias) {
                        candidate.aliases.push(alias.clone());
                    }
                    break;
                }
            }
        }

        let mut emb_stmt = conn.prepare_cached(
            "SELECT canonical_name, embedding_blob, dim FROM name_embeddings WHERE entity_id = ?1",
        )?;
        let emb_rows = emb_stmt.query_map(params![entity_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, i64>(2)? as usize,
            ))
        })?;
        for row in emb_rows {
            let (cname, blob, dim) = row?;
            if let Some(idx) = name_index.get(&cname.to_ascii_lowercase()) {
                let embedding: Vec<f32> = blob
                    .chunks(4)
                    .take(dim)
                    .filter_map(|chunk| {
                        if chunk.len() == 4 {
                            Some(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        } else {
                            None
                        }
                    })
                    .collect();
                if embedding.len() == dim {
                    candidates[*idx].embedding = Some(embedding);
                }
            }
        }

        Ok(candidates)
    }

    fn proposal_id(entity_id: &str, from: &str, to: &str, now_ms: u64) -> String {
        format!("merge::{}::{}::{}::{}", entity_id, from, to, now_ms)
    }

    /// Create a pending merge proposal. Returns the proposal_id if created.
    pub fn create_merge_proposal(
        &self,
        entity_id: &str,
        from_name: &str,
        to_name: &str,
        tier: &str,
        confidence: f32,
    ) -> Result<Option<String>> {
        let now =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        let pid = Self::proposal_id(entity_id, from_name, to_name, now);
        let conn = self.get_conn()?;
        let inserted = conn.execute(
            "INSERT OR IGNORE INTO merge_proposals (proposal_id, entity_id, from_name, to_name, tier, confidence, status, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', ?7)",
            params![pid, entity_id, from_name, to_name, tier, confidence as f64, now as i64],
        )?;
        if inserted > 0 {
            Ok(Some(pid))
        } else {
            Ok(None)
        }
    }

    /// Run the tiered resolver against the entity registry.
    /// Creates merge proposals for non-exact matches.
    pub fn resolve_and_propose(
        &self,
        entity_id: &str,
        name: &str,
        name_embedding: Option<&[f32]>,
        config: &crate::storage::entity_resolver::ResolutionConfig,
    ) -> Result<crate::storage::entity_resolver::EntityResolution> {
        let candidates = self.load_entity_candidates(entity_id)?;
        let resolution = crate::storage::entity_resolver::resolve_name(
            name,
            &candidates,
            name_embedding,
            config,
        );
        if let Some(ref matched) = resolution.matched_name {
            let tier_label = match resolution.tier {
                crate::storage::entity_resolver::ResolverTier::Exact => return Ok(resolution),
                crate::storage::entity_resolver::ResolverTier::Fuzzy(_) => "fuzzy",
                crate::storage::entity_resolver::ResolverTier::Phonetic => "phonetic",
                crate::storage::entity_resolver::ResolverTier::Embedding(_) => "embedding",
            };
            if !name.eq_ignore_ascii_case(matched) {
                let _ = self.create_merge_proposal(
                    entity_id,
                    name,
                    matched,
                    tier_label,
                    resolution.tier.confidence(),
                );
            }
        }
        Ok(resolution)
    }

    /// Batch check which content hashes already exist.
    /// Returns a set of hashes that are already stored.
    /// Stored `content_hash` per memory id (ids not stored are absent).
    pub fn stored_content_hashes(&self, memory_ids: &[String]) -> Result<HashMap<String, String>> {
        let mut out = HashMap::with_capacity(memory_ids.len());
        let conn = self.get_conn()?;
        for chunk in memory_ids.chunks(256) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT memory_id, content_hash FROM memories WHERE memory_id IN ({placeholders})"
            ))?;
            let rows = stmt.query_map(rusqlite::params_from_iter(chunk.iter()), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            for row in rows {
                let (id, hash) = row?;
                out.insert(id, hash);
            }
        }
        Ok(out)
    }

    /// Source turns (not derived records) of one session with
    /// `lo <= turn_index <= hi`, ordered by turn: `(memory_id, turn, role, content)`.
    pub fn session_turn_window(
        &self,
        entity_id: &str,
        session_id: &str,
        lo: u32,
        hi: u32,
    ) -> Result<Vec<(String, u32, String, String)>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT memory_id, turn_index, role, content FROM memories
             WHERE entity_id = ?1 AND session_id = ?2 AND turn_index BETWEEN ?3 AND ?4
               AND (parent_memory_id IS NULL OR memory_id = parent_memory_id || '::c0')
             ORDER BY turn_index, rowid",
        )?;
        let rows = stmt.query_map(params![entity_id, session_id, lo, hi], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Replaces stored embeddings of existing memories and returns
    /// `(vector_id, entity_id, embedding)` for updating the vector index.
    /// Memories stored without a vector are skipped.
    pub fn update_embeddings(
        &self,
        updates: &[(String, Vec<f32>)],
    ) -> Result<Vec<(u64, String, Vec<f32>)>> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut applied = Vec::with_capacity(updates.len());
        {
            let mut select = tx.prepare_cached(
                "SELECT vector_id, entity_id FROM vector_lookup WHERE memory_id = ?1",
            )?;
            let mut update =
                tx.prepare_cached("UPDATE vector_lookup SET embedding = ?1 WHERE vector_id = ?2")?;
            for (memory_id, embedding) in updates {
                let found = match select.query_row(params![memory_id], |row| {
                    Ok((row.get::<_, i64>(0)? as u64, row.get::<_, String>(1)?))
                }) {
                    Ok(found) => found,
                    Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                    Err(err) => return Err(err.into()),
                };
                update.execute(params![vec_f32_to_bytes(embedding), found.0 as i64])?;
                applied.push((found.0, found.1, embedding.clone()));
            }
        }
        tx.commit()?;
        Ok(applied)
    }

    pub fn existing_content_hashes(
        &self,
        hashes: &[String],
    ) -> Result<std::collections::HashSet<String>> {
        let conn = self.get_conn()?;
        let mut found = std::collections::HashSet::new();
        let mut stmt =
            conn.prepare_cached("SELECT content_hash FROM memories WHERE content_hash = ?1")?;
        for h in hashes {
            let exists: bool = stmt
                .query_row(params![h], |row| row.get::<_, String>(0))
                .ok()
                .map(|s| !s.is_empty())
                .unwrap_or(false);
            if exists {
                found.insert(h.clone());
            }
        }
        Ok(found)
    }

    // ── Memory Links ──

    pub fn set_memory_links_batch(&self, links: &[(String, String, String)]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR IGNORE INTO memory_links (source_memory_id, target_memory_id, link_type)
                 VALUES (?1, ?2, ?3)",
            )?;
            for (src, tgt, link_type) in links {
                stmt.execute(params![src, tgt, link_type])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn get_linked_memories(&self, memory_id: &str) -> Result<Vec<String>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT target_memory_id FROM memory_links WHERE source_memory_id = ?1
             UNION ALL
             SELECT source_memory_id FROM memory_links WHERE target_memory_id = ?1",
        )?;
        let rows = stmt.query_map(params![memory_id], |row| row.get::<_, String>(0))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    pub fn get_link_cluster_scores(
        &self,
        seed_memory_id: &str,
        max_depth: usize,
    ) -> Result<HashMap<String, f32>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "WITH RECURSIVE
               bfs(node, depth, path_weight) AS (
                 SELECT ?1, 0, 1.0
                 -- UNION (not ALL): links are stored in both directions, so the
                 -- same neighbour is reached twice per hop and was double-counted.
                 UNION
                 SELECT
                   CASE WHEN ml.source_memory_id = bfs.node THEN ml.target_memory_id ELSE ml.source_memory_id END,
                   bfs.depth + 1,
                   bfs.path_weight * 0.6
                 FROM bfs
                 JOIN memory_links ml ON ml.source_memory_id = bfs.node OR ml.target_memory_id = bfs.node
                 WHERE bfs.depth < ?2
               )
             SELECT node, SUM(path_weight) FROM bfs WHERE depth > 0 GROUP BY node;"
        )?;
        let rows = stmt.query_map(params![seed_memory_id, max_depth as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
        })?;
        let mut result = HashMap::new();
        for row in rows {
            let (node, weight) = row?;
            result.insert(node, weight);
        }
        Ok(result)
    }

    #[allow(dead_code)]
    pub fn get_edge_cluster_neighbors(
        &self,
        seed_memory_id: &str,
        edge_type_filter: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "WITH seed_nodes AS (
                 SELECT source AS node FROM edges WHERE memory_id = ?1
                 UNION
                 SELECT target AS node FROM edges WHERE memory_id = ?1
             )
             SELECT DISTINCT e.memory_id, e.weight
             FROM edges e
             JOIN seed_nodes sn ON (e.source = sn.node OR e.target = sn.node)
             WHERE e.memory_id != ?1
               AND (?2 IS NULL OR e.edge_type = ?2)
             ORDER BY e.weight DESC
             LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![seed_memory_id, edge_type_filter, limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
            })?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Like `get_edge_cluster_neighbors` but also returns the edge type so the
    /// caller can apply intent-aware re-weighting. The `edge_type_filter`
    /// parameter is still honored: pass `None` to return all edge types.
    pub fn get_edge_cluster_neighbors_typed(
        &self,
        seed_memory_id: &str,
        edge_type_filter: Option<&str>,
        limit: usize,
        max_node_degree: usize,
    ) -> Result<Vec<(String, f32, String)>> {
        let conn = self.get_conn()?;
        // Nodes with more than `max_node_degree` edges are not traversed. Such
        // hubs (the entity id, speaker labels like "assistant", header words
        // from derived text, the empty name) connect nearly every memory, carry
        // no relational signal, and made each hop scan thousands of edges.
        // Degree counts are capped so checking a hub stays cheap.
        let mut stmt = conn.prepare_cached(
            "WITH seed_nodes AS (
                 SELECT node FROM (
                     SELECT source AS node FROM edges WHERE memory_id = ?1
                     UNION ALL
                     SELECT target AS node FROM edges WHERE memory_id = ?1
                 )
                 WHERE node != ''
                   AND (SELECT COUNT(*) FROM (SELECT 1 FROM edges x WHERE x.source = node LIMIT ?4 + 1))
                     + (SELECT COUNT(*) FROM (SELECT 1 FROM edges y WHERE y.target = node LIMIT ?4 + 1))
                     <= ?4
             )
             SELECT memory_id, weight, edge_type FROM (
                 SELECT e.memory_id, e.weight, e.edge_type
                 FROM edges e
                 JOIN seed_nodes sn ON e.source = sn.node
                 WHERE e.memory_id != ?1
                   AND (?2 IS NULL OR e.edge_type = ?2)
                 UNION ALL
                 SELECT e.memory_id, e.weight, e.edge_type
                 FROM edges e
                 JOIN seed_nodes sn ON e.target = sn.node
                 WHERE e.memory_id != ?1
                   AND (?2 IS NULL OR e.edge_type = ?2)
             )
             ORDER BY weight DESC, memory_id
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(
            params![seed_memory_id, edge_type_filter, limit as i64, max_node_degree as i64],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, f64>(1)? as f32,
                    row.get::<_, String>(2)?,
                ))
            },
        )?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Batched form of `get_edge_cluster_neighbors_typed` for many memories:
    /// the same rows (including multiplicities) and ordering per memory, from
    /// a handful of `IN` queries instead of one query per memory.
    pub fn get_edge_cluster_neighbors_batch(
        &self,
        memory_ids: &[String],
        edge_type_filter: Option<&str>,
        limit: usize,
        max_node_degree: usize,
    ) -> Result<HashMap<String, Vec<EdgeNeighbour>>> {
        const CHUNK: usize = 400;
        let conn = self.get_conn()?;
        let placeholders = |n: usize| vec!["?"; n].join(",");

        // Node multiset per memory (a node listed once per edge endpoint).
        let mut nodes_of: HashMap<String, Vec<String>> = HashMap::new();
        for ids in memory_ids.chunks(CHUNK) {
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT memory_id, source, target FROM edges WHERE memory_id IN ({})",
                placeholders(ids.len())
            ))?;
            let rows = stmt.query_map(rusqlite::params_from_iter(ids), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
            })?;
            for row in rows {
                let (memory_id, source, target) = row?;
                let nodes = nodes_of.entry(memory_id).or_default();
                nodes.extend([source, target].into_iter().filter(|n| !n.is_empty()));
            }
        }

        let mut unique: Vec<String> = nodes_of.values().flatten().cloned().collect();
        unique.sort();
        unique.dedup();

        // Degree = edges with the node as source + as target; hubs are skipped.
        let mut degree: HashMap<String, usize> = HashMap::new();
        for column in ["source", "target"] {
            for nodes in unique.chunks(CHUNK) {
                let mut stmt = conn.prepare_cached(&format!(
                    "SELECT {column}, COUNT(*) FROM edges WHERE {column} IN ({}) GROUP BY {column}",
                    placeholders(nodes.len())
                ))?;
                let rows = stmt.query_map(rusqlite::params_from_iter(nodes), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
                })?;
                for row in rows {
                    let (node, count) = row?;
                    *degree.entry(node).or_default() += count;
                }
            }
        }
        let traversable: Vec<String> = unique
            .into_iter()
            .filter(|n| degree.get(n).copied().unwrap_or(0) <= max_node_degree)
            .collect();

        type Incident = HashMap<String, Vec<(String, f32, String)>>;
        let mut incident: [Incident; 2] = [HashMap::new(), HashMap::new()];
        for (slot, column) in ["source", "target"].into_iter().enumerate() {
            for nodes in traversable.chunks(CHUNK) {
                // The filter is bound last: bare `?` markers number from 1.
                let filter_idx = nodes.len() + 1;
                let mut stmt = conn.prepare_cached(&format!(
                    "SELECT {column}, memory_id, weight, edge_type FROM edges
                     WHERE {column} IN ({}) AND memory_id IS NOT NULL
                       AND (?{filter_idx} IS NULL OR edge_type = ?{filter_idx})",
                    placeholders(nodes.len())
                ))?;
                let mut params: Vec<&dyn rusqlite::types::ToSql> =
                    nodes.iter().map(|n| n as &dyn rusqlite::types::ToSql).collect();
                params.push(&edge_type_filter);
                let rows = stmt.query_map(params.as_slice(), |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, f64>(2)? as f32,
                        row.get::<_, String>(3)?,
                    ))
                })?;
                for row in rows {
                    let (node, memory_id, weight, edge_type) = row?;
                    incident[slot].entry(node).or_default().push((memory_id, weight, edge_type));
                }
            }
        }

        let traversable: std::collections::HashSet<&str> =
            traversable.iter().map(String::as_str).collect();
        let mut result = HashMap::with_capacity(memory_ids.len());
        for memory_id in memory_ids {
            let mut rows: Vec<(String, f32, String)> = Vec::new();
            for node in nodes_of.get(memory_id).into_iter().flatten() {
                if !traversable.contains(node.as_str()) {
                    continue;
                }
                for side in &incident {
                    rows.extend(
                        side.get(node)
                            .into_iter()
                            .flatten()
                            .filter(|(mid, _, _)| mid != memory_id)
                            .cloned(),
                    );
                }
            }
            rows.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            rows.truncate(limit);
            result.insert(memory_id.clone(), rows);
        }
        Ok(result)
    }

    // ── Session Router ──

    pub fn merge_session_router_records_batch(
        &self,
        updates: &[SessionRouterRecord],
    ) -> Result<Vec<SessionRouterRecord>> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut merged_results = Vec::new();
        {
            let mut select_stmt = tx.prepare_cached(
                "SELECT rowid, record_json FROM session_router WHERE session_id = ?1 AND entity_id = ?2",
            )?;
            let mut upsert_stmt = tx.prepare_cached(
                "INSERT INTO session_router (session_id, entity_id, record_json, router_text, created_at_ms, updated_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(session_id, entity_id) DO UPDATE SET
                    record_json = excluded.record_json,
                    router_text = excluded.router_text,
                    updated_at_ms = excluded.updated_at_ms
                 RETURNING rowid"
            )?;
            let mut fts = tx.prepare_cached(
                "INSERT OR REPLACE INTO fts_session_router (rowid, session_id, entity_id, router_text) VALUES (?1, ?2, ?3, ?4)"
            )?;
            let now = unix_timestamp_ms()? as u64;
            for record in updates {
                // Fetch existing record if any and merge
                let merged = match select_stmt
                    .query_row(params![record.session_id, record.entity_id], |row| {
                        row.get::<_, String>(1)
                    }) {
                    Ok(existing_json) => {
                        if let Ok(existing) =
                            serde_json::from_str::<SessionRouterRecord>(&existing_json)
                        {
                            merge_router_records(&existing, record)
                        } else {
                            record.clone()
                        }
                    }
                    Err(_) => record.clone(),
                };
                let json = serde_json::to_string(&merged)?;
                let rowid: i64 = upsert_stmt.query_row(
                    params![
                        merged.session_id,
                        merged.entity_id,
                        json,
                        &merged.router_text,
                        merged.created_at_ms.min(now),
                        now,
                    ],
                    |row| row.get(0),
                )?;

                fts.execute(params![
                    rowid,
                    &merged.session_id,
                    &merged.entity_id,
                    &merged.router_text
                ])?;
                merged_results.push(merged);
            }
        }
        tx.commit()?;
        Ok(merged_results)
    }

    pub fn search_session_router(
        &self,
        entity_id: &str,
        query: &str,
        lexical_terms: &[String],
        temporal_terms: &[String],
        entities: &[String],
        limit: usize,
    ) -> Result<Vec<SessionRouterSearchHit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let query_lower = query.to_ascii_lowercase();
        let conn = self.get_conn()?;

        // Use FTS5 to get candidate pool instead of full table scan
        let terms: Vec<&str> =
            query_lower.split_whitespace().filter(|t| t.len() > SEARCH_MIN_TERM_LEN).collect();
        let hits = if terms.is_empty() {
            // Fall back to full scan if no substantial terms
            let mut stmt =
                conn.prepare_cached("SELECT record_json FROM session_router WHERE entity_id = ?1")?;
            let rows = stmt.query_map(params![entity_id], |row| row.get::<_, String>(0))?;
            Self::score_session_router_rows(
                rows,
                &query_lower,
                lexical_terms,
                temporal_terms,
                entities,
            )
        } else {
            let fts_query =
                terms.iter().map(|t| format!("\"{}\"", t)).collect::<Vec<_>>().join(" OR ");
            let mut stmt = conn.prepare_cached(
                "SELECT sr.record_json
                 FROM fts_session_router fsr
                 JOIN session_router sr ON sr.session_id = fsr.session_id AND sr.entity_id = fsr.entity_id
                 WHERE fsr.fts_session_router MATCH ?1 AND fsr.entity_id = ?2
                 ORDER BY rank LIMIT ?3"
            )?;
            let rows = stmt.query_map(
                params![fts_query, entity_id, (limit.saturating_mul(3)) as i64],
                |row| row.get::<_, String>(0),
            )?;
            Self::score_session_router_rows(
                rows,
                &query_lower,
                lexical_terms,
                temporal_terms,
                entities,
            )
        };

        let mut hits = hits;
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.session_id.cmp(&b.session_id))
        });
        hits.truncate(limit);
        Ok(hits)
    }

    fn score_session_router_rows(
        rows: impl Iterator<Item = Result<String, rusqlite::Error>>,
        query_lower: &str,
        lexical_terms: &[String],
        temporal_terms: &[String],
        entities: &[String],
    ) -> Vec<SessionRouterSearchHit> {
        let mut hits = Vec::new();
        for row in rows {
            let json = match row {
                Ok(j) => j,
                Err(_) => continue,
            };
            let Ok(record) = serde_json::from_str::<SessionRouterRecord>(&json) else {
                continue;
            };
            let router_text = if record.router_text.is_empty() {
                build_session_router_text(&record)
            } else {
                record.router_text.clone()
            };
            let lower = router_text.to_ascii_lowercase();
            let lexical_hits = contains_term_count(&lower, lexical_terms);
            let temporal_hits = contains_term_count(&lower, temporal_terms);
            let entity_hits = contains_term_count(&lower, entities);
            let exact_focus_hit = !record.session_focus.is_empty()
                && query_lower
                    .split_whitespace()
                    .filter(|part| part.len() >= FOCUS_MATCH_MIN_LEN)
                    .any(|part| record.session_focus.to_ascii_lowercase().contains(part));

            if lexical_hits == 0 && temporal_hits == 0 && entity_hits == 0 && !exact_focus_hit {
                // Last-ditch accept: if the router_text has ANY of the raw query
                // terms (not just the classified lexical/temporal/entity terms),
                // keep the row. The classified term lists are often empty for
                // short or open-vocabulary questions.
                let lower_terms: Vec<&str> = query_lower
                    .split_whitespace()
                    .filter(|t| t.len() > SEARCH_MIN_TERM_LEN)
                    .collect();
                let has_raw_term =
                    !lower_terms.is_empty() && lower_terms.iter().any(|t| lower.contains(t));
                if !has_raw_term {
                    continue;
                }
            }

            let lexical_coverage = if lexical_terms.is_empty() {
                0.0
            } else {
                lexical_hits as f32 / lexical_terms.len() as f32
            };
            let temporal_coverage = if temporal_terms.is_empty() {
                0.0
            } else {
                temporal_hits as f32 / temporal_terms.len() as f32
            };
            let entity_coverage =
                if entities.is_empty() { 0.0 } else { entity_hits as f32 / entities.len() as f32 };
            let source_depth =
                (record.source_memory_ids.len() as f32 / SOURCE_DEPTH_DIVISOR).min(1.0);
            let score = lexical_coverage.min(1.0) * SESSION_LEXICAL_WEIGHT
                + temporal_coverage.min(1.0) * SESSION_TEMPORAL_WEIGHT
                + entity_coverage.min(1.0) * SESSION_ENTITY_WEIGHT
                + source_depth * SESSION_DEPTH_WEIGHT
                + if exact_focus_hit { SESSION_FOCUS_BONUS } else { 0.0 };

            hits.push(SessionRouterSearchHit {
                session_id: record.session_id,
                score,
                lexical_hits,
                temporal_hits,
                entity_hits,
            });
        }
        hits
    }

    pub fn sessions_in_time_window(
        &self,
        entity_id: &str,
        start_ms: u64,
        end_ms: u64,
    ) -> Result<Vec<SessionRouterSearchHit>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT record_json FROM session_router
             WHERE entity_id = ?1 AND created_at_ms >= ?2 AND created_at_ms <= ?3",
        )?;
        let rows = stmt.query_map(params![entity_id, start_ms as i64, end_ms as i64], |row| {
            row.get::<_, String>(0)
        })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            let json = row?;
            if let Ok(record) = serde_json::from_str::<SessionRouterRecord>(&json) {
                results.push(SessionRouterSearchHit {
                    session_id: record.session_id.clone(),
                    score: 1.0,
                    lexical_hits: 0,
                    temporal_hits: 1,
                    entity_hits: 0,
                });
            }
        }
        Ok(results)
    }

    pub fn entity_pivot_sessions(
        &self,
        entity_id: &str,
        subject_entities: &[String],
    ) -> Result<Vec<SessionRouterSearchHit>> {
        if subject_entities.is_empty() {
            return Ok(Vec::new());
        }
        // Use FTS5 OR-query so we tokenize properly and the planner can use the
        // fts_session_router index. Fall back to LIKE if all terms are too short
        // for FTS5 (less than SEARCH_MIN_TERM_LEN).
        let conn = self.get_conn()?;
        let fts_terms: Vec<String> = subject_entities
            .iter()
            .filter(|e| e.len() > SEARCH_MIN_TERM_LEN)
            .map(|e| format!("\"{}\"", e.to_ascii_lowercase()))
            .collect();

        let mut session_to_hits: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        if !fts_terms.is_empty() {
            let fts_query = fts_terms.join(" OR ");
            let mut stmt = conn.prepare_cached(
                "SELECT sr.session_id, fsr.fts_session_router
                 FROM fts_session_router fsr
                 JOIN session_router sr
                   ON sr.session_id = fsr.session_id AND sr.entity_id = fsr.entity_id
                 WHERE fsr.fts_session_router MATCH ?1 AND fsr.entity_id = ?2
                 ORDER BY rank LIMIT ?3",
            )?;
            let rows = stmt.query_map(
                params![fts_query, entity_id, (subject_entities.len().saturating_mul(8)) as i64],
                |row| row.get::<_, String>(0),
            )?;
            for row in rows.flatten() {
                *session_to_hits.entry(row).or_insert(0) += 1;
            }
        }

        // Backstop: LIKE-based scan in case FTS5 missed something due to token
        // boundaries. Cheap because session_router is one row per session.
        for entity in subject_entities {
            if entity.len() < 3 {
                continue;
            }
            let needle = entity.to_ascii_lowercase();
            let mut stmt = conn.prepare_cached(
                "SELECT session_id FROM session_router
                 WHERE entity_id = ?1 AND LOWER(router_text) LIKE ?2",
            )?;
            let rows = stmt.query_map(params![entity_id, format!("%{}%", needle)], |row| {
                row.get::<_, String>(0)
            })?;
            for row in rows.flatten() {
                *session_to_hits.entry(row).or_insert(0) += 1;
            }
        }

        let mut results: Vec<SessionRouterSearchHit> = session_to_hits
            .into_iter()
            .map(|(session_id, hits)| SessionRouterSearchHit {
                score: hits as f32,
                lexical_hits: 0,
                temporal_hits: 0,
                entity_hits: hits,
                session_id,
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.session_id.cmp(&b.session_id))
        });
        Ok(results)
    }

    // ── Preferences ──

    pub fn set_preference_memories_batch(
        &self,
        entity_id: &str,
        items: &[(String, f32)],
    ) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO preferences (entity_id, memory_id, strength) VALUES (?1, ?2, ?3)",
            )?;
            for (memory_id, strength) in items {
                stmt.execute(params![entity_id, memory_id, strength])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn get_preference_memories(
        &self,
        entity_id: &str,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let conn = self.get_conn()?;
        let mut stmt =
            conn.prepare_cached("SELECT memory_id, strength FROM preferences WHERE entity_id = ?1 ORDER BY strength DESC LIMIT ?2")?;
        let rows = stmt.query_map(params![entity_id, limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f32>(1)?))
        })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Fact versions relevant to these memories, keyed by the id asked for.
    ///
    /// Facts are registered against derived records (cards, fact companions),
    /// so a conversation turn is matched through its derived children. When a
    /// turn has several, the version with the latest `valid_from_ms` wins. The
    /// rows carry what superseded them and which memories state the same
    /// value, for `why_stale` in results.
    pub fn fact_versions_for_memories(
        &self,
        memory_ids: &[String],
    ) -> Result<HashMap<String, FactVersionRow>> {
        if memory_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let conn = self.get_conn()?;
        let mut rows_by_memory: HashMap<String, (String, FactVersionRow)> = HashMap::new();
        for chunk in memory_ids.chunks(200) {
            let placeholders = vec!["?"; chunk.len()].join(",");
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT COALESCE(m.parent_memory_id, v.memory_id) AS asked_for,
                        v.memory_id, v.fact_key, v.entity_id, v.object, v.status,
                        v.valid_from_ms, v.valid_to_ms,
                        -- Report the turn the client sent, not the derived
                        -- record the fact was registered against.
                        COALESCE(
                            (SELECT sm.parent_memory_id FROM memories sm
                              WHERE sm.memory_id = v.superseded_by),
                            v.superseded_by
                        ),
                        (SELECT c.object FROM fact_versions c
                          WHERE c.fact_key = v.fact_key AND c.entity_id = v.entity_id
                            AND c.status = 'current' LIMIT 1),
                        (SELECT s.timestamp_ms FROM fact_versions s
                          WHERE s.fact_key = v.fact_key AND s.memory_id = v.superseded_by LIMIT 1)
                 FROM fact_versions v
                 LEFT JOIN memories m ON m.memory_id = v.memory_id
                 WHERE v.memory_id IN ({placeholders})
                    OR m.parent_memory_id IN ({placeholders})
                 ORDER BY v.valid_from_ms, v.memory_id"
            ))?;
            let params: Vec<&dyn rusqlite::types::ToSql> = chunk
                .iter()
                .chain(chunk.iter())
                .map(|s| s as &dyn rusqlite::types::ToSql)
                .collect();
            let mapped = stmt.query_map(params.as_slice(), |row| {
                let status: String = row.get(5)?;
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    FactVersionRow {
                        fact_key: row.get(2)?,
                        entity_id: row.get(3)?,
                        object: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                        is_current: status == "current",
                        valid_from_ms: row.get::<_, Option<i64>>(6)?.unwrap_or(0) as u64,
                        valid_to_ms: row.get::<_, Option<i64>>(7)?.map(|v| v as u64),
                        superseded_by: row.get(8)?,
                        current_object: row.get(9)?,
                        superseded_at_ms: row.get::<_, Option<i64>>(10)?.map(|v| v as u64),
                        evidence: Vec::new(),
                    },
                ))
            })?;
            let wanted: std::collections::HashSet<&str> =
                chunk.iter().map(String::as_str).collect();
            for row in mapped {
                let (asked_for, version_memory_id, version) = row?;
                // Direct matches win over a turn's derived records; among
                // derived records the ordering above leaves the latest.
                let key = if wanted.contains(version_memory_id.as_str()) {
                    version_memory_id.clone()
                } else {
                    asked_for
                };
                rows_by_memory.insert(key, (version_memory_id, version));
            }
        }

        // Evidence per version, newest first.
        let mut stmt = conn.prepare_cached(
            "SELECT COALESCE(
                        (SELECT em.parent_memory_id FROM memories em
                          WHERE em.memory_id = e.memory_id),
                        e.memory_id
                    )
             FROM fact_evidence e
             WHERE e.entity_id = ?1 AND e.fact_key = ?2 AND e.version_memory_id = ?3
             ORDER BY e.timestamp_ms DESC, e.memory_id",
        )?;
        let mut result = HashMap::with_capacity(rows_by_memory.len());
        for (asked_for, (version_memory_id, mut version)) in rows_by_memory {
            version.evidence = stmt
                .query_map(
                    params![version.entity_id, version.fact_key, version_memory_id],
                    |row| row.get(0),
                )?
                .collect::<rusqlite::Result<_>>()?;
            result.insert(asked_for, version);
        }
        Ok(result)
    }

    /// Every version of one fact, oldest first: what the value was, when it
    /// held, and which memories stated it. Ids are source turns where the
    /// fact came from a derived record.
    pub fn fact_history(&self, entity_id: &str, fact_key: &str) -> Result<Vec<FactHistoryEntry>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT v.memory_id,
                    COALESCE(
                        (SELECT m.parent_memory_id FROM memories m
                          WHERE m.memory_id = v.memory_id),
                        v.memory_id
                    ),
                    COALESCE(v.object, ''), v.status, v.valid_from_ms, v.valid_to_ms
             FROM fact_versions v
             WHERE v.entity_id = ?1 AND v.fact_key = ?2
             ORDER BY v.valid_from_ms, v.memory_id",
        )?;
        let rows: Vec<(String, String, String, String, u64, Option<u64>)> = stmt
            .query_map(params![entity_id, fact_key], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get::<_, Option<i64>>(4)?.unwrap_or(0) as u64,
                    row.get::<_, Option<i64>>(5)?.map(|v| v as u64),
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;

        let mut evidence_stmt = conn.prepare_cached(
            "SELECT COALESCE(
                        (SELECT em.parent_memory_id FROM memories em
                          WHERE em.memory_id = e.memory_id),
                        e.memory_id
                    )
             FROM fact_evidence e
             WHERE e.entity_id = ?1 AND e.fact_key = ?2 AND e.version_memory_id = ?3
             ORDER BY e.timestamp_ms DESC, e.memory_id",
        )?;
        let mut history = Vec::with_capacity(rows.len());
        for (version_id, memory_id, object, status, valid_from_ms, valid_to_ms) in rows {
            history.push(FactHistoryEntry {
                memory_id,
                object,
                is_current: status == "current",
                valid_from_ms,
                valid_to_ms,
                evidence: evidence_stmt
                    .query_map(params![entity_id, fact_key, version_id], |row| row.get(0))?
                    .collect::<rusqlite::Result<_>>()?,
            });
        }
        Ok(history)
    }

    /// Groups predicates by meaning: a predicate whose embedding is within
    /// `tau` of a known group joins it, otherwise it starts its own. Returns
    /// the canonical predicate for each input. Assignments are stored, so a
    /// predicate keeps its group once chosen.
    pub fn canonicalize_predicates(
        &self,
        entity_id: &str,
        predicates: &[(String, Vec<f32>)],
        tau: f32,
    ) -> Result<HashMap<String, String>> {
        let mut assigned = HashMap::with_capacity(predicates.len());
        if predicates.is_empty() {
            return Ok(assigned);
        }
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut known = tx.prepare_cached(
                "SELECT canonical FROM predicate_canon WHERE entity_id = ?1 AND predicate = ?2",
            )?;
            let mut groups = tx.prepare_cached(
                "SELECT canonical, embedding FROM predicate_canon
                 WHERE entity_id = ?1 AND predicate = canonical AND embedding IS NOT NULL",
            )?;
            let mut insert = tx.prepare_cached(
                "INSERT OR REPLACE INTO predicate_canon (entity_id, predicate, canonical, embedding)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;
            for (predicate, embedding) in predicates {
                match known.query_row(params![entity_id, predicate], |row| row.get::<_, String>(0))
                {
                    Ok(canonical) => {
                        assigned.insert(predicate.clone(), canonical);
                        continue;
                    }
                    Err(rusqlite::Error::QueryReturnedNoRows) => {}
                    Err(err) => return Err(err.into()),
                }
                let mut best: Option<(String, f32)> = None;
                let candidates = groups.query_map(params![entity_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
                })?;
                for candidate in candidates {
                    let (canonical, bytes) = candidate?;
                    let other = bytes_to_vec_f32(&bytes);
                    if other.len() != embedding.len() {
                        continue;
                    }
                    let similarity = crate::ml::cosine_similarity(embedding, &other);
                    if similarity >= tau && best.as_ref().map_or(true, |(_, s)| similarity > *s) {
                        best = Some((canonical, similarity));
                    }
                }
                let canonical = match best {
                    Some((canonical, _)) => canonical,
                    None => predicate.clone(),
                };
                let stored_embedding =
                    (canonical == *predicate).then(|| vec_f32_to_bytes(embedding));
                insert.execute(params![entity_id, predicate, canonical, stored_embedding])?;
                assigned.insert(predicate.clone(), canonical);
            }
        }
        tx.commit()?;
        Ok(assigned)
    }

    // ── Fact Versions ──

    /// Records fact versions and recomputes the validity chain of every
    /// affected `(entity, fact_key)`: versions are ordered by timestamp, each is
    /// valid until the next one starts, and only the latest is `current`.
    /// Recomputing the whole chain keeps intervals correct when versions
    /// arrive out of order (an earlier implementation only compared against
    /// the current version, leaving overlapping intervals for backfilled
    /// facts). Among equal timestamps the earlier-registered version stays
    /// latest.
    pub fn register_fact_versions_batch(
        &self,
        entity_id: &str,
        registrations: &[(&str, u64, &str, &str, &str, &str)],
    ) -> Result<Vec<FactVersionStatus>> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let recorded_at = unix_timestamp_ms()?;
        let mut statuses = Vec::with_capacity(registrations.len());
        {
            let mut insert = tx.prepare_cached(
                "INSERT INTO fact_versions (fact_key, memory_id, entity_id, subject, predicate, object, status, timestamp_ms, valid_from_ms, recorded_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'current', ?7, ?7, ?8)
                 ON CONFLICT(fact_key, memory_id) DO NOTHING",
            )?;
            let mut latest_stmt = tx.prepare_cached(
                "SELECT memory_id, timestamp_ms FROM fact_versions
                 WHERE fact_key = ?1 AND entity_id = ?2 AND status = 'current'",
            )?;
            let mut chain_stmt = tx.prepare_cached(
                "SELECT memory_id, timestamp_ms, COALESCE(object, '') FROM fact_versions
                 WHERE fact_key = ?1 AND entity_id = ?2
                 ORDER BY timestamp_ms ASC, rowid DESC",
            )?;
            let mut evidence = tx.prepare_cached(
                "INSERT OR IGNORE INTO fact_evidence
                     (fact_key, entity_id, version_memory_id, memory_id, timestamp_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            let mut update = tx.prepare_cached(
                "UPDATE fact_versions
                 SET status = ?1, valid_from_ms = ?2, valid_to_ms = ?3, superseded_by = ?4, supersedes = ?5
                 WHERE fact_key = ?6 AND memory_id = ?7",
            )?;

            for (fact_key, ts, memory_id, subject, predicate, object) in registrations {
                // A memory that restates the value already covering its
                // timestamp confirms that version instead of starting a new
                // one, so repeating "I live in Seattle" does not look like a
                // change of residence.
                let existing: Vec<(String, u64, String)> = chain_stmt
                    .query_map(params![fact_key, entity_id], |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, i64>(1)? as u64,
                            row.get::<_, String>(2)?,
                        ))
                    })?
                    .collect::<rusqlite::Result<_>>()?;
                let covering = existing
                    .iter()
                    .rev()
                    .find(|(_, version_ts, _)| version_ts <= ts)
                    .or_else(|| existing.first());
                if let Some((version_id, version_ts, version_object)) = covering {
                    if version_id != memory_id && same_fact_object(version_object, object) {
                        evidence.execute(params![
                            fact_key, entity_id, version_id, memory_id, *ts as i64
                        ])?;
                        statuses.push(FactVersionStatus::Confirmed {
                            version: (*version_ts, version_id.clone()),
                        });
                        continue;
                    }
                }

                let previous_latest: Option<(String, u64)> = match latest_stmt
                    .query_row(params![fact_key, entity_id], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
                    }) {
                    Ok(latest) => Some(latest),
                    Err(rusqlite::Error::QueryReturnedNoRows) => None,
                    Err(err) => return Err(err.into()),
                };

                insert.execute(params![
                    fact_key,
                    memory_id,
                    entity_id,
                    subject,
                    predicate,
                    object,
                    *ts as i64,
                    recorded_at
                ])?;

                evidence.execute(params![fact_key, entity_id, memory_id, memory_id, *ts as i64])?;

                let chain: Vec<(String, u64)> = chain_stmt
                    .query_map(params![fact_key, entity_id], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
                    })?
                    .collect::<rusqlite::Result<_>>()?;
                for (idx, (version_id, version_ts)) in chain.iter().enumerate() {
                    let next = chain.get(idx + 1);
                    let prev = idx.checked_sub(1).map(|p| &chain[p]);
                    update.execute(params![
                        if next.is_none() { "current" } else { "stale" },
                        *version_ts as i64,
                        next.map(|(_, next_ts)| *next_ts as i64),
                        next.map(|(next_id, _)| next_id.as_str()),
                        prev.map(|(prev_id, _)| prev_id.as_str()),
                        fact_key,
                        version_id,
                    ])?;
                }

                let (latest_id, latest_ts) = chain.last().expect("chain contains the new version");
                statuses.push(if latest_id == memory_id {
                    FactVersionStatus::Current {
                        superseded: previous_latest
                            .filter(|(id, _)| id != *memory_id)
                            .map(|(id, t)| (t, id)),
                    }
                } else {
                    FactVersionStatus::Stale { current: (*latest_ts, latest_id.clone()) }
                });
            }
        }
        tx.commit()?;
        Ok(statuses)
    }

    /// Direct fact lookup for pre-synthesized retrieval paths.
    /// Returns the `object` column of the most recent current row in `fact_versions`
    /// matching the given fact_key and entity_id.
    pub fn get_current_fact_value(
        &self,
        entity_id: &str,
        fact_key: &str,
    ) -> Result<Option<String>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT object FROM fact_versions
             WHERE fact_key = ?1 AND entity_id = ?2 AND status = 'current'
             ORDER BY timestamp_ms DESC LIMIT 1",
        )?;
        let res =
            stmt.query_row(params![fact_key, entity_id], |row| row.get::<_, Option<String>>(0));
        match res {
            Ok(Some(value)) => Ok(Some(value)),
            Ok(None) => Ok(None),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    // ── Core Profile ──

    pub fn get_core_profile(&self, entity_id: &str) -> Result<Option<String>> {
        let conn = self.get_conn()?;
        let mut stmt =
            conn.prepare_cached("SELECT profile_json FROM core_profiles WHERE entity_id = ?1")?;
        let res = stmt.query_row(params![entity_id], |row| row.get::<_, String>(0));
        match res {
            Ok(json) => Ok(Some(json)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Read-modify-write of one entity's core profile inside a single
    /// IMMEDIATE transaction, so concurrent ingests cannot overwrite each
    /// other's updates. `update` returns `None` to leave the profile unchanged.
    pub fn update_core_profile(
        &self,
        entity_id: &str,
        update: impl FnOnce(Option<String>) -> Option<String>,
    ) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let current = match tx.query_row(
            "SELECT profile_json FROM core_profiles WHERE entity_id = ?1",
            params![entity_id],
            |row| row.get::<_, String>(0),
        ) {
            Ok(json) => Some(json),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(err) => return Err(err.into()),
        };
        if let Some(next) = update(current) {
            tx.execute(
                "INSERT OR REPLACE INTO core_profiles (entity_id, profile_json, updated_at_ms) VALUES (?1, ?2, ?3)",
                params![entity_id, next, unix_timestamp_ms()?],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn set_core_profile(&self, entity_id: &str, profile_json: &str) -> Result<()> {
        let conn = self.get_conn()?;
        let now = unix_timestamp_ms()?;
        conn.execute(
            "INSERT OR REPLACE INTO core_profiles (entity_id, profile_json, updated_at_ms) VALUES (?1, ?2, ?3)",
            params![entity_id, profile_json, now],
        )?;
        Ok(())
    }

    // ── Deletion ──

    pub fn delete_observation(
        &self,
        _timestamp: u64,
        memory_id: &str,
        reason: &str,
    ) -> Result<DeletedObservation> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;

        // Get vector_id before deleting
        let vector_id: Option<i64> = match tx.query_row(
            "SELECT vector_id FROM vector_lookup WHERE memory_id = ?1",
            params![memory_id],
            |row| row.get(0),
        ) {
            Ok(id) => Some(id),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(err) => return Err(err.into()),
        };

        let entity_id: String = match tx.query_row(
            "SELECT entity_id FROM memories WHERE memory_id = ?1",
            params![memory_id],
            |row| row.get(0),
        ) {
            Ok(entity_id) => entity_id,
            Err(rusqlite::Error::QueryReturnedNoRows) => String::new(),
            Err(err) => return Err(err.into()),
        };

        // Create tombstone
        let tombstone_id_val = format!("tombstone::{}", memory_id);
        let now = unix_timestamp_ms()?;
        let tombstone = crate::lifecycle::DeletionTombstone {
            tombstone_id: tombstone_id_val.clone(),
            scope: "memory".to_string(),
            target_id: memory_id.to_string(),
            deleted_at_ms: now as u64,
            reason: reason.to_string(),
            cascade_count: 0,
            proof_hash: String::new(),
        };
        let tombstone_json =
            serde_json::to_string(&tombstone).context("failed to serialize deletion tombstone")?;
        tx.execute(
            "INSERT OR REPLACE INTO deletion_tombstones (tombstone_id, target_memory_id, reason, timestamp_ms, tombstone_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![tombstone_id_val, memory_id, reason, now, tombstone_json],
        )?;

        // Delete from memories and clean up FTS, centroids, and disambiguation vectors
        tx.execute("DELETE FROM memories WHERE memory_id = ?1", params![memory_id])?;
        tx.execute("DELETE FROM metrics WHERE memory_id = ?1", params![memory_id])?;
        tx.execute("DELETE FROM fts_memories WHERE rowid = ?1", params![fts_rowid(memory_id)])?;
        tx.execute("DELETE FROM negative_centroids WHERE memory_id = ?1", params![memory_id])?;
        tx.execute("DELETE FROM disambiguation_vectors WHERE memory_id = ?1", params![memory_id])?;
        tx.execute("DELETE FROM vector_lookup WHERE memory_id = ?1", params![memory_id])?;
        tx.execute("DELETE FROM memory_cards WHERE card_id = ?1", params![memory_id])?;
        tx.execute(
            "DELETE FROM memory_links WHERE source_memory_id = ?1 OR target_memory_id = ?1",
            params![memory_id],
        )?;
        // Cascade to every record derived from this memory (chunks, companions,
        // cards): their ids extend "{parent}::". Match the prefix literally
        // (LIKE would treat `%`/`_` inside the id as wildcards).
        let chunk_prefix = format!("{}::", memory_id);
        let chunks: Vec<(String, Option<i64>)> = {
            let mut stmt = tx.prepare(
                "SELECT m.memory_id, v.vector_id FROM memories m
                 LEFT JOIN vector_lookup v ON v.memory_id = m.memory_id
                 WHERE substr(m.memory_id, 1, length(?1)) = ?1",
            )?;
            let rows =
                stmt.query_map(params![chunk_prefix], |row| Ok((row.get(0)?, row.get(1)?)))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        tx.execute(
            "DELETE FROM memory_cards WHERE source_memory_id = ?1 OR substr(card_id, 1, length(?2)) = ?2",
            params![memory_id, chunk_prefix],
        )?;
        let mut chunk_vector_ids = Vec::new();
        for (chunk_id, chunk_vector_id) in &chunks {
            tx.execute("DELETE FROM fts_memories WHERE rowid = ?1", params![fts_rowid(chunk_id)])?;
            tx.execute("DELETE FROM vector_lookup WHERE memory_id = ?1", params![chunk_id])?;
            tx.execute("DELETE FROM memories WHERE memory_id = ?1", params![chunk_id])?;
            chunk_vector_ids.extend(chunk_vector_id.map(|v| v as u64));
        }

        tx.commit()?;

        Ok(DeletedObservation {
            vector_id: vector_id.map(|v| v as u64),
            chunk_vector_ids,
            entity_id,
            tombstone: Some(tombstone),
        })
    }

    // ── Turn / Ledger ──

    /// Conversation turns as stored in `memories`, keyed by the requested id.
    /// A chunked memory is reassembled from its `::cN` chunks.
    pub fn get_ledger_turns_batch(
        &self,
        turn_ids: &[String],
    ) -> Result<std::collections::HashMap<String, LedgerTurn>> {
        let mut result = std::collections::HashMap::new();
        if turn_ids.is_empty() {
            return Ok(result);
        }
        let conn = self.get_conn()?;
        let wanted: std::collections::HashSet<&str> = turn_ids.iter().map(String::as_str).collect();
        for ids in turn_ids.chunks(400) {
            let placeholders = vec!["?"; ids.len()].join(",");
            let sql = format!(
                "SELECT memory_id, parent_memory_id, entity_id, session_id, role, turn_index,
                        content, created_at_ms, content_hash
                 FROM memories
                 WHERE memory_id IN ({placeholders})
                    OR (parent_memory_id IN ({placeholders})
                        AND memory_id GLOB parent_memory_id || '::c[0-9]*')
                 ORDER BY rowid"
            );
            let mut stmt = conn.prepare_cached(&sql)?;
            let params: Vec<&dyn rusqlite::types::ToSql> =
                ids.iter().chain(ids.iter()).map(|s| s as &dyn rusqlite::types::ToSql).collect();
            let rows = stmt.query_map(params.as_slice(), memory_turn_row)?;
            for row in rows {
                let (memory_id, parent, turn) = row?;
                let key = if wanted.contains(memory_id.as_str()) {
                    memory_id
                } else {
                    match parent {
                        Some(parent) => parent,
                        None => continue,
                    }
                };
                merge_turn(&mut result, key, turn);
            }
        }
        Ok(result)
    }

    /// Source turns of a session within `radius` of `turn_index`, in order.
    pub fn get_turn_window(
        &self,
        entity_id: &str,
        session_id: &str,
        turn_index: u32,
        radius: u32,
    ) -> Result<Vec<LedgerTurn>> {
        let conn = self.get_conn()?;
        let min_idx = turn_index.saturating_sub(radius);
        let max_idx = turn_index.saturating_add(radius);
        let mut stmt = conn.prepare_cached(
            "SELECT memory_id, parent_memory_id, entity_id, session_id, role, turn_index,
                    content, created_at_ms, content_hash
             FROM memories
             WHERE entity_id = ?1 AND session_id = ?2 AND turn_index BETWEEN ?3 AND ?4
               AND (parent_memory_id IS NULL
                    OR memory_id GLOB parent_memory_id || '::c[0-9]*')
             ORDER BY turn_index, rowid",
        )?;
        let rows =
            stmt.query_map(params![entity_id, session_id, min_idx, max_idx], memory_turn_row)?;
        let mut by_turn = std::collections::HashMap::new();
        let mut order = Vec::new();
        for row in rows {
            let (memory_id, parent, turn) = row?;
            let key = parent.unwrap_or(memory_id);
            if !by_turn.contains_key(&key) {
                order.push(key.clone());
            }
            merge_turn(&mut by_turn, key, turn);
        }
        Ok(order.into_iter().filter_map(|key| by_turn.remove(&key)).collect())
    }

    // ── Deletion tombstones ──

    pub fn get_deletion_tombstones_for_target(
        &self,
        memory_id: &str,
        limit: usize,
    ) -> Result<Vec<crate::lifecycle::DeletionTombstone>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT tombstone_json FROM deletion_tombstones WHERE target_memory_id = ?1 LIMIT ?2",
        )?;
        let rows =
            stmt.query_map(params![memory_id, limit as i64], |row| row.get::<_, String>(0))?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            let json = row?;
            if let Ok(tombstone) =
                serde_json::from_str::<crate::lifecycle::DeletionTombstone>(&json)
            {
                results.push(tombstone);
            }
        }
        Ok(results)
    }

    // ── Search / Query ──

    pub fn search_memory_cards(
        &self,
        query: &MemoryCardSearchInput<'_>,
    ) -> Result<Vec<MemoryCardSearchHit>> {
        if query.limit == 0 {
            return Ok(Vec::new());
        }
        let now_ms = unix_timestamp_ms()? as u64;
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT card_id, source_memory_id, source_session_id, subject, predicate, object,
                    memory_text, card_type, confidence, is_latest, expires_at, created_at_ms
             FROM memory_cards WHERE entity_id = ?1",
        )?;
        let rows = stmt.query_map(params![query.entity_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, f64>(8)? as f32,
                row.get::<_, i32>(9)? != 0,
                row.get::<_, Option<i64>>(10)?.map(|v| v as u64),
                row.get::<_, i64>(11)? as u64,
            ))
        })?;

        let mut hits = Vec::new();
        for row in rows {
            let (
                card_id,
                source_memory_id,
                source_session_id,
                subject,
                predicate,
                object,
                memory_text,
                card_type,
                confidence,
                is_latest,
                expires_at,
                created_at_ms,
            ) = row?;

            if !query.include_stale && !is_latest {
                continue;
            }
            if expires_at.map(|exp| exp <= now_ms).unwrap_or(false) {
                continue;
            }

            let text = format!(
                "{} {} {} {} {} {}",
                subject, predicate, object, memory_text, card_type, source_session_id
            );
            let lower = text.to_ascii_lowercase();
            let lexical_hits = contains_term_count(&lower, query.lexical_terms);
            let temporal_hits = contains_term_count(&lower, query.temporal_terms);
            let entity_hits = contains_term_count(&lower, query.entities);
            let routed = query.route_sessions.contains(&source_session_id);

            if lexical_hits == 0 && temporal_hits == 0 && entity_hits == 0 && !routed {
                continue;
            }

            let lexical_coverage = if query.lexical_terms.is_empty() {
                0.0
            } else {
                lexical_hits as f32 / query.lexical_terms.len() as f32
            };
            let temporal_coverage = if query.temporal_terms.is_empty() {
                0.0
            } else {
                temporal_hits as f32 / query.temporal_terms.len() as f32
            };
            let entity_coverage = if query.entities.is_empty() {
                0.0
            } else {
                entity_hits as f32 / query.entities.len() as f32
            };

            let type_boost = match card_type.as_str() {
                "fact" => FACT_TYPE_BOOST,
                "preference" | "profile" => PREFERENCE_TYPE_BOOST,
                "event" | "episode" => EVENT_TYPE_BOOST,
                "decision" => DECISION_TYPE_BOOST,
                "inference" => INFERENCE_TYPE_BOOST,
                _ => OTHER_TYPE_BOOST,
            };
            let latest_boost = if is_latest { CARD_LATEST_BOOST } else { CARD_STALE_PENALTY };
            let route_boost = if routed { CARD_ROUTE_BOOST } else { 0.0 };
            let score = lexical_coverage.min(1.0) * CARD_LEXICAL_WEIGHT
                + temporal_coverage.min(1.0) * CARD_TEMPORAL_WEIGHT
                + entity_coverage.min(1.0) * CARD_ENTITY_WEIGHT
                + route_boost
                + type_boost
                + latest_boost
                + confidence.clamp(0.0, 1.0) * CARD_CONFIDENCE_WEIGHT;

            hits.push(MemoryCardSearchHit {
                card_id,
                source_memory_id,
                source_session_id,
                timestamp: created_at_ms,
                score,
                lexical_hits,
                temporal_hits,
                entity_hits,
            });
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.card_id.cmp(&b.card_id))
        });
        hits.truncate(query.limit);
        Ok(hits)
    }

    // ── FTS5 methods (delegated here) ──

    pub fn fts_search(
        &self,
        query: &str,
        limit: usize,
        entity_id: Option<&str>,
    ) -> Result<Vec<(String, f32)>> {
        let conn = self.get_conn()?;

        let cleaned = query.replace(|c: char| !c.is_alphanumeric() && c != ' ', " ");
        let mut terms: Vec<String> = cleaned
            .split_whitespace()
            .filter(|t| t.len() > FTS_MIN_TERM_LEN)
            .map(|t| t.to_lowercase())
            .filter(|t| !crate::api::utils::is_low_signal_keyword(t))
            .map(fts_quote)
            .collect();

        if terms.is_empty() {
            // The cleaned pass dropped everything, so fall back to the raw
            // words. These have not been stripped of punctuation, so a term
            // may contain a quote; `fts_quote` doubles it rather than letting
            // it close the phrase and make the whole expression invalid.
            terms = query
                .split_whitespace()
                .filter(|t| t.len() > FTS_MIN_TERM_LEN)
                .map(fts_quote)
                .collect();
        }

        if terms.is_empty() {
            return Ok(Vec::new());
        }

        // Terms are matched against `content` only, so an entity token can
        // never be matched by a content word that happens to look like one.
        let terms = format!("{{content}}:({})", terms.join(" OR "));
        let fts_query = match entity_id {
            Some(eid) => format!("entity_tok:{} AND {}", fts_entity_tok(eid), terms),
            None => terms,
        };

        let mut stmt = conn.prepare_cached(
            "SELECT memory_id, bm25(fts_memories) as score
             FROM fts_memories WHERE fts_memories MATCH ?1
             ORDER BY score LIMIT ?2",
        )?;
        let results = stmt
            .query_map(params![fts_query, limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(results)
    }

    pub fn fts_index_text(&self, memory_id: &str, content: &str, entity_id: &str) -> Result<()> {
        let conn = self.get_conn()?;
        conn.execute(
            "INSERT OR REPLACE INTO fts_memories (rowid, memory_id, entity_id, entity_tok, content) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                fts_rowid(memory_id),
                memory_id,
                entity_id,
                fts_entity_tok(entity_id),
                content
            ],
        )?;
        Ok(())
    }

    pub fn fts_index_batch(&self, batch: &[(String, String, String)]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO fts_memories (rowid, memory_id, entity_id, entity_tok, content) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;
            for (memory_id, entity_id, content) in batch {
                stmt.execute(params![
                    fts_rowid(memory_id),
                    memory_id,
                    entity_id,
                    fts_entity_tok(entity_id),
                    content
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn fts_remove_document(&self, memory_id: &str) -> Result<()> {
        let conn = self.get_conn()?;
        conn.execute("DELETE FROM fts_memories WHERE rowid = ?1", params![fts_rowid(memory_id)])?;
        Ok(())
    }

    pub fn fts_clear(&self) -> Result<()> {
        let conn = self.get_conn()?;
        conn.execute("DELETE FROM fts_memories", [])?;
        Ok(())
    }

    // ── Graph methods (delegated here) ──

    pub fn graph_upsert_memory_batch(&self, batch: &GraphEdgeBatch<'_>) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO edges (edge_id, source, target, edge_type, label, status, timestamp_ms, memory_id, weight)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )?;
            for entry in batch {
                // An empty node name would join every such edge into one hub.
                if entry.subject.trim().is_empty() || entry.object.trim().is_empty() {
                    continue;
                }
                let edge_id =
                    format!("edge::{}::{}::{}", entry.memory_id, entry.subject, entry.predicate);
                let label = format!("{} {} {}", entry.subject, entry.predicate, entry.object);
                let weight = crate::graph::EdgeType::from_str(entry.predicate).default_weight();
                stmt.execute(params![
                    edge_id,
                    entry.subject,
                    entry.object,
                    entry.predicate,
                    label,
                    entry.status,
                    entry.timestamp as i64,
                    entry.memory_id,
                    weight as f64,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Insert a single typed edge using owned strings (no lifetime issues).
    /// Inserts `(memory_id, subject, predicate, object, timestamp_ms)` edges
    /// in one transaction and returns how many were new; edges with an empty
    /// part are skipped.
    pub fn graph_insert_edges_batch(
        &self,
        edges: &[(&str, &str, &str, &str, u64)],
    ) -> Result<usize> {
        let mut written = 0;
        if edges.is_empty() {
            return Ok(written);
        }
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR IGNORE INTO edges (edge_id, source, target, edge_type, label, status, timestamp_ms, memory_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, 'current', ?6, ?7)",
            )?;
            for (memory_id, subject, predicate, object, timestamp_ms) in edges {
                if subject.is_empty() || predicate.is_empty() || object.is_empty() {
                    continue;
                }
                written += stmt.execute(params![
                    format!("edge::{memory_id}::{subject}::{predicate}"),
                    subject,
                    object,
                    predicate,
                    format!("{subject} {predicate} {object}"),
                    *timestamp_ms as i64,
                    memory_id
                ])?;
            }
        }
        tx.commit()?;
        Ok(written)
    }

    pub fn graph_upsert_fact_status_batch(
        &self,
        _entity_id: &str,
        batch: &GraphEdgeBatch<'_>,
    ) -> Result<()> {
        self.graph_upsert_memory_batch(batch)
    }

    pub fn graph_edge_summaries_for_label(
        &self,
        entity_id: &str,
        label: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT label FROM edges WHERE (source = ?1 OR target = ?1) AND label LIKE ?2 LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![entity_id, format!("%{}%", label), limit as i64], |row| {
                row.get::<_, String>(0)
            })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn graph_query_edges(
        &self,
        entity: &str,
        _label: Option<&str>,
        direction: &str,
        limit: usize,
    ) -> Result<Vec<GraphEdge>> {
        let conn = self.get_conn()?;
        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match direction {
            "Inbound" => (
                "SELECT edge_id, source, target, edge_type, label, weight, timestamp_ms, memory_id
                 FROM edges WHERE target = ?1 ORDER BY timestamp_ms DESC LIMIT ?2"
                    .to_string(),
                vec![Box::new(entity.to_string()), Box::new(limit as i64)],
            ),
            "Both" => (
                format!(
                    "SELECT edge_id, source, target, edge_type, label, weight, timestamp_ms, memory_id
                     FROM edges WHERE (source = ?1 OR target = ?1) AND edge_type != '{}'
                     ORDER BY timestamp_ms DESC LIMIT ?2",
                    crate::graph::EdgeType::Default.as_str()
                ),
                vec![Box::new(entity.to_string()), Box::new(limit as i64)],
            ),
            _ => (
                "SELECT edge_id, source, target, edge_type, label, weight, timestamp_ms, memory_id
                 FROM edges WHERE source = ?1 ORDER BY timestamp_ms DESC LIMIT ?2"
                    .to_string(),
                vec![Box::new(entity.to_string()), Box::new(limit as i64)],
            ),
        };
        let mut stmt = conn.prepare_cached(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok(GraphEdge {
                edge_id: row.get(0)?,
                source: row.get(1)?,
                target: row.get(2)?,
                edge_type: row.get(3)?,
                label: row.get(4)?,
                weight: row.get::<_, f64>(5)? as f32,
                timestamp_ms: row.get::<_, i64>(6)? as u64,
                memory_id: row.get(7)?,
            })
        })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn graph_remove_memory(&self, memory_id: &str) -> Result<usize> {
        let conn = self.get_conn()?;
        let count = conn.execute("DELETE FROM edges WHERE memory_id = ?1", params![memory_id])?;
        Ok(count)
    }

    pub fn graph_clear(&self) -> Result<()> {
        let conn = self.get_conn()?;
        conn.execute("DELETE FROM edges", [])?;
        Ok(())
    }

    // ── Clear / Reset ──

    /// Deletes every row in every tenant table (FTS shadow tables are
    /// cleared through their virtual table). Enumerating `sqlite_master`
    /// keeps `/reset` complete as tables are added; a hand-written list had
    /// already missed centroids, registries and three FTS indexes.
    pub fn clear_all(&self) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let tables: Vec<String> = {
            let mut stmt = tx.prepare(
                "SELECT name FROM sqlite_master
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                 AND (sql LIKE 'CREATE VIRTUAL TABLE%' OR name NOT IN (
                     SELECT m.name FROM sqlite_master v, sqlite_master m
                     WHERE v.sql LIKE 'CREATE VIRTUAL TABLE%' AND m.name LIKE v.name || '_%'
                 ))",
            )?;
            let rows = stmt.query_map([], |row| row.get(0))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        for table in tables {
            tx.execute(&format!("DELETE FROM \"{}\"", table.replace('"', "\"\"")), [])?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn db_stats(&self) -> Result<super::types::CoreClusterStats> {
        let conn = self.get_conn()?;

        let memory_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0)).unwrap_or(0);

        let entity_count: i64 = conn
            .query_row("SELECT COUNT(DISTINCT entity_id) FROM memories", [], |row| row.get(0))
            .unwrap_or(0);

        let fact_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM fact_versions WHERE status = 'current'", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);

        // Storage size in bytes
        let page_count: i64 =
            conn.query_row("PRAGMA page_count", [], |row| row.get(0)).unwrap_or(0);
        let page_size: i64 = conn.query_row("PRAGMA page_size", [], |row| row.get(0)).unwrap_or(0);
        let storage_bytes = page_count * page_size;

        Ok(super::types::CoreClusterStats {
            memory_count: memory_count as usize,
            entity_count: entity_count as usize,
            fact_count: fact_count as usize,
            storage_bytes: storage_bytes as usize,
            request_count: 0,
            ingest_count: 0,
            query_count: 0,
        })
    }

    pub fn detailed_db_stats(&self) -> Result<crate::api::types::StorageStatsResponse> {
        let conn = self.get_conn()?;
        let count = |table: &str| -> i64 {
            conn.query_row(&format!("SELECT COUNT(*) FROM {}", table), [], |row| row.get(0))
                .unwrap_or(0)
        };
        let page_count: i64 =
            conn.query_row("PRAGMA page_count", [], |row| row.get(0)).unwrap_or(0);
        let page_size: i64 = conn.query_row("PRAGMA page_size", [], |row| row.get(0)).unwrap_or(0);
        let storage_bytes = page_count * page_size;
        let free_pages: i64 =
            conn.query_row("PRAGMA freelist_count", [], |row| row.get(0)).unwrap_or(0);
        Ok(crate::api::types::StorageStatsResponse {
            used_bytes: ((page_count - free_pages).max(0) * page_size) as usize,
            memory_card_count: count("memory_cards") as usize,
            edge_count: count("edges") as usize,
            memory_count: count("memories") as usize,
            metric_count: count("metrics") as usize,
            session_router_count: count("session_router") as usize,
            fact_version_count: count("fact_versions") as usize,
            memory_link_count: count("memory_links") as usize,
            alias_count: count("aliases") as usize,
            preference_count: count("preferences") as usize,
            core_profile_count: count("core_profiles") as usize,
            deletion_tombstone_count: count("deletion_tombstones") as usize,
            storage_bytes: storage_bytes as usize,
        })
    }

    // ── Get memory cards batch ──

    /// Look up the highest-confidence, most-recent memory card for a given
    /// source memory_id. Returns the most recent `is_latest` card, or the
    /// most recent card of any kind if no `is_latest` row exists.
    pub fn get_memory_card_by_source(&self, source_memory_id: &str) -> Result<Option<MemoryCard>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT card_id, entity_id, user_id, source_memory_id, source_session_id,
                    subject, predicate, object, memory_text, card_type, confidence,
                    is_latest, is_static, is_inference, expires_at, root_card_id, parent_card_id,
                    lifecycle, created_at_ms, updated_at_ms
             FROM memory_cards WHERE source_memory_id = ?1
             ORDER BY is_latest DESC, updated_at_ms DESC LIMIT 1",
        )?;
        let res = stmt.query_row(params![source_memory_id], |row| {
            Ok(MemoryCard {
                card_id: row.get(0)?,
                entity_id: row.get(1)?,
                user_id: row.get(2)?,
                source_memory_id: row.get(3)?,
                source_session_id: row.get(4)?,
                source_turn_index: 0,
                document_time: 0,
                conversation_time: 0,
                event_time: None,
                subject: row.get(5)?,
                predicate: row.get(6)?,
                object: row.get(7)?,
                memory_text: row.get(8)?,
                card_type: row.get(9)?,
                confidence: row.get(10)?,
                is_latest: row.get::<_, i32>(11)? != 0,
                is_static: row.get::<_, i32>(12)? != 0,
                is_inference: row.get::<_, i32>(13)? != 0,
                expires_at: row.get(14)?,
                root_card_id: row.get(15)?,
                parent_card_id: row.get(16)?,
                lifecycle: row
                    .get::<_, Option<String>>(17)?
                    .and_then(|s| serde_json::from_str(&s).ok()),
                created_at_ms: row.get(18)?,
                updated_at_ms: row.get(19)?,
            })
        });
        match res {
            Ok(card) => Ok(Some(card)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn get_memory_cards_batch(
        &self,
        card_ids: &[String],
    ) -> Result<std::collections::HashMap<String, MemoryCard>> {
        if card_ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let conn = self.get_conn()?;
        let placeholders: Vec<String> = card_ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT card_id, entity_id, user_id, source_memory_id, source_session_id,
                    subject, predicate, object, memory_text, card_type, confidence,
                    is_latest, is_static, is_inference, expires_at, root_card_id, parent_card_id,
                    lifecycle, created_at_ms, updated_at_ms
             FROM memory_cards WHERE card_id IN ({})",
            placeholders.join(",")
        );
        let mut stmt = conn.prepare_cached(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            card_ids.iter().map(|s| s as &dyn rusqlite::types::ToSql).collect();
        let rows = stmt.query_map(
            param_refs.as_slice(),
            |row| -> rusqlite::Result<(String, MemoryCard)> {
                let lifecycle_str: Option<String> = row.get(17)?;
                Ok((
                    row.get::<_, String>(0)?,
                    MemoryCard {
                        card_id: row.get(0)?,
                        entity_id: row.get(1)?,
                        user_id: row.get(2)?,
                        source_memory_id: row.get(3)?,
                        source_session_id: row.get(4)?,
                        subject: row.get(5)?,
                        predicate: row.get(6)?,
                        object: row.get(7)?,
                        memory_text: row.get(8)?,
                        card_type: row.get(9)?,
                        confidence: row.get(10)?,
                        is_latest: row.get::<_, i32>(11)? != 0,
                        is_static: row.get::<_, i32>(12)? != 0,
                        is_inference: row.get::<_, i32>(13)? != 0,
                        expires_at: row.get(14)?,
                        root_card_id: row.get(15)?,
                        parent_card_id: row.get(16)?,
                        lifecycle: lifecycle_str.and_then(|s| serde_json::from_str(&s).ok()),
                        source_turn_index: 0,
                        document_time: 0,
                        conversation_time: 0,
                        event_time: None,
                        created_at_ms: row.get(18)?,
                        updated_at_ms: row.get(19)?,
                    },
                ))
            },
        )?;
        let mut results = std::collections::HashMap::new();
        for row in rows {
            let (card_id, card) = row?;
            results.insert(card_id, card);
        }
        Ok(results)
    }

    /// Which of `memory_ids` are stale fact versions now.
    pub fn invalidated_set(
        &self,
        memory_ids: &[String],
    ) -> Result<std::collections::HashSet<String>> {
        self.invalidated_among(memory_ids, "status = 'stale'", None)
    }

    /// Which of `memory_ids` were not the valid version at `point_in_time_ms`
    /// (superseded before it, or not yet recorded).
    pub fn invalidated_set_at_time(
        &self,
        point_in_time_ms: u64,
        memory_ids: &[String],
    ) -> Result<std::collections::HashSet<String>> {
        self.invalidated_among(
            memory_ids,
            "((valid_to_ms IS NOT NULL AND valid_to_ms <= ?1) OR COALESCE(valid_from_ms, 0) > ?1)",
            Some(point_in_time_ms),
        )
    }

    /// Checks only the candidates a query is scoring instead of loading every
    /// stale version in the tenant on each query.
    fn invalidated_among(
        &self,
        memory_ids: &[String],
        condition: &str,
        point_in_time_ms: Option<u64>,
    ) -> Result<std::collections::HashSet<String>> {
        const CHUNK: usize = 500;
        let mut set = std::collections::HashSet::new();
        if memory_ids.is_empty() {
            return Ok(set);
        }
        let conn = self.get_conn()?;
        for chunk in memory_ids.chunks(CHUNK) {
            let first_id_param = if point_in_time_ms.is_some() { 2 } else { 1 };
            let placeholders: Vec<String> =
                (0..chunk.len()).map(|i| format!("?{}", i + first_id_param)).collect();
            let sql = format!(
                "SELECT DISTINCT memory_id FROM fact_versions WHERE {condition} AND memory_id IN ({})",
                placeholders.join(",")
            );
            let mut stmt = conn.prepare_cached(&sql)?;
            let mut values: Vec<rusqlite::types::Value> = Vec::with_capacity(chunk.len() + 1);
            if let Some(pit) = point_in_time_ms {
                values.push(rusqlite::types::Value::Integer(pit as i64));
            }
            values.extend(chunk.iter().map(|m| rusqlite::types::Value::Text(m.clone())));
            let rows =
                stmt.query_map(rusqlite::params_from_iter(values), |row| row.get::<_, String>(0))?;
            for row in rows {
                set.insert(row?);
            }
        }
        Ok(set)
    }
}

fn dedupe_append<T: Clone + PartialEq + Eq + std::hash::Hash>(base: &[T], extra: &[T]) -> Vec<T> {
    let mut seen: std::collections::HashSet<&T> = base.iter().collect();
    let mut result: Vec<T> = base.to_vec();
    for item in extra {
        if seen.insert(item) {
            result.push(item.clone());
        }
    }
    result
}

fn merge_router_records(
    existing: &SessionRouterRecord,
    incoming: &SessionRouterRecord,
) -> SessionRouterRecord {
    let mut merged = existing.clone();
    merged.canonical_facts = dedupe_append(&merged.canonical_facts, &incoming.canonical_facts);
    merged.events = dedupe_append(&merged.events, &incoming.events);
    merged.source_memory_ids =
        dedupe_append(&merged.source_memory_ids, &incoming.source_memory_ids);
    merged.persons = dedupe_append(&merged.persons, &incoming.persons);
    merged.speakers = dedupe_append(&merged.speakers, &incoming.speakers);
    merged.salient_terms = dedupe_append(&merged.salient_terms, &incoming.salient_terms);
    merged.objects = dedupe_append(&merged.objects, &incoming.objects);
    merged.places = dedupe_append(&merged.places, &incoming.places);
    merged.activities = dedupe_append(&merged.activities, &incoming.activities);
    merged.preference_signals =
        dedupe_append(&merged.preference_signals, &incoming.preference_signals);
    merged.router_text = build_session_router_text(&merged);
    merged.updated_at_ms = std::cmp::max(merged.updated_at_ms, incoming.updated_at_ms);
    merged.session_focus = if incoming.session_focus.is_empty() {
        merged.session_focus
    } else {
        incoming.session_focus.clone()
    };
    merged.session_date = if incoming.session_date.is_empty() || incoming.session_date == "unknown"
    {
        merged.session_date
    } else {
        incoming.session_date.clone()
    };
    merged
}

fn contains_term_count(lower_haystack: &str, terms: &[String]) -> usize {
    terms
        .iter()
        .filter(|term| {
            let needle = term.trim().to_ascii_lowercase();
            !needle.is_empty() && lower_haystack.contains(needle.as_str())
        })
        .count()
}

fn parse_kind_enum(kind: &str) -> MemoryKind {
    if kind.contains("Preference") {
        MemoryKind::Preference
    } else if kind.contains("Decision") {
        MemoryKind::Decision
    } else if kind.contains("Lesson") {
        MemoryKind::Lesson
    } else if kind.contains("Fact") {
        MemoryKind::Fact
    } else if kind.contains("SessionSummary") {
        MemoryKind::SessionSummary
    } else {
        MemoryKind::Conversational
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub edge_id: String,
    pub source: String,
    pub target: String,
    pub edge_type: String,
    pub label: String,
    pub weight: f32,
    pub timestamp_ms: u64,
    pub memory_id: String,
}

impl TenantStore {
    pub fn get_all_edges(&self, limit: usize) -> Result<Vec<GraphEdge>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT edge_id, source, target, edge_type, label, weight, timestamp_ms, memory_id FROM edges ORDER BY weight DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit as i64], |row| {
            Ok(GraphEdge {
                edge_id: row.get(0)?,
                source: row.get(1)?,
                target: row.get(2)?,
                edge_type: row.get(3)?,
                label: row.get(4)?,
                weight: row.get(5)?,
                timestamp_ms: row.get(6)?,
                memory_id: row.get(7)?,
            })
        })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn expire_records(&self, now_ms: u64) -> Result<usize> {
        let mut conn = self.get_conn()?;
        let mut updates = Vec::new();
        {
            let mut stmt = conn.prepare_cached(
                "SELECT card_id, lifecycle FROM memory_cards WHERE expires_at IS NOT NULL AND expires_at <= ?1"
            )?;
            let rows = stmt.query_map(params![now_ms as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;

            for row in rows {
                let (card_id, lifecycle_json) = row?;
                if let Ok(mut lifecycle) =
                    serde_json::from_str::<crate::lifecycle::LifecycleMetadata>(&lifecycle_json)
                {
                    if lifecycle.lifecycle_state != crate::lifecycle::LifecycleState::Expired
                        && matches!(
                            lifecycle.retention_class,
                            crate::lifecycle::RetentionClass::Ephemeral
                                | crate::lifecycle::RetentionClass::Working
                        )
                    {
                        lifecycle.lifecycle_state = crate::lifecycle::LifecycleState::Expired;
                        if let Ok(updated_json) = serde_json::to_string(&lifecycle) {
                            updates.push((card_id, updated_json));
                        }
                    }
                }
            }
        }

        let count = updates.len();
        if count > 0 {
            let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            {
                let mut update_stmt = tx.prepare_cached(
                    "UPDATE memory_cards SET lifecycle = ?1, updated_at_ms = ?2 WHERE card_id = ?3",
                )?;
                for (card_id, updated_json) in updates {
                    update_stmt.execute(params![updated_json, now_ms as i64, card_id])?;
                }
            }
            tx.commit()?;
        }
        Ok(count)
    }
}

// Re-import needed for artifact versions
use serde::{Deserialize, Serialize};

fn vec_f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes
}

fn bytes_to_vec_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Reads per-entity embeddings from `vector_lookup` for the vector index.
struct SqliteVectorSource {
    pool: Pool<SqliteConnectionManager>,
}

impl crate::vector_index::VectorSource for SqliteVectorSource {
    fn entity_vectors(&self, entity_id: &str) -> Result<Vec<(u64, Vec<f32>)>> {
        let conn = self.pool.get().context("failed to get connection")?;
        let mut stmt = conn.prepare_cached(
            "SELECT vector_id, embedding FROM vector_lookup
             WHERE entity_id = ?1 AND embedding IS NOT NULL ORDER BY vector_id",
        )?;
        let rows = stmt.query_map(params![entity_id], |row| {
            Ok((row.get::<_, i64>(0)? as u64, bytes_to_vec_f32(&row.get::<_, Vec<u8>>(1)?)))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    fn entities(&self) -> Result<Vec<String>> {
        let conn = self.pool.get().context("failed to get connection")?;
        let mut stmt = conn.prepare_cached(
            "SELECT DISTINCT entity_id FROM vector_lookup WHERE embedding IS NOT NULL ORDER BY entity_id",
        )?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    fn vectors_by_id(&self, ids: &[u64]) -> Result<HashMap<u64, Vec<f32>>> {
        let conn = self.pool.get().context("failed to get connection")?;
        let mut out = HashMap::with_capacity(ids.len());
        for chunk in ids.chunks(500) {
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT vector_id, embedding FROM vector_lookup
                 WHERE embedding IS NOT NULL AND vector_id IN ({})",
                vec!["?"; chunk.len()].join(",")
            ))?;
            let params: Vec<i64> = chunk.iter().map(|id| *id as i64).collect();
            let rows = stmt.query_map(rusqlite::params_from_iter(params), |row| {
                Ok((row.get::<_, i64>(0)? as u64, bytes_to_vec_f32(&row.get::<_, Vec<u8>>(1)?)))
            })?;
            for row in rows {
                let (id, vector) = row?;
                out.insert(id, vector);
            }
        }
        Ok(out)
    }
}

/// Whether two fact objects state the same value (whitespace, case and
/// trailing punctuation ignored).
fn same_fact_object(a: &str, b: &str) -> bool {
    let normalize = |text: &str| {
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .trim_matches(|c: char| c.is_ascii_punctuation())
            .to_ascii_lowercase()
    };
    !a.is_empty() && normalize(a) == normalize(b)
}

/// `(memory_id, weight, edge_type)` reached through a shared graph node.
pub type EdgeNeighbour = (String, f32, String);

type MemoryTurnRow = (String, Option<String>, LedgerTurn);

fn memory_turn_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryTurnRow> {
    let memory_id: String = row.get(0)?;
    let parent: Option<String> = row.get(1)?;
    let role: String = row.get(4)?;
    let created_at_ms = row.get::<_, i64>(7)? as u64;
    let turn = LedgerTurn {
        turn_id: parent.clone().unwrap_or_else(|| memory_id.clone()),
        entity_id: row.get(2)?,
        session_id: row.get(3)?,
        speaker: (!role.is_empty()).then_some(role),
        turn_index: row.get::<_, i64>(5)? as u32,
        raw_text: row.get(6)?,
        document_time_ms: created_at_ms,
        ingest_time_ms: created_at_ms,
        source_type: "memory".to_string(),
        source_uri: None,
        raw_sha256: row.get(8)?,
        redaction_state: "none".to_string(),
        lifecycle: None,
        schema_version: 2,
    };
    Ok((memory_id, parent, turn))
}

/// Adds a row to its turn; chunks of one memory are joined in rowid order.
fn merge_turn(
    turns: &mut std::collections::HashMap<String, LedgerTurn>,
    key: String,
    turn: LedgerTurn,
) {
    match turns.entry(key) {
        std::collections::hash_map::Entry::Occupied(mut existing) => {
            let existing = existing.get_mut();
            existing.raw_text.push('\n');
            existing.raw_text.push_str(&turn.raw_text);
        }
        std::collections::hash_map::Entry::Vacant(slot) => {
            let turn_id = slot.key().clone();
            slot.insert(LedgerTurn { turn_id, ..turn });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn turn_obs(
        session: &str,
        turn: u32,
        role: &str,
        text: &str,
        parent: Option<&str>,
    ) -> AgentObservation {
        AgentObservation {
            entity_id: "alice".into(),
            textual_content: text.into(),
            created_at_ms: 1_000 + u64::from(turn),
            session_id: session.into(),
            turn_index: turn,
            role: role.into(),
            parent_memory_id: parent.map(str::to_string),
            ..Default::default()
        }
    }

    fn register(
        store: &TenantStore,
        entity: &str,
        items: &[(&str, u64, &str, &str)],
    ) -> Vec<FactVersionStatus> {
        let registrations: Vec<(&str, u64, &str, &str, &str, &str)> = items
            .iter()
            .map(|(key, ts, memory_id, object)| (*key, *ts, *memory_id, entity, *key, *object))
            .collect();
        store.register_fact_versions_batch(entity, &registrations).unwrap()
    }

    #[test]
    fn restatements_become_evidence_instead_of_new_versions() {
        let dir = tempfile::tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("t.db")).unwrap();
        let statuses = register(
            &store,
            "alice",
            &[
                ("residence", 100, "m1", "Austin"),
                // Same value restated, with different spacing and case.
                ("residence", 150, "m2", " austin "),
                ("residence", 200, "m3", "Seattle"),
            ],
        );
        assert!(matches!(statuses[0], FactVersionStatus::Current { superseded: None }));
        assert!(
            matches!(&statuses[1], FactVersionStatus::Confirmed { version } if version.1 == "m1")
        );
        assert!(
            matches!(&statuses[2], FactVersionStatus::Current { superseded: Some((_, id)) } if id == "m1")
        );

        let rows =
            store.fact_versions_for_memories(&["m1".into(), "m2".into(), "m3".into()]).unwrap();
        // The restatement has no version row of its own.
        assert!(!rows.contains_key("m2"));
        let stale = &rows["m1"];
        assert!(!stale.is_current);
        assert_eq!(stale.object, "Austin");
        assert_eq!(stale.superseded_by.as_deref(), Some("m3"));
        assert_eq!(stale.current_object.as_deref(), Some("Seattle"));
        assert_eq!(stale.superseded_at_ms, Some(200));
        assert_eq!(stale.evidence, vec!["m2".to_string(), "m1".to_string()]);
        assert_eq!(stale.superseded_by.as_deref(), Some("m3"), "no derived parent to map to");
        assert!(rows["m3"].is_current);

        // A turn is matched through its derived records: the fact lives on
        // `turn::card0`, but asking for the turn finds it.
        store
            .insert_observations_batch(&[(
                300,
                "turn::card0".to_string(),
                AgentObservation {
                    entity_id: "alice".into(),
                    textual_content: "Atomic memory card: I work at Globex".into(),
                    parent_memory_id: Some("turn".into()),
                    ..Default::default()
                },
            )])
            .unwrap();
        register(&store, "alice", &[("employer", 300, "turn::card0", "Globex")]);
        let via_turn = store.fact_versions_for_memories(&["turn".into()]).unwrap();
        assert_eq!(via_turn["turn"].object, "Globex");
        assert!(via_turn["turn"].is_current);
        assert_eq!(
            store.get_current_fact_value("alice", "residence").unwrap().as_deref(),
            Some("Seattle")
        );
    }

    #[test]
    fn predicate_variants_join_one_group() {
        let dir = tempfile::tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("t.db")).unwrap();
        // "job title" and "job_title" get near-identical embeddings; "pet
        // name" is unrelated.
        let job = vec![1.0, 0.0, 0.0];
        let job_variant = vec![0.99, 0.10, 0.0];
        let pet = vec![0.0, 1.0, 0.0];
        let assigned = store
            .canonicalize_predicates(
                "alice",
                &[
                    ("job title".to_string(), job),
                    ("job_title".to_string(), job_variant.clone()),
                    ("pet name".to_string(), pet),
                ],
                0.86,
            )
            .unwrap();
        assert_eq!(assigned["job title"], "job title");
        assert_eq!(assigned["job_title"], "job title", "variant joins the first group");
        assert_eq!(assigned["pet name"], "pet name");

        // Assignments are stable across calls, even with a different vector.
        let again = store
            .canonicalize_predicates(
                "alice",
                &[("job_title".to_string(), vec![0.0, 0.0, 1.0])],
                0.86,
            )
            .unwrap();
        assert_eq!(again["job_title"], "job title");
        // Another entity groups independently.
        let other = store
            .canonicalize_predicates("bob", &[("job_title".to_string(), job_variant)], 0.86)
            .unwrap();
        assert_eq!(other["job_title"], "job_title");
    }

    #[test]
    fn parent_lookups_do_not_scan_the_memories_table() {
        let dir = tempfile::tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("t.db")).unwrap();
        let conn = store.get_conn().unwrap();
        // `get_ledger_turns_batch` reaches derived records through their
        // parent; a full scan here is a latency cliff at real corpus sizes.
        let plan: Vec<String> = conn
            .prepare(
                "EXPLAIN QUERY PLAN SELECT memory_id FROM memories
                 WHERE memory_id IN (?1)
                    OR (parent_memory_id IN (?1)
                        AND memory_id GLOB parent_memory_id || '::c[0-9]*')",
            )
            .unwrap()
            .query_map(rusqlite::params!["x"], |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        let plan = plan.join(" | ");
        assert!(
            plan.contains("idx_memories_parent"),
            "parent lookup must use the index, got: {plan}"
        );
        assert!(!plan.contains("SCAN memories"), "plan still scans: {plan}");
    }

    #[test]
    fn turn_window_reads_stored_turns_and_joins_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("t.db")).unwrap();
        let items = vec![
            (1_000, "m0".to_string(), turn_obs("s", 0, "user", "hello", None)),
            (1_001, "m1::c0".to_string(), turn_obs("s", 1, "assistant", "part one", Some("m1"))),
            (1_001, "m1::c1".to_string(), turn_obs("s", 1, "assistant", "part two", Some("m1"))),
            (1_001, "m1::gist".to_string(), turn_obs("s", 1, "", "a gist", Some("m1"))),
            (1_002, "m2".to_string(), turn_obs("s", 2, "user", "bye", None)),
            (1_003, "other".to_string(), turn_obs("t", 1, "user", "elsewhere", None)),
        ];
        store.insert_observations_batch(&items).unwrap();

        let window = store.get_turn_window("alice", "s", 1, 1).unwrap();
        let summary: Vec<(String, u32, Option<String>, String)> =
            window.into_iter().map(|t| (t.turn_id, t.turn_index, t.speaker, t.raw_text)).collect();
        assert_eq!(
            summary,
            vec![
                ("m0".into(), 0, Some("user".into()), "hello".into()),
                ("m1".into(), 1, Some("assistant".into()), "part one\npart two".into()),
                ("m2".into(), 2, Some("user".into()), "bye".into()),
            ]
        );

        let by_id =
            store.get_ledger_turns_batch(&["m1".into(), "m2".into(), "nope".into()]).unwrap();
        assert_eq!(by_id["m1"].raw_text, "part one\npart two");
        assert_eq!(by_id["m2"].session_id, "s");
        assert!(!by_id.contains_key("nope"));
    }

    #[test]
    fn batched_edge_neighbours_match_per_memory_query() {
        let dir = tempfile::tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("t.db")).unwrap();
        // Deterministic pseudo-random graph with a hub ("hub") and repeats.
        let names = ["hub", "a", "b", "c", "d", "e", ""];
        let mut state = 7u64;
        let mut next = |m: u64| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        let mut owned = Vec::new();
        for i in 0..160 {
            let memory = format!("m{}", next(40));
            let source =
                if i % 3 == 0 { "hub".to_string() } else { names[next(7) as usize].to_string() };
            let target = names[next(7) as usize].to_string();
            let predicate = ["p", "q"][next(2) as usize].to_string();
            owned.push((memory, source, predicate, target, i as u64));
        }
        {
            let conn = store.get_conn().unwrap();
            for (i, (m, s, p, t, ts)) in owned.iter().enumerate() {
                conn.execute(
                    "INSERT INTO edges (edge_id, source, target, edge_type, weight, timestamp_ms, memory_id)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    rusqlite::params![format!("e{i}"), s, t, p, 1.0 + (i % 4) as f64, *ts as i64, m],
                )
                .unwrap();
            }
        }
        let ids: Vec<String> = (0..42).map(|i| format!("m{i}")).collect();
        for (filter, limit, degree) in [(None, 50, 40), (Some("p"), 7, 40), (None, 1000, 1000)] {
            let batch =
                store.get_edge_cluster_neighbors_batch(&ids, filter, limit, degree).unwrap();
            for id in &ids {
                let single =
                    store.get_edge_cluster_neighbors_typed(id, filter, limit, degree).unwrap();
                let key = |rows: &[(String, f32, String)]| {
                    let mut v: Vec<(String, i64)> =
                        rows.iter().map(|(m, w, _)| (m.clone(), (*w * 100.0) as i64)).collect();
                    v.sort();
                    v
                };
                assert_eq!(
                    key(&batch[id]),
                    key(&single),
                    "memory {id} filter {filter:?} limit {limit}"
                );
            }
        }
    }

    #[test]
    fn edge_batch_inserts_once_and_skips_incomplete_edges() {
        let dir = tempfile::tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("t.db")).unwrap();
        let edges = [("m1", "alice", "lives_in", "Denver", 5), ("m1", "alice", "", "x", 5)];
        assert_eq!(store.graph_insert_edges_batch(&edges).unwrap(), 1);
        assert_eq!(store.graph_insert_edges_batch(&edges).unwrap(), 0);
    }

    #[test]
    fn test_tenant_memory_upsert_rowid_stability() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("tenant.db");
        let store = TenantStore::new(&db_path).unwrap();

        let obs1 = AgentObservation {
            entity_id: "user-123".to_string(),
            textual_content: "Sharjeel is developing AletheiaDB".to_string(),
            embedding: vec![1.0, 2.0, 3.0],
            kind: MemoryKind::Fact,
            content_hash: String::new(),
            created_at_ms: 1000,
            ..Default::default()
        };

        // Ingest first time
        let ids1 = store.insert_observations_batch(&[(1000, "mem-001".to_string(), obs1)]).unwrap();
        assert_eq!(ids1.len(), 1);
        let rid1 = ids1[0].unwrap();

        // Ingest update to the same memory_id
        let obs2 = AgentObservation {
            entity_id: "user-123".to_string(),
            textual_content: "Sharjeel is developing AletheiaDB in Rust".to_string(),
            embedding: vec![4.0, 5.0, 6.0],
            kind: MemoryKind::Fact,
            content_hash: String::new(),
            created_at_ms: 2000,
            ..Default::default()
        };

        let ids2 = store.insert_observations_batch(&[(2000, "mem-001".to_string(), obs2)]).unwrap();
        assert_eq!(ids2.len(), 1);
        let rid2 = ids2[0].unwrap();

        // Verify the rowid remains the same (stability!)
        assert_eq!(rid1, rid2);

        // Verify contents are updated in the SQLite table
        let conn = store.get_conn().unwrap();
        let (content, created_at): (String, u64) = conn
            .query_row(
                "SELECT content, created_at_ms FROM memories WHERE rowid = ?1",
                params![rid1 as i64],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(content, "Sharjeel is developing AletheiaDB in Rust");
        assert_eq!(created_at, 2000);

        // Verify vector_lookup is updated
        let (mem_id, ts): (String, u64) = conn
            .query_row(
                "SELECT memory_id, timestamp_ms FROM vector_lookup WHERE vector_id = ?1",
                params![rid1 as i64],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();

        assert_eq!(mem_id, "mem-001");
        assert_eq!(ts, 2000);
    }

    #[test]
    fn test_fact_versions_point_in_time() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("tenant.db");
        let store = TenantStore::new(&db_path).unwrap();

        let registrations1 =
            vec![("fact_key_1", 100, "mem-100", "Caroline", "prefers", "counseling")];
        let statuses1 = store.register_fact_versions_batch("Caroline", &registrations1).unwrap();
        assert_eq!(statuses1.len(), 1);

        let registrations2 =
            vec![("fact_key_1", 200, "mem-200", "Caroline", "prefers", "coaching")];
        let statuses2 = store.register_fact_versions_batch("Caroline", &registrations2).unwrap();
        assert_eq!(statuses2.len(), 1);

        let ids = vec!["mem-100".to_string(), "mem-200".to_string()];
        let stale_at_150 = store.invalidated_set_at_time(150, &ids).unwrap();
        assert!(stale_at_150.contains("mem-200"));
        assert!(!stale_at_150.contains("mem-100"));

        let stale_at_250 = store.invalidated_set_at_time(250, &ids).unwrap();
        assert!(stale_at_250.contains("mem-100"));
        assert!(!stale_at_250.contains("mem-200"));
    }

    #[test]
    fn test_get_edge_cluster_neighbors() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("tenant.db");
        let store = TenantStore::new(&db_path).unwrap();

        let entry1 = GraphEdgeEntry {
            memory_id: "mem-1",
            subject: "Caroline",
            predicate: "caused_by",
            object: "stress",
            status: "current",
            ref_info: None,
            timestamp: 100,
        };
        let entry2 = GraphEdgeEntry {
            memory_id: "mem-2",
            subject: "stress",
            predicate: "leads_to",
            object: "counseling",
            status: "current",
            ref_info: None,
            timestamp: 100,
        };
        store.graph_upsert_memory_batch(&[entry1, entry2]).unwrap();

        let neighbors = store.get_edge_cluster_neighbors("mem-1", None, 10).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, "mem-2");
        assert!((neighbors[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bitemporal_preference_supersession() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("tenant.db");
        let store = TenantStore::new(&db_path).unwrap();

        let registrations1 =
            vec![("pref_key_1", 100, "mem-pref-1", "Caroline", "prefers", "counseling")];
        let statuses1 = store.register_fact_versions_batch("Caroline", &registrations1).unwrap();
        assert_eq!(statuses1.len(), 1);

        let registrations2 =
            vec![("pref_key_1", 200, "mem-pref-2", "Caroline", "prefers", "coaching")];
        let statuses2 = store.register_fact_versions_batch("Caroline", &registrations2).unwrap();
        assert_eq!(statuses2.len(), 1);

        let ids = vec!["mem-pref-1".to_string(), "mem-pref-2".to_string()];
        let stale_at_250 = store.invalidated_set_at_time(250, &ids).unwrap();
        assert!(stale_at_250.contains("mem-pref-1"));
        assert!(!stale_at_250.contains("mem-pref-2"));
    }

    #[test]
    fn test_lifecycle_expiration_sweeper() {
        let temp = tempdir().unwrap();
        let db_path = temp.path().join("tenant.db");
        let store = TenantStore::new(&db_path).unwrap();

        // Create an ephemeral memory card expiring at timestamp 1000
        let mut lifecycle = crate::lifecycle::evaluate_lifecycle(
            "ephemeral test memory",
            MemoryKind::Conversational,
            100,
            None,
            true,
        );
        lifecycle.expires_at_ms = Some(1000);
        assert_eq!(lifecycle.retention_class, crate::lifecycle::RetentionClass::Ephemeral);

        let card = MemoryCard {
            card_id: "card-123".to_string(),
            entity_id: "user-123".to_string(),
            user_id: "user-123".to_string(),
            source_memory_id: "mem-123".to_string(),
            source_session_id: "session-123".to_string(),
            subject: "subject".to_string(),
            predicate: "predicate".to_string(),
            object: "object".to_string(),
            memory_text: "text".to_string(),
            card_type: "Fact".to_string(),
            confidence: 0.9,
            is_latest: true,
            is_static: false,
            is_inference: true,
            expires_at: Some(1000),
            root_card_id: None,
            parent_card_id: None,
            lifecycle: Some(lifecycle),
            source_turn_index: 0,
            document_time: 100,
            conversation_time: 100,
            event_time: None,
            created_at_ms: 100,
            updated_at_ms: 100,
        };

        store.ingest_cards(std::slice::from_ref(&card)).unwrap();

        // Sweep at time 500 (card has not expired)
        let swept = store.expire_records(500).unwrap();
        assert_eq!(swept, 0);

        let card_loaded = store.get_memory_card("card-123").unwrap().unwrap();
        assert_ne!(
            card_loaded.lifecycle.unwrap().lifecycle_state,
            crate::lifecycle::LifecycleState::Expired
        );

        // Sweep at time 1500 (card has expired)
        let swept = store.expire_records(1500).unwrap();
        assert_eq!(swept, 1);

        let card_loaded_after = store.get_memory_card("card-123").unwrap().unwrap();
        assert_eq!(
            card_loaded_after.lifecycle.unwrap().lifecycle_state,
            crate::lifecycle::LifecycleState::Expired
        );

        // Sweep again (already marked Expired, should not be returned again)
        let swept = store.expire_records(1500).unwrap();
        assert_eq!(swept, 0);
    }

    fn fact_chain(store: &TenantStore, fact_key: &str) -> Vec<(String, String, u64, Option<u64>)> {
        let conn = store.get_conn().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT memory_id, status, valid_from_ms, valid_to_ms FROM fact_versions
                 WHERE fact_key = ?1 ORDER BY valid_from_ms, rowid DESC",
            )
            .unwrap();
        stmt.query_map(params![fact_key], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get::<_, i64>(2)? as u64,
                row.get::<_, Option<i64>>(3)?.map(|v| v as u64),
            ))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
    }

    #[test]
    fn out_of_order_fact_versions_form_a_consistent_chain() {
        let temp = tempdir().unwrap();
        let store = TenantStore::new(&temp.path().join("tenant.db")).unwrap();
        let register = |ts: u64, id: &str, city: &str| {
            store
                .register_fact_versions_batch(
                    "user",
                    &[("residence", ts, id, "user", "lives_in", city)],
                )
                .unwrap()
                .remove(0)
        };

        assert!(matches!(
            register(100, "m100", "Austin"),
            FactVersionStatus::Current { superseded: None }
        ));
        assert!(matches!(register(50, "m50", "Boston"), FactVersionStatus::Stale { .. }));
        assert!(matches!(register(75, "m75", "Denver"), FactVersionStatus::Stale { .. }));

        let chain = fact_chain(&store, "residence");
        assert_eq!(
            chain,
            vec![
                ("m50".into(), "stale".into(), 50, Some(75)),
                ("m75".into(), "stale".into(), 75, Some(100)),
                ("m100".into(), "current".into(), 100, None),
            ]
        );

        // As of t=80 only the t=75 version is valid (it used to overlap with t=50).
        let ids: Vec<String> = ["m50", "m75", "m100"].iter().map(|s| s.to_string()).collect();
        let invalid = store.invalidated_set_at_time(80, &ids).unwrap();
        assert!(invalid.contains("m50") && invalid.contains("m100") && !invalid.contains("m75"));

        match register(150, "m150", "Seattle") {
            FactVersionStatus::Current { superseded: Some((ts, id)) } => {
                assert_eq!((ts, id.as_str()), (100, "m100"));
            }
            other => panic!("unexpected status {other:?}"),
        }
        // Re-registering an existing version is idempotent.
        assert!(matches!(
            register(150, "m150", "Seattle"),
            FactVersionStatus::Current { superseded: None }
        ));
        assert_eq!(fact_chain(&store, "residence").len(), 4);
    }

    proptest::proptest! {
        #[test]
        fn fact_chain_invariants_hold_for_any_insertion_order(
            timestamps in proptest::collection::vec(0u64..50, 1..8)
        ) {
            let temp = tempdir().unwrap();
            let store = TenantStore::new(&temp.path().join("tenant.db")).unwrap();
            // Distinct objects: equal values are merged as evidence for the
            // version they restate (covered by its own test).
            for (i, ts) in timestamps.iter().enumerate() {
                let id = format!("m{i}");
                let object = format!("o{i}");
                store
                    .register_fact_versions_batch(
                        "e",
                        &[("k", *ts, id.as_str(), "e", "p", object.as_str())],
                    )
                    .unwrap();
            }
            let chain = fact_chain(&store, "k");
            proptest::prop_assert_eq!(chain.len(), timestamps.len());
            proptest::prop_assert_eq!(chain.iter().filter(|c| c.1 == "current").count(), 1);
            let last = chain.last().unwrap();
            proptest::prop_assert_eq!(last.1.as_str(), "current");
            proptest::prop_assert_eq!(last.2, *timestamps.iter().max().unwrap());
            proptest::prop_assert_eq!(last.3, None);
            for pair in chain.windows(2) {
                // Each version ends exactly where the next begins.
                proptest::prop_assert_eq!(pair[0].3, Some(pair[1].2));
                proptest::prop_assert!(pair[0].2 <= pair[1].2);
            }
        }
    }

    #[test]
    fn link_cluster_counts_bidirectional_link_once() {
        let temp = tempdir().unwrap();
        let store = TenantStore::new(&temp.path().join("tenant.db")).unwrap();
        store
            .set_memory_links_batch(&[
                ("a".into(), "b".into(), "derived_from".into()),
                ("b".into(), "a".into(), "derived_variant".into()),
            ])
            .unwrap();
        let scores = store.get_link_cluster_scores("a", 1).unwrap();
        assert!((scores["b"] - 0.6).abs() < 1e-6, "got {:?}", scores);
    }

    fn fts_store(dir: &tempfile::TempDir) -> TenantStore {
        let store = TenantStore::new(&dir.path().join("t.db")).unwrap();
        // Two entities with the same words, so a leaking entity filter shows
        // up as another entity's rows rather than as a missing result.
        let batch: Vec<(String, String, String)> = vec![
            ("alice::s1::0", "alice", "the garden plan for spring"),
            ("alice::s1::1", "alice", "a recipe for bread"),
            ("bob::s1::0", "bob", "the garden plan for spring"),
            ("bob::s1::1", "bob", "a recipe for bread"),
        ]
        .into_iter()
        .map(|(m, e, c)| (m.to_string(), e.to_string(), c.to_string()))
        .collect();
        store.fts_index_batch(&batch).unwrap();
        store
    }

    #[test]
    fn fts_search_is_scoped_to_one_entity() {
        let dir = tempfile::tempdir().unwrap();
        let store = fts_store(&dir);
        let hits = store.fts_search("garden plan", 10, Some("alice")).unwrap();
        assert!(!hits.is_empty(), "expected alice's rows");
        assert!(
            hits.iter().all(|(id, _)| id.starts_with("alice::")),
            "entity filter leaked: {hits:?}"
        );
        // Unscoped still spans both entities.
        let all = store.fts_search("garden plan", 10, None).unwrap();
        assert!(all.iter().any(|(id, _)| id.starts_with("bob::")));
    }

    #[test]
    fn entity_token_cannot_be_matched_by_content() {
        let dir = tempfile::tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("t.db")).unwrap();
        let tok = fts_entity_tok("alice");
        // A memory whose *text* is another entity's token must not be
        // returned when searching that entity.
        store.fts_index_text("bob::s1::0", &tok, "bob").unwrap();
        store.fts_index_text("alice::s1::0", "unrelated words", "alice").unwrap();
        let hits = store.fts_search(&tok, 10, Some("alice")).unwrap();
        assert!(
            hits.iter().all(|(id, _)| id.starts_with("alice::")),
            "content matched an entity token: {hits:?}"
        );
    }

    #[test]
    fn quoted_query_does_not_break_the_match_expression() {
        let dir = tempfile::tempdir().unwrap();
        let store = fts_store(&dir);
        // `a"b` cleans to two one-character terms, so both are dropped and
        // the raw-word fallback runs. That fallback used to wrap the word in
        // quotes without escaping, leaving an odd number of quotes: FTS5
        // rejects the expression and the caller's `unwrap_or_default` turns
        // the error into a silently empty lane.
        let hits = store.fts_search("a\"b", 10, Some("alice"));
        assert!(hits.is_ok(), "odd-quote query errored: {:?}", hits.err());
    }

    #[test]
    fn entity_tokens_are_single_tokens_and_distinct() {
        assert_eq!(fts_entity_tok("ab"), "e6162");
        assert_ne!(fts_entity_tok("alice"), fts_entity_tok("bob"));
        // Ids that tokenize differently must not collapse to one token.
        assert_ne!(fts_entity_tok("a b"), fts_entity_tok("ab"));
        assert!(fts_entity_tok("a b/c-d").chars().all(|c| c.is_ascii_alphanumeric()));
    }
}
