use anyhow::{Context, Result};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use super::types::*;

type GraphEdgeBatch<'a> = [GraphEdgeEntry<'a>];

const PRAGMA_CACHE_SIZE: i64 = -262144;
const PRAGMA_MMAP_SIZE: i64 = 1073741824;
const PRAGMA_BUSY_TIMEOUT: i64 = 10000;
const PRAGMA_PAGE_SIZE: i64 = 8192;

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
}

impl TenantStore {
    pub fn new(path: &Path) -> Result<Self> {
        let manager = SqliteConnectionManager::file(path).with_init(|conn| {
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
        Ok(Self { pool })
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
                created_at_ms INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity_id);
            CREATE INDEX IF NOT EXISTS idx_memories_memory_id ON memories(memory_id);
            CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);

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
                timestamp_ms INTEGER,
                entity_id TEXT,
                label TEXT,
                value REAL,
                unit TEXT,
                content_hash TEXT,
                confidence REAL DEFAULT 1.0,
                source TEXT DEFAULT 'deterministic',
                PRIMARY KEY(timestamp_ms, entity_id, label)
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_entity_label ON metrics(entity_id, label, timestamp_ms);

            -- FTS5
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_memories USING fts5(
                memory_id UNINDEXED,
                entity_id UNINDEXED,
                content,
                tokenize='porter unicode61'
            );

            -- Vector lookup
            CREATE TABLE IF NOT EXISTS vector_lookup (
                vector_id INTEGER PRIMARY KEY,
                memory_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_vector_lookup_memory ON vector_lookup(memory_id);

            -- Ledger turns
            CREATE TABLE IF NOT EXISTS ledger_turns (
                turn_id TEXT PRIMARY KEY,
                entity_id TEXT,
                session_id TEXT,
                speaker TEXT,
                turn_index INTEGER,
                raw_text TEXT,
                document_time_ms INTEGER,
                ingest_time_ms INTEGER,
                source_type TEXT,
                source_uri TEXT,
                raw_sha256 TEXT,
                redaction_state TEXT DEFAULT 'none',
                lifecycle TEXT,
                schema_version INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_ledger_turns_session ON ledger_turns(session_id);
            CREATE INDEX IF NOT EXISTS idx_ledger_turns_entity ON ledger_turns(entity_id);

            -- Memory artifacts
            CREATE TABLE IF NOT EXISTS memory_artifacts (
                artifact_id TEXT PRIMARY KEY,
                artifact_type TEXT,
                entity_id TEXT,
                source_turn_ids TEXT DEFAULT '[]',
                source_memory_ids TEXT DEFAULT '[]',
                source_session_ids TEXT DEFAULT '[]',
                compiler_name TEXT,
                compiler_version TEXT,
                embedding_model TEXT,
                embedding_dim INTEGER,
                index_namespace TEXT,
                lifecycle TEXT,
                created_at_ms INTEGER,
                updated_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_artifacts_source ON memory_artifacts(source_memory_ids);

            -- Artifact versions
            CREATE TABLE IF NOT EXISTS artifact_versions (
                version_id TEXT PRIMARY KEY,
                artifact_id TEXT,
                entity_id TEXT,
                version_data TEXT,
                lifecycle TEXT,
                created_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_artifact_versions_artifact ON artifact_versions(artifact_id);

            -- Temporal events
            CREATE TABLE IF NOT EXISTS temporal_events (
                event_id TEXT PRIMARY KEY,
                entity_id TEXT,
                source_session_id TEXT,
                source_memory_id TEXT,
                subject TEXT,
                relation TEXT,
                object TEXT,
                document_time_ms INTEGER,
                event_time_ms INTEGER,
                event_text TEXT,
                event_type TEXT,
                confidence REAL,
                lifecycle TEXT,
                created_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_events_entity ON temporal_events(entity_id);

            -- Shadow questions
            CREATE TABLE IF NOT EXISTS shadow_questions (
                shadow_id TEXT PRIMARY KEY,
                entity_id TEXT,
                source_session_id TEXT,
                source_memory_id TEXT,
                question_text TEXT,
                answer_type TEXT,
                confidence REAL,
                created_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_shadow_entity ON shadow_questions(entity_id);

            -- Facet postings
            CREATE TABLE IF NOT EXISTS facet_postings (
                posting_id TEXT PRIMARY KEY,
                entity_id TEXT,
                facet_type TEXT,
                facet_value TEXT,
                target_id TEXT,
                target_type TEXT,
                session_id TEXT,
                memory_id TEXT,
                weight REAL DEFAULT 1.0
            );
            CREATE INDEX IF NOT EXISTS idx_facets_entity ON facet_postings(entity_id);

            -- Memory cells
            CREATE TABLE IF NOT EXISTS mem_cells (
                cell_id TEXT PRIMARY KEY,
                entity_id TEXT,
                source_session_id TEXT,
                cell_text TEXT,
                cell_type TEXT,
                document_time_ms INTEGER,
                confidence REAL,
                saliency REAL,
                lifecycle TEXT,
                created_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_cells_entity ON mem_cells(entity_id);

            -- Memory scenes
            CREATE TABLE IF NOT EXISTS mem_scenes (
                scene_id TEXT PRIMARY KEY,
                entity_id TEXT,
                scene_title TEXT,
                scene_summary TEXT,
                scene_type TEXT,
                saliency REAL,
                lifecycle TEXT,
                created_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_scenes_entity ON mem_scenes(entity_id);

            -- Profile facts
            CREATE TABLE IF NOT EXISTS profile_facts (
                profile_fact_id TEXT PRIMARY KEY,
                entity_id TEXT,
                category TEXT,
                value TEXT,
                source_session_id TEXT,
                source_memory_id TEXT,
                confidence REAL,
                document_time_ms INTEGER,
                is_latest INTEGER DEFAULT 1,
                lifecycle TEXT,
                created_at_ms INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_profile_entity ON profile_facts(entity_id);

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

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_temporal_events USING fts5(
                event_id UNINDEXED,
                entity_id UNINDEXED,
                source_session_id UNINDEXED,
                event_text,
                tokenize='porter unicode61'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_shadow_questions USING fts5(
                shadow_id UNINDEXED,
                entity_id UNINDEXED,
                source_session_id UNINDEXED,
                question_text,
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

            -- Card relations
            CREATE TABLE IF NOT EXISTS card_relations (
                source_card_id TEXT NOT NULL,
                target_card_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                PRIMARY KEY(source_card_id, target_card_id, relation_type)
            );

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

        // Migration: add content_hash column if upgrading from an older schema.
        // Wrapped in a closure so prepare errors don't propagate.
        let _ = (|| -> std::result::Result<(), rusqlite::Error> {
            let has_hash_col: bool = conn
                .prepare("SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name = 'content_hash'")?
                .query_row([], |row| row.get::<_, i64>(0))
                .unwrap_or(0)
                > 0;
            if !has_hash_col {
                conn.execute_batch("ALTER TABLE memories ADD COLUMN content_hash TEXT NOT NULL DEFAULT '';")?;
            }
            let has_hash_idx: bool = conn
                .prepare("SELECT COUNT(*) FROM pragma_index_list('memories') WHERE name = 'idx_memories_content_hash'")?
                .query_row([], |row| row.get::<_, i64>(0))
                .unwrap_or(0)
                > 0;
            if !has_hash_idx {
                conn.execute_batch("CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash);")?;
            }
            Ok(())
        })();

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
                "UPDATE memories SET content = ?1, kind = ?2, created_at_ms = ?3, entity_id = ?4, content_hash = ?5 WHERE rowid = ?6",
            )?;
            let mut insert_stmt = tx.prepare_cached(
                "INSERT INTO memories (memory_id, entity_id, content, kind, content_hash, created_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            let mut vec_stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO vector_lookup (vector_id, memory_id, entity_id, timestamp_ms)
                 VALUES (?1, ?2, ?3, ?4)",
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
                        rid
                    ])?;
                    rid
                } else {
                    insert_stmt.execute(params![
                        mem_id,
                        obs.entity_id,
                        obs.textual_content,
                        format!("{:?}", obs.kind),
                        obs.content_hash,
                        ts
                    ])?;
                    tx.last_insert_rowid()
                };

                if !obs.embedding.is_empty() {
                    vec_stmt.execute(params![rid, mem_id, obs.entity_id, ts])?;
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

    pub fn lookup_by_memory_id(&self, memory_id: &str) -> Result<Option<(u64, u64)>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT v.vector_id, m.created_at_ms
             FROM memories m
             LEFT JOIN vector_lookup v ON v.memory_id = m.memory_id
             WHERE m.memory_id = ?1",
        )?;
        let res = stmt.query_row(params![memory_id], |row| {
            Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64))
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
            result.insert(mid, (vid_opt.unwrap_or(0), ts));
        }
        Ok(result)
    }

    pub fn get_observation(
        &self,
        _timestamp: u64,
        memory_id: &str,
    ) -> Result<Option<AgentObservation>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT entity_id, content, kind, created_at_ms FROM memories WHERE memory_id = ?1",
        )?;
        let res = stmt.query_row(params![memory_id], |row| {
            Ok(AgentObservation {
                entity_id: row.get(0)?,
                textual_content: row.get(1)?,
                embedding: Vec::new(),
                kind: parse_kind_enum(row.get::<_, String>(2)?.as_str()),
                content_hash: String::new(),
                created_at_ms: row.get(3)?,
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
            "SELECT memory_id, entity_id, content, kind, created_at_ms FROM memories WHERE memory_id IN ({})",
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

    pub fn set_memory_card_relations_batch(
        &self,
        relations: &[(String, String, String)],
    ) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR IGNORE INTO card_relations (source_card_id, target_card_id, relation_type)
                 VALUES (?1, ?2, ?3)",
            )?;
            for (src, tgt, rel) in relations {
                stmt.execute(params![src, tgt, rel])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    // ── Combined Ingest Upsert ──

    pub fn combined_ingest_upsert(&self, input: &CombinedIngestUpsertInput) -> Result<()> {
        self.ingest_cards(input.cards)?;
        self.ingest_temporal_events(input.events)?;
        self.ingest_shadow_questions(input.shadow_questions)?;
        self.ingest_facet_postings(input.facet_postings)?;
        self.ingest_mem_cells(input.mem_cells)?;
        self.ingest_mem_scenes(input.mem_scenes)?;
        self.ingest_profile_facts(input.profile_facts)?;
        Ok(())
    }

    pub fn ingest_temporal_events(&self, events: &[TemporalEvent]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO temporal_events (event_id, entity_id, source_session_id, source_memory_id,
                 subject, relation, object, document_time_ms, event_time_ms, event_text, event_type,
                 confidence, lifecycle, created_at_ms) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)"
            )?;
            let mut fts = tx.prepare_cached(
                "INSERT OR REPLACE INTO fts_temporal_events (event_id, entity_id, source_session_id, event_text) VALUES (?1, ?2, ?3, ?4)"
            )?;
            for e in events {
                stmt.execute(params![
                    e.event_id,
                    e.entity_id,
                    e.source_session_id,
                    e.source_memory_id,
                    e.subject,
                    e.relation,
                    e.object,
                    e.document_time_ms,
                    e.event_time_ms,
                    e.event_text,
                    e.event_type,
                    e.confidence,
                    e.lifecycle
                        .as_ref()
                        .map(serde_json::to_string)
                        .transpose()
                        .context("failed to serialize temporal event lifecycle")?
                        .as_deref()
                        .unwrap_or(""),
                    e.created_at_ms,
                ])?;
                fts.execute(params![e.event_id, e.entity_id, e.source_session_id, e.event_text])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn ingest_shadow_questions(&self, questions: &[ShadowQuestion]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO shadow_questions (shadow_id, entity_id, source_session_id, source_memory_id,
                 question_text, answer_type, confidence, created_at_ms)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8)"
            )?;
            let mut fts = tx.prepare_cached(
                "INSERT OR REPLACE INTO fts_shadow_questions (shadow_id, entity_id, source_session_id, question_text) VALUES (?1, ?2, ?3, ?4)"
            )?;
            for q in questions {
                stmt.execute(params![
                    q.shadow_id,
                    q.entity_id,
                    q.source_session_id,
                    q.source_memory_id,
                    q.question_text,
                    q.answer_type,
                    q.confidence,
                    q.created_at_ms,
                ])?;
                fts.execute(params![
                    q.shadow_id,
                    q.entity_id,
                    q.source_session_id,
                    q.question_text
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn ingest_facet_postings(&self, postings: &[FacetPosting]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO facet_postings (posting_id, entity_id, facet_type, facet_value,
                 target_id, target_type, session_id, memory_id, weight)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)"
            )?;
            for p in postings {
                let posting_id = format!("fp::{}::{}::{}", p.entity_id, p.facet_type, p.target_id);
                stmt.execute(params![
                    posting_id,
                    p.entity_id,
                    p.facet_type,
                    p.facet_value,
                    p.target_id,
                    p.target_type,
                    p.session_id,
                    p.memory_id,
                    p.weight,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn ingest_mem_cells(&self, cells: &[MemCell]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO mem_cells (cell_id, entity_id, source_session_id, cell_text,
                 cell_type, document_time_ms, confidence, saliency, lifecycle, created_at_ms)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)"
            )?;
            for c in cells {
                stmt.execute(params![
                    c.cell_id,
                    c.entity_id,
                    c.source_session_id,
                    c.cell_text,
                    c.cell_type,
                    c.document_time_ms,
                    c.confidence,
                    c.saliency,
                    c.lifecycle
                        .as_ref()
                        .map(serde_json::to_string)
                        .transpose()
                        .context("failed to serialize mem cell lifecycle")?
                        .as_deref()
                        .unwrap_or(""),
                    c.created_at_ms,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn ingest_mem_scenes(&self, scenes: &[MemSceneRecord]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO mem_scenes (scene_id, entity_id, scene_title, scene_summary,
                 scene_type, saliency, lifecycle, created_at_ms)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8)"
            )?;
            for s in scenes {
                stmt.execute(params![
                    s.scene_id,
                    s.entity_id,
                    s.scene_title,
                    s.scene_summary,
                    s.scene_type,
                    s.saliency,
                    s.lifecycle
                        .as_ref()
                        .map(serde_json::to_string)
                        .transpose()
                        .context("failed to serialize mem scene lifecycle")?
                        .as_deref()
                        .unwrap_or(""),
                    s.created_at_ms,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn ingest_profile_facts(&self, facts: &[ProfileFact]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO profile_facts (profile_fact_id, entity_id, category, value,
                 source_session_id, source_memory_id, confidence, document_time_ms, is_latest, lifecycle, created_at_ms)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)"
            )?;
            for f in facts {
                stmt.execute(params![
                    f.profile_fact_id,
                    f.entity_id,
                    f.category,
                    f.value,
                    f.source_session_id,
                    f.source_memory_id,
                    f.confidence,
                    f.document_time_ms,
                    f.is_latest as i32,
                    f.lifecycle
                        .as_ref()
                        .map(serde_json::to_string)
                        .transpose()
                        .context("failed to serialize profile fact lifecycle")?
                        .as_deref()
                        .unwrap_or(""),
                    f.created_at_ms,
                ])?;
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
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
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
        let mut name_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

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

        let mut alias_stmt = conn.prepare_cached(
            "SELECT alias FROM aliases WHERE entity_id = ?1",
        )?;
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
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?, row.get::<_, i64>(2)? as usize))
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
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let pid = Self::proposal_id(entity_id, from_name, to_name, now);
        let conn = self.get_conn()?;
        let inserted = conn.execute(
            "INSERT OR IGNORE INTO merge_proposals (proposal_id, entity_id, from_name, to_name, tier, confidence, status, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', ?7)",
            params![pid, entity_id, from_name, to_name, tier, confidence as f64, now as i64],
        )?;
        if inserted > 0 { Ok(Some(pid)) } else { Ok(None) }
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
        let resolution = crate::storage::entity_resolver::resolve_name(name, &candidates, name_embedding, config);
        if let Some(ref matched) = resolution.matched_name {
            let tier_label = match resolution.tier {
                crate::storage::entity_resolver::ResolverTier::Exact => return Ok(resolution),
                crate::storage::entity_resolver::ResolverTier::Fuzzy(_) => "fuzzy",
                crate::storage::entity_resolver::ResolverTier::Phonetic => "phonetic",
                crate::storage::entity_resolver::ResolverTier::Embedding(_) => "embedding",
            };
            if name.to_ascii_lowercase() != matched.to_ascii_lowercase() {
                let _ = self.create_merge_proposal(entity_id, name, matched, tier_label, resolution.tier.confidence());
            }
        }
        Ok(resolution)
    }

    /// Batch check which content hashes already exist.
    /// Returns a set of hashes that are already stored.
    pub fn existing_content_hashes(&self, hashes: &[String]) -> Result<std::collections::HashSet<String>> {
        let conn = self.get_conn()?;
        let mut found = std::collections::HashSet::new();
        let mut stmt = conn.prepare_cached(
            "SELECT content_hash FROM memories WHERE content_hash = ?1",
        )?;
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

    pub fn get_linked_memories(&self, memory_id: &str) -> Result<Vec<String>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT target_memory_id FROM memory_links WHERE source_memory_id = ?1
             UNION
             SELECT source_memory_id FROM memory_links WHERE target_memory_id = ?1",
        )?;
        let rows = stmt.query_map(params![memory_id], |row| row.get::<_, String>(0))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
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
             LIMIT ?3"
        )?;
        let rows = stmt.query_map(
            params![seed_memory_id, edge_type_filter, limit as i64],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
        )?;
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
    ) -> Result<Vec<(String, f32, String)>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "WITH seed_nodes AS (
                 SELECT source AS node FROM edges WHERE memory_id = ?1
                 UNION
                 SELECT target AS node FROM edges WHERE memory_id = ?1
             )
             SELECT memory_id, weight, edge_type FROM (
                 SELECT e.memory_id, e.weight, e.edge_type
                 FROM edges e
                 JOIN seed_nodes sn ON e.source = sn.node
                 WHERE e.memory_id != ?1
                   AND (?2 IS NULL OR e.edge_type = ?2)
                 UNION
                 SELECT e.memory_id, e.weight, e.edge_type
                 FROM edges e
                 JOIN seed_nodes sn ON e.target = sn.node
                 WHERE e.memory_id != ?1
                   AND (?2 IS NULL OR e.edge_type = ?2)
             )
             ORDER BY weight DESC
             LIMIT ?3"
        )?;
        let rows = stmt.query_map(
            params![seed_memory_id, edge_type_filter, limit as i64],
            |row| Ok((
                row.get::<_, String>(0)?,
                row.get::<_, f64>(1)? as f32,
                row.get::<_, String>(2)?,
            ))
        )?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
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
                let rowid: i64 = upsert_stmt.query_row(params![
                    merged.session_id,
                    merged.entity_id,
                    json,
                    &merged.router_text,
                    merged.created_at_ms.min(now),
                    now,
                ], |row| row.get(0))?;
                
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
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
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

            if lexical_hits == 0
                && temporal_hits == 0
                && entity_hits == 0
                && !exact_focus_hit
            {
                // Last-ditch accept: if the router_text has ANY of the raw query
                // terms (not just the classified lexical/temporal/entity terms),
                // keep the row. The classified term lists are often empty for
                // short or open-vocabulary questions.
                let lower_terms: Vec<&str> = query_lower
                    .split_whitespace()
                    .filter(|t| t.len() > SEARCH_MIN_TERM_LEN)
                    .collect();
                let has_raw_term = !lower_terms.is_empty()
                    && lower_terms.iter().any(|t| lower.contains(t));
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
                |row| Ok(row.get::<_, String>(0)?),
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
            let rows = stmt.query_map(
                params![entity_id, format!("%{}%", needle)],
                |row| Ok(row.get::<_, String>(0)?),
            )?;
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
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
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

    // ── Fact Versions ──

    pub fn register_fact_versions_batch(
        &self,
        entity_id: &str,
        registrations: &[(&str, u64, &str, &str, &str, &str)],
    ) -> Result<Vec<FactVersionStatus>> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let mut statuses = Vec::new();
        {
            let mut select_stmt = tx.prepare_cached(
                "SELECT memory_id, timestamp_ms FROM fact_versions WHERE fact_key = ?1 AND entity_id = ?2 AND status = 'current'",
            )?;
            let mut update_stmt = tx.prepare_cached(
                "UPDATE fact_versions SET status = 'stale', superseded_by = ?1, valid_to_ms = ?4 WHERE fact_key = ?2 AND entity_id = ?3 AND status = 'current'",
            )?;
            let mut insert_current = tx.prepare_cached(
                "INSERT OR REPLACE INTO fact_versions (fact_key, memory_id, entity_id, subject, predicate, object, status, timestamp_ms, valid_from_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'current', ?7, ?7)",
            )?;
            let mut insert_stale = tx.prepare_cached(
                "INSERT OR REPLACE INTO fact_versions (fact_key, memory_id, entity_id, subject, predicate, object, status, timestamp_ms, superseded_by, valid_from_ms, valid_to_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'stale', ?7, ?8, ?7, ?9)",
            )?;
            // NOTE: We intentionally do NOT delete from fts_memories here.
            // fts_memories is an FTS5 virtual table with memory_id UNINDEXED,
            // so DELETE by memory_id requires a full FTS index scan — O(N) in
            // total DB size. As the database grows this dominated ingest time
            // (400-500ms per batch). Stale FTS entries are harmless: the
            // dedicated FTS indexing phase handles inserts correctly (INSERT OR
            // REPLACE), and queries are not affected by leftover stale rows.

            for (fact_key, ts, memory_id, subject, predicate, object) in registrations {
                let existing: Option<(String, u64)> = select_stmt
                    .query_row(params![fact_key, entity_id], |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, u64>(1)?))
                    })
                    .ok();

                match existing {
                    Some((old_id, old_ts)) if old_id != *memory_id => {
                        if *ts > old_ts {
                            // Incoming is newer: supersede the old version
                            update_stmt.execute(params![memory_id, fact_key, entity_id, *ts as i64])?;
                            insert_current.execute(params![
                                fact_key, memory_id, entity_id, subject, predicate, object, ts
                            ])?;
                            statuses.push(FactVersionStatus::Current {
                                superseded: Some((*ts, old_id)),
                            });
                        } else {
                            // Incoming is older or equal: mark incoming as stale, keep existing current
                            insert_stale.execute(params![
                                fact_key, memory_id, entity_id, subject, predicate, object, ts,
                                old_id, old_ts as i64
                            ])?;
                            statuses.push(FactVersionStatus::Stale { current: (old_ts, old_id) });
                        }
                    }
                    None => {
                        insert_current.execute(params![
                            fact_key, memory_id, entity_id, subject, predicate, object, ts
                        ])?;
                        statuses.push(FactVersionStatus::Current { superseded: None });
                    }
                    _ => {
                        statuses.push(FactVersionStatus::Current { superseded: None });
                    }
                }
            }
        }
        tx.commit()?;
        Ok(statuses)
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
        let vector_id: Option<i64> = tx
            .query_row(
                "SELECT vector_id FROM vector_lookup WHERE memory_id = ?1",
                params![memory_id],
                |row| row.get(0),
            )
            .ok();

        let entity_id: String = tx
            .query_row(
                "SELECT entity_id FROM memories WHERE memory_id = ?1",
                params![memory_id],
                |row| row.get(0),
            )
            .unwrap_or_default();

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
        tx.execute("DELETE FROM fts_memories WHERE memory_id = ?1", params![memory_id])?;
        tx.execute("DELETE FROM negative_centroids WHERE memory_id = ?1", params![memory_id])?;
        tx.execute("DELETE FROM disambiguation_vectors WHERE memory_id = ?1", params![memory_id])?;
        tx.execute("DELETE FROM vector_lookup WHERE memory_id = ?1", params![memory_id])?;
        tx.execute("DELETE FROM memory_cards WHERE card_id = ?1", params![memory_id])?;
        tx.execute(
            "DELETE FROM memory_links WHERE source_memory_id = ?1 OR target_memory_id = ?1",
            params![memory_id],
        )?;
        // Cascade chunks: if this memory was a parent that was chunked, the
        // child chunks have card_id = "{parent}::ct{N}" and source_memory_id
        // = parent. Clean them up.
        let chunk_pattern = format!("{}::ct%", memory_id);
        
        let mut chunk_rowids: Vec<i64> = Vec::new();
        {
            let mut stmt = tx.prepare("SELECT rowid FROM memories WHERE source_memory_id = ?1 OR memory_id LIKE ?2")?;
            let mut rows = stmt.query(params![memory_id, chunk_pattern])?;
            while let Some(row) = rows.next()? {
                chunk_rowids.push(row.get(0)?);
            }
        }
        
        let _chunk_rows = tx.execute(
            "DELETE FROM memory_cards WHERE source_memory_id = ?1 OR card_id LIKE ?2",
            params![memory_id, chunk_pattern],
        )?;
        
        if !chunk_rowids.is_empty() {
            let placeholders: Vec<String> = chunk_rowids.iter().map(|_| "?".to_string()).collect();
            let sql = format!("DELETE FROM fts_memories WHERE rowid IN ({})", placeholders.join(","));
            let params: Vec<&dyn rusqlite::types::ToSql> = chunk_rowids.iter().map(|p| p as &dyn rusqlite::types::ToSql).collect();
            tx.execute(&sql, params.as_slice())?;
            
            let sql2 = format!("DELETE FROM memories WHERE rowid IN ({})", placeholders.join(","));
            tx.execute(&sql2, params.as_slice())?;
        }


        tx.commit()?;

        Ok(DeletedObservation {
            vector_id: vector_id.map(|v| v as u64),
            entity_id,
            tombstone: Some(tombstone),
        })
    }

    // ── Turn / Ledger ──

    pub fn get_ledger_turns_batch(
        &self,
        turn_ids: &[String],
    ) -> Result<std::collections::HashMap<String, LedgerTurn>> {
        if turn_ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let conn = self.get_conn()?;
        let placeholders: Vec<String> = turn_ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT turn_id, entity_id, session_id, speaker, turn_index, raw_text,
                    document_time_ms, ingest_time_ms, source_type, source_uri, raw_sha256,
                    redaction_state, lifecycle, schema_version
             FROM ledger_turns WHERE turn_id IN ({})",
            placeholders.join(",")
        );
        let mut stmt = conn.prepare_cached(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            turn_ids.iter().map(|s| s as &dyn rusqlite::types::ToSql).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            let lifecycle_str: Option<String> = row.get(12)?;
            Ok(LedgerTurn {
                turn_id: row.get(0)?,
                entity_id: row.get(1)?,
                session_id: row.get(2)?,
                speaker: row.get(3)?,
                turn_index: row.get::<_, i32>(4)? as u32,
                raw_text: row.get(5)?,
                document_time_ms: row.get::<_, i64>(6)? as u64,
                ingest_time_ms: row.get::<_, i64>(7)? as u64,
                source_type: row.get(8)?,
                source_uri: row.get(9)?,
                raw_sha256: row.get(10)?,
                redaction_state: row.get(11)?,
                lifecycle: lifecycle_str.and_then(|s| serde_json::from_str(&s).ok()),
                schema_version: row.get::<_, i32>(13)? as u32,
            })
        })?;
        let mut result = std::collections::HashMap::new();
        for row in rows {
            let turn = row?;
            result.insert(turn.turn_id.clone(), turn);
        }
        Ok(result)
    }

    pub fn get_turn_window(
        &self,
        entity_id: &str,
        session_id: &str,
        turn_index: u32,
        radius: u32,
    ) -> Result<Vec<LedgerTurn>> {
        let conn = self.get_conn()?;
        let min_idx = (turn_index as i64 - radius as i64).max(0);
        let max_idx = turn_index as i64 + radius as i64;
        let mut stmt = conn.prepare_cached(
            "SELECT turn_id, entity_id, session_id, speaker, turn_index, raw_text,
                    document_time_ms, ingest_time_ms, source_type, source_uri, raw_sha256,
                    redaction_state, lifecycle, schema_version
             FROM ledger_turns
             WHERE entity_id = ?1 AND session_id = ?2 AND turn_index >= ?3 AND turn_index <= ?4
             ORDER BY turn_index ASC",
        )?;
        let rows = stmt.query_map(params![entity_id, session_id, min_idx, max_idx], |row| {
            let lifecycle_str: Option<String> = row.get(12)?;
            Ok(LedgerTurn {
                turn_id: row.get(0)?,
                entity_id: row.get(1)?,
                session_id: row.get(2)?,
                speaker: row.get(3)?,
                turn_index: row.get::<_, i32>(4)? as u32,
                raw_text: row.get(5)?,
                document_time_ms: row.get::<_, i64>(6)? as u64,
                ingest_time_ms: row.get::<_, i64>(7)? as u64,
                source_type: row.get(8)?,
                source_uri: row.get(9)?,
                raw_sha256: row.get(10)?,
                redaction_state: row.get(11)?,
                lifecycle: lifecycle_str.and_then(|s| serde_json::from_str(&s).ok()),
                schema_version: row.get::<_, i32>(13)? as u32,
            })
        })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Memory Artifacts ──

    pub fn get_memory_artifacts_for_source(
        &self,
        memory_id: &str,
        limit: usize,
    ) -> Result<Vec<MemoryArtifact>> {
        let conn = self.get_conn()?;
        let search = format!("%{}%", memory_id);
        let mut stmt = conn.prepare_cached(
            "SELECT artifact_id, artifact_type, entity_id, source_turn_ids, source_memory_ids,
                    source_session_ids, compiler_name, compiler_version, embedding_model,
                    embedding_dim, index_namespace, lifecycle, created_at_ms, updated_at_ms
             FROM memory_artifacts
             WHERE source_memory_ids LIKE ?1
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![search, limit as i64], |row| {
            let lifecycle_str: Option<String> = row.get(11)?;
            Ok(MemoryArtifact {
                artifact_id: row.get(0)?,
                artifact_type: row.get(1)?,
                entity_id: row.get(2)?,
                source_turn_ids: serde_json::from_str::<Vec<String>>(
                    &row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                )
                .unwrap_or_default(),
                source_memory_ids: serde_json::from_str::<Vec<String>>(
                    &row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                )
                .unwrap_or_default(),
                source_session_ids: serde_json::from_str::<Vec<String>>(
                    &row.get::<_, Option<String>>(5)?.unwrap_or_default(),
                )
                .unwrap_or_default(),
                compiler_name: row.get(6)?,
                compiler_version: row.get(7)?,
                embedding_model: row.get(8)?,
                embedding_dim: row.get::<_, Option<i64>>(9)?.map(|v| v as usize),
                index_namespace: row.get(10)?,
                lifecycle: lifecycle_str.and_then(|s| serde_json::from_str(&s).ok()),
                created_at_ms: row.get::<_, i64>(12)? as u64,
                updated_at_ms: row.get::<_, i64>(13)? as u64,
            })
        })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn get_artifact_versions_for_artifacts(
        &self,
        artifact_ids: &[String],
        limit: usize,
    ) -> Result<Vec<crate::lifecycle::ArtifactVersionRecord>> {
        if artifact_ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.get_conn()?;
        let placeholders: Vec<String> = artifact_ids.iter().map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT version_id, artifact_id, entity_id, version_data, created_at_ms
             FROM artifact_versions WHERE artifact_id IN ({}) ORDER BY created_at_ms DESC LIMIT ?",
            placeholders.join(",")
        );
        let mut stmt = conn.prepare_cached(&sql)?;
        let mut param_refs: Vec<&dyn rusqlite::types::ToSql> =
            artifact_ids.iter().map(|s| s as &dyn rusqlite::types::ToSql).collect();
        let limit_i64 = limit as i64;
        param_refs.push(&limit_i64);
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok(crate::lifecycle::ArtifactVersionRecord {
                version_id: row.get::<_, String>(0)?,
                artifact_id: row.get::<_, String>(1)?,
                operation: "version".to_string(),
                previous_version_id: None,
                compiler_version: row.get::<_, String>(3).unwrap_or_default(),
                reason: String::new(),
                created_at_ms: row.get::<_, i64>(4)? as u64,
            })
        })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

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
            let entity_coverage =
                if query.entities.is_empty() { 0.0 } else { entity_hits as f32 / query.entities.len() as f32 };

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

        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
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
            .map(|t| format!("\"{}\"", t))
            .collect();

        if terms.is_empty() {
            terms = query
                .split_whitespace()
                .filter(|t| t.len() > FTS_MIN_TERM_LEN)
                .map(|t| format!("\"{}\"", t))
                .collect();
        }

        let fts_query = terms.join(" OR ");

        if fts_query.is_empty() {
            return Ok(Vec::new());
        }

        let sql = if entity_id.is_some() {
            "SELECT memory_id, bm25(fts_memories) as score
             FROM fts_memories WHERE fts_memories MATCH ?1 AND entity_id = ?2
             ORDER BY score LIMIT ?3"
        } else {
            "SELECT memory_id, bm25(fts_memories) as score
             FROM fts_memories WHERE fts_memories MATCH ?1
             ORDER BY score LIMIT ?2"
        };

        let mut stmt = conn.prepare_cached(sql)?;
        let results = if let Some(eid) = entity_id {
            stmt.query_map(params![fts_query, eid, limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
            })?
            .collect::<Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(params![fts_query, limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
            })?
            .collect::<Result<Vec<_>, _>>()?
        };
        Ok(results)
    }

    pub fn fts_index_text(&self, memory_id: &str, content: &str, entity_id: &str) -> Result<()> {
        let conn = self.get_conn()?;
        conn.execute(
            "INSERT OR REPLACE INTO fts_memories (memory_id, entity_id, content) VALUES (?1, ?2, ?3)",
            params![memory_id, entity_id, content],
        )?;
        Ok(())
    }

    pub fn fts_index_batch(&self, batch: &[(String, String, String)]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO fts_memories (memory_id, entity_id, content) VALUES (?1, ?2, ?3)",
            )?;
            for (memory_id, entity_id, content) in batch {
                stmt.execute(params![memory_id, entity_id, content])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn fts_remove_document(&self, memory_id: &str) -> Result<()> {
        let conn = self.get_conn()?;
        conn.execute("DELETE FROM fts_memories WHERE memory_id = ?1", params![memory_id])?;
        Ok(())
    }

    pub fn fts_clear(&self) -> Result<()> {
        let conn = self.get_conn()?;
        conn.execute("DELETE FROM fts_memories", [])?;
        Ok(())
    }

    // ── Graph methods (delegated here) ──

    pub fn graph_upsert_memory_batch(
        &self,
        batch: &GraphEdgeBatch<'_>,
    ) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO edges (edge_id, source, target, edge_type, label, status, timestamp_ms, memory_id, weight)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )?;
            for entry in batch {
                let edge_id = format!("edge::{}::{}::{}", entry.memory_id, entry.subject, entry.predicate);
                let label = format!("{} {} {}", entry.subject, entry.predicate, entry.object);
                let weight = crate::graph::EdgeType::from_str(entry.predicate).default_weight();
                stmt.execute(params![
                    edge_id, entry.subject, entry.object, entry.predicate, label, entry.status, entry.timestamp as i64, entry.memory_id, weight as f64,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Insert a single typed edge using owned strings (no lifetime issues).
    pub fn graph_insert_edge(
        &self,
        _entity_scope: &str,
        memory_id: &str,
        subject: &str,
        predicate: &str,   // becomes edge_type
        object: &str,
        timestamp_ms: u64,
    ) -> Result<()> {
        let edge_id = format!("edge::{}::{}::{}", memory_id, subject, predicate);
        let label = format!("{} {} {}", subject, predicate, object);
        let conn = self.get_conn()?;
        conn.execute(
            "INSERT OR IGNORE INTO edges (edge_id, source, target, edge_type, label, status, timestamp_ms, memory_id)
             VALUES (?1, ?2, ?3, ?4, ?5, 'current', ?6, ?7)",
            params![edge_id, subject, object, predicate, label, timestamp_ms as i64, memory_id],
        )?;
        Ok(())
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

    pub fn clear_all(&self) -> Result<()> {
        let conn = self.get_conn()?;
        conn.execute_batch(
            "DELETE FROM memories;
             DELETE FROM fts_memories;
             DELETE FROM vector_lookup;
             DELETE FROM memory_cards;
             DELETE FROM edges;
             DELETE FROM ledger_turns;
             DELETE FROM memory_artifacts;
             DELETE FROM artifact_versions;
             DELETE FROM temporal_events;
             DELETE FROM shadow_questions;
             DELETE FROM facet_postings;
             DELETE FROM mem_cells;
             DELETE FROM mem_scenes;
             DELETE FROM profile_facts;
             DELETE FROM session_router;
             DELETE FROM aliases;
             DELETE FROM preferences;
             DELETE FROM memory_links;
             DELETE FROM fact_versions;
             DELETE FROM card_relations;
             DELETE FROM core_profiles;
             DELETE FROM entity_embeddings;
             DELETE FROM deletion_tombstones;
             DELETE FROM metrics;",
        )?;
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
            .query_row("SELECT COUNT(*) FROM profile_facts WHERE is_latest = 1", [], |row| {
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
        Ok(crate::api::types::StorageStatsResponse {
            memory_card_count: count("memory_cards") as usize,
            edge_count: count("edges") as usize,
            memory_count: count("memories") as usize,
            metric_count: count("metrics") as usize,
            ledger_turn_count: count("ledger_turns") as usize,
            memory_artifact_count: count("memory_artifacts") as usize,
            temporal_event_count: count("temporal_events") as usize,
            shadow_question_count: count("shadow_questions") as usize,
            facet_posting_count: count("facet_postings") as usize,
            mem_cell_count: count("mem_cells") as usize,
            mem_scene_count: count("mem_scenes") as usize,
            profile_fact_count: count("profile_facts") as usize,
            session_router_count: count("session_router") as usize,
            fact_version_count: count("fact_versions") as usize,
            card_relation_count: count("card_relations") as usize,
            memory_link_count: count("memory_links") as usize,
            alias_count: count("aliases") as usize,
            preference_count: count("preferences") as usize,
            core_profile_count: count("core_profiles") as usize,
            deletion_tombstone_count: count("deletion_tombstones") as usize,
            storage_bytes: storage_bytes as usize,
        })
    }

    // ── Get memory cards batch ──

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

    pub fn get_disambiguation_vectors_batch(
        &self,
        entity_id: &str,
    ) -> Result<Vec<(String, Vec<f32>)>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT memory_id, vector_blob FROM disambiguation_vectors WHERE entity_id = ?1",
        )?;
        let rows = stmt.query_map(params![entity_id], |row| {
            let memory_id: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let vector = bytes_to_vec_f32(&blob);
            Ok((memory_id, vector))
        })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn get_negative_centroids_batch(&self, entity_id: &str) -> Result<Vec<(String, Vec<f32>)>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT memory_id, centroid_blob FROM negative_centroids WHERE entity_id = ?1",
        )?;
        let rows = stmt.query_map(params![entity_id], |row| {
            let memory_id: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            let vector = bytes_to_vec_f32(&blob);
            Ok((memory_id, vector))
        })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn invalidated_set(&self) -> Result<std::collections::HashSet<String>> {
        let conn = self.get_conn()?;
        let mut stmt =
            conn.prepare_cached("SELECT memory_id FROM fact_versions WHERE status = 'stale'")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut set = std::collections::HashSet::new();
        for r in rows {
            set.insert(r?);
        }
        Ok(set)
    }

    pub fn invalidated_set_at_time(&self, point_in_time_ms: u64) -> Result<std::collections::HashSet<String>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT memory_id FROM fact_versions 
             WHERE (valid_to_ms IS NOT NULL AND valid_to_ms <= ?1)
                OR (COALESCE(valid_from_ms, 0) > ?1)"
        )?;
        let rows = stmt.query_map(params![point_in_time_ms as i64], |row| row.get::<_, String>(0))?;
        let mut set = std::collections::HashSet::new();
        for r in rows {
            set.insert(r?);
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
                if let Ok(mut lifecycle) = serde_json::from_str::<crate::lifecycle::LifecycleMetadata>(&lifecycle_json) {
                    if lifecycle.lifecycle_state != crate::lifecycle::LifecycleState::Expired
                        && matches!(
                            lifecycle.retention_class,
                            crate::lifecycle::RetentionClass::Ephemeral | crate::lifecycle::RetentionClass::Working
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
                    "UPDATE memory_cards SET lifecycle = ?1, updated_at_ms = ?2 WHERE card_id = ?3"
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

#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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

        let registrations1 = vec![
            ("fact_key_1".to_string(), 100, "mem-100".to_string(), "Caroline".to_string(), "prefers".to_string(), "counseling".to_string())
        ];
        let statuses1 = store.register_fact_versions_batch("Caroline", &registrations1).unwrap();
        assert_eq!(statuses1.len(), 1);

        let registrations2 = vec![
            ("fact_key_1".to_string(), 200, "mem-200".to_string(), "Caroline".to_string(), "prefers".to_string(), "coaching".to_string())
        ];
        let statuses2 = store.register_fact_versions_batch("Caroline", &registrations2).unwrap();
        assert_eq!(statuses2.len(), 1);

        let stale_at_150 = store.invalidated_set_at_time(150).unwrap();
        assert!(stale_at_150.contains("mem-200"));
        assert!(!stale_at_150.contains("mem-100"));

        let stale_at_250 = store.invalidated_set_at_time(250).unwrap();
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

        let registrations1 = vec![
            ("pref_key_1".to_string(), 100, "mem-pref-1".to_string(), "Caroline".to_string(), "prefers".to_string(), "counseling".to_string())
        ];
        let statuses1 = store.register_fact_versions_batch("Caroline", &registrations1).unwrap();
        assert_eq!(statuses1.len(), 1);

        let registrations2 = vec![
            ("pref_key_1".to_string(), 200, "mem-pref-2".to_string(), "Caroline".to_string(), "prefers".to_string(), "coaching".to_string())
        ];
        let statuses2 = store.register_fact_versions_batch("Caroline", &registrations2).unwrap();
        assert_eq!(statuses2.len(), 1);

        let stale_at_250 = store.invalidated_set_at_time(250).unwrap();
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

        store.ingest_cards(&[card.clone()]).unwrap();

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
}
