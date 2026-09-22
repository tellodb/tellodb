use anyhow::{Context, Result};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::info;

use super::types::{
    build_session_router_text, AgentObservation, GraphEdgeEntry, LedgerTurn, MemoryCard,
    SessionRouterRecord,
};
use std::fmt::Write as _;

pub(crate) type GraphEdgeBatch<'a> = [GraphEdgeEntry<'a>];

const PRAGMA_CACHE_SIZE: i64 = -32_768;
const PRAGMA_MMAP_SIZE: i64 = 1_073_741_824;
const PRAGMA_BUSY_TIMEOUT: i64 = 10000;
const PRAGMA_PAGE_SIZE: i64 = 8192;
const PRAGMA_WAL_AUTOCHECKPOINT: i64 = 2000;
const STATEMENT_CACHE_CAPACITY: usize = 512;

pub(crate) const METRICS_DDL: &str = "CREATE TABLE IF NOT EXISTS metrics (
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

pub(crate) const SOURCE_DEPTH_DIVISOR: f32 = 8.0;

pub(crate) const SESSION_LEXICAL_WEIGHT: f32 = 0.48;
pub(crate) const SESSION_TEMPORAL_WEIGHT: f32 = 0.18;
pub(crate) const SESSION_ENTITY_WEIGHT: f32 = 0.24;
pub(crate) const SESSION_DEPTH_WEIGHT: f32 = 0.04;
pub(crate) const SESSION_FOCUS_BONUS: f32 = 0.06;

pub(crate) const CARD_LEXICAL_WEIGHT: f32 = 0.42;
pub(crate) const CARD_TEMPORAL_WEIGHT: f32 = 0.16;
pub(crate) const CARD_ENTITY_WEIGHT: f32 = 0.24;
pub(crate) const CARD_ROUTE_BOOST: f32 = 0.12;
pub(crate) const CARD_LATEST_BOOST: f32 = 0.04;
pub(crate) const CARD_STALE_PENALTY: f32 = -0.06;
pub(crate) const CARD_CONFIDENCE_WEIGHT: f32 = 0.08;

pub(crate) const FACT_TYPE_BOOST: f32 = 0.10;
pub(crate) const PREFERENCE_TYPE_BOOST: f32 = 0.09;
pub(crate) const EVENT_TYPE_BOOST: f32 = 0.07;
pub(crate) const DECISION_TYPE_BOOST: f32 = 0.06;
pub(crate) const INFERENCE_TYPE_BOOST: f32 = 0.04;
pub(crate) const OTHER_TYPE_BOOST: f32 = 0.03;

pub(crate) const FOCUS_MATCH_MIN_LEN: usize = 4;
pub(crate) const FTS_MIN_TERM_LEN: usize = 1;
pub(crate) const SEARCH_MIN_TERM_LEN: usize = 2;

pub(crate) fn unix_timestamp_ms() -> Result<i64> {
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
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Wraps a term as an FTS5 string, doubling any embedded quote.
///
/// An unescaped `"` closes the phrase early and makes the whole MATCH
/// expression invalid, which the callers turn into an empty lane.
pub(crate) fn fts_quote(term: impl AsRef<str>) -> String {
    format!("\"{}\"", term.as_ref().replace('"', "\"\""))
}

/// Stable FTS5 rowid for a document key. FTS tables have no unique key on
/// `memory_id`, so `INSERT OR REPLACE` only replaces when the rowid matches;
/// deriving it from the key makes re-ingest replace instead of duplicate and
/// makes deletes an O(log n) rowid lookup.
pub(crate) fn fts_rowid(key: &str) -> i64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in key.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    // Positive and non-zero.
    ((hash >> 1) | 1) as i64
}

/// Schema version recorded in `PRAGMA user_version`.
pub(crate) const SCHEMA_VERSION: i64 = 7;

impl TenantStore {
    pub fn new(path: &Path) -> Result<Self> {
        let synchronous = match std::env::var("TELLODB_DURABILITY")
            .unwrap_or_else(|_| "full".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "full" => "FULL",
            "normal" => "NORMAL",
            value => anyhow::bail!("unknown TELLODB_DURABILITY {value:?} (full, normal)"),
        };
        let manager = SqliteConnectionManager::file(path).with_init(move |conn| {
            // The store uses ~70 distinct cached statements plus IN-list
            // queries; with rusqlite's default capacity of 16 they were
            // evicted and re-parsed constantly.
            conn.set_prepared_statement_cache_capacity(STATEMENT_CACHE_CAPACITY);
            conn.execute_batch(&format!(
                "PRAGMA page_size = {PRAGMA_PAGE_SIZE};
                     PRAGMA journal_mode = WAL;
                     PRAGMA synchronous = {synchronous};
                     PRAGMA foreign_keys = ON;
                     PRAGMA temp_store = MEMORY;
                     PRAGMA cache_size = {PRAGMA_CACHE_SIZE};
                     PRAGMA mmap_size = {PRAGMA_MMAP_SIZE};
                     PRAGMA busy_timeout = {PRAGMA_BUSY_TIMEOUT};
                     PRAGMA wal_autocheckpoint = {PRAGMA_WAL_AUTOCHECKPOINT};",
            ))
        });
        let max_size = num_cpus::get().clamp(4, 16) as u32;
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
        self.vectors
            .set(index)
            .map_err(|_existing_index| anyhow::anyhow!("vector index already attached"))
    }

    pub fn vectors(&self) -> Result<&crate::vector_index::VectorIndex> {
        self.vectors.get().context("tenant has no vector index attached")
    }

    pub fn vector_source(&self) -> std::sync::Arc<dyn crate::vector_index::VectorSource> {
        std::sync::Arc::new(SqliteVectorSource { pool: self.pool.clone() })
    }

    pub fn stored_vector_counts(&self) -> Result<(usize, usize)> {
        let conn = self.get_conn()?;
        Ok(conn.query_row(
            "SELECT COUNT(embedding), COUNT(*) - COUNT(embedding) FROM vector_lookup",
            [],
            |row| Ok((row.get::<_, i64>(0)? as usize, row.get::<_, i64>(1)? as usize)),
        )?)
    }

    pub fn get_conn(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.pool.get().context("Failed to get connection from pool")
    }

    pub fn checkpoint(&self) -> Result<()> {
        let conn = self.get_conn()?;
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }

    pub(crate) fn allocate_vector_ids<'a>(
        &self,
        items: impl IntoIterator<Item = (u64, &'a str, &'a AgentObservation)>,
    ) -> Result<Vec<Option<u64>>> {
        let items = items
            .into_iter()
            .map(|(timestamp, memory_id, observation)| {
                (timestamp, memory_id.to_string(), observation.clone())
            })
            .collect::<Vec<_>>();
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let rowids = crate::storage::repo::ingest::insert_observations_tx(&tx, &items)?;
        tx.commit()?;
        Ok(rowids)
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

pub(crate) fn merge_router_records(
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

pub(crate) fn contains_term_count(lower_haystack: &str, terms: &[String]) -> usize {
    terms
        .iter()
        .filter(|term| {
            let needle = term.trim().to_ascii_lowercase();
            !needle.is_empty() && lower_haystack.contains(needle.as_str())
        })
        .count()
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

// Re-import needed for artifact versions
use serde::{Deserialize, Serialize};

pub(crate) fn vec_f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes
}

pub(crate) fn bytes_to_vec_f32(bytes: &[u8]) -> Vec<f32> {
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
        for chunk in ids.chunks(crate::storage::repo::IN_CHUNK) {
            let values = crate::storage::repo::padded_in_chunk(chunk);
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT vector_id, embedding FROM vector_lookup
                 WHERE embedding IS NOT NULL AND vector_id IN ({})",
                crate::storage::repo::in_placeholders(crate::storage::repo::IN_CHUNK)
            ))?;
            let params = values.into_iter().map(|id| *id as i64);
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
pub(crate) fn same_fact_object(a: &str, b: &str) -> bool {
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
pub(crate) type EdgeNeighbour = (String, f32, String);

pub(crate) type MemoryTurnRow = (String, Option<String>, LedgerTurn);

pub(crate) fn memory_turn_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryTurnRow> {
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

pub(crate) fn memory_card_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryCard> {
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
        source_turn_index: row.get::<_, i64>(20)? as usize,
        document_time: row.get(21)?,
        conversation_time: row.get(22)?,
        event_time: row.get(23)?,
        created_at_ms: row.get(18)?,
        updated_at_ms: row.get(19)?,
    })
}

/// Adds a row to its turn; chunks of one memory are joined in rowid order.
pub(crate) fn merge_turn(
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
    use crate::storage::types::{FactVersionStatus, MemoryKind};
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
    fn migrate_is_idempotent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tenant.db");

        for _ in 0..2 {
            let store = TenantStore::new(&path).unwrap();
            let conn = store.get_conn().unwrap();
            let version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0)).unwrap();
            assert_eq!(version, SCHEMA_VERSION);
        }
    }

    #[test]
    fn new_databases_use_durable_bounded_pragmas() {
        let temp = tempdir().unwrap();
        let store = TenantStore::new(&temp.path().join("tenant.db")).unwrap();
        let conn = store.get_conn().unwrap();
        let synchronous: i64 = conn.query_row("PRAGMA synchronous", [], |row| row.get(0)).unwrap();
        let page_size: i64 = conn.query_row("PRAGMA page_size", [], |row| row.get(0)).unwrap();
        let cache_size: i64 = conn.query_row("PRAGMA cache_size", [], |row| row.get(0)).unwrap();
        let autocheckpoint: i64 =
            conn.query_row("PRAGMA wal_autocheckpoint", [], |row| row.get(0)).unwrap();

        assert_eq!(synchronous, 2);
        assert_eq!(page_size, 8192);
        assert_eq!(cache_size, -32_768);
        assert_eq!(autocheckpoint, 2_000);
    }

    /// Builds a `memories` table as it stood before the columns named in
    /// `missing` existed, stamps an old `user_version`, and inserts one row --
    /// so opening it exercises the real `has_column` migration path rather
    /// than a table that already has everything.
    #[cfg(test)]
    fn aged_fixture(path: &std::path::Path, user_version: i64, missing: &[&str]) {
        let all: Vec<(&str, &str)> = vec![
            ("content_hash", "content_hash TEXT NOT NULL DEFAULT ''"),
            ("session_id", "session_id TEXT NOT NULL DEFAULT ''"),
            ("turn_index", "turn_index INTEGER NOT NULL DEFAULT 0"),
            ("role", "role TEXT NOT NULL DEFAULT ''"),
            ("parent_memory_id", "parent_memory_id TEXT"),
            ("indexed", "indexed INTEGER NOT NULL DEFAULT 0"),
            ("recorded_at_ms", "recorded_at_ms INTEGER NOT NULL DEFAULT 0"),
            ("expires_at_ms", "expires_at_ms INTEGER"),
        ];
        let columns: Vec<&str> =
            all.iter().filter(|(name, _)| !missing.contains(name)).map(|(_, ddl)| *ddl).collect();
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute_batch(&format!(
            "CREATE TABLE memories (
                 rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                 memory_id TEXT NOT NULL UNIQUE,
                 entity_id TEXT NOT NULL,
                 content TEXT NOT NULL,
                 kind TEXT NOT NULL,
                 created_at_ms INTEGER NOT NULL{}{}
             );
             PRAGMA user_version = {user_version};",
            if columns.is_empty() { "" } else { ", " },
            columns.join(", ")
        ))
        .unwrap();
        conn.execute(
            "INSERT INTO memories (memory_id, entity_id, content, kind, created_at_ms)
             VALUES ('legacy::s1::0', 'legacy', 'a memory written by an older build', 'fact', 100)",
            [],
        )
        .unwrap();
        drop(conn);
    }

    #[test]
    fn databases_from_older_schema_versions_still_open() {
        // Each entry is a shape the store actually shipped: the columns listed
        // did not exist yet at that user_version.
        let generations: [(i64, &[&str]); 4] = [
            (
                1,
                &[
                    "content_hash",
                    "session_id",
                    "turn_index",
                    "role",
                    "parent_memory_id",
                    "indexed",
                    "recorded_at_ms",
                    "expires_at_ms",
                ],
            ),
            (3, &["content_hash", "indexed", "recorded_at_ms", "expires_at_ms"]),
            (4, &["content_hash", "recorded_at_ms", "expires_at_ms"]),
            (6, &["expires_at_ms"]),
        ];

        for (version, missing) in generations {
            let temp = tempdir().unwrap();
            let path = temp.path().join("tenant.db");
            aged_fixture(&path, version, missing);

            let store = TenantStore::new(&path)
                .unwrap_or_else(|e| panic!("v{version} database must open: {e}"));
            let conn = store.get_conn().unwrap();

            let migrated: i64 =
                conn.query_row("PRAGMA user_version", [], |row| row.get(0)).unwrap();
            assert_eq!(migrated, SCHEMA_VERSION, "v{version} did not reach current schema");

            // Every column the migration was supposed to add is present...
            for column in missing {
                let present: i64 = conn
                    .query_row(
                        "SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name = ?1",
                        params![column],
                        |row| row.get(0),
                    )
                    .unwrap();
                assert_eq!(present, 1, "v{version} migration did not add {column}");
            }

            // ...and the row written by the older build is still readable.
            let content: String = conn
                .query_row(
                    "SELECT content FROM memories WHERE memory_id = 'legacy::s1::0'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(content, "a memory written by an older build", "v{version} lost data");
        }
    }

    #[test]
    fn migrating_an_old_database_twice_is_a_no_op() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("tenant.db");
        aged_fixture(&path, 3, &["content_hash", "indexed", "recorded_at_ms", "expires_at_ms"]);

        drop(TenantStore::new(&path).unwrap());
        let store = TenantStore::new(&path).expect("reopening a migrated database must work");
        let conn = store.get_conn().unwrap();
        let rows: i64 =
            conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0)).unwrap();
        assert_eq!(rows, 1, "second open duplicated or dropped rows");
    }

    #[test]
    fn newer_schema_versions_are_rejected() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("tenant.db");
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch(&format!("PRAGMA user_version = {};", SCHEMA_VERSION + 1)).unwrap();
        drop(conn);

        let error = TenantStore::new(&path).err().expect("newer schema must fail");
        assert!(error.to_string().contains("newer than supported"));
    }

    #[test]
    fn retention_expiry_physically_deletes_the_memory() {
        let temp = tempdir().unwrap();
        let store = TenantStore::new(&temp.path().join("tenant.db")).unwrap();
        store
            .insert_observation(
                100,
                "expired",
                &AgentObservation {
                    entity_id: "alice".into(),
                    textual_content: "temporary secret".into(),
                    expires_at_ms: Some(200),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(store.purge_expired_memories(199, 10).unwrap(), 0);
        assert_eq!(store.purge_expired_memories(200, 10).unwrap(), 1);
        assert!(store.get_observation(100, "expired").unwrap().is_none());
        assert_eq!(store.purge_expired_memories(200, 10).unwrap(), 0);
    }

    #[test]
    fn content_hash_migration_preserves_dedup_identity() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tenant.db");
        {
            let store = TenantStore::new(&path).unwrap();
            store
                .insert_observations_batch(&[(
                    100,
                    "m1".to_string(),
                    AgentObservation {
                        entity_id: "alice".to_string(),
                        textual_content: "I live in Denver.".to_string(),
                        kind: MemoryKind::Fact,
                        content_hash: "legacy-debug-hash".to_string(),
                        created_at_ms: 100,
                        ..Default::default()
                    },
                )])
                .unwrap();
            let conn = store.get_conn().unwrap();
            conn.execute("PRAGMA user_version = 4", []).unwrap();
        }

        let store = TenantStore::new(&path).unwrap();
        let expected =
            crate::storage::types::content_hash("I live in Denver.", "alice", MemoryKind::Fact);
        let hashes = store.stored_content_hashes(&["m1".to_string()]).unwrap();
        assert_eq!(hashes.get("m1"), Some(&expected));
        assert_eq!(store.existing_content_hashes(&[expected]).unwrap().len(), 1);
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
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
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
            textual_content: "Ada is building a database".to_string(),
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
            textual_content: "Ada is building a database in Rust".to_string(),
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

        assert_eq!(content, "Ada is building a database in Rust");
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
    fn lookup_by_memory_ids_batch_handles_two_thousand_ids() {
        let temp = tempdir().unwrap();
        let store = TenantStore::new(&temp.path().join("tenant.db")).unwrap();
        let items: Vec<(u64, String, AgentObservation)> = (0..2_000)
            .map(|index| {
                (
                    index as u64,
                    format!("memory-{index}"),
                    AgentObservation {
                        entity_id: "alice".to_string(),
                        textual_content: format!("memory {index}"),
                        created_at_ms: index as u64,
                        ..Default::default()
                    },
                )
            })
            .collect();
        store.insert_observations_batch(&items).unwrap();
        let ids: Vec<String> = items.into_iter().map(|(_, memory_id, _)| memory_id).collect();
        let found = store.lookup_by_memory_ids_batch(&ids).unwrap();
        assert_eq!(found.len(), 2_000);
        assert_eq!(found["memory-1999"].0, 1_999);
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
        assert_eq!(card_loaded.source_turn_index, 0);
        assert_eq!(card_loaded.document_time, 100);
        assert_eq!(card_loaded.conversation_time, 100);
        assert_eq!(card_loaded.event_time, None);
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
                 WHERE fact_key = ?1 ORDER BY valid_from_ms, recorded_at_ms, rowid",
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
        let ids: Vec<String> = ["m50", "m75", "m100"].iter().map(|s| (*s).to_string()).collect();
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

    #[test]
    fn later_recorded_fact_wins_when_event_times_match() {
        let temp = tempdir().unwrap();
        let store = TenantStore::new(&temp.path().join("tenant.db")).unwrap();
        store
            .register_fact_versions_batch(
                "user",
                &[("residence", 100, "first", "user", "lives_in", "Austin")],
            )
            .unwrap();
        store
            .register_fact_versions_batch(
                "user",
                &[("residence", 100, "second", "user", "lives_in", "Boston")],
            )
            .unwrap();

        assert_eq!(
            store.get_current_fact_value("user", "residence").unwrap().as_deref(),
            Some("Boston")
        );
        assert_eq!(
            fact_chain(&store, "residence"),
            vec![
                ("first".into(), "stale".into(), 100, Some(100)),
                ("second".into(), "current".into(), 100, None),
            ]
        );
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

    proptest::proptest! {
        /// The sequential test above registers one version per batch, so the
        /// chain is always settled before the next call sees it. Registering a
        /// whole set in ONE batch is the case that panicked with "fact chain
        /// contains a current version" on a LongMemEval ingest: the status for
        /// a superseded row used to read the chain's head mid-loop, while later
        /// rows in the same batch could still supersede each other and the
        /// predecessor update could leave the chain transiently with no current
        /// row at all.
        #[test]
        fn one_batch_of_versions_settles_into_a_consistent_chain(
            timestamps in proptest::collection::vec(0u64..50, 1..8)
        ) {
            let temp = tempdir().unwrap();
            let store = TenantStore::new(&temp.path().join("tenant.db")).unwrap();
            let ids: Vec<String> = (0..timestamps.len()).map(|i| format!("m{i}")).collect();
            let objects: Vec<String> = (0..timestamps.len()).map(|i| format!("o{i}")).collect();
            let batch: Vec<(&str, u64, &str, &str, &str, &str)> = timestamps
                .iter()
                .enumerate()
                .map(|(i, ts)| ("k", *ts, ids[i].as_str(), "e", "p", objects[i].as_str()))
                .collect();

            let statuses = store.register_fact_versions_batch("e", &batch).unwrap();

            let chain = fact_chain(&store, "k");
            proptest::prop_assert_eq!(chain.len(), timestamps.len());
            proptest::prop_assert_eq!(chain.iter().filter(|c| c.1 == "current").count(), 1);
            let head = chain.last().unwrap();
            proptest::prop_assert_eq!(head.1.as_str(), "current");
            proptest::prop_assert_eq!(head.2, *timestamps.iter().max().unwrap());
            for pair in chain.windows(2) {
                proptest::prop_assert_eq!(pair[0].3, Some(pair[1].2));
                proptest::prop_assert!(pair[0].2 <= pair[1].2);
            }

            // Every superseded row points at the chain's FINAL head, not at
            // whichever row happened to be current partway through the batch.
            for status in &statuses {
                if let FactVersionStatus::Stale { current: (_, id) } = status {
                    proptest::prop_assert_eq!(id.as_str(), head.0.as_str());
                }
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
        assert!((scores["b"] - 0.6).abs() < 1e-6, "got {scores:?}");
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
    fn point_in_time_fts_fills_the_requested_limit() {
        let dir = tempfile::tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("t.db")).unwrap();
        let items: Vec<_> = (1..=30)
            .map(|timestamp| {
                let memory_id = format!("alice::s::{timestamp}");
                (
                    timestamp,
                    memory_id,
                    AgentObservation {
                        entity_id: "alice".into(),
                        textual_content: "shared timeline term".into(),
                        created_at_ms: timestamp,
                        ..Default::default()
                    },
                )
            })
            .collect();
        store.insert_observations_batch(&items).unwrap();
        let fts: Vec<_> = items
            .iter()
            .map(|(_, memory_id, observation)| {
                (
                    memory_id.clone(),
                    observation.entity_id.clone(),
                    observation.textual_content.clone(),
                )
            })
            .collect();
        store.fts_index_batch(&fts).unwrap();

        let hits = store.fts_search_at("timeline", 10, Some("alice"), Some(10), None).unwrap();
        assert_eq!(hits.len(), 10);
        let identities = store
            .lookup_by_memory_ids_batch(
                &hits.iter().map(|(memory_id, _)| memory_id.clone()).collect::<Vec<_>>(),
            )
            .unwrap();
        assert!(identities.values().all(|(timestamp, _)| *timestamp <= 10));
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
    fn entity_tokens_survive_the_porter_tokenizer() {
        // The token goes through `porter unicode61`, whose suffix rules
        // rewrite endings like ED, EED and AT. Hex digits include a-f, so a
        // token can consist only of letters and be eligible for stemming. If
        // two entities stemmed alike, one entity would read another's rows.
        let dir = tempfile::tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("t.db")).unwrap();
        // Characters whose UTF-8 bytes are all a-f, so the hex is all letters.
        let ids = ["\u{caa}", "\u{cab}", "\u{cae}", "\u{caa}\u{cab}", "feed", "deed", "aed"];
        for id in ids {
            store.fts_index_text(&format!("{id}::s::0"), "shared words", id).unwrap();
        }
        for id in ids {
            let hits = store.fts_search("shared", 10, Some(id)).unwrap();
            assert_eq!(hits.len(), 1, "entity {id:?} matched {hits:?}");
            assert!(hits[0].0.starts_with(&format!("{id}::")), "{id:?} -> {hits:?}");
        }
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
