use super::prelude::*;

impl TenantStore {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn init_schema(conn: &rusqlite::Connection) -> Result<()> {
        let version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if version > SCHEMA_VERSION {
            anyhow::bail!(
                "database schema version {version} is newer than supported version {SCHEMA_VERSION}"
            );
        }
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
                recorded_at_ms INTEGER NOT NULL,
                expires_at_ms INTEGER,
                session_id TEXT NOT NULL DEFAULT '',
                turn_index INTEGER NOT NULL DEFAULT 0,
                role TEXT NOT NULL DEFAULT '',
                parent_memory_id TEXT,
                indexed INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity_id);
            CREATE INDEX IF NOT EXISTS idx_memories_memory_id ON memories(memory_id);
            -- Indexes on recorded_at_ms and expires_at_ms belong to migrate(),
            -- not here. This batch runs first, so on a database written before
            -- those columns existed the CREATE INDEX fails and the migration
            -- that would have added them never runs. migrate() creates both
            -- once the columns are in place.

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
                updated_at_ms INTEGER,
                source_turn_index INTEGER NOT NULL DEFAULT 0,
                document_time INTEGER NOT NULL DEFAULT 0,
                conversation_time INTEGER NOT NULL DEFAULT 0,
                event_time INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_memory_cards_entity ON memory_cards(entity_id);
            CREATE INDEX IF NOT EXISTS idx_memory_cards_session ON memory_cards(source_session_id);
            CREATE INDEX IF NOT EXISTS idx_memory_cards_source ON memory_cards(source_memory_id);
            CREATE INDEX IF NOT EXISTS idx_memory_cards_entity_latest
                ON memory_cards(entity_id, is_latest, expires_at);

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

            CREATE TABLE IF NOT EXISTS session_router_sources (
                session_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                PRIMARY KEY(session_id, entity_id, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_router_sources_memory
                ON session_router_sources(memory_id);

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
                recorded_at_ms INTEGER NOT NULL,
                PRIMARY KEY(fact_key, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_fact_entity ON fact_versions(entity_id);
            CREATE INDEX IF NOT EXISTS idx_fact_versions_lookup ON fact_versions(fact_key, entity_id, status);
            CREATE INDEX IF NOT EXISTS idx_fact_versions_order
                ON fact_versions(fact_key, entity_id, timestamp_ms, recorded_at_ms);

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

            CREATE TABLE IF NOT EXISTS consolidation_queue (
                memory_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                textual_content TEXT NOT NULL
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

        conn.execute_batch(METRICS_DDL)?;
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

    #[allow(clippy::too_many_lines)]
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
            ("indexed", "ALTER TABLE memories ADD COLUMN indexed INTEGER NOT NULL DEFAULT 0"),
            (
                "recorded_at_ms",
                "ALTER TABLE memories ADD COLUMN recorded_at_ms INTEGER NOT NULL DEFAULT 0",
            ),
            ("expires_at_ms", "ALTER TABLE memories ADD COLUMN expires_at_ms INTEGER"),
        ] {
            if !Self::has_column(conn, "memories", column)? {
                conn.execute_batch(ddl)?;
            }
        }
        conn.execute(
            "UPDATE memories SET recorded_at_ms = created_at_ms WHERE recorded_at_ms = 0",
            [],
        )?;
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_memories_recorded_at ON memories(recorded_at_ms);
             CREATE INDEX IF NOT EXISTS idx_memories_expiry ON memories(expires_at_ms);",
        )?;
        let version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if version < 4 {
            conn.execute("UPDATE memories SET indexed = 1", [])?;
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
            conn.execute_batch(&format!("DROP TABLE metrics; {METRICS_DDL}"))?;
        }
        if !Self::has_column(conn, "fact_versions", "recorded_at_ms")? {
            conn.execute_batch("ALTER TABLE fact_versions ADD COLUMN recorded_at_ms INTEGER;")?;
        }
        conn.execute(
            "UPDATE fact_versions SET recorded_at_ms = timestamp_ms WHERE recorded_at_ms IS NULL",
            [],
        )?;
        for (column, ddl) in [
            (
                "source_turn_index",
                "ALTER TABLE memory_cards ADD COLUMN source_turn_index INTEGER NOT NULL DEFAULT 0",
            ),
            (
                "document_time",
                "ALTER TABLE memory_cards ADD COLUMN document_time INTEGER NOT NULL DEFAULT 0",
            ),
            (
                "conversation_time",
                "ALTER TABLE memory_cards ADD COLUMN conversation_time INTEGER NOT NULL DEFAULT 0",
            ),
            ("event_time", "ALTER TABLE memory_cards ADD COLUMN event_time INTEGER"),
        ] {
            if !Self::has_column(conn, "memory_cards", column)? {
                conn.execute_batch(ddl)?;
            }
        }
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_fact_versions_memory ON fact_versions(memory_id);
             CREATE INDEX IF NOT EXISTS idx_fact_versions_order
                 ON fact_versions(fact_key, entity_id, timestamp_ms, recorded_at_ms);",
        )?;

        if version < 5 {
            let tx = conn.unchecked_transaction()?;
            let mut last_rowid = 0_i64;
            loop {
                let rows: Vec<(i64, String, String, String)> = {
                    let mut stmt = tx.prepare_cached(
                        "SELECT rowid, content, entity_id, kind FROM memories
                         WHERE rowid > ?1 ORDER BY rowid LIMIT 10000",
                    )?;
                    let mapped = stmt.query_map(params![last_rowid], |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                    })?;
                    mapped.collect::<rusqlite::Result<Vec<_>>>()?
                };
                if rows.is_empty() {
                    break;
                }
                let mut update =
                    tx.prepare_cached("UPDATE memories SET content_hash = ?1 WHERE rowid = ?2")?;
                for (rowid, content, entity_id, kind) in &rows {
                    update.execute(params![
                        content_hash(content, entity_id, MemoryKind::parse(kind)),
                        rowid,
                    ])?;
                }
                last_rowid = rows.last().expect("batch is non-empty").0;
            }
            tx.commit()?;
        }

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

        if version < 3 {
            // The v3 FTS rebuild supersedes the v2 rowid rewrite because it
            // drops the table and re-ingest restores its contents.
            tracing::warn!("dropping FTS index for the entity-token schema; re-ingest to restore");
            let tx = conn.unchecked_transaction()?;
            tx.execute_batch(
                "DROP TABLE IF EXISTS fts_memories;
                 CREATE VIRTUAL TABLE fts_memories USING fts5(
                     memory_id UNINDEXED,
                     entity_id UNINDEXED,
                     entity_tok,
                     content,
                     tokenize='porter unicode61'
                 );",
            )?;
            tx.commit()?;
        }
        if version < SCHEMA_VERSION {
            conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION};"))?;
        }
        Ok(())
    }
}
