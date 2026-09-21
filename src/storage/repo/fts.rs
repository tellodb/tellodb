use super::prelude::*;

fn ensure_fts_rowid_available(
    conn: &rusqlite::Connection,
    rowid: i64,
    memory_id: &str,
) -> Result<()> {
    let existing = match conn.query_row(
        "SELECT memory_id FROM fts_memories WHERE rowid = ?1",
        params![rowid],
        |row| row.get::<_, String>(0),
    ) {
        Ok(memory_id) => Some(memory_id),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(error) => return Err(error.into()),
    };
    if existing.as_deref().is_some_and(|value| value != memory_id) {
        anyhow::bail!("FTS rowid collision for memory {memory_id}");
    }
    Ok(())
}

impl TenantStore {
    pub fn fts_search(
        &self,
        query: &str,
        limit: usize,
        entity_id: Option<&str>,
    ) -> Result<Vec<(String, f32)>> {
        self.fts_search_at(query, limit, entity_id, None, None)
    }

    pub fn fts_search_at(
        &self,
        query: &str,
        limit: usize,
        entity_id: Option<&str>,
        point_in_time_ms: Option<u64>,
        known_as_of_ms: Option<u64>,
    ) -> Result<Vec<(String, f32)>> {
        let conn = self.get_conn()?;

        let cleaned = query.replace(|c: char| !c.is_alphanumeric() && c != ' ', " ");
        let mut terms: Vec<String> = cleaned
            .split_whitespace()
            .filter(|t| t.len() > FTS_MIN_TERM_LEN)
            .map(str::to_lowercase)
            .filter(|t| !crate::core::text::is_low_signal_keyword(t))
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
            "SELECT f.memory_id, bm25(fts_memories) as score
             FROM fts_memories f
             LEFT JOIN memories m ON m.memory_id = f.memory_id
             WHERE fts_memories MATCH ?1
               AND (?3 IS NULL OR m.created_at_ms <= ?3)
               AND (?4 IS NULL OR m.recorded_at_ms <= ?4)
             ORDER BY score LIMIT ?2",
        )?;
        let results = stmt
            .query_map(
                params![
                    fts_query,
                    limit as i64,
                    point_in_time_ms.map(|value| value as i64),
                    known_as_of_ms.map(|value| value as i64)
                ],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32)),
            )?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(results)
    }

    pub fn fts_index_text(&self, memory_id: &str, content: &str, entity_id: &str) -> Result<()> {
        let conn = self.get_conn()?;
        ensure_fts_rowid_available(&conn, fts_rowid(memory_id), memory_id)?;
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
                ensure_fts_rowid_available(&tx, fts_rowid(memory_id), memory_id)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_to_overwrite_a_colliding_fts_rowid() {
        let temp = tempfile::tempdir().unwrap();
        let store = TenantStore::new(&temp.path().join("tenant.db")).unwrap();
        let conn = store.get_conn().unwrap();
        conn.execute(
            "INSERT INTO fts_memories (rowid, memory_id, entity_id, entity_tok, content)
             VALUES (?1, 'other', 'alice', 'alice', 'original')",
            params![fts_rowid("target")],
        )
        .unwrap();
        drop(conn);

        assert!(store.fts_index_text("target", "replacement", "alice").is_err());
        let conn = store.get_conn().unwrap();
        let stored: String = conn
            .query_row(
                "SELECT memory_id FROM fts_memories WHERE rowid = ?1",
                params![fts_rowid("target")],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stored, "other");
    }
}
