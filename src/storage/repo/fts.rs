use super::prelude::*;

impl TenantStore {
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
}
