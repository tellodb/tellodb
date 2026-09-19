use super::prelude::*;

impl TenantStore {
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
        let mut lookup: std::collections::HashMap<u64, (u64, String)> =
            std::collections::HashMap::with_capacity(vector_ids.len());
        for chunk in vector_ids.chunks(IN_CHUNK) {
            let values = padded_in_chunk(chunk);
            let sql = format!(
                "SELECT vector_id, memory_id, timestamp_ms FROM vector_lookup WHERE vector_id IN ({})",
                in_placeholders(IN_CHUNK)
            );
            let mut stmt = conn.prepare_cached(&sql)?;
            let params = values.into_iter().map(|value| *value as i64);
            let rows = stmt.query_map(rusqlite::params_from_iter(params), |row| {
                Ok((
                    row.get::<_, i64>(0)? as u64,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)? as u64,
                ))
            })?;
            for row in rows {
                let (vid, memory_id, ts) = row?;
                lookup.insert(vid, (ts, memory_id));
            }
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
        for chunk in memory_ids.chunks(IN_CHUNK) {
            let values = padded_in_chunk(chunk);
            let sql = format!(
                "SELECT m.memory_id, v.vector_id, m.created_at_ms
                 FROM memories m
                 LEFT JOIN vector_lookup v ON v.memory_id = m.memory_id
                 WHERE m.memory_id IN ({})",
                in_placeholders(IN_CHUNK)
            );
            let mut stmt = conn.prepare_cached(&sql)?;
            let rows = stmt.query_map(rusqlite::params_from_iter(values), |row| {
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
        }
        Ok(result)
    }

    pub fn memory_identity_batch(
        &self,
        memory_ids: &[String],
    ) -> Result<HashMap<String, (String, u32)>> {
        let mut out = HashMap::with_capacity(memory_ids.len());
        if memory_ids.is_empty() {
            return Ok(out);
        }
        let conn = self.get_conn()?;
        for chunk in memory_ids.chunks(IN_CHUNK) {
            let values = padded_in_chunk(chunk);
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT memory_id, session_id, turn_index FROM memories WHERE memory_id IN ({})",
                in_placeholders(IN_CHUNK)
            ))?;
            let rows = stmt.query_map(rusqlite::params_from_iter(values), |row| {
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
                kind: MemoryKind::parse(row.get::<_, String>(2)?.as_str()),
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
        let conn = self.get_conn()?;
        let mut result = std::collections::HashMap::new();
        for chunk in keys.chunks(IN_CHUNK) {
            let memory_ids: Vec<&str> = chunk.iter().map(|(_, mid)| mid.as_str()).collect();
            let values = padded_in_chunk(&memory_ids);
            let sql = format!(
                "SELECT memory_id, entity_id, content, kind, created_at_ms, session_id, turn_index, role, parent_memory_id
                 FROM memories WHERE memory_id IN ({})",
                in_placeholders(IN_CHUNK)
            );
            let mut stmt = conn.prepare_cached(&sql)?;
            let rows = stmt.query_map(rusqlite::params_from_iter(values), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    AgentObservation {
                        entity_id: row.get::<_, String>(1)?,
                        textual_content: row.get::<_, String>(2)?,
                        embedding: Vec::new(),
                        kind: MemoryKind::parse(row.get::<_, String>(3)?.as_str()),
                        content_hash: String::new(),
                        created_at_ms: row.get::<_, i64>(4)? as u64,
                        session_id: row.get(5)?,
                        turn_index: row.get(6)?,
                        role: row.get(7)?,
                        parent_memory_id: row.get(8)?,
                    },
                ))
            })?;
            for row in rows {
                let (memory_id, obs) = row?;
                result.insert(memory_id, obs);
            }
        }
        Ok(result)
    }

    pub fn stored_content_hashes(&self, memory_ids: &[String]) -> Result<HashMap<String, String>> {
        let mut out = HashMap::with_capacity(memory_ids.len());
        let conn = self.get_conn()?;
        for chunk in memory_ids.chunks(IN_CHUNK) {
            let values = padded_in_chunk(chunk);
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT memory_id, content_hash FROM memories WHERE memory_id IN ({})",
                in_placeholders(IN_CHUNK)
            ))?;
            let rows = stmt.query_map(rusqlite::params_from_iter(values), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            for row in rows {
                let (id, hash) = row?;
                out.insert(id, hash);
            }
        }
        Ok(out)
    }

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
}
