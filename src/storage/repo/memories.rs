use super::prelude::*;

impl TenantStore {
    pub fn insert_observations_batch(
        &self,
        items: &[(u64, String, AgentObservation)],
    ) -> Result<Vec<Option<u64>>> {
        self.allocate_vector_ids(items.iter().map(|(timestamp, memory_id, observation)| {
            (*timestamp, memory_id.as_str(), observation)
        }))
    }

    pub fn insert_observation(
        &self,
        timestamp: u64,
        memory_id: &str,
        obs: &AgentObservation,
    ) -> Result<()> {
        self.allocate_vector_ids(std::iter::once((timestamp, memory_id, obs)))?;
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
        self.lookup_by_vector_ids_batch_at(vector_ids, None, None)
    }

    pub fn lookup_by_vector_ids_batch_at(
        &self,
        vector_ids: &[u64],
        point_in_time_ms: Option<u64>,
        known_as_of_ms: Option<u64>,
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
                "SELECT v.vector_id, v.memory_id, v.timestamp_ms
                 FROM vector_lookup v
                 JOIN memories m ON m.memory_id = v.memory_id
                 WHERE v.vector_id IN ({})
                   AND (?501 IS NULL OR v.timestamp_ms <= ?501)
                   AND (?502 IS NULL OR m.recorded_at_ms <= ?502)",
                in_placeholders(IN_CHUNK)
            );
            let mut stmt = conn.prepare_cached(&sql)?;
            let params = values
                .into_iter()
                .map(|value| rusqlite::types::Value::Integer(*value as i64))
                .chain([
                    point_in_time_ms.map_or(rusqlite::types::Value::Null, |value| {
                        rusqlite::types::Value::Integer(value as i64)
                    }),
                    known_as_of_ms.map_or(rusqlite::types::Value::Null, |value| {
                        rusqlite::types::Value::Integer(value as i64)
                    }),
                ]);
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
            "SELECT entity_id, content, kind, created_at_ms, session_id, turn_index, role,
                    parent_memory_id, recorded_at_ms, expires_at_ms
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
                recorded_at_ms: row.get(8)?,
                expires_at_ms: row.get::<_, Option<i64>>(9)?.map(|value| value as u64),
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
                "SELECT memory_id, entity_id, content, kind, created_at_ms, session_id, turn_index, role,
                        parent_memory_id, recorded_at_ms, expires_at_ms
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
                        recorded_at_ms: row.get::<_, i64>(9)? as u64,
                        expires_at_ms: row.get::<_, Option<i64>>(10)?.map(|value| value as u64),
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

    pub fn stored_content_states(
        &self,
        memory_ids: &[String],
    ) -> Result<HashMap<String, (String, i64)>> {
        let mut out = HashMap::with_capacity(memory_ids.len());
        let conn = self.get_conn()?;
        for chunk in memory_ids.chunks(IN_CHUNK) {
            let values = padded_in_chunk(chunk);
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT memory_id, content_hash, indexed FROM memories WHERE memory_id IN ({})",
                in_placeholders(IN_CHUNK)
            ))?;
            let rows = stmt.query_map(rusqlite::params_from_iter(values), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?))
            })?;
            for row in rows {
                let (id, hash, indexed) = row?;
                out.insert(id, (hash, indexed));
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
        for chunk in hashes.chunks(IN_CHUNK) {
            let values = padded_in_chunk(chunk);
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT DISTINCT content_hash FROM memories WHERE content_hash IN ({})",
                in_placeholders(IN_CHUNK)
            ))?;
            let rows = stmt.query_map(rusqlite::params_from_iter(values), |row| row.get(0))?;
            for row in rows {
                found.insert(row?);
            }
        }
        Ok(found)
    }

    #[allow(clippy::too_many_lines)]
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
        let tombstone_id_val = format!("tombstone::{memory_id}");
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
        let mut fts_removed =
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
        let chunk_prefix = format!("{memory_id}::");
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
        let affected_facts: Vec<(String, String)> = {
            let mut stmt = tx.prepare(
                "SELECT DISTINCT fact_key, entity_id FROM fact_versions
                 WHERE memory_id = ?1 OR substr(memory_id, 1, length(?2)) = ?2",
            )?;
            let rows = stmt.query_map(params![memory_id, chunk_prefix], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        let mut affected_routers = std::collections::HashSet::new();
        {
            let mut stmt =
                tx.prepare("SELECT session_id, entity_id, record_json FROM session_router")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
            })?;
            for row in rows {
                let (session_id, router_entity_id, json) = row?;
                let matches =
                    serde_json::from_str::<SessionRouterRecord>(&json).ok().is_some_and(|record| {
                        record
                            .source_memory_ids
                            .iter()
                            .any(|source| source == memory_id || source.starts_with(&chunk_prefix))
                    });
                if matches {
                    affected_routers.insert((session_id, router_entity_id));
                }
            }
        }
        {
            let mut stmt = tx.prepare(
                "SELECT DISTINCT session_id, entity_id FROM session_router_sources
                 WHERE memory_id = ?1 OR substr(memory_id, 1, length(?2)) = ?2",
            )?;
            let rows = stmt.query_map(params![memory_id, chunk_prefix], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            for row in rows {
                affected_routers.insert(row?);
            }
        }
        tx.execute(
            "DELETE FROM memory_cards WHERE source_memory_id = ?1 OR substr(card_id, 1, length(?2)) = ?2",
            params![memory_id, chunk_prefix],
        )?;
        tx.execute(
            "DELETE FROM preferences
             WHERE memory_id = ?1 OR substr(memory_id, 1, length(?2)) = ?2",
            params![memory_id, chunk_prefix],
        )?;
        for table in
            ["metrics", "negative_centroids", "disambiguation_vectors", "consolidation_queue"]
        {
            tx.execute(
                &format!(
                    "DELETE FROM {table}
                     WHERE memory_id = ?1 OR substr(memory_id, 1, length(?2)) = ?2"
                ),
                params![memory_id, chunk_prefix],
            )?;
        }
        tx.execute(
            "DELETE FROM memory_links
             WHERE source_memory_id = ?1 OR target_memory_id = ?1
                OR substr(source_memory_id, 1, length(?2)) = ?2
                OR substr(target_memory_id, 1, length(?2)) = ?2",
            params![memory_id, chunk_prefix],
        )?;
        let graph_edges_removed = tx.execute(
            "DELETE FROM edges
             WHERE memory_id = ?1 OR ref_source = ?1 OR ref_target = ?1
                OR substr(memory_id, 1, length(?2)) = ?2
                OR substr(COALESCE(ref_source, ''), 1, length(?2)) = ?2
                OR substr(COALESCE(ref_target, ''), 1, length(?2)) = ?2",
            params![memory_id, chunk_prefix],
        )?;
        tx.execute(
            "DELETE FROM fact_evidence
             WHERE memory_id = ?1 OR version_memory_id = ?1
                OR substr(memory_id, 1, length(?2)) = ?2
                OR substr(version_memory_id, 1, length(?2)) = ?2",
            params![memory_id, chunk_prefix],
        )?;
        tx.execute(
            "DELETE FROM fact_versions
             WHERE memory_id = ?1 OR substr(memory_id, 1, length(?2)) = ?2",
            params![memory_id, chunk_prefix],
        )?;
        for (fact_key, fact_entity_id) in affected_facts {
            rebuild_fact_chain_after_delete(&tx, &fact_key, &fact_entity_id)?;
        }
        tx.execute(
            "DELETE FROM session_router_sources
             WHERE memory_id = ?1 OR substr(memory_id, 1, length(?2)) = ?2",
            params![memory_id, chunk_prefix],
        )?;
        remove_deleted_from_core_profiles(&tx, memory_id, &chunk_prefix)?;
        for (session_id, router_entity_id) in affected_routers {
            rebuild_router_after_delete(&tx, &session_id, &router_entity_id)?;
        }
        let mut chunk_vector_ids = Vec::new();
        for (chunk_id, chunk_vector_id) in &chunks {
            fts_removed += tx.execute(
                "DELETE FROM fts_memories WHERE rowid = ?1",
                params![fts_rowid(chunk_id)],
            )?;
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
            fts_removed,
            graph_edges_removed,
        })
    }
}

fn remove_deleted_from_core_profiles(
    tx: &rusqlite::Transaction<'_>,
    memory_id: &str,
    derived_prefix: &str,
) -> Result<()> {
    let profiles = {
        let mut stmt = tx.prepare_cached("SELECT entity_id, profile_json FROM core_profiles")?;
        let rows =
            stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    let mut update = tx.prepare_cached(
        "UPDATE core_profiles SET profile_json = ?1, updated_at_ms = ?2 WHERE entity_id = ?3",
    )?;
    for (entity_id, profile_json) in profiles {
        let Ok(mut profile) = serde_json::from_str::<serde_json::Value>(&profile_json) else {
            continue;
        };
        let Some(facts) = profile.get_mut("facts").and_then(serde_json::Value::as_array_mut) else {
            continue;
        };
        let original_len = facts.len();
        facts.retain(|fact| match fact.get("memory_id").and_then(serde_json::Value::as_str) {
            Some(id) => id != memory_id && !id.starts_with(derived_prefix),
            None => true,
        });
        if facts.len() != original_len {
            update.execute(params![
                serde_json::to_string(&profile)?,
                unix_timestamp_ms()?,
                entity_id
            ])?;
        }
    }
    Ok(())
}

fn rebuild_fact_chain_after_delete(
    tx: &rusqlite::Transaction<'_>,
    fact_key: &str,
    entity_id: &str,
) -> Result<()> {
    let chain: Vec<(String, u64)> = {
        let mut stmt = tx.prepare_cached(
            "SELECT memory_id, timestamp_ms FROM fact_versions
             WHERE fact_key = ?1 AND entity_id = ?2
             ORDER BY timestamp_ms, COALESCE(recorded_at_ms, timestamp_ms), rowid",
        )?;
        let rows = stmt.query_map(params![fact_key, entity_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
        })?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    let mut update = tx.prepare_cached(
        "UPDATE fact_versions
         SET status = ?1, valid_from_ms = ?2, valid_to_ms = ?3,
             superseded_by = ?4, supersedes = ?5
         WHERE fact_key = ?6 AND entity_id = ?7 AND memory_id = ?8",
    )?;
    for (index, (memory_id, timestamp)) in chain.iter().enumerate() {
        let previous = index.checked_sub(1).and_then(|value| chain.get(value));
        let next = chain.get(index + 1);
        update.execute(params![
            if next.is_none() { "current" } else { "stale" },
            *timestamp as i64,
            next.map(|(_, value)| *value as i64),
            next.map(|(id, _)| id.as_str()),
            previous.map(|(id, _)| id.as_str()),
            fact_key,
            entity_id,
            memory_id
        ])?;
    }
    let mut rows = tx.prepare_cached(
        "SELECT memory_id, COALESCE(subject, ''), COALESCE(predicate, ''),
                COALESCE(object, ''), status, timestamp_ms, superseded_by, supersedes
         FROM fact_versions WHERE fact_key = ?1 AND entity_id = ?2",
    )?;
    let facts = rows
        .query_map(params![fact_key, entity_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, Option<String>>(7)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut upsert = tx.prepare_cached(
        "INSERT OR REPLACE INTO edges
            (edge_id, source, target, edge_type, label, status, ref_source,
             ref_target, timestamp_ms, memory_id, weight)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
    )?;
    for (memory_id, subject, predicate, object, status, timestamp, next, previous) in facts {
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            continue;
        }
        let (ref_source, ref_target) = if status == "current" {
            (previous.as_ref().map(|_| crate::graph::EdgeType::Supersedes.as_str()), previous)
        } else {
            (next.as_ref().map(|_| crate::graph::EdgeType::SupersededBy.as_str()), next)
        };
        upsert.execute(params![
            format!("edge::{memory_id}::{subject}::{predicate}"),
            subject,
            object,
            predicate,
            format!("{subject} {predicate} {object}"),
            status,
            ref_source,
            ref_target,
            timestamp,
            memory_id,
            f64::from(crate::graph::EdgeType::from_str(&predicate).default_weight())
        ])?;
    }
    Ok(())
}

fn rebuild_router_after_delete(
    tx: &rusqlite::Transaction<'_>,
    session_id: &str,
    entity_id: &str,
) -> Result<()> {
    let fragments: Vec<SessionRouterRecord> = {
        let mut stmt = tx.prepare_cached(
            "SELECT record_json FROM session_router_sources
             WHERE session_id = ?1 AND entity_id = ?2 ORDER BY memory_id",
        )?;
        let rows = stmt.query_map(params![session_id, entity_id], |row| row.get::<_, String>(0))?;
        rows.filter_map(|row| row.ok().and_then(|json| serde_json::from_str(&json).ok())).collect()
    };
    let router_fts_rowid = fts_rowid(&format!("router::{entity_id}::{session_id}"));
    let router_id = crate::core::memory_id::MemoryId::new(entity_id, session_id, 0)
        .derived(crate::core::memory_id::Tag::Named("router".to_string()))
        .as_str()
        .to_string();
    if fragments.is_empty() {
        tx.execute(
            "DELETE FROM fts_session_router WHERE session_id = ?1 AND entity_id = ?2",
            params![session_id, entity_id],
        )?;
        tx.execute(
            "DELETE FROM session_router WHERE session_id = ?1 AND entity_id = ?2",
            params![session_id, entity_id],
        )?;
        tx.execute("DELETE FROM fts_memories WHERE rowid = ?1", params![fts_rowid(&router_id)])?;
        return Ok(());
    }
    let mut fragments = fragments.into_iter();
    let mut merged = fragments.next().expect("fragments is non-empty");
    for fragment in fragments {
        merged = merge_router_records(&merged, &fragment);
    }
    merged.router_text = build_session_router_text(&merged);
    let json = serde_json::to_string(&merged)?;
    tx.query_row(
        "INSERT INTO session_router
            (session_id, entity_id, record_json, router_text, created_at_ms, updated_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(session_id, entity_id) DO UPDATE SET
            record_json = excluded.record_json,
            router_text = excluded.router_text,
            updated_at_ms = excluded.updated_at_ms
         RETURNING rowid",
        params![
            session_id,
            entity_id,
            json,
            merged.router_text,
            merged.created_at_ms,
            merged.updated_at_ms
        ],
        |row| row.get::<_, i64>(0),
    )?;
    tx.execute(
        "DELETE FROM fts_session_router WHERE session_id = ?1 AND entity_id = ?2",
        params![session_id, entity_id],
    )?;
    tx.execute(
        "INSERT OR REPLACE INTO fts_session_router
            (rowid, session_id, entity_id, router_text) VALUES (?1, ?2, ?3, ?4)",
        params![router_fts_rowid, session_id, entity_id, merged.router_text],
    )?;
    tx.execute(
        "INSERT OR REPLACE INTO fts_memories
            (rowid, memory_id, entity_id, entity_tok, content) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            fts_rowid(&router_id),
            router_id,
            entity_id,
            fts_entity_tok(entity_id),
            merged.router_text
        ],
    )?;
    Ok(())
}

/// Every place a deleted token could still be hiding, found by walking the
/// live schema: user tables plus the FTS5 shadow tables, every column cast to
/// text. Returns `(table, column, rows)` for each column that still matches,
/// so a failure names the leak instead of just asserting a count.
#[cfg(test)]
fn residue_sweep(conn: &rusqlite::Connection, token: &str) -> Vec<(String, String, i64)> {
    let pattern = format!("%{token}%");
    let mut tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
        .and_then(|mut stmt| {
            stmt.query_map([], |row| row.get::<_, String>(0))?.collect::<rusqlite::Result<_>>()
        })
        .unwrap_or_default();
    tables.sort();

    let mut found = Vec::new();
    for table in tables {
        let columns: Vec<String> = match conn.prepare(&format!("PRAGMA table_info(\"{table}\")")) {
            Ok(mut stmt) => stmt
                .query_map([], |row| row.get::<_, String>(1))
                .and_then(std::iter::Iterator::collect)
                .unwrap_or_default(),
            Err(_) => continue,
        };
        for column in columns {
            // Shadow tables store blobs; CAST lets one probe cover both. A
            // column that cannot be scanned at all is skipped rather than
            // failing the sweep.
            let sql = format!(
                "SELECT COUNT(*) FROM \"{table}\" WHERE CAST(\"{column}\" AS TEXT) LIKE ?1"
            );
            if let Ok(count) = conn.query_row(&sql, params![pattern], |row| row.get::<_, i64>(0)) {
                if count > 0 {
                    found.push((table.clone(), column, count));
                }
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::repo::ingest::{
        IngestBatches, IngestConsolidationTask, IngestFactRegistration,
    };

    #[test]
    #[allow(clippy::too_many_lines)]
    fn deleted_memory_leaves_no_text_anywhere() {
        let dir = tempfile::tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("tenant.db")).unwrap();
        let token = "deletion_secret_token";
        let parent = "alice::s1::1";
        let child = "alice::s1::1::fact0";
        let observation = |text: &str, parent_memory_id: Option<&str>| AgentObservation {
            entity_id: "alice".into(),
            textual_content: text.into(),
            kind: MemoryKind::Fact,
            content_hash: text.into(),
            created_at_ms: 100,
            session_id: "s1".into(),
            turn_index: 1,
            parent_memory_id: parent_memory_id.map(str::to_string),
            ..Default::default()
        };
        let card = MemoryCard {
            card_id: child.into(),
            entity_id: "alice".into(),
            user_id: "alice".into(),
            source_memory_id: parent.into(),
            source_session_id: "s1".into(),
            source_turn_index: 1,
            document_time: 100,
            conversation_time: 100,
            event_time: Some(100),
            subject: "alice".into(),
            predicate: "stores".into(),
            object: token.into(),
            memory_text: token.into(),
            card_type: "fact".into(),
            confidence: 1.0,
            is_latest: true,
            is_static: false,
            is_inference: false,
            expires_at: None,
            root_card_id: Some(parent.into()),
            parent_card_id: Some(parent.into()),
            lifecycle: None,
            created_at_ms: 100,
            updated_at_ms: 100,
        };
        let router = SessionRouterRecord {
            session_id: "s1".into(),
            entity_id: "alice".into(),
            canonical_facts: vec![token.into()],
            source_memory_ids: vec![parent.into()],
            router_text: format!("facts {token}"),
            created_at_ms: 100,
            updated_at_ms: 100,
            ..Default::default()
        };
        let batches = IngestBatches {
            observations: vec![
                (50, "alice::s1::0".into(), observation("Austin", None)),
                (100, parent.into(), observation(token, None)),
                (100, child.into(), observation(token, Some(parent))),
            ],
            fts_batch: vec![
                (parent.into(), "alice".into(), token.into()),
                (child.into(), "alice".into(), token.into()),
            ],
            memory_card_batch: vec![card],
            session_router_updates: vec![router],
            preference_batch: HashMap::from([("alice".into(), vec![(parent.into(), 1.0)])]),
            memory_links_batch: vec![(parent.into(), child.into(), "derived_variant".into())],
            fact_batch: vec![
                IngestFactRegistration {
                    entity_id: "alice".into(),
                    fact_key: "location".into(),
                    timestamp: 50,
                    memory_id: "alice::s1::0".into(),
                    subject: "alice".into(),
                    predicate: "lives_in".into(),
                    object: "Austin".into(),
                },
                IngestFactRegistration {
                    entity_id: "alice".into(),
                    fact_key: "location".into(),
                    timestamp: 100,
                    memory_id: child.into(),
                    subject: "alice".into(),
                    predicate: "lives_in".into(),
                    object: token.into(),
                },
            ],
            consolidation_tasks: vec![IngestConsolidationTask {
                entity_id: "alice".into(),
                memory_id: parent.into(),
                timestamp: 100,
                textual_content: token.into(),
            }],
            ..Default::default()
        };
        let outcome = store.commit_ingest(&batches).unwrap();
        store.fts_index_batch(&outcome.fts_batch).unwrap();
        store
            .set_core_profile(
                "alice",
                &serde_json::json!({
                    "facts": [{"memory_id": child, "text": token, "timestamp_ms": 100}]
                })
                .to_string(),
            )
            .unwrap();

        // Control: the sweep must be able to see the token while it is still
        // there. Without this, a broken probe and a clean delete are
        // indistinguishable -- both report nothing.
        let planted = {
            let conn = store.get_conn().unwrap();
            residue_sweep(&conn, token)
        };
        assert!(
            planted.len() >= 5,
            "probe found the planted token in only {} place(s); it is not searching what it should: {:?}",
            planted.len(),
            planted
        );

        store.delete_observation(100, parent, "test").unwrap();

        let conn = store.get_conn().unwrap();
        let edge_residue: Vec<(String, String, String)> = conn
            .prepare("SELECT memory_id, label, target FROM edges WHERE label LIKE ?1")
            .unwrap()
            .query_map(params![format!("%{token}%")], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert!(edge_residue.is_empty(), "edge residue: {edge_residue:?}");
        // Sweep every column the schema actually has rather than a list
        // written by hand: a hand-kept list silently stops covering each new
        // table, and "we checked ten tables" is not the claim -- "the token is
        // nowhere in the file" is.
        let residue = residue_sweep(&conn, token);
        assert!(
            residue.is_empty(),
            "deleted text still present in {} place(s): {:?}",
            residue.len(),
            residue
        );
        let evidence: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM fact_evidence WHERE memory_id = ?1 OR version_memory_id = ?1",
                params![child],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(evidence, 0);
        assert_eq!(
            store.get_current_fact_value("alice", "location").unwrap().as_deref(),
            Some("Austin")
        );
    }
}
