use super::prelude::*;

impl TenantStore {
    pub fn fact_versions_for_memories(
        &self,
        memory_ids: &[String],
    ) -> Result<HashMap<String, FactVersionRow>> {
        if memory_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let conn = self.get_conn()?;
        let mut rows_by_memory: HashMap<String, (String, FactVersionRow)> = HashMap::new();
        for chunk in memory_ids.chunks(IN_CHUNK) {
            let values = padded_in_chunk(chunk);
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
                 WHERE v.memory_id IN ({})
                    OR m.parent_memory_id IN ({})
                 ORDER BY v.valid_from_ms, v.memory_id",
                in_placeholders(IN_CHUNK),
                in_placeholders(IN_CHUNK)
            ))?;
            let params = values.iter().chain(values.iter());
            let mapped = stmt.query_map(rusqlite::params_from_iter(params), |row| {
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

    pub fn canonicalize_predicates(
        &self,
        entity_id: &str,
        predicates: &[(String, Vec<f32>)],
        tau: f32,
    ) -> Result<HashMap<String, String>> {
        if predicates.is_empty() {
            return Ok(HashMap::new());
        }
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let assigned = Self::canonicalize_predicates_tx(&tx, entity_id, predicates, tau)?;
        tx.commit()?;
        Ok(assigned)
    }

    pub(crate) fn canonicalize_predicates_tx(
        tx: &rusqlite::Transaction<'_>,
        entity_id: &str,
        predicates: &[(String, Vec<f32>)],
        tau: f32,
    ) -> Result<HashMap<String, String>> {
        let mut assigned = HashMap::with_capacity(predicates.len());
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
            match known.query_row(params![entity_id, predicate], |row| row.get::<_, String>(0)) {
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
            let stored_embedding = (canonical == *predicate).then(|| vec_f32_to_bytes(embedding));
            insert.execute(params![entity_id, predicate, canonical, stored_embedding])?;
            assigned.insert(predicate.clone(), canonical);
        }
        Ok(assigned)
    }

    #[allow(clippy::too_many_lines)]
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

    #[allow(clippy::too_many_lines)]
    pub(crate) fn register_fact_versions_tx(
        tx: &rusqlite::Transaction<'_>,
        entity_id: &str,
        registrations: &[(&str, u64, &str, &str, &str, &str)],
        recorded_at: u64,
    ) -> Result<Vec<FactVersionStatus>> {
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

        Ok(statuses)
    }

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
            Ok(None) | Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn invalidated_set(
        &self,
        memory_ids: &[String],
    ) -> Result<std::collections::HashSet<String>> {
        self.invalidated_among(memory_ids, "status = 'stale'", None)
    }

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

    fn invalidated_among(
        &self,
        memory_ids: &[String],
        condition: &str,
        point_in_time_ms: Option<u64>,
    ) -> Result<std::collections::HashSet<String>> {
        let mut set = std::collections::HashSet::new();
        if memory_ids.is_empty() {
            return Ok(set);
        }
        let conn = self.get_conn()?;
        for chunk in memory_ids.chunks(IN_CHUNK) {
            let values = padded_in_chunk(chunk);
            let sql = format!(
                "SELECT DISTINCT memory_id FROM fact_versions WHERE {condition} AND memory_id IN ({})",
                in_placeholders(IN_CHUNK)
            );
            let mut stmt = conn.prepare_cached(&sql)?;
            let mut params: Vec<rusqlite::types::Value> = Vec::with_capacity(IN_CHUNK + 1);
            if let Some(pit) = point_in_time_ms {
                params.push(rusqlite::types::Value::Integer(pit as i64));
            }
            params.extend(
                values.into_iter().map(|memory_id| rusqlite::types::Value::Text(memory_id.clone())),
            );
            let rows =
                stmt.query_map(rusqlite::params_from_iter(params), |row| row.get::<_, String>(0))?;
            for row in rows {
                set.insert(row?);
            }
        }
        Ok(set)
    }
}
