use super::prelude::*;

type FactPosition = (String, u64, u64, i64, String);

/// Pad a chunk of (entity_id, fact_key, version_memory_id) triples up to
/// IN_CHUNK by repeating the last one, mirroring `padded_in_chunk` above —
/// a fixed-shape query lets `prepare_cached` actually reuse the statement.
/// Repeated triples are harmless: `IN (...)` is a set-membership test, so a
/// duplicated triple cannot make a row match (and thus appear) twice.
fn padded_triples(chunk: &[(String, String, String)]) -> Vec<(String, String, String)> {
    let Some(last) = chunk.last().cloned() else {
        return Vec::new();
    };
    let mut padded = Vec::with_capacity(IN_CHUNK);
    padded.extend(chunk.iter().cloned());
    padded.resize(IN_CHUNK, last);
    padded
}

fn fact_neighbour(
    tx: &rusqlite::Transaction<'_>,
    fact_key: &str,
    entity_id: &str,
    condition: &str,
    ordering: &str,
    timestamp: u64,
) -> Result<Option<FactPosition>> {
    let sql = format!(
        "SELECT memory_id, timestamp_ms, recorded_at_ms, rowid, COALESCE(object, '')
         FROM fact_versions
         WHERE fact_key = ?1 AND entity_id = ?2 AND {condition}
         ORDER BY {ordering} LIMIT 1"
    );
    match tx.query_row(&sql, params![fact_key, entity_id, timestamp as i64], |row| {
        Ok((
            row.get(0)?,
            row.get::<_, i64>(1)? as u64,
            row.get::<_, i64>(2)? as u64,
            row.get(3)?,
            row.get(4)?,
        ))
    }) {
        Ok(position) => Ok(Some(position)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn fact_version(
    tx: &rusqlite::Transaction<'_>,
    fact_key: &str,
    memory_id: &str,
) -> Result<FactPosition> {
    Ok(tx.query_row(
        "SELECT memory_id, timestamp_ms, recorded_at_ms, rowid, COALESCE(object, '')
         FROM fact_versions WHERE fact_key = ?1 AND memory_id = ?2",
        params![fact_key, memory_id],
        |row| {
            Ok((
                row.get(0)?,
                row.get::<_, i64>(1)? as u64,
                row.get::<_, i64>(2)? as u64,
                row.get(3)?,
                row.get(4)?,
            ))
        },
    )?)
}

fn current_fact_version(
    tx: &rusqlite::Transaction<'_>,
    fact_key: &str,
    entity_id: &str,
) -> Result<Option<(String, u64)>> {
    match tx.query_row(
        "SELECT memory_id, timestamp_ms FROM fact_versions
         WHERE fact_key = ?1 AND entity_id = ?2 AND status = 'current'
         ORDER BY timestamp_ms DESC, recorded_at_ms DESC, rowid DESC LIMIT 1",
        params![fact_key, entity_id],
        |row| Ok((row.get(0)?, row.get::<_, i64>(1)? as u64)),
    ) {
        Ok(version) => Ok(Some(version)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn ordered_fact_neighbour(
    tx: &rusqlite::Transaction<'_>,
    fact_key: &str,
    entity_id: &str,
    timestamp: u64,
    recorded_at: u64,
    rowid: i64,
    predecessor: bool,
) -> Result<Option<FactPosition>> {
    let (comparison, ordering) = if predecessor { ("<", "DESC") } else { (">", "ASC") };
    let sql = format!(
        "SELECT memory_id, timestamp_ms, recorded_at_ms, rowid, COALESCE(object, '')
         FROM fact_versions
         WHERE fact_key = ?1 AND entity_id = ?2
           AND ((timestamp_ms {comparison} ?3)
             OR (timestamp_ms = ?3 AND recorded_at_ms {comparison} ?4)
             OR (timestamp_ms = ?3 AND recorded_at_ms = ?4 AND rowid {comparison} ?5))
         ORDER BY timestamp_ms {ordering}, recorded_at_ms {ordering}, rowid {ordering} LIMIT 1"
    );
    match tx.query_row(
        &sql,
        params![fact_key, entity_id, timestamp as i64, recorded_at as i64, rowid],
        |row| {
            Ok((
                row.get(0)?,
                row.get::<_, i64>(1)? as u64,
                row.get::<_, i64>(2)? as u64,
                row.get(3)?,
                row.get(4)?,
            ))
        },
    ) {
        Ok(position) => Ok(Some(position)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

impl TenantStore {
    #[allow(clippy::too_many_lines)]
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

        // Evidence per version, newest first. Batched into one IN-chunked
        // query over the (entity_id, fact_key, version_memory_id) triples
        // instead of one prepared-statement round trip per row — that N+1
        // was the dominant cost of the hydrate/factver query stage (see
        // AUDIT-2026-09-21.md, "fact_evidence N+1"). idx_fact_evidence_version
        // covers the lookup.
        let mut evidence_keys: Vec<(String, String, String)> = rows_by_memory
            .values()
            .map(|(version_memory_id, version)| {
                (version.entity_id.clone(), version.fact_key.clone(), version_memory_id.clone())
            })
            .collect();
        evidence_keys.sort();
        evidence_keys.dedup();

        let mut evidence_by_key: HashMap<(String, String, String), Vec<String>> =
            HashMap::with_capacity(evidence_keys.len());
        for chunk in evidence_keys.chunks(IN_CHUNK) {
            let padded = padded_triples(chunk);
            let placeholders =
                std::iter::repeat("(?,?,?)").take(padded.len()).collect::<Vec<_>>().join(",");
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT e.entity_id, e.fact_key, e.version_memory_id,
                        COALESCE(
                            (SELECT em.parent_memory_id FROM memories em
                              WHERE em.memory_id = e.memory_id),
                            e.memory_id
                        )
                 FROM fact_evidence e
                 WHERE (e.entity_id, e.fact_key, e.version_memory_id) IN ({placeholders})
                 ORDER BY e.entity_id, e.fact_key, e.version_memory_id,
                          e.timestamp_ms DESC, e.memory_id"
            ))?;
            let bind_params: Vec<&str> = padded
                .iter()
                .flat_map(|(entity_id, fact_key, version_memory_id)| {
                    [entity_id.as_str(), fact_key.as_str(), version_memory_id.as_str()]
                })
                .collect();
            let mapped = stmt.query_map(rusqlite::params_from_iter(bind_params), |row| {
                Ok((
                    (row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?),
                    row.get::<_, String>(3)?,
                ))
            })?;
            for row in mapped {
                let (key, evidence_memory_id) = row?;
                evidence_by_key.entry(key).or_default().push(evidence_memory_id);
            }
        }

        let mut result = HashMap::with_capacity(rows_by_memory.len());
        for (asked_for, (version_memory_id, mut version)) in rows_by_memory {
            let key = (version.entity_id.clone(), version.fact_key.clone(), version_memory_id);
            version.evidence = evidence_by_key.get(&key).cloned().unwrap_or_default();
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

    pub fn register_fact_versions_batch(
        &self,
        entity_id: &str,
        registrations: &[(&str, u64, &str, &str, &str, &str)],
    ) -> Result<Vec<FactVersionStatus>> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let statuses = Self::register_fact_versions_tx(
            &tx,
            entity_id,
            registrations,
            unix_timestamp_ms()? as u64,
        )?;
        tx.commit()?;
        Ok(statuses)
    }

    /// Marks the newest version of a fact current and clears its supersession
    /// pointers. Used only to repair a chain that ended up with no current row
    /// at all, which the ordering rules are supposed to prevent.
    fn promote_newest_to_current(
        tx: &rusqlite::Transaction<'_>,
        fact_key: &str,
        entity_id: &str,
    ) -> Result<()> {
        let newest: Option<String> = tx
            .query_row(
                "SELECT memory_id FROM fact_versions
                 WHERE fact_key = ?1 AND entity_id = ?2
                 ORDER BY timestamp_ms DESC, recorded_at_ms DESC, rowid DESC
                 LIMIT 1",
                params![fact_key, entity_id],
                |row| row.get(0),
            )
            .ok();
        if let Some(memory_id) = newest {
            tracing::warn!(
                fact_key,
                entity_id,
                memory_id = %memory_id,
                "fact chain had no current version after a batch; promoting the newest"
            );
            tx.execute(
                "UPDATE fact_versions
                 SET status = 'current', valid_to_ms = NULL, superseded_by = NULL
                 WHERE fact_key = ?1 AND entity_id = ?2 AND memory_id = ?3",
                params![fact_key, entity_id, memory_id],
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn register_fact_versions_tx(
        tx: &rusqlite::Transaction<'_>,
        entity_id: &str,
        registrations: &[(&str, u64, &str, &str, &str, &str)],
        recorded_at: u64,
    ) -> Result<Vec<FactVersionStatus>> {
        let mut statuses = Vec::with_capacity(registrations.len());
        // Stale entries whose `current` pointer is resolved after the loop.
        let mut deferred_stale: Vec<(usize, String, String)> = Vec::new();
        for (fact_key, timestamp, memory_id, subject, predicate, object) in registrations {
            let covering = match fact_neighbour(
                tx,
                fact_key,
                entity_id,
                "timestamp_ms <= ?3",
                "timestamp_ms DESC, recorded_at_ms DESC, rowid DESC",
                *timestamp,
            )? {
                Some(position) => Some(position),
                None => fact_neighbour(
                    tx,
                    fact_key,
                    entity_id,
                    "?3 = ?3",
                    "timestamp_ms ASC, recorded_at_ms ASC, rowid ASC",
                    *timestamp,
                )?,
            };
            if let Some((version_id, version_timestamp, _, _, version_object)) = covering {
                if version_id != *memory_id && same_fact_object(&version_object, object) {
                    tx.execute(
                        "INSERT OR IGNORE INTO fact_evidence
                             (fact_key, entity_id, version_memory_id, memory_id, timestamp_ms)
                         VALUES (?1, ?2, ?3, ?4, ?5)",
                        params![fact_key, entity_id, version_id, memory_id, *timestamp as i64],
                    )?;
                    statuses.push(FactVersionStatus::Confirmed {
                        version: (version_timestamp, version_id),
                    });
                    continue;
                }
            }

            let previous_latest = current_fact_version(tx, fact_key, entity_id)?;
            tx.execute(
                "INSERT INTO fact_versions
                     (fact_key, memory_id, entity_id, subject, predicate, object, status,
                      timestamp_ms, valid_from_ms, recorded_at_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'current', ?7, ?7, ?8)
                 ON CONFLICT(fact_key, memory_id) DO NOTHING",
                params![
                    fact_key,
                    memory_id,
                    entity_id,
                    subject,
                    predicate,
                    object,
                    *timestamp as i64,
                    recorded_at as i64
                ],
            )?;
            tx.execute(
                "INSERT OR IGNORE INTO fact_evidence
                     (fact_key, entity_id, version_memory_id, memory_id, timestamp_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![fact_key, entity_id, memory_id, memory_id, *timestamp as i64],
            )?;

            let (_, _, stored_recorded_at, rowid, _) = fact_version(tx, fact_key, memory_id)?;
            let predecessor = ordered_fact_neighbour(
                tx,
                fact_key,
                entity_id,
                *timestamp,
                stored_recorded_at,
                rowid,
                true,
            )?;
            let successor = ordered_fact_neighbour(
                tx,
                fact_key,
                entity_id,
                *timestamp,
                stored_recorded_at,
                rowid,
                false,
            )?;

            tx.execute(
                "UPDATE fact_versions
                 SET status = ?1, valid_from_ms = ?2, valid_to_ms = ?3,
                     superseded_by = ?4, supersedes = ?5
                 WHERE fact_key = ?6 AND memory_id = ?7",
                params![
                    if successor.is_none() { "current" } else { "stale" },
                    *timestamp as i64,
                    successor.as_ref().map(|(_, ts, _, _, _)| *ts as i64),
                    successor.as_ref().map(|(id, _, _, _, _)| id.as_str()),
                    predecessor.as_ref().map(|(id, _, _, _, _)| id.as_str()),
                    fact_key,
                    memory_id
                ],
            )?;
            if let Some((predecessor_id, _, _, _, _)) = &predecessor {
                tx.execute(
                    "UPDATE fact_versions
                     SET status = 'stale', valid_to_ms = ?1, superseded_by = ?2
                     WHERE fact_key = ?3 AND memory_id = ?4",
                    params![*timestamp as i64, memory_id, fact_key, predecessor_id],
                )?;
            }
            if let Some((successor_id, _, _, _, _)) = &successor {
                tx.execute(
                    "UPDATE fact_versions SET supersedes = ?1
                     WHERE fact_key = ?2 AND memory_id = ?3",
                    params![memory_id, fact_key, successor_id],
                )?;
            }

            if successor.is_none() {
                statuses.push(FactVersionStatus::Current {
                    superseded: previous_latest
                        .filter(|(id, _)| id != memory_id)
                        .map(|(id, timestamp)| (timestamp, id)),
                });
            } else {
                // Which version is current cannot be read yet: later rows in
                // this batch may still supersede each other, and the
                // predecessor update above can leave the chain transiently
                // without a `current` row. Resolve after the whole batch has
                // settled instead of querying mid-loop, which made the answer
                // depend on the order registrations happened to arrive in.
                deferred_stale.push((statuses.len(), fact_key.to_string(), entity_id.to_string()));
                statuses.push(FactVersionStatus::Stale { current: (0, String::new()) });
            }
        }

        for (index, fact_key, entity_id) in deferred_stale {
            let current = match current_fact_version(tx, &fact_key, &entity_id)? {
                Some(found) => found,
                None => {
                    // Every row in the chain came out stale. The newest version
                    // is the current one by definition, so promote it rather
                    // than failing the whole ingest batch -- a real LongMemEval
                    // corpus reaches this state, and refusing the write loses
                    // the memory over a bookkeeping disagreement.
                    Self::promote_newest_to_current(tx, &fact_key, &entity_id)?;
                    current_fact_version(tx, &fact_key, &entity_id)?.ok_or_else(|| {
                        anyhow::anyhow!(
                            "fact chain for {fact_key} (entity {entity_id}) has no versions at \
                             all, yet a version in this batch was marked superseded"
                        )
                    })?
                }
            };
            statuses[index] = FactVersionStatus::Stale { current: (current.1, current.0) };
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

#[cfg(test)]
mod tests {
    use crate::storage::TenantStore;
    use tempfile::tempdir;

    // Evidence lookup used to be one prepared-statement round trip per fact
    // version (facts.rs "fact_evidence N+1"); this pins that the batched,
    // IN-chunked replacement still groups evidence correctly per
    // (entity_id, fact_key, version_memory_id) and keeps different entities'
    // evidence lists from bleeding into each other.
    #[test]
    fn fact_evidence_batches_without_crossing_entities() {
        let temp = tempdir().unwrap();
        let store = TenantStore::new(&temp.path().join("tenant.db")).unwrap();

        store
            .register_fact_versions_batch(
                "alice",
                &[("residence", 100, "m1", "alice", "lives_in", "Austin")],
            )
            .unwrap();
        // Restates the same value later: merged as evidence for m1, not a
        // new version.
        store
            .register_fact_versions_batch(
                "alice",
                &[("residence", 150, "m2", "alice", "lives_in", "Austin")],
            )
            .unwrap();

        store
            .register_fact_versions_batch(
                "bob",
                &[("residence", 100, "m3", "bob", "lives_in", "Chicago")],
            )
            .unwrap();
        store
            .register_fact_versions_batch(
                "bob",
                &[("residence", 150, "m4", "bob", "lives_in", "Chicago")],
            )
            .unwrap();

        let result =
            store.fact_versions_for_memories(&["m1".to_string(), "m3".to_string()]).unwrap();

        // Newest evidence first, and bob's confirmation never lands on
        // alice's version (or vice versa).
        assert_eq!(result["m1"].evidence, vec!["m2".to_string(), "m1".to_string()]);
        assert_eq!(result["m3"].evidence, vec!["m4".to_string(), "m3".to_string()]);
    }
}
