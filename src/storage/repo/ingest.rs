use super::prelude::*;
use crate::core::memory_id::{MemoryId, Tag};
use crate::graph::EdgeType;
use rusqlite::Transaction;
#[cfg(test)]
use std::cell::Cell;
use std::collections::HashMap;

#[cfg(test)]
thread_local! {
    // Thread-local, not a process-global atomic: `cargo test` runs tests in
    // parallel, and a shared flag let one test's injected failure fire inside
    // another test's ingest. That made the suite non-deterministic, which is
    // an especially bad failure mode for an engine whose central claim is
    // deterministic replay. Set and consumed on the same thread, since
    // `commit_ingest` runs synchronously on its caller.
    static FAIL_NEXT_FACT_WRITE: Cell<bool> = const { Cell::new(false) };
}

#[derive(Clone)]
pub struct IngestFactRegistration {
    pub entity_id: String,
    pub fact_key: String,
    pub timestamp: u64,
    pub memory_id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

#[derive(Clone)]
pub struct IngestGraphEdge {
    pub memory_id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub status: String,
    pub ref_info: Option<(String, String)>,
    pub timestamp: u64,
}

#[derive(Clone)]
pub struct IngestMetric {
    pub entity_id: String,
    pub memory_id: String,
    pub timestamp: u64,
    pub label: String,
    pub value: f64,
    pub unit: String,
    pub source_text: String,
}

#[derive(Clone)]
pub struct IngestConsolidationTask {
    pub entity_id: String,
    pub memory_id: String,
    pub timestamp: u64,
    pub textual_content: String,
}

#[derive(Default)]
pub struct IngestBatches {
    pub observations: Vec<(u64, String, AgentObservation)>,
    pub fts_batch: Vec<(String, String, String)>,
    pub memory_card_batch: Vec<MemoryCard>,
    pub memory_card_latest_updates: Vec<(String, bool, u64)>,
    pub session_router_updates: Vec<SessionRouterRecord>,
    pub preference_batch: HashMap<String, Vec<(String, f32)>>,
    pub memory_links_batch: Vec<(String, String, String)>,
    pub predicate_canon_batch: HashMap<String, Vec<(String, Vec<f32>)>>,
    pub predicate_canon_tau: f32,
    pub fact_batch: Vec<IngestFactRegistration>,
    pub metric_batch: Vec<IngestMetric>,
    pub metric_memory_ids: Vec<String>,
    pub neighbour_embedding_updates: Vec<(String, Vec<f32>)>,
    pub consolidation_tasks: Vec<IngestConsolidationTask>,
}

pub struct CommitOutcome {
    pub vector_ids: Vec<Option<u64>>,
    pub fts_batch: Vec<(String, String, String)>,
    pub vector_batch: HashMap<String, Vec<(u64, Vec<f32>)>>,
    pub indexed_memory_ids: Vec<String>,
}

type PendingIndexRow =
    (String, String, String, MemoryKind, String, String, Option<u64>, Option<Vec<u8>>);
type NeighbourVector = (String, u64, String, Vec<f32>);

#[cfg(test)]
pub(crate) fn fail_next_fact_write_for_test() {
    FAIL_NEXT_FACT_WRITE.with(|flag| flag.set(true));
}

impl TenantStore {
    pub fn commit_ingest(&self, batches: &IngestBatches) -> Result<CommitOutcome> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let vector_ids = insert_observations_tx(&tx, &batches.observations)?;
        insert_cards_tx(&tx, &batches.memory_card_batch)?;
        // Router FTS docs are written durably inside merge_router_records_tx
        // itself (see its comment), so fts_batch here only ever carries the
        // observation documents, which are recovered by reindex_unindexed.
        let fts_batch = batches.fts_batch.clone();
        merge_router_records_tx(&tx, &batches.session_router_updates)?;
        insert_preferences_tx(&tx, &batches.preference_batch)?;
        insert_memory_links_tx(&tx, &batches.memory_links_batch)?;
        insert_metrics_tx(&tx, &batches.metric_memory_ids, &batches.metric_batch)?;
        insert_consolidation_tasks_tx(&tx, &batches.consolidation_tasks)?;
        let neighbour_vectors =
            update_neighbour_embeddings_tx(&tx, &batches.neighbour_embedding_updates)?;

        let mut fact_batch = batches.fact_batch.clone();
        let mut canonical_by_entity = HashMap::new();
        for (entity_id, predicates) in &batches.predicate_canon_batch {
            canonical_by_entity.extend(
                TenantStore::canonicalize_predicates_tx(
                    &tx,
                    entity_id,
                    predicates,
                    batches.predicate_canon_tau,
                )?
                .into_iter()
                .map(|(predicate, canonical)| ((entity_id.clone(), predicate), canonical)),
            );
        }
        for fact in &mut fact_batch {
            if let Some(canonical) = canonical_by_entity
                .get(&(fact.entity_id.clone(), fact.fact_key.clone()))
                .filter(|canonical| **canonical != fact.fact_key)
            {
                fact.predicate = fact.fact_key.replace('_', " ");
                fact.fact_key = canonical.clone();
            }
        }
        let fact_updates = register_facts_tx(&tx, &fact_batch)?;
        let mut latest_updates = batches.memory_card_latest_updates.clone();
        latest_updates.extend(fact_updates);
        update_card_latest_tx(&tx, &latest_updates)?;

        insert_card_edges_tx(&tx, &batches.memory_card_batch, &fact_batch)?;
        tx.commit()?;

        let mut vector_batch: HashMap<String, Vec<(u64, Vec<f32>)>> = HashMap::new();
        for ((_, _, observation), vector_id) in batches.observations.iter().zip(&vector_ids) {
            if let Some(vector_id) = vector_id {
                if !observation.embedding.is_empty() {
                    vector_batch
                        .entry(observation.entity_id.clone())
                        .or_default()
                        .push((*vector_id, observation.embedding.clone()));
                }
            }
        }
        for (memory_id, vector_id, entity_id, embedding) in &neighbour_vectors {
            vector_batch
                .entry(entity_id.clone())
                .or_default()
                .push((*vector_id, embedding.clone()));
            let _ = memory_id;
        }

        let mut indexed_memory_ids = batches
            .observations
            .iter()
            .map(|(_, memory_id, _)| memory_id.clone())
            .collect::<Vec<_>>();
        indexed_memory_ids
            .extend(neighbour_vectors.iter().map(|(memory_id, _, _, _)| memory_id.clone()));
        indexed_memory_ids.sort();
        indexed_memory_ids.dedup();

        Ok(CommitOutcome { vector_ids, fts_batch, vector_batch, indexed_memory_ids })
    }

    pub fn mark_indexed(&self, memory_ids: &[String]) -> Result<()> {
        if memory_ids.is_empty() {
            return Ok(());
        }
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt =
                tx.prepare_cached("UPDATE memories SET indexed = 1 WHERE memory_id = ?1")?;
            for memory_id in memory_ids {
                stmt.execute(params![memory_id])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn reindex_unindexed(&self, limit: usize) -> Result<usize> {
        if limit == 0 {
            return Ok(0);
        }
        let rows = {
            let conn = self.get_conn()?;
            let mut stmt = conn.prepare_cached(
                "SELECT m.memory_id, m.entity_id, m.content, m.kind, m.content_hash,
                        m.session_id, v.vector_id, v.embedding
                 FROM memories m
                 LEFT JOIN vector_lookup v ON v.memory_id = m.memory_id
                 WHERE m.indexed = 0
                 ORDER BY m.rowid
                 LIMIT ?1",
            )?;
            let rows = stmt.query_map(params![limit as i64], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    MemoryKind::parse(&row.get::<_, String>(3)?),
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<i64>>(6)?.map(|value| value as u64),
                    row.get::<_, Option<Vec<u8>>>(7)?,
                ))
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        if rows.is_empty() {
            return Ok(0);
        }

        self.reindex_rows(&rows)
    }

    pub fn reindex_memory_ids(&self, memory_ids: &[String]) -> Result<usize> {
        if memory_ids.is_empty() {
            return Ok(0);
        }
        let conn = self.get_conn()?;
        let mut rows = Vec::new();
        for chunk in memory_ids.chunks(IN_CHUNK) {
            let values = padded_in_chunk(chunk);
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT m.memory_id, m.entity_id, m.content, m.kind, m.content_hash,
                        m.session_id, v.vector_id, v.embedding
                 FROM memories m
                 LEFT JOIN vector_lookup v ON v.memory_id = m.memory_id
                 WHERE m.indexed = 0 AND m.memory_id IN ({})",
                in_placeholders(IN_CHUNK)
            ))?;
            let mapped = stmt.query_map(rusqlite::params_from_iter(values), |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    MemoryKind::parse(&row.get::<_, String>(3)?),
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<i64>>(6)?.map(|value| value as u64),
                    row.get::<_, Option<Vec<u8>>>(7)?,
                ))
            })?;
            rows.extend(mapped.collect::<rusqlite::Result<Vec<_>>>()?);
        }
        self.reindex_rows(&rows)
    }

    fn reindex_rows(&self, rows: &[PendingIndexRow]) -> Result<usize> {
        if rows.is_empty() {
            return Ok(0);
        }

        let mut fts_batch: Vec<(String, String, String)> = rows
            .iter()
            .filter_map(|(memory_id, entity_id, content, kind, _, _, _, _)| {
                crate::core::text::index_text_for(*kind, content)
                    .map(|text| (memory_id.clone(), entity_id.clone(), text.to_string()))
            })
            .collect();
        let sessions: std::collections::HashSet<(&str, &str)> = rows
            .iter()
            .filter(|(_, _, _, _, _, session_id, _, _)| !session_id.is_empty())
            .map(|(_, entity_id, _, _, _, session_id, _, _)| {
                (entity_id.as_str(), session_id.as_str())
            })
            .collect();
        if !sessions.is_empty() {
            let conn = self.get_conn()?;
            let mut stmt = conn.prepare_cached(
                "SELECT router_text FROM session_router WHERE entity_id = ?1 AND session_id = ?2",
            )?;
            for (entity_id, session_id) in sessions {
                if let Ok(router_text) =
                    stmt.query_row(params![entity_id, session_id], |row| row.get::<_, String>(0))
                {
                    let router_id = MemoryId::new(entity_id, session_id, 0)
                        .derived(Tag::Named("router".to_string()))
                        .as_str()
                        .to_string();
                    fts_batch.push((router_id, entity_id.to_string(), router_text));
                }
            }
        }
        self.fts_index_batch(&fts_batch)?;

        let mut vectors_by_entity: HashMap<String, Vec<(u64, Vec<f32>)>> = HashMap::new();
        for (_, entity_id, _, _, _, _, vector_id, embedding) in rows {
            if let (Some(vector_id), Some(embedding)) = (vector_id, embedding) {
                vectors_by_entity
                    .entry(entity_id.clone())
                    .or_default()
                    .push((*vector_id, bytes_to_vec_f32(embedding)));
            }
        }
        if !vectors_by_entity.is_empty() {
            let vectors = self.vectors()?;
            for (entity_id, items) in vectors_by_entity {
                vectors.insert_batch(&entity_id, &items)?;
            }
        }

        let repaired = rows.len();
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut finish = tx.prepare_cached(
                "UPDATE memories SET indexed = 1
                 WHERE memory_id = ?1 AND content_hash = ?2 AND indexed = 0",
            )?;
            let mut reschedule = tx.prepare_cached(
                "UPDATE memories SET indexed = 0
                 WHERE memory_id = ?1 AND content_hash <> ?2",
            )?;
            for (memory_id, _, _, _, content_hash, _, _, _) in rows {
                if finish.execute(params![memory_id, content_hash])? == 0 {
                    reschedule.execute(params![memory_id, content_hash])?;
                }
            }
        }
        tx.commit()?;
        Ok(repaired)
    }
}

pub(crate) fn insert_observations_tx(
    tx: &Transaction<'_>,
    items: &[(u64, String, AgentObservation)],
) -> Result<Vec<Option<u64>>> {
    let recorded_at = unix_timestamp_ms()?;
    let mut rowids = Vec::with_capacity(items.len());
    let mut select_stmt = tx.prepare_cached("SELECT rowid FROM memories WHERE memory_id = ?1")?;
    let mut update_stmt = tx.prepare_cached(
        "UPDATE memories SET content = ?1, kind = ?2, created_at_ms = ?3, entity_id = ?4,
         content_hash = ?5, session_id = ?7, turn_index = ?8, role = ?9,
         parent_memory_id = ?10, recorded_at_ms = ?11, expires_at_ms = ?12,
         indexed = 0 WHERE rowid = ?6",
    )?;
    let mut insert_stmt = tx.prepare_cached(
        "INSERT INTO memories (
            memory_id, entity_id, content, kind, content_hash, created_at_ms,
            session_id, turn_index, role, parent_memory_id, recorded_at_ms, expires_at_ms, indexed
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, 0)",
    )?;
    let mut vec_stmt = tx.prepare_cached(
        "INSERT OR REPLACE INTO vector_lookup
            (vector_id, memory_id, entity_id, timestamp_ms, embedding)
         VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;
    let mut del_vec_stmt = tx.prepare_cached("DELETE FROM vector_lookup WHERE vector_id = ?1")?;

    for &(timestamp, ref memory_id, ref observation) in items {
        let existing_rowid: Option<i64> =
            select_stmt.query_row(params![memory_id], |row| row.get(0)).ok();
        let rowid = if let Some(rowid) = existing_rowid {
            update_stmt.execute(params![
                observation.textual_content,
                observation.kind.as_str(),
                timestamp as i64,
                observation.entity_id,
                observation.content_hash,
                rowid,
                observation.session_id,
                observation.turn_index,
                observation.role,
                observation.parent_memory_id,
                recorded_at,
                observation.expires_at_ms.map(|value| value as i64)
            ])?;
            rowid
        } else {
            insert_stmt.execute(params![
                memory_id,
                observation.entity_id,
                observation.textual_content,
                observation.kind.as_str(),
                observation.content_hash,
                timestamp as i64,
                observation.session_id,
                observation.turn_index,
                observation.role,
                observation.parent_memory_id,
                recorded_at,
                observation.expires_at_ms.map(|value| value as i64)
            ])?;
            tx.last_insert_rowid()
        };

        if observation.embedding.is_empty() {
            del_vec_stmt.execute(params![rowid])?;
        } else {
            vec_stmt.execute(params![
                rowid,
                memory_id,
                observation.entity_id,
                timestamp as i64,
                vec_f32_to_bytes(&observation.embedding)
            ])?;
        }
        rowids.push(Some(rowid as u64));
    }
    Ok(rowids)
}

fn insert_cards_tx(tx: &Transaction<'_>, cards: &[MemoryCard]) -> Result<()> {
    let mut stmt = tx.prepare_cached(
        "INSERT OR REPLACE INTO memory_cards (
            card_id, entity_id, user_id, source_memory_id, source_session_id,
            subject, predicate, object, memory_text, card_type, confidence,
            is_latest, is_static, is_inference, expires_at, root_card_id, parent_card_id,
            lifecycle, created_at_ms, updated_at_ms, source_turn_index, document_time,
            conversation_time, event_time
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24)",
    )?;
    for card in cards {
        let lifecycle =
            card.lifecycle.as_ref().map(serde_json::to_string).transpose()?.unwrap_or_default();
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
            i32::from(card.is_latest),
            i32::from(card.is_static),
            i32::from(card.is_inference),
            card.expires_at,
            card.root_card_id,
            card.parent_card_id,
            lifecycle,
            card.created_at_ms,
            card.updated_at_ms,
            card.source_turn_index as i64,
            card.document_time,
            card.conversation_time,
            card.event_time
        ])?;
    }
    Ok(())
}

fn insert_metrics_tx(
    tx: &Transaction<'_>,
    memory_ids: &[String],
    metrics: &[IngestMetric],
) -> Result<()> {
    let mut delete = tx.prepare_cached("DELETE FROM metrics WHERE memory_id = ?1")?;
    let mut insert = tx.prepare_cached(
        "INSERT INTO metrics
            (memory_id, ordinal, entity_id, timestamp_ms, label, value, unit, source_text)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    )?;
    let mut ordinals = HashMap::new();
    for memory_id in memory_ids {
        delete.execute(params![memory_id])?;
    }
    for metric in metrics {
        let ordinal = ordinals.entry(metric.memory_id.as_str()).or_insert(0_i64);
        insert.execute(params![
            metric.memory_id,
            *ordinal,
            metric.entity_id,
            metric.timestamp as i64,
            metric.label,
            metric.value,
            metric.unit,
            metric.source_text
        ])?;
        *ordinal += 1;
    }
    Ok(())
}

fn insert_consolidation_tasks_tx(
    tx: &Transaction<'_>,
    tasks: &[IngestConsolidationTask],
) -> Result<()> {
    let mut insert = tx.prepare_cached(
        "INSERT OR REPLACE INTO consolidation_queue
            (memory_id, entity_id, timestamp_ms, textual_content)
         VALUES (?1, ?2, ?3, ?4)",
    )?;
    for task in tasks {
        insert.execute(params![
            task.memory_id,
            task.entity_id,
            task.timestamp as i64,
            task.textual_content
        ])?;
    }
    Ok(())
}

fn update_neighbour_embeddings_tx(
    tx: &Transaction<'_>,
    updates: &[(String, Vec<f32>)],
) -> Result<Vec<NeighbourVector>> {
    let mut applied = Vec::new();
    let mut select =
        tx.prepare_cached("SELECT vector_id, entity_id FROM vector_lookup WHERE memory_id = ?1")?;
    let mut update =
        tx.prepare_cached("UPDATE vector_lookup SET embedding = ?1 WHERE vector_id = ?2")?;
    let mut pending = tx.prepare_cached("UPDATE memories SET indexed = 0 WHERE memory_id = ?1")?;
    for (memory_id, embedding) in updates {
        let found = match select.query_row(params![memory_id], |row| {
            Ok((row.get::<_, i64>(0)? as u64, row.get::<_, String>(1)?))
        }) {
            Ok(found) => found,
            Err(rusqlite::Error::QueryReturnedNoRows) => continue,
            Err(error) => return Err(error.into()),
        };
        update.execute(params![vec_f32_to_bytes(embedding), found.0 as i64])?;
        pending.execute(params![memory_id])?;
        applied.push((memory_id.clone(), found.0, found.1, embedding.clone()));
    }
    Ok(applied)
}

pub(super) fn merge_router_records_tx(
    tx: &Transaction<'_>,
    updates: &[SessionRouterRecord],
) -> Result<Vec<SessionRouterRecord>> {
    let mut merged_records = Vec::with_capacity(updates.len());
    let mut select_stmt = tx.prepare_cached(
        "SELECT record_json FROM session_router WHERE session_id = ?1 AND entity_id = ?2",
    )?;
    let mut upsert_stmt = tx.prepare_cached(
        "INSERT INTO session_router (
            session_id, entity_id, record_json, router_text, created_at_ms, updated_at_ms
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(session_id, entity_id) DO UPDATE SET
            record_json = excluded.record_json,
            router_text = excluded.router_text,
            updated_at_ms = excluded.updated_at_ms
         RETURNING rowid",
    )?;
    let mut fts_stmt = tx.prepare_cached(
        "INSERT OR REPLACE INTO fts_session_router
            (rowid, session_id, entity_id, router_text)
         VALUES (?1, ?2, ?3, ?4)",
    )?;
    let mut delete_fts_stmt = tx.prepare_cached(
        "DELETE FROM fts_session_router WHERE session_id = ?1 AND entity_id = ?2",
    )?;
    // Router documents have no row in `memories`, so `reindex_unindexed`
    // (which scans `memories WHERE indexed = 0`) can never find or rebuild
    // one. Previously this doc was only queued into the post-commit
    // `fts_batch` and written by a separate `fts_index_batch` call after this
    // transaction committed — a crash in that window permanently lost the
    // router's searchability. Write it into `fts_memories` here instead,
    // inside the same transaction as `session_router`, mirroring
    // `rebuild_router_after_delete` in memories.rs.
    let mut fts_memories_stmt = tx.prepare_cached(
        "INSERT OR REPLACE INTO fts_memories
            (rowid, memory_id, entity_id, entity_tok, content)
         VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;
    let mut source_stmt = tx.prepare_cached(
        "INSERT OR REPLACE INTO session_router_sources
            (session_id, entity_id, memory_id, record_json)
         VALUES (?1, ?2, ?3, ?4)",
    )?;
    let now = unix_timestamp_ms()? as u64;
    for record in updates {
        let source_json = serde_json::to_string(record)?;
        for memory_id in &record.source_memory_ids {
            source_stmt.execute(params![
                record.session_id,
                record.entity_id,
                memory_id,
                source_json
            ])?;
        }
        let merged = match select_stmt
            .query_row(params![record.session_id, record.entity_id], |row| row.get::<_, String>(0))
        {
            Ok(existing) => serde_json::from_str::<SessionRouterRecord>(&existing).map_or_else(
                |_| record.clone(),
                |previous| merge_router_records(&previous, record),
            ),
            Err(_) => record.clone(),
        };
        let json = serde_json::to_string(&merged)?;
        upsert_stmt.query_row(
            params![
                merged.session_id,
                merged.entity_id,
                json,
                merged.router_text,
                merged.created_at_ms.min(now),
                now
            ],
            |row| row.get::<_, i64>(0),
        )?;
        delete_fts_stmt.execute(params![merged.session_id, merged.entity_id])?;
        fts_stmt.execute(params![
            fts_rowid(&format!("router::{}::{}", merged.entity_id, merged.session_id)),
            merged.session_id,
            merged.entity_id,
            merged.router_text
        ])?;
        if !merged.router_text.is_empty() {
            let router_id = MemoryId::new(&merged.entity_id, &merged.session_id, 0)
                .derived(Tag::Named("router".to_string()))
                .as_str()
                .to_string();
            fts_memories_stmt.execute(params![
                fts_rowid(&router_id),
                router_id,
                merged.entity_id,
                fts_entity_tok(&merged.entity_id),
                merged.router_text
            ])?;
        }
        merged_records.push(merged);
    }
    Ok(merged_records)
}

fn insert_preferences_tx(
    tx: &Transaction<'_>,
    preferences: &HashMap<String, Vec<(String, f32)>>,
) -> Result<()> {
    let mut stmt = tx.prepare_cached(
        "INSERT OR REPLACE INTO preferences (entity_id, memory_id, strength)
         VALUES (?1, ?2, ?3)",
    )?;
    for (entity_id, items) in preferences {
        for (memory_id, strength) in items {
            stmt.execute(params![entity_id, memory_id, strength])?;
        }
    }
    Ok(())
}

fn insert_memory_links_tx(tx: &Transaction<'_>, links: &[(String, String, String)]) -> Result<()> {
    let mut stmt = tx.prepare_cached(
        "INSERT OR IGNORE INTO memory_links
            (source_memory_id, target_memory_id, link_type)
         VALUES (?1, ?2, ?3)",
    )?;
    for (source, target, link_type) in links {
        stmt.execute(params![source, target, link_type])?;
    }
    Ok(())
}

fn register_facts_tx(
    tx: &Transaction<'_>,
    facts: &[IngestFactRegistration],
) -> Result<Vec<(String, bool, u64)>> {
    if facts.is_empty() {
        return Ok(Vec::new());
    }
    let recorded_at = unix_timestamp_ms()? as u64;
    let mut grouped: HashMap<String, Vec<&IngestFactRegistration>> = HashMap::new();
    for fact in facts {
        grouped.entry(fact.entity_id.clone()).or_default().push(fact);
    }
    let mut latest_updates = Vec::new();
    let mut graph_entries = Vec::new();

    for (entity_id, registrations) in grouped {
        #[cfg(test)]
        if FAIL_NEXT_FACT_WRITE.with(|flag| flag.replace(false)) {
            return Err(anyhow::anyhow!("injected fact write failure"));
        }
        let inputs: Vec<(&str, u64, &str, &str, &str, &str)> = registrations
            .iter()
            .map(|fact| {
                (
                    fact.fact_key.as_str(),
                    fact.timestamp,
                    fact.memory_id.as_str(),
                    fact.subject.as_str(),
                    fact.predicate.as_str(),
                    fact.object.as_str(),
                )
            })
            .collect();
        let statuses =
            TenantStore::register_fact_versions_tx(tx, &entity_id, &inputs, recorded_at)?;
        for (status, fact) in statuses.into_iter().zip(registrations) {
            match status {
                FactVersionStatus::Current { superseded: Some((_, old_id)) } => {
                    latest_updates.push((old_id.clone(), false, fact.timestamp));
                    latest_updates.push((fact.memory_id.clone(), true, fact.timestamp));
                    graph_entries.push(IngestGraphEdge {
                        memory_id: fact.memory_id.clone(),
                        subject: fact.subject.clone(),
                        predicate: fact.predicate.clone(),
                        object: fact.object.clone(),
                        status: "current".to_string(),
                        ref_info: Some((EdgeType::Supersedes.as_str().to_string(), old_id.clone())),
                        timestamp: fact.timestamp,
                    });
                    graph_entries.push(IngestGraphEdge {
                        memory_id: old_id,
                        subject: fact.subject.clone(),
                        predicate: fact.predicate.clone(),
                        object: fact.object.clone(),
                        status: "stale".to_string(),
                        ref_info: Some((
                            EdgeType::SupersededBy.as_str().to_string(),
                            fact.memory_id.clone(),
                        )),
                        timestamp: fact.timestamp,
                    });
                }
                FactVersionStatus::Stale { current: (_, current_id) } => {
                    latest_updates.push((fact.memory_id.clone(), false, fact.timestamp));
                    graph_entries.push(IngestGraphEdge {
                        memory_id: fact.memory_id.clone(),
                        subject: fact.subject.clone(),
                        predicate: fact.predicate.clone(),
                        object: fact.object.clone(),
                        status: "stale".to_string(),
                        ref_info: Some((EdgeType::SupersededBy.as_str().to_string(), current_id)),
                        timestamp: fact.timestamp,
                    });
                }
                FactVersionStatus::Current { superseded: None } => {
                    latest_updates.push((fact.memory_id.clone(), true, fact.timestamp));
                    graph_entries.push(IngestGraphEdge {
                        memory_id: fact.memory_id.clone(),
                        subject: fact.subject.clone(),
                        predicate: fact.predicate.clone(),
                        object: fact.object.clone(),
                        status: "current".to_string(),
                        ref_info: None,
                        timestamp: fact.timestamp,
                    });
                }
                FactVersionStatus::Confirmed { .. } => {}
            }
        }
    }

    insert_graph_entries_tx(tx, &graph_entries)?;
    Ok(latest_updates)
}

fn update_card_latest_tx(tx: &Transaction<'_>, updates: &[(String, bool, u64)]) -> Result<()> {
    let mut stmt = tx.prepare_cached(
        "UPDATE memory_cards SET is_latest = ?1, updated_at_ms = ?2 WHERE card_id = ?3",
    )?;
    for (card_id, is_latest, timestamp) in updates {
        stmt.execute(params![i32::from(*is_latest), *timestamp as i64, card_id])?;
    }
    Ok(())
}

fn insert_card_edges_tx(
    tx: &Transaction<'_>,
    cards: &[MemoryCard],
    facts: &[IngestFactRegistration],
) -> Result<()> {
    let mut stmt = tx.prepare_cached(
        "INSERT OR IGNORE INTO edges
            (edge_id, source, target, edge_type, label, status, timestamp_ms, memory_id)
         VALUES (?1, ?2, ?3, ?4, ?5, 'current', ?6, ?7)",
    )?;
    for (memory_id, subject, predicate, object, timestamp) in cards
        .iter()
        .map(|card| {
            (
                card.source_memory_id.as_str(),
                card.subject.as_str(),
                card.predicate.as_str(),
                card.object.as_str(),
                card.created_at_ms,
            )
        })
        .chain(facts.iter().map(|fact| {
            (
                fact.memory_id.as_str(),
                fact.subject.as_str(),
                fact.predicate.as_str(),
                fact.object.as_str(),
                fact.timestamp,
            )
        }))
    {
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            continue;
        }
        stmt.execute(params![
            format!("edge::{memory_id}::{subject}::{predicate}"),
            subject,
            object,
            predicate,
            format!("{subject} {predicate} {object}"),
            timestamp as i64,
            memory_id
        ])?;
    }
    Ok(())
}

fn insert_graph_entries_tx(tx: &Transaction<'_>, entries: &[IngestGraphEdge]) -> Result<()> {
    let mut stmt = tx.prepare_cached(
        "INSERT OR REPLACE INTO edges
            (edge_id, source, target, edge_type, label, status, ref_source, ref_target, timestamp_ms, memory_id, weight)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
    )?;
    for entry in entries {
        if entry.subject.trim().is_empty() || entry.object.trim().is_empty() {
            continue;
        }
        let edge_id = format!("edge::{}::{}::{}", entry.memory_id, entry.subject, entry.predicate);
        let label = format!("{} {} {}", entry.subject, entry.predicate, entry.object);
        let weight = EdgeType::from_str(&entry.predicate).default_weight();
        stmt.execute(params![
            edge_id,
            entry.subject,
            entry.object,
            entry.predicate,
            label,
            entry.status,
            entry.ref_info.as_ref().map(|(kind, _)| kind),
            entry.ref_info.as_ref().map(|(_, target)| target),
            entry.timestamp as i64,
            entry.memory_id,
            f64::from(weight)
        ])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        fail_next_fact_write_for_test, IngestBatches, IngestConsolidationTask,
        IngestFactRegistration, IngestMetric,
    };
    use crate::storage::{
        AgentObservation, MemoryCard, MemoryKind, SessionRouterRecord, TenantStore,
    };
    use tempfile::tempdir;

    fn observation(memory_id: &str, text: &str) -> (u64, String, AgentObservation) {
        (
            1_000,
            memory_id.to_string(),
            AgentObservation {
                entity_id: "alice".to_string(),
                textual_content: text.to_string(),
                kind: MemoryKind::Fact,
                content_hash: "hash".to_string(),
                created_at_ms: 1_000,
                ..Default::default()
            },
        )
    }

    fn card(memory_id: &str) -> MemoryCard {
        MemoryCard {
            card_id: memory_id.to_string(),
            entity_id: "alice".to_string(),
            user_id: "alice".to_string(),
            source_memory_id: memory_id.to_string(),
            source_session_id: String::new(),
            source_turn_index: 0,
            document_time: 1_000,
            conversation_time: 1_000,
            event_time: None,
            subject: "alice".to_string(),
            predicate: "lives in".to_string(),
            object: "Austin".to_string(),
            memory_text: "Alice lives in Austin".to_string(),
            card_type: "fact".to_string(),
            confidence: 0.9,
            is_latest: true,
            is_static: true,
            is_inference: false,
            expires_at: None,
            root_card_id: None,
            parent_card_id: None,
            lifecycle: None,
            created_at_ms: 1_000,
            updated_at_ms: 1_000,
        }
    }

    #[test]
    fn ingest_is_atomic_across_sqlite_writes() {
        let directory = tempdir().unwrap();
        let store = TenantStore::new(&directory.path().join("tenant.db")).unwrap();
        let memory_id = "alice::session::1";
        let batches = IngestBatches {
            observations: vec![observation(memory_id, "Alice lives in Austin")],
            memory_card_batch: vec![card(memory_id)],
            fact_batch: vec![IngestFactRegistration {
                entity_id: "alice".to_string(),
                fact_key: "residence".to_string(),
                timestamp: 1_000,
                memory_id: memory_id.to_string(),
                subject: "alice".to_string(),
                predicate: "lives in".to_string(),
                object: "Austin".to_string(),
            }],
            metric_memory_ids: vec![memory_id.to_string()],
            metric_batch: vec![IngestMetric {
                entity_id: "alice".into(),
                memory_id: memory_id.into(),
                timestamp: 1_000,
                label: "distance".into(),
                value: 1.0,
                unit: "meters".into(),
                source_text: "1 meter".into(),
            }],
            consolidation_tasks: vec![IngestConsolidationTask {
                entity_id: "alice".into(),
                memory_id: memory_id.into(),
                timestamp: 1_000,
                textual_content: "Alice lives in Austin".into(),
            }],
            ..Default::default()
        };

        fail_next_fact_write_for_test();
        assert!(store.commit_ingest(&batches).is_err());
        let conn = store.get_conn().unwrap();
        for table in
            ["memories", "memory_cards", "fact_versions", "edges", "metrics", "consolidation_queue"]
        {
            let count: i64 = conn
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| row.get(0))
                .unwrap();
            assert_eq!(count, 0, "{table} retained a rolled-back row");
        }
    }

    #[test]
    fn neighbour_embeddings_join_the_atomic_commit_and_repair_set() {
        let directory = tempdir().unwrap();
        let store = TenantStore::new(&directory.path().join("tenant.db")).unwrap();
        let mut existing = observation("existing", "context");
        existing.2.embedding = vec![1.0, 0.0];
        store
            .commit_ingest(&IngestBatches { observations: vec![existing], ..Default::default() })
            .unwrap();
        store.mark_indexed(&["existing".into()]).unwrap();

        let outcome = store
            .commit_ingest(&IngestBatches {
                neighbour_embedding_updates: vec![("existing".into(), vec![0.0, 1.0])],
                ..Default::default()
            })
            .unwrap();

        assert_eq!(outcome.indexed_memory_ids, vec!["existing"]);
        assert_eq!(outcome.vector_batch["alice"][0].1, vec![0.0, 1.0]);
        let indexed: i64 = store
            .get_conn()
            .unwrap()
            .query_row("SELECT indexed FROM memories WHERE memory_id = 'existing'", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(indexed, 0);
    }

    #[test]
    fn reindex_unindexed_recovers_a_partial_ingest() {
        let directory = tempdir().unwrap();
        let store = TenantStore::new(&directory.path().join("tenant.db")).unwrap();
        let memory_id = "alice::session::2";
        let batches = IngestBatches {
            observations: vec![observation(memory_id, "Alice enjoys hiking")],
            fts_batch: vec![(
                memory_id.to_string(),
                "alice".to_string(),
                "Alice enjoys hiking".to_string(),
            )],
            ..Default::default()
        };

        store.commit_ingest(&batches).unwrap();
        let indexed: i64 = store
            .get_conn()
            .unwrap()
            .query_row("SELECT indexed FROM memories WHERE memory_id = ?1", [memory_id], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(indexed, 0);
        assert_eq!(store.reindex_unindexed(1).unwrap(), 1);
        let indexed: i64 = store
            .get_conn()
            .unwrap()
            .query_row("SELECT indexed FROM memories WHERE memory_id = ?1", [memory_id], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(indexed, 1);
        assert_eq!(store.fts_search("hiking", 10, Some("alice")).unwrap().len(), 1);
    }

    #[test]
    fn reindex_uses_the_same_text_policy_as_live_ingest() {
        let dir = tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("tenant.db")).unwrap();
        let mut normal = observation("m1", "[Session Focus: metadataonlytoken]\nbodytoken");
        normal.2.kind = MemoryKind::Conversational;
        let mut synthetic = observation("m2", "syntheticword");
        synthetic.2.kind = MemoryKind::SyntheticQuery;
        let batches = IngestBatches { observations: vec![normal, synthetic], ..Default::default() };

        store.commit_ingest(&batches).unwrap();
        assert_eq!(store.reindex_unindexed(10).unwrap(), 2);

        assert!(store.fts_search("metadataonlytoken", 10, Some("alice")).unwrap().is_empty());
        assert_eq!(store.fts_search("bodytoken", 10, Some("alice")).unwrap().len(), 1);
        assert!(store.fts_search("syntheticword", 10, Some("alice")).unwrap().is_empty());
    }

    #[test]
    fn ingest_records_event_and_database_time_separately() {
        let dir = tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("tenant.db")).unwrap();
        let batches = IngestBatches {
            observations: vec![observation("m1", "backdated")],
            ..Default::default()
        };
        store.commit_ingest(&batches).unwrap();
        let conn = store.get_conn().unwrap();
        let (event_time, recorded_at): (u64, u64) = conn
            .query_row(
                "SELECT created_at_ms, recorded_at_ms FROM memories WHERE memory_id = 'm1'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(event_time, 1_000);
        assert!(recorded_at > event_time);
    }

    #[test]
    fn stale_repair_cannot_mark_newer_content_complete() {
        let dir = tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("tenant.db")).unwrap();
        let mut first = observation("m1", "oldversiontoken");
        first.2.content_hash = "hash-a".into();
        store
            .commit_ingest(&IngestBatches {
                observations: vec![first.clone()],
                ..Default::default()
            })
            .unwrap();
        let stale_row = (
            "m1".to_string(),
            "alice".to_string(),
            "oldversiontoken".to_string(),
            MemoryKind::Fact,
            "hash-a".to_string(),
            String::new(),
            None,
            None,
        );

        let mut second = observation("m1", "newversiontoken");
        second.2.content_hash = "hash-b".into();
        store
            .commit_ingest(&IngestBatches { observations: vec![second], ..Default::default() })
            .unwrap();
        store.mark_indexed(&["m1".to_string()]).unwrap();

        store.reindex_rows(&[stale_row]).unwrap();
        let state = store.stored_content_states(&["m1".to_string()]).unwrap();
        assert_eq!(state["m1"].1, 0);

        store.reindex_memory_ids(&["m1".to_string()]).unwrap();
        assert!(store.fts_search("oldversiontoken", 10, Some("alice")).unwrap().is_empty());
        assert_eq!(store.fts_search("newversiontoken", 10, Some("alice")).unwrap().len(), 1);
    }

    // AUDIT-2026-09-21.md "Router FTS documents are unrecoverable after a
    // crash": the router doc has no row in `memories`, so it used to only
    // become searchable once a separate, post-commit fts_index_batch call
    // ran — a crash between the two lost it for good, and reindex_unindexed
    // could never rebuild it. This pins that commit_ingest alone, with no
    // follow-up indexing call, is enough.
    #[test]
    fn router_fts_document_is_searchable_immediately_after_commit_with_no_follow_up_index_call() {
        let dir = tempdir().unwrap();
        let store = TenantStore::new(&dir.path().join("tenant.db")).unwrap();
        let record = SessionRouterRecord {
            session_id: "session".to_string(),
            entity_id: "alice".to_string(),
            router_text: "routerdurabilitytoken".to_string(),
            ..Default::default()
        };
        let outcome = store
            .commit_ingest(&IngestBatches {
                session_router_updates: vec![record],
                ..Default::default()
            })
            .unwrap();

        // The router doc must not even be in the deferred batch: it was
        // already written durably inside commit_ingest's own transaction.
        assert!(!outcome
            .fts_batch
            .iter()
            .any(|(_, _, text)| text.contains("routerdurabilitytoken")));
        assert_eq!(store.fts_search("routerdurabilitytoken", 10, Some("alice")).unwrap().len(), 1);
    }
}
