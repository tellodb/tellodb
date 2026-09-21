use super::prelude::*;

impl TenantStore {
    pub fn session_turn_window(
        &self,
        entity_id: &str,
        session_id: &str,
        lo: u32,
        hi: u32,
    ) -> Result<Vec<(String, u32, String, String)>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT memory_id, turn_index, role, content FROM memories
             WHERE entity_id = ?1 AND session_id = ?2 AND turn_index BETWEEN ?3 AND ?4
               AND (parent_memory_id IS NULL OR memory_id = parent_memory_id || '::c0')
             ORDER BY turn_index, rowid",
        )?;
        let rows = stmt.query_map(params![entity_id, session_id, lo, hi], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn merge_session_router_records_batch(
        &self,
        updates: &[SessionRouterRecord],
    ) -> Result<Vec<SessionRouterRecord>> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        // merge_router_records_tx now writes the router FTS document into
        // fts_memories inside this same transaction (durability fix — see
        // its comment in ingest.rs), so there is no longer a post-commit
        // fts_index_batch step to run here.
        let merged_results = super::ingest::merge_router_records_tx(&tx, updates)?;
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
                terms.iter().map(|t| format!("\"{t}\"")).collect::<Vec<_>>().join(" OR ");
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
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.session_id.cmp(&b.session_id))
        });
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
            let Ok(json) = row else { continue };
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

            if lexical_hits == 0 && temporal_hits == 0 && entity_hits == 0 && !exact_focus_hit {
                // Last-ditch accept: if the router_text has ANY of the raw query
                // terms (not just the classified lexical/temporal/entity terms),
                // keep the row. The classified term lists are often empty for
                // short or open-vocabulary questions.
                let lower_terms: Vec<&str> = query_lower
                    .split_whitespace()
                    .filter(|t| t.len() > SEARCH_MIN_TERM_LEN)
                    .collect();
                let has_raw_term =
                    !lower_terms.is_empty() && lower_terms.iter().any(|t| lower.contains(t));
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
        // Query FTS5 per entity, rather than one OR-combined query, so we know
        // per entity whether FTS found anything. The LIKE backstop below used
        // to run unconditionally for every entity regardless of the FTS
        // result, doubling the cost of every pivot lookup with a scan that
        // can't use an index beyond the entity_id prefix (AUDIT-2026-09-21.md
        // "entity_pivot_sessions unconditional full scan"). Entities too short
        // for FTS5 (<= SEARCH_MIN_TERM_LEN) skip straight to the backstop, same
        // as before.
        let conn = self.get_conn()?;
        let mut session_to_hits: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut needs_backstop: Vec<&String> = Vec::new();

        let mut fts_stmt = conn.prepare_cached(
            "SELECT sr.session_id
             FROM fts_session_router fsr
             JOIN session_router sr
               ON sr.session_id = fsr.session_id AND sr.entity_id = fsr.entity_id
             WHERE fsr.fts_session_router MATCH ?1 AND fsr.entity_id = ?2
             ORDER BY rank LIMIT ?3",
        )?;
        for entity in subject_entities {
            if entity.len() <= SEARCH_MIN_TERM_LEN {
                needs_backstop.push(entity);
                continue;
            }
            let fts_query = format!("\"{}\"", entity.to_ascii_lowercase());
            let rows = fts_stmt
                .query_map(params![fts_query, entity_id, 8i64], |row| row.get::<_, String>(0))?;
            let mut any_hit = false;
            for row in rows.flatten() {
                any_hit = true;
                *session_to_hits.entry(row).or_insert(0) += 1;
            }
            if !any_hit {
                needs_backstop.push(entity);
            }
        }

        // Backstop: LIKE-based scan for entities FTS5's tokenizer missed
        // (very short or oddly-cased names) or that had zero FTS hits. Cheap
        // because session_router is one row per session, but no longer paid
        // for entities FTS already resolved.
        for entity in needs_backstop {
            if entity.len() < 3 {
                continue;
            }
            let needle = entity.to_ascii_lowercase();
            let mut stmt = conn.prepare_cached(
                "SELECT session_id FROM session_router
                 WHERE entity_id = ?1 AND LOWER(router_text) LIKE ?2",
            )?;
            let rows = stmt.query_map(params![entity_id, format!("%{}%", needle)], |row| {
                row.get::<_, String>(0)
            })?;
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
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.session_id.cmp(&b.session_id))
        });
        Ok(results)
    }

    pub fn get_ledger_turns_batch(
        &self,
        turn_ids: &[String],
    ) -> Result<std::collections::HashMap<String, LedgerTurn>> {
        let mut result = std::collections::HashMap::new();
        if turn_ids.is_empty() {
            return Ok(result);
        }
        let conn = self.get_conn()?;
        let wanted: std::collections::HashSet<&str> = turn_ids.iter().map(String::as_str).collect();
        for ids in turn_ids.chunks(IN_CHUNK) {
            let values = padded_in_chunk(ids);
            let sql = format!(
                "SELECT memory_id, parent_memory_id, entity_id, session_id, role, turn_index,
                        content, created_at_ms, content_hash
                 FROM memories
                 WHERE memory_id IN ({})
                    OR (parent_memory_id IN ({})
                        AND memory_id GLOB parent_memory_id || '::c[0-9]*')
                 ORDER BY rowid",
                in_placeholders(IN_CHUNK),
                in_placeholders(IN_CHUNK)
            );
            let mut stmt = conn.prepare_cached(&sql)?;
            let params = values.iter().chain(values.iter());
            let rows = stmt.query_map(rusqlite::params_from_iter(params), memory_turn_row)?;
            for row in rows {
                let (memory_id, parent, turn) = row?;
                let key = if wanted.contains(memory_id.as_str()) {
                    memory_id
                } else {
                    match parent {
                        Some(parent) => parent,
                        None => continue,
                    }
                };
                merge_turn(&mut result, key, turn);
            }
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
        let min_idx = turn_index.saturating_sub(radius);
        let max_idx = turn_index.saturating_add(radius);
        let mut stmt = conn.prepare_cached(
            "SELECT memory_id, parent_memory_id, entity_id, session_id, role, turn_index,
                    content, created_at_ms, content_hash
             FROM memories
             WHERE entity_id = ?1 AND session_id = ?2 AND turn_index BETWEEN ?3 AND ?4
               AND (parent_memory_id IS NULL
                    OR memory_id GLOB parent_memory_id || '::c[0-9]*')
             ORDER BY turn_index, rowid",
        )?;
        let rows =
            stmt.query_map(params![entity_id, session_id, min_idx, max_idx], memory_turn_row)?;
        let mut by_turn = std::collections::HashMap::new();
        let mut order = Vec::new();
        for row in rows {
            let (memory_id, parent, turn) = row?;
            let key = parent.unwrap_or(memory_id);
            if !by_turn.contains_key(&key) {
                order.push(key.clone());
            }
            merge_turn(&mut by_turn, key, turn);
        }
        Ok(order.into_iter().filter_map(|key| by_turn.remove(&key)).collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::{SessionRouterRecord, TenantStore};
    use tempfile::tempdir;

    // Pins the AUDIT-2026-09-21.md "entity_pivot_sessions unconditional full
    // scan" fix: the LIKE backstop must be skipped once FTS already found an
    // entity for a session, but must still run when FTS's tokenizer misses.
    #[test]
    fn entity_pivot_sessions_skips_like_backstop_after_fts_hit() {
        let temp = tempdir().unwrap();
        let store = TenantStore::new(&temp.path().join("tenant.db")).unwrap();

        // "paris" is its own FTS token; "sam" only appears glued inside
        // "xsamx", so FTS5's tokenizer can never match it as a whole token.
        let record = SessionRouterRecord {
            session_id: "s1".to_string(),
            entity_id: "alice".to_string(),
            router_text: "paris and xsamx".to_string(),
            ..Default::default()
        };
        store.merge_session_router_records_batch(&[record]).unwrap();

        let hits = store
            .entity_pivot_sessions("alice", &["paris".to_string(), "sam".to_string()])
            .unwrap();
        assert_eq!(hits.len(), 1);
        // "paris" contributes exactly one hit (FTS only — the backstop must
        // not also count it, or every FTS hit would be silently
        // double-weighted). "sam" contributes exactly one hit via the LIKE
        // backstop, since FTS missed it — recall for names FTS5 swallows
        // into a larger token must survive this change.
        assert_eq!(hits[0].entity_hits, 2);
    }
}
