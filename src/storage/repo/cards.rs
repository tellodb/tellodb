use super::prelude::*;

impl TenantStore {
    pub fn ingest_cards(&self, cards: &[MemoryCard]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO memory_cards (
                    card_id, entity_id, user_id, source_memory_id, source_session_id,
                    subject, predicate, object, memory_text, card_type, confidence,
                    is_latest, is_static, is_inference, expires_at, root_card_id, parent_card_id,
                    lifecycle, created_at_ms, updated_at_ms
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20)",
            )?;
            for card in cards {
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
                    card.is_latest as i32,
                    card.is_static as i32,
                    card.is_inference as i32,
                    card.expires_at,
                    card.root_card_id,
                    card.parent_card_id,
                    card.lifecycle
                        .as_ref()
                        .map(serde_json::to_string)
                        .transpose()
                        .context("failed to serialize card lifecycle")?
                        .as_deref()
                        .unwrap_or(""),
                    card.created_at_ms,
                    card.updated_at_ms,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn get_memory_card(&self, card_id: &str) -> Result<Option<MemoryCard>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT card_id, entity_id, user_id, source_memory_id, source_session_id,
                    subject, predicate, object, memory_text, card_type, confidence,
                    is_latest, is_static, is_inference, expires_at, root_card_id, parent_card_id,
                    lifecycle, created_at_ms, updated_at_ms
             FROM memory_cards WHERE card_id = ?1",
        )?;
        let res = stmt.query_row(params![card_id], memory_card_row);
        match res {
            Ok(card) => Ok(Some(card)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn set_memory_card_latest_batch(&self, updates: &[(String, bool, u64)]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "UPDATE memory_cards SET is_latest = ?1, updated_at_ms = ?2 WHERE card_id = ?3",
            )?;
            for (card_id, is_latest, ts) in updates {
                stmt.execute(params![*is_latest as i32, *ts as i64, card_id])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn search_memory_cards(
        &self,
        query: &MemoryCardSearchInput<'_>,
    ) -> Result<Vec<MemoryCardSearchHit>> {
        if query.limit == 0 {
            return Ok(Vec::new());
        }
        let now_ms = unix_timestamp_ms()? as u64;
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT card_id, source_memory_id, source_session_id, subject, predicate, object,
                    memory_text, card_type, confidence, is_latest, expires_at, created_at_ms
             FROM memory_cards WHERE entity_id = ?1",
        )?;
        let rows = stmt.query_map(params![query.entity_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, f64>(8)? as f32,
                row.get::<_, i32>(9)? != 0,
                row.get::<_, Option<i64>>(10)?.map(|v| v as u64),
                row.get::<_, i64>(11)? as u64,
            ))
        })?;

        let mut hits = Vec::new();
        for row in rows {
            let (
                card_id,
                source_memory_id,
                source_session_id,
                subject,
                predicate,
                object,
                memory_text,
                card_type,
                confidence,
                is_latest,
                expires_at,
                created_at_ms,
            ) = row?;

            if !query.include_stale && !is_latest {
                continue;
            }
            if expires_at.map(|exp| exp <= now_ms).unwrap_or(false) {
                continue;
            }

            let text = format!(
                "{} {} {} {} {} {}",
                subject, predicate, object, memory_text, card_type, source_session_id
            );
            let lower = text.to_ascii_lowercase();
            let lexical_hits = contains_term_count(&lower, query.lexical_terms);
            let temporal_hits = contains_term_count(&lower, query.temporal_terms);
            let entity_hits = contains_term_count(&lower, query.entities);
            let routed = query.route_sessions.contains(&source_session_id);

            if lexical_hits == 0 && temporal_hits == 0 && entity_hits == 0 && !routed {
                continue;
            }

            let lexical_coverage = if query.lexical_terms.is_empty() {
                0.0
            } else {
                lexical_hits as f32 / query.lexical_terms.len() as f32
            };
            let temporal_coverage = if query.temporal_terms.is_empty() {
                0.0
            } else {
                temporal_hits as f32 / query.temporal_terms.len() as f32
            };
            let entity_coverage = if query.entities.is_empty() {
                0.0
            } else {
                entity_hits as f32 / query.entities.len() as f32
            };

            let type_boost = match card_type.as_str() {
                "fact" => FACT_TYPE_BOOST,
                "preference" | "profile" => PREFERENCE_TYPE_BOOST,
                "event" | "episode" => EVENT_TYPE_BOOST,
                "decision" => DECISION_TYPE_BOOST,
                "inference" => INFERENCE_TYPE_BOOST,
                _ => OTHER_TYPE_BOOST,
            };
            let latest_boost = if is_latest { CARD_LATEST_BOOST } else { CARD_STALE_PENALTY };
            let route_boost = if routed { CARD_ROUTE_BOOST } else { 0.0 };
            let score = lexical_coverage.min(1.0) * CARD_LEXICAL_WEIGHT
                + temporal_coverage.min(1.0) * CARD_TEMPORAL_WEIGHT
                + entity_coverage.min(1.0) * CARD_ENTITY_WEIGHT
                + route_boost
                + type_boost
                + latest_boost
                + confidence.clamp(0.0, 1.0) * CARD_CONFIDENCE_WEIGHT;

            hits.push(MemoryCardSearchHit {
                card_id,
                source_memory_id,
                source_session_id,
                timestamp: created_at_ms,
                score,
                lexical_hits,
                temporal_hits,
                entity_hits,
            });
        }

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.card_id.cmp(&b.card_id))
        });
        hits.truncate(query.limit);
        Ok(hits)
    }

    pub fn get_memory_card_by_source(&self, source_memory_id: &str) -> Result<Option<MemoryCard>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT card_id, entity_id, user_id, source_memory_id, source_session_id,
                    subject, predicate, object, memory_text, card_type, confidence,
                    is_latest, is_static, is_inference, expires_at, root_card_id, parent_card_id,
                    lifecycle, created_at_ms, updated_at_ms
             FROM memory_cards WHERE source_memory_id = ?1
             ORDER BY is_latest DESC, updated_at_ms DESC LIMIT 1",
        )?;
        let res = stmt.query_row(params![source_memory_id], memory_card_row);
        match res {
            Ok(card) => Ok(Some(card)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn get_memory_cards_batch(
        &self,
        card_ids: &[String],
    ) -> Result<std::collections::HashMap<String, MemoryCard>> {
        if card_ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let conn = self.get_conn()?;
        let mut results = std::collections::HashMap::new();
        for chunk in card_ids.chunks(IN_CHUNK) {
            let values = padded_in_chunk(chunk);
            let sql = format!(
                "SELECT card_id, entity_id, user_id, source_memory_id, source_session_id,
                        subject, predicate, object, memory_text, card_type, confidence,
                        is_latest, is_static, is_inference, expires_at, root_card_id, parent_card_id,
                        lifecycle, created_at_ms, updated_at_ms
                 FROM memory_cards WHERE card_id IN ({})",
                in_placeholders(IN_CHUNK)
            );
            let mut stmt = conn.prepare_cached(&sql)?;
            let rows = stmt.query_map(rusqlite::params_from_iter(values), |row| {
                let card = memory_card_row(row)?;
                Ok((card.card_id.clone(), card))
            })?;
            for row in rows {
                let (card_id, card) = row?;
                results.insert(card_id, card);
            }
        }
        Ok(results)
    }
}
