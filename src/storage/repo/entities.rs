use super::prelude::*;

impl TenantStore {
    pub fn set_aliases_batch(&self, entity_id: &str, aliases: &[(String, String)]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR IGNORE INTO aliases (entity_id, alias) VALUES (?1, ?2)",
            )?;
            for (alias, _canonical) in aliases {
                stmt.execute(params![entity_id, alias])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn register_entity(&self, entity_id: &str, canonical_name: &str) -> Result<()> {
        let now =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        let sk = crate::storage::entity_resolver::phonetic_key(canonical_name);
        let conn = self.get_conn()?;
        conn.execute(
            "INSERT OR IGNORE INTO entity_registry (entity_id, canonical_name, soundex_key, updated_at_ms)
             VALUES (?1, ?2, ?3, ?4)",
            params![entity_id, canonical_name, sk, now as i64],
        )?;
        Ok(())
    }

    pub fn load_entity_candidates(
        &self,
        entity_id: &str,
    ) -> Result<Vec<crate::storage::entity_resolver::EntityCandidate>> {
        let conn = self.get_conn()?;

        let mut reg_stmt = conn.prepare_cached(
            "SELECT canonical_name, soundex_key FROM entity_registry WHERE entity_id = ?1",
        )?;
        let rows = reg_stmt.query_map(params![entity_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        let mut candidates: Vec<crate::storage::entity_resolver::EntityCandidate> = Vec::new();
        let mut name_index: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for row in rows {
            let (name, sk) = row?;
            let idx = candidates.len();
            candidates.push(crate::storage::entity_resolver::EntityCandidate {
                name: name.clone(),
                aliases: Vec::new(),
                soundex_key: sk,
                embedding: None,
            });
            name_index.insert(name.to_ascii_lowercase(), idx);
        }

        let mut alias_stmt =
            conn.prepare_cached("SELECT alias FROM aliases WHERE entity_id = ?1")?;
        let alias_rows = alias_stmt.query_map(params![entity_id], |row| row.get::<_, String>(0))?;
        for alias in alias_rows {
            let alias = alias?;
            for (lower_name, idx) in &name_index {
                let candidate = &mut candidates[*idx];
                if alias.to_ascii_lowercase().contains(lower_name)
                    || lower_name.contains(&alias.to_ascii_lowercase())
                {
                    if !candidate.aliases.contains(&alias) {
                        candidate.aliases.push(alias.clone());
                    }
                    break;
                }
            }
        }

        let mut emb_stmt = conn.prepare_cached(
            "SELECT canonical_name, embedding_blob, dim FROM name_embeddings WHERE entity_id = ?1",
        )?;
        let emb_rows = emb_stmt.query_map(params![entity_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, i64>(2)? as usize,
            ))
        })?;
        for row in emb_rows {
            let (cname, blob, dim) = row?;
            if let Some(idx) = name_index.get(&cname.to_ascii_lowercase()) {
                let embedding: Vec<f32> = blob
                    .chunks(4)
                    .take(dim)
                    .filter_map(|chunk| {
                        if chunk.len() == 4 {
                            Some(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        } else {
                            None
                        }
                    })
                    .collect();
                if embedding.len() == dim {
                    candidates[*idx].embedding = Some(embedding);
                }
            }
        }

        Ok(candidates)
    }

    fn proposal_id(entity_id: &str, from: &str, to: &str, now_ms: u64) -> String {
        format!("merge::{entity_id}::{from}::{to}::{now_ms}")
    }

    pub fn create_merge_proposal(
        &self,
        entity_id: &str,
        from_name: &str,
        to_name: &str,
        tier: &str,
        confidence: f32,
    ) -> Result<Option<String>> {
        let now =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        let pid = Self::proposal_id(entity_id, from_name, to_name, now);
        let conn = self.get_conn()?;
        let inserted = conn.execute(
            "INSERT OR IGNORE INTO merge_proposals (proposal_id, entity_id, from_name, to_name, tier, confidence, status, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', ?7)",
            params![pid, entity_id, from_name, to_name, tier, f64::from(confidence), now as i64],
        )?;
        if inserted > 0 {
            Ok(Some(pid))
        } else {
            Ok(None)
        }
    }

    pub fn resolve_and_propose(
        &self,
        entity_id: &str,
        name: &str,
        name_embedding: Option<&[f32]>,
        config: &crate::storage::entity_resolver::ResolutionConfig,
    ) -> Result<crate::storage::entity_resolver::EntityResolution> {
        let candidates = self.load_entity_candidates(entity_id)?;
        let resolution = crate::storage::entity_resolver::resolve_name(
            name,
            &candidates,
            name_embedding,
            config,
        );
        if let Some(ref matched) = resolution.matched_name {
            let tier_label = match resolution.tier {
                crate::storage::entity_resolver::ResolverTier::Exact => return Ok(resolution),
                crate::storage::entity_resolver::ResolverTier::Fuzzy(_) => "fuzzy",
                crate::storage::entity_resolver::ResolverTier::Phonetic => "phonetic",
                crate::storage::entity_resolver::ResolverTier::Embedding(_) => "embedding",
            };
            if !name.eq_ignore_ascii_case(matched) {
                let _ = self.create_merge_proposal(
                    entity_id,
                    name,
                    matched,
                    tier_label,
                    resolution.tier.confidence(),
                );
            }
        }
        Ok(resolution)
    }

    pub fn get_core_profile(&self, entity_id: &str) -> Result<Option<String>> {
        let conn = self.get_conn()?;
        let mut stmt =
            conn.prepare_cached("SELECT profile_json FROM core_profiles WHERE entity_id = ?1")?;
        let res = stmt.query_row(params![entity_id], |row| row.get::<_, String>(0));
        match res {
            Ok(json) => Ok(Some(json)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn update_core_profile(
        &self,
        entity_id: &str,
        update: impl FnOnce(Option<String>) -> Option<String>,
    ) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let current = match tx.query_row(
            "SELECT profile_json FROM core_profiles WHERE entity_id = ?1",
            params![entity_id],
            |row| row.get::<_, String>(0),
        ) {
            Ok(json) => Some(json),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(err) => return Err(err.into()),
        };
        if let Some(next) = update(current) {
            tx.execute(
                "INSERT OR REPLACE INTO core_profiles (entity_id, profile_json, updated_at_ms) VALUES (?1, ?2, ?3)",
                params![entity_id, next, unix_timestamp_ms()?],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn set_core_profile(&self, entity_id: &str, profile_json: &str) -> Result<()> {
        let conn = self.get_conn()?;
        let now = unix_timestamp_ms()?;
        conn.execute(
            "INSERT OR REPLACE INTO core_profiles (entity_id, profile_json, updated_at_ms) VALUES (?1, ?2, ?3)",
            params![entity_id, profile_json, now],
        )?;
        Ok(())
    }
}
