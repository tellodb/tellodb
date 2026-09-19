use super::prelude::*;

impl TenantStore {
    pub fn get_deletion_tombstones_for_target(
        &self,
        memory_id: &str,
        limit: usize,
    ) -> Result<Vec<crate::lifecycle::DeletionTombstone>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT tombstone_json FROM deletion_tombstones WHERE target_memory_id = ?1 LIMIT ?2",
        )?;
        let rows =
            stmt.query_map(params![memory_id, limit as i64], |row| row.get::<_, String>(0))?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            let json = row?;
            if let Ok(tombstone) =
                serde_json::from_str::<crate::lifecycle::DeletionTombstone>(&json)
            {
                results.push(tombstone);
            }
        }
        Ok(results)
    }

    pub fn clear_all(&self) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let tables: Vec<String> = {
            let mut stmt = tx.prepare(
                "SELECT name FROM sqlite_master
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                 AND (sql LIKE 'CREATE VIRTUAL TABLE%' OR name NOT IN (
                     SELECT m.name FROM sqlite_master v, sqlite_master m
                     WHERE v.sql LIKE 'CREATE VIRTUAL TABLE%' AND m.name LIKE v.name || '_%'
                 ))",
            )?;
            let rows = stmt.query_map([], |row| row.get(0))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        for table in tables {
            tx.execute(&format!("DELETE FROM \"{}\"", table.replace('"', "\"\"")), [])?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn db_stats(&self) -> Result<crate::storage::types::CoreClusterStats> {
        let conn = self.get_conn()?;

        let memory_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0)).unwrap_or(0);

        let entity_count: i64 = conn
            .query_row("SELECT COUNT(DISTINCT entity_id) FROM memories", [], |row| row.get(0))
            .unwrap_or(0);

        let fact_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM fact_versions WHERE status = 'current'", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);

        // Storage size in bytes
        let page_count: i64 =
            conn.query_row("PRAGMA page_count", [], |row| row.get(0)).unwrap_or(0);
        let page_size: i64 = conn.query_row("PRAGMA page_size", [], |row| row.get(0)).unwrap_or(0);
        let storage_bytes = page_count * page_size;

        Ok(crate::storage::types::CoreClusterStats {
            memory_count: memory_count as usize,
            entity_count: entity_count as usize,
            fact_count: fact_count as usize,
            storage_bytes: storage_bytes as usize,
            request_count: 0,
            ingest_count: 0,
            query_count: 0,
        })
    }

    pub fn detailed_db_stats(&self) -> Result<crate::api::types::StorageStatsResponse> {
        let conn = self.get_conn()?;
        let count = |table: &str| -> i64 {
            conn.query_row(&format!("SELECT COUNT(*) FROM {}", table), [], |row| row.get(0))
                .unwrap_or(0)
        };
        let page_count: i64 =
            conn.query_row("PRAGMA page_count", [], |row| row.get(0)).unwrap_or(0);
        let page_size: i64 = conn.query_row("PRAGMA page_size", [], |row| row.get(0)).unwrap_or(0);
        let storage_bytes = page_count * page_size;
        let free_pages: i64 =
            conn.query_row("PRAGMA freelist_count", [], |row| row.get(0)).unwrap_or(0);
        Ok(crate::api::types::StorageStatsResponse {
            used_bytes: ((page_count - free_pages).max(0) * page_size) as usize,
            memory_card_count: count("memory_cards") as usize,
            edge_count: count("edges") as usize,
            memory_count: count("memories") as usize,
            metric_count: count("metrics") as usize,
            session_router_count: count("session_router") as usize,
            fact_version_count: count("fact_versions") as usize,
            memory_link_count: count("memory_links") as usize,
            alias_count: count("aliases") as usize,
            preference_count: count("preferences") as usize,
            core_profile_count: count("core_profiles") as usize,
            deletion_tombstone_count: count("deletion_tombstones") as usize,
            storage_bytes: storage_bytes as usize,
        })
    }

    pub fn expire_records(&self, now_ms: u64) -> Result<usize> {
        let mut conn = self.get_conn()?;
        let mut updates = Vec::new();
        {
            let mut stmt = conn.prepare_cached(
                "SELECT card_id, lifecycle FROM memory_cards WHERE expires_at IS NOT NULL AND expires_at <= ?1"
            )?;
            let rows = stmt.query_map(params![now_ms as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;

            for row in rows {
                let (card_id, lifecycle_json) = row?;
                if let Ok(mut lifecycle) =
                    serde_json::from_str::<crate::lifecycle::LifecycleMetadata>(&lifecycle_json)
                {
                    if lifecycle.lifecycle_state != crate::lifecycle::LifecycleState::Expired
                        && matches!(
                            lifecycle.retention_class,
                            crate::lifecycle::RetentionClass::Ephemeral
                                | crate::lifecycle::RetentionClass::Working
                        )
                    {
                        lifecycle.lifecycle_state = crate::lifecycle::LifecycleState::Expired;
                        if let Ok(updated_json) = serde_json::to_string(&lifecycle) {
                            updates.push((card_id, updated_json));
                        }
                    }
                }
            }
        }

        let count = updates.len();
        if count > 0 {
            let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            {
                let mut update_stmt = tx.prepare_cached(
                    "UPDATE memory_cards SET lifecycle = ?1, updated_at_ms = ?2 WHERE card_id = ?3",
                )?;
                for (card_id, updated_json) in updates {
                    update_stmt.execute(params![updated_json, now_ms as i64, card_id])?;
                }
            }
            tx.commit()?;
        }
        Ok(count)
    }
}
