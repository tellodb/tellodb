use super::prelude::*;

impl TenantStore {
    pub fn set_preference_memories_batch(
        &self,
        entity_id: &str,
        items: &[(String, f32)],
    ) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO preferences (entity_id, memory_id, strength) VALUES (?1, ?2, ?3)",
            )?;
            for (memory_id, strength) in items {
                stmt.execute(params![entity_id, memory_id, strength])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn get_preference_memories(
        &self,
        entity_id: &str,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let conn = self.get_conn()?;
        let mut stmt =
            conn.prepare_cached("SELECT memory_id, strength FROM preferences WHERE entity_id = ?1 ORDER BY strength DESC LIMIT ?2")?;
        let rows = stmt.query_map(params![entity_id, limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f32>(1)?))
        })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}
