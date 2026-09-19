use super::prelude::*;

impl TenantStore {
    pub fn set_memory_links_batch(&self, links: &[(String, String, String)]) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR IGNORE INTO memory_links (source_memory_id, target_memory_id, link_type)
                 VALUES (?1, ?2, ?3)",
            )?;
            for (src, tgt, link_type) in links {
                stmt.execute(params![src, tgt, link_type])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn get_linked_memories(&self, memory_id: &str) -> Result<Vec<String>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT target_memory_id FROM memory_links WHERE source_memory_id = ?1
             UNION ALL
             SELECT source_memory_id FROM memory_links WHERE target_memory_id = ?1",
        )?;
        let rows = stmt.query_map(params![memory_id], |row| row.get::<_, String>(0))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    pub fn get_link_cluster_scores(
        &self,
        seed_memory_id: &str,
        max_depth: usize,
    ) -> Result<HashMap<String, f32>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "WITH RECURSIVE
               bfs(node, depth, path_weight) AS (
                 SELECT ?1, 0, 1.0
                 -- UNION (not ALL): links are stored in both directions, so the
                 -- same neighbour is reached twice per hop and was double-counted.
                 UNION
                 SELECT
                   CASE WHEN ml.source_memory_id = bfs.node THEN ml.target_memory_id ELSE ml.source_memory_id END,
                   bfs.depth + 1,
                   bfs.path_weight * 0.6
                 FROM bfs
                 JOIN memory_links ml ON ml.source_memory_id = bfs.node OR ml.target_memory_id = bfs.node
                 WHERE bfs.depth < ?2
               )
             SELECT node, SUM(path_weight) FROM bfs WHERE depth > 0 GROUP BY node;"
        )?;
        let rows = stmt.query_map(params![seed_memory_id, max_depth as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
        })?;
        let mut result = HashMap::new();
        for row in rows {
            let (node, weight) = row?;
            result.insert(node, weight);
        }
        Ok(result)
    }

    pub fn get_edge_cluster_neighbors(
        &self,
        seed_memory_id: &str,
        edge_type_filter: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "WITH seed_nodes AS (
                 SELECT source AS node FROM edges WHERE memory_id = ?1
                 UNION
                 SELECT target AS node FROM edges WHERE memory_id = ?1
             )
             SELECT DISTINCT e.memory_id, e.weight
             FROM edges e
             JOIN seed_nodes sn ON (e.source = sn.node OR e.target = sn.node)
             WHERE e.memory_id != ?1
               AND (?2 IS NULL OR e.edge_type = ?2)
             ORDER BY e.weight DESC
             LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![seed_memory_id, edge_type_filter, limit as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)? as f32))
            })?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    pub fn get_edge_cluster_neighbors_typed(
        &self,
        seed_memory_id: &str,
        edge_type_filter: Option<&str>,
        limit: usize,
        max_node_degree: usize,
    ) -> Result<Vec<(String, f32, String)>> {
        let conn = self.get_conn()?;
        // Nodes with more than `max_node_degree` edges are not traversed. Such
        // hubs (the entity id, speaker labels like "assistant", header words
        // from derived text, the empty name) connect nearly every memory, carry
        // no relational signal, and made each hop scan thousands of edges.
        // Degree counts are capped so checking a hub stays cheap.
        let mut stmt = conn.prepare_cached(
            "WITH seed_nodes AS (
                 SELECT node FROM (
                     SELECT source AS node FROM edges WHERE memory_id = ?1
                     UNION ALL
                     SELECT target AS node FROM edges WHERE memory_id = ?1
                 )
                 WHERE node != ''
                   AND (SELECT COUNT(*) FROM (SELECT 1 FROM edges x WHERE x.source = node LIMIT ?4 + 1))
                     + (SELECT COUNT(*) FROM (SELECT 1 FROM edges y WHERE y.target = node LIMIT ?4 + 1))
                     <= ?4
             )
             SELECT memory_id, weight, edge_type FROM (
                 SELECT e.memory_id, e.weight, e.edge_type
                 FROM edges e
                 JOIN seed_nodes sn ON e.source = sn.node
                 WHERE e.memory_id != ?1
                   AND (?2 IS NULL OR e.edge_type = ?2)
                 UNION ALL
                 SELECT e.memory_id, e.weight, e.edge_type
                 FROM edges e
                 JOIN seed_nodes sn ON e.target = sn.node
                 WHERE e.memory_id != ?1
                   AND (?2 IS NULL OR e.edge_type = ?2)
             )
             ORDER BY weight DESC, memory_id
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(
            params![seed_memory_id, edge_type_filter, limit as i64, max_node_degree as i64],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, f64>(1)? as f32,
                    row.get::<_, String>(2)?,
                ))
            },
        )?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn get_edge_cluster_neighbors_batch(
        &self,
        memory_ids: &[String],
        edge_type_filter: Option<&str>,
        limit: usize,
        max_node_degree: usize,
    ) -> Result<HashMap<String, Vec<EdgeNeighbour>>> {
        const CHUNK: usize = 400;
        let conn = self.get_conn()?;
        let placeholders = |n: usize| vec!["?"; n].join(",");

        // Node multiset per memory (a node listed once per edge endpoint).
        let mut nodes_of: HashMap<String, Vec<String>> = HashMap::new();
        for ids in memory_ids.chunks(CHUNK) {
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT memory_id, source, target FROM edges WHERE memory_id IN ({})",
                placeholders(ids.len())
            ))?;
            let rows = stmt.query_map(rusqlite::params_from_iter(ids), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
            })?;
            for row in rows {
                let (memory_id, source, target) = row?;
                let nodes = nodes_of.entry(memory_id).or_default();
                nodes.extend([source, target].into_iter().filter(|n| !n.is_empty()));
            }
        }

        let mut unique: Vec<String> = nodes_of.values().flatten().cloned().collect();
        unique.sort();
        unique.dedup();

        // Degree = edges with the node as source + as target; hubs are skipped.
        let mut degree: HashMap<String, usize> = HashMap::new();
        for column in ["source", "target"] {
            for nodes in unique.chunks(CHUNK) {
                let mut stmt = conn.prepare_cached(&format!(
                    "SELECT {column}, COUNT(*) FROM edges WHERE {column} IN ({}) GROUP BY {column}",
                    placeholders(nodes.len())
                ))?;
                let rows = stmt.query_map(rusqlite::params_from_iter(nodes), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
                })?;
                for row in rows {
                    let (node, count) = row?;
                    *degree.entry(node).or_default() += count;
                }
            }
        }
        let traversable: Vec<String> = unique
            .into_iter()
            .filter(|n| degree.get(n).copied().unwrap_or(0) <= max_node_degree)
            .collect();

        type Incident = HashMap<String, Vec<(String, f32, String)>>;
        let mut incident: [Incident; 2] = [HashMap::new(), HashMap::new()];
        for (slot, column) in ["source", "target"].into_iter().enumerate() {
            for nodes in traversable.chunks(CHUNK) {
                // The filter is bound last: bare `?` markers number from 1.
                let filter_idx = nodes.len() + 1;
                let mut stmt = conn.prepare_cached(&format!(
                    "SELECT {column}, memory_id, weight, edge_type FROM edges
                     WHERE {column} IN ({}) AND memory_id IS NOT NULL
                       AND (?{filter_idx} IS NULL OR edge_type = ?{filter_idx})",
                    placeholders(nodes.len())
                ))?;
                let mut params: Vec<&dyn rusqlite::types::ToSql> =
                    nodes.iter().map(|n| n as &dyn rusqlite::types::ToSql).collect();
                params.push(&edge_type_filter);
                let rows = stmt.query_map(params.as_slice(), |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, f64>(2)? as f32,
                        row.get::<_, String>(3)?,
                    ))
                })?;
                for row in rows {
                    let (node, memory_id, weight, edge_type) = row?;
                    incident[slot].entry(node).or_default().push((memory_id, weight, edge_type));
                }
            }
        }

        let traversable: std::collections::HashSet<&str> =
            traversable.iter().map(String::as_str).collect();
        let mut result = HashMap::with_capacity(memory_ids.len());
        for memory_id in memory_ids {
            let mut rows: Vec<(String, f32, String)> = Vec::new();
            for node in nodes_of.get(memory_id).into_iter().flatten() {
                if !traversable.contains(node.as_str()) {
                    continue;
                }
                for side in &incident {
                    rows.extend(
                        side.get(node)
                            .into_iter()
                            .flatten()
                            .filter(|(mid, _, _)| mid != memory_id)
                            .cloned(),
                    );
                }
            }
            rows.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            rows.truncate(limit);
            result.insert(memory_id.clone(), rows);
        }
        Ok(result)
    }

    pub fn graph_upsert_memory_batch(&self, batch: &GraphEdgeBatch<'_>) -> Result<()> {
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO edges (edge_id, source, target, edge_type, label, status, timestamp_ms, memory_id, weight)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )?;
            for entry in batch {
                // An empty node name would join every such edge into one hub.
                if entry.subject.trim().is_empty() || entry.object.trim().is_empty() {
                    continue;
                }
                let edge_id =
                    format!("edge::{}::{}::{}", entry.memory_id, entry.subject, entry.predicate);
                let label = format!("{} {} {}", entry.subject, entry.predicate, entry.object);
                let weight = crate::graph::EdgeType::from_str(entry.predicate).default_weight();
                stmt.execute(params![
                    edge_id,
                    entry.subject,
                    entry.object,
                    entry.predicate,
                    label,
                    entry.status,
                    entry.timestamp as i64,
                    entry.memory_id,
                    weight as f64,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn graph_insert_edges_batch(
        &self,
        edges: &[(&str, &str, &str, &str, u64)],
    ) -> Result<usize> {
        let mut written = 0;
        if edges.is_empty() {
            return Ok(written);
        }
        let mut conn = self.get_conn()?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR IGNORE INTO edges (edge_id, source, target, edge_type, label, status, timestamp_ms, memory_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, 'current', ?6, ?7)",
            )?;
            for (memory_id, subject, predicate, object, timestamp_ms) in edges {
                if subject.is_empty() || predicate.is_empty() || object.is_empty() {
                    continue;
                }
                written += stmt.execute(params![
                    format!("edge::{memory_id}::{subject}::{predicate}"),
                    subject,
                    object,
                    predicate,
                    format!("{subject} {predicate} {object}"),
                    *timestamp_ms as i64,
                    memory_id
                ])?;
            }
        }
        tx.commit()?;
        Ok(written)
    }

    pub fn graph_upsert_fact_status_batch(
        &self,
        _entity_id: &str,
        batch: &GraphEdgeBatch<'_>,
    ) -> Result<()> {
        self.graph_upsert_memory_batch(batch)
    }

    pub fn graph_edge_summaries_for_label(
        &self,
        entity_id: &str,
        label: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT label FROM edges WHERE (source = ?1 OR target = ?1) AND label LIKE ?2 LIMIT ?3",
        )?;
        let rows = stmt
            .query_map(params![entity_id, format!("%{}%", label), limit as i64], |row| {
                row.get::<_, String>(0)
            })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn graph_query_edges(
        &self,
        entity: &str,
        _label: Option<&str>,
        direction: &str,
        limit: usize,
    ) -> Result<Vec<GraphEdge>> {
        let conn = self.get_conn()?;
        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match direction {
            "Inbound" => (
                "SELECT edge_id, source, target, edge_type, label, weight, timestamp_ms, memory_id
                 FROM edges WHERE target = ?1 ORDER BY timestamp_ms DESC LIMIT ?2"
                    .to_string(),
                vec![Box::new(entity.to_string()), Box::new(limit as i64)],
            ),
            "Both" => (
                format!(
                    "SELECT edge_id, source, target, edge_type, label, weight, timestamp_ms, memory_id
                     FROM edges WHERE (source = ?1 OR target = ?1) AND edge_type != '{}'
                     ORDER BY timestamp_ms DESC LIMIT ?2",
                    crate::graph::EdgeType::Default.as_str()
                ),
                vec![Box::new(entity.to_string()), Box::new(limit as i64)],
            ),
            _ => (
                "SELECT edge_id, source, target, edge_type, label, weight, timestamp_ms, memory_id
                 FROM edges WHERE source = ?1 ORDER BY timestamp_ms DESC LIMIT ?2"
                    .to_string(),
                vec![Box::new(entity.to_string()), Box::new(limit as i64)],
            ),
        };
        let mut stmt = conn.prepare_cached(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), |row| {
            Ok(GraphEdge {
                edge_id: row.get(0)?,
                source: row.get(1)?,
                target: row.get(2)?,
                edge_type: row.get(3)?,
                label: row.get(4)?,
                weight: row.get::<_, f64>(5)? as f32,
                timestamp_ms: row.get::<_, i64>(6)? as u64,
                memory_id: row.get(7)?,
            })
        })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn graph_remove_memory(&self, memory_id: &str) -> Result<usize> {
        let conn = self.get_conn()?;
        let count = conn.execute("DELETE FROM edges WHERE memory_id = ?1", params![memory_id])?;
        Ok(count)
    }

    pub fn graph_clear(&self) -> Result<()> {
        let conn = self.get_conn()?;
        conn.execute("DELETE FROM edges", [])?;
        Ok(())
    }

    pub fn get_all_edges(&self, limit: usize) -> Result<Vec<GraphEdge>> {
        let conn = self.get_conn()?;
        let mut stmt = conn.prepare_cached(
            "SELECT edge_id, source, target, edge_type, label, weight, timestamp_ms, memory_id FROM edges ORDER BY weight DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map([limit as i64], |row| {
            Ok(GraphEdge {
                edge_id: row.get(0)?,
                source: row.get(1)?,
                target: row.get(2)?,
                edge_type: row.get(3)?,
                label: row.get(4)?,
                weight: row.get(5)?,
                timestamp_ms: row.get(6)?,
                memory_id: row.get(7)?,
            })
        })?;
        let mut results = Vec::with_capacity(rows.size_hint().0);
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}
