use anyhow::Context;
use anyhow::Result;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::error;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

#[derive(Clone)]
pub struct VectorIndex {
    index: Arc<RwLock<Index>>,
    vector_to_entity: Arc<RwLock<HashMap<u64, String>>>,
    entity_counts: Arc<RwLock<HashMap<String, usize>>>,
    dimensions: usize,
    max_elements: usize,
    connectivity: usize,
    expansion_add: usize,
    expansion_search: usize,
    persist_dir: Option<Arc<PathBuf>>,
    dirty: Arc<AtomicBool>,
}

impl VectorIndex {
    const ENTITY_COUNT_SMALL_THRESHOLD: usize = 50;
    const ENTITY_COUNT_MEDIUM_THRESHOLD: usize = 500;
    const EXPANSION_MIN_LIMIT: usize = 10;
    const EXPANSION_MULTIPLIER_SMALL: usize = 20;
    const EXPANSION_MULTIPLIER_MEDIUM: usize = 10;
    const EXPANSION_MULTIPLIER_LARGE: usize = 5;

    pub fn new(
        dimensions: usize,
        max_elements: usize,
        connectivity: usize,
        expansion_add: usize,
        expansion_search: usize,
        persist_dir: Option<&str>,
    ) -> Result<Self> {
        let options = IndexOptions {
            dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity,
            expansion_add,
            expansion_search,
            multi: false,
        };
        let index = Index::new(&options).context("failed to create usearch index")?;
        index.reserve(max_elements).context("failed to reserve capacity in usearch index")?;

        let persist_dir_path = persist_dir.map(|p| Arc::new(PathBuf::from(p)));
        let mut vector_to_entity: HashMap<u64, String> = HashMap::new();

        if let Some(dir) = &persist_dir_path {
            let unified_path = dir.join("unified.hnsw");
            let mapping_path = dir.join("unified.mapping");

            if unified_path.exists() {
                if std::fs::metadata(&unified_path).map(|m| m.len()).unwrap_or(0) > 0 {
                    index
                        .load(unified_path.to_string_lossy().as_ref())
                        .context("failed to load hnsw index from disk")?;
                }
                if mapping_path.exists() {
                    if let Ok(contents) = std::fs::read_to_string(&mapping_path) {
                        for line in contents.lines() {
                            let parts: Vec<&str> = line.split(',').collect();
                            if parts.len() == 2 {
                                if let Ok(id) = parts[0].parse::<u64>() {
                                    vector_to_entity.insert(id, parts[1].to_string());
                                }
                            }
                        }
                    }
                }
            } else {
                std::fs::create_dir_all(&**dir).context("failed to create persist directory")?;
            }
        }

        let entity_counts = {
            let mut counts: HashMap<String, usize> = HashMap::new();
            for eid in vector_to_entity.values() {
                *counts.entry(eid.clone()).or_insert(0) += 1;
            }
            counts
        };

        Ok(Self {
            index: Arc::new(RwLock::new(index)),
            vector_to_entity: Arc::new(RwLock::new(vector_to_entity)),
            entity_counts: Arc::new(RwLock::new(entity_counts)),
            dimensions,
            max_elements,
            connectivity,
            expansion_add,
            expansion_search,
            persist_dir: persist_dir_path,
            dirty: Arc::new(AtomicBool::new(false)),
        })
    }

    fn expanded_limit(&self, entity_id: Option<&str>, requested: usize) -> usize {
        let Some(eid) = entity_id else {
            return requested;
        };
        let entity_count = self.entity_counts.read().get(eid).copied().unwrap_or(0);

        if entity_count == 0 {
            return requested;
        }

        let expansion = if entity_count < Self::ENTITY_COUNT_SMALL_THRESHOLD {
            requested
                .max(Self::EXPANSION_MIN_LIMIT)
                .saturating_mul(Self::EXPANSION_MULTIPLIER_SMALL)
        } else if entity_count < Self::ENTITY_COUNT_MEDIUM_THRESHOLD {
            requested
                .max(Self::EXPANSION_MIN_LIMIT)
                .saturating_mul(Self::EXPANSION_MULTIPLIER_MEDIUM)
        } else {
            requested
                .max(Self::EXPANSION_MIN_LIMIT)
                .saturating_mul(Self::EXPANSION_MULTIPLIER_LARGE)
        };
        expansion.min(self.max_elements)
    }

    pub fn len(&self, entity_id: Option<&str>) -> usize {
        if let Some(eid) = entity_id {
            self.entity_counts.read().get(eid).copied().unwrap_or(0)
        } else {
            let idx = self.index.read();
            idx.size()
        }
    }

    pub fn insert(&self, entity_id: &str, id: u64, vector: &[f32]) -> Result<()> {
        self.insert_batch(entity_id, &[(id, vector.to_vec())])
    }

    pub fn insert_batch(&self, entity_id: &str, items: &[(u64, Vec<f32>)]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        debug_assert!(
            items.iter().all(|(_, v)| v.len() == self.dimensions),
            "vector dimension mismatch in insert_batch"
        );

        let index = self.index.write();
        let mut mapping = self.vector_to_entity.write();
        let mut removed_entities = Vec::with_capacity(items.len());
        for (id, vector) in items {
            if index.contains(*id) {
                let _ = index.remove(*id);
                if let Some(old_eid) = mapping.remove(id) {
                    removed_entities.push(old_eid);
                }
            }
            index.add(*id, vector).context("failed to add vector to index")?;
            mapping.insert(*id, entity_id.to_string());
        }
        drop(mapping);
        drop(index);

        let mut counts = self.entity_counts.write();
        *counts.entry(entity_id.to_string()).or_insert(0) += items.len();
        for old_eid in removed_entities {
            if let Some(count) = counts.get_mut(&old_eid) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    counts.remove(&old_eid);
                }
            }
        }
        drop(counts);

        self.dirty.store(true, Ordering::Release);
        Ok(())
    }

    pub fn search(
        &self,
        entity_id: Option<&str>,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<(u64, f32)>> {
        debug_assert!(
            query.len() == self.dimensions,
            "query vector dimension mismatch: expected {}, got {}",
            self.dimensions,
            query.len()
        );

        let search_limit = self.expanded_limit(entity_id, limit);
        let index = self.index.read();
        if index.size() == 0 {
            return Ok(Vec::new());
        }
        let matches = index.search(query, search_limit).context("failed to search index")?;
        let mut results = Vec::with_capacity(matches.keys.len());
        drop(index);

        if let Some(eid) = entity_id {
            let mapping = self.vector_to_entity.read();
            for (&key, &dist) in matches.keys.iter().zip(matches.distances.iter()) {
                if mapping.get(&key).map(|s| s.as_str()) == Some(eid) {
                    results.push((key, dist));
                    if results.len() >= limit {
                        break;
                    }
                }
            }
        } else {
            for (&key, &dist) in matches.keys.iter().zip(matches.distances.iter()) {
                results.push((key, dist));
            }
            results.truncate(limit);
        }
        Ok(results)
    }

    pub fn remove(&self, _entity_id: &str, id: u64) -> Result<usize> {
        let index = self.index.write();
        let removed = index.remove(id).context("failed to remove vector from index")?;
        drop(index);
        if removed > 0 {
            let mut mapping = self.vector_to_entity.write();
            if let Some(eid) = mapping.remove(&id) {
                drop(mapping);
                let mut counts = self.entity_counts.write();
                if let Some(count) = counts.get_mut(&eid) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        counts.remove(&eid);
                    }
                }
            }
            self.dirty.store(true, Ordering::Release);
        }
        Ok(removed)
    }

    pub fn rebuild(&self, entity_id: &str, items: &[(u64, Vec<f32>)]) -> Result<()> {
        debug_assert!(
            items.iter().all(|(_, v)| v.len() == self.dimensions),
            "vector dimension mismatch in rebuild"
        );

        let ids_to_remove: Vec<u64> = {
            let mapping = self.vector_to_entity.read();
            mapping.iter().filter(|(_, eid)| eid.as_str() == entity_id).map(|(id, _)| *id).collect()
        };

        let index = self.index.write();
        for id in &ids_to_remove {
            let _ = index.remove(*id);
        }
        drop(index);

        let mut mapping = self.vector_to_entity.write();
        for id in &ids_to_remove {
            mapping.remove(id);
        }
        for (id, _vector) in items {
            mapping.insert(*id, entity_id.to_string());
        }
        drop(mapping);
        // Update counts after rebuild
        let mut counts = self.entity_counts.write();
        counts.insert(entity_id.to_string(), items.len());

        let index = self.index.write();
        for (id, vector) in items {
            if !index.contains(*id) {
                index.add(*id, vector).context("failed to add vector during rebuild")?;
            }
        }
        drop(index);

        self.dirty.store(true, Ordering::Release);
        Ok(())
    }

    pub fn clear(&self, entity_id: Option<&str>) -> Result<usize> {
        let total_removed;
        if let Some(eid) = entity_id {
            let ids_to_remove: Vec<u64> = {
                let mapping = self.vector_to_entity.read();
                mapping
                    .iter()
                    .filter(|(_, entity)| entity.as_str() == eid)
                    .map(|(id, _)| *id)
                    .collect()
            };
            total_removed = ids_to_remove.len();

            let index = self.index.write();
            for id in &ids_to_remove {
                let _ = index.remove(*id);
            }
            drop(index);

            let mut mapping = self.vector_to_entity.write();
            for id in &ids_to_remove {
                mapping.remove(id);
            }
            drop(mapping);
            let mut counts = self.entity_counts.write();
            counts.remove(eid);
        } else {
            let index = self.index.write();
            total_removed = index.size();
            index.reset().context("failed to reset index")?;
            index.reserve(self.max_elements).context("failed to reserve capacity after reset")?;
            drop(index);

            let mut mapping = self.vector_to_entity.write();
            mapping.clear();
            drop(mapping);
            let mut counts = self.entity_counts.write();
            counts.clear();
        }
        if total_removed > 0 {
            self.dirty.store(true, Ordering::Release);
        }
        Ok(total_removed)
    }

    pub fn checkpoint_if_dirty(&self) -> Result<bool> {
        if !self.dirty.load(Ordering::Acquire) {
            return Ok(false);
        }

        let persist_result = if let Some(dir) = &self.persist_dir {
            std::fs::create_dir_all(&**dir)
                .context("failed to create persist directory for checkpoint")?;
            let unified_path = dir.join("unified.hnsw");
            let mapping_path = dir.join("unified.mapping");

            let index_read = self.index.read();
            let snapshot_size = index_read.size();
            if snapshot_size > 0 {
                let save_result = index_read.save(unified_path.to_string_lossy().as_ref());
                drop(index_read);
                save_result.context("failed to save index during checkpoint")?;
            } else {
                drop(index_read);
            }

            let mapping_read = self.vector_to_entity.read();
            let snapshot: Vec<(u64, String)> =
                mapping_read.iter().map(|(k, v)| (*k, v.clone())).collect();
            drop(mapping_read);

            let mut mapping_lines = Vec::with_capacity(snapshot.len());
            for (id, entity_id) in &snapshot {
                mapping_lines.push(format!("{},{}", id, entity_id));
            }
            std::fs::write(&mapping_path, mapping_lines.join("\n"))
                .context("failed to write mapping during checkpoint")?;

            self.dirty.store(false, Ordering::Release);
            Ok(())
        } else {
            self.dirty.store(false, Ordering::Release);
            Ok(())
        };

        if let Err(ref e) = persist_result {
            error!(error = ?e, "Checkpoint persist failed");
        }
        persist_result.map(|_| true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_vector_index() {
        let temp = tempdir().unwrap();
        let persist_dir = temp.path().to_str().unwrap();
        let index = VectorIndex::new(3, 100, 16, 128, 64, Some(persist_dir)).unwrap();

        index.insert("test_entity", 1, &[0.8, 0.2, 0.1]).unwrap();
        index.insert("test_entity", 2, &[0.1, 0.9, 0.2]).unwrap();

        let results = index.search(Some("test_entity"), &[0.8, 0.2, 0.1], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);

        let results_other = index.search(Some("other_entity"), &[0.8, 0.2, 0.1], 2).unwrap();
        assert_eq!(results_other.len(), 0);

        let results_global = index.search(None, &[0.8, 0.2, 0.1], 2).unwrap();
        assert_eq!(results_global.len(), 2);
    }

    #[test]
    fn test_vector_index_edge_cases() {
        let temp = tempdir().unwrap();
        let persist_dir = temp.path().to_str().unwrap();
        let index = VectorIndex::new(3, 100, 16, 128, 64, Some(persist_dir)).unwrap();

        let res = index.search(Some("test_entity"), &[0.1, 0.2, 0.3], 5).unwrap();
        assert_eq!(res.len(), 0);

        index.insert("test_entity", 10, &[1.0, 0.0, 0.0]).unwrap();
        index.insert("test_entity", 20, &[1.0, 0.0, 0.0]).unwrap();

        let res2 = index.search(Some("test_entity"), &[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(res2.len(), 2);
    }

    #[test]
    fn test_vector_index_clear() {
        let temp = tempdir().unwrap();
        let persist_dir = temp.path().to_str().unwrap();
        let index = VectorIndex::new(3, 100, 16, 128, 64, Some(persist_dir)).unwrap();

        index.insert("test_entity", 1, &[0.8, 0.2, 0.1]).unwrap();
        index.insert("test_entity", 2, &[0.1, 0.9, 0.2]).unwrap();
        assert_eq!(index.clear(Some("test_entity")).unwrap(), 2);
        assert_eq!(index.len(Some("test_entity")), 0);
        assert!(index.search(Some("test_entity"), &[0.8, 0.2, 0.1], 5).unwrap().is_empty());
    }

    #[test]
    fn test_vector_index_persistence() {
        let temp = tempdir().unwrap();
        let persist_dir = temp.path().to_str().unwrap();

        {
            let index = VectorIndex::new(3, 100, 16, 128, 64, Some(persist_dir)).unwrap();
            index.insert("entity_a", 1, &[1.0, 0.0, 0.0]).unwrap();
            index.insert("entity_b", 2, &[0.0, 1.0, 0.0]).unwrap();
            index.checkpoint_if_dirty().unwrap();
        }

        {
            let index = VectorIndex::new(3, 100, 16, 128, 64, Some(persist_dir)).unwrap();
            let a_results = index.search(Some("entity_a"), &[1.0, 0.0, 0.0], 1).unwrap();
            assert_eq!(a_results.len(), 1);
            assert_eq!(a_results[0].0, 1);

            let b_results = index.search(Some("entity_b"), &[0.0, 1.0, 0.0], 1).unwrap();
            assert_eq!(b_results.len(), 1);
            assert_eq!(b_results[0].0, 2);
        }
    }

    #[test]
    fn test_vector_index_rebuild() {
        let temp = tempdir().unwrap();
        let persist_dir = temp.path().to_str().unwrap();
        let index = VectorIndex::new(3, 100, 16, 128, 64, Some(persist_dir)).unwrap();

        index.insert("entity_a", 1, &[1.0, 0.0, 0.0]).unwrap();
        index.insert("entity_a", 2, &[0.9, 0.1, 0.0]).unwrap();
        index.insert("entity_b", 3, &[0.0, 1.0, 0.0]).unwrap();

        index.rebuild("entity_a", &[(1, vec![1.0, 0.0, 0.0])]).unwrap();

        let a_results = index.search(Some("entity_a"), &[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(a_results.len(), 1);
        assert_eq!(a_results[0].0, 1);

        let b_results = index.search(Some("entity_b"), &[0.0, 1.0, 0.0], 10).unwrap();
        assert_eq!(b_results.len(), 1);
        assert_eq!(b_results[0].0, 3);
    }

    #[test]
    fn test_vector_index_remove() {
        let temp = tempdir().unwrap();
        let persist_dir = temp.path().to_str().unwrap();
        let index = VectorIndex::new(3, 100, 16, 128, 64, Some(persist_dir)).unwrap();

        index.insert("entity_a", 1, &[1.0, 0.0, 0.0]).unwrap();
        index.insert("entity_a", 2, &[0.9, 0.1, 0.0]).unwrap();

        assert_eq!(index.remove("entity_a", 1).unwrap(), 1);

        let results = index.search(Some("entity_a"), &[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_vector_index_update() {
        let temp = tempdir().unwrap();
        let persist_dir = temp.path().to_str().unwrap();
        let index = VectorIndex::new(3, 100, 16, 128, 64, Some(persist_dir)).unwrap();

        index.insert("entity_a", 1, &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.len(Some("entity_a")), 1);

        index.insert("entity_a", 1, &[0.0, 1.0, 0.0]).unwrap();
        assert_eq!(index.len(Some("entity_a")), 1);

        let results = index.search(Some("entity_a"), &[0.0, 1.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }
}
