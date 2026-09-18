use super::tenant::TenantStore;
use crate::runtime_paths::RuntimePaths;
use crate::vector_index::{VectorConfig, VectorIndex};
use anyhow::Result;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::warn;

pub struct TenantDatabaseManager {
    paths: RuntimePaths,
    vector_config: VectorConfig,
    tenants: RwLock<HashMap<String, Arc<TenantStore>>>,
}

/// Tenant ids become directory names, so only a safe character set is
/// accepted (no path separators, `..`, or empty ids).
pub fn validate_tenant_id(tenant_id: &str) -> Result<()> {
    let valid = !tenant_id.is_empty()
        && tenant_id.len() <= 128
        && tenant_id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if !valid {
        anyhow::bail!("invalid tenant id {tenant_id:?}");
    }
    Ok(())
}

impl TenantDatabaseManager {
    pub fn new(paths: RuntimePaths, vector_config: VectorConfig) -> Self {
        let legacy_index = paths.vector_index().join("unified.hnsw");
        if legacy_index.exists() {
            warn!(
                path = %legacy_index.display(),
                "ignoring legacy vector index files; vectors are loaded from tenant databases"
            );
        }
        Self { paths, vector_config, tenants: RwLock::new(HashMap::new()) }
    }

    pub fn get_tenant(&self, tenant_id: &str) -> Result<Arc<TenantStore>> {
        validate_tenant_id(tenant_id)?;
        {
            let read = self.tenants.read();
            if let Some(store) = read.get(tenant_id) {
                return Ok(store.clone());
            }
        }

        let mut write = self.tenants.write();
        // Double check
        if let Some(store) = write.get(tenant_id) {
            return Ok(store.clone());
        }

        self.paths.ensure_tenant_dir(tenant_id)?;
        let store = Arc::new(TenantStore::new(&self.paths.tenant_db(tenant_id))?);
        let (_, without_embedding) = store.stored_vector_counts()?;
        if without_embedding > 0 {
            warn!(
                tenant_id,
                count = without_embedding,
                "vectors from an older build have no stored embedding; re-ingest to restore them"
            );
        }
        store.attach_vectors(VectorIndex::new(self.vector_config, store.vector_source()))?;
        write.insert(tenant_id.to_string(), store.clone());

        Ok(store)
    }

    pub fn all_tenants(&self) -> Vec<Arc<TenantStore>> {
        let read = self.tenants.read();
        read.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager(root: &std::path::Path) -> TenantDatabaseManager {
        let paths = RuntimePaths::from_root(root.to_path_buf());
        TenantDatabaseManager::new(paths, VectorConfig::new(3))
    }

    #[test]
    fn rejects_path_traversal_tenant_ids() {
        let temp = tempfile::tempdir().unwrap();
        let mgr = manager(temp.path());
        for bad in ["../x", "a/b", "", "..", "a b"] {
            assert!(mgr.get_tenant(bad).is_err(), "{bad:?} should be rejected");
        }
        assert!(!temp.path().join("x").exists());
        assert!(mgr.get_tenant("usr_abc-1").is_ok());
    }

    #[test]
    fn tenants_have_independent_vector_indexes() {
        let temp = tempfile::tempdir().unwrap();
        let mgr = manager(temp.path());
        let a = mgr.get_tenant("a").unwrap();
        let b = mgr.get_tenant("b").unwrap();
        let obs = |v: Vec<f32>| crate::storage::AgentObservation {
            entity_id: "e".into(),
            textual_content: "x".into(),
            embedding: v,
            ..Default::default()
        };
        let a_ids =
            a.insert_observations_batch(&[(1, "m".into(), obs(vec![1.0, 0.0, 0.0]))]).unwrap();
        let b_ids =
            b.insert_observations_batch(&[(1, "m".into(), obs(vec![0.0, 1.0, 0.0]))]).unwrap();
        assert_eq!(a_ids, b_ids, "rowids overlap across tenants");
        let a_hits = a.vectors().unwrap().search(Some("e"), &[1.0, 0.0, 0.0], 5).unwrap();
        let b_hits = b.vectors().unwrap().search(Some("e"), &[1.0, 0.0, 0.0], 5).unwrap();
        assert!(a_hits[0].1 < 1e-5, "tenant a finds its own vector");
        assert!((b_hits[0].1 - 1.0).abs() < 1e-5, "tenant b only sees its orthogonal vector");
    }
}
