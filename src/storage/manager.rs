use super::tenant::TenantStore;
use crate::runtime_paths::RuntimePaths;
use anyhow::Result;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub struct TenantDatabaseManager {
    paths: RuntimePaths,
    tenants: RwLock<HashMap<String, Arc<TenantStore>>>,
}

impl TenantDatabaseManager {
    pub fn new(paths: RuntimePaths) -> Self {
        Self { paths, tenants: RwLock::new(HashMap::new()) }
    }

    pub fn get_tenant(&self, tenant_id: &str) -> Result<Arc<TenantStore>> {
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
        let db_path = self.paths.tenant_db(tenant_id);
        let store = Arc::new(TenantStore::new(&db_path)?);
        write.insert(tenant_id.to_string(), store.clone());

        Ok(store)
    }

    pub fn all_tenants(&self) -> Vec<Arc<TenantStore>> {
        let read = self.tenants.read();
        read.values().cloned().collect()
    }
}
