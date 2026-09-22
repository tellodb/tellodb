use super::tenant::TenantStore;
use crate::runtime_paths::RuntimePaths;
use crate::vector_index::{VectorConfig, VectorIndex};
use anyhow::Result;
use lru::LruCache;
use parking_lot::RwLock;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tracing::warn;

pub struct TenantDatabaseManager {
    paths: RuntimePaths,
    vector_config: VectorConfig,
    tenants: RwLock<LruCache<String, Arc<TenantStore>>>,
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
        let capacity = std::env::var("TELLODB_MAX_OPEN_TENANTS")
            .ok()
            .and_then(|value| value.parse().ok())
            .and_then(NonZeroUsize::new)
            .unwrap_or(NonZeroUsize::new(32).expect("non-zero default"));
        Self::with_capacity(paths, vector_config, capacity)
    }

    fn with_capacity(
        paths: RuntimePaths,
        vector_config: VectorConfig,
        capacity: NonZeroUsize,
    ) -> Self {
        Self { paths, vector_config, tenants: RwLock::new(LruCache::new(capacity)) }
    }

    pub fn migrate_existing_tenants(&self) -> Result<usize> {
        let tenants_dir = self.paths.root().join("tenants");
        if !tenants_dir.exists() {
            return Ok(0);
        }
        let mut migrated = 0;
        for entry in std::fs::read_dir(tenants_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let Some(tenant_id) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            validate_tenant_id(&tenant_id)?;
            let database = self.paths.tenant_db(&tenant_id);
            if database.exists() {
                TenantStore::new(&database)?;
                migrated += 1;
            }
        }
        Ok(migrated)
    }

    pub fn get_tenant(&self, tenant_id: &str) -> Result<Arc<TenantStore>> {
        validate_tenant_id(tenant_id)?;
        let mut write = self.tenants.write();
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
        write.put(tenant_id.to_string(), store.clone());

        Ok(store)
    }

    pub fn all_tenants(&self) -> Vec<Arc<TenantStore>> {
        let read = self.tenants.read();
        read.iter().map(|(_, store)| store.clone()).collect()
    }

    pub fn tenant_ids(&self) -> Result<Vec<String>> {
        let tenants_dir = self.paths.root().join("tenants");
        if !tenants_dir.exists() {
            return Ok(Vec::new());
        }
        let mut tenant_ids = Vec::new();
        for entry in std::fs::read_dir(tenants_dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let Some(tenant_id) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            validate_tenant_id(&tenant_id)?;
            if self.paths.tenant_db(&tenant_id).exists() {
                tenant_ids.push(tenant_id);
            }
        }
        tenant_ids.sort();
        Ok(tenant_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager(root: &std::path::Path) -> TenantDatabaseManager {
        let paths = RuntimePaths::from_root(root.to_path_buf());
        TenantDatabaseManager::new(paths, VectorConfig::new(3))
    }

    /// "One tenant, one file" is a claim the project makes about its on-disk
    /// shape, and it is the sort of claim that quietly stops being true the
    /// first time something persists an index or a cache beside the database.
    /// SQLite's own -wal and -shm are part of the database, not sidecars, and
    /// they disappear on a clean checkpoint.
    #[test]
    fn a_tenant_directory_holds_nothing_but_its_database() {
        let temp = tempfile::tempdir().unwrap();
        let mgr = manager(temp.path());
        let store = mgr.get_tenant("alice").unwrap();

        // Exercise the vector index, which is the thing most likely to want a
        // file of its own, then force a checkpoint the way a clean shutdown
        // would.
        store
            .vectors()
            .unwrap()
            .insert_batch("alice", &[(1, vec![1.0, 0.0, 0.0]), (2, vec![0.0, 1.0, 0.0])])
            .unwrap();
        store.set_memory_links_batch(&[("a".into(), "b".into(), "derived_from".into())]).unwrap();
        store.checkpoint().unwrap();

        let dir = temp.path().join("tenants").join("alice");
        let mut found: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        found.sort();

        let unexpected: Vec<&String> = found
            .iter()
            .filter(|name| {
                !matches!(name.as_str(), "tellodb.db" | "tellodb.db-wal" | "tellodb.db-shm")
            })
            .collect();
        assert!(
            unexpected.is_empty(),
            "tenant directory gained a sidecar: {unexpected:?} (all entries: {found:?})"
        );
        assert!(found.iter().any(|n| n == "tellodb.db"), "no database at all: {found:?}");
    }

    /// E7 -- what one tenant's write storm does to another tenant's reads.
    ///
    /// Isolation here is structural: one SQLite file per tenant, so tenant B's
    /// writers hold a lock tenant A's readers never ask for. That is an
    /// argument, not a measurement, and it is worth measuring because writers
    /// inside a single tenant demonstrably do interfere -- two concurrent
    /// ingesters on one file drove the storage stage from 400ms to 2.2s and
    /// eventually returned "database is locked".
    ///
    /// Run it on demand; it is timing-sensitive and does not belong in CI:
    ///   cargo test --lib tenant_tail_latency -- --ignored --nocapture
    #[test]
    #[ignore = "timing measurement, run on demand"]
    fn tenant_tail_latency_under_a_neighbours_write_storm() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::time::Instant;

        fn percentile(sorted: &[u128], p: f64) -> u128 {
            if sorted.is_empty() {
                return 0;
            }
            let index = ((sorted.len() as f64 - 1.0) * p).round() as usize;
            sorted[index]
        }

        println!(
            "{:>8} {:>10} {:>10} {:>10} {:>12}",
            "tenants", "p50 us", "p95 us", "p99 us", "reads"
        );
        for neighbours in [0usize, 1, 4, 16] {
            let temp = tempfile::tempdir().unwrap();
            let mgr = Arc::new(manager(temp.path()));
            let reader = mgr.get_tenant("reader").unwrap();
            reader
                .set_memory_links_batch(&[("seed".into(), "target".into(), "derived_from".into())])
                .unwrap();

            let stop = Arc::new(AtomicBool::new(false));
            let mut writers = Vec::new();
            for n in 0..neighbours {
                let mgr = Arc::clone(&mgr);
                let stop = Arc::clone(&stop);
                writers.push(std::thread::spawn(move || {
                    let store = mgr.get_tenant(&format!("writer{n}")).unwrap();
                    let mut i = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        let links: Vec<(String, String, String)> = (0..64)
                            .map(|k| {
                                (format!("a{i}_{k}"), format!("b{i}_{k}"), "derived_from".into())
                            })
                            .collect();
                        let _ = store.set_memory_links_batch(&links);
                        i += 1;
                    }
                }));
            }

            let mut samples = Vec::with_capacity(400);
            for _ in 0..400 {
                let started = Instant::now();
                reader.get_link_cluster_scores("seed", 1).unwrap();
                samples.push(started.elapsed().as_micros());
            }
            stop.store(true, Ordering::Relaxed);
            for writer in writers {
                let _ = writer.join();
            }

            samples.sort_unstable();
            println!(
                "{:>8} {:>10} {:>10} {:>10} {:>12}",
                neighbours + 1,
                percentile(&samples, 0.50),
                percentile(&samples, 0.95),
                percentile(&samples, 0.99),
                samples.len()
            );
        }
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

    #[test]
    fn evicts_the_least_recently_used_tenant() {
        let temp = tempfile::tempdir().unwrap();
        let mgr = TenantDatabaseManager::with_capacity(
            RuntimePaths::from_root(temp.path().to_path_buf()),
            VectorConfig::new(3),
            NonZeroUsize::new(2).unwrap(),
        );
        let first = mgr.get_tenant("first").unwrap();
        mgr.get_tenant("second").unwrap();
        mgr.get_tenant("third").unwrap();

        let reopened = mgr.get_tenant("first").unwrap();
        assert!(!Arc::ptr_eq(&first, &reopened));
        assert_eq!(mgr.all_tenants().len(), 2);
    }

    #[test]
    fn migrates_existing_tenants_before_they_are_requested() {
        let temp = tempfile::tempdir().unwrap();
        let paths = RuntimePaths::from_root(temp.path().to_path_buf());
        paths.ensure_tenant_dir("existing").unwrap();
        let database = paths.tenant_db("existing");
        drop(TenantStore::new(&database).unwrap());
        let connection = rusqlite::Connection::open(&database).unwrap();
        connection.execute("DROP TABLE consolidation_queue", []).unwrap();
        connection.execute("PRAGMA user_version = 6", []).unwrap();
        drop(connection);

        let mgr = TenantDatabaseManager::new(paths, VectorConfig::new(3));
        assert_eq!(mgr.migrate_existing_tenants().unwrap(), 1);
        let connection = rusqlite::Connection::open(database).unwrap();
        let version: i64 =
            connection.query_row("PRAGMA user_version", [], |row| row.get(0)).unwrap();
        assert_eq!(version, crate::storage::tenant::SCHEMA_VERSION);
        let queue_exists: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'consolidation_queue'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(queue_exists, 1);
    }
}
