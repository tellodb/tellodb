use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const TEMPORAL_DB_FILE: &str = "temporal.redb";
const GRAPH_DB_FILE: &str = "graph.redb";
const FTS_DIR: &str = "fts_tantivy";
const PLATFORM_DB_FILE: &str = "platform.db";
const ANALYTICS_DB_FILE: &str = "analytics.db";
const VECTOR_INDEX_FILE: &str = "vector.hnsw";
const EMBEDDING_CACHE_FILE: &str = "embedding_cache.redb";

#[derive(Debug, Clone)]
pub struct RuntimePaths {
    root: PathBuf,
    explicit_root: bool,
    temporal_db: PathBuf,
    graph_db: PathBuf,
    fts_dir: PathBuf,
    platform_db: PathBuf,
    analytics_db: PathBuf,
    vector_index: PathBuf,
    embedding_cache: PathBuf,
    hf_home: PathBuf,
    hf_hub_cache: PathBuf,
    xdg_cache_home: PathBuf,
}

impl RuntimePaths {
    pub fn from_env() -> Result<Self> {
        let configured_root = env::var("TEMPORAL_MEMORY_DATA_DIR")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                env::var("TELLODB_DATA_DIR").ok().filter(|value| !value.trim().is_empty())
            });

        let (root, explicit_root) = match configured_root {
            Some(root) => (PathBuf::from(root), true),
            None => (PathBuf::from("."), false),
        };

        let root = if root.is_absolute() {
            root
        } else {
            env::current_dir().context("failed to read current working directory")?.join(root)
        };

        let hf_home = root.join("hf-home");
        let hf_hub_cache = hf_home.join("hub");
        let xdg_cache_home = root.join("cache");

        Ok(Self {
            temporal_db: root.join(TEMPORAL_DB_FILE),
            graph_db: root.join(GRAPH_DB_FILE),
            fts_dir: root.join(FTS_DIR),
            platform_db: root.join(PLATFORM_DB_FILE),
            analytics_db: root.join(ANALYTICS_DB_FILE),
            vector_index: root.join(VECTOR_INDEX_FILE),
            embedding_cache: root.join(EMBEDDING_CACHE_FILE),
            root,
            explicit_root,
            hf_home,
            hf_hub_cache,
            xdg_cache_home,
        })
    }

    pub fn ensure_dirs(&self) -> Result<()> {
        fs::create_dir_all(&self.root).with_context(|| {
            format!("failed to create runtime data directory {}", self.root.display())
        })?;

        if self.explicit_root {
            for dir in [&self.hf_home, &self.hf_hub_cache, &self.xdg_cache_home] {
                fs::create_dir_all(dir).with_context(|| {
                    format!("failed to create runtime cache directory {}", dir.display())
                })?;
            }
        }

        Ok(())
    }

    pub fn apply_process_env_defaults(&self) {
        if !self.explicit_root {
            return;
        }

        // SAFETY: `set_var` is safe in Rust 1.62+; we limit env var writes to process
        // bootstrap before any worker tasks are spawned, avoiding race conditions.
        if env::var_os("HF_HOME").is_none() {
            env::set_var("HF_HOME", &self.hf_home);
        }
        if env::var_os("HUGGINGFACE_HUB_CACHE").is_none() {
            env::set_var("HUGGINGFACE_HUB_CACHE", &self.hf_hub_cache);
        }
        if env::var_os("XDG_CACHE_HOME").is_none() {
            env::set_var("XDG_CACHE_HOME", &self.xdg_cache_home);
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

        pub fn temporal_db(&self) -> &Path {
        &self.temporal_db
    }

        pub fn graph_db(&self) -> &Path {
        &self.graph_db
    }

        pub fn fts_dir(&self) -> &Path {
        &self.fts_dir
    }

    pub fn platform_db(&self) -> &Path {
        &self.platform_db
    }

        pub fn analytics_db(&self) -> &Path {
        &self.analytics_db
    }

    pub fn vector_index(&self) -> &Path {
        &self.vector_index
    }

        pub fn session_vector_index(&self) -> PathBuf {
        let mut path = self.vector_index.clone();
        path.set_extension("session_hnsw");
        path
    }

        pub fn event_vector_index(&self) -> PathBuf {
        let mut path = self.vector_index.clone();
        path.set_extension("event_hnsw");
        path
    }

        pub fn shadow_vector_index(&self) -> PathBuf {
        let mut path = self.vector_index.clone();
        path.set_extension("shadow_hnsw");
        path
    }

        pub fn embedding_cache(&self) -> &Path {
        &self.embedding_cache
    }

    pub fn tenant_dir(&self, tenant_id: &str) -> PathBuf {
        self.root.join("tenants").join(tenant_id)
    }

    pub fn tenant_db(&self, tenant_id: &str) -> PathBuf {
        self.tenant_dir(tenant_id).join("tellodb.db")
    }

        pub fn tenant_vector_index(&self, tenant_id: &str) -> PathBuf {
        self.tenant_dir(tenant_id).join("vectors.hnsw")
    }

    pub fn ensure_tenant_dir(&self, tenant_id: &str) -> Result<PathBuf> {
        let dir = self.tenant_dir(tenant_id);
        fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create tenant directory {}", dir.display()))?;
        Ok(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use tempfile::tempdir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn clear_all_env_vars() {
        env::remove_var("TEMPORAL_MEMORY_DATA_DIR");
        env::remove_var("TELLODB_DATA_DIR");
        env::remove_var("TEMPORAL_EMBED_CACHE_PATH");
        env::remove_var("HF_HOME");
        env::remove_var("HUGGINGFACE_HUB_CACHE");
        env::remove_var("XDG_CACHE_HOME");
    }

    fn lock_env() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn set_tellodb_dir(path: &str) -> RuntimePaths {
        clear_all_env_vars();
        env::set_var("TELLODB_DATA_DIR", path);
        RuntimePaths::from_env().unwrap()
    }

    #[test]
    fn default_paths_relative_to_cwd() {
        let _lock = lock_env();
        clear_all_env_vars();
        let paths = RuntimePaths::from_env().unwrap();
        let cwd = env::current_dir().unwrap();
        assert_eq!(paths.root(), cwd.join(".").as_path());
        assert!(!paths.explicit_root);
    }

    #[test]
    fn custom_root_via_temporal_memory_data_dir() {
        let _lock = lock_env();
        clear_all_env_vars();
        env::set_var("TEMPORAL_MEMORY_DATA_DIR", "/tmp/test_temporal");
        let paths = RuntimePaths::from_env().unwrap();
        assert_eq!(paths.root(), Path::new("/tmp/test_temporal"));
        assert!(paths.explicit_root);
    }

    #[test]
    fn custom_root_via_tellodb_data_dir() {
        let _lock = lock_env();
        clear_all_env_vars();
        env::set_var("TELLODB_DATA_DIR", "/tmp/test_tellodb");
        let paths = RuntimePaths::from_env().unwrap();
        assert_eq!(paths.root(), Path::new("/tmp/test_tellodb"));
        assert!(paths.explicit_root);
    }

    #[test]
    fn temporal_memory_data_dir_takes_precedence_over_tellodb_data_dir() {
        let _lock = lock_env();
        clear_all_env_vars();
        env::set_var("TEMPORAL_MEMORY_DATA_DIR", "/tmp/precedence_temporal");
        env::set_var("TELLODB_DATA_DIR", "/tmp/precedence_tellodb");
        let paths = RuntimePaths::from_env().unwrap();
        assert_eq!(paths.root(), Path::new("/tmp/precedence_temporal"));
    }

    #[test]
    fn root_getter_returns_correct_path() {
        let paths = {
            let _lock = lock_env();
            set_tellodb_dir("/tmp/root_test")
        };
        assert_eq!(paths.root(), Path::new("/tmp/root_test"));
    }

    #[test]
    fn temporal_db_getter_returns_correct_filename() {
        let paths = {
            let _lock = lock_env();
            set_tellodb_dir("/tmp/getter_test")
        };
        assert_eq!(paths.temporal_db(), Path::new("/tmp/getter_test/temporal.redb"));
    }

    #[test]
    fn graph_db_getter_returns_correct_filename() {
        let paths = {
            let _lock = lock_env();
            set_tellodb_dir("/tmp/graph_test")
        };
        assert_eq!(paths.graph_db(), Path::new("/tmp/graph_test/graph.redb"));
    }

    #[test]
    fn fts_dir_getter_returns_correct_filename() {
        let paths = {
            let _lock = lock_env();
            set_tellodb_dir("/tmp/fts_test")
        };
        assert_eq!(paths.fts_dir(), Path::new("/tmp/fts_test/fts_tantivy"));
    }

    #[test]
    fn platform_db_getter_returns_correct_filename() {
        let paths = {
            let _lock = lock_env();
            set_tellodb_dir("/tmp/platform_test")
        };
        assert_eq!(paths.platform_db(), Path::new("/tmp/platform_test/platform.db"));
    }

    #[test]
    fn analytics_db_getter_returns_correct_filename() {
        let paths = {
            let _lock = lock_env();
            set_tellodb_dir("/tmp/analytics_test")
        };
        assert_eq!(paths.analytics_db(), Path::new("/tmp/analytics_test/analytics.db"));
    }

    #[test]
    fn vector_index_getter_returns_correct_filename() {
        let paths = {
            let _lock = lock_env();
            set_tellodb_dir("/tmp/vector_test")
        };
        assert_eq!(paths.vector_index(), Path::new("/tmp/vector_test/vector.hnsw"));
    }

    #[test]
    fn embedding_cache_getter_returns_correct_filename() {
        let paths = {
            let _lock = lock_env();
            set_tellodb_dir("/tmp/cache_test")
        };
        assert_eq!(paths.embedding_cache(), Path::new("/tmp/cache_test/embedding_cache.redb"));
    }

    #[test]
    fn session_vector_index_has_session_hnsw_extension() {
        let paths = {
            let _lock = lock_env();
            set_tellodb_dir("/tmp/vidx_test")
        };
        assert_eq!(
            paths.session_vector_index(),
            PathBuf::from("/tmp/vidx_test/vector.session_hnsw")
        );
    }

    #[test]
    fn event_vector_index_has_event_hnsw_extension() {
        let paths = {
            let _lock = lock_env();
            set_tellodb_dir("/tmp/vidx_test")
        };
        assert_eq!(paths.event_vector_index(), PathBuf::from("/tmp/vidx_test/vector.event_hnsw"));
    }

    #[test]
    fn shadow_vector_index_has_shadow_hnsw_extension() {
        let paths = {
            let _lock = lock_env();
            set_tellodb_dir("/tmp/vidx_test")
        };
        assert_eq!(paths.shadow_vector_index(), PathBuf::from("/tmp/vidx_test/vector.shadow_hnsw"));
    }

    #[test]
    fn ensure_dirs_creates_root_directory() {
        let dir;
        let paths;
        {
            let _lock = lock_env();
            dir = tempdir().unwrap();
            let root = dir.path().join("_data");
            paths = set_tellodb_dir(root.to_str().unwrap());
        }
        assert!(!dir.path().join("_data").exists());
        paths.ensure_dirs().unwrap();
        assert!(dir.path().join("_data").exists());
    }

    #[test]
    fn ensure_dirs_creates_subdirectories_when_explicit_root() {
        let dir;
        let root;
        let paths;
        {
            let _lock = lock_env();
            dir = tempdir().unwrap();
            root = dir.path().join("explicit_test");
            paths = set_tellodb_dir(root.to_str().unwrap());
        }
        assert!(paths.explicit_root);
        paths.ensure_dirs().unwrap();
        assert!(root.join("hf-home").exists());
        assert!(root.join("hf-home/hub").exists());
        assert!(root.join("cache").exists());
    }

    #[test]
    fn ensure_dirs_does_not_create_subdirs_when_default_root() {
        let _lock = lock_env();
        let original_cwd = env::current_dir().unwrap();
        let dir = tempdir().unwrap();
        clear_all_env_vars();
        env::set_current_dir(dir.path()).unwrap();
        let paths = RuntimePaths::from_env().unwrap();
        assert!(!paths.explicit_root);
        paths.ensure_dirs().unwrap();
        assert!(!dir.path().join("hf-home").exists());
        assert!(!dir.path().join("cache").exists());
        env::set_current_dir(&original_cwd).unwrap();
    }

    #[test]
    fn apply_process_env_defaults_sets_hf_home() {
        let _lock = lock_env();
        clear_all_env_vars();
        env::set_var("TELLODB_DATA_DIR", "/tmp/env_defaults_test");
        let paths = RuntimePaths::from_env().unwrap();
        paths.apply_process_env_defaults();
        assert_eq!(env::var("HF_HOME").unwrap(), "/tmp/env_defaults_test/hf-home");
        assert_eq!(
            env::var("HUGGINGFACE_HUB_CACHE").unwrap(),
            "/tmp/env_defaults_test/hf-home/hub"
        );
        assert_eq!(env::var("XDG_CACHE_HOME").unwrap(), "/tmp/env_defaults_test/cache");
    }

    #[test]
    fn apply_process_env_defaults_does_not_overwrite_existing_env_vars() {
        let _lock = lock_env();
        clear_all_env_vars();
        env::set_var("TELLODB_DATA_DIR", "/tmp/no_overwrite");
        env::set_var("HF_HOME", "/custom/hf");
        let paths = RuntimePaths::from_env().unwrap();
        paths.apply_process_env_defaults();
        assert_eq!(env::var("HF_HOME").unwrap(), "/custom/hf");
    }

    #[test]
    fn apply_process_env_defaults_skipped_when_not_explicit_root() {
        let _lock = lock_env();
        clear_all_env_vars();
        let paths = RuntimePaths::from_env().unwrap();
        assert!(!paths.explicit_root);
        paths.apply_process_env_defaults();
        assert!(env::var_os("HF_HOME").is_none());
    }

    #[test]
    fn debug_output_contains_all_paths() {
        let paths = {
            let _lock = lock_env();
            set_tellodb_dir("/tmp/debug_test")
        };
        let debug = format!("{:?}", paths);
        assert!(debug.contains("/tmp/debug_test"));
        assert!(debug.contains("temporal.redb"));
        assert!(debug.contains("graph.redb"));
        assert!(debug.contains("fts_tantivy"));
        assert!(debug.contains("platform.db"));
        assert!(debug.contains("analytics.db"));
        assert!(debug.contains("vector.hnsw"));
        assert!(debug.contains("embedding_cache.redb"));
        assert!(debug.contains("hf-home"));
        assert!(debug.contains("cache"));
    }

    #[test]
    fn relative_root_is_resolved_against_cwd() {
        let _lock = lock_env();
        clear_all_env_vars();
        let dir = tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();
        let original_cwd = env::current_dir().unwrap();
        env::set_current_dir(&dir_path).unwrap();
        let paths = RuntimePaths::from_env().unwrap();
        let expected = env::current_dir().unwrap().join(".");
        assert_eq!(paths.root(), expected.as_path());
        env::set_current_dir(&original_cwd).unwrap();
    }
}
