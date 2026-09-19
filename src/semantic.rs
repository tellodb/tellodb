use anyhow::{Context, Result};
use fastembed::{
    EmbeddingModel, InitOptionsUserDefined, RerankInitOptions, RerankerModel, TextEmbedding,
    TextInitOptions, TextRerank, TokenizerFiles, UserDefinedEmbeddingModel,
};
use ort::ep::CUDA;

use crate::config::{Config, EmbeddingConfig, RerankConfig};
use lru::LruCache;
use parking_lot::{Condvar, Mutex};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// Device the models were initialised on; readable without an instance.
static DEVICE_LABEL: OnceLock<&'static str> = OnceLock::new();
static DEVICE_FLAGS: OnceLock<(bool, bool)> = OnceLock::new();

fn hash_text(text: &str) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    Sha256::digest(text.as_bytes()).into()
}

fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_f32s(bytes: &[u8]) -> Option<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Counting semaphore usable from blocking threads. The tokio semaphore this
/// replaces was acquired with `Handle::current().block_on`, which panics
/// outside a runtime and parks a blocking-pool thread inside one.
struct ComputePermits {
    available: Mutex<usize>,
    released: Condvar,
}

impl ComputePermits {
    fn new(permits: usize) -> Self {
        Self { available: Mutex::new(permits.max(1)), released: Condvar::new() }
    }

    fn acquire(&self) -> PermitGuard<'_> {
        let mut available = self.available.lock();
        while *available == 0 {
            self.released.wait(&mut available);
        }
        *available -= 1;
        PermitGuard { permits: self }
    }
}

struct PermitGuard<'a> {
    permits: &'a ComputePermits,
}

impl Drop for PermitGuard<'_> {
    fn drop(&mut self) {
        *self.permits.available.lock() += 1;
        self.permits.released.notify_one();
    }
}

/// Persistent text → embedding cache. Survives `/reset`, so re-running a
/// benchmark does not recompute embeddings. Cache failures are logged and
/// treated as misses; they never fail an ingest.
pub struct EmbeddingCache {
    conn: Option<Mutex<rusqlite::Connection>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl EmbeddingCache {
    pub fn new(path: Option<PathBuf>, enabled: bool) -> Self {
        let conn = if enabled { path.and_then(Self::open) } else { None };
        Self { conn: conn.map(Mutex::new), hits: AtomicU64::new(0), misses: AtomicU64::new(0) }
    }

    fn open(path: PathBuf) -> Option<rusqlite::Connection> {
        if let Some(parent) = path.parent() {
            if let Err(err) = std::fs::create_dir_all(parent) {
                tracing::warn!(path = %parent.display(), error = %err, "embedding cache dir");
                return None;
            }
        }
        let open = || -> rusqlite::Result<rusqlite::Connection> {
            let conn = rusqlite::Connection::open(&path)?;
            conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 CREATE TABLE IF NOT EXISTS embeddings (
                     model_key TEXT NOT NULL,
                     text_hash BLOB NOT NULL,
                     embedding BLOB NOT NULL,
                     PRIMARY KEY (model_key, text_hash)
                 ) WITHOUT ROWID;",
            )?;
            Ok(conn)
        };
        match open() {
            Ok(conn) => Some(conn),
            Err(err) => {
                tracing::warn!(path = %path.display(), error = %err, "embedding cache disabled");
                None
            }
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.conn.is_some()
    }

    /// Looks up every hash under one lock; `None` entries are misses.
    pub fn get_many(
        &self,
        model_key: &str,
        hashes: &[[u8; 32]],
        dim: usize,
    ) -> Vec<Option<Vec<f32>>> {
        let Some(conn) = self.conn.as_ref() else {
            return vec![None; hashes.len()];
        };
        let conn = conn.lock();
        let lookup = || -> rusqlite::Result<Vec<Option<Vec<f32>>>> {
            let mut stmt = conn.prepare_cached(
                "SELECT embedding FROM embeddings WHERE model_key = ?1 AND text_hash = ?2",
            )?;
            let mut out = Vec::with_capacity(hashes.len());
            for hash in hashes {
                let blob: Option<Vec<u8>> = match stmt
                    .query_row(rusqlite::params![model_key, &hash[..]], |row| row.get(0))
                {
                    Ok(blob) => Some(blob),
                    Err(rusqlite::Error::QueryReturnedNoRows) => None,
                    Err(err) => return Err(err),
                };
                out.push(blob.and_then(|b| bytes_to_f32s(&b)).filter(|v| v.len() == dim));
            }
            Ok(out)
        };
        let result = lookup().unwrap_or_else(|err| {
            tracing::warn!(error = %err, "embedding cache read failed");
            vec![None; hashes.len()]
        });
        let hits = result.iter().filter(|r| r.is_some()).count() as u64;
        self.hits.fetch_add(hits, Ordering::Relaxed);
        self.misses.fetch_add(result.len() as u64 - hits, Ordering::Relaxed);
        result
    }

    pub fn put_many(&self, model_key: &str, items: &[([u8; 32], &[f32])]) {
        let Some(conn) = self.conn.as_ref() else {
            return;
        };
        let mut conn = conn.lock();
        let mut write = || -> rusqlite::Result<()> {
            let tx = conn.transaction()?;
            {
                let mut stmt = tx.prepare_cached(
                    "INSERT OR REPLACE INTO embeddings (model_key, text_hash, embedding) VALUES (?1, ?2, ?3)",
                )?;
                for (hash, embedding) in items {
                    stmt.execute(rusqlite::params![
                        model_key,
                        &hash[..],
                        f32s_to_bytes(embedding)
                    ])?;
                }
            }
            tx.commit()
        };
        if let Err(err) = write() {
            tracing::warn!(error = %err, "embedding cache write failed");
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    pub fn clear(&self) {
        if let Some(conn) = self.conn.as_ref() {
            if let Err(err) = conn.lock().execute("DELETE FROM embeddings", []) {
                tracing::warn!(error = %err, "embedding cache clear failed");
            }
        }
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }
}

pub fn parse_embedding_model(id: &str) -> Result<EmbeddingModel> {
    if let Ok(model) = EmbeddingModel::try_from(id.to_string()) {
        return Ok(model);
    }

    let normalized = id.trim().to_lowercase();
    for m in TextEmbedding::list_supported_models() {
        if m.model_code.to_lowercase() == normalized {
            return Ok(m.model);
        }
        let code_suffix = m.model_code.rsplit('/').next().unwrap_or("").to_lowercase();
        if code_suffix == normalized.rsplit('/').next().unwrap_or("") {
            return Ok(m.model);
        }
    }

    match normalized.as_str() {
        s if s.contains("bge-small-en") || s == "bge-small" => Ok(EmbeddingModel::BGESmallENV15),
        s if s.contains("bge-base-en") || s == "bge-base" => Ok(EmbeddingModel::BGEBaseENV15),
        s if s.contains("bge-large-en") || s == "bge-large" => Ok(EmbeddingModel::BGELargeENV15),
        s if s.contains("minilm-l6") => Ok(EmbeddingModel::AllMiniLML6V2),
        s if s.contains("minilm-l12") => Ok(EmbeddingModel::AllMiniLML12V2),
        s if s.contains("bge-m3") => Ok(EmbeddingModel::BGEM3),
        _ => anyhow::bail!(
            "Unsupported embedding model '{}'. Please specify a valid FastEmbed model identifier.",
            id
        ),
    }
}

/// Embedding model files loaded from bytes instead of the Hugging Face cache.
struct ModelFiles {
    onnx: Vec<u8>,
    tokenizer: Vec<u8>,
    config: Vec<u8>,
    special_tokens_map: Vec<u8>,
    tokenizer_config: Vec<u8>,
}

impl ModelFiles {
    fn tokenizer(&self) -> TokenizerFiles {
        TokenizerFiles {
            tokenizer_file: self.tokenizer.clone(),
            config_file: self.config.clone(),
            special_tokens_map_file: self.special_tokens_map.clone(),
            tokenizer_config_file: self.tokenizer_config.clone(),
        }
    }

    /// Reads a Hugging Face model snapshot directory (`onnx/model.onnx` or
    /// `model.onnx`, plus the tokenizer JSON files).
    fn read_dir(dir: &std::path::Path) -> Result<Self> {
        let read = |name: &str| {
            std::fs::read(dir.join(name))
                .with_context(|| format!("reading {}", dir.join(name).display()))
        };
        let onnx = if dir.join("onnx/model.onnx").exists() {
            read("onnx/model.onnx")?
        } else {
            read("model.onnx")?
        };
        Ok(Self {
            onnx,
            tokenizer: read("tokenizer.json")?,
            config: read("config.json")?,
            special_tokens_map: read("special_tokens_map.json")?,
            tokenizer_config: read("tokenizer_config.json")?,
        })
    }
}

/// Model files compiled into the binary (`--features bundled-models`, with
/// `TELLODB_BUNDLE_DIR` pointing at a model snapshot at build time), then
/// `TELLODB_MODEL_DIR` at run time. `None` downloads through fastembed.
fn local_model_files(model_dir: Option<&Path>) -> Result<Option<(String, ModelFiles)>> {
    #[cfg(feature = "bundled-models")]
    {
        let files = ModelFiles {
            onnx: include_bytes!(concat!(env!("TELLODB_BUNDLE_DIR"), "/onnx/model.onnx")).to_vec(),
            tokenizer: include_bytes!(concat!(env!("TELLODB_BUNDLE_DIR"), "/tokenizer.json"))
                .to_vec(),
            config: include_bytes!(concat!(env!("TELLODB_BUNDLE_DIR"), "/config.json")).to_vec(),
            special_tokens_map: include_bytes!(concat!(
                env!("TELLODB_BUNDLE_DIR"),
                "/special_tokens_map.json"
            ))
            .to_vec(),
            tokenizer_config: include_bytes!(concat!(
                env!("TELLODB_BUNDLE_DIR"),
                "/tokenizer_config.json"
            ))
            .to_vec(),
        };
        return Ok(Some(("bundled".to_string(), files)));
    }
    #[allow(unreachable_code)]
    match model_dir {
        Some(dir) => Ok(Some((dir.display().to_string(), ModelFiles::read_dir(dir)?))),
        None => Ok(None),
    }
}

/// `TELLODB_RERANK_MODEL`: a fastembed cross-encoder, or `none`.
fn parse_reranker_model(name: &str) -> Result<Option<(&'static str, RerankerModel)>> {
    Ok(Some(match name.trim().to_ascii_lowercase().as_str() {
        "none" | "off" => return Ok(None),
        "bge-reranker-base" | "baai/bge-reranker-base" => {
            ("BAAI/bge-reranker-base", RerankerModel::BGERerankerBase)
        }
        "bge-reranker-v2-m3" => ("rozgo/bge-reranker-v2-m3", RerankerModel::BGERerankerV2M3),
        "jina-reranker-v1-turbo-en" => {
            ("jinaai/jina-reranker-v1-turbo-en", RerankerModel::JINARerankerV1TurboEn)
        }
        "jina-reranker-v2-base-multilingual" => (
            "jinaai/jina-reranker-v2-base-multilingual",
            RerankerModel::JINARerankerV2BaseMultiligual,
        ),
        other => anyhow::bail!(
            "unknown TELLODB_RERANK_MODEL '{other}' (bge-reranker-base, bge-reranker-v2-m3, \
             jina-reranker-v1-turbo-en, jina-reranker-v2-base-multilingual, none)"
        ),
    }))
}

/// Execution providers for the selected device, newest first. Shared so every
/// ONNX model in the process (embedder, reranker, extractor) runs on the same
/// device.
pub(crate) fn execution_providers_for(
    use_gpu: bool,
    use_coreml: bool,
) -> Vec<ort::execution_providers::ExecutionProviderDispatch> {
    let mut eps: Vec<ort::execution_providers::ExecutionProviderDispatch> = Vec::new();
    if use_gpu {
        eps.push(CUDA::default().into());
    } else if use_coreml {
        #[cfg(target_os = "macos")]
        eps.push(ort::ep::CoreML::default().into());
    }
    let _ = use_coreml;
    eps
}

/// The device selected during model initialization, as `(use_gpu, use_coreml)`.
pub(crate) fn selected_device() -> (bool, bool) {
    DEVICE_FLAGS.get().copied().unwrap_or((false, false))
}

const BGE_QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";

/// `TELLODB_QUERY_INSTRUCTION`: unset uses the model's recommended instruction
/// (BGE English v1.5 models), `off` or empty disables it, anything else is used
/// verbatim.
fn query_instruction_for(model_id: &str, setting: Option<&str>) -> String {
    match setting.map(str::trim) {
        Some(v) if v.is_empty() || v.eq_ignore_ascii_case("off") => String::new(),
        Some(v) if !v.eq_ignore_ascii_case("auto") => format!("{v} "),
        _ => {
            let id = model_id.to_ascii_lowercase();
            if id.contains("bge-") && id.contains("-en") {
                BGE_QUERY_INSTRUCTION.to_string()
            } else {
                String::new()
            }
        }
    }
}

pub struct SemanticInference {
    embedding_model_id: String,
    /// Cache namespace: model id plus every setting that changes the vector.
    cache_model_key: String,
    embedding_dim: usize,
    embed_batch: usize,
    max_tokens: usize,
    /// Reranker model id, when reranking is enabled.
    rerank_model_id: Option<&'static str>,
    /// Prepended to queries (not documents) for asymmetric retrieval models.
    query_instruction: String,
    executors: Vec<Mutex<TextEmbedding>>,
    next_executor: AtomicUsize,
    rerankers: Vec<Arc<Mutex<TextRerank>>>,
    rerank_cache: Mutex<LruCache<u64, Arc<Vec<f32>>>>,
    /// Caps concurrent model calls (1 on GPU so batches don't compete for memory).
    permits: ComputePermits,
    cache: EmbeddingCache,
    device_label: &'static str,
}

impl SemanticInference {
    pub async fn new() -> Result<Self> {
        let config = Config::from_env()?;
        let cache_path = config.embedding.cache_path.clone().or_else(|| {
            crate::runtime_paths::RuntimePaths::from_env()
                .ok()
                .map(|paths| paths.embedding_cache().to_path_buf())
        });
        Self::with_config(cache_path, &config.embedding, &config.rerank).await
    }

    pub async fn with_cache_path(cache_path: Option<PathBuf>) -> Result<Self> {
        let config = Config::from_env()?;
        Self::with_config(cache_path, &config.embedding, &config.rerank).await
    }

    pub async fn with_config(
        cache_path: Option<PathBuf>,
        embedding: &EmbeddingConfig,
        rerank: &RerankConfig,
    ) -> Result<Self> {
        let embedding_model_id = embedding.model_id.clone();
        if cfg!(debug_assertions) && embedding_model_id.trim().eq_ignore_ascii_case("test") {
            return Ok(Self::test_stub(cache_path, embedding));
        }
        let model_name = parse_embedding_model(&embedding_model_id)?;
        let embedding_dim =
            TextEmbedding::get_model_info(&model_name).map(|info| info.dim).unwrap_or_else(|_| {
                embedding_dimensions_for_model(&embedding_model_id, embedding.dimension)
            });

        let threads = embedding.threads.max(1);
        // Sizes the rayon pool used by ingest NLP. ONNX Runtime threads are set
        // by fastembed to all visible CPUs per session; on Linux restrict them
        // with `taskset`, which `available_parallelism` respects.
        if let Err(err) = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global() {
            tracing::debug!(error = %err, "rayon global pool already initialised");
        }

        let device_env = embedding.device.trim().to_ascii_lowercase();
        let use_gpu = device_env == "gpu" || device_env == "cuda";
        let use_coreml = device_env == "coreml" || device_env == "mps" || device_env == "mac";
        let device_label: &'static str = if use_gpu {
            "CUDA"
        } else if use_coreml {
            "CoreML"
        } else {
            "CPU"
        };
        let _ = DEVICE_LABEL.set(device_label);
        let _ = DEVICE_FLAGS.set((use_gpu, use_coreml));

        let n_embed = embedding.executors.clamp(1, 32);
        let max_tokens = embedding.max_tokens.clamp(16, 8192);
        let embed_batch = embedding.batch.max(1);

        let rerank_model = parse_reranker_model(&rerank.model)?;
        let rerank_enabled = rerank.enabled && rerank_model.is_some();
        let n_rerank = if rerank_enabled { rerank.executors.clamp(1, 16) } else { 0 };
        let cache_size = rerank.cache_size.max(64);

        tracing::info!(
            model = %embedding_model_id,
            dim = embedding_dim,
            device = device_label,
            embed_executors = n_embed,
            rerank_executors = n_rerank,
            max_tokens,
            embed_batch,
            rayon_threads = threads,
            "initialising semantic models"
        );

        let execution_providers = || execution_providers_for(use_gpu, use_coreml);

        let local_files = local_model_files(embedding.model_dir.as_deref())?;
        let mut executors = Vec::with_capacity(n_embed);
        for i in 0..n_embed {
            let model = match &local_files {
                Some((source, files)) => {
                    if i == 0 {
                        tracing::info!(source = %source, "loading embedding model without download");
                    }
                    let pooling = TextEmbedding::get_default_pooling_method(&model_name);
                    let mut user_model =
                        UserDefinedEmbeddingModel::new(files.onnx.clone(), files.tokenizer());
                    if let Some(pooling) = pooling {
                        user_model = user_model.with_pooling(pooling);
                    }
                    let options = InitOptionsUserDefined::new()
                        .with_max_length(max_tokens)
                        .with_execution_providers(execution_providers());
                    TextEmbedding::try_new_from_user_defined(user_model, options)
                }
                None => {
                    let mut options = TextInitOptions::default();
                    options.model_name = model_name.clone();
                    options.max_length = max_tokens;
                    options.show_download_progress = i == 0;
                    options.execution_providers.splice(0..0, execution_providers());
                    TextEmbedding::try_new(options)
                }
            }
            .with_context(|| format!("failed to load embedding model {embedding_model_id}"))?;
            executors.push(Mutex::new(model));
        }

        let probe = executors[0].lock().embed(["probe"], None)?;
        let actual_dim = probe.first().map(Vec::len).unwrap_or(0);
        if actual_dim != embedding_dim {
            anyhow::bail!(
                "Embedding model dimension mismatch for '{}': expected {}, got {}",
                embedding_model_id,
                embedding_dim,
                actual_dim
            );
        }

        let mut rerankers = Vec::with_capacity(n_rerank);
        for i in 0..n_rerank {
            let mut options = RerankInitOptions::default();
            options.model_name = rerank_model.clone().expect("rerank enabled implies a model").1;
            options.show_download_progress = i == 0;
            options.execution_providers.splice(0..0, execution_providers());
            rerankers.push(Arc::new(Mutex::new(TextRerank::try_new(options)?)));
        }

        let cache = EmbeddingCache::new(cache_path, embedding.cache_enabled);

        let query_instruction =
            query_instruction_for(&embedding_model_id, embedding.query_instruction.as_deref());

        Ok(Self {
            rerank_model_id: rerank_model.filter(|_| n_rerank > 0).map(|(id, _)| id),
            query_instruction,
            cache_model_key: format!("{embedding_model_id}@{max_tokens}"),
            embedding_model_id,
            embedding_dim,
            embed_batch,
            max_tokens,
            permits: ComputePermits::new(if use_gpu { 1 } else { n_embed.max(n_rerank) }),
            executors,
            next_executor: AtomicUsize::new(0),
            rerankers,
            rerank_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_size).expect("cache size must be non-zero"),
            )),
            cache,
            device_label,
        })
    }

    fn test_stub(cache_path: Option<PathBuf>, embedding: &EmbeddingConfig) -> Self {
        let embedding_dim = embedding_dimensions_for_model("test", embedding.dimension);
        Self {
            embedding_model_id: "test".to_string(),
            cache_model_key: format!("test@{}", embedding.max_tokens.max(1)),
            embedding_dim,
            embed_batch: embedding.batch.max(1),
            max_tokens: embedding.max_tokens.max(1),
            rerank_model_id: None,
            query_instruction: String::new(),
            executors: Vec::new(),
            next_executor: AtomicUsize::new(0),
            rerankers: Vec::new(),
            rerank_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(64).expect("literal is non-zero"),
            )),
            permits: ComputePermits::new(1),
            cache: EmbeddingCache::new(cache_path, embedding.cache_enabled),
            device_label: "CPU",
        }
    }

    pub fn is_rerank_enabled(&self) -> bool {
        !self.rerankers.is_empty()
    }

    /// The loaded reranker model id, or `off`.
    pub fn rerank_mode(&self) -> &'static str {
        self.rerank_model_id.unwrap_or("off")
    }

    pub fn embed_cache_hits(&self) -> u64 {
        self.cache.hits()
    }

    pub fn embed_cache_misses(&self) -> u64 {
        self.cache.misses()
    }

    pub fn clear_embedding_cache(&self) {
        self.cache.clear();
    }

    pub fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_texts(&[text])?.pop().context("embedding model returned no vector")
    }

    pub fn generate_query_embedding(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_queries(&[text])?.pop().context("embedding model returned no vector")
    }

    /// Embeds search queries, applying the model's query instruction.
    pub fn embed_queries(&self, queries: &[&str]) -> Result<Vec<Vec<f32>>> {
        if self.query_instruction.is_empty() {
            return self.embed_texts(queries);
        }
        let prefixed: Vec<String> =
            queries.iter().map(|q| format!("{}{}", self.query_instruction, q)).collect();
        let refs: Vec<&str> = prefixed.iter().map(String::as_str).collect();
        self.embed_texts(&refs)
    }

    pub fn query_instruction(&self) -> &str {
        &self.query_instruction
    }

    /// Embeds `texts` in input order. Cached vectors are reused; the rest are
    /// sorted by length and embedded in batches of `embed_batch`, spread over
    /// the executors. Every returned vector has `embedding_dim` finite values.
    pub fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let hashes: Vec<[u8; 32]> = texts.iter().map(|t| hash_text(t)).collect();
        let mut results = self.cache.get_many(&self.cache_model_key, &hashes, self.embedding_dim);

        let mut missing: Vec<usize> = (0..texts.len()).filter(|&i| results[i].is_none()).collect();
        if missing.is_empty() {
            return Ok(results.into_iter().flatten().collect());
        }
        missing.sort_by_key(|&i| texts[i].len());

        if self.executors.is_empty() {
            for &idx in &missing {
                results[idx] = Some(test_embedding(texts[idx], self.embedding_dim));
            }
        } else {
            let batches: Vec<&[usize]> = missing.chunks(self.embed_batch).collect();
            let computed: Vec<Vec<Vec<f32>>> = if self.executors.len() == 1 || batches.len() == 1 {
                batches
                    .iter()
                    .map(|batch| self.embed_on_executor(texts, batch))
                    .collect::<Result<_>>()?
            } else {
                std::thread::scope(|scope| {
                    let handles: Vec<_> = batches
                        .iter()
                        .map(|batch| scope.spawn(move || self.embed_on_executor(texts, batch)))
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| {
                            h.join()
                                .map_err(|_panic| anyhow::anyhow!("embedding thread panicked"))?
                        })
                        .collect::<Result<_>>()
                })?
            };

            for (batch, vectors) in batches.iter().zip(computed) {
                for (&idx, vector) in batch.iter().zip(vectors) {
                    results[idx] = Some(vector);
                }
            }
        }
        let cache_items: Vec<([u8; 32], &[f32])> = missing
            .iter()
            .filter_map(|&idx| results[idx].as_deref().map(|v| (hashes[idx], v)))
            .collect();
        self.cache.put_many(&self.cache_model_key, &cache_items);

        let embedded: Vec<Vec<f32>> = results.into_iter().flatten().collect();
        if embedded.len() != texts.len() {
            anyhow::bail!("embedded {} of {} texts", embedded.len(), texts.len());
        }
        Ok(embedded)
    }

    fn embed_on_executor(&self, texts: &[&str], batch: &[usize]) -> Result<Vec<Vec<f32>>> {
        let _permit = self.permits.acquire();
        let idx = self.next_executor.fetch_add(1, Ordering::Relaxed) % self.executors.len();
        let inputs: Vec<&str> = batch.iter().map(|&i| texts[i]).collect();
        let vectors = self.executors[idx].lock().embed(&inputs, Some(inputs.len()))?;
        if vectors.len() != inputs.len() {
            anyhow::bail!(
                "embedding model returned {} vectors for {} texts",
                vectors.len(),
                inputs.len()
            );
        }
        if let Some(bad) = vectors
            .iter()
            .find(|v| v.len() != self.embedding_dim || v.iter().any(|x| !x.is_finite()))
        {
            anyhow::bail!(
                "embedding model returned an invalid vector (len {}, expected {})",
                bad.len(),
                self.embedding_dim
            );
        }
        Ok(vectors)
    }

    /// `embed_texts` on the blocking pool, for async callers.
    pub async fn embed_texts_async(self: &Arc<Self>, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let this = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            this.embed_texts(&refs)
        })
        .await
        .context("embedding task panicked")?
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn embed_max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn embed_batch_size(&self) -> usize {
        self.embed_batch
    }

    pub fn embedding_model_id(&self) -> &str {
        &self.embedding_model_id
    }

    pub fn device_label(&self) -> &str {
        self.device_label
    }

    /// Device of the initialised models ("CPU" before initialisation).
    pub fn device_label_static() -> &'static str {
        DEVICE_LABEL.get().copied().unwrap_or("CPU")
    }

    pub fn executor_count(&self) -> usize {
        self.executors.len()
    }

    #[allow(dead_code)]
    pub fn extract_entities(&self, _text: &str) -> Result<Vec<(String, String)>> {
        Ok(Vec::new())
    }

    /// Rerank `texts` against `q`. Uses an LRU cache keyed by
    /// (q, sorted(texts) hashes) so repeated questions with overlapping
    /// candidate sets skip the cross-encoder call entirely.
    ///
    /// If more than one rerank executor is configured, `texts` is split
    /// into roughly equal chunks and each chunk is run on a different
    /// executor in parallel via `std::thread::scope`.
    pub fn predict_scores_batch(&self, q: &str, texts: &[String]) -> Result<Vec<f32>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if self.rerankers.is_empty() {
            anyhow::bail!("reranker is disabled (TELLODB_RERANK=off)");
        }

        let cache_key = rerank_cache_key(q, texts);
        if let Some(cached) = self.rerank_cache.lock().get(&cache_key) {
            return Ok((**cached).clone());
        }

        let n_exec = self.rerankers.len();
        let chunks = split_for_rerank(texts, n_exec);
        let _permit = self.permits.acquire();
        let results: Vec<(usize, Vec<f32>)> = if chunks.len() == 1 {
            chunks
                .into_iter()
                .enumerate()
                .map(|(i, (offset, chunk))| Ok((offset, self.rerank_on_executor(i, q, &chunk)?)))
                .collect::<Result<_>>()?
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = chunks
                    .into_iter()
                    .enumerate()
                    .map(|(i, (offset, chunk))| {
                        scope.spawn(move || Ok((offset, self.rerank_on_executor(i, q, &chunk)?)))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().map_err(|_panic| anyhow::anyhow!("rerank thread panicked"))?
                    })
                    .collect::<Result<Vec<_>>>()
            })?
        };

        let mut scores = vec![0.0f32; texts.len()];
        for (offset, chunk_scores) in results {
            for (i, s) in chunk_scores.into_iter().enumerate() {
                scores[offset + i] = s;
            }
        }

        self.rerank_cache.lock().put(cache_key, Arc::new(scores.clone()));
        Ok(scores)
    }

    fn rerank_on_executor(
        &self,
        executor_idx: usize,
        q: &str,
        texts: &[String],
    ) -> Result<Vec<f32>> {
        let rr = &self.rerankers[executor_idx % self.rerankers.len()];
        let mut reranker = rr.lock();
        let doc_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let results = reranker.rerank(q, doc_refs, false, None)?;
        let mut scores = vec![0.0f32; texts.len()];
        for res in results {
            if res.index < scores.len() {
                scores[res.index] = res.score;
            }
        }
        Ok(scores)
    }
}

/// Split `texts` into at most `n` roughly-equal chunks. Returns (offset, chunk)
/// pairs so the caller can stitch results back into the original order.
fn split_for_rerank(texts: &[String], n: usize) -> Vec<(usize, Vec<String>)> {
    if n <= 1 || texts.len() <= 8 {
        return vec![(0, texts.to_vec())];
    }
    let n = n.min(texts.len());
    let chunk_size = texts.len().div_ceil(n);
    let mut out = Vec::with_capacity(n);
    for (i, chunk) in texts.chunks(chunk_size).enumerate() {
        out.push((i * chunk_size, chunk.to_vec()));
    }
    out
}

/// Cache key = hash(query) XOR hash(sorted texts). Cheap, no need for crypto.
fn rerank_cache_key(q: &str, texts: &[String]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    q.hash(&mut hasher);
    let mut sorted: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    sorted.sort();
    for t in &sorted {
        t.hash(&mut hasher);
    }
    hasher.finish()
}

fn embedding_dimensions_for_model(id: &str, configured: Option<usize>) -> usize {
    if let Some(dim) = configured {
        return dim;
    }
    match id {
        s if s.contains("bge-small") => 384,
        s if s.contains("bge-base") => 768,
        s if s.contains("bge-large") => 1024,
        s if s.contains("MiniLM-L6") => 384,
        s if s.contains("MiniLM-L12") => 384,
        s if s.contains("e5-small") => 384,
        s if s.contains("e5-base") => 768,
        s if s.contains("e5-large") => 1024,
        _ => 384,
    }
}

fn test_embedding(text: &str, dimension: usize) -> Vec<f32> {
    use sha2::{Digest, Sha256};

    let mut vector = vec![0.0; dimension.max(1)];
    for token in text.split_whitespace().map(|token| token.to_ascii_lowercase()) {
        let digest = Sha256::digest(token.as_bytes());
        let mut seed = [0; 8];
        seed.copy_from_slice(&digest[..8]);
        let index = u64::from_le_bytes(seed) as usize % vector.len();
        vector[index] += 1.0;
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    } else {
        vector[0] = 1.0;
    }
    vector
}

#[cfg(test)]
mod tests {
    #[test]
    fn query_instruction_defaults_to_bge_prompt() {
        assert_eq!(
            super::query_instruction_for("BAAI/bge-small-en-v1.5", None),
            super::BGE_QUERY_INSTRUCTION
        );
        assert_eq!(super::query_instruction_for("BAAI/bge-small-en-v1.5", Some("off")), "");
        assert_eq!(
            super::query_instruction_for("sentence-transformers/all-MiniLM-L6-v2", None),
            ""
        );
        assert_eq!(super::query_instruction_for("x", Some("query:")), "query: ");
    }

    use super::*;

    #[test]
    fn rerank_cache_key_stable_for_permuted_texts() {
        // Cache key must be order-independent so that calling rerank with the
        // same set of candidates in any order hits the same cache entry.
        let a = rerank_cache_key("query", &["foo".into(), "bar".into(), "baz".into()]);
        let b = rerank_cache_key("query", &["baz".into(), "foo".into(), "bar".into()]);
        assert_eq!(a, b);
    }

    #[test]
    fn rerank_cache_key_differs_per_query() {
        let a = rerank_cache_key("q1", &["x".into()]);
        let b = rerank_cache_key("q2", &["x".into()]);
        assert_ne!(a, b);
    }

    #[test]
    fn rerank_cache_key_differs_per_text() {
        let a = rerank_cache_key("q", &["x".into()]);
        let b = rerank_cache_key("q", &["y".into()]);
        assert_ne!(a, b);
    }

    #[test]
    fn split_for_rerank_serializes_short_input() {
        // Fewer than 8 texts should never be split, regardless of n.
        let texts: Vec<String> = (0..5).map(|i| format!("t{i}")).collect();
        let out = split_for_rerank(&texts, 4);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 0);
        assert_eq!(out[0].1, texts);
    }

    #[test]
    fn split_for_rerank_splits_long_input() {
        let texts: Vec<String> = (0..20).map(|i| format!("t{i}")).collect();
        let out = split_for_rerank(&texts, 4);
        assert_eq!(out.len(), 4);
        // Verify offsets are correct and chunks cover the input.
        let total: usize = out.iter().map(|(_, c)| c.len()).sum();
        assert_eq!(total, 20);
        assert_eq!(out[0].0, 0);
        assert_eq!(out[1].0, 5);
        assert_eq!(out[2].0, 10);
        assert_eq!(out[3].0, 15);
    }

    #[test]
    fn test_unsupported_model_validation_fails() {
        let res = parse_embedding_model("unsupported-fake-model-xyz");
        assert!(res.is_err());
        let err_msg = res.err().unwrap().to_string();
        assert!(err_msg.contains("Unsupported embedding model"));
    }

    #[test]
    fn test_supported_model_validation_succeeds() {
        assert!(parse_embedding_model("BAAI/bge-small-en-v1.5").is_ok());
        assert!(parse_embedding_model("bge-small").is_ok());
        assert!(parse_embedding_model("BGESmallENV15").is_ok());
        assert!(parse_embedding_model("AllMiniLML6V2").is_ok());
        assert!(parse_embedding_model("Qdrant/all-MiniLM-L6-v2-onnx").is_ok());
        assert!(parse_embedding_model("all-MiniLM-L6-v2").is_ok());
    }

    #[test]
    fn embedding_cache_round_trip_and_clear() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cache = EmbeddingCache::new(Some(temp_dir.path().join("cache.sqlite")), true);
        assert!(cache.is_enabled());

        let (h1, h2, h3) = (hash_text("a"), hash_text("b"), hash_text("c"));
        let emb1 = vec![1.0f32, 2.0, 3.0];
        let emb2 = vec![4.0f32, 5.0, 6.0];
        cache.put_many("m@512", &[(h1, &emb1), (h2, &emb2)]);

        let got = cache.get_many("m@512", &[h1, h2, h3], 3);
        assert_eq!(got, vec![Some(emb1.clone()), Some(emb2), None]);
        assert_eq!((cache.hits(), cache.misses()), (2, 1));

        // A different model key (e.g. another max_tokens) never reuses vectors.
        assert_eq!(cache.get_many("m@256", &[h1], 3), vec![None]);
        // Wrong dimension is treated as a miss.
        assert_eq!(cache.get_many("m@512", &[h1], 4), vec![None]);

        cache.clear();
        assert_eq!(cache.get_many("m@512", &[h1], 3), vec![None]);
    }

    #[test]
    fn disabled_cache_always_misses() {
        let cache = EmbeddingCache::new(None, true);
        assert!(!cache.is_enabled());
        assert_eq!(cache.get_many("m", &[hash_text("a")], 3), vec![None]);
    }

    #[test]
    fn compute_permits_limit_concurrency() {
        let permits = Arc::new(ComputePermits::new(2));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let (permits, active, peak) = (permits.clone(), active.clone(), peak.clone());
                scope.spawn(move || {
                    let _guard = permits.acquire();
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });
        assert!(peak.load(Ordering::SeqCst) <= 2);
    }
}
