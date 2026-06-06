use anyhow::Result;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions, TextRerank, RerankInitOptions, RerankerModel};
use ort::ep::CUDA;

use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;
use std::sync::Arc;



/// Default number of embedding executors. Each holds a copy of the embed
/// model in memory; for BGE-small-en-v1.5 (~130MB) this is fine. Override with
/// `TEMPORAL_MEMORY_EMBED_EXECUTORS` to tune.
const DEFAULT_EMBED_EXECUTORS: usize = 4;

/// Default number of rerank executors. Each holds a copy of BGE-reranker-base
/// (~700MB on CPU). Override with `TEMPORAL_MEMORY_RERANK_EXECUTORS`.
const DEFAULT_RERANK_EXECUTORS: usize = 2;

/// Default size of the rerank-result LRU cache. Each entry holds a Vec<f32>
/// of length ≤ 32 (one NEURAL_BATCH chunk). Override with
/// `TEMPORAL_MEMORY_RERANK_CACHE_SIZE`.
const DEFAULT_RERANK_CACHE_SIZE: usize = 4096;

pub struct SemanticInference {
    embedding_model_id: String,
    embedding_dim: usize,
    executors: Vec<Arc<SemanticExecutor>>,
    next_executor: std::sync::atomic::AtomicUsize,
    rerankers: Vec<Arc<Mutex<TextRerank>>>,
    rerank_cache: Mutex<LruCache<u64, Arc<Vec<f32>>>>,
    /// Limits the number of concurrent ONNX embedding inference calls.
    /// On GPU (1 executor) the per-model Mutex already serialises calls, but
    /// ingest and query threads can concurrently pick DIFFERENT executors,
    /// causing two simultaneous GPU MatMul ops that together exceed GPU memory
    /// (each needs 90–220 MB of intermediate tensor space). This semaphore
    /// caps total in-flight embed calls to min(n_exec, 2) on CPU or 1 on GPU.
    embed_sem: Arc<tokio::sync::Semaphore>,
}

struct SemanticExecutor {
    fast_embedding: Option<Mutex<TextEmbedding>>,
    execution_device_label: &'static str,
}

impl SemanticInference {
    pub async fn new() -> Result<Self> {
        let embedding_model_id = "BAAI/bge-small-en-v1.5".to_string();
        let embedding_dim = embedding_dimensions_for_model(&embedding_model_id);

        // Use GPU if TEMPORAL_MEMORY_DEVICE=gpu or cuda is set.
        let device_env = std::env::var("TEMPORAL_MEMORY_DEVICE").unwrap_or_default().to_lowercase();
        let use_gpu = device_env == "gpu" || device_env == "cuda";
        let use_coreml = device_env == "coreml" || device_env == "mps" || device_env == "mac";

        let device_label: &'static str = if use_gpu { "CUDA" } else if use_coreml { "CoreML" } else { "CPU" };

        let default_n_embed = if use_gpu || use_coreml { 1 } else { DEFAULT_EMBED_EXECUTORS };
        let default_n_rerank = if use_gpu || use_coreml { 1 } else { DEFAULT_RERANK_EXECUTORS };

        let n_embed = std::env::var("TEMPORAL_MEMORY_EMBED_EXECUTORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n: &usize| *n >= 1 && *n <= 32)
            .unwrap_or(default_n_embed);

        let n_rerank = std::env::var("TEMPORAL_MEMORY_RERANK_EXECUTORS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n: &usize| *n >= 1 && *n <= 16)
            .unwrap_or(default_n_rerank);

        let cache_size = std::env::var("TEMPORAL_MEMORY_RERANK_CACHE_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n: &usize| *n >= 64)
            .unwrap_or(DEFAULT_RERANK_CACHE_SIZE);

        info_msg(&format!(
            "Initializing {n_embed} embed executors and {n_rerank} rerank executors on {device_label}, \
             rerank cache size {cache_size}"
        ));

        // Build embed executors
        let mut executors = Vec::with_capacity(n_embed);
        for i in 0..n_embed {
            let mut options = TextInitOptions::default();
            options.model_name = EmbeddingModel::try_from(embedding_model_id.clone())
                .unwrap_or(EmbeddingModel::AllMiniLML6V2);
            options.show_download_progress = i == 0; // only show progress on the first
            if use_gpu {
                let cuda_ep: ort::execution_providers::ExecutionProviderDispatch =
                    CUDA::default().into();
                options.execution_providers.insert(0, cuda_ep);
            } else if use_coreml {
                #[cfg(target_os = "macos")]
                {
                    let coreml_ep: ort::execution_providers::ExecutionProviderDispatch =
                        ort::ep::CoreML::default().into();
                    options.execution_providers.insert(0, coreml_ep);
                }
            }
            let model = TextEmbedding::try_new(options)?;
            executors.push(Arc::new(SemanticExecutor {
                fast_embedding: Some(Mutex::new(model)),
                execution_device_label: device_label,
            }));
        }

        // Build rerank executors
        let mut rerankers = Vec::with_capacity(n_rerank);
        for i in 0..n_rerank {
            let mut rerank_options = RerankInitOptions::default();
            rerank_options.model_name = RerankerModel::BGERerankerBase;
            rerank_options.show_download_progress = i == 0;
            if use_gpu {
                let cuda_ep: ort::execution_providers::ExecutionProviderDispatch =
                    CUDA::default().into();
                rerank_options.execution_providers.insert(0, cuda_ep);
            } else if use_coreml {
                #[cfg(target_os = "macos")]
                {
                    let coreml_ep: ort::execution_providers::ExecutionProviderDispatch =
                        ort::ep::CoreML::default().into();
                    rerank_options.execution_providers.insert(0, coreml_ep);
                }
            }
            let rr = TextRerank::try_new(rerank_options)?;
            rerankers.push(Arc::new(Mutex::new(rr)));
        }

        // Maximum concurrent embedding calls. On GPU we allow exactly 1;
        // on CPU we allow up to n_embed (each executor runs independently).
        // This prevents two simultaneous CUDA MatMuls from fighting over
        // GPU memory (each needs 90-220 MB of intermediate tensor space).
        let sem_permits = if use_gpu { 1 } else { n_embed };

        Ok(Self {
            embedding_model_id,
            embedding_dim,
            executors,
            next_executor: std::sync::atomic::AtomicUsize::new(0),
            rerankers,
            rerank_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_size).expect("cache size must be non-zero"),
            )),
            embed_sem: Arc::new(tokio::sync::Semaphore::new(sem_permits)),
        })
    }

    fn next_executor_arc(&self) -> Arc<SemanticExecutor> {
        let idx = self.next_executor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.executors[idx % self.executors.len()].clone()
    }

    fn next_executor(&self) -> &SemanticExecutor {
        let idx = self.next_executor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        &self.executors[idx % self.executors.len()]
    }


    pub fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let executor = self.next_executor();
        if let Some(ref model) = executor.fast_embedding {
            let mut model = model.lock();
            let embeddings: Vec<Vec<f32>> = model.embed([text], None)?;
            Ok(embeddings.into_iter().next().unwrap_or_default())
        } else {
            anyhow::bail!("ORT model not loaded")
        }
    }

    pub fn generate_query_embedding(&self, text: &str) -> Result<Vec<f32>> {
        self.generate_embedding(text)
    }

    pub async fn generate_query_embedding_async(&self, text: String) -> Result<Vec<f32>> {
        let executor = self.next_executor_arc();
        tokio::task::spawn_blocking(move || {
            if let Some(ref model) = executor.fast_embedding {
                let mut model = model.lock();
                let embeddings: Vec<Vec<f32>> = model.embed([text.as_str()], None)?;
                Ok(embeddings.into_iter().next().unwrap_or_default())
            } else {
                anyhow::bail!("ORT model not loaded")
            }
        })
        .await?
    }

    /// Batch-embed multiple texts in a single ONNX inference call.
    /// This is significantly faster than calling `generate_query_embedding` N times
    /// because the matrix multiplications are batched across the batch dimension.
    pub fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        if texts.is_empty() {
            return Vec::new();
        }
        let executor = self.next_executor();
        if let Some(ref model) = executor.fast_embedding {
            let mut model = model.lock();
            model.embed(texts, None).unwrap_or_default()
        } else {
            vec![Vec::new(); texts.len()]
        }
    }

    pub async fn embed_batch_async(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        if texts.is_empty() {
            return Vec::new();
        }
        let _permit = self.embed_sem.acquire().await.ok();
        let executor = self.next_executor_arc();
        let len = texts.len();
        tokio::task::spawn_blocking(move || {
            if let Some(ref model) = executor.fast_embedding {
                let mut model = model.lock();
                let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                model.embed(refs, None).unwrap_or_default()
            } else {
                vec![Vec::new(); len]
            }
        })
        .await
        .unwrap_or_default()
    }

    /// Embed multiple texts across all configured executors in parallel.
    /// Splits `texts` into roughly equal chunks (one per executor) and runs
    /// each chunk on a different executor. This is much faster than
    /// `embed_batch_async` for large batches when N>1 executors are
    /// configured, because each executor runs an independent ONNX inference.
    ///
    /// The returned Vec is in the same order as `texts`.
    pub async fn embed_batch_parallel(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        if texts.is_empty() {
            return Vec::new();
        }
        let n_exec = self.executors.len();
        // Acquire the global semaphore BEFORE splitting into per-executor
        // chunks. On GPU (sem_permits=1) this serialises all embed calls;
        // on CPU it allows n_exec parallel calls. Holding the permit for the
        // duration ensures no two concurrent batches compete for GPU memory.
        let _permit = self.embed_sem.acquire().await.ok();
        if n_exec == 1 || texts.len() <= 16 {
            // Single-executor or small batch: run synchronously on one executor.
            let executor = self.next_executor_arc();
            let len = texts.len();
            return tokio::task::spawn_blocking(move || {
                if let Some(ref model) = executor.fast_embedding {
                    let mut model = model.lock();
                    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
                    model.embed(refs, None).unwrap_or_default()
                } else {
                    vec![Vec::new(); len]
                }
            })
            .await
            .unwrap_or_default();
        }

        // Split into chunks aligned with executor count.
        let chunk_size = (texts.len() + n_exec - 1) / n_exec;
        let mut chunks: Vec<Vec<String>> = Vec::with_capacity(n_exec);
        for c in texts.chunks(chunk_size) {
            chunks.push(c.to_vec());
        }
        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let executor = self.next_executor_arc();
            handles.push(tokio::task::spawn_blocking(move || -> Vec<Vec<f32>> {
                if let Some(ref model) = executor.fast_embedding {
                    let mut model = model.lock();
                    let refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
                    model.embed(refs, None).unwrap_or_default()
                } else {
                    vec![Vec::new(); chunk.len()]
                }
            }));
        }
        let mut ordered: Vec<Vec<Vec<f32>>> = Vec::with_capacity(handles.len());
        let total = texts.len();
        for h in handles {
            match h.await {
                Ok(v) => ordered.push(v),
                Err(e) => {
                    tracing::warn!("embed chunk join failed: {:?}", e);
                    return vec![Vec::new(); total];
                }
            }
        }
        let mut out = Vec::with_capacity(total);
        for chunk_result in ordered {
            out.extend(chunk_result);
        }
        out
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn embedding_model_id(&self) -> &str {
        &self.embedding_model_id
    }

    pub fn device_label(&self) -> &str {
        self.executors.first().map(|e| e.execution_device_label).unwrap_or("CPU")
    }

    pub fn device_label_static() -> &'static str {
        "CPU"
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

        // Cache lookup keyed on (query hash, sorted set of text hashes).
        let cache_key = rerank_cache_key(q, texts);
        {
            let mut cache = self.rerank_cache.lock();
            if let Some(cached) = cache.get(&cache_key) {
                return Ok((**cached).clone());
            }
        }

        // Decide whether to run in parallel. The cost of std::thread::spawn
        // is ~30-50us; for tiny inputs we serialize.
        let n_exec = self.rerankers.len();
        let chunks: Vec<(usize, Vec<String>)> = split_for_rerank(texts, n_exec);

        let results: Vec<(usize, Vec<f32>)> = if chunks.len() == 1 || n_exec == 1 {
            // Serial path.
            let mut out = Vec::with_capacity(chunks.len());
            for (i, (offset, chunk)) in chunks.into_iter().enumerate() {
                let _permit = tokio::runtime::Handle::current().block_on(self.embed_sem.acquire()).ok();
                let scores = self.rerank_on_executor(i, q, &chunk)?;
                out.push((offset, scores));
            }
            out
        } else {
            // Parallel path: spawn one thread per chunk, each grabbing a
            // different rerank executor. The executor is held only for the
            // duration of the ONNX call, then released.
            use std::thread;
            thread::scope(|s| {
                let mut handles = Vec::with_capacity(chunks.len());
                for (i, (offset, chunk)) in chunks.into_iter().enumerate() {
                    let rr = self.rerankers[i % n_exec].clone();
                    let q_owned = q.to_string();
                    let sem = self.embed_sem.clone();
                    let h = s.spawn(move || -> Result<(usize, Vec<f32>)> {
                        let _permit = tokio::runtime::Handle::current().block_on(sem.acquire()).ok();
                        let mut reranker = rr.lock();
                        let doc_refs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
                        let results = reranker.rerank(q_owned.as_str(), doc_refs, false, None)?;
                        let mut scores = vec![0.0f32; chunk.len()];
                        for res in results {
                            if res.index < scores.len() {
                                scores[res.index] = res.score;
                            }
                        }
                        Ok((offset, scores))
                    });
                    handles.push(h);
                }
                let mut out = Vec::with_capacity(handles.len());
                for h in handles {
                    match h.join() {
                        Ok(Ok(pair)) => out.push(pair),
                        Ok(Err(e)) => return Err(e),
                        Err(_) => return Err(anyhow::anyhow!("rerank thread panicked")),
                    }
                }
                Ok::<_, anyhow::Error>(out)
            })?
        };

        // Stitch results back into a single Vec<f32> in original order.
        let mut scores = vec![0.0f32; texts.len()];
        for (offset, chunk_scores) in results {
            for (i, s) in chunk_scores.into_iter().enumerate() {
                scores[offset + i] = s;
            }
        }

        // Cache the result.
        let arc = Arc::new(scores.clone());
        self.rerank_cache.lock().put(cache_key, arc);

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
    let chunk_size = (texts.len() + n - 1) / n;
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

fn info_msg(msg: &str) {
    eprintln!("[semantic] {msg}");
}

#[cfg(test)]
mod tests {
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
}

fn embedding_dimensions_for_model(id: &str) -> usize {
    if let Ok(dim) = std::env::var("TEMPORAL_MEMORY_EMBEDDING_DIM") {
        if let Ok(d) = dim.parse::<usize>() {
            return d;
        }
    }
    match id {
        s if s.contains("bge-small") => 384,
        s if s.contains("bge-base") => 768,
        s if s.contains("bge-large") => 1024,
        s if s.contains("Qwen3-Embedding-0.6B") => 1024,
        s if s.contains("MiniLM-L6") => 384,
        s if s.contains("MiniLM-L12") => 384,
        s if s.contains("e5-small") => 384,
        s if s.contains("e5-base") => 768,
        s if s.contains("e5-large") => 1024,
        _ => 384,
    }
}
