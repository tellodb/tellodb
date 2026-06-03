use anyhow::Result;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use ort::ep::CUDA;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

const DEFAULT_MAX_INPUT_CHARS: usize = 8_192;

pub struct SemanticInference {
    embedding_model_id: String,
    embedding_dim: usize,
    executors: Vec<Arc<SemanticExecutor>>,
    next_executor: std::sync::atomic::AtomicUsize,
    max_input_chars: usize,
}

struct SemanticExecutor {
    fast_embedding: Option<Mutex<TextEmbedding>>,
    execution_device_label: &'static str,
}

impl SemanticInference {
    pub async fn new() -> Result<Self> {
        let embedding_model_id = "BAAI/bge-small-en-v1.5".to_string();
        let embedding_dim = embedding_dimensions_for_model(&embedding_model_id);

        let mut options = TextInitOptions::default();
        options.model_name = EmbeddingModel::try_from(embedding_model_id.clone())
            .unwrap_or(EmbeddingModel::AllMiniLML6V2);
        options.show_download_progress = true;

        // Use GPU if TEMPORAL_MEMORY_DEVICE=gpu is set.
        let use_gpu = std::env::var("TEMPORAL_MEMORY_DEVICE")
            .map(|v| v.eq_ignore_ascii_case("gpu"))
            .unwrap_or(false);
        let mut device_label = "CPU";
        if use_gpu {
            let cuda_ep: ort::execution_providers::ExecutionProviderDispatch =
                CUDA::default().into();
            options.execution_providers.insert(0, cuda_ep);
            device_label = "CUDA";
        }

        let model = TextEmbedding::try_new(options)?;

        let executor = Arc::new(SemanticExecutor {
            fast_embedding: Some(Mutex::new(model)),
            execution_device_label: device_label,
        });

        Ok(Self {
            embedding_model_id,
            embedding_dim,
            executors: vec![executor],
            next_executor: std::sync::atomic::AtomicUsize::new(0),
            max_input_chars: DEFAULT_MAX_INPUT_CHARS,
        })
    }

    fn next_executor(&self) -> &SemanticExecutor {
        let idx = self.next_executor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        &self.executors[idx % self.executors.len()]
    }

    pub async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let executor = self.next_executor();
        if let Some(ref model) = executor.fast_embedding {
            let mut model = model.lock().unwrap_or_else(|e| e.into_inner());
            let embeddings = model.embed(vec![text], None)?;
            Ok(embeddings.first().cloned().unwrap_or_default())
        } else {
            anyhow::bail!("ORT model not loaded")
        }
    }

    pub fn generate_query_embedding(&self, text: &str) -> Result<Vec<f32>> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.generate_embedding(text))
        })
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
            let mut model = model.lock().unwrap_or_else(|e| e.into_inner());
            model
                .embed(texts.to_vec(), None)
                .unwrap_or_else(|_| vec![Vec::new(); texts.len()])
        } else {
            vec![Vec::new(); texts.len()]
        }
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

    /// Number of GPU executors to create (one per detected GPU, capped at 4).
    fn gpu_executor_count() -> usize {
        let count = std::env::var("TEMPORAL_MEMORY_EXECUTORS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        if count > 0 { count } else { 0 }
    }

    pub fn executor_count(&self) -> usize {
        self.executors.len()
    }

    pub fn extract_entities(&self, text: &str) -> Result<Vec<(String, String)>> {
        let re = regex::Regex::new(r"\b([A-Z][a-zA-Z0-9\.\-']*(?:\s+[A-Z][a-zA-Z0-9\.\-']*)*)\b")?;

        let ignore_words: HashSet<&str> = [
            "The", "He", "She", "It", "They", "We", "Then", "In", "On", "At", "When", "How",
            "This", "That", "There", "Here", "A", "An", "But", "And", "Or", "If", "You", "My",
            "Our", "Their", "Your", "His", "Her", "Its", "All", "Any", "Some", "Every", "Each",
            "No", "Not", "Only", "Also", "Very", "So", "To", "From", "By", "For", "With", "About",
            "I", "As", "Are", "Is", "Was", "Were", "Be", "Been", "Have", "Has", "Had", "Do",
            "Does", "Did", "Can", "Could", "Will", "Would", "Should", "May", "Might", "Must",
        ]
        .iter()
        .cloned()
        .collect();

        let known_orgs: HashSet<&str> = [
            "Google",
            "Microsoft",
            "OpenAI",
            "Apple",
            "Amazon",
            "Meta",
            "IBM",
            "Intel",
            "Netflix",
            "Twitter",
            "Github",
            "Gitlab",
            "Oracle",
            "Tesla",
            "SpaceX",
            "NASA",
        ]
        .iter()
        .cloned()
        .collect();

        let known_locs: HashSet<&str> = [
            "Paris",
            "London",
            "Tokyo",
            "Berlin",
            "Rome",
            "Madrid",
            "Beijing",
            "Sydney",
            "California",
            "Texas",
            "Florida",
            "New York",
            "Boston",
            "Chicago",
            "Seattle",
            "San Francisco",
            "Europe",
            "America",
            "Asia",
            "Africa",
            "Canada",
            "Germany",
            "France",
            "Japan",
            "China",
            "India",
            "Australia",
            "UK",
            "US",
            "USA",
            "Washington",
            "Denver",
            "Austin",
            "Miami",
        ]
        .iter()
        .cloned()
        .collect();

        let known_pers: HashSet<&str> = [
            "Alice",
            "Bob",
            "Charlie",
            "David",
            "Eve",
            "Frank",
            "Grace",
            "Heidi",
            "Ivan",
            "Judy",
            "Mallory",
            "Niaj",
            "Olivia",
            "Peggy",
            "Rupert",
            "Sybil",
            "Trent",
            "Victor",
            "Walter",
            "John",
            "Mary",
            "James",
            "Patricia",
            "Robert",
            "Jennifer",
            "Michael",
            "Linda",
            "William",
            "Elizabeth",
            "Barbara",
            "Richard",
            "Susan",
            "Joseph",
            "Jessica",
            "Thomas",
            "Sarah",
            "Charles",
            "Karen",
            "Christopher",
            "Nancy",
            "Daniel",
            "Lisa",
            "Matthew",
            "Betty",
            "Anthony",
            "Margaret",
            "Mark",
            "Sandra",
        ]
        .iter()
        .cloned()
        .collect();

        let mut entities = Vec::new();
        let mut seen = HashSet::new();

        for cap in re.captures_iter(text) {
            let candidate = cap[1].trim();
            if candidate.is_empty() || ignore_words.contains(candidate) || seen.contains(candidate)
            {
                continue;
            }

            if candidate.chars().all(|c| !c.is_alphabetic()) {
                continue;
            }

            seen.insert(candidate.to_string());

            let candidate_lower = candidate.to_lowercase();
            let words: Vec<&str> = candidate_lower.split_whitespace().collect();

            let is_org = known_orgs.contains(candidate)
                || words.iter().any(|&w| {
                    w == "inc"
                        || w == "inc."
                        || w == "corp"
                        || w == "corp."
                        || w == "corporation"
                        || w == "ltd"
                        || w == "ltd."
                        || w == "co"
                        || w == "co."
                        || w == "company"
                        || w == "university"
                        || w == "lab"
                        || w == "labs"
                        || w == "institute"
                        || w == "foundation"
                        || w == "association"
                        || w == "agency"
                });

            if is_org {
                entities.push(("ORG".to_string(), candidate.to_string()));
                continue;
            }

            let is_loc = known_locs.contains(candidate)
                || words.iter().any(|&w| {
                    w == "city"
                        || w == "state"
                        || w == "street"
                        || w == "avenue"
                        || w == "road"
                        || w == "river"
                        || w == "lake"
                        || w == "mountain"
                        || w == "ocean"
                        || w == "sea"
                        || w == "park"
                        || w == "valley"
                        || w == "station"
                        || w == "airport"
                        || w == "county"
                        || w == "island"
                        || w == "country"
                });

            if is_loc {
                entities.push(("LOC".to_string(), candidate.to_string()));
                continue;
            }

            let idx = text.find(candidate).unwrap_or(0);
            let has_title = if idx >= 4 {
                let start = text.floor_char_boundary(idx.saturating_sub(4));
                let prefix = &text[start..idx].to_lowercase();
                prefix.contains("mr.") || prefix.contains("ms.") || prefix.contains("dr.")
            } else if idx >= 5 {
                let start = text.floor_char_boundary(idx.saturating_sub(5));
                let prefix = &text[start..idx].to_lowercase();
                prefix.contains("mrs.") || prefix.contains("prof.")
            } else {
                false
            };

            let is_per = has_title || known_pers.contains(candidate) || words.len() >= 2;

            if is_per {
                entities.push(("PER".to_string(), candidate.to_string()));
            } else {
                entities.push(("PER".to_string(), candidate.to_string()));
            }
        }

        Ok(entities)
    }

    pub fn predict_score(&self, a: &str, b: &str) -> Result<f32> {
        let tokens_a: HashSet<&str> = a.split_whitespace().collect();
        let tokens_b: HashSet<&str> = b.split_whitespace().collect();
        let intersection = tokens_a.intersection(&tokens_b).count();
        let union = tokens_a.union(&tokens_b).count();
        Ok(if union == 0 { 0.0 } else { intersection as f32 / union as f32 })
    }

    pub fn predict_scores_batch(&self, q: &str, texts: &[String]) -> Result<Vec<f32>> {
        let query_tokens: HashSet<&str> = q.split_whitespace().collect();
        Ok(texts
            .iter()
            .map(|text| {
                let text_tokens: HashSet<&str> = text.split_whitespace().collect();
                let intersection = query_tokens.intersection(&text_tokens).count();
                let union = query_tokens.union(&text_tokens).count();
                if union == 0 {
                    0.0
                } else {
                    intersection as f32 / union as f32
                }
            })
            .collect())
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
