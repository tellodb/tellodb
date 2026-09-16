use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tempfile::tempdir;

use tellodb::api::ingest_utils::{infer_fact_key, split_atomic_claims};
use tellodb::retrieval::rrf_fuse;
use tellodb::storage::TenantStore;
use tellodb::vector_index::VectorIndex;

// Deterministic PRNG for generating benchmark data
struct BenchRng(u64);

impl BenchRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    fn random_vector(&mut self, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| self.next_f32() - 0.5).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

#[derive(Clone, Copy, PartialEq)]
struct ScoredItem {
    score: f32,
    id: u64,
}

impl Eq for ScoredItem {}

impl Ord for ScoredItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ScoredItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn brute_force_cosine_search(
    query: &[f32],
    vectors: &[(u64, String, Vec<f32>)],
    scoped_entity: Option<&str>,
    top_k: usize,
) -> Vec<(u64, f32)> {
    let mut heap: BinaryHeap<std::cmp::Reverse<ScoredItem>> = BinaryHeap::with_capacity(top_k + 1);

    for (id, entity, vec) in vectors {
        if let Some(target_eid) = scoped_entity {
            if entity != target_eid {
                continue;
            }
        }
        let score: f32 = query.iter().zip(vec.iter()).map(|(&a, &b)| a * b).sum();
        let item = ScoredItem { score, id: *id };
        heap.push(std::cmp::Reverse(item));
        if heap.len() > top_k {
            heap.pop();
        }
    }

    let mut results = Vec::with_capacity(heap.len());
    while let Some(std::cmp::Reverse(item)) = heap.pop() {
        results.push((item.id, item.score));
    }
    results.reverse();
    results
}

fn bench_vector_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_index");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let dim = 384;
    let mut rng = BenchRng::new(12345);

    for &total_vectors in &[10_000, 100_000] {
        let index = VectorIndex::new(dim, total_vectors + 5000, 16, 128, 256, None)
            .expect("Failed to create VectorIndex");

        // Populate index distributed across entities
        let entity_counts = [1, 100, 10_000];
        let mut entity_id_pool = Vec::new();
        for &num_entities in &entity_counts {
            for e in 0..num_entities {
                entity_id_pool.push(format!("entity_{:05}", e));
            }
        }

        let mut items = Vec::with_capacity(total_vectors);
        for i in 0..total_vectors {
            let entity_id = format!("entity_{:05}", i % 100);
            let vec = rng.random_vector(dim);
            items.push((i as u64, entity_id, vec));
        }

        // Insert in batches of 256
        for chunk in items.chunks(256) {
            let batch: Vec<(u64, Vec<f32>)> =
                chunk.iter().map(|(id, _, v)| (*id, v.clone())).collect();
            index.insert_batch(&chunk[0].1, &batch).expect("Failed to insert batch");
        }

        // Benchmark insert_batch
        let batch_to_insert: Vec<(u64, Vec<f32>)> =
            (0..100).map(|i| ((total_vectors + i) as u64, rng.random_vector(dim))).collect();
        group.bench_with_input(
            BenchmarkId::new("insert_batch_100", total_vectors),
            &total_vectors,
            |b, _| {
                b.iter(|| {
                    let _ = index.insert_batch("bench_entity", black_box(&batch_to_insert));
                });
            },
        );

        // Benchmark search: unscoped, and scoped with 1, 100, and 10k entities
        let query = rng.random_vector(dim);

        group.bench_with_input(
            BenchmarkId::new("search_unscoped", total_vectors),
            &total_vectors,
            |b, _| {
                b.iter(|| {
                    let res = index.search(None, black_box(&query), 20).unwrap();
                    black_box(res);
                });
            },
        );

        for &num_entities in &[1, 100, 10_000] {
            if num_entities > total_vectors {
                continue;
            }
            let target_eid = "entity_00000";
            let bench_id = format!("search_scoped_{}_entities", num_entities);
            group.bench_with_input(
                BenchmarkId::new(bench_id, total_vectors),
                &total_vectors,
                |b, _| {
                    b.iter(|| {
                        let res = index.search(Some(target_eid), black_box(&query), 20).unwrap();
                        black_box(res);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_flat_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_scan");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let dim = 384;
    let mut rng = BenchRng::new(54321);

    for &total_vectors in &[10_000, 100_000] {
        let mut vectors = Vec::with_capacity(total_vectors);
        for i in 0..total_vectors {
            let entity_id = format!("entity_{:05}", i % 100);
            let vec = rng.random_vector(dim);
            vectors.push((i as u64, entity_id, vec));
        }

        let query = rng.random_vector(dim);

        group.bench_with_input(
            BenchmarkId::new("exact_cosine_unscoped", total_vectors),
            &total_vectors,
            |b, _| {
                b.iter(|| {
                    let res = brute_force_cosine_search(black_box(&query), &vectors, None, 20);
                    black_box(res);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("exact_cosine_scoped_100_entities", total_vectors),
            &total_vectors,
            |b, _| {
                b.iter(|| {
                    let res = brute_force_cosine_search(
                        black_box(&query),
                        &vectors,
                        Some("entity_00000"),
                        20,
                    );
                    black_box(res);
                });
            },
        );
    }
    group.finish();
}

fn bench_fts(c: &mut Criterion) {
    let mut group = c.benchmark_group("fts");
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(10);

    let dir = tempdir().expect("Failed to create tempdir");
    let db_path = dir.path().join("tenant.db");
    let store = TenantStore::new(&db_path).expect("Failed to create TenantStore");

    // Populate 10,000 synthetic rows
    let vocab = [
        "quantum",
        "database",
        "temporal",
        "memory",
        "inference",
        "engine",
        "vector",
        "search",
        "retrieval",
        "ranking",
        "semantic",
        "graph",
        "session",
        "context",
        "latency",
        "throughput",
    ];

    let mut rows = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        let mem_id = format!("mem_{:05}", i);
        let entity_id = format!("entity_{:03}", i % 50);
        let content = format!(
            "Memory record {} discusses {} {} algorithms and {} optimizations for temporal retrieval.",
            i,
            vocab[i % vocab.len()],
            vocab[(i + 3) % vocab.len()],
            vocab[(i + 7) % vocab.len()]
        );
        rows.push((mem_id, entity_id, content));
    }

    for chunk in rows.chunks(1000) {
        store.fts_index_batch(chunk).expect("Failed to batch index FTS");
    }

    group.bench_function("fts_search_unscoped_10k", |b| {
        b.iter(|| {
            let res = store.fts_search(black_box("quantum database algorithms"), 20, None).unwrap();
            black_box(res);
        });
    });

    group.bench_function("fts_search_scoped_10k", |b| {
        b.iter(|| {
            let res = store
                .fts_search(black_box("quantum database algorithms"), 20, Some("entity_010"))
                .unwrap();
            black_box(res);
        });
    });

    group.finish();
}

fn bench_rrf_fuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrf_fuse");
    group.measurement_time(Duration::from_secs(2));

    // 5 result lists of 200 items each
    let lanes: Vec<Vec<(String, f32)>> = (0..5)
        .map(|lane_idx| {
            (0..200)
                .map(|item_idx| {
                    let id = format!("item_{:04}", (lane_idx * 50 + item_idx) % 300);
                    let score = 1.0 / (item_idx as f32 + 1.0);
                    (id, score)
                })
                .collect()
        })
        .collect();

    group.bench_function("rrf_fuse_5_lanes_200_items", |b| {
        b.iter(|| {
            let res = rrf_fuse(black_box(&lanes), 60.0);
            black_box(res);
        });
    });

    group.finish();
}

fn bench_fact_rules(c: &mut Criterion) {
    let mut group = c.benchmark_group("fact_rules");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let templates = [
        "I moved to London recently.",
        "I work at Acme Corp as a senior architect.",
        "I have two children and one dog at home.",
        "My favorite beverage is pour-over coffee; I also love tea.",
        "I bought a new car yesterday for $25000.",
        "I got a certificate in machine learning from Coursera.",
        "I research AI and temporal knowledge graphs.",
        "My nickname is Bob and people call me that often.",
        "The quick brown fox jumps over the lazy dog.",
        "General meeting scheduled for tomorrow at 10 AM.",
    ];

    let mut sentences = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        sentences.push(templates[i % templates.len()].to_string());
    }

    group.bench_function("infer_fact_key_10k_sentences", |b| {
        b.iter(|| {
            for sentence in &sentences {
                let res = infer_fact_key(black_box(sentence));
                black_box(res);
            }
        });
    });

    group.bench_function("split_atomic_claims_10k_sentences", |b| {
        b.iter(|| {
            for sentence in &sentences {
                let res = split_atomic_claims(black_box(sentence));
                black_box(res);
            }
        });
    });

    group.finish();
}

fn bench_models(c: &mut Criterion) {
    if std::env::var("TELLODB_BENCH_MODELS").ok().as_deref() != Some("1") {
        eprintln!("Skipping bench_models: TELLODB_BENCH_MODELS=1 is not set.");
        return;
    }

    let mut group = c.benchmark_group("models");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    let semantic = rt.block_on(async {
        tellodb::semantic::SemanticInference::new().await.expect("Failed to init SemanticInference")
    });

    let sample_texts: Vec<String> = (0..32)
        .map(|i| {
            format!(
                "This is test document number {} containing various semantic claims about agent memory.",
                i
            )
        })
        .collect();
    let text_refs: Vec<&str> = sample_texts.iter().map(|s| s.as_str()).collect();

    for &batch_size in &[1, 8, 32] {
        let input_slice = &text_refs[..batch_size];
        group.bench_with_input(BenchmarkId::new("embed_batch", batch_size), &batch_size, |b, _| {
            b.iter(|| {
                let embs = semantic.embed_batch(black_box(input_slice));
                black_box(embs);
            });
        });
    }

    group.bench_function("rerank_32_pairs", |b| {
        let query = "What is agent memory architecture?";
        b.iter(|| {
            let scores =
                semantic.predict_scores_batch(black_box(query), black_box(&sample_texts)).unwrap();
            black_box(scores);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_vector_index,
    bench_flat_scan,
    bench_fts,
    bench_rrf_fuse,
    bench_fact_rules,
    bench_models
);
criterion_main!(benches);
