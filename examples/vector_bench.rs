//! ANN recall and cost per vector configuration (roadmap WP6).
//!
//! For each quantization mode and segment size, compares scoped top-10 search
//! against exact f32 search and reports recall@10, latency and memory.
//!
//! `cargo run --profile fastrelease --example vector_bench -- [dims] [queries]`
//!
//! Vectors are random mixtures of 64 cluster centres, so near neighbours
//! exist (uniform random vectors make every method look alike).
use std::time::Instant;
use tellodb::vector_index::{Quantization, VectorConfig, VectorIndex};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
    fn vector(&mut self, dims: usize) -> Vec<f32> {
        (0..dims).map(|_| self.next()).collect()
    }
}

fn clustered(rng: &mut Rng, centres: &[Vec<f32>], dims: usize) -> Vec<f32> {
    let c = &centres[((rng.next() + 0.5) * centres.len() as f32) as usize % centres.len()];
    let noise = rng.vector(dims);
    let v: Vec<f32> = c.iter().zip(noise).map(|(a, n)| a + 0.35 * n).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / norm).collect()
}

fn percentile(samples: &mut [f64], p: f64) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[((samples.len() as f64 - 1.0) * p).round() as usize]
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dims: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(384);
    let queries: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(200);
    let mut rng = Rng(7);
    let centres: Vec<Vec<f32>> = (0..64).map(|_| rng.vector(dims)).collect();

    println!("dims={dims} queries={queries}");
    println!("| segment | layout | quant | recall@10 | p50 µs | p99 µs | bytes/vector | load ms |");
    println!("|---|---|---|---|---|---|---|---|");
    for &segment in &[1_000usize, 10_000, 100_000] {
        let rows: Vec<(u64, Vec<f32>)> =
            (0..segment).map(|i| (i as u64, clustered(&mut rng, &centres, dims))).collect();
        let qs: Vec<Vec<f32>> = (0..queries).map(|_| clustered(&mut rng, &centres, dims)).collect();

        let exact = VectorIndex::in_memory(VectorConfig {
            flat_threshold: usize::MAX,
            ..VectorConfig::new(dims)
        });
        exact.insert_batch("e", &rows)?;
        let truth: Vec<Vec<u64>> = qs
            .iter()
            .map(|q| exact.search(Some("e"), q, 10).map(|h| h.iter().map(|x| x.0).collect()))
            .collect::<anyhow::Result<_>>()?;
        drop(exact);

        for (layout, flat_threshold) in [("flat", usize::MAX), ("hnsw", 0)] {
            if layout == "flat" && segment > 100_000 {
                continue;
            }
            for quantization in
                [Quantization::F32, Quantization::F16, Quantization::I8, Quantization::Binary]
            {
                // Inserting into an entity that was never searched only records
                // the rows, so the first `len` measures building the segment.
                let index = VectorIndex::in_memory(VectorConfig {
                    quantization,
                    flat_threshold,
                    ..VectorConfig::new(dims)
                });
                index.insert_batch("e", &rows)?;
                let load = Instant::now();
                index.len(Some("e"))?;
                let load_ms = load.elapsed().as_secs_f64() * 1e3;

                let mut latencies = Vec::with_capacity(queries);
                let mut found = 0usize;
                for (q, truth) in qs.iter().zip(&truth) {
                    let t = Instant::now();
                    let hits = index.search(Some("e"), q, 10)?;
                    latencies.push(t.elapsed().as_secs_f64() * 1e6);
                    found += hits.iter().filter(|h| truth.contains(&h.0)).count();
                }
                let bytes = index.loaded_bytes() as f64 / segment as f64;
                println!(
                    "| {segment} | {layout} | {} | {:.3} | {:.0} | {:.0} | {:.0} | {:.1} |",
                    quantization.name(),
                    found as f64 / (10 * queries) as f64,
                    percentile(&mut latencies, 0.5),
                    percentile(&mut latencies, 0.99),
                    bytes,
                    load_ms,
                );
            }
        }
    }
    Ok(())
}
