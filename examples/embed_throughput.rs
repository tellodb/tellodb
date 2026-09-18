//! Embedding throughput probe: texts/sec by text length and batching strategy.
//!
//! `cargo run --profile fastrelease --example embed_throughput`
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::time::Instant;

fn words(n: usize) -> String {
    (0..n)
        .map(|i| ["memory", "engine", "seattle", "moved", "coffee", "project"][i % 6])
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() -> anyhow::Result<()> {
    let mut options = TextInitOptions::default();
    options.model_name = EmbeddingModel::BGESmallENV15;
    if let Ok(dir) = std::env::var("HF_HOME") {
        options.cache_dir = dir.into();
    }
    let mut model = TextEmbedding::try_new(options)?;
    model.embed(vec![words(10)], None)?;

    let short: Vec<String> = (0..64).map(|_| words(15)).collect();
    let long: Vec<String> = (0..64).map(|_| words(400)).collect();
    let mut mixed: Vec<String> = (0..8).map(|_| words(400)).collect();
    mixed.extend((0..63).map(|_| words(15)));

    let run = |name: &str,
               model: &mut TextEmbedding,
               texts: &[String],
               batch: Option<usize>,
               sort: bool|
     -> anyhow::Result<()> {
        let mut t: Vec<&String> = texts.iter().collect();
        if sort {
            t.sort_by_key(|s| s.len());
        }
        let start = Instant::now();
        model.embed(t, batch)?;
        let secs = start.elapsed().as_secs_f64();
        println!(
            "{name:<32} n={:<3} {:>8.1} ms {:>8.1} texts/s",
            texts.len(),
            secs * 1e3,
            texts.len() as f64 / secs
        );
        Ok(())
    };
    run("short x64, one batch", &mut model, &short, None, false)?;
    run("long x64, one batch", &mut model, &long, None, false)?;
    run("long x64, batch 16", &mut model, &long, Some(16), false)?;
    run("mixed x71, one batch", &mut model, &mixed, None, false)?;
    run("mixed x71, sorted, batch 16", &mut model, &mixed, Some(16), true)?;
    Ok(())
}
