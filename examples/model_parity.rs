//! Checks that loading the embedder from `TELLODB_MODEL_DIR` gives the same
//! vectors as the fastembed download path.
//!
//! `TELLODB_MODEL_DIR=<snapshot> cargo run --profile fastrelease --example model_parity`
use tellodb::semantic::SemanticInference;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let texts = ["I just moved to Denver for a new job.", "where do I live"];
    let dir = std::env::var("TELLODB_MODEL_DIR")?;
    let local = SemanticInference::with_cache_path(None).await?.embed_texts(&texts)?;
    std::env::remove_var("TELLODB_MODEL_DIR");
    let downloaded = SemanticInference::with_cache_path(None).await?.embed_texts(&texts)?;
    for (i, (a, b)) in local.iter().zip(&downloaded).enumerate() {
        let max_diff = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "text {i}: max |diff| = {max_diff:.2e}, norms local {norm_a:.4} downloaded {norm_b:.4}"
        );
    }
    println!("model dir: {dir}");
    Ok(())
}
