//! Runs the GLiNER extractor over a few sentences and prints the spans.
//!
//! `TELLODB_EXTRACTOR_MODEL_DIR=<dir> cargo run --profile fastrelease --example gliner_probe`
use tellodb::gliner::GlinerModel;

fn main() -> anyhow::Result<()> {
    let dir = std::env::var("TELLODB_EXTRACTOR_MODEL_DIR")
        .unwrap_or_else(|_| format!("{}/.cache/tellodb/models/gliner_small", env!("HOME")));
    let model = GlinerModel::load(std::path::Path::new(&dir))?;
    let labels: Vec<String> = ["city of residence", "employer", "job title", "person", "pet"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    for text in ["I live in Austin.", "My home city is Austin.", "I now live in Seattle."] {
        let start = std::time::Instant::now();
        let spans = model.predict(text, &labels, 0.3)?;
        println!("\n{text}  ({} ms)", start.elapsed().as_millis());
        for span in spans {
            println!("   {:<20} {:<28} {:.3}", span.label, format!("{:?}", span.text), span.score);
        }
    }
    Ok(())
}
