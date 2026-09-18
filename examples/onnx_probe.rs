//! Prints the input/output signature of an ONNX model.
//!
//! `cargo run --profile fastrelease --example onnx_probe -- path/to/model.onnx`
fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: onnx_probe <model.onnx>");
    let session = ort::session::Session::builder()?.commit_from_file(&path)?;
    println!("inputs:");
    for input in session.inputs() {
        println!("  {} : {:?}", input.name(), input.dtype());
    }
    println!("outputs:");
    for output in session.outputs() {
        println!("  {} : {:?}", output.name(), output.dtype());
    }
    Ok(())
}
