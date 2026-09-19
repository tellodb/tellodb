pub mod analytics;
pub mod api;
pub mod db;
pub mod doctor;
pub mod engine;
pub mod extract;
pub mod features;
pub mod fts;
pub mod gliner;
pub mod graph;
pub mod heuristics;
pub mod lifecycle;
pub mod mcp_stdio;
pub mod metrics;
pub mod ml;
pub mod platform;
pub mod retrieval;
pub mod runtime_paths;
pub mod semantic;
pub mod storage;
pub mod vector_index;

pub fn init_tracing_subscriber() {
    init_tracing_with_default_level(tracing::Level::INFO);
}

/// Logging to stderr at `level` unless `RUST_LOG` says otherwise.
pub fn init_tracing_with_default_level(level: tracing::Level) {
    use tracing_subscriber::{filter::EnvFilter, fmt, prelude::*, Registry};

    let env_filter = EnvFilter::builder().with_default_directive(level.into()).from_env_lossy();

    // Logs go to stderr so stdout stays free for command output and the
    // stdio MCP protocol.
    let fmt_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true);

    let subscriber = Registry::default().with(env_filter).with(fmt_layer);

    let _ = tracing::subscriber::set_global_default(subscriber);
}
