pub mod analytics;
pub mod api;
pub mod fts;
pub mod graph;
pub mod graph_inference;
pub mod lifecycle;
pub mod metrics;
pub mod ml;
pub mod platform;
pub mod retrieval;
pub mod runtime_paths;
pub mod semantic;
pub mod storage;
pub mod vector_index;

pub fn init_tracing_subscriber() {
    use tracing_subscriber::{filter::EnvFilter, fmt, prelude::*, Registry};

    let env_filter =
        EnvFilter::builder().with_default_directive(tracing::Level::INFO.into()).from_env_lossy();

    let fmt_layer =
        fmt::layer().with_target(true).with_thread_ids(true).with_file(true).with_line_number(true);

    let subscriber = Registry::default().with(env_filter).with(fmt_layer);

    let _ = tracing::subscriber::set_global_default(subscriber);
}
