use prometheus::{Encoder, Histogram, HistogramOpts, IntCounter, Registry, TextEncoder};
use std::sync::OnceLock;

fn instance() -> &'static Metrics {
    static METRICS: OnceLock<Metrics> = OnceLock::new();
    METRICS.get_or_init(Metrics::new)
}

struct Metrics {
    registry: Registry,
    ingest_total: IntCounter,
    query_total: IntCounter,
    query_duration: Histogram,
}

impl Metrics {
    fn new() -> Self {
        let registry = Registry::new();
        let ingest_total =
            IntCounter::new("tellodb_ingest_total", "Total ingest requests").expect("failed to create ingest_total metric");
        let query_total = IntCounter::new("tellodb_query_total", "Total query requests").expect("failed to create query_total metric");
        let query_duration = Histogram::with_opts(
            HistogramOpts::new("tellodb_query_duration_seconds", "Query duration in seconds")
                .buckets(vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
        )
        .expect("failed to create query_duration histogram");
        registry.register(Box::new(ingest_total.clone())).expect("failed to register ingest_total metric");
        registry.register(Box::new(query_total.clone())).expect("failed to register query_total metric");
        registry.register(Box::new(query_duration.clone())).expect("failed to register query_duration metric");
        Self { registry, ingest_total, query_total, query_duration }
    }
}

pub fn increment_ingest() {
    instance().ingest_total.inc();
}

pub fn increment_query() {
    instance().query_total.inc();
}

pub fn observe_query_duration(secs: f64) {
    instance().query_duration.observe(secs);
}

pub fn gather() -> Vec<prometheus::proto::MetricFamily> {
    instance().registry.gather()
}

pub fn render() -> String {
    let encoder = TextEncoder::new();
    let metric_families = gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap_or_default();
    String::from_utf8(buffer).unwrap_or_default()
}
