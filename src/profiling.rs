//! Opt-in structured profiling. Enable with `RUST_LOG=tellodb::profile=debug`.
//! Durations are inclusive wall times; nested and parallel stages overlap.
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Emits paired events even on an early return or panic. `exit` is deliberately
/// not labelled success: callers emit outcomes separately.
pub(crate) struct Timer {
    operation: &'static str,
    call_id: u64,
    start: Option<Instant>,
    span: tracing::Span,
}

impl Timer {
    pub(crate) fn new(operation: &'static str) -> Self {
        let enabled = tracing::enabled!(target: "tellodb::profile", tracing::Level::DEBUG);
        let call_id = if enabled { next_id() } else { 0 };
        if enabled {
            tracing::debug!(target: "tellodb::profile", event = "stage_enter", operation, call_id);
        }
        Self {
            operation,
            call_id,
            start: enabled.then(Instant::now),
            span: if enabled { tracing::Span::current() } else { tracing::Span::none() },
        }
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed_us = start.elapsed().as_micros() as u64;
            let _entered = self.span.enter();
            tracing::debug!(target: "tellodb::profile", event = "stage_exit",
                operation = self.operation, call_id = self.call_id, elapsed_us,
                panicking = std::thread::panicking());
        }
    }
}

/// Preserves request correlation and separates blocking-pool queue time from
/// execution time. Callers retain Tokio's normal JoinHandle/error behaviour.
#[track_caller]
pub(crate) fn spawn_blocking<F, R>(f: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let span = tracing::Span::current();
    let dispatch = tracing::dispatcher::get_default(Clone::clone);
    let queued =
        tracing::enabled!(target: "tellodb::profile", tracing::Level::DEBUG).then(Instant::now);
    let location = std::panic::Location::caller();
    tokio::task::spawn_blocking(move || {
        tracing::dispatcher::with_default(&dispatch, || {
            let _entered = span.enter();
            if let Some(queued) = queued {
                tracing::debug!(target: "tellodb::profile", event = "blocking_queue",
                file = location.file(), line = location.line(),
                elapsed_us = queued.elapsed().as_micros() as u64);
            }
            let _profile = Timer::new("runtime.blocking_execute");
            f()
        })
    })
}

pub(crate) fn fallback<T: Default, E: std::fmt::Debug>(stage: &'static str, error: E) -> T {
    tracing::warn!(target: "tellodb::profile", event = "quality_fallback", stage,
        error = ?error, "stage failed; retaining existing empty fallback behaviour");
    T::default()
}

pub(crate) fn spawn_scoped<'scope, 'env, F, T>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    f: F,
) -> std::thread::ScopedJoinHandle<'scope, T>
where
    F: FnOnce() -> T + Send + 'scope,
    T: Send + 'scope,
{
    let span = tracing::Span::current();
    let dispatch = tracing::dispatcher::get_default(Clone::clone);
    scope.spawn(move || {
        tracing::dispatcher::with_default(&dispatch, || {
            let _entered = span.enter();
            let _profile = Timer::new("runtime.scoped_worker");
            f()
        })
    })
}

pub(crate) fn spawn<F>(future: F) -> tokio::task::JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    use tracing::Instrument;
    tokio::spawn(future.in_current_span())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);
    impl Write for Buffer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Buffer {
        type Writer = Self;
        fn make_writer(&'a self) -> Self {
            self.clone()
        }
    }

    #[tokio::test]
    async fn worker_logs_keep_request_id_and_early_returns_close_timers() {
        let buffer = Buffer(Arc::new(Mutex::new(Vec::new())));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(buffer.clone())
            .finish();
        let dispatch = tracing::Dispatch::new(subscriber);
        let handle = tracing::dispatcher::with_default(&dispatch, || {
            let span =
                tracing::debug_span!(target: "tellodb::profile", "request", request_id = 42u64);
            let _entered = span.enter();
            spawn_blocking(|| {
                std::thread::scope(|scope| {
                    spawn_scoped(scope, || {
                        let _timer = Timer::new("test.early_error");
                        Err::<(), _>("expected error")
                    })
                    .join()
                    .unwrap()
                })
            })
        });
        assert_eq!(handle.await.unwrap(), Err("expected error"));
        let bytes = buffer.0.lock().unwrap().clone();
        let records: Vec<serde_json::Value> = String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert!(records.iter().any(|row| row["fields"]["event"] == "blocking_queue"));
        let exits: Vec<_> =
            records.iter().filter(|row| row["fields"]["event"] == "stage_exit").collect();
        assert!(exits.iter().any(|row| row["fields"]["operation"] == "test.early_error"));
        for row in exits {
            assert!(row["spans"].as_array().unwrap().iter().any(|span| span["request_id"] == 42));
            assert!(row["fields"].get("elapsed_us").is_some());
        }
    }
}
