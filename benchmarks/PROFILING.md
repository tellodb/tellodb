# Profiling benchmark runs

Run the full dev ablation matrix with structured diagnostics:

```bash
source .env.tellodb
PROFILE_LOGS=1 RUNS_OVERRIDE=1 CONCURRENCY_OVERRIDE=1 \
  bash benchmarks/run_ablation.sh --tier dev
```

The working databases default to `$HOME/.cache/tellodb/benchmark-work` on disk.
Each finished arm deletes its scratch database. Original run records and logs
remain in `benchmarks/runs/<timestamp>_dev_ablation/`.

Profiling is opt-in. `PROFILE_LOGS=1` sets JSON logging and enables
`tellodb::profile=debug`, appending that target to any existing `RUST_LOG`. Set
`TELLODB_LOG_FORMAT=json RUST_LOG=warn,tellodb::profile=debug` when running the
engine directly. Default engine logging retains its previous text format.

Over 500 function scopes cover ingestion preparation, extraction, embedding,
cache access, SQLite operations, FTS, vector indexing, routing, retrieval,
fusion, scoring, proofs, authentication, and background work. Timing scopes
emit `stage_enter` and `stage_exit` with matching `call_id`, inclusive
microseconds, and panic state. Exit means the scope ended, not that it succeeded.
Every call contributes to `stage_aggregate` counts, total/max durations and
histograms, flushed in process-wide windows after HTTP requests. Windows can
include background work and concurrent requests and have no request parent.
Fast individual calls are sampled every 100 calls by default; every call taking
at least 1 ms and every panicking call emits an exit event. Set
`TELLODB_PROFILE_SAMPLE_EVERY=1` for every individual event (very large logs).
The summary uses aggregate windows rather than double-counting sampled exits.

HTTP spans carry a process-local `request_id`, returned as `x-tm-request-id`.
Blocking-pool and scoped workers inherit it. `blocking_queue` measures delay
before execution; dedicated timers measure GPU permit and executor lock waits,
inference, SQLite `BEGIN IMMEDIATE`, commit, and connection acquisition.
Database transaction helpers have individual timers.

Quality events include embedding cache hits/misses, derived structure counts,
dedup skips, retrieval candidate counts, rerank decisions, filtering reasons,
candidate score components, final ranks, proof verification, conflicts, and
abstention. Formerly silent optional lookup failures emit `quality_fallback`
while keeping their existing fallback behavior. The evaluator emits
`question_quality` with request ID, question ID, expected/retrieved session IDs,
recall and nDCG to connect benchmark failures to engine traces. New diagnostic
events omit raw memory/query text and credentials.

Artifacts:

- `engine_<arm>.log`: JSON engine events.
- `resources_<arm>.jsonl`: engine RSS, I/O, available RAM/swap, scratch/WAL size,
  disk free space, GPU utilization/memory/temperature/power sampled every 15s.
- `PROFILE_<arm>.md`: operation counts, inclusive total/mean/max durations,
  histogram percentile upper bounds, event counts, warning/error counts.
- `PROFILE_CONFIG.txt`: logging, storage, concurrency and seed settings.
- Per-dataset JSON records and final `REPORT.md`: normal benchmark output.

Generate a summary while a run is active or after interruption:

```bash
python3 benchmarks/profile_logs.py summarize path/to/engine_baseline.log /tmp/profile.md
```

Inclusive durations overlap across nested calls and parallel workers; summing
them does not give end-to-end time. Percentiles use bounded-memory power-of-two
microsecond buckets and report their upper bounds. Heavy diagnostic logging
affects throughput and latency; compare profiling runs with each other. A fresh
profiled baseline is preferable to reusing earlier unprofiled latency results.

The `router_batch` event reports input `updates` and `persisted_sessions` after
coalescing session-router writes within each transaction. Source fragments and
intermediate return values retain their original order and deletion provenance.
Schema v8 normalizes legacy router FTS row IDs once on opening a database; this
allows replacement and deletion by stable row ID without per-update index scans.

Runs before the card-query fix can contain `retrieve.optional_lookup` warnings
for ambiguous `created_at_ms` in the card search join. Those runs effectively
had an empty memory-card retrieval lane. Use a fresh baseline when comparing
card ablations after the fix; do not interpret those older runs as evidence that
cards provide no quality benefit. Ranking and temporal-gate changes still need
controlled experiments; the storage fixes alone do not establish a recall gain.
