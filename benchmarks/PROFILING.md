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
`tellodb::profile=debug` (unless `RUST_LOG` is explicitly set). Set
`TELLODB_LOG_FORMAT=json RUST_LOG=warn,tellodb::profile=debug` when running the
engine directly. Default engine logging retains its previous text format.

Over 500 function scopes cover ingestion preparation, extraction, embedding,
cache access, SQLite operations, FTS, vector indexing, routing, retrieval,
fusion, scoring, proofs, authentication, and background work. Timing scopes
emit `stage_enter` and `stage_exit` with matching `call_id`, inclusive
microseconds, and panic state. Exit means the scope ended, not that it succeeded.

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
