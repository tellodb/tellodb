# Tellodb engine roadmap v2

Status: WP2, WP3 (tooling), WP4, WP6, WP7 and most of WP8 implemented on
`feat/wp0-measurement-harness`; see "Implementation status" below. WP1 and WP9
(GPU runs) are the owner's; WP5 is not started.

## Implementation status (2026-09-18, laptop smoke numbers only)

Quality figures below come from the 60-question synthetic set and LongMemEval
dev `--limit 10` on an M2 Pro. They are for direction only; the dev-split
numbers still have to come from the GPU box (WP1, WP9).

**Landed**

- **WP2 memory representation.** Explicit `session_id` / `turn_index` / `role`
  on payloads, observations and the `memories` table; `TELLODB_EMBED_TEXT`
  (`legacy` | `turn` | `context`, default `context`) with
  `TELLODB_CONTEXT_WINDOW`; context text is built from stored turns, so
  embeddings no longer depend on client batching (regression test:
  `context_texts_do_not_depend_on_batching`). Evaluator gained
  `--client-context window|off`. BGE query instruction is applied to queries
  only. Companions and identity now work for opaque (UUID) memory ids.
- **WP3 ablations.** `TELLODB_DISABLE` switches for 17 derived structures
  (creation *and* the matching retrieval lane), per-structure counts in
  `x-tm-ingest-counts` and run records, database growth per memory,
  `benchmarks/run_matrix.sh` with `run_ablation.sh` / `run_rerank_sweep.sh`
  wrappers, and `rust_evaluator ablation-report` (paired bootstrap deltas).
  Six structures were **deleted** rather than measured, because nothing read
  them: temporal events, shadow questions, facet postings, mem cells, mem
  scenes and profile facts. Also removed: card relations, memory artifacts and
  artifact versions (written, never read), the ledger-turn table (its readers
  now use `memories`, so evidence windows actually work), and a dead
  alias/entity-resolver block that could never run.
- **WP4 graph latency.** Level-synchronous BFS across all seeds with batched
  neighbour queries (about 6 SQL queries per stage instead of ~1,200),
  `TELLODB_GRAPH_SEEDS` / `TELLODB_GRAPH_MAX_DEPTH`, and sub-stage timings
  (`x-tm-graph-{links,edges,entities,lookup}-us`). Synthetic: query p50
  115 ms → 10 ms, p95 281 ms → 15 ms, graph stage 107 ms → 3 ms, with
  identical recall and nDCG.
- **WP6 vector segments.** Per-entity segments loaded lazily from
  `vector_lookup.embedding` (SQLite is the only source of truth: no
  checkpoints, no rebuild path, no global lock), exact SIMD scans up to
  `TELLODB_FLAT_THRESHOLD`, per-entity HNSW above it, and
  `TELLODB_VECTOR_QUANT=f32|f16|i8|binary` with f32 rescoring of the top
  `k × TELLODB_RESCORE_FACTOR`. `examples/vector_bench.rs` reports recall@10
  against exact search: at 384 dims, flat i8 holds recall 1.000 at 2.2–2.5x
  the speed of f32 and 3.9x less memory; binary collapses beyond ~10k vectors;
  f16 is exact but slower than f32 on this CPU. Default stays f32 because the
  query path asks for 240+ candidates, which makes SQLite rescoring the
  dominant cost. i8 and f32 give identical end-to-end results.
- **WP7 rerank cascade.** `TELLODB_RERANK_POLICY=heuristic|always|gate` with
  `TELLODB_RERANK_MARGIN`, `TELLODB_RERANK_TOP` and `TELLODB_RERANK_MODEL`;
  per-question decisions recorded (`x-tm-rerank-reason`, `rerank_reasons` in
  run records). The gate reranked 48% of synthetic queries.
- **WP8 library-first.** `tellodb::db::{Db, Engine, Memory, Query}` embedded
  API (no HTTP, no API key), engine construction moved into the library, a CLI
  (`serve`, `mcp`, `doctor`, `ingest`, `query`), a spec-compliant stdio MCP
  server with `remember` / `recall` / `current_fact`, offline model loading
  (`TELLODB_MODEL_DIR`, verified bit-identical to the download path) and a
  `bundled-models` feature, a C ABI crate (`crates/tellodb-ffi`) and a
  ctypes-only Python binding.

**Bugs found and fixed while measuring**

- Cross-encoder logits were used as `base_score` for reranked candidates while
  everything else used fused RRF scores, so reranking *hurt*: on synthetic,
  always-rerank scored 48.1 recall against 79.6 without it. Scores now come
  from the fusion, where the reranker has its own weighted lane; always-rerank
  is now 85.2 / 67.3 nDCG against 79.6 / 63.0.
- Results were not reproducible run to run (identical configurations scored
  79.6–85.2) because candidates sorted out of `HashMap`s inherited the
  process's random hash order. Every score sort now has an id tie-breaker and
  session grouping uses a `BTreeMap`; two runs are now byte-identical per
  question.
- Ranking dropped any conversation older than 90 days (60 for summaries, 365
  for lessons, 730 for facts) through a hard TTL on *event* time, so imported
  history was invisible. Age now only decays the score, with a floor;
  retention remains lifecycle's job, measured from storage time.
- Pre-synthesized answer rows were dated "now" instead of the memory they
  restate, which distorted recency ranking.
- `graph_insert_edge` committed one transaction per edge and swallowed every
  error; edges are now written in one batch that propagates failures.

**Not done**

- **WP5 fact extraction** (encoder-only tiers, `fact_evidence`, `why_stale`,
  predicate canonicalization). Not started. Note for the spike: the graph
  edges today come from parsing derived card text, so their subjects are words
  like `Session`, `Canonical`, `user` and `assistant` and their objects are
  whole sentences — which is why disabling `graph_edges` costs no measurable
  recall. Fixing extraction is what should make the graph worth its cost.
- **Workspace crate split** (`tellodb-core` / `-models` / `-server` / `-cli`).
  The library API and CLI exist, but the code still lives in one crate with
  the retrieval pipeline under `api::handlers`; splitting it is a packaging
  refactor worth doing once the API settles.
- **Node binding** and the **OpenAI-compatible proxy** (README now marks the
  proxy as planned).
- **Disk cache for HNSW segments.** Large segments rebuild on first use
  (~11 s for 100k i8 vectors at 384 dims).
- **WP1 / WP9 GPU runs**, and therefore every default that was left where the
  laptop could not decide it (embed-text mode, rerank policy, quantization).

## Goal

Make Tellodb a fast, local-first, single-binary temporal memory engine that is
good enough to publish on. This round covers:

- how memories are represented and embedded,
- fact replacement driven by extraction instead of keyword rules,
- query latency (graph stage, reranker cascade) and ingest cost (ablations),
- vector storage (per-entity segments, quantization),
- a library-first workspace that builds one binary,
- evaluation on a GPU box with enough questions for confidence intervals to mean something.

Out of scope this round:
- benchmark-contamination cleanup (rules written for specific LoCoMo questions stay as they are);
- any generative LLM inside the engine (extraction, reranking, planning). Only small encoder models (embedder, cross-encoder, span extractor) run in-process.

Existing data does not need to be preserved. Schema and ID changes may drop and
recreate tables; no migrations are required.

## Ground rules

- **Measure first.** Every work package (WP) starts from a run record and ends with one produced the same way.
- **Default behavior changes only when numbers support it.** New behavior ships behind an environment flag. The default flips only when dev-split recall is within the confidence interval (or better) and at least one cost metric improves.
- **One WP per commit (or small commit series)**, with `cargo fmt`, `cargo clippy --all-targets -- -D warnings` and `cargo test` passing in both crates.
- **Quality numbers come from the dev split on the GPU box.** The laptop runs smoke tests only.

## Order and dependencies

```
WP1 GPU dev baseline ──┬─> WP2 memory representation ──> WP3 ingest ablations
                       │                                     │
                       ├─> WP4 graph latency                 v
                       │                               WP5 fact extractor
                       ├─> WP6 vector segments + quantization
                       └─> WP7 retrieval cascade
WP2..WP7 ──> WP8 library-first workspace + single binary ──> WP9 final dev + scaling runs
```

WP4, WP6 and WP7 are independent of WP2 and can run in parallel once WP1 exists.
WP8 comes last so code isn't moved while its behavior is still changing.

---

## WP1 — GPU dev-tier baseline

**Why:** smoke runs use n=10 per dataset, so the confidence intervals span ±30 points. Every later decision needs a dev-split baseline.

**Machine:** one RTX 4090 or L4 (24 GB), Ubuntu 24.04 (RunPod/Vast). Setup uses `scripts/linux_ubuntu`.

**Tasks**
1. **Verify CUDA embedding is actually used.** Run `TEMPORAL_MEMORY_DEVICE=cuda` and check that `/version` reports `device: CUDA`. Run `examples/embed_throughput.rs` on the box and record texts/s at short, 256 and 512 tokens, next to the M2 numbers.
2. **Add GPU metadata to run records.** `host.gpu` is already collected; add the ORT execution provider in use (from `/version`).
3. **Run the dev tier twice:** `bash benchmarks/run_baseline.sh --tier dev`, once with `TIMESTAMPS=wallclock` and once with `TIMESTAMPS=session`. That's LongMemEval-S dev (149 questions), LoCoMo dev (3 conversations, 432 questions), and synthetic at 1k and 10k.
4. **Commit** `benchmarks/BASELINE.md` and the run records for this baseline. Ignore everything else under `benchmarks/runs/` and `benchmarks/snapshots/` in git.

**Acceptance**
- Two identical dev runs agree within their confidence intervals.
- The baseline table has per-question-type recall_any, recall_all and nDCG, plus p50/p95/p99 per query stage and ingest memories/s.

---

## WP2 — Memory representation: raw turns, explicit identity, stored-turn context

**Problem**
- **Header-dominated embeddings.** The evaluator embeds a 3-turn window prefixed with `[Session ID] [Session Date] [Session Focus] [Window Turns]` (`build_enriched_window`, `benchmarks/rust_evaluator/src/main.rs`). On short turns the header dominates the embedding, which is why synthetic recall is stuck around 80%.
- **Batch-dependent context.** The engine adds a second, batch-dependent context header (`expand_and_enrich_payloads` → `build_context_header`). Its neighbours are whatever else is in the same HTTP request, companions included. The same conversation gets different vectors depending on batch size.
- **Identity parsed from IDs.** Session and turn are parsed out of `memory_id` (`entity::session::turn[::tag]`). Clients with UUID memory IDs get no session routing, and the scoped ANN filter drops them.

**Design**
1. **Explicit identity fields.**
   - Add `session_id: Option<String>`, `turn_index: Option<u32>`, `role: Option<String>` (`user`/`assistant`/…) and `event_time_ms` to `IngestPayload`.
   - Store them as columns on `memories` and `vector_lookup`.
   - `memory_id` becomes an opaque identifier. Derived records keep the `{parent}::{tag}` suffix only for cascading deletes and never parse it for meaning.
   - Remove `session_id_from_memory_id`, `turn_index_from_memory_id`, `split_memory_id` and `routed_session_from_memory_id` from the retrieval path. Use the columns instead (fall back to parsing only when the client sent nothing).
   - The scoped ANN filter uses `vector_lookup.entity_id`, not the `"{entity}::"` string prefix.
2. **What gets embedded (`TELLODB_EMBED_TEXT`):**
   - `turn` (new default candidate): the turn's own text, prefixed with the role only (`user: …`).
   - `turn+context`: the turn plus up to `TELLODB_CONTEXT_WINDOW` (0|1|2, default 1) neighbouring turns, read from **stored** turns of the same `(entity_id, session_id)` ordered by `turn_index` (original turns only, never companions). Neighbours arriving in later batches are handled by recomputing the embeddings of the affected earlier turns (bounded to `window` turns back).
   - `legacy`: today's behavior, for comparison.
   - Dates and session IDs are metadata columns, never embedded text.
3. **Evaluator.** Add `--client-context off|window` (default `off`). With `off`, the evaluator sends one payload per turn with `session_id`, `turn_index`, `role` and the parsed session date, and no headers. `window` keeps the current behavior for the ablation table.
4. **FTS content.** Index the raw turn text (plus role). Session router text is built from stored turns of the session, not from payload headers.
5. **Deterministic ingest.** Embeddings must not depend on batch size or request order.

**Code touched:** `src/api/types.rs` (payload), `src/storage/tenant.rs` (schema: `memories`, `vector_lookup` columns and indexes on `(entity_id, session_id, turn_index)`), `src/api/handlers/ingest.rs` (`expand_and_enrich_payloads`, `generate_embeddings`), `src/api/ingest/companion.rs` (`build_context_header` removed or replaced), `src/api/handlers/query.rs` (session derivation, scoped ANN filter), `src/api/utils.rs` (ID parsers), evaluator `ingest_instance`.

**Tests**
- **Batching invariance:** ingesting a session in batches of 1, 8 or 50 produces bitwise-identical stored embeddings (CPU).
- **UUID memory IDs:** with explicit session/turn fields, recall matches the structured-ID form on a fixture.
- **Late neighbours:** a turn whose neighbour arrives in a later batch gets its embedding recomputed.

**Experiments (dev split, GPU):** a table with `legacy` vs `turn` vs `turn+context` (window 1, 2), crossed with client context `off` vs `window`. Report recall_any/all, nDCG, per question type, ingest texts/s and database size.

**Acceptance:** pick the default from the table. Expected: synthetic current-value and as-of recall rises well above 80% without lowering LongMemEval/LoCoMo recall beyond their confidence intervals.

---

## WP3 — Ingest ablations

**Problem:** each ingested turn fans out into up to about 9 embedded texts:
- chunks
- gist and keyword companions
- fact companions
- atomic memory cards
- event companions
- relation companions

Plus non-embedded derived structures:
- session router
- temporal events
- shadow questions
- facet postings
- mem cells and scenes
- profile facts
- retrospective links
- preferences

None of them has been measured. Embedding cost scales with the fan-out.

**Tasks**
1. **One switch per structure.** `TELLODB_DISABLE=chunks,gist,keywords,fact_companions,cards,events,relations,session_router,temporal_events,shadow_questions,facets,mem_cells,mem_scenes,profile_facts,retrospective_links,preferences`. Each switch skips both creation and the matching retrieval lane or bonus. Parse it once at startup into a `IngestFeatures` struct on `EngineState`, and report it in `/version` and run records.
2. **Diagnostics.** Record `expanded_count`, `embedded_count` and bytes written per structure in ingest diagnostics and run records.
3. **Ablation script** `benchmarks/run_ablation.sh`: for the chosen WP2 representation, run dev once with everything on, then once per disabled structure. The embedding cache makes repeats cheap.
4. **Report table:** Δ recall_any, Δ nDCG, Δ ingest texts/s, Δ DB bytes/memory, Δ p95 query latency, each with the confidence interval.

**Acceptance and decision rule**
- A structure whose removal keeps every dataset's recall and nDCG within the confidence interval is **deleted from the code** (not just disabled).
- A structure that clearly helps stays and is listed in the paper's system description with its measured contribution.

---

## WP4 — Graph stage latency

**Problem:** the graph stage (in `fusion_phase`) takes 135–165 ms of a ~200 ms query, even on tiny synthetic tenants.

**Tasks**
1. **Instrument sub-stages first:**
   - link-cluster CTE per seed
   - edge BFS (`collect_edge_cluster_scores_with_intent`)
   - entity-seed lane (`graph_query_edges` per named phrase)
   - lookups
   - thread-pool overhead (`rayon par_iter` over 24 seeds)

   Add `x-tm-graph-{links,edges,entities,lookup}-us` headers and include them in run records.
2. **Likely fixes (apply only what the measurements justify):**
   - **Query-local neighbour memo:** many seeds share neighbours. A `HashMap<memory_id, neighbours>` for the query avoids repeated SQL.
   - **Batch depth-1 frontier queries** with a fixed-size `IN` list of 32. Plain indexed lookups only; the CTE/window-function batch version was measured 50× slower.
   - **Per-tenant node degrees:** keep a `graph_node_degree(node PRIMARY KEY, degree)` table maintained on edge insert, so hub pruning is one indexed lookup instead of two count subqueries per node.
   - **Single connection for the stage:** borrow one pooled connection and run the traversal serially. Measure against `par_iter`; the per-seed work is ~1 ms and thread handoff may cost more than it saves.
   - **Skip graph when nothing uses it:** when normalized graph weight × candidate count can't change the top-k (for example, all candidates already come from one session), skip the stage. Record `graph_skipped` in diagnostics.
3. **Tunable hop limits:** `TELLODB_GRAPH_SEEDS` (default 24) and `TELLODB_GRAPH_MAX_DEPTH` (default 2). Include both in the ablation table.

**Acceptance:** graph p50 ≤ 20 ms on LongMemEval dev with recall/nDCG within the confidence interval of WP1/WP2.

---

## WP5 — Fact extraction for replacement (temporal truth)

**Problem:** facts replace each other only when keyword rules in `infer_fact_key` (`src/api/ingest/fact.rs`) produce the same key. Most real phrasings ("I just relocated my home to Dublin", "Taking on the role of Security Analyst") produce no key, so there is no replacement, and the fact history and current-value results stay wrong.

**Design**
1. **Triple model.** Each extracted fact is `(subject, predicate, object, confidence, extractor)`. The `fact_versions` chain logic in `register_fact_versions_batch` stays; its key becomes `(entity_id, subject_norm, predicate_canon)`.
2. **`Extractor` trait** (`fn extract(&self, turns: &[Turn], ctx: &ExtractCtx) -> Result<Vec<Triple>>`). **No generative LLMs anywhere in the engine**, local or remote: they add too much latency per ingested turn. Extraction uses only small encoder models (single forward pass, no token-by-token decoding) and rules. Tiers:
   - **T0 `RuleExtractor`:** today's `infer_fact_key` rules, as a fast path and a baseline.
   - **T1 `EncoderExtractor` (default candidate).** A span-extraction encoder run through ONNX Runtime on the same device as the embedder, batched like embeddings. Run a spike comparing:
     - (a) GLiNER-style span and relation extraction (e.g. the `gline-rs` crate, or a direct `ort` session with a small GLiNER model, ~50–200M parameters)
     - (b) a cheaper two-step pipeline: a token-classification (NER) encoder finds the object span, and predicate candidates come from the verb phrase around it, canonicalized in step 3

     Criteria: triple precision/recall on the held-out paraphrase set, extraction ms/turn on CPU and GPU, model size. Pick the fastest option that meets the quality bar.
   - **Gating to keep ingest fast:** T1 runs only on user-authored turns that pass a cheap first-person/state-change filter (e.g. "I …", "my …", "we …" plus a verb). Assistant turns and filler skip extraction. Record the fraction of turns extracted in ingest diagnostics.
3. **Predicate canonicalization.** Embed predicate strings. Assign each to an existing canonical predicate when cosine ≥ τ (default 0.8, tuned on dev), else create one. Stored in `predicate_canon(entity_id, canonical, variants_json, embedding)`. The same object restated confirms the current version (adds evidence) instead of creating a new version.
4. **Evidence table.** `fact_evidence(entity_id, fact_key, memory_id, confidence)`, so answers can cite every supporting memory.
5. **Query integration**
   - **Current-value questions** (`plan.prefers_latest`): filter candidates whose facts are superseded, and boost memories that are evidence for current versions of predicates matching the query (predicate embedding vs query).
   - **As-of questions** use `invalidated_set_at_time` on the extracted chain.
   - **Explanations:** `QueryResult` gains `why_stale` — which fact replaced it, when, and from which memory.
6. **Benchmark:** extend `synth.rs` with a `--paraphrase-file` input. The owner writes about 300 held-out paraphrases; the implementer does not. Report replacement precision/recall, stale-in-top-k rate and as-of accuracy per extractor tier.

**Acceptance:** T1 beats T0 on held-out replacement F1 by a clear margin. Synthetic current-value and as-of recall reach at least 90% on dev with T1. Extraction adds ≤ 15% to GPU ingest time and ≤ 25% on CPU. Query latency is unchanged, because extraction never runs at query time.

---

## WP6 — Per-entity vector segments and quantization

**Problem:** each tenant has one HNSW index. Entity-scoped searches over-fetch and filter afterwards, under one lock. Vectors are stored as f32 in SQLite and again in usearch.

**Design**
1. **Per-entity segments.** `EntitySegment { ids, vectors }` per `(tenant, entity_id)`, loaded lazily from `vector_lookup.embedding`.
   - Below `TELLODB_FLAT_THRESHOLD` (start at 20k; tune from the benchmark), exact SIMD dot-product search (`simsimd` or `wide`).
   - Above it, a per-entity usearch HNSW built on demand and cached on disk next to the database.
   - Unscoped queries search across segments with a bounded merge.
   - No more filtering after search and no global lock.
2. **Quantization** (`TELLODB_VECTOR_QUANT=f32|f16|i8|binary`). Store the quantized vector in SQLite plus f32 for rescoring when not `f32`. Search on the quantized vectors, then rescore the top `k × TELLODB_RESCORE_FACTOR` (default 4) with f32.
3. **Recall accounting:** a benchmark mode that compares each search against exact f32 search and reports recall@10 of the ANN step alone.
4. **Criterion benchmarks** (extend `benches/benchmarks.rs`): 10³–10⁶ vectors per entity and 1–10k entities, for flat vs HNSW and each quantization mode. Report latency p50/p99, bytes/vector and cold load time.

**Acceptance**
- Scoped search latency no longer depends on other entities' sizes.
- The default quantization keeps ANN recall@10 ≥ 0.98 against exact search.
- End-to-end dev recall stays within the confidence interval.

---

## WP7 — Retrieval cost cascade

**Problem:** stage 2 (cross-encoder rerank) turns on for most questions through keyword heuristics (`should_apply_neural_rerank`), and the reranker is the largest model on the query path.

**Design**
1. **Stage 1 (cheap):** BM25 + ANN + routing, fused with RRF, then scoring. Unchanged.
2. **Confidence gate.** Rerank only when stage 1 is uncertain: normalized score gap between top-1 and top-k below `TELLODB_RERANK_MARGIN`, or low evidence confidence (`compute_evidence_confidence`). Remove the keyword heuristics once the gate is measured to be at least as good. Record `rerank_applied` and the reason per question (already partially in run records).
3. **Configurable reranker:** `TELLODB_RERANK_MODEL` (`bge-reranker-base`, a smaller fastembed-supported reranker, or `none`) and `TELLODB_RERANK_TOP` (default 25).
4. **Trade-off curve:** sweep the margin and report recall/nDCG against p95 latency and rerank rate. The chosen default is the knee of the curve.

**Acceptance:** at the default, dev recall/nDCG match "always rerank" within the confidence interval, and p95 query latency drops by ≥ 40% against always-rerank on GPU.

---

## WP8 — Library-first workspace and single binary

**Design:** a Cargo workspace:

```
crates/tellodb-core     storage, temporal model, vector segments, retrieval pipeline, metrics (no axum)
crates/tellodb-models   Embedder / Reranker / Extractor traits + ONNX implementations, model registry
crates/tellodb-server   axum HTTP API, MCP (HTTP + stdio), OpenAI-compatible proxy, auth, rate limiting, platform
crates/tellodb-cli      the `tellodb` binary: serve | mcp | ingest | query | bench | doctor | rebuild-index
crates/tellodb-ffi      C ABI (cbindgen) for Python/Node bindings
```

**Tasks**
1. **Core API** (synchronous, `Send + Sync`):
   ```rust
   let db = tellodb::Db::open("agent.tello", Options::default())?;
   db.ingest(&[Memory { entity, session, turn, role, text, event_time_ms, .. }])?;
   let hits = db.query(Query::new(entity, "where do I live now?").as_of(ts).k(8))?;
   let fact = db.current_fact(entity, "residence")?;
   let agg  = db.aggregate(entity, "money", range)?;
   ```
   `EngineState`, HTTP types and `StatusCode` leave the core. Core returns `tellodb_core::Error`.
2. **One database file per tenant:** `<name>.tello`, a SQLite file. The HNSW cache sits alongside as `<name>.tello-hnsw` and is rebuilt when missing or stale (epoch stored in a `meta` table).
3. **Models ship with the binary.** A `bundled-models` feature embeds quantized bge-small (int8 ONNX) and the tokenizer via `include_bytes!`. Otherwise the first run downloads to a user cache and verifies a pinned SHA-256. The device is picked automatically: CUDA, then CoreML, then CPU, with a `--device` override.
4. **stdio MCP server:** `tellodb mcp` speaks MCP over stdin/stdout for Claude Code, Cursor and others. The HTTP `/mcp` route stays in the server crate.
5. **OpenAI-compatible proxy (optional):** `tellodb serve --proxy <upstream>` injects retrieved memories into the caller's own `/v1/chat/completions` request and forwards it. Tellodb makes no model calls of its own; the only added latency is one retrieval.
6. **`tellodb doctor`:** prints device and execution provider, model hashes, per-tenant memory/vector counts and index health, WAL size, and the embedding cache hit rate.
7. **Bindings:** Python via maturin/pyo3 on `tellodb-core` directly, Node via napi-rs. These replace `../python` and `../node`.
8. **Evaluator:** gains an `--in-process` mode that links `tellodb-core`, so benchmarks can skip HTTP overhead. The HTTP mode stays for end-to-end numbers.

**Acceptance**
- `cargo build --release -p tellodb-cli` produces one binary that runs `serve`, `mcp` and `doctor` with no other files present (with `bundled-models`).
- The HTTP API and evaluator results are unchanged against WP7's run records.
- The Python binding passes an ingest → query → current_fact smoke test.

---

## WP9 — Final dev evaluation and scaling runs

1. **Full dev tier on the GPU box** with the WP2–WP8 defaults, 3 repeats, run records committed.
2. **Scaling curves** (synthetic plus random vectors): 10³ → 10⁷ memories, and 1 → 10k entities. Measure ingest memories/s, query p50/p95/p99, bytes/memory, cold start. GPU (L4) and CPU (c7i.4xlarge-class) machines.
3. **Paper tables generated only from run records** (`rust_evaluator report`): the representation table (WP2), ablations (WP3), extractor tiers (WP5), quantization (WP6), cascade curve (WP7), scaling.

## Risks

- **Context-window recomputation (WP2)** adds write amplification on streaming ingest. Bound it by the window size and measure it.
- **Extractor quality (WP5):** encoder-only extraction may not beat rules on held-out phrasings. The spike has explicit go/no-go criteria; T0 stays as fallback. No generative LLM is added as a fallback.
- **Workspace split (WP8)** is a large mechanical move. Do it in one branch with no behavior changes and compare run records before and after.
- **GPU nondeterminism** can move recall slightly between runs. Always report 3 repeats with confidence intervals.
