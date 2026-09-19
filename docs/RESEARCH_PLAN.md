# TelloDB Research Plan — "SQLite for agent memory"

Status: draft v1 · 2026-09-19 · target: one flagship systems paper + one empirical paper

---

## 1. The claim we are trying to earn

Everything below exists to support or falsify one thesis:

> **A single embeddable artifact — deterministic, no LLM in the ingest or retrieval loop — can match hosted LLM-orchestrated memory systems on long-horizon benchmarks at 1–2 orders of magnitude lower cost and latency, and the *same* file format and binary runs from a 512 MB ARM board to a multi-tenant server.**

Three separable legs, each falsifiable on its own:

| Leg | Claim | Killed by |
|---|---|---|
| **L1 Quality parity** | Deterministic extraction + lifecycle reaches ≥ parity with Mem0/Zep-class systems on LongMemEval + LoCoMo | Losing outside CI on the frozen test split |
| **L2 Cost frontier** | Pareto-dominant on accuracy per dollar, per ms, per byte, per joule | A baseline that is cheaper *and* as accurate |
| **L3 Scale invariance** | Identical quality across device tiers; only latency moves. One file, no sidecars | Quality that degrades when the memory budget shrinks |

L3 is the actual "SQLite" analogy, and it is the least-contested territory in this literature — nobody in the agent-memory space reports crash safety, byte budgets, or energy. That is where the defensible novelty is.

### What is genuinely new here

Positioned against Mem0, Zep/Graphiti, MemGPT/Letta, A-MEM, HippoRAG and the LongMemEval paper itself:

1. **Determinism as a design axis.** `lifecycle.rs` carries a versioned policy (`lifecycle-v1-deterministic`). Nobody has published "how far can you get with zero LLM calls in the memory path."
2. **Structure-level ablation at scale.** `src/features.rs` has 18 switches, each disabled at *both* ingest and query time — so a run measures quality contribution *and* cost. No published agent-memory system reports which of its structures actually earn their keep. This is the highest novelty-per-effort result available to us.
3. **Memory under a fixed byte budget.** Forgetting is a first-class admission policy, not an afterthought. Accuracy-vs-retention-budget is an evaluation axis that does not exist in the literature yet.
4. **Deletion completeness.** Provable erasure of a fact and all its derivatives (derived links, graph edges, vector segments, FTS). Testable, differentiated, and immediately relevant to anyone deploying memory under GDPR.

---

## 2. Ideas to add, ranked by paper value × effort

### Tier A — do these, they are the paper

**A1. Frozen test discipline (blocking, do first).**
`ScoringWeights` is documented as "empirically tuned on the LoCoMo benchmark." Any reviewer kills the paper for this. Split every dataset into train / dev / **sealed test**; all tuning on dev; test unsealed once per paper revision, logged. `benchmarks/splits/` already has the machinery — it needs a sealed tier and a commit-hash-stamped unsealing log.

**A2. Graph-lane latency (biggest systems win on the table).**
Current breakdown: graph is 138–164 ms of a ~200 ms query. It is ~70% of latency across all three datasets. Options to test, in order of expected value:
- Precomputed per-entity neighborhoods (materialized 1–2 hop closure), refreshed on write.
- Roaring-bitmap adjacency instead of row-wise SQLite traversal.
- **Budget-gated expansion**: only pay graph cost when the router predicts multi-hop. This converts a fixed cost into a *knob*, which gives us the anytime-retrieval Pareto curve in §3.
- Personalized-PageRank-style expansion (HippoRAG comparison point) as the quality ceiling to measure against.

**A3. The 18-switch ablation, cost-adjusted.**
`run_ablation.sh` already generates the matrix. What is missing for publication: ≥3 seeds, Holm–Bonferroni correction across the 18 comparisons, effect sizes, and a **cost-adjusted ranking** (Δquality per Δms, per Δbyte, per Δingest-second). Deliverable: a single ranked table, "what earns its keep," plus the pruned configuration it implies.

**A4. One file, no sidecars.**
Today: sharded SQLite + a usearch sidecar + an embedding cache. Target: vectors live in SQLite pages in a quantized, mmap-able layout so a memory is one portable file. `vector_index.rs` already does f32/f16/i8/binary with rescore — extend to product quantization / RaBitQ and publish the **recall-vs-bytes-per-memory curve**. Headline metric candidate: *bytes per memory at 95% of full-precision recall*.

**A5. Hardware-budget autotuner.**
`doctor.rs` is the seed. Probe the host, then pick quantization level, flat/HNSW threshold, HNSW `M`/`ef`, embed batch and token cap to hit a declared quality SLO within a memory ceiling. Claim: *a memory engine that self-configures to a hardware budget with a stated quality SLO*. This is what makes L3 a result rather than an anecdote.

**A6. Retention-budget curves.**
Sweep the admission threshold (`INDEX_VECTOR_MIN_ADMISSION`, `ADMISSION_SCORE_*`) and plot accuracy vs. retained bytes. Two outcomes, both publishable: forgetting merely saves space (cost story), or forgetting *improves* precision (a much stronger story — "less memory, better answers").

**A7. Deletion completeness harness.**
Insert a marked fact, let it propagate into cards/facts/edges/vectors/FTS, issue erasure, then attempt recovery through every lane. Report a completeness rate and worst-case residual. Nobody else reports this.

### Tier B — strong, fold in if time allows

- **B1. Learned-but-frozen fusion.** Replace hand-tuned `ScoringWeights` with a ≤50 KB logistic/LambdaMART model over the existing lane features. Keep it frozen at inference so determinism survives. Test transfer: train on LoCoMo, evaluate on LongMemEval.
- **B2. Deterministic vs. LLM extraction.** `extract.rs` + `gliner.rs` make this testable today. Headline: "deterministic extraction reaches X% of LLM-extractor quality at 1/100 the cost." This directly quantifies L1's central bet.
- **B3. Temporal / knowledge-update reporting.** Fact supersession chains should shine exactly where RAG baselines fail. Report LongMemEval *per category* (temporal reasoning, knowledge update, multi-session) — aggregate numbers hide our best result.
- **B4. Energy.** Joules per query and per 1k ingests on an ARM board with an inline power meter. Rare in this literature, cheap to produce, and it is the figure people will screenshot.
- **B5. Multi-tenant isolation.** `tenants/` + per-entity vector segments already give the mechanism. Quantify noisy-neighbor: p99 under N tenants, with the tail of one tenant's queries under another's ingest storm.
- **B6. Crash safety.** `fuzz/` and `proptest-regressions/` exist. Turn them into a reported table: kill -9 under concurrent ingest at N injection points, recovery rate, corruption rate.

### Tier C — explore, likely future work

- Sleep-time / background consolidation: amortized quality gain vs. background CPU.
- Embedding-model size sweep × device tier → the full quality/cost frontier surface.
- Cross-session entity resolution error analysis (`entity_resolver.rs`) — likely a qualitative section.
- Sensitivity tiers (`SENSITIVITY_RISK_*`) as a policy-aware retrieval story.

---

## 3. What to test — the experimental matrix

**Datasets.** Keep LongMemEval (S and M), LoCoMo, synth-1k. Add: MSC (multi-session chat) for breadth, a multi-hop set for the graph lane, and a **stress corpus at 10M memories** for L3 — the existing `synth` generator can be extended rather than replaced. Sealed test split on every one.

**Baselines.** Non-negotiable for L1 credibility, run with each author's own recommended configuration on a single node:
no-memory · BM25 · dense-only · hybrid RAG · full-context long-window LLM · Mem0 / Mem0^g · Zep · Letta/MemGPT.
Every baseline gets the same cost/latency accounting we apply to ourselves — API calls included.

**Metrics.** recall@k · nDCG · answer accuracy · p50/p95/p99 query latency · ingest memories/s · **bytes per memory** · peak RSS · **$ per 1k queries** · **joules per query**. The last three are where we win; they must be first-class columns, not an appendix.

**Statistics.** ≥3 seeds everywhere. Paired bootstrap CIs (already implemented). Holm–Bonferroni across the ablation family. Effect sizes reported alongside p-values.

**Judge validation.** The LLM judge needs a human-agreement study on a stratified sample (~200 items, two annotators, Cohen's κ). Without it, every accuracy number in the paper is contestable. This is small work that protects the whole results section.

---

## 4. Phased plan (~16 weeks)

Each phase has an exit gate. Do not start the next phase until the gate is met — the gates are what keep the paper honest.

**Phase 0 · Measurement hygiene (weeks 1–2)** — A1. Sealed splits + unsealing log; seed control; judge-agreement study; cost model (dollars, joules, bytes) wired into the evaluator; hardware tiers provisioned (ARM board, laptop, server); `paper` tier in `run_baseline.sh` promoted to the canonical run.
*Gate: BASELINE.md reproduced on a clean machine within CI, from a single command.*

**Phase 1 · Baselines and parity (weeks 3–5)** — L1. Port/run all external baselines; build the unified cost table.
*Gate: ≥1 dataset where TelloDB is within CI or better, with the full cost column filled in.*

**Phase 2 · The ablation study (weeks 4–8, overlaps Phase 1)** — A3, B3. Full 18-switch × 3-dataset × 3-seed matrix; category-level LongMemEval reporting.
*Gate: ranked "earns its keep" table with corrected significance, and a pruned default configuration that loses nothing on dev.*

**Phase 3 · Systems frontier (weeks 6–10)** — A2, A4, A5, B4, B5, B6. Graph-lane work, quantization curve, autotuner, energy, isolation, crash safety.
*Gate: p95 < 100 ms on laptop tier at unchanged quality; full pipeline running on the ARM tier with a measured bytes/memory and joules/query figure.*

**Phase 4 · Forgetting and budgets (weeks 9–12)** — A6, A7, B2. Retention curves, deletion completeness, deterministic-vs-LLM extraction.
*Gate: the accuracy-vs-retained-bytes curve, and a deletion completeness number we are willing to print.*

**Phase 5 · Write and package (weeks 12–16)** — paper, artifact, reproducibility bundle (Docker image, sealed-split runner, one-command reproduction of every table).

---

## 5. Paper strategy

- **P1 — flagship systems paper.** The engine, the ablation, the cost/scale frontier. Venue: VLDB / SIGMOD, or MLSys. CIDR if we want a shorter vision-shaped version first.
- **P2 — empirical.** "What actually earns its keep in agent memory?" The 18-switch study stands alone and is the most citable thing here. Venue: EMNLP / ACL / an eval-focused workshop.
- **P3 — future.** Deterministic lifecycle and forgetting as belief revision, if Phase 4 produces the "less memory, better answers" result.

Lead with P2's content *inside* P1 if we only have bandwidth for one paper — the ablation is the part reviewers will not have seen before.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Weights tuned on what becomes test data | Phase 0 sealed splits, unsealing log — blocking |
| LLM-judge variance swamps effects | Judge-agreement study + fixed judge version + seeds |
| Benchmark contamination in the base LLM | Report a no-memory control; contamination probe on a sample |
| Baselines run unfairly (ours tuned, theirs not) | Authors' own configs, single node, documented; invite pre-submission review |
| Graph optimization trades quality for latency silently | Every latency change re-runs the quality gate; no latency claim without a paired quality delta |
| 16 weeks is optimistic | Phases 2 and 3 are the paper; 4 is cuttable to future work |

## 7. Immediate next actions

1. Seal the test splits and add the unsealing log (`benchmarks/rust_evaluator/src/splits.rs`).
2. Add bytes/memory, peak RSS, and $/1k-queries columns to the evaluator record (`record.rs`).
3. Run the full 18-switch ablation at `dev` tier, 3 seeds, overnight — it tells us where the rest of the effort goes.
4. Profile the graph lane to find where 140 ms actually goes before optimizing it.
5. Stand up the ARM tier and get one end-to-end run on it, however slow.

---
---

# Part II — Experiments, architecture, and findings from the code

Part I set the claims. This part is what we actually build and run, grounded in a read of the query hot path (`src/api/handlers/query.rs`, `src/storage/tenant.rs`, `src/vector_index.rs`).

## 8. The architectural pillar: determinism as a storage primitive

We already have a versioned, deterministic lifecycle policy (`lifecycle-v1-deterministic`). Today that determinism is used as a *cost* argument — no LLM in the loop. The proposal is to make it a **storage primitive**, which is what turns a good engine into a paper.

### 8.1 The Memory Log + deterministic replay

> Store the raw turns and the policy version. Everything else — cards, facts, edges, gists, vectors, FTS — is a **materialized view** that can be rebuilt bit-identically by replaying the log under a named policy.

```
memory_log (append-only, the only durable truth)
      │
      │  replay(policy = lifecycle-v1-deterministic, seed = 0)
      ▼
derived: cards · facts · edges · vectors · fts · router · profiles
```

Four properties fall out of one mechanism, which is exactly the shape a reviewer rewards:

| Property | How the log gives it |
|---|---|
| **Provable erasure** | Delete from the log, replay. Nothing derived can survive, by construction — no per-table cascade to get right, no residue to argue about |
| **Policy migration** | Upgrading `lifecycle-v2` is a replay, not a reingest. Old and new policies are comparable on identical inputs |
| **Archival footprint** | Cold memories keep only the log. Derived structures rebuild on demand. Bytes/memory collapses for the long tail |
| **Reproducibility** | Reviewers rebuild every table in the paper from the log + a policy version. Bit-identical or it's a bug |

Erasure is the one to lead with. §A7 proposed *measuring* deletion completeness; replay makes it **structurally guaranteed** rather than empirically estimated, which is a much stronger claim to print.

Cost to check: replay throughput must stay above ingest throughput or cold-tier rebuild becomes unusable. That is an experiment (E7), not an assumption.

### 8.2 The data structure: entity-segmented CSR graph

The strongest observation from reading the code is an **asymmetry the codebase already resolved once**. `vector_index.rs` says, explicitly:

> "Each entity's segment is searched on its own, so an entity-scoped query never scans or filters other entities' vectors and entities never contend on one lock."

The graph lane never got that treatment. `edges` has no `entity_id` column at all — only `source`, `target`, `memory_id`. So graph traversal is tenant-wide, and the entity scope is enforced *afterwards* (`query.rs:2616`), by discarding what the traversal already paid to produce. That single asymmetry explains most of the 138–164 ms.

**Proposal — apply the vector-segment design to the graph:**

1. **Intern identifiers.** `memory_id` and node labels become `u32` symbols in a `symbols` table. Edges store integers. Kills the `String`-keyed `HashMap` and per-edge clone storm in `collect_edge_cluster_scores_for_seeds`.
2. **Add `entity_id` to `edges`**, with `(entity_id, source)` and `(entity_id, target)` composite indexes. Traversal becomes entity-scoped at the storage layer, matching vectors.
3. **Materialize `node_degree`** on write. Degree is a static graph property, currently recomputed on *every query* with two chunked `GROUP BY` passes over ~1.2k nodes.
4. **In-memory CSR adjacency per entity**, loaded lazily on first use and invalidated on write — the exact lifecycle `vector_index.rs` already implements for vectors. A 2-hop BFS becomes pointer-chasing over two `Vec<u32>`, not SQL.
5. **`EdgeType` as the `Copy` enum that already exists** in `graph.rs`, instead of a `String` materialized per row per hop.

Expected: graph lane from ~140 ms to single digits, and — because hub suppression finally measures *per-entity* degree — a recall change too. Note the direction is unknown, which is why it is E1 and not a refactor.

This also gives the paper its architectural through-line: **segment everything by entity; scope at the storage layer, never after the fact.**

### 8.3 The budget scheduler

With lanes cost-annotated, the planner becomes a scheduler over a latency budget instead of a fixed pipeline. `TELLODB_GRAPH_SEEDS` / `MAX_DEPTH` / `MAX_NODE_DEGREE` stop being env vars and become scheduler inputs. This is what produces the anytime-retrieval Pareto curve (E4) and makes the autotuner (A5) a scheduling policy rather than a pile of heuristics.

---

## 9. Findings from the code

Confidence is stated per item. Verified = traced end to end in this session; likely = read but not executed.

### 9.1 Correctness

**F1 — the entity-scope filter fails open. (verified; low likelihood, high severity)**
`query.rs:2616`, with a comment that already names the risk:

```rust
observations.get(mid).map_or(true, |o| o.entity_id == scope)
    && cards.get(mid).map_or(true, |c| c.entity_id == scope)
```

A candidate absent from *both* `observations` and `memory_cards` passes the scope check. Graph expansion inserts ids into `fused_map` straight from the `edges` table (`query.rs:2475`), and `lookup_by_memory_ids_batch` does not filter by entity. A dangling edge — one pointing at a deleted or never-materialized memory — therefore survives an entity-scoped query. The blast radius is small today because such a candidate usually dies later for lack of an observation, but **a scope filter must fail closed.** Default to `false` and make the absent case an explicit, logged decision.

**F2 — graph scores are normalized per query. (verified; ranking correctness)**
`query.rs:2534`: `score /= max_graph`. The top graph score in *every* query is rescaled to exactly 1.0, whether the underlying evidence was overwhelming or negligible. A query with only weak graph signal has that weak signal amplified to full strength before fusion. The comment explains the fix for the *previous* bug (unbounded sums reaching the hundreds) but overshoots. Use a saturating absolute transform — `s / (s + k)` — so "no good graph evidence" stays weak. Cheap to test; plausibly worth real accuracy. This is E2.

**F3 — panic handling is dead code under the release profile. (verified; minor)**
`Cargo.toml` sets `panic = "abort"` for release. `score_hydrate` catches thread panics via `join(...)` and maps them to a 500 — unreachable in any shipped binary, since a panic aborts the process. Either drop `panic = "abort"` (and keep the isolation, which matters for a *server*) or delete the handler. Right now the code advertises a resilience it does not have. For a multi-tenant deployment I'd drop `abort`: one tenant's panic should not take the node down.

### 9.2 Performance

**F4 — node degree recomputed on every query. (verified)** §8.2 item 3. Two chunked SQL passes over every unique node, per query, for a value that only changes on write.

**F5 — traversal is tenant-wide, results are entity-scoped. (verified)** §8.2. Work is done and then thrown away; worse, hub suppression (`max_node_degree = 128`) measures *tenant-global* degree, so a node that is quiet within this entity but busy tenant-wide is wrongly skipped. **This is a recall bug hiding inside a performance problem** — the most interesting single finding here, because fixing the latency may move quality in either direction.

**F6 — `prepare_cached` on `format!`-built SQL with variable placeholder counts. (verified; minor)** `get_edge_cluster_neighbors_batch` chunks at 400, so the final partial chunk generates a distinct SQL string and a distinct cache entry. Across five query shapes this can churn rusqlite's statement cache. Pad chunks to fixed sizes (400/100/25/1) so the shapes are bounded.

**F7 — `nodes_of` is an undeduplicated multiset. (verified; minor)** Built as "a node listed once per edge endpoint," then walked without dedup in the assembly loop, so a memory with many edges to the same node redoes that node's incident scan.

**F8 — allocation storm in the BFS. (verified)** `HashMap<String, Vec<(String, f32, String)>>` with `.cloned()` per incident edge, per node, per seed, and `edge_type` re-materialized as a `String` on every row. Subsumed by the interning in §8.2 — noted separately because it is measurable on its own and makes a clean before/after profile for the paper.

**F9 — rayon inside the blocking pool. (likely; measure before acting)** `seed_top.par_iter()` runs on a thread already borrowed from the blocking pool. Under concurrency this may contend rather than parallelize. Worth a flamegraph before it is worth a change.

### 9.3 Methodology

**F10 — tuned constants, untracked.** `ScoringWeights` and the lifecycle admission weights (`ADMISSION_WEIGHT_*`, `SALIENCE_*`, thresholds) are hand-tuned floats spread across two files, with provenance only in comments. Before any of this is a paper: one versioned config struct, serialized into every benchmark record, so a result is always attributable to an exact parameter set. This is a prerequisite for Phase 0, not a nicety.

---

## 10. Experiments

Ordered by information per unit of effort. Each states the question, the method, and what would falsify the expectation — an experiment that cannot come out badly is not an experiment.

| # | Question | Method | Falsified if |
|---|---|---|---|
| **E1** | Is the graph lane's cost buying anything? | Sweep `GRAPH_SEEDS` × `MAX_DEPTH` × `MAX_NODE_DEGREE` on the full grid; report quality vs. ms | Depth 1 matches depth 2 → we were paying 70% of latency for nothing |
| **E2** | Does per-query graph normalization hurt ranking? | F2 fix; paired delta on all three datasets | No change → the signal was saturated anyway |
| **E3** | Does entity-scoped traversal change recall, not just latency? | Implement §8.2 (2)+(3); measure quality *and* latency separately | Quality moves — then hub suppression was silently dropping real evidence, and that is the more interesting paper |
| **E4** | What is the accuracy/latency Pareto frontier? | Budget scheduler at 10/25/50/100/200 ms caps | Flat curve → latency was never the binding constraint |
| **E5** | What does each of the 18 structures earn? | `run_ablation.sh`, 3 seeds, Holm–Bonferroni, cost-adjusted | Everything significant → no pruning story, but a strong "it's all load-bearing" result |
| **E6** | How far can quantization go? | f32/f16/i8/binary × rescore factor → recall vs. bytes/memory | Binary + rescore matches f32 → headline footprint number |
| **E7** | Is deterministic replay fast enough to be a storage tier? | Rebuild all derived structures from the log; replay/ingest throughput ratio; verify bit-identical | Replay slower than ingest → §8.1 is a correctness story only, not a footprint one |
| **E8** | Does forgetting help or only save space? | Sweep admission thresholds → accuracy vs. retained bytes | Monotone decreasing → cost story. Non-monotone → "less memory, better answers" |
| **E9** | Deterministic vs. LLM extraction | `extract.rs` vs. `gliner.rs` vs. an LLM extractor, equal budget | Deterministic ≪ LLM → the central bet needs restating |
| **E10** | Does quality hold across device tiers? | Identical workload, ARM / laptop / server | Quality moves with hardware → L3 dies, and we need to know early |
| **E11** | Learned fusion vs. hand-tuned weights | Frozen ≤50 KB model over existing lane features; train LoCoMo, test LongMemEval | No transfer → hand-tuning was fitting dataset quirks, which is itself worth reporting |
| **E12** | Multi-tenant isolation | p99 for tenant A under tenant B's ingest storm, N = 1…64 | Tail blows up → per-entity segmentation isn't enough and the scale-up claim needs qualifying |

**Run order.** E1, E2 first — days of work, they retune the whole latency budget, and E1 might delete a subsystem. Then E3 and E5 (paper core). E6, E7, E10 for the scale story. E8, E9, E11, E12 as capacity allows.

**Standing rule:** no latency change ships without a paired quality delta on the dev split. F5 is precisely the case where the two are entangled, and a latency win that quietly costs recall is the easiest way to embarrass ourselves at review.

---

## 11. How we do it

Ordering is chosen so each step de-risks the next, and so nothing large gets built on an unmeasured assumption.

1. **Instrument before optimizing.** The `x-tm-graph-*` diagnostic headers already split links / edges / entities / lookup. Get a flamegraph of the 140 ms and confirm the split between SQL time, degree recomputation, and allocation. F4/F5/F8 are hypotheses with good evidence, not measurements. *One day. Do it first.*
2. **Land the cheap correctness fixes** — F1 fail-closed, F2 saturating normalization, F3 decision — each with a paired benchmark delta. Small diffs, isolated effects, and F2 may pay for itself immediately.
3. **Schema migration for the graph** — `entity_id` on `edges`, materialized `node_degree`, composite indexes. Behind a feature switch so E3 is a paired A/B rather than a one-way door.
4. **Interning + CSR segments**, mirroring `vector_index.rs`'s segment lifecycle. Only after step 3 proves the scoping thesis.
5. **Budget scheduler**, once lanes carry measured costs from step 1.
6. **Memory log + replay**, largest change, last. Land it as an *additional* durable path first, verify bit-identical rebuild against the live tables, and only then make it the source of truth.
7. **Config unification (F10)** runs alongside everything, because every result above is worthless if we cannot say which parameters produced it.

Every step: feature-switched, paired delta on dev, sealed test untouched.

---

## 12. What it should look like at the end

### The artifact

```
$ tellodb open ./alice.tellodb          # one file. no sidecars, no daemon
$ tellodb doctor --budget 512MB --slo recall@10=0.90
  → quant=i8 rescore=4 flat_threshold=8k ef=64 graph_depth=1
    projected: p95 34ms · 180 B/memory · fits in 380MB

$ tellodb forget --fact "works at Acme" --verify
  → replayed 12,481 turns under lifecycle-v1-deterministic
    0 residual references across cards/facts/edges/vectors/fts ✓
```

One file that moves between a phone and a server unchanged. One binary, embeddable as a library, a CLI, an MCP server, or an HTTP service. A `doctor` that configures itself to a hardware budget against a declared quality SLO. Erasure that is *verified by replay*, not asserted.

### The paper

Six artifacts, in the order a reader meets them:

1. **Figure 1 — the frontier.** Accuracy vs. cost, us against Mem0 / Zep / Letta / full-context / hybrid RAG. Cost on a log axis in dollars per 1k queries. This figure is the paper; everything else is support.
2. **Table 1 — quality parity.** LongMemEval (per category) and LoCoMo, sealed test, paired CIs. Temporal-reasoning and knowledge-update rows are where deterministic fact supersession should visibly win.
3. **Table 2 — what earns its keep.** The 18 structures ranked by Δquality per Δms and per Δbyte, significance-corrected. The most citable table here, and nobody else has run it.
4. **Figure 2 — scale invariance.** Identical quality bars across ARM / laptop / server; only the latency axis moves. The SQLite analogy, made empirical.
5. **Figure 3 — memory under budget.** Accuracy vs. retained bytes, with the recall-vs-quantization curve inset. A new evaluation axis, and the one most likely to be adopted by others.
6. **Table 3 — the properties nobody reports.** Deletion completeness, crash recovery, p99 under N tenants, joules per query. Uncontested territory; four rows of table that competing systems cannot fill in.

### The result, in one sentence

> Deterministic memory is not a compromise made to avoid LLM costs — it is the mechanism that makes erasure provable, migration safe, replay reproducible, and the whole system small enough to run where the data already lives.

If the experiments hold, that sentence is the abstract. If E9 fails and deterministic extraction lags LLM extraction badly, the paper becomes an honest cost/quality frontier study instead — still publishable, and we would know by Phase 4.
