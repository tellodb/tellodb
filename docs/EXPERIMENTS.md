# tellodb — experiment plan

**Audience: an implementing agent (a small model is fine).** Every experiment
below is self-contained: it states the question, the exact command, what to
record, and an acceptance check. Run them in order. **Never skip the acceptance
check.** If an experiment fails its check, stop and report — do not proceed to
the next one.

Written 2026-09-21, after the eight-workstream audit in `docs/AUDIT-2026-09-21.md`.

---

## 1. What the research is about

Everyone building agent memory today — Mem0, Zep/Graphiti, Letta/MemGPT, A-MEM —
runs a language model to decide what to store on write, and another to decide
what to retrieve on read, as a network service. Those systems are built from many
derived structures (summaries, extracted facts, typed graph edges, session
indexes, dense vectors). **Nobody has published which of those structures
actually earn their cost.**

tellodb is the instrument that makes that measurement possible: a memory engine
with **no generative model anywhere in the ingest or retrieval path**, running
embedded in one process against one SQLite file per tenant, with **18 independent
switches** (`src/features.rs`) that disable each derived structure at *both*
write time and read time. Flip one, and you get that structure's contribution to
quality *and* its cost in milliseconds and bytes, on the same run.

### The claim

> Agent memory is a storage problem. Treating it as one yields properties that
> LLM-orchestrated memory services cannot offer at any price, and makes it
> possible to measure — for the first time — which parts of a memory system are
> worth what they cost.

### Two contributions

1. **A deterministic, embeddable memory engine with database guarantees**: one
   transaction covering every relational write, an `indexed` watermark that makes
   partial-failure and crash recovery the same code path, vectors in relational
   pages with the ANN index as a disposable cache, deletion that cascades 13
   tables in one commit, and refusal to open a file written by a newer schema.
2. **The first structure-level ablation of agent memory, cost-adjusted** — the
   headline result.

### What we are NOT claiming

State these in the limitations section rather than letting a reviewer find them:

- No quality-parity claim against Mem0/Zep/Letta. **No external baseline harness
  exists.** `benchmarks/run_baselines.sh` compares tellodb to subsets of itself.
- No cost or energy numbers. There is no dollar or joule instrumentation.
- No scale-invariance claim. One x86 host, a CUDA-only Dockerfile, no ARM build.
- Replay is proposed, not implemented. Say "verifiable erasure", never "provable".

---

## 2. Standing rules

These are not optional. They are what makes the results publishable.

1. **No change ships without a paired quality delta on the dev split.** A latency
   win that quietly costs recall is the easiest way to be embarrassed at review.
2. **The sealed test split is unsealed exactly once**, at the very end, logged
   with a commit hash. As of this writing **no run record has `split == "test"`**
   — the seal is intact. Keep it that way. `--split dev` for everything below.
3. **Every run must come from a clean git tree.** `record.rs` stamps the commit
   *and a dirty flag*; a dirty run is unattributable and unusable.
4. **One protocol, pinned.** See E0.2 — a single harness flag has been observed to
   move LoCoMo recall_any by 10 points.
5. **Record everything.** Runs land in `benchmarks/runs/<timestamp>_<tier>/`.
   Never delete them, never edit them by hand.

---

## 3. Phase 0 — prep (BLOCKING, do all of it first)

Nothing measured before Phase 0 completes is usable.

### E0.1 — Clean tree

**Why:** a dirty tree makes every run record unattributable.

```bash
cd /Users/sharjeel/projects/rust/tellodb
git status --porcelain
```

**Action:** if anything is listed, commit it before running any experiment.

**Acceptance:** `git status --porcelain` prints nothing.

---

### E0.2 — Pin one canonical protocol

**Why:** two committed LoCoMo dev runs on the *same commit* report recall_any
**93.1** and **83.1**. The difference is the `--client-context` flag. The higher
number came from a configuration that applied conversational context **twice**,
because each payload is already a 3-turn block. If this is not pinned, no number
below means anything.

**Action:** fix these for every experiment in this document and never vary them:

```
--split dev
--client-context off      # NOT "window": window double-applies context
--timestamps session
--reset-first             # on the first run of a suite only
```

Record the chosen protocol in `benchmarks/BASELINE.md` as a header block.

**Acceptance:** `grep -rn "client_context" benchmarks/runs/*/**/*.json` shows the
same value for every new run.

---

### E0.3 — Regenerate `BASELINE.md`

**Why:** the committed `BASELINE.md` is a **smoke tier, n=10, RUNS=1** table from
a dirty commit three phases old. It is the first file a reader opens and every
number in it is stale. It is the source of the false "graph lane is 70% of
latency" belief that misdirected a week of planning.

```bash
bash benchmarks/run_baseline.sh --tier dev --only longmemeval,locomo
```

**Record:** the new `benchmarks/runs/<ts>_dev/` directory.

**Acceptance:** `BASELINE.md` regenerated at dev tier, with the E0.2 protocol
block at the top, and no row citing a dirty commit.

---

### E0.4 — Add a real run seed

**Why:** there is **no engine seed parameter**. `RUNS=3` currently repeats three
*identical deterministic* runs. That is not three seeds and a reviewer will say so.

**Action:** add `--seed <u64>` to the evaluator (`benchmarks/rust_evaluator/src/main.rs`,
alongside the other `global = true` args near line 300). It must actually control
something: shuffle the post-split question order with a splitmix64 keyed on the
seed, and record the seed at the top level of the run record.

**Acceptance:** three runs at `--seed 1/2/3` produce three run records with
different `seed` values and non-identical per-question row orders.

---

### E0.5 — Conversation-clustered bootstrap

**Why:** LoCoMo's 432 dev questions come from **3 conversations**, and the test
split is **7**. `mean_with_ci` in `benchmarks/rust_evaluator/src/record.rs`
resamples questions i.i.d., which overstates precision by a large factor. This is
the single most likely thing to sink the results section.

**Action:** add a clustered variant that resamples *conversations*, not questions,
and use it for LoCoMo. Report `n = number of conversations` in every LoCoMo caption.

**Acceptance:** a LoCoMo CI computed with clustering is visibly wider than the
i.i.d. one on the same data.

---

## 4. Phase 1 — the core experiments (this is the paper)

### E1 — The 18-structure ablation  ← **THE HEADLINE RESULT**

**Question:** what does each derived structure earn, per millisecond and per byte?

**Why it matters:** nobody has run this. It is the most citable thing in the
project and the reason the engine exists as an instrument.

```bash
bash benchmarks/run_ablation.sh --tier dev
```

This runs `baseline` plus one arm per switch via `TELLODB_DISABLE=<name>`, across
all 18: `chunks, gist, keywords, fact_companions, atomic_cards, event_companions,
relation_companions, memory_cards, session_router, preferences,
retrospective_links, derived_links, graph_edges, facts, semantic_dedup,
consolidation, metrics, predicate_canon`.

**Repeat for 3 seeds** once E0.4 lands.

**Record per arm:** Δrecall_any with 95% CI, ΔnDCG, Δbytes/memory, Δembedded/memory,
Δingest memories/sec, Δquery p95.

**Then:** apply **Holm–Bonferroni** across the 18 arms. `ablation_report` in
`record.rs` currently does paired bootstrap with **no multiplicity correction**
and a naive "CI spans zero → drop" verdict. At α=0.05 over 18 arms you expect
roughly one false "keep" by chance.

**Acceptance:** a ranked table of all 18, significance-corrected, with a pruned
default configuration that loses nothing on dev.

**Falsified if:** every structure is significant → no pruning story. Still
publishable as "it is all load-bearing", but a weaker paper. You need to know this
early, which is why E1 is first.

**Prior evidence this will be interesting** (from `benchmarks/runs/20260919_093324_dev_baselines/`,
a 6-arm *lane* ablation, query-side only):
- `bm25_only` on LongMemEval: **+1.3 recall (CI −2.0…+4.7) at −53% p95** → keyword
  search alone ties the full pipeline at half the cost.
- `vector_only` loses to `bm25_only` on **both** datasets (−3.4 vs +1.3; −8.6 vs −3.7).
- `no_rerank`: −0.7 and −0.9, both CIs touching zero, for −21% and −27% p95 → the
  cross-encoder probably does not earn its cost.
- `no_graph`: −2.7 and −1.4, both significant, for only −5.7% and −2.5% p95 → the
  graph lane is cheap and load-bearing. **Do not delete it.**

---

### E2 — A/B the uncommitted audit fixes

**Question:** did the 2026-09-21 changes cost recall?

**Why:** two changes carry real recall risk and are currently **unvalidated**:
1. **Pre-hydration truncation** (`src/query/fuse.rs`, `FUSED_HYDRATION_FLOOR`) —
   caps the fused set before hydration. It can remove exactly the low-score,
   facet-unique candidates that session/facet coverage selection wants.
2. **LIKE-backstop gating** (`src/storage/repo/sessions.rs`) — the entity-pivot
   full scan now only runs when FTS returns zero hits. Recall-relevant for names
   FTS5's tokenizer mishandles (very short or oddly-cased).

**Method:** both arms on the **same machine**, same protocol. Arm A = current
HEAD. Arm B = HEAD with those two changes reverted. Paired delta over questions.

**Acceptance:** Δrecall_any CI for each change stated explicitly.

**Action if it fails:** if either loses recall beyond noise, revert that change
and re-run. Do not keep a latency win that costs recall.

---

### E3 — Calibrate `GRAPH_SCORE_SATURATION_K`

**Question:** what is the right saturation constant for graph scores?

**Why:** `src/query/fuse.rs` sets `GRAPH_SCORE_SATURATION_K = 2.0` as an
**unmeasured placeholder**. The old behaviour (divide by per-query max) forced the
top graph score to exactly 1.0 on every query regardless of evidence strength —
a query-independent popularity prior. The saturating transform is structurally
correct but the constant is a guess.

**Method:** dump raw (pre-calibration) graph scores over the dev split, take the
**median top-1 raw graph score**, set K to it, re-run. Use
`--dump-candidates-jsonl` to capture per-candidate scores.

**Acceptance:** K set from data, with the distribution recorded; a paired delta
against K=2.0.

---

### E4 — Quantization / footprint curve

**Question:** how far can vector quantization go before recall breaks?

**Why:** fully implemented (`src/vector_index.rs`), never exercised. Highest
result-per-effort item in the project. **Caveat:** quantization only attacks
~7 KB of the ~51 KB/memory (4.54 embeddings × 384 dims × 4 bytes). Plot **both**
vector bytes and total bytes or a reviewer will say you hid the difference.

```bash
MATRIX=$'baseline\nq_f16 TELLODB_VECTOR_QUANT=f16\nq_i8 TELLODB_VECTOR_QUANT=i8\nq_binary TELLODB_VECTOR_QUANT=binary\nq_i8_r8 TELLODB_VECTOR_QUANT=i8 TELLODB_RESCORE_FACTOR=8' \
  NAME=quant bash benchmarks/run_matrix.sh --tier dev
```

**Record:** recall_any, nDCG, bytes/memory (vector and total), query p95 per level.

**Acceptance:** a recall-vs-bytes curve with the rescore factor as an inset.

---

## 5. Phase 2 — the properties table (NO GPU NEEDED — run locally, in parallel with Phase 1)

This is the uncontested territory. Competing systems cannot fill these rows in.

### E5 — Deletion completeness

**Why:** the strongest differentiated claim. The mechanism already exists:
`delete_observation` (`src/storage/repo/memories.rs`) cascades 13 tables in one
`Immediate` transaction, and `deleted_memory_leaves_no_text_anywhere` already
plants a token and greps for residue.

**Action:** generalise that test into a harness. Instead of a hand-enumerated
table list (which will rot as tables are added), **discover** every table and
TEXT column reflectively via `sqlite_master` + `pragma_table_info`, then probe
each for the planted token. Additionally probe the **live vector lane** — run a
similarity search and assert the deleted id does not come back, since residue
there lives in the in-memory index, not in a text column.

**Record:** completeness rate, worst-case residual (table name + sample row).

**Acceptance:** a number you are willing to print, plus a concurrency variant
(delete racing an in-flight ingest and query).

---

### E6 — Crash recovery

**Why:** `synchronous = FULL` is now the shipped default (`TELLODB_DURABILITY`),
so an acknowledged write should survive `kill -9`. That is currently a claim
based on a pragma, not a measurement.

**Action:** kill -9 at N injection points under concurrent ingest, restart, then
check: schema version, `reindex_unindexed` convergence, no orphaned FTS rowid
without a `memories` row, no `vector_lookup` row without a `memories` row, no
fact chain with two `current` rows for one key.

Reuse the existing `#[cfg(test)] FAIL_NEXT_FACT_WRITE` hook in
`src/storage/repo/ingest.rs` — it is the right shape; it needs a `kill` variant
and more injection sites.

**Run the matrix twice: `TELLODB_DURABILITY=full` and `=normal`.** That gives a
*measured* loss window for NORMAL instead of an estimate, which is itself a result.

**Known exception to document:** router FTS documents were unrecoverable before
the 2026-09-21 fix (they have no `memories` row, so the repair scan could never
find them). Verify the fix holds under injection.

**Acceptance:** recovery rate and corruption rate per injection point, per mode.

---

### E7 — Multi-tenant tail latency

**Why:** isolation is structurally sound (one file per tenant, per-tenant vector
index, a colliding-rowid regression test in `src/storage/manager.rs`) but has
never been quantified.

**Action:** p99 for tenant A's queries while tenant B runs an ingest storm,
N = 1…64 tenants. Reuse the app builder in `tests/http.rs`.

**Note:** `TenantDatabaseManager` has **no eviction** — every tenant ever seen
keeps its pool and vector index resident. Expect memory growth with N and report
it honestly.

**Acceptance:** a p99-vs-N curve, and the measured per-tenant resident cost.

---

### E8 — File-format compatibility

**Why:** forward-compat refusal is implemented and tested (`src/storage/repo/schema.rs`
bails when `version > SCHEMA_VERSION`). Backward compat is only exercised against
*synthetically aged* databases, never a real file from an older binary.

**Action:** build fixture `.db` files from each historical `SCHEMA_VERSION`, check
them into `tests/fixtures/`, and add a test that opens each on today's binary.

**Acceptance:** every fixture opens and migrates; a `version + 1` fixture is
refused with the exact expected error string.

---

## 6. Phase 3 — after Phase 1 and 2 land

### E9 — CPU-tier, cold-cache run

**Why:** **every latency number you have is a warm-cache GPU number.** The dev
run reports `"device": "CUDA"` with **363,422 embedding-cache hits against 264
misses**. For a thesis that says the same engine runs from a small board to a
server, the CPU measurement does not currently exist.

```bash
TELLODB_DEVICE=cpu bash benchmarks/run_baseline.sh --tier dev --only longmemeval,locomo
```

Use `--clear-embedding-cache` for a genuine cold run.

**Acceptance:** a CPU-tier row in `BASELINE.md` beside the GPU row.

---

### E10 — External baselines  ← **THE VENUE GATE**

**Why:** this decides whether the paper is VLDB-class or something smaller.
`benchmarks/run_baselines.sh` compares tellodb **to subsets of itself**. There is
no external harness at all. Porting Mem0 alone is 3–4 weeks.

Secondary problem: your metric is **session recall**; they report **judged answer
accuracy**, and `accuracy.n == 0` in every record you have. You cannot appear in
the same table until the evaluator's `Llm` mode runs.

**Decision point — make it explicitly, not by drift:** if external baselines
cannot exist within the schedule, **drop every parity claim** and make Figure 1 an
*internal* frontier — quality vs bytes/memory and vs p95 across your own 18
ablation × 6 lane × 4 quantization points. That is still a frontier nobody has
published, at a smaller venue. Both are honest. Only one is achievable on a given
calendar.

---

### E11 — LLM judge agreement

**Why:** without a human-agreement study every accuracy number is contestable.

**Action:** ~200 stratified items, two annotators, report Cohen's κ. One day of
work that protects the entire results section.

---

## 7. Deferred — needs a quality gate before shipping

Do **not** change these defaults without a paired dev-split delta first.

| Item | Why deferred |
|---|---|
| `DEFAULT_RERANK_MODEL` → `jina-reranker-v1-turbo-en` | Already supported (`src/semantic.rs`). Cuts install from 1.21 GiB to ~150 MB. Changes retrieval quality. Note E1's prior evidence suggests rerank may not earn its cost at all — measure before optimising it. |
| Cut atomic-card embeddings | ~5.5 cards per input memory, each with its own 384-dim vector, despite being a slice of an already-embedded parent. **This is where the 51 KB/memory actually lives.** Changes retrieval quality. |
| Static embeddings (Model2Vec/potion class) | Removes the ONNX runtime entirely (~70 MB static lib). Large quality question. |
| Fix `src/db.rs` random memory ids | `rand::random::<u128>()` cascades into FTS rowids and edge ids. **Blocks any byte-identical replay claim.** Low risk, do it early. |

---

## 8. Which experiment produces which paper artifact

| Artifact | Produced by | Cuttable? |
|---|---|---|
| **Table 2** — what earns its keep | E1 | **No.** This is the paper. |
| **Table 1** — quality, sealed test | E0.4, E0.5, then unseal once | **No.** |
| **Table 3** — properties nobody reports | E5, E6, E7, E8 | Row by row; keep E5 |
| **Figure 1** — quality/cost frontier | E10, or internal fallback | No — but the fallback is legitimate |
| **Figure 2** — memory under budget | E4 | Keep the quantization inset |
| **Scale invariance** | E9 + an ARM build | Yes — cutting it cuts the L3 claim |

---

## 9. Order of work, condensed

1. **E0.1–E0.5** — prep. Blocking. Nothing before this counts.
2. **E1** — the ablation. Run it first; it decides how strong the paper is.
3. **E2** — validate the uncommitted fixes. Revert anything that costs recall.
4. **E5, E6, E7, E8** — the properties table. No GPU. Run in parallel with E1.
5. **E3, E4** — calibration and the footprint curve.
6. **E9** — CPU tier.
7. **E10** — the venue gate. Decide explicitly at this point.
8. **E11** — judge agreement, only if answer accuracy is being reported.
9. **Freeze the engine.** After this point no code changes except correctness
   fixes, or the ablation must be re-run.
10. **Unseal the test split once.** Log commit, date, operator. Regenerate Table 1.

## Related work found 2026-09-23 (web search)
- FluctlightDB (arXiv 2608.12365): embedded Rust engine for agent memory — closest prior art to our "engine" thesis. Differentiate: we contribute DB guarantees (crash/isolation tests) + structure-level cost-adjusted ablation; they don't ablate structures.
- Memanto (2604.22085): 5-stage *progressive* ablation of retrieval knobs (limits, thresholds, prompts) — not structure-level, not significance-tested.
- EverMemOS-style ablations (MemScenes/MemCells): component removal, single seed, no CIs.
- SwiftMem (2601.08160), MemForest (2605.23986): latency/build-rate focus; cite for cost axis.
- Agent Zero Memory (2608.29606): 95.6 LME / 93.6 LoCoMo end-to-end QA; retrieval-channel ablation only.
- DimMem (2605.15759), AtomMem (2606.19847), ByteRover (2604.01599): structure-rich memory designs; none report multi-seed, Holm-corrected, cost-adjusted structure ablation.

## Status 2026-09-23 — experiments stopped by user
All runs stopped on the GPU box; all run dirs pulled into benchmarks/runs/ and committed.

### LoCoMo family ablation (runs/20260922_185547_dev_groups, 3 seeds, n=432)
Families: companions={gist,keywords,relation_companions}; links={retrospective_links,derived_links,graph_edges};
routing={session_router,preferences}; facts={facts,predicate_canon,semantic_dedup,consolidation}; misc={chunks,metrics}.
| arm | Δrecall | p_adj | ΔnDCG | p_adj | Δbytes |
|---|---|---|---|---|---|
| g_links | −6.6 | <0.001 | −6.3 | <0.001 | −6% |
| g_companions | −0.8 | 1.0 | −0.7 | 1.0 | −13% |
| g_routing | −0.9 | 1.0 | +1.6 | 1.0 | −30% |
| g_facts | 0.0 | 1.0 | +1.2 | 1.0 | −6% |
| g_misc | 0.0 | 1.0 | 0.0 | 1.0 | 0 |
| pruned_all | −7.3 | <0.001 | −6.7 | <0.001 | −48% |
Family effects ~sum to pruned loss; nearly all of it is the links family.

### LoCoMo links decomposition (runs/20260923_040524_dev_links, 3 seeds, n=432)
| arm (members kept) | Δrecall | ΔnDCG (p_adj) |
|---|---|---|
| only_graph_edges | −0.9 | −0.9 (<0.001) |
| only_derived | −2.9 (p=0.54) | +1.0 (<0.001) |
| only_retrospective | −6.6 (<0.001) | −6.3 (<0.001) |
| no_retrospective | 0.0 | 0.0 |
| no_derived | −0.9 | −0.9 (<0.001) |
| no_graph_edges | −2.9 (p=0.54) | +1.0 (<0.001) |
| g_links (none) | −6.6 (<0.001) | −6.3 (<0.001) |
Finding: retrospective_links is inert (identical results with/without). graph_edges and derived_links are
substitutes — either alone recovers most of the loss, removing both costs −6.6. This is the mechanism behind
"one-at-a-time ablation is not a pruning recipe".

### LongMemEval ablation_full (runs/20260922_185404_dev_ablation_full, 1 seed)
Completed: baseline, pruned_all, g_links, g_companions, g_facts, g_routing, g_misc. no_memory_cards partial (unusable).
Not run: 17 single-structure arms. Report not yet generated for these arms (run rust_evaluator ablation-report on the dir).

### Caveats
Latency/throughput columns from 2026-09-22/23 box runs are contaminated (3 concurrent jobs); quality metrics are fine.
