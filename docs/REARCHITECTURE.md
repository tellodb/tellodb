# tellodb — rearchitecture, bug register, and implementation plan

Audience: an implementing agent (small model is fine). Every task below is
self-contained: it names the files, the exact change, and a command that
proves it worked. Do the tasks in order within a phase; phases are ordered by
risk. **Never skip the acceptance check.**

Analysed at commit `8977baa`. Baseline measured on this machine:

| Fact | Value |
|---|---|
| Rust source | 28,607 lines across 59 files (`src/` + `crates/`) |
| `cargo check --all-targets` | clean |
| `cargo clippy --all-targets` (default lints) | clean |
| `cargo clippy -- -W clippy::pedantic` | ~250 warnings |
| `cargo test --lib` | 595 passed, 0 failed (~30 s) |
| Integration tests (HTTP level) | **0** |
| Distinct environment variables read | **58**, under two prefixes |
| Bare `StatusCode::INTERNAL_SERVER_ERROR` returns | 88 (36 discard the cause entirely) |

## 0. Honest verdict

This is **not** uniformly AI slop. `src/features.rs`, `src/heuristics.rs`,
`src/retrieval/lanes.rs`, `src/extract.rs` and `src/vector_index.rs` are
genuinely well-designed: real abstractions, honest doc comments, tests that
assert behaviour. Commit hygiene is good. CI exists.

The damage is concentrated and it has one shape. Features were added by
appending to whatever file was open, and removed by deleting the *reader*
while leaving the *writer*, the *config*, the *struct field* and the *test*
behind. The result:

1. **Three god modules** hold 32% of the code: `storage/tenant.rs` (4,177),
   `api/handlers/query.rs` (3,266), `api/handlers/ingest.rs` (1,804).
2. **Dependency direction is inverted.** The library API (`src/db.rs`) depends
   on the HTTP layer (`api::handlers::*`) and its error type is
   `axum::http::StatusCode`. Domain logic lives inside request handlers.
3. **Configuration is ambient global state.** 58 env vars read from arbitrary
   call depths through `OnceLock`s, so a process can only ever hold one
   configuration and tests need `thread_local` escape hatches.
4. **Dead features were half-removed**, and one `#![allow(dead_code,
   unused_imports)]` at `src/api/handlers/query.rs:1` hides it from the
   compiler *and from CI* (`cargo clippy -- -D warnings` passes only because
   of that line).

The plan below is ~7 phases. Phases 1–2 are mechanical and safe. Phase 3 is
the real rearchitecture. Phases 4–7 harden.

---

## 1. Architecture assessment, through *A Philosophy of Software Design*

### 1.1 `TenantStore` is a shallow, very wide module (Ousterhout ch. 4)

`src/storage/tenant.rs` exposes **~90 public methods** on one struct, covering
nine unrelated concerns: observations, memory cards, entity resolution, merge
proposals, session routing, bitemporal facts, core profiles, FTS, the graph,
lifecycle expiry, and stats. The interface is as large as the implementation —
the definition of a shallow module. Nothing in the type system stops the query
path from calling an ingest-only method.

Most methods are also thin wrappers over one SQL statement, so the module
provides no abstraction: callers must still know that `memory_cards` and
`fact_versions` are separate tables that must be updated together.

### 1.2 The query pipeline is temporal decomposition over a mutable blackboard (ch. 5, ch. 9)

`QueryPipelineState` (`query.rs:1349`) is a **40-field mutable struct**,
hand-initialised field by field across 150 lines even though every field has a
natural `Default`. Six free functions then mutate it in sequence:

```rust
fn plan_phase(s: &mut QueryPipelineState)
fn route_phase(s: &mut QueryPipelineState)
fn retrieval_phase(s: &mut QueryPipelineState)
fn rerank_phase(s: &mut QueryPipelineState)
fn fusion_phase(s: &mut QueryPipelineState)
fn score_phase(s: &mut QueryPipelineState) -> Result<Vec<QueryResult>, StatusCode>
```

Every phase can read and write every field. There is no way to tell from a
signature what a phase consumes or produces, no way to unit-test one phase,
and no way to reorder or skip one safely. This is exactly the "temporal
decomposition" Ousterhout warns about: the code is organised by *when things
happen* rather than by *what knowledge they encapsulate*. It is also why this
3,266-line file — the heart of the product — has **3 tests**.

### 1.3 Information leakage: the `entity::session::turn::tag` convention (ch. 5)

Memory identity is a stringly-typed convention parsed ad hoc in at least five
places with no type, no validation, and no escaping:

- `api/utils.rs:299` `session_id_from_memory_id`
- `api/utils.rs:308` `turn_index_from_memory_id` (returns `0` on parse failure)
- `api/utils.rs:317` `derived_memory_id`
- `api/utils.rs:323` `normalize_payload_identity`
- `api/utils.rs:339` `split_memory_id`
- `api/handlers/query.rs:404` `is_synthetic_query_memory` (segment 3 == `"sq"`)
- `db.rs:252` `starts_with("__pre_synth_")`

The design has already drawn blood once. `api/auth.rs:156` reads:

```rust
/// Always `None`: ... The former `"{user_id}::"` prefix also broke the
/// `entity::session::turn` id layout, making every platform user's session
/// ids resolve to the entity name.
pub fn principal_namespace_prefix(_principal: &RequestPrincipal) -> Option<String> {
    None
}
```

The fix was to neuter the function rather than fix the format — so
`scope_entity_id` and `principal_namespace_prefix` are now **dead
pass-through abstractions threaded through every handler** doing nothing
(APoSD ch. 7: pass-through methods add no value and must be removed).

### 1.4 The `retrieval::scoring` module's documented abstraction is false (ch. 13)

`src/retrieval/scoring.rs:1` claims "all tunable scoring parameters live here
so they can be reasoned about, benchmarked, and versioned". In reality there
are **85 tuning constants defined outside that file** — `lifecycle.rs` alone
has ~45, `storage/tenant.rs` has 15 session/card ranking weights, `query.rs`
has `FourSignalWeights`, plus 9 live fields in `RankingConfig`. A comment that
is wrong is worse than no comment.

### 1.5 Benchmark overfitting is compiled into the retrieval core

`src/api/plan/expansions.rs:94` defines `EXPANSION_RULES`: hand-written English
lexicons keyed to LoCoMo personas and topics (`turtle`, `dinosaur`,
`smartwatch`, `indigestion`). `src/heuristics.rs` is refreshingly honest about
this and gates the worst of it behind `Profile::LegacyTuned`. But the rules are
still `const` Rust data in the hot path: they cannot be versioned, swapped per
tenant, localised, or A/B-tested without a recompile.

### 1.6 What is already good — preserve these patterns

| File | Why it is the model to copy |
|---|---|
| `src/extract.rs` | A real `trait Extractor` seam with two implementations; deep module, narrow interface |
| `src/features.rs`, `src/retrieval/lanes.rs` | Bitset config, typo-is-a-startup-error, exhaustive `ALL` arrays with a test that proves every variant has a distinct bit |
| `src/heuristics.rs` | Documents its own epistemic limits instead of hiding them |
| `src/vector_index.rs` | Segment abstraction with a `VectorSource` trait; lazily rebuilds from SQLite, which makes the index disposable |
| `src/retrieval/mod.rs::rrf_fuse` | Cites its source, and has a test proving tie-breaks don't depend on hash order |

---

## 2. Bug register

Severity: **S1** data loss / security · **S2** wrong results or outage ·
**S3** performance or maintenance hazard.

### S1-1 — `random_token` has modulo bias (session tokens and API keys)

`src/platform.rs:744`

```rust
let idx = (rand::random::<u8>() as usize) % CHARS.len();   // CHARS.len() == 62
```

`256 % 62 == 8`, so the first 8 characters (`a`–`h`) are drawn ~1.5× more
often than the rest. This biases every `sess_*` token (48 chars), every `ak_*`
API key (48 chars) and every `usr_*`/`key_*` id. The absolute entropy loss is
small, but it is a textbook CSPRNG misuse in the one function that generates
all bearer credentials. **Fix: rejection sampling, or `% 64` over a 64-char
alphabet.**

### S1-2 — Generated memory ids can collide and silently overwrite

`src/db.rs:83`

```rust
let session = session_id.clone().unwrap_or_else(||
    format!("mem-{timestamp}-{:04x}", rand::random::<u16>()));
```

Only **16 bits** of randomness. Two sessionless memories ingested in the same
millisecond collide with probability 1/65536. `insert_observations_batch`
upserts by `memory_id`, so a collision **silently replaces** the earlier
memory. At 1,000 sessionless writes/second you lose a memory roughly every
90 seconds. **Fix: 128 bits (`rand::random::<u128>()`), and use a reserved
segment that cannot look like a session id.**

### S1-3 — `MemoryKind` is persisted via `Debug` formatting

`src/storage/tenant.rs:627,643` write `format!("{:?}", obs.kind)` into
`memories.kind`. `src/api/handlers/ingest.rs:645` feeds the same string into
`content_hash`. `src/storage/tenant.rs:3265` reads it back with
**substring matching**:

```rust
fn parse_kind_enum(kind: &str) -> MemoryKind {
    if kind.contains("Preference") { ... } else if kind.contains("Fact") { ... }
}
```

Three defects in one: (a) renaming an enum variant silently rewrites the
on-disk format; (b) renaming a variant changes every `content_hash`, so after
an upgrade every existing memory looks new and dedup breaks; (c) `contains`
means a future variant named `FactCandidate` parses as `Fact`. **Fix: an
explicit `as_str()`/`from_str()` pair with a round-trip test, exactly as
`graph.rs::EdgeType` already does.**

### S1-4 — Ingest is not atomic and has no repair path

`src/api/handlers/ingest.rs:816` `commit_batches` performs **eight independent
committed writes** in sequence: memory cards → session router → (FTS ‖ vectors)
→ preferences → memory links → predicate canon → fact versions → graph edges.
A failure at step 5 leaves steps 1–4 committed and returns `500`. The code
knows:

```rust
// Failures fail the request: the rows are already committed, and a 200 here
// would hide memories that can never be retrieved.
```

…but returning 500 does not undo them, and there is no reconciliation job, no
idempotency key, and no "indexed" watermark. A client retry re-runs all eight
steps; most are upserts so it usually converges, but nothing guarantees it.
**Fix: Phase 3 task 3.6.**

### S1-5 — Unbounded password length feeds Argon2

`src/platform.rs:172` checks only `password.len() >= 8`. The body limit is
10 MB (`api/handlers/mod.rs`), so `POST /signup` with a 10 MB password makes
the server run Argon2 over 10 MB on a request thread. `/signup` and `/login`
sit behind only the generic limiter (50 rps per address). **Fix: cap password
at 4 KB (bytes) and username at 64; add a dedicated 5 rpm limiter bucket for
`/login` and `/signup`.**

### S2-1 — An uncached `env::var` runs inside the per-candidate scoring loop

`src/api/utils.rs:64`

```rust
pub fn temporal_recency_scoring_enabled() -> bool {
    env_bool("TEMPORAL_MEMORY_ENABLE_TEMPORAL_RECENCY_SCORING", true)   // no OnceLock
}
```

Called at `src/api/handlers/query.rs:2773`, **inside** `for (mid, ts,
rrf_score) in &s.fused` (loop opens at 2699). Every candidate takes the
process-wide environment lock and allocates a `String`. With a 3,000-candidate
budget that is 3,000 lock acquisitions per query.

This directly violates the invariant documented six lines above it in the same
file (`src/api/utils.rs:37`: *"These are read on the query path, and
re-reading them per query lets a mutation mid-run change retrieval behaviour
between two queries of the same benchmark"*). Commit `362d48e` ("read tuning
settings once, not per query") converted every other setting and missed this
one. **Fix: wrap it in the existing `env_setting!` macro.**

### S2-2 — 9 of 15 documented `RankingConfig` fields are silently ignored

`src/api/types.rs:12`. Verified by grep — these are never read anywhere:

`session_ann_weight`, `event_weight`, `shadow_weight`, `facet_weight`,
`profile_weight`, `graph_weight`, `evidence_density_weight`, `stale_penalty`,
`contradiction_penalty`.

They are documented with defaults ("Weight for graph traversal proximity
score. Default: 1.0") and loaded from a user-editable
`ranking_config.json` (`src/engine.rs:83`). An operator who tunes
`graph_weight` gets no error and no effect. Worse, the six that *are* read are
then diluted by hardcoded multipliers — `card_boost * 1.3`, `card_boost *
1.15`, `session_boost * 1.5` (`query.rs:2154,2367,2370`).

### S2-3 — `migrate()` leaves a transaction open on the error path

`src/storage/tenant.rs:520-560`. The `version < 2` block issues
`BEGIN IMMEDIATE` and `DELETE FROM {table}` via `execute_batch`, then runs
fallible statement preparation and inserts with `?`, and only then
`COMMIT`. Any error between them returns with the transaction open and no
`ROLLBACK`. Today the blast radius is limited (the whole pool is dropped when
`TenantStore::new` fails), but it is one refactor away from returning a
poisoned connection to `r2d2`.

Separately: for a database at `user_version = 0`, the `version < 2` block
rebuilds `fts_memories` row by row and the `version < 3` block
(`tenant.rs:558`) then **drops the table entirely**. The whole migration is
wasted work. The migration chain was never read end to end.

### S2-4 — `#![allow(dead_code, unused_imports)]` defeats CI

`src/api/handlers/query.rs:1`. CI runs `cargo clippy --all-targets -- -D
warnings`. I removed that one line and re-ran clippy; it immediately reports:

```
warning: multiple fields are never read
   --> src/api/handlers/query.rs:715:5
    | event_limit, shadow_limit, facet_limit, scene_limit,
    | session_ann_limit, event_vector_limit, shadow_vector_limit
warning: unused import: `serde::Serialize`
warning: unused imports: `session_id_from_memory_id` and `turn_index_from_memory_id`
```

Those seven `RetrievalBudget` fields belong to features whose tables
`migrate()` explicitly `DROP`s (`temporal_events`, `shadow_questions`,
`facet_postings`, `mem_scenes` — `tenant.rs:501`). They are still initialised
in **13 separate struct literals**, which is most of why
`retrieval_budget_for_plan` is **258 lines long**.

### S2-5 — `METRICS_DDL` is defined twice and the copies can diverge

`src/storage/tenant.rs:20` defines `const METRICS_DDL`. `init_schema` at
`tenant.rs:236` contains the *same* `CREATE TABLE metrics` inline. `migrate()`
uses the constant; `init_schema` uses the literal. Editing one does not touch
the other.

### S2-6 — Blocking SQLite calls on Tokio worker threads

`src/main.rs` spawns two 300-second tickers that call `tenant.checkpoint()`
and `tenant.expire_records(now_ms)` **directly in async context** — no
`spawn_blocking`. `PRAGMA wal_checkpoint(TRUNCATE)` over N tenants blocks a
runtime worker for the duration. The final shutdown checkpoint in the same
file *does* use `spawn_blocking`, which shows the author knew.

### S2-7 — CLI turn counter advances for rows that don't use it

`src/main.rs::ingest_command`. `turn += 1` runs for every line, but
`memory.turn_index.get_or_insert(turn)` only executes inside
`if memory.session_id.is_none()`. Mixed JSON/plain input therefore produces
non-contiguous turn indices, and `session_turn_window` ordering degrades.

### S2-8 — Sessions are never deleted; there is no logout

`src/platform.rs` has no `DELETE FROM sessions` anywhere and no logout
endpoint. `resolve_session` filters on `expires_at_ms` at read time, so
expired rows accumulate forever with a 30-day TTL
(`api/handlers/platform.rs:12`), and a leaked token cannot be revoked.

### S2-9 — `"protected"` router applies no authentication

`src/api/handlers/mod.rs::build_api` builds a `Router` named `protected` whose
only layer is `rate_limit_middleware`. Authentication is done **inside each
handler** by calling `authorize_request`. I audited all 20 of them and every
one currently does it — but a new route added to that list is unauthenticated
by default, and the name says otherwise. Safety by convention, not by
construction.

### S3-1 — Unbounded `IN (...)` lists with per-length `prepare_cached`

`src/storage/tenant.rs:719, 759, 851` (and ~12 more sites) build
`IN (?,?,?…)` by allocating **one `String` per element**, then call
`prepare_cached` on the resulting SQL. Two consequences: (a) every distinct
list length is a distinct cache entry, so the 512-slot statement cache thrashes;
(b) `SQLITE_MAX_VARIABLE_NUMBER` is 32,766 — a batch ingest above that hard
errors. Some call sites chunk (`tenant.rs:796, 1181, 1921`), most do not.

### S3-2 — Whole ingest batches are deep-cloned per commit stage

`src/api/handlers/ingest.rs:816` clones `memory_card_batch`, `fts_batch`,
`vector_batch`, `preference_batch` and `memory_links_batch` to move into
`spawn_blocking`, even though `batches: &mut ArtifactBatches` is exclusively
owned. `std::mem::take` is free. Similarly `insert_observations_batch`
(`tenant.rs:672`) clones every `memory_id` `String` to build a `&`-view of a
slice it already has.

### S3-3 — Per-candidate re-tokenisation, twice

`ScorableObservation::new` (`api/plan/types.rs:105`) lowercases, tokenises,
extracts temporal terms **and** runs `extract_named_phrases` on every
candidate. It is called once per candidate at `query.rs:2716` and again at
`plan/scoring.rs:84` — the same text, tokenised twice per query, for work that
was already done at ingest.

### S3-4 — Four dead `RuntimePaths` accessors, tested only by themselves

`src/runtime_paths.rs` still models the removed redb/tantivy backends:
`temporal_db` (`temporal.redb`), `graph_db` (`graph.redb`), `fts_dir`
(`fts_tantivy`), `analytics_db`. Each has **exactly one caller: its own unit
test** (lines 265, 274, 283, 301). `ensure_dirs` still creates them, which is
why an empty `fts_tantivy/` directory sits in the repo root next to a stale
20 MB `vector.hnsw/unified.hnsw`.

### S3-5 — Two modules exist only to re-export another module

- `src/api/planner.rs` — one line, `pub use crate::api::plan::*;`, plus 98
  lines of tests for `api::plan`. Consumers then do
  `use crate::api::planner::*` on top of `api/plan/mod.rs`'s own globs, so a
  symbol passes through **two** glob re-export layers.
- `src/api/ingest_utils.rs` — six `pub use` lines plus **1,458 lines of
  tests** for code in `api/ingest/*`.
- `src/storage/tables.rs` — a single comment line,
  `// Legacy redb tables removed in favor of Sharded SQLite`, declared as
  `pub mod tables;`.

### S3-6 — `ingest_handler` and `batch_ingest_handler` are ~90% identical

`src/api/handlers/ingest.rs:234` and `:299`. The namespace-prefix block and
the entire 20-line response-header block are duplicated verbatim.

### S3-7 — Month-name tables duplicated three times

`src/lifecycle.rs:321`, `src/api/utils.rs:395`, `src/api/plan/intent.rs:15`.

### S3-8 — `MemoryCard` row mapping duplicated three times

`src/storage/tenant.rs:946`, `:3062`, `:3115` — identical 15-field
`row.get(n)?` blocks. `memory_turn_row` (`tenant.rs:3452`) already shows the
right pattern; it was just never applied here.

### S3-9 — `EdgeType` exists but the boundary still passes strings

`src/graph.rs` defines the enum, and ingest uses it. The query side compares
raw strings: `intent_weight_for_edge` (`query.rs:1128`) does
`et == "caused_by" || et == "leads_to" || et == "prefers"`, and
`parse_graph_direction_str` (`query.rs:1153`) returns `&'static str`
`"Inbound"`/`"Both"`/`"Outbound"` that `graph_query_edges` re-parses.

### S3-10 — `panic = "abort"` plus a C ABI plus 338 `unwrap()`s

`Cargo.toml` sets `panic = "abort"` for `release`. `crates/tellodb-ffi`
exposes a C ABI. Any panic in the engine therefore **aborts the host process**
of any FFI or Python consumer, and `catch_unwind` cannot help because
unwinding is disabled. There are 338 `.unwrap()` and 23 `.expect()` calls in
`src/`.

### S3-11 — Test coverage is inverted

| File | Lines | Tests |
|---|---|---|
| `src/api/handlers/query.rs` | 3,266 | **3** |
| `src/api/handlers/ingest.rs` | 1,804 | **0** |
| `src/storage/tenant.rs` | 4,177 | 19 |
| `src/api/ingest_utils.rs` | 1,467 (all tests) | 212 |
| `src/api/utils.rs` | 1,589 | 102 |
| `src/runtime_paths.rs` | 468 | 23 (4 test dead accessors) |

There are **zero** end-to-end HTTP tests; `tower` is declared as a
`dev-dependency` and never used.

### S3-12 — Stale identity and leftovers

`src/retrieval/scoring.rs:5` and `src/platform.rs:99` still say "Aletheia".
`src/storage/tenant.rs:3761` hardcodes `"Sharjeel is developing AletheiaDB"`
as a test fixture. There is **no README**, although `src/main.rs`'s `USAGE`
string tells the user to "see README". 7 `.DS_Store` files are tracked in git.

---

## 3. Target architecture

Four crates in the existing workspace. The rule is one arrow: **nothing in
`core` may name `axum`, `StatusCode`, or `std::env`.**

```
tellodb-core/     domain: ids, kinds, facts, scoring, lifecycle, planning
    |             errors: thiserror enum. config: passed in, never read.
    v
tellodb-store/    SQLite + FTS + vectors. Repository traits, one per concern.
    |
    v
tellodb-engine/   orchestration: ingest pipeline, query pipeline. The only
    |             place that composes store + models. Returns domain errors.
    v
tellodb-server/   axum router, auth, rate limiting, HTTP mapping of errors.
                  The ONLY crate that knows about StatusCode.
```

`src/db.rs` becomes the public face of `tellodb-engine`, not a client of the
HTTP layer. `crates/tellodb-ffi` depends on `tellodb-engine`.

Three cross-cutting changes make the split possible:

**(a) `Config` replaces ambient env.** One `struct Config` assembled in
`Config::from_env()` at startup, stored in `EngineState`, passed by reference.
58 env reads collapse to one. Tests construct a `Config` literal; the
`#[cfg(test)] thread_local` hacks in `heuristics.rs` disappear.

**(b) `MemoryId` becomes a type.** A struct with `entity`, `session`, `turn`,
`tag: Option<Tag>` and a single parse/format implementation. Components are
percent-escaped so `::` inside an entity id is representable. This retires
`session_id_from_memory_id`, `turn_index_from_memory_id`, `split_memory_id`,
`derived_memory_id`, `normalize_payload_identity`, `is_synthetic_query_memory`,
`scope_entity_id` and `principal_namespace_prefix` in one stroke.

**(c) The query pipeline becomes data-flow, not a blackboard.** Each stage is
a function with an explicit input and output type:

```rust
fn plan(q: &QueryRequest, cfg: &Config) -> QueryPlan;
fn route(plan: &QueryPlan, store: &dyn SessionRepo, cfg: &Config) -> RouteResult;
fn retrieve(plan: &QueryPlan, route: &RouteResult, ...) -> Candidates;
fn rerank(c: Candidates, ...) -> Candidates;
fn fuse(c: Candidates, cfg: &Config) -> Fused;
fn score(f: Fused, ctx: &ScoringContext) -> Vec<ScoredMemory>;
```

Every stage becomes testable in isolation, which is the point: it is how
`query.rs` gets from 3 tests to meaningful coverage.

---

## 4. Implementation plan

Conventions for the implementing agent:

- One task = one commit. Message format: `phase-N/task-M: <imperative>`.
- After **every** task run `cargo check --all-targets && cargo test --lib`.
  Both must be clean before the next task.
- Never change behaviour and structure in the same commit. If a task says
  "no behaviour change", the test count must not drop.
- If a task's acceptance check fails, **stop and report**. Do not improvise.

---

### Phase 1 — Stop lying to the compiler (half a day, zero behaviour change)

Do this first. It is what makes every later phase verifiable.

#### Task 1.1 — Delete the blanket `allow` and the dead code it hides

1. Delete line 1 of `src/api/handlers/query.rs`:
   `#![allow(dead_code, unused_imports)]`
2. Run `cargo clippy --all-targets`. Fix exactly what it reports:
   - Remove the unused `use serde::Serialize;`.
   - Remove `session_id_from_memory_id` and `turn_index_from_memory_id` from
     the `crate::api::utils::{...}` import list.
   - Delete these seven fields from `struct RetrievalBudget` (`query.rs:709`):
     `event_limit`, `shadow_limit`, `facet_limit`, `scene_limit`,
     `session_ann_limit`, `event_vector_limit`, `shadow_vector_limit`.
   - Delete the corresponding initialisers from **all 13**
     `RetrievalBudget { ... }` literals: 12 inside `retrieval_budget_for_plan`
     (lines 772, 792, 812, 832, 856, 876, 896, 916, 939, 959, 979, 999) and
     1 in `QueryPipelineState::new` (line 1445). `cargo check` names every
     one you miss.

**Acceptance:** `cargo clippy --all-targets -- -D warnings` exits 0, and
`grep -c '#!\[allow' src/api/handlers/query.rs` prints 0.
Expected result: `retrieval_budget_for_plan` drops from 258 to ~160 lines.

#### Task 1.2 — Collapse `retrieval_budget_for_plan` into a table

`retrieval_budget_for_plan` is a 3 (profile) × 4 (shape) matrix written as 12
struct literals. Replace it:

1. Add `#[derive(Clone, Copy)] enum QueryShape { Hard, Temporal, Numeric, Simple }`
   and `fn query_shape(plan: &QueryPlan) -> QueryShape` holding the existing
   `hard` / `temporal` / `numeric` predicates (currently `query.rs:758-770`),
   checked in that same priority order.
2. Add `const BUDGETS: [[RetrievalBudget; 4]; 3]` indexed
   `[profile as usize][shape as usize]`, filled from the existing literals.
3. `retrieval_budget_for_plan` becomes:
   `BUDGETS[profile as usize][query_shape(plan) as usize]` plus the two
   `if temporal { .. } else { .. }` adjustments for `event_limit`… which are
   now deleted by task 1.1, so it is a pure lookup.
4. Add `#[derive(Clone, Copy)]` to `RetrievalProfile` with explicit
   `Fast = 0, Balanced = 1, Research = 2`.

**Acceptance:** `cargo test --lib` still 595 passed. Add one test asserting
`retrieval_budget_for_plan` returns identical values to the pre-refactor
constants for all 12 combinations (write the expected table by hand from git
history before you change the code).

#### Task 1.3 — Derive `Default` instead of hand-zeroing 40 fields

In `query.rs`, add `#[derive(Default)]` to `QueryPlan` (`api/plan/types.rs:6`),
`QueryAdaptiveProfile`, and `RetrievalBudget`. `QueryIntent` needs
`#[derive(Default)]` with `#[default] General`. Then rewrite
`QueryPipelineState::new` (`query.rs:1394`) to:

```rust
Self {
    payload, state, tenant, limit, enable_neural_rerank,
    weights: ScoringWeights { ambiguity_delta_threshold, ..Default::default() },
    total_start: Instant::now(),
    route_start: Instant::now(),
    now_ms,
    ..Default::default()
}
```

`QueryPipelineState` itself cannot derive `Default` (it holds `EngineState`),
so keep the explicit fields above and add `#[derive(Default)]` only to the
plain-data members. If that is awkward, instead just replace each
`field: Vec::new()` / `field: 0` / `field: String::new()` line with
`..Default::default()` coverage per sub-struct.

**Acceptance:** `QueryPipelineState::new` is under 30 lines;
`cargo test --lib` unchanged.

#### Task 1.4 — Delete dead modules and dead paths

1. Delete `src/storage/tables.rs` and its `pub mod tables;` line in
   `src/storage/mod.rs`.
2. Delete from `src/runtime_paths.rs`: the fields `temporal_db`, `graph_db`,
   `fts_dir`, `analytics_db`; their accessors; the consts
   `TEMPORAL_DB_FILE`, `GRAPH_DB_FILE`, `FTS_DIR`, `ANALYTICS_DB_FILE`; their
   creation in `ensure_dirs`; and the four unit tests at lines 265, 274, 283,
   301 that are their only other callers.
3. `vector_index` has one real caller (`storage/manager.rs:30`, a legacy-file
   warning). Keep the accessor, or inline the path and delete it too —
   prefer deleting: remove the warning block, it has served its purpose.
4. Remove `tower` from `[dev-dependencies]` in `Cargo.toml` (0 uses) — but
   **re-add it in Phase 6**, where it is needed for HTTP tests. Simpler: leave
   it and note it. Choose: leave it.
5. `rm -rf fts_tantivy vector.hnsw` from the repo root; add
   `.DS_Store` to `.gitignore` and `git rm --cached` the 7 tracked ones.

**Acceptance:** `cargo test --lib` passes with 591 tests (4 fewer — the
deleted path tests). `ls fts_tantivy` fails.

#### Task 1.5 — Dissolve the two re-export shim modules

1. `src/api/planner.rs`: move its `#[cfg(test)] mod tests` block into
   `src/api/plan/mod.rs`. Delete `planner.rs` and `pub mod planner;`.
   Replace every `use crate::api::planner::` with `use crate::api::plan::`
   (`grep -rl 'api::planner' src`).
2. `src/api/ingest_utils.rs`: move its 1,458-line `mod tests` block into the
   modules it tests — `api/ingest/chunking.rs`, `companion.rs`, `datetime.rs`,
   `dialogue.rs`, `fact.rs`, `alias.rs`, `salient.rs`. Put each test next to
   the function it exercises. Delete `ingest_utils.rs` and its `pub mod` line.
   Replace `use crate::api::ingest_utils::` with the concrete module paths.
3. Remove the two `#[allow(unused_imports)]` on the globs in
   `src/api/plan/mod.rs:9,11` and fix whatever they were hiding.

**Acceptance:** `cargo test --lib` still reports the same test count (tests
moved, not deleted). `find src -name 'planner.rs' -o -name 'ingest_utils.rs'`
prints nothing.

---

### Phase 2 — Fix the bugs (one day)

Each task is independent. Each needs a regression test.

#### Task 2.1 — Fix `random_token` (S1-1)

`src/platform.rs:744`. Replace with rejection sampling:

```rust
fn random_token(length: usize) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const LIMIT: u8 = (256 / CHARS.len() * CHARS.len()) as u8; // 248
    let mut out = String::with_capacity(length);
    while out.len() < length {
        let byte = rand::random::<u8>();
        if byte < LIMIT {
            out.push(CHARS[(byte % CHARS.len() as u8) as usize] as char);
        }
    }
    out
}
```

**Test:** generate 620,000 characters, assert every character's count is
within 5% of 10,000 (`chi-square` is overkill; a range check catches the 1.5×
bias immediately — the old code fails it, the new one passes).

#### Task 2.2 — Widen generated memory ids (S1-2)

`src/db.rs:83`. Change `rand::random::<u16>()` / `{:04x}` to
`rand::random::<u128>()` / `{:032x}`. Keep the `mem-{timestamp}-` prefix.

**Test:** generate 10,000 ids with a fixed timestamp and assert all distinct.

#### Task 2.3 — Persist `MemoryKind` explicitly (S1-3)

1. In `src/storage/types.rs`, add to `impl MemoryKind`:

```rust
pub fn as_str(self) -> &'static str {
    match self {
        MemoryKind::Conversational => "conversational",
        MemoryKind::Decision => "decision",
        MemoryKind::Lesson => "lesson",
        MemoryKind::Preference => "preference",
        MemoryKind::SessionSummary => "session_summary",
        MemoryKind::Fact => "fact",
    }
}
pub fn parse(s: &str) -> MemoryKind { /* exact match on as_str(), plus the
    legacy Debug spellings "Conversational".."Fact" for existing rows;
    default Conversational */ }
pub const ALL: [MemoryKind; 6] = [ /* ... */ ];
```

2. Replace `format!("{:?}", obs.kind)` at `tenant.rs:627,643` with
   `obs.kind.as_str()`.
3. Replace `parse_kind_enum` (`tenant.rs:3265`) with `MemoryKind::parse` and
   delete it.
4. **Leave `content_hash` alone for now** — `ingest.rs:645` must keep using
   `format!("{:?}", kind)` or every existing row's hash changes and dedup
   breaks. Add a comment saying exactly that, and file it as a Phase 7
   migration.

**Test:** `for k in MemoryKind::ALL { assert_eq!(MemoryKind::parse(k.as_str()), k) }`
plus a test that each legacy Debug spelling still parses to the same variant.

#### Task 2.4 — Cache the query-loop env read (S2-1)

`src/api/utils.rs:64`. Rewrite using the macro already in that file:

```rust
env_setting!(temporal_recency_scoring_enabled_cached, bool, {
    env_bool("TEMPORAL_MEMORY_ENABLE_TEMPORAL_RECENCY_SCORING", true)
});
pub fn temporal_recency_scoring_enabled() -> bool {
    temporal_recency_scoring_enabled_cached()
}
```

Then **audit the rest**: `grep -rn 'env::var\|env_bool' src/api/ src/storage/`
and confirm every remaining call site is inside a `OnceLock`, an
`env_setting!`, or startup code. Fix any that are not.

**Test:** add `#[test] fn no_uncached_env_reads_on_the_query_path()` is not
practical; instead assert the acceptance below manually.
**Acceptance:** `grep -n 'env::var' src/api/handlers/query.rs` returns only
lines inside `OnceLock::get_or_init` closures.

#### Task 2.5 — Make `RankingConfig` honest (S2-2)

`src/api/types.rs:12`. Delete the nine never-read fields (`session_ann_weight`,
`event_weight`, `shadow_weight`, `facet_weight`, `profile_weight`,
`graph_weight`, `evidence_density_weight`, `stale_penalty`,
`contradiction_penalty`) and their `Default` entries.

Then make unknown keys loud: in `src/engine.rs:83`, parse with
`serde_json::from_str::<RankingConfig>` where `RankingConfig` carries
`#[serde(deny_unknown_fields)]`, so an operator editing a removed key gets a
startup warning naming it rather than silence.

Finally, fold the hardcoded multipliers into the config
(`query.rs:2154,2367,2370`): either add `card_boost_strong` /
`card_boost_medium` / `session_boost_routed` fields, or delete the multipliers.
Prefer adding fields — the numbers are load-bearing.

**Test:** `serde_json::from_str::<RankingConfig>(r#"{"graph_weight":1.0}"#)`
returns `Err`.

#### Task 2.6 — Make `migrate()` transactional (S2-3)

`src/storage/tenant.rs:520`. Replace the raw `BEGIN IMMEDIATE` /
`COMMIT` strings with a real `rusqlite` transaction so the `Drop` impl rolls
back on error:

```rust
let tx = conn.unchecked_transaction()?;   // conn is &Connection here
// ... all the work, using tx ...
tx.commit()?;
```

Also: move the `version < 3` check **before** the `version < 2` block, and
skip the v2 rewrite entirely when the v3 branch will drop the table anyway.
Add a comment explaining the ordering.

**Test:** `migrate_is_idempotent` — open a `TenantStore` on a temp path twice
in a row and assert both succeed and `PRAGMA user_version == 3`.

#### Task 2.7 — Single source for `METRICS_DDL` (S2-5)

`src/storage/tenant.rs`. Delete the inline `CREATE TABLE metrics (...)` and
its index from the `init_schema` batch (around line 236) and instead run
`conn.execute_batch(METRICS_DDL)?;` right after the main batch.

**Acceptance:** `grep -c 'CREATE TABLE IF NOT EXISTS metrics' src/storage/tenant.rs`
prints 1.

#### Task 2.8 — Move background SQLite work off the runtime (S2-6)

`src/main.rs::serve`. Merge the two 300-second tickers into one, and wrap the
body in `spawn_blocking`:

```rust
let maintenance = tenant_manager.clone();
tokio::spawn(async move {
    let mut ticker = interval(Duration::from_secs(300));
    loop {
        ticker.tick().await;
        let tenants = maintenance.all_tenants();
        let _ = tokio::task::spawn_blocking(move || {
            let now_ms = /* ... */;
            for tenant in tenants {
                if let Err(e) = tenant.checkpoint() { error!(...); }
                if let Err(e) = tenant.expire_records(now_ms) { error!(...); }
            }
        }).await;
    }
});
```

Also delete the `tokio::time::sleep(Duration::from_millis(500))` at the end of
`shutdown_signal()` — it delays the shutdown signal rather than draining
anything, and `with_graceful_shutdown` already drains.

**Acceptance:** `grep -c 'tokio::spawn' src/main.rs` drops from 2 to 1 in
`serve`; no `sleep` remains in `shutdown_signal`.

#### Task 2.9 — Fix the CLI turn counter (S2-7)

`src/main.rs::ingest_command`. Move `turn += 1;` so it only runs when the turn
was actually consumed, or better: track `turn` per session id in a
`HashMap<String, u32>` and assign from that map for every memory lacking an
explicit `turn_index`.

**Test:** feed three lines (plain, JSON-with-session, plain) and assert turn
indices are 0, 0, 1 for the default session.

#### Task 2.10 — Bound credential inputs and rate-limit auth (S1-5)

1. `src/platform.rs::create_user`: add
   `if password.len() > 4096 { bail!("password too long") }` and
   `if username.len() > 64 { bail!("username too long") }`. Add the same
   length guard to `login` before the hash comparison.
2. `src/api/auth.rs`: add
   `pub fn check_auth_rate_limit(state, headers, peer) -> Result<(), StatusCode>`
   using `allow_with(&format!("auth:{addr}"), 0.1, 5.0)` — 5 burst, 6/minute.
3. Call it at the top of `platform_signup_handler` and
   `platform_login_handler` (`src/api/handlers/platform.rs`).

**Test:** six rapid `check_auth_rate_limit` calls for one address — the sixth
returns `TOO_MANY_REQUESTS`.

#### Task 2.11 — Session lifecycle (S2-8)

1. `src/platform.rs`: add
   `pub fn delete_session(&self, token: &str) -> Result<()>` and
   `pub fn purge_expired_sessions(&self, now_ms: u64) -> Result<usize>`
   (`DELETE FROM sessions WHERE expires_at_ms <= ?1`).
2. Add `POST /logout` in `api/handlers/platform.rs` calling `delete_session`
   for the bearer token; register it in `build_api`.
3. Call `purge_expired_sessions` from the Phase-2.8 maintenance tick.

**Test:** create a session with `ttl_secs = 0`, assert `resolve_session` is
`None`, assert `purge_expired_sessions` removes 1 row.

#### Task 2.12 — Make authentication structural (S2-9)

Replace per-handler `authorize_request` with an axum middleware on the
`protected` router that authorises once and inserts `RequestPrincipal` into
request extensions; handlers then take
`Extension(principal): Extension<RequestPrincipal>`.

`/signup` and `/login` must move **out** of `protected` into a third router
that has the rate-limit layer but not the auth layer.

Do this task **last in Phase 2** — it touches 20 handlers. If it feels too
large, the minimal version is: rename `protected` to `rate_limited` and add a
doc comment stating that every handler there must call `authorize_request`
itself. Prefer the middleware.

**Test:** an HTTP-level test (see Phase 6) asserting `POST /query` with no
`x-api-key` returns 401. Until Phase 6 exists, assert manually with `curl`.

---

### Phase 3 — The rearchitecture (the real work, ~1 week)

#### Task 3.1 — Introduce `Config`

Create `src/config.rs`:

```rust
#[derive(Debug, Clone)]
pub struct Config {
    pub features: crate::features::Features,
    pub heuristics: crate::heuristics::Profile,
    pub lanes: crate::retrieval::lanes::Lanes,
    pub ranking: crate::api::types::RankingConfig,
    pub scoring: crate::retrieval::ScoringWeights,
    pub retrieval: RetrievalConfig,   // profile, scoped-ANN knobs, graph knobs
    pub vector: crate::vector_index::VectorConfig,
    pub embedding: EmbeddingConfig,   // model id, batch, max tokens, executors
    pub rerank: RerankConfig,         // model, policy, top, margin
    pub server: ServerConfig,         // host, port, cors, timeout, trust_proxy
    pub extractor: ExtractorConfig,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> { /* every env read, once, here */ }
}
impl Default for Config { /* the documented defaults, no env */ }
```

Then, **incrementally**, one group per commit:

1. Add `pub config: Arc<Config>` to `EngineState`; build it in
   `engine::build_state`.
2. Pick one config group (start with `retrieval`). Replace each
   `OnceLock`-backed accessor with a read from `s.state.config.retrieval.*`.
   Delete the accessor.
3. Repeat for `graph`, `rerank`, `vector`, `embedding`, `server`, `extractor`.
4. Last, fold `features`, `heuristics` and `lanes` in and delete their
   `OnceLock` statics, `init_from_env`, free `enabled()` functions and — the
   payoff — the `#[cfg(test)] thread_local TEST_PROFILE` block in
   `heuristics.rs`. Tests now pass a `Config`.

**Acceptance after the final sub-task:**
`grep -rn 'env::var' src/ | grep -v 'src/config.rs' | grep -v runtime_paths`
returns nothing.

#### Task 3.2 — Introduce `MemoryId`

Create `src/core/memory_id.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryId {
    entity: EntityId,
    session: SessionId,
    turn: u32,
    tags: Vec<Tag>,       // chunk "c3", companion "gist", card "card2", "sq"
}
impl MemoryId {
    pub fn new(entity: EntityId, session: SessionId, turn: u32) -> Self;
    pub fn derived(&self, tag: Tag) -> Self;
    pub fn parse(s: &str) -> Result<Self, MemoryIdError>;
    pub fn as_str(&self) -> &str;     // cached rendering; components escaped
    pub fn entity(&self) -> &EntityId;
    pub fn session(&self) -> &SessionId;
    pub fn turn(&self) -> u32;
    pub fn tags(&self) -> &[Tag];
}
```

Escaping: percent-encode `%` and `:` inside each component before joining with
`::`. `parse` un-escapes. This is what makes `principal_namespace_prefix`
implementable again.

Migration strategy — the DB stores strings, so:
- Keep the column as `TEXT`.
- `MemoryId::parse` must accept **unescaped legacy ids**: if a component
  contains no `%`, it round-trips unchanged. Add a property test proving that
  for all components matching `[A-Za-z0-9_-]+`, `parse(format(id)) == id` and
  the rendered string equals the legacy `a::b::0` form byte for byte.

Then delete, replacing each call site:
`session_id_from_memory_id`, `turn_index_from_memory_id`, `split_memory_id`,
`derived_memory_id`, `normalize_payload_identity`,
`is_synthetic_query_memory` (→ `id.tags().contains(&Tag::SyntheticQuery)`),
`scope_entity_id`, `principal_namespace_prefix`.

Also replace the `"__pre_synth_"` string sentinel: give `QueryResult` a
`pub origin: ResultOrigin` enum (`Stored`, `SynthesizedFact`,
`SynthesizedCard`) and filter on that in `db.rs:252` and `query.rs:2999,3020,3148`.

**Acceptance:** `grep -rn '"::"' src/` returns only `core/memory_id.rs`.
`grep -rn '__pre_synth_' src/` returns nothing.

#### Task 3.3 — Split `TenantStore` into repositories

Do **not** move code between files in the same commit as changing it. Do it in
two passes.

Pass A (mechanical, one commit per repo): split
`src/storage/tenant.rs` into `src/storage/repo/`:

| New file | Methods moved (from the `impl TenantStore` block) |
|---|---|
| `memories.rs` | `insert_observation*`, `lookup_by_*`, `get_observation*`, `memory_identity_batch`, `delete_observation`, `stored_content_hashes`, `existing_content_hashes`, `session_turn_window`, `get_turn_window`, `get_ledger_turns_batch`, `update_embeddings` |
| `cards.rs` | `ingest_cards`, `get_memory_card*`, `set_memory_card_latest_batch`, `search_memory_cards`, `get_memory_cards_batch` |
| `facts.rs` | `fact_versions_for_memories`, `fact_history`, `register_fact_versions_batch`, `get_current_fact_value`, `canonicalize_predicates`, `invalidated_*` |
| `entities.rs` | `register_entity`, `load_entity_candidates`, `set_aliases_batch`, `create_merge_proposal`, `resolve_and_propose`, `get_core_profile`, `update_core_profile`, `set_core_profile` |
| `sessions.rs` | `merge_session_router_records_batch`, `search_session_router`, `score_session_router_rows`, `sessions_in_time_window`, `entity_pivot_sessions` |
| `graph.rs` | `graph_*`, `get_link_cluster_scores`, `get_edge_cluster_neighbors*`, `set_memory_links_batch`, `get_linked_memories`, `get_all_edges` |
| `fts.rs` | `fts_search`, `fts_index_text`, `fts_index_batch`, `fts_remove_document`, `fts_clear`, `fts_quote`, `fts_entity_tok`, `fts_rowid` |
| `prefs.rs` | `set_preference_memories_batch`, `get_preference_memories` |
| `admin.rs` | `clear_all`, `db_stats`, `detailed_db_stats`, `expire_records`, `checkpoint`, `get_deletion_tombstones_for_target` |
| `schema.rs` | `init_schema`, `migrate`, `has_column`, `SCHEMA_VERSION`, all DDL consts |

Each file holds `impl TenantStore { ... }` for its own methods — Rust allows
multiple `impl` blocks. Move the matching `#[cfg(test)] mod tests` with them.
`git mv`-style: nothing but line moves. **`cargo test --lib` must still report
595 (or 591 after Phase 1).**

Pass B (one commit per repo): define a trait and a concrete type.

```rust
pub trait MemoryRepo: Send + Sync {
    fn insert_batch(&self, items: &[(u64, MemoryId, AgentObservation)]) -> Result<Vec<Option<u64>>>;
    fn get_batch(&self, ids: &[MemoryId]) -> Result<HashMap<MemoryId, AgentObservation>>;
    // ...
}
```

`TenantStore` keeps the pool and hands out `&dyn MemoryRepo` etc. Query and
ingest take the narrow traits, not `TenantStore`. This is the change that
makes the pipeline stages unit-testable with in-memory fakes.

**Acceptance:** `wc -l src/storage/repo/*.rs` — no file over 700 lines.
`grep -c 'pub fn' src/storage/repo/memories.rs` under 20.

#### Task 3.4 — Extract the domain error type and reverse the dependency

1. Create `src/error.rs`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("storage: {0}")] Storage(#[from] rusqlite::Error),
    #[error("vector index: {0}")] Vector(String),
    #[error("embedding model: {0}")] Embedding(String),
    #[error("not found: {0}")] NotFound(String),
    #[error("invalid request: {0}")] BadRequest(String),
    #[error("unauthorized")] Unauthorized,
    #[error("rate limited")] RateLimited,
    #[error(transparent)] Other(#[from] anyhow::Error),
}
pub type EngineResult<T> = Result<T, EngineError>;
```

2. Change `execute_query_pipeline` and `process_ingest_batch` (and everything
   they call) to return `EngineResult<_>` instead of `Result<_, StatusCode>`.
   Replace each of the 88 `StatusCode::INTERNAL_SERVER_ERROR` sites with the
   right variant — most become `?` on the underlying error, which is the point.
3. Add the HTTP mapping in **one** place:

```rust
impl IntoResponse for EngineError {
    fn into_response(self) -> Response {
        let status = match &self {
            EngineError::NotFound(_)     => StatusCode::NOT_FOUND,
            EngineError::BadRequest(_)   => StatusCode::BAD_REQUEST,
            EngineError::Unauthorized    => StatusCode::UNAUTHORIZED,
            EngineError::RateLimited     => StatusCode::TOO_MANY_REQUESTS,
            _ => { tracing::error!(error = ?self, "request failed");
                   StatusCode::INTERNAL_SERVER_ERROR }
        };
        (status, Json(json!({"error": self.to_string()}))).into_response()
    }
}
```

4. `src/db.rs` now returns `EngineError` instead of
   `anyhow!("query failed ({status})")`. Embedded and FFI callers finally get
   a cause.

**Acceptance:** `grep -rn 'StatusCode' src/ | grep -v 'src/api/'` returns
nothing. `grep -c 'map_err(|_|' src/` drops from 36 to under 5.

#### Task 3.5 — Turn the query blackboard into a data-flow pipeline

Move `query.rs` into `src/query/` as `plan.rs`, `route.rs`, `retrieve.rs`,
`rerank.rs`, `fuse.rs`, `score.rs`, `respond.rs`, `mod.rs`.

Define the types that flow between them (in `src/query/mod.rs`):

```rust
pub struct RouteResult { pub session_scores: HashMap<SessionId, f32>,
                         pub routed_memories: HashMap<MemoryId, f32>,
                         pub adaptive: QueryAdaptiveProfile }
pub struct Candidates  { pub semantic: Vec<(f32, Vec<RankedItem>)>,
                         pub lexical:  Vec<(f32, Vec<RankedItem>)>,
                         pub cards:    Vec<RankedItem>,
                         pub ann_raw:  Vec<(u64, f32)> }
pub struct Fused       { pub items: Vec<(MemoryId, u64, f32)>,
                         pub graph_scores: HashMap<MemoryId, f32> }
pub struct ScoringContext<'a> { /* observations, cards, invalidated, config */ }
```

Then convert one stage per commit, in this order (each is independently
testable): `plan` → `fuse` → `score` → `retrieve` → `route` → `rerank`.
`QueryDiagnostics` becomes a separate `&mut Diagnostics` argument rather than
a field of the state, and **drops its 30 redundant `_ms` fields** — keep only
the `_us` values and derive milliseconds at render time.

For each converted stage, add at least **three** tests: empty input, the
normal path, and one edge case named in this document.

**Acceptance:** `wc -l src/query/*.rs` — no file over 600 lines.
`grep -c '#\[test\]' src/query/*.rs` — at least 18 in total (up from 3).

#### Task 3.6 — Make ingest atomic where it can be, and recoverable where it cannot

SQLite writes (cards, router, preferences, links, facts, edges, predicate
canon) **can** be one transaction; FTS and the vector index cannot join it.

1. In `src/storage/repo/`, add
   `fn commit_ingest(&self, batches: &IngestBatches) -> Result<CommitOutcome>`
   which opens **one** `TransactionBehavior::Immediate` transaction and does
   every SQLite write inside it.
2. Add an `indexed` column to `memories` (migration, `SCHEMA_VERSION = 4`),
   default `0`. `commit_ingest` inserts with `indexed = 0`.
3. After the transaction commits, index FTS and vectors. On success, a second
   tiny transaction sets `indexed = 1` for those ids.
4. Add `fn reindex_unindexed(&self, limit: usize)` and call it from the
   Phase-2.8 maintenance tick. Crash recovery and partial-failure recovery are
   now the same code path.
5. `commit_batches` in `ingest.rs` becomes ~60 lines calling those two steps.
   Replace every `batches.x.clone()` with `std::mem::take(&mut batches.x)`.

**Test:** `ingest_is_atomic_across_sqlite_writes` — inject a failure in the
fact-version write (feature-gate a `#[cfg(test)]` hook), then assert that
`memory_cards` has no rows for that batch. And
`reindex_unindexed_recovers_a_partial_ingest`.

#### Task 3.7 — De-duplicate the ingest handlers (S3-6)

`ingest_handler` becomes:

```rust
pub async fn ingest_handler(state, headers, Json(payload): Json<IngestPayload>)
    -> Result<impl IntoResponse, EngineError>
{
    batch_ingest_handler(state, headers, Json(BatchIngestPayload { items: vec![payload] })).await
}
```

and the 20-line response-header block moves into
`fn ingest_timing_headers(diag: &IngestDiagnostics) -> HeaderMap`.

**Acceptance:** `ingest_handler` is under 6 lines.

---

### Phase 4 — Performance (half a day, after Phase 3)

#### Task 4.1 — One `IN`-list helper, always chunked (S3-1)

Add to `src/storage/repo/mod.rs`:

```rust
/// SQLite's parameter ceiling is 32766; chunk well under it so the prepared
/// statement cache holds one entry per chunk size, not one per call.
pub const IN_CHUNK: usize = 500;

pub fn in_placeholders(n: usize) -> String {
    let mut s = String::with_capacity(n * 2);
    for i in 0..n { if i > 0 { s.push(','); } s.push('?'); }
    s
}
```

Convert **every** `IN (...)` site to chunk at `IN_CHUNK`, padding the final
chunk by repeating its last element so only one SQL string per query shape
enters the cache. List of sites: `tenant.rs:719, 759, 796, 851, 1181, 1441,
1448, 1470, 1496, 1921, 2455` and any others `grep -n 'placeholders' ` finds.

**Test:** `lookup_by_memory_ids_batch` with 2,000 ids returns 2,000 rows.
(Today it raises `too many SQL variables` above 32,766 and thrashes the cache
well before that.)

#### Task 4.2 — Stop re-tokenising every candidate (S3-3)

`ScorableObservation::new` is called once per candidate in `score_loop`
(`query.rs:2716`) and again in `plan/scoring.rs:84`. Build it once per
candidate in `score_hydrate` and pass `&ScorableObservation` into both.
Cheaper still: cache `tokens` and `entities` on `AgentObservation` at hydrate
time.

**Benchmark before and after:** `cargo bench --bench benchmarks` — record the
numbers in the commit message.

#### Task 4.3 — Take, don't clone (S3-2)

`commit_batches` and `insert_observations_batch`, per Task 3.6 step 5.
`insert_observations_batch` should take
`&mut Vec<(u64, MemoryId, AgentObservation)>` or borrow directly instead of
building a cloned `&`-view.

---

### Phase 5 — Consistency and naming (half a day)

#### Task 5.1 — `EdgeType` at every boundary (S3-9)

- `intent_weight_for_edge` (`query.rs:1128`) takes `EdgeType`, not `&str`.
- `parse_graph_direction_str` becomes
  `enum Direction { In, Out, Both }` with `parse` and `as_str`;
  `graph_query_edges` takes `Direction`.
- Add `#[test] fn edge_type_round_trips()` over an `EdgeType::ALL` array.

#### Task 5.2 — One calendar module (S3-7)

Create `src/core/calendar.rs` holding `MONTHS: [&str; 12]`, `month_index`,
`days_in_month`, `days_since_epoch`, `month_to_ms`. Delete the three copies
(`lifecycle.rs:321`, `api/utils.rs:395`, `api/plan/intent.rs:15`) and the
hand-rolled date math currently inside `api/utils.rs:440-666`
(`parse_temporal_window` is 196 lines; the calendar half belongs here).

Consider `jiff` or `chrono` instead — but **only** if you also delete the
hand-rolled functions. Two implementations is worse than either one.

#### Task 5.3 — One `MemoryCard` row mapper (S3-8)

Add `fn memory_card_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryCard>`
next to the existing `memory_turn_row` and use it at the three sites
(`tenant.rs:946, 3062, 3115`).

#### Task 5.4 — Dissolve `api/utils.rs` (1,589 lines, no coherent subject)

Redistribute:

| To | What |
|---|---|
| `src/config.rs` | every `env_setting!` and const default (Phase 3.1 already takes most) |
| `src/core/calendar.rs` | `parse_temporal_window`, `extract_temporal_terms`, `month_to_ms`, `days_*` |
| `src/core/text.rs` | `normalize_fact_text`, `normalize_alpha_tokens`, `singularize_token`, `dedupe_preserve_order`, `has_token`, `is_low_signal_keyword`, `extract_salient_terms`, `extract_named_phrases` |
| `src/core/memory_id.rs` | the id helpers (Phase 3.2 already takes them) |
| `src/core/decay.rs` | `apply_time_decay`, `decay_policy`, `apply_decay_with_policy`, `cosine_similarity_from_distance` |
| `src/api/http.rs` | `insert_u64_header`, `insert_f32_header`, `insert_stage_timing_headers`, `elapsed_ms_and_us`, `ok_or_500` (delete `ok_or_500` — Phase 3.4 removes its reason to exist) |

Move the 102 tests with their functions. Delete `api/utils.rs`.

#### Task 5.5 — Rename the ghosts

`grep -rni aletheia src/` → `src/retrieval/scoring.rs:5`,
`src/platform.rs:99`, and the test fixture at `src/storage/tenant.rs:3761`.
Replace with `tellodb`, and replace the personal-name fixture with a neutral
one (`"Ada is building a database in Rust"`).

Pick **one** env prefix. `TELLODB_` is the product name; `TEMPORAL_MEMORY_` is
the old one. In `Config::from_env`, read `TELLODB_*` first, fall back to
`TEMPORAL_MEMORY_*` with a deprecation warning naming both, and document the
removal date. 21 vars need the alias.

#### Task 5.6 — API surface hygiene

`build_api` registers six alias routes (`/reset`, `/admin/reset`,
`/v1/admin/reset`; `/ingest/batch`, `/batch-ingest`; `/memory/*` and
`/v1/memory/*`; `/query`, `/query/semantic`). Pick the `/v1/` form as
canonical, keep the others with a `Deprecation` response header, and write
them down in the README (Task 7.1).

Replace the `UNTIMED: [&str; 9]` array plus the trailing
`|| path == "/v1/admin/reset"` special case in `request_timeout_middleware`
with a single `const UNTIMED: &[&str]` containing all ten. Read the timeout
from `Config`, not `env::var`, on every request.

---

### Phase 6 — Tests that match the risk (two days)

#### Task 6.1 — Add end-to-end HTTP tests

Create `tests/http.rs`. `tower` is already a dev-dependency.

```rust
async fn test_app() -> (Router, TempDir) { /* build_state on a tempdir, tiny
    VectorConfig, embedding model stubbed via Config */ }
```

Minimum set (one test each):
- `query_without_api_key_is_401`
- `ingest_then_query_returns_the_memory`
- `batch_ingest_and_single_ingest_agree` (same payload, same stored rows)
- `superseded_fact_is_marked_stale`
- `point_in_time_query_excludes_later_memories`
- `oversized_body_is_413`
- `login_is_rate_limited_after_five_attempts`
- `unknown_ranking_config_key_fails_startup`

#### Task 6.2 — Property tests for the new types

- `MemoryId`: `parse(format(id)) == id` for arbitrary components including
  `::`, `%`, empty strings and unicode.
- `MemoryKind`: round-trip over `ALL`, plus every legacy Debug spelling.
- `rrf_fuse`: already has one; add "adding an empty lane changes nothing" and
  "score is monotone in rank".
- `Config`: `from_env` with an empty environment equals `Config::default()`.

#### Task 6.3 — Fill the ingest test hole

`src/api/handlers/ingest.rs` has **0** tests. After Phase 3.6 splits it, add
tests for `expand_and_enrich_payloads`, `build_observations`,
`build_artifacts`, `build_memory_card_from_payload`,
`classify_retrospective_link` and `card_type_for_kind`. Target 25 tests.

#### Task 6.4 — Tighten CI

In `.github/workflows/ci.yml`, change the clippy step to:

```yaml
- run: cargo clippy --all-targets -- -D warnings -W clippy::pedantic
       -A clippy::missing_errors_doc -A clippy::missing_panics_doc
       -A clippy::module_name_repetitions -A clippy::must_use_candidate
       -A clippy::doc_markdown -A clippy::cast_precision_loss
       -A clippy::cast_possible_truncation -A clippy::cast_sign_loss
       -A clippy::cast_possible_wrap
```

Add a job that fails on a re-introduced blanket allow:

```yaml
- name: no blanket allows
  run: "! grep -rn '#!\\[allow' src/ crates/"
```

Add `cargo test --all-targets` (currently only `cargo test` runs, which does
include integration tests once `tests/` exists — verify).

---

### Phase 7 — Deferred and optional

- **7.1 Write a README.** `src/main.rs`'s `USAGE` string points at one that
  does not exist. It needs: what tellodb is, the three entry points (HTTP,
  embedded `Db`, MCP stdio), the full env-var table generated from `Config`,
  and the API route table from Task 5.6.
- **7.2 `content_hash` migration** (deferred from Task 2.3). Changing the hash
  input invalidates every stored hash. Do it as a `SCHEMA_VERSION` bump that
  recomputes `memories.content_hash` for every row in one pass, with a test
  proving dedup still works across the boundary.
- **7.3 Move `EXPANSION_RULES` out of the binary.** `api/plan/expansions.rs`'s
  lexicons become a versioned data file (`rules/expansions.v1.json`) loaded by
  `Config`, with the current table as the built-in default. This is what makes
  the benchmark-tuned rules swappable per tenant and testable without a
  recompile — and it is the honest version of what `heuristics.rs` is already
  trying to do.
- **7.4 Reconsider `panic = "abort"`** (S3-10). With a C ABI and 338
  `unwrap()`s, a panic kills the host process. Either switch the release
  profile to `panic = "unwind"` and wrap every `extern "C"` entry point in
  `catch_unwind`, or audit the `unwrap()`s on the request path to zero. Do the
  former; it is one line plus a wrapper.
- **7.5 Consider extracting the four crates** described in §3. Only worth
  doing once Phases 3–5 have already enforced the dependency direction inside
  the single crate — at that point the split is mechanical and the compiler
  proves it stayed split.

---

## 5. Rust "deslop" checklist

`dabit3/deslop` is a regex scanner written for JavaScript (`console.log`,
`response.data.data`, `=== null || === undefined`); its pattern set does not
apply to Rust. The *idea* does. These are the Rust equivalents, all of which
this codebase exhibits, with a grep that finds each one. Run them before every
PR.

| # | Slop signature | Detector |
|---|---|---|
| 1 | Blanket `allow` hiding dead code | `grep -rn '#!\[allow' src/` |
| 2 | Struct fields initialised everywhere, read nowhere | remove #1, then `cargo clippy` |
| 3 | Module that only re-exports another module | `grep -rlc 'pub use' src/ \| xargs wc -l` — flag files under 20 non-test lines |
| 4 | Config fields documented but never read | `for f in $(fields); do grep -c "\.$f" src/; done` |
| 5 | `Debug` formatting used as a serialisation format | `grep -rn 'format!("{:?}"' src/` near a SQL `execute` |
| 6 | Parallel `_ms` and `_us` (or `_str` and typed) fields | `grep -n '_ms: u64' <file> \| wc -l` vs `_us` |
| 7 | Hand-zeroed struct literal that `Default` covers | literal over 20 lines where every value is `0`/`new()` |
| 8 | Near-identical `match` arms / struct literals | the 8-line block hasher in §2 (47 hits here) |
| 9 | `env::var` below the startup layer | `grep -rn 'env::var' src/ \| grep -v config.rs` |
| 10 | Transport types in domain signatures | `grep -rn 'StatusCode' src/ \| grep -v 'src/api/'` |
| 11 | `map_err(\|_\|` — cause thrown away | `grep -rn 'map_err(|_|' src/` |
| 12 | Tests that only exercise the dead thing they test | accessor with exactly one caller, in `mod tests` |
| 13 | Comment that contradicts the code beneath it | read the doc comment on `fts_entity_tok` (`tenant.rs:72`) — it describes `fts_rowid` |
| 14 | Giant function (>150 lines) | `cargo clippy -- -W clippy::too_many_lines` |
| 15 | Stringly-typed identity with ad-hoc parsing | `grep -rn 'split("::")\|starts_with("__' src/` |

Item 13 is real and unfixed: `src/storage/tenant.rs:72-80` carries a doc
comment beginning *"Stable FTS5 rowid for a document key…"* attached to
`fts_entity_tok`, which computes no rowid. It belongs to `fts_rowid` 25 lines
below. Fix it in Phase 1.

---

## 6. Suggested order and effort

| Phase | What | Effort | Risk |
|---|---|---|---|
| 1 | Stop lying to the compiler | 0.5 d | none |
| 2 | Fix the 12 bugs | 1 d | low, each tested |
| 3 | Rearchitecture (Config, MemoryId, repos, errors, pipeline, atomic ingest) | 5–7 d | medium — do one sub-task per commit |
| 4 | Performance | 0.5 d | low |
| 5 | Consistency and naming | 0.5 d | low |
| 6 | Tests | 2 d | none |
| 7 | Deferred | — | — |

Phases 1, 2, 4, 5 and 6 are each independently shippable. Phase 3 is the only
one that must be done as a sequence, and the order inside it (3.1 Config →
3.2 MemoryId → 3.3 repos → 3.4 errors → 3.5 pipeline → 3.6 atomic ingest →
3.7 handlers) is load-bearing: each step removes a dependency the next one
would otherwise have to work around.
