# tellodb

tellodb is a local-first temporal memory engine for AI agents. It stores
conversation turns, facts, preferences, decisions, lessons, session summaries,
memory cards, full-text indexes, graph edges, and embeddings in tenant-scoped
SQLite databases. Retrieval combines lexical search, vector search, cards,
session routing, graph expansion, fact history, and optional reranking.

The engine keeps event time separate from ingest time, so queries can be run as
of a point in time and superseded facts can be identified instead of silently
overwriting history.

## Entry points

### HTTP server

Run the server with:

```text
cargo run -- serve
```

The default listener is `127.0.0.1:3000`. Set `TELLODB_HOST=0.0.0.0` when a
container or another machine must reach it. The HTTP API uses `TELLODB_API_KEY`
for engine routes. Platform routes also support username/password sessions and
Bearer-token logout.

### Embedded `Db`

Use `tellodb::db::Db` when the application owns the process and wants blocking
ingest and query methods without HTTP:

```rust
use tellodb::db::{Db, Memory, Query};

let db = Db::open("./agent-memory")?;
db.ingest(vec![Memory::new("alice", "I moved to Denver.")])?;
let hits = db.query(Query::new("where do I live?").entity("alice"))?;
# anyhow::Ok(())
```

Applications already running Tokio can use `tellodb::db::Engine`, which exposes
the asynchronous form used by the server and MCP adapter.

### MCP over stdio

Run the Model Context Protocol adapter with:

```text
cargo run -- mcp --entity alice
```

It reads newline-delimited JSON-RPC messages from stdin and writes responses to
stdout. Available tools are `remember`, `recall`, `get_memory`,
`explore_graph`, `fact_history`, and `current_fact`.

## Configuration

Configuration is read once at startup by `Config::from_env()`. Canonical
variables use the `TELLODB_` prefix. The `TEMPORAL_MEMORY_` names in the legacy
column remain accepted for compatibility and emit a deprecation warning; they
are scheduled for removal on 2027-01-01. When both names are set, `TELLODB_`
wins. An unset value uses the default shown below.

### Runtime and retrieval

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELLODB_DATA_DIR` | platform data directory | Root directory for tenant databases, indexes, caches, and platform state. |
| `TELLODB_DISABLE` | empty | Comma-separated ingest structures to disable: `chunks`, `gist`, `keywords`, `fact_companions`, `atomic_cards`, `event_companions`, `relation_companions`, `memory_cards`, `session_router`, `preferences`, `retrospective_links`, `derived_links`, `graph_edges`, `facts`, `semantic_dedup`, `consolidation`, `metrics`, `predicate_canon`. |
| `TELLODB_HEURISTICS` | `generic` | Heuristic profile: `generic` or `legacy-tuned`. |
| `TELLODB_LANES` | `all` | Query lanes allowlist: `vector`, `fts`, `cards`, `rerank`, `graph`, and `route`. |
| `TELLODB_RETRIEVAL_PROFILE` | `fast` | Retrieval budget profile: `fast`, `balanced`, or `research`. |
| `TELLODB_AUTO_RERANK` | unset | Optional boolean override for automatic reranking. |
| `TELLODB_SCOPED_SEMANTIC_TOP` | `3000` | Maximum scoped vector candidates. |
| `TELLODB_SCOPED_SEMANTIC_START` | `256` | Initial scoped vector search size. |
| `TELLODB_SCOPED_SEMANTIC_STEP` | `256` | Scoped vector search growth step. |
| `TELLODB_SCOPED_MIN_HITS` | unset | Minimum scoped hits before convergence can stop expansion. |
| `TELLODB_SCOPED_STOP_MAX_ATTEMPTS` | `3` | Maximum scoped vector expansion attempts. |
| `TELLODB_SCOPED_STOP_MIN_SIM` | `0.70` | Similarity required for scoped convergence. |
| `TELLODB_SCOPED_STOP_MAX_HIT_GAIN` | `2` | Maximum hit-count gain considered converged. |
| `TELLODB_SCOPED_STOP_MIN_SIM_GAIN` | `0.01` | Maximum similarity gain considered converged. |
| `TELLODB_GRAPH_SEEDS` | `24` | Graph seeds used during retrieval. |
| `TELLODB_GRAPH_MAX_DEPTH` | `2` | Maximum graph expansion depth. |
| `TELLODB_GRAPH_MAX_NODE_DEGREE` | `128` | Hub-degree cutoff for graph traversal. |
| `TELLODB_LATEST_RECENCY_WEIGHT` | `0.35` | Weight for latest-value recency. |
| `TELLODB_ENABLE_TEMPORAL_RECENCY_SCORING` | `true` | Enable temporal recency scoring. |

The following legacy retrieval and temporal aliases are accepted:
`TEMPORAL_MEMORY_RETRIEVAL_PROFILE`, `TEMPORAL_MEMORY_AUTO_RERANK`,
`TEMPORAL_MEMORY_SCOPED_SEMANTIC_TOP`, `TEMPORAL_MEMORY_SCOPED_SEMANTIC_START`,
`TEMPORAL_MEMORY_SCOPED_SEMANTIC_STEP`, `TEMPORAL_MEMORY_SCOPED_MIN_HITS`,
`TEMPORAL_MEMORY_SCOPED_STOP_MAX_ATTEMPTS`, `TEMPORAL_MEMORY_SCOPED_STOP_MIN_SIM`,
`TEMPORAL_MEMORY_SCOPED_STOP_MAX_HIT_GAIN`,
`TEMPORAL_MEMORY_SCOPED_STOP_MIN_SIM_GAIN`, and
`TEMPORAL_MEMORY_ENABLE_TEMPORAL_RECENCY_SCORING`.

### Embeddings, vectors, and reranking

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELLODB_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed embedding model identifier. |
| `TELLODB_MODEL_DIR` | unset | Local embedding model directory. |
| `TELLODB_EMBEDDING_CACHE_PATH` | runtime embedding cache | SQLite embedding-cache path. |
| `TELLODB_DEVICE` | automatic | Device selection, such as `cpu`, `cuda`, or `coreml`. |
| `TELLODB_THREADS` | host CPU count | Model and NLP thread count. |
| `TELLODB_EMBED_EXECUTORS` | `1` | Concurrent embedding executors. |
| `TELLODB_EMBED_MAX_TOKENS` | `512` | Maximum embedding input tokens. |
| `TELLODB_EMBED_BATCH` | `32` | Embedding batch size. |
| `TELLODB_EMBEDDING_DIM` | model dimension (`384` by default) | Embedding dimension and vector index width. |
| `TELLODB_QUERY_INSTRUCTION` | model default | Query instruction passed to the embedding model. |
| `TELLODB_EMBED_CACHE` | `true` | Enable the persistent embedding cache. |
| `TELLODB_EMBED_TEXT` | `context` | Embedding text mode: `legacy`, `turn`, or `context`. |
| `TELLODB_CONTEXT_WINDOW` | `1` | Neighbor-turn context window, capped at `4`. |
| `TELLODB_VECTOR_QUANT` | `f32` | Vector storage/search quantization: `f32`, `f16`, `i8`, or `binary`. |
| `TELLODB_FLAT_THRESHOLD` | `20000` | Segment size above which HNSW is used. |
| `TELLODB_RESCORE_FACTOR` | `4` | Quantized candidate rescore multiplier. |
| `TELLODB_HNSW_CONNECTIVITY` | `16` | HNSW graph connectivity. |
| `TELLODB_HNSW_EF_ADD` | `128` | HNSW construction expansion factor. |
| `TELLODB_HNSW_EF_SEARCH` | `256` | HNSW search expansion factor. |
| `TELLODB_RERANK_MODEL` | `bge-reranker-base` | Cross-encoder reranker model. |
| `TELLODB_RERANK` | `true` | Enable reranking. |
| `TELLODB_RERANK_EXECUTORS` | `1` | Concurrent reranking executors. |
| `TELLODB_RERANK_CACHE_SIZE` | `4096` | Reranking cache capacity. |
| `TELLODB_RERANK_POLICY` | `gate` | Rerank policy: `heuristic`, `always`, or `gate`. |
| `TELLODB_RERANK_MARGIN` | `0.05` | Score margin used by the rerank gate. |
| `TELLODB_RERANK_TOP` | `25` | Candidates sent to the cross-encoder, from `2` to `500`. |

Legacy embedding and reranking aliases are accepted for
`EMBEDDING_MODEL`, `DEVICE`, `EMBED_EXECUTORS`, `RERANK_EXECUTORS`, and
`RERANK_CACHE_SIZE` using the corresponding `TEMPORAL_MEMORY_` prefix.

### Server and ingest

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELLODB_HOST` | `127.0.0.1` | HTTP bind host. |
| `TELLODB_PORT` | `3000` | HTTP bind port. `PORT` is also accepted as a deployment fallback. |
| `TELLODB_REQUEST_TIMEOUT_SECS` | `30` | Timeout for timed HTTP routes. |
| `TELLODB_TRUST_PROXY` | `false` | Trust forwarded client-address headers. |
| `TELLODB_CORS_ALLOW_ORIGINS` | `https://tellodb.com` | Comma-separated CORS origins. |
| `TELLODB_API_KEY` | unset in release; test key in debug | Global engine API key. |
| `TELLODB_ML_INTENT` | `false` | Enable the optional ML intent classifier. |
| `TELLODB_EXTRACTOR` | `rules` | Entity extractor: `rules` or `encoder`. |
| `TELLODB_EXTRACTOR_LABELS` | empty | Comma-separated encoder labels. |
| `TELLODB_EXTRACTOR_MODEL_DIR` | unset | Local encoder model directory. |
| `TELLODB_EXTRACTOR_THRESHOLD` | `0.5` | Encoder extraction threshold from `0.0` to `1.0`. |
| `TELLODB_PREDICATE_TAU` | `0.86` | Predicate canonicalization similarity threshold. |

Legacy aliases are accepted for `HOST`, `PORT`, `CORS_ALLOW_ORIGINS`,
`API_KEY`, and `ML_INTENT` using the `TEMPORAL_MEMORY_` prefix. `PORT` is the
only non-prefixed fallback.

The `ranking` and `scoring` fields in `Config` currently use their typed
defaults; they are not populated from environment variables.

## HTTP API

Unless marked public, routes under the protected surface require the engine
API key or an authenticated platform session. Responses for deprecated aliases
include `Deprecation: true`.

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| GET | `/healthz` | public | Liveness probe. |
| GET | `/metrics` | public | Prometheus-style metrics. |
| POST | `/signup` | public | Create a platform user and session. |
| POST | `/login` | public | Create a platform session. |
| POST | `/logout` | bearer session | Revoke the current platform session. |
| GET | `/me` | bearer session | Current platform user. |
| POST | `/api-keys` | bearer session | Create a user API key. |
| GET | `/api-keys` | bearer session | List user API keys. |
| POST | `/api-keys/{prefix}` | bearer session | Revoke a user API key. |
| GET | `/stats` | bearer session | Platform usage statistics. |
| GET | `/profile` | bearer session | Platform profile. |
| GET | `/health` | engine auth | Detailed health status. |
| GET | `/version` | engine auth | Engine, model, feature, lane, and ranking metadata. |
| GET | `/status` | engine auth | Engine status and data root. |
| POST | `/warmup` | engine auth | Warm embedding models. |
| POST | `/ingest` | engine auth | Ingest one memory payload. |
| POST | `/ingest/batch` | engine auth | Ingest a batch. |
| POST | `/batch-ingest` | engine auth, deprecated | Alias for `/ingest/batch`. |
| POST | `/memory/inspect` | engine auth, deprecated | Inspect one memory. |
| POST | `/v1/memory/inspect` | engine auth | Inspect one memory. |
| POST | `/memory/delete` | engine auth, deprecated | Delete one memory. |
| POST | `/v1/memory/delete` | engine auth | Delete one memory. |
| POST | `/query` | engine auth | Hybrid memory query. |
| POST | `/query/semantic` | engine auth, deprecated | Alias for `/query`. |
| POST | `/graph/query` | engine auth | Query graph edges. |
| POST | `/graph/walk` | engine auth | Walk graph edges. |
| POST | `/graph/export` | engine auth | Export graph edges. |
| POST | `/analytics/query` | engine auth | Query extracted metrics. |
| GET | `/facts/current` | engine auth | Resolve the current value of a fact. |
| GET | `/facts/history` | engine auth | Read fact versions over time. |
| POST | `/mcp` | engine auth | MCP JSON-RPC over HTTP. |
| POST | `/reset` | engine auth, deprecated | Clear the current tenant. |
| POST | `/admin/reset` | engine auth, deprecated | Alias for tenant reset. |
| POST | `/v1/admin/reset` | engine auth | Canonical tenant reset. |
| GET | `/admin/clusters/{cluster_id}/stats` | global engine key | Cluster statistics. |
| GET | `/admin/clusters/{cluster_id}/storage-stats` | global engine key | Detailed storage statistics. |
| GET | `/admin/clusters/{cluster_id}/graph-edges` | global engine key | Cluster graph edges. |
| GET | `/admin/stats/hardware` | global engine key | Hardware telemetry. |
| POST | `/admin/api_keys` | global engine key | Inject an API key. |
| DELETE | `/admin/api_keys/{key_id}` | global engine key | Revoke an API key. |

## Storage and data model

Each tenant has a SQLite database under the configured data root. SQLite is the
source of truth for observations, fact versions, cards, FTS rows, graph edges,
metrics, sessions, and vector lookup rows. Vector segments and embedding caches
are rebuildable accelerators. Ingest writes the primary row and derived records
atomically where the repository supports the operation; retries are safe for
stable memory IDs and content hashes.

`MemoryId` is structured as entity, session, turn, and optional derived tags.
Use the typed `MemoryId` API when constructing or parsing IDs instead of
splitting the rendered string yourself.

## Development

```text
cargo fmt --all -- --check
cargo check --all-targets
cargo test --all-targets --no-fail-fast
cargo clippy --all-targets -- -D warnings
```

The CI workflow also runs pedantic Clippy checks, checks for reintroduced
crate-level blanket allows, and executes all targets.
