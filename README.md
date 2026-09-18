# Tellodb

**Tellodb** is a temporal memory database for AI agents. It stores memories as evolving evidence, tracks which facts are currently true, invalidates stale facts, computes numeric answers deterministically, and retrieves context through a hybrid of vector, lexical, graph, and temporal search.

Unlike generic memory APIs that just wrap embeddings and return stale information, Tellodb is an actual database engine built in Rust. It focuses on the core problem of long-term agent memory: **knowing what is currently true, rather than just recalling what was said in the past.**

## What Tellodb Does Better (The Benefits)

- **Temporal Truth & Fact Supersession:** Tellodb doesn't just store "persistent memory". It understands when a new fact supersedes an old one (e.g., "I moved to Seattle" invalidates "I live in Austin"). Stale facts are filtered out, giving your agents accurate context.
- **Deterministic Numeric Memory:** It computes numeric answers (counts, sums) deterministically using a metric vault, rather than relying on the LLM to guess the right number from a context window.
- **Local-First, Single-Binary Engine:** One Rust binary and one SQLite database per tenant, holding memories, facts, graph edges and the embeddings themselves. Your data stays on your machine.
- **True Hybrid Retrieval:** Per-entity vector segments (exact scans when small, HNSW when large), BM25 full-text search, graph walks and time-aware ranking, fused into one ranking.
- **Evidence-Cited Answers:** Memory responses can include evidence IDs, source snippets, and current/stale status, allowing agents to cite their sources.
- **Library, server, CLI or MCP:** Embed it in a Rust program, run the HTTP API, drive it from the command line, or plug it into an MCP client over stdio. (An OpenAI-compatible proxy is planned, not built.)

## Setting Up a GPU Box

On a fresh Linux box (Ubuntu 24.04 recommended — 22.04's glibc is too old for
the ONNX Runtime GPU binaries), one script installs system packages and Rust,
builds the engine and evaluator, downloads the models and the LongMemEval
dataset, and verifies that the CUDA execution provider actually loaded:

```bash
bash scripts/setup_gpu_box.sh
source .env.tellodb
```

It is safe to re-run; every step is skipped if already done. `SKIP_DATASETS=1`
and `SKIP_EXTRACTOR_MODEL=1` skip the two large downloads. The script ends by
printing the benchmark commands to run next.

If it warns that a GPU is present but the engine reports `device=CPU`, stop:
the CUDA provider did not load and any numbers you collect would be CPU
numbers. `tellodb doctor` reports the resolved device at any time.

## Recommended Local GPU Setup

Tellodb is intended to run locally as a Rust binary. For GPU embedding with ONNX Runtime, use Ubuntu 24.04. Ubuntu 22.04 is not recommended for the ORT GPU provider binaries because its glibc is too old.

The practical default is:

```text
Model: BAAI/bge-small-en-v1.5
Runtime: ONNX Runtime through fastembed
Execution provider: CPU or CUDA, depending on TEMPORAL_MEMORY_DEVICE and build features
Embedding dimension: 384
```

This is the current default because it keeps demos and benchmark iteration fast. `bge-base-en-v1.5` remains a stronger middle option, `bge-large-en-v1.5` is much slower on 3070 Ti/3080-class GPUs, and `Qwen3-Embedding-0.6B` is a higher-quality candidate but substantially heavier.

### Ubuntu 24.04 Bootstrap

Use an Ubuntu 24.04 CUDA image, for example on Vast.ai:

```text
vastai/base-image:cuda-12.6.3-cudnn-devel-ubuntu24.04-py310
```

Then:

```bash
cd /root
git clone <YOUR_REPO_URL> Tellodb
cd /root/Tellodb
bash scripts/linux_ubuntu
source ~/.bashrc
```

The setup script installs CUDA 12.6, cuDNN 9, TensorRT 10 runtime libraries, Rust, and writes the default Tellodb embedding environment to `~/.bashrc`.

### Run Tellodb

```bash
cd /root/Tellodb
# Linux with TensorRT:
cargo run --release --features gpu-tensorrt
# Linux with CUDA:
cargo run --release --features gpu-cuda
# macOS with CoreML:
cargo run --release --features coreml
```

Warm up before benchmarking. TensorRT may spend the first run building engines and cache files.

```bash
curl -i -X POST http://localhost:3000/warmup -H 'x-api-key: XXX1111AAA'

curl -i -X POST http://localhost:3000/warmup \
  -H 'x-api-key: XXX1111AAA'
```

Verify that TensorRT and CUDA are loaded:

```bash
grep -E 'libonnxruntime_providers_tensorrt|libonnxruntime_providers_cuda|libnvinfer' \
  /proc/$(pgrep -n )/maps | sort -u

nvidia-smi dmon -s pucm -d 1
```

You should see `libonnxruntime_providers_tensorrt.so`, `libonnxruntime_providers_cuda.so`, and `libnvinfer.so.10`. During ingest or warmup, GPU `sm` should rise above zero.

## Model Selection

Use environment variables before starting Tellodb to switch models.

### Default: BGE Small

```bash
export TEMPORAL_MEMORY_DEVICE=cuda
export TEMPORAL_MEMORY_EMBEDDING_BACKEND=candle
export TEMPORAL_MEMORY_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
export TEMPORAL_MEMORY_EMBEDDING_TOKENIZER_MODEL=BAAI/bge-small-en-v1.5
export TEMPORAL_MEMORY_EMBEDDING_DIM=384

cargo run --release --features gpu-cuda
```

### Middle Option: BGE Base

```bash
export TEMPORAL_MEMORY_EMBEDDING_BACKEND=candle
export TEMPORAL_MEMORY_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
export TEMPORAL_MEMORY_EMBEDDING_TOKENIZER_MODEL=BAAI/bge-base-en-v1.5
export TEMPORAL_MEMORY_EMBEDDING_DIM=768
```

### Higher Quality Candidate: Qwen3 0.6B

```bash
export TEMPORAL_MEMORY_DEVICE=cuda
export TEMPORAL_MEMORY_EMBEDDING_BACKEND=ort
export TEMPORAL_MEMORY_ORT_EP=cuda
export TEMPORAL_MEMORY_EMBEDDING_MODEL=onnx-community/Qwen3-Embedding-0.6B-ONNX
export TEMPORAL_MEMORY_EMBEDDING_TOKENIZER_MODEL=Qwen/Qwen3-Embedding-0.6B
export TEMPORAL_MEMORY_EMBEDDING_DIM=1024

cargo run --release --features gpu-cuda
```

Qwen3 can be tested with TensorRT, but some ONNX exports may fail TensorRT initialization with a missing-shape error. If that happens, use CUDA EP first to measure recall, then decide whether shape-inferred ONNX/TensorRT work is worth it.

When changing embedding dimensions, reset the engine state before benchmarking because existing vectors are incompatible.

## Benchmarks

Run the evaluator from the repo root while Tellodb is running.

### LongMemEval Recall

```bash
cargo run --release --manifest-path ./benchmarks/rust_evaluator/Cargo.toml -- \
  --dataset-kind longmemeval \
  --dataset ./benchmarks/LongMemEval/data/longmemeval_s_cleaned.json \
  --engine-url http://localhost:3000 \
  --engine-api-key XXX1111AAA \
  --reset-first \
  --start-index 0 --limit 500 \
  --ingest-concurrency 4 \
  --top-k 8 \
  --max-chunks-per-session 4 \
  recall
```

### LoCoMo Recall

```bash
cargo run --release --manifest-path ./benchmarks/rust_evaluator/Cargo.toml -- \
  --dataset-kind locomo \
  --dataset ./benchmarks/LoCoMo/data/locomo10.json \
  --engine-url http://localhost:3000 \
  --engine-api-key XXX1111AAA \
  --reset-first \
  --start-index 0 --limit 9999 \
  --ingest-concurrency 4 \
  --top-k 8 \
  --max-chunks-per-session 4 \
  recall
```

### LoCoMo LLM 
export OPENROUTER_API_KEY=********
export LLM_BASE_URL=https://opencode.ai/zen/go/v1/chat/completions

cargo run --release --manifest-path ./benchmarks/rust_evaluator/Cargo.toml -- \
  --dataset-kind locomo \
  --dataset ./benchmarks/LoCoMo/data/locomo10.json \
  --engine-api-key ************ \
  --reset-first --start-index 0 --limit 9999 \
  --top-k 8 --max-chunks-per-session 4 \
  llm --openrouter-model deepseek-v4-flash --openrouter-judge-model deepseek-v4-flash

  
Start with `--ingest-concurrency 4` for BGE small/base. For Qwen3 0.6B, start with `--ingest-concurrency 2`.

## Integration Modes

### 1. Embedded library

Tellodb is a library first; the server is a wrapper around it. No HTTP, no API
key, one data directory:

```rust
use tellodb::db::{Db, Memory, Query};

let db = Db::open("./agent-memory")?;
db.ingest(vec![Memory::new("alice", "I just moved to Denver.").session("chat-1", 0)])?;
let hits = db.query(Query::new("where do I live?").entity("alice").limit(5))?;
let city = db.current_fact("alice", "residence")?;
```

`Engine` is the same API for callers that already run Tokio.

### 2. Command line

```bash
tellodb serve                                   # HTTP API (default)
tellodb mcp --entity alice                      # MCP server over stdio
tellodb doctor                                  # models, device, tenant health
tellodb ingest --entity alice --session s1 turns.jsonl
tellodb query --entity alice --limit 5 "where do I live?"
```

`--data-dir DIR` selects the data directory for any command (same as
`TELLODB_DATA_DIR`). `ingest` reads JSON lines
(`{"text", "role", "session_id", "turn_index", "timestamp_ms", "kind"}`) or
plain text lines, from a file or stdin.

### 3. Model Context Protocol (MCP)

`tellodb mcp` speaks JSON-RPC over stdio, which is what Claude Desktop, Claude
Code, Cursor and Windsurf expect. Tools: `remember`, `recall` and
`current_fact`. Logs go to stderr, so stdout carries only protocol traffic.

```json
{
  "mcpServers": {
    "tellodb": {
      "command": "/path/to/tellodb",
      "args": ["--data-dir", "/path/to/agent-memory", "mcp", "--entity", "alice"]
    }
  }
}
```

`POST /mcp` on the server speaks the same JSON-RPC with API-key auth.

### 4. Language bindings

`crates/tellodb-ffi` exposes a C ABI (`crates/tellodb-ffi/tellodb.h`), and
`bindings/python/tellodb.py` wraps it with ctypes only — no build step beyond
the library:

```bash
cargo build --release -p tellodb-ffi
PYTHONPATH=bindings/python python3 -c "
from tellodb import Tellodb
with Tellodb('./agent-memory') as db:
    db.remember('alice', 'I live in Lisbon.')
    print(db.recall('where do I live?', entity_id='alice'))"
```

A Node binding over the same C ABI is not written yet.

### 5. Drop-in proxy (planned)

An OpenAI-compatible route that injects memory context into chat completions
is designed but not implemented; there is no `/v1/chat/completions` endpoint
today.

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `TELLODB_DATA_DIR` | `.` | Data directory (databases, caches) |
| `TELLODB_API_KEY` | — | Required to `serve`; unused by the library and CLI |
| `TELLODB_THREADS` | physical cores | Thread budget for models and rayon |
| `TEMPORAL_MEMORY_DEVICE` | cpu | `cuda`, `coreml` or cpu |
| `TELLODB_EMBED_TEXT` | `context` | What gets embedded: `legacy`, `turn`, `context` |
| `TELLODB_CONTEXT_WINDOW` | 1 | Neighbouring turns added in `context` mode |
| `TELLODB_RERANK` | auto | `off` skips loading the cross-encoder |
| `TELLODB_RERANK_POLICY` | `heuristic` | `always`, `gate` (confidence-gated) or `heuristic` |
| `TELLODB_RERANK_MARGIN` | 0.05 | Gate: rerank when top-1 and top-5 are this close |
| `TELLODB_RERANK_MODEL` | bge-reranker-base | Cross-encoder, or `none` |
| `TELLODB_VECTOR_QUANT` | `f32` | Segment precision: `f32`, `f16`, `i8`, `binary` |
| `TELLODB_FLAT_THRESHOLD` | 20000 | Above this many vectors an entity gets its own HNSW |
| `TELLODB_DISABLE` | — | Comma-separated derived structures to switch off (ablations) |
| `TELLODB_EXTRACTOR` | `rules` | Fact extraction tier: `rules` or `encoder` (GLiNER spans) |
| `TELLODB_EXTRACTOR_MODEL_DIR` | — | GLiNER export directory, required by the `encoder` tier |
| `TELLODB_EXTRACTOR_LABELS` | generic attributes | Comma-separated fact slots the encoder fills |
| `TELLODB_EXTRACTOR_THRESHOLD` | 0.5 | Score floor for an extracted span |
| `TELLODB_MODEL_DIR` | — | Load the embedder from this directory instead of downloading |
| `TELLODB_EMBEDDING_CACHE_PATH` | data dir | Persistent embedding cache |

`tellodb doctor` prints the resolved configuration.

### Running without downloads

Point `TELLODB_MODEL_DIR` at a model snapshot (`onnx/model.onnx` plus the
tokenizer JSON files) to skip the Hugging Face download, or build with
`--features bundled-models` and `TELLODB_BUNDLE_DIR` set to that directory to
compile the embedder into the binary.

### Quick Platform Test
You can interact with the engine directly using the REST API.

**Signup & Get Token:**
```bash
curl -sS -X POST http://localhost:3000/platform/signup \
  -H "Content-Type: application/json" \
  -d '{"username":"demo_user","password":"demo_pass_123"}'
```

**Get User Profile:**
```bash
curl -sS http://localhost:3000/platform/profile \
  -H "Authorization: Bearer <SESSION_TOKEN>"
```

## Fact Extraction

Facts drive supersession: each one lands in a slot, and a new value for that
slot makes the old one stale. Two tiers produce them, neither using a
generative model:

- **`rules`** (default) — pattern rules over atomic claims. No model, no
  latency, but it keys facts on whole sentences, so two phrasings of one fact
  land in different slots and never supersede each other.
- **`encoder`** — GLiNER zero-shot span extraction through ONNX Runtime. The
  label set is the schema: each label is a slot, and the span the model marks
  is that slot's value. Because values are spans (`Austin`, not `I live in
  Austin`), restatements compare equal and merge into evidence while real
  changes supersede.

```bash
export TELLODB_EXTRACTOR=encoder
export TELLODB_EXTRACTOR_MODEL_DIR=~/.cache/tellodb/models/gliner_small
export TELLODB_EXTRACTOR_LABELS="city of residence,employer,job title,pet"
```

The encoder tier costs roughly 8 ms per memory on CPU and is off by default.
Selecting it without a usable model is an error at startup, never a silent
fall back to rules. See `docs/roadmap-v2.md` for measurements.

## Architecture Overview
- **Storage:** one SQLite database per tenant (WAL), holding memories, facts,
  cards, links, graph edges and the embeddings themselves.
- **Vector search:** per-entity segments loaded lazily from SQLite — exact SIMD
  scans up to `TELLODB_FLAT_THRESHOLD` vectors, a per-entity `usearch` HNSW
  above it, with optional f16/i8/binary quantization and f32 rescoring.
- **Lexical search:** SQLite FTS5 (porter, unicode61) with entity-scoped queries.
- **Temporal model:** fact version chains with validity intervals, so a query
  can ask what was true at a point in time and see what superseded a fact.
- **Inference:** local ONNX Runtime models through fastembed (CUDA, CoreML or
  CPU); embeddings are cached on disk, keyed by model and text.
- **Graph:** subject/predicate/object edges walked breadth-first with hub
  pruning, batched per level.
- **API:** `axum` and `tokio`, with the retrieval pipeline on blocking threads.

---
*Tellodb ensures your agents don't just remember everything—they know what is actually true.*
