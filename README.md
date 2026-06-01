# Tellodb

**Tellodb** is a temporal memory database for AI agents. It stores memories as evolving evidence, tracks which facts are currently true, invalidates stale facts, computes numeric answers deterministically, and retrieves context through a hybrid of vector, lexical, graph, and temporal search.

Unlike generic memory APIs that just wrap embeddings and return stale information, Tellodb is an actual database engine built in Rust. It focuses on the core problem of long-term agent memory: **knowing what is currently true, rather than just recalling what was said in the past.**

## What Tellodb Does Better (The Benefits)

- **Temporal Truth & Fact Supersession:** Tellodb doesn't just store "persistent memory". It understands when a new fact supersedes an old one (e.g., "I moved to Seattle" invalidates "I live in Austin"). Stale facts are filtered out, giving your agents accurate context.
- **Deterministic Numeric Memory:** It computes numeric answers (counts, sums) deterministically using a metric vault, rather than relying on the LLM to guess the right number from a context window.
- **Local-First, Single-Binary Engine:** Deployed via a highly performant Rust binary. Keep your data private and local, with no dependencies on complex multi-database Frankenstein architectures.
- **True Hybrid Retrieval:** Combines HNSW vector search, BM25 full-text search, graph knowledge retrieval, and time-aware ranking in one unified system.
- **Evidence-Cited Answers:** Memory responses can include evidence IDs, source snippets, and current/stale status, allowing agents to cite their sources.
- **Drop-in Proxy:** Add memory to existing OpenAI SDK apps by simply changing the base URL. Tellodb automatically injects context and forwards the request to your LLM provider.

## Recommended Local GPU Setup

Tellodb is intended to run locally as a Rust binary. For GPU embedding with ONNX Runtime, use Ubuntu 24.04. Ubuntu 22.04 is not recommended for the ORT GPU provider binaries because its glibc is too old.

The practical default is:

```text
Model: BAAI/bge-small-en-v1.5
Backend: Candle
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
cargo run --release --features gpu-tensorrt
cargo run --release --features gpu-cuda
TEMPORAL_MEMORY_DEVICE=metal TEMPORAL_MEMORY_API_KEY=XXX1111AAA cargo run --features gpu-metal --release
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

Tellodb provides three main ways to integrate with your agents:

### 1. Local Engine

Run Tellodb privately on your local machine or GPU server. This is the recommended path for privacy-focused developers and local coding agents.

### 2. Drop-In Proxy

For developers already using OpenAI-style APIs, you can add memory to your agents without rewriting your application code.

Just change your SDK's base URL to point to your Tellodb instance, and pass your Tellodb API key.
- Tellodb intercepts the request.
- Retrieves relevant memories and current facts.
- Injects the compact memory context.
- Forwards to the OpenAI (or compatible) provider and returns the response.

### 3. Model Context Protocol (MCP) Server
Tellodb is designed to integrate seamlessly into environments like Claude Desktop, Claude Code, Cursor, and Windsurf via MCP. It exposes tools like `_ingest`, `_query`, `_current_fact`, and more directly to your AI IDEs.

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

## Architecture Overview
- **Storage:** MVCC temporal storage using `redb`.
- **Vector Search:** `usearch` for extreme SIMD-accelerated HNSW indexing (up to 1,000,000 vectors).
- **Inference:** Local embedding models through ONNX Runtime with CUDA/TensorRT, with Candle still available for compatible BERT-style models.
- **Graph:** RDF Adjacency Lists stored locally for fast associative recall.
- **API:** Robust asynchronous routing with `axum` and `tokio`.

---
*Tellodb ensures your agents don't just remember everything—they know what is actually true.*
