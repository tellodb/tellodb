# Benchmark Guide (Current Spec)

This folder contains the benchmark wrappers and evaluator for LoCoMo and LongMemEval.

## LLM Spec (default)

- Answer model: `openai/gpt-5-mini`
- Judge model: `openai/gpt-4.1-mini`
- OpenRouter preflight: enabled by default in dataset wrappers
- Ingest concurrency: `1`
- Retrieval breadth: `top-k=8`, `max-chunks-per-session=4`


These defaults come from:

- `.env.benchmarks.example`
- `run_locomo_gpt5mini.sh`
- `run_longmemeval_gpt5mini.sh`
- `run_model_smoke.sh`
- `rust_evaluator` default CLI values

### Provider Flags

`rust_evaluator` now supports provider selection for answer and judge independently:

- Answer provider/model:
  - `--use-groq` + `--groq-model <model>`
  - default remains OpenRouter via `--openrouter-model`
- Judge provider/model:
  - `--use-groq-judge` + `--groq-judge-model <model>`
  - default remains OpenRouter via `--openrouter-judge-model`

## Tooling Spec

- Rust: `cargo run --release` (primary path)
- Python: optional, only for upstream `LongMemEval/src/*` scripts (`python=3.9` per upstream LongMemEval setup)
- Node: not required for current benchmark pipeline

## Quick Start

1. Create local env file:

```bash
cp benchmarks/.env.benchmarks.example benchmarks/.env.benchmarks.local
```

2. Set secrets in `benchmarks/.env.benchmarks.local`:

- `OPENROUTER_API_KEY`
- `TELLODB_ENGINE_URL`
- `TELLODB_API_KEY`

3. Run wrappers:

```bash
bash benchmarks/run_model_smoke.sh
bash benchmarks/run_locomo_gpt5mini.sh
bash benchmarks/run_longmemeval_gpt5mini.sh
```

### Dataset Bootstrap (Server-Friendly)

To fetch benchmark datasets on a fresh server:

```bash
bash scripts/setup_bench_data.sh
```

Variants:

```bash
# only LongMemEval
bash scripts/setup_bench_data.sh --dataset longmemeval

# only LoCoMo
bash scripts/setup_bench_data.sh --dataset locomo

# force re-download
bash scripts/setup_bench_data.sh --force
```

## Direct Evaluator Examples

Use one-line commands to avoid shell line-continuation mistakes.

```bash
cargo run --release --manifest-path benchmarks/rust_evaluator/Cargo.toml -- --dataset-kind locomo --dataset benchmarks/LoCoMo/data/locomo10.json --engine-url "$TELLODB_ENGINE_URL" --engine-api-key "$TELLODB_API_KEY" --reset-first --limit 999999 --ingest-concurrency 1 --top-k 8 --max-chunks-per-session 4 llm --openrouter-model openai/gpt-5-mini --openrouter-judge-model openai/gpt-4.1-mini
```

```bash
cargo run --release --manifest-path benchmarks/rust_evaluator/Cargo.toml -- --dataset-kind longmemeval --dataset benchmarks/LongMemEval/data/longmemeval_s_cleaned.json --engine-url "$TELLODB_ENGINE_URL" --engine-api-key "$TELLODB_API_KEY" --reset-first --limit 999999 --ingest-concurrency 1 --top-k 8 --max-chunks-per-session 4 llm --openrouter-model openai/gpt-5-mini --openrouter-judge-model openai/gpt-4.1-mini
```

```bash
# Groq for both answer and judge
cargo run --release --manifest-path benchmarks/rust_evaluator/Cargo.toml -- --dataset-kind longmemeval --dataset benchmarks/LongMemEval/data/longmemeval_s_cleaned.json --engine-url "$TELLODB_ENGINE_URL" --engine-api-key "$TELLODB_API_KEY" --reset-first --limit 999999 --ingest-concurrency 1 --top-k 8 --max-chunks-per-session 4 llm --use-groq --groq-model openai/gpt-oss-120b --use-groq-judge --groq-judge-model openai/gpt-oss-120b
```

## GPU Engine Notes

If you run the memory engine on GPU, build the engine binary with CUDA features (this is separate from the evaluator):

```bash
cargo run --release --features gpu-cuda
```

Runtime device selection:

- `TEMPORAL_MEMORY_DEVICE=auto|cpu|cuda`
- `TEMPORAL_MEMORY_CUDA_DEVICE=<gpu-index>`

Optional engine feature toggles for A/B testing:

- `TEMPORAL_MEMORY_ENABLE_FACT_GRAPH_SEMANTICS=true|false` (default: `true`)
- `TEMPORAL_MEMORY_ENABLE_CONTENT_TYPE_CHUNKING=true|false` (default: `true`)

## Common Pitfalls

- `unrecognized subcommand ' '`: usually caused by an accidental trailing space after `\` in a multiline shell command.
- `nvcc --version failed`: CUDA toolkit is missing from `PATH`; install toolkit and export CUDA bin path.
- Very slow ingest on LongMemEval: expected if each question ingests many sessions; embed time dominates ingest time.
