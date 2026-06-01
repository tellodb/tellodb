#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ENV_FILE="${BENCHMARK_ENV_FILE:-$SCRIPT_DIR/.env.benchmarks.local}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

ENGINE_URL="${TELLODB_ENGINE_URL:-http://localhost:3000}"
ENGINE_API_KEY="${TELLODB_API_KEY:-XXX1111AAA}"
ANSWER_MODEL="${OPENROUTER_MODEL:-openai/gpt-5-mini}"
JUDGE_MODEL="${OPENROUTER_JUDGE_MODEL:-openai/gpt-4.1-mini}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY must be set." >&2
  exit 1
fi

exec cargo run --release --manifest-path "$SCRIPT_DIR/rust_evaluator/Cargo.toml" -- \
  --engine-url "$ENGINE_URL" \
  --engine-api-key "$ENGINE_API_KEY" \
  --reset-first \
  --top-k 4 \
  --max-chunks-per-session 2 \
  smoke \
  --openrouter-model "$ANSWER_MODEL" \
  --openrouter-judge-model "$JUDGE_MODEL" \
  "$@"
