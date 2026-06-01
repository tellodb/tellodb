#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ENV_FILE="${BENCHMARK_ENV_FILE:-$SCRIPT_DIR/.env.benchmarks.local}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
ENGINE_URL="${TELLODB_ENGINE_URL:-http://localhost:3000}"
ENGINE_API_KEY="${TELLODB_API_KEY:-XXX1111AAA}"
DATASET_PATH="${LOCOMO_DATASET:-$SCRIPT_DIR/LoCoMo/data/locomo10.json}"
ANSWER_MODEL="${OPENROUTER_MODEL:-openai/gpt-5-mini}"
JUDGE_MODEL="${OPENROUTER_JUDGE_MODEL:-openai/gpt-4.1-mini}"
OPENROUTER_PREFLIGHT="${OPENROUTER_PREFLIGHT:-1}"
INGEST_CONCURRENCY="${INGEST_CONCURRENCY:-1}"
TOP_K="${TOP_K:-8}"
MAX_CHUNKS_PER_SESSION="${MAX_CHUNKS_PER_SESSION:-4}"
READER_MODE="${READER_MODE:-con-separate}"
LIMIT="${LIMIT:-999999}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_JSONL="$OUTPUT_DIR/locomo_${TIMESTAMP}.jsonl"

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "LoCoMo dataset not found at: $DATASET_PATH" >&2
  echo "Download locomo10.json from https://github.com/snap-research/locomo and place it under benchmarks/LoCoMo/data/." >&2
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY must be set." >&2
  exit 1
fi

openrouter_preflight() {
  local model="$1"
  local response
  local payload

  payload=$(cat <<JSON
{"model":"$model","messages":[{"role":"user","content":"Reply with OK only."}],"temperature":0.0,"max_tokens":16}
JSON
)

  if ! response="$(curl -fsS https://openrouter.ai/api/v1/chat/completions \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    -H "Content-Type: application/json" \
    -d "$payload")"; then
    echo "OpenRouter preflight failed for model: $model" >&2
    exit 1
  fi

  if command -v jq >/dev/null 2>&1; then
    local text
    local has_choices
    text="$(printf '%s' "$response" | jq -r '.choices[0].message.content // .choices[0].text // empty' 2>/dev/null)"
    has_choices="$(printf '%s' "$response" | jq -r 'has("choices")' 2>/dev/null)"
    if [[ "$has_choices" != "true" ]]; then
      echo "OpenRouter preflight returned no choices for model: $model" >&2
      printf '%s\n' "$response" | head -c 800 >&2
      echo >&2
      exit 1
    fi
    if [[ -z "$text" || "$text" == "null" ]]; then
      echo "OpenRouter preflight OK for $model (accepted; no final text in tiny probe)"
      return
    fi
    echo "OpenRouter preflight OK for $model: $text"
    return
  fi

  if [[ "$response" != *'"choices"'* ]]; then
    echo "OpenRouter preflight returned an unexpected payload for model: $model" >&2
    printf '%s\n' "$response" | head -c 800 >&2
    echo >&2
    exit 1
  fi

  echo "OpenRouter preflight OK for $model"
}

if [[ "$OPENROUTER_PREFLIGHT" != "0" ]]; then
  openrouter_preflight "$ANSWER_MODEL"
  if [[ "$JUDGE_MODEL" != "$ANSWER_MODEL" ]]; then
    openrouter_preflight "$JUDGE_MODEL"
  fi
fi

mkdir -p "$OUTPUT_DIR"

exec cargo run --release --manifest-path "$SCRIPT_DIR/rust_evaluator/Cargo.toml" -- \
  --dataset-kind locomo \
  --dataset "$DATASET_PATH" \
  --engine-url "$ENGINE_URL" \
  --engine-api-key "$ENGINE_API_KEY" \
  --reset-first \
  --limit "$LIMIT" \
  --ingest-concurrency "$INGEST_CONCURRENCY" \
  --top-k "$TOP_K" \
  --max-chunks-per-session "$MAX_CHUNKS_PER_SESSION" \
  --reader-mode "$READER_MODE" \
  llm \
  --openrouter-model "$ANSWER_MODEL" \
  --openrouter-judge-model "$JUDGE_MODEL" \
  --output-jsonl "$OUTPUT_JSONL" \
  "$@"
