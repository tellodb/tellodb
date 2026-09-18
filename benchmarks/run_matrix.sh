#!/usr/bin/env bash
# Runs the dev split under several engine configurations and writes one
# paired-delta table per dataset against the first configuration.
#
# MATRIX holds one configuration per line: a label, then engine environment
# assignments. The engine restarts for each configuration.
#
#   MATRIX=$'baseline\nno_gist TELLODB_DISABLE=gist' NAME=demo \
#     bash benchmarks/run_matrix.sh --tier smoke
#
# run_ablation.sh and run_rerank_sweep.sh build MATRIX for common studies.
# The embedding cache is shared across configurations, so only texts that a
# configuration newly produces are embedded.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TIER="smoke"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier|-t) TIER="$2"; shift 2 ;;
    --tier=*) TIER="${1#*=}"; shift 1 ;;
    *) echo "Usage: $0 [--tier <smoke|dev>]" >&2; exit 1 ;;
  esac
done

case "$TIER" in
  smoke)
    PROFILE=fastrelease
    LIMIT=10
    SYNTH_LIMIT=60
    CONCURRENCY=1
    TELLODB_THREADS="${TELLODB_THREADS:-6}"
    TELLODB_RERANK="${TELLODB_RERANK:-off}"
    TELLODB_EMBED_MAX_TOKENS="${TELLODB_EMBED_MAX_TOKENS:-256}"
    ;;
  dev)
    PROFILE=fastrelease
    LIMIT=500
    SYNTH_LIMIT=100000
    CONCURRENCY=2
    TELLODB_THREADS="${TELLODB_THREADS:-8}"
    TELLODB_RERANK="${TELLODB_RERANK:-auto}"
    TELLODB_EMBED_MAX_TOKENS="${TELLODB_EMBED_MAX_TOKENS:-512}"
    ;;
  *) echo "Unsupported tier: $TIER (smoke|dev)" >&2; exit 1 ;;
esac

if [[ -z "${MATRIX:-}" ]]; then
    echo "MATRIX is empty; see the header of $0" >&2
    exit 1
fi
NAME="${NAME:-matrix}"
DATASETS="${DATASETS:-longmemeval,locomo,synthetic}"
TIMESTAMPS="${TIMESTAMPS:-session}"
TELLODB_EMBED_TEXT="${TELLODB_EMBED_TEXT:-context}"
TELLODB_CONTEXT_WINDOW="${TELLODB_CONTEXT_WINDOW:-1}"
CLIENT_CONTEXT="${CLIENT_CONTEXT:-window}"

export HF_HOME="${HF_HOME:-$HOME/.cache/tellodb/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache/tellodb/xdg}"
export TELLODB_EMBEDDING_CACHE_PATH="${TELLODB_EMBEDDING_CACHE_PATH:-$HOME/.cache/tellodb/embedding_cache.sqlite}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME"

ENGINE_BIN="${REPO_ROOT}/target/${PROFILE}/tellodb"
EVALUATOR_BIN="${REPO_ROOT}/benchmarks/rust_evaluator/target/release/rust_evaluator"
SYNTH_BIN="${REPO_ROOT}/benchmarks/rust_evaluator/target/release/synth"
ENGINE_PORT="${PORT:-3000}"
ENGINE_URL="http://127.0.0.1:${ENGINE_PORT}"
ENGINE_API_KEY="${TELLODB_API_KEY:-XXX1111AAA}"

RUNS_DIR="${REPO_ROOT}/benchmarks/runs/$(date +%Y%m%d_%H%M%S)_${TIER}_${NAME}"
WORK_DIR="$(mktemp -d /tmp/tellodb_ablation.XXXXXX)"
mkdir -p "$RUNS_DIR"
ENGINE_PID=""

stop_engine() {
    if [[ -n "$ENGINE_PID" ]] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        kill "$ENGINE_PID" 2>/dev/null || true
        wait "$ENGINE_PID" 2>/dev/null || true
    fi
    ENGINE_PID=""
}
cleanup() {
    stop_engine
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT INT TERM

echo "Building..."
cargo build --profile "$PROFILE" --bin tellodb --manifest-path "${REPO_ROOT}/Cargo.toml"
cargo build --release --manifest-path "${REPO_ROOT}/benchmarks/rust_evaluator/Cargo.toml"
"$SYNTH_BIN" --entities 50 --memories-per-entity 20 --seed 101 --output "$WORK_DIR/synth.json"

start_engine() {
    local label="$1" log="$2"
    shift 2
    local data_dir="$WORK_DIR/data_${label}"
    rm -rf "$data_dir"
    mkdir -p "$data_dir"
    env TELLODB_DATA_DIR="$data_dir" \
        PORT="$ENGINE_PORT" \
        TEMPORAL_MEMORY_API_KEY="$ENGINE_API_KEY" \
        TELLODB_THREADS="$TELLODB_THREADS" \
        TELLODB_RERANK="$TELLODB_RERANK" \
        TELLODB_EMBED_MAX_TOKENS="$TELLODB_EMBED_MAX_TOKENS" \
        TELLODB_EMBED_TEXT="$TELLODB_EMBED_TEXT" \
        TELLODB_CONTEXT_WINDOW="$TELLODB_CONTEXT_WINDOW" \
        "$@" \
        "$ENGINE_BIN" >"$log" 2>&1 &
    ENGINE_PID=$!
    for _ in $(seq 1 120); do
        if curl -sf "${ENGINE_URL}/healthz" >/dev/null 2>&1; then
            curl -s -X POST "${ENGINE_URL}/warmup" -H "x-api-key: ${ENGINE_API_KEY}" >/dev/null
            return 0
        fi
        if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
            echo "Engine exited during startup:" >&2
            tail -20 "$log" >&2
            exit 1
        fi
        sleep 1
    done
    echo "Engine did not become healthy; last log lines:" >&2
    tail -20 "$log" >&2
    exit 1
}

run_dataset() {
    local dataset="$1" out_dir="$2"
    local args=(--split dev --limit "$LIMIT" --timestamps "$TIMESTAMPS")
    case "$dataset" in
      longmemeval) args=(--dataset-kind longmemeval "${args[@]}") ;;
      locomo) args=(--dataset-kind locomo "${args[@]}") ;;
      synthetic) args=(--dataset-kind longmemeval --dataset "$WORK_DIR/synth.json" --split all --limit "$SYNTH_LIMIT" --timestamps session) ;;
    esac
    mkdir -p "$out_dir"
    "$EVALUATOR_BIN" "${args[@]}" \
        --tier "$TIER" \
        --client-context "$CLIENT_CONTEXT" \
        --ingest-concurrency "$CONCURRENCY" \
        --engine-url "$ENGINE_URL" \
        --engine-api-key "$ENGINE_API_KEY" \
        --runs-dir "$out_dir" \
        --reset-first \
        recall
}

IFS=, read -ra DATASET_LIST <<< "$DATASETS"
LABELS=()
while IFS= read -r line; do
    [[ -z "${line// }" ]] && continue
    read -ra parts <<< "$line"
    label="${parts[0]}"
    LABELS+=("$label")
    echo "============================================================"
    echo "Configuration: ${line}"
    echo "============================================================"
    start_engine "$label" "$RUNS_DIR/engine_${label}.log" "${parts[@]:1}"
    for dataset in "${DATASET_LIST[@]}"; do
        echo "--- ${label} / ${dataset} ---"
        run_dataset "$dataset" "$RUNS_DIR/$dataset/$label"
    done
    stop_engine
done <<< "$MATRIX"

REPORT="$RUNS_DIR/REPORT.md"
{
    echo "# ${NAME} (${TIER}, $(date -u +%Y-%m-%dT%H:%MZ))"
    echo
    echo "Embed text: ${TELLODB_EMBED_TEXT} (window ${TELLODB_CONTEXT_WINDOW}), client context: ${CLIENT_CONTEXT}, timestamps: ${TIMESTAMPS}."
    echo
    echo "Configurations:"
    echo '```'
    echo "$MATRIX"
    echo '```'
    echo "Deltas are paired over questions against \`${LABELS[0]}\`; \"drop\" means neither quality delta's 95% CI excludes zero."
    for dataset in "${DATASET_LIST[@]}"; do
        echo
        echo "## ${dataset}"
        echo
        baseline=$(ls "$RUNS_DIR/$dataset/${LABELS[0]}"/[0-9]*.json | head -1)
        others=()
        for label in "${LABELS[@]:1}"; do
            others+=("$(ls "$RUNS_DIR/$dataset/$label"/[0-9]*.json | head -1)")
        done
        "$EVALUATOR_BIN" ablation-report --baseline "$baseline" "${others[@]}"
    done
} > "$REPORT"
cat "$REPORT"
echo "Report: $REPORT"
