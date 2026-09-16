#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENGINE_BIN="${REPO_ROOT}/target/release/tellodb"
EVALUATOR_BIN="${REPO_ROOT}/benchmarks/rust_evaluator/target/release/rust_evaluator"
SYNTH_BIN="${REPO_ROOT}/benchmarks/rust_evaluator/target/release/synth"

ENGINE_PORT="${PORT:-3000}"
ENGINE_URL="http://localhost:${ENGINE_PORT}"
ENGINE_API_KEY="${TELLODB_API_KEY:-XXX1111AAA}"
RUNS_DIR="${REPO_ROOT}/benchmarks/runs"
SYNTH_DIR="$(mktemp -d /tmp/tellodb_synth.XXXXXX)"
DATA_DIR="$(mktemp -d /tmp/tellodb_baseline_data.XXXXXX)"

mkdir -p "$RUNS_DIR"

cleanup() {
    echo "Stopping Tellodb engine..."
    if [[ -n "${ENGINE_PID:-}" ]] && kill -0 "$ENGINE_PID" 2>/dev/null; then
        kill "$ENGINE_PID" 2>/dev/null || true
        wait "$ENGINE_PID" 2>/dev/null || true
    fi
    rm -rf "$SYNTH_DIR"
    rm -rf "$DATA_DIR"
}
trap cleanup EXIT INT TERM

echo "Starting Tellodb engine on port ${ENGINE_PORT} with data dir ${DATA_DIR}..."
TELLODB_DATA_DIR="$DATA_DIR" \
PORT="$ENGINE_PORT" \
TEMPORAL_MEMORY_API_KEY="$ENGINE_API_KEY" \
"$ENGINE_BIN" &
ENGINE_PID=$!

echo "Waiting for engine healthz..."
for i in {1..60}; do
    if curl -s -f "${ENGINE_URL}/healthz" >/dev/null 2>&1; then
        echo "Engine healthy!"
        break
    fi
    sleep 1
done

if ! curl -s -f "${ENGINE_URL}/healthz" >/dev/null 2>&1; then
    echo "Engine failed to become healthy within 60s" >&2
    exit 1
fi

echo "Warming up engine..."
curl -s -X POST "${ENGINE_URL}/warmup" -H "x-api-key: ${ENGINE_API_KEY}" >/dev/null

echo "=== Running LongMemEval-S (dev split, 3 runs) ==="
for run in 1 2 3; do
    echo "--- LongMemEval-S run $run/3 ---"
    "$EVALUATOR_BIN" \
        --dataset-kind longmemeval \
        --split dev \
        --engine-url "$ENGINE_URL" \
        --engine-api-key "$ENGINE_API_KEY" \
        --runs-dir "$RUNS_DIR" \
        --reset-first \
        recall
done

echo "=== Running LoCoMo (dev split, 3 runs) ==="
for run in 1 2 3; do
    echo "--- LoCoMo run $run/3 ---"
    "$EVALUATOR_BIN" \
        --dataset-kind locomo \
        --split dev \
        --engine-url "$ENGINE_URL" \
        --engine-api-key "$ENGINE_API_KEY" \
        --runs-dir "$RUNS_DIR" \
        --reset-first \
        recall
done

echo "=== Running Synthetic Recall (1k, 10k, 100k memories) ==="
echo "Generating synthetic 1k memories..."
"$SYNTH_BIN" --entities 50 --memories-per-entity 20 --seed 101 --output "$SYNTH_DIR/synth_1k.json"
echo "Evaluating synthetic 1k..."
"$EVALUATOR_BIN" \
    --dataset-kind longmemeval \
    --dataset "$SYNTH_DIR/synth_1k.json" \
    --split all \
    --engine-url "$ENGINE_URL" \
    --engine-api-key "$ENGINE_API_KEY" \
    --runs-dir "$RUNS_DIR" \
    --reset-first \
    recall

echo "Generating synthetic 10k memories..."
"$SYNTH_BIN" --entities 200 --memories-per-entity 50 --seed 102 --output "$SYNTH_DIR/synth_10k.json"
echo "Evaluating synthetic 10k..."
"$EVALUATOR_BIN" \
    --dataset-kind longmemeval \
    --dataset "$SYNTH_DIR/synth_10k.json" \
    --split all \
    --engine-url "$ENGINE_URL" \
    --engine-api-key "$ENGINE_API_KEY" \
    --runs-dir "$RUNS_DIR" \
    --reset-first \
    recall

echo "Generating synthetic 100k memories..."
"$SYNTH_BIN" --entities 1000 --memories-per-entity 100 --seed 103 --output "$SYNTH_DIR/synth_100k.json"
echo "Evaluating synthetic 100k..."
"$EVALUATOR_BIN" \
    --dataset-kind longmemeval \
    --dataset "$SYNTH_DIR/synth_100k.json" \
    --split all \
    --engine-url "$ENGINE_URL" \
    --engine-api-key "$ENGINE_API_KEY" \
    --runs-dir "$RUNS_DIR" \
    --reset-first \
    recall

echo "=== Generating BASELINE.md ==="
"$EVALUATOR_BIN" report "$RUNS_DIR"/*.json > "$REPO_ROOT/benchmarks/BASELINE.md"
echo "Baseline written to $REPO_ROOT/benchmarks/BASELINE.md"
