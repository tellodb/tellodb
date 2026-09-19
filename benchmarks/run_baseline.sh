#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TIER="smoke"
# Which suites to run. Empty means all of them; the tier still picks the
# sizes. Iterating on one dataset should not cost a full matrix.
ONLY=""
RUNS_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier|-t)
      TIER="$2"
      shift 2
      ;;
    --tier=*)
      TIER="${1#*=}"
      shift 1
      ;;
    --only)
      ONLY="$2"
      shift 2
      ;;
    --only=*)
      ONLY="${1#*=}"
      shift 1
      ;;
    --runs)
      RUNS_OVERRIDE="$2"
      shift 2
      ;;
    --runs=*)
      RUNS_OVERRIDE="${1#*=}"
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--tier <smoke|dev|paper>] [--only <longmemeval,locomo,synth>] [--runs N]" >&2
      exit 1
      ;;
  esac
done

# `--only a,b` -> a shell function we can ask about each suite.
want_suite() {
  [[ -z "$ONLY" ]] && return 0
  local suite
  for suite in ${ONLY//,/ }; do
    [[ "$suite" == "$1" ]] && return 0
  done
  return 1
}

case "$TIER" in
  smoke)
    PROFILE=fastrelease
    LIMIT=10
    # Synthetic data is cheap to ingest; cover several entities, not one.
    SYNTH_LIMIT=60
    RUNS=1
    CONCURRENCY=1
    TELLODB_THREADS="${TELLODB_THREADS:-6}"
    TELLODB_RERANK="${TELLODB_RERANK:-off}"
    # Smoke checks that the pipeline works, not quality: shorter texts keep a
    # laptop run to minutes (attention cost grows with text length).
    TELLODB_EMBED_MAX_TOKENS="${TELLODB_EMBED_MAX_TOKENS:-256}"
    TELLODB_EMBED_BATCH="${TELLODB_EMBED_BATCH:-8}"
    RUN_SYNTH_10K=0
    RUN_SYNTH_100K=0
    ;;
  dev)
    PROFILE=fastrelease
    LIMIT=500
    SYNTH_LIMIT=100000
    RUNS=3
    CONCURRENCY=2
    TELLODB_THREADS="${TELLODB_THREADS:-8}"
    TELLODB_RERANK="${TELLODB_RERANK:-auto}"
    # Attention memory scales with batch x seq^2, and each embed executor
    # holds its own CUDA arena alongside the reranker. 32 x 512 tokens asks
    # for ~354 MB per buffer, which OOMs an 8 GB card once the embedding
    # cache is cold and inference actually runs.
    TELLODB_EMBED_MAX_TOKENS="${TELLODB_EMBED_MAX_TOKENS:-512}"
    TELLODB_EMBED_BATCH="${TELLODB_EMBED_BATCH:-8}"
    RUN_SYNTH_10K=1
    RUN_SYNTH_100K=0
    ;;
  paper)
    PROFILE=release
    LIMIT=500
    SYNTH_LIMIT=100000
    RUNS=5
    CONCURRENCY=4
    TELLODB_THREADS="${TELLODB_THREADS:-8}"
    TELLODB_RERANK="${TELLODB_RERANK:-auto}"
    # Attention memory scales with batch x seq^2, and each embed executor
    # holds its own CUDA arena alongside the reranker. 32 x 512 tokens asks
    # for ~354 MB per buffer, which OOMs an 8 GB card once the embedding
    # cache is cold and inference actually runs.
    TELLODB_EMBED_MAX_TOKENS="${TELLODB_EMBED_MAX_TOKENS:-512}"
    TELLODB_EMBED_BATCH="${TELLODB_EMBED_BATCH:-8}"
    RUN_SYNTH_10K=1
    RUN_SYNTH_100K=1
    ;;
  *)
    echo "Unsupported tier: $TIER. Choose from smoke, dev, paper." >&2
    exit 1
    ;;
esac

echo "============================================================"
echo "Tellodb Baseline Runner: Tier = ${TIER}"
echo "Threads = ${TELLODB_THREADS}, Rerank = ${TELLODB_RERANK}, Max tokens = ${TELLODB_EMBED_MAX_TOKENS}, Limit = ${LIMIT}, Runs = ${RUNS}"
echo "Embed text = ${TELLODB_EMBED_TEXT:-context} (window ${TELLODB_CONTEXT_WINDOW:-1}), Client context = ${CLIENT_CONTEXT:-window}"
echo "============================================================"

# Persistent cache directories to avoid re-downloading model weights
export HF_HOME="${HF_HOME:-$HOME/.cache/tellodb/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache/tellodb/xdg}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME"
# Embedding cache outlives the throwaway data dir, so re-runs skip embedding.
export TELLODB_EMBEDDING_CACHE_PATH="${TELLODB_EMBEDDING_CACHE_PATH:-$HOME/.cache/tellodb/embedding_cache.sqlite}"
# wallclock (comparable with older runs) or session (real event dates) for
# LongMemEval/LoCoMo. Synthetic runs always use session dates: their
# current-value and as-of questions are meaningless without real event times.
if [[ -n "$RUNS_OVERRIDE" ]]; then
    RUNS="$RUNS_OVERRIDE"
fi

# Retrieval lanes and heuristics profile. Both are recorded in the run record
# via /version, so a baseline cannot be mislabelled.
TELLODB_LANES="${TELLODB_LANES:-all}"
TELLODB_HEURISTICS="${TELLODB_HEURISTICS:-generic}"
# `off` makes the run measure encoder inference instead of a cache lookup.
TELLODB_EMBED_CACHE="${TELLODB_EMBED_CACHE:-on}"

# Real event times, not ingest time. `wallclock` stamps every memory at
# roughly now, which makes temporal decay and recency scoring inert and
# flatters the engine by ~2 recall points; it is not a setting any result
# should be reported under. run_matrix.sh already defaulted to `session`, and
# the mismatch made runs from the two scripts silently incomparable.
TIMESTAMPS="${TIMESTAMPS:-session}"
# Memory representation (WP2): what the engine embeds (legacy|turn|context),
# how many neighbouring turns `context` adds, and whether the evaluator also
# prepends its own header/window to each turn (window|off).
TELLODB_EMBED_TEXT="${TELLODB_EMBED_TEXT:-context}"
TELLODB_CONTEXT_WINDOW="${TELLODB_CONTEXT_WINDOW:-1}"
# The engine assembles context itself from stored turns
# (TELLODB_EMBED_TEXT=context), so a client-side window applies it twice: each
# payload is already a 3-turn block, and the engine then windows those blocks
# against their neighbours. `off` sends the turn as written and lets the
# engine do it once.
CLIENT_CONTEXT="${CLIENT_CONTEXT:-off}"

ENGINE_BIN="${REPO_ROOT}/target/${PROFILE}/tellodb"
EVALUATOR_BIN="${REPO_ROOT}/benchmarks/rust_evaluator/target/release/rust_evaluator"
SYNTH_BIN="${REPO_ROOT}/benchmarks/rust_evaluator/target/release/synth"

ENGINE_PORT="${PORT:-3000}"
ENGINE_URL="http://127.0.0.1:${ENGINE_PORT}"
ENGINE_API_KEY="${TELLODB_API_KEY:-XXX1111AAA}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUNS_DIR="${REPO_ROOT}/benchmarks/runs/${TIMESTAMP}_${TIER}"
SNAPSHOT_DIR="${REPO_ROOT}/benchmarks/snapshots/${TIMESTAMP}_${TIER}"
SYNTH_DIR="$(mktemp -d /tmp/tellodb_synth.XXXXXX)"
DATA_DIR="$(mktemp -d /tmp/tellodb_baseline_data.XXXXXX)"

mkdir -p "$RUNS_DIR" "$SNAPSHOT_DIR"

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

echo "Building release binaries..."
# `fastrelease` skips fat LTO so local rebuilds take ~1 min instead of many.
cargo build --profile "$PROFILE" --bin tellodb --manifest-path "${REPO_ROOT}/Cargo.toml"
cargo build --release --manifest-path "${REPO_ROOT}/benchmarks/rust_evaluator/Cargo.toml"

echo "Starting Tellodb engine on port ${ENGINE_PORT} with data dir ${DATA_DIR}..."
TELLODB_DATA_DIR="$DATA_DIR" \
PORT="$ENGINE_PORT" \
TEMPORAL_MEMORY_API_KEY="$ENGINE_API_KEY" \
TELLODB_THREADS="$TELLODB_THREADS" \
TELLODB_RERANK="$TELLODB_RERANK" \
TELLODB_EMBED_MAX_TOKENS="$TELLODB_EMBED_MAX_TOKENS" \
        TELLODB_EMBED_BATCH="$TELLODB_EMBED_BATCH" \
TELLODB_EMBED_TEXT="$TELLODB_EMBED_TEXT" \
TELLODB_LANES="$TELLODB_LANES" \
TELLODB_HEURISTICS="$TELLODB_HEURISTICS" \
TELLODB_EMBED_CACHE="$TELLODB_EMBED_CACHE" \
TELLODB_CONTEXT_WINDOW="$TELLODB_CONTEXT_WINDOW" \
HF_HOME="$HF_HOME" \
HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
XDG_CACHE_HOME="$XDG_CACHE_HOME" \
TELLODB_EMBEDDING_CACHE_PATH="$TELLODB_EMBEDDING_CACHE_PATH" \
"$ENGINE_BIN" >"$RUNS_DIR/engine.log" 2>&1 &
ENGINE_PID=$!
echo "Engine log: $RUNS_DIR/engine.log"

echo "Waiting for engine healthz..."
for i in {1..60}; do
    if curl -s -f "${ENGINE_URL}/healthz" >/dev/null 2>&1; then
        echo "Engine healthy!"
        break
    fi
    sleep 1
done

if ! curl -s -f "${ENGINE_URL}/healthz" >/dev/null 2>&1; then
    echo "Engine failed to become healthy within 60s; last log lines:" >&2
    tail -20 "$RUNS_DIR/engine.log" >&2
    exit 1
fi

echo "Warming up engine..."
curl -s -X POST "${ENGINE_URL}/warmup" -H "x-api-key: ${ENGINE_API_KEY}" >/dev/null

if want_suite longmemeval; then
echo "=== Running LongMemEval-S (dev split, ${RUNS} run(s)) ==="
for run in $(seq 1 "$RUNS"); do
    echo "--- LongMemEval-S run $run/$RUNS ---"
    "$EVALUATOR_BIN" \
        --dataset-kind longmemeval \
        --split dev \
        --tier "$TIER" \
        --client-context "$CLIENT_CONTEXT" \
        --timestamps "$TIMESTAMPS" \
        --limit "$LIMIT" \
        --ingest-concurrency "$CONCURRENCY" \
        --engine-url "$ENGINE_URL" \
        --engine-api-key "$ENGINE_API_KEY" \
        --runs-dir "$RUNS_DIR" \
        --reset-first \
        recall
done
fi

if want_suite locomo; then
echo "=== Running LoCoMo (dev split, ${RUNS} run(s)) ==="
for run in $(seq 1 "$RUNS"); do
    echo "--- LoCoMo run $run/$RUNS ---"
    "$EVALUATOR_BIN" \
        --dataset-kind locomo \
        --split dev \
        --tier "$TIER" \
        --client-context "$CLIENT_CONTEXT" \
        --timestamps "$TIMESTAMPS" \
        --limit "$LIMIT" \
        --ingest-concurrency "$CONCURRENCY" \
        --engine-url "$ENGINE_URL" \
        --engine-api-key "$ENGINE_API_KEY" \
        --runs-dir "$RUNS_DIR" \
        --reset-first \
        recall
done
fi

if want_suite synth; then
echo "=== Running Synthetic Recall (1k memories) ==="
echo "Generating synthetic 1k memories..."
"$SYNTH_BIN" --entities 50 --memories-per-entity 20 --seed 101 --output "$SYNTH_DIR/synth_1k.json"
echo "Evaluating synthetic 1k..."
"$EVALUATOR_BIN" \
    --dataset-kind longmemeval \
    --dataset "$SYNTH_DIR/synth_1k.json" \
    --split all \
    --tier "$TIER" \
        --client-context "$CLIENT_CONTEXT" \
    --timestamps session \
    --limit "$SYNTH_LIMIT" \
    --ingest-concurrency "$CONCURRENCY" \
    --engine-url "$ENGINE_URL" \
    --engine-api-key "$ENGINE_API_KEY" \
    --runs-dir "$RUNS_DIR" \
    --reset-first \
    recall

if [[ "$RUN_SYNTH_10K" -eq 1 ]]; then
    echo "=== Running Synthetic Recall (10k memories) ==="
    echo "Generating synthetic 10k memories..."
    "$SYNTH_BIN" --entities 200 --memories-per-entity 50 --seed 102 --output "$SYNTH_DIR/synth_10k.json"
    echo "Evaluating synthetic 10k..."
    "$EVALUATOR_BIN" \
        --dataset-kind longmemeval \
        --dataset "$SYNTH_DIR/synth_10k.json" \
        --split all \
        --tier "$TIER" \
        --client-context "$CLIENT_CONTEXT" \
        --timestamps session \
        --limit "$SYNTH_LIMIT" \
        --ingest-concurrency "$CONCURRENCY" \
        --engine-url "$ENGINE_URL" \
        --engine-api-key "$ENGINE_API_KEY" \
        --runs-dir "$RUNS_DIR" \
        --reset-first \
        recall
fi

if [[ "$RUN_SYNTH_100K" -eq 1 ]]; then
    echo "=== Running Synthetic Recall (100k memories) ==="
    echo "Generating synthetic 100k memories..."
    "$SYNTH_BIN" --entities 1000 --memories-per-entity 100 --seed 103 --output "$SYNTH_DIR/synth_100k.json"
    echo "Evaluating synthetic 100k..."
    "$EVALUATOR_BIN" \
        --dataset-kind longmemeval \
        --dataset "$SYNTH_DIR/synth_100k.json" \
        --split all \
        --tier "$TIER" \
        --client-context "$CLIENT_CONTEXT" \
        --timestamps session \
        --limit "$SYNTH_LIMIT" \
        --ingest-concurrency "$CONCURRENCY" \
        --engine-url "$ENGINE_URL" \
        --engine-api-key "$ENGINE_API_KEY" \
        --runs-dir "$RUNS_DIR" \
        --reset-first \
        recall
fi

fi

echo "=== Snapshotting Run Records ==="
cp "$RUNS_DIR"/[0-9]*.json "$SNAPSHOT_DIR/"
echo "Snapshotted $(ls -1 "$SNAPSHOT_DIR"/*.json | wc -l | tr -d ' ') record(s) to $SNAPSHOT_DIR"

echo "=== Generating BASELINE.md ==="
"$EVALUATOR_BIN" report "$RUNS_DIR"/[0-9]*.json > "$REPO_ROOT/benchmarks/BASELINE.md"
echo "Embedding cache: $(curl -s "${ENGINE_URL}/version" -H "x-api-key: ${ENGINE_API_KEY}" | grep -o '"embed_cache_[a-z]*":[0-9]*' | tr '\n' ' ')"
cat "$REPO_ROOT/benchmarks/BASELINE.md"
echo "Baseline written to $REPO_ROOT/benchmarks/BASELINE.md"
