#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:3000}"
API_KEY="${API_KEY:-XXX1111AAA}"
TIMEOUT="${TIMEOUT:-10}"
ENTITY_ID="${ENTITY_ID:-tellodb-bench-$(date +%s)}"
SINGLE_RUNS="${SINGLE_RUNS:-5}"
BATCH_RUNS="${BATCH_RUNS:-3}"
BATCH_ITEMS="${BATCH_ITEMS:-8}"
COLD_RUNS="${COLD_RUNS:-5}"
WARM_RUNS="${WARM_RUNS:-10}"
RERANK_RUNS="${RERANK_RUNS:-5}"
RESET_FIRST="${RESET_FIRST:-0}"
OUTPUT_CSV="${OUTPUT_CSV:-}"

TMP_DIR=""
LAST_HEADERS=""
LAST_BODY=""
LAST_META=""

DOCS=(
  "Aletheia stores temporal records in redb with timestamp ordered keys."
  "Aletheia uses an HNSW vector index for semantic retrieval across memories."
  "Aletheia combines BM25 full text search with semantic retrieval for hybrid ranking."
  "Aletheia applies time decay so recent memories outrank stale conversational noise."
  "Aletheia tracks superseded facts so newer knowledge invalidates stale facts."
  "Aletheia ranks memory candidates with reciprocal rank fusion over semantic and lexical hits."
  "Aletheia reranks top semantic candidates with a cross encoder when results are ambiguous."
  "Aletheia fuses BM25 and vector candidates before applying time aware ranking."
  "Aletheia caches embeddings to speed repeated queries and repeated ingests."
  "Aletheia exposes timing headers for embed ann rerank fts fuse and hydrate stages."
  "Aletheia attaches session summaries as companion memories for compressed recall."
  "Aletheia keeps provenance edges between derived memories and source memories."
)

usage() {
  cat <<'EOF'
Usage:
  bash temporal_memory/benchmarks/endpoint_bench.sh [options]

Options:
  -u, --url URL            Base URL for the deployed engine.
  -k, --api-key KEY        API key sent as x-api-key.
  -t, --timeout SEC        curl timeout in seconds.
  -e, --entity-id ID       Override benchmark entity id.
  --single-runs N          Number of single-ingest samples.
  --batch-runs N           Number of batch-ingest samples.
  --batch-items N          Documents per batch ingest sample.
  --cold-runs N            Number of unique cold-query samples.
  -w, --warm-runs N        Number of repeated warm-query samples.
  --rerank-runs N          Number of rerank off/on sample pairs.
  -r, --reset-first        Call /admin/reset before benchmarking.
  -o, --output-csv PATH    Write raw sample rows to CSV.
  -h, --help               Show help.

Environment overrides:
  BASE_URL API_KEY TIMEOUT ENTITY_ID SINGLE_RUNS BATCH_RUNS BATCH_ITEMS
  COLD_RUNS WARM_RUNS RERANK_RUNS RESET_FIRST OUTPUT_CSV

Example:
  BASE_URL=http://localhost:3000 \
  API_KEY=XXX1111AAA \
  bash temporal_memory/benchmarks/endpoint_bench.sh --reset-first
EOF
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

cleanup() {
  if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
  fi
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    -u|--url)
      BASE_URL="$2"
      shift 2
      ;;
    -k|--api-key)
      API_KEY="$2"
      shift 2
      ;;
    -t|--timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    -e|--entity-id)
      ENTITY_ID="$2"
      shift 2
      ;;
    --single-runs)
      SINGLE_RUNS="$2"
      shift 2
      ;;
    --batch-runs)
      BATCH_RUNS="$2"
      shift 2
      ;;
    --batch-items)
      BATCH_ITEMS="$2"
      shift 2
      ;;
    --cold-runs)
      COLD_RUNS="$2"
      shift 2
      ;;
    -w|--warm-runs)
      WARM_RUNS="$2"
      shift 2
      ;;
    --rerank-runs)
      RERANK_RUNS="$2"
      shift 2
      ;;
    -r|--reset-first)
      RESET_FIRST=1
      shift
      ;;
    -o|--output-csv)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_command curl
require_command awk
require_command sed
require_command grep
require_command sort
require_command mktemp
require_command tr
require_command date
require_command dirname
require_command mkdir

TMP_DIR="$(mktemp -d)"

if [[ -n "$OUTPUT_CSV" ]]; then
  mkdir -p "$(dirname "$OUTPUT_CSV")"
  printf '%s\n' "scenario,sample,engine_total_us,http_total_ms,embed_us,storage_us,ann_us,rerank_us,fts_us,fuse_us,hydrate_us,vector_us,graph_us,fact_us,rerank_applied,correct" > "$OUTPUT_CSV"
fi

log() {
  printf '%s\n' "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

header_value() {
  local file="$1"
  local name="$2"
  awk -F': ' -v key="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')" '
    {
      line = tolower($0)
      sub(/\r$/, "", line)
      if (index(line, key ":") == 1) {
        value = substr($0, length($1) + 3)
        sub(/\r$/, "", value)
        print value
      }
    }
  ' "$file"
}

duration_header_us() {
  local headers_file="$1"
  local base="$2"
  local us
  us="$(header_value "$headers_file" "${base}-us")"
  if [[ -n "$us" ]]; then
    printf '%s\n' "$us"
    return
  fi

  local ms
  ms="$(header_value "$headers_file" "${base}-ms")"
  if [[ -n "$ms" ]]; then
    awk -v v="$ms" 'BEGIN { printf "%.0f\n", v * 1000 }'
  else
    printf '0\n'
  fi
}

status_and_time_ms() {
  local meta="$1"
  local http_code time_total
  http_code="${meta%% *}"
  time_total="${meta##* }"
  local time_ms
  time_ms="$(awk -v seconds="$time_total" 'BEGIN { printf "%.0f", seconds * 1000 }')"
  printf '%s %s\n' "$http_code" "$time_ms"
}

percentile() {
  local pct="$1"
  shift
  if [[ $# -eq 0 ]]; then
    printf '0\n'
    return
  fi
  printf '%s\n' "$@" | sort -n | awk -v p="$pct" '
    { values[NR] = $1 }
    END {
      idx = int((p * NR + 99) / 100)
      if (idx < 1) idx = 1
      if (idx > NR) idx = NR
      print values[idx]
    }
  '
}

average() {
  if [[ $# -eq 0 ]]; then
    printf '0\n'
    return
  fi
  printf '%s\n' "$@" | awk '{ sum += $1 } END { printf "%.0f", sum / NR }'
}

format_us() {
  awk -v us="$1" 'BEGIN { printf "%.3fms", us / 1000.0 }'
}

format_ms() {
  awk -v ms="$1" 'BEGIN { printf "%.1fms", ms + 0.0 }'
}

percent_delta() {
  local before="$1"
  local after="$2"
  awk -v a="$before" -v b="$after" '
    BEGIN {
      if (a == 0) {
        print "n/a"
      } else {
        printf "%.1f%%", ((a - b) / a) * 100.0
      }
    }
  '
}

percent_overhead() {
  local base="$1"
  local current="$2"
  awk -v a="$base" -v b="$current" '
    BEGIN {
      if (a == 0) {
        print "n/a"
      } else {
        printf "%.1f%%", ((b - a) / a) * 100.0
      }
    }
  '
}

docs_per_sec() {
  local docs="$1"
  local us="$2"
  awk -v docs="$docs" -v us="$us" '
    BEGIN {
      if (us == 0) {
        print "0.0"
      } else {
        printf "%.1f", (docs * 1000000.0) / us
      }
    }
  '
}

perform_request() {
  local label="$1"
  local method="$2"
  local path="$3"
  local data="${4:-}"
  local headers_file="$TMP_DIR/${label}.headers"
  local body_file="$TMP_DIR/${label}.body"
  local meta

  if [[ "$method" == "GET" ]]; then
    meta="$(
      curl -sS -D "$headers_file" -o "$body_file" \
        --max-time "$TIMEOUT" \
        -H "x-api-key: $API_KEY" \
        "$BASE_URL$path" \
        -w '%{http_code} %{time_total}'
    )"
  else
    meta="$(
      curl -sS -D "$headers_file" -o "$body_file" \
        --max-time "$TIMEOUT" \
        -X POST \
        -H "content-type: application/json" \
        -H "x-api-key: $API_KEY" \
        "$BASE_URL$path" \
        -d "$data" \
        -w '%{http_code} %{time_total}'
    )"
  fi

  LAST_HEADERS="$headers_file"
  LAST_BODY="$body_file"
  LAST_META="$meta"
}

assert_status() {
  local expected="$1"
  local actual="$2"
  local body_file="$3"
  if [[ "$actual" != "$expected" ]]; then
    log "Response body:"
    sed -n '1,160p' "$body_file" >&2
    die "Expected HTTP $expected but got $actual"
  fi
}

record_csv() {
  if [[ -z "$OUTPUT_CSV" ]]; then
    return
  fi
  printf '%s\n' "$*" >> "$OUTPUT_CSV"
}

build_doc_text() {
  local doc_index="$1"
  local sample_kind="$2"
  local sample_number="$3"
  local base="${DOCS[$((doc_index % ${#DOCS[@]}))]}"
  printf '%s Benchmark run %s %s sample %s.' "$base" "$ENTITY_ID" "$sample_kind" "$sample_number"
}

build_ingest_payload() {
  local session="$1"
  local turn="$2"
  local timestamp_ms="$3"
  local text="$4"
  printf '{"entity_id":"%s","memory_id":"%s::%s::%s","timestamp":%s,"textual_content":"%s","relations":[]}' \
    "$ENTITY_ID" "$ENTITY_ID" "$session" "$turn" "$timestamp_ms" "$text"
}

build_batch_payload() {
  local run="$1"
  local start_index="$2"
  local timestamp_base_ms="$3"
  local payload='{"items":['
  local i idx ts text item
  for ((i = 0; i < BATCH_ITEMS; i++)); do
    idx=$((start_index + i))
    ts=$((timestamp_base_ms + idx * 1000))
    text="$(build_doc_text "$idx" "batch" "$run")"
    item="$(build_ingest_payload "session-b-${run}" "$i" "$ts" "$text")"
    if (( i > 0 )); then
      payload+=","
    fi
    payload+="$item"
  done
  payload+="]}"
  printf '%s' "$payload"
}

body_contains() {
  local file="$1"
  local needle="$2"
  if grep -Fqi "$needle" "$file"; then
    printf '1'
  else
    printf '0'
  fi
}

summarize_distribution() {
  local label="$1"
  shift
  local avg p50 p95
  avg="$(average "$@")"
  p50="$(percentile 50 "$@")"
  p95="$(percentile 95 "$@")"
  log "$label avg=$(format_us "$avg") p50=$(format_us "$p50") p95=$(format_us "$p95")"
}

summarize_http_distribution() {
  local label="$1"
  shift
  local avg p50 p95
  avg="$(average "$@")"
  p50="$(percentile 50 "$@")"
  p95="$(percentile 95 "$@")"
  log "$label avg=$(format_ms "$avg") p50=$(format_ms "$p50") p95=$(format_ms "$p95")"
}

reset_system() {
  log "[reset] Calling /admin/reset"
  perform_request "reset" "POST" "/admin/reset" '{"confirm":"delete-all-data","clear_embedding_cache":true}'
  local status http_ms
  read -r status http_ms <<<"$(status_and_time_ms "$LAST_META")"
  assert_status "200" "$status" "$LAST_BODY"
  local engine_us
  engine_us="$(duration_header_us "$LAST_HEADERS" "x-tm-total")"
  local body
  body="$(tr -d '\n' < "$LAST_BODY")"
  log "reset: engine=$(format_us "$engine_us") http=$(format_ms "$http_ms") -> $body"
  log
}

declare -a single_engine_us=()
declare -a single_http_ms=()
declare -a single_embed_us=()
declare -a batch_engine_us=()
declare -a batch_http_ms=()
declare -a batch_per_doc_us=()
declare -a cold_engine_us=()
declare -a cold_http_ms=()
declare -a cold_embed_us=()
declare -a cold_ann_us=()
declare -a cold_fts_us=()
declare -a warm_engine_us=()
declare -a warm_http_ms=()
declare -a warm_embed_us=()
declare -a rerank_off_engine_us=()
declare -a rerank_off_http_ms=()
declare -a rerank_on_engine_us=()
declare -a rerank_on_http_ms=()
declare -a rerank_stage_us=()

single_doc_cursor=0
batch_doc_cursor=1000
exact_hits=0
hybrid_hits=0
rerank_applied_hits=0

log "Benchmark target: $BASE_URL"
log "Benchmark entity: $ENTITY_ID"
log

if [[ "$RESET_FIRST" == "1" ]]; then
  reset_system
fi

log "[1/7] Probing /health"
perform_request "health" "GET" "/health"
read -r health_status health_time_ms <<<"$(status_and_time_ms "$LAST_META")"
assert_status "200" "$health_status" "$LAST_BODY"
log "health: HTTP $health_status in $(format_ms "$health_time_ms") -> $(tr -d '\n' < "$LAST_BODY")"
log

log "[2/7] Probing /version"
perform_request "version" "GET" "/version"
read -r version_status version_time_ms <<<"$(status_and_time_ms "$LAST_META")"
assert_status "200" "$version_status" "$LAST_BODY"
version_body="$(tr -d '\n' < "$LAST_BODY")"
engine_version="$(printf '%s' "$version_body" | sed -n 's/.*"engine_version":"\([^"]*\)".*/\1/p')"
log "version: HTTP $version_status in $(format_ms "$version_time_ms") -> $version_body"
log

log "[3/7] Single ingest latency samples"
for ((run = 1; run <= SINGLE_RUNS; run++)); do
  text="$(build_doc_text "$single_doc_cursor" "single" "$run")"
  payload="$(build_ingest_payload "session-a" "$run" "$((1774718743000 + run * 1000))" "$text")"
  perform_request "single-ingest-$run" "POST" "/ingest" "$payload"
  read -r status http_ms <<<"$(status_and_time_ms "$LAST_META")"
  assert_status "201" "$status" "$LAST_BODY"
  engine_us="$(duration_header_us "$LAST_HEADERS" "x-tm-total")"
  embed_us="$(duration_header_us "$LAST_HEADERS" "x-tm-embed")"
  storage_us="$(duration_header_us "$LAST_HEADERS" "x-tm-storage")"
  vector_us="$(duration_header_us "$LAST_HEADERS" "x-tm-vector")"
  graph_us="$(duration_header_us "$LAST_HEADERS" "x-tm-graph")"
  fact_us="$(duration_header_us "$LAST_HEADERS" "x-tm-fact")"
  single_engine_us+=("$engine_us")
  single_http_ms+=("$http_ms")
  single_embed_us+=("$embed_us")
  record_csv "single_ingest,$run,$engine_us,$http_ms,$embed_us,$storage_us,0,0,0,0,0,$vector_us,$graph_us,$fact_us,0,1"
  log "single[$run]: engine=$(format_us "$engine_us") embed=$(format_us "$embed_us") http=$(format_ms "$http_ms")"
  single_doc_cursor=$((single_doc_cursor + 1))
done
log

log "[4/7] Batch ingest latency and throughput samples"
for ((run = 1; run <= BATCH_RUNS; run++)); do
  payload="$(build_batch_payload "$run" "$batch_doc_cursor" $((1774719750000 + run * 100000)))"
  perform_request "batch-ingest-$run" "POST" "/ingest/batch" "$payload"
  read -r status http_ms <<<"$(status_and_time_ms "$LAST_META")"
  assert_status "201" "$status" "$LAST_BODY"
  engine_us="$(duration_header_us "$LAST_HEADERS" "x-tm-total")"
  embed_us="$(duration_header_us "$LAST_HEADERS" "x-tm-embed")"
  per_doc_us="$(awk -v us="$engine_us" -v docs="$BATCH_ITEMS" 'BEGIN { printf "%.0f", us / docs }')"
  throughput="$(docs_per_sec "$BATCH_ITEMS" "$engine_us")"
  batch_engine_us+=("$engine_us")
  batch_http_ms+=("$http_ms")
  batch_per_doc_us+=("$per_doc_us")
  record_csv "batch_ingest,$run,$engine_us,$http_ms,$embed_us,0,0,0,0,0,0,0,0,0,0,1"
  log "batch[$run]: engine=$(format_us "$engine_us") per_doc=$(format_us "$per_doc_us") http=$(format_ms "$http_ms") throughput=${throughput} docs/s"
  batch_doc_cursor=$((batch_doc_cursor + BATCH_ITEMS))
done
log

log "[5/7] Cold query samples"
for ((run = 1; run <= COLD_RUNS; run++)); do
  exact_query="For benchmark run ${ENTITY_ID}, cold sample ${run}, which database does Aletheia use for temporal records?"
  payload="$(printf '{"textual_query":"%s","limit":5,"entity_id":"%s","enable_neural_rerank":false}' "$exact_query" "$ENTITY_ID")"
  perform_request "cold-query-$run" "POST" "/query/semantic" "$payload"
  read -r status http_ms <<<"$(status_and_time_ms "$LAST_META")"
  assert_status "200" "$status" "$LAST_BODY"
  engine_us="$(duration_header_us "$LAST_HEADERS" "x-tm-total")"
  embed_us="$(duration_header_us "$LAST_HEADERS" "x-tm-embed")"
  ann_us="$(duration_header_us "$LAST_HEADERS" "x-tm-ann")"
  fts_us="$(duration_header_us "$LAST_HEADERS" "x-tm-fts")"
  correct="$(body_contains "$LAST_BODY" "redb")"
  exact_hits=$((exact_hits + correct))
  cold_engine_us+=("$engine_us")
  cold_http_ms+=("$http_ms")
  cold_embed_us+=("$embed_us")
  cold_ann_us+=("$ann_us")
  cold_fts_us+=("$fts_us")
  record_csv "cold_query,$run,$engine_us,$http_ms,$embed_us,0,$ann_us,0,$fts_us,0,0,0,0,0,0,$correct"
  log "cold[$run]: engine=$(format_us "$engine_us") embed=$(format_us "$embed_us") ann=$(format_us "$ann_us") fts=$(format_us "$fts_us") http=$(format_ms "$http_ms") exact_hit=$correct"
done
log

log "[6/7] Warm query samples"
warm_query="For benchmark run ${ENTITY_ID}, which database does Aletheia use for temporal records?"
warm_payload="$(printf '{"textual_query":"%s","limit":5,"entity_id":"%s","enable_neural_rerank":false}' "$warm_query" "$ENTITY_ID")"
perform_request "warm-query-seed" "POST" "/query/semantic" "$warm_payload"
read -r warm_seed_status warm_seed_http_ms <<<"$(status_and_time_ms "$LAST_META")"
assert_status "200" "$warm_seed_status" "$LAST_BODY"
warm_seed_engine_us="$(duration_header_us "$LAST_HEADERS" "x-tm-total")"
warm_seed_embed_us="$(duration_header_us "$LAST_HEADERS" "x-tm-embed")"
log "warm seed: engine=$(format_us "$warm_seed_engine_us") embed=$(format_us "$warm_seed_embed_us") http=$(format_ms "$warm_seed_http_ms")"

for ((run = 1; run <= WARM_RUNS; run++)); do
  perform_request "warm-query-$run" "POST" "/query/semantic" "$warm_payload"
  read -r status http_ms <<<"$(status_and_time_ms "$LAST_META")"
  assert_status "200" "$status" "$LAST_BODY"
  engine_us="$(duration_header_us "$LAST_HEADERS" "x-tm-total")"
  embed_us="$(duration_header_us "$LAST_HEADERS" "x-tm-embed")"
  warm_engine_us+=("$engine_us")
  warm_http_ms+=("$http_ms")
  warm_embed_us+=("$embed_us")
  record_csv "warm_query,$run,$engine_us,$http_ms,$embed_us,0,0,0,0,0,0,0,0,0,0,1"
  log "warm[$run]: engine=$(format_us "$engine_us") embed=$(format_us "$embed_us") http=$(format_ms "$http_ms")"
done
log

log "[7/7] Hybrid and rerank samples"
for ((run = 1; run <= RERANK_RUNS; run++)); do
  hybrid_query="For benchmark run ${ENTITY_ID}, rerank sample ${run}, what does Aletheia combine with semantic retrieval for hybrid ranking?"
  hybrid_payload="$(printf '{"textual_query":"%s","limit":5,"entity_id":"%s","enable_neural_rerank":false}' "$hybrid_query" "$ENTITY_ID")"
  perform_request "hybrid-query-$run" "POST" "/query/semantic" "$hybrid_payload"
  read -r status http_ms <<<"$(status_and_time_ms "$LAST_META")"
  assert_status "200" "$status" "$LAST_BODY"
  correct="$(body_contains "$LAST_BODY" "BM25")"
  hybrid_hits=$((hybrid_hits + correct))
  record_csv "hybrid_query,$run,$(duration_header_us "$LAST_HEADERS" "x-tm-total"),$http_ms,$(duration_header_us "$LAST_HEADERS" "x-tm-embed"),0,$(duration_header_us "$LAST_HEADERS" "x-tm-ann"),0,$(duration_header_us "$LAST_HEADERS" "x-tm-fts"),$(duration_header_us "$LAST_HEADERS" "x-tm-fuse"),$(duration_header_us "$LAST_HEADERS" "x-tm-hydrate"),0,0,0,0,$correct"

  rerank_query="For benchmark run ${ENTITY_ID}, rerank sample ${run}, how does Aletheia rank ambiguous memory candidates?"
  rerank_off_payload="$(printf '{"textual_query":"%s","limit":5,"entity_id":"%s","enable_neural_rerank":false}' "$rerank_query" "$ENTITY_ID")"
  perform_request "rerank-off-$run" "POST" "/query/semantic" "$rerank_off_payload"
  read -r status http_ms <<<"$(status_and_time_ms "$LAST_META")"
  assert_status "200" "$status" "$LAST_BODY"
  off_engine_us="$(duration_header_us "$LAST_HEADERS" "x-tm-total")"
  rerank_off_engine_us+=("$off_engine_us")
  rerank_off_http_ms+=("$http_ms")
  record_csv "rerank_off,$run,$off_engine_us,$http_ms,$(duration_header_us "$LAST_HEADERS" "x-tm-embed"),0,$(duration_header_us "$LAST_HEADERS" "x-tm-ann"),0,$(duration_header_us "$LAST_HEADERS" "x-tm-fts"),$(duration_header_us "$LAST_HEADERS" "x-tm-fuse"),$(duration_header_us "$LAST_HEADERS" "x-tm-hydrate"),0,0,0,0,1"

  rerank_on_payload="$(printf '{"textual_query":"%s","limit":5,"entity_id":"%s","enable_neural_rerank":true}' "$rerank_query" "$ENTITY_ID")"
  perform_request "rerank-on-$run" "POST" "/query/semantic" "$rerank_on_payload"
  read -r status http_ms <<<"$(status_and_time_ms "$LAST_META")"
  assert_status "200" "$status" "$LAST_BODY"
  on_engine_us="$(duration_header_us "$LAST_HEADERS" "x-tm-total")"
  stage_us="$(duration_header_us "$LAST_HEADERS" "x-tm-rerank")"
  rerank_applied="$(header_value "$LAST_HEADERS" "x-tm-rerank-applied")"
  if [[ "$rerank_applied" == "1" ]]; then
    rerank_applied_hits=$((rerank_applied_hits + 1))
  fi
  rerank_on_engine_us+=("$on_engine_us")
  rerank_on_http_ms+=("$http_ms")
  rerank_stage_us+=("$stage_us")
  record_csv "rerank_on,$run,$on_engine_us,$http_ms,$(duration_header_us "$LAST_HEADERS" "x-tm-embed"),0,$(duration_header_us "$LAST_HEADERS" "x-tm-ann"),$stage_us,$(duration_header_us "$LAST_HEADERS" "x-tm-fts"),$(duration_header_us "$LAST_HEADERS" "x-tm-fuse"),$(duration_header_us "$LAST_HEADERS" "x-tm-hydrate"),0,0,0,$rerank_applied,1"

  log "rerank[$run]: off=$(format_us "$off_engine_us") on=$(format_us "$on_engine_us") rerank_stage=$(format_us "$stage_us") applied=$rerank_applied"
done
log

single_p50_us="$(percentile 50 "${single_engine_us[@]}")"
batch_per_doc_p50_us="$(percentile 50 "${batch_per_doc_us[@]}")"
cold_p50_us="$(percentile 50 "${cold_engine_us[@]}")"
warm_p50_us="$(percentile 50 "${warm_engine_us[@]}")"
rerank_off_p50_us="$(percentile 50 "${rerank_off_engine_us[@]}")"
rerank_on_p50_us="$(percentile 50 "${rerank_on_engine_us[@]}")"

log "Summary"
log "-------"
log "Engine version: ${engine_version:-unknown}"
log "Health RTT: $(format_ms "$health_time_ms")"
log "Version RTT: $(format_ms "$version_time_ms")"
summarize_distribution "Single ingest engine" "${single_engine_us[@]}"
summarize_http_distribution "Single ingest HTTP" "${single_http_ms[@]}"
summarize_distribution "Single ingest embed stage" "${single_embed_us[@]}"
summarize_distribution "Batch ingest engine" "${batch_engine_us[@]}"
summarize_http_distribution "Batch ingest HTTP" "${batch_http_ms[@]}"
summarize_distribution "Batch ingest per-doc engine" "${batch_per_doc_us[@]}"
log "Batch ingest throughput avg=$(docs_per_sec "$BATCH_ITEMS" "$(average "${batch_engine_us[@]}")") docs/s p50=$(docs_per_sec "$BATCH_ITEMS" "$(percentile 50 "${batch_engine_us[@]}")") docs/s"
log "Batch vs single per-doc engine improvement: $(percent_delta "$single_p50_us" "$batch_per_doc_p50_us")"
summarize_distribution "Cold query engine" "${cold_engine_us[@]}"
summarize_http_distribution "Cold query HTTP" "${cold_http_ms[@]}"
summarize_distribution "Cold query embed stage" "${cold_embed_us[@]}"
summarize_distribution "Cold query ANN stage" "${cold_ann_us[@]}"
summarize_distribution "Cold query FTS stage" "${cold_fts_us[@]}"
log "Exact retrieval correctness: ${exact_hits}/${COLD_RUNS}"
log "Warm query seed engine: $(format_us "$warm_seed_engine_us")"
summarize_distribution "Warm query engine" "${warm_engine_us[@]}"
summarize_http_distribution "Warm query HTTP" "${warm_http_ms[@]}"
summarize_distribution "Warm query embed stage" "${warm_embed_us[@]}"
log "Warm speedup vs cold query p50 engine latency: $(percent_delta "$cold_p50_us" "$warm_p50_us")"
summarize_distribution "Rerank off engine" "${rerank_off_engine_us[@]}"
summarize_distribution "Rerank on engine" "${rerank_on_engine_us[@]}"
summarize_http_distribution "Rerank off HTTP" "${rerank_off_http_ms[@]}"
summarize_http_distribution "Rerank on HTTP" "${rerank_on_http_ms[@]}"
summarize_distribution "Rerank stage" "${rerank_stage_us[@]}"
log "Rerank overhead vs rerank-off p50 engine latency: $(percent_overhead "$rerank_off_p50_us" "$rerank_on_p50_us")"
log "Rerank applied: ${rerank_applied_hits}/${RERANK_RUNS}"
log "Hybrid retrieval correctness: ${hybrid_hits}/${RERANK_RUNS}"
if [[ -n "$OUTPUT_CSV" ]]; then
  log "Raw samples CSV: $OUTPUT_CSV"
fi
