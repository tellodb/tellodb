#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarks"

LONGMEMEVAL_URL_DEFAULT="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
LOCOMO_URL_DEFAULT="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"

LONGMEMEVAL_URL="${LONGMEMEVAL_URL:-$LONGMEMEVAL_URL_DEFAULT}"
LOCOMO_URL="${LOCOMO_URL:-$LOCOMO_URL_DEFAULT}"

MODE="all"
FORCE="false"

usage() {
  cat <<'EOF'
Usage: scripts/setup_bench_data.sh [--dataset all|longmemeval|locomo] [--force]

Options:
  --dataset   Dataset to fetch (default: all)
  --force     Re-download even if file already exists
  -h, --help  Show this help

Environment overrides:
  LONGMEMEVAL_URL  Override LongMemEval source URL
  LOCOMO_URL       Override LoCoMo source URL
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      MODE="${2:-}"
      shift 2
      ;;
    --force)
      FORCE="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi

download_if_needed() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out" && "$FORCE" != "true" ]]; then
    echo "skip  $out (already exists)"
    return 0
  fi
  echo "fetch $url"
  curl -fL "$url" -o "$out"
  echo "saved $out"
}

case "$MODE" in
  all|longmemeval)
    download_if_needed \
      "$LONGMEMEVAL_URL" \
      "${BENCH_DIR}/LongMemEval/data/longmemeval_s_cleaned.json"
    ;;
esac

case "$MODE" in
  all|locomo)
    download_if_needed \
      "$LOCOMO_URL" \
      "${BENCH_DIR}/LoCoMo/data/locomo10.json"
    ;;
esac

echo "Done. Benchmark data setup complete."
