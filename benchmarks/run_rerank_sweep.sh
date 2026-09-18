#!/usr/bin/env bash
# Rerank cost cascade (roadmap WP7): always-rerank as the reference, then the
# keyword heuristic, the confidence gate at several margins, and no rerank.
# Pick the margin at the knee of recall/nDCG against p95 latency.
#
#   bash benchmarks/run_rerank_sweep.sh --tier dev
#   MARGINS="0.02 0.05" RERANK_MODEL=jina-reranker-v1-turbo-en bash benchmarks/run_rerank_sweep.sh --tier dev
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

MARGINS="${MARGINS:-0.01 0.02 0.05 0.1 0.2}"
MODEL="TELLODB_RERANK_MODEL=${RERANK_MODEL:-bge-reranker-base}"
TOP="TELLODB_RERANK_TOP=${RERANK_TOP:-25}"

MATRIX="always TELLODB_RERANK=auto TELLODB_RERANK_POLICY=always $MODEL $TOP"
MATRIX+=$'\n'"heuristic TELLODB_RERANK=auto TELLODB_RERANK_POLICY=heuristic $MODEL $TOP"
for m in $MARGINS; do
    MATRIX+=$'\n'"gate_${m} TELLODB_RERANK=auto TELLODB_RERANK_POLICY=gate TELLODB_RERANK_MARGIN=${m} $MODEL $TOP"
done
MATRIX+=$'\n'"no_rerank TELLODB_RERANK=off"
MATRIX="$MATRIX" NAME=rerank_sweep exec bash "$SCRIPT_DIR/run_matrix.sh" "$@"
