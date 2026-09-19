#!/usr/bin/env bash
# Retrieval baselines: the full pipeline, then the standard comparisons a
# paper needs, each a subset of the same engine (TELLODB_LANES).
#
# Every number the engine produces is otherwise absolute, with nothing to
# compare it against. These are the internal baselines; external systems
# (Mem0, Zep, Letta) are a separate harness.
#
#   bash benchmarks/run_baselines.sh --tier dev
#   DATASETS=longmemeval bash benchmarks/run_baselines.sh --tier smoke
#
# Two rows are not lane subsets but policy checks that belong in the same
# table: `no_recency_decay` tests whether ranking decay penalises correct
# evidence (a 30-day half-life over haystacks that span months), and
# `rerank_heuristic` is the pre-gate behaviour that reranked 93% of queries.
#
# run_matrix.sh reports every row as a paired delta against the first, so the
# full pipeline must stay first.
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

MATRIX="full
bm25_only TELLODB_LANES=fts
vector_only TELLODB_LANES=vector
hybrid_rrf TELLODB_LANES=vector,fts
hybrid_rerank TELLODB_LANES=vector,fts,rerank
no_graph TELLODB_LANES=vector,fts,cards,rerank,route
no_rerank TELLODB_LANES=vector,fts,cards,graph,route
no_recency_decay TEMPORAL_MEMORY_ENABLE_TEMPORAL_RECENCY_SCORING=0
rerank_heuristic TELLODB_RERANK_POLICY=heuristic"

MATRIX="$MATRIX" NAME=baselines exec bash "$SCRIPT_DIR/run_matrix.sh" "$@"
