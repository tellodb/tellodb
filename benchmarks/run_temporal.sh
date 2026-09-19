#!/usr/bin/env bash
# Why real event times score worse than fake ones.
#
# Under `--timestamps session` (real conversation dates) LongMemEval dev loses
# ~2.1 recall and ~3.2 nDCG against `wallclock`, which stamps every memory at
# roughly ingest time. Under wallclock every memory has age ~0, so ranking
# decay is inert; under session it is not.
#
# Conversational memories decay with a 30-day half-life to a 0.35 floor, so a
# six-month-old turn is ranked at a third of its score for age alone — and in
# LongMemEval the answer is routinely in an old session. The hypothesis is
# that recency decay penalises correct evidence, and that supersession
# (fact_versions) is the mechanism that should decide staleness instead.
#
#   bash benchmarks/run_temporal.sh --tier dev
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

MATRIX="baseline
no_recency_decay TEMPORAL_MEMORY_ENABLE_TEMPORAL_RECENCY_SCORING=0
rerank_heuristic TELLODB_RERANK_POLICY=heuristic
rerank_off TELLODB_LANES=vector,fts,cards,graph,route"

MATRIX="$MATRIX" NAME=temporal exec bash "$SCRIPT_DIR/run_matrix.sh" "$@"
