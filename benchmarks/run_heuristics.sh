#!/usr/bin/env bash
# What the benchmark-derived rules bought (TELLODB_HEURISTICS).
#
# Rules keyed on proper nouns, brands and phrases taken from LoCoMo questions
# are off under `generic` and on under `legacy-tuned`. Running both quantifies
# the hand-tuning on the benchmark it was tuned on (LoCoMo) and on one it was
# not (LongMemEval), which is the honest way to report a contaminated rule set
# without deleting it.
#
#   bash benchmarks/run_heuristics.sh --tier dev
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

MATRIX="generic TELLODB_HEURISTICS=generic
legacy_tuned TELLODB_HEURISTICS=legacy-tuned"

MATRIX="$MATRIX" NAME=heuristics exec bash "$SCRIPT_DIR/run_matrix.sh" "$@"
