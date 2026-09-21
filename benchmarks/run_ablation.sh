#!/usr/bin/env bash
# Ingest-structure ablations (roadmap WP3): everything on, then each
# structure disabled in turn (TELLODB_DISABLE).
#
#   bash benchmarks/run_ablation.sh --tier dev
#   STRUCTURES="gist,keywords" DATASETS=synthetic bash benchmarks/run_ablation.sh --tier smoke
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# Every switch in src/features.rs.
STRUCTURES="${STRUCTURES:-chunks,gist,keywords,fact_companions,atomic_cards,event_companions,relation_companions,memory_cards,session_router,preferences,retrospective_links,derived_links,graph_edges,facts,semantic_dedup,consolidation,metrics,predicate_canon}"

MATRIX="baseline"
IFS=, read -ra LIST <<< "$STRUCTURES"
for s in "${LIST[@]}"; do
    MATRIX+=$'\n'"no_${s} TELLODB_DISABLE=${s}"
done
MATRIX="$MATRIX" NAME=ablation exec bash "$SCRIPT_DIR/run_matrix.sh" "$@"
