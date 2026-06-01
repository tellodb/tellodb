#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${TELLODB_DATA_DIR:-${TEMPORAL_MEMORY_DATA_DIR:-/runpod-volume/tellodb}}"

mkdir -p "${DATA_DIR}"

exec /usr/local/bin/temporal_memory
