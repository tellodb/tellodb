#!/usr/bin/env bash
# One-shot setup for a fresh Linux GPU box (Ubuntu 22.04/24.04; 24.04 preferred
# — ONNX Runtime's GPU provider binaries need its glibc).
#
# Installs system packages, Rust, builds tellodb and the evaluator, fetches the
# models and the LongMemEval dataset, then verifies the CUDA execution provider
# actually loaded. Safe to re-run: every step is skipped if already done.
#
#   curl -fsSL <raw-url>/scripts/setup_gpu_box.sh | bash
#   # or, from a clone:
#   bash scripts/setup_gpu_box.sh
#
# Options (environment):
#   TELLODB_REPO=<git url>     repo to clone when not run from inside one
#   TELLODB_DIR=<path>         where to clone (default ~/tellodb)
#   TELLODB_BRANCH=<branch>    branch to check out (default main)
#   SKIP_DATASETS=1            don't download LongMemEval (277 MB)
#   SKIP_EXTRACTOR_MODEL=1     don't download the GLiNER extractor (183 MB)
#   CACHE_DIR=<path>           model/dataset cache (default ~/.cache/tellodb)
set -euo pipefail

TELLODB_REPO="${TELLODB_REPO:-https://github.com/tellodb/tellodb.git}"
TELLODB_DIR="${TELLODB_DIR:-$HOME/tellodb}"
TELLODB_BRANCH="${TELLODB_BRANCH:-main}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/tellodb}"
EXTRACTOR_MODEL_DIR="$CACHE_DIR/models/gliner_small"
GLINER_REPO="https://huggingface.co/onnx-community/gliner_small-v2.1/resolve/main"
LONGMEMEVAL_URL="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mwarning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

sudo_if_needed() {
    if [[ $EUID -eq 0 ]]; then "$@"; else sudo "$@"; fi
}

# ── Preflight ────────────────────────────────────────────────────────────────
step "Checking the box"
[[ "$(uname -s)" == "Linux" ]] || die "this script is for Linux; on macOS just run cargo build"
if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    echo "    OS: ${PRETTY_NAME:-unknown}"
    case "${VERSION_ID:-}" in
        24.04|24.10|25.04) ;;
        22.04) warn "Ubuntu 22.04's glibc is too old for some ORT GPU binaries; 24.04 is recommended" ;;
        *) warn "untested distro ${VERSION_ID:-unknown}; expecting Ubuntu 24.04" ;;
    esac
fi
echo "    CPU cores: $(nproc)   RAM: $(free -g | awk '/^Mem:/{print $2}') GB   Disk free: $(df -h --output=avail "$HOME" | tail -1 | tr -d ' ')"

GPU_PRESENT=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_PRESENT=1
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    GPU: /'
else
    warn "no working nvidia-smi — continuing, but the engine will run on CPU"
fi

# ── System packages ──────────────────────────────────────────────────────────
step "Installing system packages"
if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    sudo_if_needed apt-get update -qq
    sudo_if_needed apt-get install -y -qq --no-install-recommends \
        build-essential pkg-config libssl-dev cmake git curl ca-certificates \
        sqlite3 python3 jq bc unzip
else
    warn "no apt-get; install build-essential/pkg-config/libssl-dev/cmake/git/curl/sqlite3 yourself"
fi

# ── Rust ─────────────────────────────────────────────────────────────────────
step "Installing Rust"
if command -v cargo >/dev/null 2>&1; then
    echo "    already present: $(cargo --version)"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
    echo "    installed: $("$HOME/.cargo/bin/cargo" --version)"
fi
# shellcheck disable=SC1091
[[ -f "$HOME/.cargo/env" ]] && . "$HOME/.cargo/env"
command -v cargo >/dev/null 2>&1 || die "cargo not on PATH after install"

# ── Repository ───────────────────────────────────────────────────────────────
step "Getting the source"
if git rev-parse --show-toplevel >/dev/null 2>&1 && [[ -f "$(git rev-parse --show-toplevel)/Cargo.toml" ]]; then
    REPO_ROOT="$(git rev-parse --show-toplevel)"
    echo "    using the current clone: $REPO_ROOT"
elif [[ -d "$TELLODB_DIR/.git" ]]; then
    REPO_ROOT="$TELLODB_DIR"
    git -C "$REPO_ROOT" fetch --quiet origin "$TELLODB_BRANCH"
    git -C "$REPO_ROOT" checkout --quiet "$TELLODB_BRANCH"
    git -C "$REPO_ROOT" pull --quiet --ff-only
    echo "    updated: $REPO_ROOT ($(git -C "$REPO_ROOT" rev-parse --short HEAD))"
else
    git clone --quiet --branch "$TELLODB_BRANCH" "$TELLODB_REPO" "$TELLODB_DIR"
    REPO_ROOT="$TELLODB_DIR"
    echo "    cloned into $REPO_ROOT"
fi
cd "$REPO_ROOT"

# ── Caches ───────────────────────────────────────────────────────────────────
# Keep models, the embedding cache and datasets outside the throwaway data dir
# so benchmark runs never re-download or re-embed.
step "Preparing caches under $CACHE_DIR"
mkdir -p "$CACHE_DIR/hf" "$CACHE_DIR/xdg" "$CACHE_DIR/models"
ENV_FILE="$REPO_ROOT/.env.tellodb"
cat > "$ENV_FILE" <<ENVEOF
# Written by scripts/setup_gpu_box.sh — source this before running benchmarks.
# cargo first: benchmark scripts build, and a non-interactive shell (tmux, ssh
# command, cron) does not source ~/.cargo/env on its own.
export PATH="\$HOME/.cargo/bin:\$PATH"
export HF_HOME="$CACHE_DIR/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR/hf/hub"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export TELLODB_EMBEDDING_CACHE_PATH="$CACHE_DIR/embedding_cache.sqlite"
export TEMPORAL_MEMORY_DEVICE=$([[ $GPU_PRESENT -eq 1 ]] && echo cuda || echo cpu)
export TELLODB_THREADS=$(nproc)
export TELLODB_API_KEY=\${TELLODB_API_KEY:-$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')}
# Encoder fact-extraction tier (opt in with TELLODB_EXTRACTOR=encoder):
export TELLODB_EXTRACTOR_MODEL_DIR="$EXTRACTOR_MODEL_DIR"
ENVEOF
echo "    wrote $ENV_FILE"
# shellcheck disable=SC1090
. "$ENV_FILE"

# ── Build ────────────────────────────────────────────────────────────────────
step "Building tellodb (this takes a few minutes on a cold cache)"
cargo build --profile fastrelease --bin tellodb
cargo build --release --manifest-path benchmarks/rust_evaluator/Cargo.toml
echo "    engine:    $REPO_ROOT/target/fastrelease/tellodb"
echo "    evaluator: $REPO_ROOT/benchmarks/rust_evaluator/target/release/rust_evaluator"

# ── Datasets ─────────────────────────────────────────────────────────────────
step "Fetching datasets"
LME_PATH="$REPO_ROOT/benchmarks/LongMemEval/data/longmemeval_s_cleaned.json"
if [[ "${SKIP_DATASETS:-0}" == "1" ]]; then
    echo "    skipped (SKIP_DATASETS=1)"
elif [[ -s "$LME_PATH" ]]; then
    echo "    LongMemEval already present ($(du -h "$LME_PATH" | cut -f1))"
else
    mkdir -p "$(dirname "$LME_PATH")"
    echo "    downloading LongMemEval-S cleaned (~277 MB)..."
    curl -fL --progress-bar -o "$LME_PATH.part" "$LONGMEMEVAL_URL"
    mv "$LME_PATH.part" "$LME_PATH"
    echo "    saved to $LME_PATH"
fi
if [[ -s "$REPO_ROOT/benchmarks/LoCoMo/data/locomo10.json" ]]; then
    echo "    LoCoMo ships with the repo — present"
else
    warn "benchmarks/LoCoMo/data/locomo10.json is missing; LoCoMo runs will fail"
fi

# ── Models ───────────────────────────────────────────────────────────────────
# The embedder and reranker download themselves on first use via fastembed; the
# GLiNER extractor is fetched here because nothing downloads it automatically.
step "Fetching the encoder extractor model"
if [[ "${SKIP_EXTRACTOR_MODEL:-0}" == "1" ]]; then
    echo "    skipped (SKIP_EXTRACTOR_MODEL=1)"
elif [[ -s "$EXTRACTOR_MODEL_DIR/onnx/model_int8.onnx" ]]; then
    echo "    already present at $EXTRACTOR_MODEL_DIR"
else
    mkdir -p "$EXTRACTOR_MODEL_DIR/onnx"
    for f in gliner_config.json tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json config.json; do
        curl -fsSL -o "$EXTRACTOR_MODEL_DIR/$f" "$GLINER_REPO/$f" || warn "could not fetch $f"
    done
    echo "    downloading GLiNER small int8 (~183 MB)..."
    curl -fL --progress-bar -o "$EXTRACTOR_MODEL_DIR/onnx/model_int8.onnx" "$GLINER_REPO/onnx/model_int8.onnx"
    echo "    saved to $EXTRACTOR_MODEL_DIR"
fi

# ── Verify ───────────────────────────────────────────────────────────────────
step "Verifying (loads the models — first run also downloads the embedder)"
DOCTOR_DIR="$(mktemp -d)"
trap 'rm -rf "$DOCTOR_DIR"' EXIT
DOCTOR_JSON="$("$REPO_ROOT/target/fastrelease/tellodb" --data-dir "$DOCTOR_DIR" doctor 2>/dev/null || true)"
if [[ -z "$DOCTOR_JSON" ]]; then
    die "tellodb doctor produced no output; run it manually to see the error"
fi
echo "$DOCTOR_JSON" | jq '{device, embedding: .embedding.model, dims: .embedding.dimensions, reranker, vectors}' 2>/dev/null || echo "$DOCTOR_JSON"

DEVICE="$(echo "$DOCTOR_JSON" | jq -r '.device' 2>/dev/null || echo unknown)"
if [[ $GPU_PRESENT -eq 1 && "$DEVICE" != "CUDA" ]]; then
    warn "a GPU is present but the engine reports device=$DEVICE."
    warn "The CUDA execution provider did not load — benchmark numbers would be CPU numbers."
    warn "Check: nvidia-smi works, driver >= 550, and you are on Ubuntu 24.04."
elif [[ "$DEVICE" == "CUDA" ]]; then
    echo "    CUDA execution provider is live."
fi

if [[ -s "$EXTRACTOR_MODEL_DIR/onnx/model_int8.onnx" ]]; then
    echo "    checking the encoder extractor loads..."
    TELLODB_EXTRACTOR=encoder "$REPO_ROOT/target/fastrelease/tellodb" \
        --data-dir "$DOCTOR_DIR" query --entity setup-probe "smoke" >/dev/null 2>&1 \
        && echo "    encoder extractor OK" \
        || warn "the encoder extractor failed to load; TELLODB_EXTRACTOR=rules still works"
fi

# ── Next steps ───────────────────────────────────────────────────────────────
cat <<NEXT

$(printf '\033[1;32m==> Ready\033[0m')

  cd $REPO_ROOT
  source .env.tellodb          # device, threads, caches, API key

Baseline (start here — WP1):
  bash benchmarks/run_baseline.sh --tier dev

Then the decisions waiting on GPU numbers:
  bash benchmarks/run_ablation.sh --tier dev      # which derived structures earn their cost
  bash benchmarks/run_rerank_sweep.sh --tier dev  # rerank policy + margin knee

Embed-text default (run once per client-context setting):
  MATRIX=\$'legacy TELLODB_EMBED_TEXT=legacy\\nturn TELLODB_EMBED_TEXT=turn\\ncontext TELLODB_EMBED_TEXT=context' \\
    NAME=embed_text CLIENT_CONTEXT=window bash benchmarks/run_matrix.sh --tier dev

Fact extraction tiers (WP5):
  MATRIX=\$'rules TELLODB_EXTRACTOR=rules\\nencoder TELLODB_EXTRACTOR=encoder' \\
    NAME=extractor bash benchmarks/run_matrix.sh --tier dev

Reports land in benchmarks/runs/<timestamp>_<tier>_<name>/REPORT.md
NEXT
