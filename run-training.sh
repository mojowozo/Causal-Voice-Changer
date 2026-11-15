#!/usr/bin/env bash
# Sets up everything on first run (Python packages, pretrained files), then opens the training window.
set -e
cd "$(dirname "$(readlink -f "$0")")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv (the Python package manager this project uses) was not found. Installing it from https://astral.sh/uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Checking Python packages..."
uv sync
echo "Checking pretrained files..."
uv run --no-sync python -m causalvc.weights

# The COSMIC window (training/ui), built on first run; it shares compiled libraries with the
# voice changer's. Without Rust, or if the build fails, the small Tk window opens instead.
binary=realtime/ui/target/release/causalvc-training
if command -v cargo >/dev/null 2>&1; then
    [[ -x $binary ]] || echo "Building the training window (first run only)…" >&2
    if ! cargo build --release --quiet --manifest-path training/ui/Cargo.toml; then
        echo "The training window could not be built; opening the small window instead." >&2
    fi
fi
if [[ -x $binary ]]; then
    CAUSALVC_DIR="$PWD" exec "$binary" "$@"
fi
uv run --no-sync python -m causalvc.gui
