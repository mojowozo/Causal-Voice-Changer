#!/usr/bin/env bash
# Sets up everything on first run (Python packages, pretrained files), then opens the training window.
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv (the Python package manager this project uses) was not found. Installing it from https://astral.sh/uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Checking Python packages..."
uv sync
echo "Checking pretrained files..."
uv run --no-sync python -m causalvc.weights
uv run --no-sync python -m causalvc.gui
