#!/usr/bin/env bash
# Shared first-run setup for both apps, run by run-realtime.sh and run-training.sh:
#
#     bash scripts/setup.sh realtime|training
#
# 1. uv (the Python package manager this project uses), installed into ~/.local/bin when missing.
# 2. The Python packages both apps share (.venv), installed or brought up to date; quick when
#    nothing changed.
# 3. The pretrained files the app needs in weights/: the base every voice runs on (both apps), and
#    RMVPE, which labels training audio (training only).
#
# Whichever app is opened first installs what the other needs too, so either can be run first.
set -euo pipefail
app=${1:?usage: scripts/setup.sh realtime|training}
cd "$(dirname "$(readlink -f "$0")")/.."

if ! command -v uv >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv (the Python package manager this project uses) was not found. Installing it from https://astral.sh/uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Checking Python packages..."
uv sync
echo "Checking pretrained files..."
uv run --no-sync python -m causalvc.weights "$app"
