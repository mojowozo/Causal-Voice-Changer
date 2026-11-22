#!/usr/bin/env bash
# Launch the CausalVC realtime voice changer (the COSMIC front end; it starts the Python engine itself).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [[ ! -x ../.venv/bin/python ]]; then
    echo "The project's Python environment is missing. Run ../run.sh once first: it installs everything." >&2
    exit 1
fi
binary=ui/target/release/causalvc-realtime
if [[ ! -x $binary ]]; then
    echo "Building the front end (first run only)…" >&2
    cargo build --release --manifest-path ui/Cargo.toml
fi
# Ask PipeWire for a 480-sample period (10 ms at 48 kHz) while this runs: blocks are whole
# multiples of 10 ms, so the output buffer stays level (fewer gaps, less delay).
# The graph runs at the smallest period any client requests, and returns to normal on exit.
# Set PIPEWIRE_QUANTUM yourself to override.
export PIPEWIRE_QUANTUM="${PIPEWIRE_QUANTUM:-480/48000}"
CAUSALVC_REALTIME_DIR="$PWD" exec "$binary" "$@"
