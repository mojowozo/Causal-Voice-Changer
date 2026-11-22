#!/usr/bin/env bash
# Launch the CausalVC realtime voice changer (the COSMIC front end; it starts the Python engine itself).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [[ ! -x .venv/bin/python ]]; then
    echo "The project's Python environment is missing. Run ./run-training.sh once first: it installs everything." >&2
    exit 1
fi
binary=ui/target/release/causalvc-realtime
stamp=realtime/$binary.source-hash
# A front end that came built in a release download holds a fingerprint of the sources it was
# built from (cargo has no record of building it, so it would rebuild it all): start it as is
# while those still match.
if [[ -x realtime/$binary && -f $stamp ]] && [[ $(bash scripts/ui-source-hash.sh realtime/ui) == "$(<"$stamp")" ]]; then
    :
# Built on first run, and again after an update (git pull): cargo rebuilds only when a source
# file changed, and otherwise finishes in well under a second.
elif command -v cargo >/dev/null 2>&1; then
    rm -f "$stamp"
    [[ -x realtime/$binary ]] || echo "Building the front end (first run only)…" >&2
    if ! cargo build --release --quiet --manifest-path realtime/ui/Cargo.toml; then
        if [[ -x realtime/$binary ]]; then
            echo "The front end could not be rebuilt; starting the copy built before." >&2
        else
            exit 1
        fi
    fi
elif [[ ! -x realtime/$binary ]]; then
    echo "Rust (cargo) is needed to build the front end the first time: https://rustup.rs" >&2
    exit 1
elif [[ -f $stamp ]]; then
    echo "The front end's sources changed since it was built; starting it anyway (install Rust to rebuild it: https://rustup.rs)." >&2
fi
# Ask PipeWire for a 480-sample period (10 ms at 48 kHz) while this runs: blocks are whole
# multiples of 10 ms, so the output buffer stays level (fewer gaps, less delay).
# The graph runs at the smallest period any client requests, and returns to normal on exit.
# Set PIPEWIRE_QUANTUM yourself to override.
export PIPEWIRE_QUANTUM="${PIPEWIRE_QUANTUM:-480/48000}"
CAUSALVC_REALTIME_DIR="$PWD/realtime" exec "realtime/$binary" "$@"
