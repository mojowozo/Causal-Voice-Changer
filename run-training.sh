#!/usr/bin/env bash
# Sets up everything on first run (Python packages, pretrained files), then opens the training window.
set -e
cd "$(dirname "$(readlink -f "$0")")"

# Python packages and pretrained files, shared with the voice changer (run-realtime.sh does the same).
bash scripts/setup.sh training
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"  # where setup.sh may have just installed uv

# The COSMIC window (training/ui), built on first run; it shares compiled libraries with the
# voice changer's. Without Rust, or if the build fails, the small Tk window opens instead.
binary=realtime/ui/target/release/causalvc-training
stamp=$binary.source-hash
# A window that came built in a release download holds a fingerprint of the sources it was built
# from (cargo has no record of building it, so it would rebuild it all): start it as is while
# those still match.
if [[ -x $binary && -f $stamp ]] && [[ $(bash scripts/ui-source-hash.sh training/ui) == "$(<"$stamp")" ]]; then
    :
elif command -v cargo >/dev/null 2>&1; then
    rm -f "$stamp"
    [[ -x $binary ]] || echo "Building the training window (first run only)…" >&2
    # Run from training/ui: cargo reads .cargo/config.toml (the shared target folder) from the
    # directory it runs in, not from the manifest's.
    if ! (cd training/ui && cargo build --release --quiet); then
        echo "The training window could not be built; opening the small window instead." >&2
    fi
fi
if [[ -x $binary ]]; then
    CAUSALVC_DIR="$PWD" exec "$binary" "$@"
fi
uv run --no-sync python -m causalvc.gui
