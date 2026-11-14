#!/usr/bin/env bash
# Convert the eval set with every baseline, then score. Run on an idle GPU:
# per-block compute times are part of the results.
#
# Voices must already be built (same commands with --build-only), see README.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

APPLIO=${APPLIO:-/mnt/ssd/Applio-3.7.1}
REDPANDA=${REDPANDA:-/mnt/ssd/Redpanda-vc}
RVC="$APPLIO/.venv/bin/python -m vcbench.convert --system vcbench.systems.rvc_applio:RvcApplio"
BEA="$REDPANDA/env/bin/python -m vcbench.convert --system vcbench.systems.beatrice_redpanda:BeatriceRedpanda"

# RVC with Applio's realtime defaults (2.5 s of extra context per chunk).
$RVC --run rvc --blocks offline,20,40,80,250

# RVC tuned for low latency (0.5 s of extra context): same voices, less compute.
mkdir -p runs/rvc-lowctx
ln -sfn ../rvc/voices runs/rvc-lowctx/voices
$RVC --run rvc-lowctx --system-args '{"extra_s":0.5}' --blocks offline,10,20,40,80

# Beatrice 2 on GPU and, as it is designed for, on CPU.
$BEA --run beatrice --blocks 5,10,20,40
mkdir -p runs/beatrice-cpu
ln -sfn ../beatrice/voices runs/beatrice-cpu/voices
$BEA --run beatrice-cpu --system-args '{"device":"cpu"}' --blocks 10,20

uv run vcbench score
