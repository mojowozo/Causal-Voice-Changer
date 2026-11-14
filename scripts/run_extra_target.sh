#!/usr/bin/env bash
# Add one extra target voice (see `vcbench data --extra-target`) to every baseline:
# RVC from an existing Applio build (no retraining), Beatrice as its own
# single-speaker model. Run on an idle GPU.
#
#   scripts/run_extra_target.sh NAME APPLIO_MODEL.pth APPLIO_INDEX.index
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

NAME=${1:?target name, as given to --extra-target}
MODEL=${2:?path to the Applio model (.pth) of that voice}
INDEX=${3:?path to its .index file}
APPLIO=${APPLIO:?set APPLIO to an Applio install}
REDPANDA=${REDPANDA:?set REDPANDA to a Redpanda-vc install}
PRE="\"prebuilt\":{\"$NAME\":{\"model\":\"$MODEL\",\"index\":\"$INDEX\"}}"
RVC="$APPLIO/.venv/bin/python -m vcbench.convert --system vcbench.systems.rvc_applio:RvcApplio --targets $NAME"
BEA="$REDPANDA/env/bin/python -m vcbench.convert --system vcbench.systems.beatrice_redpanda:BeatriceRedpanda --targets $NAME"

$RVC --run rvc --system-args "{$PRE}" --blocks offline,20,40,80,250
$RVC --run rvc-lowctx --system-args "{\"extra_s\":0.5,$PRE}" --blocks offline,10,20,40,80

$BEA --run beatrice --system-args "{\"solo\":[\"$NAME\"]}" --build-only
$BEA --run beatrice --system-args "{\"solo\":[\"$NAME\"]}" --blocks 5,10,20,40
$BEA --run beatrice-cpu --system-args "{\"solo\":[\"$NAME\"],\"device\":\"cpu\"}" --blocks 10,20

uv run vcbench score
