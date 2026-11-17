#!/usr/bin/env bash
# A fingerprint of a Linux front end's source files (a Cargo project folder, without its target/).
# A pre-built front end from a release download carries the fingerprint of the sources it was
# built from (BINARY.source-hash beside it), so run-realtime.sh and run-training.sh start it as is
# while the sources still match, instead of having cargo rebuild it from scratch (a download holds
# none of cargo's records of what it built). Line endings don't count.
#
#   scripts/ui-source-hash.sh realtime/ui        prints the fingerprint
#
# Run from the project folder (paths in the fingerprint are relative to it).
set -euo pipefail
folder=${1:?usage: ui-source-hash.sh FOLDER}
find "$folder" -path "$folder/target" -prune -o -type f -print | LC_ALL=C sort |
    while IFS= read -r file; do
        printf '%s %s\n' "$(tr -d '\r' <"$file" | sha256sum | cut -d' ' -f1)" "$file"
    done | sha256sum | cut -d' ' -f1
