#!/usr/bin/env bash
# The voice changer's launcher is ../run-realtime.sh (it sets up the project first); this starts it.
exec "$(dirname "$(readlink -f "$0")")/../run-realtime.sh" "$@"
