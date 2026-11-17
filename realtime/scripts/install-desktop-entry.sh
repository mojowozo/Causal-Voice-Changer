#!/usr/bin/env bash
# Add CausalVC Realtime to the application launcher for the current user.
set -euo pipefail
# The repository's launcher, which rebuilds the front end after an update before starting it.
launcher="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)/run-realtime.sh"
app_id=io.github.causalvc.Realtime
target="${XDG_DATA_HOME:-$HOME/.local/share}/applications/$app_id.desktop"
mkdir -p "$(dirname "$target")"
cat > "$target" <<DESKTOP
[Desktop Entry]
Type=Application
Name=CausalVC Realtime
Comment=Realtime voice changer
Exec="$launcher"
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Audio;
StartupWMClass=$app_id
DESKTOP
echo "Installed $target"
