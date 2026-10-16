#!/bin/bash
# Writes "Onion AI Inspection System.desktop" onto this user's desktop.
# Paths are taken from this script's folder and from xdg-user-dir. Nothing is guessed.

set -euo pipefail

APP_DIR="$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || printf '%s' "$0")")" && pwd)"
LAUNCH="$APP_DIR/launch-onion-inspection.sh"
ENTRY="$APP_DIR/final.py"
PYTHON="$APP_DIR/.venv/bin/python"

if [ "$(uname -s)" != "Linux" ]; then
  echo "Run this installer on the Raspberry Pi, from the RaspberryPi5 folder."
  exit 1
fi

if [ ! -f "$LAUNCH" ] || [ ! -f "$ENTRY" ]; then
  echo "The launch script or final.py is missing in $APP_DIR"
  exit 1
fi

chmod +x "$LAUNCH" "$APP_DIR/run.sh" "$APP_DIR/setup.sh" "$0"

DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
if [ -z "$DESKTOP_DIR" ] || [ ! -d "$DESKTOP_DIR" ]; then
  DESKTOP_DIR="$HOME/Desktop"
fi
mkdir -p "$DESKTOP_DIR"

quote_exec() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

TARGET="$DESKTOP_DIR/Onion AI Inspection System.desktop"
EXEC_VALUE="$(quote_exec "$LAUNCH")"
OFFICIAL="$APP_DIR/SRC/resources/official-logo.png"
UNIVERSITY="$APP_DIR/SRC/New smartlab logo.png"
ICON_LINE=""
if [ -f "$OFFICIAL" ]; then
  ICON_LINE="Icon=$OFFICIAL"
elif [ -f "$UNIVERSITY" ]; then
  ICON_LINE="Icon=$UNIVERSITY"
fi

cat > "$TARGET" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Onion AI Inspection System
GenericName=Onion inspection
Comment=Open the Onion AI Inspection System
Exec=$EXEC_VALUE
Path=$APP_DIR
Terminal=false
Categories=Utility;
StartupNotify=true
$ICON_LINE
EOF

chmod +x "$TARGET"
if command -v gio >/dev/null 2>&1; then
  gio set "$TARGET" metadata::trusted true
else
  echo "gio is not installed, so the desktop may still ask for permission once."
  echo "Install it with: sudo apt-get install -y gvfs"
  echo "Then run this installer again. Do not start the inspection app with sudo."
fi

echo "Desktop file: $TARGET"
echo "Exec: $EXEC_VALUE"
echo "Working directory: $APP_DIR"
echo "Entry point: $ENTRY"
if [ -x "$PYTHON" ]; then
  echo "Interpreter: $PYTHON"
else
  echo "Interpreter is not installed yet: $PYTHON"
  echo "Run setup.sh before double-clicking the icon."
fi
echo "Startup log: $APP_DIR/logs/startup.log"

test -d "$APP_DIR"
test -f "$ENTRY"
test -x "$LAUNCH"
test -x "$TARGET"
grep -q '^Type=Application$' "$TARGET"
grep -q '^Name=Onion AI Inspection System$' "$TARGET"
grep -q '^Terminal=false$' "$TARGET"
grep -q "^Exec=$EXEC_VALUE$" "$TARGET"
grep -q "^Path=$APP_DIR$" "$TARGET"
echo "Launcher checks passed."
