#!/bin/bash
# Starts Onion AI Inspection from the folder that contains this script.
# Entry point: final.py
# Interpreter: .venv/bin/python (created by setup.sh with system site packages)
# Optional environment: .venv/realsense-path.sh adds the built RealSense module.

APP_DIR="$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || printf '%s' "$0")")" && pwd)"
LOG_DIR="$APP_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/startup.log"
exec >>"$LOG_FILE" 2>&1

echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="
cd "$APP_DIR" || {
  echo "Cannot open the project folder: $APP_DIR"
  exit 1
}

if [ ! -f "$APP_DIR/.venv/bin/activate" ]; then
  echo "The virtual environment is missing: $APP_DIR/.venv"
  echo "Run setup.sh in this folder, then open the desktop icon again."
  exit 1
fi
# shellcheck disable=SC1091
source "$APP_DIR/.venv/bin/activate"

if [ -f "$APP_DIR/.venv/realsense-path.sh" ]; then
  # shellcheck disable=SC1091
  source "$APP_DIR/.venv/realsense-path.sh"
  echo "Loaded RealSense path from .venv/realsense-path.sh"
fi

PYTHON="$APP_DIR/.venv/bin/python"
ENTRY="$APP_DIR/final.py"
if [ ! -x "$PYTHON" ]; then
  echo "Python interpreter is missing or not executable: $PYTHON"
  exit 1
fi
if [ ! -f "$ENTRY" ]; then
  echo "Entry point is missing: $ENTRY"
  exit 1
fi

echo "cwd=$APP_DIR"
echo "python=$PYTHON"
echo "entry=$ENTRY"
echo "PYTHONPATH=${PYTHONPATH:-}"
exec "$PYTHON" "$ENTRY"
