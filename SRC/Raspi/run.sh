#!/bin/bash
# Double-click this file on the Raspberry Pi desktop to open the application.
APP_DIR="$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || printf "%s" "$0")")" && pwd)"
cd "$APP_DIR"

if [ ! -t 1 ]; then
  if command -v lxterminal >/dev/null 2>&1; then
    exec lxterminal -e bash "$APP_DIR/run.sh"
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    exec x-terminal-emulator -e bash "$APP_DIR/run.sh"
  elif command -v gnome-terminal >/dev/null 2>&1; then
    exec gnome-terminal -- bash "$APP_DIR/run.sh"
  fi
fi

pause_window() {
  echo
  echo "Press Enter to close this window."
  read -r _ || true
}

if [ ! -d "$APP_DIR/.venv" ]; then
  echo "The Pi packages are not installed yet."
  echo "Starting setup. This can take a long time the first time."
  echo
  bash "$APP_DIR/setup.sh" || {
    pause_window
    exit 1
  }
fi

# shellcheck disable=SC1091
source "$APP_DIR/.venv/bin/activate"
if [ -f "$APP_DIR/.venv/realsense-path.sh" ]; then
  # shellcheck disable=SC1091
  source "$APP_DIR/.venv/realsense-path.sh"
fi

if ! python -c "import numpy, cv2, PyQt5, tensorflow, matplotlib, reportlab, openpyxl, pyrealsense2" >/dev/null 2>&1; then
  echo "A required library is missing. Running setup again..."
  bash "$APP_DIR/setup.sh" || {
    pause_window
    exit 1
  }
  # shellcheck disable=SC1091
  source "$APP_DIR/.venv/bin/activate"
fi

if ! python -c "from PyQt5 import QtSvg" >/dev/null 2>&1; then
  echo "The graph library needs PyQt5 QtSvg. Installing it..."
  sudo apt-get install -y python3-pyqt5.qtsvg || {
    echo "Run this on the Raspberry Pi, then start the application again:"
    echo "  sudo apt-get install -y python3-pyqt5.qtsvg"
    pause_window
    exit 1
  }
fi

echo "Opening Onion AI Inspection..."
python "$APP_DIR/final.py"
status=$?
if [ "$status" -ne 0 ]; then
  echo
  echo "The application closed with an error (code $status)."
  pause_window
  exit "$status"
fi
