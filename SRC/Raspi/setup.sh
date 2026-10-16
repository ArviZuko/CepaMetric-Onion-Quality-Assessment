#!/bin/bash
# Double-click this file on the Raspberry Pi desktop.
# It installs every library the inspection window needs, then you start run.sh.
APP_DIR="$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || printf "%s" "$0")")" && pwd)"
cd "$APP_DIR"

if [ ! -t 1 ]; then
  if command -v lxterminal >/dev/null 2>&1; then
    exec lxterminal -e bash "$APP_DIR/setup.sh"
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    exec x-terminal-emulator -e bash "$APP_DIR/setup.sh"
  elif command -v gnome-terminal >/dev/null 2>&1; then
    exec gnome-terminal -- bash "$APP_DIR/setup.sh"
  fi
fi

pause_window() {
  echo
  echo "Press Enter to close this window."
  read -r _ || true
}

echo "================================================"
echo "  Onion AI Inspection  -  Raspberry Pi 5 setup"
echo "================================================"
echo "Folder: $APP_DIR"
echo

if ! command -v sudo >/dev/null 2>&1; then
  echo "This setup must be run on the Raspberry Pi, not on Windows."
  pause_window
  exit 1
fi

ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
  echo "This setup is for a 64-bit Raspberry Pi 5 (aarch64). This machine is $ARCH."
  echo "Do not install the Windows requirements.txt here."
  pause_window
  exit 1
fi

echo "Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
  python3 python3-pip python3-venv python3-dev \
  python3-pyqt5 python3-pyqt5.qtsvg python3-pyqt5.sip python3-numpy \
  python3-opencv python3-matplotlib python3-h5py python3-pil \
  python3-lgpio python3-gpiozero \
  libgl1 libglib2.0-0 libusb-1.0-0 \
  cmake build-essential git pkg-config \
  libssl-dev libusb-1.0-0-dev libudev-dev \
  pybind11-dev python3-pybind11 \
  v4l-utils || {
    echo "Some system packages could not be installed."
    pause_window
    exit 1
  }

echo
echo "Creating the Python environment..."
if [ ! -d ".venv" ]; then
  python3 -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r "$APP_DIR/requirements-pi.txt" || {
  echo "The Python packages did not finish installing."
  echo "On the Pi, TensorFlow must be tensorflow-aarch64, not the x86 tensorflow package."
  pause_window
  exit 1
}

echo
echo "Installing the RealSense camera library from source..."
echo "The PyPI pyrealsense2 wheel is not an ARM64 build, so it is not used."
if ! python -c "import pyrealsense2" >/dev/null 2>&1; then
  RS_DIR="$APP_DIR/.librealsense"
  if [ ! -d "$RS_DIR/.git" ]; then
    rm -rf "$RS_DIR"
    git clone --depth 1 https://github.com/IntelRealSense/librealsense.git "$RS_DIR"
  fi
  mkdir -p "$RS_DIR/build"
  cmake -S "$RS_DIR" -B "$RS_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE="$APP_DIR/.venv/bin/python"
  cmake --build "$RS_DIR/build" -j"$(nproc)"
  cmake --install "$RS_DIR/build"
  if [ -f "$RS_DIR/config/99-realsense-libusb.rules" ]; then
    sudo cp "$RS_DIR/config/99-realsense-libusb.rules" /etc/udev/rules.d/
    sudo udevadm control --reload-rules || true
    sudo udevadm trigger || true
  fi
  RS_PY="$(find "$RS_DIR/build" -name 'pyrealsense2*.so' -printf '%h\n' 2>/dev/null | head -n 1)"
  if [ -n "$RS_PY" ]; then
    echo "export PYTHONPATH=\"$RS_PY\${PYTHONPATH:+:\$PYTHONPATH}\"" > "$APP_DIR/.venv/realsense-path.sh"
  fi
fi
if [ -f "$APP_DIR/.venv/realsense-path.sh" ]; then
  # shellcheck disable=SC1091
  source "$APP_DIR/.venv/realsense-path.sh"
fi

echo
echo "Checking that the application can start..."
python - <<'PY'
missing = []
checks = [
    ("numpy", "import numpy"),
    ("cv2", "import cv2"),
    ("PyQt5", "import PyQt5"),
    ("PyQt5.QtSvg", "from PyQt5 import QtSvg"),
    ("tensorflow", "import tensorflow"),
    ("matplotlib", "import matplotlib"),
    ("reportlab", "import reportlab"),
    ("openpyxl", "import openpyxl"),
    ("pyrealsense2", "import pyrealsense2"),
    ("gpiozero", "import gpiozero"),
]
for name, statement in checks:
    try:
        exec(statement, {})
        print("  OK", name)
    except Exception as exc:
        missing.append(name)
        print("  MISSING", name, "-", exc)
if missing:
    raise SystemExit(1)
PY

if [ $? -ne 0 ]; then
  echo
  echo "Setup did not finish. Read the MISSING lines above."
  pause_window
  exit 1
fi

chmod +x "$APP_DIR/run.sh" "$APP_DIR/setup.sh" "$APP_DIR/launch-onion-inspection.sh" "$APP_DIR/install-desktop-launcher.sh" 2>/dev/null || true
bash "$APP_DIR/install-desktop-launcher.sh"

echo
echo "Setup finished."
echo "Plug in the RealSense camera, then double-click Onion AI Inspection System on the desktop."
echo "If it does not open, read logs/startup.log in this folder."
pause_window
