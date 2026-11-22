#!/usr/bin/env python3
"""
================================================================================
ONION AI INSPECTION SYSTEM — Production PyQt Dashboard
================================================================================
Target Platform: Linux / Raspberry Pi 5 / Windows
Primary Implementation File: main.py

Features:
  1. Live Video Stream & OpenCV Camera Worker with visualization zoom controls.
  2. Deep Learning Classification with 5 Base Models, Stacking Meta-Classifier,
     and Ensemble Probability Averaging:
     - ConvNeXt Tiny
     - EfficientNetV2-B0
     - DenseNet121
     - ResNet
     - Xception
     - Stacking Meta-Classifier (20-feature base model probability vector)
     - Ensemble Model (5-model probability averaging)
  3. 3D Spatial Geometry Sizing & Physical Metric Measurement.
  4. Persistent Spatial Object Tracking: Ensures each unique physical onion
     is counted ONLY ONCE (prevents duplicate frame counting).
  5. Asynchronous Worker Threads for Camera, Model Inference, and XAI Grad-CAM
     so the PyQt UI remains 100% responsive.
  6. On-Demand XAI (Grad-CAM) Visual Explanation Heatmaps for captured records.
  7. Capture History Navigation: [<-], [->], [Restart], and [Capture No] lookup.
  8. Batch Operation Workflow with Input Validation and Automatic Slot Completion.
================================================================================
"""

import sys
import os
import atexit
import time
import math
import json
import dataclasses
from collections import Counter
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2
import pyrealsense2 as rs
import queue
import threading

# ==============================================================================
# RASPBERRY PI 5 GPIO SERVO CONTROLLER
# ==============================================================================
# Controls 4 sorting servos directly via Raspberry Pi 5 GPIO pins.
# Uses gpiozero (supported by lgpio on Raspberry Pi 5 RP1 chip).
# Automatically falls back to Simulation Mode on Windows/PC without errors.

try:
    from gpiozero import AngularServo
    GPIO_AVAILABLE = True
except (ImportError, Exception):
    AngularServo = None
    GPIO_AVAILABLE = False

try:
    import lgpio
    LGPIO_AVAILABLE = True
except (ImportError, Exception):
    lgpio = None
    LGPIO_AVAILABLE = False


def _load_hardware_config():
    """Load the single hardware file from beside this script, or from SRC when launched as final.py."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = (
        os.path.join(here, "hardware_config.py"),
        os.path.join(here, "SRC", "hardware_config.py"),
        os.path.join(here, "..", "config", "hardware_config.py"),
        os.path.join(here, "..", "Application", "config", "hardware_config.py"),
    )
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("onion_hardware_config", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ImportError("hardware_config.py is missing. Expected it beside this file or in SRC.")


_HW = _load_hardware_config()
SERVO_PINS = _HW.SERVO_PINS
BOARD_PIN_DESC = _HW.SERVO_HEADER_LABELS
SERVO_ANGLES = _HW.SERVO_ANGLES
SERVO_MIN_PW = _HW.SERVO_MIN_PW
SERVO_MAX_PW = _HW.SERVO_MAX_PW
SERVO_FRAME_WIDTH = _HW.SERVO_FRAME_WIDTH
servo_pulse_us = _HW.servo_pulse_us
PUSH_HOLD_TIME = _HW.SERVO_HOLD_SECONDS
CONVEYOR_RUN_TIME = _HW.CONVEYOR_RUN_SECONDS
CONVEYOR_TIMEOUT_GRACE = _HW.CONVEYOR_TIMEOUT_GRACE
GPIO_NUMBERING_MODE = _HW.GPIO_NUMBERING_MODE

# One move to the recorded sorting angle, one hold, then one return to rest.
FRONT_DELAY = 0.0
REAR_DELAY = 0.0
SETTLE_TIME = _HW.SERVO_SETTLE_SECONDS
DETACH_AFTER_REST = False


class RaspberryPiServoManager:
    """Non-blocking, multi-threaded servo manager for Raspberry Pi 5.
    Supports both gpiozero and native lgpio (RP1 Southbridge) drivers.
    """
    def __init__(self):
        self.servos = {}
        self.lgpio_pins = {}
        self.queues = {}
        self.running = True
        self._cancel = 0
        self._started_ids = set()
        self._io_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self.lgpio_handle = None
        self.is_simulation = AngularServo is None

        print("=" * 65)
        if self.is_simulation:
            print("[SERVO] Running in SIMULATION MODE (No Raspberry Pi GPIO detected)")
        else:
            print("[SERVO] Initializing physical servos on Raspberry Pi 5 with gpiozero (BCM numbering)")

        for cat, pin in SERVO_PINS.items():
            rest_angle = SERVO_ANGLES[cat]["rest"]
            push_angle = SERVO_ANGLES[cat]["push"]
            pin_desc = BOARD_PIN_DESC.get(pin, f"GPIO {pin}")
            servo_obj = None
            attached_driver = "Simulation"
            if AngularServo is not None:
                try:
                    servo_obj = AngularServo(
                        pin,
                        min_angle=0,
                        max_angle=180,
                        min_pulse_width=SERVO_MIN_PW,
                        max_pulse_width=SERVO_MAX_PW,
                        frame_width=SERVO_FRAME_WIDTH,
                    )
                    servo_obj.angle = rest_angle
                    attached_driver = "gpiozero"
                    self.is_simulation = False
                except Exception as exc:
                    servo_obj = None
                    log_actuator(f"servo BCM {pin}", "error", PUSH_HOLD_TIME, detail=str(exc))
            self.servos[cat] = servo_obj
            print(
                f"  Category {cat} -> BCM GPIO {pin} ({pin_desc}) | Driver: [{attached_driver}] "
                f"| Rest: {rest_angle}° | Push: {push_angle}°"
            )
            q = queue.Queue()
            self.queues[cat] = q
            threading.Thread(target=self._worker, args=(cat, q), daemon=True).start()

        # Open the RP1 gpiochip only for servos gpiozero could not claim.
        # Opening it first blocks gpiozero's own lgpio pin factory on a Pi 5.
        if any(self.servos[cat] is None for cat in SERVO_PINS) and LGPIO_AVAILABLE and lgpio is not None:
            for chip_num in (4, 0):
                try:
                    handle = lgpio.gpiochip_open(chip_num)
                    if handle >= 0:
                        self.lgpio_handle = handle
                        break
                except Exception:
                    pass
            if self.lgpio_handle is not None:
                self.is_simulation = False
                for cat, pin in SERVO_PINS.items():
                    if self.servos.get(cat) is not None:
                        continue
                    try:
                        lgpio.tx_servo(
                            self.lgpio_handle, pin, servo_pulse_us(SERVO_ANGLES[cat]["rest"]),
                            pulse_frequency=50,
                        )
                        self.lgpio_pins[cat] = pin
                        print(f"  Category {cat} -> BCM GPIO {pin} | Driver: [lgpio]")
                    except Exception as exc:
                        log_actuator(f"servo BCM {pin}", "error", PUSH_HOLD_TIME, detail=str(exc))

        print("[SERVO] All sorting servos ready and holding rest positions.")
        print("=" * 65)

    def _worker(self, cat: int, q: queue.Queue):
        pin = SERVO_PINS[cat]
        rest = SERVO_ANGLES[cat]["rest"]
        push = SERVO_ANGLES[cat]["push"]

        while self.running:
            try:
                task = q.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:
                break

            delay = task.get("delay", 0.0)
            timestamp = task.get("timestamp", time.time())
            onion_id = task.get("onion_id") or ""

            token = self._cancel
            active = lambda: self.running and token == self._cancel
            if delay > 0:
                waited = wait_interruptible(
                    delay, time.sleep, time.time, active, started_at=timestamp,
                )
                if waited != "ok":
                    self._write_angle(cat, rest)
                    q.task_done()
                    continue

            try:
                run_servo_sort(
                    writer=lambda angle, category=cat: self._write_angle(category, angle),
                    sleeper=time.sleep,
                    clock=time.time,
                    should_continue=active,
                    rest_angle=rest,
                    push_angle=push,
                    onion_id=onion_id,
                    pin=pin,
                )
            except Exception as exc:
                log_actuator(f"servo BCM {pin}", "error", PUSH_HOLD_TIME, detail=f"onion={onion_id} error={exc}")
                self._write_angle(cat, rest)
            if DETACH_AFTER_REST:
                self._detach(cat)
            q.task_done()

    def _write_angle(self, cat: int, angle: int):
        clamped = max(0, min(180, int(angle)))
        with self._io_lock:
            servo = self.servos.get(cat)
            if servo is not None:
                try:
                    servo.angle = clamped
                    return
                except Exception as exc:
                    print(f"[SERVO ERROR] gpiozero angle error on Cat {cat}: {exc}")
            if self.lgpio_handle is not None and cat in self.lgpio_pins and lgpio is not None:
                try:
                    lgpio.tx_servo(
                        self.lgpio_handle, self.lgpio_pins[cat], servo_pulse_us(clamped),
                        pulse_frequency=50,
                    )
                    return
                except Exception as exc:
                    print(f"[SERVO ERROR] lgpio tx_servo error on Cat {cat}: {exc}")
            if not self.is_simulation:
                print(f"  [HARDWARE WARNING] Servo {cat} has no physical GPIO driver attached!")

    def _detach(self, cat: int):
        servo = self.servos.get(cat)
        if servo is not None:
            try:
                servo.value = None
            except Exception:
                pass
        if self.lgpio_handle is not None and cat in self.lgpio_pins and lgpio is not None:
            try:
                lgpio.tx_servo(self.lgpio_handle, self.lgpio_pins[cat], 0, pulse_frequency=50)
            except Exception:
                pass

    def trigger(self, cat: int, onion_id=None):
        if cat not in self.queues:
            log_actuator(f"servo category {cat}", "error", PUSH_HOLD_TIME, detail="invalid category")
            return False
        with self._command_lock:
            if onion_id:
                if onion_id in self._started_ids:
                    log_actuator(
                        f"servo BCM {SERVO_PINS[cat]}", "ignored", PUSH_HOLD_TIME,
                        detail=f"onion={onion_id} already started",
                    )
                    return False
                self._started_ids.add(onion_id)
            delay = FRONT_DELAY if cat in (1, 2) else REAR_DELAY
            self.queues[cat].put({"delay": delay, "timestamp": time.time(), "onion_id": onion_id})
        log_actuator(
            f"servo BCM {SERVO_PINS[cat]}", "queued", PUSH_HOLD_TIME,
            detail=f"category={cat} onion={onion_id or 'unspecified'} hold={PUSH_HOLD_TIME:.1f}s",
        )
        return True

    def halt(self):
        """Stop the current hold and drop queued moves. Pins stay available for the next batch."""
        self._cancel += 1
        with self._command_lock:
            for q in self.queues.values():
                while True:
                    try:
                        q.get_nowait()
                        q.task_done()
                    except queue.Empty:
                        break
        for cat, angles in SERVO_ANGLES.items():
            self._write_angle(cat, angles["rest"])
        log_actuator("servo", "halt", PUSH_HOLD_TIME, time.time(), time.time(), detail="returned to rest")

    def cleanup(self):
        self.running = False
        for q in self.queues.values():
            q.put(None)

        # Park gpiozero servos
        for cat, servo in self.servos.items():
            if servo is not None:
                try:
                    servo.angle = SERVO_ANGLES[cat]["rest"]
                    time.sleep(0.1)
                    servo.close()
                except Exception:
                    pass

        # Close native lgpio handle
        if self.lgpio_handle is not None and lgpio is not None:
            for cat, pin in self.lgpio_pins.items():
                try:
                    lgpio.tx_servo(self.lgpio_handle, pin, 0, pulse_frequency=50)
                except Exception:
                    pass
            try:
                lgpio.gpiochip_close(self.lgpio_handle)
            except Exception:
                pass


# Singleton instance
_servo_manager = None
_servo_manager_lock = threading.Lock()


def get_servo_manager() -> RaspberryPiServoManager:
    global _servo_manager
    with _servo_manager_lock:
        if _servo_manager is None:
            _servo_manager = RaspberryPiServoManager()
        return _servo_manager


def send_servo_command(cmd_char, onion_id=None):
    """
    Queue one recorded sorting position.
    Inspection calls this only after hardware_config marks the signal wire verified.
    """
    if not _HW.SERVO_SIGNAL_VERIFIED:
        log_actuator(
            "servo", "disabled", PUSH_HOLD_TIME,
            detail="servo signal wire is not confirmed in hardware_config.py",
        )
        return False
    try:
        cat = int(str(cmd_char).strip())
        if cat not in SERVO_PINS:
            return False
        manager = get_servo_manager()
        return bool(manager.trigger(cat, onion_id=onion_id))
    except Exception as e:
        log_actuator(f"servo command {cmd_char}", "error", PUSH_HOLD_TIME, detail=str(e))
        return False


# ==============================================================================
# CONVEYOR INTERVAL
# ==============================================================================
# The 8-second interval is kept. Physical motor output is not.
# Wiring lives only in hardware_config.py. This controller does not claim a GPIO pin.


def log_actuator(device, state, duration, started=None, stopped=None, detail=""):
    """Record one motor or servo change. Times are seconds since the epoch."""
    print(
        f"[ACTUATOR] device={device} state={state} start={started} "
        f"duration={float(duration):.1f}s stop={stopped} detail={detail}"
    )


def hardware_configuration():
    """Return the centralized hardware snapshot. Callers cannot change the stored pins."""
    return _HW.configuration()


def execute_servo_diagnostic(category, writer, sleeper, clock, should_continue=None):
    """Move one sorting servo once. The caller supplies the writer, so tests do not open GPIO."""
    if category not in SERVO_PINS:
        return "invalid"
    angles = SERVO_ANGLES[category]
    return run_servo_sort(
        writer,
        sleeper,
        clock,
        should_continue or (lambda: True),
        rest_angle=angles["rest"],
        push_angle=angles["push"],
        pin=SERVO_PINS[category],
        onion_id="diagnostic",
    )


def run_attached_servo_diagnostic(argv):
    """Print the recorded servo test. GPIO moves only after --confirm-signal-pin. The belt stays off."""
    try:
        category = int(argv[argv.index("--servo-test") + 1])
    except (IndexError, ValueError):
        print("Usage: python final.py --servo-test <1-4> [--confirm-signal-pin]")
        print("Without --confirm-signal-pin this command only prints the recorded servo settings.")
        return 2
    report = hardware_configuration()
    overlap = set(SERVO_PINS.values()) & set(report["motor_control_pins_bcm"].values())
    if overlap:
        print(f"Refusing the servo test because these BCM lines are also motor pins: {sorted(overlap)}")
        return 1
    if category not in SERVO_PINS:
        print(f"Category {category} is not configured. Choose one of {sorted(SERVO_PINS)}.")
        return 2
    pin = SERVO_PINS[category]
    angles = SERVO_ANGLES[category]
    print(f"[SERVO TEST] Recorded signal line: BCM GPIO {pin} ({BOARD_PIN_DESC.get(pin, 'header position not labeled')})")
    print(
        f"[SERVO TEST] {report['servo_min_pulse_us']}-{report['servo_max_pulse_us']} us "
        f"at {report['servo_frequency_hz']} Hz"
    )
    print(
        f"[SERVO TEST] rest {angles['rest']} degrees, push {angles['push']} degrees, "
        f"hold {PUSH_HOLD_TIME:.0f} seconds, then one return to rest"
    )
    print("[SERVO TEST] The conveyor is not started.")
    print("[SERVO TEST] Confirm the signal wire is that BCM GPIO before adding --confirm-signal-pin.")
    print("[SERVO TEST] Power the servo from an external supply. Do not power it from a GPIO pin.")
    if "--confirm-signal-pin" not in argv:
        print("[SERVO TEST] No pulse was generated. The signal wire is not marked verified.")
        return 2
    manager = get_servo_manager()

    def writer(angle, selected=category):
        manager._write_angle(selected, angle)

    try:
        status = execute_servo_diagnostic(category, writer, time.sleep, time.time)
    except KeyboardInterrupt:
        manager.halt()
        print("[SERVO TEST] Interrupted. The servo was sent back toward rest.")
        return 1
    finally:
        manager.halt()
    print(f"[SERVO TEST] finished status={status}")
    return 0 if status == "ok" else 1


def wait_interruptible(seconds, sleeper, clock, should_continue, started_at=None):
    """Sleep in short steps so Stop can end a hold without blocking the camera thread."""
    origin = clock() if started_at is None else float(started_at)
    deadline = origin + float(seconds)
    while clock() < deadline:
        if not should_continue():
            return "aborted"
        step = min(0.1, deadline - clock())
        if step <= 0:
            break
        sleeper(step)
    if not should_continue():
        return "aborted"
    return "ok"


def run_servo_sort(
    writer, sleeper, clock, should_continue, rest_angle, push_angle,
    hold_seconds=None, settle_seconds=None, onion_id="", pin=None,
):
    """Move once to the sorting angle, hold, then return once to rest."""
    hold = float(PUSH_HOLD_TIME if hold_seconds is None else hold_seconds)
    settle = float(SETTLE_TIME if settle_seconds is None else settle_seconds)
    started = clock()
    device = f"servo BCM {pin}" if pin is not None else "servo"
    log_actuator(device, "start", hold, started, detail=f"onion={onion_id} push={push_angle}")
    status = "ok"
    try:
        if not should_continue():
            status = "aborted"
        else:
            writer(push_angle)
            status = wait_interruptible(hold, sleeper, clock, should_continue)
        writer(rest_angle)
        if status == "ok" and should_continue():
            settled = wait_interruptible(settle, sleeper, clock, should_continue)
            if settled != "ok":
                status = settled
    except Exception as exc:
        try:
            writer(rest_angle)
        except Exception:
            pass
        log_actuator(device, "error", hold, started, clock(), detail=f"onion={onion_id} error={exc}")
        raise
    log_actuator(device, status, hold, started, clock(), detail=f"onion={onion_id} rest={rest_angle}")
    return status


class ConveyorBeltManager:
    """Keeps the 8-second belt interval. It does not drive a GPIO pin."""
    def __init__(self):
        self.physical_enabled = bool(_HW.MOTOR_PHYSICAL_ENABLED and _HW.MOTOR_WIRING_VERIFIED)
        self.is_simulation = not self.physical_enabled
        self.on_finished = None
        self._running = True
        self._abort = False
        if self.physical_enabled:
            print("[CONVEYOR] Motor wiring is marked verified, but this program has no driver for it.")
            print("[CONVEYOR] Physical output stays off. A GPIO pin will not power the motor.")
            self.physical_enabled = False
            self.is_simulation = True
        else:
            print(f"[CONVEYOR] {_HW.MOTOR_DISABLED_REASON}")

        self._cycles = ConveyorCycleQueue(
            motor_on=self._motor_on,
            motor_off=self._motor_off,
            should_continue=lambda: self._running and not self._abort,
        )
        self._wake = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def enqueue(self, onion_id):
        if not self.physical_enabled:
            log_actuator("conveyor", "disabled", CONVEYOR_RUN_TIME, detail=_HW.MOTOR_DISABLED_REASON)
            return False
        accepted = self._cycles.enqueue(onion_id)
        if accepted:
            self._wake.set()
        return accepted

    def trigger(self, onion_id=None):
        """Queue one cycle for this onion. A bare trigger is not a sort."""
        return self.enqueue(onion_id)

    def _motor_on(self):
        self._cycle_started = time.time()
        log_actuator(
            "conveyor", "disabled", CONVEYOR_RUN_TIME, self._cycle_started,
            detail="no verified motor driver; GPIO was not changed",
        )

    def _motor_off(self):
        log_actuator(
            "conveyor", "stop", CONVEYOR_RUN_TIME,
            getattr(self, "_cycle_started", None), time.time(),
            detail="no motor GPIO is claimed",
        )

    def _loop(self):
        while self._running:
            self._wake.wait(0.2)
            self._wake.clear()
            while self._running:
                outcome = self._cycles.run_next()
                if outcome is None:
                    if self._abort:
                        self._abort = False
                        self._motor_off()
                    break
                onion_id, ok, reason = outcome
                if reason == "aborted":
                    self._abort = False
                callback = self.on_finished
                if callback is not None:
                    try:
                        callback(onion_id, ok, reason)
                    except Exception as exc:
                        print(f"[CONVEYOR] Completion callback failed: {exc}")

    def halt(self):
        """Stop the belt immediately and drop cycles that have not started."""
        self._abort = True
        self._cycles.clear_pending()
        self._wake.set()
        self._motor_off()
        log_actuator("conveyor", "halt", CONVEYOR_RUN_TIME, detail="pending cycles cancelled")

    def cleanup(self):
        self._running = False
        self._abort = True
        self._cycles.clear_pending()
        self._wake.set()
        self._motor_off()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)


_conveyor_manager = None
_conveyor_manager_lock = threading.Lock()


def halt_motion():
    """Stop the belt and return servos to rest without opening GPIO if it was never used."""
    if _conveyor_manager is not None:
        _conveyor_manager.halt()
    if _servo_manager is not None:
        _servo_manager.halt()


def _release_gpio_on_exit():
    halt_motion()
    if _conveyor_manager is not None:
        _conveyor_manager.cleanup()
    if _servo_manager is not None:
        _servo_manager.cleanup()


atexit.register(_release_gpio_on_exit)


def get_conveyor_manager() -> ConveyorBeltManager:
    global _conveyor_manager
    with _conveyor_manager_lock:
        if _conveyor_manager is None:
            _conveyor_manager = ConveyorBeltManager()
        return _conveyor_manager


def trigger_conveyor(onion_id=None):
    """Refuse a belt start until hardware_config names a verified motor driver."""
    if not onion_id:
        print("[CONVEYOR] Ignored a belt start that is not tied to one onion.")
        return False
    if not (_HW.MOTOR_PHYSICAL_ENABLED and _HW.MOTOR_WIRING_VERIFIED):
        log_actuator("conveyor", "disabled", CONVEYOR_RUN_TIME, detail=_HW.MOTOR_DISABLED_REASON)
        return False
    try:
        return bool(get_conveyor_manager().enqueue(onion_id))
    except Exception as e:
        print(f"[CONVEYOR ERROR] Failed to trigger conveyor: {e}")
        return False


# Set TensorFlow log level to reduce console noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() or 1))
except Exception:
    pass
import keras
from keras import layers, models

from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QSize, QRect, QPoint, QStandardPaths
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QLineEdit, QSpinBox, QGridLayout, QHBoxLayout, QVBoxLayout, QBoxLayout, QFrame,
    QGroupBox, QMessageBox, QStatusBar, QSizePolicy, QScrollArea, QDesktopWidget,
    QInputDialog, QLayout, QSplitter, QScroller
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ==============================================================================
# APPLICATION CONFIGURATION & PATHS
# ==============================================================================

if getattr(sys, "frozen", False):
    # Double-click exe: models, spectrum files, and the logo sit beside the program.
    PROJECT_DIR = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    BASE_DIR = os.path.join(PROJECT_DIR, "SRC")
    TRAINING_DIR = os.path.join(PROJECT_DIR, "Weight Files")
else:
    # Source_App/app.py -> Windows_App -> Application -> 2_SRC -> project root
    _source_dir = os.path.dirname(os.path.abspath(__file__))
    _windows_app = os.path.dirname(_source_dir)
    _application = os.path.dirname(_windows_app)
    _src_root = os.path.dirname(_application)
    BASE_DIR = os.path.join(_windows_app, "assets")
    PROJECT_DIR = os.path.join(_application, "Common")
    TRAINING_DIR = os.path.join(_src_root, "Training_DL_Models")

UNIVERSITY_LOGO_PATH = os.path.join(BASE_DIR, "New smartlab logo.png")
OFFICIAL_LOGO_PATH = os.path.join(BASE_DIR, "resources", "official-logo.png")
OFFICIAL_ICON_PATH = os.path.join(BASE_DIR, "resources", "official-logo.ico")
LOGO_PATH = UNIVERSITY_LOGO_PATH
EXISTING_SIZE_APP_PATH = os.path.join(PROJECT_DIR, "Edge Detection", "D455f")
XAI_FILE_PATH = os.path.join(PROJECT_DIR, "XAI Gradcam")
SPECTRUM_WEIGHTS = os.path.join(PROJECT_DIR, "Spectrum_single", "weights")
REFLECTANCE_EXCEL_DIR = os.path.join(PROJECT_DIR, "Spectrum_single", "excel")

# Fixed camera-to-conveyor-belt distance for the current physical setup (mm)
# Camera is mounted 24 cm above the conveyor belt.
CONVEYOR_DEPTH_MM = 240.0

def _weight_file(model_folder, filename):
    """Frozen builds keep every weight in one folder. The source tree keeps one folder per model."""
    if getattr(sys, "frozen", False):
        return os.path.join(TRAINING_DIR, filename)
    return os.path.join(TRAINING_DIR, model_folder, "weights", filename)


MODEL_PATHS = {
    "ConvNeXt Tiny": _weight_file("Model_1", "convnexttiny_model.weights (1).h5"),
    "EfficientNetV2-B0": _weight_file("Model_2", "effnetv2b0model.weights.h5"),
    "DenseNet121": _weight_file("Model_3", "onion_densenet121_best_macro_f1.weights.h5"),
    "ResNet": _weight_file("Model_4", "resnet_model.weights.h5"),
    "Xception": _weight_file("Model_5", "xception_model.weights.h5"),
    "Stacking Meta-Classifier": _weight_file("Evaluation", "onion_stacking_meta_classifier_best.keras"),
    "Ensemble Model": None
}

CAMERA_INDEX = 0
CAPTURE_INTERVAL_MS = 3000

# Four disease classes: Moldy (molded), Normal, Rotten, Sprouted
CLASS_NAMES = ["Moldy", "Normal", "Rotten", "Sprouted"]

# Current-frame onion checks. A blob must pass these limits before a diameter,
# disease label, counter, or sorting command is allowed.
ONION_VALIDATION = {
    "min_area": 900.0,
    "max_area": 80000.0,
    "min_circularity": 0.46,
    "max_aspect": 1.55,
    "min_radius_px": 18.0,
    "max_radius_px": 150.0,
    "min_fill": 0.40,
    "hue_low": 42,
    "hue_high": 150,
    "min_saturation": 40,
    "min_value": 45,
    "max_value": 245,
    "min_interior_fraction": 0.40,
    "min_score": 0.55,
    "depth_raise_mm": 8.0,
    "edge_margin_px": 8,
    "min_disease_confidence": 45.0,
    "min_track_frames": 3,
    "drop_after_misses": 4,
    "min_diameter_mm": 15.0,
    "max_diameter_mm": 130.0,
    "diameter_attempt_limit": 8,
    "diameter_timeout_frames": 45,
    "diameter_min_consistent": 3,
    "diameter_pixel_noise": 2.0,
    "min_crop_side": 24,
    "min_crop_sharpness": 8.0,
    "roi_top": 0.28,
    "roi_bottom": 0.72,
    "roi_left": 0.12,
    "roi_right": 0.88,
}


def diameter_tolerance_mm(focal_px, depth_mm=None, pixel_noise=None):
    """Acceptance band in millimetres from the camera height and focal length.

    One pixel changes the sphere diameter by about 2 * height / focal.
    The band is that change times the configured pixel noise. The millimetre
    conversion itself is unchanged.
    """
    noise = ONION_VALIDATION["diameter_pixel_noise"] if pixel_noise is None else pixel_noise
    try:
        noise = float(noise)
        depth = float(CONVEYOR_DEPTH_MM if depth_mm is None else depth_mm)
        focal = float(focal_px)
    except (TypeError, ValueError):
        return 3.5
    if not math.isfinite(noise) or not math.isfinite(depth) or not math.isfinite(focal):
        return 3.5
    if focal <= 1.0 or depth <= 0.0 or noise <= 0.0:
        return 3.5
    per_pixel = (2.0 * depth) / focal
    return min(6.0, max(2.0, per_pixel * noise))


def diameter_is_valid(diameter_mm):
    """A measured diameter is usable only inside the calibrated range."""
    if isinstance(diameter_mm, bool) or diameter_mm is None:
        return False
    try:
        value = float(diameter_mm)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(value):
        return False
    low = float(ONION_VALIDATION["min_diameter_mm"])
    high = float(ONION_VALIDATION["max_diameter_mm"])
    return low <= value <= high


def size_category_for_diameter(diameter_mm):
    """Map a validated diameter to one size category. A failed measurement has no category."""
    if not diameter_is_valid(diameter_mm):
        return None
    value = float(diameter_mm)
    if value < 40.0:
        return "Category 1"
    if value < 50.0:
        return "Category 2"
    if value <= 60.0:
        return "Category 3"
    return "Category 4"


def record_inspection_counts(counts, counted_ids, onion_id, predicted_class, confidence, diameter_mm):
    """Increment the disease and size counters once for one finalized inspection."""
    if not onion_id or onion_id in counted_ids:
        return None
    disease = disease_display_label(predicted_class, confidence)
    category = size_category_for_diameter(diameter_mm)
    if disease == "UNCERTAIN" or category is None:
        return None
    disease_key = {
        "Normal": "normal",
        "Rotten": "rotten",
        "Sprouted": "sprouted",
        "Moldy": "moldy",
    }[disease]
    size_key = {
        "Category 1": "cat1",
        "Category 2": "cat2",
        "Category 3": "cat3",
        "Category 4": "cat4",
    }[category]
    counts[disease_key] = int(counts.get(disease_key, 0)) + 1
    counts[size_key] = int(counts.get(size_key, 0)) + 1
    counted_ids.add(onion_id)
    return category


_CATEGORY_SERVO = {
    "Category 1": "1",
    "Category 2": "2",
    "Category 3": "3",
    "Category 4": "4",
}
_REJECT_CLASSES = ("Rotten", "Sprouted", "Moldy")


def _record_value(record, name):
    if record is None:
        return None
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def sorting_route(record):
    """Choose one route from a finalized inspection. An incomplete record is not sortable."""
    onion_id = _record_value(record, "onion_id")
    if not onion_id:
        return None
    disease = disease_display_label(
        _record_value(record, "predicted_class"),
        _record_value(record, "confidence"),
    )
    category = size_category_for_diameter(_record_value(record, "size_mm"))
    if disease == "UNCERTAIN" or category is None:
        return None
    if disease == "Normal":
        return {
            "onion_id": onion_id,
            "route": "sort",
            "category": category,
            "servo": _CATEGORY_SERVO[category],
            "conveyor": True,
        }
    if disease in _REJECT_CLASSES:
        return {
            "onion_id": onion_id,
            "route": "reject",
            "category": category,
            "servo": None,
            "conveyor": True,
        }
    return None


def execute_sorting_route(route, servo_fn, conveyor_fn):
    """Run a route through injected hardware. A failure does not count as sorted."""
    if not route or not route.get("onion_id"):
        return False
    try:
        servo_cmd = route.get("servo")
        if servo_cmd:
            if servo_fn(servo_cmd) is False:
                return False
        if route.get("conveyor"):
            if conveyor_fn(route["onion_id"]) is False:
                return False
        return True
    except Exception as exc:
        print(f"[SORT] Actuator failed for {route.get('onion_id')}: {exc}")
        return False


class BatchFlow:
    """Detected, inspected, and physically sorted onions for one batch."""

    def __init__(self, slot_target=0):
        self.slot_target = max(0, int(slot_target or 0))
        self.detected = set()
        self.inspected = {}
        self.queued = set()
        self.sorted = set()
        self.failed = set()

    def note_detected(self, onion_id):
        if onion_id is None or onion_id == "":
            return False
        self.detected.add(onion_id)
        return True

    def queue_finalized(self, record):
        """Store one inspection and queue it once. Classification is not physical completion."""
        route = sorting_route(record)
        if route is None:
            return None
        onion_id = route["onion_id"]
        if onion_id in self.inspected or onion_id in self.queued or onion_id in self.sorted:
            return None
        self.detected.add(onion_id)
        self.inspected[onion_id] = route
        self.queued.add(onion_id)
        return route

    def mark_physically_processed(self, onion_id, success):
        """Mark a queued onion sorted only after its actuator cycle reports success."""
        if onion_id not in self.queued or onion_id not in self.inspected:
            return False
        if onion_id in self.sorted or onion_id in self.failed:
            return False
        if not success:
            self.failed.add(onion_id)
            return False
        self.sorted.add(onion_id)
        return True

    def counts(self):
        detected = len(self.detected)
        inspected = len(self.inspected)
        sorted_count = len(self.sorted)
        remaining = max(0, self.slot_target - sorted_count)
        return {
            "detected": detected,
            "inspected": inspected,
            "sorted": sorted_count,
            "remaining": remaining,
            "consistent": detected >= inspected >= sorted_count and remaining == max(0, self.slot_target - sorted_count),
        }


class ConveyorCycleQueue:
    """One belt cycle at a time. Motor functions are injected so tests never open GPIO."""

    def __init__(self, run_seconds=None, sleeper=None, clock=None, motor_on=None, motor_off=None, should_continue=None):
        self.run_seconds = float(CONVEYOR_RUN_TIME if run_seconds is None else run_seconds)
        self.sleeper = sleeper or time.sleep
        self.clock = clock or time.time
        self.motor_on = motor_on or (lambda: None)
        self.motor_off = motor_off or (lambda: None)
        self.should_continue = should_continue or (lambda: True)
        self._lock = threading.Lock()
        self.pending = []
        self._seen = set()

    def enqueue(self, onion_id):
        if not onion_id:
            return False
        with self._lock:
            if onion_id in self._seen:
                return False
            self._seen.add(onion_id)
            self.pending.append(onion_id)
        return True

    def clear_pending(self):
        with self._lock:
            self.pending.clear()

    def run_next(self):
        with self._lock:
            if not self.pending:
                return None
            onion_id = self.pending.pop(0)
        if not self.should_continue():
            return onion_id, False, "aborted"
        try:
            self.motor_on()
            waited = self._wait(self.run_seconds)
        except Exception:
            self._safe_off()
            return onion_id, False, "actuator"
        self._safe_off()
        if waited != "ok":
            return onion_id, False, waited
        return onion_id, True, "ok"

    def _wait(self, seconds):
        started = self.clock()
        deadline = started + float(seconds)
        timeout_at = started + float(seconds) + CONVEYOR_TIMEOUT_GRACE
        while self.clock() < deadline:
            if not self.should_continue():
                return "aborted"
            step = min(0.1, deadline - self.clock())
            if step <= 0:
                break
            self.sleeper(step)
            if self.clock() > timeout_at:
                return "timeout"
        if self.clock() > timeout_at:
            return "timeout"
        return "ok"

    def _safe_off(self):
        try:
            self.motor_off()
        except Exception:
            pass


def format_detection_log(frame_number, frame_shape, state, track_id, score, bbox, reason):
    """One diagnostic line for the current camera frame. It does not create a detection."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    score_text = "none" if score is None else f"{float(score):.2f}"
    return (
        f"[DETECT] frame={int(frame_number)} size={width}x{height} bgr8 "
        f"state={state} id={track_id} score={score_text} bbox={bbox} reason={reason}"
    )


def describe_disease_model(model_name="DenseNet121"):
    """The disease classifier inspects a crop. It is not the onion detector."""
    side = 299 if model_name == "Xception" else 224
    return {
        "model": model_name,
        "classes": list(CLASS_NAMES),
        "input_size": (side, side, 3),
        "color": "RGB",
        "preprocessing": "resize to the model input, then rescale by 1/255 in the trained head",
        "min_confidence": float(ONION_VALIDATION["min_disease_confidence"]),
        "weights": MODEL_PATHS.get(model_name),
    }


def inspection_action(status, diameter_mm):
    """Disease classification is allowed only after diameter validation succeeds."""
    if status == "valid" and diameter_is_valid(diameter_mm):
        return "classify"
    if status == "failed":
        return "diameter_failed"
    return "pending"


def inspection_stage(
    onion_visible,
    frames_tracked,
    diameter_status,
    diameter_mm,
    classification_state="",
    sort_started=False,
):
    """ONION DETECTED → TRACKING → MEASURING DIAMETER → DIAMETER VERIFIED → DISEASE CLASSIFICATION → INSPECTION COMPLETE → SORTING."""
    if not onion_visible:
        return "cleared"
    if diameter_status == "unstable":
        return "unstable"
    if diameter_status == "failed":
        return "diameter_failed"
    tracked = int(frames_tracked or 0)
    if tracked <= 0:
        return "detected"
    if tracked == 1 and diameter_status != "valid":
        return "tracking"
    if diameter_status != "valid" or not diameter_is_valid(diameter_mm):
        return "measuring"
    if classification_state == "timed_out":
        return "classification_timeout"
    if classification_state != "done":
        return "classifying"
    if not sort_started:
        return "complete"
    return "sorting"


def classification_is_allowed(stage, diameter_status, diameter_mm):
    """The disease model stays closed until this onion has a verified diameter."""
    return (
        stage == "classifying"
        and inspection_action(diameter_status, diameter_mm) == "classify"
    )


def same_onion_measurement(track_label, locked_diameter, result_onion_id, result_diameter):
    """The diameter, crop, and disease result must name one tracking ID."""
    if not track_label or result_onion_id != track_label:
        return False
    if not diameter_is_valid(locked_diameter) or not diameter_is_valid(result_diameter):
        return False
    return abs(float(locked_diameter) - float(result_diameter)) <= 0.2


def classify_onion_disease(model_manager, model_name, image_rgb, diameter_mm):
    """Run the disease model only when this onion has a validated diameter."""
    if not diameter_is_valid(diameter_mm):
        return None
    if image_rgb is None or getattr(image_rgb, "size", 0) == 0:
        return None
    return model_manager.predict(model_name, image_rgb)


def disease_display_label(predicted_class, confidence):
    """Return the disease name, or UNCERTAIN when the result cannot be trusted."""
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return "UNCERTAIN"
    if not math.isfinite(confidence_value):
        return "UNCERTAIN"
    if confidence_value < float(ONION_VALIDATION["min_disease_confidence"]):
        return "UNCERTAIN"
    if predicted_class not in ("Normal", "Rotten", "Sprouted", "Moldy"):
        return "UNCERTAIN"
    return predicted_class


def label_obj_id(onion_label):
    text = str(onion_label or "")
    if "-" in text and text.rsplit("-", 1)[-1].isdigit():
        return int(text.rsplit("-", 1)[-1])
    return None


def prediction_belongs_to_visible_onion(result_onion_id, visible_labels, cancelled_ids):
    """A disease result may update the live panel only for the onion in this frame."""
    if not result_onion_id or result_onion_id in set(cancelled_ids or ()):
        return False
    return result_onion_id in set(visible_labels or ())


def result_belongs_to_current_onion(result_onion_id, current_target_id, visible_labels):
    """A finished classification counts only for the onion that is still being inspected."""
    if not result_onion_id or result_onion_id != current_target_id:
        return False
    return result_onion_id in set(visible_labels or ())


def inspection_is_busy(classification_state, sort_started, sort_settled):
    """Disease inference or a conveyor cycle still belongs to this onion."""
    if classification_state == "running":
        return True
    return bool(sort_started) and not bool(sort_settled)


def retain_target_on_visibility_change(current_target, classification_state, sort_started, sort_settled):
    """Keep the in-progress onion after it leaves the camera. Release it once its work has finished."""
    if not current_target:
        return None
    if inspection_is_busy(classification_state, sort_started, sort_settled):
        return current_target
    return None


def track_release_allowed(onion_label, visible_labels, classification_state, sort_started, sort_settled):
    """Drop the tracker entry only after the onion has left and sorting is no longer pending."""
    if not onion_label or onion_label in set(visible_labels or ()):
        return False
    return not inspection_is_busy(classification_state, sort_started, sort_settled)


def result_disposition(result_onion_id, submitted_ids, finalized_ids, visible_labels):
    """A late result may finish its own onion. It must not replace another onion or be stored twice."""
    if not result_onion_id or result_onion_id not in set(submitted_ids or ()):
        return "reject"
    if result_onion_id in set(finalized_ids or ()):
        return "duplicate"
    if result_onion_id in set(visible_labels or ()):
        return "accept_and_show"
    return "accept_quietly"


def counter_milestone(event):
    """Disease and size counters move when the record is finalized. Processed moves after the belt cycle."""
    if event == "inspection_finalized":
        return "disease_and_size"
    if event == "physical_sort_finished":
        return "processed"
    return None


REVIEW_WAITING = "WAITING FOR VALID ONION"
REVIEW_MEASURING = "MEASURING DIAMETER"
REVIEW_ANALYZING = "DIAMETER VERIFIED — ANALYZING DISEASE"


def camera_offline_response(error):
    """A disconnected camera or a failed read must not be replaced with a synthetic onion."""
    detail = str(error).strip() or "frame read failed"
    return {
        "simulate": False,
        "running": False,
        "message": f"Camera unavailable: {detail}",
    }


def prepare_live_frame(frame_bgr):
    """Copy one camera frame and convert BGR to RGB. The camera buffer is not reused."""
    if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3 or frame_bgr.shape[2] < 3:
        return None, None
    if int(frame_bgr.shape[0]) < 32 or int(frame_bgr.shape[1]) < 32:
        return None, None
    bgr = np.ascontiguousarray(frame_bgr[:, :, :3])
    if bgr.dtype != np.uint8:
        return None, None
    bgr = bgr.copy()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bgr, rgb


def clamp_crop_box(bbox, width, height):
    """Keep a crop box inside the current frame."""
    if bbox is None or len(bbox) != 4:
        return None
    try:
        x, y, box_w, box_h = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    except (TypeError, ValueError):
        return None
    if box_w <= 0 or box_h <= 0 or width <= 0 or height <= 0:
        return None
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    box_w = min(box_w, width - x)
    box_h = min(box_h, height - y)
    minimum = int(ONION_VALIDATION["min_crop_side"])
    if box_w < minimum or box_h < minimum:
        return None
    return x, y, box_w, box_h


def crop_sharpness(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_is_usable(image_rgb):
    """Reject an empty, tiny, or blurry crop before it can become an inspection image."""
    if image_rgb is None or getattr(image_rgb, "ndim", 0) != 3 or image_rgb.shape[2] < 3:
        return False
    height, width = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    minimum = int(ONION_VALIDATION["min_crop_side"])
    if height < minimum or width < minimum or image_rgb.size == 0:
        return False
    if image_rgb.dtype != np.uint8:
        return False
    if float(np.std(image_rgb)) < 2.0:
        return False
    if crop_sharpness(image_rgb) < float(ONION_VALIDATION["min_crop_sharpness"]):
        return False
    return True


def extract_inspection_crop(frame_rgb, bbox, track_id, frame_token):
    """Cut this frame's RGB crop for this tracking ID. A stale token cannot be reused."""
    if frame_rgb is None or track_id is None or frame_token is None:
        return None
    if getattr(frame_rgb, "ndim", 0) != 3 or frame_rgb.shape[2] < 3:
        return None
    height, width = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
    box = clamp_crop_box(bbox, width, height)
    if box is None:
        return None
    x, y, box_w, box_h = box
    crop = np.ascontiguousarray(frame_rgb[y:y + box_h, x:x + box_w, :3]).copy()
    if not crop_is_usable(crop):
        return None
    return {
        "image": crop,
        "track_id": track_id,
        "frame_token": frame_token,
        "bbox": box,
    }


def crop_belongs_to_track(crop_info, track_id, frame_token):
    """Accept a crop only when it came from the current frame and the current onion."""
    if not crop_info or crop_info.get("track_id") != track_id:
        return False
    if crop_info.get("frame_token") != frame_token:
        return False
    image = crop_info.get("image")
    return crop_is_usable(image) and PhysicalOnionTracker.crop_matches_onion(image)


def review_panel_message(onion_visible, diameter_status, disease_complete, diameter_mm=None):
    """Status text for the review panel. The disease label is not one of these messages."""
    if not onion_visible:
        return REVIEW_WAITING
    if diameter_status == "unstable":
        return "MEASUREMENT UNSTABLE"
    if diameter_status == "failed":
        return "DIAMETER NOT AVAILABLE"
    if diameter_status != "valid":
        return REVIEW_MEASURING
    if not disease_complete:
        if diameter_is_valid(diameter_mm):
            return f"DIAMETER VERIFIED — {float(diameter_mm):.1f} mm — ANALYZING DISEASE"
        return REVIEW_ANALYZING
    return None


class SpectrumTransfer:
    """RGB to reflectance, same transfer as Spectrum_single."""

    def __init__(self):
        self.M = None
        self.pca_eigenvectors = None
        self.pca_mean = None
        self.TransMin = 0.0
        self.TransMax = 1.0
        self.extend_mode = 0
        self.TransM = None

    def get_extend_color(self, colors):
        r = colors[:, 0] / 255.0
        g = colors[:, 1] / 255.0
        b = colors[:, 2] / 255.0
        if self.extend_mode == 0:
            extend = np.array([r, g, b])
        elif self.extend_mode == 1:
            extend = np.array([r, g, b, r ** 2, g ** 2, b ** 2, r * g, r * b, g * b])
        else:
            extend = np.array([
                r, g, b, r ** 2, g ** 2, b ** 2, r * g, r * b, g * b, r ** 3, g ** 3, b ** 3
            ])
        return extend

    def prepare(self):
        self.TransM = np.dot(self.M.T, self.pca_eigenvectors)

    def load_files(self, weights_path):
        self.M = np.load(os.path.join(weights_path, "a.npy"))
        self.pca_eigenvectors = np.load(os.path.join(weights_path, "b.npy"))
        self.pca_mean = np.load(os.path.join(weights_path, "c.npy"))
        d = np.load(os.path.join(weights_path, "d.npy"))
        self.TransMin = float(d[0])
        self.TransMax = float(d[1])
        e = np.load(os.path.join(weights_path, "e.npy"))
        self.extend_mode = int(e[0])
        self.prepare()

    def transfer(self, src_data):
        src_data = src_data.astype(np.float32)
        src_shape = src_data.shape
        flat = src_data.reshape(-1, src_shape[-1]) if len(src_shape) >= 3 else src_data
        tar_spec = np.dot(self.get_extend_color(flat).T, self.TransM) + self.pca_mean
        if len(src_shape) >= 3:
            tar_spec = tar_spec.reshape(src_shape[0:-1] + (self.pca_mean.shape[0],))
        return (tar_spec - self.TransMin) / (self.TransMax - self.TransMin)


def reflectance_curve_is_valid(curve):
    """A measurement is a finite spectrum. The fixed reference file is not checked here."""
    if curve is None:
        return False
    values = np.asarray(curve, dtype=np.float64).ravel()
    return values.size >= 10 and bool(np.all(np.isfinite(values)))


def reflectance_sample_blocks(image_rgb):
    """Onion-skin blocks used for a live curve. Too little onion skin means there is no measurement."""
    if image_rgb is None or getattr(image_rgb, "ndim", 0) != 3 or image_rgb.shape[2] < 3:
        return None
    if image_rgb.dtype != np.uint8:
        return None
    image = np.ascontiguousarray(image_rgb[:, :, :3])
    height, width = image.shape[:2]
    if height < 8 or width < 8:
        return None
    block = 8
    grid_h, grid_w = height // block, width // block
    if grid_h < 1 or grid_w < 1:
        return None
    image = image[:grid_h * block, :grid_w * block]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    onion = (
        ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 155))
        & (hsv[:, :, 1] >= 28)
        & (hsv[:, :, 2] >= 25)
    )
    block_img = (
        image.reshape(grid_h, block, grid_w, block, 3)
        .mean(axis=(1, 3))
        .astype(np.uint8)
    )
    block_onion = onion.reshape(grid_h, block, grid_w, block).mean(axis=(1, 3)) >= 0.35
    if int(np.count_nonzero(block_onion)) < 3:
        return None
    return block_img, block_onion


def reflectance_plot_series(reference_curve, sample_curve):
    """Keep the Excel normal curve labeled as a reference. Plot a sample only when it is a measurement."""
    series = []
    if reflectance_curve_is_valid(reference_curve):
        reference = np.asarray(reference_curve, dtype=np.float64).ravel()
        series.append(("Normal reference", reference))
    else:
        reference = None
    if not reflectance_curve_is_valid(sample_curve):
        return series
    sample = np.asarray(sample_curve, dtype=np.float64).ravel()
    if reference is not None and sample.shape == reference.shape and np.allclose(sample, reference):
        return series
    series.append(("Current onion", sample))
    return series


def gradcam_class_index(predicted_class):
    """Index of the disease prediction that an explanation must use."""
    try:
        return CLASS_NAMES.index(predicted_class)
    except ValueError:
        return None


def xai_result_matches(request_capture, request_onion, request_class, item_capture, item_onion, item_class, overlay, request_token, active_token):
    """Accept a heatmap only for the onion and prediction that were requested."""
    if overlay is None or request_token != active_token:
        return False
    if gradcam_class_index(request_class) is None or request_class != item_class:
        return False
    if not request_onion or request_onion != item_onion:
        return False
    return int(request_capture) == int(item_capture)


class SpectrumEngine:
    """One reflectance curve for the onion in a captured image."""

    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.transfer = SpectrumTransfer()
        self.ready = False

    def ensure(self):
        if self.ready:
            return
        self.transfer.load_files(self.weights_path)
        self.ready = True

    def mean_curve(self, rgb_uint8: np.ndarray):
        """Reflectance of onion-skin pixels. Background and the Excel reference are not measurements."""
        blocks = reflectance_sample_blocks(rgb_uint8)
        if blocks is None:
            return None
        block_img, block_onion = blocks
        self.ensure()
        spectrum = self.transfer.transfer(block_img).astype(np.float32)
        curve = spectrum[block_onion].mean(axis=0)
        if not reflectance_curve_is_valid(curve):
            return None
        return curve


class ReflectanceReferences:
    """Normal and one Abnormal curve from the Excel workbooks.

    Molded, sprouted, and rotten stay one Abnormal curve. They are not separate classes.
    """

    def __init__(self, excel_dir):
        self.wavelengths = np.arange(380, 781, dtype=np.float64)
        self.normal = None
        self.abnormal = None
        self.error = ""
        try:
            normal_wave, normal = self._load_average(os.path.join(excel_dir, "Normal.xlsx"))
            abnormal_curves = []
            for name in ("Abnormal.xlsx", "Molded.xlsx", "Sprouted.xlsx", "Roten.xlsx"):
                path = os.path.join(excel_dir, name)
                if name == "Abnormal.xlsx" and os.path.isfile(path):
                    wave, curve = self._load_average(path)
                    self.wavelengths = wave
                    self.abnormal = curve
                    abnormal_curves = []
                    break
                if os.path.isfile(path) and name != "Abnormal.xlsx":
                    wave, curve = self._load_average(path)
                    abnormal_curves.append(np.interp(normal_wave, wave, curve))
            self.wavelengths = normal_wave
            self.normal = normal
            if self.abnormal is None and abnormal_curves:
                self.abnormal = np.mean(np.vstack(abnormal_curves), axis=0)
            if self.normal is None or self.abnormal is None:
                raise FileNotFoundError(excel_dir)
        except Exception as exc:
            self.error = str(exc)

    @staticmethod
    def _load_average(path):
        import openpyxl
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheet = workbook["Ellipse ROI"] if "Ellipse ROI" in workbook.sheetnames else workbook.active
        wavelengths = []
        values = []
        first = True
        for row in sheet.iter_rows(values_only=True):
            if first:
                first = False
                continue
            if not row or row[0] is None:
                continue
            text = str(row[0]).lower().replace("nm", "").strip()
            try:
                wave = float(text)
            except ValueError:
                continue
            if wave < 380 or wave > 780:
                continue
            number = row[-1]
            if not isinstance(number, (int, float)):
                numbers = [value for value in row[1:] if isinstance(value, (int, float))]
                if not numbers:
                    continue
                number = sum(numbers) / len(numbers)
            wavelengths.append(wave)
            values.append(float(number))
        workbook.close()
        if len(values) < 10:
            raise ValueError(f"No reflectance rows in {path}")
        return np.asarray(wavelengths, dtype=np.float64), np.asarray(values, dtype=np.float64)

    @staticmethod
    def _shape(curve):
        values = np.asarray(curve, dtype=np.float64).ravel()
        values = values - np.mean(values)
        scale = np.linalg.norm(values)
        if scale <= 1e-8:
            return values
        return values / scale

    def label_for(self, curve) -> str:
        if self.normal is None or self.abnormal is None or curve is None or len(curve) == 0:
            return ""
        sample = np.asarray(curve, dtype=np.float64).ravel()
        if sample.shape[0] != self.normal.shape[0]:
            sample = np.interp(
                np.linspace(0, 1, self.normal.shape[0]),
                np.linspace(0, 1, sample.shape[0]),
                sample,
            )
        sample = self._shape(sample)
        normal_distance = np.linalg.norm(sample - self._shape(self.normal))
        abnormal_distance = np.linalg.norm(sample - self._shape(self.abnormal))
        return "NORMAL" if normal_distance <= abnormal_distance else "ABNORMAL"


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class InspectionResult:
    capture_number: int
    onion_id: str
    timestamp: str
    image: np.ndarray             # RGB uint8 image array
    predicted_class: str         # "Normal", "Moldy", "Rotten", "Sprouted"
    confidence: float            # Percentage confidence e.g. 95.4
    class_probabilities: Dict[str, float]
    model_name: str
    size_mm: float = 0.0
    size_category: str = "UNKNOWN" # "0-2 mm", "2-4 mm", "4-6 mm", "6-8 mm", ">8 mm"
    usda_grade: str = "UNKNOWN"    # "SMALL", "MEDIUM", "LARGE"
    xai_image: Optional[np.ndarray] = None
    batch_no: str = ""
    inference_image: Optional[np.ndarray] = None
    reflectance: Optional[np.ndarray] = None
    spectral_label: str = ""


# ==============================================================================
# DEEP LEARNING MODEL MANAGER
# ==============================================================================

class ModelManager:
    """
    Handles model construction, weight loading, caching, and inference.
    Supports single base models, Stacking Meta-Classifier, and Ensemble.
    """
    def __init__(self, model_paths: Dict[str, str]):
        self.model_paths = model_paths
        self.model_cache: Dict[str, Any] = {}

    def get_model(self, model_name: str):
        """Loads and caches requested model on demand."""
        if model_name in self.model_cache:
            return self.model_cache[model_name]

        path = self.model_paths.get(model_name)
        if not path and model_name not in ["Ensemble Model", "Stacking Meta-Classifier"]:
            raise ValueError(f"No weight file specified for model {model_name}")

        print(f"[ModelManager] Loading {model_name}...")

        if model_name == "ConvNeXt Tiny":
            model = self._build_base_model(tf.keras.applications.ConvNeXtTiny, (224, 224, 3), name="convnext_tiny")
            model.load_weights(path, by_name=True, skip_mismatch=True)

        elif model_name == "EfficientNetV2-B0":
            model = self._build_base_model(tf.keras.applications.EfficientNetV2B0, (224, 224, 3), name="efficientnetv2-b0")
            model.load_weights(path)

        elif model_name == "DenseNet121":
            model = self._build_base_model(tf.keras.applications.DenseNet121, (224, 224, 3), name="densenet121")
            model.load_weights(path)

        elif model_name == "ResNet":
            # ResNet50V2 architecture matches trained resnet_model.weights.h5
            model = self._build_base_model(tf.keras.applications.ResNet50V2, (224, 224, 3), name="resnet50v2")
            model.load_weights(path)

        elif model_name == "Xception":
            # Full saved model or weights load
            if path and path.endswith(".keras") and os.path.exists(path):
                model = keras.models.load_model(path)
            else:
                model = self._build_base_model(tf.keras.applications.Xception, (299, 299, 3), name="xception")
                model.load_weights(path)

        elif model_name == "Stacking Meta-Classifier":
            if path and os.path.exists(path):
                model = keras.models.load_model(path)
            else:
                raise FileNotFoundError(f"Stacking Meta-Classifier file not found at {path}")

        elif model_name == "Ensemble Model":
            # Ensemble uses probability averaging across all 5 base models
            model = "ENSEMBLE_PIPELINE"

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.model_cache[model_name] = model
        if model_name != "Ensemble Model":
            side = 299 if model_name == "Xception" else 224
            warm = np.zeros((1, side, side, 3), dtype=np.float32)
            try:
                model(warm, training=False)
            except Exception:
                pass
        print(f"[ModelManager] {model_name} loaded successfully.")
        return model

    def _build_base_model(self, base_app, input_shape=(224, 224, 3), name=None):
        """Constructs canonical classification model with custom head matching training architecture."""
        base_model = base_app(include_top=False, weights=None, pooling='avg', input_shape=input_shape)
        if name:
            try:
                base_model._name = name
            except Exception:
                pass

        aug = keras.Sequential([layers.Identity()], name='sequential')
        inputs = layers.Input(shape=input_shape, name='input_layer')
        x = aug(inputs)
        x = layers.Rescaling(1.0 / 255.0, name='rescaling')(x)
        x = base_model(x)
        x = layers.BatchNormalization(name='head_bn')(x)
        x = layers.Dropout(0.3, name='head_dropout_1')(x)
        x = layers.Dense(256, activation='relu', name='head_dense')(x)
        x = layers.Dropout(0.3, name='head_dropout_2')(x)
        outputs = layers.Dense(4, activation='softmax', name='predictions')(x)

        return keras.Model(inputs=inputs, outputs=outputs)

    def predict(self, model_name: str, img_rgb: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Executes model inference on input RGB image array.
        Returns: (predicted_class_name, confidence_percentage, dict_of_all_probabilities)
        """
        base_models = ["ConvNeXt Tiny", "EfficientNetV2-B0", "DenseNet121", "ResNet", "Xception"]

        if model_name in base_models:
            probs = self._predict_single_base(model_name, img_rgb)

        elif model_name == "Stacking Meta-Classifier":
            # Collect base model prediction probability vectors in exact training order
            base_probs_list = []
            for b_name in base_models:
                p_vec = self._predict_single_base(b_name, img_rgb)
                base_probs_list.append(p_vec)

            # Concatenate 5 x 4 = 20 features
            meta_input = np.concatenate(base_probs_list, axis=0).reshape(1, 20).astype(np.float32)
            meta_model = self.get_model("Stacking Meta-Classifier")
            probs_out = meta_model.predict(meta_input, verbose=0)[0]
            probs = probs_out

        elif model_name == "Ensemble Model":
            # Average probabilities across all 5 base models
            all_p = []
            for b_name in base_models:
                p_vec = self._predict_single_base(b_name, img_rgb)
                all_p.append(p_vec)
            probs = np.mean(all_p, axis=0)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Extract predicted class index and confidence
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx] * 100.0)
        pred_class = CLASS_NAMES[pred_idx]

        prob_dict = {CLASS_NAMES[i]: float(probs[i] * 100.0) for i in range(len(CLASS_NAMES))}
        return pred_class, confidence, prob_dict

    def _predict_single_base(self, model_name: str, img_rgb: np.ndarray) -> np.ndarray:
        """Preprocesses image and runs single base model inference."""
        model = self.get_model(model_name)
        target_size = (299, 299) if model_name == "Xception" else (224, 224)

        if img_rgb.shape[:2] != target_size:
            resized = cv2.resize(img_rgb, target_size)
        else:
            resized = img_rgb.copy()

        input_tensor = np.expand_dims(resized.astype(np.float32), axis=0)
        probs = model(input_tensor, training=False).numpy()[0]
        return probs


# ==============================================================================
# EXPLAINABLE AI (GRAD-CAM XAI ENGINE)
# ==============================================================================

class GradCAMEngine:
    """Computes Grad-CAM explainability heatmaps for onion quality inspection."""
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self._prepared = {}

    def _prepare(self, model_name: str):
        """Build the Grad-CAM graph once per model and reuse it."""
        target_model_name = "Xception" if model_name in ["Ensemble Model", "Stacking Meta-Classifier"] else model_name
        if target_model_name in self._prepared:
            return target_model_name, self._prepared[target_model_name]

        model = self.model_manager.get_model(target_model_name)
        base_model = None
        for layer in model.layers:
            if not isinstance(layer, keras.Model):
                continue
            # Skip the tiny augmentation wrapper. The real backbone has many layers.
            if len(getattr(layer, "layers", [])) < 8:
                continue
            base_model = layer
            break
        if base_model is None:
            raise ValueError("Base model not found in the loaded model.")

        last_conv_layer = None
        for layer in reversed(base_model.layers):
            try:
                shape = layer.output_shape
            except Exception:
                continue
            if isinstance(shape, tuple) and len(shape) == 4:
                last_conv_layer = layer
                break
        if last_conv_layer is None:
            raise ValueError("Could not find a convolutional layer.")

        grad_model = keras.Model(inputs=base_model.inputs, outputs=[last_conv_layer.output, base_model.output])
        head_layers = []
        found_base = False
        for layer in model.layers:
            if layer == base_model:
                found_base = True
                continue
            if found_base:
                head_layers.append(layer)

        pre_layers = []
        for layer in model.layers:
            if layer == base_model:
                break
            if layer.name != "input_layer":
                pre_layers.append(layer)

        prepared = (grad_model, pre_layers, head_layers)
        self._prepared[target_model_name] = prepared
        return target_model_name, prepared

    def generate_gradcam(self, img_rgb: np.ndarray, model_name: str = "Xception", predicted_class: str = None):
        """Overlay Grad-CAM for this image and this disease prediction. Failure returns no image."""
        class_index = gradcam_class_index(predicted_class)
        if class_index is None or img_rgb is None or getattr(img_rgb, "size", 0) == 0:
            return None
        try:
            target_model_name, (grad_model, pre_layers, head_layers) = self._prepare(model_name)
            target_size = (299, 299) if target_model_name == "Xception" else (224, 224)
            img_resized = cv2.resize(img_rgb, target_size)
            img_tensor = np.expand_dims(img_resized.astype(np.float32), axis=0)

            with tf.GradientTape() as tape:
                x = img_tensor
                for layer in pre_layers:
                    x = layer(x)

                conv_outputs, base_output = grad_model(x)
                tape.watch(conv_outputs)

                h = base_output
                for layer in head_layers:
                    h = layer(h, training=False) if 'dropout' in layer.name or 'bn' in layer.name else layer(h)

                loss = h[:, class_index]

            grads = tape.gradient(loss, conv_outputs)
            if grads is None:
                return None
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_outputs_val = conv_outputs[0]
            heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(heatmap, 0.0)
            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val
            heatmap_np = heatmap.numpy()
            if heatmap_np is None or not np.all(np.isfinite(heatmap_np)):
                return None

            heatmap_resized = cv2.resize(heatmap_np, (img_rgb.shape[1], img_rgb.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            color_heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            return cv2.addWeighted(img_rgb, 0.55, color_heatmap_rgb, 0.45, 0)
        except Exception as exc:
            print(f"[GradCAMEngine] No explanation produced ({exc})")
            return None


# ==============================================================================
# SPATIAL GEOMETRY & OBJECT TRACKING (NO DOUBLE COUNTING)
# ==============================================================================

class MockIntrinsics:
    def __init__(self, w=640, h=480, fx=385.0, fy=385.0):
        self.width = w
        self.height = h
        self.fx = fx
        self.fy = fy
        self.ppx = w / 2.0
        self.ppy = h / 2.0
        self.coeffs = [0.0] * 5

import sys
if EXISTING_SIZE_APP_PATH not in sys.path:
    sys.path.append(EXISTING_SIZE_APP_PATH)
try:
    import importlib.util
    tracker_file = os.path.join(EXISTING_SIZE_APP_PATH, "main.py")
    spec = importlib.util.spec_from_file_location("d455f_main", tracker_file)
    d455f_main = importlib.util.module_from_spec(spec)
    sys.modules["d455f_main"] = d455f_main
    spec.loader.exec_module(d455f_main)
    MultiObjectTracker = d455f_main.MultiObjectTracker
except Exception as e:
    print(f"Failed to load MultiObjectTracker: {e}")
    MultiObjectTracker = None

class TrackedOnion:
    def __init__(self, obj_id):
        self.obj_id = obj_id
        self.diameters = []
        self.frames_tracked = 0
        self.best_crop = None
        self.best_bbox = None
        self.min_dist_to_center = float('inf')
        self.size_cat = ""
        self.usda_grade = ""
        self.misses = 0


class OnionDiameterFilter:
    """One fixed millimetre diameter per onion, measured from several agreeing frames."""

    MIN_CLUSTER = 3
    CLUSTER_MM = 3.5
    MAX_SAMPLES = 20
    HOLD_FRAMES = 15

    def __init__(self):
        self._samples = {}
        self._sample_radius = {}
        self._locked = {}
        self._locked_radius = {}
        self._positions = {}
        self._failed = set()
        self._unstable = set()
        self._bad_frames = {}
        self._pending_frames = {}
        self._log_signature = None

    def clear(self, obj_id=None):
        if obj_id is None:
            self._samples.clear()
            self._sample_radius.clear()
            self._locked.clear()
            self._locked_radius.clear()
            self._positions.clear()
            self._failed.clear()
            self._unstable.clear()
            self._bad_frames.clear()
            self._pending_frames.clear()
            self._log_signature = None
            return
        self._samples.pop(obj_id, None)
        self._sample_radius.pop(obj_id, None)
        self._locked.pop(obj_id, None)
        self._locked_radius.pop(obj_id, None)
        self._positions.pop(obj_id, None)
        self._failed.discard(obj_id)
        self._unstable.discard(obj_id)
        self._bad_frames.pop(obj_id, None)
        self._pending_frames.pop(obj_id, None)

    def begin_frame(self):
        for position in self._positions.values():
            position[2] += 1

    def end_frame(self):
        expired = [obj_id for obj_id, position in self._positions.items() if position[2] > self.HOLD_FRAMES]
        for obj_id in expired:
            self.clear(obj_id)

    def radius_px(self, obj_id):
        return self._locked_radius.get(obj_id)

    def status(self, obj_id):
        locked = self._locked.get(obj_id)
        if diameter_is_valid(locked):
            return "valid"
        if obj_id in self._failed:
            return "failed"
        if obj_id in self._unstable:
            return "unstable"
        return "pending"

    def filtered(self, obj_id):
        """Rolling median of the accepted readings, or the locked value once it is stable."""
        if obj_id in self._failed or obj_id in self._unstable:
            return None
        locked = self._locked.get(obj_id)
        if diameter_is_valid(locked):
            return float(locked)
        samples = self._samples.get(obj_id) or []
        if not samples:
            return None
        return float(np.median(samples[-7:]))

    def invalidate_pending(self, obj_id):
        """Drop an unfinished measurement. A saved inspection record is not stored here."""
        if obj_id is None:
            return
        self._samples.pop(obj_id, None)
        self._sample_radius.pop(obj_id, None)
        self._locked.pop(obj_id, None)
        self._locked_radius.pop(obj_id, None)
        self._pending_frames.pop(obj_id, None)
        self._bad_frames.pop(obj_id, None)
        self._failed.discard(obj_id)
        self._unstable.discard(obj_id)

    def _note(self, obj_id, raw_mm, status, frame_number=None):
        filtered = self.filtered(obj_id)
        raw_text = "none" if not diameter_is_valid(raw_mm) else f"{float(raw_mm):.1f}"
        filtered_text = "none" if filtered is None else f"{float(filtered):.1f}"
        signature = (obj_id, raw_text, filtered_text, status)
        if signature == self._log_signature:
            return
        self._log_signature = signature
        frame_text = "none" if frame_number is None else int(frame_number)
        print(
            f"[DIAMETER] frame={frame_text} id=ONION-{int(obj_id):03d} "
            f"raw={raw_text} filtered={filtered_text} status={status}"
        )

    def update(self, obj_id, raw_mm, cx, cy, radius_px=None, present_ids=None, focal_px=None, frame_number=None):
        """Lock a rolling median only after enough readings agree. A jump is not accepted."""
        if obj_id in self._failed or obj_id in self._unstable:
            self._note(obj_id, raw_mm, self.status(obj_id), frame_number)
            return None
        if obj_id in self._locked and diameter_is_valid(self._locked[obj_id]):
            self._positions[obj_id] = [float(cx), float(cy), 0]
            self._pending_frames.pop(obj_id, None)
            self._note(obj_id, raw_mm, "valid", frame_number)
            return self._locked[obj_id]

        self._positions[obj_id] = [float(cx), float(cy), 0]
        self._pending_frames[obj_id] = self._pending_frames.get(obj_id, 0) + 1
        if self._pending_frames[obj_id] >= int(ONION_VALIDATION["diameter_timeout_frames"]):
            self._locked.pop(obj_id, None)
            if self._samples.get(obj_id):
                self._unstable.add(obj_id)
            else:
                self._failed.add(obj_id)
            self._note(obj_id, raw_mm, self.status(obj_id), frame_number)
            self._samples.pop(obj_id, None)
            return None
        if not diameter_is_valid(raw_mm):
            self._bad_frames[obj_id] = self._bad_frames.get(obj_id, 0) + 1
            if self._bad_frames[obj_id] >= int(ONION_VALIDATION["diameter_attempt_limit"]):
                self._failed.add(obj_id)
            self._note(obj_id, raw_mm, self.status(obj_id), frame_number)
            return None
        self._bad_frames[obj_id] = 0

        samples = self._samples.setdefault(obj_id, [])
        raw_mm = float(raw_mm)
        tolerance = diameter_tolerance_mm(focal_px)
        if samples and abs(raw_mm - float(np.median(samples[-7:]))) > tolerance:
            self._note(obj_id, raw_mm, "pending", frame_number)
            return None

        samples.append(raw_mm)
        if len(samples) > self.MAX_SAMPLES:
            del samples[:-self.MAX_SAMPLES]

        if radius_px is not None and radius_px >= 8:
            self._sample_radius[obj_id] = float(radius_px)

        recent = samples[-7:]
        median_mm = float(np.median(recent))
        inliers = [value for value in recent if abs(value - median_mm) <= tolerance]
        needed = int(ONION_VALIDATION["diameter_min_consistent"])
        if len(inliers) >= needed:
            locked = self._lock(obj_id, inliers)
            if not diameter_is_valid(locked):
                self._failed.add(obj_id)
                self._locked.pop(obj_id, None)
                self._note(obj_id, raw_mm, "failed", frame_number)
                return None
            self._note(obj_id, raw_mm, "valid", frame_number)
            return locked
        if len(samples) >= self.MAX_SAMPLES:
            self._unstable.add(obj_id)
            self._locked.pop(obj_id, None)
            self._note(obj_id, raw_mm, "unstable", frame_number)
            return None
        self._note(obj_id, raw_mm, "pending", frame_number)
        return None

    def _lock(self, obj_id, samples):
        locked = round(float(np.median(samples)), 1)
        self._locked[obj_id] = locked
        if obj_id in self._sample_radius:
            self._locked_radius[obj_id] = self._sample_radius[obj_id]
        self._samples[obj_id] = []
        return locked

    def _nearby_lock(self, obj_id, cx, cy, present_ids):
        """Keep the frozen size if this is the same onion with a new track id."""
        best_id = None
        best_dist = 50.0
        for other_id, (px, py, misses) in self._positions.items():
            if other_id == obj_id or other_id not in self._locked or other_id in present_ids:
                continue
            if misses < 1 or misses > 8:
                continue
            dist = math.hypot(px - cx, py - cy)
            if dist < best_dist:
                best_dist = dist
                best_id = other_id
        return best_id

class PhysicalOnionTracker:
    """
    Optical centroid tracker with persistent object IDs.
    Ensures every physical onion is classified and counted EXACTLY ONCE.
    """
    def __init__(self):
        self.counted_ids = set()
        self.tracker = MultiObjectTracker() if MultiObjectTracker else None
        self.mock_intrin = MockIntrinsics()
        self.diameter_filter = OnionDiameterFilter()
        self.active_tracks = {}
        self._belt_mask = None
        self._belt_outline = None
        self._belt_hold = 0
        self._blob_tracks = []
        self._next_blob_id = 1

    @property
    def next_object_id(self):
        return self.tracker.next_obj_id if self.tracker else 1

    def process_frame_objects(self, frame_bgr: np.ndarray, depth_raw=None, intrinsics=None,
                              depth_scale=0.001) -> List[Tuple[int, Tuple[int, int, int, int], float, str, str]]:
        """
        Detects circular/spherical onion contours and assigns persistent IDs.
        Returns: list of (obj_id, bbox_xywh, diameter_mm, size_category, usda_grade)
        """
        h, w = frame_bgr.shape[:2]
        self._frame_number = int(getattr(self, "_frame_number", 0)) + 1
        if intrinsics is not None and hasattr(intrinsics, 'width'):
            self.mock_intrin = intrinsics
        else:
            self.mock_intrin.width = w
            self.mock_intrin.height = h
            self.mock_intrin.ppx = w / 2.0
            self.mock_intrin.ppy = h / 2.0
        
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        belt_mask = self._conveyor_belt_mask(gray)
        belt_blobs = self._belt_onion_blobs(frame_bgr, gray, belt_mask, depth_raw, depth_scale)
        detected_circles = [(cx, cy, radius * 2.0) for cx, cy, radius in belt_blobs]
        onions = self._track_belt_blobs(belt_blobs)
        present_ids = {obj.obj_id for obj in onions}
        self.diameter_filter.begin_frame()
        needs_measure = any(obj.obj_id not in self.diameter_filter._locked for obj in onions)
        belt_level = None
        if needs_measure:
            if belt_mask is not None:
                belt_pixels = gray[belt_mask > 0]
                belt_level = float(np.percentile(belt_pixels, 30)) if belt_pixels.size else None
            if belt_level is None:
                belt_level = float(np.percentile(gray, 20))
        depth_mm = None
        if needs_measure and depth_raw is not None:
            scale_mm = (depth_scale * 1000.0) if depth_scale else 1.0
            depth_mm = depth_raw.astype(np.float32) * scale_mm

        frame_h, frame_w = frame_bgr.shape[:2]
        current_frame_objects = []
        for obj in onions:
            bx, by, bw, bh = obj.bbox
            if bw <= 0 or bh <= 0:
                continue
            if bx >= frame_w or by >= frame_h or (bx + bw) <= 0 or (by + bh) <= 0:
                continue
            bx = max(0, int(bx))
            by = max(0, int(by))
            bw = min(int(bw), frame_w - bx)
            bh = min(int(bh), frame_h - by)
            if bw < 4 or bh < 4:
                continue
            obj.bbox = (bx, by, bw, bh)
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            pixel_radius = max(8.0, min(bw, bh) / 2.0)
            if obj.obj_id in self.diameter_filter._locked:
                raw_dia_mm = None
            else:
                # The camera is fixed 240 mm above the conveyor.  At this close
                # mounting distance the raw D455 depth at the onion surface can
                # be unstable, so use the known camera height + camera intrinsics
                # for the primary diameter estimate.  The spherical correction
                # accounts for the onion centre being closer to the camera than
                # the belt plane.
                raw_dia_mm = self._fixed_height_diameter_mm(pixel_radius, intrinsics)
                if not diameter_is_valid(raw_dia_mm):
                    raw_dia_mm = self._fast_diameter_mm(cx, cy, pixel_radius, depth_mm, intrinsics)
                if not diameter_is_valid(raw_dia_mm):
                    raw_dia_mm = None
            focal_px = None
            if intrinsics is not None:
                fx = float(getattr(intrinsics, "fx", 0.0) or 0.0)
                fy = float(getattr(intrinsics, "fy", 0.0) or 0.0)
                if fx > 1.0 and fy > 1.0:
                    focal_px = (fx + fy) / 2.0
                elif fx > 1.0 or fy > 1.0:
                    focal_px = fx if fx > 1.0 else fy
            dia_mm = self.diameter_filter.update(
                obj.obj_id, raw_dia_mm, cx, cy, pixel_radius, present_ids,
                focal_px=focal_px, frame_number=self._frame_number,
            )
            if dia_mm is None:
                current_frame_objects.append((obj.obj_id, obj.bbox, None, "", ""))
                continue

            # Assign size category and USDA grade from this onion's locked diameter.
            cat = size_category_for_diameter(dia_mm)
            if cat is None:
                current_frame_objects.append((obj.obj_id, obj.bbox, None, "", ""))
                continue

            if dia_mm < 45.0:
                usda = "SMALL"
            elif dia_mm >= 70.0:
                usda = "LARGE"
            else:
                usda = "MEDIUM"

            current_frame_objects.append((obj.obj_id, obj.bbox, dia_mm, cat, usda))

        self.diameter_filter.end_frame()
        self._publish_detection_log(frame_bgr.shape, current_frame_objects)
        return current_frame_objects

    def _publish_detection_log(self, frame_shape, current_frame_objects):
        debug = getattr(self, "_candidate_debug", {}) or {}
        score = debug.get("score")
        if current_frame_objects:
            obj_id, bbox, dia_mm, _cat, _usda = current_frame_objects[0]
            status = self.diameter_filter.status(obj_id)
            if status == "valid" and diameter_is_valid(dia_mm):
                state = "diameter_ready"
                reason = "diameter validated"
            elif status == "unstable":
                state = "unstable"
                reason = "diameter did not stabilize"
            elif status == "failed":
                state = "diameter_failed"
                reason = "diameter measurement failed"
            else:
                state = "measuring"
                reason = "waiting for a stable diameter"
            track_id = f"ONION-{obj_id:03d}"
            bbox_text = tuple(int(v) for v in bbox)
        else:
            state = "waiting"
            track_id = "none"
            bbox_text = "none"
            reason = debug.get("reason") or "no onion on the conveyor"
        report = {
            "frame": self._frame_number,
            "state": state,
            "track_id": track_id,
            "score": score,
            "bbox": bbox_text,
            "reason": reason,
        }
        self._detection_report = report
        signature = (state, track_id, reason)
        if signature != getattr(self, "_log_signature", None):
            self._log_signature = signature
            print(format_detection_log(
                report["frame"], frame_shape, state, track_id, score, bbox_text, reason
            ))

    @staticmethod
    def _measure_onion_diameter_3d(obj, depth_raw, intrinsics, depth_scale, detected_circles=None, depth_mm=None):
        """Measure one onion from its real depth edge and the camera focal length."""
        if depth_raw is None or intrinsics is None:
            return None

        h, w = depth_raw.shape[:2]
        bx, by, bw, bh = obj.bbox
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        est_radius = max(8.0, min(bw, bh) / 2.0)
        if detected_circles:
            best_diam = 0.0
            for circ_x, circ_y, eq_diam in detected_circles:
                dist = math.hypot(circ_x - cx, circ_y - cy)
                if dist < max(60.0, est_radius) and eq_diam > best_diam:
                    best_diam = float(eq_diam)
            if best_diam > 0:
                est_radius = max(est_radius, best_diam / 2.0)

        if depth_mm is None:
            scale_mm = (depth_scale * 1000.0) if depth_scale else 1.0
            depth_mm = depth_raw.astype(np.float32) * scale_mm
        top_depth = PhysicalOnionTracker._crown_depth_mm(depth_mm, cx, cy, est_radius)
        if top_depth is None:
            return None
        support_depth = PhysicalOnionTracker._belt_depth_mm(depth_mm, cx, cy, est_radius, top_depth)
        pixel_radius = None
        if support_depth is not None and support_depth >= top_depth + 8.0:
            pixel_radius = PhysicalOnionTracker._radius_at_depth_step(
                depth_mm, cx, cy, est_radius, top_depth, support_depth
            )
        if pixel_radius is None:
            pixel_radius = est_radius

        fx = float(getattr(intrinsics, "fx", 0.0) or 0.0)
        fy = float(getattr(intrinsics, "fy", 0.0) or 0.0)
        if fx > 1.0 and fy > 1.0:
            focal = (fx + fy) / 2.0
        else:
            focal = fx if fx > 1.0 else fy
        if focal <= 1.0:
            return None

        # Same projection as the calibrated RealSense measurement: pixels * depth / focal length.
        diameter_mm = (2.0 * pixel_radius * top_depth) / focal
        return float(diameter_mm) if diameter_is_valid(diameter_mm) else None

    @staticmethod
    def _fixed_height_diameter_mm(pixel_radius, intrinsics):
        """Estimate onion diameter from the fixed 240 mm camera height.

        The onion is approximated as a sphere resting on the belt.  This avoids
        the large size error caused by unreliable close-range depth values while
        still using the RealSense colour-camera focal length for metric scaling.
        """
        if intrinsics is None or pixel_radius is None or pixel_radius <= 0:
            return None

        fx = float(getattr(intrinsics, "fx", 0.0) or 0.0)
        fy = float(getattr(intrinsics, "fy", 0.0) or 0.0)
        if fx > 1.0 and fy > 1.0:
            focal = (fx + fy) / 2.0
        else:
            focal = fx if fx > 1.0 else fy
        if focal <= 1.0:
            return None

        # q is the projected image radius divided by focal length.  For a
        # spherical onion resting on a belt H mm below the camera, the apparent
        # radius is perspective-corrected to recover the physical diameter.
        q = float(pixel_radius) / focal
        if q <= 0.0:
            return None
        diameter_mm = (2.0 * CONVEYOR_DEPTH_MM * q) / (math.sqrt(1.0 + q * q) + q)
        if not diameter_is_valid(diameter_mm):
            return None
        return float(diameter_mm)

    def _fast_diameter_mm(self, cx, cy, pixel_radius, depth_mm, intrinsics):
        """Millimetres from the onion outline already found on the belt."""
        if depth_mm is None or intrinsics is None:
            return None
        top_depth = self._crown_depth_mm(depth_mm, cx, cy, pixel_radius)
        if top_depth is None:
            return None
        fx = float(getattr(intrinsics, "fx", 0.0) or 0.0)
        fy = float(getattr(intrinsics, "fy", 0.0) or 0.0)
        focal = (fx + fy) / 2.0 if fx > 1.0 and fy > 1.0 else (fx if fx > 1.0 else fy)
        if focal <= 1.0:
            return None
        diameter_mm = (2.0 * pixel_radius * top_depth) / focal
        return float(diameter_mm) if diameter_is_valid(diameter_mm) else None

    def _conveyor_belt_mask(self, gray):
        """The dark belt between the wooden rails. The floor and the frame stay outside it."""
        self._belt_hold += 1
        if self._belt_mask is not None and self._belt_hold % 8 != 1:
            return self._belt_mask
        h, w = gray.shape[:2]
        thresh = float(np.percentile(gray, 34))
        thresh = min(max(thresh, 35.0), 115.0)
        dark = (gray <= thresh).astype(np.uint8) * 255
        open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, open_k)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 0.12 * h * w
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        if not contours:
            return self._belt_mask

        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        if cv2.countNonZero(inner) > 0.08 * h * w:
            mask = inner
            inner_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if inner_contours:
                hull = max(inner_contours, key=cv2.contourArea)
        self._belt_mask = mask
        self._belt_outline = hull
        return mask

    @staticmethod
    def _point_on_belt(mask, x, y):
        if mask is None:
            return True
        ix, iy = int(round(x)), int(round(y))
        h, w = mask.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return False
        return mask[iy, ix] > 0

    def _track_belt_blobs(self, blobs):
        """Give each onion on the belt a stable id without the slow full-frame detector."""
        used = set()
        onions = []
        for cx, cy, radius in blobs:
            best_i = None
            best_dist = 80.0
            for index, track in enumerate(self._blob_tracks):
                if index in used or track.get("misses", 0) > 2:
                    continue
                dist = math.hypot(track["cx"] - cx, track["cy"] - cy)
                if dist < best_dist:
                    best_dist = dist
                    best_i = index
            if best_i is None:
                track = {"id": self._next_blob_id, "cx": cx, "cy": cy, "misses": 0}
                self._next_blob_id += 1
                self._blob_tracks.append(track)
                best_i = len(self._blob_tracks) - 1
            else:
                track = self._blob_tracks[best_i]
                track["cx"] = cx
                track["cy"] = cy
                track["misses"] = 0
            used.add(best_i)
            draw_r = max(8, int(round(radius)))
            onion = type("BeltOnion", (), {})()
            onion.obj_id = track["id"]
            onion.shape_type = "ONION"
            onion.bbox = (int(round(cx)) - draw_r, int(round(cy)) - draw_r, draw_r * 2, draw_r * 2)
            onions.append(onion)

        kept = []
        for index, track in enumerate(self._blob_tracks):
            if index not in used:
                track["misses"] += 1
            if track["misses"] <= 6:
                kept.append(track)
            else:
                self.diameter_filter.clear(track["id"])
                self.active_tracks.pop(track["id"], None)
        self._blob_tracks = kept
        return onions

    @staticmethod
    def _region_is_conveyor(mask):
        """A conveyor region is a wide band. A narrow dark object is not the belt."""
        if mask is None or cv2.countNonZero(mask) < 500:
            return False
        ys, xs = np.where(mask > 0)
        if xs.size < 500:
            return False
        height, width = mask.shape[:2]
        span_x = int(xs.max() - xs.min())
        span_y = int(ys.max() - ys.min())
        if span_x < int(0.45 * width):
            return False
        if span_x < span_y and span_x < int(0.70 * width):
            return False
        return True

    def _inspection_region(self, belt_mask, height, width):
        """Use the dark belt when it is conveyor-shaped. Otherwise use the center ROI."""
        if self._region_is_conveyor(belt_mask):
            return belt_mask, False
        cfg = ONION_VALIDATION
        roi = np.zeros((height, width), dtype=np.uint8)
        y0, y1 = int(height * cfg["roi_top"]), int(height * cfg["roi_bottom"])
        x0, x1 = int(width * cfg["roi_left"]), int(width * cfg["roi_right"])
        if y1 <= y0 or x1 <= x0:
            return None, True
        roi[y0:y1, x0:x1] = 255
        return roi, True

    @staticmethod
    def _interior_color_fraction(hue, sat, val, cx, cy, radius):
        cfg = ONION_VALIDATION
        frame_h, frame_w = hue.shape[:2]
        reach = max(2, int(round(radius)))
        x0, x1 = max(0, int(cx) - reach), min(frame_w, int(cx) + reach + 1)
        y0, y1 = max(0, int(cy) - reach), min(frame_h, int(cy) + reach + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius * radius)
        inside = int(np.count_nonzero(disk))
        if inside <= 0:
            return 0.0
        hh = hue[y0:y1, x0:x1]
        ss = sat[y0:y1, x0:x1]
        vv = val[y0:y1, x0:x1]
        match = (
            ((hh <= cfg["hue_low"]) | (hh >= cfg["hue_high"]))
            & (ss >= cfg["min_saturation"])
            & (vv >= cfg["min_value"])
            & (vv <= cfg["max_value"])
            & disk
        )
        return float(np.count_nonzero(match)) / float(inside)

    @staticmethod
    def crop_matches_onion(image_rgb):
        """True when the center of this crop still looks like onion skin."""
        if image_rgb is None or getattr(image_rgb, "size", 0) == 0 or getattr(image_rgb, "ndim", 0) != 3:
            return False
        height, width = image_rgb.shape[:2]
        if height < 16 or width < 16 or image_rgb.shape[2] < 3:
            return False
        y0, y1 = int(height * 0.20), int(height * 0.80)
        x0, x1 = int(width * 0.20), int(width * 0.80)
        if y1 <= y0 or x1 <= x0:
            return False
        hsv = cv2.cvtColor(image_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        cfg = ONION_VALIDATION
        match = (
            ((hue <= cfg["hue_low"]) | (hue >= cfg["hue_high"]))
            & (sat >= cfg["min_saturation"])
            & (val >= cfg["min_value"])
            & (val <= cfg["max_value"])
        )
        return float(np.mean(match)) >= cfg["min_interior_fraction"]

    def _belt_onion_blobs(self, frame_bgr, gray, belt_mask, depth_raw=None, depth_scale=0.001):
        """One current-frame onion inside the inspection region, or none."""
        cfg = ONION_VALIDATION
        h, w = gray.shape[:2]
        belt_mask, require_raised = self._inspection_region(belt_mask, h, w)
        if belt_mask is None or cv2.countNonZero(belt_mask) < 500:
            self._candidate_debug = {"reason": "inspection region is empty", "score": None}
            return []
        inner = cv2.erode(belt_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        if cv2.countNonZero(inner) < 500:
            inner = belt_mask
        strongest_rejection = [0.0, "no onion-colored contour"]

        def reject(area, reason):
            if area >= strongest_rejection[0]:
                strongest_rejection[0] = area
                strongest_rejection[1] = reason
        depth_mm = None
        if depth_raw is not None:
            scale_mm = (float(depth_scale) * 1000.0) if depth_scale else 1.0
            depth_mm = np.asarray(depth_raw, dtype=np.float32) * scale_mm
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        color = (
            ((hue <= cfg["hue_low"]) | (hue >= cfg["hue_high"]))
            & (sat >= cfg["min_saturation"])
            & (val >= cfg["min_value"])
            & (val <= cfg["max_value"])
            & (inner > 0)
        ).astype(np.uint8) * 255
        onion = cv2.morphologyEx(color, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        onion = cv2.morphologyEx(onion, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(onion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        margin = int(cfg["edge_margin_px"])
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < cfg["min_area"] or area > cfg["max_area"]:
                reject(area, "contour size is outside the onion range")
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                reject(area, "contour has no perimeter")
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < cfg["min_circularity"]:
                reject(area, f"shape is not round enough ({circularity:.2f})")
                continue
            (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(cnt)
            short_side = max(1.0, min(rect_w, rect_h))
            aspect = max(rect_w, rect_h) / short_side
            if aspect > cfg["max_aspect"]:
                reject(area, f"shape is too elongated ({aspect:.2f})")
                continue
            (cx, cy), enc_r = cv2.minEnclosingCircle(cnt)
            eq_radius = math.sqrt(max(area, 1.0) / math.pi)
            short_radius = short_side / 2.0
            radius = min(float(enc_r), 0.70 * eq_radius + 0.30 * short_radius)
            if radius < cfg["min_radius_px"] or radius > cfg["max_radius_px"]:
                reject(area, f"radius {radius:.1f}px is outside the onion range")
                continue
            fill = area / max(1.0, math.pi * radius * radius)
            if fill < cfg["min_fill"]:
                reject(area, f"outline is too hollow ({fill:.2f})")
                continue
            if not (margin <= cx < w - margin and margin <= cy < h - margin):
                reject(area, "onion center is outside the frame")
                continue
            hidden = max(
                0.0,
                margin - (cx - radius),
                margin - (cy - radius),
                (cx + radius) - (w - margin),
                (cy + radius) - (h - margin),
            )
            if hidden > 0.45 * radius:
                reject(area, "too little of the onion is inside the frame")
                continue
            if not self._point_on_belt(inner, cx, cy):
                reject(area, "center is outside the conveyor")
                continue
            check_r = max(4.0, radius * 0.72)
            belt_points = (
                (cx + check_r, cy), (cx - check_r, cy),
                (cx, cy + check_r), (cx, cy - check_r),
            )
            points_on_belt = sum(1 for px, py in belt_points if self._point_on_belt(inner, px, py))
            if points_on_belt < 3:
                reject(area, "onion is not resting on the conveyor")
                continue
            interior = self._interior_color_fraction(hue, sat, val, cx, cy, radius * 0.75)
            if interior < cfg["min_interior_fraction"]:
                reject(area, f"surface color is not onion skin ({interior:.2f})")
                continue
            raised = None
            if depth_mm is not None:
                raised = self._raised_above_belt(depth_mm, cx, cy, radius, belt_mask)
            if raised is False:
                reject(area, "object is flat on the belt")
                continue
            if raised is None and require_raised:
                reject(area, "depth did not confirm an onion above the belt")
                continue
            circ_score = min(1.0, circularity / 0.85)
            fill_score = min(1.0, fill / 0.85)
            color_score = min(1.0, interior / 0.85)
            depth_score = 1.0 if raised is True else 0.70
            score = (0.30 * circ_score) + (0.20 * fill_score) + (0.30 * color_score) + (0.20 * depth_score)
            if score < cfg["min_score"]:
                reject(area, f"detection score {score:.2f} is below {cfg['min_score']:.2f}")
                continue
            blobs.append((int(round(cx)), int(round(cy)), float(radius), float(area), float(score)))
        if not blobs:
            self._candidate_debug = {"reason": strongest_rejection[1], "score": None}
            return []
        blobs.sort(key=lambda item: (item[4], item[3]), reverse=True)
        best = blobs[0]
        self._candidate_debug = {
            "reason": "onion visible",
            "score": best[4],
            "center": (best[0], best[1], best[2]),
        }
        return [(best[0], best[1], best[2])]

    @staticmethod
    def _onion_edge_radius(gray, cx, cy, seed_radius, belt_level):
        """Grow the small highlight circle out to the dark belt around the onion."""
        h, w = gray.shape[:2]
        edge_level = belt_level + 12.0
        radii = []
        start = max(4, int(seed_radius))
        limit = min(start + 50, 180)
        for angle in range(0, 360, 30):
            cos_a = math.cos(math.radians(angle))
            sin_a = math.sin(math.radians(angle))
            streak = 0
            found = None
            for step in range(start, limit):
                px = int(round(cx + step * cos_a))
                py = int(round(cy + step * sin_a))
                if px < 0 or py < 0 or px >= w or py >= h:
                    break
                if float(gray[py, px]) <= edge_level:
                    streak += 1
                    if streak >= 2:
                        found = float(step)
                        break
                else:
                    streak = 0
            if found is not None:
                radii.append(found)
        if len(radii) < 4:
            return float(seed_radius)
        return float(np.median(radii))

    @staticmethod
    def _raised_above_belt(depth_mm, cx, cy, radius, belt_mask):
        """An onion is closer to the camera than the belt around it."""
        h, w = depth_mm.shape[:2]
        r = max(4, int(round(radius)))
        x0, x1 = max(0, int(cx) - r * 2), min(w, int(cx) + r * 2 + 1)
        y0, y1 = max(0, int(cy) - r * 2), min(h, int(cy) + r * 2 + 1)
        if x1 <= x0 or y1 <= y0:
            return False
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        patch = depth_mm[y0:y1, x0:x1]
        on_belt = belt_mask[y0:y1, x0:x1] > 0
        crown = patch[(dist2 <= (r * 0.45) ** 2) & (patch > 100.0) & (patch < 3000.0)]
        ring = patch[(dist2 >= (r * 1.25) ** 2) & (dist2 <= (r * 2.0) ** 2) & on_belt & (patch > 100.0) & (patch < 3000.0)]
        if crown.size < 8 or ring.size < 12:
            return None
        gap = float(np.median(ring)) - float(np.median(crown))
        raise_mm = float(ONION_VALIDATION["depth_raise_mm"])
        if gap >= raise_mm:
            return True
        # A confident flat reading rejects a stain or shadow. A noisy close-range
        # reading is not evidence that the visible onion is absent.
        if gap <= 2.0 and crown.size >= 20 and ring.size >= 20:
            return False
        return None

    @staticmethod
    def _full_onion_radius(frame_bgr, cx, cy, gray=None, belt_mask=None, belt_level=None):
        """Stop at the belt just outside the onion, not on the onion skin or the wooden rail."""
        if gray is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        ix, iy = int(round(cx)), int(round(cy))
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        if belt_level is None:
            if belt_mask is not None and np.count_nonzero(belt_mask) > 0:
                belt_level = float(np.percentile(gray[belt_mask > 0], 30))
            else:
                belt_level = float(np.percentile(gray, 20))
        edge_level = belt_level + 14.0
        radii = []
        for angle in np.linspace(0, 2 * math.pi, 36, endpoint=False):
            cos_a = math.cos(float(angle))
            sin_a = math.sin(float(angle))
            streak = 0
            found = None
            for step in range(8, 180):
                px = int(round(ix + step * cos_a))
                py = int(round(iy + step * sin_a))
                if px < 0 or py < 0 or px >= w or py >= h:
                    break
                if belt_mask is not None and belt_mask[py, px] == 0:
                    found = float(max(8, step - 1))
                    break
                if float(gray[py, px]) <= edge_level:
                    streak += 1
                    if streak >= 2:
                        found = float(step)
                        break
                else:
                    streak = 0
            if found is not None and found >= 12:
                radii.append(found)
        if len(radii) < 8:
            return None
        median_radius = float(np.median(radii))
        inliers = [radius for radius in radii if abs(radius - median_radius) <= max(4.0, 0.2 * median_radius)]
        if len(inliers) < 8:
            return None
        return float(np.median(inliers))

    @staticmethod
    def _depths_in_disk(depth_mm, cx, cy, radius):
        h, w = depth_mm.shape[:2]
        r = max(1, int(math.ceil(radius)))
        x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
        if x1 <= x0 or y1 <= y0:
            return np.empty(0, dtype=np.float32)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        patch = depth_mm[y0:y1, x0:x1]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (radius * radius)
        values = patch[mask]
        return values[values > 100.0]

    @staticmethod
    def _crown_depth_mm(depth_mm, cx, cy, est_radius):
        """Depth of the onion top, ignoring the belt and depth holes."""
        values = PhysicalOnionTracker._depths_in_disk(depth_mm, cx, cy, max(4.0, est_radius * 0.35))
        values = values[values < 2500.0]
        if values.size < 8:
            return None
        median_depth = float(np.median(values))
        band = values[np.abs(values - median_depth) <= 30.0]
        if band.size < 8:
            band = values
        return float(np.median(band))

    @staticmethod
    def _belt_depth_mm(depth_mm, cx, cy, est_radius, top_depth):
        """Depth of the belt around the onion, not the onion surface itself."""
        h, w = depth_mm.shape[:2]
        inner_r = est_radius * 1.25
        outer_r = est_radius * 2.1
        r = max(1, int(math.ceil(outer_r)))
        x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)
        if x1 > x0 and y1 > y0:
            yy, xx = np.ogrid[y0:y1, x0:x1]
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            mask = (dist2 >= inner_r * inner_r) & (dist2 <= outer_r * outer_r)
            ring = depth_mm[y0:y1, x0:x1][mask]
            ring = ring[(ring > top_depth + 8.0) & (ring < 2500.0)]
            if ring.size >= 12:
                return float(np.percentile(ring, 70))
        if CONVEYOR_DEPTH_MM > top_depth + 8.0:
            return float(CONVEYOR_DEPTH_MM)
        return None

    @staticmethod
    def _radius_at_depth_step(depth_mm, cx, cy, est_radius, top_depth, support_depth):
        """Pixel radius where the onion surface falls away to the belt."""
        h, w = depth_mm.shape[:2]
        threshold = top_depth + (support_depth - top_depth) * 0.45
        start = max(2, int(est_radius * 0.15))
        limit = min(int(max(cx, cy, w - cx, h - cy)), int(est_radius * 2.0) + 6)
        if limit <= start + 2:
            return None

        radii = []
        for angle in np.linspace(0, 2 * math.pi, 36, endpoint=False):
            cos_a = math.cos(float(angle))
            sin_a = math.sin(float(angle))
            streak = 0
            found = None
            for step in range(start, limit):
                px = int(round(cx + step * cos_a))
                py = int(round(cy + step * sin_a))
                if px < 0 or py < 0 or px >= w or py >= h:
                    break
                depth = float(depth_mm[py, px])
                if depth <= 100.0:
                    continue
                if depth >= threshold:
                    streak += 1
                    if streak >= 2:
                        found = step - 1
                        break
                else:
                    streak = 0
            if found is not None and found < limit - 1:
                radii.append(float(found))

        if len(radii) < 10:
            return None
        median_radius = float(np.median(radii))
        inliers = [radius for radius in radii if abs(radius - median_radius) <= max(2.0, 0.12 * median_radius)]
        if len(inliers) < 8:
            return None
        return float(np.median(inliers))

    def reset(self):
        """Resets tracking history for new inspection slot."""
        self.counted_ids.clear()
        self.diameter_filter.clear()
        self.active_tracks.clear()
        self._belt_mask = None
        self._belt_outline = None
        self._belt_hold = 0
        self._blob_tracks = []
        self._next_blob_id = 1
        self._log_signature = None
        self._candidate_debug = {}
        self._detection_report = {}
        if self.tracker:
            self.tracker.active_objects.clear()
            self.tracker.total_objects_counted = 0
            self.tracker.next_obj_id = 1

    def process_continuous(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray, depth_raw=None, intrinsics=None, depth_scale=0.001, hold_ids=None):
        """
        Runs the continuous temporal tracker. 
        Returns: 
          current_frame_objects: for UI rendering
          finalized_onions: list of onions that have left the screen and are ready for inference
        """
        current_frame_objects = self.process_frame_objects(frame_bgr, depth_raw, intrinsics, depth_scale)
        
        current_ids = set()
        h, w = frame_rgb.shape[:2]
        img_cx, img_cy = w // 2, h // 2
        
        for obj_id, bbox, dia_mm, cat, usda in current_frame_objects:
            current_ids.add(obj_id)
            if obj_id not in self.active_tracks:
                self.active_tracks[obj_id] = TrackedOnion(obj_id)
            
            track = self.active_tracks[obj_id]
            track.misses = 0
            track.frames_tracked += 1
            if diameter_is_valid(dia_mm):
                track.diameters.append(float(dia_mm))
                
            bx, by, bw, bh = bbox
            cx, cy = bx + bw//2, by + bh//2
            dist = (cx - img_cx)**2 + (cy - img_cy)**2
            previous_area = 0
            if track.best_bbox is not None:
                previous_area = track.best_bbox[2] * track.best_bbox[3]
            
            if dist < track.min_dist_to_center:
                track.min_dist_to_center = dist
                track.best_bbox = bbox
                # Keep the best actual onion crop for the second (analysis) box.
                # The previous code stored the complete camera frame here.
                pad = max(12, int(0.15 * max(bw, bh)))
                x0 = max(0, bx - pad)
                y0 = max(0, by - pad)
                x1 = min(w, bx + bw + pad)
                y1 = min(h, by + bh + pad)
                crop = frame_rgb[y0:y1, x0:x1]
                if crop.size > 0:
                    track.best_crop = crop.copy()
                
        # A missed frame is not a new onion and is not drawn as a current detection.
        # An onion in disease inference or on the conveyor stays tracked until that work finishes.
        drop_after = int(ONION_VALIDATION["drop_after_misses"])
        held = set(hold_ids or ())
        finished_ids = []
        for obj_id, track in self.active_tracks.items():
            if obj_id not in current_ids:
                track.misses += 1
                if track.misses >= drop_after and obj_id not in held:
                    finished_ids.append(obj_id)
        for fid in finished_ids:
            self.active_tracks.pop(fid, None)
            self.diameter_filter.clear(fid)

        return current_frame_objects, []


# ==============================================================================
# WORKER THREADS (NON-BLOCKING ASYNCHRONOUS PROCESSING)
# ==============================================================================

class CameraWorker(QThread):
    """QThread worker for RealSense camera streaming."""
    # (bgr, rgb, depth_raw, intrinsics, timestamp)
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, object, object, float) 
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_index=CAMERA_INDEX):
        super().__init__()
        self.camera_index = camera_index
        self.is_running = False
        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001
        self.ui_ready = threading.Event()
        self.ui_ready.set()

    def run(self):
        self.is_running = True
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        except Exception as e:
            response = camera_offline_response(e)
            print(f"[CameraWorker] {response['message']}")
            self.is_running = response["running"]
            self.error_occurred.emit(response["message"])
            return

        read_failures = 0
        while self.is_running:
            handed_off = False
            try:
                frames = self.pipeline.wait_for_frames(5000)
                for _ in range(5):
                    try:
                        newer = self.pipeline.poll_for_frames()
                    except Exception:
                        break
                    if newer is None:
                        break
                    try:
                        if not newer.get_color_frame():
                            break
                    except Exception:
                        break
                    frames = newer
                if not self.ui_ready.is_set():
                    continue

                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                    
                # Depth is aligned to the color image, so the colour focal length is the correct scale.
                intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
                depth_raw = np.asanyarray(depth_frame.get_data()).copy()
                frame_bgr = np.asanyarray(color_frame.get_data()).copy()
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                self.ui_ready.clear()
                handed_off = True
                self.frame_ready.emit(frame_bgr, frame_rgb, depth_raw, intrinsics, time.time())
                read_failures = 0
            except Exception as e:
                if handed_off:
                    self.ui_ready.set()
                read_failures += 1
                if read_failures >= 3:
                    response = camera_offline_response(e)
                    print(f"[CameraWorker] {response['message']}")
                    self.is_running = False
                    self.error_occurred.emit(response["message"])
                    break
                time.sleep(0.03)

        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

    def _run_simulator(self):
        """Do not invent onions when the RealSense camera is unavailable."""
        response = camera_offline_response("RealSense camera is not connected")
        self.is_running = response["running"]
        self.error_occurred.emit(response["message"])

    def stop(self):
        self.is_running = False
        self.wait()


class InferenceWorker(QThread):
    """QThread worker for deep learning inference & 3D metric sizing."""
    inspection_completed = pyqtSignal(object) # InspectionResult
    error_occurred = pyqtSignal(str)

    def __init__(self, model_manager: ModelManager, gradcam_engine: GradCAMEngine = None):
        super().__init__()
        self.model_manager = model_manager
        self.gradcam_engine = gradcam_engine
        self.pending_task = None
        self._is_running = True

    def process_capture(self, capture_num: int, onion_id: str, frame_rgb: np.ndarray,
                        display_img: np.ndarray,
                        model_name: str, batch_no: str, size_mm: float,
                        size_cat: str, usda_grade: str):
        if not diameter_is_valid(size_mm):
            return
        self.pending_task = {
            "capture_num": capture_num,
            "onion_id": onion_id,
            "frame_rgb": frame_rgb,
            "display_img": display_img,
            "model_name": model_name,
            "batch_no": batch_no,
            "size_mm": size_mm,
            "size_cat": size_cat,
            "usda_grade": usda_grade
        }
        self.start()

    def _classify_task(self, task):
        """Call the disease model only after this task's diameter is valid."""
        return classify_onion_disease(
            self.model_manager,
            task.get("model_name"),
            task.get("frame_rgb"),
            task.get("size_mm"),
        )

    def run(self):
        if not self.pending_task:
            return

        task = self.pending_task
        self.pending_task = None

        try:
            classified = self._classify_task(task)
            if classified is None:
                return
            pred_class, conf, prob_dict = classified

            ts_str = time.strftime("%H:%M:%S")

            result = InspectionResult(
                capture_number=task["capture_num"],
                onion_id=task["onion_id"],
                timestamp=ts_str,
                image=task["display_img"].copy(),
                inference_image=task["frame_rgb"].copy(),
                predicted_class=pred_class,
                confidence=conf,
                class_probabilities=prob_dict,
                model_name=task["model_name"],
                size_mm=task["size_mm"],
                size_category=task["size_cat"],
                usda_grade=task["usda_grade"],
                batch_no=task["batch_no"]
            )

            self.inspection_completed.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))


class XAIWorker(QThread):
    """QThread worker for background Grad-CAM explanation generation."""
    xai_completed = pyqtSignal(int, str, str, object, int)  # capture, onion id, class, overlay, token
    error_occurred = pyqtSignal(str)

    def __init__(self, gradcam_engine: GradCAMEngine):
        super().__init__()
        self.gradcam_engine = gradcam_engine
        self._request_id = 0
        self._pending = None

    def generate_xai(self, capture_num: int, onion_id: str, img_rgb: np.ndarray, model_name: str, predicted_class: str):
        if img_rgb is None or not onion_id or gradcam_class_index(predicted_class) is None:
            return
        self._request_id += 1
        self._pending = (
            self._request_id,
            int(capture_num),
            onion_id,
            img_rgb.copy(),
            model_name,
            predicted_class,
        )
        if not self.isRunning():
            self.start()

    def run(self):
        while self._pending is not None:
            request_id, capture_num, onion_id, image, model_name, predicted_class = self._pending
            try:
                overlay = self.gradcam_engine.generate_gradcam(image, model_name, predicted_class)
            except Exception as exc:
                self.error_occurred.emit(str(exc))
                overlay = None
            if request_id != self._request_id:
                continue
            self._pending = None
            self.xai_completed.emit(capture_num, onion_id, predicted_class, overlay, request_id)


# ==============================================================================
# MAIN DASHBOARD WINDOW (PYQT5 GUI ARCHITECTURE)
# ==============================================================================

class FlowLayout(QLayout):
    """Wraps controls onto extra rows so a narrow window does not clip them."""

    def __init__(self, parent=None, margin=0, h_spacing=8, v_spacing=8):
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self._h_space = h_spacing
        self._v_space = v_spacing
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _do_layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0
        for item in self._items:
            widget = item.widget()
            if widget is not None and not widget.isVisible():
                continue
            hint = item.sizeHint()
            space_x = self._h_space
            space_y = self._v_space
            next_x = x + hint.width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + hint.width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


class MainWindow(QMainWindow):
    physical_route_finished = pyqtSignal(str, bool)
    """
    Full-Screen Industrial Dashboard Window for Onion AI Inspection System.
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ONION AI INSPECTION SYSTEM")
        self._apply_window_icon()
        
        desktop = QApplication.desktop()
        screen_rect = desktop.availableGeometry()
        self.screen_w = screen_rect.width()
        self.screen_h = screen_rect.height()
        
        base_w, base_h = 1280.0, 800.0
        self.scale_f = min(self.screen_w / base_w, self.screen_h / base_h)
        if self.scale_f > 1.0:
            self.scale_f = 1.0
            
        self.is_small_screen = self.scale_f < 0.95
        self._style_bucket = None
        self._applying_layout = False
        # The window may be resized below the design size. Content scrolls instead of clipping.
        self.setMinimumSize(640, 420)

        # Initialize Managers and Engines
        self.model_manager = ModelManager(MODEL_PATHS)
        self.gradcam_engine = GradCAMEngine(self.model_manager)
        self.onion_tracker = PhysicalOnionTracker()
        self.spectrum_engine = SpectrumEngine(SPECTRUM_WEIGHTS)
        self.reflectance_references = ReflectanceReferences(REFLECTANCE_EXCEL_DIR)

        # State Variables
        self.is_operation_active = False
        self.batch_number = ""
        self.batch_started_at = None
        self.slot_target_quantity = 0
        self.unique_onions_processed = 0
        self._pending_onion_captures = []
        self._queued_onion_ids = set()
        self._sorted_servo_ids = set()
        self._counted_inspection_ids = set()
        self._dispatch_started = set()
        self._sorting_refused = set()
        self._submitted_labels = set()
        self._submitted_diameter = {}
        self._cancelled_inspections = set()
        self._batch_flow = BatchFlow(0)
        self._visible_onion_labels = set()
        self._no_onion_shown = False
        self._live_track_signature = None
        self._inspection_target_id = None
        self._browsing_history = False
        self._disease_shown_for = None
        self._review_message = ""
        self._live_frame_token = 0
        self._classification_state = {}

        self.inspection_history: List[InspectionResult] = []
        self.current_history_index = -1
        self.captured_count = 0

        # Visualization Zoom Factors
        self.left_zoom_factor = 1.0
        self.center_zoom_factor = 1.0
        self._last_live_frame = None
        self._last_capture_frame = None

        # Disease Counters (Pass = Normal)
        self.normal_count = 0
        self.rotten_count = 0
        self.sprouted_count = 0
        self.moldy_count = 0

        # Size Category Counters
        self.size_cat1 = 0
        self.size_cat2 = 0
        self.size_cat3 = 0
        self.size_cat4 = 0

        # Workers & Timers
        self.camera_worker = None
        self.inference_worker = InferenceWorker(self.model_manager, self.gradcam_engine)
        self.inference_worker.inspection_completed.connect(self._on_inspection_completed)
        self.inference_worker.error_occurred.connect(self._on_inference_error)
        self.physical_route_finished.connect(self._on_physical_route_finished)

        self.xai_worker = XAIWorker(self.gradcam_engine)
        self.xai_worker.xai_completed.connect(self._on_xai_completed)
        self.xai_worker.error_occurred.connect(self._on_error_occurred)

        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(CAPTURE_INTERVAL_MS)
        self.capture_timer.timeout.connect(self._on_capture_timer_tick)

        self.latest_live_frame_rgb = None
        self.latest_live_frame_bgr = None
        self.latest_depth_raw = None
        self.latest_intrinsics = None

        # Build UI & Apply Styling
        self._setup_ui()
        self._apply_stylesheet()
        self._update_button_states()
        self._draw_reflectance(None, "")

    # --------------------------------------------------------------------------
    # UI INITIALIZATION & LAYOUT BUILDER
    # --------------------------------------------------------------------------
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("dashboardScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        try:
            QScroller.grabGesture(self._scroll.viewport(), QScroller.TouchGesture)
        except Exception:
            pass

        self._page = QWidget()
        self._page.setObjectName("dashboardPage")
        main_layout = QVBoxLayout(self._page)
        self._main_layout = main_layout
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # 1. TOP HEADER TITLE
        header_frame = QFrame()
        header_frame.setObjectName("headerFrame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 10, 12, 10)

        # University branding stays on the left. The supplied official logo sits on the right.
        logo_w = max(60, int(120 * self.scale_f))
        logo_h = max(30, int(60 * self.scale_f))
        self.top_logo_label = QLabel()
        self.top_logo_label.setFixedSize(logo_w, logo_h)
        self.top_logo_label.setAlignment(Qt.AlignCenter)
        self.top_logo_label.setScaledContents(False)
        self._show_logo_on(self.top_logo_label, UNIVERSITY_LOGO_PATH, logo_w, logo_h, "SMART LAB")

        header_layout.addWidget(self.top_logo_label)

        self.title_label = QLabel("ONION AI INSPECTION SYSTEM")
        self.title_label.setObjectName("appTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        self.title_label.setMinimumWidth(0)
        header_layout.addWidget(self.title_label, stretch=1)

        self.official_logo_label = QLabel()
        self.official_logo_label.setFixedSize(logo_w, logo_h)
        self.official_logo_label.setAlignment(Qt.AlignCenter)
        self.official_logo_label.setScaledContents(False)
        self._official_logo_present = self._show_logo_on(
            self.official_logo_label, OFFICIAL_LOGO_PATH, logo_w, logo_h, "",
        )
        header_layout.addWidget(self.official_logo_label)

        main_layout.addWidget(header_frame)

        # 2. THREE DISPLAY PANELS (LEFT: Live Video, CENTER: AI Result, RIGHT: Reflectance)
        self._panels_splitter = QSplitter(Qt.Horizontal)
        self._panels_splitter.setObjectName("panelSplitter")
        self._panels_splitter.setChildrenCollapsible(False)
        self._panels_splitter.setHandleWidth(10)

        # LEFT PANEL — LIVE VIDEO
        left_box = QGroupBox("LIVE VIDEO FEED")
        left_box.setObjectName("panelGroup")
        left_box.setMinimumWidth(200)
        left_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(8, 12, 8, 8)

        self.live_video_label = QLabel("Camera Offline\nPress Start Operation")
        self.live_video_label.setObjectName("videoViewport")
        self.live_video_label.setAlignment(Qt.AlignCenter)
        self.live_video_label.setWordWrap(True)
        self.live_video_label.setScaledContents(False)
        self.live_video_label.setMinimumSize(160, 110)
        self.live_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.live_video_label, stretch=1)

        # Left Zoom Controls
        left_zoom_layout = FlowLayout(h_spacing=8, v_spacing=6)
        self.btn_left_zoom_in = QPushButton("Zoom In")
        self.btn_left_zoom_out = QPushButton("Zoom Out")
        self.btn_left_zoom_in.clicked.connect(self._left_zoom_in)
        self.btn_left_zoom_out.clicked.connect(self._left_zoom_out)
        self._style_control(self.btn_left_zoom_in, 34)
        self._style_control(self.btn_left_zoom_out, 34)
        left_zoom_layout.addWidget(self.btn_left_zoom_in)
        left_zoom_layout.addWidget(self.btn_left_zoom_out)
        left_layout.addLayout(left_zoom_layout)

        # Model Selector Dropdown
        model_select_layout = QHBoxLayout()
        lbl_model = QLabel("Model:")
        lbl_model.setStyleSheet("font-weight: bold; font-size: 13px; color: #334155;")
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "ConvNeXt Tiny",
            "EfficientNetV2-B0",
            "DenseNet121",
            "ResNet",
            "Xception",
            "Stacking Meta-Classifier",
            "Ensemble Model"
        ])
        self.combo_model.setCurrentText("DenseNet121")
        self.combo_model.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_model.setMinimumContentsLength(10)
        self.combo_model.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo_model.setMinimumHeight(34)
        self.combo_model.currentTextChanged.connect(self._on_model_changed)
        model_select_layout.addWidget(lbl_model)
        model_select_layout.addWidget(self.combo_model, stretch=1)
        left_layout.addLayout(model_select_layout)

        self._panels_splitter.addWidget(left_box)

        # CENTER PANEL — CAPTURED IMAGE & RESULT
        center_box = QGroupBox("ANALYSIS & CAPTURE REVIEW")
        center_box.setObjectName("panelGroup")
        center_box.setMinimumWidth(200)
        center_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        center_layout = QVBoxLayout(center_box)
        center_layout.setContentsMargins(8, 12, 8, 8)

        self.prediction_text_label = QLabel("No Inspection Active")
        self.prediction_text_label.setObjectName("predictionHeader")
        self.prediction_text_label.setAlignment(Qt.AlignCenter)
        self.prediction_text_label.setWordWrap(True)
        self.prediction_text_label.setMinimumWidth(0)
        self.prediction_text_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        center_layout.addWidget(self.prediction_text_label)

        self.captured_image_label = QLabel("Captured Frame View")
        self.captured_image_label.setObjectName("videoViewport")
        self.captured_image_label.setAlignment(Qt.AlignCenter)
        self.captured_image_label.setWordWrap(True)
        self.captured_image_label.setScaledContents(False)
        self.captured_image_label.setMinimumSize(160, 110)
        self.captured_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        center_layout.addWidget(self.captured_image_label, stretch=1)

        # Center Zoom Controls
        center_zoom_layout = FlowLayout(h_spacing=8, v_spacing=6)
        self.btn_center_zoom_in = QPushButton("Zoom In")
        self.btn_center_zoom_out = QPushButton("Zoom Out")
        self.btn_center_zoom_in.clicked.connect(self._center_zoom_in)
        self.btn_center_zoom_out.clicked.connect(self._center_zoom_out)
        self._style_control(self.btn_center_zoom_in, 34)
        self._style_control(self.btn_center_zoom_out, 34)
        center_zoom_layout.addWidget(self.btn_center_zoom_in)
        center_zoom_layout.addWidget(self.btn_center_zoom_out)
        center_layout.addLayout(center_zoom_layout)

        # Navigation Controls
        nav_layout = FlowLayout(h_spacing=8, v_spacing=6)
        self.btn_prev_capture = QPushButton("<-")
        self.btn_next_capture = QPushButton("->")
        self.btn_restart_history = QPushButton("Restart")
        self.btn_xai = QPushButton("XAI")

        self.btn_prev_capture.clicked.connect(self._navigate_prev)
        self.btn_next_capture.clicked.connect(self._navigate_next)
        self.btn_restart_history.clicked.connect(self._navigate_restart)
        self.btn_xai.clicked.connect(self._trigger_xai)
        for button in (self.btn_prev_capture, self.btn_next_capture, self.btn_restart_history, self.btn_xai):
            self._style_control(button, 34)

        lbl_cap_no = QLabel("Capture No:")
        lbl_cap_no.setStyleSheet("font-size: 13px; color: #334155;")
        self.input_cap_no = QLineEdit()
        self.input_cap_no.setFixedWidth(56)
        self.input_cap_no.setMinimumHeight(34)
        self.input_cap_no.setPlaceholderText("#")
        self.input_cap_no.returnPressed.connect(self._lookup_capture_no)

        nav_layout.addWidget(self.btn_prev_capture)
        nav_layout.addWidget(self.btn_next_capture)
        nav_layout.addWidget(self.btn_restart_history)
        nav_layout.addWidget(lbl_cap_no)
        nav_layout.addWidget(self.input_cap_no)
        nav_layout.addWidget(self.btn_xai)
        center_layout.addLayout(nav_layout)

        self._panels_splitter.addWidget(center_box)

        # RIGHT PANEL — reflectance graph inside the existing third box
        right_frame = QFrame()
        right_frame.setObjectName("blankPanel")
        right_frame.setMinimumWidth(180)
        right_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(6, 6, 6, 6)
        self.reflectance_figure = Figure(figsize=(3.2, 2.4), dpi=80)
        self.reflectance_figure.patch.set_facecolor("#ffffff")
        self.reflectance_canvas = FigureCanvasQTAgg(self.reflectance_figure)
        self.reflectance_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.reflectance_canvas.setMinimumSize(140, 100)
        self.reflectance_ax = self.reflectance_figure.add_subplot(111)
        right_layout.addWidget(self.reflectance_canvas)
        self._panels_splitter.addWidget(right_frame)
        for index in range(3):
            self._panels_splitter.setStretchFactor(index, 1)

        main_layout.addWidget(self._panels_splitter, stretch=2)

        # 3. DISEASE & SIZE COUNTERS SECTION
        self._counters_layout = QBoxLayout(QBoxLayout.LeftToRight)
        self._counters_layout.setSpacing(10)

        # Disease Counters Box
        disease_box = QGroupBox("DISEASE COUNTERS")
        disease_box.setObjectName("panelGroup")
        disease_grid = QGridLayout(disease_box)

        self.lbl_count_normal = QLabel("PASS / NORMAL:\nNormal: 0")
        self.lbl_count_rotten = QLabel("ROTTEN:\nRotten: 0")
        self.lbl_count_sprouted = QLabel("SPROUTED:\nSprouted: 0")
        self.lbl_count_moldy = QLabel("MOLDY:\nMoldy: 0")

        for lbl in [self.lbl_count_normal, self.lbl_count_rotten, self.lbl_count_sprouted, self.lbl_count_moldy]:
            lbl.setObjectName("counterBadge")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            lbl.setMinimumWidth(0)
            lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        disease_grid.addWidget(self.lbl_count_normal, 0, 0)
        disease_grid.addWidget(self.lbl_count_sprouted, 0, 1)
        disease_grid.addWidget(self.lbl_count_rotten, 1, 0)
        disease_grid.addWidget(self.lbl_count_moldy, 1, 1)
        disease_grid.setColumnStretch(0, 1)
        disease_grid.setColumnStretch(1, 1)

        self._counters_layout.addWidget(disease_box, stretch=1)

        # Size Counters Box
        size_box = QGroupBox("SIZE MEASUREMENT COUNTERS")
        size_box.setObjectName("panelGroup")
        size_grid = QGridLayout(size_box)

        self.lbl_size_cat1 = QLabel("Category 1\n< 40 mm\nServo 8\n0")
        self.lbl_size_cat2 = QLabel("Category 2\n40–50 mm\nServo 7\n0")
        self.lbl_size_cat3 = QLabel("Category 3\n50–60 mm\nServo 9\n0")
        self.lbl_size_cat4 = QLabel("Category 4\n> 60 mm\nServo 6\n0")

        for lbl in [self.lbl_size_cat1, self.lbl_size_cat2, self.lbl_size_cat3, self.lbl_size_cat4]:
            lbl.setObjectName("counterBadge")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            lbl.setMinimumWidth(0)
            lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        size_grid.addWidget(self.lbl_size_cat1, 0, 0)
        size_grid.addWidget(self.lbl_size_cat2, 0, 1)
        size_grid.addWidget(self.lbl_size_cat3, 1, 0)
        size_grid.addWidget(self.lbl_size_cat4, 1, 1)
        size_grid.setColumnStretch(0, 1)
        size_grid.setColumnStretch(1, 1)

        self._counters_layout.addWidget(size_box, stretch=1)

        main_layout.addLayout(self._counters_layout, stretch=0)

        # 4. BOTTOM OPERATION BAR
        self._bottom_bar = QFrame()
        self._bottom_bar.setObjectName("bottomBar")
        bottom_layout = FlowLayout(self._bottom_bar, margin=10, h_spacing=10, v_spacing=8)

        # Logo Display
        b_logo_w = max(60, int(110 * self.scale_f))
        b_logo_h = max(26, int(48 * self.scale_f))
        self.logo_label = QLabel()
        self.logo_label.setFixedSize(b_logo_w, b_logo_h)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setScaledContents(False)
        self._show_logo_on(self.logo_label, UNIVERSITY_LOGO_PATH, b_logo_w, b_logo_h, "SMART LAB")

        bottom_layout.addWidget(self.logo_label)

        # Batch No Input
        lbl_batch = QLabel("Batch No:")
        lbl_batch.setStyleSheet("font-weight: bold; font-size: 13px; color: #334155;")
        self.input_batch_no = QLineEdit()
        self.input_batch_no.setPlaceholderText("e.g. B001")
        self.input_batch_no.setMinimumWidth(88)
        self.input_batch_no.setMaximumWidth(140)
        self.input_batch_no.setMinimumHeight(36)

        bottom_layout.addWidget(lbl_batch)
        bottom_layout.addWidget(self.input_batch_no)

        # Start Operation Button. The same button becomes STOP OPERATION while a batch is running.
        self.btn_start_operation = QPushButton("START OPERATION")
        self.btn_start_operation.setObjectName("btnPrimary")
        self._style_control(self.btn_start_operation, 40)
        self.btn_start_operation.clicked.connect(self._on_start_operation_clicked)
        bottom_layout.addWidget(self.btn_start_operation)

        # Report Button
        self.btn_report = QPushButton("REPORT")
        self._style_control(self.btn_report, 40)
        self.btn_report.clicked.connect(self._on_report_clicked)
        bottom_layout.addWidget(self.btn_report)

        # Calibrate Button
        self.btn_calibrate = QPushButton("CALIBRATE SIZE")
        self._style_control(self.btn_calibrate, 40)
        self.btn_calibrate.clicked.connect(self._on_calibrate_clicked)
        self.btn_calibrate.setStyleSheet("background-color: #f59e0b; color: #ffffff;")
        bottom_layout.addWidget(self.btn_calibrate)

        # Slot Quantity Input
        lbl_slot = QLabel("Onions in slot:")
        lbl_slot.setStyleSheet("font-weight: bold; font-size: 13px; color: #334155;")
        self.input_slot_quantity = QLineEdit()
        self.input_slot_quantity.setPlaceholderText("e.g. 100")
        self.input_slot_quantity.setMinimumWidth(72)
        self.input_slot_quantity.setMaximumWidth(110)
        self.input_slot_quantity.setMinimumHeight(36)

        bottom_layout.addWidget(lbl_slot)
        bottom_layout.addWidget(self.input_slot_quantity)

        # Slot Progress Display
        self.lbl_slot_progress = QLabel("Processed: 0 / 0")
        self.lbl_slot_progress.setObjectName("progressStatus")
        self.lbl_slot_progress.setAlignment(Qt.AlignCenter)
        self.lbl_slot_progress.setMinimumHeight(36)
        bottom_layout.addWidget(self.lbl_slot_progress)

        main_layout.addWidget(self._bottom_bar)
        self._scroll.setWidget(self._page)
        outer_layout.addWidget(self._scroll, stretch=1)

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Status: Ready")

    def _style_control(self, widget, min_height):
        widget.setMinimumHeight(min_height)
        widget.setMinimumWidth(0)
        widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

    def _show_logo_on(self, label, path, width, height, fallback_text):
        """Draw one logo at its own proportions. A missing file leaves the label usable."""
        label.setScaledContents(False)
        pix = QPixmap(path) if path and os.path.isfile(path) else QPixmap()
        if pix.isNull():
            label.setPixmap(QPixmap())
            label.setText(fallback_text)
            if fallback_text:
                label.setStyleSheet("font-weight: bold; color: #38bdf8;")
            return False
        label.setText("")
        label.setPixmap(pix.scaled(int(width), int(height), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        return True

    def _apply_window_icon(self):
        """Use the supplied onion logo as the window icon."""
        for path in (
            OFFICIAL_ICON_PATH,
            OFFICIAL_LOGO_PATH,
            os.path.join(BASE_DIR, "app.ico"),
            UNIVERSITY_LOGO_PATH,
        ):
            if not path or not os.path.isfile(path):
                continue
            icon = QIcon(path)
            if not icon.isNull():
                self.setWindowIcon(icon)
                return

    def _fit_logos(self):
        scale = getattr(self, "scale_f", 1.0)
        top_w = max(48, int(110 * scale))
        top_h = max(24, int(52 * scale))
        self.top_logo_label.setFixedSize(top_w, top_h)
        self.official_logo_label.setFixedSize(top_w, top_h)
        bot_w = max(48, int(100 * scale))
        bot_h = max(22, int(44 * scale))
        self.logo_label.setFixedSize(bot_w, bot_h)
        self._show_logo_on(self.top_logo_label, UNIVERSITY_LOGO_PATH, top_w, top_h, "SMART LAB")
        self._official_logo_present = self._show_logo_on(
            self.official_logo_label, OFFICIAL_LOGO_PATH, top_w, top_h, "",
        )
        self._show_logo_on(self.logo_label, UNIVERSITY_LOGO_PATH, bot_w, bot_h, "SMART LAB")

    def _apply_responsive_layout(self):
        """Fit the three panels and controls to the current window without clipping them."""
        if getattr(self, "_applying_layout", False) or not hasattr(self, "_scroll"):
            return
        viewport = self._scroll.viewport()
        width = viewport.width()
        height = viewport.height()
        if width < 160 or height < 160:
            return
        self._applying_layout = True
        try:
            scale = max(0.62, min(1.0, width / 1280.0, height / 800.0))
            bucket = int(round(scale * 20))
            if bucket != self._style_bucket:
                self.scale_f = scale
                self.is_small_screen = scale < 0.95
                self._style_bucket = bucket
                self._apply_stylesheet()
                self._fit_logos()
            self._panels_splitter.setHandleWidth(14 if scale < 0.85 else 8)
            orientation = Qt.Vertical if width < 720 else Qt.Horizontal
            if self._panels_splitter.orientation() != orientation:
                self._panels_splitter.setOrientation(orientation)
                span = self._panels_splitter.height() if orientation == Qt.Vertical else self._panels_splitter.width()
                part = max(1, int(span / 3))
                self._panels_splitter.setSizes([part, part, part])
            direction = QBoxLayout.TopToBottom if width < 760 else QBoxLayout.LeftToRight
            if self._counters_layout.direction() != direction:
                self._counters_layout.setDirection(direction)
            margin = 8 if scale < 0.85 else 14
            self._main_layout.setContentsMargins(margin, margin, margin, margin)
            self._main_layout.setSpacing(8 if scale < 0.85 else 12)
            if self._official_logo_present:
                self.official_logo_label.setVisible(True)
            else:
                self.official_logo_label.setVisible(width >= 900)
            available = width
            content_height = self._page.minimumSizeHint().height()
            if content_height > height:
                available = max(200, width - self._scroll.verticalScrollBar().sizeHint().width())
            self._bottom_bar.setMinimumHeight(max(48, self._bottom_bar.heightForWidth(available)))
            self._page.setMinimumWidth(0)
            self._page.setMaximumWidth(available)
            self._page.setMinimumHeight(0)
        finally:
            self._applying_layout = False

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_responsive_layout()
        self._refit_displayed_images()
        QTimer.singleShot(0, self._apply_responsive_layout)

    # --------------------------------------------------------------------------
    # QSS INDUSTRIAL STYLESHEET
    # --------------------------------------------------------------------------
    def _apply_stylesheet(self):
        def scaled_px(px):
            return max(10, int(px * getattr(self, 'scale_f', 1.0)))
            
        style = f"""
        QMainWindow {{
            background-color: #f8fafc;
        }}
        #headerFrame {{
            background: linear-gradient(135deg, #ffffff, #f1f5f9);
            border: 1px solid #cbd5e1;
            border-radius: 8px;
        }}
        #dashboardPage, #dashboardScroll {{
            background-color: #f8fafc;
            border: none;
        }}
        #appTitle {{
            font-size: {scaled_px(24)}px;
            font-weight: 800;
            color: #0284c7;
            letter-spacing: {0.4 if getattr(self, 'scale_f', 1.0) < 0.85 else 1.5}px;
        }}
        QSplitter#panelSplitter::handle {{
            background-color: #cbd5e1;
        }}
        QScrollBar:vertical {{
            width: 16px;
            background: #f1f5f9;
        }}
        QScrollBar:horizontal {{
            height: 16px;
            background: #f1f5f9;
        }}
        QScrollBar::handle {{
            background: #94a3b8;
            border-radius: 6px;
            min-height: 28px;
            min-width: 28px;
        }}
        QGroupBox#panelGroup {{
            font-size: {scaled_px(13)}px;
            font-weight: bold;
            color: #475569;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            margin-top: 6px;
            padding-top: 10px;
            background-color: #ffffff;
        }}
        #blankPanel {{
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
        }}
        #videoViewport {{
            background-color: #e2e8f0;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            color: #475569;
            font-size: {scaled_px(15)}px;
            font-weight: bold;
        }}
        #predictionHeader {{
            font-size: {scaled_px(22)}px;
            font-weight: bold;
            color: #16a34a;
            padding: 4px;
        }}
        #counterBadge {{
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            color: #0f172a;
            font-size: {scaled_px(14)}px;
            font-weight: bold;
            padding: {max(2, int(8 * getattr(self, 'scale_f', 1.0)))}px;
        }}
        #bottomBar {{
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
        }}
        #btnPrimary {{
            background-color: #0ea5e9;
            color: #ffffff;
            font-weight: bold;
            font-size: {scaled_px(14)}px;
            border-radius: 6px;
            padding: 6px 18px;
        }}
        #btnPrimary:hover {{
            background-color: #0284c7;
        }}
        #btnPrimary:disabled {{
            background-color: #e2e8f0;
            color: #94a3b8;
        }}
        QPushButton {{
            background-color: #f1f5f9;
            color: #334155;
            font-weight: bold;
            border-radius: 6px;
            padding: 6px 12px;
            border: 1px solid #cbd5e1;
        }}
        QPushButton:hover {{
            background-color: #e2e8f0;
        }}
        QPushButton:disabled {{
            background-color: #f8fafc;
            color: #94a3b8;
            border: 1px solid #e2e8f0;
        }}
        QComboBox, QLineEdit {{
            background-color: #ffffff;
            color: #334155;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            padding: 5px 8px;
            font-size: {scaled_px(13)}px;
        }}
        #progressStatus {{
            font-size: {scaled_px(14)}px;
            font-weight: bold;
            color: #0284c7;
            padding: 4px 10px;
            border: 1px solid #bae6fd;
            border-radius: 6px;
            background-color: #f0f9ff;
        }}
        QStatusBar {{
            color: #475569;
            background-color: #f8fafc;
        }}
        """
        self.setStyleSheet(style)

    # --------------------------------------------------------------------------
    # CAMERA / SIZE CALIBRATION
    # --------------------------------------------------------------------------
    def _on_calibrate_clicked(self):
        """Calibrate the metric size measurement using a known-size reference onion.

        The old button measured an arbitrary 60x60-pixel depth patch and called
        that the size calibration. That only calibrated the belt height and could
        silently reset the installation to 240 mm, so it did not correct actual
        onion diameter errors. This routine uses a real reference onion and the
        same pixel-geometry equation used by the live diameter measurement.
        """
        if self.is_operation_active:
            QMessageBox.warning(
                self, "Calibration",
                "Please STOP the current operation before calibrating.\n\n"
                "Place one reference onion on the conveyor and keep it fully visible."
            )
            return

        frame_bgr = getattr(self, "latest_live_frame_bgr", None)
        depth_raw = getattr(self, "latest_depth_raw", None)
        intrinsics = getattr(self, "latest_intrinsics", None)

        if frame_bgr is None or intrinsics is None:
            QMessageBox.warning(
                self, "Calibration",
                "Camera feed is not ready. Start the camera/live preview first, "
                "then place the reference onion on the conveyor."
            )
            return

        actual_diameter, ok = QInputDialog.getDouble(
            self,
            "Calibrate Onion Size",
            "Enter the actual diameter of the reference onion (mm):",
            60.0, 20.0, 150.0, 1
        )
        if not ok:
            return

        # Detect the reference onion using exactly the same belt detector used
        # during normal operation. This avoids calibrating from the empty belt.
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            belt_mask = self.onion_tracker._conveyor_belt_mask(gray)
            blobs = self.onion_tracker._belt_onion_blobs(
                frame_bgr, gray, belt_mask, depth_raw,
                float(getattr(self.camera_worker, "depth_scale", 0.001) or 0.001)
            )
        except Exception as exc:
            QMessageBox.warning(self, "Calibration Failed", f"Could not analyze the reference onion.\n\n{exc}")
            return

        if not blobs:
            QMessageBox.warning(
                self, "Calibration Failed",
                "No onion was detected on the conveyor.\n\n"
                "Place one complete reference onion in the camera view and try again."
            )
            return

        # The detector intentionally returns at most one blob. Use its measured
        # pixel radius and the current RealSense colour focal length.
        _cx, _cy, pixel_radius = blobs[0]
        fx = float(getattr(intrinsics, "fx", 0.0) or 0.0)
        fy = float(getattr(intrinsics, "fy", 0.0) or 0.0)
        focal = (fx + fy) / 2.0 if fx > 1.0 and fy > 1.0 else (fx if fx > 1.0 else fy)
        if focal <= 1.0 or pixel_radius <= 0:
            QMessageBox.warning(self, "Calibration Failed", "Invalid camera intrinsics were received from the RealSense camera.")
            return

        # Invert the same spherical/perspective equation used by
        # _fixed_height_diameter_mm() to obtain the effective camera height.
        q = float(pixel_radius) / focal
        denominator = 2.0 * q
        if denominator <= 0:
            QMessageBox.warning(self, "Calibration Failed", "Invalid reference onion geometry.")
            return

        global CONVEYOR_DEPTH_MM
        calibrated_height = float(actual_diameter) * (math.sqrt(1.0 + q * q) + q) / denominator

        # Sanity range prevents a mistaken reference diameter or bad detection
        # from corrupting the metric scale.
        if not 150.0 <= calibrated_height <= 350.0:
            QMessageBox.warning(
                self, "Calibration Failed",
                f"The calculated camera height ({calibrated_height:.1f} mm) is outside "
                "the safe 150–350 mm range.\n\n"
                "Check the reference diameter and make sure the onion is fully visible."
            )
            return

        CONVEYOR_DEPTH_MM = calibrated_height

        # Any previously locked onion sizes were calculated with the old scale.
        # Clear them so the next onion is measured again with the new calibration.
        try:
            self.onion_tracker.diameter_filter.clear()
            self.onion_tracker.active_tracks.clear()
            self.onion_tracker._blob_tracks = []
            self.onion_tracker._next_blob_id = 1
        except Exception:
            pass

        # Keep the depth baseline in sync when the optional external tracker
        # provides the calibration hook.
        if depth_raw is not None and self.onion_tracker.tracker is not None:
            calibrate_baseline = getattr(self.onion_tracker.tracker, "calibrate_baseline", None)
            if callable(calibrate_baseline):
                try:
                    calibrate_baseline(depth_raw)
                except Exception:
                    pass

        QMessageBox.information(
            self, "Size Calibration Successful",
            f"Reference onion: {actual_diameter:.1f} mm\n"
            f"Detected radius: {pixel_radius:.1f} px\n"
            f"Effective camera height: {CONVEYOR_DEPTH_MM:.1f} mm\n\n"
            "The new calibration will be used for the next onion."
        )
        self.statusBar.showMessage(
            f"Status: Onion size calibrated using {actual_diameter:.1f} mm reference "
            f"→ effective height {CONVEYOR_DEPTH_MM:.1f} mm."
        )

    # --------------------------------------------------------------------------
    # START OPERATION & VALIDATION LOGIC
    # --------------------------------------------------------------------------
    def _on_start_operation_clicked(self):
        if self.is_operation_active:
            self._stop_operation()
            return

        batch_no = self.input_batch_no.text().strip()
        slot_qty_str = self.input_slot_quantity.text().strip()

        # Input Validation
        if not batch_no and not slot_qty_str:
            QMessageBox.warning(self, "Input Validation", "Please enter the Batch Number and number of onions in one slot.")
            return
        if not batch_no:
            QMessageBox.warning(self, "Input Validation", "Please enter the Batch Number.")
            return
        if not slot_qty_str or not slot_qty_str.isdigit() or int(slot_qty_str) <= 0:
            QMessageBox.warning(self, "Input Validation", "Please enter how many onions are in one slot.")
            return

        self.batch_number = batch_no
        self.slot_target_quantity = int(slot_qty_str)

        self._start_operation()

    def _start_operation(self):
        print(f"[Operation] Starting Batch {self.batch_number} (Target: {self.slot_target_quantity} onions)...")
        model_spec = describe_disease_model(self.combo_model.currentText())
        print(
            f"[MODEL] {model_spec['model']} classes={model_spec['classes']} "
            f"input={model_spec['input_size']} color={model_spec['color']} "
            f"preprocess={model_spec['preprocessing']} "
            f"min_confidence={model_spec['min_confidence']} weights={model_spec['weights']}"
        )
        self.batch_started_at = datetime.now()
        self.is_operation_active = True

        # Reset Operation Counters
        self.unique_onions_processed = 0
        self.captured_count = 0
        self._pending_onion_captures = []
        self._queued_onion_ids = set()
        self._sorted_servo_ids = set()
        self._counted_inspection_ids = set()
        self._dispatch_started = set()
        self._sorting_refused = set()
        self._submitted_labels = set()
        self._submitted_diameter = {}
        self._cancelled_inspections = set()
        self._batch_flow = BatchFlow(self.slot_target_quantity)
        self._visible_onion_labels = set()
        self._no_onion_shown = False
        self._live_track_signature = None
        self._inspection_target_id = None
        self._browsing_history = False
        self._disease_shown_for = None
        self._review_message = ""
        self._live_frame_token = 0
        self._classification_state = {}
        self._classification_waits = {}
        self._last_servo_onion = None
        self._last_end_onion = None

        self.normal_count = 0
        self.rotten_count = 0
        self.sprouted_count = 0
        self.moldy_count = 0

        self.size_cat1 = 0
        self.size_cat2 = 0
        self.size_cat3 = 0
        self.size_cat4 = 0

        self._update_counter_labels()
        self.inspection_history.clear()
        self.current_history_index = -1
        self.onion_tracker.reset()

        # UI Lock during operation
        self.input_batch_no.setEnabled(False)
        self.input_slot_quantity.setEnabled(False)
        self.combo_model.setEnabled(False)

        self.btn_start_operation.setText("STOP OPERATION")
        self.btn_start_operation.setStyleSheet("background-color: #dc2626; color: #ffffff;")

        # Start Camera Worker
        if self.camera_worker is None or not self.camera_worker.isRunning():
            self.camera_worker = CameraWorker(CAMERA_INDEX)
            self.camera_worker.frame_ready.connect(self._on_frame_ready)
            self.camera_worker.error_occurred.connect(self._on_camera_failed)
            self.camera_worker.start()

        # Capture history is now driven by confirmed tracked onions, not by a
        # periodic full-frame timer.
        if hasattr(self, 'capture_timer'):
            self.capture_timer.stop()

        self.lbl_slot_progress.setText(f"Processed: 0 / {self.slot_target_quantity}")
        self.statusBar.showMessage(f"Status: RUNNING - Model: {self.combo_model.currentText()}")

    def _stop_operation(self, completed_normally=False):
        print("[Operation] Stopping operation...")
        halt_motion()
        self.is_operation_active = False

        if hasattr(self, 'capture_timer'):
            self.capture_timer.stop()

        if self.camera_worker is not None and self.camera_worker.isRunning():
            self.camera_worker.stop()
            self.camera_worker = None

        # Unlock Controls
        self.input_batch_no.setEnabled(True)
        self.input_slot_quantity.setEnabled(True)
        self.combo_model.setEnabled(True)

        self.btn_start_operation.setText("START OPERATION")
        self.btn_start_operation.setObjectName("btnPrimary")
        self.btn_start_operation.setStyleSheet("")

        if completed_normally:
            self.lbl_slot_progress.setText(f"Processed: {self.unique_onions_processed} / {self.slot_target_quantity} | Status: SLOT COMPLETED")
            QMessageBox.information(self, "Slot Completed", f"Slot Completed – {self.slot_target_quantity} onions processed.")
            self.statusBar.showMessage("Status: SLOT COMPLETED")
        else:
            self.statusBar.showMessage("Status: STOPPED")

        self._update_button_states()

    def _on_model_changed(self, new_model: str):
        if hasattr(self, '_is_navigating') and self._is_navigating:
            return
        self.statusBar.showMessage(f"Status: Switched model to {new_model}.")
        
        if self.current_history_index >= 0 and not self.is_operation_active:
            current_item = self.inspection_history[self.current_history_index]
            try:
                img_for_inference = current_item.inference_image if current_item.inference_image is not None else current_item.image
                pred_class, conf, prob_dict = self.model_manager.predict(new_model, img_for_inference)
                current_item.predicted_class = pred_class
                current_item.confidence = conf
                current_item.class_probabilities = prob_dict
                current_item.model_name = new_model
                current_item.xai_image = None
                self._display_capture_result(current_item)
                self.statusBar.showMessage(f"Status: Re-evaluated Capture #{current_item.capture_number} with {new_model}.")
            except Exception as e:
                self.statusBar.showMessage(f"Error updating model: {str(e)}")

    # --------------------------------------------------------------------------
    # CAMERA & INSPECTION PIPELINE CALLBACKS
    # --------------------------------------------------------------------------
    def _on_frame_ready(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray, depth_raw: object, intrinsics: object, timestamp: float):
        try:
            self._handle_live_frame(frame_bgr, frame_rgb, depth_raw, intrinsics, timestamp)
        finally:
            if self.camera_worker is not None:
                self.camera_worker.ui_ready.set()

    def _handle_live_frame(self, frame_bgr: np.ndarray, frame_rgb: np.ndarray, depth_raw: object, intrinsics: object, timestamp: float):
        fresh_bgr, fresh_rgb = prepare_live_frame(frame_bgr)
        if fresh_bgr is None or fresh_rgb is None:
            return
        frame_bgr = fresh_bgr
        frame_rgb = fresh_rgb
        if depth_raw is not None:
            depth_raw = np.array(depth_raw, copy=True)
        self._live_frame_token = int(getattr(self, "_live_frame_token", 0)) + 1
        frame_token = self._live_frame_token
        self.latest_live_frame_bgr = frame_bgr
        self.latest_live_frame_rgb = frame_rgb
        self.latest_depth_raw = depth_raw
        self.latest_intrinsics = intrinsics

        if not self.is_operation_active:
            self._last_live_frame = frame_rgb
            pixmap = self._cv_image_to_pixmap(frame_rgb, self.left_zoom_factor, self.live_video_label)
            self.live_video_label.setPixmap(pixmap)
            return
            
        current_objects, _departed = self.onion_tracker.process_continuous(
            frame_bgr, frame_rgb, depth_raw, intrinsics, getattr(self.camera_worker, 'depth_scale', 0.001),
            hold_ids=self._held_track_ids(),
        )

        display_img = frame_rgb.copy()
        frame_h, frame_w = display_img.shape[:2]
        incoming = tuple(item[0] for item in current_objects)
        if incoming != self._live_track_signature:
            self._live_track_signature = incoming
            retained = self._inspection_target_id
            if retain_target_on_visibility_change(
                retained,
                self._classification_state.get(retained, ""),
                retained in self._dispatch_started,
                self._sort_is_settled(retained),
            ) is None:
                self._inspection_target_id = None
                self._browsing_history = False
                self._disease_shown_for = None
                self._classification_state = {}
                self._classification_waits = {}
                self._crop_rejection_logged = set()
                self._clear_current_inspection()
        drawn = 0
        tracked_labels = set()
        review_crop = None
        failed_diameter = False
        diameter_ready = False
        panel = None
        for obj_id, bbox, dia_mm, cat, usda in current_objects:
            bx, by, bw, bh = [int(v) for v in bbox]
            if bw < 4 or bh < 4 or bx >= frame_w or by >= frame_h or bx + bw <= 0 or by + bh <= 0:
                continue
            bx = max(0, min(bx, frame_w - 1))
            by = max(0, min(by, frame_h - 1))
            bw = min(bw, frame_w - bx)
            bh = min(bh, frame_h - by)
            center_x, center_y = bx + bw // 2, by + bh // 2
            obj_r = max(1, min(bw, bh) // 2)
            cv2.circle(display_img, (center_x, center_y), obj_r, (0, 255, 0), 2)
            drawn += 1
            onion_label = f"ONION-{obj_id:03d}"
            tracked_labels.add(onion_label)
            self._batch_flow.note_detected(onion_label)
            status = self.onion_tracker.diameter_filter.status(obj_id)
            action = inspection_action(status, dia_mm)
            if panel is None:
                track = self.onion_tracker.active_tracks.get(obj_id)
                panel = {
                    "label": onion_label,
                    "status": status,
                    "diameter": dia_mm,
                    "frames": track.frames_tracked if track is not None else 1,
                }
            crop_info = extract_inspection_crop(frame_rgb, (bx, by, bw, bh), obj_id, frame_token)
            if crop_belongs_to_track(crop_info, obj_id, frame_token):
                review_crop = crop_info["image"]
            elif action == "classify" and onion_label not in getattr(self, "_crop_rejection_logged", set()):
                self._crop_rejection_logged = getattr(self, "_crop_rejection_logged", set())
                self._crop_rejection_logged.add(onion_label)
                print(
                    f"[INSPECT] frame={frame_token} state=diameter_ready id={onion_label} "
                    f"bbox={(bx, by, bw, bh)} score=none reason=inspection crop was rejected"
                )
            if action == "diameter_failed":
                failed_diameter = True
                cv2.putText(
                    display_img, "DIAMETER NOT AVAILABLE", (bx, max(20, by - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA,
                )
                continue
            if action != "classify":
                continue
            diameter_ready = True
            text = f"#{obj_id} Dia: {dia_mm:.1f}mm"
            cv2.putText(display_img, text, (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(display_img, text, (bx, max(20, by - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if review_crop is not None:
                self._submit_visible_onion(obj_id, (bx, by, bw, bh), dia_mm, cat, usda, review_crop)

        self._visible_onion_labels = tracked_labels
        active_label = panel["label"] if panel is not None else None
        state = self._classification_state.get(active_label)
        if state == "running" and active_label:
            waits = getattr(self, "_classification_waits", {})
            waits[active_label] = waits.get(active_label, 0) + 1
            self._classification_waits = waits
            if waits[active_label] >= 90:
                self._classification_state[active_label] = "timed_out"
                state = "timed_out"
                print(
                    f"[INSPECT] frame={frame_token} state=classification_timeout "
                    f"id={active_label} reason=disease classification timed out"
                )
        disease_ready = state == "done" and self._disease_shown_for == active_label
        if drawn == 0:
            cv2.putText(
                display_img, "NO ONION DETECTED", (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA,
            )
            cv2.putText(
                display_img, "NO ONION DETECTED", (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 204, 21), 2, cv2.LINE_AA,
            )
            self._show_waiting_for_onion()
        elif not disease_ready and not self._browsing_history:
            stage = inspection_stage(
                True,
                panel["frames"] if panel else 0,
                panel["status"] if panel else "pending",
                panel["diameter"] if panel else None,
                state or "",
                active_label in self._dispatch_started,
            )
            if state == "uncertain":
                self._show_uncertain()
            elif stage == "unstable":
                self._show_measurement_unstable()
            elif stage == "diameter_failed":
                self._show_diameter_unavailable()
            elif stage in ("classifying", "classification_timeout"):
                message = review_panel_message(True, "valid", False, panel["diameter"] if panel else None)
                if stage == "classification_timeout" and message:
                    message = message.replace(" — ANALYZING DISEASE", "")
                self._set_review_message(message, review_crop)
            else:
                self._set_review_message(REVIEW_MEASURING, review_crop)

        self._last_live_frame = display_img
        pixmap = self._cv_image_to_pixmap(display_img, self.left_zoom_factor, self.live_video_label)
        self.live_video_label.setPixmap(pixmap)

    def _sort_is_settled(self, onion_label):
        if not onion_label:
            return False
        return onion_label in self._batch_flow.sorted or onion_label in self._batch_flow.failed

    def _held_track_ids(self):
        """Keep a track that has left the picture while its disease call or belt cycle is still open."""
        held = set()
        for label, state in self._classification_state.items():
            if not inspection_is_busy(state, label in self._dispatch_started, self._sort_is_settled(label)):
                continue
            if "-" in str(label) and str(label).rsplit("-", 1)[-1].isdigit():
                held.add(int(str(label).rsplit("-", 1)[-1]))
        return held

    def _clear_current_inspection(self):
        """Drop the previous onion's diameter, crop, disease label, and confidence."""
        self._displayed_onion_id = None
        self._no_onion_shown = False
        self.prediction_text_label.setText("")
        self.prediction_text_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #e2e8f0;")
        self.captured_image_label.setPixmap(QPixmap())
        self.captured_image_label.setText("")
        self._last_capture_frame = None
        self._draw_reflectance(None, "")

    def _log_review_state(self, message):
        if message == getattr(self, "_logged_review_state", None):
            return
        self._logged_review_state = message
        report = getattr(self.onion_tracker, "_detection_report", {}) or {}
        score = report.get("score")
        score_text = "none" if score is None else f"{float(score):.2f}"
        print(
            f"[INSPECT] frame={getattr(self, '_live_frame_token', 0)} state={message} "
            f"id={report.get('track_id', 'none')} score={score_text} "
            f"bbox={report.get('bbox', 'none')} reason={report.get('reason', message)}"
        )

    def _set_review_message(self, message, image=None):
        """Update the review panel without replacing a history item the operator is viewing."""
        if self._browsing_history:
            return
        if message != self._review_message:
            self._review_message = message
            self._no_onion_shown = False
            self.prediction_text_label.setText(message)
            self.prediction_text_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #facc15;")
            self._log_review_state(message)
        if image is not None:
            self._show_capture_preview(image, None)
        elif message in (REVIEW_WAITING, "DIAMETER NOT AVAILABLE"):
            self.captured_image_label.setPixmap(QPixmap())
            self.captured_image_label.setText(message)
            self._last_capture_frame = None

    def _release_lost_measurements(self):
        """A track that has left is not allowed to keep a pending diameter or an unsaved prediction."""
        active = set(self.onion_tracker.active_tracks)
        for label in list(self._classification_state):
            obj_id = label_obj_id(label)
            if obj_id in active:
                continue
            if label in self._counted_inspection_ids or label in self._batch_flow.inspected:
                continue
            self._cancelled_inspections.add(label)
            self._classification_state.pop(label, None)
            self.onion_tracker.diameter_filter.invalidate_pending(obj_id)
            if self._inspection_target_id == label:
                self._inspection_target_id = None
            print(
                f"[INSPECT] frame={getattr(self, '_live_frame_token', 0)} state=cleared "
                f"id={label} reason=onion left before the inspection was saved"
            )

    def _show_waiting_for_onion(self):
        """Clear the current onion view. Disease and size counters stay as history."""
        if self._browsing_history:
            return
        self._release_lost_measurements()
        if self._review_message == REVIEW_WAITING and self._no_onion_shown and self._last_capture_frame is None:
            return
        self._disease_shown_for = None
        self._clear_current_inspection()
        self._no_onion_shown = True
        self._review_message = REVIEW_WAITING
        self.prediction_text_label.setText(REVIEW_WAITING)
        self.prediction_text_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #facc15;")
        self.captured_image_label.setPixmap(QPixmap())
        self.captured_image_label.setText(REVIEW_WAITING)
        self._last_capture_frame = None
        self._log_review_state(REVIEW_WAITING)

    def _show_no_onion(self):
        """Clear the current result when this frame has no valid onion."""
        self._show_waiting_for_onion()

    def _show_measurement_unstable(self):
        self._no_onion_shown = False
        self._review_message = "MEASUREMENT UNSTABLE"
        self._log_review_state("MEASUREMENT UNSTABLE")
        self.prediction_text_label.setText("MEASUREMENT UNSTABLE")
        self.prediction_text_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #facc15;")
        self.captured_image_label.setPixmap(QPixmap())
        self.captured_image_label.setText("MEASUREMENT UNSTABLE")
        self._last_capture_frame = None

    def _show_diameter_unavailable(self):
        self._no_onion_shown = False
        self._review_message = "DIAMETER NOT AVAILABLE"
        self._log_review_state("DIAMETER NOT AVAILABLE")
        self.prediction_text_label.setText("DIAMETER NOT AVAILABLE")
        self.prediction_text_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #facc15;")
        self.captured_image_label.setPixmap(QPixmap())
        self.captured_image_label.setText("DIAMETER NOT AVAILABLE")
        self._last_capture_frame = None

    def _show_uncertain(self):
        self._no_onion_shown = False
        self._review_message = "UNCERTAIN"
        self._log_review_state("UNCERTAIN")
        self._disease_shown_for = None
        self.prediction_text_label.setText("UNCERTAIN")
        self.prediction_text_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #facc15;")

    def _submit_visible_onion(self, obj_id, bbox, dia_mm, cat, usda, crop_img):
        """Classify this tracking ID only after its own diameter is valid."""
        if not diameter_is_valid(dia_mm) or crop_img is None:
            return
        if inspection_action(self.onion_tracker.diameter_filter.status(obj_id), dia_mm) != "classify":
            return
        if obj_id in self._queued_onion_ids or obj_id in self.onion_tracker.counted_ids:
            return
        if not PhysicalOnionTracker.crop_matches_onion(crop_img):
            return
        if self.inference_worker.isRunning():
            return
        onion_label = f"ONION-{obj_id:03d}"
        retained = self._inspection_target_id
        if retained and retained != onion_label and inspection_is_busy(
            self._classification_state.get(retained, ""),
            retained in self._dispatch_started,
            self._sort_is_settled(retained),
        ):
            return
        if not crop_belongs_to_track(
            {"image": crop_img, "track_id": obj_id, "frame_token": self._live_frame_token},
            obj_id,
            self._live_frame_token,
        ):
            return
        stamped = self._stamp_diameter(crop_img, dia_mm)
        self._queued_onion_ids.add(obj_id)
        self._submitted_labels.add(onion_label)
        self._submitted_diameter[onion_label] = float(dia_mm)
        self._inspection_target_id = onion_label
        self._displayed_onion_id = onion_label
        self._no_onion_shown = False
        self._classification_state[onion_label] = "running"
        self.captured_count += 1
        self.inference_worker.process_capture(
            capture_num=self.captured_count,
            onion_id=onion_label,
            frame_rgb=crop_img,
            display_img=stamped,
            model_name=self.combo_model.currentText(),
            batch_no=self.batch_number,
            size_mm=float(dia_mm),
            size_cat=cat,
            usda_grade=usda,
        )

    def _accept_inspection(self, result):
        """Reject a model result that is not a measured onion in a valid crop."""
        if result is None:
            return False
        if disease_display_label(result.predicted_class, result.confidence) == "UNCERTAIN":
            return False
        if not diameter_is_valid(result.size_mm):
            return False
        image = result.inference_image if result.inference_image is not None else result.image
        return PhysicalOnionTracker.crop_matches_onion(image)

    def _sort_onion_now(self, onion_id, cat=None):
        """Dispatch the stored route once. A category alone cannot move hardware."""
        route = self._batch_flow.inspected.get(onion_id)
        if route is None:
            return
        if cat is not None and route.get("category") != cat:
            return
        if not _HW.sorting_authorized():
            if onion_id not in self._sorting_refused:
                self._sorting_refused.add(onion_id)
                print(f"[SORT] {onion_id} was inspected. Sorting is not authorized. {_HW.MOTOR_DISABLED_REASON}")
                self.statusBar.showMessage(
                    "Sorting is off. The motor wiring is not verified, so the belt and servos stay idle."
                )
            return
        if onion_id in self._dispatch_started:
            return
        self._dispatch_started.add(onion_id)
        self._dispatch_route(route)

    def _dispatch_route(self, route):
        """Ask the actuators to run. The onion stays unsorted until the belt cycle finishes."""
        def work():
            try:
                servo_cmd = route.get("servo")
                if servo_cmd and send_servo_command(servo_cmd, route["onion_id"]) is False:
                    self.physical_route_finished.emit(route["onion_id"], False)
                    return
                if route.get("conveyor"):
                    manager = get_conveyor_manager()
                    manager.on_finished = self._relay_conveyor_result
                    if not manager.enqueue(route["onion_id"]):
                        self.physical_route_finished.emit(route["onion_id"], False)
                    return
                self.physical_route_finished.emit(route["onion_id"], True)
            except Exception as exc:
                print(f"[SORT] {exc}")
                halt_motion()
                self.physical_route_finished.emit(route["onion_id"], False)

        threading.Thread(target=work, daemon=True).start()

    def _relay_conveyor_result(self, onion_id, success, reason):
        if reason and reason != "ok":
            print(f"[CONVEYOR] {onion_id} finished with {reason}")
        self.physical_route_finished.emit(str(onion_id), bool(success))

    def _on_physical_route_finished(self, onion_id, success):
        """Slot progress follows a finished actuator cycle, not the classifier."""
        if not self._batch_flow.mark_physically_processed(onion_id, bool(success)):
            if not success:
                halt_motion()
                print(f"[SORT] {onion_id} was not physically processed.")
                self.statusBar.showMessage(
                    f"Sorting stopped for {onion_id}. The conveyor is off and pending moves were cancelled."
                )
            return
        self.unique_onions_processed = len(self._batch_flow.sorted)
        self._sorted_servo_ids.add(onion_id)
        self.lbl_slot_progress.setText(
            f"Processed: {self.unique_onions_processed} / {self.slot_target_quantity}"
        )
        if self.slot_target_quantity > 0 and self.unique_onions_processed >= self.slot_target_quantity:
            self._stop_operation(completed_normally=True)

    def _size_labels(self, dia_mm: float):
        cat = size_category_for_diameter(dia_mm)
        if cat is None:
            return None, None
        if dia_mm < 45.0:
            usda = "SMALL"
        elif dia_mm >= 70.0:
            usda = "LARGE"
        else:
            usda = "MEDIUM"
        return cat, usda

    def _queue_onion_capture(self, obj_id, bbox, dia_mm, cat, usda, crop_img):
        if crop_img is None:
            return
        if obj_id in self.onion_tracker.counted_ids or obj_id in self._queued_onion_ids:
            return
        if diameter_is_valid(dia_mm):
            crop_img = self._stamp_diameter(crop_img, dia_mm)
        self._queued_onion_ids.add(obj_id)
        self._pending_onion_captures.append((obj_id, bbox, dia_mm, cat, usda, crop_img))
        self._show_capture_preview(crop_img, dia_mm)

    def _stamp_diameter(self, crop_rgb, dia_mm):
        """Write the measured diameter onto the picture kept in the review history."""
        stamped = crop_rgb.copy()
        label = f"Dia: {float(dia_mm):.1f} mm"
        scale = max(0.45, stamped.shape[1] / 900.0)
        origin = (12, int(28 * scale) + 8)
        cv2.putText(stamped, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(stamped, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 220, 0), 1, cv2.LINE_AA)
        return stamped

    def _queue_best_live_onion(self):
        ready = []
        for obj_id, track in self.onion_tracker.active_tracks.items():
            if obj_id in self.onion_tracker.counted_ids or obj_id in self._queued_onion_ids:
                continue
            if track.frames_tracked < 2 or track.best_crop is None:
                continue
            ready.append(track)
        if not ready:
            return
        track = min(ready, key=lambda item: item.min_dist_to_center)
        if not track.diameters:
            return
        dia_mm = float(np.median(track.diameters))
        cat, usda = self._size_labels(dia_mm)
        if cat is None:
            return
        self._queue_onion_capture(track.obj_id, track.best_bbox, dia_mm, cat, usda, track.best_crop)

    def _process_one_onion(self, obj_id, dia_mm, cat, usda, crop_img):
        if not diameter_is_valid(dia_mm) or size_category_for_diameter(dia_mm) is None:
            return
        if inspection_action("valid", dia_mm) != "classify":
            return
        if obj_id in self._queued_onion_ids or obj_id in self.onion_tracker.counted_ids:
            return
        self._queued_onion_ids.add(obj_id)
        self.captured_count += 1
        self.inference_worker.process_capture(
            capture_num=self.captured_count,
            onion_id=f"ONION-{obj_id:03d}",
            frame_rgb=crop_img,
            display_img=crop_img,
            model_name=self.combo_model.currentText(),
            batch_no=self.batch_number,
            size_mm=dia_mm,
            size_cat=cat,
            usda_grade=usda
        )

    def _apply_finalized_counts(self, result):
        """Count this onion once. The disease total and the size total use this same record."""
        counts = {
            "normal": self.normal_count,
            "rotten": self.rotten_count,
            "sprouted": self.sprouted_count,
            "moldy": self.moldy_count,
            "cat1": self.size_cat1,
            "cat2": self.size_cat2,
            "cat3": self.size_cat3,
            "cat4": self.size_cat4,
        }
        category = record_inspection_counts(
            counts,
            self._counted_inspection_ids,
            result.onion_id,
            result.predicted_class,
            result.confidence,
            result.size_mm,
        )
        if category is None:
            return None
        self.normal_count = counts["normal"]
        self.rotten_count = counts["rotten"]
        self.sprouted_count = counts["sprouted"]
        self.moldy_count = counts["moldy"]
        self.size_cat1 = counts["cat1"]
        self.size_cat2 = counts["cat2"]
        self.size_cat3 = counts["cat3"]
        self.size_cat4 = counts["cat4"]
        result.size_category = category
        return category

    def _on_capture_timer_tick(self):
        """Inspection is driven by a validated onion in the current frame."""
        return

    def _on_inspection_completed(self, result: InspectionResult):
        """Callback when inference completes in worker thread."""
        if result is None:
            return
        finalized = set(self._counted_inspection_ids) | set(self._batch_flow.inspected)
        decision = result_disposition(
            getattr(result, "onion_id", None),
            self._submitted_labels,
            finalized,
            self._visible_onion_labels,
        )
        if decision in ("reject", "duplicate"):
            return
        if result.onion_id in self._cancelled_inspections:
            return
        if self._classification_state.get(result.onion_id) == "timed_out":
            return
        locked_diameter = self._submitted_diameter.get(result.onion_id)
        if not same_onion_measurement(result.onion_id, locked_diameter, result.onion_id, result.size_mm):
            return
        if not diameter_is_valid(result.size_mm):
            self._show_diameter_unavailable()
            return
        if disease_display_label(result.predicted_class, result.confidence) == "UNCERTAIN" or not self._accept_inspection(result):
            self._classification_state[result.onion_id] = "uncertain"
            self._disease_shown_for = None
            self._show_uncertain()
            return
        if result.onion_id in self._counted_inspection_ids or result.onion_id in self._batch_flow.inspected:
            return
        if any(saved.onion_id == result.onion_id for saved in self.inspection_history):
            return
        route = self._batch_flow.queue_finalized(result)
        if route is None:
            return
        category = self._apply_finalized_counts(result)
        if category is None:
            return
        result.size_category = category
        self.inspection_history.append(result)
        self.current_history_index = len(self.inspection_history) - 1

        self._update_counter_labels()
        onion_key = result.onion_id
        if route["route"] == "sort":
            self._last_servo_onion = onion_key
        elif route["route"] == "reject":
            self._last_end_onion = onion_key
        self._sort_onion_now(onion_key, category)
        onion_text = str(result.onion_id)
        if "-" in onion_text and onion_text.rsplit("-", 1)[-1].isdigit():
            self.onion_tracker.counted_ids.add(int(onion_text.rsplit("-", 1)[-1]))
        self.lbl_slot_progress.setText(
            f"Processed: {self.unique_onions_processed} / {self.slot_target_quantity}"
        )
        if prediction_belongs_to_visible_onion(
            result.onion_id, self._visible_onion_labels, self._cancelled_inspections
        ):
            self._no_onion_shown = False
            self._browsing_history = False
            self._displayed_onion_id = result.onion_id
            self._disease_shown_for = result.onion_id
            self._classification_state[result.onion_id] = "done"
            self._review_message = ""
            self._display_capture_result(result)
        else:
            self._classification_state[result.onion_id] = "done"
        self._update_button_states()

    # --------------------------------------------------------------------------
    # DISPLAY & COUNTER UPDATES
    # --------------------------------------------------------------------------
    def _show_capture_preview(self, image, dia_mm):
        """Show this onion's current crop. The disease label is applied only after classification."""
        if image is None:
            return
        self._last_capture_frame = image
        pixmap = self._cv_image_to_pixmap(image, self.center_zoom_factor, self.captured_image_label, fill=False)
        self.captured_image_label.setPixmap(pixmap)

    def _display_capture_result(self, result: InspectionResult):
        # This label is the upper area of the second (Analysis & Capture Review) box.
        # Keep the onion image immediately below it.
        if result.predicted_class == "Normal":
            header = (
                f"NORMAL  |  {result.confidence:.1f}%  |  {result.onion_id}"
                f"  |  Dia: {result.size_mm:.1f} mm  |  {result.size_category}"
            )
            self.prediction_text_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #4ade80;")
        elif result.predicted_class in ("Rotten", "Sprouted", "Moldy"):
            header = (
                f"DISEASE  |  {result.predicted_class.upper()}  |  {result.confidence:.1f}%"
                f"  |  {result.onion_id}  |  DIRECT TO END"
            )
            self.prediction_text_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #f87171;")
        else:
            header = "UNCERTAIN"
            self.prediction_text_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #facc15;")
        self.prediction_text_label.setText(header)

        display_img = result.xai_image if result.xai_image is not None else result.image
        self._last_capture_frame = display_img
        pixmap = self._cv_image_to_pixmap(display_img, self.center_zoom_factor, self.captured_image_label, fill=False)
        self.captured_image_label.setPixmap(pixmap)
        self._update_reflectance_graph(result)

    def _measurement_for_result(self, result):
        """Return this onion's curve only. A missing or non-onion image leaves the sample empty."""
        if result is None or disease_display_label(result.predicted_class, result.confidence) == "UNCERTAIN":
            return None, ""
        source = result.inference_image if result.inference_image is not None else result.image
        if reflectance_sample_blocks(source) is None:
            result.reflectance = None
            result.spectral_label = ""
            return None, ""
        if not reflectance_curve_is_valid(result.reflectance):
            try:
                result.reflectance = self.spectrum_engine.mean_curve(source)
            except Exception as exc:
                print(f"[Reflectance] {exc}")
                result.reflectance = None
        if not reflectance_curve_is_valid(result.reflectance):
            result.reflectance = None
            result.spectral_label = ""
            return None, ""
        label = self.reflectance_references.label_for(result.reflectance)
        result.spectral_label = label
        return result.reflectance, label

    def _update_reflectance_graph(self, result=None):
        """Keep the Excel reference fixed, and replace the sample only for a valid measurement."""
        sample, label = self._measurement_for_result(result)
        onion_id = getattr(result, "onion_id", "") if result is not None else ""
        self._draw_reflectance(sample, label, onion_id)

    def _draw_reflectance(self, sample, label, onion_id=""):
        ax = self.reflectance_ax
        ax.clear()
        ax.set_facecolor("#ffffff")
        references = self.reflectance_references
        series = reflectance_plot_series(references.normal, sample)
        curves = []
        for name, values in series:
            if name == "Normal reference":
                ax.plot(
                    references.wavelengths, values,
                    color="#16a34a", linewidth=2.2, label="Normal reference", zorder=3,
                )
            else:
                wavelengths = references.wavelengths
                if values.shape[0] != wavelengths.shape[0]:
                    values = np.interp(
                        np.linspace(0, 1, wavelengths.shape[0]),
                        np.linspace(0, 1, values.shape[0]),
                        values,
                    )
                color = "#dc2626" if label == "ABNORMAL" else "#2563eb"
                ax.plot(wavelengths, values, color=color, linewidth=2.0, label="Current onion", zorder=4)
            curves.append(values)
        ax.set_xlim(380, 780)
        ax.set_xlabel("Wavelength (nm)", fontsize=8, color="#334155")
        ax.set_ylabel("Reflectance", fontsize=8, color="#334155")
        if any(name == "Current onion" for name, _values in series):
            title = f"Current onion {onion_id}".strip()
            if label in ("NORMAL", "ABNORMAL"):
                title = f"{title}: {label}"
            title_color = "#b91c1c" if label == "ABNORMAL" else "#1d4ed8"
        else:
            title = "Normal reference"
            title_color = "#166534"
        ax.set_title(title, fontsize=11, fontweight="bold", color=title_color)
        if curves:
            stacked = np.concatenate([np.asarray(curve, dtype=np.float64).ravel() for curve in curves])
            finite = stacked[np.isfinite(stacked)]
            if finite.size:
                low, high = float(np.min(finite)), float(np.max(finite))
                pad = max(0.02, (high - low) * 0.08)
                ax.set_ylim(low - pad, high + pad)
        ax.tick_params(labelsize=7, colors="#475569")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, frameon=False, loc="upper right")
        self.reflectance_figure.tight_layout()
        self.reflectance_canvas.draw_idle()

    def _update_counter_labels(self):
        self.lbl_count_normal.setText(f"PASS / NORMAL:\nNormal: {self.normal_count}")
        self.lbl_count_rotten.setText(f"ROTTEN:\nRotten: {self.rotten_count}")
        self.lbl_count_sprouted.setText(f"SPROUTED:\nSprouted: {self.sprouted_count}")
        self.lbl_count_moldy.setText(f"MOLDY:\nMoldy: {self.moldy_count}")

        self.lbl_size_cat1.setText(f"Category 1\n< 40 mm\nServo 8\n{self.size_cat1}")
        self.lbl_size_cat2.setText(f"Category 2\n40–50 mm\nServo 7\n{self.size_cat2}")
        self.lbl_size_cat3.setText(f"Category 3\n50–60 mm\nServo 9\n{self.size_cat3}")
        self.lbl_size_cat4.setText(f"Category 4\n> 60 mm\nServo 6\n{self.size_cat4}")

    def _update_button_states(self):
        has_history = len(self.inspection_history) > 0
        self.btn_xai.setEnabled(has_history)
        self.btn_center_zoom_in.setEnabled(has_history)
        self.btn_center_zoom_out.setEnabled(has_history)
        self.btn_prev_capture.setEnabled(has_history and self.current_history_index > 0)
        self.btn_next_capture.setEnabled(has_history and self.current_history_index < len(self.inspection_history) - 1)
        self.btn_restart_history.setEnabled(has_history)

    # --------------------------------------------------------------------------
    # ZOOM CONTROLS
    # --------------------------------------------------------------------------
    def _left_zoom_in(self):
        self.left_zoom_factor = min(3.0, self.left_zoom_factor + 0.2)
        self._refit_displayed_images()

    def _left_zoom_out(self):
        self.left_zoom_factor = max(1.0, self.left_zoom_factor - 0.2)
        self._refit_displayed_images()

    def _center_zoom_in(self):
        self.center_zoom_factor = min(3.0, self.center_zoom_factor + 0.2)
        self._refit_displayed_images()

    def _center_zoom_out(self):
        self.center_zoom_factor = max(1.0, self.center_zoom_factor - 0.2)
        self._refit_displayed_images()

    # --------------------------------------------------------------------------
    # HISTORY NAVIGATION CONTROLS
    # --------------------------------------------------------------------------
    def _navigate_prev(self):
        if self.current_history_index > 0:
            self._browsing_history = True
            self.current_history_index -= 1
            self._is_navigating = True
            self.combo_model.setCurrentText(self.inspection_history[self.current_history_index].model_name)
            self._is_navigating = False
            self._display_capture_result(self.inspection_history[self.current_history_index])
            self._update_button_states()

    def _navigate_next(self):
        if self.current_history_index < len(self.inspection_history) - 1:
            self._browsing_history = True
            self.current_history_index += 1
            self._is_navigating = True
            self.combo_model.setCurrentText(self.inspection_history[self.current_history_index].model_name)
            self._is_navigating = False
            self._display_capture_result(self.inspection_history[self.current_history_index])
            self._update_button_states()

    def _navigate_restart(self):
        """Restarts history review back to Capture #1 without resetting batch/counters."""
        if self.inspection_history:
            self._browsing_history = True
            self.current_history_index = 0
            self._is_navigating = True
            self.combo_model.setCurrentText(self.inspection_history[0].model_name)
            self._is_navigating = False
            self._display_capture_result(self.inspection_history[0])
            self._update_button_states()

    def _lookup_capture_no(self):
        val_str = self.input_cap_no.text().strip()
        if not val_str.isdigit():
            return

        target_no = int(val_str)
        found_idx = -1
        for idx, item in enumerate(self.inspection_history):
            if item.capture_number == target_no:
                found_idx = idx
                break

        if found_idx != -1:
            self._browsing_history = True
            self.current_history_index = found_idx
            self._is_navigating = True
            self.combo_model.setCurrentText(self.inspection_history[found_idx].model_name)
            self._is_navigating = False
            self._display_capture_result(self.inspection_history[found_idx])
            self._update_button_states()
        else:
            QMessageBox.information(self, "Lookup Result", "Capture number not available.")

    # --------------------------------------------------------------------------
    # ON-DEMAND XAI GRAD-CAM TRIGGER
    # --------------------------------------------------------------------------
    def _trigger_xai(self):
        if self.current_history_index < 0 or not self.inspection_history:
            return

        current_item = self.inspection_history[self.current_history_index]
        if disease_display_label(current_item.predicted_class, current_item.confidence) == "UNCERTAIN":
            self.statusBar.showMessage("Status: XAI needs a completed disease prediction.")
            return
        img_to_explain = current_item.inference_image if current_item.inference_image is not None else current_item.image
        if reflectance_sample_blocks(img_to_explain) is None and not PhysicalOnionTracker.crop_matches_onion(img_to_explain):
            self.statusBar.showMessage("Status: XAI needs the captured onion image.")
            return
        current_item.xai_image = None
        self.statusBar.showMessage(
            f"Status: Generating Grad-CAM XAI for {current_item.onion_id} ({current_item.predicted_class})..."
        )
        self.xai_worker.generate_xai(
            capture_num=current_item.capture_number,
            onion_id=current_item.onion_id,
            img_rgb=img_to_explain,
            model_name=current_item.model_name,
            predicted_class=current_item.predicted_class,
        )

    def _on_xai_completed(self, capture_num: int, onion_id: str, predicted_class: str, xai_overlay, request_token: int):
        if request_token != self.xai_worker._request_id:
            self.statusBar.showMessage("Status: Stale XAI result was ignored.")
            return
        if xai_overlay is None:
            self.statusBar.showMessage("Status: XAI explanation was not available for this onion.")
            return
        matched = None
        for item in self.inspection_history:
            if xai_result_matches(
                capture_num, onion_id, predicted_class,
                item.capture_number, item.onion_id, item.predicted_class,
                xai_overlay, request_token, self.xai_worker._request_id,
            ):
                matched = item
                break
        if matched is None:
            self.statusBar.showMessage("Status: Stale XAI result was ignored.")
            return
        matched.xai_image = xai_overlay
        current = None
        if 0 <= self.current_history_index < len(self.inspection_history):
            current = self.inspection_history[self.current_history_index]
        if current is matched:
            self._display_capture_result(current)
        self.statusBar.showMessage(f"Status: Grad-CAM XAI computed for {onion_id}.")

    # --------------------------------------------------------------------------
    # REPORT BUTTON
    # --------------------------------------------------------------------------
    def _on_report_clicked(self):
        records = list(self.inspection_history)
        if not records:
            QMessageBox.warning(self, "No Inspection Data", "Inspect at least one onion before generating a report.")
            return

        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
            from xml.sax.saxutils import escape

            now = datetime.now()
            desktop = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
            desktop_dir = Path(desktop) if desktop else Path.home() / "Desktop"
            desktop_dir.mkdir(parents=True, exist_ok=True)
            safe_batch = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.batch_number).strip("_") or "Batch"
            output = desktop_dir / f"Onion_Quality_Report_{safe_batch}_{now:%Y%m%d_%H%M%S_%f}.pdf"

            navy = colors.HexColor("#12324A")
            teal = colors.HexColor("#0E8276")
            pale = colors.HexColor("#EAF5F2")
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=navy, spaceAfter=7))
            styles.add(ParagraphStyle(name="ReportSection", parent=styles["Heading2"], fontSize=10.5, leading=13, textColor=teal, spaceBefore=13, spaceAfter=6))
            styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"], fontSize=8.8, leading=12, spaceAfter=5))
            styles.add(ParagraphStyle(name="ReportSmall", parent=styles["BodyText"], fontSize=7.5, leading=10))
            styles.add(ParagraphStyle(name="ReportHeader", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.white))
            story = []

            def paragraph(value, style="ReportSmall"):
                return Paragraph(escape(str(value)), styles[style])

            def add_table(rows, widths):
                cells = [[paragraph(value, "ReportHeader" if row_index == 0 else "ReportSmall")
                          for value in row] for row_index, row in enumerate(rows)]
                table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), navy),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D7DC")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(table)

            total = len(records)
            counts = Counter(record.predicted_class for record in records)
            failures = sum(counts.get(label, 0) for label in ("Rotten", "Sprouted", "Moldy"))
            share = lambda n: f"{100 * n / total:.1f}%"
            complete = not self.is_operation_active and total >= self.slot_target_quantity
            status = "Complete" if complete else "In progress / partial"
            report_id = f"{safe_batch}-{now:%Y%m%d%H%M%S}"

            story.append(paragraph("ONION QUALITY ASSESSMENT REPORT", "ReportTitle"))
            story.append(paragraph("Digital inspection report | Results available at the time of export"))
            story.append(Spacer(1, 7 * mm))
            add_table([
                ["Report ID", "Batch number", "Report date and time"],
                [report_id, self.batch_number, now.strftime("%d %b %Y, %H:%M:%S")],
                ["Inspection status", "Model used", "Batch started at"],
                [status, records[0].model_name, self.batch_started_at.strftime("%d %b %Y, %H:%M:%S") if self.batch_started_at else "Not recorded"],
            ], [49 * mm, 49 * mm, 73 * mm])

            story.append(paragraph("01  Batch summary", "ReportSection"))
            summary = [["Inspection result", "Count", "Share of sample"],
                       ["Sample size (inspections completed)", total, "100%"],
                       ["Passed / Normal", counts.get("Normal", 0), share(counts.get("Normal", 0))]]
            for label in ("Rotten", "Sprouted", "Moldy"):
                summary.append([f"Failed - {label}", counts.get(label, 0), share(counts.get(label, 0))])
            summary.append(["Total failed", failures, share(failures)])
            add_table(summary, [87 * mm, 38 * mm, 46 * mm])
            story.append(paragraph(f"Target sample size: {self.slot_target_quantity}. Percentages use {total} completed inspections as the denominator.", "ReportBody"))

            story.append(paragraph("02  Size distribution", "ReportSection"))
            size_counts = Counter(record.size_category for record in records)
            sizes = [["Size category", "Measured diameter", "Count", "Share"]]
            for category, bounds in (("Category 1", "Below 40 mm"), ("Category 2", "40 mm to below 50 mm"),
                                     ("Category 3", "50-60 mm"), ("Category 4", "Above 60 mm")):
                n = size_counts.get(category, 0)
                sizes.append([category, bounds, n, share(n)])
            add_table(sizes, [51 * mm, 51 * mm, 34 * mm, 35 * mm])
            story.append(paragraph("Undersized: Not calculated - configure the buyer's minimum diameter first.", "ReportBody"))

            story.append(paragraph("03  Procurement grade estimate", "ReportSection"))
            add_table([["Outcome", "Count", "Share"],
                       ["Grade A", "Not calculated", "-"],
                       ["URS", "Not calculated", "-"],
                       ["Requires grading rule", total, "100%"]], [87 * mm, 38 * mm, 46 * mm])
            story.append(paragraph("Grade A and URS rules are not configured in this app. Condition and size counts above are measured independently.", "ReportSmall"))

            story.append(paragraph("04  Individual sample results and failures", "ReportSection"))
            story.append(paragraph("One row per completed inspection. Failure type is the predicted condition.", "ReportSmall"))
            detail = [["Sample ID", "Inspected at", "Result / failure", "Confidence", "Diameter", "Size category"]]
            for record in records:
                label = "Pass / Normal" if record.predicted_class == "Normal" else f"Fail / {record.predicted_class}"
                detail.append([record.onion_id, record.timestamp, label,
                               f"{record.confidence:.1f}%", f"{record.size_mm:.1f} mm", record.size_category])
            add_table(detail, [28 * mm, 27 * mm, 49 * mm, 23 * mm, 21 * mm, 23 * mm])
            story.append(Spacer(1, 4 * mm))
            story.append(paragraph("Review note: AI results are estimates from captured images. Low-confidence predictions should be checked manually. The current condition classes are Normal, Rotten, Sprouted, and Moldy; a separate Damaged class is not available.", "ReportSmall"))

            def footer(canvas, document):
                canvas.saveState()
                canvas.setStrokeColor(colors.HexColor("#CBD5DC"))
                canvas.line(19 * mm, 15 * mm, 191 * mm, 15 * mm)
                canvas.setFont("Helvetica", 7)
                canvas.setFillColor(colors.HexColor("#526272"))
                canvas.drawString(19 * mm, 11 * mm, "ONION QUALITY ASSESSMENT")
                canvas.drawRightString(191 * mm, 11 * mm, f"Page {document.page}")
                canvas.restoreState()

            document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=19 * mm, rightMargin=19 * mm,
                                         topMargin=17 * mm, bottomMargin=18 * mm)
            document.build(story, onFirstPage=footer, onLaterPages=footer)
        except Exception as exc:
            QMessageBox.critical(self, "Report Error", f"Could not save the PDF report:\n{exc}\n\nInstall ReportLab if necessary: pip install reportlab")
            return

        self.statusBar.showMessage(f"Report saved: {output}")
        QMessageBox.information(self, "Report Saved", f"PDF report saved to Desktop:\n{output}")

    # --------------------------------------------------------------------------
    # HELPER & CLEANUP METHODS
    # --------------------------------------------------------------------------
    def _label_target_size(self, label):
        rect = label.contentsRect()
        width = rect.width()
        height = rect.height()
        if width < 20 or height < 20:
            width = max(label.width(), label.minimumWidth())
            height = max(label.height(), label.minimumHeight())
        return max(20, width), max(20, height)

    def _refit_displayed_images(self):
        if self._last_live_frame is not None:
            self.live_video_label.setPixmap(
                self._cv_image_to_pixmap(self._last_live_frame, self.left_zoom_factor, self.live_video_label)
            )
        if self._last_capture_frame is not None:
            self.captured_image_label.setPixmap(
                self._cv_image_to_pixmap(
                    self._last_capture_frame, self.center_zoom_factor, self.captured_image_label, fill=False
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout()
        self._refit_displayed_images()

    def _cv_image_to_pixmap(self, img_rgb: np.ndarray, zoom_factor: float = 1.0, target_label=None, fill: bool = False) -> QPixmap:
        img_rgb = np.ascontiguousarray(img_rgb)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

        pixmap = QPixmap.fromImage(q_img)
        if zoom_factor > 1.0:
            crop_w = int(w / zoom_factor)
            crop_h = int(h / zoom_factor)
            x = (w - crop_w) // 2
            y = (h - crop_h) // 2
            pixmap = pixmap.copy(QRect(x, y, crop_w, crop_h))

        if target_label is not None:
            tw, th = self._label_target_size(target_label)
        else:
            tw = max(100, int(360 * getattr(self, 'scale_f', 1.0)))
            th = max(100, int(260 * getattr(self, 'scale_f', 1.0)))
        # Letterbox the frame. KeepAspectRatio never stretches the camera or capture image.
        return pixmap.scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _on_error_occurred(self, err_msg: str):
        print(f"[ERROR] {err_msg}")
        self.statusBar.showMessage(f"Error: {err_msg}")

    def _on_inference_error(self, err_msg: str):
        """A model failure does not invent a disease class and does not start the belt."""
        print(f"[ERROR] Disease classification failed: {err_msg}")
        self.statusBar.showMessage(f"Disease classification failed: {err_msg}")
        target = self._inspection_target_id
        if target and self._classification_state.get(target) == "running":
            self._classification_state[target] = "error"

    def _on_camera_failed(self, err_msg: str):
        """Stop the belt and the camera when a frame cannot be read."""
        message = camera_offline_response(err_msg)["message"]
        print(f"[ERROR] {message}")
        if self.is_operation_active:
            self._stop_operation()
        self.statusBar.showMessage(message)
        QMessageBox.warning(self, "Camera unavailable", message)

    def closeEvent(self, event):
        """Safe application shutdown."""
        self._stop_operation()
        if self.inference_worker.isRunning():
            self.inference_worker.quit()
            self.inference_worker.wait()
        if self.xai_worker.isRunning():
            self.xai_worker.quit()
            self.xai_worker.wait()
        global _servo_manager
        if _servo_manager is not None:
            _servo_manager.cleanup()
        global _conveyor_manager
        if _conveyor_manager is not None:
            _conveyor_manager.cleanup()
        event.accept()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    if "--servo-test" in sys.argv:
        sys.exit(run_attached_servo_diagnostic(sys.argv))
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    window = MainWindow()

    if "--test-mode" in sys.argv:
        print("[TEST MODE] Performing quick automated UI and model verification...")
        window.show()
        QTimer.singleShot(1000, app.quit)
        return app.exec_()

    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
