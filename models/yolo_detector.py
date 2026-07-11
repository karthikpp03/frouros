"""
models/yolo_detector.py
=======================
Loads the YOLO model and exposes a single track() wrapper used by the
main loop.  The tracker yaml (botsort.yaml / strongsort.yaml) is kept
here so the main loop never needs to import ultralytics directly.

Device selection is never hardcoded — YOLO runs on whichever device
utils/device.py resolved at startup (CUDA if available, else CPU),
same as every other model in the project.
"""

from ultralytics import YOLO
from config.settings import YOLO_MODEL_PATH
from utils.device import DEVICE


def load_yolo():
    """Load and return the YOLO model. Called once at startup."""
    model = YOLO(YOLO_MODEL_PATH)
    return model


def run_tracking(model, frame):
    """
    Run YOLO tracking on a single frame.
    Returns the results list exactly as model.track() does.
    Tracker is botsort.yaml — unchanged from v4 main loop.
    Device is passed explicitly (str(DEVICE) -> "cuda" or "cpu") instead
    of relying on ultralytics' own auto-detection, so the whole project
    shares one single source of truth for device selection.
    """
    return model.track(
        frame, persist=True, tracker="botsort.yaml", classes=[0],
        device=str(DEVICE),
    )
