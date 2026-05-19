"""
models/yolo_detector.py
=======================
Loads the YOLO model and exposes a single track() wrapper used by the
main loop.  The tracker yaml (botsort.yaml / strongsort.yaml) is kept
here so the main loop never needs to import ultralytics directly.
"""

from ultralytics import YOLO
from config.settings import YOLO_MODEL_PATH


def load_yolo():
    """Load and return the YOLO model. Called once at startup."""
    model = YOLO(YOLO_MODEL_PATH)
    return model


def run_tracking(model, frame):
    """
    Run YOLO tracking on a single frame.
    Returns the results list exactly as model.track() does.
    Tracker is botsort.yaml — unchanged from v4 main loop.
    """
    return model.track(frame, persist=True, tracker="botsort.yaml", classes=[0])
