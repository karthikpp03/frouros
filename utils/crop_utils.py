"""
utils/crop_utils.py
===================
Person crop manager.

State:
  best_crops  — dict: reid_id → {frame, score, track_id}

Exports:
  crop_update(frame, track_id, reid_id, x1, y1, x2, y2, confidence)
  crop_save_by_reid(reid_id, ev_id, person_index) → path | None
  crop_clear()

All logic preserved verbatim from the original monolith.
Keyed by reid_id (not track_id) to prevent duplicate crops on track splits.
"""

import cv2
from config.settings import PERSON_CROPS_DIR

# Module-level mutable state
best_crops = {}   # reid_id → {frame, score, track_id}


def crop_update(frame, track_id, reid_id, x1, y1, x2, y2, confidence):
    """
    Keep only the highest-scoring crop per persistent reid_id.
    Identical logic to the original crop_update().
    """
    w     = x2 - x1
    h     = y2 - y1
    area  = w * h
    score = area * confidence

    if reid_id not in best_crops or score > best_crops[reid_id]["score"]:
        H, W = frame.shape[:2]
        px1  = max(0, x1 - 10)
        py1  = max(0, y1 - 10)
        px2  = min(W, x2 + 10)
        py2  = min(H, y2 + 10)
        crop = frame[py1:py2, px1:px2].copy()
        best_crops[reid_id] = {
            "frame":    crop,
            "score":    score,
            "track_id": track_id,
        }


def crop_save_by_reid(reid_id, ev_id, person_index):
    """Save best crop for a reid_id.  Returns saved path or None."""
    if reid_id not in best_crops:
        return None
    crop     = best_crops[reid_id]["frame"]
    filename = f"{PERSON_CROPS_DIR}/event{ev_id}_person{person_index}_{reid_id}.jpg"
    cv2.imwrite(filename, crop)
    return filename


def crop_clear():
    """Reset best_crops at the end of an event."""
    best_crops.clear()
