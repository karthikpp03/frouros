"""
utils/debug_utils.py
====================
Debug utilities for rejected detections.

Exports:
  save_rejected_detection(frame, frame_index, track_id, confidence, area, x1, y1, x2, y2)

Logic is preserved verbatim from the original main loop's inline debug block.
"""

import os
import cv2
from config.settings import DEBUG_DIR


def save_rejected_detection(frame, frame_index, track_id, confidence, area,
                             x1, y1, x2, y2):
    """
    Draw a red bounding box with a REJECTED label and save to debug_rejected/.
    Called whenever a detection fails the area/confidence filter.
    Identical to the original inline debug block.
    """
    debug_frame = frame.copy()

    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    label = f"REJECTED | ID:{track_id} conf:{confidence:.2f} area:{area}"
    cv2.putText(
        debug_frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(
        f"{DEBUG_DIR}/frame_{frame_index}_id_{track_id}.jpg",
        debug_frame,
    )
