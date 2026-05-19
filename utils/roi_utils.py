"""
utils/roi_utils.py
==================
ROI polygon utilities.
The polygon itself lives in config/settings.py (ROI_POINTS).
This module exposes a convenience wrapper used by the main loop.
"""

import cv2
from config.settings import ROI_POINTS


def is_inside_roi(cx, cy):
    """
    Return True if the point (cx, cy) is inside or on the ROI polygon.
    Wraps cv2.pointPolygonTest — identical behaviour to the inline call
    in the original main loop.
    """
    result = cv2.pointPolygonTest(ROI_POINTS, (cx, cy), False)
    return result >= 0


def draw_roi(frame):
    """Draw the ROI polygon on a frame in-place and return the frame."""
    cv2.polylines(frame, [ROI_POINTS], True, (255, 0, 0), 2)
    return frame
