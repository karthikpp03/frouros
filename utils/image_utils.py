"""
utils/image_utils.py
====================
Manages the best-frame tracker used for snapshot selection during an event.

State:
  best_frame  — np.ndarray | None
  best_score  — float

Exports:
  reset_best_frame()
  try_update_best_frame(frame, score) → bool   (True if updated)
  get_best_frame()                    → np.ndarray | None

Keeping these as module-level state (rather than in the main loop) makes
the main loop's per-detection block cleaner without changing any logic.
"""

# Module-level mutable state — mirrors the original globals
best_frame = None
best_score = 0


def reset_best_frame():
    """Reset to initial state at the start / end of each event."""
    global best_frame, best_score
    best_frame = None
    best_score = 0


def try_update_best_frame(frame, score):
    """
    Replace best_frame if score is higher.
    Returns True when updated.
    """
    global best_frame, best_score
    if score > best_score:
        best_score = score
        best_frame = frame.copy()
        return True
    return False


def get_best_frame():
    """Return current best_frame (may be None before first detection)."""
    return best_frame
