"""
face/recognizer.py
===================
PLACEHOLDER — face recognition is not implemented yet.

This file only exists so services/summary_router.py has something to
call when ENABLE_FACE_RECOGNITION=true. Right now that flag defaults to
false in .env, so this module is not even imported by the router.

HOW TO WIRE UP YOUR REAL MODEL LATER
-------------------------------------
When your trained face-recognition model is ready:
  1. Replace ONLY this file.
  2. Keep the function name, signature, and return contract below
     exactly the same: `recognize_face(frame_paths) -> "known" | "unknown"`.
  3. Nothing else in the project needs to change — summary_router.py,
     event_manager.py, config/settings.py etc. already call this
     function correctly and only need ENABLE_FACE_RECOGNITION=true in
     .env to start using it.

Feel free to load your model as a lazy-initialized module-level
singleton here (same load-once pattern as models/qwen_vl.py), so it's
loaded on first use rather than at import time.
"""


def recognize_face(frame_paths):
    """
    Decide whether the person in this event is a known or unknown face.

    Args:
        frame_paths: the same list of VideoMAE smart-frame image paths
                      the rest of the pipeline already uses for this
                      event (chronological order, exactly 3 frames).

    Returns:
        str — "known" or "unknown".

    PLACEHOLDER BEHAVIOUR: always returns "unknown" until a real model
    is wired in here, so every event safely falls back to the OpenAI
    branch (i.e. gets a full-quality summary) rather than silently
    skipping detail for someone who isn't actually a recognized face.
    Replace this body with real inference when your model is ready.
    """
    return "unknown"
