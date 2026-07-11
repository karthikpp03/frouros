"""
face/recognizer.py
===================
Face Recognition module — a completely independent unit whose ONLY job
is to answer: "who is this, and how confident are you?"

Detection + embedding: InsightFace's 'buffalo_l' model pack — SCRFD for
face detection, ArcFace for the 512-d recognition embedding. Runs
entirely locally; no external API is ever called.

Contract
--------
    recognize_face(frame_paths) -> (person_name: str, confidence: float)

  person_name : the real, registered identity (e.g. "Dad", "Mom",
                "Priya" — whatever folder name was used under faces/),
                or the literal string "Unknown". NEVER a placeholder
                like "Known" / "Known Person" / "Matched Person" /
                "Recognized Person".
  confidence  : float in [0.0, 100.0] — how confident the match is.

Nothing else is ever returned. This module does NOT decide routing —
services/summary_router.py reads (person_name, confidence) and decides
Known -> Qwen / Unknown -> OpenAI itself; that logic is not duplicated
or moved here.

ENABLE_FACE_RECOGNITION=false
------------------------------
When the flag is false, this module never imports InsightFace, never
loads the face database, never generates an embedding, and never
compares anything — recognize_face() returns ("Unknown", 0.0)
immediately, and load_face_recognizer() (the startup hook) is a no-op.
The rest of the pipeline continues exactly as it works today.

ENABLE_FACE_RECOGNITION=true
------------------------------
src/main.py calls load_face_recognizer() exactly ONCE at startup. That
loads the InsightFace model and the face database (see face/face_db.py)
into module-level singletons and never reloads either again during
runtime — every recognize_face() call afterwards reuses them.
"""

import os

import numpy as np
import cv2

from config.settings import (
    ENABLE_FACE_RECOGNITION,
    USE_OPENAI,
    FACE_RECOGNITION_THRESHOLD,
    INSIGHTFACE_HOME,
)
from utils.device import DEVICE

# Module-level singletons — populated exactly once by
# load_face_recognizer(), never reloaded during runtime.
_face_app = None   # InsightFace FaceAnalysis instance (SCRFD + ArcFace)
_face_db  = None   # {person_name: {"embedding": np.ndarray, "registered_at": str}}

# Face recognition only ever matters on the branch where the router
# can actually choose between Qwen and OpenAI based on identity (see
# services/summary_router.py) — mirrors the router's own condition
# exactly, so nothing is ever loaded for a config where it could never
# be used.
_ENABLED = USE_OPENAI and ENABLE_FACE_RECOGNITION


def load_face_recognizer() -> None:
    """
    One-time startup hook — call this exactly once from src/main.py.

    When face recognition is disabled (ENABLE_FACE_RECOGNITION=false,
    or USE_OPENAI=false so face recognition could never be reached by
    the router anyway), this is a complete no-op: InsightFace is never
    imported, the face database is never read, no embedding is ever
    generated.

    When enabled, loads the InsightFace model and the face database
    once and caches both in module-level singletons — never reloaded
    during runtime.
    """
    global _face_app, _face_db

    if not _ENABLED:
        return

    if _face_app is not None and _face_db is not None:
        return  # already loaded — never reload during runtime

    print("Loading InsightFace...")
    from insightface.app import FaceAnalysis

    # Must exist BEFORE FaceAnalysis() is constructed — InsightFace
    # resolves its model path at construction time, and if
    # INSIGHTFACE_HOME doesn't exist yet on a fresh runtime, it can
    # fall back to its own default cache dir (~/.insightface) instead
    # of the project directory. Registration (face/face_db.py) creates
    # this directory at import time, but that import used to happen
    # only *after* FaceAnalysis() ran here — too late for the runtime
    # path. Always create it right here, immediately before use.
    os.makedirs(INSIGHTFACE_HOME, exist_ok=True)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if DEVICE.type == "cuda"
        else ["CPUExecutionProvider"]
    )
    _face_app = FaceAnalysis(
        name="buffalo_l",
        root=INSIGHTFACE_HOME,
        providers=providers,
    )
    _face_app.prepare(ctx_id=0 if DEVICE.type == "cuda" else -1)

    print("Loading Face Database...")
    from face.face_db import load_face_database
    _face_db = load_face_database()

    print(f"Registered Faces : {len(_face_db)}")
    print("Face Recognition Ready.")


def _best_match(embedding: np.ndarray):
    """
    Compare one detected-face embedding against every registered
    embedding (cosine similarity — ArcFace embeddings are already
    L2-normalized, so this is just a dot product) and return the best
    (person_name, similarity) pair, or (None, 0.0) if the database is
    empty.
    """
    if not _face_db:
        return None, 0.0

    best_name, best_sim = None, -1.0
    for person_name, record in _face_db.items():
        sim = float(np.dot(embedding, record["embedding"]))
        if sim > best_sim:
            best_name, best_sim = person_name, sim

    return best_name, best_sim


def recognize_face(frame_paths):
    """
    Decide who (if anyone) the person in this event is, using the
    InsightFace model and face database loaded once by
    load_face_recognizer().

    Args:
        frame_paths: the same list of VideoMAE smart-frame image paths
                      the rest of the pipeline already uses for this
                      event (chronological order, exactly 3 frames).

    Returns:
        (person_name, confidence) tuple.
          person_name: str — the recognized person's real name (e.g.
                       "Dad", "Mom", "Priya"), or "Unknown" if no
                       registered face matched with sufficient
                       confidence. Never a placeholder.
          confidence:  float — 0.0-100.0, the best match's confidence
                       score.
    """
    if not _ENABLED:
        return "Unknown", 0.0

    if _face_app is None or _face_db is None:
        raise RuntimeError(
            "Face recognition is enabled but not loaded. Call "
            "load_face_recognizer() once at startup (see src/main.py) "
            "before recognize_face() — it is never loaded lazily/"
            "per-call, only once at startup."
        )

    if not _face_db:
        # No one registered yet — every event is Unknown until
        # face/face_db.build_face_database() has been run.
        return "Unknown", 0.0

    best_name, best_sim = None, -1.0

    for path in frame_paths or []:
        img = cv2.imread(path)
        if img is None:
            continue

        detected = _face_app.get(img)
        if not detected:
            continue

        # Largest detected face per frame — the ROI is already scoped
        # to one subject by the rest of the pipeline, so this simply
        # guards against a stray background face winning by accident.
        face = max(
            detected,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        name, sim = _best_match(face.normed_embedding)
        if name is not None and sim > best_sim:
            best_name, best_sim = name, sim

    confidence = max(best_sim, 0.0) * 100.0

    if best_name is None or best_sim < FACE_RECOGNITION_THRESHOLD:
        return "Unknown", confidence

    return best_name, confidence