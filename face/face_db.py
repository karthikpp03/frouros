"""
face/face_db.py
================
Face REGISTRATION utility + local face database (pickle) I/O.

This module is what turns folders of reference photos into the small,
local embedding database that face/recognizer.py loads ONCE at startup
and matches against at runtime. It is completely independent from the
matching/routing logic in face/recognizer.py — this file only ever
builds or reads the database, it never decides anything about an
event.

Detection + embedding: InsightFace's 'buffalo_l' model pack (SCRFD for
detection, ArcFace for the 512-d recognition embedding). Everything
runs fully locally — no external API is ever called.

Input layout expected on disk
------------------------------
    faces/
      Dad/
        1.jpg
        2.jpg
      Mom/
        1.jpg
      Priya/
        1.jpg

Output
------
face_database.pkl (config.settings.FACE_DATABASE_FILE) — a pickled
dict:

    {
      "Dad":   {"embedding": np.ndarray(512,), "registered_at": "2026-07-11T12:00:00"},
      "Mom":   {"embedding": np.ndarray(512,), "registered_at": "2026-07-11T12:00:00"},
      "Priya": {"embedding": np.ndarray(512,), "registered_at": "2026-07-11T12:00:00"},
    }

Each person's embedding is the average of their per-photo ArcFace
embeddings, L2-re-normalized.

Usage
-----
Build (or rebuild) the database whenever reference photos change —
this is a deliberately OFFLINE / on-demand step, never run
automatically at application startup (startup only ever READS this
file once — see face/recognizer.py.load_face_recognizer()):

    python -m face.face_db
    # or
    from face.face_db import build_face_database
    build_face_database()
"""

import os
import pickle
from datetime import datetime

import cv2
import numpy as np

from config.settings import FACES_DIR, FACE_DATABASE_FILE
from config.settings import INSIGHTFACE_HOME
from utils.device import DEVICE
import os

os.makedirs(INSIGHTFACE_HOME, exist_ok=True)
_SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _load_insightface_app():
    """
    Construct an InsightFace FaceAnalysis app — SCRFD for detection,
    ArcFace for the recognition embedding, via the 'buffalo_l' model
    pack — on whichever device utils/device.py resolved (never a
    hardcoded "cuda"). Only imported here, so registration is the only
    path that needs InsightFace loaded for this purpose.
    """
    from insightface.app import FaceAnalysis

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if DEVICE.type == "cuda"
        else ["CPUExecutionProvider"]
    )
    app = FaceAnalysis(
    name="buffalo_l",
    root=INSIGHTFACE_HOME,
    providers=providers,
)
    app.prepare(ctx_id=0 if DEVICE.type == "cuda" else -1)
    return app


def build_face_database(faces_dir: str = FACES_DIR, output_path: str = FACE_DATABASE_FILE) -> dict:
    """
    Scan `faces_dir/<person_name>/*.jpg` (etc.), detect the face (SCRFD)
    and generate an ArcFace embedding for every reference photo, and
    store ONE averaged, L2-normalized embedding per person, plus a
    registration timestamp, to `output_path` as a pickle file.

    Multiple faces detected in the same photo are handled by keeping
    only the largest bounding box (assumed to be the intended
    subject). Photos with no detected face, or that fail to load, are
    skipped with a warning rather than aborting the whole run.

    Returns:
        dict — {person_name: {"embedding": np.ndarray, "registered_at": str}}
        — the same content written to output_path.
    """
    if not os.path.isdir(faces_dir):
        print(f"[FACE] No faces directory found at '{faces_dir}' — "
              f"nothing to register. Create faces/<person_name>/*.jpg "
              f"folders and re-run.")
        return {}

    print("[FACE] Loading InsightFace (SCRFD + ArcFace) for registration...")
    app = _load_insightface_app()
    print("[FACE] InsightFace loaded.")

    database: dict = {}
    registered_at = datetime.now().isoformat(timespec="seconds")

    for person_name in sorted(os.listdir(faces_dir)):
        person_dir = os.path.join(faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith(_SUPPORTED_EXTS):
                continue

            img_path = os.path.join(person_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[FACE] WARNING — could not read '{img_path}', skipping.")
                continue

            faces = app.get(img)
            if not faces:
                print(f"[FACE] WARNING — no face detected in '{img_path}', skipping.")
                continue

            # If more than one face is in the reference photo, assume
            # the largest bounding box is the intended subject.
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            embeddings.append(face.normed_embedding)

        if not embeddings:
            print(f"[FACE] WARNING — no usable embeddings for '{person_name}', "
                  f"skipping this person entirely.")
            continue

        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = (avg_embedding / np.linalg.norm(avg_embedding)).astype(np.float32)

        database[person_name] = {
            "embedding": avg_embedding,
            "registered_at": registered_at,
        }
        print(f"[FACE] Registered '{person_name}' from {len(embeddings)} image(s).")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(database, f)

    print(f"[FACE] Face database saved -> '{output_path}' "
          f"({len(database)} person(s) registered).")
    return database


def load_face_database(path: str = FACE_DATABASE_FILE) -> dict:
    """
    Read the pre-built face database (pickle) from disk. Pure I/O —
    does NOT import or load InsightFace, and does not regenerate
    anything; it only unpickles the file build_face_database() already
    produced.

    face/recognizer.py calls this exactly ONCE per process (the result
    is cached in a module-level singleton there) — this function
    itself is stateless and safe to call more than once if ever
    needed.

    Returns:
        dict — {person_name: {"embedding": np.ndarray, "registered_at": str}},
        empty if no database file exists yet.
    """
    if not os.path.exists(path):
        print(f"[FACE] No face database found at '{path}' — "
              f"run `python -m face.face_db` to build one. Every event "
              f"will be treated as Unknown until then.")
        return {}

    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # `python -m face.face_db` — rebuild the database from faces/<name>/
    # folders. Not run automatically by the main application.
    build_face_database()