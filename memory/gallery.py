"""
memory/gallery.py
=================
Manages the persistent ReID identity gallery (reid_gallery.json) and
the in-memory gallery_data dict.

Exports:
  gallery_data          — dict: rid → {embedding, count, last_seen, first_seen}
  gallery_counter       — int, monotonically increasing identity counter
  gallery_load()
  gallery_save()
  gallery_match(embedding)   → (reid_id, similarity)
  gallery_update(reid_id, embedding)
  gallery_was_recent(reid_id) → bool
  gallery_purge_stale()

All logic is preserved verbatim from the original monolith.
FAISS operations are delegated to memory/faiss_index.py.
"""

import os
import json
import time
import numpy as np
from datetime import datetime

from config.settings import (
    REID_GALLERY_FILE,
    REID_SIMILARITY_THRESHOLD,
    REID_GRACE_PERIOD,
    REID_STALE_TIMEOUT,
)
from memory.faiss_index import _faiss_rebuild, _faiss_add, faiss_search


# Module-level mutable state
gallery_data    = {}   # rid → {embedding, count, last_seen, first_seen}
gallery_counter = 0


# --------------------------------------------------
def gallery_load():
    """Load gallery from disk; rebuild FAISS index."""
    global gallery_data, gallery_counter

    if not os.path.exists(REID_GALLERY_FILE):
        _faiss_rebuild(gallery_data)
        return

    with open(REID_GALLERY_FILE, "r") as f:
        raw = json.load(f)

    for rid, data in raw.items():
        gallery_data[rid] = {
            "embedding":  np.array(data["embedding"], dtype=np.float32),
            "count":      data["count"],
            "last_seen":  data["last_seen"],
            "first_seen": data["first_seen"],
        }

    if gallery_data:
        nums = [int(k.split("_")[1]) for k in gallery_data if "_" in k]
        gallery_counter = max(nums) if nums else 0

    _faiss_rebuild(gallery_data)

    import config.settings as _cfg
    print(f"[ReID] Gallery loaded: {len(gallery_data)} identities, FAISS dim={_cfg.REID_DIM}")


def gallery_save():
    """Serialise gallery_data to disk."""
    serializable = {}
    for rid, data in gallery_data.items():
        serializable[rid] = {
            "embedding":  data["embedding"].tolist(),
            "count":      data["count"],
            "last_seen":  data["last_seen"],
            "first_seen": data["first_seen"],
        }
    with open(REID_GALLERY_FILE, "w") as f:
        json.dump(serializable, f)


def gallery_match(embedding):
    """
    FAISS cosine search → return (reid_id, similarity).
    Creates a new identity if no match above threshold.
    Identical logic to the original gallery_match().
    """
    global gallery_counter

    hits = faiss_search(embedding, k=1)

    if hits:
        best_id, best_sim = hits[0]
        if best_sim >= REID_SIMILARITY_THRESHOLD:
            gallery_update(best_id, embedding)
            return best_id, best_sim

    # No match above threshold — new identity
    gallery_counter += 1
    rid = f"person_{gallery_counter:04d}"
    gallery_data[rid] = {
        "embedding":  embedding.copy(),
        "count":      1,
        "last_seen":  time.time(),
        "first_seen": datetime.now().isoformat(),
    }
    _faiss_add(rid, embedding)
    return rid, 0.0


def gallery_update(reid_id, embedding):
    """EMA update of stored embedding; no FAISS in-place update needed."""
    if reid_id not in gallery_data:
        return
    data  = gallery_data[reid_id]
    alpha = 0.1
    data["embedding"] = (1 - alpha) * data["embedding"] + alpha * embedding
    norm = np.linalg.norm(data["embedding"])
    if norm > 0:
        data["embedding"] /= norm
    data["count"]    += 1
    data["last_seen"] = time.time()


def gallery_was_recent(reid_id):
    """Return True if reid_id was seen within REID_GRACE_PERIOD seconds."""
    if reid_id not in gallery_data:
        return False
    return (time.time() - gallery_data[reid_id]["last_seen"]) < REID_GRACE_PERIOD


def gallery_purge_stale():
    """Remove identities not seen for REID_STALE_TIMEOUT seconds; rebuild FAISS."""
    now   = time.time()
    stale = [
        rid for rid, data in gallery_data.items()
        if (now - data["last_seen"]) > REID_STALE_TIMEOUT
    ]
    for rid in stale:
        del gallery_data[rid]
    if stale:
        print(f"[ReID] Purged {len(stale)} stale identities.")
        _faiss_rebuild(gallery_data)
