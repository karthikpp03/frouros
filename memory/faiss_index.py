"""
memory/faiss_index.py
=====================
Manages the FAISS flat inner-product index used for:
  1. Fast duplicate-identity detection before gallery insert.
  2. Embedding-based image retrieval in find_matching_crops().

State:
  _faiss_index  — faiss.IndexFlatIP instance (or None before first build)
  _faiss_id_map — list of reid_ids in FAISS insertion order

All logic is preserved verbatim from the original monolith.
REID_DIM is read from config.settings each time it is needed so that the
value updated by models/reid.py (after its load) is always seen correctly.
"""

import numpy as np
import faiss


# Module-level FAISS state
_faiss_index = None    # faiss.IndexFlatIP
_faiss_id_map = []     # list of reid_ids in FAISS insertion order


def _get_dim():
    """Read REID_DIM from settings at call-time (after reid.py may have updated it)."""
    import config.settings as _cfg
    return _cfg.REID_DIM


def _faiss_rebuild(gallery_data):
    """
    Rebuild the FAISS index from scratch using current gallery_data.
    Called by gallery.py after load and after stale-purge.
    Signature differs from the original monolith because gallery_data is
    passed in explicitly to avoid a circular import.
    """
    global _faiss_index, _faiss_id_map

    _faiss_id_map = []
    dim = _get_dim()

    if not gallery_data:
        _faiss_index = faiss.IndexFlatIP(dim)
        return

    vecs = []
    for rid, data in gallery_data.items():
        emb  = data["embedding"].astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        vecs.append(emb)
        _faiss_id_map.append(rid)

    mat          = np.vstack(vecs).astype(np.float32)
    _faiss_index = faiss.IndexFlatIP(mat.shape[1])
    _faiss_index.add(mat)


def _faiss_add(reid_id, embedding):
    """
    Add a single new embedding to the live FAISS index.
    If the index has not been initialised yet, a rebuild is triggered.
    gallery_data is NOT needed here — only the new vector is added.
    """
    global _faiss_index

    dim = _get_dim()

    if _faiss_index is None:
        _faiss_index = faiss.IndexFlatIP(dim)

    emb  = embedding.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    _faiss_index.add(emb.reshape(1, -1))
    _faiss_id_map.append(reid_id)


def faiss_search(query_embedding, k=1):
    """
    Return [(reid_id, cosine_similarity), ...] for the top-k nearest neighbours.
    Returns [] if the index is empty.
    Identical logic to the original faiss_search().
    """
    if _faiss_index is None or _faiss_index.ntotal == 0:
        return []

    emb  = query_embedding.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    k      = min(k, _faiss_index.ntotal)
    D, I   = _faiss_index.search(emb.reshape(1, -1), k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if 0 <= idx < len(_faiss_id_map):
            results.append((_faiss_id_map[idx], float(dist)))
    return results
