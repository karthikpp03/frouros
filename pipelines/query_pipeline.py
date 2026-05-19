"""
pipelines/query_pipeline.py
============================
Operator query processing:

  classify_intent(query)                            → list[str]
  build_structured_context(memory, query, intents)  → str
  find_matching_crops(query, memory, query_emb)     → [(path, desc)]
  query_memory(query, query_embedding=None)         → (answer_str, [img_paths])

All logic is preserved verbatim from the original monolith.
Intent patterns and the Llama system prompt live in prompts/query_prompts.py.
"""

import os
import re
from datetime import datetime

from config.settings import REID_SIMILARITY_THRESHOLD
from memory.event_memory import load_memory
from memory.faiss_index  import faiss_search, _faiss_index
from models.groq_query_engine import _llama_infer
from prompts.query_prompts import LLAMA_SYSTEM_PROMPT, INTENT_PATTERNS


# --------------------------------------------------
# INTENT CLASSIFIER  (unchanged)
# --------------------------------------------------

def classify_intent(query):
    q       = query.lower()
    matched = []
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q):
                matched.append(intent)
                break
    return matched or ["general"]


# --------------------------------------------------
# TIME HELPERS  (unchanged)
# --------------------------------------------------

def _parse_hour(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").hour
    except Exception:
        return None


def _time_matches_query(hour, query):
    q = query.lower()
    m = re.search(r"after (\d{1,2})\s*(am|pm)?", q)
    if m:
        h = int(m.group(1))
        if m.group(2) == "pm" and h < 12: h += 12
        return hour >= h
    m = re.search(r"before (\d{1,2})\s*(am|pm)?", q)
    if m:
        h = int(m.group(1))
        if m.group(2) == "pm" and h < 12: h += 12
        return hour < h
    if "at night"       in q: return hour >= 20 or hour < 6
    if "in the morning" in q: return 6 <= hour < 12
    if "in the evening" in q: return 17 <= hour < 21
    return True


# --------------------------------------------------
# STRUCTURED CONTEXT BUILDER  (unchanged)
# --------------------------------------------------

def build_structured_context(memory, query, intents):
    if not memory:
        return "No events recorded yet."

    parts = []
    for event in memory:
        ts      = event.get("timestamp", "unknown")
        eid     = event.get("event_id",  "?")
        dur     = event.get("duration",  0)
        summary = event.get("summary",   "")
        persons = event.get("persons",   [])

        if "time_query" in intents:
            hour = _parse_hour(ts)
            if hour is not None and not _time_matches_query(hour, query):
                continue

        s = f"[Event #{eid} | {ts} | {dur}s]\nSummary: {summary}\n"
        for p in persons:
            pid    = p.get("person_id", "?")
            appear = p.get("appearance") or "unknown"
            acts   = ", ".join(p.get("actions", [])) or "none"
            objs   = ", ".join(p.get("objects", [])) or "none"
            move   = p.get("movement") or "unknown"
            wait   = "yes" if p.get("waiting") else "no"
            crop   = p.get("crop_image")
            reid   = p.get("reid_id", "?")

            s += (
                f"  Person {pid} [reid={reid}]: appearance='{appear}', "
                f"actions=[{acts}], objects=[{objs}], "
                f"movement='{move}', waiting={wait}"
            )
            if crop:
                s += f", [has_image={crop}]"
            s += "\n"
        parts.append(s)

    return "\n".join(parts) if parts else "No matching events found."


# --------------------------------------------------
# IMAGE RETRIEVAL — FAISS + keyword fallback  (unchanged)
# --------------------------------------------------

def find_matching_crops(query, memory, query_embedding=None):
    """
    Returns [(crop_path, description), ...] top-3 matches.
    Uses FAISS cosine search when query_embedding provided,
    keyword scoring otherwise.
    Identical to the original find_matching_crops().
    """
    q       = query.lower()
    results = []

    # ---- FAISS path ----
    if query_embedding is not None and _faiss_index is not None \
            and _faiss_index.ntotal > 0:
        hits     = faiss_search(query_embedding, k=10)
        hit_rids = {rid for rid, sim in hits
                    if sim >= REID_SIMILARITY_THRESHOLD * 0.8}

        for event in memory:
            for p in event.get("persons", []):
                crop    = p.get("crop_image")
                reid_id = p.get("reid_id")
                if not crop or not os.path.exists(crop):
                    continue
                if reid_id in hit_rids:
                    ts   = event.get("timestamp", "unknown")
                    desc = (
                        f"Event #{event['event_id']} @ {ts} | "
                        f"{p.get('appearance','?')} | reid={reid_id}"
                    )
                    results.append((crop, desc, 10.0))

    # ---- keyword scoring path (always runs as supplement) ----
    seen_crops = {r[0] for r in results}
    for event in memory:
        for p in event.get("persons", []):
            crop = p.get("crop_image")
            if not crop or not os.path.exists(crop) or crop in seen_crops:
                continue

            score  = 0
            appear = (p.get("appearance") or "").lower()
            acts   = " ".join(p.get("actions", [])).lower()
            objs   = " ".join(p.get("objects", [])).lower()

            for word in q.split():
                if word in appear: score += 3
                if word in acts:   score += 2
                if word in objs:   score += 2

            for obj in ["bag", "phone", "backpack", "umbrella",
                        "helmet", "laptop", "bottle"]:
                if obj in q and obj in objs:
                    score += 5

            if score > 0:
                ts   = event.get("timestamp", "unknown")
                desc = (
                    f"Event #{event['event_id']} @ {ts} | "
                    f"{p.get('appearance','?')} | "
                    f"actions: {', '.join(p.get('actions',[])) or 'none'}"
                )
                results.append((crop, desc, score))

    results.sort(key=lambda x: x[2], reverse=True)
    return [(path, desc) for path, desc, _ in results[:3]]


# --------------------------------------------------
# MAIN QUERY ENTRY POINT  (unchanged)
# --------------------------------------------------

def query_memory(query, query_embedding=None):
    """
    Full query pipeline: classify → build context → Llama inference.
    Returns (answer_str, [img_paths]).
    Identical to the original query_memory().
    """
    memory      = load_memory()
    intents     = classify_intent(query)
    context     = build_structured_context(memory, query, intents)
    image_paths = []

    if "image_request" in intents:
        crops       = find_matching_crops(query, memory, query_embedding)
        image_paths = [c[0] for c in crops]

    intent_note = ", ".join(intents)

    user_prompt = (
        f"Query intent detected: {intent_note}\n\n"
        "=== SURVEILLANCE MEMORY ===\n"
        f"{context}\n"
        "=== END OF MEMORY ===\n\n"
        f"Operator question: {query}"
    )

    output = _llama_infer(LLAMA_SYSTEM_PROMPT, user_prompt, max_new_tokens=300)
    return output, image_paths
