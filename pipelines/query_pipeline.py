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
from datetime import datetime, timedelta

from config.settings import REID_SIMILARITY_THRESHOLD
from memory.event_memory import load_memory
from memory.faiss_index  import faiss_search, _faiss_index
from models.groq_query_engine import _llama_infer
from prompts.query_prompts import LLAMA_SYSTEM_PROMPT, INTENT_PATTERNS

# Phase 2 — Telegram/Groq now answers from SQLite instead of the JSON
# event memory file. Per the integration spec: Telegram/Groq must NEVER
# call OpenAI Vision, never re-analyze images, and never regenerate
# summaries — it only reads whatever is already in the `events` /
# `persons` / `movement` / `actions` / `objects` / `vehicles` /
# `keywords` tables (written once, at event-close time, by
# services/db_writer.py) and asks Groq to phrase an answer from that.
#
# find_matching_crops() below still reads memory/event_memory.py (JSON)
# + the FAISS ReID index for photo lookups — that is a pre-existing,
# working feature unrelated to this integration (crop-image retrieval,
# not summary generation) and is left untouched.
#
# All SQLite reads go through pipelines/retrieval.py's named helper
# functions (get_event_by_id, get_events_by_date, get_unknown_visitors,
# get_delivery_events, get_vehicle_events, get_parcel_events,
# get_keyword_matches, ...) — this module never opens a DatabaseManager
# or runs SQL directly, and neither does telegram/bot.py.
from pipelines import retrieval
from utils.event_logger import log_block

# Objects worth trying as a `keywords` table lookup before falling back
# to "give me every recent event" — keeps Telegram/Groq's SQLite reads
# narrow ("only the relevant records") instead of dumping the whole DB.
_KNOWN_OBJECT_KEYWORDS = [
    "bag", "backpack", "phone", "umbrella", "bottle",
    "helmet", "laptop", "box", "parcel", "package",
]

# Hard cap on how many events we ever pull back for a single query, so
# context sent to Groq stays small regardless of how large the DB gets.
_MAX_EVENTS_PER_QUERY = 25


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
# PERSON NAME DETECTION  (Task 9 — person-aware retrieval)
# --------------------------------------------------

def _detect_person_name(query):
    """
    Detect a registered person's name mentioned in the operator's
    question (e.g. "Dad", "test_1"), matched as a whole word,
    case-insensitively, against every name currently registered in
    SQLite (pipelines.retrieval.get_known_person_names() — populated
    from face/face_db.py registrations, never a placeholder).

    Returns:
        str | None — the exact stored name (correct casing) if found,
        else None. "Unknown" is intentionally not matched here — "show
        unknown visitors" style questions are already handled by the
        existing get_unknown_visitors() branch below.
    """
    names = retrieval.get_known_person_names()
    if not names:
        return None

    q = query.lower()
    for name in names:
        if re.search(rf"\b{re.escape(name.lower())}\b", q):
            return name
    return None


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
# SQLITE RETRIEVAL  (Phase 2 — Telegram/Groq's only data source)
# --------------------------------------------------

def _relevant_events_from_db(query, intents):
    """
    Python-side retrieval step in the Telegram flow:
        Groq (intent) -> Python retrieves ONLY relevant records via
        pipelines/retrieval.py's named helpers -> Groq (answer).

    Dispatches to the narrowest matching helper — a specific event id,
    a date, a category (unknown visitors / delivery / vehicle /
    parcel), or a named keyword — before ever falling back to "most
    recent events". Never fetches the whole database, and never opens
    a DatabaseManager or runs SQL directly (that only happens inside
    pipelines/retrieval.py).

    Returns: list[database.models.Event]
    """
    q = query.lower()

    # 0) Person-aware retrieval — the MOST specific possible filter,
    # checked before every other category below. If the question names
    # a registered person (e.g. "Dad", "test_1"), every subsequent
    # lookup is narrowed to ONLY that person's events — other people's
    # events (Mom, Unknown, Priya, ...) are never retrieved, and Groq
    # never sees them. See _detect_person_name() above and
    # pipelines/retrieval.py's get_events_by_person* helpers.
    person_name = _detect_person_name(query)
    if person_name:
        if "yesterday" in q:
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            hits = retrieval.get_events_by_person_and_date(person_name, yesterday)
            if hits:
                return hits

        for kw in _KNOWN_OBJECT_KEYWORDS:
            if kw in q:
                hits = retrieval.get_events_by_person_and_keyword(person_name, kw)
                if hits:
                    return hits

        hits = retrieval.get_events_by_person(person_name)
        if hits:
            return hits[:_MAX_EVENTS_PER_QUERY]

        # A registered person was named but has zero events on record —
        # return empty rather than silently falling through to another
        # person's events (e.g. "most recent events" below).
        return []

    # 1) A specific event id — "explain event 5", "what happened in
    #    event #12" — is the narrowest possible retrieval.
    m = re.search(r"event\s*#?\s*(\d+)", q)
    if m:
        event = retrieval.get_event_by_id(m.group(1))
        if event:
            return [event]

    # 2) Category-specific helpers, tried in order of specificity.
    if "unknown" in q and ("visitor" in q or "person" in q or "people" in q):
        hits = retrieval.get_unknown_visitors()
        if hits:
            return hits

    if any(w in q for w in ("delivery", "courier", "postman")):
        hits = retrieval.get_delivery_events()
        if hits:
            return hits

    if "parcel" in q or "package" in q:
        hits = retrieval.get_parcel_events()
        if hits:
            return hits

    if any(w in q for w in ("vehicle", "car", "bike", "motorcycle", "van", "truck")):
        hits = retrieval.get_vehicle_events()
        if hits:
            return hits

    # 3) Named-object keyword lookup — "who carried a bag / phone / ..."
    for kw in _KNOWN_OBJECT_KEYWORDS:
        if kw in q:
            hits = retrieval.get_keyword_matches(kw)
            if hits:
                return hits[:_MAX_EVENTS_PER_QUERY]

    # 4) "yesterday" — the single most common relative-date question.
    if "yesterday" in q:
        hits = retrieval.get_yesterdays_events()
        if hits:
            return hits

    # 5) time_query intent — pull everything in range and let the
    #    existing hour-of-day filter (below, in build_structured_context)
    #    narrow it further; SQLite only helps us cap volume here.
    #    (No explicit date parsing beyond hour-of-day is attempted, to
    #    keep behaviour identical to the pre-Phase-2 JSON-memory path.)

    # 6) Fallback — most recent events, still capped, never the full table.
    return retrieval.get_recent_events(_MAX_EVENTS_PER_QUERY)


def build_structured_context_from_db(events, query, intents):
    """
    SQLite equivalent of build_structured_context() below — pulls each
    event's persons/movement/actions/objects/vehicles/keywords via
    DatabaseManager and renders the same kind of plain-text block for
    Groq. Only ever called with the ALREADY-FILTERED `events` list from
    _relevant_events_from_db() — never queries every table for every
    event in the database.
    """
    if not events:
        return "No events recorded yet."

    parts = []
    for event in events:
        if "time_query" in intents:
            hour = _parse_hour(event.start_time or "")
            if hour is not None and not _time_matches_query(hour, query):
                continue

        s = (
            f"[Event #{event.event_id} | {event.start_time} | "
            f"{event.duration_seconds or 0:.0f}s | provider={event.ai_provider}]\n"
            f"Summary: {event.summary}\n"
        )

        details = retrieval.get_event_details(event.event_id)

        for p in details["persons"]:
            s += (
                f"  Person [{p.known_status or 'unknown'}]"
                f"{' — ' + p.person_name if p.person_name else ''}: "
                f"gender={p.gender}, age={p.estimated_age}, "
                f"top={p.top_clothing}, bottom={p.bottom_clothing}, "
                f"bag={p.bag}, headwear={p.headwear}\n"
            )

        for m in details["movement"]:
            s += (
                f"  Movement: entry={m.entry_point}, exit={m.exit_point}, "
                f"direction={m.direction}, loitering={bool(m.loitering)}\n"
            )

        acts = details["actions"]
        if acts:
            s += "  Actions: " + ", ".join(a.action for a in acts) + "\n"

        objs = details["objects"]
        if objs:
            s += "  Objects: " + ", ".join(o.object_type for o in objs) + "\n"

        vehicles = details["vehicles"]
        if vehicles:
            s += "  Vehicles: " + ", ".join(
                f"{v.color or ''} {v.vehicle_type or ''}".strip() for v in vehicles
            ) + "\n"

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
    Full Telegram/Groq query pipeline (Phase 2):
        Groq/regex classify_intent() -> Python retrieves ONLY the
        relevant records from SQLite (_relevant_events_from_db) ->
        those records are rendered into a small context block
        (build_structured_context_from_db) -> Groq/Llama answers.

    OpenAI Vision is NEVER called here, no image is ever re-analyzed,
    and no summary is ever regenerated — this function only reads rows
    that services/db_writer.py already wrote to SQLite when each event
    closed. Returns (answer_str, [img_paths]).

    Photo lookups (image_request intent) still use the pre-existing
    JSON-memory + FAISS ReID crop index (find_matching_crops below) —
    that is a separate, already-working feature (returning an existing
    saved photo, not analyzing a new one) and is left untouched.
    """
    intents     = classify_intent(query)

    log_block("GROQ", "Generating response...")

    events      = _relevant_events_from_db(query, intents)

    log_block("DATABASE", "Searching SQLite...", f"Retrieved {len(events)} Events")

    context     = build_structured_context_from_db(events, query, intents)
    image_paths = []

    if "image_request" in intents:
        memory      = load_memory()
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
    log_block("GROQ", "Answer generated.")
    return output, image_paths
