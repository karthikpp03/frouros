"""
services/db_writer.py
======================
Phase 2 — the ONE place that turns a finished event (summary +
structured data, from either provider) into rows in SQLite.

This module is purely additive: nothing in the existing pipeline is
modified to make this work, it is only ever *called* (from
pipelines/event_manager.py, at the very end of _finalize_event(),
after the existing JSON-memory/Telegram steps have already run).

Routing recap (see services/summary_router.py for the source of truth):
  OpenAI branch -> generate_event_summary() already returns a rich
                   `structured` dict shaped like the OpenAI JSON
                   contract in models/openai_vl.py.
  Qwen branch   -> `structured` is None. We still want an `events` row
                   in SQLite (so Telegram/Groq queries — which now read
                   ONLY from SQLite, see pipelines/query_pipeline.py —
                   can see every event, not just OpenAI ones), so
                   build_structured_from_qwen() assembles a minimal
                   equivalent structure from the data the Qwen pipeline
                   already produced (pipelines/summary_pipeline.
                   extract_person_attributes()) — no new AI calls, no
                   behaviour change to Qwen itself.

Every insert goes through DatabaseManager.insert_full_event(), which
wraps the whole event (+ all its children) in ONE transaction — commit
all-or-nothing, rollback on any failure, so a bad record can never
half-write itself into the database. See database/db_manager.py.

Failures here are caught and logged, never re-raised — a DB write
problem must never take down event finalisation (Telegram alert /
JSON memory / gallery persistence must all still complete). The JSON
memory file (memory/event_memory.py) remains the source of truth of
record; SQLite is an additive, queryable mirror of it for Telegram/Groq.
"""

from typing import Optional

from database.db_manager import DatabaseManager
from database.models import Action, Camera, Event, Keyword, Movement, ObjectRecord, Person, Vehicle
from utils.event_logger import log_block

_NOT_VISIBLE = "Not Clearly Visible"

# Lazily initialised singleton — created tables are idempotent
# (CREATE TABLE IF NOT EXISTS), so calling this repeatedly is safe.
_db: Optional[DatabaseManager] = None


def _get_db() -> DatabaseManager:
    global _db
    if _db is None:
        _db = DatabaseManager()
        _db.create_database()
        _db.create_tables()
    return _db


def build_structured_from_qwen(summary_text, persons_attrs):
    """
    Build a structured dict with the same shape OpenAI returns, from
    the Qwen pipeline's existing output — so both providers can be
    persisted through the exact same insert path below.

    Args:
        summary_text:  str  — Qwen's plain-text summary.
        persons_attrs: list[dict] — pipelines.summary_pipeline.
                        extract_person_attributes() output. Each dict
                        has keys: appearance, actions, objects,
                        movement, waiting.
    """
    persons = []
    actions = []
    objects = []
    movement = None

    for i, attrs in enumerate(persons_attrs or []):
        if not isinstance(attrs, dict):
            continue

        persons.append({
            "known_status": "unknown",
            "gender": _NOT_VISIBLE,
            "estimated_age": _NOT_VISIBLE,
            "height": _NOT_VISIBLE,
            "body_build": attrs.get("appearance") or _NOT_VISIBLE,
            "top_clothing": _NOT_VISIBLE,
            "bottom_clothing": _NOT_VISIBLE,
            "footwear": _NOT_VISIBLE,
            "headwear": _NOT_VISIBLE,
            "hair": _NOT_VISIBLE,
            "beard": _NOT_VISIBLE,
            "glasses": _NOT_VISIBLE,
            "mask": _NOT_VISIBLE,
            "accessories": _NOT_VISIBLE,
            "bag": _NOT_VISIBLE,
            "dominant_hand": _NOT_VISIBLE,
            "confidence": None,
        })

        for seq, act in enumerate(attrs.get("actions") or [], start=1):
            actions.append({"sequence_number": len(actions) + 1, "action": act})

        for obj in attrs.get("objects") or []:
            objects.append({
                "object_type": obj,
                "description": None,
                "held_by_person": i,   # list index, resolved by db_manager
                "confidence": None,
            })

        # Only one `movement` row per event in the schema — first
        # person's free-text movement description wins (matches the
        # single-shared-movement assumption OpenAI's contract also
        # makes for a single-subject event).
        if movement is None and attrs.get("movement"):
            movement = {
                "entry_point": _NOT_VISIBLE,
                "exit_point": _NOT_VISIBLE,
                "direction": attrs.get("movement"),
                "speed": None,
                "final_position": _NOT_VISIBLE,
                "loitering": bool(attrs.get("waiting")),
                "duration": None,
            }

    return {
        "summary": summary_text,
        "event_type": "person_detected",
        "confidence": None,
        "persons": persons,
        "movement": movement,
        "actions": actions,
        "objects": objects,
        "vehicles": [],
        "keywords": [],
    }


def persist_event(
    event_id,
    camera_id,
    provider,
    summary_text,
    structured,
    start_time,
    end_time,
    duration_seconds,
    video_path,
    merged_image_path=None,
):
    """
    Insert one finished event + all of its structured children into
    SQLite, as ONE atomic transaction. Never raises — logs and returns
    False on failure so callers (pipelines/event_manager.py) never
    need a try/except of their own.

    Args:
        event_id:          the event's id (str/int — coerced to str,
                            matching Event.event_id: TEXT PRIMARY KEY).
        camera_id:          str | None.
        provider:            "qwen" | "openai" | "none".
        summary_text:        str — plain-text CCTV summary.
        structured:           dict | None — see build_structured_from_qwen()
                              / models/openai_vl.py for the shape.
        start_time, end_time: ISO-8601 strings.
        duration_seconds:     float.
        video_path:           str.
        merged_image_path:    str | None — only set on the OpenAI branch.

    Returns:
        bool — True if the transaction committed, False if it failed
        (already logged).
    """
    structured = structured or {}

    try:
        db = _get_db()
        if camera_id:
            # events.camera_id has a FOREIGN KEY -> cameras.camera_id.
            # insert_camera() is an upsert (ON CONFLICT DO UPDATE, see
            # database/queries.INSERT_CAMERA), so this is always safe
            # and cheap even if src/main.py already registered this
            # camera at startup — it just guarantees the FK never fails
            # because of load/import ordering.
            db.insert_camera(Camera(camera_id=camera_id, camera_name=camera_id))
    except Exception as e:
        print(f"[DB] ERROR — failed to ensure camera '{camera_id}' exists: {e}")
        return False

    event = Event(
        event_id=str(event_id),
        camera_id=camera_id,
        ai_provider=provider,
        event_type=structured.get("event_type") or "person_detected",
        event_date=(start_time or "")[:10] or None,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration_seconds,
        summary=summary_text,
        confidence=structured.get("confidence"),
        video_path=video_path,
        merged_image_path=merged_image_path,
    )

    persons = [
        Person(event_id=event.event_id, **{
            k: v for k, v in p.items()
            if k in Person.__dataclass_fields__
        })
        for p in (structured.get("persons") or [])
    ]

    movement_dict = structured.get("movement")
    movement = None
    if movement_dict:
        movement = Movement(event_id=event.event_id, **{
            k: v for k, v in movement_dict.items()
            if k in Movement.__dataclass_fields__
        })

    actions = [
        Action(
            event_id=event.event_id,
            sequence_number=a.get("sequence_number", i + 1),
            action=a.get("action") or _NOT_VISIBLE,
            timestamp=a.get("timestamp"),
        )
        for i, a in enumerate(structured.get("actions") or [])
    ]

    objects = [
        ObjectRecord(
            event_id=event.event_id,
            object_type=o.get("object_type") or _NOT_VISIBLE,
            description=o.get("description"),
            held_by_person=o.get("held_by_person"),
            confidence=o.get("confidence"),
        )
        for o in (structured.get("objects") or [])
    ]

    vehicles = [
        Vehicle(event_id=event.event_id, **{
            k: v for k, v in v_.items()
            if k in Vehicle.__dataclass_fields__
        })
        for v_ in (structured.get("vehicles") or [])
    ]

    keywords = list(structured.get("keywords") or [])

    log_block(
        "DATABASE",
        "Saving Event...",
        "Saving Persons...",
        "Saving Objects...",
        "Saving Actions...",
        "Saving Keywords...",
    )

    try:
        db.insert_full_event(
            event=event,
            persons=persons,
            movement=movement,
            actions=actions,
            objects=objects,
            vehicles=vehicles,
            keywords=keywords,
        )
        print(f"[DB] Event {event.event_id} persisted to SQLite "
              f"({len(persons)} person(s), {len(actions)} action(s), "
              f"{len(objects)} object(s), {len(vehicles)} vehicle(s)).")
        log_block("DATABASE", "SQLite Save Success")
        return True
    except Exception as e:
        # Never let a DB problem take down event finalisation — the
        # JSON memory record (already saved by the caller) remains the
        # source of truth even if this mirror write fails.
        print(f"[DB] ERROR — failed to persist event {event_id} to SQLite: {e}")
        log_block("DATABASE", f"SQLite Save Failed: {e}")
        return False
