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


def build_structured_from_qwen(summary_text, persons_attrs, person_name=None, confidence=None, participants=None):
    """
    Build a structured dict with the same shape OpenAI returns, from
    the Qwen pipeline's existing output — so both providers can be
    persisted through the exact same insert path below.

    Args:
        summary_text:  str  — Qwen's plain-text summary.
        persons_attrs: list[dict] — pipelines.summary_pipeline.
                        extract_person_attributes() output. Each dict
                        has keys: appearance, top_clothing,
                        bottom_clothing, footwear, headwear, bag,
                        accessories, actions, objects, movement,
                        waiting (ISSUE 1 — extracted directly from the
                        summary text, so these can never disagree with
                        it).
        participants:  list[dict] | None — ISSUE 3 (Qwen must become
                        scene aware). The Scene Event's COMPLETE
                        participant list, in the SAME order as
                        `persons_attrs` (both are built from
                        list(SceneEvent.participants.values()), so
                        index i in one is the same person as index i
                        in the other). Each dict has "name" (real
                        recognized identity or None) and "confidence".
                        When given, EVERY participant's real name is
                        stored — not just the first — so a scene with
                        Dad, Mom, and an unknown visitor stores all
                        three identities correctly instead of only
                        Dad's. Takes priority over the legacy
                        person_name/confidence args below.
        person_name:   str | None — LEGACY single-name fallback, kept
                        for backward compatibility when `participants`
                        isn't supplied. The real, recognized identity
                        from face/recognizer.py (e.g. "Dad") applied to
                        only the primary (first) person record.
        confidence:    float | None — the legacy recognition confidence
                        score (0-100) that goes with person_name.
    """
    persons = []
    actions = []
    objects = []
    movement = None

    for i, attrs in enumerate(persons_attrs or []):
        if not isinstance(attrs, dict):
            continue

        if participants is not None:
            # ISSUE 3: every participant's own real name/confidence,
            # index-aligned with persons_attrs — never just the first.
            p_info = participants[i] if i < len(participants) else None
            p_name = (p_info or {}).get("name")
            p_conf = (p_info or {}).get("confidence")
            is_known = bool(p_name)
        else:
            # Legacy single-name fallback (participants not supplied).
            is_known = (i == 0 and bool(person_name))
            p_name   = person_name if is_known else None
            p_conf   = confidence if is_known else None

        persons.append({
            "known_status": "known" if is_known else "unknown",
            "person_name": p_name if is_known else "Unknown",
            "gender": _NOT_VISIBLE,
            "estimated_age": _NOT_VISIBLE,
            "height": _NOT_VISIBLE,
            "body_build": attrs.get("appearance") or _NOT_VISIBLE,
            # ISSUE 1 FIX: these five used to be hardcoded to
            # _NOT_VISIBLE regardless of what the summary/attrs
            # actually said, which is exactly the summary/structured-
            # data mismatch bug reported (a summary clearly describing
            # a shirt/pants/bag while the structured row said "Not
            # Clearly Visible" for all three). `attrs` now comes from
            # prompts.summary_prompts.build_attribute_extraction_prompt(),
            # which extracts these fields directly from the summary
            # text, so they are read straight through here instead of
            # being discarded.
            "top_clothing": attrs.get("top_clothing") or _NOT_VISIBLE,
            "bottom_clothing": attrs.get("bottom_clothing") or _NOT_VISIBLE,
            "footwear": attrs.get("footwear") or _NOT_VISIBLE,
            "headwear": attrs.get("headwear") or _NOT_VISIBLE,
            "hair": _NOT_VISIBLE,
            "beard": _NOT_VISIBLE,
            "glasses": _NOT_VISIBLE,
            "mask": _NOT_VISIBLE,
            "accessories": attrs.get("accessories") or _NOT_VISIBLE,
            "bag": attrs.get("bag") or _NOT_VISIBLE,
            "dominant_hand": _NOT_VISIBLE,
            "confidence": p_conf if is_known else None,
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


def persist_initial_event(
    event_id,
    camera_id,
    start_time,
    video_path,
    persons=None,
    person_name=None,
    confidence=None,
):
    """
    Worker 1's ("Real-Time Scene Manager") lightweight "Save Initial
    Event" step — see pipelines/event_manager.py's
    EventManager._process_join(). Called exactly ONCE per Scene Event,
    for the FIRST tracked person to enter the ROI — inserts a minimal
    `events` row (+ that person's `persons` row) immediately after
    Face Recognition completes for them, well before OpenAI/Qwen/the
    scene-wide AI summary ever run, so SQLite — the single source of
    truth — has a real record of this Scene Event from the moment it
    exists. `ai_provider` is stored as "pending" and `summary` is left
    None until Worker 2 ("AI Processing Worker") replaces this row with
    the finished scene summary + structured data (covering every
    participant) using persist_event() below, under the SAME event_id.

    Every person who joins this same Scene Event AFTER this first one
    is added separately via persist_join_person() below — this
    function is never called again for an already-open scene, so the
    initial row is never overwritten to add a second/third person.

    Never raises — logs and returns False on failure, exactly like
    persist_event(), so a DB hiccup here can never delay or break the
    immediate Telegram alert (which does not depend on this call
    succeeding).

    Args:
        event_id:    the event's id (coerced to str, matching
                     Event.event_id: TEXT PRIMARY KEY).
        camera_id:   str | None.
        start_time:  ISO-8601 string ('YYYY-MM-DD HH:MM:SS').
        video_path:  str.
        persons:     list[dict] | None — participant(s) already known
                     at scene-creation time, each shaped
                     {"name": str | None, "confidence": float | None}.
                     pipelines/event_manager.py always passes exactly
                     one entry here (the first person to enter the
                     ROI).
        person_name / confidence: back-compat single-person shorthand,
                     used only when `persons` is not supplied.
    """
    try:
        db = _get_db()
        if camera_id:
            db.insert_camera(Camera(camera_id=camera_id, camera_name=camera_id))
    except Exception as e:
        print(f"[DB] ERROR — failed to ensure camera '{camera_id}' exists: {e}")
        return False

    event = Event(
        event_id=str(event_id),
        camera_id=camera_id,
        ai_provider="pending",
        event_type="scene_detected",
        event_date=(start_time or "")[:10] or None,
        start_time=start_time,
        end_time=None,
        duration_seconds=None,
        summary=None,
        confidence=None,
        video_path=video_path,
        merged_image_path=None,
    )

    if persons is None:
        persons = [{"name": person_name, "confidence": confidence}]

    person_rows = [
        Person(
            event_id=event.event_id,
            known_status="known" if p.get("name") else "unknown",
            person_name=p.get("name") or "Unknown",
            confidence=p.get("confidence"),
        )
        for p in persons
    ]

    try:
        # Belt-and-braces: if an initial row for this event_id somehow
        # already exists (e.g. a retried job), replace it rather than
        # raising an IntegrityError — mirrors the same delete-then-
        # insert approach persist_event() uses below.
        db.delete_event(event.event_id)
        db.insert_full_event(event=event, persons=person_rows)
        names = ", ".join(p.get("name") or "Unknown" for p in persons)
        print(f"[DB] Event {event.event_id} initial row persisted to "
              f"SQLite (persons={names}).")
        return True
    except Exception as e:
        print(f"[DB] ERROR — failed to persist initial event {event_id}: {e}")
        return False


def persist_join_person(event_id, camera_id, person_name=None, confidence=None):
    """
    "NEW PERSON ENTERS ACTIVE SCENE" — add ONE additional participant
    to an already-open Scene Event's SQLite row, without touching (or
    re-inserting) the `events` row itself or any person already
    recorded. See pipelines/event_manager.py's EventManager.
    _process_join() for when this is called instead of
    persist_initial_event() above.

    Uses DatabaseManager.insert_person() directly — a single plain
    INSERT — unlike persist_initial_event()/persist_event(), which
    both delete-then-reinsert the whole event and would otherwise wipe
    out every participant added so far.

    Never raises — logs and returns False on failure, exactly like
    every other function in this module.
    """
    try:
        db = _get_db()
        if camera_id:
            db.insert_camera(Camera(camera_id=camera_id, camera_name=camera_id))
        db.insert_person(Person(
            event_id=str(event_id),
            known_status="known" if person_name else "unknown",
            person_name=person_name or "Unknown",
            confidence=confidence,
        ))
        print(f"[DB] Event {event_id} — added participant "
              f"'{person_name or 'Unknown'}' to SQLite.")
        return True
    except Exception as e:
        print(f"[DB] ERROR — failed to add participant to event {event_id}: {e}")
        return False


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
            k: v for k, v in {**{"person_name": "Unknown"}, **p}.items()
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
        # Two-worker flow: Worker 1 (pipelines/event_manager.py's
        # _process_detection()) already inserted a minimal "initial"
        # row for this exact event_id via persist_initial_event()
        # above, before this (Worker 2's) call ever runs. A bare
        # INSERT would collide with that row (events.event_id is a
        # TEXT PRIMARY KEY with no ON CONFLICT clause — see
        # database/queries.INSERT_EVENT), so the initial row — and
        # its child rows, via each table's ON DELETE CASCADE in
        # schema.sql — is deleted first. This is a no-op (rowcount 0)
        # if no initial row exists (e.g. persist_initial_event()
        # itself failed), so insert_full_event() below always ends up
        # being the single, final row for this event_id — SQLite is
        # never left with two rows for one event, and the schema
        # itself is untouched.
        db.delete_event(event.event_id)
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