"""
pipelines/retrieval.py
=======================
The smart-retrieval layer: reusable, narrowly-scoped helper functions
over the SQLite database. This is the ONLY module that should ever
import database.db_manager for read access on the Telegram/Groq query
path — telegram/bot.py and pipelines/query_pipeline.py must go through
these functions, never open a DatabaseManager / run SQL directly.

Every helper here retrieves ONLY the minimum records a question needs
— never the whole table:

    Explain event 5              -> get_event_by_id(5)
    What happened yesterday?     -> get_events_by_date(...)
    Did any delivery come?       -> get_delivery_events()
    Show unknown visitors.       -> get_unknown_visitors()
    Show people carrying parcels -> get_parcel_events()

A single hard cap (_MAX_EVENTS) is applied everywhere a helper could
otherwise return an unbounded number of rows, so context sent to Groq
always stays small regardless of how large the database grows.
"""

from datetime import datetime, timedelta
from typing import List, Optional

from database.db_manager import DatabaseManager
from database.models import Event

# Single shared DatabaseManager instance for the whole retrieval layer.
_db = DatabaseManager()

# Hard cap on how many events any one helper ever returns.
_MAX_EVENTS = 25

# Keyword patterns used to identify "delivery" / "parcel" style events.
# These map a natural-language concept onto the LIKE patterns used
# against the objects/keywords tables — not an exhaustive vocabulary,
# just the common cases the router is expected to see.
_DELIVERY_KEYWORDS = ["delivery", "courier", "postman", "parcel", "package"]
_PARCEL_OBJECT_PATTERNS = ["%parcel%", "%package%", "%box%", "%delivery%"]
_VEHICLE_KEYWORDS = ["vehicle", "car", "bike", "motorcycle", "van", "truck"]


# --------------------------------------------------
# Single-event lookup
# --------------------------------------------------

def get_event_by_id(event_id) -> Optional[Event]:
    """Retrieve exactly ONE event by id — e.g. "explain event 5"."""
    return _db.get_event(str(event_id))


# --------------------------------------------------
# Date / time scoped lookups
# --------------------------------------------------

def get_events_by_date(event_date: str) -> List[Event]:
    """
    Events on a single calendar day. `event_date` must be 'YYYY-MM-DD'.
    Callers resolve relative phrases ("yesterday", "today") to a
    concrete date before calling this (see query_pipeline.py).
    """
    return _db.get_events_by_date(event_date)[:_MAX_EVENTS]


def get_events_by_time_range(start_time: str, end_time: str) -> List[Event]:
    """Events between two timestamps ('YYYY-MM-DD HH:MM:SS')."""
    return _db.get_events_by_time_range(start_time, end_time)[:_MAX_EVENTS]


def get_yesterdays_events() -> List[Event]:
    """Convenience wrapper for the extremely common "what happened
    yesterday?" question — resolves the date itself."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    return get_events_by_date(yesterday)


# --------------------------------------------------
# Category-scoped lookups
# --------------------------------------------------

def get_unknown_visitors() -> List[Event]:
    """Events where at least one person is NOT a recognized/known
    face — e.g. "show unknown visitors"."""
    return _db.get_unknown_visitor_events()[:_MAX_EVENTS]


def get_delivery_events() -> List[Event]:
    """
    Events that look like a delivery/courier visit — tries a keyword
    match first (cheap, precise), then a fuzzy LIKE fallback so a
    slightly-differently-worded delivery event still surfaces.
    """
    for kw in _DELIVERY_KEYWORDS:
        hits = _db.search_events(keyword=kw)
        if hits:
            return hits[:_MAX_EVENTS]

    for kw in _DELIVERY_KEYWORDS:
        hits = _db.get_events_by_keyword_like(f"%{kw}%")
        if hits:
            return hits[:_MAX_EVENTS]

    return []


def get_vehicle_events() -> List[Event]:
    """Events with at least one linked vehicle row — e.g. "did any
    vehicle come by"."""
    return _db.get_events_with_vehicles()[:_MAX_EVENTS]


def get_parcel_events() -> List[Event]:
    """Events where someone was seen carrying a parcel/package/box —
    e.g. "show people carrying parcels"."""
    for pattern in _PARCEL_OBJECT_PATTERNS:
        hits = _db.get_events_by_object_type(pattern)
        if hits:
            return hits[:_MAX_EVENTS]
    return []


def get_keyword_matches(keyword: str) -> List[Event]:
    """
    Generic named-object / free-keyword lookup, e.g. "bag", "phone",
    "umbrella". Tries an exact keyword-table match first, then falls
    back to a fuzzy LIKE match.
    """
    hits = _db.search_events(keyword=keyword)
    if hits:
        return hits[:_MAX_EVENTS]
    return _db.get_events_by_keyword_like(f"%{keyword}%")[:_MAX_EVENTS]


# --------------------------------------------------
# Person-scoped lookups (Task 9 — person-aware retrieval)
# --------------------------------------------------

def get_known_person_names() -> List[str]:
    """
    Every distinct registered person name currently in SQLite (e.g.
    ["Dad", "Mom", "test_1"]) — used by pipelines/query_pipeline.py to
    detect which person (if any) a Telegram question is about, BEFORE
    any event rows are fetched. Does not include "Unknown" (that is
    handled by get_unknown_visitors() above, which already matches
    questions like "show unknown visitors").
    """
    return _db.get_known_person_names()


def get_events_by_person(person_name: str) -> List[Event]:
    """Every event with a person row matching `person_name` — e.g.
    "When did Dad come?" -> get_events_by_person("Dad")."""
    return _db.get_events_by_person_name(person_name)[:_MAX_EVENTS]


def get_events_by_person_and_date(person_name: str, event_date: str) -> List[Event]:
    """Only `person_name`'s events on a single calendar day
    ('YYYY-MM-DD') — e.g. "What was Dad doing yesterday?"."""
    events = get_events_by_person(person_name)
    return [e for e in events if e.event_date == event_date][:_MAX_EVENTS]


def get_events_by_person_and_keyword(person_name: str, keyword: str) -> List[Event]:
    """Only `person_name`'s events that also match a keyword/object —
    e.g. "Did Dad carry a parcel?" ->
    get_events_by_person_and_keyword("Dad", "parcel")."""
    person_event_ids = {e.event_id for e in get_events_by_person(person_name)}
    if not person_event_ids:
        return []
    keyword_hits = get_keyword_matches(keyword)
    return [e for e in keyword_hits if e.event_id in person_event_ids][:_MAX_EVENTS]


def get_events_by_person_and_timerange(person_name: str, start_time: str, end_time: str) -> List[Event]:
    """Only `person_name`'s events within a timestamp window
    ('YYYY-MM-DD HH:MM:SS')."""
    person_event_ids = {e.event_id for e in get_events_by_person(person_name)}
    if not person_event_ids:
        return []
    ranged = get_events_by_time_range(start_time, end_time)
    return [e for e in ranged if e.event_id in person_event_ids][:_MAX_EVENTS]


# --------------------------------------------------
# Fallback
# --------------------------------------------------

def get_recent_events(limit: int = _MAX_EVENTS) -> List[Event]:
    """
    Most recent events, still capped — used only when no narrower
    helper above produced a match, so a question never comes back
    empty just because it didn't hit a known keyword/category.
    """
    return _db.search_events()[:limit]


# --------------------------------------------------
# Event children (persons/movement/actions/objects/vehicles/keywords)
# --------------------------------------------------

def get_event_details(event_id):
    """
    Bundle every child record for one event (persons, movement,
    actions, objects, vehicles, keywords) — used once a specific
    event has already been selected by one of the helpers above, to
    render its full detail without ever querying every table for
    every event in the database.
    """
    event_id = str(event_id)
    return {
        "persons":   _db.get_persons_by_event(event_id),
        "movement":  _db.get_movement_by_event(event_id),
        "actions":   _db.get_actions_by_event(event_id),
        "objects":   _db.get_objects_by_event(event_id),
        "vehicles":  _db.get_vehicles_by_event(event_id),
        "keywords":  _db.get_keywords_by_event(event_id),
    }
