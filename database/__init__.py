"""
database/
=========
Phase 1 — standalone SQLite database layer for Frouros.

STATUS: infrastructure only. Nothing in the existing pipeline imports
this package yet, and it does not import anything from the existing
pipeline (config/settings.py, event_manager.py, etc.) either. It exists
so Phase 2 can wire it in without another schema/architecture redesign.

Quick start (Phase 2, not yet wired anywhere):

    from database import DatabaseManager, Event, Person

    db = DatabaseManager()
    db.create_database()
    db.create_tables()

    db.insert_event(Event(
        event_id="evt_0001",
        camera_id="cam_front_door",
        ai_provider="qwen",
        event_type="person_detected",
        event_date="2026-07-11",
        summary="A person walked past the front door.",
    ))

Modules:
    database.py   — low-level connection handling + schema bootstrap
    db_manager.py — DatabaseManager: all insert_*/get_*/search_*/update_*/delete_* helpers
    models.py     — one dataclass per table
    queries.py    — every raw parameterized SQL statement
    schema.sql    — table/index/foreign-key definitions
"""

from database.database import (
    DB_PATH,
    connection_scope,
    create_database,
    create_tables,
    get_connection,
    initialize_database,
)
from database.db_manager import DatabaseManager
from database.models import (
    Action,
    Camera,
    Event,
    Keyword,
    Movement,
    ObjectRecord,
    Person,
    Vehicle,
)

__all__ = [
    "DatabaseManager",
    "DB_PATH",
    "create_database",
    "create_tables",
    "initialize_database",
    "get_connection",
    "connection_scope",
    "Camera",
    "Event",
    "Person",
    "Movement",
    "Action",
    "ObjectRecord",
    "Vehicle",
    "Keyword",
]
