"""
database/db_manager.py
=======================
High-level DatabaseManager: every reusable helper function this phase
asked for (create_database, create_tables, insert_*, get_*, search_events,
update_event, delete_event), built on top of database.py's connection
handling and queries.py's SQL text.

This class is the ONLY thing Phase 2 should ever need to import from
this package:

    from database.db_manager import DatabaseManager
    db = DatabaseManager()
    db.create_database()
    db.create_tables()
    db.insert_event(Event(event_id="evt_123", ...))

Nothing in the existing pipeline imports this yet — see the Phase 1
task notes in schema.sql / this module's docstring.
"""

import json
import sqlite3
from dataclasses import asdict
from typing import List, Optional

from database import queries
from database.database import DB_PATH, connection_scope, create_database, create_tables
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


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(**dict(row))


def _row_to_person(row: sqlite3.Row) -> Person:
    return Person(**dict(row))


def _row_to_movement(row: sqlite3.Row) -> Movement:
    return Movement(**dict(row))


def _row_to_action(row: sqlite3.Row) -> Action:
    return Action(**dict(row))


def _row_to_object(row: sqlite3.Row) -> ObjectRecord:
    return ObjectRecord(**dict(row))


def _row_to_vehicle(row: sqlite3.Row) -> Vehicle:
    return Vehicle(**dict(row))


def _row_to_keyword(row: sqlite3.Row) -> Keyword:
    return Keyword(**dict(row))


class DatabaseManager:
    """
    Thin, explicit data-access layer over the SQLite database. Every
    method opens its own short-lived connection via
    database.connection_scope() (commit on success, rollback on error,
    always closed) — no long-lived shared connection/state to manage.

    All SQL is parameterized (see database/queries.py); this class
    never builds a query by concatenating user-supplied values into a
    string.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    def create_database(self) -> None:
        """Create the .db file (and its parent folder) if missing."""
        create_database(self.db_path)

    def create_tables(self) -> None:
        """Create every table/index from schema.sql if missing."""
        create_tables(self.db_path)

    # ------------------------------------------------------------
    # cameras
    # ------------------------------------------------------------

    def insert_camera(self, camera: Camera) -> str:
        """Insert a camera, or update it in place if camera_id already
        exists (see ON CONFLICT in queries.INSERT_CAMERA)."""
        with connection_scope(self.db_path) as conn:
            conn.execute(queries.INSERT_CAMERA, asdict(camera))
        return camera.camera_id

    def get_camera(self, camera_id: str) -> Optional[Camera]:
        with connection_scope(self.db_path) as conn:
            row = conn.execute(queries.GET_CAMERA, {"camera_id": camera_id}).fetchone()
        return Camera(**dict(row)) if row else None

    # ------------------------------------------------------------
    # events
    # ------------------------------------------------------------

    def insert_event(self, event: Event) -> str:
        """Insert a new event. Raises sqlite3.IntegrityError if
        event_id already exists — use update_event() to modify one."""
        params = asdict(event)
        params.pop("created_at", None)  # DB-generated (DEFAULT datetime('now'))
        with connection_scope(self.db_path) as conn:
            conn.execute(queries.INSERT_EVENT, params)
        return event.event_id

    def get_event(self, event_id: str) -> Optional[Event]:
        with connection_scope(self.db_path) as conn:
            row = conn.execute(queries.GET_EVENT, {"event_id": event_id}).fetchone()
        return _row_to_event(row) if row else None

    def get_events_by_date(self, event_date: str) -> List[Event]:
        """event_date format: 'YYYY-MM-DD'."""
        with connection_scope(self.db_path) as conn:
            rows = conn.execute(queries.GET_EVENTS_BY_DATE, {"event_date": event_date}).fetchall()
        return [_row_to_event(r) for r in rows]

    def get_events_by_camera(self, camera_id: str) -> List[Event]:
        with connection_scope(self.db_path) as conn:
            rows = conn.execute(queries.GET_EVENTS_BY_CAMERA, {"camera_id": camera_id}).fetchall()
        return [_row_to_event(r) for r in rows]

    def get_events_between_dates(self, date_from: str, date_to: str) -> List[Event]:
        """Inclusive range. Dates in 'YYYY-MM-DD' format."""
        with connection_scope(self.db_path) as conn:
            rows = conn.execute(
                queries.GET_EVENTS_BETWEEN_DATES,
                {"date_from": date_from, "date_to": date_to},
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    def search_events(
        self,
        keyword: Optional[str] = None,
        event_type: Optional[str] = None,
        camera_id: Optional[str] = None,
        ai_provider: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Event]:
        """
        Flexible multi-filter search — every argument is optional and
        AND-combined. Passing `keyword` joins against the keywords
        table (matches keywords.keyword exactly); swap for a LIKE
        clause here later if fuzzy matching is needed.

        All filter values are still passed as parameters, never
        interpolated into the SQL string, regardless of how many are
        supplied.
        """
        clauses: List[str] = []
        params: dict = {}

        sql = queries.SEARCH_EVENTS_BASE
        if keyword is not None:
            sql += " JOIN keywords ON keywords.event_id = events.event_id"
            clauses.append("keywords.keyword = :keyword")
            params["keyword"] = keyword
        if event_type is not None:
            clauses.append("events.event_type = :event_type")
            params["event_type"] = event_type
        if camera_id is not None:
            clauses.append("events.camera_id = :camera_id")
            params["camera_id"] = camera_id
        if ai_provider is not None:
            clauses.append("events.ai_provider = :ai_provider")
            params["ai_provider"] = ai_provider
        if date_from is not None:
            clauses.append("events.event_date >= :date_from")
            params["date_from"] = date_from
        if date_to is not None:
            clauses.append("events.event_date <= :date_to")
            params["date_to"] = date_to

        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY events.event_date DESC, events.start_time DESC"

        with connection_scope(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_event(r) for r in rows]

    def update_event(self, event_id: str, **fields) -> bool:
        """
        Update arbitrary columns on an existing event, e.g.:
            db.update_event("evt_123", summary="Updated text", confidence=0.92)

        Only columns that exist on the Event model are accepted, to
        keep this from ever building an arbitrary/unsafe SQL statement.
        Returns True if a row was updated, False if event_id wasn't found.
        """
        if not fields:
            return False

        valid_columns = set(Event.__dataclass_fields__.keys()) - {"event_id", "created_at"}
        unknown = set(fields) - valid_columns
        if unknown:
            raise ValueError(f"Unknown event field(s): {sorted(unknown)}")

        set_clause = ", ".join(f"{col} = :{col}" for col in fields)
        sql = f"UPDATE events SET {set_clause} WHERE event_id = :event_id"
        params = {**fields, "event_id": event_id}

        with connection_scope(self.db_path) as conn:
            cursor = conn.execute(sql, params)
        return cursor.rowcount > 0

    def delete_event(self, event_id: str) -> bool:
        """Delete an event and cascade-delete all related persons/
        movement/actions/objects/vehicles/keywords (ON DELETE CASCADE
        in schema.sql). Returns True if a row was deleted."""
        with connection_scope(self.db_path) as conn:
            cursor = conn.execute(queries.DELETE_EVENT, {"event_id": event_id})
        return cursor.rowcount > 0

    # ------------------------------------------------------------
    # persons
    # ------------------------------------------------------------

    def insert_person(self, person: Person) -> int:
        params = asdict(person)
        params.pop("person_id", None)  # auto-assigned
        with connection_scope(self.db_path) as conn:
            cursor = conn.execute(queries.INSERT_PERSON, params)
        return cursor.lastrowid

    def get_person(self, person_id: int) -> Optional[Person]:
        with connection_scope(self.db_path) as conn:
            row = conn.execute(queries.GET_PERSON, {"person_id": person_id}).fetchone()
        return _row_to_person(row) if row else None

    def get_persons_by_event(self, event_id: str) -> List[Person]:
        with connection_scope(self.db_path) as conn:
            rows = conn.execute(queries.GET_PERSONS_BY_EVENT, {"event_id": event_id}).fetchall()
        return [_row_to_person(r) for r in rows]

    # ------------------------------------------------------------
    # movement
    # ------------------------------------------------------------

    def insert_movement(self, movement: Movement) -> int:
        params = asdict(movement)
        params.pop("movement_id", None)  # auto-assigned
        # `path` is a list of points in the model — persist as JSON text.
        if isinstance(params.get("path"), (list, tuple, dict)):
            params["path"] = json.dumps(params["path"])
        with connection_scope(self.db_path) as conn:
            cursor = conn.execute(queries.INSERT_MOVEMENT, params)
        return cursor.lastrowid

    def get_movement_by_event(self, event_id: str) -> List[Movement]:
        with connection_scope(self.db_path) as conn:
            rows = conn.execute(queries.GET_MOVEMENT_BY_EVENT, {"event_id": event_id}).fetchall()
        return [_row_to_movement(r) for r in rows]

    # ------------------------------------------------------------
    # actions
    # ------------------------------------------------------------

    def insert_action(self, action: Action) -> int:
        params = asdict(action)
        params.pop("action_id", None)  # auto-assigned
        with connection_scope(self.db_path) as conn:
            cursor = conn.execute(queries.INSERT_ACTION, params)
        return cursor.lastrowid

    def get_actions_by_event(self, event_id: str) -> List[Action]:
        with connection_scope(self.db_path) as conn:
            rows = conn.execute(queries.GET_ACTIONS_BY_EVENT, {"event_id": event_id}).fetchall()
        return [_row_to_action(r) for r in rows]

    # ------------------------------------------------------------
    # objects
    # ------------------------------------------------------------

    def insert_object(self, obj: ObjectRecord) -> int:
        params = asdict(obj)
        params.pop("object_id", None)  # auto-assigned
        with connection_scope(self.db_path) as conn:
            cursor = conn.execute(queries.INSERT_OBJECT, params)
        return cursor.lastrowid

    def get_objects_by_event(self, event_id: str) -> List[ObjectRecord]:
        with connection_scope(self.db_path) as conn:
            rows = conn.execute(queries.GET_OBJECTS_BY_EVENT, {"event_id": event_id}).fetchall()
        return [_row_to_object(r) for r in rows]

    # ------------------------------------------------------------
    # vehicles
    # ------------------------------------------------------------

    def insert_vehicle(self, vehicle: Vehicle) -> int:
        params = asdict(vehicle)
        params.pop("vehicle_id", None)  # auto-assigned
        with connection_scope(self.db_path) as conn:
            cursor = conn.execute(queries.INSERT_VEHICLE, params)
        return cursor.lastrowid

    def get_vehicles_by_event(self, event_id: str) -> List[Vehicle]:
        with connection_scope(self.db_path) as conn:
            rows = conn.execute(queries.GET_VEHICLES_BY_EVENT, {"event_id": event_id}).fetchall()
        return [_row_to_vehicle(r) for r in rows]

    # ------------------------------------------------------------
    # keywords
    # ------------------------------------------------------------

    def insert_keyword(self, keyword: Keyword) -> int:
        params = asdict(keyword)
        params.pop("keyword_id", None)  # auto-assigned
        with connection_scope(self.db_path) as conn:
            cursor = conn.execute(queries.INSERT_KEYWORD, params)
        return cursor.lastrowid

    def get_keywords_by_event(self, event_id: str) -> List[Keyword]:
        with connection_scope(self.db_path) as conn:
            rows = conn.execute(queries.GET_KEYWORDS_BY_EVENT, {"event_id": event_id}).fetchall()
        return [_row_to_keyword(r) for r in rows]

    # ------------------------------------------------------------
    # Phase 2 — atomic multi-table insert
    # ------------------------------------------------------------

    def insert_full_event(
        self,
        event: Event,
        persons: Optional[List[Person]] = None,
        movement: Optional[Movement] = None,
        actions: Optional[List[Action]] = None,
        objects: Optional[List[ObjectRecord]] = None,
        vehicles: Optional[List[Vehicle]] = None,
        keywords: Optional[List[str]] = None,
    ) -> str:
        """
        Insert one event and every one of its child rows (persons,
        movement, actions, objects, vehicles, keywords) as a SINGLE
        atomic transaction — used by services/db_writer.py to persist
        the OpenAI Vision (or Qwen fallback) result for a finished
        event.

        All statements run on ONE connection_scope(), so:
          - Either every row commits together, or
          - Any failure (bad FK, bad column, etc.) rolls back the
            ENTIRE transaction — the event row is never left half
            -written with some children missing. No data is ever lost
            or partially committed.

        `objects[i].held_by_person`, if set, must be an INDEX into the
        `persons` list (0-based) — it is resolved to the real
        auto-assigned person_id after persons are inserted, since
        person_id doesn't exist until insert time.

        Returns:
            The event_id that was inserted.
        """
        persons  = persons or []
        actions  = actions or []
        objects  = objects or []
        vehicles = vehicles or []
        keywords = keywords or []

        with connection_scope(self.db_path) as conn:
            # events
            params = asdict(event)
            params.pop("created_at", None)
            conn.execute(queries.INSERT_EVENT, params)

            # persons — collect the real auto-assigned person_ids in
            # list order so objects[i].held_by_person (a list index)
            # can be resolved to a real FK below.
            person_ids: List[int] = []
            for person in persons:
                p_params = asdict(person)
                p_params.pop("person_id", None)
                cursor = conn.execute(queries.INSERT_PERSON, p_params)
                person_ids.append(cursor.lastrowid)

            # movement (0..1 per event)
            if movement is not None:
                m_params = asdict(movement)
                m_params.pop("movement_id", None)
                if isinstance(m_params.get("path"), (list, tuple, dict)):
                    m_params["path"] = json.dumps(m_params["path"])
                conn.execute(queries.INSERT_MOVEMENT, m_params)

            # actions
            for action in actions:
                a_params = asdict(action)
                a_params.pop("action_id", None)
                conn.execute(queries.INSERT_ACTION, a_params)

            # objects — resolve held_by_person (list index -> real person_id)
            for obj in objects:
                o_params = asdict(obj)
                o_params.pop("object_id", None)
                held_idx = o_params.get("held_by_person")
                if isinstance(held_idx, int) and 0 <= held_idx < len(person_ids):
                    o_params["held_by_person"] = person_ids[held_idx]
                elif held_idx is not None and held_idx not in person_ids:
                    o_params["held_by_person"] = None
                conn.execute(queries.INSERT_OBJECT, o_params)

            # vehicles
            for vehicle in vehicles:
                v_params = asdict(vehicle)
                v_params.pop("vehicle_id", None)
                conn.execute(queries.INSERT_VEHICLE, v_params)

            # keywords
            for kw in keywords:
                if not kw:
                    continue
                conn.execute(queries.INSERT_KEYWORD, {
                    "event_id": event.event_id,
                    "keyword": kw,
                })

        return event.event_id
