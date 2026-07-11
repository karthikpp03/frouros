-- database/schema.sql
-- =====================================================================
-- Frouros CCTV Surveillance Platform — SQLite Schema (Phase 1)
--
-- This is pure database infrastructure. Nothing in the existing
-- pipeline reads from or writes to these tables yet — this schema is
-- prepared ahead of Phase 2 integration, when JSON event memory
-- (memory/event_memory.py) will be replaced by this database.
--
-- Design notes:
--   * event_id is the hub every other table hangs off, via FOREIGN KEY
--     ... ON DELETE CASCADE, so deleting an event cleans up all of its
--     persons/movement/actions/objects/vehicles/keywords automatically.
--   * "known" vs "unknown" persons, "qwen" vs "openai" providers, and
--     multiple cameras/persons/vehicles per event are all first-class,
--     so future face recognition, multi-provider routing, and a
--     dashboard can all be built on top without another redesign.
--   * TEXT is used for timestamps (ISO-8601 strings) rather than a
--     dedicated datetime type, since SQLite has no native datetime —
--     this keeps sorting/filtering simple with plain string comparison.
-- =====================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------
-- cameras — one row per physical camera / ROI source.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cameras (
    camera_id   TEXT PRIMARY KEY,
    camera_name TEXT NOT NULL,
    location    TEXT,
    roi_name    TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------
-- events — one row per detected/tracked event. The hub table.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
    event_id          TEXT PRIMARY KEY,
    camera_id         TEXT,
    ai_provider       TEXT,               -- 'qwen' | 'openai' | 'none'
    event_type        TEXT,
    event_date        TEXT,               -- 'YYYY-MM-DD'
    start_time        TEXT,               -- ISO-8601
    end_time          TEXT,               -- ISO-8601
    duration_seconds  REAL,
    summary           TEXT,
    confidence        REAL,
    video_path        TEXT,
    merged_image_path TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (camera_id) REFERENCES cameras (camera_id)
        ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_events_camera_id   ON events (camera_id);
CREATE INDEX IF NOT EXISTS idx_events_event_date  ON events (event_date);
CREATE INDEX IF NOT EXISTS idx_events_event_type  ON events (event_type);
CREATE INDEX IF NOT EXISTS idx_events_ai_provider ON events (ai_provider);

-- ---------------------------------------------------------------------
-- persons — 0..N per event (multiple people in one event).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS persons (
    person_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL,
    known_status    TEXT,                 -- 'known' | 'unknown'
    face_id         TEXT,                 -- future face-recognition identity key
    person_name     TEXT,
    gender          TEXT,
    estimated_age   TEXT,
    height          TEXT,
    body_build      TEXT,
    top_clothing    TEXT,
    bottom_clothing TEXT,
    footwear        TEXT,
    headwear        TEXT,
    hair            TEXT,
    beard           TEXT,
    glasses         TEXT,
    mask            TEXT,
    accessories     TEXT,
    bag             TEXT,
    dominant_hand   TEXT,
    confidence      REAL,
    FOREIGN KEY (event_id) REFERENCES events (event_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_persons_event_id     ON persons (event_id);
CREATE INDEX IF NOT EXISTS idx_persons_face_id      ON persons (face_id);
CREATE INDEX IF NOT EXISTS idx_persons_known_status ON persons (known_status);

-- ---------------------------------------------------------------------
-- movement — 0..1 per event (or per person, if tracked individually
-- later). "path" is stored as a JSON-encoded string (list of
-- [x, y, timestamp] points), since SQLite has no native array type.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS movement (
    movement_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       TEXT NOT NULL,
    entry_point    TEXT,
    exit_point     TEXT,
    direction      TEXT,
    path           TEXT,                  -- JSON-encoded list of points
    speed          REAL,
    final_position TEXT,
    loitering      INTEGER,               -- 0/1 boolean
    duration       REAL,
    FOREIGN KEY (event_id) REFERENCES events (event_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_movement_event_id ON movement (event_id);

-- ---------------------------------------------------------------------
-- actions — 0..N per event, ordered by sequence_number.
-- e.g. "Entered ROI", "Walking", "Stopped", "Opened Door", "Exited".
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS actions (
    action_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    timestamp       TEXT,
    action          TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events (event_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_actions_event_id ON actions (event_id);
CREATE INDEX IF NOT EXISTS idx_actions_sequence ON actions (event_id, sequence_number);

-- ---------------------------------------------------------------------
-- objects — 0..N per event. e.g. "Parcel", "Helmet", "Phone", "Umbrella".
-- held_by_person optionally links to the specific person carrying it.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS objects (
    object_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       TEXT NOT NULL,
    object_type    TEXT NOT NULL,
    description    TEXT,
    held_by_person INTEGER,               -- FK -> persons.person_id, nullable
    confidence     REAL,
    FOREIGN KEY (event_id) REFERENCES events (event_id)
        ON DELETE CASCADE,
    FOREIGN KEY (held_by_person) REFERENCES persons (person_id)
        ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_objects_event_id    ON objects (event_id);
CREATE INDEX IF NOT EXISTS idx_objects_object_type ON objects (object_type);

-- ---------------------------------------------------------------------
-- vehicles — 0..N per event.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT NOT NULL,
    vehicle_type TEXT,
    color        TEXT,
    number_plate TEXT,
    confidence   REAL,
    FOREIGN KEY (event_id) REFERENCES events (event_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vehicles_event_id     ON vehicles (event_id);
CREATE INDEX IF NOT EXISTS idx_vehicles_number_plate ON vehicles (number_plate);

-- ---------------------------------------------------------------------
-- keywords — 0..N per event. Powers fast text/tag search over events
-- without a full-text scan of the summary column.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS keywords (
    keyword_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   TEXT NOT NULL,
    keyword    TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events (event_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_keywords_event_id ON keywords (event_id);
CREATE INDEX IF NOT EXISTS idx_keywords_keyword  ON keywords (keyword);
