"""
database/queries.py
====================
Every raw SQL statement the database layer uses, as named constants.

Keeping SQL text here (instead of inline inside db_manager.py methods)
keeps business logic and SQL fully separate, makes every query easy to
find/audit in one place, and means db_manager.py never builds SQL by
string-concatenation — every statement below uses ":named" placeholders
so callers always pass parameters, never interpolate values.
"""

# ---------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------

INSERT_CAMERA = """
INSERT INTO cameras (camera_id, camera_name, location, roi_name)
VALUES (:camera_id, :camera_name, :location, :roi_name)
ON CONFLICT(camera_id) DO UPDATE SET
    camera_name = excluded.camera_name,
    location    = excluded.location,
    roi_name    = excluded.roi_name
"""

GET_CAMERA = "SELECT * FROM cameras WHERE camera_id = :camera_id"

# ---------------------------------------------------------------------
# events
# ---------------------------------------------------------------------

INSERT_EVENT = """
INSERT INTO events (
    event_id, camera_id, ai_provider, event_type, event_date,
    start_time, end_time, duration_seconds, summary, confidence,
    video_path, merged_image_path
) VALUES (
    :event_id, :camera_id, :ai_provider, :event_type, :event_date,
    :start_time, :end_time, :duration_seconds, :summary, :confidence,
    :video_path, :merged_image_path
)
"""

GET_EVENT = "SELECT * FROM events WHERE event_id = :event_id"

GET_EVENTS_BY_DATE = """
SELECT * FROM events WHERE event_date = :event_date
ORDER BY start_time DESC
"""

GET_EVENTS_BY_CAMERA = """
SELECT * FROM events WHERE camera_id = :camera_id
ORDER BY event_date DESC, start_time DESC
"""

GET_EVENTS_BETWEEN_DATES = """
SELECT * FROM events
WHERE event_date BETWEEN :date_from AND :date_to
ORDER BY event_date DESC, start_time DESC
"""

GET_EVENTS_BY_TIME_RANGE = """
SELECT * FROM events
WHERE start_time BETWEEN :start_time AND :end_time
ORDER BY start_time DESC
"""

GET_UNKNOWN_VISITOR_EVENTS = """
SELECT DISTINCT events.* FROM events
JOIN persons ON persons.event_id = events.event_id
WHERE persons.known_status = 'unknown' OR persons.known_status IS NULL
ORDER BY events.event_date DESC, events.start_time DESC
"""

DELETE_EVENT = "DELETE FROM events WHERE event_id = :event_id"

# search_events() builds its WHERE clause dynamically (optional filters),
# so its base SELECT lives here and db_manager.py appends conditions —
# still always through parameterized "?"/":name" placeholders, never
# raw string interpolation of values.
SEARCH_EVENTS_BASE = "SELECT DISTINCT events.* FROM events"

GET_EVENTS_WITH_VEHICLES = """
SELECT DISTINCT events.* FROM events
JOIN vehicles ON vehicles.event_id = events.event_id
ORDER BY events.event_date DESC, events.start_time DESC
"""

GET_EVENTS_BY_OBJECT_TYPE = """
SELECT DISTINCT events.* FROM events
JOIN objects ON objects.event_id = events.event_id
WHERE objects.object_type LIKE :pattern
ORDER BY events.event_date DESC, events.start_time DESC
"""

GET_EVENTS_BY_KEYWORD_LIKE = """
SELECT DISTINCT events.* FROM events
JOIN keywords ON keywords.event_id = events.event_id
WHERE keywords.keyword LIKE :pattern
ORDER BY events.event_date DESC, events.start_time DESC
"""

# Person-aware retrieval (Task 9) — every event that has at least one
# person row matching the given name, case-insensitively. person_name
# is always a real stored name ("Dad", "test_1", ...) or the literal
# "Unknown" — never a placeholder (see services/db_writer.py).
GET_EVENTS_BY_PERSON_NAME = """
SELECT DISTINCT events.* FROM events
JOIN persons ON persons.event_id = events.event_id
WHERE persons.person_name = :person_name COLLATE NOCASE
ORDER BY events.event_date DESC, events.start_time DESC
"""

# Every distinct registered ("known") person name currently in SQLite —
# used to detect which person (if any) a Telegram question is about,
# BEFORE any event rows are fetched (see pipelines/query_pipeline.py).
GET_KNOWN_PERSON_NAMES = """
SELECT DISTINCT person_name FROM persons
WHERE known_status = 'known'
  AND person_name IS NOT NULL
  AND person_name != 'Unknown'
"""

# ---------------------------------------------------------------------
# persons
# ---------------------------------------------------------------------

INSERT_PERSON = """
INSERT INTO persons (
    event_id, known_status, face_id, person_name, gender, estimated_age,
    height, body_build, top_clothing, bottom_clothing, footwear,
    headwear, hair, beard, glasses, mask, accessories, bag,
    dominant_hand, confidence
) VALUES (
    :event_id, :known_status, :face_id, :person_name, :gender, :estimated_age,
    :height, :body_build, :top_clothing, :bottom_clothing, :footwear,
    :headwear, :hair, :beard, :glasses, :mask, :accessories, :bag,
    :dominant_hand, :confidence
)
"""

GET_PERSON = "SELECT * FROM persons WHERE person_id = :person_id"

GET_PERSONS_BY_EVENT = "SELECT * FROM persons WHERE event_id = :event_id"

# ---------------------------------------------------------------------
# movement
# ---------------------------------------------------------------------

INSERT_MOVEMENT = """
INSERT INTO movement (
    event_id, entry_point, exit_point, direction, path, speed,
    final_position, loitering, duration
) VALUES (
    :event_id, :entry_point, :exit_point, :direction, :path, :speed,
    :final_position, :loitering, :duration
)
"""

GET_MOVEMENT_BY_EVENT = "SELECT * FROM movement WHERE event_id = :event_id"

# ---------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------

INSERT_ACTION = """
INSERT INTO actions (event_id, sequence_number, timestamp, action)
VALUES (:event_id, :sequence_number, :timestamp, :action)
"""

GET_ACTIONS_BY_EVENT = """
SELECT * FROM actions WHERE event_id = :event_id ORDER BY sequence_number ASC
"""

# ---------------------------------------------------------------------
# objects
# ---------------------------------------------------------------------

INSERT_OBJECT = """
INSERT INTO objects (event_id, object_type, description, held_by_person, confidence)
VALUES (:event_id, :object_type, :description, :held_by_person, :confidence)
"""

GET_OBJECTS_BY_EVENT = "SELECT * FROM objects WHERE event_id = :event_id"

# ---------------------------------------------------------------------
# vehicles
# ---------------------------------------------------------------------

INSERT_VEHICLE = """
INSERT INTO vehicles (event_id, vehicle_type, color, number_plate, confidence)
VALUES (:event_id, :vehicle_type, :color, :number_plate, :confidence)
"""

GET_VEHICLES_BY_EVENT = "SELECT * FROM vehicles WHERE event_id = :event_id"

# ---------------------------------------------------------------------
# keywords
# ---------------------------------------------------------------------

INSERT_KEYWORD = """
INSERT INTO keywords (event_id, keyword)
VALUES (:event_id, :keyword)
"""

GET_KEYWORDS_BY_EVENT = "SELECT * FROM keywords WHERE event_id = :event_id"
