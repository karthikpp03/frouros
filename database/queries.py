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

DELETE_EVENT = "DELETE FROM events WHERE event_id = :event_id"

# search_events() builds its WHERE clause dynamically (optional filters),
# so its base SELECT lives here and db_manager.py appends conditions —
# still always through parameterized "?"/":name" placeholders, never
# raw string interpolation of values.
SEARCH_EVENTS_BASE = "SELECT DISTINCT events.* FROM events"

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
