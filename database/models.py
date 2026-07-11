"""
database/models.py
===================
Plain Python dataclasses mirroring every table in schema.sql, one-to-one.

These are pure data containers — no SQL, no I/O, no business logic.
database/db_manager.py converts between these and sqlite3.Row objects.
Keeping them here (rather than inline dicts) gives Phase 2 callers
type hints, autocomplete, and a single place to update if a column is
ever added or renamed.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Camera:
    camera_id: str
    camera_name: str
    location: Optional[str] = None
    roi_name: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Event:
    event_id: str
    camera_id: Optional[str] = None
    ai_provider: Optional[str] = None          # 'qwen' | 'openai' | 'none'
    event_type: Optional[str] = None
    event_date: Optional[str] = None           # 'YYYY-MM-DD'
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    summary: Optional[str] = None
    confidence: Optional[float] = None
    video_path: Optional[str] = None
    merged_image_path: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Person:
    event_id: str
    person_id: Optional[int] = None            # auto-assigned on insert
    known_status: Optional[str] = None         # 'known' | 'unknown'
    face_id: Optional[str] = None
    person_name: Optional[str] = None
    gender: Optional[str] = None
    estimated_age: Optional[str] = None
    height: Optional[str] = None
    body_build: Optional[str] = None
    top_clothing: Optional[str] = None
    bottom_clothing: Optional[str] = None
    footwear: Optional[str] = None
    headwear: Optional[str] = None
    hair: Optional[str] = None
    beard: Optional[str] = None
    glasses: Optional[str] = None
    mask: Optional[str] = None
    accessories: Optional[str] = None
    bag: Optional[str] = None
    dominant_hand: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class Movement:
    event_id: str
    movement_id: Optional[int] = None          # auto-assigned on insert
    entry_point: Optional[str] = None
    exit_point: Optional[str] = None
    direction: Optional[str] = None
    path: Optional[str] = None                 # JSON-encoded list of points
    speed: Optional[float] = None
    final_position: Optional[str] = None
    loitering: Optional[bool] = None
    duration: Optional[float] = None


@dataclass
class Action:
    event_id: str
    sequence_number: int
    action: str
    action_id: Optional[int] = None            # auto-assigned on insert
    timestamp: Optional[str] = None


@dataclass
class ObjectRecord:
    """Represents a row in the `objects` table. Named ObjectRecord
    (rather than Object) to avoid shadowing the Python builtin."""
    event_id: str
    object_type: str
    object_id: Optional[int] = None            # auto-assigned on insert
    description: Optional[str] = None
    held_by_person: Optional[int] = None       # FK -> Person.person_id
    confidence: Optional[float] = None


@dataclass
class Vehicle:
    event_id: str
    vehicle_id: Optional[int] = None           # auto-assigned on insert
    vehicle_type: Optional[str] = None
    color: Optional[str] = None
    number_plate: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class Keyword:
    event_id: str
    keyword: str
    keyword_id: Optional[int] = None           # auto-assigned on insert
