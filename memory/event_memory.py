"""
memory/event_memory.py
======================
Manages the structured JSON event memory (event_memory.json).

Exports:
  empty_person_record(person_id, track_id, ev_id)  → dict
  empty_event_record(ev_id, timestamp)              → dict
  load_memory()                                     → list
  save_memory_append(event_record)

All logic is preserved verbatim from the original monolith.
"""

import os
import json
from config.settings import MEMORY_FILE


def empty_person_record(person_id, track_id, ev_id):
    return {
        "person_id":      person_id,
        "track_id":       track_id,
        "reid_id":        None,
        "appearance":     None,
        "actions":        [],
        "objects":        [],
        "movement":       None,
        "waiting":        False,
        "interaction":    None,
        "first_seen":     None,
        "last_seen":      None,
        "crop_image":     None,
        "frames":         [],
        "reid_embedding": None,
    }


def empty_event_record(ev_id, timestamp):
    return {
        "event_id":  ev_id,
        "timestamp": timestamp,
        "duration":  0,
        "summary":   "",
        "snapshot":  None,
        "video":     None,
        "persons":   [],
    }


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memory_append(event_record):
    memory = load_memory()
    memory = [e for e in memory if e["event_id"] != event_record["event_id"]]
    memory.append(event_record)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)
