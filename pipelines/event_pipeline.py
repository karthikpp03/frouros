"""
pipelines/event_pipeline.py
============================
Per-event lifecycle state and the close_event() routine.

State managed here (mirrors the originals — all mutable, reset on close):
  current_event        — dict | None
  track_to_reid_id     — {track_id: reid_id}
  track_to_person_idx  — {track_id: person_index}
  reid_to_person_idx   — {reid_id: person_index}
  person_counter       — int

Exports:
  close_event(ev_id, ev_output_path, ev_start_time, ev_record)
  reset_event_state()

close_event() is preserved verbatim from the original monolith.
"""

import cv2
import time
from datetime import datetime

from config.settings           import SMART_FRAMES_DIR, CHAT_ID
from memory.event_memory       import save_memory_append, empty_person_record
from memory.gallery            import gallery_save
from models.videomae           import extract_smart_frames
from pipelines.summary_pipeline import generate_summary, extract_person_attributes
from telegram.alerts           import send_telegram_alert, tg_send_message
from utils.crop_utils          import crop_save_by_reid, crop_clear
from utils.image_utils         import get_best_frame, reset_best_frame


# --------------------------------------------------
# PER-EVENT MUTABLE STATE
# Exposed as module-level vars so main.py can read/write them
# exactly as the original globals were.
# --------------------------------------------------

current_event       = None
track_to_reid_id    = {}   # track_id → reid_id
track_to_person_idx = {}   # track_id → person index in current_event
reid_to_person_idx  = {}   # reid_id  → person index (dedup key)
person_counter      = 0


def reset_event_state():
    """Reset all per-event tracking state.  Called at the end of close_event."""
    global current_event, track_to_reid_id, track_to_person_idx
    global reid_to_person_idx, person_counter
    current_event       = None
    track_to_reid_id    = {}
    track_to_person_idx = {}
    reid_to_person_idx  = {}
    person_counter      = 0


# --------------------------------------------------
# CLOSE EVENT  (verbatim from original monolith)
# --------------------------------------------------

def close_event(ev_id, ev_output_path, ev_start_time, ev_record):
    """
    Finalise a completed event:
      1. Write snapshot
      2. Send immediate Telegram alert
      3. Extract VideoMAE smart frames
      4. Generate Qwen summary
      5. Send full summary to Telegram
      6. Save best crop per reid_id
      7. Extract person attributes
      8. Persist to JSON memory
      9. Reset all per-event state
    """
    global track_to_person_idx, reid_to_person_idx, person_counter

    ev_end      = time.time()
    ev_duration = int(ev_end - ev_start_time)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    best_frame = get_best_frame()

    if best_frame is None:
        print(f"[WARNING] Event {ev_id} has no best_frame — skipping close.")
        return

    # STEP 1: snapshot
    snapshot_path = f"{SMART_FRAMES_DIR}/event_{ev_id}.jpg"
    write_ok      = cv2.imwrite(snapshot_path, best_frame)
    if not write_ok:
        print(f"[WARNING] Snapshot write failed for Event {ev_id}")

    # STEP 2: immediate Telegram alert
    if write_ok:
        send_telegram_alert(snapshot_path, "⏳ Processing event summary...",
                            ev_duration, timestamp, ev_id)

    # STEP 3: VideoMAE
    smart_frames = extract_smart_frames(ev_output_path, ev_id)

    # STEP 4: Qwen summary
    summary = generate_summary(smart_frames) if smart_frames else "No frames available."
    print("\n" + "=" * 50 + "\nAI SUMMARY\n" + "=" * 50)
    print(summary)

    # STEP 5: full summary message
    if write_ok:
        tg_send_message(CHAT_ID, f"📋 *Event #{ev_id} Full Report*\n\n{summary[:1000]}")

    # STEP 6: Save best crop PER REID IDENTITY (dedup fix)
    seen_reid_ids = set()
    for p in ev_record["persons"]:
        reid_id = p.get("reid_id")
        if reid_id and reid_id not in seen_reid_ids:
            seen_reid_ids.add(reid_id)
            pidx      = reid_to_person_idx.get(reid_id, p.get("track_id", 0))
            crop_path = crop_save_by_reid(reid_id, ev_id, pidx)
            p["crop_image"] = crop_path
            p["last_seen"]  = timestamp
            p["frames"]     = smart_frames[:3]

    # STEP 7: Attribute extraction
    persons_attrs  = extract_person_attributes(summary)
    actual_persons = ev_record["persons"]

    for i, attrs in enumerate(persons_attrs):
        if attrs is None or not isinstance(attrs, dict):
            continue
        if i < len(actual_persons):
            p = actual_persons[i]
        else:
            p = empty_person_record(f"event{ev_id}_person{i+1}", -1, ev_id)
            actual_persons.append(p)

        p["appearance"] = attrs.get("appearance")
        p["actions"]    = attrs.get("actions", [])
        p["objects"]    = attrs.get("objects", [])
        p["movement"]   = attrs.get("movement")
        p["waiting"]    = attrs.get("waiting", False)
        if not p["first_seen"]:
            p["first_seen"] = timestamp
        p["last_seen"] = timestamp

    # STEP 8: Finalise
    ev_record["summary"]   = summary
    ev_record["duration"]  = ev_duration
    ev_record["timestamp"] = timestamp
    ev_record["snapshot"]  = snapshot_path
    ev_record["video"]     = ev_output_path

    save_memory_append(ev_record)
    gallery_save()

    # STEP 9: Reset
    reset_best_frame()
    crop_clear()
    reset_event_state()

    print(f"[INFO] Event {ev_id} saved with {len(actual_persons)} person records.")
