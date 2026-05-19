"""
main.py
=======
CCTV Surveillance System — v4  (modularized)

Startup order:
  1. Ensure data directories exist + seed empty memory file
  2. Load models  (YOLO → VideoMAE → ReID → Qwen → Groq)
  3. Load gallery (requires REID_DIM set by ReID loader)
  4. Start Telegram bot thread
  5. Open video capture
  6. Main tracking loop (identical to original monolith)
  7. Finalise last open event (if any)
  8. Cleanup, bot keepalive, test query

All logic inside the main loop is preserved verbatim — only imports changed.
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import os
import sys
import cv2
import time
import json
import torch

# ── ensure src/ is on the path when running as  python src/main.py ──────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── configuration ────────────────────────────────────────────────────────────
from config.settings import (
    VIDEO_PATH,
    EVENTS_DIR, SMART_FRAMES_DIR, PERSON_CROPS_DIR, DEBUG_DIR,
    MEMORY_FILE, REID_GALLERY_FILE,
    FRAME_WIDTH, FRAME_HEIGHT,
    MIN_AREA, MIN_CONFIDENCE,
    REID_GRACE_PERIOD, NO_PERSON_TIMEOUT,
    ROI_POINTS,
    CHAT_ID,
)

# ── models ────────────────────────────────────────────────────────────────────
from models.yolo_detector    import load_yolo, run_tracking
from models.videomae         import load_videomae
from models.reid             import load_reid, reid_fn as _reid_fn_placeholder
from models.qwen_vl          import load_qwen
from models.groq_query_engine import load_groq

# ── memory ────────────────────────────────────────────────────────────────────
from memory.gallery      import (
    gallery_load, gallery_save,
    gallery_match, gallery_was_recent, gallery_purge_stale,
)
from memory.event_memory import empty_event_record

# ── pipelines ─────────────────────────────────────────────────────────────────
import pipelines.event_pipeline as ep   # mutable state lives here

# ── telegram ─────────────────────────────────────────────────────────────────
from telegram.bot    import start_bot_thread, stop_bot
from telegram.alerts import tg_send_message

# ── utils ─────────────────────────────────────────────────────────────────────
from utils.roi_utils   import is_inside_roi, draw_roi
from utils.crop_utils  import crop_update
from utils.image_utils import try_update_best_frame, reset_best_frame
from utils.debug_utils import save_rejected_detection

# ── query pipeline (for the post-run test) ───────────────────────────────────
from pipelines.query_pipeline import query_memory

# ── torch perf flag ──────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True


# ==============================================================================
# 1. DATA DIRECTORIES + SEED MEMORY FILE
# ==============================================================================

os.makedirs(EVENTS_DIR,       exist_ok=True)
os.makedirs(SMART_FRAMES_DIR, exist_ok=True)
os.makedirs(PERSON_CROPS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR,        exist_ok=True)

if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump([], f)


# ==============================================================================
# 2. MODEL LOADING  (identical startup sequence to the original)
# ==============================================================================

load_qwen()         # Qwen2.5-VL-7B  — heaviest; load first so OOM surfaces early
load_groq()         # Groq / Llama-3.1-8B-instant
load_videomae()     # VideoMAE (CPU)
load_reid()         # FastReID → OSNet → ResNet18 fallback chain

# After load_reid() REID_DIM is finalised in config.settings — gallery can now build index
gallery_load()

# Import reid_fn *after* load_reid() so the module-level binding is resolved
from models.reid import reid_fn


# ==============================================================================
# 3. YOLO MODEL
# ==============================================================================

model = load_yolo()


# ==============================================================================
# 4. TELEGRAM BOT THREAD
# ==============================================================================

bot_thread = start_bot_thread()


# ==============================================================================
# 5. VIDEO CAPTURE
# ==============================================================================

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))


# ==============================================================================
# 6. EVENT / RECORDING STATE
# (mirrors the original module-level globals exactly)
# ==============================================================================

recording        = False
video_writer     = None
event_id         = 0
event_start_time = 0
output_path      = ""

last_detection_time = 0
active_reid_ids     = set()

frame_index = 0   # used by debug_utils


# ==============================================================================
# 7. MAIN TRACKING LOOP
# Logic is byte-for-byte identical to the original — only the inline helpers
# (draw_roi, save_rejected_detection, is_inside_roi, try_update_best_frame,
#  crop_update, gallery_match, gallery_was_recent, gallery_purge_stale,
#  empty_event_record, close_event, reset_event_state)
# are now imported from their respective modules.
# ==============================================================================

from datetime import datetime

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    draw_roi(frame)

    results = run_tracking(model, frame)

    detections_in_roi = []

    if results[0].boxes.id is not None:
        boxes       = results[0].boxes.xyxy.cpu()
        track_ids   = results[0].boxes.id.cpu().int().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        for box, track_id, confidence in zip(boxes, track_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)

            # ── DEBUG REJECTED DETECTIONS ──────────────────────────────────
            if area < MIN_AREA or confidence < MIN_CONFIDENCE:
                save_rejected_detection(
                    frame, frame_index, track_id, confidence, area,
                    x1, y1, x2, y2
                )
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if not is_inside_roi(cx, cy):
                continue

            detections_in_roi.append((x1, y1, x2, y2, track_id, confidence))

            score = area * confidence
            try_update_best_frame(frame, score)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    person_in_roi = len(detections_in_roi) > 0

    if person_in_roi:
        for (x1, y1, x2, y2, track_id, conf) in detections_in_roi:
            crop = frame[max(0, y1-10):y2+10, max(0, x1-10):x2+10]
            if crop.size == 0:
                continue

            # FastReID embedding
            emb          = reid_fn(crop)
            reid_id, sim = gallery_match(emb)
            active_reid_ids.add(reid_id)
            ep.track_to_reid_id[track_id] = reid_id

            # Best crop keyed by reid_id (dedup fix)
            crop_update(frame, track_id, reid_id, x1, y1, x2, y2, conf)

        last_detection_time = time.time()

        if not recording:
            print(f"[EVENT START] Event {event_id}")
            event_start_time    = time.time()
            output_path         = f"{EVENTS_DIR}/event_{event_id}.mp4"
            fourcc              = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer        = cv2.VideoWriter(
                output_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT)
            )
            recording           = True
            ep.current_event    = empty_event_record(
                event_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            ep.track_to_person_idx = {}
            ep.reid_to_person_idx  = {}
            ep.person_counter      = 0

        # ── DEDUP via reid_to_person_idx ──────────────────────────────────
        for (x1, y1, x2, y2, track_id, conf) in detections_in_roi:
            reid_id = ep.track_to_reid_id.get(track_id)
            if reid_id is None:
                continue

            if reid_id in ep.reid_to_person_idx:
                pidx = ep.reid_to_person_idx[reid_id]
                ep.track_to_person_idx[track_id] = pidx
                continue

            if track_id not in ep.track_to_person_idx:
                ep.person_counter += 1
                ep.track_to_person_idx[track_id] = ep.person_counter
                ep.reid_to_person_idx[reid_id]   = ep.person_counter

                from memory.event_memory import empty_person_record
                pid   = f"event{event_id}_person{ep.person_counter}"
                p_rec = empty_person_record(pid, track_id, event_id)
                p_rec["reid_id"]    = reid_id
                p_rec["first_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ep.current_event["persons"].append(p_rec)

    if recording:
        video_writer.write(frame)

    if recording:
        elapsed = time.time() - last_detection_time
        grace   = REID_GRACE_PERIOD if active_reid_ids else NO_PERSON_TIMEOUT

        if elapsed > grace:
            identity_returned = any(
                gallery_was_recent(rid) for rid in active_reid_ids
            )

            if identity_returned and elapsed < REID_GRACE_PERIOD * 2:
                pass
            else:
                print(f"[EVENT END] Event {event_id}")
                recording = False
                video_writer.release()

                ep.close_event(
                    event_id, output_path, event_start_time, ep.current_event
                )

                active_reid_ids  = set()
                ep.track_to_reid_id = {}
                ep.current_event = None
                event_id        += 1

                gallery_purge_stale()


# ==============================================================================
# 8. FINALIZE LAST EVENT  (unchanged)
# ==============================================================================

if recording:
    print(f"[FINAL EVENT END] Event {event_id}")
    recording = False
    video_writer.release()
    ep.close_event(event_id, output_path, event_start_time, ep.current_event)

cap.release()
if video_writer is not None:
    video_writer.release()

gallery_save()

print("\n[INFO] Video pipeline finished.")
print("[INFO] Telegram bot remains alive for queries. Press Ctrl+C to exit.\n")


# ==============================================================================
# 9. BOT KEEPALIVE + GRACEFUL SHUTDOWN
# ==============================================================================

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[INFO] Shutting down...")
    stop_bot()
    bot_thread.join(timeout=3)
    print("[INFO] Telegram bot stopped. Goodbye.")


# ==============================================================================
# 10. TEST QUERY  (unchanged)
# ==============================================================================

print("\n" + "=" * 50 + "\nAI QUERY TEST\n" + "=" * 50)
response, images = query_memory("Who carried a bag?")
print(response)
if images:
    print(f"\nMatching images: {images}")
