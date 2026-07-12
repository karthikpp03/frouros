"""
main.py
=======
CCTV Surveillance System — v5  (modularized + parallel Event Manager)

Startup order:
  1. Ensure data directories exist + seed empty memory file
  2. Load models  (Groq → ReID)  — Qwen and VideoMAE are NO LONGER loaded
     here; see the GPU memory note below.
  3. Load gallery (requires REID_DIM set by ReID loader)
  4. Load YOLO
  5. Start Telegram bot thread
  6. Open video capture
  7. Main tracking loop — real-time detection/tracking/ReID. Every Track ID
     seen in the ROI gets its own independent Event via
     pipelines.event_manager.event_manager: its own recording, its own best
     frame/crop selection, and its own AI finalisation, all running without
     ever blocking the live feed or each other. AI processing (VideoMAE,
     Qwen, Groq, Telegram) happens asynchronously on a background worker
     thread — see pipelines/event_manager.py for the full design.
  8. Finalise any Events still open when the video ends
  9. Cleanup, bot keepalive, test query

Detection/tracking/ROI/ReID logic is preserved verbatim from the original
monolith — only the event lifecycle (previously one shared global recording
slot) has changed, to support multiple concurrent, non-blocking Events.

GPU MEMORY OPTIMISATION
------------------------
YOLO (continuous, every frame) and ReID (CPU-resident) still load here at
startup, same as before. Qwen2.5-VL-7B and VideoMAE, however, are NO LONGER
loaded eagerly here — they used to sit on the GPU for the entire process
lifetime alongside YOLO, leaving too little headroom for Qwen's
`.generate()` activation spike and causing CUDA OOM. They are now loaded
on-demand inside pipelines/event_manager.py._finalize_event() (only for the
brief window they're actually needed, one at a time, never together), and
fully released again immediately after. See models/qwen_vl.py and
models/videomae.py for the load_*()/unload_*() pair each one now exposes.
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
    VIDEO_PATH, INPUT_MODE, CAMERA_INDEX, RTSP_URL,WEBCAM_DEVICE,
    EVENTS_DIR, SMART_FRAMES_DIR, PERSON_CROPS_DIR, DEBUG_DIR,
    MEMORY_FILE, REID_GALLERY_FILE,
    FRAME_WIDTH, FRAME_HEIGHT,
    MIN_AREA, MIN_CONFIDENCE,
    ROI_POINTS,
    CHAT_ID,
)

# ── models ────────────────────────────────────────────────────────────────────
from models.yolo_detector    import load_yolo, run_tracking
from models.reid             import load_reid, reid_fn as _reid_fn_placeholder
from models.groq_query_engine import load_groq
# NOTE: models.qwen_vl.load_qwen() and models.videomae.load_videomae() are
# intentionally NOT imported/called here anymore. Both are heavy models that
# used to be loaded once at startup and held on the GPU for the entire
# process lifetime, alongside YOLO — this permanent co-residency was the
# root cause of Qwen's CUDA OOM (too little headroom left during
# `.generate()`). They are now loaded on-demand, one at a time, inside
# pipelines/event_manager.py._finalize_event() and fully released right
# after use. See models/qwen_vl.py and models/videomae.py for details.

# ── memory ────────────────────────────────────────────────────────────────────
from memory.gallery      import gallery_load, gallery_save, gallery_match

# ── pipelines ─────────────────────────────────────────────────────────────────
# NOTE: pipelines/event_pipeline.py (the single-global-event pipeline) is kept
# in place untouched, but main.py no longer drives the event lifecycle through
# it directly. It has been superseded by pipelines/event_manager.py, which
# gives every Track ID its own independent Event (own recording, own best
# frame/crop, own AI finalisation) so multiple people in the ROI can be
# tracked and summarised in parallel without blocking each other or the live
# feed. See pipelines/event_manager.py for the full design rationale.
from pipelines.event_manager import event_manager

# ── telegram ─────────────────────────────────────────────────────────────────
from telegram.bot    import start_bot_thread, stop_bot

# ── utils ─────────────────────────────────────────────────────────────────────
from utils.roi_utils   import is_inside_roi, draw_roi
from utils.debug_utils import save_rejected_detection

# ── query pipeline (for the post-run test) ───────────────────────────────────
from pipelines.query_pipeline import query_memory

# ── torch perf flag ──────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True

# ── device diagnostics (printed once, at startup) ────────────────────────────
from utils.device import log_device_info
log_device_info()


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

# ── Phase 2 (additive): SQLite database bootstrap ────────────────────────────
# Creates data/frouros.db + every table in database/schema.sql if missing
# (safe/idempotent — every CREATE TABLE uses IF NOT EXISTS). Nothing else
# in this file changes: pipelines/event_manager.py writes to this database
# on its own via services/db_writer.py, and pipelines/query_pipeline.py
# reads from it for Telegram/Groq queries.
from database import DatabaseManager, Camera

_db = DatabaseManager()
_db.create_database()
_db.create_tables()
_db.insert_camera(Camera(
    camera_id="cam_default",
    camera_name="Default Camera",
    location=None,
    roi_name="main_roi",
))


# ==============================================================================
# 2. MODEL LOADING
# Qwen2.5-VL-7B and VideoMAE are intentionally NOT loaded here anymore — see
# the "GPU MEMORY OPTIMISATION" note in the module docstring above. They are
# loaded on-demand, one at a time, during event finalisation instead.
# ==============================================================================

load_groq()         # Groq / Llama-3.1-8B-instant (cloud API — no local GPU/RAM cost)
load_reid()         # FastReID → OSNet → ResNet18 fallback chain (CPU-resident)

# Face recognition (InsightFace / ArcFace) — loaded exactly once here,
# and only when it could actually be used (USE_OPENAI=true AND
# ENABLE_FACE_RECOGNITION=true, matching services/summary_router.py's
# own condition). A complete no-op otherwise: InsightFace is never
# imported, the face database is never read. See face/recognizer.py.
from face.recognizer import load_face_recognizer
load_face_recognizer()

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
# Source is chosen ONLY via INPUT_MODE (config/settings.py / .env) —
# "video" (VIDEO_PATH), "webcam" (CAMERA_INDEX), or "rtsp" (RTSP_URL).
# Nothing downstream (ROI, tracking, events, AI) is aware of which one
# is active; only this capture line changes.
# ==============================================================================

if INPUT_MODE == "webcam":
    print(f"[INFO] Input source: webcam ({WEBCAM_DEVICE})")
    cap = cv2.VideoCapture(WEBCAM_DEVICE)
elif INPUT_MODE == "rtsp":
    print(f"[INFO] Input source: RTSP ({RTSP_URL})")
    cap = cv2.VideoCapture(RTSP_URL)
else:
    print(f"[INFO] Input source: video file ({VIDEO_PATH})")
    cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20   # webcams/RTSP often report 0


# ==============================================================================
# 6. FRAME COUNTER
# (recording / event-id / best-frame / best-crop state used to live here as
#  module-level globals sized for ONE concurrent event. That state now lives
#  per-Track-ID inside pipelines/event_manager.py's TrackEvent instances, so
#  multiple people in the ROI get independent, non-interfering Events.)
# ==============================================================================

frame_index = 0   # used by debug_utils


# ==============================================================================
# 7. MAIN TRACKING LOOP
# Detection / ROI-filtering / debug-rejects / ROI drawing logic is unchanged.
# The single shared "recording" slot has been replaced by pipelines/event_
# manager.event_manager, which gives every Track ID its own independent Event
# (own VideoWriter, own best frame/crop, own AI finalisation running on a
# background thread) — so this loop never waits on AI, and N people in the
# ROI get N events running in parallel instead of one shared/blocking event.
# ==============================================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    now   = time.time()
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

            print(f"[YOLO] track_id={track_id} conf={confidence:.2f} "
                  f"box=({x1},{y1},{x2},{y2}) area={area}")

            # ── DEBUG REJECTED DETECTIONS ──────────────────────────────────
            if area < MIN_AREA or confidence < MIN_CONFIDENCE:
                reason = "LOW_CONFIDENCE" if confidence < MIN_CONFIDENCE else "MIN_BOX_SIZE"
                print(f"    -> REJECTED ({reason})")
                save_rejected_detection(
                    frame, frame_index, track_id, confidence, area,
                    x1, y1, x2, y2
                )
                continue

            # ROI containment must be tested against the person's FEET
            # (bottom-center of the box), not the box's geometric center.
            # ROI_POINTS traces a floor-plane polygon (see config/settings.py
            # / data/roi.json) — it marks out an area of *floor*, not a
            # region of image-space centered on a standing person's torso.
            # A standing adult's bbox center sits roughly at chest/waist
            # height, which is well above the floor-level polygon, so
            # cv2.pointPolygonTest on the box center fails even when the
            # person is standing squarely inside the ROI on the ground.
            # The bottom-center point is where the person actually contacts
            # the floor, which is what the polygon represents.
            cx, cy = (x1 + x2) // 2, y2
            inside = is_inside_roi(cx, cy)
            print(f"    -> ROI point=({cx},{cy}) inside={inside}")
            if not inside:
                print("    -> REJECTED (OUTSIDE_ROI)")
                save_rejected_detection(
                    frame, frame_index, track_id, confidence, area,
                    x1, y1, x2, y2
                )
                continue

            print("    -> ACCEPTED")


            # ── ReID: identify WHO this track_id belongs to (unchanged) ────
            crop = frame[max(0, y1-10):y2+10, max(0, x1-10):x2+10]
            reid_id = None
            if crop.size:
                emb          = reid_fn(crop)
                reid_id, sim = gallery_match(emb)

            detections_in_roi.append(
                (x1, y1, x2, y2, track_id, confidence, reid_id)
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ── EVENT MANAGER: create/update one independent Event per Track ID ────
    # Non-blocking — only opens VideoWriters and updates in-memory trackers.
    # A brand-new Track ID gets a brand-new Event immediately; an existing
    # Track ID just gets its own event fed this frame.
    # NOTE: called every frame, even when detections_in_roi is empty for
    # THIS specific frame. A momentary missed YOLO detection (occlusion,
    # motion blur, a confidence dip) does not mean the active Scene Event
    # has ended — tick()/grace-period logic below is what actually decides
    # that. If this call is skipped on frames with no detection, the scene
    # recording silently drops those frames, producing a video with far
    # fewer frames than the scene's real wall-clock duration.
    event_manager.update_detections(frame, detections_in_roi, fps, now)

    # ── Close any Track ID that left the ROI / timed out. Each event is
    # evaluated and closed independently — closing one never blocks or
    # delays any other still-active event. Closed events are handed off to
    # a background worker for AI summarisation; this call returns instantly.
    event_manager.tick(now)


# ==============================================================================
# 8. FINALIZE ALL STILL-OPEN EVENTS
# (generalises the original single "finalize last event" step to however
#  many Track IDs are still active when the video ends)
# ==============================================================================

event_manager.close_all()

cap.release()

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