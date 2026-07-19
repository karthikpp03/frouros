"""
config/settings.py
==================
All project-wide constants, credentials, paths, thresholds, and device settings.
Nothing here should contain logic — pure configuration values only.
Imported by every other module that needs a constant.
"""

import os
import torch
from dotenv import load_dotenv

from utils.device import DEVICE

load_dotenv()

# ==================================================
# TELEGRAM
# ==================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ==================================================
# GROQ API
# ==================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ==================================================
# AI MODE ROUTING — Qwen (local, free) vs OpenAI Vision (cloud, paid)
# See services/summary_router.py for the single place this is used.
#
# USE_OPENAI=false               -> always Qwen, OpenAI is never
#                                    imported/initialized/called.
# USE_OPENAI=true  + face OFF    -> always OpenAI (testing mode, used
#                                    before the face-recognition model
#                                    exists — see face/recognizer.py).
# USE_OPENAI=true  + face ON     -> known face -> Qwen (free)
#                                    unknown face -> OpenAI (paid)
# ==================================================

USE_OPENAI              = os.getenv("USE_OPENAI", "false").strip().lower() == "true"
ENABLE_FACE_RECOGNITION = os.getenv("ENABLE_FACE_RECOGNITION", "false").strip().lower() == "true"

# Never hardcode a key or model name — always read from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# ==================================================
# MODEL PATHS / IDS
# ==================================================

YOLO_MODEL_PATH = "/app/models/yolo26m.pt"
QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
#QWEN_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
VIDEOMAE_MODEL   = "MCG-NJU/videomae-base"

# FastReID weight paths (adjust to your fastreid clone)
FASTREID_CONFIG  = "configs/MSMT17/bagtricks_R50.yml"
FASTREID_WEIGHTS = "pretrained/market_bot_R50.pth"

# ==================================================
# INPUT SOURCE
# ==================================================
# The rest of the project must keep working without modification no
# matter which of these is active — only the capture line in
# src/main.py reads these; no AI model is ever aware of the input
# source. Switch source ONLY through .env / these three values:
#
#   INPUT_MODE=video   VIDEO_PATH=/app/video/sample.mp4
#   INPUT_MODE=webcam  CAMERA_INDEX=0
#   INPUT_MODE=rtsp     RTSP_URL=rtsp://...
# ==================================================

INPUT_MODE   = os.getenv("INPUT_MODE", "video").strip().lower()   # "video" | "webcam" | "rtsp"
VIDEO_PATH   = os.getenv("VIDEO_PATH", "/app/video/sample.mp4")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
RTSP_URL     = os.getenv("RTSP_URL", "")
WEBCAM_DEVICE = os.getenv("WEBCAM_DEVICE", "/dev/video0")
# ==================================================
# DATA DIRECTORIES
# ==================================================

BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR          = os.path.join(BASE_DIR, "data")

EVENTS_DIR        = os.path.join(DATA_DIR, "events")
SMART_FRAMES_DIR  = os.path.join(DATA_DIR, "smart_frames")
PERSON_CROPS_DIR  = os.path.join(DATA_DIR, "person_crops")
DEBUG_DIR         = os.path.join(DATA_DIR, "debug_rejected")

# Per-event debugging artifacts (Smart Frame Selection + AI-processing
# audit trail) — see utils/debug_artifacts.py. One subfolder per event:
#   debug/event_<id>/{original_video.mp4, all_frames/, selected_frames/,
#                      videomae_scores.csv, frame_mapping.csv,
#                      processing_log.txt}
DEBUG_EVENTS_DIR  = os.path.join(DATA_DIR, "debug")

MEMORY_FILE       = os.path.join(DATA_DIR, "event_memory.json")
REID_GALLERY_FILE = os.path.join(DATA_DIR, "reid_gallery.json")

# ==================================================
# FACE RECOGNITION (InsightFace / ArcFace)
# ==================================================
# FACES_DIR: reference-photo folders, one per registered person —
#   faces/Dad/1.jpg, faces/Dad/2.jpg, faces/Mom/1.jpg, ...
#   face/face_db.build_face_database() scans this to build the database.
# FACE_DATABASE_FILE: the pre-built local face database (pickle) —
#   {person_name: {"embedding": np.ndarray, "registered_at": iso_str}}
#   — that face/recognizer.py loads exactly once at startup.
# FACE_RECOGNITION_THRESHOLD: minimum cosine similarity (ArcFace
#   normed embeddings, so cosine similarity == dot product) for a
#   match to count as a recognized person rather than "Unknown".
FACES_DIR                  = os.path.join(BASE_DIR, "faces")
FACE_DATABASE_FILE         = os.path.join(DATA_DIR, "face_database.pkl")
FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.45"))
INSIGHTFACE_HOME = os.getenv(
    "INSIGHTFACE_HOME",
    os.path.join(DATA_DIR, "insightface")
)

# Output folder for the merged (3-frames-in-1) images sent to OpenAI
# Vision. Only used on the OpenAI branch of services/summary_router.py
# — Qwen keeps using the 3 separate smart frames as before.
MERGED_EVENTS_DIR = os.path.join(DATA_DIR, "merged_events")

# ==================================================
# VIDEO SETTINGS
# ==================================================

FRAME_WIDTH  = 640
FRAME_HEIGHT = 360

# ==================================================
# DETECTION THRESHOLDS
# ==================================================

MIN_AREA       = 8000
MIN_CONFIDENCE = 0.6

# ==================================================
# REID THRESHOLDS
# ==================================================

REID_SIMILARITY_THRESHOLD = 0.70   # FastReID ResNet50
REID_GRACE_PERIOD         = 12     # seconds
REID_STALE_TIMEOUT        = 300    # seconds
NO_PERSON_TIMEOUT         = 8      # seconds

# ==================================================
# REID FEATURE DIMENSIONS (updated by reid.py at load time)
# ==================================================

REID_DIM = 2048   # ResNet50 default; overwritten to 512 for fallback

# ==================================================
# QUANTIZATION CONFIG
# Swap load_in_4bit → load_in_8bit for Jetson Orin
#
# 4-bit (bitsandbytes) quantization only works on CUDA — on a CPU-only
# system BNB_CONFIG is None and models/qwen_vl.py / models/smolvlm.py
# load the model unquantized (fp32) on CPU instead. This is the only
# device-dependent behaviour difference, and it exists because 4-bit
# quantization has no CPU backend, not because of a routing/pipeline
# change.
# ==================================================

from transformers import BitsAndBytesConfig

if DEVICE.type == "cuda":
    BNB_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
else:
    BNB_CONFIG = None

# ==================================================
# ROI POLYGON  (pixel coordinates, closed polygon)
# ==================================================
# Configurable — no longer hardcoded. tools/roi_selector.py writes
# data/roi.json; this loads it automatically on every startup (video
# file, webcam, or future RTSP — identically, since ROI_POINTS is the
# only thing any of them ever read). Falls back to the previous
# hardcoded polygon if data/roi.json doesn't exist yet or is invalid,
# so a fresh checkout with no roi.json still runs.
# ==================================================

import json
import numpy as np

ROI_CONFIG_FILE = os.path.join(DATA_DIR, "roi.json")

_DEFAULT_ROI_POINTS = [
(225, 357),
    (297, 320),
    (295, 262),
    (339, 243),
    (475, 274),
    (533, 278),
    (594, 191),
    (636, 207),
    (636, 357),
    (228, 356),
]


def _load_roi_points():
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, "r") as f:
                data = json.load(f)
            pts = data.get("points") or []
            if len(pts) >= 3:
                return np.array([tuple(p) for p in pts])
            print(f"[settings] {ROI_CONFIG_FILE} has fewer than 3 points — "
                  f"falling back to the default ROI_POINTS.")
        except Exception as e:
            print(f"[settings] Could not load {ROI_CONFIG_FILE} ({e}) — "
                  f"falling back to the default ROI_POINTS.")
    return np.array(_DEFAULT_ROI_POINTS)


ROI_POINTS = _load_roi_points()

