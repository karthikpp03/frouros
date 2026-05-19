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
GROQ_MODEL = "llama-3.1-8b-instant"

# ==================================================
# MODEL PATHS / IDS
# ==================================================

YOLO_MODEL_PATH  = "/content/drive/MyDrive/cctv/model/yolo26m.pt"
QWEN_MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"
VIDEOMAE_MODEL   = "MCG-NJU/videomae-base"

# FastReID weight paths (adjust to your fastreid clone)
FASTREID_CONFIG  = "configs/MSMT17/bagtricks_R50.yml"
FASTREID_WEIGHTS = "pretrained/market_bot_R50.pth"

# ==================================================
# VIDEO SOURCE
# ==================================================

VIDEO_PATH = "/content/drive/MyDrive/cctv/events/test.mp4"

# ==================================================
# DATA DIRECTORIES
# ==================================================

BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR          = os.path.join(BASE_DIR, "data")

EVENTS_DIR        = os.path.join(DATA_DIR, "events")
SMART_FRAMES_DIR  = os.path.join(DATA_DIR, "smart_frames")
PERSON_CROPS_DIR  = os.path.join(DATA_DIR, "person_crops")
DEBUG_DIR         = os.path.join(DATA_DIR, "debug_rejected")

MEMORY_FILE       = os.path.join(DATA_DIR, "event_memory.json")
REID_GALLERY_FILE = os.path.join(DATA_DIR, "reid_gallery.json")

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
# ==================================================

from transformers import BitsAndBytesConfig

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ==================================================
# ROI POLYGON  (pixel coordinates, closed polygon)
# ==================================================

import numpy as np

ROI_POINTS = np.array([
    (3, 226),
    (37, 231),
    (202, 185),
    (232, 155),
    (396, 151),
    (383, 170),
    (392, 183),
    (498, 355),
    (3, 355),
    (2, 225),
])
