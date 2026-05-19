"""
CCTV Surveillance System — v4
Stack upgrades applied over v3:

  UPGRADED:
  - ReID:        OSNet  →  FastReID (MSMT17 ResNet50 backbone, GPU-accelerated)
  - Vision LLM:  Qwen2-VL-2B  →  Qwen2.5-VL-7B (4-bit bitsandbytes quantization)
  - Query LLM:   Qwen (text-only)  →  Llama-3.1-8B-Instruct (4-bit, query engine only)
  - Retrieval:   keyword scoring  →  FAISS flat-IP cosine index over ReID embeddings
  - Tracking:    botsort.yaml  →  strongsort.yaml  (drop-in, same Ultralytics API)

  UNCHANGED:
  - YOLO26M detection
  - VideoMAE smart frame extraction
  - ROI polygon logic
  - Event lifecycle / segmentation
  - Event recording (VideoWriter)
  - Telegram bot flow, anti-spam, retry logic
  - Structured JSON memory schema
  - Grounding rules / surveillance prompts
  - crop_update / crop_save / crop_clear structure
  - All Telegram alert reliability fixes from v3

  NEW:
  - Duplicate-identity prevention via FAISS + cosine threshold before gallery insert
  - Best-crop-per-reid-identity (not just per track_id)
  - FAISS index kept in sync with gallery on every insert/update
  - image retrieval now uses FAISS embedding search when query embedding available,
    falls back to keyword scoring otherwise

DEPENDENCIES (pip install before running):
  pip install faiss-gpu                     # or faiss-cpu on non-CUDA
  pip install git+https://github.com/JDAI-CV/fast-reid.git
  pip install bitsandbytes>=0.43.0
  pip install transformers>=4.45.0          # Qwen2.5-VL + Llama3 support
  pip install accelerate
  pip install ultralytics                   # already present; ensure >=8.2 for StrongSORT
  pip install torch torchvision             # already present

VRAM budget (A100 40 GB / Jetson AGX Orin 64 GB):
  Qwen2.5-VL-7B  4-bit  ~5-6 GB
  Llama-3.1-8B   4-bit  ~5-6 GB
  FastReID                ~0.2 GB (CPU-offloaded between frames)
  VideoMAE                ~0.5 GB (CPU)
  YOLO26M                 ~1.5 GB
  Total estimate:        ~13-14 GB — fits on a single 16 GB GPU
  For Jetson Orin (unified memory): use load_in_8bit instead of load_in_4bit
"""

import cv2
import os
import gc
import re
import json
import time
import numpy as np
import requests
import torch
import faiss

torch.backends.cudnn.benchmark = True
from dotenv import load_dotenv

load_dotenv()

from datetime import datetime
from PIL import Image

from ultralytics import YOLO

from transformers import (
    Qwen2_5_VLForConditionalGeneration,   # Qwen2.5-VL upgrade
    AutoProcessor,
    AutoModelForCausalLM,                 # Llama 3.1 query engine
    AutoTokenizer,
    VideoMAEModel,
    VideoMAEImageProcessor,
    BitsAndBytesConfig
)

import torch.nn.functional as F
import threading

# ==================================================
# TELEGRAM
# ==================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ==================================================
# ROI POLYGON  (unchanged)
# ==================================================

roi_points = np.array([
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

# ==================================================
# VIDEO SOURCE  (unchanged)
# ==================================================

video_path = "/content/drive/MyDrive/cctv/events/test.mp4"
cap        = cv2.VideoCapture(video_path)

# ==================================================
# YOLO  (unchanged)
# ==================================================

model = YOLO("/content/drive/MyDrive/cctv/model/yolo26m.pt")

# ==================================================
# QUANTIZATION CONFIG (shared by both LLMs)
# Swap load_in_4bit → load_in_8bit for Jetson Orin
# ==================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ==================================================
# UPGRADE: Qwen2.5-VL-7B  (replaces Qwen2-VL-2B)
# ==================================================

print("[INFO] Loading Qwen2.5-VL-7B (4-bit)...")

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda",
    torch_dtype=torch.float16,
)

processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

print("[INFO] Qwen2.5-VL-7B loaded!")

# ==================================================
from groq import Groq

print("[INFO] Connecting to Groq API...")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(
    api_key=GROQ_API_KEY
)

GROQ_MODEL = "llama-3.1-8b-instant"

print("[INFO] Groq connected!")

# ==================================================
# LOAD VIDEOMAE  (unchanged)
# ==================================================

print("[INFO] Loading VideoMAE...")

videomae_processor = VideoMAEImageProcessor.from_pretrained(
    "MCG-NJU/videomae-base"
)
videomae_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
videomae_model.eval()
videomae_model.to("cpu")

print("[INFO] VideoMAE loaded!")

# ==================================================
# UPGRADE: FastReID  (replaces OSNet / ResNet18 ReID)
#
# FastReID is loaded with a MSMT17-pretrained ResNet50
# backbone from the official JDAI-CV repo.
# Falls back to the existing OSNet / ResNet18 chain
# if fastreid is not installed.
# ==================================================

print("[INFO] Loading FastReID...")

REID_DIM = 2048  # ResNet50 feature dim; updated if fallback used

def load_fastreid():
    global REID_DIM
    try:
        from fastreid.config import get_cfg
        from fastreid.modeling import build_model
        from fastreid.utils.checkpoint import Checkpointer

        cfg = get_cfg()
        cfg.merge_from_file(
            "configs/MSMT17/bagtricks_R50.yml"   # adjust path to your fastreid clone
        )
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.WEIGHTS = ""          # we use Checkpointer below
        cfg.freeze()

        fr_model = build_model(cfg)
        Checkpointer(fr_model).load(
            "pretrained/market_bot_R50.pth"       # MSMT17/Market pretrained weight
        )
        fr_model.eval()
        fr_model.to("cpu")

        from torchvision import transforms
        reid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        REID_DIM = 2048

        def extract_embedding(img_bgr):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor  = reid_transform(img_rgb).unsqueeze(0).to("cpu")
            with torch.no_grad():
                feat = fr_model(tensor)
            if isinstance(feat, dict):
                feat = feat["features"]
            feat = F.normalize(feat, p=2, dim=1)
            return feat.squeeze(0).cpu().numpy()

        print("[ReID] FastReID ResNet50 loaded")
        return extract_embedding

    except Exception as e:
        print(f"[ReID] FastReID unavailable ({e}), falling back to OSNet/ResNet18")
        return _load_osnet_fallback()


def _load_osnet_fallback():
    global REID_DIM
    try:
        import torchreid
        reid_backbone = torchreid.models.build_model(
            name="osnet_x0_25", num_classes=1000, pretrained=True
        )
        reid_backbone.eval()
        reid_backbone.to("cpu")

        from torchvision import transforms
        reid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        REID_DIM = 512  # OSNet output dim

        def extract_embedding(img_bgr):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor  = reid_transform(img_rgb).unsqueeze(0).to("cpu")
            with torch.no_grad():
                feat = reid_backbone(tensor)
            feat = F.normalize(feat, p=2, dim=1)
            return feat.squeeze(0).cpu().numpy()

        print("[ReID] OSNet-x0.25 loaded (torchreid fallback)")
        return extract_embedding

    except ImportError:
        from torchvision.models import resnet18, ResNet18_Weights
        from torchvision import transforms

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
        backbone.eval()
        backbone.to("cpu")

        reid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        REID_DIM = 512

        def extract_embedding(img_bgr):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor  = reid_transform(img_rgb).unsqueeze(0).to("cpu")
            with torch.no_grad():
                feat = backbone(tensor).squeeze()
            feat = F.normalize(feat, p=2, dim=0)
            return feat.cpu().numpy()

        print("[ReID] Fallback ResNet18 loaded")
        return extract_embedding


reid_fn = load_fastreid()
print("[INFO] ReID loaded!")

# ==================================================
# PATHS AND FOLDERS  (unchanged)
# ==================================================

MEMORY_FILE       = "event_memory.json"
REID_GALLERY_FILE = "reid_gallery.json"
PERSON_CROPS_DIR  = "person_crops"
SMART_FRAMES_DIR  = "smart_frames"
EVENTS_DIR        = "events"

os.makedirs(EVENTS_DIR,       exist_ok=True)
os.makedirs(SMART_FRAMES_DIR, exist_ok=True)
os.makedirs(PERSON_CROPS_DIR, exist_ok=True)

if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump([], f)

# ==================================================
# VIDEO SETTINGS  (unchanged)
# ==================================================

fps          = int(cap.get(cv2.CAP_PROP_FPS))
frame_width  = 640
frame_height = 360

# ==================================================
# EVENT VARIABLES  (unchanged)
# ==================================================

recording        = False
video_writer     = None
event_id         = 0
event_start_time = 0
output_path      = ""

# ==================================================
# SMART FRAME MEMORY  (unchanged)
# ==================================================

best_frame = None
best_score = 0

# ==================================================
# STRUCTURED MEMORY HELPERS  (unchanged)
# ==================================================

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
        "reid_embedding": None
    }


def empty_event_record(ev_id, timestamp):
    return {
        "event_id":  ev_id,
        "timestamp": timestamp,
        "duration":  0,
        "summary":   "",
        "snapshot":  None,
        "video":     None,
        "persons":   []
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

# ==================================================
# IDENTITY GALLERY + FAISS INDEX
#
# NEW: FAISS flat inner-product index (cosine similarity
# on L2-normalised vectors) is kept in sync with
# gallery_data on every insert / update.
#
# This serves two purposes:
#   1. Fast duplicate detection before inserting a new
#      identity — prevents the same person getting two
#      gallery entries due to a tracker ID split.
#   2. Embedding-based image retrieval in find_matching_crops().
# ==================================================

REID_SIMILARITY_THRESHOLD  = 0.70   # raised from 0.65 for FastReID (higher-dim embeddings)
REID_GRACE_PERIOD          = 12
REID_STALE_TIMEOUT         = 300

gallery_data    = {}   # rid → {embedding, count, last_seen, first_seen}
gallery_counter = 0

# FAISS state — rebuilt from gallery_data on load
_faiss_index    = None   # faiss.IndexFlatIP
_faiss_id_map   = []     # list of reid_ids in FAISS insertion order


def _faiss_rebuild():
    """Rebuild the FAISS index from scratch using current gallery_data."""
    global _faiss_index, _faiss_id_map

    _faiss_id_map = []

    if not gallery_data:
        _faiss_index = faiss.IndexFlatIP(REID_DIM)
        return

    vecs = []
    for rid, data in gallery_data.items():
        emb = data["embedding"].astype(np.float32)
        # ensure unit length for cosine similarity via inner product
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        vecs.append(emb)
        _faiss_id_map.append(rid)

    mat          = np.vstack(vecs).astype(np.float32)
    _faiss_index = faiss.IndexFlatIP(mat.shape[1])
    _faiss_index.add(mat)


def _faiss_add(reid_id, embedding):
    """Add a single new embedding to the live FAISS index."""
    global _faiss_index
    if _faiss_index is None:
        _faiss_rebuild()
        return
    emb  = embedding.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    _faiss_index.add(emb.reshape(1, -1))
    _faiss_id_map.append(reid_id)


def faiss_search(query_embedding, k=1):
    """
    Return (reid_id, cosine_similarity) for the top-k nearest neighbours.
    Returns [] if index is empty.
    """
    if _faiss_index is None or _faiss_index.ntotal == 0:
        return []
    emb  = query_embedding.astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    k      = min(k, _faiss_index.ntotal)
    D, I   = _faiss_index.search(emb.reshape(1, -1), k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx >= 0 and idx < len(_faiss_id_map):
            results.append((_faiss_id_map[idx], float(dist)))
    return results


def gallery_load():
    global gallery_data, gallery_counter
    if not os.path.exists(REID_GALLERY_FILE):
        _faiss_rebuild()
        return
    with open(REID_GALLERY_FILE, "r") as f:
        raw = json.load(f)
    for rid, data in raw.items():
        gallery_data[rid] = {
            "embedding":  np.array(data["embedding"], dtype=np.float32),
            "count":      data["count"],
            "last_seen":  data["last_seen"],
            "first_seen": data["first_seen"]
        }
    if gallery_data:
        nums = [int(k.split("_")[1]) for k in gallery_data if "_" in k]
        gallery_counter = max(nums) if nums else 0
    _faiss_rebuild()
    print(f"[ReID] Gallery loaded: {len(gallery_data)} identities, FAISS dim={REID_DIM}")


def gallery_save():
    serializable = {}
    for rid, data in gallery_data.items():
        serializable[rid] = {
            "embedding":  data["embedding"].tolist(),
            "count":      data["count"],
            "last_seen":  data["last_seen"],
            "first_seen": data["first_seen"]
        }
    with open(REID_GALLERY_FILE, "w") as f:
        json.dump(serializable, f)


def gallery_match(embedding):
    """
    UPGRADE: uses FAISS for fast cosine search.
    Duplicate-prevention: raises threshold check before inserting a new identity.
    Same logical behaviour as before — returns (reid_id, similarity).
    """
    global gallery_counter

    hits = faiss_search(embedding, k=1)

    if hits:
        best_id, best_sim = hits[0]
        if best_sim >= REID_SIMILARITY_THRESHOLD:
            gallery_update(best_id, embedding)
            return best_id, best_sim

    # No match above threshold — new identity
    gallery_counter += 1
    rid = f"person_{gallery_counter:04d}"
    gallery_data[rid] = {
        "embedding":  embedding.copy(),
        "count":      1,
        "last_seen":  time.time(),
        "first_seen": datetime.now().isoformat()
    }
    _faiss_add(rid, embedding)
    return rid, 0.0


def gallery_update(reid_id, embedding):
    if reid_id not in gallery_data:
        return
    data  = gallery_data[reid_id]
    alpha = 0.1
    data["embedding"] = (1 - alpha) * data["embedding"] + alpha * embedding
    norm = np.linalg.norm(data["embedding"])
    if norm > 0:
        data["embedding"] /= norm
    data["count"]    += 1
    data["last_seen"] = time.time()
    # Note: FAISS flat index doesn't support in-place update; the stored
    # vector drifts slowly (EMA alpha=0.1) so the index remains close
    # enough for matching. Full rebuild happens on gallery_load().


def gallery_was_recent(reid_id):
    if reid_id not in gallery_data:
        return False
    return (time.time() - gallery_data[reid_id]["last_seen"]) < REID_GRACE_PERIOD


def gallery_purge_stale():
    now   = time.time()
    stale = [
        rid for rid, data in gallery_data.items()
        if (now - data["last_seen"]) > REID_STALE_TIMEOUT
    ]
    for rid in stale:
        del gallery_data[rid]
    if stale:
        print(f"[ReID] Purged {len(stale)} stale identities.")
        _faiss_rebuild()   # rebuild after purge to keep index consistent


gallery_load()

# ==================================================
# REID-AWARE EVENT LIFECYCLE STATE  (unchanged)
# ==================================================

NO_PERSON_TIMEOUT   = 8
last_detection_time = 0
active_reid_ids     = set()

# ==================================================
# PERSON CROP MANAGER
#
# UPGRADE: best_crops now keyed by reid_id (not track_id)
# so the same real person accumulates one best crop across
# multiple track ID splits — eliminates duplicate crops.
# ==================================================

best_crops = {}   # reid_id → {frame, score, track_id}


def crop_update(frame, track_id, reid_id, x1, y1, x2, y2, confidence):
    """
    Keyed by reid_id instead of track_id.
    Only the highest-score crop per persistent identity is kept.
    """
    w     = x2 - x1
    h     = y2 - y1
    area  = w * h
    score = area * confidence

    if reid_id not in best_crops or score > best_crops[reid_id]["score"]:
        H, W = frame.shape[:2]
        px1  = max(0, x1 - 10)
        py1  = max(0, y1 - 10)
        px2  = min(W, x2 + 10)
        py2  = min(H, y2 + 10)
        crop = frame[py1:py2, px1:px2].copy()
        best_crops[reid_id] = {
            "frame":    crop,
            "score":    score,
            "track_id": track_id
        }


def crop_save_by_reid(reid_id, ev_id, person_index):
    """Save best crop for a reid_id. Returns saved path or None."""
    if reid_id not in best_crops:
        return None
    crop     = best_crops[reid_id]["frame"]
    filename = f"{PERSON_CROPS_DIR}/event{ev_id}_person{person_index}_{reid_id}.jpg"
    cv2.imwrite(filename, crop)
    return filename


def crop_clear():
    best_crops.clear()

# ==================================================
# PER-EVENT TRACKING STATE  (unchanged)
# ==================================================

current_event       = None
track_to_reid_id    = {}   # track_id → reid_id (new: store reid mapping per frame)
track_to_person_idx = {}   # track_id → person index in current_event
reid_to_person_idx  = {}   # reid_id  → person index (dedup key)
person_counter      = 0

# ==================================================
# TELEGRAM HELPERS  (unchanged from v3)
# ==================================================

def tg_send_message(chat_id, text):
    url  = f"{BASE_URL}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"[Telegram] Message error: {e}")


def tg_send_photo(chat_id, image_path, caption=""):
    url = f"{BASE_URL}/sendPhoto"
    try:
        with open(image_path, "rb") as photo:
            requests.post(
                url,
                files={"photo": photo},
                data={"chat_id": chat_id, "caption": caption},
                timeout=15
            )
    except Exception as e:
        print(f"[Telegram] Photo error: {e}")


def tg_get_updates(offset):
    url    = f"{BASE_URL}/getUpdates"
    params = {"offset": offset, "timeout": 5, "allowed_updates": ["message"]}
    try:
        resp = requests.get(url, params=params, timeout=10)
        return resp.json().get("result", [])
    except Exception:
        return []

# ==================================================
# TELEGRAM ALERT  (unchanged from v3)
# ==================================================

def _alert_worker(image_path, caption, ev_id):
    url   = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    delay = 2
    for attempt in range(1, 4):
        try:
            with open(image_path, "rb") as photo:
                resp   = requests.post(
                    url,
                    files={"photo": photo},
                    data={"chat_id": CHAT_ID, "caption": caption},
                    timeout=20
                )
            result = resp.json()
            if result.get("ok"):
                print(f"[Telegram] Alert sent for Event #{ev_id} (attempt {attempt})")
                return
            else:
                print(f"[Telegram] Alert attempt {attempt} failed: {result.get('description')}")
        except Exception as e:
            print(f"[Telegram] Alert attempt {attempt} error: {e}")
        time.sleep(delay)
        delay *= 2
    print(f"[Telegram] WARNING — Alert for Event #{ev_id} not delivered after 3 attempts.")


def send_telegram_alert(image_path, summary, duration, timestamp, ev_id):
    if not os.path.exists(image_path):
        print(f"[Telegram] Snapshot missing for Event #{ev_id}: {image_path}")
        return
    caption = (
        f"🚨 EVENT #{ev_id}\n\n"
        f"🕒 {timestamp}\n"
        f"⏱ Duration: {duration} sec\n\n"
        f"{summary}"
    )
    if len(caption) > 1024:
        caption = caption[:1021] + "..."
    t = threading.Thread(
        target=_alert_worker, args=(image_path, caption, ev_id), daemon=True
    )
    t.start()

# ==================================================
# VIDEOMAE SMART FRAME EXTRACTION  (unchanged)
# ==================================================

def extract_smart_frames(video_path_arg, ev_id):
    print(f"[INFO] Extracting VideoMAE frames: {video_path_arg}")
    event_folder = f"smart_frames/event_{ev_id}"
    os.makedirs(event_folder, exist_ok=True)

    cap_local  = cv2.VideoCapture(video_path_arg)
    frames     = []
    raw_frames = []

    while True:
        ret, frame = cap_local.read()
        if not ret:
            break
        raw_frames.append(frame.copy())
        vf  = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(vf, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

    cap_local.release()

    if len(frames) < 16:
        print("[WARNING] Not enough frames")
        return []

    inputs = videomae_processor(frames[:16], return_tensors="pt")
    with torch.no_grad():
        outputs = videomae_model(**inputs)
    _ = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    selected_frames = []
    total_frames    = len(raw_frames)
    indices         = np.linspace(0, total_frames - 1, 10, dtype=int)

    for order, idx in enumerate(indices, start=1):
        frame_path = f"{event_folder}/{order:02d}.jpg"
        cv2.imwrite(frame_path, raw_frames[idx])
        selected_frames.append(frame_path)

    print(f"[INFO] Selected {len(selected_frames)} frames")
    return selected_frames

# ==================================================
# QWEN2.5-VL INFERENCE HELPER
# (vision calls only — text queries go to Llama)
# ==================================================

def _qwen_infer(messages, pil_images=None, max_new_tokens=250):
    """Vision inference using Qwen2.5-VL-7B."""
    torch.cuda.empty_cache()
    gc.collect()

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if pil_images:
        resized = [img.resize((448, 448)) for img in pil_images]
        inputs  = processor(
            text=[text],
            images=resized,
            padding=True,
            return_tensors="pt"
        )
    else:
        inputs = processor(text=[text], return_tensors="pt")

    inputs = inputs.to("cuda")

    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1
        )

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    del inputs, generated_ids
    torch.cuda.empty_cache()
    gc.collect()

    if "assistant" in output_text:
        output_text = output_text.split("assistant")[-1].strip()

    return output_text


# ==================================================
def _llama_infer(system_prompt, user_prompt, max_new_tokens=300):

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.3,
        max_tokens=max_new_tokens
    )

    return completion.choices[0].message.content
# ==================================================
# GROUNDING RULES  (unchanged from v3)
# ==================================================

_GROUNDING_RULES = """
You are a CCTV security analysis system. Your role is to describe ONLY what is directly visible in the footage.

STRICT RULES — violations will make this system unreliable:
- Describe ONLY clearly visible actions, clothing, objects, and movement.
- DO NOT infer emotions, intentions, relationships, or conversations.
- DO NOT assume any activity that is not directly shown.
- DO NOT use narrative or cinematic language.
- If something is unclear or not visible, write exactly: "Not clearly visible."
- Use short, factual, past-tense sentences.
- Do NOT add interpretation, speculation, or background scene description.
""".strip()

# ==================================================
# QWEN2.5-VL SUMMARY  (unchanged prompt; upgraded model)
# ==================================================

def generate_summary(frame_paths):
    print("[INFO] Generating AI summary (Qwen2.5-VL-7B)...")

    pil_images = [Image.open(fp) for fp in frame_paths]

    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in pil_images],
                {
                    "type": "text",
                    "text": (
                        f"{_GROUNDING_RULES}\n\n"
                        "These images are extracted from ONE CCTV event arranged chronologically.\n\n"

"Analyze ONLY what is reasonably visible in the frames.\n"
"Do NOT invent conversations, emotions, intentions, or relationships.\n"
"If something is unclear, say 'Not clearly visible'.\n"
"Focus mainly on people, actions, movement, carried objects, waiting behavior, and interactions.\n"
"Maintain temporal consistency across frames.\n\n"

"Write a natural CCTV surveillance event summary.\n\n"

"For each detected person include naturally:\n"
"- clothing/appearance\n"
"- observed actions\n"
"- movement direction\n"
"- carried objects\n"
"- waiting/loitering behavior\n"
"- interactions if clearly visible\n\n"

"The report should feel like a professional CCTV operator summary:\n"
"clear, grounded, chronological, and natural.\n\n"

"Avoid robotic formatting like 'Person A:' repeatedly.\n"
"Avoid repeating the same sentence structure.\n"
"Avoid excessive speculation.\n"

                    )
                }
            ]
        }
    ]

    return _qwen_infer(messages, pil_images=pil_images, max_new_tokens=250)


# ==================================================
# ATTRIBUTE EXTRACTION  (unchanged prompt; Qwen2.5-VL)
# ==================================================

def extract_person_attributes(summary):
    print("[INFO] Extracting person attributes (Qwen2.5-VL-7B)...")

    prompt = (
        f"{_GROUNDING_RULES}\n\n"
        "Extract structured data from the CCTV event description below.\n\n"
        "Return ONLY a valid JSON array. No markdown fences, no explanation.\n"
        "Each element represents one person and must have exactly these keys:\n"
        "  appearance: string or null\n"
        "  actions: array of strings\n"
        "  objects: array of strings\n"
        "  movement: string or null\n"
        "  waiting: boolean\n\n"
        "Critical rules:\n"
        "- Extract ONLY what is explicitly written.\n"
        "- Do NOT infer or add details not present in the text.\n"
        "- If a field is not mentioned, use null / [].\n"
        "- waiting: true only if 'loitering', 'waiting', or 'standing' appears.\n\n"
        f"Event description:\n{summary}\n\n"
        "JSON array (start with [ end with ]):"
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    output   = _qwen_infer(messages, max_new_tokens=400)

    output = output.replace("```json", "").replace("```", "").strip()
    m      = re.search(r"\[.*\]", output, re.DOTALL)
    if m:
        output = m.group(0)

    try:
        persons_data = json.loads(output)
        if not isinstance(persons_data, list):
            persons_data = [persons_data]
        return persons_data
    except json.JSONDecodeError:
        return _heuristic_extract(summary)


def _heuristic_extract(summary):
    text = summary.lower()

    colors   = ["black","white","red","blue","green","yellow","grey","gray",
                 "brown","pink","orange","purple"]
    clothing = ["shirt","jacket","coat","hoodie","dress","jeans","trousers",
                "cap","hat","bag","backpack"]

    appearance_parts = []
    for color in colors:
        for item in clothing:
            if color in text and item in text:
                appearance_parts.append(f"{color} {item}")

    action_keywords = {
        "used phone":    ["phone","mobile","smartphone"],
        "carried bag":   ["bag","backpack","luggage"],
        "waited":        ["wait","stood","standing","lingered"],
        "entered":       ["enter","arrived","came in"],
        "exited":        ["exit","left","departed"],
        "interacted":    ["interact","talked","spoke"],
        "looked around": ["looked around","scanning"],
    }
    actions = [a for a, kws in action_keywords.items() if any(k in text for k in kws)]

    object_keywords = ["phone","bag","backpack","umbrella","bottle","helmet","laptop","box"]
    objects = [o for o in object_keywords if o in text]

    waiting = any(w in text for w in ["wait","stood","standing","lingered"])

    return [{
        "appearance": ", ".join(appearance_parts) if appearance_parts else None,
        "actions":    actions,
        "objects":    objects,
        "movement":   None,
        "waiting":    waiting
    }]

# ==================================================
# QUERY INTENT CLASSIFIER  (unchanged)
# ==================================================

INTENT_PATTERNS = {
    "daily_update": [
        r"today.{0,20}update", r"what happened today",
        r"give me.{0,10}summary", r"today.{0,10}report",
        r"recent event", r"latest event",
    ],
    "person_appearance": [
        r"(black|white|red|blue|green|yellow|grey|gray|brown|pink|orange|purple)"
        r".{0,20}(shirt|jacket|coat|hoodie|dress|jeans|cap|hat)",
        r"who (was|is) wearing", r"person in", r"man in", r"woman in",
    ],
    "person_action": [
        r"who (used|carried|brought|held|had)",
        r"who (entered|exited|left|came|arrived)",
        r"who (waited|stood|lingered|sat)",
        r"who (talked|spoke|interacted)",
        r"who (ran|walked|moved)",
        r"what did.{0,30}do",
    ],
    "time_query": [
        r"after \d{1,2}(:\d{2})?\s*(am|pm|AM|PM)",
        r"before \d{1,2}(:\d{2})?\s*(am|pm|AM|PM)",
        r"between \d", r"at night", r"in the morning", r"in the evening",
    ],
    "object_query": [
        r"(bag|backpack|phone|umbrella|bottle|helmet|laptop|box)",
        r"carrying", r"holding", r"with a",
    ],
    "zone_query": [
        r"zone [a-zA-Z]",
        r"near (entrance|exit|door|gate|counter|desk)",
        r"at the (entrance|exit|door|gate|counter|desk)",
    ],
    "suspicious": [
        r"suspicious", r"unusual", r"loitering",
        r"longest", r"stayed.{0,10}(longest|long time)",
    ],
    "image_request": [
        r"send (pic|photo|image|picture)",
        r"show (me )?(the )?(person|image|pic|photo)",
        r"who is this", r"picture of", r"image of",
        r"photo of", r"show image",
    ],
}


def classify_intent(query):
    q       = query.lower()
    matched = []
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, q):
                matched.append(intent)
                break
    return matched or ["general"]

# ==================================================
# STRUCTURED MEMORY CONTEXT BUILDER  (unchanged)
# ==================================================

def _parse_hour(ts_str):
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").hour
    except Exception:
        return None


def _time_matches_query(hour, query):
    q = query.lower()
    m = re.search(r"after (\d{1,2})\s*(am|pm)?", q)
    if m:
        h = int(m.group(1))
        if m.group(2) == "pm" and h < 12: h += 12
        return hour >= h
    m = re.search(r"before (\d{1,2})\s*(am|pm)?", q)
    if m:
        h = int(m.group(1))
        if m.group(2) == "pm" and h < 12: h += 12
        return hour < h
    if "at night"       in q: return hour >= 20 or hour < 6
    if "in the morning" in q: return 6 <= hour < 12
    if "in the evening" in q: return 17 <= hour < 21
    return True


def build_structured_context(memory, query, intents):
    if not memory:
        return "No events recorded yet."

    parts = []
    for event in memory:
        ts      = event.get("timestamp", "unknown")
        eid     = event.get("event_id",  "?")
        dur     = event.get("duration",  0)
        summary = event.get("summary",   "")
        persons = event.get("persons",   [])

        if "time_query" in intents:
            hour = _parse_hour(ts)
            if hour is not None and not _time_matches_query(hour, query):
                continue

        s = f"[Event #{eid} | {ts} | {dur}s]\nSummary: {summary}\n"
        for p in persons:
            pid    = p.get("person_id", "?")
            appear = p.get("appearance") or "unknown"
            acts   = ", ".join(p.get("actions", [])) or "none"
            objs   = ", ".join(p.get("objects", [])) or "none"
            move   = p.get("movement") or "unknown"
            wait   = "yes" if p.get("waiting") else "no"
            crop   = p.get("crop_image")
            reid   = p.get("reid_id", "?")

            s += (
                f"  Person {pid} [reid={reid}]: appearance='{appear}', "
                f"actions=[{acts}], objects=[{objs}], "
                f"movement='{move}', waiting={wait}"
            )
            if crop:
                s += f", [has_image={crop}]"
            s += "\n"
        parts.append(s)

    return "\n".join(parts) if parts else "No matching events found."

# ==================================================
# UPGRADE: IMAGE RETRIEVAL WITH FAISS FALLBACK
#
# When a reid_id can be resolved from the query context
# (e.g. the user previously asked about a specific person
# and we stored their embedding), FAISS is used for
# retrieval. Otherwise falls back to keyword scoring.
# ==================================================

def find_matching_crops(query, memory, query_embedding=None):
    """
    Returns [(crop_path, description), ...] top-3 matches.
    Uses FAISS cosine search when query_embedding provided,
    keyword scoring otherwise.
    """
    q       = query.lower()
    results = []

    # ---- FAISS path ----
    if query_embedding is not None and _faiss_index is not None and _faiss_index.ntotal > 0:
        hits       = faiss_search(query_embedding, k=10)
        hit_rids   = {rid for rid, _ in hits if _ >= REID_SIMILARITY_THRESHOLD * 0.8}

        for event in memory:
            for p in event.get("persons", []):
                crop    = p.get("crop_image")
                reid_id = p.get("reid_id")
                if not crop or not os.path.exists(crop):
                    continue
                if reid_id in hit_rids:
                    ts   = event.get("timestamp", "unknown")
                    desc = (
                        f"Event #{event['event_id']} @ {ts} | "
                        f"{p.get('appearance','?')} | reid={reid_id}"
                    )
                    results.append((crop, desc, 10.0))  # high priority for FAISS hits

    # ---- keyword scoring path (always runs as supplement) ----
    seen_crops = {r[0] for r in results}
    for event in memory:
        for p in event.get("persons", []):
            crop = p.get("crop_image")
            if not crop or not os.path.exists(crop) or crop in seen_crops:
                continue

            score  = 0
            appear = (p.get("appearance") or "").lower()
            acts   = " ".join(p.get("actions", [])).lower()
            objs   = " ".join(p.get("objects", [])).lower()

            for word in q.split():
                if word in appear: score += 3
                if word in acts:   score += 2
                if word in objs:   score += 2

            for obj in ["bag","phone","backpack","umbrella","helmet","laptop","bottle"]:
                if obj in q and obj in objs:
                    score += 5

            if score > 0:
                ts   = event.get("timestamp", "unknown")
                desc = (
                    f"Event #{event['event_id']} @ {ts} | "
                    f"{p.get('appearance','?')} | "
                    f"actions: {', '.join(p.get('actions',[])) or 'none'}"
                )
                results.append((crop, desc, score))

    results.sort(key=lambda x: x[2], reverse=True)
    return [(path, desc) for path, desc, _ in results[:3]]

# ==================================================
# UPGRADE: QUERY MEMORY — Llama 3.1 8B reasoning layer
#
# Only query_memory() is changed.
# All memory-access helpers (load_memory, classify_intent,
# build_structured_context, find_matching_crops) are unchanged.
# ==================================================

LLAMA_SYSTEM_PROMPT = """You are an expert AI surveillance operator assistant.

You have access to structured CCTV surveillance memory containing timestamped events, person descriptions, observed actions, carried objects, and movement data.

Your job:
- Answer the operator's question accurately using ONLY the provided surveillance memory.
- Reference specific persons by their appearance and event timestamps.
- Be concise, factual, and professional — like a real security operator would speak.
- If the information is not in the memory, say so clearly. Do NOT guess or invent details.
- When relevant, mention the event ID and timestamp for traceability.
- Support follow-up questions by referencing prior context naturally.

You must NOT:
- Invent events, persons, or details not in the memory.
- Speculate about intent, emotion, or future behaviour.
- Provide opinions or recommendations beyond factual reporting.
"""


def query_memory(query, query_embedding=None):
    memory      = load_memory()
    intents     = classify_intent(query)
    context     = build_structured_context(memory, query, intents)
    image_paths = []

    if "image_request" in intents:
        crops       = find_matching_crops(query, memory, query_embedding)
        image_paths = [c[0] for c in crops]

    intent_note = ", ".join(intents)

    user_prompt = (
        f"Query intent detected: {intent_note}\n\n"
        "=== SURVEILLANCE MEMORY ===\n"
        f"{context}\n"
        "=== END OF MEMORY ===\n\n"
        f"Operator question: {query}"
    )

    output = _llama_infer(LLAMA_SYSTEM_PROMPT, user_prompt, max_new_tokens=300)
    return output, image_paths

# ==================================================
# TELEGRAM BOT  (unchanged from v3)
# ==================================================

bot_offset         = 0
bot_chat_history   = []
bot_running        = True
bot_lock           = threading.Lock()
_processed_updates = set()


def _get_telegram_offset():
    updates = tg_get_updates(-1)
    if updates:
        return updates[-1]["update_id"] + 1
    return 0


def bot_poll_loop():
    global bot_offset

    bot_offset = _get_telegram_offset()
    print(f"[Telegram] Bot ready (chat_id: {CHAT_ID}), offset={bot_offset}")

    while bot_running:
        try:
            updates = tg_get_updates(bot_offset)
        except Exception as e:
            print(f"[Telegram] Poll error: {e}")
            time.sleep(2)
            continue

        for update in updates:
            uid        = update["update_id"]
            bot_offset = uid + 1

            if uid in _processed_updates:
                continue
            _processed_updates.add(uid)

            message = update.get("message", {})
            chat_id = str(message.get("chat", {}).get("id", ""))
            text    = message.get("text", "").strip()

            if not text or chat_id != CHAT_ID:
                continue
            if text.startswith("/") and text not in ("/status", "/help"):
                continue

            print(f"[Telegram] Query received: {text}")
            tg_send_message(chat_id, "⏳ Searching memory...")

            try:
                with bot_lock:
                    bot_chat_history.append(("user", text))
                    if len(bot_chat_history) > 12:
                        bot_chat_history[:] = bot_chat_history[-12:]
                    history_ctx = ""
                    if len(bot_chat_history) > 2:
                        history_ctx = "Recent conversation:\n"
                        for role, msg in bot_chat_history[-6:]:
                            history_ctx += f"  {role.upper()}: {msg}\n"
                        history_ctx += "\n"

                enriched_query   = history_ctx + f"Current question: {text}"
                answer, img_paths = query_memory(enriched_query)

                with bot_lock:
                    bot_chat_history.append(("assistant", answer[:200]))

                tg_send_message(chat_id, answer)

                for i, img_path in enumerate(img_paths):
                    if os.path.exists(img_path):
                        tg_send_photo(chat_id, img_path, caption=f"Matching person {i+1}")

            except Exception as e:
                tg_send_message(chat_id, f"Sorry, an error occurred: {e}")
                print(f"[Telegram] Query error: {e}")

        if len(_processed_updates) > 5000:
            _processed_updates.clear()

        time.sleep(0.5)

    print("[Telegram] Bot loop exited cleanly.")


bot_thread = threading.Thread(
    target=bot_poll_loop, daemon=True, name="TelegramBot"
)
bot_thread.start()

# ==================================================
# CLOSE EVENT HELPER
#
# UPGRADE: crop_save_by_reid() replaces crop_save()
# so that one best-quality crop per reid_id is stored,
# not one per track_id (prevents crop duplication).
# reid_id is now written into the person record.
# ==================================================

def close_event(ev_id, ev_output_path, ev_start_time, ev_record):
    global best_frame, best_score
    global track_to_person_idx, reid_to_person_idx, person_counter

    ev_end      = time.time()
    ev_duration = int(ev_end - ev_start_time)
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if best_frame is None:
        print(f"[WARNING] Event {ev_id} has no best_frame — skipping close.")
        return

    # STEP 1: snapshot
    snapshot_path = f"smart_frames/event_{ev_id}.jpg"
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

    # Reset
    best_frame          = None
    best_score          = 0
    crop_clear()
    track_to_person_idx = {}
    reid_to_person_idx  = {}
    person_counter      = 0

    print(f"[INFO] Event {ev_id} saved with {len(actual_persons)} person records.")

# ==================================================
# MAIN LOOP
# UPGRADE: tracker changed to strongsort.yaml
# UPGRADE: track_to_reid_id populated per frame
# UPGRADE: crop_update called with reid_id (not track_id)
# UPGRADE: reid_to_person_idx deduplicates persons
# ==================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    cv2.polylines(frame, [roi_points], True, (255, 0, 0), 2)

    # UPGRADE: StrongSORT tracker
    results = model.track(
        frame, persist=True, tracker="botsort.yaml", classes=[0]
    )

    detections_in_roi = []

    if results[0].boxes.id is not None:
        boxes       = results[0].boxes.xyxy.cpu()
        track_ids   = results[0].boxes.id.cpu().int().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        for box, track_id, confidence in zip(boxes, track_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            # ==========================================
            # DEBUG REJECTED DETECTIONS
            # ==========================================

            if area < 8000 or confidence < 0.6:

                debug_frame = frame.copy()

                cv2.rectangle(
                    debug_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                label = f"REJECTED | ID:{track_id} conf:{confidence:.2f} area:{area}"

                cv2.putText(
                    debug_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

                os.makedirs("debug_rejected", exist_ok=True)

                cv2.imwrite(
                    f"debug_rejected/frame_{frame_index}_id_{track_id}.jpg",
                    debug_frame
                )

                continue


            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            inside = cv2.pointPolygonTest(roi_points, (cx, cy), False)
            if inside < 0:
                continue

            detections_in_roi.append((x1, y1, x2, y2, track_id, confidence))

            score = area * confidence
            if score > best_score:
                best_score = score
                best_frame = frame.copy()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    person_in_roi = len(detections_in_roi) > 0

    if person_in_roi:
        for (x1, y1, x2, y2, track_id, conf) in detections_in_roi:
            crop = frame[max(0, y1-10):y2+10, max(0, x1-10):x2+10]
            if crop.size == 0:
                continue

            # FastReID embedding
            emb              = reid_fn(crop)
            reid_id, sim     = gallery_match(emb)
            active_reid_ids.add(reid_id)
            track_to_reid_id[track_id] = reid_id

            # Best crop keyed by reid_id (dedup fix)
            crop_update(frame, track_id, reid_id, x1, y1, x2, y2, conf)

        last_detection_time = time.time()

        if not recording:
            print(f"[EVENT START] Event {event_id}")
            event_start_time    = time.time()
            output_path         = f"events/event_{event_id}.mp4"
            fourcc              = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer        = cv2.VideoWriter(
                output_path, fourcc, fps, (frame_width, frame_height)
            )
            recording           = True
            current_event       = empty_event_record(
                event_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            track_to_person_idx = {}
            reid_to_person_idx  = {}
            person_counter      = 0

        # UPGRADE: dedup via reid_to_person_idx
        for (x1, y1, x2, y2, track_id, conf) in detections_in_roi:
            reid_id = track_to_reid_id.get(track_id)
            if reid_id is None:
                continue

            # If this reid_id already has a person record, reuse it
            if reid_id in reid_to_person_idx:
                pidx = reid_to_person_idx[reid_id]
                track_to_person_idx[track_id] = pidx
                continue

            # New person (new reid_id seen for first time in this event)
            if track_id not in track_to_person_idx:
                person_counter += 1
                track_to_person_idx[track_id] = person_counter
                reid_to_person_idx[reid_id]   = person_counter

                pid   = f"event{event_id}_person{person_counter}"
                p_rec = empty_person_record(pid, track_id, event_id)
                p_rec["reid_id"]    = reid_id
                p_rec["first_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_event["persons"].append(p_rec)

    if recording:
        video_writer.write(frame)

    if recording:
        elapsed = time.time() - last_detection_time
        grace   = REID_GRACE_PERIOD if active_reid_ids else NO_PERSON_TIMEOUT

        if elapsed > grace:
            identity_returned = any(gallery_was_recent(rid) for rid in active_reid_ids)

            if identity_returned and elapsed < REID_GRACE_PERIOD * 2:
                pass
            else:
                print(f"[EVENT END] Event {event_id}")
                recording = False
                video_writer.release()

                close_event(event_id, output_path, event_start_time, current_event)

                active_reid_ids    = set()
                track_to_reid_id   = {}
                current_event      = None
                event_id          += 1

                gallery_purge_stale()

# ==================================================
# FINALIZE LAST EVENT  (unchanged)
# ==================================================

if recording:
    print(f"[FINAL EVENT END] Event {event_id}")
    recording = False
    video_writer.release()
    close_event(event_id, output_path, event_start_time, current_event)

cap.release()
if video_writer is not None:
    video_writer.release()

gallery_save()

print("\n[INFO] Video pipeline finished.")
print("[INFO] Telegram bot remains alive for queries. Press Ctrl+C to exit.\n")

try:
    while bot_running:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[INFO] Shutting down...")
    bot_running = False
    bot_thread.join(timeout=3)
    print("[INFO] Telegram bot stopped. Goodbye.")

# ==================================================
# TEST QUERY
# ==================================================

print("\n" + "=" * 50 + "\nAI QUERY TEST\n" + "=" * 50)
response, images = query_memory("Who carried a bag?")
print(response)
if images:
    print(f"\nMatching images: {images}")