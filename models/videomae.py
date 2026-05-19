"""
models/videomae.py
==================
Loads VideoMAE and exposes extract_smart_frames() used by the summary
pipeline after every event ends.

Logic is preserved verbatim from the original monolith.
"""

import os
import cv2
import torch
import numpy as np
from transformers import VideoMAEModel, VideoMAEImageProcessor
from config.settings import VIDEOMAE_MODEL, SMART_FRAMES_DIR

# Module-level singletons
videomae_processor = None
videomae_model     = None


def load_videomae():
    """Load VideoMAE model and processor to CPU.  Called once at startup."""
    global videomae_processor, videomae_model

    print("[INFO] Loading VideoMAE...")

    videomae_processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL)
    videomae_model     = VideoMAEModel.from_pretrained(VIDEOMAE_MODEL)
    videomae_model.eval()
    videomae_model.to("cpu")

    print("[INFO] VideoMAE loaded!")


def extract_smart_frames(video_path_arg, ev_id):
    """
    Extract smart frames from a recorded event video using VideoMAE.
    Identical logic to the original extract_smart_frames() — no changes.
    """
    print(f"[INFO] Extracting VideoMAE frames: {video_path_arg}")
    event_folder = f"{SMART_FRAMES_DIR}/event_{ev_id}"
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
