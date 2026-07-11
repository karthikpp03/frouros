"""
models/videomae.py
==================
Loads VideoMAE and exposes extract_smart_frames() used by the summary
pipeline after every event ends.

Feature-extraction logic (frame sampling, resize, model forward pass,
frame selection) is preserved verbatim from the original monolith.

GPU MEMORY OPTIMISATION (new)
------------------------------
VideoMAE was already CPU-resident (`videomae_model.to("cpu")` below),
so it never actually competed with Qwen for GPU memory in this
codebase. Even so, per the requirement that VideoMAE and Qwen must
never be considered loaded "at the same time", load_videomae() /
unload_videomae() are now a matched on-demand pair, called by
pipelines/event_manager.py immediately before and after smart-frame
extraction — i.e. VideoMAE is loaded, used, and fully released BEFORE
Qwen is loaded for the same event. This also means VideoMAE no longer
sits in RAM for the whole process lifetime, only while an event is
actually being finalised, and the pattern stays correct even if
VideoMAE were ever moved to GPU in the future.
"""

import os
import gc
import cv2
import torch
import numpy as np
from transformers import VideoMAEModel, VideoMAEImageProcessor
from config.settings import VIDEOMAE_MODEL, SMART_FRAMES_DIR
from utils.event_logger import log_block

# Module-level singletons — populated by load_videomae(), cleared by
# unload_videomae(). No longer loaded at import time / startup.
videomae_processor = None
videomae_model     = None


def load_videomae():
    """
    Load VideoMAE model and processor to CPU.

    Called on-demand right before smart-frame extraction for an event
    (not once at startup), so it's only resident in memory for as long
    as it's actually needed.
    """
    global videomae_processor, videomae_model

    if videomae_model is not None:
        return  # already loaded

    print("[INFO] Loading VideoMAE...")

    videomae_processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_MODEL)
    videomae_model     = VideoMAEModel.from_pretrained(VIDEOMAE_MODEL)
    videomae_model.eval()
    videomae_model.to("cpu")

    print("[INFO] VideoMAE loaded!")


def unload_videomae():
    """
    Fully release VideoMAE (model + processor) from memory.

    Called right after extract_smart_frames() finishes for an event,
    before Qwen gets loaded for that same event — guarantees the two
    heavy models are never resident at once, and frees the CPU RAM
    VideoMAE was holding in the meantime.
    """
    global videomae_processor, videomae_model

    if videomae_model is None:
        return  # nothing to unload

    print("[INFO] Unloading VideoMAE...")

    del videomae_model
    del videomae_processor
    videomae_model     = None
    videomae_processor = None

    gc.collect()
    # VideoMAE runs on CPU in this pipeline, so there's normally nothing
    # in the CUDA allocator to reclaim here — but this call is cheap and
    # harmless, and keeps the "release GPU memory" step in place even if
    # VideoMAE's device ever changes later.
    torch.cuda.empty_cache()

    print("[INFO] VideoMAE unloaded.")
    print("After VideoMAE unload")
    print(torch.cuda.memory_summary())


def extract_smart_frames(video_path_arg, ev_id):
    """
    Extract smart frames from a recorded event video using VideoMAE.
    Identical logic to the original extract_smart_frames() — no changes.
    Callers must call load_videomae() before this and unload_videomae()
    after (see pipelines/event_manager.py._finalize_event()).
    """
    if videomae_model is None or videomae_processor is None:
        raise RuntimeError(
            "VideoMAE model is not loaded. Call load_videomae() before "
            "extract_smart_frames() — it is now loaded on-demand during "
            "event finalisation instead of at startup."
        )

    log_block("VideoMAE", "Selecting smart frames...")
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
    indices         = np.linspace(0, total_frames - 1, 3, dtype=int)

    for order, idx in enumerate(indices, start=1):
        frame_path = f"{event_folder}/{order:02d}.jpg"
        cv2.imwrite(frame_path, raw_frames[idx])
        selected_frames.append(frame_path)

    log_block("VideoMAE", f"Selected : {len(selected_frames)} frames")
    return selected_frames
