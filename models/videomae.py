"""
models/videomae.py
==================
Loads VideoMAE and exposes extract_smart_frames() used by the summary
pipeline after every event ends.

GPU MEMORY OPTIMISATION
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

VIDEOMAE-DRIVEN SMART FRAME SELECTION
------------------------------------------------------
Originally VideoMAE only ever looked at the FIRST 16 raw frames of the
event (`frames[:16]`) — usually well under a second of a multi-second
recording — ran its forward pass, and then threw the resulting
embedding away (`_ = outputs.last_hidden_state.mean(...)`). The 3
"smart frames" actually sent to Qwen/OpenAI were picked afterwards by
a completely separate, handcrafted heuristic (evenly-spaced thirds +
per-frame detection-quality score + a Laplacian sharpness check) that
never looked at VideoMAE's output at all.

FULL-EVENT COVERAGE (this revision)
------------------------------------
A single 16-frame clip — even one sampled evenly across the whole
event — still only ever SEES 16 instants out of what can be hundreds
of recorded frames. Anything happening between two sampled frames
(e.g. a parcel handover at 2.3s that lands between two evenly-spaced
sample points) never reaches VideoMAE at all. This revision has
VideoMAE analyze the ENTIRE recording instead of one sampled clip:

  1. The whole recording is decoded (unchanged) and split into
     CONTIGUOUS, back-to-back clips of VideoMAE's expected input
     length (16 frames each) that together cover every single decoded
     frame — see _split_into_clips(). The last clip overlaps slightly
     backward (instead of being padded/invented) whenever the event
     length isn't an exact multiple of the clip size, so no frame is
     ever skipped and no frame is ever fabricated.
  2. VideoMAE runs its forward pass on EVERY clip. Each clip's
     `last_hidden_state` is reshaped into one embedding per temporal
     tubelet (VideoMAE groups every `tubelet_size` consecutive input
     frames into a single spatio-temporal token) — see
     _temporal_embeddings() — instead of being discarded.
  3. Every clip's tubelet embeddings are concatenated into ONE
     complete-event representation spanning the entire recording, not
     just one clip.
  4. Each tubelet's embedding is scored by how much it stands out from
     that COMPLETE event's own mean embedding (`_temporal_saliency()`)
     — a tubelet that looks like every other moment (an empty/static
     stretch) scores low; a tubelet VideoMAE's representation says is
     temporally/visually distinctive (someone entering, the main
     interaction, someone leaving) scores high.
  5. The 3 final frames are chosen using THAT score, one from the
     first/middle/last third of the event each (keeps the "entry,
     main interaction, exit" temporal spread), instead of the old
     handcrafted quality+Laplacian ranking, and instead of only being
     able to choose among 16 sparse candidates.

`quality_scores` (frame-level, real-detection-confidence signal that
pipelines/event_manager.py already computes for free during capture)
is kept only as a cheap VALIDITY GATE — skip a candidate with no
confirmed detection at all (an empty/grace-period frame) — never as
the thing that picks the winner. VideoMAE's saliency score is what
picks the winner among valid candidates. This preserves the "avoid
empty frames" requirement without reintroducing a handcrafted ranking
algorithm as the actual decision-maker.
"""

import os
import gc
import cv2
import torch
import numpy as np
from transformers import VideoMAEModel, VideoMAEImageProcessor
from config.settings import VIDEOMAE_MODEL, SMART_FRAMES_DIR
from utils.event_logger import log_block
from utils.device import empty_cache, log_gpu_memory

# Module-level singletons — populated by load_videomae(), cleared by
# unload_videomae(). No longer loaded at import time / startup.
videomae_processor = None
videomae_model     = None

# VideoMAE-base's expected clip length. The full recording is split
# into back-to-back clips of this length (see _split_into_clips())
# so every frame of the event is analyzed, not just one sampled clip.
_NUM_MODEL_FRAMES = 16


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
    # harmless (and skipped entirely on CPU-only systems), and keeps the
    # "release GPU memory" step in place even if VideoMAE's device ever
    # changes later.
    empty_cache()

    print("[INFO] VideoMAE unloaded.")
    log_gpu_memory("After VideoMAE unload")


def extract_smart_frames(video_path_arg, ev_id, quality_scores=None):
    """
    Extract smart frames from a recorded event video using VideoMAE.

    VideoMAE now analyzes the COMPLETE recording — every decoded frame
    is covered by at least one contiguous clip run through the model
    (see _split_into_clips()) — and its own output decides WHICH 3 raw
    frames are the final "smart frames" (see _temporal_embeddings()/
    _temporal_saliency()/_select_indices() below), instead of running
    a forward pass on one sampled clip whose output is thrown away in
    favour of a separate handcrafted heuristic.

    Args:
        video_path_arg: path to the finished scene recording.
        ev_id: event id, used for the output folder name.
        quality_scores: optional list[float], index-aligned with the
            video's frames — one score per frame, higher = a real,
            confident, on-target detection; 0.0 = no detection that
            frame (e.g. a grace-period gap). Produced for free during
            capture by pipelines/event_manager.py (SceneEvent.
            frame_quality) — no extra inference here. Used ONLY as a
            validity gate (skip a frame with no confirmed detection at
            all) — VideoMAE's own saliency score below is what picks
            the winner among valid candidates. When omitted, absent,
            or mismatched in length, every frame is treated as valid.
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
    raw_frames = []

    while True:
        ret, frame = cap_local.read()
        if not ret:
            break
        raw_frames.append(frame.copy())

    cap_local.release()

    total_frames = len(raw_frames)
    if total_frames < _NUM_MODEL_FRAMES:
        print("[WARNING] Not enough frames")
        return []

    try:
        # Run VideoMAE on the COMPLETE event — every contiguous clip,
        # covering every decoded frame — instead of one sampled clip,
        # so nothing happening BETWEEN sample points can ever be missed.
        clips = _split_into_clips(total_frames, _NUM_MODEL_FRAMES)

        all_embeds, all_frame_idx = [], []
        for clip_idx in clips:
            model_frames = [
                cv2.cvtColor(cv2.resize(raw_frames[i], (224, 224)), cv2.COLOR_BGR2RGB)
                for i in clip_idx
            ]
            inputs = videomae_processor(model_frames, return_tensors="pt")
            with torch.no_grad():
                outputs = videomae_model(**inputs)

            tubelet_embeds, tubelet_frame_idx = _temporal_embeddings(
                outputs.last_hidden_state, clip_idx
            )
            all_embeds.append(tubelet_embeds)
            all_frame_idx.extend(tubelet_frame_idx)

        # ONE complete-event representation, spanning every clip —
        # this is what makes saliency below reflect the WHOLE
        # recording rather than just whichever single clip was sampled.
        combined_embeds = torch.cat(all_embeds, dim=0)
        saliency = _temporal_saliency(combined_embeds)
        indices = _select_indices(
            total_frames, all_frame_idx, saliency, quality_scores
        )
    except Exception as e:
        # Defensive fallback — a shape mismatch (e.g. a differently
        # configured VideoMAE checkpoint) must never crash event
        # finalisation. Falls back to an evenly-spaced selection rather
        # than the old discarded-embedding behaviour.
        print(f"[WARNING] VideoMAE-driven frame selection failed ({e}); "
              f"falling back to evenly-spaced frames.")
        indices = list(np.linspace(0, total_frames - 1, 3, dtype=int))

    selected_frames = []
    for order, idx in enumerate(indices, start=1):
        frame_path = f"{event_folder}/{order:02d}.jpg"
        cv2.imwrite(frame_path, raw_frames[idx])
        selected_frames.append(frame_path)

    log_block("VideoMAE", f"Selected : {len(selected_frames)} frames")
    return selected_frames


def _split_into_clips(total_frames, clip_len):
    """
    Split the WHOLE event into contiguous, back-to-back clips of
    `clip_len` frames each, covering every single decoded frame —
    this is what lets VideoMAE analyze the complete recording instead
    of one sampled clip.

    Clips are non-overlapping except possibly the LAST one: when
    total_frames isn't an exact multiple of clip_len, the final clip
    is shifted backward just enough to still be a full `clip_len`
    frames (VideoMAE's fixed input size) — a small, bounded overlap
    with the previous clip — rather than skipping the leftover frames
    or padding the clip with invented/duplicated frames.

    Requires total_frames >= clip_len (checked by the caller).
    """
    clips = []
    start = 0
    while start < total_frames:
        end = start + clip_len
        if end > total_frames:
            end = total_frames
            start = max(0, end - clip_len)
        clips.append(list(range(start, end)))
        if end >= total_frames:
            break
        start = end
    return clips


def _temporal_embeddings(last_hidden_state, sample_idx):
    """
    Turn VideoMAE's raw patch-level output into ONE embedding per
    temporal tubelet, instead of collapsing/discarding it.

    VideoMAE groups every `tubelet_size` consecutive input frames (2,
    for the standard MCG-NJU/videomae-base checkpoint this project
    uses) into a spatio-temporal tubelet, then splits each tubelet into
    a spatial grid of patches. `last_hidden_state` is
    (1, num_temporal_tubelets * num_spatial_patches, hidden_size),
    temporal-major. Mean-pooling over the spatial patches within each
    tubelet yields exactly one embedding per tubelet — a genuine
    VideoMAE-derived representation of that slice of the event,
    keeping the model's own learned features as the selection signal.

    Returns:
        tubelet_embeds:    Tensor (num_tubelets, hidden_size)
        tubelet_frame_idx: list[int] — the raw-frame index (from
            `sample_idx`, i.e. actual position in the original
            recording) at the centre of each tubelet.
    """
    model = videomae_model
    cfg = model.config
    tubelet_size = getattr(cfg, "tubelet_size", 2)
    patch_size   = cfg.patch_size
    image_size   = cfg.image_size
    grid         = image_size // patch_size
    num_spatial_patches = grid * grid
    num_tubelets = len(sample_idx) // tubelet_size

    patches = last_hidden_state[0]  # (num_patches, hidden_size)
    expected = num_tubelets * num_spatial_patches
    if patches.shape[0] != expected:
        raise ValueError(
            f"Unexpected VideoMAE patch count {patches.shape[0]} "
            f"(expected {expected} for {num_tubelets} tubelets x "
            f"{num_spatial_patches} spatial patches) — checkpoint's "
            f"patch/tubelet geometry differs from the assumed default."
        )

    patches = patches.view(num_tubelets, num_spatial_patches, -1)
    tubelet_embeds = patches.mean(dim=1)  # (num_tubelets, hidden_size)

    tubelet_frame_idx = []
    for t in range(num_tubelets):
        start = t * tubelet_size
        end   = min(start + tubelet_size, len(sample_idx)) - 1
        tubelet_frame_idx.append((sample_idx[start] + sample_idx[end]) // 2)

    return tubelet_embeds, tubelet_frame_idx


def _temporal_saliency(tubelet_embeds):
    """
    Score each tubelet by how much VideoMAE's own representation of it
    differs from the event's overall mean embedding.

    A tubelet whose embedding sits close to the mean looks like "more
    of the same" as the rest of the event — an empty stretch, someone
    standing still, background with no activity. A tubelet whose
    embedding sits far from the mean is something VideoMAE's
    representation treats as temporally/visually distinctive —
    someone entering, the main interaction, someone leaving — exactly
    the moments the user wants the Smart Frames to cover. This is the
    "VideoMAE decides" signal that replaces the old Laplacian-sharpness
    heuristic.
    """
    mean_embed = tubelet_embeds.mean(dim=0)
    return torch.norm(tubelet_embeds - mean_embed, dim=1).cpu().numpy()


def _select_indices(total_frames, tubelet_frame_idx, saliency, quality_scores):
    """
    Pick 3 raw-frame indices — one each from the first / middle / last
    third of the event, to keep the "entering, main interaction,
    leaving" temporal spread — using VideoMAE's own saliency score
    (see _temporal_saliency()) to choose the winner within each third.

    `quality_scores`, when available, is used ONLY to skip a candidate
    tubelet whose centre frame has no confirmed detection at all (an
    empty/grace-period gap) — never to rank or override VideoMAE's own
    saliency score, which is what actually decides the winner.
    """
    fallback = list(np.linspace(0, total_frames - 1, 3, dtype=int))
    valid_quality = (
        quality_scores is not None and len(quality_scores) == total_frames
    )

    bounds = [0, total_frames // 3, (2 * total_frames) // 3, total_frames]
    indices = []

    for b in range(3):
        lo, hi = bounds[b], bounds[b + 1]
        if hi <= lo:
            indices.append(fallback[b])
            continue

        # Every tubelet whose centre frame falls inside this third.
        candidates = [
            t for t, f in enumerate(tubelet_frame_idx) if lo <= f < hi
        ]
        if valid_quality:
            with_detection = [
                t for t in candidates if quality_scores[tubelet_frame_idx[t]] > 0
            ]
            if with_detection:
                candidates = with_detection

        if not candidates:
            indices.append(fallback[b])
            continue

        # VideoMAE's own saliency score picks the winner.
        best_t = max(candidates, key=lambda t: saliency[t])
        indices.append(tubelet_frame_idx[best_t])

    return indices