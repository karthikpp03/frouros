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

ISSUE 1 ROOT CAUSE (this revision) — why back/blurry/empty frames
still won even with full-event coverage
------------------------------------------------------------------
Investigation (see the task's 10-point checklist) confirmed items 1-8
were already correct as of the previous revision above: VideoMAE does
see the whole event, clips are contiguous and cover every frame,
embeddings are converted into a real per-tubelet score, indices map
back to the correct original frame (`tubelet_frame_idx` always indexes
into the SAME `raw_frames` list the clips were sliced from — no
cross-clip index mismatch), and timestamps derive from those same
frame indices.

Items 9-10 were the actual root cause: **face visibility and blur
were never part of the score at all.** `_temporal_saliency()` only
measures how far a tubelet's embedding sits from the event's OWN mean
embedding — "how different does this instant look from the rest of
this event". That is a pure *novelty* signal, not a *quality* signal,
and the two are frequently anti-correlated in CCTV footage: the moment
someone turns their back to leave, or a motion-blurred transition
between two poses, is often the MOST different-looking instant in the
whole clip precisely because it's a brief transient — so it can
out-score a longer, static, front-facing stretch that looks similar
frame to frame. The `quality_scores` validity gate didn't fix this
either, since it only measures "YOLO found a confident person-shaped
box here" (`area * confidence`) — a back view or a blurry frame with a
big, confident detection box passes that gate just fine.

Fix: `_select_indices()` below now combines THREE signals per
candidate instead of one — VideoMAE's saliency (representativeness), a
Laplacian-variance sharpness score (`_blur_score()`), and a face-
visibility score from the already-loaded InsightFace detector
(`_face_score()`, reusing face/recognizer.py — no second model, no
extra load) — so a sharp, front-facing candidate is preferred over a
"novel-looking" but back-turned or blurry one, while VideoMAE's own
representation still participates in the decision (per the task's
requirement to "still utilize VideoMAE where appropriate"). Blur/face
scoring runs only on a small saliency-ranked shortlist per third (not
every candidate) to keep this cheap.
"""

import os
import gc
import csv
import time
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


def extract_smart_frames_fallback(video_path_arg, ev_id, quality_scores=None, debug=None):
    """
    Public entrypoint used ONLY when VideoMAE itself failed to LOAD for
    this event (see ISSUE 2 / pipelines/event_manager.py, which calls
    this instead of extract_smart_frames() whenever
    model_manager.safe_load("videomae", load_videomae) returns False).

    Does the same frame-decode + 3-frame-per-thirds selection as
    extract_smart_frames(), but scores candidates with sharpness +
    the detection-quality gate only (see _fallback_select()) since no
    VideoMAE embedding is available at all. Smart Frame Selection
    degrades gracefully instead of the whole event failing outright.
    """
    log_block("VideoMAE", "VideoMAE unavailable — using fallback frame selection...")
    event_folder = f"{SMART_FRAMES_DIR}/event_{ev_id}"
    os.makedirs(event_folder, exist_ok=True)

    cap_local  = cv2.VideoCapture(video_path_arg)
    fps        = cap_local.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 15.0
    raw_frames = []

    while True:
        ret, frame = cap_local.read()
        if not ret:
            break
        raw_frames.append(frame.copy())
    cap_local.release()

    total_frames = len(raw_frames)
    if debug is not None:
        debug.save_all_frames(raw_frames)
        debug.save_original_video(video_path_arg)
        debug.log(f"[INFO]\nDecoded {total_frames} frames at {fps:.2f} fps "
                   f"(VideoMAE unavailable this event).")

    if total_frames == 0:
        print("[WARNING] Not enough frames")
        return []

    indices, score_rows = _fallback_select(total_frames, raw_frames, quality_scores)

    selected_frames = []
    mapping_rows = []
    for order, idx in enumerate(indices, start=1):
        frame_path = f"{event_folder}/{order:02d}.jpg"
        cv2.imwrite(frame_path, raw_frames[idx])
        selected_frames.append(frame_path)
        mapping_rows.append({
            "order": order,
            "frame_index": idx,
            "timestamp_sec": round(idx / fps, 3),
            "output_path": frame_path,
        })

    if debug is not None:
        selected_set = set(indices)
        for row in score_rows:
            row["selected"] = 1 if row.get("frame_index") in selected_set else 0
            row.setdefault("timestamp_sec", round(row.get("frame_index", 0) / fps, 3))
        debug.write_videomae_scores(score_rows)
        debug.write_frame_mapping(mapping_rows)
        debug.save_selected_frames(selected_frames)

    log_block("VideoMAE", f"Selected (fallback) : {len(selected_frames)} frames")
    return selected_frames


def extract_smart_frames(video_path_arg, ev_id, quality_scores=None, debug=None):
    """
    Extract smart frames from a recorded event video using VideoMAE.

    VideoMAE analyzes the COMPLETE recording — every decoded frame is
    covered by at least one contiguous clip run through the model (see
    _split_into_clips()) — and the final winners are chosen by
    combining VideoMAE's own saliency score with sharpness and face-
    visibility (see _select_indices()), instead of a single novelty-
    only signal or a discarded embedding.

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
            all) — the combined score below is what picks the winner
            among valid candidates. When omitted, absent, or
            mismatched in length, every frame is treated as valid.
        debug: optional utils.debug_artifacts.EventDebug instance. When
            given, this call writes videomae_scores.csv, frame_mapping.csv,
            all_frames/, and selected_frames/ for this event.
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
    fps        = cap_local.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 15.0  # sane default when the container doesn't report one
    raw_frames = []

    while True:
        ret, frame = cap_local.read()
        if not ret:
            break
        raw_frames.append(frame.copy())

    cap_local.release()

    total_frames = len(raw_frames)
    if debug is not None:
        debug.save_all_frames(raw_frames)
        debug.save_original_video(video_path_arg)
        debug.log(f"[INFO]\nDecoded {total_frames} frames at {fps:.2f} fps.")

    if total_frames < _NUM_MODEL_FRAMES:
        print("[WARNING] Not enough frames")
        if debug is not None:
            debug.log("[WARNING]\nNot enough frames for VideoMAE (need at "
                       f"least {_NUM_MODEL_FRAMES}, got {total_frames}).")
        return []

    score_rows = []
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
        indices, score_rows = _select_indices(
            total_frames, all_frame_idx, saliency, quality_scores, raw_frames
        )
        if debug is not None:
            debug.log(f"[SUCCESS]\nVideoMAE scored {len(all_frame_idx)} "
                       f"tubelets across {len(clips)} clips.")
    except Exception as e:
        # Defensive fallback — a shape mismatch (e.g. a differently
        # configured VideoMAE checkpoint) must never crash event
        # finalisation. Falls back to a blur/quality-only selection
        # (see _fallback_select()) rather than the old discarded-
        # embedding behaviour.
        print(f"[WARNING] VideoMAE-driven frame selection failed ({e}); "
              f"falling back to blur/quality-only selection.")
        if debug is not None:
            debug.log(f"[ERROR]\nVideoMAE-driven frame selection failed.\n"
                       f"Reason:\n{type(e).__name__}: {e}")
        indices, score_rows = _fallback_select(total_frames, raw_frames, quality_scores)

    selected_frames = []
    mapping_rows = []
    for order, idx in enumerate(indices, start=1):
        frame_path = f"{event_folder}/{order:02d}.jpg"
        cv2.imwrite(frame_path, raw_frames[idx])
        selected_frames.append(frame_path)
        mapping_rows.append({
            "order": order,
            "frame_index": idx,
            "timestamp_sec": round(idx / fps, 3),
            "output_path": frame_path,
        })

    if debug is not None:
        selected_set = set(indices)
        for row in score_rows:
            row["selected"] = 1 if row.get("frame_index") in selected_set else 0
            row.setdefault("timestamp_sec", round(row.get("frame_index", 0) / fps, 3))
        debug.write_videomae_scores(score_rows)
        debug.write_frame_mapping(mapping_rows)
        debug.save_selected_frames(selected_frames)
        debug.log(f"[SUCCESS]\nSelected {len(selected_frames)} Smart Frames: "
                   f"{[r['frame_index'] for r in mapping_rows]}")

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


# How many saliency-ranked candidates per third get the (more
# expensive) blur + face-detection scoring. Keeps the extra cost
# bounded regardless of how long an event runs, instead of scoring
# every tubelet in the whole recording.
_SHORTLIST_SIZE = 6

# Below this Laplacian-variance value a frame is considered too
# blurry/motion-smeared to be a good representative frame, unless
# every shortlisted candidate in that third is equally blurry (in
# which case there's nothing better to fall back to).
_MIN_SHARPNESS = 40.0


def _blur_score(frame) -> float:
    """
    Laplacian-variance sharpness score — the standard, cheap
    "how in-focus is this frame" measure. Higher = sharper. A frame
    with heavy motion blur or an out-of-focus subject scores low here
    even if VideoMAE's saliency score liked that instant.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _face_score(frame) -> float:
    """
    Face-visibility score for one candidate frame — reuses the SAME
    already-loaded InsightFace detector as face/recognizer.py (no
    second model, no extra load; returns 0.0 immediately if face
    recognition is disabled or not yet loaded, per detect_faces()'s
    own contract). Combines the largest detected face's area (bigger,
    closer face = more useful) with SCRFD's own detection confidence
    (`det_score`) so a clear frontal face outscores a barely-detected
    sliver of a profile. 0.0 whenever no face is visible at all —
    e.g. a back view.
    """
    try:
        from face.recognizer import detect_faces
        faces = detect_faces(frame)
    except Exception:
        return 0.0

    if not faces:
        return 0.0

    best = max(faces, key=lambda f: f["area"] * f.get("det_score", 1.0))
    return float(best["area"] * best.get("det_score", 1.0))


def _normalize(values):
    """Min-max normalize to [0, 1]; a constant (or empty) list maps to
    all-zeros rather than dividing by zero."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _select_indices(total_frames, tubelet_frame_idx, saliency, quality_scores, raw_frames):
    """
    Pick 3 raw-frame indices — one each from the first / middle / last
    third of the event, to keep the "entering, main interaction,
    leaving" temporal spread.

    Within each third, the winner is chosen by a WEIGHTED COMBINATION
    of three signals (see the module docstring's "ISSUE 1 ROOT CAUSE"
    section for why saliency alone picked back/blurry frames):

      - VideoMAE's saliency score (_temporal_saliency())   — representativeness
      - Sharpness (_blur_score())                           — is it in focus
      - Face visibility (_face_score())                     — is a face visible

    To keep this cheap on a long event, blur/face are only computed
    for the top `_SHORTLIST_SIZE` saliency-ranked candidates in each
    third, not every tubelet. `quality_scores`, when available, is
    still used first as the validity gate (skip a candidate with no
    confirmed detection at all).

    Returns:
        (indices, score_rows) — `indices` is the 3 chosen raw-frame
        indices; `score_rows` is a list[dict] (one row per candidate
        that was scored) suitable for utils.debug_artifacts.
        EventDebug.write_videomae_scores() — used for debugging, not
        by the selection logic itself.
    """
    fallback = list(np.linspace(0, total_frames - 1, 3, dtype=int))
    valid_quality = (
        quality_scores is not None and len(quality_scores) == total_frames
    )

    bounds = [0, total_frames // 3, (2 * total_frames) // 3, total_frames]
    indices = []
    score_rows = []

    for b in range(3):
        lo, hi = bounds[b], bounds[b + 1]
        if hi <= lo:
            indices.append(fallback[b])
            continue

        # Every tubelet whose centre frame falls inside this third.
        candidates = [
            t for t, f in enumerate(tubelet_frame_idx) if lo <= f < hi
        ]
        gated_out = []
        if valid_quality:
            with_detection = [
                t for t in candidates if quality_scores[tubelet_frame_idx[t]] > 0
            ]
            if with_detection:
                gated_out = [t for t in candidates if t not in with_detection]
                candidates = with_detection

        # Record the frames that were gated out purely for the debug
        # trail (quality_gate=0, never scored further).
        for t in gated_out:
            score_rows.append({
                "frame_index": tubelet_frame_idx[t],
                "saliency": float(saliency[t]),
                "quality_gate": 0,
            })

        if not candidates:
            indices.append(fallback[b])
            continue

        # Shortlist by VideoMAE saliency first — keeps blur/face
        # scoring bounded regardless of event length.
        shortlist = sorted(candidates, key=lambda t: saliency[t], reverse=True)[:_SHORTLIST_SIZE]

        blur_vals = [_blur_score(raw_frames[tubelet_frame_idx[t]]) for t in shortlist]
        face_vals = [_face_score(raw_frames[tubelet_frame_idx[t]]) for t in shortlist]
        saliency_vals = [float(saliency[t]) for t in shortlist]

        norm_saliency = _normalize(saliency_vals)
        norm_blur     = _normalize(blur_vals)
        norm_face     = _normalize(face_vals)

        has_face_signal = max(face_vals) > 0.0

        combined = []
        for i in range(len(shortlist)):
            if has_face_signal:
                score = 0.45 * norm_face[i] + 0.40 * norm_saliency[i] + 0.15 * norm_blur[i]
            else:
                # Face recognition disabled, or nobody's face visible
                # anywhere in this third — fall back to saliency+blur.
                score = 0.6 * norm_saliency[i] + 0.4 * norm_blur[i]

            # Hard-deprioritize genuinely blurry candidates unless
            # every option in the shortlist is equally blurry.
            if blur_vals[i] < _MIN_SHARPNESS and max(blur_vals) >= _MIN_SHARPNESS:
                score *= 0.1

            combined.append(score)
            score_rows.append({
                "frame_index": tubelet_frame_idx[shortlist[i]],
                "saliency": saliency_vals[i],
                "blur": blur_vals[i],
                "face_score": face_vals[i],
                "quality_gate": 1,
                "combined_score": round(score, 6),
            })

        best_i = max(range(len(shortlist)), key=lambda i: combined[i])
        indices.append(tubelet_frame_idx[shortlist[best_i]])

    return indices, score_rows


def _fallback_select(total_frames, raw_frames, quality_scores):
    """
    Used ONLY when VideoMAE itself is unavailable for this event (see
    ISSUE 2 — models/model_manager.py) or its forward pass raised.
    Picks one frame per first/middle/last third using JUST sharpness
    + the detection-quality gate (no VideoMAE saliency term, since
    VideoMAE didn't run) — still far better than a blind evenly-spaced
    pick, and still avoids empty/no-detection frames.
    """
    valid_quality = (
        quality_scores is not None and len(quality_scores) == total_frames
    )
    bounds = [0, total_frames // 3, (2 * total_frames) // 3, total_frames]
    indices = []
    score_rows = []

    for b in range(3):
        lo, hi = bounds[b], bounds[b + 1]
        if hi <= lo:
            indices.append(min(max(lo, 0), total_frames - 1))
            continue

        candidates = list(range(lo, hi))
        if valid_quality:
            with_detection = [i for i in candidates if quality_scores[i] > 0]
            if with_detection:
                candidates = with_detection

        blur_vals = [_blur_score(raw_frames[i]) for i in candidates]
        for i, idx in enumerate(candidates):
            score_rows.append({
                "frame_index": idx,
                "blur": blur_vals[i],
                "quality_gate": 1,
                "combined_score": blur_vals[i],
            })

        best_i = max(range(len(candidates)), key=lambda i: blur_vals[i])
        indices.append(candidates[best_i])

    return indices, score_rows