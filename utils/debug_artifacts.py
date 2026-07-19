"""
utils/debug_artifacts.py
==========================
Per-event debugging artifacts for Smart Frame Selection + AI
processing, so a bad Smart Frame (or a failed model) can be diagnosed
after the fact without reproducing the event.

For every processed event, produces:

    debug/event_<id>/
        original_video.mp4     - copy of the exact recording that was analyzed
        all_frames/            - every decoded frame (NNNN.jpg), for eyeballing
                                  what VideoMAE actually saw
        selected_frames/       - the final Smart Frames, in selection order
        videomae_scores.csv    - per-candidate: frame_index, timestamp,
                                  saliency, blur, face score, quality gate,
                                  combined score, selected (1/0)
        frame_mapping.csv      - order -> raw frame index -> timestamp ->
                                  output path, for the final selected frames
        processing_log.txt     - the same [INFO]/[SUCCESS]/[ERROR] lines
                                  printed to stdout for this event, saved

This module only writes files — it has no opinion on the Smart Frame
algorithm itself (see models/videomae.py).
"""

import csv
import os
import shutil

import cv2

from config.settings import DEBUG_EVENTS_DIR


class EventDebug:
    """One instance per event. Cheap to construct; every write is
    best-effort (a failed debug write must never break real event
    processing — see the try/except in every method)."""

    def __init__(self, event_id):
        self.event_id = event_id
        self.root = os.path.join(DEBUG_EVENTS_DIR, f"event_{event_id}")
        self.all_frames_dir = os.path.join(self.root, "all_frames")
        self.selected_frames_dir = os.path.join(self.root, "selected_frames")
        self._log_lines = []

        try:
            os.makedirs(self.all_frames_dir, exist_ok=True)
            os.makedirs(self.selected_frames_dir, exist_ok=True)
        except Exception as e:
            print(f"[WARNING] Could not create debug folder for event {event_id}: {e}")

    # ------------------------------------------------------------------
    def log(self, message: str):
        """Record one line for processing_log.txt (also visible on
        stdout already via the callers' own print()/log_block() calls
        — this just persists a copy alongside the other artifacts)."""
        self._log_lines.append(str(message))

    def flush_log(self):
        try:
            with open(os.path.join(self.root, "processing_log.txt"), "w") as f:
                f.write("\n".join(self._log_lines) + "\n")
        except Exception as e:
            print(f"[WARNING] Could not write processing_log.txt for event {self.event_id}: {e}")

    # ------------------------------------------------------------------
    def save_original_video(self, video_path):
        try:
            shutil.copy2(video_path, os.path.join(self.root, "original_video.mp4"))
        except Exception as e:
            print(f"[WARNING] Could not copy original video for event {self.event_id}: {e}")

    def save_all_frames(self, frames):
        """`frames`: list of BGR numpy arrays, in original order."""
        try:
            for i, frame in enumerate(frames):
                cv2.imwrite(os.path.join(self.all_frames_dir, f"{i:05d}.jpg"), frame)
        except Exception as e:
            print(f"[WARNING] Could not save all_frames for event {self.event_id}: {e}")

    def save_selected_frames(self, frame_paths):
        """`frame_paths`: the final Smart Frame file paths, in order."""
        try:
            for order, path in enumerate(frame_paths, start=1):
                if os.path.exists(path):
                    shutil.copy2(
                        path,
                        os.path.join(self.selected_frames_dir, f"{order:02d}.jpg"),
                    )
        except Exception as e:
            print(f"[WARNING] Could not save selected_frames for event {self.event_id}: {e}")

    def write_videomae_scores(self, rows):
        """
        `rows`: list[dict] with keys frame_index, timestamp_sec,
        saliency, blur, face_score, quality_gate, combined_score,
        selected. Missing keys are written as blank cells (e.g. a
        candidate that was never shortlisted for blur/face scoring).
        """
        fieldnames = [
            "frame_index", "timestamp_sec", "saliency", "blur",
            "face_score", "quality_gate", "combined_score", "selected",
        ]
        try:
            with open(os.path.join(self.root, "videomae_scores.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
        except Exception as e:
            print(f"[WARNING] Could not write videomae_scores.csv for event {self.event_id}: {e}")

    def write_frame_mapping(self, rows):
        """`rows`: list[dict] with keys order, frame_index,
        timestamp_sec, output_path — the final selected frames only."""
        fieldnames = ["order", "frame_index", "timestamp_sec", "output_path"]
        try:
            with open(os.path.join(self.root, "frame_mapping.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
        except Exception as e:
            print(f"[WARNING] Could not write frame_mapping.csv for event {self.event_id}: {e}")
