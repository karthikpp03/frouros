"""
utils/image_merger.py
======================
Used only by the OpenAI branch of services/summary_router.py. Does not
touch, import from, or get imported by any part of the existing
Qwen/Groq pipeline — Qwen keeps using the 3 separate smart frames
exactly as before.

Takes the 3 smart-frame image paths that VideoMAE already produces and
merges them into a single side-by-side image, so the whole event can be
sent to OpenAI Vision as ONE image instead of 3 separate ones (fewer
tokens per event). Each frame gets a small label above it (BEGINNING /
MIDDLE / END) so the arrangement is unambiguous even without a text
prompt.

Exports:
  merge_frames_horizontally(frame_paths, event_id) -> str (saved path)
"""

import os
from PIL import Image, ImageDraw, ImageFont

from config.settings import MERGED_EVENTS_DIR
from utils.event_logger import log_block

# Labels shown above each frame, left to right, chronological order.
_LABELS = ["Frame 1 - BEGINNING", "Frame 2 - MIDDLE", "Frame 3 - END"]

_TARGET_HEIGHT = 360      # each frame is resized to this height (aspect preserved)
_LABEL_BAND_H  = 30       # pixels reserved above each frame for its label
_GAP           = 8        # gap between frames
_BG_COLOR      = (20, 20, 20)
_LABEL_COLOR   = (255, 255, 255)


def _load_font():
    """Best-effort load of a small readable font; falls back to PIL default."""
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, 16)
            except Exception:
                pass
    return ImageFont.load_default()


def merge_frames_horizontally(frame_paths, event_id):
    """
    Merge the 3 VideoMAE smart frames into one labelled, side-by-side
    image and save it to data/merged_events/event_<id>.jpg.

    Args:
        frame_paths: list of 3 image file paths (order = chronological,
                      exactly what models/videomae.py.extract_smart_frames()
                      already returns).
        event_id:    the event id, used only for the output filename.

    Returns:
        Path to the saved merged image (str).
    """
    if not frame_paths:
        raise ValueError("merge_frames_horizontally: no frame paths given")

    os.makedirs(MERGED_EVENTS_DIR, exist_ok=True)

    images = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGB")
        # Preserve aspect ratio while normalising height so all 3 panels
        # line up neatly regardless of the source frame's resolution.
        w, h   = img.size
        new_w  = max(1, int(w * (_TARGET_HEIGHT / h)))
        img    = img.resize((new_w, _TARGET_HEIGHT))
        images.append(img)

    font = _load_font()

    total_w = sum(im.width for im in images) + _GAP * (len(images) - 1)
    total_h = _TARGET_HEIGHT + _LABEL_BAND_H

    canvas = Image.new("RGB", (total_w, total_h), _BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    x = 0
    for i, img in enumerate(images):
        label = _LABELS[i] if i < len(_LABELS) else f"Frame {i + 1}"

        # Center the label above its frame.
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
        except Exception:
            text_w = len(label) * 7
        text_x = x + max(0, (img.width - text_w) // 2)
        draw.text((text_x, 6), label, fill=_LABEL_COLOR, font=font)

        canvas.paste(img, (x, _LABEL_BAND_H))
        x += img.width + _GAP

    out_path = os.path.join(MERGED_EVENTS_DIR, f"event_{event_id}.jpg")
    canvas.save(out_path, quality=92)

    log_block("MERGE", "Merged image created", out_path)
    return out_path
