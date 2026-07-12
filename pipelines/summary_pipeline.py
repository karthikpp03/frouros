"""
pipelines/summary_pipeline.py
==============================
Post-event AI processing:
  generate_summary(frame_paths)          → str
  extract_person_attributes(summary)     → list[dict]
  _heuristic_extract(summary)            → list[dict]   (fallback)

Uses Qwen2.5-VL-7B for both steps.
All logic and prompts are preserved verbatim from the original monolith.
Prompts are sourced from prompts/summary_prompts.py.
"""

import re
import json
from PIL import Image

from models.qwen_vl import _qwen_infer
#from models.smolvlm import _qwen_infer
from prompts.summary_prompts import (
    build_summary_messages,
    build_attribute_extraction_prompt,
)


def generate_summary(frame_paths, participants=None):
    """
    Given a list of frame image paths (from VideoMAE), produce a natural
    CCTV event summary using Qwen2.5-VL-7B.

    ISSUE 3 FIX — Qwen must become scene aware: this used to accept a
    single `person_name` and could only ever describe one person. It
    now accepts the Scene Event's COMPLETE participant list instead, so
    Qwen always describes the whole scene (every known and unknown
    participant, in join order) in one combined summary.

    Args:
        frame_paths:  unchanged — the 3 VideoMAE smart-frame paths.
        participants: list[dict] | None — every participant who was
                      ever part of this Scene Event, forwarded straight
                      to prompts.summary_prompts.build_summary_messages()
                      (see there for the exact shape). None preserves
                      the original single-unknown-person behaviour.
    """
    print("[INFO] Generating AI summary (Qwen2.5-VL-7B)...")
    pil_images = [Image.open(fp) for fp in frame_paths]
    messages   = build_summary_messages(pil_images, participants=participants)
    return _qwen_infer(messages, pil_images=pil_images, max_new_tokens=250)


def extract_person_attributes(summary):
    """
    Extract structured person data from a free-text summary.
    Returns a list of dicts with keys: appearance, actions, objects,
    movement, waiting.
    Falls back to _heuristic_extract() on JSON parse failure.
    Identical to the original extract_person_attributes().
    """
    print("[INFO] Extracting person attributes (Qwen2.5-VL-7B)...")

    messages = build_attribute_extraction_prompt(summary)
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
    """
    Keyword-based fallback when JSON parsing fails.
    Identical to the original _heuristic_extract() for `appearance`/
    `actions`/`objects`/`movement`/`waiting`, plus new per-field
    clothing/bag detection (ISSUE 1) so this fallback path — used only
    when Qwen's JSON extraction fails to parse — can never leave
    top_clothing/bottom_clothing/bag null while the summary text
    plainly describes them (the same consistency guarantee the primary
    JSON-extraction prompt now enforces; see
    prompts/summary_prompts.build_attribute_extraction_prompt()).
    """
    text = summary.lower()

    colors   = ["black", "white", "red", "blue", "green", "yellow", "grey",
                 "gray", "brown", "pink", "orange", "purple", "light",
                 "dark", "navy", "beige"]
    clothing = ["shirt", "jacket", "coat", "hoodie", "dress", "jeans",
                 "trousers", "cap", "hat", "bag", "backpack"]

    appearance_parts = []
    for color in colors:
        for item in clothing:
            if color in text and item in text:
                appearance_parts.append(f"{color} {item}")

    def _first_match(items):
        """First color+item phrase found in `text` for this category,
        falling back to a bare item mention (no color) so a summary
        like "wearing a shirt" (no color given) still populates the
        field instead of leaving it null."""
        for color in colors:
            for item in items:
                if color in text and item in text:
                    return f"{color} {item}".title()
        for item in items:
            if item in text:
                return item.title()
        return None

    top_items      = ["shirt", "jacket", "coat", "hoodie", "sweater", "blouse", "t-shirt"]
    bottom_items    = ["jeans", "trousers", "pants", "shorts", "skirt"]
    footwear_items  = ["shoes", "boots", "sneakers", "sandals"]
    headwear_items  = ["cap", "hat", "helmet"]

    top_clothing    = _first_match(top_items)
    bottom_clothing = _first_match(bottom_items)
    footwear        = _first_match(footwear_items)
    headwear        = _first_match(headwear_items)

    # Bag detection, most-specific phrase first, so "shoulder bag" is
    # reported as exactly that rather than the generic "bag".
    bag = None
    for phrase in ["shoulder bag", "handbag", "backpack", "bag"]:
        if phrase in text:
            bag = phrase.title()
            break

    action_keywords = {
        "used phone":    ["phone", "mobile", "smartphone"],
        "carried bag":   ["bag", "backpack", "luggage"],
        "waited":        ["wait", "stood", "standing", "lingered"],
        "entered":       ["enter", "arrived", "came in"],
        "exited":        ["exit", "left", "departed"],
        "interacted":    ["interact", "talked", "spoke"],
        "looked around": ["looked around", "scanning"],
    }
    actions = [
        a for a, kws in action_keywords.items()
        if any(k in text for k in kws)
    ]

    object_keywords = [
        "phone", "bag", "backpack", "umbrella", "bottle",
        "helmet", "laptop", "box",
    ]
    objects = [o for o in object_keywords if o in text]

    waiting = any(w in text for w in ["wait", "stood", "standing", "lingered"])

    return [{
        "appearance":      ", ".join(appearance_parts) if appearance_parts else None,
        "top_clothing":    top_clothing,
        "bottom_clothing": bottom_clothing,
        "footwear":        footwear,
        "headwear":        headwear,
        "bag":             bag,
        "accessories":     None,
        "actions":    actions,
        "objects":    objects,
        "movement":   None,
        "waiting":    waiting,
    }]