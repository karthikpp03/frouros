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


def generate_summary(frame_paths):
    """
    Given a list of frame image paths (from VideoMAE), produce a natural
    CCTV event summary using Qwen2.5-VL-7B.
    Identical to the original generate_summary().
    """
    print("[INFO] Generating AI summary (Qwen2.5-VL-7B)...")
    pil_images = [Image.open(fp) for fp in frame_paths]
    messages   = build_summary_messages(pil_images)
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
    Identical to the original _heuristic_extract().
    """
    text = summary.lower()

    colors   = ["black", "white", "red", "blue", "green", "yellow", "grey",
                 "gray", "brown", "pink", "orange", "purple"]
    clothing = ["shirt", "jacket", "coat", "hoodie", "dress", "jeans",
                 "trousers", "cap", "hat", "bag", "backpack"]

    appearance_parts = []
    for color in colors:
        for item in clothing:
            if color in text and item in text:
                appearance_parts.append(f"{color} {item}")

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
        "appearance": ", ".join(appearance_parts) if appearance_parts else None,
        "actions":    actions,
        "objects":    objects,
        "movement":   None,
        "waiting":    waiting,
    }]
