"""
prompts/summary_prompts.py
==========================
Prompt builders for the Qwen2.5-VL summary and attribute-extraction steps.
Functions return fully-formed message dicts ready for _qwen_infer().
"""

from prompts.grounding_rules import _GROUNDING_RULES

def build_summary_messages(pil_images, person_name=None):
    """
    Returns the messages list passed to _qwen_infer() for event summarisation.
    Images are interleaved with explicit stage labels (Beginning/Middle/End)
    so the model treats them as one temporal sequence instead of three
    independent images.

    Args:
        pil_images:  the 3 chronological smart frames (unchanged).
        person_name: str | None — the real, recognized identity from
                      face/recognizer.py (e.g. "Dad", "test_1") if this
                      event was routed to Qwen because a registered face
                      was recognized. When provided, Qwen is explicitly
                      told to refer to the individual by this name
                      throughout the summary instead of a generic term
                      ("the person", "a person", "someone", "individual").
                      None when face recognition is disabled/unavailable
                      for this event — behaviour is then identical to
                      before this parameter existed.
    """
    stage_labels = ["BEGINNING of the event", "MIDDLE of the event", "END of the event"]

    content = []
    for i, img in enumerate(pil_images):
        label = stage_labels[i] if i < len(stage_labels) else f"Frame {i+1}"
        content.append({"type": "text", "text": f"--- Frame {i+1} ({label}) ---"})
        content.append({"type": "image", "image": img})

    identity_instruction = ""
    if person_name:
        identity_instruction = (
            f"IDENTITY: The person in these frames has been positively "
            f"identified by the face-recognition system as \"{person_name}\". "
            f"You MUST refer to them by this exact name (\"{person_name}\") "
            f"everywhere in the summary — e.g. \"{person_name} entered the "
            f"house\", \"{person_name} walked toward the door\". Do NOT refer "
            f"to them as \"the person\", \"a person\", \"someone\", "
            f"\"individual\", or \"known person\" anywhere in the summary.\n\n"
        )

    content.append({
        "type": "text",
        "text": (
            f"{_GROUNDING_RULES}\n\n"
            f"{identity_instruction}"
            "The 3 images above are consecutive frames from ONE SINGLE CCTV event, "
            "in strict chronological order: Frame 1 = beginning, Frame 2 = middle, "
            "Frame 3 = end.\n\n"

            "Do NOT describe each frame separately or in isolation. Instead, "
            "COMPARE the frames against each other to infer motion, action "
            "progression, and behavior change over time, then write ONE "
            "combined narrative summary of the whole event.\n\n"

            "Specifically:\n"
            "- Compare the person's position/pose in Frame 1 vs Frame 2 vs Frame 3.\n"
            "- If the person's position changes between frames, explicitly describe "
            "the transition (e.g. 'started standing near the entrance, then began "
            "walking toward the corridor, and finally exited the frame').\n"
            "- If the person's position does NOT change across all 3 frames, "
            "explicitly state that they remained stationary throughout the event.\n"
            "- Describe the overall direction of movement (e.g. left-to-right, "
            "toward/away from camera) if visible.\n"
            "- Note any objects carried and whether that changes across frames.\n"
            "- Note clothing/appearance once (it won't change across frames).\n"
            "- Describe the final outcome/state as seen in Frame 3.\n\n"

            "Analyze ONLY what is reasonably visible in the frames.\n"
            "Do NOT invent conversations, emotions, intentions, or relationships.\n"
            "If something is unclear, say 'Not clearly visible'.\n\n"

            "Write a natural CCTV surveillance event summary as ONE continuous "
            "paragraph (or short set of paragraphs) describing the event's motion "
            "and behavior progression from beginning to end — NOT three separate "
            "per-frame descriptions.\n\n"

            "The report should feel like a professional CCTV operator summary: "
            "clear, grounded, chronological, and natural.\n\n"

            "Avoid robotic formatting like 'Person A:' or 'Frame 1:' repeatedly "
            "in the output — the frame labels above are for your reference only, "
            "not for the final summary text.\n"
            "Avoid repeating the same sentence structure.\n"
            "Avoid excessive speculation.\n"
        )
    })

    return [{"role": "user", "content": content}]

def build_attribute_extraction_prompt(summary):
    """
    Returns the messages list passed to _qwen_infer() for structured
    attribute extraction from a free-text summary.
    Exactly preserves the original inline prompt text.
    """
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
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
