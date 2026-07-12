"""
prompts/summary_prompts.py
==========================
Prompt builders for the Qwen2.5-VL summary and attribute-extraction steps.
Functions return fully-formed message dicts ready for _qwen_infer().
"""

from prompts.grounding_rules import _GROUNDING_RULES


def _label_participants(participants):
    """
    Turns the Scene Event's participant list into label-annotated dicts
    Qwen must use verbatim as the subject of every sentence about that
    participant — ISSUE 3 (Qwen must become scene aware).

    Args:
        participants: list[dict], one per participant who was ever part
                      of this Scene Event (in join order), each shaped:
                        {"name": str | None,   # real recognized name,
                                                # or None if unrecognized
                         "track_id": int,
                         "joined_at": str,     # "HH:MM:SS" or full ts
                         "left_at": str}       # "HH:MM:SS" or full ts
                      `name` is the real identity from face/recognizer.py
                      (e.g. "Dad", "Mom", "test_1") — never a placeholder.

    Returns:
        list[dict] — each participant's dict, with a "label" key added:
        the recognized name verbatim if known, otherwise "Unknown" (if
        this is the only unrecognized participant) or "Unknown visitor N"
        (N = join order among unknowns, if more than one participant is
        unrecognized) so multiple unknown participants in the same scene
        stay distinguishable from one another instead of collapsing into
        one indistinguishable "Unknown".
    """
    unknown_count = sum(1 for p in participants if not p.get("name"))
    labeled = []
    unknown_seen = 0
    for p in participants:
        name = p.get("name")
        if name:
            label = name
        else:
            unknown_seen += 1
            label = "Unknown" if unknown_count == 1 else f"Unknown visitor {unknown_seen}"
        labeled.append({**p, "label": label})
    return labeled


def build_summary_messages(pil_images, participants=None):
    """
    Returns the messages list passed to _qwen_infer() for event summarisation.
    Images are interleaved with explicit stage labels (Beginning/Middle/End)
    so the model treats them as one temporal sequence instead of three
    independent images.

    ISSUE 3 FIX — Qwen must become scene aware: previously this only
    accepted a single `person_name` and could describe exactly one
    person. A Scene Event can hold any number of participants (known
    and/or unknown) who joined and left at different times, so Qwen now
    receives the COMPLETE participant list — every real name, every
    "Unknown"/"Unknown visitor N" label for unrecognized participants,
    and each participant's join/leave time — and is instructed to
    describe the whole interaction between them in one combined,
    chronological scene summary, never one summary per person and
    never a generic placeholder ("Person", "Known Person", "Someone",
    "Recognized Person") for anyone.

    Args:
        pil_images:   the 3 chronological smart frames (unchanged).
        participants: list[dict] | None — every participant who was
                      ever part of this Scene Event, in join order (see
                      _label_participants() above for the exact shape).
                      None / empty behaves like a scene with a single
                      unrecognized participant ("Unknown person"), the
                      original pre-scene-aware behaviour.
    """
    # Internal-only stage ordering hints — these words are for the model's
    # OWN reasoning about chronology and must never leak into the output
    # text (see the explicit ban below). Deliberately NOT phrased as
    # "Frame 1/2/3" or "first/second/third image" so there's no ready-made
    # numbering vocabulary sitting right next to the images to copy from.
    stage_labels = ["earliest moment", "mid-point", "final moment"]

    content = []
    for i, img in enumerate(pil_images):
        label = stage_labels[i] if i < len(stage_labels) else f"moment {i+1}"
        content.append({"type": "text", "text": f"[Chronological reference only — {label} of the event, not for output]"})
        content.append({"type": "image", "image": img})

    # IDENTITY — ISSUE 3 FIX: resolved to the COMPLETE participant list
    # of the Scene Event, not a single subject. Every known participant
    # is named by their real, recognized identity; every unrecognized
    # participant gets "Unknown" (or "Unknown visitor N" if there is
    # more than one unrecognized participant, so they stay
    # distinguishable) — exactly the labels _label_participants()
    # above assigned. No participant, known or unknown, is ever
    # replaced with a generic placeholder such as "Person", "Known
    # Person", "Someone", or "Recognized Person".
    labeled_participants = _label_participants(participants or [])

    if labeled_participants:
        roster_lines = []
        for p in labeled_participants:
            joined = p.get("joined_at")
            left   = p.get("left_at")
            timing = f" (joined {joined}" + (f", left {left}" if left else "") + ")" if joined else ""
            roster_lines.append(f"  - \"{p['label']}\"{timing}")
        roster_block = "\n".join(roster_lines)

        identity_instruction = (
            "IDENTITY — COMPLETE PARTICIPANT LIST: this Scene Event "
            "involves the following participant(s), listed in the order "
            "they joined the scene. Every one of them may appear in the "
            "images above, at overlapping or different times:\n"
            f"{roster_block}\n\n"
            "For EVERY sentence about a participant, use their EXACT "
            "label from the list above as the subject — the real name if "
            "known (e.g. \"Dad\", \"Mom\", \"test_1\"), or their exact "
            "\"Unknown\"/\"Unknown visitor N\" label if unrecognized. "
            "Never invent a name, never merge two participants into one, "
            "and never drop a participant who is listed above.\n"
        )
    else:
        # No participant list was supplied — original single-unknown-
        # person behaviour, unchanged.
        identity_instruction = (
            "IDENTITY: No registered face was matched for this event. Refer "
            "to the subject as the exact phrase \"Unknown person\" "
            "everywhere in the summary — every sentence about them must "
            "have \"Unknown person\" as the subject, e.g. \"Unknown person "
            "entered the monitored area.\", \"Unknown person carried a "
            "backpack.\", \"Unknown person exited the monitored area.\"\n"
        )

    content.append({
        "type": "text",
        "text": (
            f"{_GROUNDING_RULES}\n\n"
            f"{identity_instruction}\n"
            "You are a professional CCTV surveillance analyst writing an "
            "incident report. The images above are three chronological "
            "snapshots of ONE SINGLE continuous scene — treat them exactly "
            "like three moments the same scene unfolded across, never as "
            "separate, independent images.\n\n"

            "ABSOLUTE OUTPUT BAN — the final summary text must NEVER "
            "contain any of the following, in any form:\n"
            "  \"Frame 1\", \"Frame 2\", \"Frame 3\", \"frame one/two/three\"\n"
            "  \"the first image\", \"the second image\", \"the third image\"\n"
            "  \"image 1/2/3\", \"picture 1/2/3\"\n"
            "  any other language that numbers, labels, or refers to the "
            "individual snapshots as separate items.\n"
            "The bracketed captions above each image are for YOUR internal "
            "ordering only — never copy them, quote them, or refer to them "
            "in the output.\n\n"

            "NEVER refer to any participant as \"the person\", \"a "
            "person\", \"someone\", \"individual\", \"known person\", or "
            "\"recognized person\" — always use the exact name/label given "
            "in the IDENTITY list above, every time you mention them.\n\n"

            "Write ONE complete, continuous professional surveillance "
            "report covering EVERY participant listed above and how they "
            "interacted — not a separate summary per person, and not "
            "focused on only one participant while ignoring the others. "
            "Describe the scene as it actually unfolded, in chronological "
            "order using each participant's join/leave timing as a guide, "
            "for example:\n\n"

            "  <Participant A> entered the monitored area "
            "<carrying/while doing X, if visible>.\n"
            "  <Participant B> joined shortly afterward.\n"
            "  <Describe any interaction between participants here, if "
            "visible — e.g. one handed an object to another, they spoke, "
            "they walked together.>\n"
            "  <Participant A> <exited the monitored area, or the final "
            "observed state, etc., continuing for every participant>.\n\n"

            "Compare each participant's position, pose, and any carried "
            "objects across the three moments to infer motion and behavior "
            "progression — describe transitions explicitly (e.g. moved from "
            "the entrance toward the corridor) rather than describing each "
            "moment in isolation. If a participant's position does not "
            "change at all, state plainly that they remained stationary. "
            "Mention each participant's clothing/appearance and carried "
            "objects once, not per sentence.\n\n"

            "Analyze ONLY what is reasonably visible. Do NOT invent "
            "conversations, emotions, intentions, or relationships between "
            "participants beyond what the images actually show. If "
            "something is unclear, say 'Not clearly visible'.\n\n"

            "Keep the writing style consistent with a professional CCTV "
            "incident report — the same tone and structure OpenAI's "
            "summaries use: clear, grounded, chronological, natural, and "
            "free of robotic labels or repeated sentence structure.\n"
        )
    })

    return [{"role": "user", "content": content}]

def build_attribute_extraction_prompt(summary):
    """
    Returns the messages list passed to _qwen_infer() for structured
    attribute extraction from a free-text summary.

    ISSUE 1 FIX: previously this only extracted a single free-text
    `appearance` string, which services/db_writer.py's
    build_structured_from_qwen() never actually mapped onto the
    per-field SQLite columns (top_clothing, bottom_clothing, bag,
    ...) — those were hardcoded to "Not Clearly Visible" regardless of
    what the summary said, so a summary that clearly described a
    "light blue shirt", "dark pants", and a "shoulder bag" still
    persisted those three fields as unknown. This extraction now asks
    Qwen for the same per-field breakdown OpenAI's contract uses
    (models/openai_vl.py's _OPENAI_PROMPT), extracted directly from
    the summary text itself — so the structured JSON can never
    disagree with the summary it was derived from, and
    build_structured_from_qwen() (below, in services/db_writer.py) now
    reads these fields directly instead of discarding them.
    """
    prompt = (
        f"{_GROUNDING_RULES}\n\n"
        "Extract structured data from the CCTV event description below.\n\n"
        "Return ONLY a valid JSON array. No markdown fences, no explanation.\n"
        "Each element represents one person and must have exactly these keys:\n"
        "  appearance: string or null   (general appearance summary)\n"
        "  top_clothing: string or null (e.g. 'Light blue shirt')\n"
        "  bottom_clothing: string or null (e.g. 'Dark pants')\n"
        "  footwear: string or null\n"
        "  headwear: string or null\n"
        "  bag: string or null          (e.g. 'Shoulder bag' — null if no bag mentioned)\n"
        "  accessories: string or null\n"
        "  actions: array of strings\n"
        "  objects: array of strings\n"
        "  movement: string or null\n"
        "  waiting: boolean\n\n"
        "Critical rules:\n"
        "- Extract ONLY what is explicitly written in the description below.\n"
        "- Do NOT infer or add details not present in the text.\n"
        "- If a field is not mentioned, use null (or [] for array fields).\n"
        "- CONSISTENCY IS MANDATORY: every structured field must exactly match "
        "what the description says. If the description mentions a top garment, "
        "top_clothing must NOT be null. If it mentions a bottom garment, "
        "bottom_clothing must NOT be null. If it mentions any kind of bag "
        "(backpack, shoulder bag, handbag, etc.), bag must NOT be null — it "
        "must be that same item. Never contradict the description and never "
        "leave a field null when the description clearly states it.\n"
        "- waiting: true only if 'loitering', 'waiting', or 'standing' appears.\n\n"
        f"Event description:\n{summary}\n\n"
        "JSON array (start with [ end with ]):"
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]