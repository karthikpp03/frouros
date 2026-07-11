"""
models/openai_vl.py
====================
OpenAI Vision client. Completely independent from models/qwen_vl.py —
neither module imports the other, and both can coexist.

SAFETY — read before touching this file
-----------------------------------------
The `openai` package is imported, and the client constructed, ONLY
inside _get_client(), which only ever runs when generate_openai_analysis()
is called. The only caller of generate_openai_analysis() is
services/summary_router.py, and it only calls it when USE_OPENAI=true.

When USE_OPENAI=false:
  - services/summary_router.py never calls into this module.
  - This module is therefore never imported, the `openai` package is
    never imported, no client is ever constructed, and no request is
    ever sent. Zero token usage, guaranteed.

_get_client() also re-checks settings.USE_OPENAI itself as a second,
independent safety net — even if some future code path called this
module directly instead of going through the router, it still refuses
to run unless USE_OPENAI=true.

PHASE 2 UPDATE
--------------
OpenAI is called exactly ONE TIME per event (see the single
`client.chat.completions.create(...)` call in
generate_openai_analysis() below). That one call now returns BOTH:
  - a professional plain-text CCTV summary, and
  - rich structured data shaped to match the SQLite schema
    (database/schema.sql: events / persons / movement / actions /
    objects / vehicles / keywords)
in a single JSON response, so services/db_writer.py can insert it
straight into SQLite without any additional OpenAI calls.
"""

import base64
import io
import json

from PIL import Image

from config.settings import USE_OPENAI, OPENAI_API_KEY, OPENAI_MODEL

# Lazily-created singleton client — mirrors the "load on demand" pattern
# used by models/qwen_vl.py / models/videomae.py. OpenAI is a cloud API
# call (no GPU/RAM footprint), so there's nothing heavy to load/unload;
# this purely exists to avoid constructing the client (and importing
# the SDK) until it's actually needed.
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    if not USE_OPENAI:
        # Should be unreachable in normal operation — summary_router.py
        # never calls this module unless USE_OPENAI=true. This is a
        # deliberate second guard against any accidental OpenAI usage.
        raise RuntimeError(
            "generate_openai_analysis() was called while USE_OPENAI=false. "
            "Refusing to contact OpenAI. Check services/summary_router.py."
        )

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file — "
            "models/openai_vl.py never hardcodes API keys."
        )

    from openai import OpenAI
    _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _image_to_data_url(image_path):
    """Encode a local image file as a base64 data URL for the Chat
    Completions vision API."""
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------
# The structured-output contract. Keys mirror database/schema.sql /
# database/models.py one-to-one so services/db_writer.py can map this
# dict straight onto Event/Person/Movement/Action/ObjectRecord/Vehicle/
# Keyword dataclasses with no guesswork.
# ---------------------------------------------------------------------
_NOT_VISIBLE = "Not Clearly Visible"

_OPENAI_PROMPT = f"""You are a professional CCTV operator writing an incident report.

This single image contains THREE CCTV frames from ONE event, arranged
LEFT to RIGHT in strict chronological order:
- Frame 1 (leftmost)  = the BEGINNING of the event
- Frame 2 (middle)    = the MIDDLE of the event
- Frame 3 (rightmost) = the END of the event

Analyze the event as a WHOLE. Do NOT describe each frame independently
— instead, compare the person's position, pose, and any carried objects
across the three frames to describe how the event progressed over time
(e.g. direction of movement, whether they entered/exited view, any
change in behaviour).

STRICT GROUNDING RULES:
- Be factual and objective. Do not speculate or invent details that are
  not visibly supported by the image.
- If something is not clearly visible, use the exact string
  "{_NOT_VISIBLE}" for that field instead of guessing.
- Never hallucinate a person, object, or vehicle that is not visible.
- Do not mention "Frame 1/2/3" or the image layout in the summary text.

IDENTITY RULE (read carefully):
- This image is ONLY ever sent to you for a person who is NOT a
  registered/recognized face — every person you see here is, by
  definition, unidentified.
- In the "summary" text, always refer to any person you describe as
  "Unknown person" (e.g. "Unknown person entered the house.",
  "Unknown person carrying a parcel."). NEVER use "a person", "the
  person", "someone", "individual", or a bare pronoun as the subject —
  always use the exact phrase "Unknown person".
- In the "persons" array, set "person_name" to the exact literal string
  "Unknown" for every person object (never null, never a guessed name).

Return ONLY a single JSON object (no markdown, no code fences, no
commentary) with EXACTLY this shape:

{{
  "summary": "<natural, flowing CCTV incident report paragraph, referring to any person as 'Unknown person'>",
  "event_type": "<short label, e.g. 'person_detected', 'delivery', 'loitering'>",
  "confidence": <float 0.0-1.0, your overall confidence in this analysis>,
  "persons": [
    {{
      "known_status": "unknown",
      "person_name": "Unknown",
      "gender": "...", "estimated_age": "...", "height": "...",
      "body_build": "...", "top_clothing": "...", "bottom_clothing": "...",
      "footwear": "...", "headwear": "...", "hair": "...", "beard": "...",
      "glasses": "...", "mask": "...", "accessories": "...", "bag": "...",
      "dominant_hand": "...", "confidence": <float 0.0-1.0>
    }}
  ],
  "movement": {{
    "entry_point": "...", "exit_point": "...", "direction": "...",
    "speed": "...", "final_position": "...", "loitering": false
  }},
  "actions": [
    {{"sequence_number": 1, "action": "..."}},
    {{"sequence_number": 2, "action": "..."}}
  ],
  "objects": [
    {{"object_type": "...", "description": "...", "confidence": <float 0.0-1.0>}}
  ],
  "vehicles": [
    {{"vehicle_type": "...", "color": "...", "number_plate": "...", "confidence": <float 0.0-1.0>}}
  ],
  "keywords": ["short", "searchable", "tags"]
}}

"persons", "actions", "objects", "vehicles", and "keywords" may be empty
arrays if nothing of that kind is visible — never invent an entry just
to fill the array. Use "{_NOT_VISIBLE}" (not null, not empty string) for
any individual text field you cannot determine from the image.
"""


def _fallback_result(summary_text=None):
    """Used only if OpenAI's response can't be parsed as JSON — keeps the
    rest of the pipeline (db_writer, Telegram alert) working with a
    degraded-but-safe result instead of crashing the event finalisation."""
    return {
        "summary": summary_text or _NOT_VISIBLE,
        "event_type": _NOT_VISIBLE,
        "confidence": None,
        "persons": [],
        "movement": None,
        "actions": [],
        "objects": [],
        "vehicles": [],
        "keywords": [],
    }


def generate_openai_analysis(merged_image_path):
    """
    Send ONE merged image (3 chronological CCTV frames side by side) to
    OpenAI Vision and return BOTH a plain-text CCTV summary AND rich
    structured data shaped to match the SQLite schema — in a single
    API call (OpenAI is called exactly once per event).

    Args:
        merged_image_path: path to the merged image produced by
                            utils/image_merger.merge_frames_horizontally().

    Returns:
        dict — see _OPENAI_PROMPT / _fallback_result() above for the
        exact shape. Always contains at least "summary".

    Raises:
        RuntimeError — with a clear, tagged message identifying whether
        the failure was an OpenAI API error, a network/connection
        error, or a timeout. Never hides the original exception (it is
        always chained via `from e`), so pipelines/event_manager.py's
        caller sees exactly what went wrong instead of a generic crash.
    """
    from openai import (
        APIConnectionError,
        APITimeoutError,
        APIStatusError,
        OpenAIError,
    )

    client = _get_client()
    data_url = _image_to_data_url(merged_image_path)

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _OPENAI_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
    except APITimeoutError as e:
        raise RuntimeError(f"[OPENAI] Request timed out: {e}") from e
    except APIConnectionError as e:
        raise RuntimeError(f"[OPENAI] Network/connection error: {e}") from e
    except APIStatusError as e:
        raise RuntimeError(f"[OPENAI] API error (status {e.status_code}): {e}") from e
    except OpenAIError as e:
        raise RuntimeError(f"[OPENAI] API error: {e}") from e

    raw = (response.choices[0].message.content or "").strip()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        print("[WARNING] OpenAI Vision did not return valid JSON — "
              "falling back to raw text as the summary only.")
        return _fallback_result(raw)

    if not isinstance(data, dict) or not data.get("summary"):
        return _fallback_result(data.get("summary") if isinstance(data, dict) else raw)

    # Defensive defaults — never let a missing key break db_writer.py.
    data.setdefault("event_type", _NOT_VISIBLE)
    data.setdefault("confidence", None)
    data.setdefault("persons", [])
    data.setdefault("movement", None)
    data.setdefault("actions", [])
    data.setdefault("objects", [])
    data.setdefault("vehicles", [])
    data.setdefault("keywords", [])
    return data


# ---------------------------------------------------------------------
# Backwards-compatible alias. Earlier Phase-2 wiring called this
# function generate_openai_summary() and treated the return value as a
# plain string; services/summary_router.py now calls
# generate_openai_analysis() directly and reads structured fields off
# the dict, but the alias is kept here in case anything else in the
# project still imports the old name.
# ---------------------------------------------------------------------
def generate_openai_summary(merged_image_path):
    """Deprecated alias — returns only the "summary" string from
    generate_openai_analysis(). Prefer generate_openai_analysis()."""
    return generate_openai_analysis(merged_image_path).get("summary", _NOT_VISIBLE)
