"""
services/summary_router.py
===========================
The ONE place in the whole project that decides which AI provider
generates an event's summary. pipelines/event_manager.py just calls
generate_event_summary() and gets a summary back — it never contains
provider-selection logic itself, and neither does anything else.

Routing table (controlled entirely by .env / config/settings.py)
------------------------------------------------------------------
USE_OPENAI=false
    -> Always Qwen (local, free, unlimited). OpenAI is never imported,
       never initialized, never called. This is the default.

USE_OPENAI=true, ENABLE_FACE_RECOGNITION=false
    -> Always OpenAI. Testing mode: lets you evaluate OpenAI Vision
       output before the face-recognition model exists.

USE_OPENAI=true, ENABLE_FACE_RECOGNITION=true
    -> face/recognizer.py decides per-event:
         known face   -> Qwen (free — known people never cost tokens)
         unknown face -> OpenAI (paid — only unknowns are ever sent
                          to OpenAI)

FUTURE PROVIDERS
-----------------
Adding another provider (Claude, Llama, a different Gemini, etc.) only
means adding one more `_run_<provider>()` function here and wiring it
into the branches above — no other file in the project needs to change.

LOGGING
-------
Every routing decision prints a [ROUTER] block (config values + which
pipeline was selected and why) before the chosen provider is ever
called, so the terminal always shows *why* a path was taken, not just
which one.
"""

from config.settings import USE_OPENAI, ENABLE_FACE_RECOGNITION
from utils.event_logger import log_block


def generate_event_summary(smart_frame_paths, event_id):
    """
    Args:
        smart_frame_paths: the 3 VideoMAE smart-frame paths for this
                            event (unchanged — VideoMAE itself is never
                            modified).
        event_id:           the event id, used only for OpenAI's merged
                             image filename.

    Returns:
        (summary, provider, face_status, structured) tuple:
          summary:     str  — the plain-text CCTV summary.
          provider:    str  — "qwen" | "openai" | "none".
          face_status: str | None — "known" / "unknown" if face
                       recognition ran for this event, else None.
          structured:  dict | None — rich structured data shaped to
                       match the SQLite schema (see
                       models/openai_vl.py._OPENAI_PROMPT), only
                       populated on the OpenAI branch. None on the
                       Qwen branch — services/db_writer.py falls back
                       to building a minimal structured record from
                       pipelines/summary_pipeline.extract_person_attributes()
                       in that case, so SQLite still gets an event row
                       either way (see pipelines/event_manager.py).
    """
    if not smart_frame_paths:
        log_block("ROUTER", "No smart frames available.", "Selected Pipeline", "NONE")
        return "No frames available.", "none", None, None

    if not USE_OPENAI:
        log_block(
            "ROUTER",
            f"USE_OPENAI = {USE_OPENAI}",
            "Selected Pipeline",
            "Qwen2.5-VL",
        )
        return _run_qwen(smart_frame_paths), "qwen", None, None

    if not ENABLE_FACE_RECOGNITION:
        # Testing mode — face model isn't wired up yet, so every event
        # goes to OpenAI so you can evaluate it ahead of time.
        log_block(
            "ROUTER",
            f"USE_OPENAI = {USE_OPENAI}",
            f"ENABLE_FACE_RECOGNITION = {ENABLE_FACE_RECOGNITION}",
            "Known Person = False",
            "Selected Pipeline",
            "OpenAI Vision",
        )
        summary, structured = _run_openai(smart_frame_paths, event_id)
        return summary, "openai", None, structured

    # Full routing: known faces stay on free local Qwen, only unknown
    # faces ever consume OpenAI tokens.
    from face.recognizer import recognize_face
    face_status = recognize_face(smart_frame_paths)
    known = face_status == "known"

    log_block(
        "ROUTER",
        f"USE_OPENAI = {USE_OPENAI}",
        f"ENABLE_FACE_RECOGNITION = {ENABLE_FACE_RECOGNITION}",
        f"Known Person = {known}",
        "Selected Pipeline",
        "Qwen2.5-VL" if known else "OpenAI Vision",
    )

    if known:
        return _run_qwen(smart_frame_paths), "qwen", face_status, None

    summary, structured = _run_openai(smart_frame_paths, event_id)
    return summary, "openai", face_status, structured


def _run_qwen(smart_frame_paths):
    """Local, free, unlimited. Reuses the existing Qwen summary pipeline
    verbatim — nothing about Qwen's behaviour changes."""
    from pipelines.summary_pipeline import generate_summary
    from config.settings import QWEN_MODEL_ID

    log_block("QWEN", "Generating summary...", f"Model : {QWEN_MODEL_ID}", "Waiting for response...")
    summary = generate_summary(smart_frame_paths)
    log_block("QWEN", "Summary generated.")
    return summary


def _run_openai(smart_frame_paths, event_id):
    """Cloud, paid. Merges the 3 smart frames into one labelled image
    (utils/image_merger.py) and sends that single image to OpenAI
    Vision (models/openai_vl.py) — ONE API call per event returns both
    the plain-text summary AND the rich structured data in one shot,
    minimizing token usage and guaranteeing OpenAI is never called a
    second time for this event.

    Returns:
        (summary_text, structured_dict) tuple.
    """
    from utils.image_merger import merge_frames_horizontally
    from models.openai_vl import generate_openai_analysis
    from config.settings import OPENAI_MODEL

    merged_path = merge_frames_horizontally(smart_frame_paths, event_id)

    log_block("OPENAI", "Sending merged image...", f"Model : {OPENAI_MODEL}", "Waiting for response...")
    analysis = generate_openai_analysis(merged_path)
    log_block("OPENAI", "Summary generated.", "Structured data generated.")

    return analysis.get("summary", "Not Clearly Visible"), analysis
