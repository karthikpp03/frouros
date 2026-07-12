"""
telegram/alerts.py
==================
Telegram alert delivery: fire-and-forget photo alerts with retry logic,
plus the low-level tg_send_message / tg_send_photo / tg_get_updates helpers.

All logic is preserved verbatim from the original monolith.
"""

import os
import re
import time
import threading
import requests
from datetime import datetime

from config.settings import BOT_TOKEN, CHAT_ID, BASE_URL


# --------------------------------------------------
# SHARED FORMATTING HELPERS
# --------------------------------------------------

def _format_telegram_datetime(timestamp):
    """
    Render a 'YYYY-MM-DD HH:MM:SS' timestamp string as the
    human-friendly (date_str, time_str) pair used across every
    Telegram message for an event — e.g. ('11 Jul 2026', '04:02:35 PM').
    Falls back to a best-effort split of the raw string if parsing
    ever fails, so a malformed timestamp can never crash an alert.
    """
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%d %b %Y"), dt.strftime("%I:%M:%S %p")
    except (ValueError, TypeError):
        date_str, _, time_str = (timestamp or "").partition(" ")
        return date_str, time_str


# ISSUE 4 — Telegram Markdown reliability.
#
# Telegram's legacy `parse_mode: "Markdown"` only treats four
# characters as special: '_', '*', '`', '['. Groq's answers are plain
# conversational text that was never meant to carry Markdown
# formatting, so any of these characters appearing "naturally" (e.g.
# "test_1", "cost was $5*2", an em dash mistyped as '_') is enough to
# break parsing and silently trigger the plain-text fallback below.
#
# Escaping them before the first send neutralises that risk up front —
# normal replies succeed on the first attempt instead of relying on
# the fallback — while the fallback itself (tg_send_message's
# `_retry_plain` path) stays in place as a safety net for anything
# this doesn't catch (e.g. Telegram-side edge cases unrelated to these
# four characters).
_MARKDOWN_SPECIAL_CHARS = re.compile(r"([_*`\[])")


def escape_telegram_markdown(text):
    """Escape the four characters Telegram's legacy Markdown mode
    treats as special, so arbitrary LLM-generated text can be sent
    with parse_mode="Markdown" without accidentally forming (or
    breaking) an entity."""
    if not text:
        return text
    return _MARKDOWN_SPECIAL_CHARS.sub(r"\\\1", text)


# --------------------------------------------------
# LOW-LEVEL TELEGRAM HELPERS  (unchanged)
# --------------------------------------------------

def tg_send_message(chat_id, text, _retry_plain=True):
    """
    Send a Telegram text message and verify it actually went through.

    Previously this only caught request-level exceptions (network
    errors) — it never inspected Telegram's own JSON response. Telegram
    returns HTTP 200 with `{"ok": false, ...}` for perfectly ordinary
    failures (e.g. a Groq-generated answer containing unmatched
    '*'/'_' characters breaks `parse_mode: "Markdown"` parsing), so a
    reply could silently fail to reach the user with no error ever
    printed — this is the bug behind "Groq answered but Telegram never
    received it".

    Now:
      1. The response is always parsed and checked (`ok` field).
      2. If Markdown parsing is what failed, the same text is retried
         once as plain text (no parse_mode) rather than being dropped.
      3. Success or failure is always logged.

    Returns:
        bool — True if the message was confirmed delivered, False
        otherwise (already logged).
    """
    url  = f"{BASE_URL}/sendMessage"
    # ISSUE 4: sanitize first, so the FIRST attempt (still parse_mode
    # "Markdown") succeeds instead of relying on the plain-text
    # fallback below. `text` itself is left untouched — only the copy
    # actually sent to Telegram is escaped, so callers/logs still see
    # the original string.
    safe_text = escape_telegram_markdown(text)
    data = {"chat_id": chat_id, "text": safe_text, "parse_mode": "Markdown"}

    print("[TELEGRAM] Sending reply...")
    try:
        resp = requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"[TELEGRAM] Failed to send reply: {e}")
        return False

    try:
        result = resp.json()
    except ValueError as e:
        print(f"[TELEGRAM] Failed to send reply: could not parse Telegram "
              f"response ({e}); HTTP {resp.status_code}: {resp.text[:200]}")
        return False

    if result.get("ok"):
        print("[TELEGRAM] Reply sent successfully.")
        return True

    description = result.get("description", "unknown error")

    # Markdown parse failures are the single most common cause of a
    # silently-dropped reply, since the message text usually comes
    # straight from an LLM (Groq) and can easily contain unmatched
    # '*'/'_'/'`' characters. Retry once as plain text rather than
    # losing the answer entirely.
    if _retry_plain and "can't parse entities" in description.lower():
        print(f"[TELEGRAM] Markdown parse error ({description}) — "
              f"retrying as plain text...")
        plain_data = {"chat_id": chat_id, "text": text}
        try:
            resp2 = requests.post(url, data=plain_data, timeout=10)
            result2 = resp2.json()
        except Exception as e:
            print(f"[TELEGRAM] Failed to send reply: {e}")
            return False
        if result2.get("ok"):
            print("[TELEGRAM] Reply sent successfully (plain text fallback).")
            return True
        print(f"[TELEGRAM] Failed to send reply: {result2.get('description', 'unknown error')}")
        return False

    print(f"[TELEGRAM] Failed to send reply: {description}")
    return False


def tg_send_photo(chat_id, image_path, caption=""):
    url = f"{BASE_URL}/sendPhoto"
    try:
        with open(image_path, "rb") as photo:
            requests.post(
                url,
                files={"photo": photo},
                data={"chat_id": chat_id, "caption": caption},
                timeout=15,
            )
    except Exception as e:
        print(f"[TELEGRAM] Photo error: {e}")


def tg_get_updates(offset):
    url    = f"{BASE_URL}/getUpdates"
    params = {"offset": offset, "timeout": 5, "allowed_updates": ["message"]}
    try:
        resp = requests.get(url, params=params, timeout=10)
        return resp.json().get("result", [])
    except Exception:
        return []


# --------------------------------------------------
# ALERT WORKER  (unchanged)
# --------------------------------------------------

def _alert_worker(image_path, caption, ev_id):
    url   = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    delay = 2
    for attempt in range(1, 4):
        try:
            with open(image_path, "rb") as photo:
                resp   = requests.post(
                    url,
                    files={"photo": photo},
                    data={"chat_id": CHAT_ID, "caption": caption},
                    timeout=20,
                )
            result = resp.json()
            if result.get("ok"):
                print(f"[TELEGRAM] Alert sent for Event #{ev_id} (attempt {attempt})")
                return
            else:
                print(f"[TELEGRAM] Alert attempt {attempt} failed: {result.get('description')}")
        except Exception as e:
            print(f"[TELEGRAM] Alert attempt {attempt} error: {e}")
        time.sleep(delay)
        delay *= 2
    print(f"[TELEGRAM] WARNING — Alert for Event #{ev_id} not delivered after 3 attempts.")


def send_telegram_alert(image_path, summary, duration, timestamp, ev_id):
    """
    Fire-and-forget Telegram photo alert.
    Spawns _alert_worker in a daemon thread — identical to the original.
    Kept for any other caller that still wants the old free-text-caption
    style alert; pipelines/event_manager.py's first notification now
    uses send_person_detected_alert() below instead (ISSUE 3 / ISSUE 5).
    """
    if not os.path.exists(image_path):
        print(f"[TELEGRAM] Snapshot missing for Event #{ev_id}: {image_path}")
        return
    caption = (
        f"🚨 EVENT #{ev_id}\n\n"
        f"🕒 {timestamp}\n"
        f"⏱ Duration: {duration} sec\n\n"
        f"{summary}"
    )
    if len(caption) > 1024:
        caption = caption[:1021] + "..."
    t = threading.Thread(
        target=_alert_worker, args=(image_path, caption, ev_id), daemon=True
    )
    t.start()


def send_scene_started_alert(image_path, ev_id, timestamp, persons_detected=1):
    """
    🟢 EVENT #<id> STARTED — the very FIRST Telegram notification for a
    new Scene Event, fired the instant the first tracked person's crop
    is saved. This is Stage 1 of the two-stage alert system: it NEVER
    performs or waits on Face Recognition and NEVER reports an
    identity (no "Unknown", no "Known Person", no person name) — at
    the very start of an event the only frame available is often a
    back view, a side view, a partial body, or motion blur, so any
    identity claimed this early is frequently wrong (this is the
    "UNKNOWN PERSON DETECTED" false alert the two-stage system exists
    to remove). Identity is resolved later, once the whole scene has
    been recorded — see the Stage 2 "scene summary" message sent from
    pipelines/event_manager.py._process_ai() after VideoMAE + Face
    Recognition have run on the complete recording.

    Renders:
        🟢 EVENT #12 STARTED
        Activity detected in the monitored area.
        Persons Detected : 1
        Time : 10:42:18 AM
        Processing AI...
    """
    if not os.path.exists(image_path):
        print(f"[TELEGRAM] Snapshot missing for Event #{ev_id}: {image_path}")
        return

    _, time_str = _format_telegram_datetime(timestamp)

    lines = [
        f"🟢 EVENT #{ev_id} STARTED",
        "Activity detected in the monitored area.",
        f"Persons Detected : {persons_detected}",
        f"Time : {time_str}",
        "Processing AI...",
    ]
    caption = "\n".join(lines)
    if len(caption) > 1024:
        caption = caption[:1021] + "..."

    t = threading.Thread(
        target=_alert_worker, args=(image_path, caption, ev_id), daemon=True
    )
    t.start()


def send_person_joined_alert(image_path, ev_id, timestamp, persons_detected):
    """
    ➕ EVENT #<id> UPDATE — fired every time an ADDITIONAL tracked
    person enters an already-ACTIVE Scene Event. Never creates a new
    event — always refers back to the same ev_id the scene started
    with (see send_scene_started_alert() above for the very first
    person). Same Stage 1 rule applies: no identity recognition, no
    "Unknown"/"Known Person"/name — only that activity/another person
    was detected and the running count for this event. Identity for
    every participant is resolved once in the single Stage 2 scene
    summary, after the whole recording is available.

    Renders:
        ➕ EVENT #12 UPDATE
        Persons Detected : 2
        Time : 10:42:26 AM
    """
    if not os.path.exists(image_path):
        print(f"[TELEGRAM] Snapshot missing for Event #{ev_id} (join): {image_path}")
        return

    _, time_str = _format_telegram_datetime(timestamp)

    lines = [
        f"➕ EVENT #{ev_id} UPDATE",
        f"Persons Detected : {persons_detected}",
        f"Time : {time_str}",
    ]
    caption = "\n".join(lines)
    if len(caption) > 1024:
        caption = caption[:1021] + "..."

    t = threading.Thread(
        target=_alert_worker, args=(image_path, caption, ev_id), daemon=True
    )
    t.start()