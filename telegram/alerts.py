"""
telegram/alerts.py
==================
Telegram alert delivery: fire-and-forget photo alerts with retry logic,
plus the low-level tg_send_message / tg_send_photo / tg_get_updates helpers.

All logic is preserved verbatim from the original monolith.
"""

import os
import time
import threading
import requests

from config.settings import BOT_TOKEN, CHAT_ID, BASE_URL


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
    data = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}

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
