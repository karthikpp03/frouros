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

def tg_send_message(chat_id, text):
    url  = f"{BASE_URL}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print(f"[TELEGRAM] Message error: {e}")


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
