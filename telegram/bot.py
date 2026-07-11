"""
telegram/bot.py
===============
Telegram bot polling loop — runs in a background daemon thread.
Dispatches operator queries to query_pipeline.query_memory() and
sends replies (text + optional images) back via Telegram.

All logic is preserved verbatim from the original monolith.
The only structural change: query_memory is imported from
pipelines/query_pipeline.py instead of being defined inline.
"""

import time
import threading

from config.settings import CHAT_ID, BASE_URL
from telegram.alerts  import tg_send_message, tg_send_photo, tg_get_updates
from utils.event_logger import log_bar, log_block

# Imported lazily inside bot_poll_loop to avoid circular imports at module load.
# pipelines.query_pipeline → memory.event_memory → (no models needed at import)

# Module-level state
bot_offset         = 0
bot_chat_history   = []
bot_running        = True
bot_lock           = threading.Lock()
_processed_updates = set()


def _get_telegram_offset():
    updates = tg_get_updates(-1)
    if updates:
        return updates[-1]["update_id"] + 1
    return 0


def bot_poll_loop():
    """Main polling loop — identical logic to the original."""
    global bot_offset

    # Import here to guarantee models are loaded before first query arrives
    from pipelines.query_pipeline import query_memory

    bot_offset = _get_telegram_offset()
    print(f"[TELEGRAM] Bot ready (chat_id: {CHAT_ID}), offset={bot_offset}")

    while bot_running:
        try:
            updates = tg_get_updates(bot_offset)
        except Exception as e:
            print(f"[TELEGRAM] Poll error: {e}")
            time.sleep(2)
            continue

        for update in updates:
            uid        = update["update_id"]
            bot_offset = uid + 1

            if uid in _processed_updates:
                continue
            _processed_updates.add(uid)

            message = update.get("message", {})
            chat_id = str(message.get("chat", {}).get("id", ""))
            text    = message.get("text", "").strip()

            if not text or chat_id != CHAT_ID:
                continue
            if text.startswith("/") and text not in ("/status", "/help"):
                continue

            log_bar()
            log_block("TELEGRAM", "Incoming Question", text)
            tg_send_message(chat_id, "⏳ Searching memory...")

            try:
                with bot_lock:
                    bot_chat_history.append(("user", text))
                    if len(bot_chat_history) > 12:
                        bot_chat_history[:] = bot_chat_history[-12:]
                    history_ctx = ""
                    if len(bot_chat_history) > 2:
                        history_ctx = "Recent conversation:\n"
                        for role, msg in bot_chat_history[-6:]:
                            history_ctx += f"  {role.upper()}: {msg}\n"
                        history_ctx += "\n"

                enriched_query    = history_ctx + f"Current question: {text}"
                answer, img_paths = query_memory(enriched_query)

                with bot_lock:
                    bot_chat_history.append(("assistant", answer[:200]))

                tg_send_message(chat_id, answer)

                for i, img_path in enumerate(img_paths):
                    if os.path.exists(img_path):
                        tg_send_photo(chat_id, img_path,
                                      caption=f"Matching person {i+1}")

                log_bar()

            except Exception as e:
                tg_send_message(chat_id, f"Sorry, an error occurred: {e}")
                print(f"[TELEGRAM] Query error: {e}")

        if len(_processed_updates) > 5000:
            _processed_updates.clear()

        time.sleep(0.5)

    print("[TELEGRAM] Bot loop exited cleanly.")


def start_bot_thread():
    """Spawn the polling loop in a background daemon thread and return it."""
    import os  # needed for os.path.exists inside the loop
    # Patch os into the module's namespace so the closure above can use it
    import telegram.bot as _self
    import os as _os
    _self.os = _os

    thread = threading.Thread(
        target=bot_poll_loop, daemon=True, name="TelegramBot"
    )
    thread.start()
    return thread


def stop_bot():
    """Signal the poll loop to exit."""
    global bot_running
    bot_running = False
