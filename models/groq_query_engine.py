"""
models/groq_query_engine.py
============================
Connects to the Groq API and exposes _llama_infer() for all text-only
query operations.  Uses llama-3.1-8b-instant via the Groq SDK.

Inference logic is unchanged from the original _llama_infer() block.
The only addition is clearer, typed error handling: Groq API errors,
network/connection errors, and timeouts are caught separately and
re-raised with an explicit, identifiable message instead of an opaque
SDK traceback — callers (pipelines/query_pipeline.py, telegram/bot.py)
never need their own try/except around this call. A small number of
retries are attempted for transient errors (timeouts, rate limits,
connection issues) before giving up.
"""

import time

import groq
from groq import Groq
from config.settings import GROQ_API_KEY, GROQ_MODEL

# Module-level singleton — initialised once at import time.
groq_client = None

# Transient-error retry policy — kept small so a stuck Telegram query
# never hangs the bot loop for long.
_MAX_RETRIES = 2
_RETRY_DELAY_SECONDS = 1.5


def load_groq():
    """Instantiate Groq client. Called once at startup."""
    global groq_client
    print("[INFO] Connecting to Groq API...")
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        raise RuntimeError(f"[GROQ] ERROR — failed to initialize Groq client: {e}") from e
    print("[INFO] Groq connected!")


def _llama_infer(system_prompt, user_prompt, max_new_tokens=300):
    """
    Text-only inference via Groq / Llama-3.1-8B-Instruct.
    Identical inference logic to the original _llama_infer() — the
    only change is explicit, retried, typed error handling below.

    Raises:
        RuntimeError — with a clear, tagged message identifying whether
        the failure was a Groq API error, a network/connection error,
        or a timeout, after retries are exhausted. Never hides the
        original exception (chained via `from e`).
    """
    if groq_client is None:
        raise RuntimeError(
            "[GROQ] ERROR — Groq client is not initialized. Call load_groq() "
            "at startup before _llama_infer()."
        )

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 2):  # e.g. 1 initial try + 2 retries
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=max_new_tokens,
            )
            return completion.choices[0].message.content

        except groq.APITimeoutError as e:
            last_error = e
            print(f"[GROQ] Timeout on attempt {attempt}/{_MAX_RETRIES + 1}: {e}")
        except groq.RateLimitError as e:
            last_error = e
            print(f"[GROQ] Rate limited on attempt {attempt}/{_MAX_RETRIES + 1}: {e}")
        except groq.APIConnectionError as e:
            last_error = e
            print(f"[GROQ] Network/connection error on attempt {attempt}/{_MAX_RETRIES + 1}: {e}")
        except groq.APIStatusError as e:
            # Non-retryable Groq API error (bad request, auth, etc.) —
            # fail immediately with a clear, identified message.
            raise RuntimeError(f"[GROQ] API error (status {e.status_code}): {e}") from e
        except groq.APIError as e:
            # Any other Groq SDK error not covered above — still
            # surfaced explicitly rather than silently swallowed.
            raise RuntimeError(f"[GROQ] API error: {e}") from e

        if attempt <= _MAX_RETRIES:
            time.sleep(_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"[GROQ] ERROR — request failed after {_MAX_RETRIES + 1} attempts: {last_error}"
    ) from last_error
