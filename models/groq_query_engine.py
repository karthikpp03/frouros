"""
models/groq_query_engine.py
============================
Connects to the Groq API and exposes _llama_infer() for all text-only
query operations.  Uses llama-3.1-8b-instant via the Groq SDK.

Exactly mirrors the original _llama_infer() block — no logic changes.
"""

from groq import Groq
from config.settings import GROQ_API_KEY, GROQ_MODEL

# Module-level singleton — initialised once at import time.
groq_client = None


def load_groq():
    """Instantiate Groq client.  Called once at startup."""
    global groq_client
    print("[INFO] Connecting to Groq API...")
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("[INFO] Groq connected!")


def _llama_infer(system_prompt, user_prompt, max_new_tokens=300):
    """
    Text-only inference via Groq / Llama-3.1-8B-Instruct.
    Identical to the original _llama_infer() — no logic changes.
    """
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
