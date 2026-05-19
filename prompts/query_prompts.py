"""
prompts/query_prompts.py
========================
System prompt for the Llama 3.1 query engine and the intent-pattern
dictionary used by classify_intent().
Both are kept here as module-level constants.
"""

# ==================================================
# LLAMA SYSTEM PROMPT  (unchanged from v4)
# ==================================================

LLAMA_SYSTEM_PROMPT = """You are an expert AI surveillance operator assistant.

You have access to structured CCTV surveillance memory containing timestamped events, person descriptions, observed actions, carried objects, and movement data.

Your job:
- Answer the operator's question accurately using ONLY the provided surveillance memory.
- Reference specific persons by their appearance and event timestamps.
- Be concise, factual, and professional — like a real security operator would speak.
- If the information is not in the memory, say so clearly. Do NOT guess or invent details.
- When relevant, mention the event ID and timestamp for traceability.
- Support follow-up questions by referencing prior context naturally.

You must NOT:
- Invent events, persons, or details not in the memory.
- Speculate about intent, emotion, or future behaviour.
- Provide opinions or recommendations beyond factual reporting.
"""

# ==================================================
# INTENT PATTERNS  (unchanged from v4)
# ==================================================

INTENT_PATTERNS = {
    "daily_update": [
        r"today.{0,20}update", r"what happened today",
        r"give me.{0,10}summary", r"today.{0,10}report",
        r"recent event", r"latest event",
    ],
    "person_appearance": [
        r"(black|white|red|blue|green|yellow|grey|gray|brown|pink|orange|purple)"
        r".{0,20}(shirt|jacket|coat|hoodie|dress|jeans|cap|hat)",
        r"who (was|is) wearing", r"person in", r"man in", r"woman in",
    ],
    "person_action": [
        r"who (used|carried|brought|held|had)",
        r"who (entered|exited|left|came|arrived)",
        r"who (waited|stood|lingered|sat)",
        r"who (talked|spoke|interacted)",
        r"who (ran|walked|moved)",
        r"what did.{0,30}do",
    ],
    "time_query": [
        r"after \d{1,2}(:\d{2})?\s*(am|pm|AM|PM)",
        r"before \d{1,2}(:\d{2})?\s*(am|pm|AM|PM)",
        r"between \d", r"at night", r"in the morning", r"in the evening",
    ],
    "object_query": [
        r"(bag|backpack|phone|umbrella|bottle|helmet|laptop|box)",
        r"carrying", r"holding", r"with a",
    ],
    "zone_query": [
        r"zone [a-zA-Z]",
        r"near (entrance|exit|door|gate|counter|desk)",
        r"at the (entrance|exit|door|gate|counter|desk)",
    ],
    "suspicious": [
        r"suspicious", r"unusual", r"loitering",
        r"longest", r"stayed.{0,10}(longest|long time)",
    ],
    "image_request": [
        r"send (pic|photo|image|picture)",
        r"show (me )?(the )?(person|image|pic|photo)",
        r"who is this", r"picture of", r"image of",
        r"photo of", r"show image",
    ],
}
