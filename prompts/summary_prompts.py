"""
prompts/summary_prompts.py
==========================
Prompt builders for the Qwen2.5-VL summary and attribute-extraction steps.
Functions return fully-formed message dicts ready for _qwen_infer().
"""

from prompts.grounding_rules import _GROUNDING_RULES


def build_summary_messages(pil_images):
    """
    Returns the messages list passed to _qwen_infer() for event summarisation.
    Exactly preserves the original inline prompt text.
    """
    return [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in pil_images],
                {
                    "type": "text",
                    "text": (
                        f"{_GROUNDING_RULES}\n\n"
                        "These images are extracted from ONE CCTV event arranged chronologically.\n\n"

                        "Analyze ONLY what is reasonably visible in the frames.\n"
                        "Do NOT invent conversations, emotions, intentions, or relationships.\n"
                        "If something is unclear, say 'Not clearly visible'.\n"
                        "Focus mainly on people, actions, movement, carried objects, waiting behavior, and interactions.\n"
                        "Maintain temporal consistency across frames.\n\n"

                        "Write a natural CCTV surveillance event summary.\n\n"

                        "For each detected person include naturally:\n"
                        "- clothing/appearance\n"
                        "- observed actions\n"
                        "- movement direction\n"
                        "- carried objects\n"
                        "- waiting/loitering behavior\n"
                        "- interactions if clearly visible\n\n"

                        "The report should feel like a professional CCTV operator summary:\n"
                        "clear, grounded, chronological, and natural.\n\n"

                        "Avoid robotic formatting like 'Person A:' repeatedly.\n"
                        "Avoid repeating the same sentence structure.\n"
                        "Avoid excessive speculation.\n"
                    )
                }
            ]
        }
    ]


def build_attribute_extraction_prompt(summary):
    """
    Returns the messages list passed to _qwen_infer() for structured
    attribute extraction from a free-text summary.
    Exactly preserves the original inline prompt text.
    """
    prompt = (
        f"{_GROUNDING_RULES}\n\n"
        "Extract structured data from the CCTV event description below.\n\n"
        "Return ONLY a valid JSON array. No markdown fences, no explanation.\n"
        "Each element represents one person and must have exactly these keys:\n"
        "  appearance: string or null\n"
        "  actions: array of strings\n"
        "  objects: array of strings\n"
        "  movement: string or null\n"
        "  waiting: boolean\n\n"
        "Critical rules:\n"
        "- Extract ONLY what is explicitly written.\n"
        "- Do NOT infer or add details not present in the text.\n"
        "- If a field is not mentioned, use null / [].\n"
        "- waiting: true only if 'loitering', 'waiting', or 'standing' appears.\n\n"
        f"Event description:\n{summary}\n\n"
        "JSON array (start with [ end with ]):"
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
