"""
models/qwen_vl.py
=================
Loads Qwen2.5-VL-7B-Instruct (4-bit quantised) and exposes the
_qwen_infer() helper used by the summary and attribute-extraction steps.

Nothing here changes the model behaviour — the same quantisation config,
processor settings, and generation parameters from v4 are preserved verbatim.
"""

import gc
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from config.settings import QWEN_MODEL_ID, BNB_CONFIG

# Module-level singletons — loaded once at import time.
qwen_model = None
processor  = None


def load_qwen():
    """Load Qwen2.5-VL-7B and its processor.  Called once at startup."""
    global qwen_model, processor

    print("[INFO] Loading Qwen2.5-VL-7B (4-bit)...")

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="cuda",
        torch_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("[INFO] Qwen2.5-VL-7B loaded!")


def _qwen_infer(messages, pil_images=None, max_new_tokens=250):
    """
    Vision inference using Qwen2.5-VL-7B.
    Identical to the original _qwen_infer() — no logic changes.
    """
    torch.cuda.empty_cache()
    gc.collect()

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if pil_images:
        resized = [img.resize((448, 448)) for img in pil_images]
        inputs  = processor(
            text=[text],
            images=resized,
            padding=True,
            return_tensors="pt"
        )
    else:
        inputs = processor(text=[text], return_tensors="pt")

    inputs = inputs.to("cuda")

    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1
        )

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    del inputs, generated_ids
    torch.cuda.empty_cache()
    gc.collect()

    if "assistant" in output_text:
        output_text = output_text.split("assistant")[-1].strip()

    return output_text
