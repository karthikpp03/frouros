"""
models/smolvlm.py
==================
Drop-in alternative to models/qwen_vl.py — loads
HuggingFaceTB/SmolVLM2-2.2B-Instruct (4-bit quantised) and exposes the
exact same public API (load_qwen / unload_qwen / _qwen_infer) so the
rest of the pipeline (pipelines/event_manager.py, pipelines/summary_pipeline.py,
prompts/summary_prompts.py, etc.) can switch between Qwen and SmolVLM2
by changing a single import line, e.g.:

    # from models.qwen_vl import load_qwen, unload_qwen, _qwen_infer
    from models.smolvlm import load_qwen, unload_qwen, _qwen_infer

This file does NOT touch qwen_vl.py, event_manager.py, main.py, or any
other module. It only depends on config.settings.BNB_CONFIG (already
present and used by qwen_vl.py for the identical purpose).

Model: HuggingFaceTB/SmolVLM2-2.2B-Instruct
Official reference implementation (transformers `image-text-to-text`
pipeline_tag):
    https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
Loaded via AutoModelForImageTextToText + AutoProcessor, using
processor.apply_chat_template(..., tokenize=True, return_dict=True)
which builds the full model inputs (text + vision) directly from the
`messages` list in one call — this is the current, non-deprecated
SmolVLM2 API (the old `processor(text=..., images=...)` two-step some
older Qwen-style code uses is not required here since the chat
template already resolves embedded PIL images in the message content).

GPU MEMORY OPTIMISATION
------------------------
Identical strategy to qwen_vl.py, sized for a 4GB GTX 1650:
- Model is NOT loaded at import time / startup.
- load_smolvlm() / unload_smolvlm() are a matched pair — load right
  before inference is needed, unload immediately after.
- 4-bit NF4 quantisation via the same BNB_CONFIG used for Qwen
  (SmolVLM2-2.2B in 4-bit is only ~1.5-2 GB, well under the 4GB budget,
  leaving headroom for YOLO + activation/KV-cache spikes).
- gc.collect() + torch.cuda.empty_cache() on both load and unload.
- Images are downsized before inference to keep the vision-tower
  activation memory low on a 4GB card.
"""

import copy
import gc

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from config.settings import BNB_CONFIG
from utils.device import DEVICE, log_gpu_memory, empty_cache

# Hardcoded per spec — SmolVLM2 is a separate, explicit opt-in model.
# (config.settings.QWEN_MODEL_ID is Qwen's own setting and is
# deliberately left untouched by this file.)
SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# Image side length (px) images are resized to before inference.
# 384 is SmolVLM2's native single-tile encoder resolution — keeping
# input at this size avoids the processor's image-splitting/tiling
# logic from generating extra tiles, which is what would blow past 4GB
# of VRAM on a GTX 1650.
_SMOLVLM_IMAGE_SIZE = (384, 384)

# Module-level singletons — populated by load_qwen(), cleared by
# unload_qwen(). Not loaded at import time, same as qwen_vl.py.
qwen_model = None
processor  = None


def load_qwen():
    """
    Load SmolVLM2-2.2B-Instruct and its processor onto the GPU.

    Named load_qwen() (not load_smolvlm()) so this module is a true
    drop-in replacement for models/qwen_vl.py — callers only change
    the import line, nothing else.
    """
    global qwen_model, processor

    if qwen_model is not None:
        # Already loaded — avoid loading it twice.
        return

    print("[INFO] Loading SmolVLM2-2.2B-Instruct (4-bit)...")
    print("===================================")
    log_gpu_memory("Before loading SmolVLM2")
    print("===================================")

    load_kwargs = dict(
        device_map=str(DEVICE),
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
    )
    if BNB_CONFIG is not None:
        load_kwargs["quantization_config"] = BNB_CONFIG

    qwen_model = AutoModelForImageTextToText.from_pretrained(
        SMOLVLM_MODEL_ID,
        **load_kwargs,
    )

    processor = AutoProcessor.from_pretrained(SMOLVLM_MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"[INFO] Loaded {SMOLVLM_MODEL_ID}...")
    print("===================================")
    log_gpu_memory("After loading SmolVLM2")
    print("===================================")


def unload_qwen():
    """
    Fully release SmolVLM2 from GPU memory.

    Same matched-pair contract as qwen_vl.unload_qwen(): called right
    after event finalisation is done with the vision model, so the
    ~1.5-2 GB it occupies is returned to the GPU allocator instead of
    sitting idle-but-reserved.
    """
    global qwen_model, processor

    if qwen_model is None:
        return  # nothing to unload

    print("[INFO] Unloading SmolVLM2-2.2B-Instruct to free GPU memory...")

    del qwen_model
    del processor
    qwen_model = None
    processor  = None

    # gc.collect() first so the model's Python objects (and any
    # tensors only reachable through them) are actually freed before
    # we ask CUDA's caching allocator to release its unused pool.
    gc.collect()
    empty_cache()

    print("[INFO] SmolVLM2-2.2B-Instruct unloaded.")


def _resize_images_in_messages(messages):
    """
    Return a deep copy of `messages` with every embedded PIL image
    (content items of the form {"type": "image", "image": <PIL.Image>})
    resized to _SMOLVLM_IMAGE_SIZE. Mirrors the `.resize((448, 448))`
    step in qwen_vl._qwen_infer(), adapted to SmolVLM2's message-embedded
    image format (SmolVLM2's chat template reads images straight out of
    the message content instead of a separate `images=` processor arg).
    """
    resized_messages = copy.deepcopy(messages)
    for message in resized_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                img = item.get("image")
                if img is not None and hasattr(img, "resize"):
                    item["image"] = img.resize(_SMOLVLM_IMAGE_SIZE)
    return resized_messages


def _qwen_infer(messages, pil_images=None, max_new_tokens=250):
    """
    Vision / text inference using SmolVLM2-2.2B-Instruct.

    Same signature, same call contract, and same return type (a plain
    decoded string with the prompt/role prefix stripped) as
    qwen_vl._qwen_infer(), so callers (summary_pipeline.py etc.) work
    unmodified regardless of which module is imported.

    `pil_images` is accepted for API parity with qwen_vl._qwen_infer()
    but is not required here — SmolVLM2's messages already carry the
    PIL images inline (see prompts/summary_prompts.py), and
    processor.apply_chat_template(..., tokenize=True, return_dict=True)
    resolves them directly. It's kept as a parameter purely so
    call-sites that pass it positionally/by-keyword don't need edits.
    """
    if qwen_model is None or processor is None:
        raise RuntimeError(
            "SmolVLM2 model is not loaded. Call load_qwen() before "
            "_qwen_infer() — it is loaded on-demand during event "
            "finalisation instead of at startup, to avoid holding it "
            "on the GPU permanently."
        )

    empty_cache()
    gc.collect()

    # Resize any embedded images down to the model's native single-tile
    # resolution before building inputs — keeps vision-tower activation
    # memory low on a 4GB card. Text-only messages pass through as-is.
    resized_messages = _resize_images_in_messages(messages)

    inputs = processor.apply_chat_template(
        resized_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(qwen_model.device)

    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    # Explicitly drop the input/output tensors for this single call
    # before returning. Combined with unload_qwen() (called by the
    # caller once ALL calls for this event are done), this keeps peak
    # GPU usage as low as possible during generation.
    del inputs, generated_ids
    empty_cache()
    gc.collect()

    # SmolVLM2's chat template renders the turn marker as "Assistant:"
    # (vs. Qwen's lowercase "assistant"). Strip whichever is present so
    # only the model's reply text is returned, matching qwen_vl's
    # output format exactly.
    if "Assistant:" in output_text:
        output_text = output_text.split("Assistant:")[-1].strip()
    elif "assistant" in output_text:
        output_text = output_text.split("assistant")[-1].strip()

    return output_text
