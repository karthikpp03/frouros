"""
models/qwen_vl.py
=================
Loads Qwen2.5-VL-7B-Instruct (4-bit quantised) and exposes the
_qwen_infer() helper used by the summary and attribute-extraction steps.

Nothing here changes the model's inference behaviour — the same
quantisation config, processor settings, and generation parameters from
v4 are preserved verbatim.

GPU MEMORY OPTIMISATION (new)
------------------------------
Qwen2.5-VL-7B is the single heaviest model in the whole pipeline
(~5-6 GB in 4-bit). Previously it was loaded once at startup and kept
resident on the GPU for the entire process lifetime, alongside YOLO
(also permanently GPU-resident) — leaving little headroom for the
activation/KV-cache spike that happens during `.generate()` on 10
images, which is what was causing the CUDA OOM.

load_qwen() / unload_qwen() are now a matched pair that the caller
(pipelines/event_manager.py, during event finalisation) uses to load
Qwen right before it's needed and fully release it right after — so
Qwen only ever occupies GPU memory for the few seconds it's actually
generating text, not for the whole session. YOLO keeps running
uninterrupted on the GPU the entire time since it never gets touched
here.
"""

import gc
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from config.settings import QWEN_MODEL_ID, BNB_CONFIG
from utils.device import DEVICE, log_gpu_memory, empty_cache

# Module-level singletons — populated by load_qwen(), cleared by
# unload_qwen(). Deliberately NOT loaded at import time / startup
# anymore — see the module docstring above.
qwen_model = None
processor  = None


def load_qwen():
    """
    Load Qwen2.5-VL-7B and its processor onto the GPU.

    Called on-demand right before an event needs summarising (not once
    at startup) so it only ever shares GPU memory with YOLO, never with
    VideoMAE — which is the "don't keep two heavy models on the GPU at
    once" requirement.
    """
    global qwen_model, processor

    if qwen_model is not None:
        # Already loaded (e.g. summary + attribute-extraction back to
        # back within the same event) — avoid loading it twice.
        return

    print("[INFO] Loading Qwen2.5-VL-3B (4-bit)...")


    load_kwargs = dict(
        device_map=str(DEVICE),
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
    )
    if BNB_CONFIG is not None:
        load_kwargs["quantization_config"] = BNB_CONFIG

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        **load_kwargs,
    )

    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print(f"[INFO] Loading {QWEN_MODEL_ID}...")
    print("===================================")
    log_gpu_memory("After loading Qwen")
    print("===================================")


def unload_qwen():
    """
    Fully release Qwen from GPU memory.

    Called right after event finalisation is done with Qwen (i.e. after
    both generate_summary() and extract_person_attributes() have run
    for that event). Dropping the last Python reference to the model
    lets CUDA's allocator reclaim the memory once we force garbage
    collection + empty the cache — this is what actually returns the
    ~5-6 GB back to the GPU instead of leaving it idle-but-reserved.
    """
    global qwen_model, processor

    if qwen_model is None:
        return  # nothing to unload

    print("[INFO] Unloading Qwen2.5-VL-3B to free GPU memory...")

    del qwen_model
    del processor
    qwen_model = None
    processor  = None

    # gc.collect() first so the model's Python objects (and any tensors
    # only reachable through them) are actually freed before we ask
    # CUDA's caching allocator to release its now-unused memory pool.
    gc.collect()
    empty_cache()

    print("[INFO] Qwen2.5-VL-3B unloaded.")


def _qwen_infer(messages, pil_images=None, max_new_tokens=64):
    """
    Vision inference using Qwen2.5-VL-7B.
    Identical inference logic to the original _qwen_infer() — no
    behavioural changes; the only addition is a clear error if it's
    called before load_qwen() (callers must load it first now that
    loading is on-demand rather than automatic at startup).
    """
    if qwen_model is None or processor is None:
        raise RuntimeError(
            "Qwen model is not loaded. Call load_qwen() before "
            "_qwen_infer() — it is now loaded on-demand during event "
            "finalisation instead of at startup, to avoid holding it "
            "on the GPU permanently."
        )

    empty_cache()
    gc.collect()

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if pil_images:
        resized = [img.resize((320,320)) for img in pil_images]
        inputs  = processor(
            text=[text],
            images=resized,
            padding=True,
            return_tensors="pt"
        )
    else:
        inputs = processor(text=[text], return_tensors="pt")

    inputs = inputs.to(DEVICE)

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

    # Explicitly drop the input/output tensors for this single call
    # before returning. Combined with unload_qwen() (called by the
    # caller once ALL Qwen calls for this event are done), this keeps
    # peak GPU usage as low as possible during generation.
    del inputs, generated_ids
    empty_cache()
    gc.collect()

    if "assistant" in output_text:
        output_text = output_text.split("assistant")[-1].strip()

    return output_text
