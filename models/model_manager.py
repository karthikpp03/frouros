"""
models/model_manager.py
========================
ISSUE 2 — Robust, independent model loading.

Problem this fixes
------------------
Before this module existed, Worker 2 called each model's loader
directly (`load_videomae()`, `load_qwen()`, ...). If ANY of them
raised — model not found on disk, a corrupt checkpoint, an
out-of-memory error, a bad HuggingFace repo id, transformers/
insightface being the wrong version, etc. — the exception propagated
all the way up through `_process_ai()`, which meant a single failed
model (often VideoMAE, which is the least critical model in the whole
pipeline) took down EVERYTHING downstream for that event: Face
Recognition never ran, the AI Summary was never generated, and the
Telegram scene-summary alert was never sent. The event was simply
marked FAILED and silently dropped.

Fix
---
`ModelManager` is a small, dependency-free wrapper used by every model
loader call site. It:

  1. Times and logs every load attempt with the exact block format the
     task calls for ([INFO] / [SUCCESS] / [ERROR] + real exception
     text — never a generic "AI Processing Error").
  2. NEVER lets a loader's exception escape — `safe_load()` /
     `safe_unload()` always return a bool and swallow the underlying
     exception after logging it in full.
  3. Tracks a per-model status (`NOT_LOADED` / `LOADED` / `FAILED`)
     and the last failure reason, queryable at any point during event
     finalisation so the rest of `_process_ai()` can decide what to
     skip vs. what to still attempt.
  4. Is generic — any future model (a new VLM, a new tracker, a new
     embedding model, ...) is onboarded by calling `safe_load("name",
     loader_fn)` / `safe_unload("name", unloader_fn)`; nothing here is
     hardcoded to VideoMAE/Qwen/InsightFace specifically.

This module intentionally contains NO model-specific logic — it only
wraps whatever loader/unloader callables the caller passes it.
"""

import time
import traceback

from utils.event_logger import log_block

# ----------------------------------------------------------------------
# Per-model state — process-wide, in-memory only (mirrors the existing
# in-memory event-status tracking in pipelines/event_manager.py; no
# new persistence layer needed for this).
# ----------------------------------------------------------------------
NOT_LOADED = "NOT_LOADED"
LOADED     = "LOADED"
FAILED     = "FAILED"

_status = {}   # model_name -> NOT_LOADED | LOADED | FAILED
_reason = {}   # model_name -> last failure reason (str), only set on FAILED


def get_status(name: str) -> str:
    """Current status for `name`; NOT_LOADED if it was never attempted."""
    return _status.get(name, NOT_LOADED)


def is_available(name: str) -> bool:
    """True only if `name`'s last load attempt succeeded."""
    return _status.get(name) == LOADED


def failure_reason(name: str):
    """The exception text from `name`'s last failed load, or None."""
    return _reason.get(name)


def safe_load(name: str, loader_fn, *args, **kwargs) -> bool:
    """
    Run `loader_fn(*args, **kwargs)` (a module's no-crash-tolerant
    `load_x()` function) with full logging and exception containment.

    Returns True/False — NEVER raises. Callers use the return value to
    decide whether to proceed with that model or fall back / skip.
    """
    print(f"[INFO]\nLoading {name}...")
    start = time.time()
    try:
        loader_fn(*args, **kwargs)
        elapsed = time.time() - start
        _status[name] = LOADED
        _reason.pop(name, None)
        print(f"[SUCCESS]\n{name} loaded in {elapsed:.1f} sec")
        return True
    except Exception as e:
        elapsed = time.time() - start
        _status[name] = FAILED
        _reason[name] = f"{type(e).__name__}: {e}"
        print(
            f"[ERROR]\n{name} failed to load after {elapsed:.1f} sec.\n"
            f"Reason:\n{_reason[name]}\n"
            f"{traceback.format_exc()}"
        )
        return False


def safe_unload(name: str, unloader_fn, *args, **kwargs) -> bool:
    """
    Mirror of safe_load() for the unload side. Unloading a model that
    failed to load (or was never loaded) is a harmless no-op for every
    existing unloader in this project (`unload_videomae`/`unload_qwen`
    already guard on `model is None`), but this still never lets an
    unexpected exception during teardown crash the caller either.
    """
    try:
        unloader_fn(*args, **kwargs)
        return True
    except Exception as e:
        print(f"[ERROR]\nFailed to cleanly unload {name}.\nReason:\n{type(e).__name__}: {e}")
        return False
    finally:
        # Whatever happened, this model is no longer usable/resident.
        if _status.get(name) != FAILED:
            _status[name] = NOT_LOADED


def run_stage(name: str, fn, *args, fallback=None, **kwargs):
    """
    Run one AI-processing STAGE (not just a model load) — e.g. the
    actual smart-frame extraction call, a face-recognition pass, a
    summary-generation call — with the same "never take the rest of
    the pipeline down" guarantee as safe_load()/safe_unload().

    On success: returns fn(*args, **kwargs).
    On failure: logs the real exception, marks `name` FAILED, and
    returns `fallback` (a plain value, or a zero-arg callable that is
    invoked to produce the fallback value) instead of raising.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        _status[name] = FAILED
        _reason[name] = f"{type(e).__name__}: {e}"
        print(
            f"[ERROR]\n{name} stage failed.\nReason:\n{_reason[name]}\n"
            f"{traceback.format_exc()}"
        )
        return fallback() if callable(fallback) else fallback


def mark_failed(name: str, exc: Exception):
    """
    Record a failure for a stage that doesn't go through safe_load()/
    run_stage() (e.g. a block guarded by its own try/except, like Face
    Recognition in pipelines/event_manager.py) — keeps that failure
    visible in report()/log_report() alongside every other model's
    status, without forcing every call site to reimplement the same
    two dict writes.
    """
    _status[name] = FAILED
    _reason[name] = f"{type(exc).__name__}: {exc}"


def report() -> str:
    """One-line-per-model status summary for end-of-event logging."""
    if not _status:
        return "No models were loaded for this event."
    lines = []
    for name, status in _status.items():
        if status == FAILED:
            lines.append(f"{name:<20} : FAILED ({_reason.get(name, 'unknown reason')})")
        else:
            lines.append(f"{name:<20} : {status}")
    return "\n".join(lines)


def log_report(event_id=None):
    header = f"MODEL STATUS — Event {event_id}" if event_id is not None else "MODEL STATUS"
    log_block(header, report())
