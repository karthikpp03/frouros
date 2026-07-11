"""
utils/device.py
================
Single source of truth for which compute device (CUDA GPU or CPU) every
PyTorch model in this project loads onto.

Nothing else in the project should hardcode "cuda" or call a CUDA-only
API directly — import DEVICE (or the helpers below) from here instead.

  DEVICE            — torch.device, resolved once at import time.
  get_device()       — returns DEVICE (kept as a function for callers
                        that prefer not to import a module-level value).
  log_device_info()  — prints CUDA availability / selected device / GPU
                        name / torch version once at startup.
  log_gpu_memory(tag) — prints CUDA memory stats ONLY when CUDA is
                        available; prints "Device : CPU" otherwise.
                        Never calls a CUDA-only API on a CPU-only system.
  empty_cache()       — safe wrapper around torch.cuda.empty_cache();
                        a no-op on CPU-only systems.
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    """Return the process-wide selected device (cuda if available, else cpu)."""
    return DEVICE


def log_device_info() -> None:
    """Print device diagnostics once at startup. Safe on both GPU and
    CPU-only systems — GPU name is only queried when CUDA is available."""
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available : {cuda_available}")
    print(f"Device         : {DEVICE}")
    if cuda_available:
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Torch Version  : {torch.__version__}")


def log_gpu_memory(tag: str = "") -> None:
    """
    Print GPU memory statistics — but ONLY when CUDA is available.
    Never calls torch.cuda.memory_summary()/memory_allocated()/etc. on
    a CPU-only system; prints "Device : CPU" instead.
    """
    label = f" ({tag})" if tag else ""
    if torch.cuda.is_available():
        print(f"[GPU MEMORY{label}]")
        print(torch.cuda.memory_summary())
    else:
        print(f"[GPU MEMORY{label}]")
        print("Device : CPU")


def empty_cache() -> None:
    """Release CUDA's cached allocator memory — only when CUDA is
    available. A harmless no-op on CPU-only systems."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
