"""Shared utilities: reproducibility, device handling, ROCm-on-Windows guards, provenance dumps."""
import json
import os
import platform
import random
import ssl
from pathlib import Path

# Windows: avoid OpenMP duplicate-runtime crash (triggered by the clip package).
# Must be set before torch/numpy import in entry scripts; harmless elsewhere.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(REPO_ROOT / "config" / "config.json") as f:
        return json.load(f)


def repo_path(*parts):
    """Absolute path from repo root; keeps all code location-independent."""
    return REPO_ROOT.joinpath(*parts)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def allow_insecure_downloads():
    """SSL cert workaround for torchvision/HF dataset downloads on this machine."""
    ssl._create_default_https_context = ssl._create_unverified_context


def gpu_smoke_test():
    """Small on-device matmul; returns a provenance dict. Raises if CUDA compute fails."""
    device = get_device()
    info = {
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
    }
    x = torch.randn(256, 256, device=device)
    y = (x @ x).sum().item()
    assert np.isfinite(y), "GPU matmul produced non-finite output"
    info["matmul_ok"] = True
    return info


def dump_runtime_summary(extra=None):
    """Write versions/GPU manifest to results/runtime_summary.json for provenance."""
    summary = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    if extra:
        summary.update(extra)
    out = repo_path("results", "runtime_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    return summary
