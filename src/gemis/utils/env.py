"""Environment capture, so every result file records how it was produced."""

from __future__ import annotations

import datetime as dt
import os
import platform
import subprocess
from importlib import metadata
from typing import Any

import torch

_PACKAGES = ("torch", "transformers", "tokenizers", "numpy", "scipy", "scikit-learn")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _memory_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def collect_env(device: torch.device | None = None) -> dict[str, Any]:
    """Library versions, hardware, git revision and timestamp."""
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None

    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": props.name, "memory_gb": round(props.total_memory / 2**30, 1)}

    # "dirty" = code or configs differ from the recorded commit
    dirty = _git("status", "--porcelain", "--", "src", "scripts", "configs")
    return {
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "gpu": gpu,
        "device": str(device) if device is not None else None,
        "versions": versions,
    }
