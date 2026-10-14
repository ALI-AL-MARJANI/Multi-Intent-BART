"""Environment capture, so every result file records how it was produced."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

_PACKAGES = ("torch", "transformers", "tokenizers", "numpy", "scipy", "scikit-learn")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def code_sha256(root: str | Path = ".") -> str:
    """SHA-256 over the Python sources in ``src/`` and the YAML files in ``configs/``.

    It identifies the code that produced a result independently of the git
    history. ``python -m gemis.utils.env`` prints the value for a checkout.
    """
    root = Path(root)
    files = sorted([*root.glob("src/**/*.py"), *root.glob("configs/*.yaml")])
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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

    # the commit is only recorded when code and configs match it exactly
    dirty = _git("status", "--porcelain", "--", "src", "scripts", "configs")
    git_sha = _git("rev-parse", "HEAD") if dirty == "" else None
    return {
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code_sha256": code_sha256(),
        "git_sha": git_sha,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "gpu": gpu,
        "device": str(device) if device is not None else None,
        "versions": versions,
    }


if __name__ == "__main__":
    print(code_sha256())
