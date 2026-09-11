"""Runtime evidence captured at evaluation time, without importing model stacks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any


def runtime_fingerprint() -> dict[str, Any]:
    """Describe the actual evaluator process; never infer past runtimes."""
    versions: dict[str, str | None] = {}
    for package in (
        "torch", "torchvision", "transformers", "numpy", "pylance", "pyarrow",
        "stable-worldmodel", "stable-pretraining", "Pillow",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result: dict[str, Any] = {
        "schema_version": 1,
        "evidence_kind": "captured_in_evaluation_process",
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }
    # Inference already imports torch. Core-only callers should not pay for it.
    torch = sys.modules.get("torch")
    if torch is not None:
        cuda = getattr(torch, "cuda", None)
        result["torch"] = {
            "version": str(torch.__version__),
            "cuda_build": getattr(torch.version, "cuda", None),
            "cuda_available": bool(cuda is not None and cuda.is_available()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
        if result["torch"]["cuda_available"]:
            result["torch"]["devices"] = [
                {"index": i, "name": cuda.get_device_name(i),
                 "capability": list(cuda.get_device_capability(i))}
                for i in range(cuda.device_count())
            ]
    requirements = Path(__file__).resolve().parents[2] / "requirements_frozen.txt"
    if requirements.is_file():
        result["requirements_frozen_sha256"] = hashlib.sha256(
            requirements.read_bytes()
        ).hexdigest()
    return result
