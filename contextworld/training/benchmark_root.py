"""Locate the same native HF snapshot for training and public evaluation."""

from __future__ import annotations

import os
from pathlib import Path


HF_BUNDLE_DIRECTORY = "ContextWorld-v3-hf"
CHECKOUT_ROOT = Path(__file__).resolve().parents[2]


def resolve_benchmark_root(
    explicit: str | Path | None,
    dataset_root: str | Path | None,
    *,
    checkout_root: Path = CHECKOUT_ROOT,
) -> Path:
    """Honor explicit paths, then the installed snapshot, then local staging.

    An explicitly selected or installed but incomplete bundle must fail the
    caller's validation; it must never silently fall back to another dataset.
    Legacy v1 directories are not automatically selected.
    """

    def absolute(value: str | Path, label: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{label} must be absolute: {value}")
        return Path(os.path.abspath(path))

    if explicit:
        return absolute(explicit, "--benchmark-root/CONTEXTWORLD_BENCHMARK_ROOT")
    installed = (
        absolute(dataset_root, "--dataset-root/CONTEXTWORLD_DATASET_ROOT")
        / HF_BUNDLE_DIRECTORY
        if dataset_root else None
    )
    if installed is not None and installed.exists():
        return installed
    candidate = checkout_root / "artifacts/releases" / HF_BUNDLE_DIRECTORY
    if candidate.exists():
        return candidate.resolve()
    if installed is not None:
        return installed  # The caller reports the missing bundle at this path.
    raise ValueError(
        "Set --benchmark-root/CONTEXTWORLD_BENCHMARK_ROOT to the downloaded "
        "ContextWorld-v3-hf snapshot, or prepare artifacts/releases/ContextWorld-v3-hf."
    )


if __name__ == "__main__":
    try:
        print(resolve_benchmark_root(
            os.environ.get("CONTEXTWORLD_BENCHMARK_ROOT"),
            os.environ.get("CONTEXTWORLD_DATASET_ROOT"),
        ))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
