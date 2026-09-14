"""Locate the same native HF snapshot for training and public evaluation."""

from __future__ import annotations

import os
from pathlib import Path


HF_BUNDLE_DIRECTORY = "ContextWorld-v3-hf"


def resolve_benchmark_root(
    explicit: str | Path | None,
    dataset_root: str | Path | None,
) -> Path:
    """Honor explicit paths, then the snapshot under the external data root.

    An explicitly selected or installed but incomplete bundle must fail the
    caller's validation; it must never silently fall back to another dataset.
    Legacy v1 directories and in-checkout staging are not automatic fallbacks.
    """

    def absolute(value: str | Path, label: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{label} must be absolute: {value}")
        return Path(os.path.abspath(path))

    if explicit:
        return absolute(explicit, "--benchmark-root/CONTEXTWORLD_BENCHMARK_ROOT")
    if dataset_root:
        return (
            absolute(dataset_root, "--dataset-root/CONTEXTWORLD_DATASET_ROOT")
            / HF_BUNDLE_DIRECTORY
        )  # The caller reports an incomplete bundle at this exact location.
    raise ValueError(
        "Set --benchmark-root/CONTEXTWORLD_BENCHMARK_ROOT to the downloaded "
        "ContextWorld-v3-hf snapshot, or set --dataset-root/CONTEXTWORLD_DATASET_ROOT "
        "to its parent data directory. Store the full dataset outside the code checkout."
    )


if __name__ == "__main__":
    try:
        print(resolve_benchmark_root(
            os.environ.get("CONTEXTWORLD_BENCHMARK_ROOT"),
            os.environ.get("CONTEXTWORLD_DATASET_ROOT"),
        ))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
