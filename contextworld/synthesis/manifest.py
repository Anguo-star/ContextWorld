from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from contextworld.paths import portable_contextworld_path

from .models import CompiledScenario


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_manifest(
    path: Path,
    scenarios: Iterable[CompiledScenario],
    *,
    repo_root: Path,
    stable_worldmodel_commit: str,
) -> None:
    records: list[str] = []
    for scenario in scenarios:
        record = scenario.to_manifest_record(root=repo_root)
        record["output_path"] = portable_contextworld_path(
            scenario.output_path, repo_root=repo_root
        )
        record["stable_worldmodel_commit"] = stable_worldmodel_commit
        record["collection_status"] = (
            "collected" if scenario.output_path.exists() else "planned"
        )
        records.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
    _atomic_write(path, "\n".join(records) + "\n")


def write_catalog(
    path: Path,
    scenarios: list[CompiledScenario],
    original_dataset: Path,
    repo_root: Path,
) -> None:
    def display(value: Path) -> str:
        return portable_contextworld_path(value, repo_root=repo_root)

    train = [display(s.output_path) for s in scenarios if s.split == "train"]
    val = [display(s.output_path) for s in scenarios if s.split == "val"]
    test = [display(s.output_path) for s in scenarios if s.split == "test"]
    regimes: dict[str, list[str]] = {}
    for scenario in scenarios:
        regime = scenario.regime or scenario.split
        regimes.setdefault(regime, []).append(display(scenario.output_path))
    codecs = {
        json.dumps(scenario.pixel_codec, sort_keys=True)
        for scenario in scenarios
    }
    if len(codecs) != 1:
        raise ValueError("All scenarios in one catalog must use one pixel codec")
    pixel_codec = json.loads(next(iter(codecs)))
    write_json(
        path,
        {
            "schema_version": 1,
            "mixing": "logical_concat_at_load_time",
            "pixel_codec": pixel_codec,
            "original_dataset_read_only": display(original_dataset),
            "train": {
                "original": [display(original_dataset)],
                "synthetic": train,
            },
            "val": {"synthetic": val},
            "ood_test": {"synthetic": test},
            "by_regime": regimes,
        },
    )
