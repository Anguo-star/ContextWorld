#!/usr/bin/env python3
"""Build a tiny paired catalog from the audited formal passage release."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.hidden_passage_validation import file_sha256
from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_tiny_overfit_data_v1.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _require_hash(path: Path, expected: str) -> None:
    observed = file_sha256(path)
    if observed != str(expected):
        raise ValueError(
            f"Frozen source hash mismatch for {path}: "
            f"expected={expected}, observed={observed}"
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Invalid JSONL row at {path}:{line_number}")
        rows.append(row)
    return rows


def _door_position(row: dict[str, Any]) -> int:
    values = row.get("factors", {}).get("door.position")
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"Manifest record lacks door.position: {row.get('scenario_id')}"
        )
    unique = {int(value) for value in values}
    if len(unique) != 1:
        raise ValueError(
            f"Manifest record has multiple door positions: {row}"
        )
    return unique.pop()


def _select(
    rows: list[dict[str, Any]],
    *,
    split: str,
    doors: list[int],
    rules: list[str],
    expected_shards: int,
    expected_clips: int,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("split")) == split
        and _door_position(row) in set(map(int, doors))
        and str(row.get("rule")) in set(map(str, rules))
    ]
    selected.sort(
        key=lambda row: (
            _door_position(row),
            str(row["rule"]),
            str(row["scenario_id"]),
        )
    )
    if len(selected) != int(expected_shards):
        raise ValueError(
            f"Expected {expected_shards} {split} shards, got {len(selected)}"
        )
    if sum(int(row["clip_count"]) for row in selected) != int(
        expected_clips
    ):
        raise ValueError(
            f"Expected {expected_clips} {split} clips, got "
            f"{sum(int(row['clip_count']) for row in selected)}"
        )
    observed = {
        (_door_position(row), str(row["rule"])) for row in selected
    }
    expected = {
        (int(door), str(rule)) for door in doors for rule in rules
    }
    if observed != expected:
        raise ValueError(
            f"Paired {split} door/rule cells differ: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return selected


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the one-train-door paired History=3 overfit diagnostic "
            "catalog from the frozen formal release"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_yaml(config_path)
    if config.get("status") != "diagnostic_frozen_before_subset_build":
        raise ValueError("Tiny-overfit data config is not frozen")

    source = config["source"]
    source_catalog_path = resolve_contextworld_path(
        source["catalog"], repo_root=ROOT
    )
    source_manifest_path = resolve_contextworld_path(
        source["manifest"], repo_root=ROOT
    )
    source_report_path = resolve_contextworld_path(
        source["formal_build_report"], repo_root=ROOT
    )
    _require_hash(source_catalog_path, source["catalog_sha256"])
    _require_hash(source_manifest_path, source["manifest_sha256"])
    _require_hash(
        source_report_path,
        source["formal_build_report_sha256"],
    )

    source_catalog = json.loads(
        source_catalog_path.read_text(encoding="utf-8")
    )
    source_report = json.loads(
        source_report_path.read_text(encoding="utf-8")
    )
    if (
        source_catalog.get("group") != "passage_mixed"
        or source_catalog.get("sampling_contract", {}).get(
            "synthetic_only"
        )
        is not True
        or source_report.get("passed") is not True
    ):
        raise ValueError("Frozen source is not the completed mixed release")
    rows = _read_jsonl(source_manifest_path)

    selection = config["selection"]
    train_spec = selection["train"]
    validation_spec = selection["validation"]
    train = _select(
        rows,
        split="train",
        doors=list(map(int, train_spec["door_positions"])),
        rules=list(map(str, train_spec["rules"])),
        expected_shards=int(train_spec["expected_shards"]),
        expected_clips=int(train_spec["expected_clips"]),
    )
    validation = _select(
        rows,
        split="val",
        doors=list(map(int, validation_spec["door_positions"])),
        rules=list(map(str, validation_spec["rules"])),
        expected_shards=int(validation_spec["expected_shards"]),
        expected_clips=int(validation_spec["expected_clips"]),
    )
    selected = train + validation
    if len({row["output_path"] for row in selected}) != len(selected):
        raise ValueError("Tiny-overfit shard paths are not unique")
    if bool(selection["require_pair_id_per_door"]):
        for split_rows in (train, validation):
            by_door: dict[int, set[str]] = {}
            for row in split_rows:
                by_door.setdefault(_door_position(row), set()).add(
                    str(row["pair_id"])
                )
            if any(len(pair_ids) != 1 for pair_ids in by_door.values()):
                raise ValueError(
                    f"Passable/blocked pair_id mismatch: {by_door}"
                )
    if bool(selection["require_equal_rule_clip_counts"]):
        by_rule = Counter()
        for row in train:
            by_rule[str(row["rule"])] += int(row["clip_count"])
        if len(set(by_rule.values())) != 1:
            raise ValueError(f"Train rule clips are imbalanced: {by_rule}")

    artifact_spec = config["artifacts"]
    output_root = resolve_contextworld_path(
        artifact_spec["output_root"], repo_root=ROOT
    )
    catalog_path = resolve_contextworld_path(
        artifact_spec["catalog"], repo_root=ROOT
    )
    manifest_path = resolve_contextworld_path(
        artifact_spec["manifest"], repo_root=ROOT
    )
    report_path = resolve_contextworld_path(
        artifact_spec["report"], repo_root=ROOT
    )
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite tiny-overfit data: {output_root}"
        )
    for path in (catalog_path, manifest_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    train_paths = [str(row["output_path"]) for row in train]
    validation_paths = [str(row["output_path"]) for row in validation]
    benchmark = str(config["benchmark"])
    catalog = {
        "schema_version": 1,
        "benchmark": benchmark,
        "diagnostic_only": True,
        "diagnostic_source_passage_release_root": (
            portable_contextworld_path(
                source_catalog_path.parent.parent,
                repo_root=ROOT,
            )
        ),
        "group": "passage_tiny_overfit",
        "scale": "tiny_paired_diagnostic",
        "pixel_codec": source_catalog["pixel_codec"],
        "model_columns": ["pixels", "action"],
        "raw_privileged_columns_excluded_from_model": (
            source_catalog["raw_privileged_columns_excluded_from_model"]
        ),
        "rule_support": {
            "names": ["passable", "blocked"],
            "passage_open_values": [1, 0],
        },
        "sampling_contract": {
            "synthetic_only": True,
            "original_samples_included": False,
            "purpose": "exact_memorization_diagnostic_only",
        },
        "train": {"original": [], "synthetic": train_paths},
        "val": {"synthetic": validation_paths},
        "ood_test": {"synthetic": []},
        "by_regime": {
            "train_tiny_paired_history3": train_paths,
            "validation_tiny_paired_history3": validation_paths,
            "test_tiny_paired_history3": [],
        },
        "counts": {
            "train": {
                "shards": len(train),
                "clips": sum(int(row["clip_count"]) for row in train),
            },
            "val": {
                "shards": len(validation),
                "clips": sum(
                    int(row["clip_count"]) for row in validation
                ),
            },
            "test": {"shards": 0, "clips": 0},
        },
    }
    write_json(catalog_path, catalog)
    _atomic_write_jsonl(manifest_path, selected)
    report = {
        "schema_version": 1,
        "benchmark": benchmark,
        "passed": True,
        "diagnostic_only": True,
        "config": portable_contextworld_path(
            config_path, repo_root=ROOT
        ),
        "config_sha256": file_sha256(config_path),
        "source": {
            "catalog_sha256": source["catalog_sha256"],
            "manifest_sha256": source["manifest_sha256"],
            "formal_build_report_sha256": source[
                "formal_build_report_sha256"
            ],
        },
        "selection": {
            "train_doors": list(map(int, train_spec["door_positions"])),
            "validation_doors": list(
                map(int, validation_spec["door_positions"])
            ),
            "train_shards": len(train),
            "validation_shards": len(validation),
            "train_clips": sum(int(row["clip_count"]) for row in train),
            "validation_clips": sum(
                int(row["clip_count"]) for row in validation
            ),
            "rules": ["passable", "blocked"],
        },
        "catalog": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
