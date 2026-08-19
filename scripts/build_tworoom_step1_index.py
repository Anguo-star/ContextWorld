#!/usr/bin/env python3
"""Build a compact, fail-closed index for the formal TwoRoom step-1 data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.paths import (  # noqa: E402
    artifact_path,
    portable_contextworld_path,
    resolve_contextworld_path,
)

DATASETS = (
    {
        "key": "speed",
        "experiment": "tworoom_speed_pixel_v2",
        "atoms": ["agent_speed"],
    },
    {
        "key": "door",
        "experiment": "tworoom_door_pixel_v1",
        "atoms": ["door_position"],
    },
    {
        "key": "speed_door_composition",
        "experiment": "tworoom_speed_door_composition_v1",
        "atoms": ["agent_speed", "door_position"],
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _relative(path: Path) -> str:
    return portable_contextworld_path(path, repo_root=ROOT)


def _dataset_entry(specification: dict[str, Any]) -> dict[str, Any]:
    experiment = specification["experiment"]
    report_path = artifact_path(
        "synthesis", "reports", f"{experiment}.json", repo_root=ROOT
    )
    catalog_path = artifact_path(
        "synthesis", "catalogs", f"{experiment}.json", repo_root=ROOT
    )
    manifest_path = artifact_path(
        "synthesis", "manifests", f"{experiment}.jsonl", repo_root=ROOT
    )
    data_root = artifact_path(
        "synthesis", "data", experiment, repo_root=ROOT
    )
    report = _read_json(report_path)
    catalog = _read_json(catalog_path)

    if not report.get("passed") or report.get("compile_only"):
        raise ValueError(
            f"{experiment} is not a passed, fully collected report"
        )
    scenarios = report.get("scenarios", [])
    if not scenarios or not all(item.get("passed") for item in scenarios):
        raise ValueError(f"{experiment} has a failed or missing scenario report")

    rows = sum(int(item["rows"]) for item in scenarios)
    exact_rows = sum(
        int(item["exact_replay"]["rows_checked"]) for item in scenarios
    )
    transitions = sum(
        int(item["exact_replay"]["transitions_checked"])
        for item in scenarios
    )
    if exact_rows != rows:
        raise ValueError(
            f"{experiment} exact replay covered {exact_rows}/{rows} rows"
        )
    if any(
        item["exact_replay"]["maximum_state_absolute_error"] != 0.0
        for item in scenarios
    ):
        raise ValueError(f"{experiment} has a non-zero exact replay error")

    manifest_records = sum(
        1
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if manifest_records != len(scenarios):
        raise ValueError(
            f"{experiment} manifest/report scenario count mismatch"
        )
    bytes_on_disk = sum(
        path.stat().st_size for path in data_root.rglob("*") if path.is_file()
    )
    episode_lengths = [
        int(length)
        for scenario in scenarios
        for length in scenario["episode_lengths"]
    ]
    split_rows: dict[str, int] = {}
    for scenario in scenarios:
        split_rows[scenario["split"]] = (
            split_rows.get(scenario["split"], 0) + int(scenario["rows"])
        )

    return {
        "key": specification["key"],
        "experiment": experiment,
        "atoms": specification["atoms"],
        "config": f"configs/synthesis/{experiment}.yaml",
        "data_root": _relative(data_root),
        "manifest": _relative(manifest_path),
        "report": _relative(report_path),
        "catalog": _relative(catalog_path),
        "summary": {
            "scenarios": len(scenarios),
            "episodes": int(report["projection"]["episodes"]),
            "rows": rows,
            "rows_by_split": dict(sorted(split_rows.items())),
            "transitions_exactly_replayed": transitions,
            "rows_exactly_replayed": exact_rows,
            "minimum_episode_rows": min(episode_lengths),
            "maximum_episode_rows": max(episode_lengths),
            "bytes_on_disk": bytes_on_disk,
        },
        "regimes": {
            name: len(paths)
            for name, paths in sorted(catalog["by_regime"].items())
        },
        "train_source_count": len(catalog["train"]["synthetic"]),
        "validation_source_count": len(catalog["val"]["synthetic"]),
        "test_source_count": len(catalog["ood_test"]["synthetic"]),
    }


def build_index() -> dict[str, Any]:
    entries = [_dataset_entry(specification) for specification in DATASETS]
    reports = [
        _read_json(resolve_contextworld_path(entry["report"], repo_root=ROOT))
        for entry in entries
    ]
    commits = {
        report["stable_worldmodel"]["commit"] for report in reports
    }
    original_paths = {
        _read_json(resolve_contextworld_path(entry["catalog"], repo_root=ROOT))[
            "original_dataset_read_only"
        ]
        for entry in entries
    }
    if len(commits) != 1:
        raise ValueError(f"Stable-WM commit mismatch: {sorted(commits)}")
    if len(original_paths) != 1:
        raise ValueError(f"Original dataset mismatch: {sorted(original_paths)}")

    return {
        "schema_version": 1,
        "benchmark": "tworoom_benchmark_step1_v1",
        "status": "formal_data_validated",
        "terminology": {
            "preferred": "训练未见组合",
            "alias": "组合留出",
        },
        "stable_worldmodel_commit": next(iter(commits)),
        "original_dataset_read_only": next(iter(original_paths)),
        "datasets": entries,
        "totals": {
            "scenarios": sum(
                entry["summary"]["scenarios"] for entry in entries
            ),
            "episodes": sum(
                entry["summary"]["episodes"] for entry in entries
            ),
            "rows": sum(entry["summary"]["rows"] for entry in entries),
            "rows_exactly_replayed": sum(
                entry["summary"]["rows_exactly_replayed"]
                for entry in entries
            ),
            "transitions_exactly_replayed": sum(
                entry["summary"]["transitions_exactly_replayed"]
                for entry in entries
            ),
            "bytes_on_disk": sum(
                entry["summary"]["bytes_on_disk"] for entry in entries
            ),
        },
        "experiment_matrix_config": (
            "configs/benchmark/tworoom_step1_v1.yaml"
        ),
        "benchmark_design": "docs/ContextWorld_ICL_Benchmark.md",
        "icl_protocol_config": "configs/benchmark/tworoom_icl_v1.yaml",
        "data_card": "docs/reference/TwoRoom_Benchmark_Step1_Data_Card_v1.md",
        "progress_checklist": "docs/ContextWorld_ICL_Benchmark.md",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_path(
            "synthesis/catalogs/tworoom_benchmark_step1_v1.json",
            repo_root=ROOT,
        ),
    )
    args = parser.parse_args()
    index = build_index()
    output = resolve_contextworld_path(args.output, repo_root=ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "datasets": len(index["datasets"]),
                "rows": index["totals"]["rows"],
                "passed": (
                    index["totals"]["rows"]
                    == index["totals"]["rows_exactly_replayed"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
