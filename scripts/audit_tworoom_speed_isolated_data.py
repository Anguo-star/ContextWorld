#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


ARTIFACT_ROOT = resolve_contextworld_path("artifacts", repo_root=ROOT)


def _manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = {str(row["seed_group"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"Duplicate seed_group in {path}")
    return result


def _artifact_path(value: str) -> Path:
    return resolve_contextworld_path(value, repo_root=ROOT)


def _scenario_readback(
    row: dict[str, Any],
) -> dict[str, Any]:
    import lance

    path = _artifact_path(str(row["output_path"]))
    dataset = lance.dataset(path)
    rows = int(dataset.count_rows())
    speed_table = dataset.to_table(
        columns=["variation_agent_speed"]
    ).to_pydict()
    speeds = sorted(
        {
            round(float(value[0]), 6)
            for value in speed_table["variation_agent_speed"]
        }
    )
    reset_table = dataset.to_table(
        columns=[
            "episode_idx",
            "variation_agent_position",
            "variation_target_position",
        ],
        filter="step_idx = 0",
    ).to_pydict()
    order = np.argsort(
        np.asarray(reset_table["episode_idx"], dtype=np.int64)
    )
    states = np.asarray(
        [
            reset_table["variation_agent_position"][index]
            for index in order
        ],
        dtype=np.float32,
    )
    goals = np.asarray(
        [
            reset_table["variation_target_position"][index]
            for index in order
        ],
        dtype=np.float32,
    )
    return {
        "path": str(path),
        "rows": rows,
        "speeds": speeds,
        "episodes": len(order),
        "reset_states": states,
        "goal_states": goals,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    single_manifest_path = _artifact_path(
        "artifacts/synthesis/manifests/"
        "tworoom_speed_single_matched_v2.jsonl"
    )
    multi_manifest_path = _artifact_path(
        "artifacts/synthesis/manifests/tworoom_speed_full_v1.jsonl"
    )
    single = _manifest(single_manifest_path)
    multi = _manifest(multi_manifest_path)
    if single.keys() != multi.keys():
        raise RuntimeError("Single/multi seed groups differ")
    fixed_fields = (
        "split",
        "regime",
        "episodes",
        "env_seed",
        "policy_seed",
        "reset_constraints",
        "image_shape",
        "max_episode_steps",
        "pixel_codec",
        "stable_worldmodel_commit",
        "variation",
    )
    manifest_failures = []
    rows_by_split = {
        "single": Counter(),
        "multi": Counter(),
    }
    speed_support = {
        "single": {"train": set(), "val": set()},
        "multi": {"train": set(), "val": set()},
    }
    geometry_failures = []
    pair_rows = []
    for index, seed_group in enumerate(sorted(single)):
        left = single[seed_group]
        right = multi[seed_group]
        mismatches = [
            field
            for field in fixed_fields
            if left[field] != right[field]
        ]
        if mismatches:
            manifest_failures.append(
                {"seed_group": seed_group, "fields": mismatches}
            )
        left_data = _scenario_readback(left)
        right_data = _scenario_readback(right)
        split = str(left["split"])
        rows_by_split["single"][split] += left_data["rows"]
        rows_by_split["multi"][split] += right_data["rows"]
        speed_support["single"][split].update(left_data["speeds"])
        speed_support["multi"][split].update(right_data["speeds"])
        geometry_equal = bool(
            np.array_equal(
                left_data["reset_states"], right_data["reset_states"]
            )
            and np.array_equal(
                left_data["goal_states"], right_data["goal_states"]
            )
        )
        if not geometry_equal:
            geometry_failures.append(seed_group)
        pair_rows.append(
            {
                "seed_group": seed_group,
                "split": split,
                "single_rows": left_data["rows"],
                "multi_rows": right_data["rows"],
                "episodes": left_data["episodes"],
                "reset_and_goal_states_identical": geometry_equal,
            }
        )
        if (index + 1) % 32 == 0:
            print(
                f"[{index + 1}/{len(single)}] audited paired scenarios",
                flush=True,
            )

    expected = config["speed_support"]
    expected_support = {
        "single": {
            "train": {5.0},
            "val": {5.0},
        },
        "multi": {
            "train": set(
                map(float, expected["multi_synthetic_train"])
            ),
            "val": set(
                map(float, expected["training_monitor_only"])
            ),
        },
    }
    support_checks = {
        model: {
            split: set(values) == expected_support[model][split]
            for split, values in splits.items()
        }
        for model, splits in speed_support.items()
    }
    preflight = {}
    for model in ("single", "multi"):
        path = _artifact_path(
            "artifacts/training/reports/"
            f"h3_speed_{model}_v2_s3072_preflight.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        group = f"speed_{model}_v2"
        exposure = payload["training_plan"]["group_exposure"][group]
        original_exposure = payload["training_plan"][
            "group_exposure"
        ]["original"]
        preflight[model] = {
            "path": str(path),
            "passed": bool(payload["passed"]),
            "raw_train_clips": int(exposure["raw_train_clips"]),
            "total_draws": int(exposure["total_draws"]),
            "mean_draws_per_raw_clip": float(
                exposure["mean_draws_per_raw_clip"]
            ),
            "original_raw_train_clips": int(
                original_exposure["raw_train_clips"]
            ),
            "original_total_draws": int(
                original_exposure["total_draws"]
            ),
        }
    passed = bool(
        not manifest_failures
        and not geometry_failures
        and all(
            check
            for model in support_checks.values()
            for check in model.values()
        )
        and all(row["passed"] for row in preflight.values())
        and preflight["single"]["total_draws"]
        == preflight["multi"]["total_draws"]
        and preflight["single"]["original_total_draws"]
        == preflight["multi"]["original_total_draws"]
    )
    output = {
        "schema_version": 1,
        "benchmark": "tworoom_speed_isolated_data_audit_v2",
        "status": "passed" if passed else "failed",
        "paired_scenarios": len(pair_rows),
        "manifest_pairing": {
            "fixed_fields": list(fixed_fields),
            "failures": manifest_failures,
            "passed": not manifest_failures,
        },
        "actual_reset_goal_pairing": {
            "failed_seed_groups": geometry_failures,
            "passed": not geometry_failures,
        },
        "actual_speed_support": {
            model: {
                split: sorted(values)
                for split, values in splits.items()
            }
            for model, splits in speed_support.items()
        },
        "speed_support_checks": support_checks,
        "actual_rows": {
            model: dict(values)
            for model, values in rows_by_split.items()
        },
        "training_preflight": preflight,
        "exposure_equality": {
            "single_synthetic_draws": preflight["single"][
                "total_draws"
            ],
            "multi_synthetic_draws": preflight["multi"]["total_draws"],
            "single_original_draws": preflight["single"][
                "original_total_draws"
            ],
            "multi_original_draws": preflight["multi"][
                "original_total_draws"
            ],
            "passed": (
                preflight["single"]["total_draws"]
                == preflight["multi"]["total_draws"]
                and preflight["single"]["original_total_draws"]
                == preflight["multi"]["original_total_draws"]
            ),
        },
        "pairs": pair_rows,
    }
    output_path = resolve_contextworld_path(
        config["artifacts"]["support_audit"], repo_root=ROOT
    )
    write_json(output_path, output)
    if not passed:
        raise RuntimeError(f"Data isolation audit failed: {output_path}")
    return {**output, "output": str(output_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT / "configs/benchmark/tworoom_speed_isolated_v2.yaml"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": result["output"],
                "paired_scenarios": result["paired_scenarios"],
                "speed_support": result["actual_speed_support"],
                "exposure": result["training_preflight"],
            },
            indent=2,
            sort_keys=True,
        )
    )
