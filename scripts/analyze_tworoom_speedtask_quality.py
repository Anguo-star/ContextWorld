#!/usr/bin/env python3
"""Produce the formal data-quality report for TwoRoom-SpeedTask-v1.

This report deliberately separates task-distribution fidelity from controller
quality.  It checks the frozen geometry-by-speed cross and reports outcome
strata without filtering unsuccessful trajectories.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.paths import portable_contextworld_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from scripts.analyze_tworoom_trajectory_quality import (
    EpisodeTable,
    _load_original_h5,
    _load_synthetic_catalog,
    _match_speed_room,
    _policy_and_environment_checks,
    _speed_outcome_coupling,
    room_relation,
    summarize_episodes,
)


def _geometry_key(start: np.ndarray, goal: np.ndarray) -> tuple[float, ...]:
    values = np.concatenate((start, goal))
    return tuple(float(value) for value in np.round(values, decimals=5))


def paired_factor_cross(episodes: EpisodeTable) -> dict[str, Any]:
    """Summarize the exact geometry-by-speed Cartesian product."""

    if episodes.speed is None:
        raise ValueError("Synthetic episodes must include speed values")

    factors_by_geometry: dict[tuple[float, ...], list[float]] = defaultdict(list)
    factor_counts: Counter[float] = Counter()
    for start, goal, speed in zip(
        episodes.start, episodes.goal, episodes.speed, strict=True
    ):
        factor = round(float(speed), 6)
        factors_by_geometry[_geometry_key(start, goal)].append(factor)
        factor_counts[factor] += 1

    factor_values = sorted(factor_counts)
    expected_factor_set = set(factor_values)
    geometry_factor_counts = [
        len(set(factors)) for factors in factors_by_geometry.values()
    ]
    geometry_episode_counts = [len(factors) for factors in factors_by_geometry.values()]
    duplicate_factor_rows = sum(
        len(factors) - len(set(factors)) for factors in factors_by_geometry.values()
    )
    complete_geometries = sum(
        set(factors) == expected_factor_set and len(factors) == len(expected_factor_set)
        for factors in factors_by_geometry.values()
    )

    unique_mask: list[int] = []
    seen: set[tuple[float, ...]] = set()
    for index, (start, goal) in enumerate(
        zip(episodes.start, episodes.goal, strict=True)
    ):
        key = _geometry_key(start, goal)
        if key not in seen:
            seen.add(key)
            unique_mask.append(index)
    unique_indices = np.asarray(unique_mask, dtype=np.int64)
    unique_cross = room_relation(
        episodes.start[unique_indices], episodes.goal[unique_indices]
    )
    left_to_right = (
        (episodes.start[unique_indices, 0] < 112.0)
        & (episodes.goal[unique_indices, 0] >= 112.0)
    )

    return {
        "factor_values": factor_values,
        "factor_value_count": len(factor_values),
        "independent_geometries": len(factors_by_geometry),
        "complete_factor_cross_geometries": int(complete_geometries),
        "minimum_unique_factors_per_geometry": int(min(geometry_factor_counts)),
        "maximum_unique_factors_per_geometry": int(max(geometry_factor_counts)),
        "minimum_episodes_per_geometry": int(min(geometry_episode_counts)),
        "maximum_episodes_per_geometry": int(max(geometry_episode_counts)),
        "duplicate_factor_rows_within_geometry": int(duplicate_factor_rows),
        "minimum_episodes_per_factor": int(min(factor_counts.values())),
        "maximum_episodes_per_factor": int(max(factor_counts.values())),
        "episodes_per_factor": {
            f"{factor:.6g}": int(factor_counts[factor]) for factor in factor_values
        },
        "unique_geometry_room_relation": {
            "cross_room": int(unique_cross.sum()),
            "same_room": int((~unique_cross).sum()),
            "left_to_right": int(left_to_right.sum()),
            "right_to_left": int((~left_to_right).sum()),
        },
    }


def _outcome_strata(episodes: EpisodeTable) -> dict[str, Any]:
    if episodes.speed is None:
        raise ValueError("Synthetic episodes must include speed values")
    rows: list[dict[str, Any]] = []
    for speed in sorted(np.unique(episodes.speed)):
        selected = np.isclose(episodes.speed, speed)
        successes = int(episodes.terminated[selected].sum())
        nontermination = int(selected.sum()) - successes
        truncations = int(episodes.truncated[selected].sum())
        terminated_and_truncated = int(
            (episodes.terminated[selected] & episodes.truncated[selected]).sum()
        )
        rows.append(
            {
                "speed": round(float(speed), 6),
                "episodes": int(selected.sum()),
                "termination_successes": successes,
                "nontermination_episodes": nontermination,
                "truncations": truncations,
                "terminated_and_truncated": terminated_and_truncated,
                "both_success_and_nontermination_represented": (
                    successes > 0 and nontermination > 0
                ),
            }
        )
    return {
        "by_speed": rows,
        "speeds_with_both_success_and_nontermination": sum(
            row["both_success_and_nontermination_represented"] for row in rows
        ),
        "minimum_termination_successes_per_speed": min(
            row["termination_successes"] for row in rows
        ),
        "minimum_nontermination_episodes_per_speed": min(
            row["nontermination_episodes"] for row in rows
        ),
        "minimum_truncations_per_speed": min(row["truncations"] for row in rows),
    }


def evaluate_quality_gates(
    episodes: EpisodeTable,
    paired: dict[str, Any],
    *,
    expected_speeds: list[float],
) -> dict[str, Any]:
    summary = summarize_episodes(episodes)
    strata = _outcome_strata(episodes)
    observed_speeds = [round(value, 6) for value in paired["factor_values"]]
    expected_speeds = sorted(round(float(value), 6) for value in expected_speeds)
    expected_geometry_count = 512
    expected_factor_count = len(expected_speeds)
    checks = {
        "all_training_episodes_are_cross_room": bool(
            np.isclose(summary["cross_room_fraction"], 1.0)
        ),
        "at_least_512_independent_geometries": (
            paired["independent_geometries"] >= expected_geometry_count
        ),
        "exact_frozen_speed_support": observed_speeds == expected_speeds,
        "every_geometry_has_the_complete_speed_cross": (
            paired["complete_factor_cross_geometries"]
            == paired["independent_geometries"]
            and paired["minimum_unique_factors_per_geometry"]
            == expected_factor_count
            and paired["duplicate_factor_rows_within_geometry"] == 0
        ),
        "every_speed_has_at_least_512_episodes": (
            paired["minimum_episodes_per_factor"] >= expected_geometry_count
        ),
        "both_directions_have_at_least_200_geometries": (
            paired["unique_geometry_room_relation"]["left_to_right"] >= 200
            and paired["unique_geometry_room_relation"]["right_to_left"] >= 200
        ),
        "every_speed_retains_success_and_nontermination_strata": (
            strata["speeds_with_both_success_and_nontermination"]
            == expected_factor_count
        ),
        "trajectory_rows_exceed_one_million": summary["rows"] >= 1_000_000,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "outcome_strata": strata,
    }


def _train_speeds(config: dict[str, Any]) -> list[float]:
    for scenario_set in config["scenario_sets"]:
        if scenario_set["split"] == "train" and scenario_set["atom"] == "agent_speed":
            return [float(value) for value in scenario_set["values"]["values"]]
    raise ValueError("No train agent_speed scenario set found")


def run(args: argparse.Namespace) -> dict[str, Any]:
    original_path = resolve_contextworld_path(args.original_h5, repo_root=REPO_ROOT)
    speedseen_path = resolve_contextworld_path(
        args.speedseen_catalog, repo_root=REPO_ROOT
    )
    speedtask_path = resolve_contextworld_path(
        args.speedtask_catalog, repo_root=REPO_ROOT
    )
    config_path = resolve_contextworld_path(args.config, repo_root=REPO_ROOT)
    stablewm_repo = resolve_contextworld_path(args.stablewm_repo, repo_root=REPO_ROOT)
    output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_speeds = _train_speeds(config)
    original = _load_original_h5(original_path)
    speedseen = _load_synthetic_catalog(speedseen_path)
    speedtask = _load_synthetic_catalog(speedtask_path)
    original_summary = summarize_episodes(original)
    speedseen_summary = summarize_episodes(speedseen)
    speedtask_summary = summarize_episodes(speedtask)
    paired = paired_factor_cross(speedtask)
    quality_gates = evaluate_quality_gates(
        speedtask, paired, expected_speeds=expected_speeds
    )

    speed5_speedseen = _match_speed_room(speedseen, speed=5.0, cross_room=True)
    speed5_speedtask = _match_speed_room(speedtask, speed=5.0, cross_room=True)
    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_speed_task_data_quality_v1",
        "status": "passed" if quality_gates["passed"] else "failed",
        "sources": {
            "original_h5": str(original_path),
            "speedseen_catalog": portable_contextworld_path(
                speedseen_path, repo_root=REPO_ROOT
            ),
            "speedtask_catalog": portable_contextworld_path(
                speedtask_path, repo_root=REPO_ROOT
            ),
            "speedtask_config": str(config_path.relative_to(REPO_ROOT)),
        },
        "frozen_collection_recipe": _policy_and_environment_checks(
            config_path, stablewm_repo
        ),
        "quality_gates": quality_gates,
        "datasets": {
            "original_tworoom_h5": original_summary,
            "speedseen_synthetic_train": speedseen_summary,
            "speedtask_synthetic_train": speedtask_summary,
        },
        "paired_geometry_speed_cross": paired,
        "speedtask_speed_outcome_strata": _speed_outcome_coupling(speedtask),
        "matched_speed5_cross_room": {
            "original": original_summary["cross_room"],
            "speedseen": summarize_episodes(speed5_speedseen),
            "speedtask": summarize_episodes(speed5_speedtask),
        },
        "controlled_comparison": {
            "fixed": [
                "32 train speed values",
                "ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05)",
                "lossless PNG codec",
                "maximum episode length 100",
            ],
            "changed": [
                "opposite-room targets are enforced",
                "independent train geometries increase from 128 to 512",
                "every geometry is crossed with every speed",
            ],
            "interpretation": (
                "This dataset repairs task semantics and effective geometry coverage. "
                "Outcome differences across speed are measured and retained rather than "
                "removed by success-only filtering; model benefit remains an empirical "
                "training/evaluation question."
            ),
        },
    }
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate TwoRoom-SpeedTask-v1 task and trajectory composition."
    )
    parser.add_argument(
        "--original-h5",
        type=Path,
        default=Path("../../data/world_model/quentinll/lewm-tworooms/tworoom.h5"),
    )
    parser.add_argument(
        "--speedseen-catalog",
        type=Path,
        default=Path("artifacts/synthesis/catalogs/tworoom_speed_seen_v1.json"),
    )
    parser.add_argument(
        "--speedtask-catalog",
        type=Path,
        default=Path("artifacts/synthesis/catalogs/tworoom_speed_task_v1.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthesis/tworoom_speed_task_v1.yaml"),
    )
    parser.add_argument(
        "--stablewm-repo", type=Path, default=Path("../stable-worldmodel")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/synthesis/reports/tworoom_speed_task_quality_v1.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "benchmark": result["benchmark"],
                "status": result["status"],
                "quality_gates": result["quality_gates"]["checks"],
            },
            sort_keys=True,
        )
    )
