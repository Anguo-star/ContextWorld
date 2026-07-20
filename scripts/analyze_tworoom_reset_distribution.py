#!/usr/bin/env python3
"""Preflight a synthetic reset distribution against frozen original-train geometry."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.config import (
    build_compiler,
    scenario_requests,
)
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.reset_constraints import (
    apply_tworoom_reset_constraints,
)
from contextworld.synthesis.stablewm import load_stable_worldmodel
from scripts.analyze_tworoom_synth5_matched import (
    _grid_counts,
    _total_variation,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specification = config["stable_worldmodel"]
    load_stable_worldmodel(
        REPO_ROOT,
        specification["repo"],
        specification.get("expected_ref"),
    )
    compiler = build_compiler(config, REPO_ROOT)
    scenarios = compiler.compile_all(scenario_requests(config))
    reference_path = resolve_contextworld_path(
        config["output"]["frozen_reference"], repo_root=REPO_ROOT
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_geometry = reference["geometry"]

    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    observed: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"starts": [], "goals": []}
    )
    reset_seeds: dict[str, set[int]] = defaultdict(set)
    env = TwoRoomEnv(render_mode="rgb_array")
    try:
        for scenario in scenarios:
            apply_tworoom_reset_constraints(env, scenario.reset_constraints)
            for episode_index in range(scenario.episodes):
                reset_seed = scenario.env_seed + episode_index
                observation, _ = env.reset(
                    seed=reset_seed,
                    options={
                        "variation": scenario.variation,
                        "variation_values": scenario.variation_values,
                    },
                )
                observed[scenario.split]["starts"].append(
                    np.asarray(observation[:2], dtype=np.float32)
                )
                observed[scenario.split]["goals"].append(
                    np.asarray(observation[2:4], dtype=np.float32)
                )
                reset_seeds[scenario.split].add(reset_seed)
    finally:
        env.close()

    gate_config = config["distribution_matching"]["gates"]
    split_reports = {}
    for split, values in sorted(observed.items()):
        starts = np.asarray(values["starts"], dtype=np.float32)
        goals = np.asarray(values["goals"], dtype=np.float32)
        distances = np.linalg.norm(starts - goals, axis=1)
        start_counts = _grid_counts(starts)
        goal_counts = _grid_counts(goals)
        distance_quantiles = {
            str(value): float(np.quantile(distances, value))
            for value in (0.1, 0.5, 0.9)
        }
        distance_differences = {
            key: float(
                distance_quantiles[key]
                - reference_geometry["initial_distance_quantiles_px"][key]
            )
            for key in distance_quantiles
        }
        start_tv = _total_variation(
            start_counts, reference_geometry["start_grid_counts"]
        )
        goal_tv = _total_variation(
            goal_counts, reference_geometry["goal_grid_counts"]
        )
        left_to_right = float(
            np.mean((starts[:, 0] < 112.0) & (goals[:, 0] >= 112.0))
        )
        cross_room = float(
            np.mean((starts[:, 0] < 112.0) != (goals[:, 0] < 112.0))
        )
        formal = split == "train"
        gates = {
            "unique_reset_seeds": len(reset_seeds[split]) == len(starts),
            "cross_room_fraction": (
                cross_room
                == float(reference_geometry["cross_room_fraction"])
            ),
        }
        if formal:
            gates.update(
                {
                    "initial_distance_quantiles": all(
                        abs(value)
                        <= float(
                            gate_config["initial_distance_quantiles_px"][
                                "maximum_absolute_difference"
                            ]
                        )
                        for value in distance_differences.values()
                    ),
                    "start_grid_total_variation": (
                        start_tv
                        <= float(
                            gate_config["start_grid_total_variation"][
                                "maximum"
                            ]
                        )
                    ),
                    "goal_grid_total_variation": (
                        goal_tv
                        <= float(
                            gate_config["goal_grid_total_variation"]["maximum"]
                        )
                    ),
                    "left_to_right_fraction": (
                        abs(
                            left_to_right
                            - float(
                                reference_geometry[
                                    "left_to_right_fraction"
                                ]
                            )
                        )
                        <= float(
                            gate_config["left_to_right_fraction"][
                                "absolute_tolerance"
                            ]
                        )
                    ),
                }
            )
        split_reports[split] = {
            "passed": all(gates.values()),
            "episodes": len(starts),
            "unique_reset_seeds": len(reset_seeds[split]),
            "cross_room_fraction": cross_room,
            "left_to_right_fraction": left_to_right,
            "initial_distance_quantiles_px": distance_quantiles,
            "initial_distance_differences_px": distance_differences,
            "start_grid_total_variation": start_tv,
            "goal_grid_total_variation": goal_tv,
            "gates": gates,
        }

    passed = all(report["passed"] for report in split_reports.values())
    payload = {
        "schema_version": 1,
        "benchmark": f"{config['experiment']}_reset_distribution_preflight",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "config": str(config_path),
        "reference": str(reference_path),
        "semantics": (
            "reset-only, before trajectory collection; uses all configured "
            "train/dev reset seeds and original-train geometry only"
        ),
        "splits": split_reports,
    }
    output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/synthesis/tworoom_synth5_matched_v2.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/synthesis/reports/"
            "tworoom_synth5_matched_v2_reset_distribution_preflight.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)
