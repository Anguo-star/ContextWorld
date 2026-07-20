#!/usr/bin/env python3
"""Freeze and evaluate the TwoRoom Synth5Matched distribution contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.paths import portable_contextworld_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.training.episode_split import (
    episode_ids_sha256,
    episode_row_indices,
    partition_episode_ids,
)
from scripts.analyze_tworoom_training_data_gap import (
    GRID,
    TraceTable,
    load_original_h5,
    summarize_trace,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grid_counts(values: np.ndarray) -> dict[str, int]:
    grid = np.floor(np.asarray(values, dtype=np.float64) / GRID).astype(np.int16)
    counts = Counter(tuple(int(item) for item in row) for row in grid)
    return {",".join(map(str, key)): int(value) for key, value in sorted(counts.items())}


def _total_variation(left: dict[str, int], right: dict[str, int]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
        for key in keys
    )


def _episode_geometry(trace: TraceTable, episode_ids: np.ndarray) -> dict[str, Any]:
    starts = trace.state[trace.offsets[episode_ids]]
    goals = trace.goal[trace.offsets[episode_ids]]
    initial_distance = np.linalg.norm(starts - goals, axis=1)
    left_to_right = (starts[:, 0] < 112.0) & (goals[:, 0] >= 112.0)
    cross_room = (starts[:, 0] < 112.0) != (goals[:, 0] < 112.0)
    return {
        "start_grid_counts": _grid_counts(starts),
        "goal_grid_counts": _grid_counts(goals),
        "left_to_right_fraction": float(left_to_right.mean()),
        "cross_room_fraction": float(cross_room.mean()),
        "initial_distance_quantiles_px": {
            str(value): float(np.quantile(initial_distance, value))
            for value in (0.1, 0.5, 0.9)
        },
        "episode_length_quantiles": {
            str(value): float(np.quantile(trace.lengths[episode_ids], value))
            for value in (0.1, 0.5, 0.9)
        },
        "model_visible_start_goal_pairs_5dp": int(
            len(
                np.unique(
                    np.round(np.concatenate([starts, goals], axis=1), 5),
                    axis=0,
                )
            )
        ),
    }


def _normalizer(
    original_h5: Path,
    *,
    rows: np.ndarray,
    train_episode_ids: np.ndarray,
) -> dict[str, Any]:
    import h5py

    output: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "tworoom_original_train_s3072_unbiased_zscore_v1",
        "source": str(original_h5),
        "source_sha256": _sha256(original_h5),
        "statistics_scope": "original_9000_train_episodes_only",
        "train_episode_ids_sha256": episode_ids_sha256(train_episode_ids),
        "rows": int(len(rows)),
        "columns": {},
    }
    with h5py.File(original_h5, "r") as handle:
        for column in ("action", "proprio"):
            values = np.asarray(handle[column][rows], dtype=np.float64)
            values = values[~np.isnan(values).any(axis=1)]
            output["columns"][column] = {
                "mean": values.mean(axis=0).tolist(),
                "std_unbiased": values.std(axis=0, ddof=1).tolist(),
                "valid_rows": int(len(values)),
            }
    return output


def _load_synthetic_split(catalog_path: Path, split: str) -> TraceTable:
    import lance

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    key = "val" if split == "val" else split
    paths = list(catalog[key]["synthetic"])
    states: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    offsets: list[int] = []
    lengths: list[int] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    speeds: list[float] = []
    resets: list[np.ndarray] = []
    scenarios: list[str] = []
    row_offset = 0
    columns = [
        "episode_idx",
        "state",
        "goal_state",
        "action",
        "terminated",
        "truncated",
        "variation_agent_speed",
        "variation_agent_position",
    ]
    for logical in paths:
        path = resolve_contextworld_path(logical, repo_root=REPO_ROOT)
        values = lance.dataset(path).to_table(columns=columns).to_pydict()
        episode_idx = np.asarray(values["episode_idx"], dtype=np.int64)
        state = np.asarray(values["state"], dtype=np.float32)
        goal = np.asarray(values["goal_state"], dtype=np.float32)
        action = np.asarray(values["action"], dtype=np.float32)
        terminal = np.asarray(values["terminated"], dtype=np.float32)[:, 0]
        truncation = np.asarray(values["truncated"], dtype=np.float32)[:, 0]
        speed = np.asarray(values["variation_agent_speed"], dtype=np.float32)[:, 0]
        reset = np.asarray(values["variation_agent_position"], dtype=np.float32)
        states.append(state)
        goals.append(goal)
        actions.append(action)
        for episode in np.unique(episode_idx):
            rows = np.flatnonzero(episode_idx == episode)
            offsets.append(row_offset + int(rows[0]))
            lengths.append(int(len(rows)))
            terminated.append(bool(terminal[rows[-1]] > 0.5))
            truncated.append(bool(truncation[rows[-1]] > 0.5))
            speeds.append(float(speed[rows[0]]))
            resets.append(reset[rows[0]])
            scenarios.append(path.stem)
        row_offset += len(state)
    return TraceTable(
        state=np.concatenate(states),
        goal=np.concatenate(goals),
        action=np.concatenate(actions),
        offsets=np.asarray(offsets, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int64),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
        speed=np.asarray(speeds, dtype=np.float32),
        reset=np.asarray(resets, dtype=np.float32),
        reset_is_observed=True,
        scenario=np.asarray(scenarios, dtype=object),
    )


def _absolute_gate(observed: float, reference: float, tolerance: float) -> dict[str, Any]:
    difference = float(observed - reference)
    return {
        "passed": abs(difference) <= tolerance,
        "observed": float(observed),
        "reference": float(reference),
        "difference": difference,
        "maximum_absolute_difference": float(tolerance),
    }


def _minimum_fraction_gate(
    observed: float, reference: float, minimum_fraction: float
) -> dict[str, Any]:
    threshold = float(reference * minimum_fraction)
    return {
        "passed": observed >= threshold,
        "observed": float(observed),
        "reference": float(reference),
        "minimum_reference_fraction": float(minimum_fraction),
        "threshold": threshold,
    }


def _quantile_gate(
    observed: dict[str, float],
    reference: dict[str, float],
    maximum_difference: float,
) -> dict[str, Any]:
    differences = {
        key: float(observed[key] - reference[key]) for key in reference
    }
    return {
        "passed": all(abs(value) <= maximum_difference for value in differences.values()),
        "observed": observed,
        "reference": reference,
        "differences": differences,
        "maximum_absolute_difference": float(maximum_difference),
    }


def freeze_reference(config: dict[str, Any]) -> dict[str, Any]:
    original_h5 = resolve_contextworld_path(
        config["original_dataset"]["path"], repo_root=REPO_ROOT
    )
    trace = load_original_h5(original_h5)
    split = config["original_dataset"]["episode_split"]
    train, heldout = partition_episode_ids(
        len(trace.lengths),
        seed=int(split["seed"]),
        train_fraction=float(split["train_fraction"]),
    )
    mask = np.zeros(len(trace.lengths), dtype=bool)
    mask[train] = True
    rows = episode_row_indices(trace.lengths, trace.offsets, train)
    payload = {
        "schema_version": 1,
        "protocol": f"{config['experiment']}_reference",
        "status": "frozen_before_synthetic_collection",
        "source": {
            "original_h5": str(original_h5),
            "original_h5_sha256": _sha256(original_h5),
            "train_episodes": int(len(train)),
            "heldout_episodes": int(len(heldout)),
            "train_episode_ids_sha256": episode_ids_sha256(train),
            "heldout_episode_ids_sha256": episode_ids_sha256(heldout),
        },
        "distribution_matching": config["distribution_matching"],
        "summary": summarize_trace(trace, mask),
        "geometry": _episode_geometry(trace, train),
    }
    reference_path = resolve_contextworld_path(
        config["output"]["frozen_reference"], repo_root=REPO_ROOT
    )
    normalizer_path = resolve_contextworld_path(
        "artifacts/splits/tworoom_original_train_s3072_normalizer.json",
        repo_root=REPO_ROOT,
    )
    write_json(reference_path, payload)
    write_json(
        normalizer_path,
        _normalizer(original_h5, rows=rows, train_episode_ids=train),
    )
    return {
        "reference": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "normalizer": str(normalizer_path),
        "normalizer_sha256": _sha256(normalizer_path),
        "train_episode_ids_sha256": episode_ids_sha256(train),
    }


def evaluate_distribution(config: dict[str, Any]) -> dict[str, Any]:
    reference_path = resolve_contextworld_path(
        config["output"]["frozen_reference"], repo_root=REPO_ROOT
    )
    catalog_path = resolve_contextworld_path(
        config["output"]["catalog"], repo_root=REPO_ROOT
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    train = _load_synthetic_split(catalog_path, "train")
    dev = _load_synthetic_split(catalog_path, "val")
    train_ids = np.arange(len(train.lengths), dtype=np.int64)
    dev_ids = np.arange(len(dev.lengths), dtype=np.int64)
    observed_summary = summarize_trace(train)
    observed_geometry = _episode_geometry(train, train_ids)
    reference_summary = reference["summary"]
    reference_geometry = reference["geometry"]
    gate_config = config["distribution_matching"]["gates"]

    gates: dict[str, Any] = {}
    gates["episode_counts"] = {
        "passed": len(train.lengths) == 9000 and len(dev.lengths) == 1000,
        "train": int(len(train.lengths)),
        "dev": int(len(dev.lengths)),
    }
    gates["fixed_speed5"] = {
        "passed": bool(
            np.allclose(train.speed, 5.0) and np.allclose(dev.speed, 5.0)
        ),
        "train_unique": np.unique(train.speed).tolist(),
        "dev_unique": np.unique(dev.speed).tolist(),
    }
    gates["cross_room_fraction"] = _absolute_gate(
        observed_geometry["cross_room_fraction"],
        reference_geometry["cross_room_fraction"],
        float(gate_config["cross_room_fraction"]["absolute_tolerance"]),
    )
    gates["left_to_right_fraction"] = _absolute_gate(
        observed_geometry["left_to_right_fraction"],
        reference_geometry["left_to_right_fraction"],
        float(gate_config["left_to_right_fraction"]["absolute_tolerance"]),
    )
    gates["initial_distance_quantiles_px"] = _quantile_gate(
        observed_geometry["initial_distance_quantiles_px"],
        reference_geometry["initial_distance_quantiles_px"],
        float(
            gate_config["initial_distance_quantiles_px"][
                "maximum_absolute_difference"
            ]
        ),
    )
    gates["episode_length_quantiles"] = _quantile_gate(
        observed_geometry["episode_length_quantiles"],
        reference_geometry["episode_length_quantiles"],
        float(
            gate_config["episode_length_quantiles"]["maximum_absolute_difference"]
        ),
    )
    for name, distribution_key in (
        ("start_grid_total_variation", "start_grid_counts"),
        ("goal_grid_total_variation", "goal_grid_counts"),
    ):
        value = _total_variation(
            observed_geometry[distribution_key],
            reference_geometry[distribution_key],
        )
        maximum = float(gate_config[name]["maximum"])
        gates[name] = {
            "passed": value <= maximum,
            "observed": value,
            "maximum": maximum,
        }
    gates["reset_goal_grid_unique_cells"] = _minimum_fraction_gate(
        observed_summary["geometry"]["reset_goal_grid_14px"]["unique_cells"],
        reference_summary["geometry"]["reset_goal_grid_14px"]["unique_cells"],
        float(
            gate_config["reset_goal_grid_unique_cells"][
                "minimum_reference_fraction"
            ]
        ),
    )
    gates["state_goal_grid_unique_cells"] = _minimum_fraction_gate(
        observed_summary["model_visible_support"]["state_goal_grid_14px"][
            "unique_cells"
        ],
        reference_summary["model_visible_support"]["state_goal_grid_14px"][
            "unique_cells"
        ],
        float(
            gate_config["state_goal_grid_unique_cells"][
                "minimum_reference_fraction"
            ]
        ),
    )
    absolute_metrics = {
        "mean_episode_rows": (
            observed_summary["mean_episode_rows"],
            reference_summary["mean_episode_rows"],
        ),
        "termination_success_rate": (
            observed_summary["termination_success_rate"],
            reference_summary["termination_success_rate"],
        ),
        "goal_side_row_fraction": (
            observed_summary["model_visible_support"]["goal_side_row_fraction"],
            reference_summary["model_visible_support"]["goal_side_row_fraction"],
        ),
        "mean_action_norm": (
            observed_summary["controller_and_transition"]["mean_action_norm"],
            reference_summary["controller_and_transition"]["mean_action_norm"],
        ),
        "saturated_action_component_fraction": (
            observed_summary["controller_and_transition"][
                "saturated_action_component_fraction"
            ],
            reference_summary["controller_and_transition"][
                "saturated_action_component_fraction"
            ],
        ),
        "exact_action_repeat_fraction": (
            observed_summary["controller_and_transition"][
                "exact_consecutive_action_repeat_fraction"
            ],
            reference_summary["controller_and_transition"][
                "exact_consecutive_action_repeat_fraction"
            ],
        ),
        "collision_residual_fraction": (
            observed_summary["controller_and_transition"][
                "collision_or_boundary_residual_fraction"
            ],
            reference_summary["controller_and_transition"][
                "collision_or_boundary_residual_fraction"
            ],
        ),
        "mean_goal_progress_px": (
            observed_summary["controller_and_transition"][
                "mean_goal_progress_per_transition_px"
            ],
            reference_summary["controller_and_transition"][
                "mean_goal_progress_per_transition_px"
            ],
        ),
        "positive_net_progress_fraction": (
            observed_summary["episodes_with_positive_net_goal_progress_fraction"],
            reference_summary["episodes_with_positive_net_goal_progress_fraction"],
        ),
    }
    for name, (observed, expected) in absolute_metrics.items():
        gates[name] = _absolute_gate(
            observed,
            expected,
            float(gate_config[name]["maximum_absolute_difference"]),
        )
    observed_efficiency = observed_summary["episode_path"][
        "door_center_reference_efficiency"
    ]["median"]
    reference_efficiency = reference_summary["episode_path"][
        "door_center_reference_efficiency"
    ]["median"]
    relative_difference = float(
        abs(observed_efficiency - reference_efficiency) / reference_efficiency
    )
    maximum_relative = float(
        gate_config["median_path_efficiency"]["maximum_relative_difference"]
    )
    gates["median_path_efficiency"] = {
        "passed": relative_difference <= maximum_relative,
        "observed": observed_efficiency,
        "reference": reference_efficiency,
        "relative_difference": relative_difference,
        "maximum_relative_difference": maximum_relative,
    }

    train_pairs = np.round(
        np.concatenate(
            [train.state[train.offsets], train.goal[train.offsets]], axis=1
        ),
        5,
    )
    dev_pairs = np.round(
        np.concatenate(
            [dev.state[dev.offsets], dev.goal[dev.offsets]], axis=1
        ),
        5,
    )
    train_pair_set = {tuple(row) for row in train_pairs.tolist()}
    dev_pair_set = {tuple(row) for row in dev_pairs.tolist()}
    gates["synthetic_train_dev_geometry_disjoint"] = {
        "passed": not train_pair_set.intersection(dev_pair_set),
        "overlap": len(train_pair_set.intersection(dev_pair_set)),
    }
    gates["independent_train_geometries"] = {
        "passed": observed_geometry["model_visible_start_goal_pairs_5dp"] == 9000,
        "observed": observed_geometry["model_visible_start_goal_pairs_5dp"],
        "required": 9000,
    }

    passed = all(value["passed"] for value in gates.values())
    payload = {
        "schema_version": 1,
        "benchmark": f"{config['experiment']}_distribution",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "sources": {
            "config": portable_contextworld_path(
                Path(config["_config_path"]), repo_root=REPO_ROOT
            ),
            "catalog": portable_contextworld_path(catalog_path, repo_root=REPO_ROOT),
            "catalog_sha256": _sha256(catalog_path),
            "frozen_reference": portable_contextworld_path(
                reference_path, repo_root=REPO_ROOT
            ),
            "frozen_reference_sha256": _sha256(reference_path),
        },
        "selection_policy": {
            "success_only_filtering": False,
            "post_collection_episode_filtering": False,
            "all_collected_episodes_included": True,
        },
        "gates": gates,
        "reference_summary": reference_summary,
        "observed_train_summary": observed_summary,
        "observed_train_geometry": observed_geometry,
        "dev_summary": summarize_trace(dev),
        "dev_geometry": _episode_geometry(dev, dev_ids),
    }
    output = resolve_contextworld_path(
        config["output"]["distribution_report"], repo_root=REPO_ROOT
    )
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
    parser.add_argument("--freeze-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = config_path
    if args.freeze_reference:
        result = freeze_reference(config)
    else:
        result = evaluate_distribution(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not args.freeze_reference and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
