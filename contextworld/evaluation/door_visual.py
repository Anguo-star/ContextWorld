from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


ACTION_BLOCK = 5
HORIZONS = (1, 2, 3, 5)
INPUT_CONDITIONS = ("query_only", "natural_history3")
TASKS = ("doorway_passage", "wall_contact")
VALIDATION_TRACKS = ("validation_seen", "validation_interpolation")
DIRECTIONS = ("left_to_right", "right_to_left")


@dataclass(frozen=True)
class DoorQueryGeometry:
    """One deterministic visual-door prediction query.

    ``query_state`` is the third History-3 frame and the only visible frame in
    the query-only condition.  The target is deliberately kept far from the
    scripted prediction trajectory so data generation cannot terminate early.
    """

    template_id: str
    door_position: int
    task: str
    direction: str
    query_state: tuple[float, float]
    target_state: tuple[float, float]
    history_vertical_sign: float


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def validation_track_rows(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tracks = config["evaluation_data"]["tracks"]
    selected = {
        name: dict(tracks[name])
        for name in VALIDATION_TRACKS
        if name in tracks
    }
    if tuple(selected) != VALIDATION_TRACKS:
        raise ValueError(
            f"Expected validation tracks {VALIDATION_TRACKS}, got {tuple(selected)}"
        )
    if any(row["split"] != "validation" for row in selected.values()):
        raise ValueError("A non-validation track entered the validation builder")
    return selected


def door_support_audit(config: dict[str, Any]) -> dict[str, Any]:
    training = set(
        map(int, config["training_data"]["multi_door_target"]["door_positions"])
    )
    tracks = validation_track_rows(config)
    seen = set(map(int, tracks["validation_seen"]["door_positions"]))
    interpolation = set(
        map(int, tracks["validation_interpolation"]["door_positions"])
    )
    checks = {
        "seen_is_training_subset": seen <= training,
        "interpolation_is_training_disjoint": interpolation.isdisjoint(training),
        "interpolation_is_inside_training_range": bool(
            interpolation
            and min(training) < min(interpolation)
            and max(interpolation) < max(training)
        ),
        "validation_tracks_are_disjoint": seen.isdisjoint(interpolation),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "multi_door_training_positions": sorted(training),
        "validation_seen_positions": sorted(seen),
        "validation_interpolation_positions": sorted(interpolation),
    }


def make_query_geometry(
    *,
    door_position: int,
    task: str,
    direction: str,
    template_index: int,
    seed: int,
) -> DoorQueryGeometry:
    if task not in TASKS:
        raise ValueError(f"Unknown prediction task: {task}")
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown passage direction: {direction}")
    sequence = np.random.SeedSequence(
        [
            int(seed),
            int(door_position),
            int(template_index),
            TASKS.index(task),
            DIRECTIONS.index(direction),
        ]
    )
    rng = np.random.default_rng(sequence)
    left_x = float(rng.uniform(88.0, 94.0))
    x = left_x if direction == "left_to_right" else 224.0 - left_x
    if task == "doorway_passage":
        passage_low = max(21.0, float(door_position) - 5.0)
        passage_high = min(203.0, float(door_position) + 5.0)
        y = float(rng.uniform(passage_low, passage_high))
    else:
        toward_interior = 1.0 if int(door_position) < 112 else -1.0
        y = float(
            door_position + toward_interior * rng.uniform(22.0, 42.0)
        )
    if not (21.0 <= x <= 203.0 and 21.0 <= y <= 203.0):
        raise RuntimeError(f"Unsafe generated query state: {(x, y)}")
    history_sign = 1.0 if y <= 112.0 else -1.0
    target_y = 196.0 if y <= 112.0 else 28.0
    target_x = 198.0 if direction == "left_to_right" else 26.0
    return DoorQueryGeometry(
        template_id=(
            f"door{int(door_position):03d}-{task}-{direction}-"
            f"{int(template_index):03d}"
        ),
        door_position=int(door_position),
        task=task,
        direction=direction,
        query_state=(x, y),
        target_state=(target_x, target_y),
        history_vertical_sign=history_sign,
    )


def natural_history_actions(
    geometry: DoorQueryGeometry, *, magnitude: float = 0.25
) -> np.ndarray:
    direction = np.asarray(
        [0.0, float(geometry.history_vertical_sign)], dtype=np.float32
    )
    outward = np.repeat(
        (np.float32(magnitude) * direction)[None], ACTION_BLOCK, axis=0
    )
    return np.stack([outward, -outward]).astype(np.float32)


def future_actions(
    direction: str = "left_to_right", *, magnitude: float = 0.5
) -> np.ndarray:
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown passage direction: {direction}")
    horizontal_sign = 1.0 if direction == "left_to_right" else -1.0
    block = np.repeat(
        np.asarray(
            [[horizontal_sign * float(magnitude), 0.0]], dtype=np.float32
        ),
        ACTION_BLOCK,
        axis=0,
    )
    return np.repeat(block[None], 5, axis=0).astype(np.float32)


def assign_eval_partitions(
    bundles: list[dict[str, Any]],
    *,
    eval_seeds: list[int],
    per_seed: int,
    assignment_seed: int,
) -> None:
    """Assign six disjoint 50-query partitions in every door/task cell.

    A template index receives the same partition in all cells.  This keeps the
    validation matrix paired without reusing a query between seed partitions.
    """

    expected = len(eval_seeds) * int(per_seed)
    by_cell: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for bundle in bundles:
        by_cell[(int(bundle["door_position"]), str(bundle["task"]))].append(bundle)
    assignments = formal_template_assignments(
        eval_seeds=eval_seeds,
        per_seed=per_seed,
        assignment_seed=assignment_seed,
    )
    for cell, rows in by_cell.items():
        if len(rows) != expected:
            raise RuntimeError(f"Cell {cell}: expected {expected} queries, got {len(rows)}")
        observed_templates = {int(row["template_index"]) for row in rows}
        if observed_templates != set(range(expected)):
            raise RuntimeError(f"Cell {cell} does not contain the frozen template set")
        for row in rows:
            expected_seed, expected_index, expected_direction = assignments[
                int(row["template_index"])
            ]
            if str(row.get("direction")) != expected_direction:
                raise RuntimeError(
                    f"Direction assignment changed for template "
                    f"{row['template_index']}: {row.get('direction')} != "
                    f"{expected_direction}"
                )
            row["eval_seed"] = expected_seed
            row["evaluation_index"] = expected_index
        for eval_seed in eval_seeds:
            counts = {
                direction: sum(
                    row["eval_seed"] == eval_seed
                    and row["direction"] == direction
                    for row in rows
                )
                for direction in DIRECTIONS
            }
            if set(counts.values()) != {int(per_seed) // 2}:
                raise RuntimeError(
                    f"Cell {cell}/seed {eval_seed} direction imbalance: {counts}"
                )


def formal_template_assignments(
    *, eval_seeds: list[int], per_seed: int, assignment_seed: int
) -> dict[int, tuple[int, int, str]]:
    if int(per_seed) % 2:
        raise ValueError("Direction-balanced per-seed count must be even")
    expected = len(eval_seeds) * int(per_seed)
    rng = np.random.default_rng(int(assignment_seed))
    permutation = rng.permutation(expected)
    assignments = {}
    for seed_index, eval_seed in enumerate(eval_seeds):
        selected = permutation[
            seed_index * int(per_seed) : (seed_index + 1) * int(per_seed)
        ]
        for evaluation_index, template_index in enumerate(selected):
            direction = (
                DIRECTIONS[0]
                if evaluation_index < int(per_seed) // 2
                else DIRECTIONS[1]
            )
            assignments[int(template_index)] = (
                int(eval_seed),
                int(evaluation_index),
                direction,
            )
    return assignments


def checkpoint_cell_summary(
    records: list[dict[str, Any]], *, eval_seeds: Iterable[int]
) -> dict[str, Any]:
    """Summarize one checkpoint without comparing its native latent scale.

    Raw MSE is retained only inside each returned checkpoint summary.  The
    normalized error is the prediction MSE divided by that checkpoint's own
    unchanged-query-frame baseline MSE.
    """

    seeds = tuple(map(int, eval_seeds))
    grouped: dict[tuple[str, str, str, int, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in records:
        for horizon in HORIZONS:
            grouped[
                (
                    str(row["track"]),
                    str(row["task"]),
                    str(row["input_condition"]),
                    int(horizon),
                    int(row["door_position"]),
                    int(row["eval_seed"]),
                )
            ].append(row)
    cells: dict[str, Any] = {}
    for key, rows in sorted(grouped.items()):
        track, task, condition, horizon, door, eval_seed = key
        normalized = np.asarray(
            [row["normalized_error_by_horizon"][str(horizon)] for row in rows],
            dtype=np.float64,
        )
        raw = np.asarray(
            [row["latent_mse_by_horizon"][str(horizon)] for row in rows],
            dtype=np.float64,
        )
        baseline = np.asarray(
            [row["unchanged_baseline_mse_by_horizon"][str(horizon)] for row in rows],
            dtype=np.float64,
        )
        cell_id = (
            f"{track}/{task}/{condition}/h{horizon}/door{door}/seed{eval_seed}"
        )
        cells[cell_id] = {
            "track": track,
            "task": task,
            "input_condition": condition,
            "horizon": horizon,
            "door_position": door,
            "eval_seed": eval_seed,
            "queries": len(rows),
            "native_latent_mse": float(raw.mean()),
            "unchanged_baseline_mse": float(baseline.mean()),
            "mean_normalized_error": float(normalized.mean()),
            "beats_unchanged_baseline_rate": float(np.mean(normalized < 1.0)),
        }
    expected_seed_values = {
        int(row["eval_seed"]) for row in records
    }
    return {
        "cells": cells,
        "eval_seeds": list(seeds),
        "observed_eval_seeds": sorted(expected_seed_values),
        "raw_latent_mse_cross_checkpoint_comparison_allowed": False,
    }


def paired_normalized_effects(
    control_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Pair target and control by the exact frozen query and input condition."""

    def index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        result = {}
        for row in rows:
            key = (str(row["query_id"]), str(row["input_condition"]))
            if key in result:
                raise RuntimeError(f"Duplicate model record: {key}")
            result[key] = row
        return result

    control = index(control_records)
    target = index(target_records)
    if set(control) != set(target):
        raise RuntimeError("Target/control result queries are not exactly paired")
    effects = []
    for key in sorted(control):
        left, right = control[key], target[key]
        identity_fields = (
            "track",
            "task",
            "direction",
            "door_position",
            "eval_seed",
            "evaluation_index",
            "static_query_id",
        )
        if any(left[field] != right[field] for field in identity_fields):
            raise RuntimeError(f"Paired result metadata changed: {key}")
        for horizon in HORIZONS:
            control_error = float(
                left["normalized_error_by_horizon"][str(horizon)]
            )
            target_error = float(
                right["normalized_error_by_horizon"][str(horizon)]
            )
            effects.append(
                {
                    "query_id": key[0],
                    "static_query_id": str(left["static_query_id"]),
                    "track": str(left["track"]),
                    "task": str(left["task"]),
                    "direction": str(left["direction"]),
                    "input_condition": key[1],
                    "door_position": int(left["door_position"]),
                    "eval_seed": int(left["eval_seed"]),
                    "horizon": int(horizon),
                    "control_normalized_error": control_error,
                    "target_normalized_error": target_error,
                    "control_minus_target": control_error - target_error,
                    "target_win": target_error < control_error,
                }
            )
    by_cell: dict[tuple[str, str, str, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in effects:
        by_cell[
            (
                row["track"],
                row["task"],
                row["input_condition"],
                row["horizon"],
                row["door_position"],
            )
        ].append(row)
    rng = np.random.default_rng(int(bootstrap_seed))
    cells = {}
    for key, rows in sorted(by_cell.items()):
        track, task, condition, horizon, door = key
        seed_rows = {}
        for eval_seed in sorted({int(row["eval_seed"]) for row in rows}):
            selected = [row for row in rows if int(row["eval_seed"]) == eval_seed]
            seed_rows[str(eval_seed)] = {
                "queries": len(selected),
                "mean_control_minus_target": float(
                    np.mean([row["control_minus_target"] for row in selected])
                ),
                "paired_query_win_rate": float(
                    np.mean([row["target_win"] for row in selected])
                ),
            }
        values = np.asarray(
            [row["control_minus_target"] for row in rows], dtype=np.float64
        )
        indices = rng.integers(
            0, len(values), size=(int(bootstrap_samples), len(values))
        )
        boot = values[indices].mean(axis=1)
        cell_id = f"{track}/{task}/{condition}/h{horizon}/door{door}"
        cells[cell_id] = {
            "track": track,
            "task": task,
            "input_condition": condition,
            "horizon": horizon,
            "door_position": door,
            "queries": len(rows),
            "mean_control_minus_target": float(values.mean()),
            "paired_query_win_rate": float(
                np.mean([row["target_win"] for row in rows])
            ),
            "all_eval_seed_directions_positive": all(
                row["mean_control_minus_target"] > 0 for row in seed_rows.values()
            ),
            "by_eval_seed": seed_rows,
            "paired_bootstrap_ci": {
                "samples": int(bootstrap_samples),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
            },
        }
    return {"effects": effects, "cells": cells}


def longest_contiguous_horizon(passes: dict[str, bool]) -> int:
    longest = 0
    for horizon in HORIZONS:
        if not passes.get(str(horizon), False):
            break
        longest = horizon
    return longest
