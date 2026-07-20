#!/usr/bin/env python3
"""Quantify why original TwoRoom ID is easy and SpeedTask E4 is not improved.

The report compares model-visible trajectory support, controller/collision
statistics, factor-cross reuse, E4 geometry support, and exact clip overlap in
the historical original-H5 ID protocol.  Pixel fidelity is covered by the
existing replay reports and is referenced rather than decoded again here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.paths import portable_contextworld_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from scripts.analyze_tworoom_trajectory_quality import (
    _e4_geometry_summary,
    _template_map,
    room_relation,
)


WALL_X = 112.0
DOOR_Y = 49.0
GRID = 14.0
HISTORY_SIZE = 3
NUM_STEPS = HISTORY_SIZE + 1
FRAMESKIP = 5
CLIP_SPAN = NUM_STEPS * FRAMESKIP


@dataclass(frozen=True)
class TraceTable:
    state: np.ndarray
    goal: np.ndarray
    action: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    speed: np.ndarray
    reset: np.ndarray
    reset_is_observed: bool
    scenario: np.ndarray


def _logical_artifact_path(value: str | Path) -> Path:
    return resolve_contextworld_path(value, repo_root=REPO_ROOT)


def load_original_h5(path: Path) -> TraceTable:
    import h5py

    with h5py.File(path, "r") as dataset:
        offsets = np.asarray(dataset["ep_offset"], dtype=np.int64)
        lengths = np.asarray(dataset["ep_len"], dtype=np.int64)
        ends = offsets + lengths - 1
        state = np.asarray(dataset["pos_agent"], dtype=np.float32)
        goal = np.asarray(dataset["pos_target"], dtype=np.float32)
        return TraceTable(
            state=state,
            goal=goal,
            action=np.asarray(dataset["action"], dtype=np.float32),
            offsets=offsets,
            lengths=lengths,
            terminated=np.asarray(dataset["terminated"][ends], dtype=bool),
            truncated=np.asarray(dataset["truncated"][ends], dtype=bool),
            speed=np.full(len(lengths), 5.0, dtype=np.float32),
            # The historical H5 does not retain pre-action reset state.  Its
            # first model-visible state is therefore the comparable geometry.
            reset=state[offsets],
            reset_is_observed=False,
            scenario=np.asarray(["original_h5"] * len(lengths), dtype=object),
        )


def load_synthetic_catalog(path: Path) -> TraceTable:
    import lance

    catalog = json.loads(path.read_text(encoding="utf-8"))
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
    for logical_path in catalog["train"]["synthetic"]:
        scenario_path = _logical_artifact_path(logical_path)
        values = lance.dataset(scenario_path).to_table(columns=columns).to_pydict()
        episode_idx = np.asarray(values["episode_idx"], dtype=np.int64)
        state = np.asarray(values["state"], dtype=np.float32)
        goal = np.asarray(values["goal_state"], dtype=np.float32)
        action = np.asarray(values["action"], dtype=np.float32)
        terminal = np.asarray(values["terminated"], dtype=np.float32)[:, 0]
        truncation = np.asarray(values["truncated"], dtype=np.float32)[:, 0]
        speed = np.asarray(
            values["variation_agent_speed"], dtype=np.float32
        )[:, 0]
        reset = np.asarray(values["variation_agent_position"], dtype=np.float32)
        states.append(state)
        goals.append(goal)
        actions.append(action)
        for episode in np.unique(episode_idx):
            rows = np.flatnonzero(episode_idx == episode)
            offsets.append(row_offset + int(rows[0]))
            lengths.append(len(rows))
            terminated.append(bool(terminal[rows[-1]] > 0.5))
            truncated.append(bool(truncation[rows[-1]] > 0.5))
            speeds.append(float(speed[rows[0]]))
            resets.append(reset[rows[0]])
            scenarios.append(scenario_path.stem)
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


def _entropy(counts: np.ndarray) -> tuple[float, float]:
    probability = counts.astype(np.float64) / counts.sum()
    entropy = float(-(probability * np.log(probability)).sum())
    maximum = math.log(len(counts)) if len(counts) > 1 else 0.0
    return entropy, entropy / maximum if maximum else 1.0


def _coverage(values: np.ndarray) -> dict[str, Any]:
    unique, counts = np.unique(values, axis=0, return_counts=True)
    entropy, normalized = _entropy(counts)
    descending = np.sort(counts)[::-1]
    return {
        "unique_cells": int(len(unique)),
        "entropy_nats": entropy,
        "normalized_entropy_over_observed_cells": normalized,
        "top_10_cell_row_fraction": float(descending[:10].sum() / counts.sum()),
        "maximum_rows_in_one_cell": int(descending[0]),
    }


def _selected_rows(trace: TraceTable, episode_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    episode_by_row = np.repeat(np.arange(len(trace.lengths)), trace.lengths)
    transition_episode = np.repeat(
        np.arange(len(trace.lengths)), np.maximum(trace.lengths - 1, 0)
    )
    return episode_by_row[episode_mask[episode_by_row]], transition_episode[
        episode_mask[transition_episode]
    ]


def summarize_trace(
    trace: TraceTable, episode_mask: np.ndarray | None = None
) -> dict[str, Any]:
    if episode_mask is None:
        episode_mask = np.ones(len(trace.lengths), dtype=bool)
    episode_indices = np.flatnonzero(episode_mask)
    if not len(episode_indices):
        return {"episodes": 0}

    episode_by_row = np.repeat(np.arange(len(trace.lengths)), trace.lengths)
    row_mask = episode_mask[episode_by_row]
    row_indices = np.flatnonzero(row_mask)
    transition_rows = np.concatenate(
        [
            np.arange(trace.offsets[index], trace.offsets[index] + trace.lengths[index] - 1)
            for index in episode_indices
        ]
    )
    transition_episode = np.repeat(
        episode_indices, trace.lengths[episode_indices] - 1
    )

    state = trace.state[row_indices]
    goal = trace.goal[row_indices]
    action = trace.action[transition_rows]
    current = trace.state[transition_rows]
    following = trace.state[transition_rows + 1]
    transition_goal = trace.goal[transition_rows]
    speed = trace.speed[transition_episode]
    movement = following - current
    movement_norm = np.linalg.norm(movement, axis=1)
    action_norm = np.linalg.norm(action, axis=1)
    distance_before = np.linalg.norm(current - transition_goal, axis=1)
    distance_after = np.linalg.norm(following - transition_goal, axis=1)
    progress = distance_before - distance_after
    expected = current + np.clip(action, -1.0, 1.0) * speed[:, None]
    collision_residual = np.linalg.norm(following - expected, axis=1)

    state_grid = np.floor(state / GRID).astype(np.int16)
    goal_grid = np.floor(goal / GRID).astype(np.int16)
    action_grid = np.floor((np.clip(action, -1.0, 1.0) + 1.0) / 0.125).astype(
        np.int16
    )

    repeat_pairs: list[np.ndarray] = []
    episode_metrics: dict[str, list[float]] = {
        "path_length": [],
        "net_goal_progress": [],
        "path_length_per_net_progress": [],
        "door_center_reference_efficiency": [],
        "goal_side_reached": [],
        "door_crossed": [],
        "first_crossing_step_fraction": [],
        "first_crossing_y_error": [],
    }
    for index in episode_indices:
        start = int(trace.offsets[index])
        length = int(trace.lengths[index])
        positions = trace.state[start : start + length]
        actions = trace.action[start : start + length]
        target = trace.goal[start]
        delta = np.diff(positions, axis=0)
        path_length = float(np.linalg.norm(delta, axis=1).sum())
        initial_distance = float(np.linalg.norm(positions[0] - target))
        final_distance = float(np.linalg.norm(positions[-1] - target))
        net_progress = initial_distance - final_distance
        repeat_pairs.append(np.linalg.norm(np.diff(actions, axis=0), axis=1))
        cross = room_relation(positions[:-1], positions[1:])
        cross_indices = np.flatnonzero(cross)
        goal_side = (positions[:, 0] < WALL_X) == (target[0] < WALL_X)
        episode_metrics["path_length"].append(path_length)
        episode_metrics["net_goal_progress"].append(net_progress)
        if net_progress > 1e-3:
            episode_metrics["path_length_per_net_progress"].append(
                path_length / net_progress
            )
        reference = (
            float(np.linalg.norm(positions[0] - np.asarray([WALL_X, DOOR_Y])))
            + float(np.linalg.norm(target - np.asarray([WALL_X, DOOR_Y])))
            if bool(room_relation(positions[:1], target[None])[0])
            else initial_distance
        )
        episode_metrics["door_center_reference_efficiency"].append(
            reference / max(path_length, 1e-6)
        )
        episode_metrics["goal_side_reached"].append(float(goal_side.any()))
        episode_metrics["door_crossed"].append(float(len(cross_indices) > 0))
        if len(cross_indices):
            cross_index = int(cross_indices[0])
            crossing_y = float(
                (positions[cross_index, 1] + positions[cross_index + 1, 1]) / 2.0
            )
            episode_metrics["first_crossing_step_fraction"].append(
                cross_index / max(length - 1, 1)
            )
            episode_metrics["first_crossing_y_error"].append(abs(crossing_y - DOOR_Y))

    repeats = np.concatenate(repeat_pairs)
    initial = trace.state[trace.offsets[episode_indices]]
    initial_goal = trace.goal[trace.offsets[episode_indices]]
    reset_goal_grid = np.concatenate(
        [
            np.floor(trace.reset[episode_indices] / GRID).astype(np.int16),
            np.floor(initial_goal / GRID).astype(np.int16),
        ],
        axis=1,
    )
    cross_room = room_relation(initial, initial_goal)
    clips = np.maximum(trace.lengths[episode_indices] - CLIP_SPAN + 1, 0)
    return {
        "episodes": int(len(episode_indices)),
        "rows": int(len(row_indices)),
        "transitions": int(len(transition_rows)),
        "model_clips_history3_frameskip5": int(clips.sum()),
        "termination_success_rate": float(trace.terminated[episode_indices].mean()),
        "truncation_rate": float(trace.truncated[episode_indices].mean()),
        "cross_room_fraction_at_first_observed_state": float(cross_room.mean()),
        "mean_episode_rows": float(trace.lengths[episode_indices].mean()),
        "geometry": {
            "reset_state_source": (
                "explicit_pre_action_variation" if trace.reset_is_observed else "first_recorded_post_action_state"
            ),
            "unique_reset_goal_pairs_5dp": int(
                len(
                    np.unique(
                        np.round(
                            np.concatenate(
                                [trace.reset[episode_indices], initial_goal], axis=1
                            ),
                            5,
                        ),
                        axis=0,
                    )
                )
            ),
            "reset_goal_grid_14px": _coverage(reset_goal_grid),
        },
        "model_visible_support": {
            "state_grid_14px": _coverage(state_grid),
            "state_goal_grid_14px": _coverage(
                np.concatenate([state_grid, goal_grid], axis=1)
            ),
            "action_grid_width_0p125": _coverage(action_grid),
            "door_corridor_row_fraction": float(
                ((np.abs(state[:, 0] - WALL_X) <= GRID) & (np.abs(state[:, 1] - DOOR_Y) <= GRID)).mean()
            ),
            "goal_side_row_fraction": float(
                ((state[:, 0] < WALL_X) == (goal[:, 0] < WALL_X)).mean()
            ),
        },
        "controller_and_transition": {
            "mean_action_norm": float(action_norm.mean()),
            "p90_action_norm": float(np.percentile(action_norm, 90)),
            "saturated_action_component_fraction": float((np.abs(action) >= 0.999).mean()),
            "exact_consecutive_action_repeat_fraction": float((repeats <= 1e-7).mean()),
            "mean_movement_px": float(movement_norm.mean()),
            "stationary_transition_fraction": float((movement_norm <= 1e-3).mean()),
            "stalled_despite_action_fraction": float(
                ((movement_norm < 0.25) & (action_norm > 0.5)).mean()
            ),
            "collision_or_boundary_residual_fraction": float((collision_residual > 1e-3).mean()),
            "mean_goal_progress_per_transition_px": float(progress.mean()),
            "negative_goal_progress_fraction": float((progress < -1e-6).mean()),
        },
        "episode_path": {
            key: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
            }
            for key, values in episode_metrics.items()
            if values
        },
        "episodes_with_positive_net_goal_progress_fraction": float(
            np.mean(
                [
                    np.linalg.norm(
                        trace.state[trace.offsets[index]]
                        - trace.goal[trace.offsets[index]]
                    )
                    > np.linalg.norm(
                        trace.state[
                            trace.offsets[index] + trace.lengths[index] - 1
                        ]
                        - trace.goal[trace.offsets[index]]
                    )
                    for index in episode_indices
                ]
            )
        ),
    }


def paired_speed_reuse(trace: TraceTable) -> dict[str, Any]:
    geometry = np.round(
        np.concatenate(
            [trace.reset, trace.goal[trace.offsets]], axis=1
        ),
        5,
    )
    _, inverse, counts = np.unique(geometry, axis=0, return_inverse=True, return_counts=True)
    first_actions = trace.action[trace.offsets]
    dispersions: list[float] = []
    rounded_unique: list[int] = []
    speed_counts: list[int] = []
    for group in range(int(inverse.max()) + 1):
        selected = inverse == group
        values = first_actions[selected]
        dispersions.append(float(np.sqrt(np.mean((values - values.mean(axis=0)) ** 2))))
        rounded_unique.append(len(np.unique(np.round(values, 2), axis=0)))
        speed_counts.append(len(np.unique(np.round(trace.speed[selected], 6))))
    overall = float(
        np.sqrt(np.mean((first_actions - first_actions.mean(axis=0)) ** 2))
    )
    return {
        "independent_reset_goal_geometries": int(len(counts)),
        "episodes_per_geometry_minimum": int(counts.min()),
        "episodes_per_geometry_maximum": int(counts.max()),
        "unique_speeds_per_geometry_minimum": int(min(speed_counts)),
        "unique_speeds_per_geometry_maximum": int(max(speed_counts)),
        "first_action_within_geometry_rms_deviation_mean": float(np.mean(dispersions)),
        "first_action_global_rms_deviation": overall,
        "within_to_global_first_action_dispersion_ratio": float(
            np.mean(dispersions) / overall if overall else 0.0
        ),
        "rounded_0p01_first_actions_per_geometry_median": float(np.median(rounded_unique)),
        "interpretation": (
            "The speed cross increases episode count without increasing independent reset-goal support. The reported first-action dispersion is high enough that this report does not claim policy-noise collapse."
        ),
    }


def template_support(
    trace: TraceTable, templates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    episode_by_row = np.repeat(np.arange(len(trace.lengths)), trace.lengths)
    first = trace.state[trace.offsets]
    first_goal = trace.goal[trace.offsets]
    first_joint = np.concatenate([first, first_goal], axis=1).astype(np.float64)
    visited_joint = np.concatenate([trace.state, trace.goal], axis=1).astype(np.float64)
    visited_grid = np.floor(visited_joint / GRID).astype(np.int16)
    output: dict[str, Any] = {}
    for template_id, template in templates.items():
        query = np.asarray(
            template["reset_state"] + template["goal_state"], dtype=np.float64
        )
        query_grid = np.floor(query / GRID).astype(np.int16)
        first_distance = np.linalg.norm(first_joint - query, axis=1)
        visited_distance = np.linalg.norm(visited_joint - query, axis=1)
        exact_rows = np.all(visited_grid == query_grid, axis=1)
        output[template_id] = {
            "room_relation": template["room_relation"],
            "nearest_episode_initial_joint_distance_px": float(first_distance.min()),
            "nearest_visited_state_goal_joint_distance_px": float(visited_distance.min()),
            "rows_in_exact_14px_state_goal_cell": int(exact_rows.sum()),
            "episodes_in_exact_14px_state_goal_cell": int(
                len(np.unique(episode_by_row[exact_rows]))
            ),
            "rows_within_20px_joint_distance": int((visited_distance <= 20.0).sum()),
        }
    return output


def historical_id_clip_overlap(
    original_h5: Path,
    *,
    split_seed: int = 3072,
    eval_seeds: tuple[int, ...] = (42, 43, 44, 45, 46, 47),
    num_eval: int = 50,
    goal_offset: int = 25,
) -> dict[str, Any]:
    import h5py
    import torch

    with h5py.File(original_h5, "r") as dataset:
        lengths = np.asarray(dataset["ep_len"], dtype=np.int64)
        offsets = np.asarray(dataset["ep_offset"], dtype=np.int64)
        ep_idx = np.asarray(dataset["ep_idx"], dtype=np.int64)
        step_idx = np.asarray(dataset["step_idx"], dtype=np.int64)

    clip_lengths = np.maximum(lengths - CLIP_SPAN + 1, 0)
    clip_offsets = np.concatenate([[0], np.cumsum(clip_lengths[:-1])])
    total_clips = int(clip_lengths.sum())
    train_clips = int(math.floor(total_clips * 0.9))
    if train_clips + int(math.floor(total_clips * 0.1)) < total_clips:
        train_clips += 1
    permutation = torch.randperm(
        total_clips, generator=torch.Generator().manual_seed(split_seed)
    ).numpy()
    selected = np.zeros(total_clips, dtype=bool)
    selected[permutation[:train_clips]] = True

    max_start = lengths - goal_offset - 1
    valid = step_idx <= max_start[ep_idx]
    valid_indices = np.flatnonzero(valid)
    sampled_rows: list[int] = []
    seed_rows: dict[str, int] = {}
    for seed in eval_seeds:
        rng = np.random.default_rng(seed)
        # Match pinned StableWM eval_wm.py, including exclusion of the final
        # valid-array position through choice(len(valid_indices) - 1).
        positions = rng.choice(len(valid_indices) - 1, size=num_eval, replace=False)
        rows = np.sort(valid_indices[positions])
        sampled_rows.extend(rows.tolist())
        seed_rows[str(seed)] = len(rows)

    exact_clip = 0
    query_state_seen = 0
    goal_state_seen = 0
    local_rows_seen: list[float] = []
    unique_samples: set[tuple[int, int]] = set()
    for row in sampled_rows:
        episode = int(ep_idx[row])
        step = int(step_idx[row])
        unique_samples.add((episode, step))
        base = int(clip_offsets[episode])
        if selected[base + step]:
            exact_clip += 1

        def state_is_exposed(target_step: int) -> bool:
            for frame in (0, 5, 10, 15):
                start = target_step - frame
                if 0 <= start < clip_lengths[episode] and selected[base + start]:
                    return True
            return False

        query_state_seen += state_is_exposed(step)
        goal_state_seen += state_is_exposed(step + goal_offset)
        local_rows_seen.append(
            float(
                np.mean(
                    [state_is_exposed(value) for value in range(step, step + goal_offset + 1)]
                )
            )
        )

    total = len(sampled_rows)
    return {
        "protocol": {
            "dataset": str(original_h5),
            "training_split": "random clip-level 90/10 split",
            "training_split_seed": split_seed,
            "frameskip": FRAMESKIP,
            "num_steps": NUM_STEPS,
            "clip_span_rows": CLIP_SPAN,
            "eval_seeds": list(eval_seeds),
            "num_eval_per_seed": num_eval,
            "goal_offset_rows": goal_offset,
            "eval_sampling": "random valid rows from the same H5",
        },
        "source_clips": total_clips,
        "training_clips": train_clips,
        "sampled_queries": total,
        "unique_episode_step_queries": len(unique_samples),
        "exact_query_start_clip_in_training": exact_clip,
        "exact_query_start_clip_fraction": exact_clip / total,
        "query_state_present_in_any_training_clip": query_state_seen,
        "query_state_present_fraction": query_state_seen / total,
        "goal_state_present_in_any_training_clip": goal_state_seen,
        "goal_state_present_fraction": goal_state_seen / total,
        "mean_fraction_of_query_to_goal_rows_present_as_training_frames": float(
            np.mean(local_rows_seen)
        ),
        "interpretation": (
            "The historical ID score measures planning on states sampled from the same trajectories used for clip-level training. It is an in-distribution retention check, not an episode-held-out generalization estimate."
        ),
    }


def _portable(path: Path) -> str:
    return portable_contextworld_path(path, repo_root=REPO_ROOT)


def run(args: argparse.Namespace) -> dict[str, Any]:
    original_path = resolve_contextworld_path(args.original_h5, repo_root=REPO_ROOT)
    speedtask_catalog = resolve_contextworld_path(
        args.speedtask_catalog, repo_root=REPO_ROOT
    )
    query_catalog = resolve_contextworld_path(args.query_catalog, repo_root=REPO_ROOT)
    e4_orig = resolve_contextworld_path(args.e4_orig, repo_root=REPO_ROOT)
    e4_speedseen = resolve_contextworld_path(args.e4_speedseen, repo_root=REPO_ROOT)
    e4_speedtask = resolve_contextworld_path(args.e4_speedtask, repo_root=REPO_ROOT)
    paired_result = resolve_contextworld_path(args.paired_result, repo_root=REPO_ROOT)
    training_report_path = resolve_contextworld_path(
        args.training_report, repo_root=REPO_ROOT
    )
    output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)

    original = load_original_h5(original_path)
    speedtask = load_synthetic_catalog(speedtask_catalog)
    templates = _template_map(query_catalog)
    speed5 = np.isclose(speedtask.speed, 5.0)
    cross5 = speed5 & room_relation(
        speedtask.state[speedtask.offsets], speedtask.goal[speedtask.offsets]
    )
    paired = json.loads(paired_result.read_text(encoding="utf-8"))
    speedtask_e4 = json.loads(e4_speedtask.read_text(encoding="utf-8"))
    training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
    logical_epochs = int(training_report["training"]["plan"]["logical_epochs"])
    new_original_draws = (
        int(training_report["data"]["epoch_group_counts"]["original"])
        * logical_epochs
    )
    new_synthetic_draws = (
        int(training_report["data"]["epoch_group_counts"]["speed"])
        * logical_epochs
    )
    baseline_total_draws = int(training_report["training"]["global_step"]) * int(
        training_report["training"]["plan"]["global_batch_size"]
    )

    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_training_data_gap_v1",
        "status": "passed",
        "sources": {
            "original_h5": str(original_path),
            "speedtask_catalog": _portable(speedtask_catalog),
            "query_catalog": _portable(query_catalog),
            "e4": {
                "H3-Orig": _portable(e4_orig),
                "H3-SpeedSeen": _portable(e4_speedseen),
                "H3-SpeedTask": _portable(e4_speedtask),
            },
            "paired_speedtask_vs_speedseen": _portable(paired_result),
            "speedtask_training_report": _portable(training_report_path),
        },
        "datasets": {
            "original_h5": summarize_trace(original),
            "speedtask_train": summarize_trace(speedtask),
            "speedtask_train_speed5_cross_room": summarize_trace(speedtask, cross5),
        },
        "speedtask_factor_cross_reuse": paired_speed_reuse(speedtask),
        "e4_training_support": {
            "original_h5": template_support(original, templates),
            "speedtask_train": template_support(speedtask, templates),
        },
        "e4_outcome_geometry": {
            "templates": templates,
            "models": {
                "H3-Orig": _e4_geometry_summary(e4_orig, templates),
                "H3-SpeedSeen": _e4_geometry_summary(e4_speedseen, templates),
                "H3-SpeedTask": _e4_geometry_summary(e4_speedtask, templates),
            },
        },
        "historical_original_id_clip_overlap": historical_id_clip_overlap(
            original_path
        ),
        "training_exposure": {
            "H3_Orig": {
                "original_draws": baseline_total_draws,
                "synthetic_draws": 0,
                "mean_original_draws_per_train_clip": baseline_total_draws
                / int(training_report["data"]["groups"]["original"]["train_clips"]),
            },
            "H3_SpeedTask": {
                "original_draws": new_original_draws,
                "synthetic_draws": new_synthetic_draws,
                "mean_original_draws_per_train_clip": new_original_draws
                / int(training_report["data"]["groups"]["original"]["train_clips"]),
                "mean_synthetic_draws_per_raw_clip": new_synthetic_draws
                / int(training_report["data"]["groups"]["speed"]["train_clips_raw"]),
            },
            "fixed_total_draws": baseline_total_draws,
            "speedtask_original_exposure_fraction_vs_h3_orig": new_original_draws
            / baseline_total_draws,
            "interpretation": (
                "The fixed total optimizer/sample budget makes the 50/50 mixture replace half of the original-H5 exposure; it does not add synthetic data on top of the full original exposure."
            ),
        },
        "observed_model_result": {
            "speedtask_e4_correct_successes": speedtask_e4["aggregate"]["correct"]["successes"],
            "speedtask_e4_wrong_successes": speedtask_e4["aggregate"]["wrong"]["successes"],
            "evaluations_per_condition": speedtask_e4["aggregate"]["evaluations_per_condition"],
            "correct_only": speedtask_e4["aggregate"]["correct_only_successes"],
            "wrong_only": speedtask_e4["aggregate"]["wrong_only_successes"],
            "speedtask_minus_speedseen_correct_mean_final_distance_px": paired[
                "conditions"
            ]["correct"]["candidate_minus_reference_mean_final_distance"],
            "speedtask_minus_speedseen_wrong_mean_final_distance_px": paired[
                "conditions"
            ]["wrong"]["candidate_minus_reference_mean_final_distance"],
            "common_failure_correct_delta_px": paired["conditions"]["correct"][
                "by_paired_success_outcome"
            ]["both_failure"][
                "candidate_minus_reference_mean_final_distance"
            ],
            "common_failure_wrong_delta_px": paired["conditions"]["wrong"][
                "by_paired_success_outcome"
            ]["both_failure"]["candidate_minus_reference_mean_final_distance"],
        },
        "conclusions": {
            "supported": [
                "SpeedTask task matching and exact speed support did not create planning-level context use under the fixed H3 recipe.",
                "The original H5 contributes far more independent geometry and model-visible state-goal support than the synthetic speed cross.",
                "The synthetic episode count overstates independent support because each reset-goal geometry is reused across all 32 speeds.",
                "Historical original-ID evaluation samples rows from the same H5 that was split at clip level for training, so its high score is retention rather than held-out evidence.",
                "E4 binary successes are concentrated in same-room geometry while the intended original task and SpeedTask episode resets are cross-room, limiting the sensitivity of pooled success to the repaired training distribution.",
            ],
            "not_supported": [
                "A weaker ExpertPolicy is the primary cause; the matched speed-5 cross-room outcome statistics are similar.",
                "The speed cross collapses policy-action diversity; observed first-action dispersion is substantial.",
                "More optimizer steps alone will fix context use.",
                "Original-ID success demonstrates OOD speed adaptation.",
            ],
            "next_control": (
                "Before another full model run, build an episode-held-out original-ID split and a training set that increases independent reset-goal trajectories while matching E4 template and room-relation strata. Keep speed exposure balanced, but do not obtain diversity mainly by crossing the same geometry with more speeds."
            ),
        },
    }
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-h5",
        type=Path,
        default=Path("../../data/world_model/quentinll/lewm-tworooms/tworoom.h5"),
    )
    parser.add_argument(
        "--speedtask-catalog",
        type=Path,
        default=Path("artifacts/synthesis/catalogs/tworoom_speed_task_v1.json"),
    )
    parser.add_argument(
        "--query-catalog",
        type=Path,
        default=Path(
            "artifacts/evaluation/icl/tworoom_icl_v1_validation_context_query_catalog.json"
        ),
    )
    parser.add_argument(
        "--e4-orig",
        type=Path,
        default=Path("artifacts/evaluation/history3/e4_speed_ctx_n50x6.json"),
    )
    parser.add_argument(
        "--e4-speedseen",
        type=Path,
        default=Path(
            "artifacts/evaluation/history3/h3_speedseen_s3072/e4_speed_ctx_n50x6.json"
        ),
    )
    parser.add_argument(
        "--e4-speedtask",
        type=Path,
        default=Path(
            "artifacts/evaluation/history3/h3_speedtask_s3072/e4_speed_ctx_n50x6.json"
        ),
    )
    parser.add_argument(
        "--paired-result",
        type=Path,
        default=Path(
            "artifacts/evaluation/history3/h3_speedtask_s3072/e4_vs_speedseen_paired.json"
        ),
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=Path("artifacts/training/reports/h3_speedtask_s3072.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/evaluation/history3/h3_speedtask_s3072/training_data_gap_v1.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "benchmark": result["benchmark"],
                "status": result["status"],
                "output": result["sources"],
            },
            sort_keys=True,
        )
    )
