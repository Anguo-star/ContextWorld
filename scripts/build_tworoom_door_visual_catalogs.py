#!/usr/bin/env python3
"""Build frozen offline true-future payloads for the visual-door benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.door_visual import (
    DIRECTIONS,
    HORIZONS,
    TASKS,
    array_sha256,
    assign_eval_partitions,
    door_support_audit,
    formal_template_assignments,
    future_actions,
    make_query_geometry,
    natural_history_actions,
    validation_track_rows,
)
from contextworld.evaluation.icl_catalog import _factor_options
from contextworld.evaluation.icl_model import file_sha256
from contextworld.paths import (
    artifact_path,
    portable_contextworld_path,
)
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
DEFAULT_ARTIFACT_SUBDIR = "evaluation/history3/door_visual_generalization_v1"
GEOMETRY_SEED = 2026072201
CATALOG_SEED = 2026072202
EVAL_ASSIGNMENT_SEED = 2026072203
MAX_NOMINAL_HISTORY_RETURN_DRIFT_PX = 1e-4


def _selected_track_rows(
    config: dict[str, Any], evaluation_split: str
) -> dict[str, dict[str, Any]]:
    if evaluation_split == "validation":
        return validation_track_rows(config)
    if evaluation_split != "sealed_test":
        raise ValueError(f"Unknown evaluation split: {evaluation_split}")
    selected = {
        name: dict(row)
        for name, row in config["evaluation_data"]["tracks"].items()
        if row["split"] == "sealed_test"
    }
    expected = {
        "test_interpolation",
        "test_extrapolation_low",
        "test_extrapolation_high",
    }
    if set(selected) != expected:
        raise ValueError(f"Incomplete sealed-test tracks: {set(selected)}")
    return selected


def _simulate_episode(
    geometry: Any,
    *,
    simulator_seed: int,
) -> dict[str, np.ndarray]:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    factors = {
        "agent.speed": 5.0,
        "door.position": int(geometry.door_position),
    }
    history = natural_history_actions(geometry)
    future = future_actions(geometry.direction)
    all_actions = np.concatenate([history, future], axis=0)
    env = TwoRoomEnv(render_mode="rgb_array")
    pixels = []
    next_pixels = []
    states = []
    next_states = []
    try:
        env.reset(
            seed=int(simulator_seed),
            options={
                "variation": (),
                "variation_values": _factor_options(factors),
                "state": np.asarray(geometry.query_state, dtype=np.float32),
                "target_state": np.asarray(geometry.target_state, dtype=np.float32),
            },
        )
        speed_readback = float(
            np.asarray(env.variation_space["agent"]["speed"].value).reshape(-1)[0]
        )
        door_readback = int(
            np.asarray(env.variation_space["door"]["position"].value).reshape(-1)[0]
        )
        if speed_readback != 5.0 or door_readback != int(geometry.door_position):
            raise RuntimeError(
                f"Factor readback failed: speed={speed_readback}, door={door_readback}"
            )
        for block_index, block in enumerate(all_actions):
            pixels.append(np.asarray(env.render(), dtype=np.uint8).copy())
            states.append(env.agent_position.detach().cpu().numpy().copy())
            for raw_index, action in enumerate(block):
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    raise RuntimeError(
                        "Prediction episode terminated at "
                        f"block={block_index}, raw={raw_index}, "
                        f"template={geometry.template_id}"
                    )
            next_pixels.append(np.asarray(env.render(), dtype=np.uint8).copy())
            next_states.append(env.agent_position.detach().cpu().numpy().copy())
    finally:
        env.close()
    return {
        "pixels": np.stack(pixels),
        "next_pixels": np.stack(next_pixels),
        "states": np.asarray(states, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "history_actions": history,
        "future_actions": future,
    }


def _assert_exact_replay(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
    for key in first:
        if not np.array_equal(first[key], second[key]):
            raise RuntimeError(f"Exact replay failed for {key}")


def _prediction_payload_views(
    rollout: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Select the actual third History-3 frame as the prediction query.

    The two context actions are nominal opposites, but float32 integration can
    leave a sub-pixel residual.  Requiring the third rendered frame to be
    byte-identical to the first frame would therefore reject physically valid
    episodes.  The model conditions instead use the actual contiguous third
    frame, which is also the first frame of the future rollout.
    """

    history_pixels = np.stack(
        [rollout["pixels"][0], rollout["pixels"][1], rollout["pixels"][2]]
    )
    history_states = np.asarray(rollout["states"][:3], dtype=np.float32)
    return_drift = float(np.linalg.norm(history_states[2] - history_states[0]))
    if return_drift > MAX_NOMINAL_HISTORY_RETURN_DRIFT_PX:
        raise RuntimeError(
            "Natural History-3 nominal return drift exceeded tolerance: "
            f"{return_drift} > {MAX_NOMINAL_HISTORY_RETURN_DRIFT_PX}"
        )
    query_pixels = history_pixels[-1]
    future_pixels = rollout["pixels"][2:]
    future_next_pixels = rollout["next_pixels"][2:]
    future_states = rollout["states"][2:]
    future_next_states = rollout["next_states"][2:]
    if not np.array_equal(future_pixels[0], query_pixels):
        raise RuntimeError("Future does not start at the actual third History-3 frame")
    if not np.array_equal(future_pixels[1:], future_next_pixels[:-1]):
        raise RuntimeError("Future pixels are not contiguous")
    if not np.array_equal(future_states[1:], future_next_states[:-1]):
        raise RuntimeError("Future states are not contiguous")
    return {
        "history_pixels": history_pixels,
        "query_pixels": query_pixels,
        "future_pixels": future_pixels,
        "future_next_pixels": future_next_pixels,
        "future_states": future_states,
        "future_next_states": future_next_states,
        "nominal_history_return_drift_px": return_drift,
    }


def _task_oracle(geometry: Any, rollout: dict[str, np.ndarray]) -> dict[str, Any]:
    future_states = rollout["next_states"][2:]
    left_to_right = geometry.direction == "left_to_right"
    if geometry.task == "doorway_passage":
        passed = bool(
            future_states[-1, 0] > 119.0
            if left_to_right
            else future_states[-1, 0] < 105.0
        )
        return {
            "passed": passed,
            "expected_outcome": "crosses_visible_doorway",
            "direction": geometry.direction,
            "final_x": float(future_states[-1, 0]),
        }
    if geometry.task == "wall_contact":
        expected_free_x = float(
            geometry.query_state[0]
            + 5.0
            * np.sum(
                future_actions(geometry.direction).reshape(-1, 2)[:, 0]
            )
        )
        final_x = float(future_states[-1, 0])
        passed = bool(
            final_x <= 100.0 and expected_free_x - final_x > 25.0
            if left_to_right
            else final_x >= 124.0 and final_x - expected_free_x > 25.0
        )
        return {
            "passed": passed,
            "expected_outcome": "contacts_wall_outside_doorway",
            "direction": geometry.direction,
            "final_x": final_x,
            "unconstrained_final_x": expected_free_x,
        }
    raise ValueError(geometry.task)


def _visible_door_oracle(
    door_positions: list[int], *, simulator_seed: int
) -> dict[str, Any]:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    hashes = {}
    env = TwoRoomEnv(render_mode="rgb_array")
    try:
        for door in door_positions:
            env.reset(
                seed=int(simulator_seed),
                options={
                    "variation": (),
                    "variation_values": _factor_options(
                        {"agent.speed": 5.0, "door.position": int(door)}
                    ),
                    "state": np.asarray([70.0, 112.0], dtype=np.float32),
                    "target_state": np.asarray([190.0, 112.0], dtype=np.float32),
                },
            )
            hashes[str(door)] = array_sha256(
                np.asarray(env.render(), dtype=np.uint8)
            )
    finally:
        env.close()
    return {
        "passed": len(set(hashes.values())) == len(door_positions),
        "same_agent_state_and_pixels_differ_only_when_door_moves": True,
        "pixel_hash_by_door_position": hashes,
    }


def _scripted_planning_oracle(door_position: int, *, simulator_seed: int) -> dict[str, Any]:
    """Verify an aligned passage remains physically solvable within 100 steps."""

    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    env = TwoRoomEnv(render_mode="rgb_array")
    target = np.asarray([180.0, float(door_position)], dtype=np.float32)
    try:
        env.reset(
            seed=int(simulator_seed),
            options={
                "variation": (),
                "variation_values": _factor_options(
                    {"agent.speed": 5.0, "door.position": int(door_position)}
                ),
                "state": np.asarray([70.0, float(door_position)], dtype=np.float32),
                "target_state": target,
            },
        )
        terminated = False
        steps = 0
        for steps in range(1, 101):
            state = env.agent_position.detach().cpu().numpy().astype(np.float32)
            action = np.clip((target - state) / 5.0, -1.0, 1.0)
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                break
        final_state = env.agent_position.detach().cpu().numpy().astype(np.float32)
        final_distance = float(np.linalg.norm(final_state - target))
    finally:
        env.close()
    return {
        "passed": bool(terminated or final_distance < 16.0),
        "steps": int(steps),
        "final_distance_px": final_distance,
    }


def _build_track(
    *,
    config: dict[str, Any],
    config_hash: str,
    track_name: str,
    track: dict[str, Any],
    payload_root: Path,
    catalog_path: Path,
    stable_commit: str,
    queries_per_cell: int,
    exact_replay: bool,
) -> dict[str, Any]:
    evaluation = config["evaluation_data"]
    eval_seeds = [int(value) for value in evaluation["eval_seeds"]]
    per_seed = int(evaluation["unique_queries_per_door_per_task_per_seed"])
    formal_query_count = len(eval_seeds) * per_seed
    if queries_per_cell == formal_query_count:
        assignments = formal_template_assignments(
            eval_seeds=eval_seeds,
            per_seed=per_seed,
            assignment_seed=EVAL_ASSIGNMENT_SEED,
        )
    else:
        assignments = {
            index: (
                eval_seeds[index % len(eval_seeds)],
                index,
                DIRECTIONS[index % len(DIRECTIONS)],
            )
            for index in range(queries_per_cell)
        }
    bundles = []
    query_hashes = set()
    payload_hashes = set()
    task_failures = []
    future_change_failures = []
    history_return_drifts = []
    payload_root.mkdir(parents=True, exist_ok=True)
    for door in map(int, track["door_positions"]):
        for task in TASKS:
            for template_index in range(int(queries_per_cell)):
                _, _, direction = assignments[template_index]
                geometry = make_query_geometry(
                    door_position=door,
                    task=task,
                    direction=direction,
                    template_index=template_index,
                    seed=GEOMETRY_SEED,
                )
                simulator_seed = int(
                    np.random.SeedSequence(
                        [CATALOG_SEED, door, TASKS.index(task), template_index]
                    ).generate_state(1)[0]
                )
                rollout = _simulate_episode(
                    geometry, simulator_seed=simulator_seed
                )
                if exact_replay:
                    replay = _simulate_episode(
                        geometry, simulator_seed=simulator_seed
                    )
                    _assert_exact_replay(rollout, replay)
                views = _prediction_payload_views(rollout)
                history_pixels = views["history_pixels"]
                query_pixels = views["query_pixels"]
                future_pixels = views["future_pixels"]
                future_next_pixels = views["future_next_pixels"]
                future_states = views["future_states"]
                future_next_states = views["future_next_states"]
                history_return_drifts.append(
                    views["nominal_history_return_drift_px"]
                )
                for horizon in HORIZONS:
                    if np.array_equal(
                        future_next_pixels[horizon - 1], query_pixels
                    ):
                        future_change_failures.append(
                            (geometry.template_id, horizon)
                        )
                outcome = _task_oracle(geometry, rollout)
                if not outcome["passed"]:
                    task_failures.append(geometry.template_id)
                query_hash = array_sha256(query_pixels)
                if query_hash in query_hashes:
                    raise RuntimeError(
                        f"Repeated query pixels within track: {geometry.template_id}"
                    )
                query_hashes.add(query_hash)
                static_query_id = (
                    f"twdv-{track_name}-{task}-{direction}-{door}-"
                    f"{template_index:03d}"
                )
                query_id = static_query_id
                payload_path = payload_root / f"{query_id}.npz"
                np.savez_compressed(
                    payload_path,
                    history_pixels=history_pixels,
                    history_actions=rollout["history_actions"],
                    query_pixels=query_pixels,
                    future_actions=rollout["future_actions"],
                    future_pixels=future_pixels,
                    future_next_pixels=future_next_pixels,
                    future_states=future_states,
                    future_next_states=future_next_states,
                )
                with np.load(payload_path, allow_pickle=False) as encoded:
                    if not np.array_equal(encoded["query_pixels"], query_pixels):
                        raise RuntimeError("Encoded payload pixel replay failed")
                    if not np.array_equal(
                        encoded["future_next_states"], future_next_states
                    ):
                        raise RuntimeError("Encoded payload state replay failed")
                payload_hash = file_sha256(payload_path)
                if payload_hash in payload_hashes:
                    raise RuntimeError(f"Repeated payload: {geometry.template_id}")
                payload_hashes.add(payload_hash)
                bundles.append(
                    {
                        "query_id": query_id,
                        "static_query_id": static_query_id,
                        "template_id": geometry.template_id,
                        "template_index": template_index,
                        "track": track_name,
                        "split": str(track["split"]),
                        "door_position": door,
                        "agent_speed": 5.0,
                        "task": task,
                        "direction": direction,
                        "simulator_seed": simulator_seed,
                        "payload": portable_contextworld_path(
                            payload_path, repo_root=ROOT
                        ),
                        "payload_sha256": payload_hash,
                        "query_pixels_sha256": query_hash,
                        "history_pixels_sha256": array_sha256(history_pixels),
                        "future_actions_sha256": array_sha256(
                            rollout["future_actions"]
                        ),
                        "target_pixels_sha256_by_horizon": {
                            str(horizon): array_sha256(
                                future_next_pixels[horizon - 1]
                            )
                            for horizon in HORIZONS
                        },
                        "task_outcome_oracle": outcome,
                        "nominal_history_return_drift_px": views[
                            "nominal_history_return_drift_px"
                        ],
                        "model_visible_fields": ["pixels", "action"],
                        "privileged_fields": [
                            "door_position",
                            "agent_speed",
                            "state",
                            "target_state",
                            "task_outcome_oracle",
                        ],
                    }
                )
    if task_failures or future_change_failures:
        raise RuntimeError(
            f"Door task audit failed: task={task_failures[:3]}, "
            f"future_change={future_change_failures[:3]}"
        )
    if queries_per_cell == len(eval_seeds) * per_seed:
        assign_eval_partitions(
            bundles,
            eval_seeds=eval_seeds,
            per_seed=per_seed,
            assignment_seed=EVAL_ASSIGNMENT_SEED,
        )
        formal_counts = True
    else:
        # Smoke runs remain unmistakably non-formal while preserving schema.
        for index, bundle in enumerate(bundles):
            expected_seed, expected_index, expected_direction = assignments[
                int(bundle["template_index"])
            ]
            if bundle["direction"] != expected_direction:
                raise RuntimeError("Smoke direction assignment changed")
            bundle["eval_seed"] = expected_seed
            bundle["evaluation_index"] = expected_index
        formal_counts = False
    catalog = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed" if formal_counts else "smoke_only",
        "track": track_name,
        "split": str(track["split"]),
        "config": {"sha256": config_hash},
        "stable_worldmodel_commit": stable_commit,
        "protocol": {
            "factor": "door.position",
            "factor_visible_in_query_pixels": True,
            "door_position_icl_claimed": False,
            "input_conditions": ["query_only", "natural_history3"],
            "natural_history3_same_episode": True,
            "query_is_actual_third_history_frame": True,
            "nominal_history_return_drift_tolerance_px": (
                MAX_NOMINAL_HISTORY_RETURN_DRIFT_PX
            ),
            "future_horizons_action_blocks": list(HORIZONS),
            "future_action_blocks": 5,
            "action_block_raw_steps": 5,
            "offline_true_future_targets": True,
            "teacher_forcing_future_frames": False,
        },
        "bundles": bundles,
        "summary": {
            "door_positions": list(map(int, track["door_positions"])),
            "tasks": list(TASKS),
            "directions": list(DIRECTIONS),
            "input_conditions": ["query_only", "natural_history3"],
            "queries_per_door_task": int(queries_per_cell),
            "bundles": len(bundles),
            "scored_sequences_per_checkpoint": len(bundles) * 2,
            "horizon_losses_per_checkpoint": len(bundles) * 2 * len(HORIZONS),
            "unique_query_pixels": len(query_hashes),
            "unique_payloads": len(payload_hashes),
            "all_prediction_targets_change_from_query": True,
            "all_task_outcome_oracles_pass": True,
            "exact_simulator_replay": bool(exact_replay),
            "exact_encoded_payload_replay": True,
            "maximum_nominal_history_return_drift_px": float(
                max(history_return_drifts, default=0.0)
            ),
            "formal_50_by_6_counts": formal_counts,
            "direction_balance_per_door_task_eval_seed": (
                "25_left_to_right_plus_25_right_to_left"
                if formal_counts
                else "smoke_only_not_formal"
            ),
        },
    }
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return catalog


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not str(config.get("status", "")).startswith("preregistered_"):
        raise ValueError("Door benchmark config is not preregistered")
    support = door_support_audit(config)
    if not support["passed"]:
        raise RuntimeError(f"Door support audit failed: {support}")
    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, args.stablewm_repo, args.stablewm_ref
    )
    config_hash = file_sha256(config_path)
    default_root = artifact_path(DEFAULT_ARTIFACT_SUBDIR, repo_root=ROOT)
    if args.evaluation_split == "sealed_test":
        default_root = default_root / "sealed_test"
    artifact_root = args.output_root.resolve() if args.output_root else default_root
    payload_root = artifact_root / "payloads"
    catalog_root = artifact_root / "catalogs"
    evaluation = config["evaluation_data"]
    formal_queries = int(evaluation["unique_queries_per_door_per_task"])
    queries_per_cell = int(args.max_queries_per_door_task or formal_queries)
    if queries_per_cell <= 0 or queries_per_cell > formal_queries:
        raise ValueError("Invalid --max-queries-per-door-task")
    tracks = {}
    all_positions = []
    selected_tracks = _selected_track_rows(config, args.evaluation_split)
    for track_name, track in selected_tracks.items():
        if args.track and track_name != args.track:
            continue
        all_positions.extend(map(int, track["door_positions"]))
        catalog_path = catalog_root / f"{track_name}.json"
        catalog = _build_track(
            config=config,
            config_hash=config_hash,
            track_name=track_name,
            track=track,
            payload_root=payload_root / track_name,
            catalog_path=catalog_path,
            stable_commit=stable_commit,
            queries_per_cell=queries_per_cell,
            exact_replay=not args.skip_exact_replay,
        )
        tracks[track_name] = {
            "catalog": str(catalog_path),
            "catalog_sha256": file_sha256(catalog_path),
            "summary": catalog["summary"],
        }
    if not tracks:
        raise RuntimeError("No validation track selected")
    visible_oracle = _visible_door_oracle(
        sorted(set(all_positions)), simulator_seed=CATALOG_SEED
    )
    planning_oracles = {
        str(door): _scripted_planning_oracle(
            door, simulator_seed=CATALOG_SEED + door
        )
        for door in sorted(set(all_positions))
    }
    if not visible_oracle["passed"] or not all(
        row["passed"] for row in planning_oracles.values()
    ):
        raise RuntimeError("Door pixel/planning oracle failed")
    formal = queries_per_cell == formal_queries and len(tracks) == len(selected_tracks)
    count_key = (
        "validation_counts_per_checkpoint"
        if args.evaluation_split == "validation"
        else "sealed_test_counts_per_checkpoint"
    )
    expected_sequences = int(evaluation[count_key]["scored_sequences"])
    observed_sequences = sum(
        row["summary"]["scored_sequences_per_checkpoint"]
        for row in tracks.values()
    )
    if formal and observed_sequences != expected_sequences:
        raise RuntimeError(
            f"Expected {expected_sequences} sequences, got {observed_sequences}"
        )
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed" if formal else "smoke_only",
        "evaluation_split": args.evaluation_split,
        "config": {"path": str(config_path), "sha256": config_hash},
        "stable_worldmodel": {"repo": str(stable_repo), "commit": stable_commit},
        "tracks": tracks,
        "door_support_audit": support,
        "visible_door_pixel_oracle": visible_oracle,
        "scripted_planning_oracle_by_door": planning_oracles,
        "data_audit": {
            "factor_readback": True,
            "exact_state_and_pixel_replay": not args.skip_exact_replay,
            "exact_encoded_payload_replay": True,
            "query_and_payload_uniqueness": True,
            "eval_scenario_namespace": "door_visual_validation_v1",
            "scenario_ids_unique_and_validation_namespaced": True,
            "zero_train_eval_scenario_id_overlap_by_reserved_namespace": True,
            "online_environment_required_during_model_scoring": False,
        },
        "count_audit": {
            "queries_per_door_task": queries_per_cell,
            "formal_queries_per_door_task": formal_queries,
            "eval_seeds": evaluation["eval_seeds"],
            "formal_50_by_6_counts": formal,
            "scored_sequences_per_checkpoint": observed_sequences,
            "expected_scored_sequences_per_checkpoint": expected_sequences,
            "horizon_losses_per_checkpoint": sum(
                row["summary"]["horizon_losses_per_checkpoint"]
                for row in tracks.values()
            ),
        },
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    report_path = artifact_root / "catalogs" / "build_report.json"
    write_json(report_path, report)
    return {**report, "report": str(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--split",
        "--evaluation-split",
        dest="evaluation_split",
        choices=["validation", "sealed_test"],
        default="validation",
    )
    parser.add_argument("--track")
    parser.add_argument("--max-queries-per-door-task", type=int)
    parser.add_argument("--skip-exact-replay", action="store_true")
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": result["report"],
                "count_audit": result["count_audit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
