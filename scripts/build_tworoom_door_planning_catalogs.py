#!/usr/bin/env python3
"""Build frozen cross-room query/candidate catalogs for the door benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.door_planning import (
    ACTION_BLOCK,
    AGENT_SPEED,
    VALID_DIRECTIONS,
    VALID_OFFSETS,
    deterministic_cem_seed,
    doorway_crossing,
    simulate_door_candidates,
)
from contextworld.evaluation.icl_catalog import _factor_options
from contextworld.evaluation.icl_model import file_sha256
from contextworld.evaluation.planner_mechanism import array_sha256
from contextworld.paths import artifact_path, portable_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"
)
DEFAULT_ROOT = "evaluation/history3/door_visual_generalization_v1/planning"


def _formal_strata(eval_seed_index: int) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for direction_index, direction in enumerate(VALID_DIRECTIONS):
        extra = (int(eval_seed_index) + direction_index) % len(VALID_OFFSETS)
        for offset_index, offset in enumerate(VALID_OFFSETS):
            count = 9 if offset_index == extra else 8
            rows.extend([(direction, int(offset))] * count)
    if len(rows) != 50:
        raise AssertionError(len(rows))
    rng = np.random.default_rng(2026072240 + int(eval_seed_index))
    return [rows[int(index)] for index in rng.permutation(len(rows))]


def _strata(eval_seed_index: int, count: int) -> list[tuple[str, int]]:
    if int(count) == 50:
        return _formal_strata(eval_seed_index)
    if count < 6:
        raise ValueError("A smoke cell needs at least six queries for all strata")
    base = [
        (direction, int(offset))
        for direction in VALID_DIRECTIONS
        for offset in VALID_OFFSETS
    ]
    rows = [base[index % len(base)] for index in range(int(count))]
    rng = np.random.default_rng(2026072240 + int(eval_seed_index))
    return [rows[int(index)] for index in rng.permutation(len(rows))]


def _relative_templates(
    eval_seeds: list[int], queries_per_seed: int
) -> dict[tuple[int, int], dict[str, Any]]:
    counters: Counter[tuple[str, int]] = Counter()
    result = {}
    for seed_index, eval_seed in enumerate(eval_seeds):
        for evaluation_index, (direction, offset) in enumerate(
            _strata(seed_index, queries_per_seed)
        ):
            occurrence = counters[(direction, offset)]
            counters[(direction, offset)] += 1
            left_x = 32.0 + 1.2 * float(occurrence)
            right_x = 192.0 - 1.2 * float(occurrence)
            result[(int(eval_seed), int(evaluation_index))] = {
                "direction": direction,
                "door_relative_vertical_offset_px": int(offset),
                "occurrence_within_stratum": int(occurrence),
                "query_x": left_x if direction == "left_to_right" else right_x,
                "goal_x": right_x if direction == "left_to_right" else left_x,
            }
    if queries_per_seed == 50:
        expected = {
            (direction, offset): 50
            for direction in VALID_DIRECTIONS
            for offset in VALID_OFFSETS
        }
        if dict(counters) != expected:
            raise RuntimeError(f"Six-seed stratum balance failed: {counters}")
    return result


def _advance_free(position: np.ndarray, action: np.ndarray) -> np.ndarray:
    return np.clip(
        position + np.clip(action, -1.0, 1.0) * AGENT_SPEED,
        21.0,
        203.0,
    )


def _scripted_actions(
    *,
    query_state: np.ndarray,
    goal_state: np.ndarray,
    door_position: int,
    doorway_y_delta: float = 0.0,
) -> np.ndarray:
    left_to_right = float(query_state[0]) < 112.0
    approach_x, crossed_x = ((95.0, 130.0) if left_to_right else (129.0, 94.0))
    requested_doorway_y = float(door_position) + float(doorway_y_delta)
    # Agent centers are clamped to [21, 203].  Without clipping the waypoint,
    # the low extrapolation door at y=24 and delta=-12 asks the controller to
    # reach y=12; it then remains stuck at y=21 forever and never advances to
    # the crossing waypoint.  Clip inside both the legal center range and the
    # exact collision opening (door half-size 14 plus margin 1.75).
    doorway_y = float(
        np.clip(
            requested_doorway_y,
            max(21.0, float(door_position) - 15.75),
            min(203.0, float(door_position) + 15.75),
        )
    )
    waypoints = [
        np.asarray([approach_x, doorway_y], dtype=np.float32),
        np.asarray([crossed_x, doorway_y], dtype=np.float32),
        np.asarray(goal_state, dtype=np.float32),
    ]
    position = np.asarray(query_state, dtype=np.float32).copy()
    actions = []
    waypoint_index = 0
    for _ in range(10 * ACTION_BLOCK):
        while (
            waypoint_index < len(waypoints) - 1
            and np.linalg.norm(waypoints[waypoint_index] - position) < 2.5
        ):
            waypoint_index += 1
        action = np.clip(
            (waypoints[waypoint_index] - position) / AGENT_SPEED,
            -1.0,
            1.0,
        ).astype(np.float32)
        actions.append(action)
        position = _advance_free(position, action)
    return np.asarray(actions, dtype=np.float32)


def _candidate_bank(
    *,
    query_state: np.ndarray,
    goal_state: np.ndarray,
    door_position: int,
    seed: int,
    candidates: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    bank = np.empty((int(candidates), 10 * ACTION_BLOCK, 2), dtype=np.float32)
    doorway_deltas = np.linspace(-12.0, 12.0, min(49, int(candidates)))
    for index, delta in enumerate(doorway_deltas):
        bank[index] = _scripted_actions(
            query_state=query_state,
            goal_state=goal_state,
            door_position=door_position,
            doorway_y_delta=float(delta),
        )
    start = len(doorway_deltas)
    block_actions = rng.normal(
        0.0, 0.65, size=(int(candidates) - start, 10, 2)
    ).astype(np.float32)
    block_actions = np.clip(block_actions, -1.0, 1.0)
    bank[start:] = np.repeat(block_actions, ACTION_BLOCK, axis=1)
    return bank


def _history_and_query(
    *,
    query_state: np.ndarray,
    goal_state: np.ndarray,
    door_position: int,
    simulator_seed: int,
    vertical_sign: float,
) -> dict[str, np.ndarray]:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    outward = np.repeat(
        np.asarray([[0.0, 0.25 * vertical_sign]], dtype=np.float32),
        ACTION_BLOCK,
        axis=0,
    )
    actions = np.stack([outward, -outward])
    env = TwoRoomEnv(render_mode="rgb_array")
    pixels = []
    try:
        env.reset(
            seed=int(simulator_seed),
            options={
                "variation": (),
                "variation_values": _factor_options(
                    {"agent.speed": 5.0, "door.position": int(door_position)}
                ),
                "state": query_state,
                "target_state": goal_state,
            },
        )
        for block in actions:
            pixels.append(np.asarray(env.render(), dtype=np.uint8).copy())
            for action in block:
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    raise RuntimeError("Natural planning history terminated")
        query_pixels = np.asarray(env.render(), dtype=np.uint8).copy()
        returned_state = env.agent_position.detach().cpu().numpy().astype(np.float32)
        goal_pixels = (
            env._target_img.detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
            .astype(np.uint8)
        )
        speed = float(
            np.asarray(env.variation_space["agent"]["speed"].value).reshape(-1)[0]
        )
        door = int(
            np.asarray(env.variation_space["door"]["position"].value).reshape(-1)[0]
        )
    finally:
        env.close()
    if speed != 5.0 or door != int(door_position):
        raise RuntimeError("Planning factor readback failed")
    if not np.allclose(returned_state, query_state, rtol=0.0, atol=1e-5):
        raise RuntimeError("Planning history did not return to query")
    if not np.array_equal(pixels[0], query_pixels):
        raise RuntimeError("Planning history query pixels did not return exactly")
    return {
        "history_pixels": np.stack(pixels),
        "history_actions": actions.astype(np.float32),
        "query_pixels": query_pixels,
        "goal_pixels": goal_pixels,
        "query_state": query_state.astype(np.float32),
        "goal_state": goal_state.astype(np.float32),
    }


def _environment_oracle(
    *,
    query_state: np.ndarray,
    goal_state: np.ndarray,
    door_position: int,
    simulator_seed: int,
    actions: np.ndarray,
) -> dict[str, Any]:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    env = TwoRoomEnv(render_mode="rgb_array")
    states = [query_state.copy()]
    terminated = False
    try:
        env.reset(
            seed=int(simulator_seed),
            options={
                "variation": (),
                "variation_values": _factor_options(
                    {"agent.speed": 5.0, "door.position": int(door_position)}
                ),
                "state": query_state,
                "target_state": goal_state,
            },
        )
        for step, action in enumerate(actions, start=1):
            _, _, terminated, truncated, _ = env.step(action)
            states.append(
                env.agent_position.detach().cpu().numpy().astype(np.float32).copy()
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    final_distance = float(np.linalg.norm(states[-1] - goal_state))
    crossing = doorway_crossing(
        np.stack(states), door_position=door_position, goal_state=goal_state
    )
    return {
        "passed": bool((terminated or final_distance < 16.0) and crossing["crossed"]),
        "steps": len(states) - 1,
        "final_distance_px": final_distance,
        "doorway_crossed": bool(crossing["crossed"]),
        "first_doorway_crossing_raw_step": crossing["first_crossing_raw_step"],
    }


def _selected_tracks(config: dict[str, Any], split: str) -> dict[str, Any]:
    expected = "validation" if split == "validation" else "sealed_test"
    rows = {
        name: dict(row)
        for name, row in config["evaluation_data"]["tracks"].items()
        if str(row["split"]) == expected
    }
    if not rows:
        raise RuntimeError(f"No door tracks for split {split}")
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT, args.stablewm_repo, args.stablewm_ref
    )
    if stable_commit != PINNED_STABLEWM:
        raise RuntimeError(f"StableWM commit mismatch: {stable_commit}")
    protocol = config["closed_loop_planning"]
    candidates = int(protocol["candidates"])
    if candidates != 300 or int(protocol["horizon_action_blocks"]) != 10:
        raise RuntimeError("Unexpected frozen door candidate protocol")
    eval_seeds = list(map(int, config["evaluation_data"]["eval_seeds"]))
    tracks = _selected_tracks(config, args.split)
    if args.track:
        tracks = {args.track: tracks[args.track]}
    if args.doors:
        selected_doors = set(map(int, args.doors))
        tracks = {
            name: {**row, "door_positions": [
                int(value) for value in row["door_positions"]
                if int(value) in selected_doors
            ]}
            for name, row in tracks.items()
        }
        tracks = {name: row for name, row in tracks.items() if row["door_positions"]}
    templates = _relative_templates(eval_seeds, args.queries_per_seed)
    artifact_dir = artifact_path(DEFAULT_ROOT, args.split, repo_root=ROOT)
    if args.output is not None:
        output = args.output.resolve()
        artifact_dir = output.parent
    else:
        output = artifact_dir / "catalog.json"
    payload_root = artifact_dir / "payloads"
    payload_root.mkdir(parents=True, exist_ok=True)

    bundles = []
    query_hashes: set[str] = set()
    payload_hashes: set[str] = set()
    distance_by_door: dict[str, list[float]] = defaultdict(list)
    cell_strata: dict[tuple[str, int, int], Counter[str]] = defaultdict(Counter)
    oracle_failures = []
    for track_name, track in tracks.items():
        for door in map(int, track["door_positions"]):
            interior_sign = 1.0 if door < 112 else -1.0
            for (eval_seed, evaluation_index), template in templates.items():
                y = float(
                    door
                    + interior_sign
                    * int(template["door_relative_vertical_offset_px"])
                )
                query_state = np.asarray([template["query_x"], y], dtype=np.float32)
                goal_state = np.asarray([template["goal_x"], y], dtype=np.float32)
                simulator_seed = int(
                    np.random.SeedSequence(
                        [2026072250, door, eval_seed, evaluation_index]
                    ).generate_state(1)[0]
                )
                arrays = _history_and_query(
                    query_state=query_state,
                    goal_state=goal_state,
                    door_position=door,
                    simulator_seed=simulator_seed,
                    vertical_sign=interior_sign,
                )
                candidate_seed = int(
                    np.random.SeedSequence(
                        [2026072260, eval_seed, evaluation_index]
                    ).generate_state(1)[0]
                )
                bank = _candidate_bank(
                    query_state=query_state,
                    goal_state=goal_state,
                    door_position=door,
                    seed=candidate_seed,
                    candidates=candidates,
                )
                arrays["fixed_candidate_raw_actions"] = bank
                vector_oracle = simulate_door_candidates(
                    query_state=query_state,
                    goal_state=goal_state,
                    raw_actions=bank,
                    door_position=door,
                )
                env_oracle = _environment_oracle(
                    query_state=query_state,
                    goal_state=goal_state,
                    door_position=door,
                    simulator_seed=simulator_seed,
                    actions=bank[0],
                )
                if not env_oracle["passed"] or not (
                    vector_oracle["success"][0]
                    and vector_oracle["doorway_crossed"][0]
                ):
                    oracle_failures.append(
                        (track_name, door, eval_seed, evaluation_index)
                    )
                query_id = (
                    f"twdp-{track_name}-d{door}-s{eval_seed}-"
                    f"e{evaluation_index:03d}"
                )
                payload_path = payload_root / f"{query_id}.npz"
                np.savez_compressed(payload_path, **arrays)
                query_hash = array_sha256(arrays["query_pixels"])
                payload_hash = file_sha256(payload_path)
                if query_hash in query_hashes:
                    raise RuntimeError(f"Duplicate planning query pixels: {query_id}")
                if payload_hash in payload_hashes:
                    raise RuntimeError(f"Duplicate planning payload: {query_id}")
                query_hashes.add(query_hash)
                payload_hashes.add(payload_hash)
                distance = float(np.linalg.norm(goal_state - query_state))
                distance_by_door[str(door)].append(distance)
                stratum = (
                    f"{template['direction']}:"
                    f"{template['door_relative_vertical_offset_px']}"
                )
                cell_strata[(track_name, door, eval_seed)][stratum] += 1
                bundle = {
                    "query_id": query_id,
                    "track": track_name,
                    "split": str(track["split"]),
                    "task": "cross_room_navigation",
                    "eval_seed": int(eval_seed),
                    "evaluation_index": int(evaluation_index),
                    "query_factors": {
                        "agent.speed": 5.0,
                        "door.position": int(door),
                    },
                    "agent_speed": 5.0,
                    "door_position": int(door),
                    "direction": template["direction"],
                    "door_relative_vertical_offset_px": int(
                        template["door_relative_vertical_offset_px"]
                    ),
                    "relative_template": template,
                    "simulator_seed": simulator_seed,
                    "cem_seed": deterministic_cem_seed(
                        eval_seed=eval_seed,
                        evaluation_index=evaluation_index,
                        query_id=query_id,
                    ),
                    "payload": portable_contextworld_path(
                        payload_path, repo_root=ROOT
                    ),
                    "payload_sha256": payload_hash,
                    "scripted_oracle": env_oracle,
                }
                bundle.update(
                    {
                        f"{key}_sha256": array_sha256(value)
                        for key, value in arrays.items()
                    }
                )
                bundles.append(bundle)
    if oracle_failures:
        raise RuntimeError(f"Scripted planning oracle failed: {oracle_failures[:5]}")

    expected_strata = {
        f"{direction}:{offset}"
        for direction in VALID_DIRECTIONS
        for offset in VALID_OFFSETS
    }
    cell_audit = {}
    for key, counts in sorted(cell_strata.items()):
        if set(counts) != expected_strata:
            raise RuntimeError(f"Missing stratum in {key}: {counts}")
        direction_totals = {
            direction: sum(
                counts[f"{direction}:{offset}"] for offset in VALID_OFFSETS
            )
            for direction in VALID_DIRECTIONS
        }
        if args.queries_per_seed == 50 and set(direction_totals.values()) != {25}:
            raise RuntimeError(f"Direction balance failed in {key}: {counts}")
        if max(counts.values()) - min(counts.values()) > 1:
            raise RuntimeError(f"Offset balance failed in {key}: {counts}")
        cell_audit["/".join(map(str, key))] = {
            "queries": sum(counts.values()),
            "direction_counts": direction_totals,
            "direction_offset_counts": dict(sorted(counts.items())),
        }
    distance_reference = next(iter(distance_by_door.values()))
    distance_matched = all(
        np.array_equal(
            np.sort(np.asarray(values)), np.sort(np.asarray(distance_reference))
        )
        for values in distance_by_door.values()
    )
    if not distance_matched:
        raise RuntimeError("Door-relative distance distributions differ by door")
    formal = int(args.queries_per_seed) == 50 and len(eval_seeds) == 6
    catalog = {
        "schema_version": 1,
        "benchmark": "tworoom_door_planning_catalog_v1",
        "status": "passed" if formal else "smoke_only",
        "split_role": args.split,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "stable_worldmodel": {"repo": str(stable_repo), "commit": stable_commit},
        "protocol": {
            "agent_speed": 5.0,
            "task": "cross_room_navigation",
            "history_tokens": 3,
            "action_block_raw_steps": 5,
            "candidates_per_query": candidates,
            "candidate_horizon_action_blocks": 10,
            "eval_seeds": eval_seeds,
            "queries_per_door_per_eval_seed": int(args.queries_per_seed),
            "same_relative_templates_across_door_positions": True,
            "sealed_test_model_scoring_performed": False,
        },
        "bundles": bundles,
        "summary": {
            "tracks": list(tracks),
            "door_positions": sorted(
                {int(bundle["door_position"]) for bundle in bundles}
            ),
            "bundles": len(bundles),
            "unique_query_pixels": len(query_hashes),
            "unique_payloads": len(payload_hashes),
            "all_scripted_oracles_pass": True,
            "distance_distributions_identical_across_doors": True,
            "formal_50_by_6_per_door": formal,
            "cell_stratum_audit": cell_audit,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, catalog)
    report_path = output.with_name(output.stem + "_build_report.json")
    report = {
        "status": "passed",
        "catalog": str(output),
        "catalog_sha256": file_sha256(output),
        "summary": catalog["summary"],
    }
    write_json(report_path, report)
    return {**report, "report": str(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--split",
        choices=("validation", "sealed_test"),
        default="validation",
    )
    parser.add_argument("--track")
    parser.add_argument("--doors", type=int, nargs="+")
    parser.add_argument("--queries-per-seed", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
