from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.manifest import write_json

from .action_delay import (
    ACTION_BLOCK,
    ActionDelayTemplate,
    array_sha256,
    canonical_sha256,
)
from .action_delay_env import (
    ACTION_DELAY_FACTOR,
    make_action_delay_env,
)
from .action_delay_validation import file_sha256


DELAYS = (0, 1, 2, 3, 4, 5)
TRAINING_SEEN_DELAYS = (0, 2, 4)
INTERPOLATION_DELAYS = (1, 3)
HIGH_ENDPOINT_DELAYS = (5,)
HORIZONS = (1, 2, 3, 5)
EVAL_SEEDS = (52, 53, 54, 55, 56, 57)
QUERIES_PER_SEED = 50
QUERY_COUNT = len(EVAL_SEEDS) * QUERIES_PER_SEED
PREDICTIONS_PER_CHECKPOINT = QUERY_COUNT * len(DELAYS)
TARGET_ENCODINGS_PER_CHECKPOINT = (
    QUERY_COUNT * len(DELAYS) * len(HORIZONS)
)
LOSS_RECORDS_PER_CHECKPOINT = (
    PREDICTIONS_PER_CHECKPOINT * len(DELAYS) * len(HORIZONS)
)


@dataclass(frozen=True)
class ActionDelayMultistepAssignment:
    query_id: str
    eval_seed: int
    evaluation_index: int
    template: ActionDelayTemplate


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _direction_action(direction: str) -> np.ndarray:
    if direction == "up":
        return np.asarray([0.0, 1.0], dtype=np.float32)
    if direction == "down":
        return np.asarray([0.0, -1.0], dtype=np.float32)
    raise ValueError(f"Unknown direction {direction!r}")


def _block(action: np.ndarray) -> np.ndarray:
    return np.repeat(
        np.asarray(action, dtype=np.float32)[None],
        ACTION_BLOCK,
        axis=0,
    ).astype(np.float32)


def _future_blocks(direction: str, magnitude: float) -> np.ndarray:
    action = float(magnitude) * _direction_action(direction)
    return np.stack(
        [_block(sign * action) for sign in (1.0, -1.0, 1.0, -1.0, 1.0)]
    ).astype(np.float32)


def _step_block(
    env: Any,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    states = []
    executed = []
    ended = False
    for action in np.asarray(actions, dtype=np.float32):
        observation, _, terminated, truncated, info = env.step(action)
        states.append(_as_numpy(observation)[:2])
        executed.append(
            np.asarray(
                info["contextworld.executed_action"],
                dtype=np.float32,
            )
        )
        ended = ended or bool(terminated or truncated)
    return (
        np.stack(states).astype(np.float32),
        np.stack(executed).astype(np.float32),
        ended,
    )


def _delay_oracle(
    commanded: np.ndarray,
    delay_steps: int,
) -> np.ndarray:
    queue = [
        np.zeros(2, dtype=np.float32)
        for _ in range(int(delay_steps))
    ]
    executed = []
    for action in np.asarray(commanded, dtype=np.float32):
        if delay_steps == 0:
            current = action.copy()
        else:
            current = queue.pop(0)
            queue.append(action.copy())
        executed.append(current)
    return np.stack(executed).astype(np.float32)


def _query_coordinate(template: ActionDelayTemplate) -> tuple[float, float]:
    direction = _direction_action(template.direction)
    value = (
        np.asarray(template.reset_state, dtype=np.float32)
        + 7.0 * ACTION_BLOCK * direction
    )
    return tuple(map(float, value))


def _source_grid(config: dict[str, Any]) -> list[tuple[str, float, float]]:
    grid = config["generation"]["source_validation_grid"]

    def values(specification: dict[str, int]) -> list[int]:
        return list(
            range(
                int(specification["start"]),
                int(specification["stop"]),
                int(specification["step"]),
            )
        )

    left = values(grid["left_x"])
    right = values(grid["right_x"])
    y_values = values(grid["query_y"])
    return [
        (room, float(x), float(y))
        for room, x_values in (("left", left), ("right", right))
        for x in x_values
        for y in y_values
    ]


def _catalog_query_key(template: dict[str, Any]) -> tuple[str, float, float]:
    direction = _direction_action(str(template["direction"]))
    query = (
        np.asarray(template["reset_state"], dtype=np.float32)
        + 7.0 * ACTION_BLOCK * direction
    )
    room = "left" if float(query[0]) < 112.0 else "right"
    return (room, float(query[0]), float(query[1]))


def _excluded_catalogs(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> list[tuple[str, Path, dict[str, Any]]]:
    identities = [
        (
            "completed_one_step_validation",
            config["source_identity"]["completed_one_step_validation"],
        )
    ]
    identities.extend(
        (
            str(row["name"]),
            row,
        )
        for row in config["generation"].get(
            "additional_excluded_catalogs", []
        )
    )
    catalogs = []
    for name, identity in identities:
        path = resolve_contextworld_path(
            identity["catalog"], repo_root=repo_root
        )
        observed = file_sha256(path)
        if observed != str(identity["catalog_sha256"]):
            raise ValueError(f"Excluded catalog hash changed: {name}")
        catalogs.append(
            (
                name,
                path,
                json.loads(path.read_text(encoding="utf-8")),
            )
        )
    return catalogs


def select_assignments(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> list[ActionDelayMultistepAssignment]:
    evaluation = config["evaluation"]
    eval_seeds = tuple(map(int, evaluation["eval_seeds"]))
    per_seed = int(evaluation["unique_queries_per_seed"])
    if eval_seeds != EVAL_SEEDS or per_seed != QUERIES_PER_SEED:
        raise ValueError("Multistep Eval requires 50 queries for seeds 52..57")

    excluded_catalogs = _excluded_catalogs(config, repo_root=repo_root)
    used = {
        _catalog_query_key(row["template"])
        for _, _, catalog in excluded_catalogs
        for row in catalog["queries"]
    }

    candidates = [
        row for row in _source_grid(config) if row not in used
    ]
    rng = np.random.default_rng(int(config["generation"]["catalog_seed"]))
    candidates = [
        candidates[index] for index in rng.permutation(len(candidates))
    ]
    selected: list[ActionDelayMultistepAssignment] = []
    selected_coordinates: set[tuple[str, float, float]] = set()
    use_room_queues = bool(
        config["generation"].get("additional_excluded_catalogs")
    )
    if use_room_queues:
        room_candidates = {
            room: [row for row in candidates if row[0] == room]
            for room in ("left", "right")
        }
        room_cursors = {"left": 0, "right": 0}
    else:
        # Preserve the already released v1 assignment algorithm byte-for-byte.
        cursor = 0
    assignment_seed = int(
        config["generation"]["eval_seed_assignment_seed"]
    )
    query_id_prefix = str(
        config["generation"].get("query_id_prefix", "action-delay-ms")
    )
    for eval_seed in eval_seeds:
        directions = ["up"] * 25 + ["down"] * 25
        direction_rng = np.random.default_rng(
            np.random.SeedSequence(
                [assignment_seed, eval_seed, 0xA5D5]
            )
        )
        direction_rng.shuffle(directions)
        for evaluation_index, direction_name in enumerate(directions):
            preferred_room = (
                "left" if evaluation_index % 2 == 0 else "right"
            )
            if use_room_queues:
                cursor = room_cursors[preferred_room]
                available = room_candidates[preferred_room]
                if cursor >= len(available):
                    raise RuntimeError(
                        f"Exhausted unused {preferred_room} Validation grid"
                    )
                room, x_position, y_position = available[cursor]
                room_cursors[preferred_room] += 1
                coordinate = (room, x_position, y_position)
            else:
                while True:
                    if cursor >= len(candidates):
                        raise RuntimeError(
                            "Exhausted unused Validation grid"
                        )
                    room, x_position, y_position = candidates[cursor]
                    cursor += 1
                    coordinate = (room, x_position, y_position)
                    if (
                        room == preferred_room
                        and coordinate not in selected_coordinates
                    ):
                        break
            selected_coordinates.add(coordinate)
            direction = _direction_action(direction_name)
            reset = (
                np.asarray([x_position, y_position], dtype=np.float32)
                - 7.0 * ACTION_BLOCK * direction
            )
            goal = (
                (200.0, 205.0 if y_position < 113.0 else 20.0)
                if room == "left"
                else (25.0, 205.0 if y_position < 113.0 else 20.0)
            )
            simulator_seed = int(
                np.random.SeedSequence(
                    [
                        int(config["generation"]["catalog_seed"]),
                        eval_seed,
                        evaluation_index,
                    ]
                ).generate_state(1)[0]
            )
            query_id = (
                f"{query_id_prefix}-s{eval_seed}-q{evaluation_index:02d}"
            )
            selected.append(
                ActionDelayMultistepAssignment(
                    query_id=query_id,
                    eval_seed=eval_seed,
                    evaluation_index=evaluation_index,
                    template=ActionDelayTemplate(
                        template_id=query_id,
                        direction=direction_name,
                        reset_state=tuple(map(float, reset)),
                        goal_state=tuple(map(float, goal)),
                        simulator_seed=simulator_seed,
                    ),
                )
            )
    if len(selected) != QUERY_COUNT:
        raise RuntimeError(f"Expected {QUERY_COUNT} assignments")
    return selected


def simulate_multistep(
    template: ActionDelayTemplate,
    *,
    delay_steps: int,
    agent_speed: float,
    query_action_magnitude: float,
) -> dict[str, Any]:
    probe = _block(_direction_action(template.direction))
    flush = np.zeros_like(probe)
    future = _future_blocks(
        template.direction,
        query_action_magnitude,
    )
    all_blocks = np.concatenate(
        [probe[None], flush[None], future],
        axis=0,
    ).astype(np.float32)
    env = make_action_delay_env(render_mode="rgb_array")
    history_pixels = []
    history_states = []
    raw_states = []
    executed_actions = []
    ended = False
    try:
        initial_observation, _ = env.reset(
            seed=int(template.simulator_seed),
            options={
                "variation": (),
                "variation_values": {
                    "agent.speed": np.asarray(
                        [float(agent_speed)], dtype=np.float32
                    ),
                    ACTION_DELAY_FACTOR: int(delay_steps),
                },
                "state": np.asarray(
                    template.reset_state, dtype=np.float32
                ),
                "target_state": np.asarray(
                    template.goal_state, dtype=np.float32
                ),
            },
        )
        history_pixels.append(np.asarray(env.render(), dtype=np.uint8))
        history_states.append(_as_numpy(initial_observation)[:2])

        states, executed, block_ended = _step_block(env, probe)
        raw_states.append(states)
        executed_actions.append(executed)
        ended = ended or block_ended
        history_pixels.append(np.asarray(env.render(), dtype=np.uint8))
        history_states.append(states[-1])

        states, executed, block_ended = _step_block(env, flush)
        raw_states.append(states)
        executed_actions.append(executed)
        ended = ended or block_ended
        history_pixels.append(np.asarray(env.render(), dtype=np.uint8))
        history_states.append(states[-1])
        pending_at_query = env.pending_actions()

        future_pixels = []
        future_states = []
        for block in future:
            states, executed, block_ended = _step_block(env, block)
            raw_states.append(states)
            executed_actions.append(executed)
            ended = ended or block_ended
            future_pixels.append(np.asarray(env.render(), dtype=np.uint8))
            future_states.append(states[-1])
        delay_readback = int(env.action_delay_steps)
    finally:
        env.close()

    commanded = all_blocks.reshape(-1, 2)
    oracle_executed = _delay_oracle(commanded, delay_steps)
    expected_states = (
        np.asarray(template.reset_state, dtype=np.float32)[None]
        + np.float32(agent_speed)
        * np.cumsum(oracle_executed, axis=0)
    )
    actual_states = np.concatenate(raw_states, axis=0)
    return {
        "delay_steps": np.asarray(delay_readback, dtype=np.int64),
        "initial_observation": _as_numpy(initial_observation).astype(
            np.float32
        ),
        "history_pixels": np.stack(history_pixels).astype(np.uint8),
        "history_states": np.stack(history_states).astype(np.float32),
        "future_pixels": np.stack(future_pixels).astype(np.uint8),
        "future_states": np.stack(future_states).astype(np.float32),
        "all_action_blocks": all_blocks,
        "future_action_blocks": future,
        "executed_actions": np.concatenate(
            executed_actions, axis=0
        ).astype(np.float32),
        "oracle_executed_actions": oracle_executed,
        "actual_raw_states": actual_states.astype(np.float32),
        "expected_raw_states": expected_states.astype(np.float32),
        "query_pixels": history_pixels[-1].copy(),
        "query_state": history_states[-1].copy(),
        "pending_actions_at_query": pending_at_query.astype(np.float32),
        "terminated_or_truncated": bool(ended),
    }


def validate_family(
    template: ActionDelayTemplate,
    rollouts: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if tuple(sorted(rollouts)) != DELAYS:
        raise ValueError(f"Expected delays {DELAYS}")
    ordered = [rollouts[delay] for delay in DELAYS]

    def all_equal(key: str) -> bool:
        return all(
            np.array_equal(ordered[0][key], row[key])
            for row in ordered[1:]
        )

    target_distinct = {}
    for horizon in HORIZONS:
        values = [
            row["future_pixels"][horizon - 1] for row in ordered
        ]
        target_distinct[str(horizon)] = len(
            {array_sha256(value) for value in values}
        ) == len(DELAYS)
    checks = {
        "delay_readback_exact": all(
            int(rollouts[delay]["delay_steps"]) == delay
            for delay in DELAYS
        ),
        "initial_observation_identical": all_equal(
            "initial_observation"
        ),
        "initial_pixels_identical": all(
            np.array_equal(
                ordered[0]["history_pixels"][0],
                row["history_pixels"][0],
            )
            for row in ordered[1:]
        ),
        "all_action_blocks_identical": all_equal("all_action_blocks"),
        "query_state_identical": all_equal("query_state"),
        "query_pixels_identical": all_equal("query_pixels"),
        "history_midpoint_states_distinct": len(
            {
                tuple(map(float, row["history_states"][1]))
                for row in ordered
            }
        )
        == len(DELAYS),
        "history_midpoint_pixels_distinct": len(
            {
                array_sha256(row["history_pixels"][1])
                for row in ordered
            }
        )
        == len(DELAYS),
        "queues_zero_at_query": all(
            row["pending_actions_at_query"].shape == (delay, 2)
            and np.array_equal(
                row["pending_actions_at_query"],
                np.zeros((delay, 2), dtype=np.float32),
            )
            for delay, row in rollouts.items()
        ),
        "physical_oracle_exact": all(
            np.array_equal(
                row["executed_actions"],
                row["oracle_executed_actions"],
            )
            and np.allclose(
                row["actual_raw_states"],
                row["expected_raw_states"],
                atol=1e-5,
            )
            for row in ordered
        ),
        "targets_distinct_at_every_scored_horizon": all(
            target_distinct.values()
        ),
        "no_termination_or_boundary_effect": not any(
            row["terminated_or_truncated"] for row in ordered
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "targets_distinct_by_horizon": target_distinct,
    }


def build_asset(
    assignment: ActionDelayMultistepAssignment,
    *,
    agent_speed: float,
    query_action_magnitude: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rollouts = {
        delay: simulate_multistep(
            assignment.template,
            delay_steps=delay,
            agent_speed=agent_speed,
            query_action_magnitude=query_action_magnitude,
        )
        for delay in DELAYS
    }
    family = validate_family(assignment.template, rollouts)
    if not family["passed"]:
        raise RuntimeError(
            f"Multistep family failed {assignment.query_id}: {family}"
        )
    arrays = {
        "history_pixels": np.stack(
            [rollouts[delay]["history_pixels"] for delay in DELAYS]
        ).astype(np.uint8),
        "action_blocks": np.stack(
            [rollouts[delay]["all_action_blocks"] for delay in DELAYS]
        ).astype(np.float32),
        "target_pixels": np.stack(
            [rollouts[delay]["future_pixels"] for delay in DELAYS]
        ).astype(np.uint8),
        "query_pixels": rollouts[0]["query_pixels"].astype(np.uint8),
        "history_states": np.stack(
            [rollouts[delay]["history_states"] for delay in DELAYS]
        ).astype(np.float32),
        "target_states": np.stack(
            [rollouts[delay]["future_states"] for delay in DELAYS]
        ).astype(np.float32),
        "history_delays": np.asarray(DELAYS, dtype=np.int64),
        "target_delays": np.asarray(DELAYS, dtype=np.int64),
        "target_horizons": np.asarray(HORIZONS, dtype=np.int64),
    }
    payload_sha256 = canonical_sha256(
        {
            key: array_sha256(value)
            for key, value in sorted(arrays.items())
        }
    )
    return arrays, {
        "query_id": assignment.query_id,
        "eval_seed": assignment.eval_seed,
        "evaluation_index": assignment.evaluation_index,
        "direction": assignment.template.direction,
        "template": asdict(assignment.template),
        "query_coordinate": list(
            _query_coordinate(assignment.template)
        ),
        "family_passed": True,
        "family_checks": family["checks"],
        "payload_sha256": payload_sha256,
        "query_pixels_sha256": array_sha256(arrays["query_pixels"]),
        "history_pixels_sha256": array_sha256(
            arrays["history_pixels"]
        ),
        "action_blocks_sha256": array_sha256(arrays["action_blocks"]),
        "target_pixels_sha256": array_sha256(arrays["target_pixels"]),
    }


def _training_query_coordinates(
    training_config_path: Path,
    *,
    repo_root: Path,
) -> set[tuple[float, float]]:
    from .action_delay_h3_data import build_shard_plans

    config = yaml.safe_load(
        training_config_path.read_text(encoding="utf-8")
    )
    plans = build_shard_plans(config, repo_root=repo_root)
    return {
        _query_coordinate(episode.template)
        for shards in plans.values()
        for shard in shards
        for episode in shard.episodes
    }


def build_release(
    *,
    config: dict[str, Any],
    config_path: Path,
    repo_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if tuple(map(int, config["protocol"]["delay_values"])) != DELAYS:
        raise ValueError("Extension must cover delays 0..5")
    if tuple(
        map(int, config["protocol"]["target_horizons_action_blocks"])
    ) != HORIZONS:
        raise ValueError("Extension must score h1/h2/h3/h5")
    if output_root.exists():
        raise FileExistsError(output_root)
    assets_root = output_root / "assets"
    assets_root.mkdir(parents=True)

    assignments = select_assignments(config, repo_root=repo_root)
    rows = []
    for index, assignment in enumerate(assignments, start=1):
        arrays, audit = build_asset(
            assignment,
            agent_speed=float(config["protocol"]["agent_speed"]),
            query_action_magnitude=float(
                config["protocol"]["query_action_magnitude"]
            ),
        )
        path = assets_root / f"{assignment.query_id}.npz"
        np.savez_compressed(path, **arrays)
        reopened = dict(np.load(path, allow_pickle=False))
        reopened_hash = canonical_sha256(
            {
                key: array_sha256(value)
                for key, value in sorted(reopened.items())
            }
        )
        rows.append(
            {
                **audit,
                "asset": portable_contextworld_path(
                    path, repo_root=repo_root
                ),
                "asset_sha256": file_sha256(path),
                "asset_reopens": set(reopened) == set(arrays),
                "asset_hash_matches": (
                    reopened_hash == audit["payload_sha256"]
                ),
            }
        )
        if index % 25 == 0:
            print(
                f"[action-delay-multistep] built {index}/{len(assignments)}",
                flush=True,
            )

    counts = Counter(int(row["eval_seed"]) for row in rows)
    directions = {
        seed: Counter(
            row["direction"]
            for row in rows
            if int(row["eval_seed"]) == seed
        )
        for seed in EVAL_SEEDS
    }
    excluded_catalogs = _excluded_catalogs(config, repo_root=repo_root)
    excluded_query_hashes = {
        name: {
            row["query_pixels_sha256"] for row in catalog["queries"]
        }
        for name, _, catalog in excluded_catalogs
    }
    all_excluded_query_hashes = set().union(
        *excluded_query_hashes.values()
    )
    new_query_hashes = {row["query_pixels_sha256"] for row in rows}
    training_config_path = (
        repo_root
        / config["source_identity"]["training_data_config"]
    ).resolve()
    training_coordinates = _training_query_coordinates(
        training_config_path,
        repo_root=repo_root,
    )
    new_coordinates = {
        tuple(map(float, row["query_coordinate"])) for row in rows
    }
    checks = {
        "exact_300_queries": len(rows) == QUERY_COUNT,
        "exact_50_queries_per_seed": counts
        == Counter({seed: QUERIES_PER_SEED for seed in EVAL_SEEDS}),
        "directions_balanced_per_seed": all(
            directions[seed] == Counter({"up": 25, "down": 25})
            for seed in EVAL_SEEDS
        ),
        "query_ids_unique": len({row["query_id"] for row in rows})
        == QUERY_COUNT,
        "query_pixels_unique": len(new_query_hashes) == QUERY_COUNT,
        "payloads_unique": len(
            {row["payload_sha256"] for row in rows}
        )
        == QUERY_COUNT,
        "all_families_pass": all(
            row["family_passed"]
            and all(row["family_checks"].values())
            for row in rows
        ),
        "all_assets_reopen_and_match": all(
            row["asset_reopens"] and row["asset_hash_matches"]
            for row in rows
        ),
        "disjoint_from_all_declared_prior_eval_queries": not (
            all_excluded_query_hashes & new_query_hashes
        ),
        "coordinates_disjoint_from_formal_training": not (
            training_coordinates & new_coordinates
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Multistep release audit failed: {checks}")

    content = {
        "benchmark": config["benchmark"],
        "protocol": {
            "history_tokens": 3,
            "raw_steps_per_action_block": ACTION_BLOCK,
            "future_action_blocks": 5,
            "target_horizons": list(HORIZONS),
            "agent_speed": float(config["protocol"]["agent_speed"]),
            "delay_values": list(DELAYS),
            "training_seen_delay_values": list(TRAINING_SEEN_DELAYS),
            "interpolation_delay_values": list(INTERPOLATION_DELAYS),
            "high_endpoint_delay_values": list(HIGH_ENDPOINT_DELAYS),
        },
        "queries": [
            {
                key: row[key]
                for key in (
                    "query_id",
                    "eval_seed",
                    "evaluation_index",
                    "direction",
                    "template",
                    "query_coordinate",
                    "asset",
                    "asset_sha256",
                    "payload_sha256",
                    "query_pixels_sha256",
                    "history_pixels_sha256",
                    "action_blocks_sha256",
                    "target_pixels_sha256",
                )
            }
            for row in rows
        ],
    }
    content_sha256 = canonical_sha256(content)
    catalog = {
        "schema_version": 1,
        **content,
        "content_manifest_sha256": content_sha256,
        "counts": {
            "queries": QUERY_COUNT,
            "history_conditions_per_query": len(DELAYS),
            "true_targets_per_query": len(DELAYS),
            "scored_horizons": len(HORIZONS),
            "model_rollouts_per_checkpoint": PREDICTIONS_PER_CHECKPOINT,
            "target_encodings_per_checkpoint": (
                TARGET_ENCODINGS_PER_CHECKPOINT
            ),
            "horizon_loss_records_per_checkpoint": (
                LOSS_RECORDS_PER_CHECKPOINT
            ),
        },
    }
    catalog_path = output_root / "catalog.json"
    write_json(catalog_path, catalog)
    training_report_path = resolve_contextworld_path(
        config["source_identity"]["training_build_report"]["path"],
        repo_root=repo_root,
    )
    expected_training_report_hash = config["source_identity"][
        "training_build_report"
    ]["sha256"]
    if file_sha256(training_report_path) != expected_training_report_hash:
        raise ValueError("Frozen training build report hash changed")
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "checks": checks,
        "identity": {
            "config": portable_contextworld_path(
                config_path, repo_root=repo_root
            ),
            "config_sha256": file_sha256(config_path),
            "excluded_eval_catalogs": {
                name: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for name, path, _ in excluded_catalogs
            },
            "training_build_report": str(training_report_path),
            "training_build_report_sha256": file_sha256(
                training_report_path
            ),
        },
        "catalog": portable_contextworld_path(
            catalog_path, repo_root=repo_root
        ),
        "catalog_sha256": file_sha256(catalog_path),
        "content_manifest_sha256": content_sha256,
        "counts": catalog["counts"],
        "query_isolation": {
            "excluded_validation_query_hashes": {
                name: len(values)
                for name, values in excluded_query_hashes.items()
            },
            "new_validation_query_hashes": len(new_query_hashes),
            "training_query_coordinates": len(training_coordinates),
            "new_query_coordinates": len(new_coordinates),
        },
    }
    write_json(output_root / "build_report.json", report)
    return report


def load_assets(
    catalog_path: Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog = json.loads(
        Path(catalog_path).read_text(encoding="utf-8")
    )
    assets = []
    for row in catalog["queries"]:
        path = resolve_contextworld_path(row["asset"], repo_root=repo_root)
        if file_sha256(path) != row["asset_sha256"]:
            raise ValueError(f"Asset hash mismatch: {path}")
        arrays = dict(np.load(path, allow_pickle=False))
        observed = canonical_sha256(
            {
                key: array_sha256(value)
                for key, value in sorted(arrays.items())
            }
        )
        if observed != row["payload_sha256"]:
            raise ValueError(f"Payload hash mismatch: {path}")
        assets.append({**row, **arrays})
    return catalog, assets


def _target_track(delay: int) -> str:
    if delay in TRAINING_SEEN_DELAYS:
        return "training_seen"
    if delay in INTERPOLATION_DELAYS:
        return "interpolation"
    if delay in HIGH_ENDPOINT_DELAYS:
        return "high_endpoint_extrapolation"
    raise ValueError(delay)


def score_assets(
    adapter: Any,
    assets: list[dict[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    input_pixels = np.concatenate(
        [asset["history_pixels"] for asset in assets], axis=0
    )
    action_blocks = np.concatenate(
        [asset["action_blocks"] for asset in assets], axis=0
    )
    predicted = np.asarray(
        adapter.rollout_latents(
            input_pixels,
            action_blocks,
            batch_size=batch_size,
        )
    )
    if predicted.shape[0] != PREDICTIONS_PER_CHECKPOINT:
        raise ValueError(f"Unexpected predictions: {predicted.shape}")
    if predicted.shape[1] != 5:
        raise ValueError(f"Expected five futures: {predicted.shape}")

    horizon_indices = [horizon - 1 for horizon in HORIZONS]
    targets = np.concatenate(
        [
            asset["target_pixels"][:, horizon_indices]
            for asset in assets
        ],
        axis=0,
    )
    flat_targets = targets.reshape(
        -1, *targets.shape[-3:]
    )
    encoded = np.asarray(
        adapter.encode_pixels(flat_targets, batch_size=batch_size)
    ).reshape(
        QUERY_COUNT,
        len(DELAYS),
        len(HORIZONS),
        -1,
    )
    predicted = predicted.reshape(
        QUERY_COUNT,
        len(DELAYS),
        5,
        -1,
    )
    records = []
    for query_index, asset in enumerate(assets):
        for history_index, history_delay in enumerate(DELAYS):
            for target_index, target_delay in enumerate(DELAYS):
                for horizon_index, horizon in enumerate(HORIZONS):
                    prediction = predicted[
                        query_index,
                        history_index,
                        horizon - 1,
                    ]
                    target = encoded[
                        query_index,
                        target_index,
                        horizon_index,
                    ]
                    records.append(
                        {
                            "query_id": asset["query_id"],
                            "eval_seed": int(asset["eval_seed"]),
                            "evaluation_index": int(
                                asset["evaluation_index"]
                            ),
                            "direction": asset["direction"],
                            "history_delay": int(history_delay),
                            "target_delay": int(target_delay),
                            "target_track": _target_track(target_delay),
                            "horizon": int(horizon),
                            "latent_mse": float(
                                np.mean((prediction - target) ** 2)
                            ),
                        }
                    )
    return {
        "records": records,
        "score_audit": {
            "queries": len(assets),
            "model_rollouts": len(input_pixels),
            "target_encodings": len(flat_targets),
            "horizon_loss_records": len(records),
            "online_environment_calls": 0,
            "privileged_fields_passed_to_adapter": [],
        },
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in records:
        grouped[(str(row["query_id"]), int(row["horizon"]))].append(row)
    if len(grouped) != QUERY_COUNT * len(HORIZONS):
        raise ValueError("Incomplete query/horizon groups")

    query_metrics = []
    for (query_id, horizon), rows in sorted(grouped.items()):
        losses = {
            (int(row["history_delay"]), int(row["target_delay"])): float(
                row["latent_mse"]
            )
            for row in rows
        }
        if len(losses) != len(DELAYS) ** 2:
            raise ValueError(f"Incomplete 6x6 matrix: {query_id}/h{horizon}")
        exemplar = rows[0]
        for target_delay in DELAYS:
            matching = losses[(target_delay, target_delay)]
            other = [
                losses[(history_delay, target_delay)]
                for history_delay in DELAYS
                if history_delay != target_delay
            ]
            selected_history = min(
                DELAYS,
                key=lambda value: (
                    losses[(value, target_delay)],
                    value,
                ),
            )
            selected_target = min(
                DELAYS,
                key=lambda value: (
                    losses[(target_delay, value)],
                    value,
                ),
            )
            query_metrics.append(
                {
                    "query_id": query_id,
                    "eval_seed": int(exemplar["eval_seed"]),
                    "direction": exemplar["direction"],
                    "horizon": horizon,
                    "target_delay": target_delay,
                    "target_track": _target_track(target_delay),
                    "matching_history_loss": matching,
                    "other_history_mean_loss": float(np.mean(other)),
                    "history_margin": float(np.mean(other) - matching),
                    "history_loss_ratio": float(
                        matching
                        / max(float(np.mean(other)), 1e-12)
                    ),
                    "matching_history_strict_win": matching < min(other),
                    "selected_history": int(selected_history),
                    "history_selection_correct": (
                        selected_history == target_delay
                    ),
                    "selected_target": int(selected_target),
                    "target_selection_correct": (
                        selected_target == target_delay
                    ),
                }
            )

    def aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
        if not values:
            raise ValueError("Cannot aggregate empty metrics")
        return {
            "queries": len(values),
            "mean_matching_history_loss": float(
                np.mean(
                    [row["matching_history_loss"] for row in values]
                )
            ),
            "mean_other_history_loss": float(
                np.mean(
                    [row["other_history_mean_loss"] for row in values]
                )
            ),
            "mean_history_margin": float(
                np.mean([row["history_margin"] for row in values])
            ),
            "mean_history_loss_ratio": float(
                np.mean([row["history_loss_ratio"] for row in values])
            ),
            "matching_history_strict_win_rate": float(
                np.mean(
                    [row["matching_history_strict_win"] for row in values]
                )
            ),
            "history_selection_accuracy": float(
                np.mean(
                    [row["history_selection_correct"] for row in values]
                )
            ),
            "target_selection_accuracy": float(
                np.mean(
                    [row["target_selection_correct"] for row in values]
                )
            ),
        }

    by_horizon = {}
    for horizon in HORIZONS:
        selected_horizon = [
            row for row in query_metrics if row["horizon"] == horizon
        ]
        by_horizon[str(horizon)] = {
            "overall": aggregate(selected_horizon),
            "by_target_delay": {
                str(delay): aggregate(
                    [
                        row
                        for row in selected_horizon
                        if row["target_delay"] == delay
                    ]
                )
                for delay in DELAYS
            },
            "by_track": {
                track: aggregate(
                    [
                        row
                        for row in selected_horizon
                        if row["target_track"] == track
                    ]
                )
                for track in (
                    "training_seen",
                    "interpolation",
                    "high_endpoint_extrapolation",
                )
            },
            "by_target_delay_and_eval_seed": {
                str(delay): {
                    str(seed): aggregate(
                        [
                            row
                            for row in selected_horizon
                            if row["target_delay"] == delay
                            and row["eval_seed"] == seed
                        ]
                    )
                    for seed in EVAL_SEEDS
                }
                for delay in DELAYS
            },
            "by_target_delay_and_direction": {
                str(delay): {
                    direction: aggregate(
                        [
                            row
                            for row in selected_horizon
                            if row["target_delay"] == delay
                            and row["direction"] == direction
                        ]
                    )
                    for direction in ("up", "down")
                }
                for delay in DELAYS
            },
        }
    return {
        "by_horizon": by_horizon,
        "query_metrics": query_metrics,
    }


__all__ = [
    "DELAYS",
    "EVAL_SEEDS",
    "HIGH_ENDPOINT_DELAYS",
    "HORIZONS",
    "INTERPOLATION_DELAYS",
    "LOSS_RECORDS_PER_CHECKPOINT",
    "PREDICTIONS_PER_CHECKPOINT",
    "QUERIES_PER_SEED",
    "QUERY_COUNT",
    "TARGET_ENCODINGS_PER_CHECKPOINT",
    "TRAINING_SEEN_DELAYS",
    "ActionDelayMultistepAssignment",
    "build_asset",
    "build_release",
    "load_assets",
    "score_assets",
    "select_assignments",
    "simulate_multistep",
    "summarize_records",
    "validate_family",
]
