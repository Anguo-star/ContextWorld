from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel

from .action_delay import array_sha256, canonical_sha256
from .action_delay_long_history import (
    ACTION_BLOCK,
    LongHistoryDelayTemplate,
    simulate_template,
)


HISTORY_TOKENS = 7
HISTORY_TRANSITIONS = HISTORY_TOKENS - 1
DELAYS = tuple(range(11))
FUTURE_HORIZONS = (1, 2, 3)
EVAL_SEEDS = (42, 43, 44, 45, 46, 47)
QUERIES_PER_SEED = 50
QUERY_COUNT = len(EVAL_SEEDS) * QUERIES_PER_SEED
MODEL_VISIBLE_FIELDS = ("pixels", "action")
ARRAY_KEYS = (
    "history_delays",
    "target_delays",
    "history_pixels",
    "action_blocks",
    "true_future_pixels",
    "query_pixels",
    "goal_state",
    "audit_history_states",
    "audit_true_future_states",
    "audit_raw_states",
    "audit_executed_actions",
    "audit_pending_actions_at_query",
    "audit_pending_action_lengths",
)


@dataclass(frozen=True)
class ActionDelayH7ValidationAssignment:
    query_id: str
    eval_seed: int
    evaluation_index: int
    room: str
    template: LongHistoryDelayTemplate


@dataclass(frozen=True)
class _AssetBuildJob:
    assignment: ActionDelayH7ValidationAssignment
    asset_path: Path
    repo_root: Path
    agent_speed: float
    action_magnitude: float
    maximum_delay_steps: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name].copy() for name in bundle.files}


def _payload_hashes(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, str], str]:
    hashes = {
        name: array_sha256(value)
        for name, value in sorted(arrays.items())
    }
    return hashes, canonical_sha256(hashes)


def _range_from_specification(
    specification: dict[str, Any],
) -> tuple[int, ...]:
    values = tuple(
        range(
            int(specification["start"]),
            int(specification["stop_exclusive"]),
            int(specification["step"]),
        )
    )
    if not values:
        raise ValueError(f"Empty Validation coordinate range: {specification}")
    return values


def _balanced_values(
    left: str,
    right: str,
    *,
    count: int,
    rng: np.random.Generator,
) -> list[str]:
    if count % 2:
        raise ValueError("Balanced Validation assignments require an even count")
    values = [left] * (count // 2) + [right] * (count // 2)
    rng.shuffle(values)
    return values


def _validate_protocol(config: dict[str, Any]) -> dict[str, Any]:
    validation = config["validation"]
    history = config["history_protocol"]
    environment = config["environment"]
    checks = {
        "history_tokens_are_seven": int(history["history_tokens"])
        == HISTORY_TOKENS,
        "context_transitions_are_six": int(history["context_transitions"])
        == HISTORY_TRANSITIONS,
        "action_block_is_five_raw_steps": int(
            history["raw_steps_per_action_block"]
        )
        == ACTION_BLOCK,
        "delays_are_zero_through_ten": tuple(
            map(int, validation["delay_values"])
        )
        == DELAYS,
        "future_horizons_are_one_two_three": tuple(
            map(int, validation["future_horizons_action_blocks"])
        )
        == FUTURE_HORIZONS,
        "eval_seeds_are_frozen": tuple(map(int, validation["eval_seeds"]))
        == EVAL_SEEDS,
        "queries_per_seed_are_fifty": int(validation["queries_per_seed"])
        == QUERIES_PER_SEED,
        "queries_per_delay_are_three_hundred": int(
            validation["independent_queries_per_delay"]
        )
        == QUERY_COUNT,
        "offline_true_futures_are_required": bool(
            validation["offline_true_futures"]
        )
        and not bool(validation["online_environment_during_model_scoring"]),
        "delay_range_is_zero_through_ten": tuple(
            map(int, environment["delay_range"])
        )
        == (0, 10),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Invalid frozen H7 Validation protocol: {failed}")
    return checks


def select_validation_assignments(
    config: dict[str, Any],
) -> list[ActionDelayH7ValidationAssignment]:
    """Select 300 distinct query geometries, partitioned 50 per Eval seed."""

    _validate_protocol(config)
    validation = config["validation"]
    geometry = validation["query_geometry"]
    catalog_seed = int(validation["catalog_seed"])
    x_by_room = {
        "left": _range_from_specification(geometry["left_x"]),
        "right": _range_from_specification(geometry["right_x"]),
    }
    y_by_direction = {
        "up": _range_from_specification(geometry["up_y"]),
        "down": _range_from_specification(geometry["down_y"]),
    }
    expected_rooms = {
        str(key): int(value)
        for key, value in geometry["rooms_per_seed"].items()
    }
    expected_directions = {
        str(key): int(value)
        for key, value in geometry["directions_per_seed"].items()
    }
    if expected_rooms != {"left": 25, "right": 25}:
        raise ValueError("Each Eval seed must contain 25 queries per room")
    if expected_directions != {"up": 25, "down": 25}:
        raise ValueError("Each Eval seed must contain 25 queries per direction")

    selected: list[ActionDelayH7ValidationAssignment] = []
    used_coordinates: set[tuple[float, float]] = set()
    used_simulator_seeds: set[int] = set()
    for eval_seed in EVAL_SEEDS:
        assignment_rng = np.random.default_rng(
            np.random.SeedSequence([catalog_seed, eval_seed, 0xA7D])
        )
        rooms = _balanced_values(
            "left",
            "right",
            count=QUERIES_PER_SEED,
            rng=assignment_rng,
        )
        directions = _balanced_values(
            "up",
            "down",
            count=QUERIES_PER_SEED,
            rng=assignment_rng,
        )
        candidate_pools: dict[tuple[str, str], list[tuple[int, int]]] = {}
        cursors: Counter[tuple[str, str]] = Counter()
        for room in ("left", "right"):
            for direction in ("up", "down"):
                values = [
                    (x_position, y_position)
                    for x_position in x_by_room[room]
                    for y_position in y_by_direction[direction]
                ]
                pool_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            catalog_seed,
                            eval_seed,
                            0 if room == "left" else 1,
                            0 if direction == "up" else 1,
                        ]
                    )
                )
                permutation = pool_rng.permutation(len(values))
                candidate_pools[(room, direction)] = [
                    values[index] for index in permutation
                ]

        for evaluation_index, (room, direction) in enumerate(
            zip(rooms, directions, strict=True)
        ):
            key = (room, direction)
            pool = candidate_pools[key]
            while True:
                cursor = cursors[key]
                if cursor >= len(pool):
                    raise RuntimeError(
                        f"Exhausted H7 Validation coordinates for {key}"
                    )
                x_position, y_position = pool[cursor]
                cursors[key] += 1
                coordinate = (float(x_position), float(y_position))
                if coordinate not in used_coordinates:
                    break
            used_coordinates.add(coordinate)
            simulator_seed = int(
                np.random.SeedSequence(
                    [catalog_seed, eval_seed, evaluation_index, 0x51A]
                ).generate_state(1)[0]
            )
            if simulator_seed in used_simulator_seeds:
                raise RuntimeError("H7 Validation simulator seed repeated")
            used_simulator_seeds.add(simulator_seed)
            goal_state = (
                (190.0, 200.0 if y_position < 112 else 24.0)
                if room == "left"
                else (30.0, 200.0 if y_position < 112 else 24.0)
            )
            query_id = (
                f"action-delay-h7-val-s{eval_seed}-q"
                f"{evaluation_index:02d}"
            )
            selected.append(
                ActionDelayH7ValidationAssignment(
                    query_id=query_id,
                    eval_seed=eval_seed,
                    evaluation_index=evaluation_index,
                    room=room,
                    template=LongHistoryDelayTemplate(
                        template_id=query_id,
                        direction=direction,
                        reset_state=coordinate,
                        goal_state=goal_state,
                        simulator_seed=simulator_seed,
                    ),
                )
            )
    if len(selected) != QUERY_COUNT:
        raise RuntimeError(
            f"Expected {QUERY_COUNT} H7 Validation queries, got {len(selected)}"
        )
    if len(used_coordinates) != QUERY_COUNT:
        raise RuntimeError("H7 Validation query coordinates repeat")
    return selected


def _unique_array_count(values: Iterable[np.ndarray]) -> int:
    return len(
        {
            (
                np.ascontiguousarray(value).dtype.str,
                np.ascontiguousarray(value).shape,
                np.ascontiguousarray(value).tobytes(),
            )
            for value in values
        }
    )


def audit_rollout_family(
    assignment: ActionDelayH7ValidationAssignment,
    rollouts: dict[int, dict[str, Any]],
    *,
    agent_speed: float,
    action_magnitude: float,
) -> dict[str, Any]:
    """Check queue semantics and future trajectories against exact physics."""

    if tuple(sorted(rollouts)) != DELAYS:
        raise ValueError(f"Expected delay family {DELAYS}")
    ordered = [rollouts[delay] for delay in DELAYS]
    history_states = [value["history_states"] for value in ordered]
    history_pixels = [value["history_pixels"] for value in ordered]
    query_states = [value["history_states"][-1] for value in ordered]
    query_pixels = [value["history_pixels"][-1] for value in ordered]
    action_blocks = [value["action_blocks"] for value in ordered]
    sign = 1.0 if assignment.template.direction == "up" else -1.0
    direction = np.asarray([0.0, sign], dtype=np.float32)
    reset_state = np.asarray(
        assignment.template.reset_state,
        dtype=np.float32,
    )
    expected_future = {
        delay: np.stack(
            [
                reset_state
                + (
                    float(agent_speed)
                    * float(action_magnitude)
                    * max(ACTION_BLOCK * horizon - delay, 0)
                    * direction
                )
                for horizon in FUTURE_HORIZONS
            ]
        ).astype(np.float32)
        for delay in DELAYS
    }
    state_group_counts = {
        str(horizon): _unique_array_count(
            value["future_states"][horizon - 1] for value in ordered
        )
        for horizon in FUTURE_HORIZONS
    }
    pixel_group_counts = {
        str(horizon): _unique_array_count(
            value["future_pixels"][horizon - 1] for value in ordered
        )
        for horizon in FUTURE_HORIZONS
    }
    expected_group_counts = {"1": 6, "2": 11, "3": 11}
    checks = {
        "delay_readback_exact": all(
            int(value["delay_steps"]) == delay
            for delay, value in zip(DELAYS, ordered, strict=True)
        ),
        "initial_state_and_pixels_identical": (
            _unique_array_count(value[0] for value in history_states) == 1
            and _unique_array_count(value[0] for value in history_pixels) == 1
        ),
        "history_actions_identical": _unique_array_count(action_blocks) == 1,
        "all_eleven_history_trajectories_are_distinct": (
            _unique_array_count(history_states) == len(DELAYS)
            and _unique_array_count(history_pixels) == len(DELAYS)
        ),
        "query_state_and_pixels_identical": (
            _unique_array_count(query_states) == 1
            and _unique_array_count(query_pixels) == 1
        ),
        "query_returns_to_reset_state": all(
            np.array_equal(value, reset_state) for value in query_states
        ),
        "pending_queue_empty": all(
            value["pending_actions_at_query"].shape == (delay, 2)
            and np.array_equal(
                value["pending_actions_at_query"],
                np.zeros((delay, 2), dtype=np.float32),
            )
            for delay, value in zip(DELAYS, ordered, strict=True)
        ),
        "final_history_transition_stationary": all(
            np.array_equal(
                value["history_states"][-2],
                value["history_states"][-1],
            )
            for value in ordered
        ),
        "executed_action_trace_exact": all(
            np.array_equal(
                value["executed_actions"],
                value["expected_executed_actions"],
            )
            for value in ordered
        ),
        "analytical_raw_state_trace_exact": all(
            np.allclose(
                value["raw_states"],
                value["expected_raw_states"],
                atol=1e-6,
            )
            for value in ordered
        ),
        "analytical_future_state_exact": all(
            np.allclose(
                rollouts[delay]["future_states"],
                expected_future[delay],
                atol=1e-6,
            )
            for delay in DELAYS
        ),
        "future_physical_group_counts_exact": (
            state_group_counts == expected_group_counts
            and pixel_group_counts == expected_group_counts
        ),
        "horizon1_delays_five_through_ten_are_equivalent": (
            _unique_array_count(
                rollouts[delay]["future_states"][0]
                for delay in range(5, 11)
            )
            == 1
            and _unique_array_count(
                rollouts[delay]["future_pixels"][0]
                for delay in range(5, 11)
            )
            == 1
        ),
        "horizon2_and_horizon3_identify_all_delays": all(
            state_group_counts[str(horizon)] == len(DELAYS)
            and pixel_group_counts[str(horizon)] == len(DELAYS)
            for horizon in (2, 3)
        ),
        "no_collision_or_early_termination": not any(
            value["terminated_or_truncated"] for value in ordered
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "future_state_group_counts": state_group_counts,
        "future_pixel_group_counts": pixel_group_counts,
        "expected_future_states_by_delay": {
            str(delay): expected_future[delay].tolist() for delay in DELAYS
        },
    }


def build_validation_asset(
    assignment: ActionDelayH7ValidationAssignment,
    *,
    agent_speed: float,
    action_magnitude: float,
    maximum_delay_steps: int = 10,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rollouts = {
        delay: simulate_template(
            assignment.template,
            history_tokens=HISTORY_TOKENS,
            delay_steps=delay,
            agent_speed=agent_speed,
            action_magnitude=action_magnitude,
            maximum_delay_steps=maximum_delay_steps,
        )
        for delay in DELAYS
    }
    physical = audit_rollout_family(
        assignment,
        rollouts,
        agent_speed=agent_speed,
        action_magnitude=action_magnitude,
    )
    if not physical["passed"]:
        failed = [
            name
            for name, passed in physical["checks"].items()
            if not passed
        ]
        raise RuntimeError(
            f"H7 Validation physics failed for {assignment.query_id}: {failed}"
        )
    shared_action_blocks = np.asarray(
        rollouts[0]["action_blocks"],
        dtype=np.float32,
    )
    pending = np.zeros((len(DELAYS), max(DELAYS), 2), dtype=np.float32)
    pending_lengths = np.zeros(len(DELAYS), dtype=np.int64)
    for delay in DELAYS:
        value = np.asarray(
            rollouts[delay]["pending_actions_at_query"],
            dtype=np.float32,
        )
        pending_lengths[delay] = len(value)
        if len(value):
            pending[delay, : len(value)] = value
    arrays = {
        "history_delays": np.asarray(DELAYS, dtype=np.int64),
        "target_delays": np.asarray(DELAYS, dtype=np.int64),
        "history_pixels": np.stack(
            [rollouts[delay]["history_pixels"] for delay in DELAYS]
        ).astype(np.uint8),
        "action_blocks": shared_action_blocks,
        "true_future_pixels": np.stack(
            [rollouts[delay]["future_pixels"] for delay in DELAYS]
        ).astype(np.uint8),
        "query_pixels": np.asarray(
            rollouts[0]["history_pixels"][-1],
            dtype=np.uint8,
        ),
        "goal_state": np.asarray(
            assignment.template.goal_state,
            dtype=np.float32,
        ),
        "audit_history_states": np.stack(
            [rollouts[delay]["history_states"] for delay in DELAYS]
        ).astype(np.float32),
        "audit_true_future_states": np.stack(
            [rollouts[delay]["future_states"] for delay in DELAYS]
        ).astype(np.float32),
        "audit_raw_states": np.stack(
            [rollouts[delay]["raw_states"] for delay in DELAYS]
        ).astype(np.float32),
        "audit_executed_actions": np.stack(
            [rollouts[delay]["executed_actions"] for delay in DELAYS]
        ).astype(np.float32),
        "audit_pending_actions_at_query": pending,
        "audit_pending_action_lengths": pending_lengths,
    }
    if tuple(sorted(arrays)) != tuple(sorted(ARRAY_KEYS)):
        raise RuntimeError("H7 Validation payload keys changed")
    array_hashes, payload_sha256 = _payload_hashes(arrays)
    audit = {
        "query_id": assignment.query_id,
        "eval_seed": assignment.eval_seed,
        "evaluation_index": assignment.evaluation_index,
        "room": assignment.room,
        "direction": assignment.template.direction,
        "template": asdict(assignment.template),
        "template_sha256": canonical_sha256(asdict(assignment.template)),
        "physical": physical,
        "array_sha256": array_hashes,
        "payload_sha256": payload_sha256,
        "query_pixels_sha256": array_hashes["query_pixels"],
        "initial_pixels_sha256": array_sha256(
            arrays["history_pixels"][0, 0]
        ),
    }
    return arrays, audit


def _initialize_asset_worker(
    repo_root: str,
    configured_repo: str,
    stable_commit: str,
) -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    load_stable_worldmodel(
        Path(repo_root),
        configured_repo,
        stable_commit,
    )


def _build_asset_worker(job: _AssetBuildJob) -> dict[str, Any]:
    arrays, audit = build_validation_asset(
        job.assignment,
        agent_speed=job.agent_speed,
        action_magnitude=job.action_magnitude,
        maximum_delay_steps=job.maximum_delay_steps,
    )
    _atomic_savez(job.asset_path, arrays)
    reopened = _load_npz(job.asset_path)
    reopened_hashes, reopened_payload = _payload_hashes(reopened)
    return {
        **audit,
        "asset": portable_contextworld_path(
            job.asset_path,
            repo_root=job.repo_root,
        ),
        "asset_sha256": file_sha256(job.asset_path),
        "asset_reopens": set(reopened) == set(arrays),
        "asset_array_hashes_match": reopened_hashes
        == audit["array_sha256"],
        "asset_payload_hash_matches": reopened_payload
        == audit["payload_sha256"],
    }


def audit_catalog_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(int(row["eval_seed"]) for row in rows)
    directions = {
        seed: Counter(
            row["direction"]
            for row in rows
            if int(row["eval_seed"]) == seed
        )
        for seed in EVAL_SEEDS
    }
    rooms = {
        seed: Counter(
            row["room"]
            for row in rows
            if int(row["eval_seed"]) == seed
        )
        for seed in EVAL_SEEDS
    }
    delay_counts = Counter()
    for row in rows:
        for delay in DELAYS:
            delay_counts[delay] += 1
    checks = {
        "exact_300_distinct_queries": len(rows) == QUERY_COUNT,
        "exact_50_queries_per_eval_seed": counts
        == Counter({seed: QUERIES_PER_SEED for seed in EVAL_SEEDS}),
        "directions_balanced_per_eval_seed": all(
            directions[seed] == Counter({"up": 25, "down": 25})
            for seed in EVAL_SEEDS
        ),
        "rooms_balanced_per_eval_seed": all(
            rooms[seed] == Counter({"left": 25, "right": 25})
            for seed in EVAL_SEEDS
        ),
        "query_ids_unique": len({row["query_id"] for row in rows})
        == len(rows),
        "template_ids_unique": len(
            {row["template"]["template_id"] for row in rows}
        )
        == len(rows),
        "simulator_seeds_unique": len(
            {int(row["template"]["simulator_seed"]) for row in rows}
        )
        == len(rows),
        "reset_states_unique": len(
            {tuple(row["template"]["reset_state"]) for row in rows}
        )
        == len(rows),
        "query_pixels_unique": len(
            {row["query_pixels_sha256"] for row in rows}
        )
        == len(rows),
        "payloads_unique": len(
            {row["payload_sha256"] for row in rows}
        )
        == len(rows),
        "each_delay_has_full_300_queries": delay_counts
        == Counter({delay: QUERY_COUNT for delay in DELAYS}),
        "every_physical_family_passed": all(
            row["physical"]["passed"]
            and all(row["physical"]["checks"].values())
            for row in rows
        ),
        "every_asset_reopens": all(row["asset_reopens"] for row in rows),
        "every_asset_array_hash_matches": all(
            row["asset_array_hashes_match"] for row in rows
        ),
        "every_asset_payload_hash_matches": all(
            row["asset_payload_hash_matches"] for row in rows
        ),
        "horizon_group_counts_are_six_eleven_eleven": all(
            row["physical"]["future_state_group_counts"]
            == {"1": 6, "2": 11, "3": 11}
            and row["physical"]["future_pixel_group_counts"]
            == {"1": 6, "2": 11, "3": 11}
            for row in rows
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "queries_by_eval_seed": {
            str(seed): counts[seed] for seed in EVAL_SEEDS
        },
        "queries_by_delay": {
            str(delay): delay_counts[delay] for delay in DELAYS
        },
        "directions_by_eval_seed": {
            str(seed): dict(sorted(directions[seed].items()))
            for seed in EVAL_SEEDS
        },
        "rooms_by_eval_seed": {
            str(seed): dict(sorted(rooms[seed].items()))
            for seed in EVAL_SEEDS
        },
    }


def build_validation_release(
    *,
    config: dict[str, Any],
    repo_root: Path,
    output_root: Path,
    workers: int,
    stable_worldmodel_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the formal 300-query, 11-delay offline Validation release."""

    protocol_checks = _validate_protocol(config)
    if workers < 1:
        raise ValueError("workers must be positive")
    output_root.mkdir(parents=True, exist_ok=False)
    assets_root = output_root / "assets"
    assets_root.mkdir()
    assignments = select_validation_assignments(config)
    agent_speed = float(config["environment"]["agent_speed"])
    action_magnitude = float(config["history_protocol"]["action_magnitude"])
    maximum_delay_steps = int(config["environment"]["delay_range"][1])
    jobs = [
        _AssetBuildJob(
            assignment=assignment,
            asset_path=assets_root / f"{assignment.query_id}.npz",
            repo_root=repo_root,
            agent_speed=agent_speed,
            action_magnitude=action_magnitude,
            maximum_delay_steps=maximum_delay_steps,
        )
        for assignment in assignments
    ]
    if workers == 1:
        rows = []
        for index, job in enumerate(jobs, start=1):
            rows.append(_build_asset_worker(job))
            if index % 25 == 0 or index == len(jobs):
                print(
                    f"[action-delay-h7] built {index}/{len(jobs)} queries",
                    flush=True,
                )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_asset_worker,
            initargs=(
                str(repo_root),
                str(config["stable_worldmodel"]["repo"]),
                stable_worldmodel_commit,
            ),
        ) as executor:
            rows = []
            for index, row in enumerate(
                executor.map(_build_asset_worker, jobs, chunksize=1),
                start=1,
            ):
                rows.append(row)
                if index % 25 == 0 or index == len(jobs):
                    print(
                        f"[action-delay-h7] built "
                        f"{index}/{len(jobs)} queries",
                        flush=True,
                    )
    rows.sort(key=lambda row: (row["eval_seed"], row["evaluation_index"]))
    audit = audit_catalog_rows(rows)
    if not audit["passed"]:
        failed = [
            name for name, passed in audit["checks"].items() if not passed
        ]
        raise RuntimeError(f"H7 Validation release audit failed: {failed}")

    counts = {
        "distinct_queries": QUERY_COUNT,
        "queries_per_eval_seed": QUERIES_PER_SEED,
        "independent_queries_per_delay": QUERY_COUNT,
        "delay_conditions_per_query": len(DELAYS),
        "physical_rollouts": QUERY_COUNT * len(DELAYS),
        "history_conditioned_model_predictions_per_checkpoint": (
            QUERY_COUNT * len(DELAYS)
        ),
        "real_future_trajectories": QUERY_COUNT * len(DELAYS),
        "real_future_frames": (
            QUERY_COUNT * len(DELAYS) * len(FUTURE_HORIZONS)
        ),
        "aggregate_three_step_target_comparisons_per_checkpoint": (
            QUERY_COUNT * len(DELAYS) * len(DELAYS)
        ),
        "per_horizon_target_loss_records_per_checkpoint": (
            QUERY_COUNT
            * len(DELAYS)
            * len(DELAYS)
            * len(FUTURE_HORIZONS)
        ),
    }
    protocol = {
        "history_tokens": HISTORY_TOKENS,
        "context_transitions": HISTORY_TRANSITIONS,
        "raw_steps_per_action_block": ACTION_BLOCK,
        "agent_speed": agent_speed,
        "action_magnitude": action_magnitude,
        "delay_values": list(DELAYS),
        "future_horizons_action_blocks": list(FUTURE_HORIZONS),
        "eval_seeds": list(EVAL_SEEDS),
        "queries_per_eval_seed": QUERIES_PER_SEED,
        "model_visible_fields": list(MODEL_VISIBLE_FIELDS),
        "privileged_audit_arrays": [
            name for name in ARRAY_KEYS if name.startswith("audit_")
        ],
        "paired_query_rule": (
            "the same 300 distinct query geometries are evaluated under "
            "every delay; each delay therefore has all 300 queries"
        ),
    }
    query_projection = [
        {
            key: row[key]
            for key in (
                "query_id",
                "eval_seed",
                "evaluation_index",
                "room",
                "direction",
                "template",
                "template_sha256",
                "asset",
                "asset_sha256",
                "payload_sha256",
                "array_sha256",
                "query_pixels_sha256",
                "initial_pixels_sha256",
            )
        }
        for row in rows
    ]
    content_projection = {
        "benchmark": config["benchmark"],
        "protocol": protocol,
        "queries": query_projection,
    }
    content_sha256 = canonical_sha256(content_projection)
    catalog = {
        "schema_version": 1,
        "status": "frozen_before_model_scoring",
        **content_projection,
        "content_manifest_sha256": content_sha256,
        "counts": counts,
    }
    catalog_path = output_root / "catalog.json"
    write_json(catalog_path, catalog)
    exclusion = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "purpose": (
            "future training builders must reject every listed template, "
            "simulator seed, reset state, and initial/query pixel hash"
        ),
        "content_manifest_sha256": content_sha256,
        "query_count": QUERY_COUNT,
        "query_records": [
            {
                "query_id": row["query_id"],
                "eval_seed": row["eval_seed"],
                "template_id": row["template"]["template_id"],
                "template_sha256": row["template_sha256"],
                "simulator_seed": row["template"]["simulator_seed"],
                "reset_state": row["template"]["reset_state"],
                "goal_state": row["template"]["goal_state"],
                "initial_pixels_sha256": row["initial_pixels_sha256"],
                "query_pixels_sha256": row["query_pixels_sha256"],
                "payload_sha256": row["payload_sha256"],
            }
            for row in rows
        ],
    }
    exclusion_path = output_root / "training_exclusion_manifest.json"
    write_json(exclusion_path, exclusion)
    report_checks = {
        **protocol_checks,
        **audit["checks"],
        "catalog_content_hash_exact": catalog["content_manifest_sha256"]
        == canonical_sha256(
            {
                "benchmark": catalog["benchmark"],
                "protocol": catalog["protocol"],
                "queries": catalog["queries"],
            }
        ),
        "training_exclusion_covers_every_query": (
            len(exclusion["query_records"]) == QUERY_COUNT
            and {
                row["query_id"] for row in exclusion["query_records"]
            }
            == {row["query_id"] for row in rows}
        ),
        "model_projection_is_pixels_and_actions_only": MODEL_VISIBLE_FIELDS
        == ("pixels", "action"),
    }
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed" if all(report_checks.values()) else "failed",
        "scope": (
            "offline_validation_data_and_physics_only; "
            "no_model_was_trained_or_scored"
        ),
        "checks": report_checks,
        "audit": audit,
        "counts": counts,
        "physical_equivalence": {
            "horizon1": {
                "distinct_groups": 6,
                "equivalent_delays": [5, 6, 7, 8, 9, 10],
            },
            "horizon2": {"distinct_groups": 11},
            "horizon3": {"distinct_groups": 11},
        },
        "content_manifest_sha256": content_sha256,
        "catalog": portable_contextworld_path(
            catalog_path,
            repo_root=repo_root,
        ),
        "catalog_sha256": file_sha256(catalog_path),
        "training_exclusion_manifest": portable_contextworld_path(
            exclusion_path,
            repo_root=repo_root,
        ),
        "training_exclusion_manifest_sha256": file_sha256(exclusion_path),
    }
    write_json(output_root / "build_report.json", report)
    return catalog, exclusion, report


def audit_validation_release(
    *,
    config: dict[str, Any],
    repo_root: Path,
    catalog_path: Path,
    replay_physics: bool,
) -> dict[str, Any]:
    """Reopen every asset and optionally replay all 3,300 trajectories."""

    _validate_protocol(config)
    catalog_path = Path(catalog_path).resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    output_root = catalog_path.parent
    exclusion_path = output_root / "training_exclusion_manifest.json"
    exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    assignments = {
        value.query_id: value
        for value in select_validation_assignments(config)
    }
    failures: list[dict[str, Any]] = []
    replayed = 0
    agent_speed = float(config["environment"]["agent_speed"])
    action_magnitude = float(config["history_protocol"]["action_magnitude"])
    maximum_delay_steps = int(config["environment"]["delay_range"][1])
    for index, row in enumerate(catalog["queries"], start=1):
        query_id = str(row["query_id"])
        reasons: list[str] = []
        assignment = assignments.get(query_id)
        if assignment is None:
            reasons.append("query_not_in_frozen_assignment")
        elif canonical_sha256(asdict(assignment.template)) != row.get(
            "template_sha256"
        ):
            reasons.append("template_changed")
        asset_path = resolve_contextworld_path(
            row["asset"],
            repo_root=repo_root,
        )
        if not asset_path.is_file():
            reasons.append("asset_missing")
            arrays = {}
        else:
            if file_sha256(asset_path) != row["asset_sha256"]:
                reasons.append("asset_file_hash_mismatch")
            arrays = _load_npz(asset_path)
            array_hashes, payload_hash = _payload_hashes(arrays)
            if tuple(sorted(arrays)) != tuple(sorted(ARRAY_KEYS)):
                reasons.append("array_keys_changed")
            if array_hashes != row["array_sha256"]:
                reasons.append("array_hash_mismatch")
            if payload_hash != row["payload_sha256"]:
                reasons.append("payload_hash_mismatch")
        if replay_physics and assignment is not None and arrays:
            replay_arrays, replay_audit = build_validation_asset(
                assignment,
                agent_speed=agent_speed,
                action_magnitude=action_magnitude,
                maximum_delay_steps=maximum_delay_steps,
            )
            replayed += len(DELAYS)
            if not replay_audit["physical"]["passed"]:
                reasons.append("physical_replay_failed")
            if set(replay_arrays) != set(arrays) or any(
                not np.array_equal(replay_arrays[name], arrays[name])
                for name in replay_arrays
            ):
                reasons.append("physical_replay_differs_from_asset")
        if reasons:
            failures.append({"query_id": query_id, "reasons": reasons})
        if index % 25 == 0 or index == len(catalog["queries"]):
            mode = "replayed" if replay_physics else "reopened"
            print(
                f"[action-delay-h7-audit] {mode} "
                f"{index}/{len(catalog['queries'])} queries",
                flush=True,
            )

    content_projection = {
        "benchmark": catalog.get("benchmark"),
        "protocol": catalog.get("protocol"),
        "queries": catalog.get("queries"),
    }
    exclusion_ids = {
        row["query_id"] for row in exclusion.get("query_records", ())
    }
    catalog_ids = {row["query_id"] for row in catalog.get("queries", ())}
    checks = {
        "catalog_status_frozen_before_model_scoring": catalog.get("status")
        == "frozen_before_model_scoring",
        "catalog_has_exact_300_queries": len(catalog.get("queries", ()))
        == QUERY_COUNT,
        "catalog_content_hash_exact": canonical_sha256(content_projection)
        == catalog.get("content_manifest_sha256"),
        "training_exclusion_content_identity_exact": exclusion.get(
            "content_manifest_sha256"
        )
        == catalog.get("content_manifest_sha256"),
        "training_exclusion_query_ids_exact": exclusion_ids == catalog_ids,
        "every_asset_hash_and_payload_reopens": not failures,
        "full_physical_replay_completed": (
            replayed == QUERY_COUNT * len(DELAYS)
            if replay_physics
            else True
        ),
    }
    return {
        "schema_version": 1,
        "benchmark": catalog.get("benchmark"),
        "status": "passed" if all(checks.values()) else "failed",
        "mode": (
            "full_physical_replay"
            if replay_physics
            else "asset_integrity_only"
        ),
        "checks": checks,
        "counts": {
            "catalog_queries": len(catalog.get("queries", ())),
            "assets_reopened": len(catalog.get("queries", ())),
            "physical_rollouts_replayed": replayed,
            "failures": len(failures),
        },
        "failures": failures,
        "catalog": portable_contextworld_path(
            catalog_path,
            repo_root=repo_root,
        ),
        "catalog_sha256": file_sha256(catalog_path),
        "training_exclusion_manifest": portable_contextworld_path(
            exclusion_path,
            repo_root=repo_root,
        ),
        "training_exclusion_manifest_sha256": file_sha256(exclusion_path),
    }


__all__ = [
    "ARRAY_KEYS",
    "DELAYS",
    "EVAL_SEEDS",
    "FUTURE_HORIZONS",
    "HISTORY_TOKENS",
    "QUERY_COUNT",
    "QUERIES_PER_SEED",
    "ActionDelayH7ValidationAssignment",
    "audit_catalog_rows",
    "audit_rollout_family",
    "audit_validation_release",
    "build_validation_asset",
    "build_validation_release",
    "file_sha256",
    "select_validation_assignments",
]
