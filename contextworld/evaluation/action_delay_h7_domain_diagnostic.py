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
from .action_delay_h7_data import (
    MULTI_DELAYS,
    training_template,
)
from .action_delay_h7_validation import file_sha256
from .action_delay_long_history import (
    ACTION_BLOCK,
    LongHistoryDelayTemplate,
    simulate_template,
)


HISTORY_TOKENS = 7
FUTURE_HORIZONS = (1, 2, 3)
DELAYS = MULTI_DELAYS
SOURCE_SPLITS = ("train", "val")
DIAGNOSTIC_EVAL_SEEDS = (142, 143, 144, 145, 146, 147)
QUERIES_PER_SEED = 50
QUERIES_PER_TRACK = len(DIAGNOSTIC_EVAL_SEEDS) * QUERIES_PER_SEED
TRACK_COUNT = len(SOURCE_SPLITS) * len(DELAYS)
QUERY_COUNT = QUERIES_PER_TRACK * TRACK_COUNT
PHYSICAL_ROLLOUTS = QUERY_COUNT * len(DELAYS)
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
class DomainDiagnosticAssignment:
    query_id: str
    track: str
    source_split: str
    source_delay: int
    diagnostic_eval_seed: int
    evaluation_index: int
    room: str
    source_shard_index: int
    source_episode_index: int
    source_table: Path
    template: LongHistoryDelayTemplate


@dataclass(frozen=True)
class _AssetBuildJob:
    assignment: DomainDiagnosticAssignment
    asset_path: Path
    repo_root: Path
    agent_speed: float
    action_magnitude: float
    maximum_delay_steps: int


def track_name(source_split: str, source_delay: int) -> str:
    prefix = (
        "training_replay"
        if source_split == "train"
        else "loader_validation"
    )
    return f"{prefix}_delay_{int(source_delay)}"


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


def _validate_config(
    config: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, bool]:
    protocol = config["protocol"]
    sampling = config["sampling"]
    checks = {
        "history_tokens_are_seven": int(protocol["history_tokens"])
        == HISTORY_TOKENS,
        "action_block_is_five": int(
            protocol["raw_steps_per_action_block"]
        )
        == ACTION_BLOCK,
        "future_horizons_are_one_two_three": tuple(
            map(int, protocol["future_horizons_action_blocks"])
        )
        == FUTURE_HORIZONS,
        "delays_are_zero_four_eight": tuple(
            map(int, protocol["delay_values"])
        )
        == DELAYS,
        "diagnostic_seeds_are_frozen": tuple(
            map(int, sampling["diagnostic_eval_seeds"])
        )
        == DIAGNOSTIC_EVAL_SEEDS,
        "fifty_queries_per_seed": int(
            sampling["queries_per_eval_seed_per_track"]
        )
        == QUERIES_PER_SEED,
        "three_hundred_queries_per_track": int(
            sampling["distinct_queries_per_track"]
        )
        == QUERIES_PER_TRACK,
        "six_tracks": tuple(config["tracks"]["track_names"])
        == tuple(
            track_name(split, delay)
            for split in SOURCE_SPLITS
            for delay in DELAYS
        ),
        "training_catalog_seed_available": int(
            training_config["catalog_seed"]
        )
        > 0,
        "source_training_delays_match": tuple(
            map(
                int,
                training_config["protocol"]["training_delay_values"],
            )
        )
        == DELAYS,
        "model_fields_are_pixels_and_action": tuple(
            protocol["model_visible_fields"]
        )
        == MODEL_VISIBLE_FIELDS,
        "offline_scoring_only": bool(
            protocol["offline_real_future_pixels_required"]
        )
        and not bool(
            protocol["online_environment_during_model_scoring"]
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Invalid H7 domain diagnostic: {failed}")
    return checks


def _source_tables(
    catalog: dict[str, Any],
    *,
    source_split: str,
    repo_root: Path,
) -> list[Path]:
    rows = (
        catalog["train"]["synthetic"]
        if source_split == "train"
        else catalog["val"]["synthetic"]
    )
    return [
        resolve_contextworld_path(value, repo_root=repo_root)
        for value in rows
    ]


def _candidate_pool(
    *,
    training_config: dict[str, Any],
    training_catalog: dict[str, Any],
    repo_root: Path,
    source_split: str,
    source_delay: int,
    selection_seed: int,
) -> dict[tuple[str, str], list[tuple[int, int, Path, LongHistoryDelayTemplate]]]:
    shard_count = int(
        training_config["counts"][source_split]["shards"]
    )
    episodes_per_shard = int(
        training_config["counts"][source_split]["episodes_per_shard"]
    )
    tables = _source_tables(
        training_catalog,
        source_split=source_split,
        repo_root=repo_root,
    )
    if len(tables) != shard_count:
        raise ValueError(
            f"{source_split}: catalog has {len(tables)} tables, "
            f"expected {shard_count}"
        )
    values: dict[
        tuple[str, str],
        list[tuple[int, int, Path, LongHistoryDelayTemplate]],
    ] = {
        (room, direction): []
        for room in ("left", "right")
        for direction in ("up", "down")
    }
    catalog_seed = int(training_config["catalog_seed"])
    for shard_index in range(shard_count):
        delay = DELAYS[shard_index % len(DELAYS)]
        if delay != int(source_delay):
            continue
        table = tables[shard_index]
        if not table.is_dir():
            raise FileNotFoundError(table)
        for episode_index in range(episodes_per_shard):
            template = training_template(
                catalog_seed=catalog_seed,
                split=source_split,
                shard_index=shard_index,
                episode_index=episode_index,
            )
            room = (
                "left"
                if float(template.reset_state[0]) < 112.0
                else "right"
            )
            values[(room, template.direction)].append(
                (
                    shard_index,
                    episode_index,
                    table,
                    template,
                )
            )
    split_index = SOURCE_SPLITS.index(source_split)
    delay_index = DELAYS.index(int(source_delay))
    for room_index, room in enumerate(("left", "right")):
        for direction_index, direction in enumerate(("up", "down")):
            key = (room, direction)
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        int(selection_seed),
                        split_index,
                        delay_index,
                        room_index,
                        direction_index,
                    ]
                )
            )
            permutation = rng.permutation(len(values[key]))
            values[key] = [values[key][index] for index in permutation]
            if len(values[key]) < 75:
                raise RuntimeError(
                    f"Insufficient source candidates for {source_split}/"
                    f"{source_delay}/{key}: {len(values[key])}"
                )
    return values


def select_domain_assignments(
    *,
    config: dict[str, Any],
    training_config: dict[str, Any],
    training_catalog: dict[str, Any],
    repo_root: Path,
) -> list[DomainDiagnosticAssignment]:
    """Select six independent 50x6 tracks from the frozen source release."""

    _validate_config(config, training_config)
    selection_seed = int(config["sampling"]["selection_seed"])
    assignments: list[DomainDiagnosticAssignment] = []
    used_templates: set[str] = set()
    used_reset_states: set[tuple[float, float]] = set()
    cells = (
        ("left", "up"),
        ("left", "down"),
        ("right", "up"),
        ("right", "down"),
    )
    for source_split in SOURCE_SPLITS:
        for source_delay in DELAYS:
            pools = _candidate_pool(
                training_config=training_config,
                training_catalog=training_catalog,
                repo_root=repo_root,
                source_split=source_split,
                source_delay=source_delay,
                selection_seed=selection_seed,
            )
            cursors: Counter[tuple[str, str]] = Counter()
            track = track_name(source_split, source_delay)
            for seed_index, diagnostic_seed in enumerate(
                DIAGNOSTIC_EVAL_SEEDS
            ):
                cell_counts = (
                    (13, 12, 12, 13)
                    if seed_index % 2 == 0
                    else (12, 13, 13, 12)
                )
                selected_for_seed: list[
                    tuple[
                        str,
                        str,
                        int,
                        int,
                        Path,
                        LongHistoryDelayTemplate,
                    ]
                ] = []
                for (room, direction), count in zip(
                    cells, cell_counts, strict=True
                ):
                    start = cursors[(room, direction)]
                    stop = start + count
                    values = pools[(room, direction)][start:stop]
                    if len(values) != count:
                        raise RuntimeError(
                            f"Candidate pool exhausted for {track}/"
                            f"{diagnostic_seed}/{room}/{direction}"
                        )
                    cursors[(room, direction)] = stop
                    selected_for_seed.extend(
                        (
                            room,
                            direction,
                            shard_index,
                            episode_index,
                            table,
                            template,
                        )
                        for (
                            shard_index,
                            episode_index,
                            table,
                            template,
                        ) in values
                    )
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            selection_seed,
                            SOURCE_SPLITS.index(source_split),
                            DELAYS.index(source_delay),
                            int(diagnostic_seed),
                            0xD0A1,
                        ]
                    )
                )
                permutation = rng.permutation(len(selected_for_seed))
                selected_for_seed = [
                    selected_for_seed[index] for index in permutation
                ]
                for evaluation_index, value in enumerate(
                    selected_for_seed
                ):
                    (
                        room,
                        direction,
                        shard_index,
                        episode_index,
                        table,
                        template,
                    ) = value
                    if template.direction != direction:
                        raise RuntimeError("Source direction changed")
                    reset_state = tuple(map(float, template.reset_state))
                    if template.template_id in used_templates:
                        raise RuntimeError("Source template repeated")
                    if reset_state in used_reset_states:
                        raise RuntimeError("Source reset state repeated")
                    used_templates.add(template.template_id)
                    used_reset_states.add(reset_state)
                    query_id = (
                        f"ad-h7-domain-{source_split}-d{source_delay}-"
                        f"s{diagnostic_seed}-q{evaluation_index:02d}"
                    )
                    assignments.append(
                        DomainDiagnosticAssignment(
                            query_id=query_id,
                            track=track,
                            source_split=source_split,
                            source_delay=int(source_delay),
                            diagnostic_eval_seed=int(diagnostic_seed),
                            evaluation_index=evaluation_index,
                            room=room,
                            source_shard_index=int(shard_index),
                            source_episode_index=int(episode_index),
                            source_table=table,
                            template=template,
                        )
                    )
    if len(assignments) != QUERY_COUNT:
        raise RuntimeError(
            f"Expected {QUERY_COUNT} assignments, got {len(assignments)}"
        )
    return assignments


def _physical_audit(
    assignment: DomainDiagnosticAssignment,
    rollouts: dict[int, dict[str, Any]],
    *,
    agent_speed: float,
    action_magnitude: float,
) -> dict[str, Any]:
    if tuple(sorted(rollouts)) != DELAYS:
        raise ValueError(f"Expected delay family {DELAYS}")
    ordered = [rollouts[delay] for delay in DELAYS]
    reset_state = np.asarray(
        assignment.template.reset_state,
        dtype=np.float32,
    )
    sign = 1.0 if assignment.template.direction == "up" else -1.0
    direction = np.asarray([0.0, sign], dtype=np.float32)
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
    history_states = [value["history_states"] for value in ordered]
    history_pixels = [value["history_pixels"] for value in ordered]
    query_states = [value["history_states"][-1] for value in ordered]
    query_pixels = [value["history_pixels"][-1] for value in ordered]
    state_groups = {
        str(horizon): _unique_array_count(
            value["future_states"][horizon - 1] for value in ordered
        )
        for horizon in FUTURE_HORIZONS
    }
    pixel_groups = {
        str(horizon): _unique_array_count(
            value["future_pixels"][horizon - 1] for value in ordered
        )
        for horizon in FUTURE_HORIZONS
    }
    checks = {
        "delay_readback_exact": all(
            int(value["delay_steps"]) == delay
            for delay, value in zip(DELAYS, ordered, strict=True)
        ),
        "initial_state_and_pixels_identical": (
            _unique_array_count(value[0] for value in history_states) == 1
            and _unique_array_count(value[0] for value in history_pixels)
            == 1
        ),
        "history_actions_identical": _unique_array_count(
            value["action_blocks"] for value in ordered
        )
        == 1,
        "three_history_trajectories_distinct": (
            _unique_array_count(history_states) == len(DELAYS)
            and _unique_array_count(history_pixels) == len(DELAYS)
        ),
        "query_state_and_pixels_identical": (
            _unique_array_count(query_states) == 1
            and _unique_array_count(query_pixels) == 1
        ),
        "query_returns_to_source_reset_state": all(
            np.array_equal(value, reset_state) for value in query_states
        ),
        "pending_action_queue_empty": all(
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
        "all_horizons_distinguish_three_delays": (
            state_groups == {"1": 3, "2": 3, "3": 3}
            and pixel_groups == {"1": 3, "2": 3, "3": 3}
        ),
        "no_collision_or_early_termination": not any(
            value["terminated_or_truncated"] for value in ordered
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "future_state_group_counts": state_groups,
        "future_pixel_group_counts": pixel_groups,
    }


def build_domain_asset(
    assignment: DomainDiagnosticAssignment,
    *,
    agent_speed: float,
    action_magnitude: float,
    maximum_delay_steps: int,
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
    physical = _physical_audit(
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
            f"{assignment.query_id}: physical audit failed: {failed}"
        )
    pending = np.zeros(
        (len(DELAYS), max(DELAYS), 2),
        dtype=np.float32,
    )
    pending_lengths = np.zeros(len(DELAYS), dtype=np.int64)
    for delay_index, delay in enumerate(DELAYS):
        value = np.asarray(
            rollouts[delay]["pending_actions_at_query"],
            dtype=np.float32,
        )
        pending_lengths[delay_index] = len(value)
        if len(value):
            pending[delay_index, : len(value)] = value
    arrays = {
        "history_delays": np.asarray(DELAYS, dtype=np.int64),
        "target_delays": np.asarray(DELAYS, dtype=np.int64),
        "history_pixels": np.stack(
            [rollouts[delay]["history_pixels"] for delay in DELAYS]
        ).astype(np.uint8),
        "action_blocks": np.asarray(
            rollouts[DELAYS[0]]["action_blocks"],
            dtype=np.float32,
        ),
        "true_future_pixels": np.stack(
            [rollouts[delay]["future_pixels"] for delay in DELAYS]
        ).astype(np.uint8),
        "query_pixels": np.asarray(
            rollouts[DELAYS[0]]["history_pixels"][-1],
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
        raise RuntimeError("H7 domain diagnostic payload keys changed")
    hashes, payload_sha256 = _payload_hashes(arrays)
    source_index = DELAYS.index(assignment.source_delay)
    source_transition = {
        "history_pixels_sha256": array_sha256(
            arrays["history_pixels"][source_index]
        ),
        "action_blocks_through_h1_sha256": array_sha256(
            arrays["action_blocks"][:HISTORY_TOKENS]
        ),
        "h1_true_future_pixels_sha256": array_sha256(
            arrays["true_future_pixels"][source_index, 0]
        ),
    }
    return arrays, {
        "query_id": assignment.query_id,
        "track": assignment.track,
        "source_split": assignment.source_split,
        "source_delay": assignment.source_delay,
        "diagnostic_eval_seed": assignment.diagnostic_eval_seed,
        "evaluation_index": assignment.evaluation_index,
        "room": assignment.room,
        "direction": assignment.template.direction,
        "source_shard_index": assignment.source_shard_index,
        "source_episode_index": assignment.source_episode_index,
        "source_table": assignment.source_table,
        "template": asdict(assignment.template),
        "template_sha256": canonical_sha256(
            asdict(assignment.template)
        ),
        "physical": physical,
        "array_sha256": hashes,
        "payload_sha256": payload_sha256,
        "query_pixels_sha256": hashes["query_pixels"],
        "initial_pixels_sha256": array_sha256(
            arrays["history_pixels"][0, 0]
        ),
        "source_supervised_transition": source_transition,
        "source_supervised_transition_sha256": canonical_sha256(
            source_transition
        ),
    }


def _initialize_asset_worker(
    repo_root: str,
    stablewm_repo: str,
    stablewm_commit: str,
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
        stablewm_repo,
        stablewm_commit,
    )


def _build_asset_worker(job: _AssetBuildJob) -> dict[str, Any]:
    arrays, audit = build_domain_asset(
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
        "source_table": portable_contextworld_path(
            audit["source_table"],
            repo_root=job.repo_root,
        ),
        "asset": portable_contextworld_path(
            job.asset_path,
            repo_root=job.repo_root,
        ),
        "asset_sha256": file_sha256(job.asset_path),
        "asset_reopens": set(reopened) == set(arrays),
        "asset_array_hashes_match": (
            reopened_hashes == audit["array_sha256"]
        ),
        "asset_payload_hash_matches": (
            reopened_payload == audit["payload_sha256"]
        ),
    }


def _catalog_audit(
    rows: list[dict[str, Any]],
    *,
    formal_validation_catalog: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    track_names = tuple(
        track_name(split, delay)
        for split in SOURCE_SPLITS
        for delay in DELAYS
    )
    counts_by_track = Counter(row["track"] for row in rows)
    counts_by_track_seed = {
        track: Counter(
            int(row["diagnostic_eval_seed"])
            for row in rows
            if row["track"] == track
        )
        for track in track_names
    }
    rooms = {
        (track, seed): Counter(
            row["room"]
            for row in rows
            if row["track"] == track
            and int(row["diagnostic_eval_seed"]) == seed
        )
        for track in track_names
        for seed in DIAGNOSTIC_EVAL_SEEDS
    }
    directions = {
        (track, seed): Counter(
            row["direction"]
            for row in rows
            if row["track"] == track
            and int(row["diagnostic_eval_seed"]) == seed
        )
        for track in track_names
        for seed in DIAGNOSTIC_EVAL_SEEDS
    }
    formal_resets = {
        tuple(map(float, row["template"]["reset_state"]))
        for row in formal_validation_catalog["queries"]
    }
    formal_query_hashes = {
        row["query_pixels_sha256"]
        for row in formal_validation_catalog["queries"]
    }
    selected_resets = {
        tuple(map(float, row["template"]["reset_state"])) for row in rows
    }
    selected_query_hashes = {
        row["query_pixels_sha256"] for row in rows
    }
    checks = {
        "exact_six_tracks": counts_by_track
        == Counter({track: QUERIES_PER_TRACK for track in track_names}),
        "exact_50_queries_per_seed_per_track": all(
            counts_by_track_seed[track]
            == Counter(
                {
                    seed: QUERIES_PER_SEED
                    for seed in DIAGNOSTIC_EVAL_SEEDS
                }
            )
            for track in track_names
        ),
        "rooms_balanced_per_seed_per_track": all(
            rooms[(track, seed)] == Counter({"left": 25, "right": 25})
            for track in track_names
            for seed in DIAGNOSTIC_EVAL_SEEDS
        ),
        "directions_balanced_per_seed_per_track": all(
            directions[(track, seed)]
            == Counter({"up": 25, "down": 25})
            for track in track_names
            for seed in DIAGNOSTIC_EVAL_SEEDS
        ),
        "query_ids_unique": len({row["query_id"] for row in rows})
        == QUERY_COUNT,
        "source_templates_unique": len(
            {row["template"]["template_id"] for row in rows}
        )
        == QUERY_COUNT,
        "source_reset_states_unique": len(selected_resets)
        == QUERY_COUNT,
        "source_tables_exist": all(
            resolve_contextworld_path(
                row["source_table"],
                repo_root=repo_root,
            ).is_dir()
            for row in rows
        ),
        "every_physical_family_passed": all(
            row["physical"]["passed"]
            and all(row["physical"]["checks"].values())
            for row in rows
        ),
        "every_asset_reopens": all(row["asset_reopens"] for row in rows),
        "every_asset_hash_matches": all(
            row["asset_array_hashes_match"]
            and row["asset_payload_hash_matches"]
            for row in rows
        ),
        "formal_validation_reset_overlap_zero": not (
            selected_resets & formal_resets
        ),
        "formal_validation_query_hash_overlap_zero": not (
            selected_query_hashes & formal_query_hashes
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "queries_by_track": dict(sorted(counts_by_track.items())),
        "queries_by_track_and_seed": {
            track: {
                str(seed): counts_by_track_seed[track][seed]
                for seed in DIAGNOSTIC_EVAL_SEEDS
            }
            for track in track_names
        },
        "formal_validation_overlap": {
            "reset_states": len(selected_resets & formal_resets),
            "query_pixel_hashes": len(
                selected_query_hashes & formal_query_hashes
            ),
        },
    }


def build_domain_diagnostic_release(
    *,
    config: dict[str, Any],
    training_config: dict[str, Any],
    training_catalog: dict[str, Any],
    formal_validation_catalog: dict[str, Any],
    repo_root: Path,
    output_root: Path,
    workers: int,
    stablewm_repo: str,
    stablewm_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checks = _validate_config(config, training_config)
    if workers < 1:
        raise ValueError("workers must be positive")
    output_root.mkdir(parents=True, exist_ok=False)
    assets_root = output_root / "assets"
    assets_root.mkdir()
    assignments = select_domain_assignments(
        config=config,
        training_config=training_config,
        training_catalog=training_catalog,
        repo_root=repo_root,
    )
    agent_speed = float(config["protocol"]["agent_speed"])
    action_magnitude = float(config["protocol"]["action_magnitude"])
    jobs = [
        _AssetBuildJob(
            assignment=assignment,
            asset_path=assets_root / f"{assignment.query_id}.npz",
            repo_root=repo_root,
            agent_speed=agent_speed,
            action_magnitude=action_magnitude,
            maximum_delay_steps=max(DELAYS),
        )
        for assignment in assignments
    ]
    if workers == 1:
        rows = [
            _build_asset_worker(job)
            for job in jobs
        ]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_asset_worker,
            initargs=(
                str(repo_root),
                stablewm_repo,
                stablewm_commit,
            ),
        ) as executor:
            rows = []
            for index, row in enumerate(
                executor.map(_build_asset_worker, jobs, chunksize=1),
                start=1,
            ):
                rows.append(row)
                if index % 100 == 0 or index == len(jobs):
                    print(
                        f"[h7-domain-data] built {index}/{len(jobs)}",
                        flush=True,
                    )
    rows.sort(
        key=lambda row: (
            row["track"],
            row["diagnostic_eval_seed"],
            row["evaluation_index"],
        )
    )
    audit = _catalog_audit(
        rows,
        formal_validation_catalog=formal_validation_catalog,
        repo_root=repo_root,
    )
    if not audit["passed"]:
        failed = [
            name for name, passed in audit["checks"].items() if not passed
        ]
        raise RuntimeError(f"H7 domain catalog audit failed: {failed}")
    row_projection = [
        {
            key: row[key]
            for key in (
                "query_id",
                "track",
                "source_split",
                "source_delay",
                "diagnostic_eval_seed",
                "evaluation_index",
                "room",
                "direction",
                "source_shard_index",
                "source_episode_index",
                "source_table",
                "template",
                "template_sha256",
                "asset",
                "asset_sha256",
                "array_sha256",
                "payload_sha256",
                "query_pixels_sha256",
                "initial_pixels_sha256",
                "source_supervised_transition",
                "source_supervised_transition_sha256",
            )
        }
        for row in rows
    ]
    protocol = {
        "history_tokens": HISTORY_TOKENS,
        "raw_steps_per_action_block": ACTION_BLOCK,
        "future_horizons_action_blocks": list(FUTURE_HORIZONS),
        "delay_values": list(DELAYS),
        "diagnostic_eval_seeds": list(DIAGNOSTIC_EVAL_SEEDS),
        "queries_per_eval_seed_per_track": QUERIES_PER_SEED,
        "queries_per_track": QUERIES_PER_TRACK,
        "track_count": TRACK_COUNT,
        "model_visible_fields": list(MODEL_VISIBLE_FIELDS),
        "online_environment_during_model_scoring": False,
    }
    content = {
        "benchmark": config["benchmark"],
        "protocol": protocol,
        "queries": row_projection,
    }
    catalog = {
        "schema_version": 1,
        "status": "frozen_before_model_scoring",
        **content,
        "content_manifest_sha256": canonical_sha256(content),
        "counts": {
            "distinct_queries": QUERY_COUNT,
            "tracks": TRACK_COUNT,
            "queries_per_track": QUERIES_PER_TRACK,
            "delay_conditions_per_query": len(DELAYS),
            "physical_rollouts": PHYSICAL_ROLLOUTS,
            "real_future_frames": (
                PHYSICAL_ROLLOUTS * len(FUTURE_HORIZONS)
            ),
            "model_predictions_per_checkpoint": PHYSICAL_ROLLOUTS,
            "target_encodings_per_checkpoint": (
                PHYSICAL_ROLLOUTS * len(FUTURE_HORIZONS)
            ),
        },
    }
    catalog_path = output_root / "catalog.json"
    write_json(catalog_path, catalog)
    report_checks = {
        **checks,
        **audit["checks"],
        "content_manifest_exact": canonical_sha256(content)
        == catalog["content_manifest_sha256"],
    }
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": (
            "passed" if all(report_checks.values()) else "failed"
        ),
        "scope": (
            "post_gate_training_domain_diagnostic_data_only; "
            "no_model_scoring"
        ),
        "checks": report_checks,
        "audit": audit,
        "counts": catalog["counts"],
        "catalog": portable_contextworld_path(
            catalog_path,
            repo_root=repo_root,
        ),
        "catalog_sha256": file_sha256(catalog_path),
        "content_manifest_sha256": catalog[
            "content_manifest_sha256"
        ],
    }
    write_json(output_root / "build_report.json", report)
    return catalog, report


def audit_domain_diagnostic_release(
    *,
    config: dict[str, Any],
    training_config: dict[str, Any],
    training_catalog: dict[str, Any],
    repo_root: Path,
    catalog_path: Path,
    replay_physics: bool,
) -> dict[str, Any]:
    _validate_config(config, training_config)
    catalog = json.loads(
        Path(catalog_path).read_text(encoding="utf-8")
    )
    assignments = {
        value.query_id: value
        for value in select_domain_assignments(
            config=config,
            training_config=training_config,
            training_catalog=training_catalog,
            repo_root=repo_root,
        )
    }
    failures: list[dict[str, Any]] = []
    replayed = 0
    for index, row in enumerate(catalog["queries"], start=1):
        query_id = str(row["query_id"])
        reasons: list[str] = []
        assignment = assignments.get(query_id)
        if assignment is None:
            reasons.append("assignment_missing")
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
            arrays: dict[str, np.ndarray] = {}
        else:
            if file_sha256(asset_path) != row["asset_sha256"]:
                reasons.append("asset_file_hash_changed")
            arrays = _load_npz(asset_path)
            hashes, payload = _payload_hashes(arrays)
            if tuple(sorted(arrays)) != tuple(sorted(ARRAY_KEYS)):
                reasons.append("array_keys_changed")
            if hashes != row["array_sha256"]:
                reasons.append("array_hash_changed")
            if payload != row["payload_sha256"]:
                reasons.append("payload_hash_changed")
        if replay_physics and assignment is not None and arrays:
            replay_arrays, replay_audit = build_domain_asset(
                assignment,
                agent_speed=float(config["protocol"]["agent_speed"]),
                action_magnitude=float(
                    config["protocol"]["action_magnitude"]
                ),
                maximum_delay_steps=max(DELAYS),
            )
            replayed += len(DELAYS)
            if not replay_audit["physical"]["passed"]:
                reasons.append("physical_replay_failed")
            if set(replay_arrays) != set(arrays) or any(
                not np.array_equal(replay_arrays[name], arrays[name])
                for name in replay_arrays
            ):
                reasons.append("physical_replay_differs")
        if reasons:
            failures.append(
                {"query_id": query_id, "reasons": reasons}
            )
        if index % 100 == 0 or index == len(catalog["queries"]):
            print(
                f"[h7-domain-audit] {index}/{len(catalog['queries'])}",
                flush=True,
            )
    content = {
        "benchmark": catalog.get("benchmark"),
        "protocol": catalog.get("protocol"),
        "queries": catalog.get("queries"),
    }
    checks = {
        "catalog_frozen_before_scoring": catalog.get("status")
        == "frozen_before_model_scoring",
        "catalog_has_exact_query_count": len(
            catalog.get("queries", ())
        )
        == QUERY_COUNT,
        "catalog_content_manifest_exact": canonical_sha256(content)
        == catalog.get("content_manifest_sha256"),
        "every_asset_reopens_exactly": not failures,
        "full_physical_replay_completed": (
            replayed == PHYSICAL_ROLLOUTS
            if replay_physics
            else True
        ),
    }
    return {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed" if all(checks.values()) else "failed",
        "mode": (
            "full_second_physical_replay"
            if replay_physics
            else "asset_integrity_only"
        ),
        "checks": checks,
        "counts": {
            "queries": len(catalog.get("queries", ())),
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
    }


__all__ = [
    "ARRAY_KEYS",
    "DELAYS",
    "DIAGNOSTIC_EVAL_SEEDS",
    "FUTURE_HORIZONS",
    "HISTORY_TOKENS",
    "PHYSICAL_ROLLOUTS",
    "QUERIES_PER_SEED",
    "QUERIES_PER_TRACK",
    "QUERY_COUNT",
    "SOURCE_SPLITS",
    "DomainDiagnosticAssignment",
    "audit_domain_diagnostic_release",
    "build_domain_asset",
    "build_domain_diagnostic_release",
    "select_domain_assignments",
    "track_name",
]
