from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.paths import resolve_contextworld_path

from .icl_model import file_sha256
from .icl_planning import QueryEpisode
from .planner_mechanism import array_sha256, spearman


ACTION_BLOCK = 5
AGENT_SPEED = 5.0
DOOR_SIZE = 14.0
WALL_CENTER_X = 112.0
VALID_DIRECTIONS = ("left_to_right", "right_to_left")
VALID_OFFSETS = (0, 20, 40)


@dataclass(frozen=True)
class DoorPlanningCell:
    catalog: dict[str, Any]
    catalog_path: Path
    catalog_sha256: str
    track: str
    door_position: int
    eval_seed: int
    task: str
    assets: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


def _required(mapping: dict[str, Any], key: str, *, where: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing {key!r} in {where}")
    return mapping[key]


def _factor(bundle: dict[str, Any], name: str) -> Any:
    factors = bundle.get("query_factors", {})
    if name in factors:
        return factors[name]
    short = name.split(".")[-1]
    if short in factors:
        return factors[short]
    aliases = {
        "agent.speed": "agent_speed",
        "door.position": "door_position",
    }
    alias = aliases.get(name)
    if alias is not None and alias in bundle:
        return bundle[alias]
    raise KeyError(f"Missing factor {name!r} in {bundle.get('query_id')}")


def _metadata(bundle: dict[str, Any], key: str) -> Any:
    if key in bundle:
        return bundle[key]
    template = bundle.get("template", {})
    if key in template:
        return template[key]
    raise KeyError(f"Missing {key!r} in {bundle.get('query_id')}")


def deterministic_cem_seed(
    *, eval_seed: int, evaluation_index: int, query_id: str
) -> int:
    """Return a stable per-query CEM seed independent of model identity."""

    digest = hashlib.sha256(str(query_id).encode("utf-8")).digest()
    query_word = int.from_bytes(digest[:4], "little", signed=False)
    return int(
        np.random.SeedSequence(
            [int(eval_seed), int(evaluation_index), query_word, 161803]
        ).generate_state(1)[0]
    )


def _assert_array_hash(
    bundle: dict[str, Any], key: str, value: np.ndarray
) -> str:
    field = f"{key}_sha256"
    expected = str(_required(bundle, field, where=bundle["query_id"]))
    actual = array_sha256(value)
    if actual != expected:
        raise RuntimeError(
            f"{bundle['query_id']} {key} hash mismatch: "
            f"expected {expected}, got {actual}"
        )
    return actual


def _validate_shapes(
    *,
    query_id: str,
    arrays: dict[str, np.ndarray],
    candidates: int,
    horizon: int,
) -> None:
    query = arrays["query_pixels"]
    goal = arrays["goal_pixels"]
    history = arrays["history_pixels"]
    if query.ndim != 3 or query.shape[-1] != 3 or query.dtype != np.uint8:
        raise ValueError(f"{query_id}: query_pixels must be uint8 HWC RGB")
    if goal.shape != query.shape or goal.dtype != np.uint8:
        raise ValueError(f"{query_id}: goal_pixels must match query_pixels")
    if (
        history.ndim != 4
        or history.shape[0] not in (2, 3)
        or history.shape[1:] != query.shape
        or history.dtype != np.uint8
    ):
        raise ValueError(
            f"{query_id}: history_pixels must contain two context frames, "
            "optionally followed by the repeated query frame"
        )
    if history.shape[0] == 3 and not np.array_equal(history[-1], query):
        raise ValueError(f"{query_id}: third history frame must equal query_pixels")
    if arrays["history_actions"].shape != (2, ACTION_BLOCK, 2):
        raise ValueError(
            f"{query_id}: history_actions must have shape (2,5,2)"
        )
    if arrays["query_state"].shape != (2,):
        raise ValueError(f"{query_id}: query_state must have shape (2,)")
    if arrays["goal_state"].shape != (2,):
        raise ValueError(f"{query_id}: goal_state must have shape (2,)")
    expected = (int(candidates), int(horizon) * ACTION_BLOCK, 2)
    if arrays["fixed_candidate_raw_actions"].shape != expected:
        raise ValueError(
            f"{query_id}: fixed_candidate_raw_actions must have shape "
            f"{expected}, got {arrays['fixed_candidate_raw_actions'].shape}"
        )
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"{query_id}: payload contains non-finite values")


def _cell_bundles(
    catalog: dict[str, Any],
    *,
    track: str,
    door_position: int,
    eval_seed: int,
    task: str,
) -> list[dict[str, Any]]:
    rows = []
    for bundle in catalog.get("bundles", []):
        if str(bundle.get("track")) != str(track):
            continue
        if int(bundle.get("eval_seed", -1)) != int(eval_seed):
            continue
        if str(bundle.get("task")) != str(task):
            continue
        if int(_factor(bundle, "door.position")) != int(door_position):
            continue
        rows.append(bundle)
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("evaluation_index", 10**9)),
            str(row.get("query_id")),
        ),
    )


def load_door_planning_cell(
    catalog_path: Path,
    *,
    repo_root: Path,
    track: str,
    door_position: int,
    eval_seed: int,
    task: str = "cross_room_navigation",
    candidates: int = 300,
    horizon: int = 10,
    expected_queries: int | None = 50,
) -> DoorPlanningCell:
    """Load and strictly validate one frozen door planning evaluation cell."""

    path = resolve_contextworld_path(catalog_path, repo_root=repo_root)
    catalog = json.loads(path.read_text(encoding="utf-8"))
    bundles = _cell_bundles(
        catalog,
        track=track,
        door_position=door_position,
        eval_seed=eval_seed,
        task=task,
    )
    if not bundles:
        raise RuntimeError(
            f"No door planning queries for track={track}, door={door_position}, "
            f"eval_seed={eval_seed}, task={task}"
        )
    if expected_queries is not None and len(bundles) != int(expected_queries):
        raise RuntimeError(
            f"Expected {expected_queries} queries in the cell, got {len(bundles)}"
        )

    query_ids: set[str] = set()
    query_pixel_hashes: set[str] = set()
    evaluation_indices: set[int] = set()
    candidate_hashes: set[str] = set()
    directions: list[str] = []
    offsets: list[int] = []
    assets: list[dict[str, Any]] = []
    payload_paths: set[str] = set()

    for bundle in bundles:
        query_id = str(_required(bundle, "query_id", where="catalog bundle"))
        if query_id in query_ids:
            raise RuntimeError(f"Duplicate query_id in cell: {query_id}")
        query_ids.add(query_id)
        evaluation_index = int(
            _required(bundle, "evaluation_index", where=query_id)
        )
        if evaluation_index in evaluation_indices:
            raise RuntimeError(
                f"Duplicate evaluation_index {evaluation_index} in cell"
            )
        evaluation_indices.add(evaluation_index)
        speed = float(_factor(bundle, "agent.speed"))
        if not np.isclose(speed, AGENT_SPEED, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"{query_id}: agent.speed must remain 5, got {speed}")

        payload_path = resolve_contextworld_path(
            _required(bundle, "payload", where=query_id), repo_root=repo_root
        )
        expected_payload_hash = str(
            _required(bundle, "payload_sha256", where=query_id)
        )
        actual_payload_hash = file_sha256(payload_path)
        if actual_payload_hash != expected_payload_hash:
            raise RuntimeError(f"{query_id}: payload file hash mismatch")
        if str(payload_path) in payload_paths:
            raise RuntimeError(f"Payload reused by multiple queries: {payload_path}")
        payload_paths.add(str(payload_path))

        required_arrays = (
            "query_pixels",
            "goal_pixels",
            "history_pixels",
            "history_actions",
            "query_state",
            "goal_state",
            "fixed_candidate_raw_actions",
        )
        with np.load(payload_path, allow_pickle=False) as payload:
            missing = [key for key in required_arrays if key not in payload]
            if missing:
                raise KeyError(f"{query_id}: payload missing arrays {missing}")
            arrays = {
                "query_pixels": np.asarray(
                    payload["query_pixels"], dtype=np.uint8
                ).copy(),
                "goal_pixels": np.asarray(
                    payload["goal_pixels"], dtype=np.uint8
                ).copy(),
                "history_pixels": np.asarray(
                    payload["history_pixels"], dtype=np.uint8
                ).copy(),
                "history_actions": np.asarray(
                    payload["history_actions"], dtype=np.float32
                ).copy(),
                "query_state": np.asarray(
                    payload["query_state"], dtype=np.float32
                ).copy(),
                "goal_state": np.asarray(
                    payload["goal_state"], dtype=np.float32
                ).copy(),
                "fixed_candidate_raw_actions": np.asarray(
                    payload["fixed_candidate_raw_actions"], dtype=np.float32
                ).copy(),
            }
        _validate_shapes(
            query_id=query_id,
            arrays=arrays,
            candidates=candidates,
            horizon=horizon,
        )
        hashes = {
            key: _assert_array_hash(bundle, key, value)
            for key, value in arrays.items()
        }
        if hashes["query_pixels"] in query_pixel_hashes:
            raise RuntimeError(f"Duplicate query pixels within cell: {query_id}")
        query_pixel_hashes.add(hashes["query_pixels"])
        candidate_hashes.add(hashes["fixed_candidate_raw_actions"])

        direction = str(_metadata(bundle, "direction"))
        offset = int(_metadata(bundle, "door_relative_vertical_offset_px"))
        if direction not in VALID_DIRECTIONS:
            raise ValueError(f"{query_id}: invalid direction {direction!r}")
        if offset not in VALID_OFFSETS:
            raise ValueError(f"{query_id}: invalid vertical offset {offset}")
        start_left = float(arrays["query_state"][0]) < WALL_CENTER_X
        goal_left = float(arrays["goal_state"][0]) < WALL_CENTER_X
        expected_direction = "left_to_right" if start_left and not goal_left else (
            "right_to_left" if not start_left and goal_left else None
        )
        if direction != expected_direction:
            raise RuntimeError(
                f"{query_id}: direction {direction} disagrees with reset/goal"
            )
        directions.append(direction)
        offsets.append(offset)

        explicit_seed = bundle.get("cem_seed")
        cem_seed = (
            int(explicit_seed)
            if explicit_seed is not None
            else deterministic_cem_seed(
                eval_seed=eval_seed,
                evaluation_index=evaluation_index,
                query_id=query_id,
            )
        )
        context_actions = arrays["history_actions"]
        assets.append(
            {
                "bundle": bundle,
                "episode": QueryEpisode(
                    query_id=query_id,
                    scenario_id=str(
                        bundle.get("source_scenario_id", query_id)
                    ),
                    template_id=str(
                        bundle.get("template_id", bundle.get("template", {}).get(
                            "template_id", query_id
                        ))
                    ),
                    speed=speed,
                    door_position=int(door_position),
                    simulator_seed=int(bundle.get("simulator_seed", 0)),
                    query_pixels=arrays["query_pixels"],
                    goal_pixels=arrays["goal_pixels"],
                    query_state=arrays["query_state"],
                    goal_state=arrays["goal_state"],
                ),
                "history_pixels": arrays["history_pixels"][:2],
                "history_raw_actions": context_actions,
                "fixed_candidate_raw_actions": arrays[
                    "fixed_candidate_raw_actions"
                ],
                "array_hashes": hashes,
                "payload": str(payload_path),
                "payload_sha256": actual_payload_hash,
                "evaluation_index": evaluation_index,
                "cem_seed": cem_seed,
                "direction": direction,
                "door_relative_vertical_offset_px": offset,
                "task": task,
            }
        )

    strata = {
        f"{direction}:{offset}": int(
            sum(
                d == direction and o == offset
                for d, o in zip(directions, offsets, strict=True)
            )
        )
        for direction in VALID_DIRECTIONS
        for offset in VALID_OFFSETS
    }
    if any(value == 0 for value in strata.values()):
        raise RuntimeError(f"Door geometry stratum missing from cell: {strata}")
    if max(strata.values()) - min(strata.values()) > 1:
        raise RuntimeError(f"Door geometry strata are not balanced: {strata}")

    audit = {
        "passed": True,
        "queries": len(assets),
        "unique_query_ids": len(query_ids),
        "unique_query_pixel_hashes": len(query_pixel_hashes),
        "unique_payloads": len(payload_paths),
        "unique_candidate_banks": len(candidate_hashes),
        "agent_speed": AGENT_SPEED,
        "candidates_per_query": int(candidates),
        "horizon_action_blocks": int(horizon),
        "geometry_strata": strata,
    }
    return DoorPlanningCell(
        catalog=catalog,
        catalog_path=path,
        catalog_sha256=file_sha256(path),
        track=str(track),
        door_position=int(door_position),
        eval_seed=int(eval_seed),
        task=str(task),
        assets=tuple(assets),
        audit=audit,
    )


def _crossing_details(
    states: np.ndarray,
    *,
    door_position: float,
    goal_state: np.ndarray | None,
    door_size: float = DOOR_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(states, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or values.shape[-1] != 2 or values.shape[1] < 2:
        raise ValueError("states must have shape (T,2) or (N,T,2)")
    before = values[:, :-1]
    after = values[:, 1:]
    x0 = before[..., 0]
    x1 = after[..., 0]
    denominator = x1 - x0
    crosses_segment = (
        ((x0 < WALL_CENTER_X) & (x1 >= WALL_CENTER_X))
        | ((x0 >= WALL_CENTER_X) & (x1 < WALL_CENTER_X))
    ) & (np.abs(denominator) > 1e-8)
    margin = float(door_size) + 1.75
    # StableWM's discrete collision rule decides whether a transition may
    # enter/cross the wall from the proposed endpoint coordinate.  Use that
    # same endpoint convention here.  Interpolating y at x=112 can reject a
    # legal diagonal transition whose endpoint has just entered the opening.
    endpoint_y = after[..., 1]
    in_opening = (
        endpoint_y >= float(door_position) - margin
    ) & (endpoint_y <= float(door_position) + margin)
    desired = crosses_segment & in_opening
    if goal_state is not None:
        goals = np.asarray(goal_state, dtype=np.float32)
        if goals.ndim == 1:
            goals = np.repeat(goals[None], values.shape[0], axis=0)
        if goals.shape != (values.shape[0], 2):
            raise ValueError("goal_state must have shape (2,) or (N,2)")
        starts_left = values[:, 0, 0] < WALL_CENTER_X
        goals_left = goals[:, 0] < WALL_CENTER_X
        direction_ok = np.where(
            (starts_left & ~goals_left)[:, None],
            denominator > 0.0,
            np.where(
                (~starts_left & goals_left)[:, None],
                denominator < 0.0,
                False,
            ),
        )
        desired &= direction_ok
    crossed = desired.any(axis=1)
    first = np.full(values.shape[0], -1, dtype=np.int32)
    if desired.shape[1]:
        indices = np.argmax(desired, axis=1) + 1
        first[crossed] = indices[crossed]
    return crossed, first


def doorway_crossing(
    states: np.ndarray,
    *,
    door_position: float,
    goal_state: np.ndarray | None = None,
    door_size: float = DOOR_SIZE,
) -> dict[str, Any]:
    crossed, first = _crossing_details(
        states,
        door_position=door_position,
        goal_state=goal_state,
        door_size=door_size,
    )
    if np.asarray(states).ndim == 2:
        return {
            "crossed": bool(crossed[0]),
            "first_crossing_raw_step": (
                int(first[0]) if crossed[0] else None
            ),
        }
    return {"crossed": crossed, "first_crossing_raw_step": first}


def simulate_door_candidates(
    *,
    query_state: np.ndarray,
    goal_state: np.ndarray,
    raw_actions: np.ndarray,
    speed: float = AGENT_SPEED,
    door_position: float,
    door_size: float = DOOR_SIZE,
) -> dict[str, np.ndarray]:
    """Exact-vectorized dynamics used by the established TwoRoom evaluator.

    This mirrors ``simulate_tworoom_candidates`` and additionally retains the
    state trajectory, wall-contact flags, and doorway-crossing outcome.
    """

    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError("raw_actions must have shape (candidates, steps, 2)")
    positions = np.repeat(
        np.asarray(query_state, dtype=np.float32)[None], actions.shape[0], axis=0
    )
    goal = np.asarray(goal_state, dtype=np.float32)
    states = [positions.copy()]
    distances = [np.linalg.norm(positions - goal[None], axis=1)]
    first_success = np.full(actions.shape[0], -1, dtype=np.int32)
    wall_contacts = np.zeros(actions.shape[0], dtype=np.int32)
    path_length = np.zeros(actions.shape[0], dtype=np.float32)

    border = 14.0
    radius = 7.0
    lower = border + radius
    upper = 224.0 - border - radius
    effective_left = WALL_CENTER_X - 5.0 - radius
    effective_right = WALL_CENTER_X + 5.0 + radius
    door_low = float(door_position) - float(door_size) - 1.75
    door_high = float(door_position) + float(door_size) + 1.75

    for step_index in range(actions.shape[1]):
        previous = positions.copy()
        proposed = positions + np.clip(
            actions[:, step_index], -1.0, 1.0
        ) * float(speed)
        proposed = np.clip(proposed, lower, upper)
        in_door = (proposed[:, 1] >= door_low) & (proposed[:, 1] <= door_high)
        started_left = previous[:, 0] < WALL_CENTER_X
        blocked_left = (
            started_left & (proposed[:, 0] > effective_left) & ~in_door
        )
        blocked_right = (
            ~started_left & (proposed[:, 0] < effective_right) & ~in_door
        )
        blocked = blocked_left | blocked_right
        wall_contacts += blocked.astype(np.int32)
        proposed[blocked_left, 0] = effective_left - 0.5
        proposed[blocked_right, 0] = effective_right + 0.5
        positions = proposed
        path_length += np.linalg.norm(positions - previous, axis=1)
        distance = np.linalg.norm(positions - goal[None], axis=1)
        newly_successful = (first_success < 0) & (distance < 16.0)
        first_success[newly_successful] = step_index + 1
        states.append(positions.copy())
        distances.append(distance)

    state_array = np.stack(states, axis=1)
    distance_array = np.stack(distances, axis=1)
    crossings = doorway_crossing(
        state_array,
        door_position=door_position,
        goal_state=goal,
        door_size=door_size,
    )
    return {
        "final_states": positions,
        "state_trajectories": state_array,
        "final_distances": distance_array[:, -1],
        "distance_trajectories": distance_array,
        "steps_to_success": first_success,
        "success": first_success > 0,
        "path_length": path_length,
        "wall_contact_steps": wall_contacts,
        "doorway_crossed": np.asarray(crossings["crossed"], dtype=bool),
        "first_doorway_crossing_step": np.asarray(
            crossings["first_crossing_raw_step"], dtype=np.int32
        ),
    }


def summarize_fixed_candidate_selection(
    costs: np.ndarray, true_dynamics: dict[str, np.ndarray]
) -> dict[str, Any]:
    predicted = np.asarray(costs, dtype=np.float64)
    selected = int(np.argmin(predicted))
    oracle = int(np.argmin(true_dynamics["final_distances"]))
    success = bool(true_dynamics["success"][selected])
    crossed = bool(true_dynamics["doorway_crossed"][selected])
    return {
        "selected_candidate": selected,
        "selected_true_final_distance_px": float(
            true_dynamics["final_distances"][selected]
        ),
        "selected_true_success": success,
        "selected_true_steps_to_success": (
            int(true_dynamics["steps_to_success"][selected])
            if success
            else None
        ),
        "selected_true_doorway_crossing": crossed,
        "selected_true_first_doorway_crossing_step": (
            int(true_dynamics["first_doorway_crossing_step"][selected])
            if crossed
            else None
        ),
        "oracle_candidate": oracle,
        "oracle_true_final_distance_px": float(
            true_dynamics["final_distances"][oracle]
        ),
        "oracle_true_success": bool(true_dynamics["success"][oracle]),
        "oracle_true_doorway_crossing": bool(
            true_dynamics["doorway_crossed"][oracle]
        ),
        "exact_environment_endpoint_regret_px": float(
            true_dynamics["final_distances"][selected]
            - true_dynamics["final_distances"][oracle]
        ),
        "predicted_cost_vs_true_endpoint_distance_spearman": spearman(
            predicted, true_dynamics["final_distances"]
        ),
        "cost_sha256": array_sha256(np.asarray(costs, dtype=np.float32)),
    }


def aggregate_door_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("Cannot aggregate an empty record list")
    success = np.asarray([bool(row["success"]) for row in rows], dtype=bool)
    crossing = np.asarray(
        [bool(row["doorway_crossing"]) for row in rows], dtype=bool
    )
    distances = np.asarray(
        [float(row["final_distance_px"]) for row in rows], dtype=np.float64
    )
    successful_steps = [
        int(row["steps_to_success"])
        for row in rows
        if row["steps_to_success"] is not None
    ]
    strata: dict[str, Any] = {}
    for direction in VALID_DIRECTIONS:
        for offset in VALID_OFFSETS:
            selected = [
                row
                for row in rows
                if row["direction"] == direction
                and int(row["door_relative_vertical_offset_px"]) == offset
            ]
            if not selected:
                continue
            strata[f"{direction}:{offset}"] = {
                "evaluations": len(selected),
                "success_rate": float(
                    np.mean([bool(row["success"]) for row in selected])
                ),
                "doorway_crossing_rate": float(
                    np.mean(
                        [bool(row["doorway_crossing"]) for row in selected]
                    )
                ),
                "mean_final_distance_px": float(
                    np.mean(
                        [float(row["final_distance_px"]) for row in selected]
                    )
                ),
            }
    return {
        "evaluations": len(rows),
        "successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "mean_final_distance_px": float(distances.mean()),
        "doorway_crossings": int(crossing.sum()),
        "doorway_crossing_rate": float(crossing.mean()),
        "steps_to_success_success_only": {
            "count": len(successful_steps),
            "mean": (
                float(np.mean(successful_steps)) if successful_steps else None
            ),
            "median": (
                float(np.median(successful_steps)) if successful_steps else None
            ),
        },
        "descriptive_geometry_strata": strata,
    }


__all__ = [
    "ACTION_BLOCK",
    "AGENT_SPEED",
    "DOOR_SIZE",
    "DoorPlanningCell",
    "aggregate_door_records",
    "deterministic_cem_seed",
    "doorway_crossing",
    "load_door_planning_cell",
    "simulate_door_candidates",
    "summarize_fixed_candidate_selection",
]
