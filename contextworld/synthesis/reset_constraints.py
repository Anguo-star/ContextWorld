from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


_SUPPORTED_KEYS = {
    "target_room",
    "exclude_wall_zone",
    "minimum_initial_distance",
    "minimum_door_path_distance",
    "agent_position_bounds",
    "target_position_bounds",
}


def normalize_reset_constraints(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("collection.reset_constraints must be a mapping")
    unknown = sorted(set(value) - _SUPPORTED_KEYS)
    if unknown:
        raise ValueError(f"Unsupported TwoRoom reset constraints: {unknown}")

    normalized: dict[str, Any] = {}
    target_room = value.get("target_room")
    if target_room is not None:
        if target_room not in {"same", "opposite"}:
            raise ValueError("target_room must be 'same' or 'opposite'")
        normalized["target_room"] = target_room
    exclude_wall_zone = value.get("exclude_wall_zone", bool(target_room))
    if not isinstance(exclude_wall_zone, bool):
        raise ValueError("exclude_wall_zone must be boolean")
    if exclude_wall_zone:
        normalized["exclude_wall_zone"] = True
    minimum_distance = float(value.get("minimum_initial_distance", 0.0))
    if not np.isfinite(minimum_distance) or minimum_distance < 0.0:
        raise ValueError("minimum_initial_distance must be finite and non-negative")
    if minimum_distance:
        normalized["minimum_initial_distance"] = minimum_distance
    minimum_path_distance = float(
        value.get("minimum_door_path_distance", 0.0)
    )
    if (
        not np.isfinite(minimum_path_distance)
        or minimum_path_distance < 0.0
    ):
        raise ValueError(
            "minimum_door_path_distance must be finite and non-negative"
        )
    if minimum_path_distance:
        normalized["minimum_door_path_distance"] = minimum_path_distance
    for key in ("agent_position_bounds", "target_position_bounds"):
        bounds = value.get(key)
        if bounds is None:
            continue
        array = np.asarray(bounds, dtype=np.float64)
        if array.shape != (2, 2) or not np.all(np.isfinite(array)):
            raise ValueError(f"{key} must be [[x_min, x_max], [y_min, y_max]]")
        if np.any(array[:, 0] >= array[:, 1]):
            raise ValueError(f"{key} lower bounds must be smaller than upper bounds")
        normalized[key] = array.tolist()
    return normalized


def _within_bounds(position: np.ndarray, bounds: Any) -> bool:
    if bounds is None:
        return True
    limits = np.asarray(bounds, dtype=np.float64)
    return bool(np.all(position >= limits[:, 0]) and np.all(position <= limits[:, 1]))


def _agent_constraint(base_constraint: Any, constraints: dict[str, Any]):
    bounds = constraints.get("agent_position_bounds")

    def predicate(agent_position: Any) -> bool:
        return bool(base_constraint(agent_position)) and _within_bounds(
            np.asarray(agent_position, dtype=np.float64), bounds
        )

    return predicate


def _apply_box_bounds(space: Any, bounds: Any) -> None:
    low_attribute = "_contextworld_base_low"
    high_attribute = "_contextworld_base_high"
    if not hasattr(space, low_attribute):
        setattr(space, low_attribute, np.asarray(space.low).copy())
        setattr(space, high_attribute, np.asarray(space.high).copy())
    base_low = np.asarray(getattr(space, low_attribute))
    base_high = np.asarray(getattr(space, high_attribute))
    if bounds is None:
        space.low[...] = base_low
        space.high[...] = base_high
        return
    limits = np.asarray(bounds, dtype=space.dtype)
    constrained_low = np.maximum(base_low, limits[:, 0])
    constrained_high = np.minimum(base_high, limits[:, 1])
    if np.any(constrained_low >= constrained_high):
        raise ValueError(f"Position bounds do not intersect the environment box: {bounds}")
    space.low[...] = constrained_low
    space.high[...] = constrained_high


def _raw_envs(value: Any) -> Iterable[Any]:
    """Yield unwrapped environments from a raw env, wrapper, pool, or World."""

    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        children = getattr(current, "envs", None)
        if children is not None:
            if isinstance(children, (list, tuple)):
                pending.extend(children)
            else:
                pending.append(children)
            continue
        raw = getattr(current, "unwrapped", current)
        if id(raw) not in seen:
            seen.add(id(raw))
        yield raw


def _target_constraint(
    raw_env: Any, base_constraint: Any, constraints: dict[str, Any]
):
    target_room = constraints.get("target_room")
    exclude_wall_zone = bool(constraints.get("exclude_wall_zone", False))
    minimum_distance = float(constraints.get("minimum_initial_distance", 0.0))
    minimum_path_distance = float(
        constraints.get("minimum_door_path_distance", 0.0)
    )

    def predicate(target_position: Any) -> bool:
        if not base_constraint(target_position):
            return False
        target = np.asarray(target_position, dtype=np.float64)
        agent = np.asarray(
            raw_env.variation_space["agent"]["position"].value,
            dtype=np.float64,
        )
        wall_axis = int(raw_env.variation_space["wall"]["axis"].value)
        coordinate = 0 if wall_axis == 1 else 1
        wall_center = float(raw_env.WALL_CENTER)

        if not _within_bounds(target, constraints.get("target_position_bounds")):
            return False

        if target_room in {"same", "opposite"}:
            agent_side = float(agent[coordinate]) < wall_center
            target_side = float(target[coordinate]) < wall_center
            if target_room == "opposite" and agent_side == target_side:
                return False
            if target_room == "same" and agent_side != target_side:
                return False

        if exclude_wall_zone:
            wall_thickness = float(
                raw_env.variation_space["wall"]["thickness"].value
            )
            agent_radius = float(
                raw_env.variation_space["agent"]["radius"].value.item()
            )
            half_thickness = float(int(wall_thickness) // 2)
            wall_min = wall_center - half_thickness - agent_radius
            wall_max = wall_center + half_thickness + agent_radius
            if wall_min <= float(target[coordinate]) <= wall_max:
                return False

        if minimum_distance and float(np.linalg.norm(target - agent)) < minimum_distance:
            return False
        if minimum_path_distance:
            num_doors = int(
                raw_env.variation_space["door"]["number"].value
            )
            door_positions = np.asarray(
                raw_env.variation_space["door"]["position"].value
            )[:num_doors]
            door_sizes = np.asarray(
                raw_env.variation_space["door"]["size"].value
            )[:num_doors]
            agent_radius = float(
                raw_env.variation_space["agent"]["radius"].value.item()
            )
            path_lengths = []
            for door_position, door_size in zip(
                door_positions, door_sizes, strict=True
            ):
                if float(door_size) < 1.1 * agent_radius:
                    continue
                door = (
                    np.asarray(
                        [wall_center, float(door_position)],
                        dtype=np.float64,
                    )
                    if wall_axis == 1
                    else np.asarray(
                        [float(door_position), wall_center],
                        dtype=np.float64,
                    )
                )
                path_lengths.append(
                    float(np.linalg.norm(agent - door))
                    + float(np.linalg.norm(target - door))
                )
            if path_lengths and min(path_lengths) < minimum_path_distance:
                return False
        return True

    return predicate


def apply_tworoom_reset_constraints(value: Any, constraints: Any) -> int:
    """Install deterministic, speed-independent target sampling constraints."""

    normalized = normalize_reset_constraints(constraints)
    configured = 0
    for raw_env in _raw_envs(value):
        variation_space = getattr(raw_env, "variation_space", None)
        if variation_space is None or not hasattr(raw_env, "WALL_CENTER"):
            raise TypeError(
                f"Reset constraints require a raw TwoRoom environment, got {type(raw_env)}"
            )
        agent_space = variation_space["agent"]["position"]
        target_space = variation_space["target"]["position"]
        base_attribute = "_contextworld_base_constrain_fn"
        if not hasattr(agent_space, base_attribute):
            setattr(agent_space, base_attribute, agent_space.constrain_fn)
        if not hasattr(target_space, base_attribute):
            setattr(target_space, base_attribute, target_space.constrain_fn)
        agent_base = getattr(agent_space, base_attribute)
        target_base = getattr(target_space, base_attribute)
        _apply_box_bounds(
            agent_space, normalized.get("agent_position_bounds")
        )
        _apply_box_bounds(
            target_space, normalized.get("target_position_bounds")
        )
        agent_space.constrain_fn = (
            _agent_constraint(agent_base, normalized)
            if normalized.get("agent_position_bounds") is not None
            else agent_base
        )
        target_space.constrain_fn = (
            _target_constraint(raw_env, target_base, normalized)
            if normalized
            else target_base
        )
        configured += 1
    if configured == 0:
        raise TypeError("No TwoRoom environments found for reset constraints")
    return configured
