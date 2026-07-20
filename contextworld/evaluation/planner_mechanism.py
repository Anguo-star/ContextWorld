from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def fixed_candidate_bank(
    *,
    eval_seed: int,
    evaluation_index: int,
    query_index: int,
    candidates: int = 300,
    horizon: int = 5,
    action_block: int = 5,
    action_dim: int = 2,
) -> np.ndarray:
    seed = np.random.SeedSequence(
        [eval_seed, evaluation_index, query_index, 271828]
    )
    rng = np.random.default_rng(seed)
    bank = rng.standard_normal(
        (candidates, horizon, action_block * action_dim)
    ).astype(np.float32)
    bank[0] = 0.0
    return bank


def simulate_tworoom_candidates(
    *,
    query_state: np.ndarray,
    goal_state: np.ndarray,
    raw_actions: np.ndarray,
    speed: float,
    door_position: float,
    door_size: float = 14.0,
) -> dict[str, np.ndarray]:
    """Vectorized exact TwoRoom dynamics for the frozen vertical-wall task."""

    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError("raw_actions must have shape (candidates, steps, 2)")
    positions = np.repeat(
        np.asarray(query_state, dtype=np.float32)[None], actions.shape[0], axis=0
    )
    goal = np.asarray(goal_state, dtype=np.float32)
    distances = [np.linalg.norm(positions - goal[None], axis=1)]
    first_success = np.full(actions.shape[0], -1, dtype=np.int16)
    path_length = np.zeros(actions.shape[0], dtype=np.float32)

    border = 14.0
    radius = 7.0
    lower = border + radius
    upper = 224.0 - border - radius
    effective_left = 112.0 - 5.0 - radius
    effective_right = 112.0 + 5.0 + radius
    door_low = door_position - door_size - 1.75
    door_high = door_position + door_size + 1.75

    for step_index in range(actions.shape[1]):
        previous = positions.copy()
        proposed = positions + np.clip(actions[:, step_index], -1.0, 1.0) * speed
        proposed = np.clip(proposed, lower, upper)
        in_door = (proposed[:, 1] >= door_low) & (proposed[:, 1] <= door_high)
        started_left = previous[:, 0] < 112.0
        blocked_left = started_left & (proposed[:, 0] > effective_left) & ~in_door
        blocked_right = (
            ~started_left & (proposed[:, 0] < effective_right) & ~in_door
        )
        proposed[blocked_left, 0] = effective_left - 0.5
        proposed[blocked_right, 0] = effective_right + 0.5
        positions = proposed
        path_length += np.linalg.norm(positions - previous, axis=1)
        distance = np.linalg.norm(positions - goal[None], axis=1)
        distances.append(distance)
        newly_successful = (first_success < 0) & (distance < 16.0)
        first_success[newly_successful] = step_index + 1

    distance_array = np.stack(distances, axis=1)
    return {
        "final_states": positions,
        "final_distances": distance_array[:, -1],
        "distance_trajectories": distance_array,
        "steps_to_success": first_success,
        "success": first_success > 0,
        "path_length": path_length,
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks


def spearman(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = rankdata(values_a)
    b = rankdata(values_b)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def topk_overlap(values_a: np.ndarray, values_b: np.ndarray, k: int = 30) -> float:
    a = set(np.argsort(values_a)[:k].tolist())
    b = set(np.argsort(values_b)[:k].tolist())
    return len(a & b) / float(k)


def summarize_costs(
    costs: np.ndarray,
    step_costs: np.ndarray,
    true_dynamics: dict[str, np.ndarray],
) -> dict[str, Any]:
    selected = int(np.argmin(costs))
    topk = np.argsort(costs)[:30].astype(np.int64)
    step_values = np.asarray(step_costs)[selected]
    return {
        "cost_sha256": array_sha256(np.asarray(costs, dtype=np.float32)),
        "selected_candidate": selected,
        "topk_indices": topk.tolist(),
        "selected_latent_step_costs": step_values.tolist(),
        "spearman_cost_vs_true_final_distance": spearman(
            costs, true_dynamics["final_distances"]
        ),
        "selected_true_final_distance": float(
            true_dynamics["final_distances"][selected]
        ),
        "selected_true_success": bool(true_dynamics["success"][selected]),
        "selected_true_steps_to_success": (
            int(true_dynamics["steps_to_success"][selected])
            if true_dynamics["success"][selected]
            else None
        ),
    }


__all__ = [
    "array_sha256",
    "fixed_candidate_bank",
    "rankdata",
    "simulate_tworoom_candidates",
    "spearman",
    "summarize_costs",
    "topk_overlap",
]
