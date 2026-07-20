from __future__ import annotations

from typing import Iterable

import numpy as np


ACTION_BLOCK = 5


def alternating_impulse_blocks(
    direction: np.ndarray | Iterable[float],
    context_transitions: int,
) -> np.ndarray:
    """Build an even-length collision-free diagnostic prefix.

    Each block contains one non-zero raw action followed by four zero actions.
    Consecutive blocks use opposite actions, so every pair returns to the same
    query state while exposing a speed-dependent intermediate observation.
    """

    vector = np.asarray(direction, dtype=np.float32)
    if vector.shape != (2,) or not np.any(vector):
        raise ValueError(f"Expected a non-zero 2D direction, got {vector}")
    if np.any(np.abs(vector) > 1.0):
        raise ValueError(f"Direction exceeds action bounds: {vector}")
    if context_transitions <= 0 or context_transitions % 2:
        raise ValueError(
            "context_transitions must be a positive even integer, got "
            f"{context_transitions}"
        )
    blocks = np.zeros((context_transitions, ACTION_BLOCK, 2), dtype=np.float32)
    signs = np.where(np.arange(context_transitions) % 2 == 0, 1.0, -1.0)
    blocks[:, 0] = signs[:, None] * vector[None]
    return blocks


def agent_centroid_from_rgb(images: np.ndarray) -> np.ndarray:
    """Locate the red TwoRoom agent using only rendered RGB pixels.

    The returned coordinate order is ``(x, y)``, matching TwoRoom state.  The
    red-excess weighting excludes the white background, black walls, and green
    goal without accessing simulator state or factor metadata.
    """

    pixels = np.asarray(images)
    if pixels.ndim < 3 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected (...,H,W,3) RGB images, got {pixels.shape}")
    rgb = pixels.astype(np.float64)
    weights = np.maximum(rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]), 0.0)
    height, width = pixels.shape[-3:-1]
    flat = weights.reshape((-1, height, width))
    totals = flat.sum(axis=(1, 2))
    if np.any(totals <= 0):
        raise ValueError("At least one image has no positive red-excess pixels")
    y_grid = np.arange(height, dtype=np.float64)[:, None]
    x_grid = np.arange(width, dtype=np.float64)[None, :]
    x = (flat * x_grid).sum(axis=(1, 2)) / totals
    y = (flat * y_grid).sum(axis=(1, 2)) / totals
    return np.stack([x, y], axis=-1).reshape(pixels.shape[:-3] + (2,))


def estimate_speed_from_transitions(
    states: np.ndarray,
    next_states: np.ndarray,
    action_blocks: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Estimate the scalar speed by least squares over observed transitions."""

    starts = np.asarray(states, dtype=np.float64)
    ends = np.asarray(next_states, dtype=np.float64)
    blocks = np.asarray(action_blocks, dtype=np.float64)
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError(
            f"Expected matching (K,2) states, got {starts.shape} and {ends.shape}"
        )
    if blocks.shape != (starts.shape[0], ACTION_BLOCK, 2):
        raise ValueError(
            f"Expected actions {(starts.shape[0], ACTION_BLOCK, 2)}, got {blocks.shape}"
        )
    effective_actions = blocks.sum(axis=1)
    displacement = ends - starts
    denominators = np.sum(effective_actions * effective_actions, axis=1)
    valid = denominators > 0
    if not np.any(valid):
        raise ValueError("No identifying non-zero action blocks")
    per_transition = np.full(starts.shape[0], np.nan, dtype=np.float64)
    per_transition[valid] = (
        np.sum(effective_actions[valid] * displacement[valid], axis=1)
        / denominators[valid]
    )
    estimate = float(
        np.sum(effective_actions[valid] * displacement[valid])
        / np.sum(effective_actions[valid] * effective_actions[valid])
    )
    return estimate, per_transition


def nearest_speed(estimate: float, candidates: Iterable[float]) -> float:
    values = np.asarray(sorted({float(value) for value in candidates}), dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one candidate speed is required")
    distances = np.abs(values - float(estimate))
    return float(values[int(np.argmin(distances))])


__all__ = [
    "ACTION_BLOCK",
    "agent_centroid_from_rgb",
    "alternating_impulse_blocks",
    "estimate_speed_from_transitions",
    "nearest_speed",
]
