"""Replay-anchored paired Push-T trajectories with hidden actuator gain.

Unlike the deliberately narrow v1 construction, this module anchors the
common query state, goal, and query action in the original expert replay.
The history probe is a rotated source action block.  Its recovery is the
nearest source block projected onto the free-space controller nullspace, so
both hidden gains naturally return to the same query within a strict numerical
tolerance without revealing the gain through the raw actions.  The simulator
state is never loaded or edited after the initial reset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from stable_worldmodel.envs.pusht.env import PushT

from contextworld.evaluation.pusht_hidden_actuation import (
    ACTION_BLOCK,
    MODEL_FRAME_ROWS,
    MODEL_STEPS,
    MODE_SCALES,
    RAW_STEPS,
    _body_snapshot,
    _future_state_gap,
    array_sha256,
    validate_hidden_actuation_pair,
)


# Calibrated once from PushT's public free-space PD controller by applying a
# unit impulse at each of ten raw action steps and measuring final position
# and velocity.  The first five columns map the probe block and the final five
# columns map the recovery block.  The existing v1 recovery profile is exactly
# the minimum-norm solution produced by this matrix.
FREE_SPACE_RESPONSE = np.asarray(
    [
        [
            4.06931947e-1,
            4.06931947e-1,
            4.06931947e-1,
            4.06931947e-1,
            4.06931947e-1,
            4.06931947e-1,
            4.06931978e-1,
            4.06927170e-1,
            4.07660961e-1,
            2.95671786e-1,
        ],
        [
            3.47901912e-16,
            3.24008904e-15,
            3.01756807e-14,
            2.81032926e-13,
            -4.53035130e-11,
            6.89636830e-09,
            -1.05252066e-6,
            1.60633017e-4,
            -2.45153966e-2,
            3.74147656,
        ],
    ],
    dtype=np.float64,
)
PROBE_RESPONSE = FREE_SPACE_RESPONSE[:, :ACTION_BLOCK]
RECOVERY_RESPONSE = FREE_SPACE_RESPONSE[:, ACTION_BLOCK:]
RECOVERY_RIGHT_INVERSE = RECOVERY_RESPONSE.T @ np.linalg.inv(
    RECOVERY_RESPONSE @ RECOVERY_RESPONSE.T
)
MINIMUM_SOURCE_BLOCK_MOTION_PX = 1.0
MINIMUM_AGENT_BLOCK_DISTANCE_PX = 25.0
MAXIMUM_AGENT_BLOCK_DISTANCE_PX = 180.0
MINIMUM_QUERY_MEAN_ACTION_NORM = 0.05


def _action_block(value: Iterable[Iterable[float]]) -> np.ndarray:
    result = np.asarray(tuple(tuple(row) for row in value), dtype=np.float64)
    if result.shape != (ACTION_BLOCK, 2):
        raise ValueError(
            f"Expected an action block with shape {(ACTION_BLOCK, 2)}, "
            f"got {result.shape}"
        )
    if not bool(np.isfinite(result).all()):
        raise ValueError("Action blocks must contain only finite values")
    if bool((np.abs(result) > 1.0 + 1e-6).any()):
        raise ValueError("Action blocks must remain inside [-1, 1]")
    return result


def _unit(value: Iterable[float]) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (2,):
        raise ValueError(f"Expected a 2-D direction, got {result.shape}")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-8:
        raise ValueError("Direction must be nonzero")
    return result / norm


def rotate_action_block_to_direction(
    actions: Iterable[Iterable[float]],
    target_direction: Iterable[float],
) -> np.ndarray:
    """Rotate a source action block while preserving all magnitudes."""

    block = _action_block(actions)
    source = block.mean(axis=0)
    if float(np.linalg.norm(source)) <= 1e-8:
        raise ValueError("Cannot orient an action block with zero mean")
    target = _unit(target_direction)
    angle = float(
        np.arctan2(target[1], target[0])
        - np.arctan2(source[1], source[0])
    )
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )
    return block @ rotation.T


def project_recovery_to_nullspace(
    probe_actions: Iterable[Iterable[float]],
    reference_actions: Iterable[Iterable[float]],
) -> np.ndarray:
    """Return the nearest recovery block that nulls position and velocity."""

    probe = _action_block(probe_actions)
    reference = _action_block(reference_actions)
    result = np.empty_like(reference)
    for dimension in range(2):
        required = (
            -PROBE_RESPONSE @ probe[:, dimension]
            - RECOVERY_RESPONSE @ reference[:, dimension]
        )
        result[:, dimension] = (
            reference[:, dimension]
            + RECOVERY_RIGHT_INVERSE @ required
        )
    if bool((np.abs(result) > 1.0 + 1e-6).any()):
        raise ValueError("Nullspace recovery leaves the action support")
    return result


def replay_candidate_rows(
    states: np.ndarray,
    actions: np.ndarray,
    episode_offsets: np.ndarray,
    episode_lengths: np.ndarray,
    episode_indices: Iterable[int],
) -> np.ndarray:
    """Select source rows that can support a contact-query pair.

    Selection uses only original replay arrays.  Physics feasibility under
    the two counterfactual gains remains a later, separately audited gate.
    """

    states = np.asarray(states)
    actions = np.asarray(actions)
    offsets = np.asarray(episode_offsets, dtype=np.int64)
    lengths = np.asarray(episode_lengths, dtype=np.int64)
    if states.ndim != 2 or states.shape[1] != 7:
        raise ValueError("states must have shape (N, 7)")
    if actions.shape != (states.shape[0], 2):
        raise ValueError("actions must have shape (N, 2)")
    if offsets.shape != lengths.shape:
        raise ValueError("episode offsets and lengths must match")

    rows = []
    for episode_index in episode_indices:
        episode_index = int(episode_index)
        if not 0 <= episode_index < len(offsets):
            raise IndexError(f"Invalid episode index {episode_index}")
        offset = int(offsets[episode_index])
        length = int(lengths[episode_index])
        # Keep 25 future steps to match the standard Push-T evaluation
        # query/goal horizon and to obtain a stable episode goal proxy.
        if length <= 25:
            continue
        rows.extend(range(offset, offset + length - 25))
    if not rows:
        return np.empty(0, dtype=np.int64)

    candidates = np.asarray(rows, dtype=np.int64)
    state = states[candidates].astype(np.float64)
    future = states[candidates + ACTION_BLOCK].astype(np.float64)
    query = np.stack(
        [actions[candidates + step] for step in range(ACTION_BLOCK)],
        axis=1,
    ).astype(np.float64)
    agent_block_distance = np.linalg.norm(
        state[:, :2] - state[:, 2:4],
        axis=1,
    )
    block_motion = np.linalg.norm(
        future[:, 2:4] - state[:, 2:4],
        axis=1,
    )
    query_mean_norm = np.linalg.norm(query.mean(axis=1), axis=1)

    # The goal overlay must remain exactly representable by PushT's declared
    # goal variation support.  The final expert state is the closest stored
    # proxy to the episode's rendered goal.
    episode_for_row = np.searchsorted(offsets, candidates, side="right") - 1
    episode_final_rows = (
        offsets[episode_for_row] + lengths[episode_for_row] - 1
    )
    goals = states[episode_final_rows].astype(np.float64)
    finite = (
        np.isfinite(state).all(axis=1)
        & np.isfinite(query).all(axis=(1, 2))
        & np.isfinite(goals).all(axis=1)
    )
    mask = (
        finite
        & (block_motion >= MINIMUM_SOURCE_BLOCK_MOTION_PX)
        & (agent_block_distance >= MINIMUM_AGENT_BLOCK_DISTANCE_PX)
        & (agent_block_distance <= MAXIMUM_AGENT_BLOCK_DISTANCE_PX)
        & (query_mean_norm >= MINIMUM_QUERY_MEAN_ACTION_NORM)
        & (np.abs(query) <= 1.0 + 1e-6).all(axis=(1, 2))
        & (state[:, :2] >= 10.0).all(axis=1)
        & (state[:, :2] <= 502.0).all(axis=1)
        & (state[:, 2:4] >= 20.0).all(axis=1)
        & (state[:, 2:4] <= 492.0).all(axis=1)
        & (goals[:, 2:4] >= 50.0).all(axis=1)
        & (goals[:, 2:4] <= 450.0).all(axis=1)
    )
    return candidates[mask]


@dataclass(frozen=True)
class ReplayMatchedHiddenActuationTemplate:
    """A paired hidden-gain query anchored in one expert replay row."""

    template_id: str
    source_row_index: int
    source_episode_index: int
    source_step_index: int
    agent_position: tuple[float, float]
    block_position: tuple[float, float]
    block_angle: float
    goal_agent_position: tuple[float, float]
    goal_block_position: tuple[float, float]
    goal_block_angle: float
    probe_actions: tuple[tuple[float, float], ...]
    recovery_actions: tuple[tuple[float, float], ...]
    query_actions: tuple[tuple[float, float], ...]
    filler_actions: tuple[tuple[float, float], ...]
    simulator_seed: int

    def __post_init__(self) -> None:
        for name in (
            "probe_actions",
            "recovery_actions",
            "query_actions",
            "filler_actions",
        ):
            _action_block(getattr(self, name))
        for name, value in (
            ("agent_position", self.agent_position),
            ("block_position", self.block_position),
            ("goal_agent_position", self.goal_agent_position),
            ("goal_block_position", self.goal_block_position),
        ):
            vector = np.asarray(value, dtype=np.float64)
            if vector.shape != (2,) or not bool(np.isfinite(vector).all()):
                raise ValueError(f"{name} must be a finite 2-D point")
        if self.source_row_index < 0:
            raise ValueError("source_row_index cannot be negative")
        if self.source_episode_index < 0 or self.source_step_index < 0:
            raise ValueError("source episode and step indices cannot be negative")

    @property
    def reset_state(self) -> np.ndarray:
        # Agent velocity is not visible to the image-only model and cannot be
        # preserved by the gain-independent nulling controller.  It is fixed
        # to zero in both modes and recorded explicitly in the data contract.
        return np.asarray(
            [
                *self.agent_position,
                *self.block_position,
                self.block_angle,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

    @property
    def goal_state(self) -> np.ndarray:
        return np.asarray(
            [
                *self.goal_agent_position,
                *self.goal_block_position,
                self.goal_block_angle,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

    @property
    def action_blocks(self) -> np.ndarray:
        return np.stack(
            [
                _action_block(self.probe_actions),
                _action_block(self.recovery_actions),
                _action_block(self.query_actions),
                _action_block(self.filler_actions),
            ]
        ).astype(np.float32)


def _variation_values(
    template: ReplayMatchedHiddenActuationTemplate,
) -> dict[str, Any]:
    # Variation values create bodies before the exact state is restored.
    # Clipping here does not change the final reset state.
    return {
        "agent.start_position": np.clip(
            np.asarray(template.agent_position, dtype=np.float64),
            50.0,
            450.0,
        ),
        "agent.velocity": np.zeros(2, dtype=np.float64),
        "block.start_position": np.clip(
            np.asarray(template.block_position, dtype=np.float64),
            100.0,
            400.0,
        ),
        "block.angle": float(template.block_angle),
        "block.shape": 2,
        "goal.position": np.asarray(
            template.goal_block_position,
            dtype=np.float64,
        ),
        "goal.angle": float(template.goal_block_angle),
    }


def _simulate(
    template: ReplayMatchedHiddenActuationTemplate,
    *,
    mode: str,
    resolution: int,
    query_state_tolerance: float,
    capture_rows: bool,
) -> dict[str, Any]:
    if mode not in MODE_SCALES:
        raise ValueError(f"Unknown hidden actuation mode {mode!r}")
    if query_state_tolerance <= 0:
        raise ValueError("query_state_tolerance must be positive")

    blocks = template.action_blocks
    raw_actions = blocks.reshape(RAW_STEPS, 2)
    env = PushT(
        resolution=int(resolution),
        with_target=True,
        render_mode="rgb_array",
    )
    if not capture_rows:
        # PushT.reset() renders the goal image unconditionally.  The fast
        # rejection gate needs only physics, so avoid paying for a 512px
        # software render for every rejected source candidate.  The formal
        # path below still uses the unchanged renderer for every stored row.
        env.render = lambda: None
    env.action_scale = float(MODE_SCALES[mode])
    rows: dict[str, list[Any]] | None = (
        {
            "pixels": [],
            "action": [],
            "proprio": [],
            "state": [],
            "goal_state": [],
            "physics_state": [],
            "n_contacts": [],
            "hidden_action_scale": [],
            "pair_id": [],
            "hidden_mode": [],
            "source_row_index": [],
            "source_episode_index": [],
            "source_step_index": [],
        }
        if capture_rows
        else None
    )
    model_pixels: list[np.ndarray] = []
    model_states: list[np.ndarray] = []
    model_physics: list[np.ndarray] = []
    block_contact_steps = [0, 0, 0, 0]
    query_natural_snapshot = None
    query_recovery_residual = None
    try:
        observation, info = env.reset(
            seed=int(template.simulator_seed),
            options={
                "variation": (),
                "variation_values": _variation_values(template),
                "state": template.reset_state,
                "goal_state": template.goal_state,
            },
        )
        query_reference_snapshot = _body_snapshot(env)
        initial_pixels = (
            np.asarray(env.render(), dtype=np.uint8).copy()
            if capture_rows
            else None
        )
        initial_state = np.asarray(observation["state"], dtype=np.float64)
        goal_pixels = (
            np.asarray(info["goal"], dtype=np.uint8).copy()
            if capture_rows
            else None
        )

        for raw_step, action in enumerate(raw_actions):
            if raw_step == 2 * ACTION_BLOCK:
                query_natural_snapshot = _body_snapshot(env)
                query_recovery_residual = float(
                    np.max(
                        np.abs(
                            query_natural_snapshot
                            - query_reference_snapshot
                        )
                    )
                )
                if query_recovery_residual > query_state_tolerance:
                    raise RuntimeError(
                        "Replay-matched recovery did not return to the "
                        "common query state: "
                        f"template={template.template_id}, mode={mode}, "
                        f"residual={query_recovery_residual:.8f}, "
                        f"tolerance={query_state_tolerance:.8f}"
                    )

            state = np.asarray(env._get_obs(), dtype=np.float64)
            physics = _body_snapshot(env)
            pixel = (
                np.asarray(env.render(), dtype=np.uint8).copy()
                if capture_rows
                else None
            )
            if raw_step in MODEL_FRAME_ROWS:
                if pixel is not None:
                    model_pixels.append(pixel)
                model_states.append(state.astype(np.float32))
                model_physics.append(physics.astype(np.float32))
            if rows is not None:
                rows["pixels"].append(pixel)
                rows["action"].append(
                    np.asarray(action, dtype=np.float32).copy()
                )
                rows["proprio"].append(
                    np.concatenate([state[:2], state[-2:]]).astype(
                        np.float32
                    )
                )
                rows["state"].append(state.astype(np.float32))
                rows["goal_state"].append(
                    template.goal_state.astype(np.float32)
                )
                rows["physics_state"].append(physics.astype(np.float32))
                rows["n_contacts"].append(
                    np.asarray([0.0], dtype=np.float32)
                )
                rows["hidden_action_scale"].append(
                    np.asarray([MODE_SCALES[mode]], dtype=np.float32)
                )
                rows["pair_id"].append(template.template_id)
                rows["hidden_mode"].append(mode)
                rows["source_row_index"].append(
                    np.asarray(
                        [template.source_row_index],
                        dtype=np.int64,
                    )
                )
                rows["source_episode_index"].append(
                    np.asarray(
                        [template.source_episode_index],
                        dtype=np.int64,
                    )
                )
                rows["source_step_index"].append(
                    np.asarray(
                        [template.source_step_index],
                        dtype=np.int64,
                    )
                )

            _, _, _, _, step_info = env.step(action)
            contacts = int(step_info["n_contacts"])
            block_contact_steps[raw_step // ACTION_BLOCK] += int(
                contacts > 0
            )
            if rows is not None:
                rows["n_contacts"][-1][0] = float(contacts)
    finally:
        env.close()

    return {
        "template": asdict(template),
        "mode": mode,
        "hidden_action_scale": float(MODE_SCALES[mode]),
        "raw_actions": raw_actions,
        "action_blocks": blocks,
        "rows": rows,
        "model_pixels": (
            np.stack(model_pixels) if capture_rows else None
        ),
        "model_states": np.stack(model_states),
        "model_physics_states": np.stack(model_physics),
        "initial_pixels": initial_pixels,
        "initial_state": initial_state.astype(np.float32),
        "goal_pixels": goal_pixels,
        "query_reference_snapshot": query_reference_snapshot,
        "query_natural_snapshot": query_natural_snapshot,
        "query_recovery_residual": query_recovery_residual,
        "query_state_tolerance": float(query_state_tolerance),
        "state_installations_after_x0": 0,
        "query_simulator_recreated": False,
        "query_contact_steps": block_contact_steps[2],
        "history_contact_steps": sum(block_contact_steps[:2]),
        "contact_steps_by_block": block_contact_steps,
    }


def simulate_replay_matched_hidden_actuation(
    template: ReplayMatchedHiddenActuationTemplate,
    *,
    mode: str,
    resolution: int = 224,
    query_state_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Render one replay-matched hidden-gain trajectory."""

    return _simulate(
        template,
        mode=mode,
        resolution=resolution,
        query_state_tolerance=query_state_tolerance,
        capture_rows=True,
    )


def fast_replay_matched_pair_audit(
    template: ReplayMatchedHiddenActuationTemplate,
    *,
    query_state_tolerance: float = 1e-5,
    minimum_middle_agent_gap_px: float = 10.0,
    minimum_future_block_gap_px: float = 2.0,
) -> dict[str, Any]:
    """Physics-only rejection gate before expensive 224px rendering."""

    try:
        low = _simulate(
            template,
            mode="low_gain",
            resolution=32,
            query_state_tolerance=query_state_tolerance,
            capture_rows=False,
        )
        high = _simulate(
            template,
            mode="high_gain",
            resolution=32,
            query_state_tolerance=query_state_tolerance,
            capture_rows=False,
        )
    except (AssertionError, RuntimeError, ValueError) as error:
        return {
            "passed": False,
            "exception": type(error).__name__,
            "message": str(error),
        }
    middle_gap = float(
        np.linalg.norm(
            low["model_states"][1, :2]
            - high["model_states"][1, :2]
        )
    )
    future_gap = _future_state_gap(
        low["model_states"][-1],
        high["model_states"][-1],
    )
    query_pair_gap = float(
        np.max(
            np.abs(
                np.asarray(
                    low["query_natural_snapshot"],
                    dtype=np.float64,
                )
                - np.asarray(
                    high["query_natural_snapshot"],
                    dtype=np.float64,
                )
            )
        )
    )
    checks = {
        "history_is_contact_free": (
            low["history_contact_steps"] == 0
            and high["history_contact_steps"] == 0
        ),
        "both_modes_make_query_contact": (
            low["query_contact_steps"] > 0
            and high["query_contact_steps"] > 0
        ),
        "middle_agent_gap_sufficient": (
            middle_gap >= minimum_middle_agent_gap_px
        ),
        "future_block_gap_sufficient": (
            future_gap["block_position_px"]
            >= minimum_future_block_gap_px
        ),
        "query_full_state_within_tolerance": (
            query_pair_gap <= query_state_tolerance
        ),
        "no_state_installations_after_x0": (
            low["state_installations_after_x0"] == 0
            and high["state_installations_after_x0"] == 0
            and not low["query_simulator_recreated"]
            and not high["query_simulator_recreated"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "middle_agent_gap_px": middle_gap,
        "future_gap": future_gap,
        "query_physics_max_abs_gap": query_pair_gap,
        "query_recovery_residual": {
            "low_gain": low["query_recovery_residual"],
            "high_gain": high["query_recovery_residual"],
        },
        "query_contact_steps": {
            "low_gain": low["query_contact_steps"],
            "high_gain": high["query_contact_steps"],
        },
    }


def validate_replay_matched_pair(
    low: dict[str, Any],
    high: dict[str, Any],
) -> dict[str, Any]:
    """Run the full pixel, physics, and source-receipt audit."""

    base = validate_hidden_actuation_pair(low, high)
    additional = {
        "history_is_contact_free": (
            low["history_contact_steps"] == 0
            and high["history_contact_steps"] == 0
        ),
        "source_query_actions_preserved": np.array_equal(
            low["action_blocks"][2],
            np.asarray(
                low["template"]["query_actions"],
                dtype=np.float32,
            ),
        ),
        "source_filler_actions_preserved": np.array_equal(
            low["action_blocks"][3],
            np.asarray(
                low["template"]["filler_actions"],
                dtype=np.float32,
            ),
        ),
    }
    checks = {**base["checks"], **additional}
    base.update(
        {
            "passed": all(checks.values()),
            "checks": checks,
            "history_contact_steps": {
                "low_gain": low["history_contact_steps"],
                "high_gain": high["history_contact_steps"],
            },
            "source": {
                "row_index": low["template"]["source_row_index"],
                "episode_index": low["template"][
                    "source_episode_index"
                ],
                "step_index": low["template"]["source_step_index"],
                "query_action_sha256": array_sha256(
                    low["action_blocks"][2]
                ),
                "filler_action_sha256": array_sha256(
                    low["action_blocks"][3]
                ),
            },
        }
    )
    return base


__all__ = [
    "FREE_SPACE_RESPONSE",
    "ReplayMatchedHiddenActuationTemplate",
    "fast_replay_matched_pair_audit",
    "project_recovery_to_nullspace",
    "replay_candidate_rows",
    "rotate_action_block_to_direction",
    "simulate_replay_matched_hidden_actuation",
    "validate_replay_matched_pair",
]
