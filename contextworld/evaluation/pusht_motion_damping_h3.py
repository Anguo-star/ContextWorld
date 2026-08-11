"""History-3 causal construction for hidden PushT motion damping.

The hidden factor is Pymunk ``Space.damping``.  A square block starts with a
mode-specific position and velocity chosen by reversing the contact-free
dynamics from one common query state.  The two real histories therefore show
different decay, naturally meet at the same query state, and then execute the
same query action.  Each branch uses one simulator from x0 through x3.  No
state is installed after the initial reset and no contact arbiter is created.

This module deliberately keeps contact friction fixed.  Motion damping acts
between contacts and must not be confused with the contact-friction component.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Iterable

import numpy as np

from contextworld.evaluation import pusht_contact_friction_h3 as friction


ACTION_BLOCK = 5
HISTORY_TOKENS = 3
HISTORY_RAW_STEPS = 10
QUERY_RAW_STEPS = 5
CLIP_ACTION_BLOCKS = 4
MODEL_FRAME_ROWS = (0, 5, 10, 15)
DAMPING_VALUES = {
    "faster_decay": 0.2,
    "no_extra_decay": 1.0,
}
ENDPOINT_MODES = tuple(DAMPING_VALUES)
EFFECTIVE_CONTACT_FRICTION = 0.25
# This tolerance covers the complete 12-D state returned by ``body_snapshot``:
# agent/block position, linear velocity, angle, and angular velocity.  Angular
# dimensions use the wrapped difference implemented by ``_snapshot_delta``.
QUERY_STATE_TOLERANCE = 1e-8
QUERY_REFERENCE_TOLERANCE = 1e-8
MINIMUM_HISTORY_GAP_PX = 3.0
MINIMUM_FUTURE_GAP_PX = 2.0
TRANSFORM_CENTER = np.asarray([255.5, 255.5], dtype=np.float64)
CATALOG_SPLIT_TAGS = {
    "train": 211,
    "loader_validation": 223,
    "validation": 227,
    "confirmation": 233,
}


@dataclass(frozen=True)
class MotionDampingTemplate:
    """One deterministic paired motion-damping query."""

    template_id: str
    faster_decay_reset_snapshot: tuple[float, ...]
    no_extra_decay_reset_snapshot: tuple[float, ...]
    goal_state: tuple[float, ...]
    history_actions: tuple[tuple[float, float], ...]
    query_actions: tuple[tuple[float, float], ...]
    expected_natural_query_snapshot: tuple[float, ...]
    simulator_seed: int
    visible_shape_id: int = friction.DIAGNOSTIC_SHAPE_ID
    visible_shape_name: str = friction.DIAGNOSTIC_SHAPE_NAME


def _array(
    value: Iterable[float],
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _reset_snapshot_for_mode(
    template: MotionDampingTemplate, mode: str
) -> tuple[float, ...]:
    if mode == "faster_decay":
        return template.faster_decay_reset_snapshot
    if mode == "no_extra_decay":
        return template.no_extra_decay_reset_snapshot
    raise ValueError(f"Unknown motion-damping mode {mode!r}")


def _friction_template(template: MotionDampingTemplate, *, mode: str):
    reset = _array(
        _reset_snapshot_for_mode(template, mode),
        shape=(12,),
        name="reset_snapshot",
    )
    return friction.ContactFrictionTemplate(
        template_id=template.template_id,
        visible_shape_id=int(template.visible_shape_id),
        visible_shape_name=template.visible_shape_name,
        reset_state=(
            float(reset[0]),
            float(reset[1]),
            float(reset[6]),
            float(reset[7]),
            float(reset[10]),
            0.0,
            0.0,
        ),
        goal_state=template.goal_state,
        history_actions=template.history_actions,
        query_actions=template.query_actions,
        simulator_seed=int(template.simulator_seed),
        # The friction helper only uses this template to create/reset the
        # environment.  It must not receive a query snapshot that could be
        # restored after x0.
        canonical_query_snapshot=None,
    )


def make_motion_damping_env(
    template: MotionDampingTemplate,
    *,
    mode: str,
    resolution: int,
):
    """Create one PushT environment with only motion damping changed."""

    if mode not in DAMPING_VALUES:
        raise ValueError(f"Unknown motion-damping mode {mode!r}")
    env, info = friction.make_contact_friction_env(
        _friction_template(template, mode=mode),
        mode="medium_friction",
        resolution=int(resolution),
    )
    env.space.damping = float(DAMPING_VALUES[mode])
    # This is the single allowed installation: the initial x0 of the branch.
    friction.restore_body_snapshot(
        env, _reset_snapshot_for_mode(template, mode)
    )
    return env, info


def _rotate_snapshot(
    snapshot: Iterable[float],
    rotation: np.ndarray,
    angle: float,
    translation: np.ndarray,
) -> np.ndarray:
    value = _array(snapshot, shape=(12,), name="snapshot").copy()
    for position_slice, velocity_slice in (
        (slice(0, 2), slice(2, 4)),
        (slice(6, 8), slice(8, 10)),
    ):
        value[position_slice] = (
            TRANSFORM_CENTER
            + rotation @ (value[position_slice] - TRANSFORM_CENTER)
            + translation
        )
        value[velocity_slice] = rotation @ value[velocity_slice]
    # The pusher is circular; keep its visually irrelevant angle at zero.
    value[4] = 0.0
    value[10] = (value[10] + angle) % (2 * np.pi)
    return value


def rigid_transform_template(
    base: MotionDampingTemplate,
    *,
    template_id: str,
    angle_rad: float,
    tangential_offset: float,
    simulator_seed: int,
) -> MotionDampingTemplate:
    """Rotate and translate one complete causal construction."""

    angle = float(angle_rad) % (2 * np.pi)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )
    translation = rotation @ np.asarray(
        [float(tangential_offset), 0.0],
        dtype=np.float64,
    )
    faster_reset = _rotate_snapshot(
        base.faster_decay_reset_snapshot,
        rotation,
        angle,
        translation,
    )
    no_extra_reset = _rotate_snapshot(
        base.no_extra_decay_reset_snapshot,
        rotation,
        angle,
        translation,
    )
    expected_query = _rotate_snapshot(
        base.expected_natural_query_snapshot,
        rotation,
        angle,
        translation,
    )
    goal = _array(base.goal_state, shape=(7,), name="goal_state").copy()
    goal[0:2] = (
        TRANSFORM_CENTER
        + rotation @ (goal[0:2] - TRANSFORM_CENTER)
        + translation
    )
    goal[2:4] = (
        TRANSFORM_CENTER
        + rotation @ (goal[2:4] - TRANSFORM_CENTER)
        + translation
    )
    goal[4] = (goal[4] + angle) % (2 * np.pi)
    history = _array(
        base.history_actions,
        shape=(HISTORY_RAW_STEPS, 2),
        name="history_actions",
    ) @ rotation.T
    query = _array(
        base.query_actions,
        shape=(QUERY_RAW_STEPS, 2),
        name="query_actions",
    ) @ rotation.T
    return MotionDampingTemplate(
        template_id=template_id,
        faster_decay_reset_snapshot=tuple(faster_reset.tolist()),
        no_extra_decay_reset_snapshot=tuple(no_extra_reset.tolist()),
        goal_state=tuple(goal.tolist()),
        history_actions=tuple(map(tuple, history.tolist())),
        query_actions=tuple(map(tuple, query.tolist())),
        expected_natural_query_snapshot=tuple(expected_query.tolist()),
        simulator_seed=int(simulator_seed),
    )


def _free_motion_inverse_coefficient(damping: float) -> float:
    """Return x0 displacement per unit x2 velocity for one history."""

    alpha = float(damping) ** 0.01
    substeps = HISTORY_RAW_STEPS * 10
    velocity_scale = alpha**substeps
    displacement_scale = 0.01 * sum(alpha**index for index in range(substeps))
    return displacement_scale / velocity_scale


def _make_contact_free_template(
    *,
    template_id: str,
    center: np.ndarray,
    query_velocity: np.ndarray,
    agent_position: np.ndarray,
    goal_position: np.ndarray,
    goal_angle: float,
    block_angle: float,
    mirrored: bool,
    simulator_seed: int,
) -> MotionDampingTemplate:
    """Build one member of an x0-RGB-balanced forward/reverse twin."""

    center = _array(center, shape=(2,), name="center")
    velocity = _array(
        query_velocity, shape=(2,), name="query_velocity"
    )
    if mirrored:
        velocity = -velocity
    coefficients = {
        mode: _free_motion_inverse_coefficient(value)
        for mode, value in DAMPING_VALUES.items()
    }
    midpoint_offset = 0.5 * sum(coefficients.values()) * velocity
    query_position = center + midpoint_offset
    expected_query = np.asarray(
        (
            float(agent_position[0]),
            float(agent_position[1]),
            0.0,
            0.0,
            0.0,
            0.0,
            float(query_position[0]),
            float(query_position[1]),
            float(velocity[0]),
            float(velocity[1]),
            float(block_angle),
            0.0,
        ),
        dtype=np.float64,
    )

    def reverse_free_motion(mode: str) -> tuple[float, ...]:
        result = expected_query.copy()
        damping = DAMPING_VALUES[mode]
        result[8:10] = expected_query[8:10] / damping
        result[6:8] = (
            expected_query[6:8]
            - coefficients[mode] * expected_query[8:10]
        )
        return tuple(result.tolist())

    zeros_history = np.zeros((HISTORY_RAW_STEPS, 2), dtype=np.float64)
    zeros_query = np.zeros((QUERY_RAW_STEPS, 2), dtype=np.float64)
    return MotionDampingTemplate(
        template_id=template_id,
        faster_decay_reset_snapshot=reverse_free_motion("faster_decay"),
        no_extra_decay_reset_snapshot=reverse_free_motion("no_extra_decay"),
        goal_state=(
            float(agent_position[0]),
            float(agent_position[1]),
            float(goal_position[0]),
            float(goal_position[1]),
            float(goal_angle),
            0.0,
            0.0,
        ),
        history_actions=tuple(map(tuple, zeros_history.tolist())),
        query_actions=tuple(map(tuple, zeros_query.tolist())),
        expected_natural_query_snapshot=tuple(expected_query.tolist()),
        simulator_seed=int(simulator_seed),
    )


def make_base_template() -> MotionDampingTemplate:
    """Return the forward member of the deterministic diagnostic twin."""

    return _make_contact_free_template(
        template_id="pmd-h3-base-contact-free-forward-v4",
        center=np.asarray([256.0, 256.0]),
        query_velocity=np.asarray([20.0, 0.0]),
        agent_position=np.asarray([256.0, 376.0]),
        goal_position=np.asarray([360.0, 360.0]),
        goal_angle=0.0,
        block_angle=0.0,
        mirrored=False,
        simulator_seed=42,
    )


def make_mirrored_base_template() -> MotionDampingTemplate:
    """Return the twin whose two x0 RGB images exchange damping labels."""

    return _make_contact_free_template(
        template_id="pmd-h3-base-contact-free-reverse-v4",
        center=np.asarray([256.0, 256.0]),
        query_velocity=np.asarray([20.0, 0.0]),
        agent_position=np.asarray([256.0, 376.0]),
        goal_position=np.asarray([360.0, 360.0]),
        goal_angle=0.0,
        block_angle=0.0,
        mirrored=True,
        simulator_seed=42,
    )


def make_confirmation_templates() -> list[MotionDampingTemplate]:
    """Return eight independent orientation/position confirmations."""

    return [
        _make_catalog_template(
            split="confirmation",
            catalog_index=index,
            catalog_seed=20260803,
        )
        for index in range(8)
    ]


def _make_catalog_template(
    *,
    split: str,
    catalog_index: int,
    catalog_seed: int,
) -> MotionDampingTemplate:
    if split not in CATALOG_SPLIT_TAGS:
        raise ValueError(f"Unknown split {split!r}")
    group_index = int(catalog_index) // 2
    mirrored = bool(int(catalog_index) % 2)
    generator = np.random.default_rng(
        np.random.SeedSequence(
            [int(catalog_seed), CATALOG_SPLIT_TAGS[split], group_index]
        )
    )
    # Each adjacent forward/reverse pair shares every rendered nuisance.  Its
    # two x0 RGB images exchange damping labels exactly, while query position,
    # velocity direction/magnitude, block angle, and goal vary by group.
    for _ in range(10_000):
        direction_angle = float(generator.uniform(0.0, 2 * np.pi))
        direction = np.asarray(
            [np.cos(direction_angle), np.sin(direction_angle)],
            dtype=np.float64,
        )
        perpendicular = np.asarray([-direction[1], direction[0]])
        speed = float(generator.uniform(14.0, 24.0))
        center = generator.uniform(105.0, 407.0, size=2)
        agent_position = center + 95.0 * perpendicular
        goal_position = generator.uniform(80.0, 432.0, size=2)
        goal_angle = float(generator.uniform(0.0, 2 * np.pi))
        block_angle = float(
            (
                (group_index % 4) * (np.pi / 2)
                + generator.uniform(-0.25, 0.25)
            )
            % (2 * np.pi)
        )
        simulator_seed = int(generator.integers(0, 2**31 - 1))
        twins = {
            is_mirrored: _make_contact_free_template(
                template_id=(
                    f"pmd-{split.replace('_', '-')}-{catalog_index:06d}-"
                    f"{'reverse' if is_mirrored else 'forward'}"
                ),
                center=center,
                query_velocity=speed * direction,
                agent_position=agent_position,
                goal_position=goal_position,
                goal_angle=goal_angle,
                block_angle=block_angle,
                mirrored=is_mirrored,
                simulator_seed=simulator_seed,
            )
            for is_mirrored in (False, True)
        }
        block_positions = []
        for candidate in twins.values():
            snapshots = (
                candidate.faster_decay_reset_snapshot,
                candidate.no_extra_decay_reset_snapshot,
                candidate.expected_natural_query_snapshot,
            )
            block_positions.extend(
                np.asarray(value)[6:8] for value in snapshots
            )
            query_state = np.asarray(candidate.expected_natural_query_snapshot)
            # Keep both the scored x3 and the unscored five-row
            # format-completion block away from walls.
            block_positions.append(
                query_state[6:8] + 1.1 * query_state[8:10]
            )
        if (
            np.all(agent_position >= 25.0)
            and np.all(agent_position <= 487.0)
            and all(
                np.all(position >= 66.0) and np.all(position <= 446.0)
                for position in block_positions
            )
        ):
            return twins[mirrored]
    raise RuntimeError("Unable to sample a contact-free motion-damping twin")


def make_catalog_template(
    *,
    split: str,
    catalog_index: int,
    catalog_seed: int,
) -> MotionDampingTemplate:
    """Create a deterministic split-specific template without overlap."""

    return _make_catalog_template(
        split=split,
        catalog_index=catalog_index,
        catalog_seed=catalog_seed,
    )


def _simulate_continuous_causal_chain(
    template: MotionDampingTemplate,
    *,
    mode: str,
    resolution: int,
    render_pixels: bool,
) -> dict[str, Any]:
    """Run x0 -> x1 -> x2 -> x3 in one simulator without a query reset."""

    history_actions = _array(
        template.history_actions,
        shape=(HISTORY_RAW_STEPS, 2),
        name="history_actions",
    )
    query_actions = _array(
        template.query_actions,
        shape=(QUERY_RAW_STEPS, 2),
        name="query_actions",
    )
    env, info = make_motion_damping_env(
        template,
        mode=mode,
        resolution=resolution,
    )
    snapshots = [friction.body_snapshot(env)]
    pixels = [np.asarray(env.render(), dtype=np.uint8).copy()]
    history_contacts: list[int] = []
    query_contacts: list[int] = []
    history_arbiter_counts: list[int] = []
    query_arbiter_counts: list[int] = []
    bounds = [friction._body_shape_bounds(env)]
    try:
        for index, action in enumerate(history_actions, start=1):
            history_contacts.append(
                friction._step_and_count_agent_block_contacts(env, action)
            )
            history_arbiter_counts.append(len(env.space._get_arbiters()))
            bounds.append(friction._body_shape_bounds(env))
            if index in (ACTION_BLOCK, HISTORY_RAW_STEPS):
                snapshots.append(friction.body_snapshot(env))
                if render_pixels:
                    pixels.append(
                        np.asarray(env.render(), dtype=np.uint8).copy()
                    )
        natural_query_snapshot = friction.body_snapshot(env)
        natural_query_pixels = (
            np.asarray(env.render(), dtype=np.uint8).copy()
            if render_pixels
            else None
        )
        for action in query_actions:
            query_contacts.append(
                friction._step_and_count_agent_block_contacts(env, action)
            )
            query_arbiter_counts.append(len(env.space._get_arbiters()))
            bounds.append(friction._body_shape_bounds(env))
        future_snapshot = friction.body_snapshot(env)
        future_pixels = (
            np.asarray(env.render(), dtype=np.uint8).copy()
            if render_pixels
            else None
        )
    finally:
        env.close()
    return {
        "snapshots": np.stack(snapshots),
        "pixels": np.stack(pixels) if render_pixels else None,
        "history_actions": history_actions,
        "query_actions": query_actions,
        "history_contacts": np.asarray(history_contacts, dtype=np.int16),
        "query_contacts": np.asarray(query_contacts, dtype=np.int16),
        "history_arbiter_counts": np.asarray(
            history_arbiter_counts, dtype=np.int16
        ),
        "query_arbiter_counts": np.asarray(query_arbiter_counts, dtype=np.int16),
        "bounds": bounds,
        "goal_pixels": np.asarray(info["goal"], dtype=np.uint8).copy(),
        "query_snapshot": natural_query_snapshot,
        "future_snapshot": future_snapshot,
        "query_pixels": natural_query_pixels,
        "future_pixels": future_pixels,
        "state_installations_after_x0": 0,
        "query_simulator_recreated": False,
    }


def _array_hash(value: np.ndarray) -> str:
    data = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(data.dtype).encode("ascii"))
    digest.update(np.asarray(data.shape, dtype=np.int64).tobytes())
    digest.update(data.tobytes())
    return digest.hexdigest()


def evaluate_template(
    template: MotionDampingTemplate,
    *,
    resolution: int = 96,
) -> dict[str, Any]:
    """Evaluate all preregistered physical and causal gates."""

    rollouts = {
        mode: _simulate_continuous_causal_chain(
            template,
            mode=mode,
            resolution=resolution,
            render_pixels=True,
        )
        for mode in ENDPOINT_MODES
    }
    low, high = (rollouts[mode] for mode in ENDPOINT_MODES)
    expected_query = _array(
        template.expected_natural_query_snapshot,
        shape=(12,),
        name="expected_natural_query_snapshot",
    )
    reference_deviations = {
        mode: float(
            np.max(
                np.abs(
                    friction._snapshot_delta(
                        rollouts[mode]["query_snapshot"],
                        expected_query,
                    )
                )
            )
        )
        for mode in ENDPOINT_MODES
    }
    history_gap = friction._visible_response_gap(
        low["snapshots"][1],
        high["snapshots"][1],
        angular_radius_px=60.0,
    )
    future_gap = friction._future_gap(low["future_snapshot"], high["future_snapshot"])
    query_state_gap = float(
        np.max(
            np.abs(
                friction._snapshot_delta(
                    low["query_snapshot"], high["query_snapshot"]
                )
            )
        )
    )
    query_pixel_difference = int(
        np.max(
            np.abs(
                low["query_pixels"].astype(np.int16)
                - high["query_pixels"].astype(np.int16)
            )
        )
    )
    query_action_difference = float(
        np.max(np.abs(low["query_actions"] - high["query_actions"]))
    )
    checks = {
        "mode_specific_initial_states": not np.array_equal(
            low["snapshots"][0], high["snapshots"][0]
        ),
        "history_actions_identical": np.array_equal(
            low["history_actions"], high["history_actions"]
        ),
        "history_contact_free": bool(
            np.all(low["history_contacts"] == 0)
            and np.all(high["history_contacts"] == 0)
        ),
        "history_has_no_cached_arbiters": bool(
            np.all(low["history_arbiter_counts"] == 0)
            and np.all(high["history_arbiter_counts"] == 0)
        ),
        "history_visible_gap_sufficient": (
            history_gap["px_equivalent"] >= MINIMUM_HISTORY_GAP_PX
        ),
        "each_trajectory_matches_expected_natural_query": all(
            value <= QUERY_REFERENCE_TOLERANCE
            for value in reference_deviations.values()
        ),
        "query_full_state_within_registered_tolerance": (
            query_state_gap <= QUERY_STATE_TOLERANCE
        ),
        "query_pixels_identical": np.array_equal(
            low["query_pixels"], high["query_pixels"]
        ),
        "query_actions_identical": np.array_equal(
            low["query_actions"], high["query_actions"]
        ),
        "query_contact_free": bool(
            np.all(low["query_contacts"] == 0)
            and np.all(high["query_contacts"] == 0)
        ),
        "query_has_no_cached_arbiters": bool(
            np.all(low["query_arbiter_counts"] == 0)
            and np.all(high["query_arbiter_counts"] == 0)
        ),
        "future_pixels_different": not np.array_equal(
            low["future_pixels"], high["future_pixels"]
        ),
        "future_gap_sufficient": (
            future_gap["block_position_px"] >= MINIMUM_FUTURE_GAP_PX
        ),
        "all_bodies_inside_playfield": all(
            friction._bounds_inside_playfield(bounds)
            for rollout in rollouts.values()
            for bounds in rollout["bounds"]
        ),
        "goal_pixels_identical": np.array_equal(
            low["goal_pixels"], high["goal_pixels"]
        ),
        "state_installations_after_x0_zero": all(
            rollout["state_installations_after_x0"] == 0
            for rollout in rollouts.values()
        ),
        "query_simulator_not_recreated": all(
            not rollout["query_simulator_recreated"]
            for rollout in rollouts.values()
        ),
    }
    return {
        "template_id": template.template_id,
        "passed": all(checks.values()),
        "checks": checks,
        "history_visible_response_gap": history_gap,
        "query_reference_deviation": reference_deviations,
        "max_pair_full_state_gap": query_state_gap,
        "max_pair_query_pixel_difference": query_pixel_difference,
        "max_pair_query_action_difference": query_action_difference,
        "query_full_state_tolerance": QUERY_STATE_TOLERANCE,
        "query_full_state_dimensions": (
            "agent_position_velocity_angle_angular_velocity_and_"
            "block_position_velocity_angle_angular_velocity"
        ),
        "state_installations_after_x0": 0,
        "query_simulator_recreated": False,
        "maximum_arbiter_count_from_x0_through_x3": int(
            max(
                max(
                    np.max(rollout["history_arbiter_counts"], initial=0),
                    np.max(rollout["query_arbiter_counts"], initial=0),
                )
                for rollout in rollouts.values()
            )
        ),
        "future_gap": future_gap,
        "contact_steps": {
            "faster_decay_history": int(
                np.count_nonzero(low["history_contacts"])
            ),
            "no_extra_decay_history": int(
                np.count_nonzero(high["history_contacts"])
            ),
            "faster_decay_query": int(
                np.count_nonzero(low["query_contacts"])
            ),
            "no_extra_decay_query": int(
                np.count_nonzero(high["query_contacts"])
            ),
        },
        "hashes": {
            "faster_decay_initial_pixels": _array_hash(low["pixels"][0]),
            "no_extra_decay_initial_pixels": _array_hash(high["pixels"][0]),
            "faster_decay_history_pixels": _array_hash(low["pixels"][1]),
            "no_extra_decay_history_pixels": _array_hash(high["pixels"][1]),
            "query_pixels": _array_hash(low["query_pixels"]),
            "faster_decay_future_pixels": _array_hash(
                low["future_pixels"]
            ),
            "no_extra_decay_future_pixels": _array_hash(
                high["future_pixels"]
            ),
            "history_actions": _array_hash(low["history_actions"]),
            "query_actions": _array_hash(low["query_actions"]),
        },
    }


def simulate_motion_damping_clip(
    template: MotionDampingTemplate,
    *,
    mode: str,
    resolution: int = 224,
) -> dict[str, Any]:
    """Render one formal 20-row Stable-WorldModel History=3 clip."""

    history = _array(
        template.history_actions,
        shape=(HISTORY_RAW_STEPS, 2),
        name="history_actions",
    )
    query = _array(
        template.query_actions,
        shape=(QUERY_RAW_STEPS, 2),
        name="query_actions",
    )
    actions = np.concatenate(
        [history, query, np.zeros((ACTION_BLOCK, 2), dtype=np.float64)]
    )
    expected_query = _array(
        template.expected_natural_query_snapshot,
        shape=(12,),
        name="expected_natural_query_snapshot",
    )
    goal_state = _array(
        template.goal_state,
        shape=(7,),
        name="goal_state",
    ).astype(np.float32)
    rows: dict[str, list[Any]] = {
        "pixels": [],
        "action": [],
        "proprio": [],
        "state": [],
        "goal_state": [],
        "physics_state": [],
        "n_contacts": [],
        "hidden_motion_damping": [],
        "pair_id": [],
        "hidden_mode": [],
    }
    model_pixels: list[np.ndarray] = []
    model_physics_states: list[np.ndarray] = []
    body_bounds: list[dict[str, list[float]]] = []
    contact_steps_by_block = [0] * CLIP_ACTION_BLOCKS
    arbiter_count_by_raw_step: list[int] = []

    def capture_and_step(env, raw_step: int, action: np.ndarray) -> None:
        state = np.asarray(env._get_obs(), dtype=np.float64)
        physics = friction.body_snapshot(env)
        pixels = np.asarray(env.render(), dtype=np.uint8).copy()
        if raw_step in MODEL_FRAME_ROWS:
            body_bounds.append(friction._body_shape_bounds(env))
            model_pixels.append(pixels)
            model_physics_states.append(physics.astype(np.float32))
        rows["pixels"].append(pixels)
        rows["action"].append(np.asarray(action, dtype=np.float32).copy())
        rows["proprio"].append(
            np.concatenate([state[:2], state[-2:]]).astype(np.float32)
        )
        rows["state"].append(state.astype(np.float32))
        rows["goal_state"].append(goal_state.copy())
        rows["physics_state"].append(physics.astype(np.float32))
        rows["n_contacts"].append(np.asarray([0.0], dtype=np.float32))
        rows["hidden_motion_damping"].append(
            np.asarray([DAMPING_VALUES[mode]], dtype=np.float32)
        )
        rows["pair_id"].append(template.template_id)
        rows["hidden_mode"].append(mode)
        contacts = friction._step_and_count_agent_block_contacts(env, action)
        arbiter_count_by_raw_step.append(len(env.space._get_arbiters()))
        rows["n_contacts"][-1][0] = float(contacts)
        contact_steps_by_block[raw_step // ACTION_BLOCK] += int(contacts > 0)

    env, info = make_motion_damping_env(
        template,
        mode=mode,
        resolution=resolution,
    )
    natural_query_snapshot: np.ndarray | None = None
    natural_future_snapshot: np.ndarray | None = None
    try:
        for raw_step, action in enumerate(actions):
            if raw_step == HISTORY_RAW_STEPS:
                natural_query_snapshot = friction.body_snapshot(env)
            if raw_step == HISTORY_RAW_STEPS + QUERY_RAW_STEPS:
                natural_future_snapshot = friction.body_snapshot(env)
            capture_and_step(env, raw_step, action)
    finally:
        env.close()
    if natural_query_snapshot is None:
        raise AssertionError("The continuous rollout did not reach x2")
    if natural_future_snapshot is None:
        raise AssertionError("The continuous rollout did not reach x3")
    reference_deviation = float(
        np.max(
            np.abs(
                friction._snapshot_delta(
                    natural_query_snapshot,
                    expected_query,
                )
            )
        )
    )
    if reference_deviation > QUERY_REFERENCE_TOLERANCE:
        raise RuntimeError(
            "History did not naturally reach the expected query region: "
            f"template={template.template_id}, mode={mode}, "
            f"deviation={reference_deviation:.10f}"
        )
    return {
        "template": asdict(template),
        "mode": mode,
        "damping": DAMPING_VALUES[mode],
        "raw_actions": actions.astype(np.float32),
        "action_blocks": actions.reshape(CLIP_ACTION_BLOCKS, ACTION_BLOCK, 2),
        "rows": rows,
        "model_pixels": np.stack(model_pixels),
        "model_physics_states": np.stack(model_physics_states),
        "body_bounds": body_bounds,
        "goal_pixels": np.asarray(info["goal"], dtype=np.uint8).copy(),
        "contact_steps_by_block": contact_steps_by_block,
        "history_contact_steps": sum(contact_steps_by_block[:2]),
        "query_contact_steps": contact_steps_by_block[2],
        "arbiter_count_by_raw_step": arbiter_count_by_raw_step,
        "natural_query_snapshot": natural_query_snapshot,
        "natural_future_snapshot": natural_future_snapshot,
        "query_reference_deviation": reference_deviation,
        "query_boundary": "single_continuous_simulator_from_x0_through_x3",
        "state_installations_after_x0": 0,
        "query_simulator_recreated": False,
    }


def validate_motion_damping_pair(
    faster: dict[str, Any],
    no_extra: dict[str, Any],
) -> dict[str, Any]:
    """Validate one stored pair before it is admitted to a split."""

    left_pixels = np.asarray(faster["model_pixels"])
    right_pixels = np.asarray(no_extra["model_pixels"])
    left_states = np.asarray(faster["model_physics_states"], dtype=np.float64)
    right_states = np.asarray(
        no_extra["model_physics_states"], dtype=np.float64
    )
    history_gap = friction._visible_response_gap(
        left_states[1], right_states[1], angular_radius_px=60.0
    )
    future_gap = friction._future_gap(left_states[3], right_states[3])
    left_query_state = np.asarray(
        faster["natural_query_snapshot"], dtype=np.float64
    )
    right_query_state = np.asarray(
        no_extra["natural_query_snapshot"], dtype=np.float64
    )
    query_gap = float(
        np.max(
            np.abs(
                friction._snapshot_delta(left_query_state, right_query_state)
            )
        )
    )
    query_pixel_difference = int(
        np.max(
            np.abs(
                left_pixels[2].astype(np.int16)
                - right_pixels[2].astype(np.int16)
            )
        )
    )
    query_action_difference = float(
        np.max(
            np.abs(
                np.asarray(faster["action_blocks"][2], dtype=np.float64)
                - np.asarray(no_extra["action_blocks"][2], dtype=np.float64)
            )
        )
    )
    checks = {
        "template_identity": faster["template"] == no_extra["template"],
        "endpoint_modes": (
            faster["mode"] == ENDPOINT_MODES[0]
            and no_extra["mode"] == ENDPOINT_MODES[1]
        ),
        "mode_specific_initial_states": not np.array_equal(
            left_states[0], right_states[0]
        ),
        "actions_identical": np.array_equal(
            faster["raw_actions"], no_extra["raw_actions"]
        ),
        "history_contact_free": (
            int(faster["history_contact_steps"]) == 0
            and int(no_extra["history_contact_steps"]) == 0
        ),
        "history_and_query_have_no_cached_arbiters": (
            max(
                faster["arbiter_count_by_raw_step"][
                    : HISTORY_RAW_STEPS + QUERY_RAW_STEPS
                ],
                default=0,
            )
            == 0
            and max(
                no_extra["arbiter_count_by_raw_step"][
                    : HISTORY_RAW_STEPS + QUERY_RAW_STEPS
                ],
                default=0,
            )
            == 0
        ),
        "history_pixels_different": not np.array_equal(
            left_pixels[1], right_pixels[1]
        ),
        "history_gap_sufficient": (
            history_gap["px_equivalent"] >= MINIMUM_HISTORY_GAP_PX
        ),
        "natural_query_matches_expected_reference": (
            float(faster["query_reference_deviation"])
            <= QUERY_REFERENCE_TOLERANCE
            and float(no_extra["query_reference_deviation"])
            <= QUERY_REFERENCE_TOLERANCE
        ),
        "query_full_state_within_registered_tolerance": (
            query_gap <= QUERY_STATE_TOLERANCE
        ),
        "query_pixels_identical": np.array_equal(
            left_pixels[2], right_pixels[2]
        ),
        "query_actions_identical": query_action_difference == 0.0,
        "state_installations_after_x0_zero": (
            int(faster["state_installations_after_x0"]) == 0
            and int(no_extra["state_installations_after_x0"]) == 0
        ),
        "query_simulator_not_recreated": (
            not bool(faster["query_simulator_recreated"])
            and not bool(no_extra["query_simulator_recreated"])
        ),
        "query_contact_free": (
            int(faster["query_contact_steps"]) == 0
            and int(no_extra["query_contact_steps"]) == 0
        ),
        "future_pixels_different": not np.array_equal(
            left_pixels[3], right_pixels[3]
        ),
        "future_gap_sufficient": (
            future_gap["block_position_px"] >= MINIMUM_FUTURE_GAP_PX
        ),
        "all_bodies_inside_playfield": all(
            friction._bounds_inside_playfield(bounds)
            for rollout in (faster, no_extra)
            for bounds in rollout["body_bounds"]
        ),
        "goal_pixels_identical": np.array_equal(
            faster["goal_pixels"], no_extra["goal_pixels"]
        ),
    }
    return {
        "template_id": faster["template"]["template_id"],
        "passed": all(checks.values()),
        "checks": checks,
        "history_visible_response_gap": history_gap,
        "future_gap": future_gap,
        "max_pair_full_state_gap": query_gap,
        "max_pair_query_pixel_difference": query_pixel_difference,
        "max_pair_query_action_difference": query_action_difference,
        "query_full_state_tolerance": QUERY_STATE_TOLERANCE,
        "query_full_state_dimensions": (
            "agent_position_velocity_angle_angular_velocity_and_"
            "block_position_velocity_angle_angular_velocity"
        ),
        "query_reference_deviation": {
            faster["mode"]: float(faster["query_reference_deviation"]),
            no_extra["mode"]: float(no_extra["query_reference_deviation"]),
        },
        "state_installations_after_x0": 0,
        "query_simulator_recreated": False,
        "maximum_arbiter_count_from_x0_through_x3": int(
            max(
                max(
                    faster["arbiter_count_by_raw_step"][
                        : HISTORY_RAW_STEPS + QUERY_RAW_STEPS
                    ],
                    default=0,
                ),
                max(
                    no_extra["arbiter_count_by_raw_step"][
                        : HISTORY_RAW_STEPS + QUERY_RAW_STEPS
                    ],
                    default=0,
                ),
            )
        ),
        "hashes": {
            "faster_decay_initial_pixels": _array_hash(left_pixels[0]),
            "no_extra_decay_initial_pixels": _array_hash(right_pixels[0]),
            "faster_decay_history_pixels": _array_hash(left_pixels[1]),
            "no_extra_decay_history_pixels": _array_hash(right_pixels[1]),
            "query_pixels": _array_hash(left_pixels[2]),
            "faster_decay_future_pixels": _array_hash(left_pixels[3]),
            "no_extra_decay_future_pixels": _array_hash(right_pixels[3]),
            "raw_actions": _array_hash(faster["raw_actions"]),
        },
    }


__all__ = [
    "ACTION_BLOCK",
    "CLIP_ACTION_BLOCKS",
    "DAMPING_VALUES",
    "ENDPOINT_MODES",
    "HISTORY_RAW_STEPS",
    "HISTORY_TOKENS",
    "MODEL_FRAME_ROWS",
    "MotionDampingTemplate",
    "QUERY_RAW_STEPS",
    "evaluate_template",
    "make_base_template",
    "make_catalog_template",
    "make_confirmation_templates",
    "make_motion_damping_env",
    "rigid_transform_template",
    "simulate_motion_damping_clip",
    "validate_motion_damping_pair",
]
