"""History-3 physical audit for hidden PushT contact friction.

This module does not build a benchmark dataset and does not evaluate a model.
It checks the causal construction that must pass before either is allowed:

* low- and high-friction trajectories receive the same history actions;
* contact friction leaves visible evidence in the history;
* each trajectory runs continuously and naturally reaches one query state;
* the same query action produces friction-dependent true futures.

Friction is assigned to Pymunk collision ``Shape`` objects.  ``space.damping``
is fixed to zero because post-release motion damping is a different hidden
factor.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from stable_worldmodel.envs.pusht.env import PushT


ACTION_BLOCK = 5
HISTORY_TOKENS = 3
HISTORY_RAW_STEPS = (HISTORY_TOKENS - 1) * ACTION_BLOCK
QUERY_RAW_STEPS = ACTION_BLOCK
MODEL_FRAME_ROWS = (0, 5, 10, 15)
CLIP_ACTION_BLOCKS = 4
CLIP_RAW_STEPS = CLIP_ACTION_BLOCKS * ACTION_BLOCK
FRICTION_VALUES = {
    "low_friction": 0.05,
    "medium_friction": 0.25,
    "high_friction": 0.80,
}
ENDPOINT_MODES = ("low_friction", "high_friction")
PRIMARY_SHAPE_ID = 2
PRIMARY_SHAPE_NAME = "T"
DIAGNOSTIC_SHAPE_ID = 4
DIAGNOSTIC_SHAPE_NAME = "square"
MODEL_VISIBLE_FIELDS = ("pixels", "action")
AGENT_COLLISION_TYPE = 101
BLOCK_COLLISION_TYPE = 102
WALL_COLLISION_TYPE = 103
PLAYFIELD_MIN = 5.0
PLAYFIELD_MAX = 506.0
CATALOG_ROTATION_CENTER = (340.0, 300.0)
BASE_CANONICAL_QUERY_SNAPSHOT = (
    351.7366779912398,
    260.0005237670546,
    -2.7816062324603763e-06,
    -7.818017313532266e-07,
    0.0,
    0.0,
    334.25714033730696,
    340.0259854176169,
    -1.4659727363899928e-07,
    1.344094388919411e-07,
    2.222083177890679,
    2.7429595225853573e-10,
)
STRICT_QUERY_FULL_STATE_TOLERANCE = 1.0e-5
STRICT_INITIAL_FULL_STATE_TOLERANCE = 0.15
STRICT_CLEAN_REPLAY_FULL_STATE_TOLERANCE = 1.0e-5
STRICT_MINIMUM_CACHE_CLEAR_STEPS = 3
STRICT_CONTINUOUS_CONSTRUCTION = (
    "continuous_forward_natural_query_no_post_x0_state_installation_v2"
)
CATALOG_SPLIT_TAGS = {
    "train": 101,
    "loader_validation": 103,
    "validation": 107,
}
STRATIFIED_TRAINING_ANGLE_BINS = 16
STRATIFIED_TRAINING_TRANSLATION_BINS = (8, 8)


def _action_array(
    value: Iterable[Iterable[float]],
    *,
    expected_steps: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(tuple(tuple(row) for row in value), dtype=np.float32)
    if result.shape != (expected_steps, 2):
        raise ValueError(
            f"{name} must have shape {(expected_steps, 2)}, got "
            f"{result.shape}"
        )
    if np.any(result < -1.0) or np.any(result > 1.0):
        raise ValueError(f"{name} must stay inside [-1, 1]")
    return result


def _state_array(
    value: Iterable[float],
    *,
    expected_size: int,
    name: str,
) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (expected_size,):
        raise ValueError(
            f"{name} must have shape {(expected_size,)}, got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite value")
    return result


@dataclass(frozen=True)
class ContactFrictionTemplate:
    """One deterministic candidate for a paired friction query."""

    template_id: str
    visible_shape_id: int
    visible_shape_name: str
    reset_state: tuple[float, ...]
    goal_state: tuple[float, ...]
    history_actions: tuple[tuple[float, float], ...]
    query_actions: tuple[tuple[float, float], ...]
    simulator_seed: int
    canonical_query_snapshot: tuple[float, ...] | None = None
    low_friction_reset_state: tuple[float, ...] | None = None
    high_friction_reset_state: tuple[float, ...] | None = None
    causal_construction: str = "legacy_shared_reset"
    strict_family_id: int | None = None

    def __post_init__(self) -> None:
        if not self.template_id:
            raise ValueError("template_id must be non-empty")
        if int(self.visible_shape_id) not in (
            PRIMARY_SHAPE_ID,
            DIAGNOSTIC_SHAPE_ID,
        ):
            raise ValueError("Only the preregistered T and square are allowed")
        _state_array(
            self.reset_state,
            expected_size=7,
            name="reset_state",
        )
        _state_array(
            self.goal_state,
            expected_size=7,
            name="goal_state",
        )
        _action_array(
            self.history_actions,
            expected_steps=HISTORY_RAW_STEPS,
            name="history_actions",
        )
        _action_array(
            self.query_actions,
            expected_steps=QUERY_RAW_STEPS,
            name="query_actions",
        )
        if self.canonical_query_snapshot is not None:
            _state_array(
                self.canonical_query_snapshot,
                expected_size=12,
                name="canonical_query_snapshot",
            )
        for name, value in (
            ("low_friction_reset_state", self.low_friction_reset_state),
            ("high_friction_reset_state", self.high_friction_reset_state),
        ):
            if value is not None:
                _state_array(value, expected_size=7, name=name)
        endpoint_resets = (
            self.low_friction_reset_state,
            self.high_friction_reset_state,
        )
        if any(value is not None for value in endpoint_resets) and not all(
            value is not None for value in endpoint_resets
        ):
            raise ValueError(
                "low/high friction reset states must be provided together"
            )


def reset_state_for_mode(
    template: ContactFrictionTemplate,
    mode: str,
) -> np.ndarray:
    """Return the physical state installed by ``env.reset`` before x0.

    A mode-specific reset is allowed only as the initial condition.  Formal
    strict clips never install a state after the first model-visible frame.
    The two endpoint reset images are required to be bitwise identical by the
    pair validator, so these sub-rendering pose offsets cannot be used as a
    static visual label.
    """

    if mode == ENDPOINT_MODES[0] and template.low_friction_reset_state:
        value = template.low_friction_reset_state
    elif mode == ENDPOINT_MODES[1] and template.high_friction_reset_state:
        value = template.high_friction_reset_state
    else:
        value = template.reset_state
    return _state_array(value, expected_size=7, name=f"{mode}_reset_state")


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _angle_delta(left: float, right: float) -> float:
    return float((float(left) - float(right) + np.pi) % (2 * np.pi) - np.pi)


def _snapshot_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(
        right,
        dtype=np.float64,
    )
    if delta.shape != (12,):
        raise ValueError(f"Expected two 12-D snapshots, got {delta.shape}")
    delta[4] = _angle_delta(left[4], right[4])
    delta[10] = _angle_delta(left[10], right[10])
    return delta


def body_snapshot(env: PushT) -> np.ndarray:
    """Capture every simulator quantity that can affect the next transition."""

    return np.asarray(
        [
            *env.agent.position,
            *env.agent.velocity,
            float(env.agent.angle),
            float(env.agent.angular_velocity),
            *env.block.position,
            *env.block.velocity,
            float(env.block.angle),
            float(env.block.angular_velocity),
        ],
        dtype=np.float64,
    )


def simulator_state_audit(env: PushT) -> dict[str, Any]:
    """Capture all registered transition-relevant non-label state.

    PushT has two dynamic bodies and no dynamic constraints.  Pymunk's active
    arbiter list is audited explicitly because a cached contact impulse would
    otherwise be an unrecorded history variable at the query boundary.
    Shape friction is intentionally omitted from the equality projection: it
    is the hidden factor being evaluated and is reported separately.
    """

    return {
        "body_state": body_snapshot(env),
        "active_arbiter_count": int(len(env.space._get_arbiters())),
        "body_count": int(len(env.space.bodies)),
        "shape_count": int(len(env.space.shapes)),
        "constraint_count": int(len(env.space.constraints)),
        "gravity": np.asarray(env.space.gravity, dtype=np.float64),
        "damping": float(env.space.damping),
        "collision_bias": float(env.space.collision_bias),
        "collision_persistence": int(env.space.collision_persistence),
        "collision_slop": float(env.space.collision_slop),
    }


def simulator_state_max_abs_gap(
    left: dict[str, Any],
    right: dict[str, Any],
) -> float:
    """Return a scalar gap for the registered simulator equality fields."""

    scalar_fields = (
        "active_arbiter_count",
        "body_count",
        "shape_count",
        "constraint_count",
        "damping",
        "collision_bias",
        "collision_persistence",
        "collision_slop",
    )
    gaps = [
        float(
            np.max(
                np.abs(
                    _snapshot_delta(
                        np.asarray(left["body_state"], dtype=np.float64),
                        np.asarray(right["body_state"], dtype=np.float64),
                    )
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    np.asarray(left["gravity"], dtype=np.float64)
                    - np.asarray(right["gravity"], dtype=np.float64)
                )
            )
        ),
    ]
    gaps.extend(
        abs(float(left[name]) - float(right[name]))
        for name in scalar_fields
    )
    return float(max(gaps))


def restore_body_snapshot(env: PushT, snapshot: Iterable[float]) -> None:
    """Restore a full state without advancing the simulator."""

    state = _state_array(
        snapshot,
        expected_size=12,
        name="physics_snapshot",
    )
    # T-shaped blocks have an off-centre centre of gravity.  In Pymunk,
    # changing their angle after their position can move the body origin.
    # Restore angles first and positions last so the requested full snapshot
    # is exact and independent of the trajectory that preceded it.
    env.agent.angle = float(state[4])
    env.agent.angular_velocity = float(state[5])
    env.block.angle = float(state[10])
    env.block.angular_velocity = float(state[11])
    env.agent.position = tuple(state[0:2])
    env.agent.velocity = tuple(state[2:4])
    env.block.position = tuple(state[6:8])
    env.block.velocity = tuple(state[8:10])
    env.space.reindex_shapes_for_body(env.agent)
    env.space.reindex_shapes_for_body(env.block)


def midpoint_snapshot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return a deterministic midpoint, including wrapped body angles."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    result = left + 0.5 * _snapshot_delta(right, left)
    result[4] = result[4] % (2 * np.pi)
    result[10] = result[10] % (2 * np.pi)
    return result


def set_effective_contact_friction(env: PushT, coefficient: float) -> None:
    """Set the requested pusher-block coefficient on collision shapes.

    Chipmunk/Pymunk multiplies the two contacting shape coefficients.  Setting
    both sides to ``sqrt(mu)`` therefore produces an effective coefficient
    ``mu``.  Static-wall friction remains zero and ``space.damping`` remains
    zero, so the hidden factor is not mixed with wall friction or free-motion
    damping.
    """

    coefficient = float(coefficient)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("Contact friction must be finite and non-negative")
    shape_value = math.sqrt(coefficient)
    for shape in env.agent.shapes:
        shape.friction = shape_value
    for shape in env.block.shapes:
        shape.friction = shape_value
    for shape in env.space.shapes:
        if shape.body not in (env.agent, env.block):
            shape.friction = 0.0
    env.space.damping = 0.0


def register_agent_block_contact_counter(env: PushT) -> None:
    """Track only pusher-block contacts, excluding walls and self contacts."""

    for shape in env.agent.shapes:
        shape.collision_type = AGENT_COLLISION_TYPE
    for shape in env.block.shapes:
        shape.collision_type = BLOCK_COLLISION_TYPE
    for shape in env.space.shapes:
        if shape.body not in (env.agent, env.block):
            shape.collision_type = WALL_COLLISION_TYPE
    env._contextworld_agent_block_contact_points = 0

    def _count_agent_block_contacts(arbiter, _space, _data):
        env._contextworld_agent_block_contact_points += len(
            arbiter.contact_point_set.points
        )

    env.space.on_collision(
        AGENT_COLLISION_TYPE,
        BLOCK_COLLISION_TYPE,
        post_solve=_count_agent_block_contacts,
    )


def _step_and_count_agent_block_contacts(
    env: PushT,
    action: np.ndarray,
) -> int:
    env._contextworld_agent_block_contact_points = 0
    env.step(action)
    return int(env._contextworld_agent_block_contact_points)


def _body_shape_bounds(env: PushT) -> dict[str, list[float]]:
    def _bounds(shapes) -> list[float]:
        boxes = [shape.cache_bb() for shape in shapes]
        return [
            float(min(box.left for box in boxes)),
            float(min(box.bottom for box in boxes)),
            float(max(box.right for box in boxes)),
            float(max(box.top for box in boxes)),
        ]

    return {
        "agent": _bounds(env.agent.shapes),
        "block": _bounds(env.block.shapes),
    }


def _bounds_inside_playfield(bounds: dict[str, list[float]]) -> bool:
    return bool(
        all(
            box[0] >= PLAYFIELD_MIN
            and box[1] >= PLAYFIELD_MIN
            and box[2] <= PLAYFIELD_MAX
            and box[3] <= PLAYFIELD_MAX
            for box in bounds.values()
        )
    )


def friction_assignment_audit(
    env: PushT,
    requested_coefficient: float,
) -> dict[str, Any]:
    agent_values = sorted(float(shape.friction) for shape in env.agent.shapes)
    block_values = sorted(float(shape.friction) for shape in env.block.shapes)
    wall_values = sorted(
        float(shape.friction)
        for shape in env.space.shapes
        if shape.body not in (env.agent, env.block)
    )
    products = sorted(
        left * right for left in agent_values for right in block_values
    )
    requested = float(requested_coefficient)
    return {
        "requested_effective_coefficient": requested,
        "agent_shape_values": agent_values,
        "block_shape_values": block_values,
        "agent_block_products": products,
        "wall_shape_values": wall_values,
        "space_damping": float(env.space.damping),
        "passed": bool(
            products
            and all(
                np.isclose(value, requested, atol=1e-12, rtol=0.0)
                for value in products
            )
            and all(value == 0.0 for value in wall_values)
            and float(env.space.damping) == 0.0
        ),
    }


def _variation_values(
    template: ContactFrictionTemplate,
    *,
    mode: str,
) -> dict[str, Any]:
    reset = reset_state_for_mode(template, mode)
    goal = _state_array(
        template.goal_state,
        expected_size=7,
        name="goal_state",
    )
    # Shape creation uses the variation values before ``state`` restores the
    # exact preregistered pose.  Clipping here only satisfies the upstream
    # variation-space bounds; it does not change the restored state.
    return {
        "agent.start_position": np.clip(reset[:2], 50.0, 450.0),
        "agent.velocity": np.zeros(2, dtype=np.float64),
        "block.start_position": np.clip(reset[2:4], 100.0, 400.0),
        "block.angle": float(reset[4]),
        "block.shape": int(template.visible_shape_id),
        "goal.position": np.clip(goal[2:4], 50.0, 450.0),
        "goal.angle": float(goal[4]),
    }


def make_contact_friction_env(
    template: ContactFrictionTemplate,
    *,
    mode: str,
    resolution: int,
) -> tuple[PushT, dict[str, Any]]:
    if mode not in FRICTION_VALUES:
        raise ValueError(f"Unknown friction mode {mode!r}")
    env = PushT(
        resolution=int(resolution),
        with_target=True,
        render_mode="rgb_array",
        damping=0.0,
    )
    env.action_scale = 100.0
    reset_state = reset_state_for_mode(template, mode)
    _, info = env.reset(
        seed=int(template.simulator_seed),
        options={
            "variation": (),
            "variation_values": _variation_values(template, mode=mode),
            "state": reset_state,
            "goal_state": _state_array(
                template.goal_state,
                expected_size=7,
                name="goal_state",
            ),
        },
    )
    set_effective_contact_friction(env, FRICTION_VALUES[mode])
    register_agent_block_contact_counter(env)
    return env, info


def simulate_history(
    template: ContactFrictionTemplate,
    *,
    mode: str,
    resolution: int = 96,
    render_pixels: bool = True,
) -> dict[str, Any]:
    """Run the ten raw history steps without any reset or canonicalization."""

    actions = _action_array(
        template.history_actions,
        expected_steps=HISTORY_RAW_STEPS,
        name="history_actions",
    )
    env, info = make_contact_friction_env(
        template,
        mode=mode,
        resolution=resolution,
    )
    snapshots = [body_snapshot(env)]
    body_bounds = [_body_shape_bounds(env)]
    pixels = (
        [np.asarray(env.render(), dtype=np.uint8).copy()]
        if render_pixels
        else []
    )
    contact_counts: list[int] = []
    try:
        for raw_index, action in enumerate(actions):
            contact_counts.append(
                _step_and_count_agent_block_contacts(env, action)
            )
            if raw_index + 1 in (ACTION_BLOCK, HISTORY_RAW_STEPS):
                snapshots.append(body_snapshot(env))
                body_bounds.append(_body_shape_bounds(env))
                if render_pixels:
                    pixels.append(
                        np.asarray(env.render(), dtype=np.uint8).copy()
                    )
        assignment = friction_assignment_audit(
            env,
            FRICTION_VALUES[mode],
        )
    finally:
        env.close()
    return {
        "template": asdict(template),
        "mode": mode,
        "friction": FRICTION_VALUES[mode],
        "actions": actions,
        "snapshots": np.stack(snapshots),
        "pixels": np.stack(pixels) if render_pixels else None,
        "contact_counts": np.asarray(contact_counts, dtype=np.int16),
        "body_bounds": body_bounds,
        "goal_pixels": np.asarray(info["goal"], dtype=np.uint8).copy(),
        "friction_assignment": assignment,
    }


def simulate_query_future(
    template: ContactFrictionTemplate,
    *,
    mode: str,
    canonical_query_snapshot: Iterable[float],
    resolution: int = 96,
    render_pixels: bool = True,
) -> dict[str, Any]:
    """Run the common query action from one explicitly frozen full state."""

    actions = _action_array(
        template.query_actions,
        expected_steps=QUERY_RAW_STEPS,
        name="query_actions",
    )
    env, _ = make_contact_friction_env(
        template,
        mode=mode,
        resolution=resolution,
    )
    contact_counts: list[int] = []
    try:
        restore_body_snapshot(env, canonical_query_snapshot)
        initial_snapshot = body_snapshot(env)
        query_body_bounds = _body_shape_bounds(env)
        query_pixels = (
            np.asarray(env.render(), dtype=np.uint8).copy()
            if render_pixels
            else None
        )
        for action in actions:
            contact_counts.append(
                _step_and_count_agent_block_contacts(env, action)
            )
        future_snapshot = body_snapshot(env)
        future_body_bounds = _body_shape_bounds(env)
        future_pixels = (
            np.asarray(env.render(), dtype=np.uint8).copy()
            if render_pixels
            else None
        )
    finally:
        env.close()
    return {
        "mode": mode,
        "actions": actions,
        "query_snapshot": initial_snapshot,
        "future_snapshot": future_snapshot,
        "query_pixels": query_pixels,
        "future_pixels": future_pixels,
        "query_body_bounds": query_body_bounds,
        "future_body_bounds": future_body_bounds,
        "contact_counts": np.asarray(contact_counts, dtype=np.int16),
    }


def simulate_contact_friction_clip(
    template: ContactFrictionTemplate,
    *,
    mode: str,
    resolution: int = 224,
    query_state_tolerance: float = STRICT_QUERY_FULL_STATE_TOLERANCE,
    audit_clean_replay: bool = True,
) -> dict[str, Any]:
    """Render one strict, continuous 20-row contact-friction clip.

    Rows 0, 5, and 10 are the three model-visible history/query frames.  Row
    15 is the true next frame after the common five-step query action.  The
    final zero-action block only completes the 20-row Stable-WorldModel clip
    format and is not part of the scored prediction.  The same simulator runs
    every row: no state is installed and no simulator is recreated after x0.
    """

    if template.canonical_query_snapshot is None:
        raise ValueError("Strict clips require a frozen natural query target")
    if template.causal_construction != STRICT_CONTINUOUS_CONSTRUCTION:
        raise ValueError(
            "Formal clips require the strict continuous construction"
        )
    history = _action_array(
        template.history_actions,
        expected_steps=HISTORY_RAW_STEPS,
        name="history_actions",
    )
    query = _action_array(
        template.query_actions,
        expected_steps=QUERY_RAW_STEPS,
        name="query_actions",
    )
    raw_actions = np.concatenate(
        [
            history,
            query,
            np.zeros((ACTION_BLOCK, 2), dtype=np.float32),
        ],
        axis=0,
    )
    canonical = _state_array(
        template.canonical_query_snapshot,
        expected_size=12,
        name="canonical_query_snapshot",
    )
    rows: dict[str, list[Any]] = {
        "pixels": [],
        "action": [],
        "proprio": [],
        "state": [],
        "goal_state": [],
        "physics_state": [],
        "n_contacts": [],
        "hidden_contact_friction": [],
        "pair_id": [],
        "hidden_mode": [],
    }
    model_pixels: list[np.ndarray] = []
    model_states: list[np.ndarray] = []
    model_physics_states: list[np.ndarray] = []
    body_bounds: list[dict[str, list[float]]] = []
    contact_steps_by_block = [0] * CLIP_ACTION_BLOCKS
    query_snapshot: np.ndarray | None = None
    query_target_residual: float | None = None
    query_simulator_state: dict[str, Any] | None = None
    continuous_future_snapshot: np.ndarray | None = None
    continuous_future_pixels: np.ndarray | None = None
    goal_state = _state_array(
        template.goal_state,
        expected_size=7,
        name="goal_state",
    ).astype(np.float32)

    def capture_and_step(
        env: PushT,
        *,
        raw_step: int,
        action: np.ndarray,
    ) -> None:
        state = np.asarray(env._get_obs(), dtype=np.float64)
        physics = body_snapshot(env)
        pixels = np.asarray(env.render(), dtype=np.uint8).copy()
        body_bounds.append(_body_shape_bounds(env))
        if raw_step in MODEL_FRAME_ROWS:
            model_pixels.append(pixels)
            model_states.append(state.astype(np.float32))
            model_physics_states.append(physics.astype(np.float32))
        nonlocal query_snapshot
        nonlocal query_target_residual
        nonlocal query_simulator_state
        nonlocal continuous_future_snapshot
        nonlocal continuous_future_pixels
        if raw_step == HISTORY_RAW_STEPS:
            query_snapshot = physics.copy()
            query_target_residual = float(
                np.max(np.abs(_snapshot_delta(physics, canonical)))
            )
            query_simulator_state = simulator_state_audit(env)
        elif raw_step == HISTORY_RAW_STEPS + QUERY_RAW_STEPS:
            continuous_future_snapshot = physics.copy()
            continuous_future_pixels = pixels.copy()

        rows["pixels"].append(pixels)
        rows["action"].append(
            np.asarray(action, dtype=np.float32).copy()
        )
        rows["proprio"].append(
            np.concatenate([state[:2], state[-2:]]).astype(np.float32)
        )
        rows["state"].append(state.astype(np.float32))
        rows["goal_state"].append(goal_state.copy())
        rows["physics_state"].append(physics.astype(np.float32))
        rows["n_contacts"].append(
            np.asarray([0.0], dtype=np.float32)
        )
        rows["hidden_contact_friction"].append(
            np.asarray([FRICTION_VALUES[mode]], dtype=np.float32)
        )
        rows["pair_id"].append(template.template_id)
        rows["hidden_mode"].append(mode)

        contacts = _step_and_count_agent_block_contacts(env, action)
        rows["n_contacts"][-1][0] = float(contacts)
        contact_steps_by_block[raw_step // ACTION_BLOCK] += int(
            contacts > 0
        )

    env, info = make_contact_friction_env(
        template,
        mode=mode,
        resolution=resolution,
    )
    try:
        for raw_step, action in enumerate(raw_actions):
            capture_and_step(
                env,
                raw_step=raw_step,
                action=action,
            )
        assignment = friction_assignment_audit(
            env,
            FRICTION_VALUES[mode],
        )
        if query_target_residual is None:
            raise RuntimeError("The query boundary was not captured")
        if query_target_residual > query_state_tolerance:
            raise RuntimeError(
                "Continuous history did not naturally reach the frozen "
                "query target: "
                f"template={template.template_id}, mode={mode}, "
                f"residual={query_target_residual:.8f}, "
                f"tolerance={query_state_tolerance:.8f}"
            )
    finally:
        env.close()

    if query_snapshot is None or query_simulator_state is None:
        raise RuntimeError("Missing strict query audit state")
    if continuous_future_snapshot is None or continuous_future_pixels is None:
        raise RuntimeError("Missing continuous true future")
    trailing_no_contact_steps = 0
    for count in reversed(rows["n_contacts"][:HISTORY_RAW_STEPS]):
        if float(np.asarray(count)[0]) != 0.0:
            break
        trailing_no_contact_steps += 1

    clean_replay: dict[str, Any] | None = None
    if audit_clean_replay:
        diagnostic_reset = (
            80.0,
            80.0,
            300.0,
            300.0,
            0.0,
            0.0,
            0.0,
        )
        diagnostic_template = ContactFrictionTemplate(
            **{
                **asdict(template),
                "reset_state": diagnostic_reset,
                "low_friction_reset_state": diagnostic_reset,
                "high_friction_reset_state": diagnostic_reset,
            }
        )
        clean_env, _ = make_contact_friction_env(
            diagnostic_template,
            mode=mode,
            resolution=resolution,
        )
        try:
            for _ in range(STRICT_MINIMUM_CACHE_CLEAR_STEPS):
                clean_env.step(np.zeros(2, dtype=np.float32))
            restore_body_snapshot(clean_env, query_snapshot)
            # Rendering optionally includes the most recent action marker.
            # Match that observable bookkeeping value without advancing time.
            clean_env.latest_action = np.asarray(
                history[-1], dtype=np.float32
            ).copy()
            clean_query_pixels = np.asarray(
                clean_env.render(), dtype=np.uint8
            ).copy()
            clean_query_state = simulator_state_audit(clean_env)
            for action in query:
                _step_and_count_agent_block_contacts(clean_env, action)
            clean_future_snapshot = body_snapshot(clean_env)
            clean_future_pixels = np.asarray(
                clean_env.render(), dtype=np.uint8
            ).copy()
        finally:
            clean_env.close()
        clean_replay_gap = float(
            np.max(
                np.abs(
                    _snapshot_delta(
                        continuous_future_snapshot,
                        clean_future_snapshot,
                    )
                )
            )
        )
        clean_replay = {
            "diagnostic_only_not_used_to_generate_rows": True,
            "query_pixels_bitwise_identical": bool(
                np.array_equal(model_pixels[2], clean_query_pixels)
            ),
            "query_registered_state_gap": simulator_state_max_abs_gap(
                query_simulator_state,
                clean_query_state,
            ),
            "future_full_state_max_abs_gap": clean_replay_gap,
            "future_pixels_bitwise_identical": bool(
                np.array_equal(
                    continuous_future_pixels,
                    clean_future_pixels,
                )
            ),
            "tolerance": STRICT_CLEAN_REPLAY_FULL_STATE_TOLERANCE,
            "passed": bool(
                np.array_equal(model_pixels[2], clean_query_pixels)
                and clean_replay_gap
                <= STRICT_CLEAN_REPLAY_FULL_STATE_TOLERANCE
                and np.array_equal(
                    continuous_future_pixels,
                    clean_future_pixels,
                )
            ),
        }

    return {
        "template": asdict(template),
        "mode": mode,
        "friction": FRICTION_VALUES[mode],
        "raw_actions": raw_actions,
        "action_blocks": raw_actions.reshape(
            CLIP_ACTION_BLOCKS,
            ACTION_BLOCK,
            2,
        ),
        "rows": rows,
        "model_pixels": np.stack(model_pixels),
        "model_states": np.stack(model_states),
        "model_physics_states": np.stack(model_physics_states),
        "body_bounds": body_bounds,
        "goal_pixels": np.asarray(
            info["goal"],
            dtype=np.uint8,
        ).copy(),
        "contact_steps_by_block": contact_steps_by_block,
        "history_contact_steps": sum(contact_steps_by_block[:2]),
        "query_contact_steps": contact_steps_by_block[2],
        "query_precanonical_snapshot": query_snapshot,
        "query_precanonical_correction": query_target_residual,
        "query_natural_target_residual": query_target_residual,
        "query_simulator_state": query_simulator_state,
        "state_installations_after_x0": 0,
        "query_simulator_recreated": False,
        "trailing_no_contact_steps_before_query": (
            trailing_no_contact_steps
        ),
        "clean_simulator_replay": clean_replay,
        "query_boundary": STRICT_CONTINUOUS_CONSTRUCTION,
        "friction_assignment": assignment,
    }


def validate_contact_friction_pair(
    low: dict[str, Any],
    high: dict[str, Any],
    *,
    query_state_tolerance: float = STRICT_QUERY_FULL_STATE_TOLERANCE,
    initial_state_tolerance: float = STRICT_INITIAL_FULL_STATE_TOLERANCE,
    minimum_history_gap_px_equivalent: float = 3.0,
    minimum_future_block_gap_px: float = 2.0,
    minimum_future_block_angle_rad: float = 1.0 / 30.0,
    angular_radius_px: float = 60.0,
) -> dict[str, Any]:
    """Audit one stored low/high-friction causal pair."""

    low_pixels = np.asarray(low["model_pixels"])
    high_pixels = np.asarray(high["model_pixels"])
    low_physics = np.asarray(
        low["model_physics_states"],
        dtype=np.float64,
    )
    high_physics = np.asarray(
        high["model_physics_states"],
        dtype=np.float64,
    )
    history_gap = _visible_response_gap(
        low_physics[1],
        high_physics[1],
        angular_radius_px=angular_radius_px,
    )
    future_gap = _future_gap(low_physics[3], high_physics[3])
    query_pair_gap = simulator_state_max_abs_gap(
        low["query_simulator_state"],
        high["query_simulator_state"],
    )
    initial_pair_gap = float(
        np.max(
            np.abs(_snapshot_delta(low_physics[0], high_physics[0]))
        )
    )
    query_pixel_difference = np.abs(
        low_pixels[2].astype(np.int16) - high_pixels[2].astype(np.int16)
    )
    action_difference = np.abs(
        np.asarray(low["raw_actions"], dtype=np.float64)
        - np.asarray(high["raw_actions"], dtype=np.float64)
    )
    checks = {
        "template_identity": low["template"] == high["template"],
        "mode_identity": (
            low["mode"] == ENDPOINT_MODES[0]
            and high["mode"] == ENDPOINT_MODES[1]
        ),
        "friction_assignment_low": bool(
            low["friction_assignment"]["passed"]
        ),
        "friction_assignment_high": bool(
            high["friction_assignment"]["passed"]
        ),
        "initial_physics_difference_within_registered_tolerance": (
            initial_pair_gap <= initial_state_tolerance
        ),
        "initial_pixels_identical": np.array_equal(
            low_pixels[0],
            high_pixels[0],
        ),
        "actions_identical": np.array_equal(
            low["raw_actions"],
            high["raw_actions"],
        ),
        "history_contact_both_modes": (
            int(low["history_contact_steps"]) > 0
            and int(high["history_contact_steps"]) > 0
        ),
        "history_pixels_different": not np.array_equal(
            low_pixels[1],
            high_pixels[1],
        ),
        "history_visible_gap_sufficient": (
            history_gap["px_equivalent"]
            >= minimum_history_gap_px_equivalent
        ),
        "each_trajectory_natural_query_residual_within_tolerance": (
            float(low["query_precanonical_correction"])
            <= query_state_tolerance
            and float(high["query_precanonical_correction"])
            <= query_state_tolerance
        ),
        "no_state_installation_after_x0": (
            int(low["state_installations_after_x0"]) == 0
            and int(high["state_installations_after_x0"]) == 0
        ),
        "query_simulator_not_recreated": (
            not bool(low["query_simulator_recreated"])
            and not bool(high["query_simulator_recreated"])
        ),
        "query_full_simulator_state_within_tolerance": (
            query_pair_gap <= query_state_tolerance
        ),
        "query_pixels_identical": np.array_equal(
            low_pixels[2],
            high_pixels[2],
        ),
        "contact_cache_naturally_expired": (
            int(low["trailing_no_contact_steps_before_query"])
            >= STRICT_MINIMUM_CACHE_CLEAR_STEPS
            and int(high["trailing_no_contact_steps_before_query"])
            >= STRICT_MINIMUM_CACHE_CLEAR_STEPS
            and int(
                low["query_simulator_state"]["active_arbiter_count"]
            )
            == 0
            and int(
                high["query_simulator_state"]["active_arbiter_count"]
            )
            == 0
        ),
        "clean_simulator_replay_low": bool(
            low.get("clean_simulator_replay")
            and low["clean_simulator_replay"]["passed"]
        ),
        "clean_simulator_replay_high": bool(
            high.get("clean_simulator_replay")
            and high["clean_simulator_replay"]["passed"]
        ),
        "query_contact_both_modes": (
            int(low["query_contact_steps"]) > 0
            and int(high["query_contact_steps"]) > 0
        ),
        "future_pixels_different": not np.array_equal(
            low_pixels[3],
            high_pixels[3],
        ),
        "future_gap_sufficient": (
            future_gap["block_position_px"]
            >= minimum_future_block_gap_px
            or future_gap["block_angle_rad"]
            >= minimum_future_block_angle_rad
        ),
        "all_bodies_inside_playfield": all(
            _bounds_inside_playfield(bounds)
            for bounds in low["body_bounds"] + high["body_bounds"]
        ),
        "goal_pixels_identical": np.array_equal(
            low["goal_pixels"],
            high["goal_pixels"],
        ),
    }
    return {
        "template_id": low["template"]["template_id"],
        "passed": all(checks.values()),
        "checks": checks,
        "history_visible_response_gap": history_gap,
        "future_gap": future_gap,
        "query_physics_max_abs_gap": query_pair_gap,
        "initial_physics_max_abs_gap": initial_pair_gap,
        "initial_physics_tolerance": float(initial_state_tolerance),
        "query_full_state_tolerance": float(query_state_tolerance),
        "query_pixel_max_abs_difference": int(
            np.max(query_pixel_difference)
        ),
        "query_pixel_different_values": int(
            np.count_nonzero(query_pixel_difference)
        ),
        "query_action_max_abs_difference": float(
            np.max(action_difference)
        ),
        "state_installations_after_x0": int(
            max(
                low["state_installations_after_x0"],
                high["state_installations_after_x0"],
            )
        ),
        "query_simulator_recreated": bool(
            low["query_simulator_recreated"]
            or high["query_simulator_recreated"]
        ),
        "trailing_no_contact_steps_before_query": {
            ENDPOINT_MODES[0]: int(
                low["trailing_no_contact_steps_before_query"]
            ),
            ENDPOINT_MODES[1]: int(
                high["trailing_no_contact_steps_before_query"]
            ),
        },
        "clean_simulator_replay": {
            ENDPOINT_MODES[0]: low["clean_simulator_replay"],
            ENDPOINT_MODES[1]: high["clean_simulator_replay"],
        },
        "query_precanonical_correction": {
            ENDPOINT_MODES[0]: float(
                low["query_precanonical_correction"]
            ),
            ENDPOINT_MODES[1]: float(
                high["query_precanonical_correction"]
            ),
        },
        "contact_steps": {
            "low_history": int(low["history_contact_steps"]),
            "high_history": int(high["history_contact_steps"]),
            "low_query": int(low["query_contact_steps"]),
            "high_query": int(high["query_contact_steps"]),
        },
        "hashes": {
            "initial_pixels": array_sha256(low_pixels[0]),
            "low_history_pixels": array_sha256(low_pixels[1]),
            "high_history_pixels": array_sha256(high_pixels[1]),
            "query_pixels": array_sha256(low_pixels[2]),
            "low_future_pixels": array_sha256(low_pixels[3]),
            "high_future_pixels": array_sha256(high_pixels[3]),
            "raw_actions": array_sha256(low["raw_actions"]),
        },
    }


def _visible_response_gap(
    low_middle: np.ndarray,
    high_middle: np.ndarray,
    *,
    angular_radius_px: float,
) -> dict[str, float]:
    agent_position = float(
        np.linalg.norm(low_middle[0:2] - high_middle[0:2])
    )
    block_position = float(
        np.linalg.norm(low_middle[6:8] - high_middle[6:8])
    )
    block_angle = abs(_angle_delta(low_middle[10], high_middle[10]))
    equivalent = float(
        max(
            agent_position,
            block_position,
            float(angular_radius_px) * block_angle,
        )
    )
    return {
        "agent_position_px": agent_position,
        "block_position_px": block_position,
        "block_angle_rad": block_angle,
        "px_equivalent": equivalent,
    }


def _future_gap(
    low_future: np.ndarray,
    high_future: np.ndarray,
) -> dict[str, float]:
    return {
        "block_position_px": float(
            np.linalg.norm(low_future[6:8] - high_future[6:8])
        ),
        "block_angle_rad": abs(
            _angle_delta(low_future[10], high_future[10])
        ),
    }


def evaluate_contact_friction_candidate(
    template: ContactFrictionTemplate,
    *,
    resolution: int = 96,
    query_state_tolerance: float = 0.002,
    minimum_history_gap_px_equivalent: float = 3.0,
    minimum_future_block_gap_px: float = 2.0,
    minimum_future_block_angle_rad: float = 1.0 / 30.0,
    angular_radius_px: float = 60.0,
    require_primary_shape: bool = False,
    query_state_gate: str = "pair_gap",
) -> dict[str, Any]:
    """Evaluate every preregistered causal gate for one candidate."""

    if query_state_gate not in ("pair_gap", "per_endpoint_correction"):
        raise ValueError(
            "query_state_gate must be 'pair_gap' or "
            "'per_endpoint_correction'"
        )
    low = simulate_history(
        template,
        mode=ENDPOINT_MODES[0],
        resolution=resolution,
    )
    high = simulate_history(
        template,
        mode=ENDPOINT_MODES[1],
        resolution=resolution,
    )
    low_prequery = low["snapshots"][-1]
    high_prequery = high["snapshots"][-1]
    pair_delta = _snapshot_delta(low_prequery, high_prequery)
    pair_max_abs = float(np.max(np.abs(pair_delta)))
    calculated_midpoint = midpoint_snapshot(low_prequery, high_prequery)
    if template.canonical_query_snapshot is None:
        canonical = calculated_midpoint
        canonical_source = "candidate_midpoint_not_yet_frozen"
    else:
        canonical = _state_array(
            template.canonical_query_snapshot,
            expected_size=12,
            name="canonical_query_snapshot",
        )
        canonical_source = "frozen_template"
    low_to_canonical = float(
        np.max(np.abs(_snapshot_delta(low_prequery, canonical)))
    )
    high_to_canonical = float(
        np.max(np.abs(_snapshot_delta(high_prequery, canonical)))
    )

    low_future = simulate_query_future(
        template,
        mode=ENDPOINT_MODES[0],
        canonical_query_snapshot=canonical,
        resolution=resolution,
    )
    high_future = simulate_query_future(
        template,
        mode=ENDPOINT_MODES[1],
        canonical_query_snapshot=canonical,
        resolution=resolution,
    )
    history_gap = _visible_response_gap(
        low["snapshots"][1],
        high["snapshots"][1],
        angular_radius_px=angular_radius_px,
    )
    future_gap = _future_gap(
        low_future["future_snapshot"],
        high_future["future_snapshot"],
    )
    query_pixels_identical = np.array_equal(
        low_future["query_pixels"],
        high_future["query_pixels"],
    )
    checks = {
        "primary_shape_if_required": (
            not require_primary_shape
            or int(template.visible_shape_id) == PRIMARY_SHAPE_ID
        ),
        "friction_assignment_low": bool(
            low["friction_assignment"]["passed"]
        ),
        "friction_assignment_high": bool(
            high["friction_assignment"]["passed"]
        ),
        "initial_physics_identical": np.array_equal(
            low["snapshots"][0],
            high["snapshots"][0],
        ),
        "initial_pixels_identical": np.array_equal(
            low["pixels"][0],
            high["pixels"][0],
        ),
        "history_actions_identical": np.array_equal(
            low["actions"],
            high["actions"],
        ),
        "history_contact_both_modes": bool(
            np.any(low["contact_counts"] > 0)
            and np.any(high["contact_counts"] > 0)
        ),
        "history_frames_inside_playfield": bool(
            all(
                _bounds_inside_playfield(bounds)
                for bounds in low["body_bounds"] + high["body_bounds"]
            )
        ),
        "history_visible_gap_sufficient": (
            history_gap["px_equivalent"]
            >= float(minimum_history_gap_px_equivalent)
        ),
        "precanonical_pair_state_within_tolerance": (
            pair_max_abs <= float(query_state_tolerance)
        ),
        "each_endpoint_within_canonicalization_tolerance": (
            low_to_canonical <= float(query_state_tolerance)
            and high_to_canonical <= float(query_state_tolerance)
        ),
        "query_physics_identical_after_canonicalization": np.array_equal(
            low_future["query_snapshot"],
            high_future["query_snapshot"],
        ),
        "query_pixels_bitwise_identical": query_pixels_identical,
        "query_bodies_inside_playfield": bool(
            _bounds_inside_playfield(low_future["query_body_bounds"])
            and _bounds_inside_playfield(high_future["query_body_bounds"])
        ),
        "query_actions_identical": np.array_equal(
            low_future["actions"],
            high_future["actions"],
        ),
        "query_contact_both_modes": bool(
            np.any(low_future["contact_counts"] > 0)
            and np.any(high_future["contact_counts"] > 0)
        ),
        "future_gap_sufficient": bool(
            future_gap["block_position_px"]
            >= float(minimum_future_block_gap_px)
            or future_gap["block_angle_rad"]
            >= float(minimum_future_block_angle_rad)
        ),
        "future_bodies_inside_playfield": bool(
            _bounds_inside_playfield(low_future["future_body_bounds"])
            and _bounds_inside_playfield(high_future["future_body_bounds"])
        ),
    }
    hashes = {
        "history_actions": array_sha256(low["actions"]),
        "query_actions": array_sha256(low_future["actions"]),
        "initial_pixels": array_sha256(low["pixels"][0]),
        "low_middle_pixels": array_sha256(low["pixels"][1]),
        "high_middle_pixels": array_sha256(high["pixels"][1]),
        "query_pixels": array_sha256(low_future["query_pixels"]),
        "low_future_pixels": array_sha256(low_future["future_pixels"]),
        "high_future_pixels": array_sha256(high_future["future_pixels"]),
    }
    ignored_checks = (
        {"precanonical_pair_state_within_tolerance"}
        if query_state_gate == "per_endpoint_correction"
        else set()
    )
    passed = all(
        value for name, value in checks.items() if name not in ignored_checks
    )
    return {
        "template_id": template.template_id,
        "visible_shape": {
            "id": int(template.visible_shape_id),
            "name": template.visible_shape_name,
        },
        "passed": bool(passed),
        "query_state_gate": query_state_gate,
        "diagnostic_checks_excluded_from_pass": sorted(ignored_checks),
        "checks": checks,
        "history_visible_response_gap": history_gap,
        "precanonical_query_state": {
            "pair_max_abs_gap": pair_max_abs,
            "low_to_canonical_max_abs": low_to_canonical,
            "high_to_canonical_max_abs": high_to_canonical,
            "tolerance": float(query_state_tolerance),
            "canonical_source": canonical_source,
            "candidate_canonical_snapshot": canonical.tolist(),
        },
        "future_gap": future_gap,
        "contact_steps": {
            "low_history": int(np.count_nonzero(low["contact_counts"])),
            "high_history": int(np.count_nonzero(high["contact_counts"])),
            "low_query": int(
                np.count_nonzero(low_future["contact_counts"])
            ),
            "high_query": int(
                np.count_nonzero(high_future["contact_counts"])
            ),
        },
        "hashes": hashes,
    }


def model_input_projection(
    history: dict[str, Any],
    query_future: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return only the arrays a benchmark adapter may expose to a model."""

    pixels = np.concatenate(
        [
            np.asarray(history["pixels"][:2], dtype=np.uint8),
            np.asarray(query_future["query_pixels"], dtype=np.uint8)[None],
        ],
        axis=0,
    )
    history_actions = np.asarray(history["actions"], dtype=np.float32).reshape(
        HISTORY_TOKENS - 1,
        ACTION_BLOCK,
        2,
    )
    query_actions = np.asarray(
        query_future["actions"],
        dtype=np.float32,
    )[None]
    return {
        "pixels": pixels,
        "action": np.concatenate(
            [history_actions, query_actions],
            axis=0,
        ),
    }


def make_frozen_search_best_template() -> ContactFrictionTemplate:
    """Return the best valid H3 candidate found by the frozen search.

    The candidate is intentionally retained even though it does not pass the
    original pairwise-distance gate.  Keeping it executable makes that result
    auditable while confirmation uses the corrected per-trajectory
    canonicalization gate.
    """

    state = (
        351.736679,
        260.000524,
        339.275633,
        324.390370,
        2.08144977,
        0.0,
        0.0,
    )
    probe = np.asarray([-0.621161791, -0.174122985], dtype=np.float64)
    recovery_profile = np.asarray(
        [
            -0.19935114,
            -0.19935120,
            -0.19934264,
            -0.20064880,
            -0.00130621,
        ],
        dtype=np.float64,
    )
    recovery = recovery_profile[:, None] / 0.4 * probe[None]
    history = np.vstack(
        [
            probe,
            probe,
            np.zeros((3, 2), dtype=np.float64),
            recovery,
        ]
    )
    query = np.vstack(
        [
            np.asarray([[-0.25, 0.25]], dtype=np.float64),
            np.zeros((QUERY_RAW_STEPS - 1, 2), dtype=np.float64),
        ]
    )
    return ContactFrictionTemplate(
        template_id="t-h3-closed-search-best-v1",
        visible_shape_id=PRIMARY_SHAPE_ID,
        visible_shape_name=PRIMARY_SHAPE_NAME,
        reset_state=state,
        goal_state=(
            state[0],
            state[1],
            300.0,
            250.0,
            0.0,
            0.0,
            0.0,
        ),
        history_actions=tuple(map(tuple, history.tolist())),
        query_actions=tuple(map(tuple, query.tolist())),
        simulator_seed=42,
    )


def make_strict_continuous_base_template(
    family_id: int = 0,
) -> ContactFrictionTemplate:
    """Return the forward-only template used by formal v2 data.

    The two sub-rendering x0 poses were solved before freezing this template.
    With one shared action sequence, both real trajectories naturally reach
    the registered x2.  The final three history steps have no active contact,
    so Pymunk's previous contact arbiter has expired before the query.
    """

    nominal = make_frozen_search_best_template()
    probe = np.asarray(nominal.history_actions, dtype=np.float64)[:5]
    families = {
        0: {
            "forward": (0.7253626053392535, 0.20524798670707306),
            "low_reset": (
                339.3047771866207,
                324.38803287819155,
                2.080896533198646,
            ),
            "high_reset": (
                339.2502270557548,
                324.40077814622526,
                2.0821531926669286,
            ),
            "canonical": (
                401.35233499524935,
                270.4444378301252,
                61.4109763476946,
                -2.3290859398457813,
                0.0,
                0.0,
                334.3802758624473,
                339.93445047988865,
                0.0,
                0.0,
                2.2223624864271443,
                0.0,
            ),
        },
        1: {
            "forward": (0.7242747465258601, 0.20458891077406405),
            "low_reset": (
                339.3204718194851,
                324.34849271585654,
                2.0805962006699903,
            ),
            "high_reset": (
                339.2341953667733,
                324.44348263940867,
                2.082493266878482,
            ),
            "canonical": (
                401.2195281392268,
                270.3639797316786,
                61.410958986575,
                -2.329096457696557,
                0.0,
                0.0,
                334.3858208686206,
                339.9393541793566,
                0.0,
                0.0,
                2.2224999379415618,
                0.0,
            ),
        },
    }
    if int(family_id) not in families:
        raise ValueError(f"Unknown strict family {family_id!r}")
    family = families[int(family_id)]
    forward = np.asarray(family["forward"], dtype=np.float64)
    retreat = np.asarray(
        [0.1651870889063084, -0.0062749148786942],
        dtype=np.float64,
    )
    history = np.vstack(
        [probe, np.tile(forward, (3, 1)), np.tile(retreat, (2, 1))]
    )
    canonical = np.asarray(family["canonical"], dtype=np.float64)
    query_direction = canonical[6:8] - canonical[0:2]
    query_direction /= np.linalg.norm(query_direction)
    query = np.vstack(
        [
            np.tile(0.4 * query_direction, (2, 1)),
            np.zeros((3, 2), dtype=np.float64),
        ]
    )

    def endpoint_reset(block_pose: tuple[float, float, float]) -> tuple:
        return (
            float(nominal.reset_state[0]),
            float(nominal.reset_state[1]),
            *map(float, block_pose),
            0.0,
            0.0,
        )

    return ContactFrictionTemplate(
        template_id=f"pcf-strict-continuous-base-f{int(family_id)}-v2",
        visible_shape_id=PRIMARY_SHAPE_ID,
        visible_shape_name=PRIMARY_SHAPE_NAME,
        reset_state=nominal.reset_state,
        goal_state=nominal.goal_state,
        history_actions=tuple(map(tuple, history.tolist())),
        query_actions=tuple(map(tuple, query.tolist())),
        simulator_seed=42,
        canonical_query_snapshot=tuple(canonical.tolist()),
        low_friction_reset_state=endpoint_reset(family["low_reset"]),
        high_friction_reset_state=endpoint_reset(family["high_reset"]),
        causal_construction=STRICT_CONTINUOUS_CONSTRUCTION,
        strict_family_id=int(family_id),
    )


def rigid_transform_contact_friction_template(
    base: ContactFrictionTemplate,
    *,
    template_id: str,
    angle_rad: float,
    translation_xy: Iterable[float],
    simulator_seed: int,
) -> ContactFrictionTemplate:
    """Rigidly transform one audited template without changing its physics."""

    if base.canonical_query_snapshot is None:
        canonical = np.asarray(
            BASE_CANONICAL_QUERY_SNAPSHOT,
            dtype=np.float64,
        )
    else:
        canonical = _state_array(
            base.canonical_query_snapshot,
            expected_size=12,
            name="canonical_query_snapshot",
        )
    angle = float(angle_rad) % (2 * np.pi)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]],
        dtype=np.float64,
    )
    translation = np.asarray(
        tuple(translation_xy),
        dtype=np.float64,
    )
    if translation.shape != (2,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation_xy must contain two finite values")
    center = np.asarray(CATALOG_ROTATION_CENTER, dtype=np.float64)

    def rotate_point(point: np.ndarray) -> np.ndarray:
        return center + rotation @ (point - center) + translation

    reset = np.asarray(base.reset_state, dtype=np.float64)
    goal = np.asarray(base.goal_state, dtype=np.float64)

    def transform_reset(value: Iterable[float]) -> np.ndarray:
        source = _state_array(value, expected_size=7, name="reset_state")
        return np.concatenate(
            [
                rotate_point(source[:2]),
                rotate_point(source[2:4]),
                [
                    (source[4] + angle) % (2 * np.pi),
                    float(source[5]),
                    float(source[6]),
                ],
            ]
        )

    transformed_reset = transform_reset(reset)
    transformed_goal = np.concatenate(
        [
            rotate_point(goal[:2]),
            rotate_point(goal[2:4]),
            [(goal[4] + angle) % (2 * np.pi), 0.0, 0.0],
        ]
    )
    transformed_canonical = canonical.copy()
    transformed_canonical[0:2] = rotate_point(canonical[0:2])
    transformed_canonical[2:4] = rotation @ canonical[2:4]
    # The pusher is circular.  Its body angle is deliberately kept at zero,
    # matching the reset contract rather than leaking the catalog rotation.
    transformed_canonical[4] = 0.0
    transformed_canonical[6:8] = rotate_point(canonical[6:8])
    transformed_canonical[8:10] = rotation @ canonical[8:10]
    transformed_canonical[10] = (
        canonical[10] + angle
    ) % (2 * np.pi)
    history_actions = (
        np.asarray(base.history_actions, dtype=np.float64) @ rotation.T
    )
    query_actions = (
        np.asarray(base.query_actions, dtype=np.float64) @ rotation.T
    )
    return ContactFrictionTemplate(
        template_id=template_id,
        visible_shape_id=PRIMARY_SHAPE_ID,
        visible_shape_name=PRIMARY_SHAPE_NAME,
        reset_state=tuple(transformed_reset.tolist()),
        goal_state=tuple(transformed_goal.tolist()),
        history_actions=tuple(map(tuple, history_actions.tolist())),
        query_actions=tuple(map(tuple, query_actions.tolist())),
        simulator_seed=int(simulator_seed),
        canonical_query_snapshot=tuple(
            transformed_canonical.tolist()
        ),
        low_friction_reset_state=(
            tuple(
                transform_reset(base.low_friction_reset_state).tolist()
            )
            if base.low_friction_reset_state is not None
            else None
        ),
        high_friction_reset_state=(
            tuple(
                transform_reset(base.high_friction_reset_state).tolist()
            )
            if base.high_friction_reset_state is not None
            else None
        ),
        causal_construction=base.causal_construction,
        strict_family_id=base.strict_family_id,
    )


def make_contact_friction_catalog_template(
    *,
    split: str,
    catalog_index: int,
    catalog_seed: int,
) -> ContactFrictionTemplate:
    """Create one deterministic, split-specific formal data template."""

    if split not in CATALOG_SPLIT_TAGS:
        raise ValueError(
            f"Unknown split {split!r}; expected {tuple(CATALOG_SPLIT_TAGS)}"
        )
    if int(catalog_index) < 0:
        raise ValueError("catalog_index must be non-negative")
    generator = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(catalog_seed),
                CATALOG_SPLIT_TAGS[split],
                int(catalog_index),
            ]
        )
    )
    family_id = int(generator.integers(0, 2))
    angle = float(generator.uniform(0.0, 2 * np.pi))
    # This central support was audited under arbitrary rotations.  It keeps
    # the complete T shape, pusher, history, query, and future away from walls.
    translation = generator.uniform(
        np.asarray([-60.0, -60.0]),
        np.asarray([40.0, 60.0]),
    )
    simulator_seed = int(generator.integers(0, 2**31 - 1))
    return rigid_transform_contact_friction_template(
        make_strict_continuous_base_template(family_id),
        template_id=(
            f"pcf-{split.replace('_', '-')}-{int(catalog_index):06d}"
        ),
        angle_rad=angle,
        translation_xy=translation,
        simulator_seed=simulator_seed,
    )


def stratified_contact_friction_training_coordinates(
    pair_index: int,
) -> dict[str, int]:
    """Map one Training pair to a balanced dynamics/geometry stratum.

    A complete cycle contains both strict physics families, sixteen angle
    bins, and an 8x8 translation grid (2,048 strata).  Larger releases repeat
    this complete grid with independently jittered templates.
    """

    pair_index = int(pair_index)
    if pair_index < 0:
        raise ValueError("pair_index must be non-negative")
    family_count = len((0, 1))
    angle_bins = STRATIFIED_TRAINING_ANGLE_BINS
    x_bins, y_bins = STRATIFIED_TRAINING_TRANSLATION_BINS
    cycle_size = family_count * angle_bins * x_bins * y_bins
    within_cycle = pair_index % cycle_size
    family_id = within_cycle % family_count
    angle_bin = (within_cycle // family_count) % angle_bins
    position_index = within_cycle // (family_count * angle_bins)
    translation_x_bin = position_index % x_bins
    translation_y_bin = position_index // x_bins
    return {
        "cycle_index": pair_index // cycle_size,
        "family_id": family_id,
        "angle_bin": angle_bin,
        "translation_x_bin": translation_x_bin,
        "translation_y_bin": translation_y_bin,
    }


def make_stratified_contact_friction_training_template(
    *,
    pair_index: int,
    attempt_index: int,
    catalog_seed: int,
) -> ContactFrictionTemplate:
    """Create a deterministic Training template inside one fixed stratum."""

    coordinates = stratified_contact_friction_training_coordinates(
        pair_index
    )
    attempt_index = int(attempt_index)
    if attempt_index < 0:
        raise ValueError("attempt_index must be non-negative")
    generator = np.random.default_rng(
        np.random.SeedSequence(
            [
                int(catalog_seed),
                CATALOG_SPLIT_TAGS["train"],
                int(pair_index),
                attempt_index,
                0xCF3,
            ]
        )
    )
    angle_width = 2 * np.pi / STRATIFIED_TRAINING_ANGLE_BINS
    angle = angle_width * (
        coordinates["angle_bin"] + generator.uniform(0.1, 0.9)
    )
    x_bins, y_bins = STRATIFIED_TRAINING_TRANSLATION_BINS
    x_edges = np.linspace(-60.0, 40.0, x_bins + 1)
    y_edges = np.linspace(-60.0, 60.0, y_bins + 1)

    def jittered_coordinate(edges: np.ndarray, index: int) -> float:
        low, high = float(edges[index]), float(edges[index + 1])
        return float(low + generator.uniform(0.1, 0.9) * (high - low))

    translation = (
        jittered_coordinate(
            x_edges,
            coordinates["translation_x_bin"],
        ),
        jittered_coordinate(
            y_edges,
            coordinates["translation_y_bin"],
        ),
    )
    simulator_seed = int(generator.integers(0, 2**31 - 1))
    return rigid_transform_contact_friction_template(
        make_strict_continuous_base_template(
            coordinates["family_id"]
        ),
        template_id=(
            f"pcf-train-strat-p{int(pair_index):06d}-"
            f"a{attempt_index:02d}"
        ),
        angle_rad=float(angle),
        translation_xy=translation,
        simulator_seed=simulator_seed,
    )


def make_frozen_confirmation_templates() -> list[ContactFrictionTemplate]:
    """Return eight H3 confirmations at four orientations and two positions.

    The common query state is frozen before these rollouts are evaluated.  It
    is the rigid transform of the searched candidate's canonical midpoint, so
    the confirmation audit measures the actual correction applied to each
    trajectory rather than deriving a new target from each confirmation pair.
    """

    base = make_frozen_search_best_template()
    translations = (
        np.asarray([-35.0, 0.0], dtype=np.float64),
        np.asarray([35.0, 0.0], dtype=np.float64),
    )

    templates: list[ContactFrictionTemplate] = []
    for rotation_index, angle in enumerate(
        (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
    ):
        for translation_index, translation in enumerate(translations):
            templates.append(
                rigid_transform_contact_friction_template(
                    base,
                    template_id=(
                        f"t-h3-confirm-r{rotation_index}-"
                        f"p{translation_index}"
                    ),
                    angle_rad=angle,
                    translation_xy=translation,
                    simulator_seed=42,
                )
            )
    return templates


__all__ = [
    "ACTION_BLOCK",
    "BASE_CANONICAL_QUERY_SNAPSHOT",
    "CATALOG_SPLIT_TAGS",
    "CLIP_ACTION_BLOCKS",
    "CLIP_RAW_STEPS",
    "DIAGNOSTIC_SHAPE_ID",
    "DIAGNOSTIC_SHAPE_NAME",
    "ENDPOINT_MODES",
    "FRICTION_VALUES",
    "HISTORY_RAW_STEPS",
    "HISTORY_TOKENS",
    "MODEL_FRAME_ROWS",
    "MODEL_VISIBLE_FIELDS",
    "PRIMARY_SHAPE_ID",
    "PRIMARY_SHAPE_NAME",
    "QUERY_RAW_STEPS",
    "STRICT_CLEAN_REPLAY_FULL_STATE_TOLERANCE",
    "STRICT_CONTINUOUS_CONSTRUCTION",
    "STRICT_INITIAL_FULL_STATE_TOLERANCE",
    "STRICT_MINIMUM_CACHE_CLEAR_STEPS",
    "STRICT_QUERY_FULL_STATE_TOLERANCE",
    "ContactFrictionTemplate",
    "array_sha256",
    "body_snapshot",
    "evaluate_contact_friction_candidate",
    "friction_assignment_audit",
    "make_contact_friction_catalog_template",
    "make_contact_friction_env",
    "make_frozen_confirmation_templates",
    "make_frozen_search_best_template",
    "make_strict_continuous_base_template",
    "midpoint_snapshot",
    "model_input_projection",
    "rigid_transform_contact_friction_template",
    "restore_body_snapshot",
    "reset_state_for_mode",
    "register_agent_block_contact_counter",
    "set_effective_contact_friction",
    "simulate_history",
    "simulate_contact_friction_clip",
    "simulate_query_future",
    "simulator_state_audit",
    "simulator_state_max_abs_gap",
    "validate_contact_friction_pair",
]
