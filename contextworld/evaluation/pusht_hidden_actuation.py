"""Paired History-3 trajectories with a visually hidden Push-T actuator gain.

The benchmark construction in this module is intentionally narrower than a
general Push-T dataset generator.  It creates two deterministic trajectories
that:

* start from the same rendered and physical state;
* execute the same raw actions;
* visibly diverge during a probe because the actuator gain is hidden;
* return to the same query state before the prediction action; and
* diverge again when the same action makes contact with the block.

The recovery profile is a finite-horizon nulling control for PushT's existing
PD-controlled kinematic agent.  It was solved once from the public simulator's
linear free-space response.  Every generated trajectory independently checks
that the recovery reaches the common query within a strict numerical
tolerance.  No simulator state is loaded or edited after reset.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from stable_worldmodel.envs.pusht.env import PushT


ACTION_BLOCK = 5
MODEL_STEPS = 4
RAW_STEPS = ACTION_BLOCK * MODEL_STEPS
MODEL_FRAME_ROWS = (0, 5, 10, 15)
MODE_SCALES = {
    'low_gain': 60.0,
    'high_gain': 140.0,
}
PHYSICS_STATE_COMPONENTS = (
    'agent.position.x',
    'agent.position.y',
    'agent.velocity.x',
    'agent.velocity.y',
    'agent.angle',
    'agent.angular_velocity',
    'block.position.x',
    'block.position.y',
    'block.velocity.x',
    'block.velocity.y',
    'block.angle',
    'block.angular_velocity',
)

# The probe is deliberately sparse, matching the two-active-step structure of
# the Door benchmark.  The recovery profile makes both final displacement and
# velocity zero in the simulator's free-space controller.
PROBE_PROFILE = np.asarray([0.4, 0.4, 0.0, 0.0, 0.0], dtype=np.float32)
RECOVERY_PROFILE = np.asarray(
    [
        -0.19935114,
        -0.19935120,
        -0.19934264,
        -0.20064880,
        -0.00130621,
    ],
    dtype=np.float32,
)
QUERY_PROFILE = PROBE_PROFILE.copy()
QUERY_SHAPE = np.asarray([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def array_sha256(value: np.ndarray) -> str:
    """Hash an array together with its dtype and shape."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f'{array.dtype.str}:{array.shape}'.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _unit_axis(value: Iterable[float]) -> np.ndarray:
    axis = np.asarray(tuple(value), dtype=np.float64)
    if axis.shape != (2,):
        raise ValueError(f'Expected a 2-D axis, got {axis.shape}')
    norm = float(np.linalg.norm(axis))
    if not np.isclose(norm, 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(f'Axis must have unit norm, got {axis.tolist()}')
    if not np.allclose(axis, np.round(axis), atol=0.0, rtol=0.0):
        raise ValueError('Only cardinal contact axes are supported')
    return axis.astype(np.float32)


@dataclass(frozen=True)
class HiddenActuationTemplate:
    """One paired Push-T query specification."""

    template_id: str
    agent_position: tuple[float, float]
    block_position: tuple[float, float]
    block_angle: float
    contact_direction: tuple[float, float]
    probe_sign: int
    goal_agent_position: tuple[float, float]
    goal_block_position: tuple[float, float]
    goal_block_angle: float
    simulator_seed: int
    query_amplitude: float = 0.4

    def __post_init__(self) -> None:
        _unit_axis(self.contact_direction)
        if int(self.probe_sign) not in (-1, 1):
            raise ValueError('probe_sign must be -1 or 1')
        if not 0.0 <= float(self.query_amplitude) <= 1.0:
            raise ValueError('query_amplitude must lie in [0, 1]')

    @property
    def reset_state(self) -> np.ndarray:
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


def action_blocks(template: HiddenActuationTemplate) -> np.ndarray:
    """Return probe, recovery, query, and filler action blocks."""

    contact = _unit_axis(template.contact_direction)
    probe = (
        np.asarray([-contact[1], contact[0]], dtype=np.float32)
        * np.float32(template.probe_sign)
    )
    blocks = np.zeros((MODEL_STEPS, ACTION_BLOCK, 2), dtype=np.float32)
    blocks[0] = PROBE_PROFILE[:, None] * probe[None, :]
    blocks[1] = RECOVERY_PROFILE[:, None] * probe[None, :]
    blocks[2] = (
        np.float32(template.query_amplitude)
        * QUERY_SHAPE[:, None]
        * contact[None, :]
    )
    return blocks


def _body_snapshot(env: PushT) -> np.ndarray:
    """Capture every dynamic quantity that can affect the next transition."""

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


def _angle_gap(left: float, right: float) -> float:
    delta = abs(float(left) - float(right)) % (2 * np.pi)
    return float(min(delta, 2 * np.pi - delta))


def _future_state_gap(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return {
        'agent_position_px': float(np.linalg.norm(left[:2] - right[:2])),
        'block_position_px': float(np.linalg.norm(left[2:4] - right[2:4])),
        'block_angle_rad': _angle_gap(left[4], right[4]),
        'combined_visible_state': float(
            np.linalg.norm(
                np.concatenate(
                    [
                        left[:4] - right[:4],
                        [_angle_gap(left[4], right[4])],
                    ]
                )
            )
        ),
    }


def _variation_values(template: HiddenActuationTemplate) -> dict[str, Any]:
    return {
        'agent.start_position': np.asarray(
            template.agent_position,
            dtype=np.float64,
        ),
        'agent.velocity': np.zeros(2, dtype=np.float64),
        'block.start_position': np.asarray(
            template.block_position,
            dtype=np.float64,
        ),
        'block.angle': float(template.block_angle),
        'block.shape': 2,
        'goal.position': np.asarray(
            template.goal_block_position,
            dtype=np.float64,
        ),
        'goal.angle': float(template.goal_block_angle),
    }


def simulate_hidden_actuation(
    template: HiddenActuationTemplate,
    *,
    mode: str,
    resolution: int = 224,
    query_state_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Simulate one hidden-gain member of a paired History-3 query."""

    if mode not in MODE_SCALES:
        raise ValueError(f'Unknown hidden actuation mode {mode!r}')
    if query_state_tolerance <= 0:
        raise ValueError('query_state_tolerance must be positive')

    scale = float(MODE_SCALES[mode])
    blocks = action_blocks(template)
    raw_actions = blocks.reshape(RAW_STEPS, 2)
    env = PushT(
        resolution=int(resolution),
        with_target=True,
        render_mode='rgb_array',
    )
    env.action_scale = scale

    rows: dict[str, list[Any]] = {
        'pixels': [],
        'action': [],
        'proprio': [],
        'state': [],
        'goal_state': [],
        'physics_state': [],
        'n_contacts': [],
        'hidden_action_scale': [],
        'pair_id': [],
        'hidden_mode': [],
    }
    query_recovery_residual = None
    query_natural_snapshot = None
    query_contact_steps = 0
    try:
        observation, info = env.reset(
            seed=int(template.simulator_seed),
            options={
                'variation': (),
                'variation_values': _variation_values(template),
                'state': template.reset_state,
                'goal_state': template.goal_state,
            },
        )
        query_reference_snapshot = _body_snapshot(env)
        initial_pixels = np.asarray(env.render(), dtype=np.uint8).copy()
        initial_state = np.asarray(observation['state'], dtype=np.float64)
        goal_pixels = np.asarray(info['goal'], dtype=np.uint8).copy()

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
                        'Recovery did not return to the common query state: '
                        f'template={template.template_id}, mode={mode}, '
                        f'residual={query_recovery_residual:.8f}, '
                        f'tolerance={query_state_tolerance:.8f}'
                    )

            state = np.asarray(env._get_obs(), dtype=np.float64)
            rows['pixels'].append(
                np.asarray(env.render(), dtype=np.uint8).copy()
            )
            rows['action'].append(np.asarray(action, dtype=np.float32).copy())
            rows['proprio'].append(
                np.concatenate([state[:2], state[-2:]]).astype(np.float32)
            )
            rows['state'].append(state.astype(np.float32))
            rows['goal_state'].append(
                np.asarray(template.goal_state, dtype=np.float32)
            )
            rows['physics_state'].append(
                _body_snapshot(env).astype(np.float32)
            )
            rows['n_contacts'].append(
                np.asarray([0.0], dtype=np.float32)
            )
            rows['hidden_action_scale'].append(
                np.asarray([scale], dtype=np.float32)
            )
            rows['pair_id'].append(template.template_id)
            rows['hidden_mode'].append(mode)

            _, _, _, _, step_info = env.step(action)
            contacts = int(step_info['n_contacts'])
            rows['n_contacts'][-1][0] = float(contacts)
            if 2 * ACTION_BLOCK <= raw_step < 3 * ACTION_BLOCK:
                query_contact_steps += int(contacts > 0)

        model_pixels = np.stack(rows['pixels'])[list(MODEL_FRAME_ROWS)]
        model_states = np.stack(rows['state'])[list(MODEL_FRAME_ROWS)]
        model_physics_states = np.stack(rows['physics_state'])[
            list(MODEL_FRAME_ROWS)
        ]
    finally:
        env.close()

    return {
        'template': asdict(template),
        'mode': mode,
        'hidden_action_scale': scale,
        'raw_actions': raw_actions,
        'action_blocks': blocks,
        'rows': rows,
        'model_pixels': model_pixels,
        'model_states': model_states,
        'model_physics_states': model_physics_states,
        'initial_pixels': initial_pixels,
        'initial_state': initial_state.astype(np.float32),
        'goal_pixels': goal_pixels,
        'query_reference_snapshot': query_reference_snapshot,
        'query_natural_snapshot': query_natural_snapshot,
        'query_recovery_residual': query_recovery_residual,
        'query_state_tolerance': float(query_state_tolerance),
        'state_installations_after_x0': 0,
        'query_simulator_recreated': False,
        'query_contact_steps': query_contact_steps,
    }


def validate_hidden_actuation_pair(
    low: dict[str, Any],
    high: dict[str, Any],
    *,
    minimum_middle_agent_gap_px: float = 10.0,
    minimum_future_block_gap_px: float = 2.0,
) -> dict[str, Any]:
    """Audit whether two rollouts form a valid condition-matched pair."""

    low_pixels = np.asarray(low['model_pixels'])
    high_pixels = np.asarray(high['model_pixels'])
    low_states = np.asarray(low['model_states'])
    high_states = np.asarray(high['model_states'])
    low_physics = np.asarray(low['model_physics_states'])
    high_physics = np.asarray(high['model_physics_states'])
    future_gap = _future_state_gap(low_states[-1], high_states[-1])
    middle_agent_gap = float(
        np.linalg.norm(low_states[1, :2] - high_states[1, :2])
    )
    query_tolerance = min(
        float(low['query_state_tolerance']),
        float(high['query_state_tolerance']),
    )
    query_physics_gap = float(
        np.max(
            np.abs(
                np.asarray(low['query_natural_snapshot'], dtype=np.float64)
                - np.asarray(
                    high['query_natural_snapshot'],
                    dtype=np.float64,
                )
            )
        )
    )
    query_pixel_difference = int(
        np.count_nonzero(low_pixels[2] != high_pixels[2])
    )
    query_action_difference = float(
        np.max(
            np.abs(
                np.asarray(low['action_blocks'][2], dtype=np.float64)
                - np.asarray(
                    high['action_blocks'][2],
                    dtype=np.float64,
                )
            )
        )
    )
    checks = {
        'template_identity': low['template'] == high['template'],
        'mode_identity': (
            low['mode'] == 'low_gain' and high['mode'] == 'high_gain'
        ),
        'initial_pixels_identical': np.array_equal(
            low_pixels[0],
            high_pixels[0],
        ),
        'initial_state_identical': np.array_equal(
            low_physics[0],
            high_physics[0],
        ),
        'actions_identical': np.array_equal(
            low['raw_actions'],
            high['raw_actions'],
        ),
        'middle_pixels_different': not np.array_equal(
            low_pixels[1],
            high_pixels[1],
        ),
        'middle_agent_gap_sufficient': (
            middle_agent_gap >= minimum_middle_agent_gap_px
        ),
        'query_pixels_identical': np.array_equal(
            low_pixels[2],
            high_pixels[2],
        ),
        'query_matches_initial_low': np.array_equal(
            low_pixels[0],
            low_pixels[2],
        ),
        'query_matches_initial_high': np.array_equal(
            high_pixels[0],
            high_pixels[2],
        ),
        'query_physics_within_numerical_tolerance': (
            query_physics_gap <= query_tolerance
        ),
        'low_recovery_natural': (
            float(low['query_recovery_residual']) <= query_tolerance
        ),
        'high_recovery_natural': (
            float(high['query_recovery_residual']) <= query_tolerance
        ),
        'no_state_installations_after_x0': (
            int(low['state_installations_after_x0']) == 0
            and int(high['state_installations_after_x0']) == 0
            and not bool(low['query_simulator_recreated'])
            and not bool(high['query_simulator_recreated'])
        ),
        'future_pixels_different': not np.array_equal(
            low_pixels[3],
            high_pixels[3],
        ),
        'future_block_gap_sufficient': (
            future_gap['block_position_px']
            >= minimum_future_block_gap_px
        ),
        'both_modes_make_contact': (
            int(low['query_contact_steps']) > 0
            and int(high['query_contact_steps']) > 0
        ),
    }
    return {
        'passed': all(checks.values()),
        'checks': checks,
        'template_id': low['template']['template_id'],
        'middle_agent_gap_px': middle_agent_gap,
        'state_installations_after_x0': 0,
        'query_simulator_recreated': False,
        'full_state_dimensions': len(PHYSICS_STATE_COMPONENTS),
        'full_state_components': list(PHYSICS_STATE_COMPONENTS),
        'query_physics_max_abs_gap': query_physics_gap,
        'query_physics_tolerance': query_tolerance,
        'pair_query_pixel_difference': query_pixel_difference,
        'pair_query_action_difference': query_action_difference,
        'history_effect': middle_agent_gap,
        'true_future_effect': future_gap['block_position_px'],
        'query_recovery_residual': {
            'low_gain': float(low['query_recovery_residual']),
            'high_gain': float(high['query_recovery_residual']),
        },
        'future_gap': future_gap,
        'query_contact_steps': {
            'low_gain': int(low['query_contact_steps']),
            'high_gain': int(high['query_contact_steps']),
        },
        'hashes': {
            'initial_pixels': array_sha256(low_pixels[0]),
            'low_middle_pixels': array_sha256(low_pixels[1]),
            'high_middle_pixels': array_sha256(high_pixels[1]),
            'query_pixels': array_sha256(low_pixels[2]),
            'low_future_pixels': array_sha256(low_pixels[3]),
            'high_future_pixels': array_sha256(high_pixels[3]),
            'raw_actions': array_sha256(low['raw_actions']),
        },
    }


def model_input_projection(rollout: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return the arrays allowed to reach LeWM/PLDM."""

    return {
        'pixels': np.asarray(rollout['model_pixels'], dtype=np.uint8),
        'action': np.asarray(rollout['action_blocks'], dtype=np.float32),
    }


__all__ = [
    'ACTION_BLOCK',
    'MODEL_FRAME_ROWS',
    'MODEL_STEPS',
    'MODE_SCALES',
    'PHYSICS_STATE_COMPONENTS',
    'RAW_STEPS',
    'HiddenActuationTemplate',
    'action_blocks',
    'array_sha256',
    'model_input_projection',
    'simulate_hidden_actuation',
    'validate_hidden_actuation_pair',
]
