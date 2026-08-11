from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .portal_exit_env import PORTAL_EXIT_FACTOR, PORTAL_EXIT_MODES, make_portal_exit_env


ACTION_BLOCK = 5
EXIT_MODES = ("near_border", "farther_from_border")
DIRECTIONS = ("forward", "reverse")
WALL_AXES = ("vertical", "horizontal")
BORDER_SIDES = ("low", "high")


@dataclass(frozen=True)
class PortalExitTemplate:
    template_id: str
    wall_axis: str
    border_side: str
    direction: str
    door_position: int
    wall_thickness: int
    agent_radius: float
    agent_color: tuple[int, int, int]
    wall_color: tuple[int, int, int]
    background_color: tuple[int, int, int]
    simulator_seed: int


def _sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def make_template(*, split: str, index: int, catalog_seed: int) -> PortalExitTemplate:
    rng = np.random.default_rng(
        np.random.SeedSequence([int(catalog_seed), int(index)])
    )
    wall_axis = WALL_AXES[index % len(WALL_AXES)]
    border_side = BORDER_SIDES[(index // 2) % len(BORDER_SIDES)]
    direction = DIRECTIONS[(index // 4) % len(DIRECTIONS)]
    door_position = int(
        rng.integers(24, 36) if border_side == "low" else rng.integers(188, 200)
    )
    radius = float((7.0, 8.0, 9.0)[(index // 8) % 3])
    agent = tuple(int(x) for x in rng.integers(30, 231, size=3))
    wall = tuple(int(x) for x in rng.integers(0, 61, size=3))
    background = tuple(int(x) for x in rng.integers(220, 256, size=3))
    return PortalExitTemplate(
        template_id=f"portal-{split}-{index:05d}",
        wall_axis=wall_axis,
        border_side=border_side,
        direction=direction,
        door_position=door_position,
        wall_thickness=int((8, 10, 12)[(index // 96) % 3]),
        agent_radius=radius,
        agent_color=agent,
        wall_color=wall,
        background_color=background,
        simulator_seed=int(rng.integers(0, 2**31 - 1)),
    )


def _geometry(template: PortalExitTemplate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_primary = 95.0 if template.direction == "forward" else 129.0
    boundary = 14.0 + template.agent_radius
    secondary = boundary if template.border_side == "low" else 224.0 - boundary
    if template.wall_axis == "vertical":
        query = np.asarray([source_primary, secondary], dtype=np.float32)
        goal = np.asarray([190.0, 112.0], dtype=np.float32)
        query_direction = np.asarray(
            [1.0 if template.direction == "forward" else -1.0, 0.0],
            dtype=np.float32,
        )
    else:
        query = np.asarray([secondary, source_primary], dtype=np.float32)
        goal = np.asarray([112.0, 190.0], dtype=np.float32)
        query_direction = np.asarray(
            [0.0, 1.0 if template.direction == "forward" else -1.0],
            dtype=np.float32,
        )
    return query, goal, query_direction


def _query_block(direction: np.ndarray) -> np.ndarray:
    block = np.zeros((ACTION_BLOCK, 2), dtype=np.float32)
    block[:3] = direction
    return block


def _recovery_block(template: PortalExitTemplate) -> np.ndarray:
    primary = -1.0 if template.direction == "forward" else 1.0
    secondary = -1.0 if template.border_side == "low" else 1.0
    action = (
        np.asarray([primary, secondary], dtype=np.float32)
        if template.wall_axis == "vertical"
        else np.asarray([secondary, primary], dtype=np.float32)
    )
    return np.repeat(action[None], ACTION_BLOCK, axis=0)


def _variations(template: PortalExitTemplate, mode: str) -> dict[str, Any]:
    return {
        "agent.speed": np.asarray([7.0], dtype=np.float32),
        "agent.position": _geometry(template)[0],
        "agent.radius": np.asarray([template.agent_radius], dtype=np.float32),
        "agent.color": np.asarray(template.agent_color, dtype=np.uint8),
        "wall.axis": 1 if template.wall_axis == "vertical" else 0,
        "wall.thickness": template.wall_thickness,
        "wall.color": np.asarray(template.wall_color, dtype=np.uint8),
        "background.color": np.asarray(template.background_color, dtype=np.uint8),
        "door.number": 1,
        "door.position": np.asarray([template.door_position] * 3, dtype=np.int64),
        "door.size": np.asarray([18, 18, 18], dtype=np.int64),
        PORTAL_EXIT_FACTOR: PORTAL_EXIT_MODES[mode],
    }


def _step_block(env: Any, block: np.ndarray) -> np.ndarray:
    states = []
    for action in block:
        env.step(action)
        states.append(env.agent_position.detach().cpu().numpy().copy())
    return np.stack(states).astype(np.float32)


def simulate_portal_exit_clip(
    template: PortalExitTemplate,
    *,
    mode: str,
) -> dict[str, Any]:
    if mode not in EXIT_MODES:
        raise ValueError(f"Unknown portal exit mode {mode!r}")
    query, goal, direction = _geometry(template)
    probe = _query_block(direction)
    recovery = _recovery_block(template)
    env = make_portal_exit_env(render_mode="rgb_array")
    try:
        env.reset(
            seed=template.simulator_seed,
            options={
                "variation": (),
                "variation_values": _variations(template, mode),
                "state": query,
                "target_state": goal,
            },
        )
        history_pixels = [np.asarray(env.render(), dtype=np.uint8).copy()]
        history_states = [env.agent_position.detach().cpu().numpy().copy()]
        first_raw = _step_block(env, probe)
        history_pixels.append(np.asarray(env.render(), dtype=np.uint8).copy())
        history_states.append(env.agent_position.detach().cpu().numpy().copy())
        second_raw = _step_block(env, recovery)
        history_pixels.append(np.asarray(env.render(), dtype=np.uint8).copy())
        history_states.append(env.agent_position.detach().cpu().numpy().copy())
        query_pixels = history_pixels[-1].copy()
        query_state = history_states[-1].copy()
        future_raw = _step_block(env, probe)
        future_pixels = np.asarray(env.render(), dtype=np.uint8).copy()
        future_state = env.agent_position.detach().cpu().numpy().copy()
    finally:
        env.close()
    return {
        "template": asdict(template),
        "mode": mode,
        "history_pixels": np.stack(history_pixels),
        "history_states": np.stack(history_states).astype(np.float32),
        "history_actions": np.stack([probe, recovery]).astype(np.float32),
        "history_raw_states": np.concatenate([first_raw, second_raw]),
        "query_pixels": query_pixels,
        "query_state": query_state.astype(np.float32),
        "query_action": probe.astype(np.float32),
        "future_pixels": future_pixels,
        "future_state": future_state.astype(np.float32),
        "future_raw_states": future_raw,
        "goal_state": goal,
    }


def simulate_portal_exit_episode(
    template: PortalExitTemplate,
    *,
    mode: str,
) -> dict[str, Any]:
    """Return the exact 20 raw rows consumed by a frameskip-5 H3 loader."""

    if mode not in EXIT_MODES:
        raise ValueError(f"Unknown portal exit mode {mode!r}")
    query, goal, direction = _geometry(template)
    actions = np.concatenate(
        [
            _query_block(direction),
            _recovery_block(template),
            _query_block(direction),
            np.zeros((ACTION_BLOCK, 2), dtype=np.float32),
        ]
    ).astype(np.float32)
    rows: dict[str, list[Any]] = {
        "pixels": [],
        "action": [],
        "proprio": [],
        "state": [],
        "goal_state": [],
        "hidden_portal_exit": [],
        "pair_id": [],
        "hidden_mode": [],
    }
    env = make_portal_exit_env(render_mode="rgb_array")
    try:
        env.reset(
            seed=template.simulator_seed,
            options={
                "variation": (),
                "variation_values": _variations(template, mode),
                "state": query,
                "target_state": goal,
            },
        )
        for action in actions:
            state = np.asarray(env._get_obs(), dtype=np.float32)
            rows["pixels"].append(np.asarray(env.render(), dtype=np.uint8).copy())
            rows["action"].append(action.copy())
            rows["proprio"].append(state[:2].copy())
            rows["state"].append(state.copy())
            rows["goal_state"].append(goal.copy())
            rows["hidden_portal_exit"].append(
                np.asarray([PORTAL_EXIT_MODES[mode]], dtype=np.float32)
            )
            rows["pair_id"].append(template.template_id)
            rows["hidden_mode"].append(mode)
            env.step(action)
    finally:
        env.close()
    model_rows = np.asarray([0, 5, 10, 15], dtype=np.int64)
    model_pixels = np.stack(rows["pixels"])[model_rows]
    model_states = np.stack(rows["proprio"])[model_rows]
    return {
        "template": asdict(template),
        "mode": mode,
        "rows": rows,
        "raw_actions": actions,
        "model_pixels": model_pixels,
        "model_states": model_states,
    }


def validate_portal_exit_episode_pair(
    near: dict[str, Any], farther: dict[str, Any]
) -> dict[str, Any]:
    left_pixels = np.asarray(near["model_pixels"])
    right_pixels = np.asarray(farther["model_pixels"])
    left_states = np.asarray(near["model_states"], dtype=np.float32)
    right_states = np.asarray(farther["model_states"], dtype=np.float32)
    middle_gap = float(np.linalg.norm(left_states[1] - right_states[1]))
    future_gap = float(np.linalg.norm(left_states[3] - right_states[3]))
    query_gap = float(np.max(np.abs(left_states[2] - right_states[2])))
    checks = {
        "template_identity": near["template"] == farther["template"],
        "initial_pixels_identical": np.array_equal(left_pixels[0], right_pixels[0]),
        "history_actions_identical": np.array_equal(
            near["raw_actions"][:10], farther["raw_actions"][:10]
        ),
        "history_exit_gap_sufficient": middle_gap >= 12.0,
        "query_state_identical": query_gap <= 1e-5,
        "query_pixels_identical": np.array_equal(left_pixels[2], right_pixels[2]),
        "query_actions_identical": np.array_equal(
            near["raw_actions"][10:15], farther["raw_actions"][10:15]
        ),
        "future_pixels_different": not np.array_equal(
            left_pixels[3], right_pixels[3]
        ),
        "future_exit_gap_sufficient": future_gap >= 12.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "middle_state_gap_px": middle_gap,
        "future_state_gap_px": future_gap,
        "maximum_query_state_gap": query_gap,
        "hashes": {
            "query_pixels": _sha(left_pixels[2]),
            "history_actions": _sha(near["raw_actions"][:10]),
            "query_action": _sha(near["raw_actions"][10:15]),
        },
    }


def validate_portal_exit_pair(
    near: dict[str, Any], farther: dict[str, Any]
) -> dict[str, Any]:
    middle_gap = float(
        np.linalg.norm(near["history_states"][1] - farther["history_states"][1])
    )
    future_gap = float(np.linalg.norm(near["future_state"] - farther["future_state"]))
    checks = {
        "same_initial_pixels": np.array_equal(
            near["history_pixels"][0], farther["history_pixels"][0]
        ),
        "same_history_actions": np.array_equal(
            near["history_actions"], farther["history_actions"]
        ),
        "history_reveals_rule": middle_gap >= 12.0,
        "same_query_state": np.allclose(
            near["query_state"], farther["query_state"], atol=1e-5, rtol=0.0
        ),
        "same_query_pixels": np.array_equal(
            near["query_pixels"], farther["query_pixels"]
        ),
        "same_query_action": np.array_equal(
            near["query_action"], farther["query_action"]
        ),
        "future_is_rule_dependent": future_gap >= 12.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "middle_state_gap_px": middle_gap,
        "future_state_gap_px": future_gap,
        "maximum_query_state_gap": float(
            np.max(np.abs(near["query_state"] - farther["query_state"]))
        ),
        "hashes": {
            "query_pixels": _sha(near["query_pixels"]),
            "history_actions": _sha(near["history_actions"]),
            "query_action": _sha(near["query_action"]),
        },
    }


__all__ = [
    "ACTION_BLOCK",
    "BORDER_SIDES",
    "DIRECTIONS",
    "EXIT_MODES",
    "PortalExitTemplate",
    "WALL_AXES",
    "make_template",
    "simulate_portal_exit_clip",
    "simulate_portal_exit_episode",
    "validate_portal_exit_pair",
    "validate_portal_exit_episode_pair",
]
