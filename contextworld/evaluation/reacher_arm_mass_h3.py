from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any

import numpy as np
from scipy.optimize import least_squares


ACTION_BLOCK = 5
MASS_MODES = ("lighter", "heavier")
ARM_DENSITIES = {"lighter": 500.0, "heavier": 1500.0}


@dataclass(frozen=True)
class ReacherArmMassCandidate:
    candidate_id: str
    split: str
    catalog_index: int
    simulator_seed: int
    initial_qpos: tuple[float, float]
    probe_action: tuple[float, float]


def make_candidate(
    *, split: str, index: int, catalog_seed: int
) -> ReacherArmMassCandidate:
    """Create one deterministic initial state and shared probe action."""

    rng = np.random.default_rng(
        np.random.SeedSequence([int(catalog_seed), int(index)])
    )
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction)
    amplitude = float(rng.uniform(0.45, 0.78))
    return ReacherArmMassCandidate(
        candidate_id=f"reacher-arm-mass-{split}-{index:06d}",
        split=str(split),
        catalog_index=int(index),
        simulator_seed=int(rng.integers(0, 2**31 - 1)),
        initial_qpos=(
            float(rng.uniform(-2.30, 2.30)),
            float(rng.uniform(-1.60, 1.60)),
        ),
        probe_action=tuple(float(value) for value in direction * amplitude),
    )


def _state(env: Any) -> np.ndarray:
    physics = env.env.physics
    return np.concatenate(
        [physics.data.qpos.copy(), physics.data.qvel.copy()]
    ).astype(np.float64)


class ReacherArmMassSimulator:
    """Generate contiguous paired trajectories for two hidden arm masses."""

    def __init__(self) -> None:
        # This machine has a working CPU OSMesa renderer but no usable EGL
        # display. Set both selectors explicitly before importing dm_control so
        # earlier tests cannot leave a conflicting default in the environment.
        os.environ["MUJOCO_GL"] = "osmesa"
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        from stable_worldmodel.envs.dmcontrol.reacher import (
            ReacherDMControlWrapper,
        )

        self.envs = {
            mode: ReacherDMControlWrapper(
                task="qpos_match",
                seed=0,
                render_mode="rgb_array",
            )
            for mode in MASS_MODES
        }

    def close(self) -> None:
        for env in self.envs.values():
            env.close()

    @staticmethod
    def _options(
        *, density: float, state: np.ndarray
    ) -> dict[str, Any]:
        return {
            "variation": (),
            "variation_values": {
                "agent.arm_density": np.asarray(
                    [density], dtype=np.float32
                ),
                "agent.finger_density": np.asarray(
                    [density], dtype=np.float32
                ),
                "rendering.render_target": 0,
            },
            "state": np.asarray(state, dtype=np.float64),
        }

    def _reset(
        self,
        mode: str,
        candidate: ReacherArmMassCandidate,
        state: np.ndarray,
    ) -> Any:
        env = self.envs[mode]
        env.reset(
            seed=candidate.simulator_seed,
            options=self._options(
                density=ARM_DENSITIES[mode],
                state=state,
            ),
        )
        env.set_target_qpos(np.zeros(2, dtype=np.float64))
        return env

    @staticmethod
    def _probe(candidate: ReacherArmMassCandidate) -> np.ndarray:
        action = np.asarray(candidate.probe_action, dtype=np.float64)
        return np.repeat(action[None], ACTION_BLOCK, axis=0)

    @staticmethod
    def _recovery(parameters: np.ndarray) -> np.ndarray:
        values = np.asarray(parameters, dtype=np.float64).reshape(2, 2)
        return np.concatenate(
            [
                np.repeat(values[0:1], 2, axis=0),
                np.repeat(values[1:2], 3, axis=0),
            ],
            axis=0,
        )

    def _state_rollout(
        self,
        mode: str,
        candidate: ReacherArmMassCandidate,
        recovery: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        initial = np.asarray(
            [*candidate.initial_qpos, 0.0, 0.0], dtype=np.float64
        )
        env = self._reset(mode, candidate, initial)
        for action in self._probe(candidate):
            env.step(action)
        middle = _state(env)
        for action in recovery:
            env.step(action)
        return middle, _state(env)

    def solve_shared_recovery(
        self,
        candidate: ReacherArmMassCandidate,
        *,
        max_nfev: int = 80,
    ) -> dict[str, Any]:
        """Find one recovery action block shared by both mass conditions."""

        probe = np.asarray(candidate.probe_action, dtype=np.float64)

        def residual(parameters: np.ndarray) -> np.ndarray:
            recovery = self._recovery(parameters)
            _, lighter = self._state_rollout(
                "lighter", candidate, recovery
            )
            _, heavier = self._state_rollout(
                "heavier", candidate, recovery
            )
            return np.concatenate(
                [
                    (lighter[:2] - heavier[:2]) / 0.01,
                    (lighter[2:] - heavier[2:]) / 0.10,
                ]
            )

        initial = np.concatenate([-probe, -probe])
        result = least_squares(
            residual,
            initial,
            bounds=(-1.0, 1.0),
            max_nfev=int(max_nfev),
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
            diff_step=2e-3,
        )
        recovery = self._recovery(result.x)
        lighter_middle, lighter_query = self._state_rollout(
            "lighter", candidate, recovery
        )
        heavier_middle, heavier_query = self._state_rollout(
            "heavier", candidate, recovery
        )
        return {
            "recovery": recovery.astype(np.float32),
            "optimizer": {
                "cost": float(result.cost),
                "function_evaluations": int(result.nfev),
                "optimality": float(result.optimality),
                "status": int(result.status),
            },
            "history_qpos_gap": float(
                np.linalg.norm(lighter_middle[:2] - heavier_middle[:2])
            ),
            "query_state_gap": float(
                np.linalg.norm(lighter_query - heavier_query)
            ),
            "query_states": {
                "lighter": lighter_query,
                "heavier": heavier_query,
            },
        }

    def _episode(
        self,
        mode: str,
        candidate: ReacherArmMassCandidate,
        recovery: np.ndarray,
    ) -> dict[str, Any]:
        initial = np.asarray(
            [*candidate.initial_qpos, 0.0, 0.0], dtype=np.float64
        )
        env = self._reset(mode, candidate, initial)
        probe = self._probe(candidate)
        actions = np.concatenate(
            [
                probe,
                np.asarray(recovery, dtype=np.float64),
                probe,
                np.zeros((ACTION_BLOCK, 2), dtype=np.float64),
            ]
        )
        rows: dict[str, list[Any]] = {
            "pixels": [],
            "action": [],
            "state": [],
            "observation": [],
            "finger_pos": [],
        }
        for action in actions:
            rows["pixels"].append(
                np.asarray(env.render(), dtype=np.uint8).copy()
            )
            rows["action"].append(action.astype(np.float32))
            rows["state"].append(_state(env))
            observation = env.env.task.get_observation(env.env.physics)
            rows["observation"].append(env._obs_to_array(observation))
            rows["finger_pos"].append(
                np.asarray(env.info["finger_pos"], dtype=np.float64)
            )
            env.step(action)
        model_rows = np.asarray([0, 5, 10, 15], dtype=np.int64)
        return {
            "candidate": asdict(candidate),
            "mode": mode,
            "density": ARM_DENSITIES[mode],
            "rows": {
                name: np.stack(values)
                for name, values in rows.items()
            },
            "model_pixels": np.stack(rows["pixels"])[model_rows],
            "model_states": np.stack(rows["state"])[model_rows],
            "model_finger_pos": np.stack(rows["finger_pos"])[model_rows],
            "raw_actions": actions.astype(np.float32),
        }

    def build_pair(
        self,
        candidate: ReacherArmMassCandidate,
        *,
        maximum_query_state_gap: float = 1e-6,
        minimum_history_qpos_gap: float = 0.008,
        maximum_history_qpos_gap: float = 0.08,
        minimum_future_qpos_gap: float = 0.008,
        maximum_future_qpos_gap: float = 0.08,
    ) -> dict[str, Any] | None:
        solution = self.solve_shared_recovery(candidate)
        if solution["query_state_gap"] > maximum_query_state_gap:
            return None
        if solution["history_qpos_gap"] < minimum_history_qpos_gap:
            return None
        if solution["history_qpos_gap"] > maximum_history_qpos_gap:
            return None
        lighter = self._episode(
            "lighter", candidate, solution["recovery"]
        )
        heavier = self._episode(
            "heavier", candidate, solution["recovery"]
        )
        future_gap = float(
            np.linalg.norm(
                lighter["model_states"][3, :2]
                - heavier["model_states"][3, :2]
            )
        )
        query_gap = float(
            np.linalg.norm(
                lighter["model_states"][2]
                - heavier["model_states"][2]
            )
        )
        if not minimum_future_qpos_gap <= future_gap <= maximum_future_qpos_gap:
            return None
        history_changed = int(
            np.count_nonzero(
                lighter["model_pixels"][1]
                != heavier["model_pixels"][1]
            )
        )
        future_changed = int(
            np.count_nonzero(
                lighter["model_pixels"][3]
                != heavier["model_pixels"][3]
            )
        )
        audit = {
            **solution["optimizer"],
            "history_qpos_gap": solution["history_qpos_gap"],
            "future_qpos_gap": future_gap,
            "query_state_gap": query_gap,
            "history_changed_rgb_values": history_changed,
            "future_changed_rgb_values": future_changed,
            "initial_pixels_equal": bool(
                np.array_equal(
                    lighter["model_pixels"][0],
                    heavier["model_pixels"][0],
                )
            ),
            "query_pixels_equal": bool(
                np.array_equal(
                    lighter["model_pixels"][2],
                    heavier["model_pixels"][2],
                )
            ),
            "actions_equal": bool(
                np.array_equal(
                    lighter["raw_actions"], heavier["raw_actions"]
                )
            ),
        }
        audit["passed"] = bool(
            audit["initial_pixels_equal"]
            and audit["query_pixels_equal"]
            and audit["actions_equal"]
            and audit["query_state_gap"] <= maximum_query_state_gap
            and audit["history_qpos_gap"] >= minimum_history_qpos_gap
            and audit["history_qpos_gap"] <= maximum_history_qpos_gap
            and audit["future_qpos_gap"] >= minimum_future_qpos_gap
            and audit["future_qpos_gap"] <= maximum_future_qpos_gap
            and audit["history_changed_rgb_values"] > 0
            and audit["future_changed_rgb_values"] > 0
        )
        if not audit["passed"]:
            return None
        return {
            "candidate": asdict(candidate),
            "lighter": lighter,
            "heavier": heavier,
            "audit": audit,
        }


def find_valid_pair(
    *, split: str, start_index: int, catalog_seed: int
) -> dict[str, Any]:
    simulator = ReacherArmMassSimulator()
    try:
        index = int(start_index)
        while True:
            candidate = make_candidate(
                split=split,
                index=index,
                catalog_seed=catalog_seed,
            )
            result = simulator.build_pair(candidate)
            if result is not None:
                return result
            index += 1
    finally:
        simulator.close()


__all__ = [
    "ACTION_BLOCK",
    "ARM_DENSITIES",
    "FINGER_DENSITY",
    "MASS_MODES",
    "ReacherArmMassCandidate",
    "ReacherArmMassSimulator",
    "find_valid_pair",
    "make_candidate",
]
