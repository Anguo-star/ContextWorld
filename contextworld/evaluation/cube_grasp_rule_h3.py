"""Causal History-3 clips for "does gripper lift move the cube?".

The current image does not reveal whether a closed gripper will carry the
cube.  A short lift in the history does.  Both rule conditions then return to
the same current query before the same lift command is applied again.

This benchmark deliberately changes a synthetic gripper-to-cube vertical
force coupling, not object mass, surface friction, or native OGBench contact.
Robot and cube motion are integrated by the public OGBench MuJoCo simulator.
Only the ``can_hold`` condition transfers the gripper's normalized vertical
command into an additional vertical force on the cube.  The object is
collision-isolated in both conditions, so the hidden coupling cannot alter the
robot state and leak into the current query.

No qpos, qvel, reset, forward, or other state installation occurs after x0.
Immediately before every OGBench control step, the environment supplies a
common gravity-compensation force plus the mode-dependent coupling force in
``qfrc_applied``.  MuJoCo then holds that external generalized force constant
during all physics substeps inside ``mj_step(..., nstep=_n_steps)``.  External
force is a transition input, not an installed simulator state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from typing import Any

import numpy as np


ACTION_BLOCK = 5
MODEL_STEPS = 4
RAW_STEPS = ACTION_BLOCK * MODEL_STEPS
MODEL_FRAME_ROWS = (0, 5, 10, 15)
GRASP_MODES = ("cannot_hold", "can_hold")
GRASP_VALUES = {"cannot_hold": 0.0, "can_hold": 1.0}

# A positive/negative impulse pair lifts the force-coupled cube and leaves it
# stationary at x1.  The reversed pair returns it to x2.  Repeating the first
# pair creates the counterfactual future x3.  This discrete double-integrator
# cycle is reversible up to floating-point roundoff without installing state.
PROBE_PROFILE = np.asarray([1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
RECOVERY_PROFILE = np.asarray([-1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
QUERY_PROFILE = PROBE_PROFILE.copy()

# Newtons per unit normalized vertical gripper command.  With the public Cube
# control/physics timesteps this yields about 9.45 mm separation at x1/x3.
VERTICAL_FORCE_COUPLING_N = 0.30
QUERY_STATE_TOLERANCE = 1e-12
CAPABILITY_NAME = "does_gripper_lift_move_the_cube"


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CubeGraspRuleCandidate:
    candidate_id: str
    split: str
    catalog_index: int
    source_row: int
    source_episode: int
    source_step: int
    simulator_seed: int
    task_id: int
    qpos: tuple[float, ...]
    control: tuple[float, ...]
    cube_color: tuple[float, float, float]
    target_position: tuple[float, float, float]


def action_blocks() -> np.ndarray:
    blocks = np.zeros(
        (MODEL_STEPS, ACTION_BLOCK, 5), dtype=np.float32
    )
    blocks[0, :, 2] = PROBE_PROFILE
    blocks[0, :, 4] = 1.0
    blocks[1, :, 2] = RECOVERY_PROFILE
    blocks[1, :, 4] = 1.0
    blocks[2, :, 2] = QUERY_PROFILE
    blocks[2, :, 4] = 1.0
    return blocks


class CubeGraspRuleSimulator:
    """Reuse one OGBench Cube simulator to build audited paired clips."""

    def __init__(self, *, resolution: int = 224) -> None:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
        import gymnasium as gym
        import stable_worldmodel.envs  # noqa: F401

        self._mujoco = __import__("mujoco")
        self.env = gym.make(
            "swm/OGBCube-v0",
            env_type="single",
            ob_type="states",
            height=int(resolution),
            width=int(resolution),
            terminate_at_goal=False,
            visualize_info=False,
        )
        self.base = self.env.unwrapped
        self.env.reset(seed=0, options={"variation": (), "task_id": 1})
        joint = self.base._model.joint("object_joint_0")
        self.object_qpos_address = int(joint.qposadr[0])
        self.object_dof_address = int(joint.dofadr[0])
        self.object_geom_id = int(self.base._model.geom("object_0").id)
        self.object_body_id = int(self.base._model.body("object_0").id)
        self.base._model.geom_contype[self.object_geom_id] = 0
        self.base._model.geom_conaffinity[self.object_geom_id] = 0
        self.target_geom_id = int(
            self.base._model.geom("target_object_0").id
        )
        self.target_mocap_id = int(
            self.base._model.body("object_target_0").mocapid[0]
        )

    def close(self) -> None:
        self.env.close()

    def _forward(self) -> None:
        self._mujoco.mj_forward(self.base._model, self.base._data)

    def _install_candidate(
        self, candidate: CubeGraspRuleCandidate
    ) -> tuple[np.ndarray, np.ndarray]:
        # A reset is permitted only here, before a trajectory starts.  Each
        # hidden-rule trajectory starts from an independently reconstructed,
        # byte-identical x0.  No reset is performed between x0 and x3.
        self.env.reset(
            seed=int(candidate.simulator_seed),
            options={
                "variation": (),
                "task_id": int(candidate.task_id),
            },
        )
        # OGBench rebuilds the MuJoCo model on reset and restores the original
        # collision masks.  Re-resolve the IDs and isolate the synthetic-rule
        # object for every trajectory; otherwise cube contact would feed force
        # back into the robot and reveal the mode in the shared query state.
        joint = self.base._model.joint("object_joint_0")
        self.object_qpos_address = int(joint.qposadr[0])
        self.object_dof_address = int(joint.dofadr[0])
        self.object_geom_id = int(self.base._model.geom("object_0").id)
        self.object_body_id = int(self.base._model.body("object_0").id)
        self.base._model.geom_contype[self.object_geom_id] = 0
        self.base._model.geom_conaffinity[self.object_geom_id] = 0
        self.target_geom_id = int(
            self.base._model.geom("target_object_0").id
        )
        self.target_mocap_id = int(
            self.base._model.body("object_target_0").mocapid[0]
        )
        qpos = np.asarray(candidate.qpos, dtype=np.float64)
        if qpos.shape != (self.base._model.nq,):
            raise ValueError(
                f"{candidate.candidate_id}: qpos has shape {qpos.shape}"
            )
        qvel = np.zeros(self.base._model.nv, dtype=np.float64)
        control = np.asarray(candidate.control, dtype=np.float64)
        if control.shape != (self.base._model.nu,):
            raise ValueError(
                f"{candidate.candidate_id}: control has shape {control.shape}"
            )
        self.base.set_state(qpos, qvel)
        self.base._data.ctrl[:] = control
        self.base._data.qfrc_applied[:] = 0.0
        self.base._data.time = 0.0
        self.base._data.mocap_pos[self.target_mocap_id] = np.asarray(
            candidate.target_position, dtype=np.float64
        )
        color = np.asarray(candidate.cube_color, dtype=np.float32)
        self.base._model.geom_rgba[self.object_geom_id, :3] = color
        self.base._model.geom_rgba[self.target_geom_id, :3] = color
        self._forward()
        self.base._data.qacc_warmstart[:] = 0.0
        self.base.pre_step()
        self.base.post_step()
        self.base._reset_next_step = False
        start = self.object_qpos_address
        canonical_qpos = self.base._data.qpos[start : start + 7].copy()
        dof = self.object_dof_address
        canonical_qvel = self.base._data.qvel[dof : dof + 6].copy()
        return canonical_qpos, canonical_qvel

    def _apply_transition_force(self, *, mode: str, action_z: float) -> None:
        """Set the generalized force consumed by the following ``mj_step``.

        ``qfrc_applied`` is MuJoCo's external-force input.  OGBench's
        ``set_control`` does not overwrite it, and ``mj_step(..., nstep=25)``
        uses the selected value throughout that control interval.  Updating it
        does not modify qpos, qvel, time, actuator activation, or warm-start
        state and therefore is not a post-x0 state installation.
        """

        self.base._data.qfrc_applied[:] = 0.0
        gravity_z = float(self.base._model.opt.gravity[2])
        mass = float(self.base._model.body_mass[self.object_body_id])
        object_z_dof = self.object_dof_address + 2
        self.base._data.qfrc_applied[object_z_dof] = -mass * gravity_z
        if mode == "can_hold":
            self.base._data.qfrc_applied[object_z_dof] += (
                VERTICAL_FORCE_COUPLING_N * float(action_z)
            )
        elif mode != "cannot_hold":
            raise ValueError(f"Unknown grasp mode: {mode}")

    def _physical_state(self) -> np.ndarray:
        info = self.base.compute_ob_info()
        effector = np.asarray(
            info["proprio/effector_pos"], dtype=np.float32
        )
        cube = np.asarray(
            info["privileged/block_0_pos"], dtype=np.float32
        )
        opening = np.asarray(
            info["proprio/gripper_opening"], dtype=np.float32
        )
        # Cube x/y occupy columns 2:4 for compatibility with the shared
        # paired-training diagnostic.  All fields remain physical SI units.
        return np.asarray(
            [
                effector[0],
                effector[1],
                cube[0],
                cube[1],
                cube[2],
                opening[0],
                effector[2],
            ],
            dtype=np.float32,
        )

    def _simulator_state(self) -> np.ndarray:
        """Return the complete dynamic state used by the equality audit."""

        return np.concatenate(
            (
                np.asarray([self.base._data.time], dtype=np.float64),
                np.asarray(self.base._data.qpos, dtype=np.float64),
                np.asarray(self.base._data.qvel, dtype=np.float64),
                np.asarray(self.base._data.act, dtype=np.float64),
                np.asarray(self.base._data.ctrl, dtype=np.float64),
                np.asarray(self.base._data.qfrc_applied, dtype=np.float64),
                np.asarray(self.base._data.xfrc_applied, dtype=np.float64).reshape(-1),
                np.asarray(self.base._data.mocap_pos, dtype=np.float64).reshape(-1),
                np.asarray(self.base._data.mocap_quat, dtype=np.float64).reshape(-1),
                np.asarray(self.base._data.qacc_warmstart, dtype=np.float64),
            )
        ).copy()

    def _run_mode(
        self,
        candidate: CubeGraspRuleCandidate,
        *,
        mode: str,
        blocks: np.ndarray,
    ) -> dict[str, Any]:
        """Run one complete x0 -> x1 -> x2 -> x3 trajectory.

        The only state installation happens before x0.  Thereafter every
        recorded state is reached by ordinary ``env.step`` calls under the
        selected transition-force coupling.  In particular, x2 is never
        copied, reset, or corrected.
        """

        canonical_qpos, canonical_qvel = self._install_candidate(candidate)
        raw_actions = blocks.reshape(RAW_STEPS, 5)
        pixels: list[np.ndarray] = []
        states: list[np.ndarray] = []
        simulator_states: list[np.ndarray] = []
        prequery_residual: float | None = None

        for raw_step, action in enumerate(raw_actions):
            if raw_step == 2 * ACTION_BLOCK:
                current_qpos = self.base._data.qpos[
                    self.object_qpos_address : self.object_qpos_address + 7
                ]
                current_qvel = self.base._data.qvel[
                    self.object_dof_address : self.object_dof_address + 6
                ]
                prequery_residual = float(
                    max(
                        np.max(np.abs(current_qpos - canonical_qpos)),
                        np.max(np.abs(current_qvel - canonical_qvel)),
                    )
                )

            if raw_step in MODEL_FRAME_ROWS:
                pixels.append(
                    np.asarray(self.base.render(), dtype=np.uint8).copy()
                )
                states.append(self._physical_state())
                simulator_states.append(self._simulator_state())

            self._apply_transition_force(mode=mode, action_z=float(action[2]))
            self.env.step(action)

        if prequery_residual is None:
            raise RuntimeError("Query boundary was not reached")
        return {
            "pixels": np.stack(pixels),
            "physical_state": np.stack(states),
            "simulator_state": np.stack(simulator_states),
            "action_blocks": blocks.copy(),
            "hidden_value": GRASP_VALUES[mode],
            "prequery_residual": prequery_residual,
            "state_installations_after_x0": 0,
            "external_force_updates_after_x0": int(RAW_STEPS),
        }

    def build_pair(
        self, candidate: CubeGraspRuleCandidate
    ) -> dict[str, Any] | None:
        blocks = action_blocks()
        payload = {
            mode: self._run_mode(
                candidate,
                mode=mode,
                blocks=blocks,
            )
            for mode in GRASP_MODES
        }
        audit = validate_cube_grasp_rule_pair(
            payload["cannot_hold"],
            payload["can_hold"],
        )
        if not audit["passed"]:
            return None
        return {
            "candidate": asdict(candidate),
            "audit": audit,
            **payload,
        }


def validate_cube_grasp_rule_pair(
    cannot_hold: dict[str, Any],
    can_hold: dict[str, Any],
) -> dict[str, Any]:
    low_pixels = np.asarray(cannot_hold["pixels"])
    high_pixels = np.asarray(can_hold["pixels"])
    low_states = np.asarray(
        cannot_hold["physical_state"], dtype=np.float32
    )
    high_states = np.asarray(
        can_hold["physical_state"], dtype=np.float32
    )
    low_simulator_states = np.asarray(
        cannot_hold["simulator_state"], dtype=np.float64
    )
    high_simulator_states = np.asarray(
        can_hold["simulator_state"], dtype=np.float64
    )
    history_gap = float(abs(high_states[1, 4] - low_states[1, 4]))
    future_gap = float(abs(high_states[3, 4] - low_states[3, 4]))
    query_gap = float(np.max(np.abs(high_states[2] - low_states[2])))
    initial_simulator_gap = float(
        np.max(np.abs(high_simulator_states[0] - low_simulator_states[0]))
    )
    query_simulator_gap = float(
        np.max(np.abs(high_simulator_states[2] - low_simulator_states[2]))
    )
    prequery_residual = max(
        float(cannot_hold["prequery_residual"]),
        float(can_hold["prequery_residual"]),
    )
    history_changed = int(np.count_nonzero(high_pixels[1] != low_pixels[1]))
    future_changed = int(np.count_nonzero(high_pixels[3] != low_pixels[3]))
    checks = {
        "initial_pixels_identical": np.array_equal(
            low_pixels[0], high_pixels[0]
        ),
        "initial_simulator_state_identical": initial_simulator_gap == 0.0,
        "history_reveals_rule": history_gap >= 0.008,
        "history_pixels_different": history_changed >= 100,
        "prequery_recovery_exact_without_state_edit": (
            float(prequery_residual) <= QUERY_STATE_TOLERANCE
        ),
        "query_physics_identical": query_gap <= QUERY_STATE_TOLERANCE,
        "query_full_simulator_state_identical": (
            query_simulator_gap <= QUERY_STATE_TOLERANCE
        ),
        "query_pixels_identical": np.array_equal(
            low_pixels[2], high_pixels[2]
        ),
        "actions_identical": np.array_equal(
            cannot_hold["action_blocks"], can_hold["action_blocks"]
        ),
        "future_reveals_rule": future_gap >= 0.008,
        "future_pixels_different": future_changed >= 100,
        "no_state_installation_after_x0": (
            int(cannot_hold["state_installations_after_x0"]) == 0
            and int(can_hold["state_installations_after_x0"]) == 0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "history_cube_height_gap_m": history_gap,
        "future_cube_height_gap_m": future_gap,
        "maximum_query_physical_gap": query_gap,
        "maximum_initial_simulator_state_gap": initial_simulator_gap,
        "maximum_query_simulator_state_gap": query_simulator_gap,
        "maximum_prequery_object_state_residual": float(prequery_residual),
        "history_changed_rgb_values": history_changed,
        "future_changed_rgb_values": future_changed,
        "state_installations_after_x0": max(
            int(cannot_hold["state_installations_after_x0"]),
            int(can_hold["state_installations_after_x0"]),
        ),
        "external_force_updates_after_x0": {
            "cannot_hold": int(cannot_hold["external_force_updates_after_x0"]),
            "can_hold": int(can_hold["external_force_updates_after_x0"]),
        },
        "hashes": {
            "query_pixels": array_sha256(low_pixels[2]),
            "action_blocks": array_sha256(
                np.asarray(cannot_hold["action_blocks"])
            ),
        },
    }


__all__ = [
    "ACTION_BLOCK",
    "CAPABILITY_NAME",
    "GRASP_MODES",
    "GRASP_VALUES",
    "QUERY_STATE_TOLERANCE",
    "VERTICAL_FORCE_COUPLING_N",
    "CubeGraspRuleCandidate",
    "CubeGraspRuleSimulator",
    "action_blocks",
    "array_sha256",
    "validate_cube_grasp_rule_pair",
]
