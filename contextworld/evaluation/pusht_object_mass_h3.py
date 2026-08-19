"""Physical identifiability audit for hidden PushT object mass.

The current PushT pusher is a Pymunk kinematic body.  Against a kinematic
body, scaling the pushed object's mass and moment together does not change
the collision trajectory.  This module freezes that negative feasibility
result before any benchmark data or model training is allowed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pymunk

from contextworld.evaluation import pusht_contact_friction_h3 as friction


MASS_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
NUMERICAL_EQUIVALENCE_TOLERANCE = 1e-9


def set_object_mass(env, mass: float) -> None:
    """Scale block mass and rotational moment without changing geometry."""

    mass = float(mass)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("Object mass must be finite and positive")
    old_mass = float(env.block.mass)
    old_moment = float(env.block.moment)
    env.block.mass = mass
    env.block.moment = old_moment * mass / old_mass


def _rollout(template, *, mass: float, query: bool) -> dict[str, Any]:
    env, _ = friction.make_contact_friction_env(
        template,
        mode="medium_friction",
        resolution=96,
    )
    set_object_mass(env, mass)
    if query:
        friction.restore_body_snapshot(
            env,
            template.canonical_query_snapshot,
        )
        actions = np.asarray(template.query_actions, dtype=np.float32)
        capture = {len(actions)}
    else:
        actions = np.asarray(template.history_actions, dtype=np.float32)
        capture = {friction.ACTION_BLOCK, friction.HISTORY_RAW_STEPS}
    snapshots = [friction.body_snapshot(env)]
    pixels = [np.asarray(env.render(), dtype=np.uint8).copy()]
    contacts = []
    try:
        for index, action in enumerate(actions, start=1):
            contacts.append(
                friction._step_and_count_agent_block_contacts(env, action)
            )
            if index in capture:
                snapshots.append(friction.body_snapshot(env))
                pixels.append(np.asarray(env.render(), dtype=np.uint8).copy())
        return {
            "mass": mass,
            "pusher_body_type": int(env.agent.body_type),
            "block_mass": float(env.block.mass),
            "block_moment": float(env.block.moment),
            "snapshots": np.stack(snapshots),
            "pixels": np.stack(pixels),
            "contact_steps": int(sum(value > 0 for value in contacts)),
        }
    finally:
        env.close()


def audit_object_mass_history3() -> dict[str, Any]:
    """Audit eight frozen geometries across five orders of mass."""

    template_rows = []
    all_identical = True
    maximum_history_gap = 0.0
    maximum_future_gap = 0.0
    pusher_is_kinematic = True
    for template in friction.make_frozen_confirmation_templates():
        histories = [
            _rollout(template, mass=mass, query=False)
            for mass in MASS_VALUES
        ]
        futures = [
            _rollout(template, mass=mass, query=True)
            for mass in MASS_VALUES
        ]
        history_reference = histories[0]
        future_reference = futures[0]
        history_gap = max(
            float(
                np.max(
                    np.abs(
                        row["snapshots"]
                        - history_reference["snapshots"]
                    )
                )
            )
            for row in histories[1:]
        )
        future_gap = max(
            float(
                np.max(
                    np.abs(
                        row["snapshots"]
                        - future_reference["snapshots"]
                    )
                )
            )
            for row in futures[1:]
        )
        pixels_identical = all(
            np.array_equal(row["pixels"], history_reference["pixels"])
            for row in histories[1:]
        ) and all(
            np.array_equal(row["pixels"], future_reference["pixels"])
            for row in futures[1:]
        )
        contacts_present = all(
            row["contact_steps"] > 0 for row in histories + futures
        )
        pusher_is_kinematic = bool(
            pusher_is_kinematic
            and all(
                row["pusher_body_type"] == pymunk.Body.KINEMATIC
                for row in histories + futures
            )
        )
        identical = bool(
            history_gap <= NUMERICAL_EQUIVALENCE_TOLERANCE
            and future_gap <= NUMERICAL_EQUIVALENCE_TOLERANCE
            and pixels_identical
            and contacts_present
        )
        all_identical = all_identical and identical
        maximum_history_gap = max(maximum_history_gap, history_gap)
        maximum_future_gap = max(maximum_future_gap, future_gap)
        template_rows.append(
            {
                "template_id": template.template_id,
                "history_state_max_abs_gap": history_gap,
                "future_state_max_abs_gap": future_gap,
                "pixels_bitwise_identical": pixels_identical,
                "contacts_present_all_masses": contacts_present,
                "mass_trajectories_identical": identical,
            }
        )
    identifiable = not all_identical
    return {
        "schema_version": 1,
        "benchmark": "pusht_object_mass_history3_feasibility_v1",
        "status": "failed_physical_identifiability",
        "mass_values": list(MASS_VALUES),
        "templates": len(template_rows),
        "pusher_body_type": "KINEMATIC",
        "pusher_is_kinematic": pusher_is_kinematic,
        "all_mass_trajectories_identical": all_identical,
        "maximum_history_state_gap": maximum_history_gap,
        "maximum_true_future_state_gap": maximum_future_gap,
        "numerical_equivalence_tolerance": (
            NUMERICAL_EQUIVALENCE_TOLERANCE
        ),
        "physically_identifiable": identifiable,
        "generate_formal_data": False,
        "register_suite_component": False,
        "reason": (
            "The PushT pusher is kinematic, so scaling object mass and "
            "moment together does not change the collision trajectory."
        ),
        "template_results": template_rows,
    }


__all__ = [
    "MASS_VALUES",
    "NUMERICAL_EQUIVALENCE_TOLERANCE",
    "audit_object_mass_history3",
    "set_object_mass",
]
