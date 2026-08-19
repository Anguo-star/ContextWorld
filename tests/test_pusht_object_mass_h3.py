from __future__ import annotations

import numpy as np

from contextworld.evaluation.pusht_contact_friction_h3 import (
    make_frozen_confirmation_templates,
    make_contact_friction_env,
)
from contextworld.evaluation.pusht_object_mass_h3 import (
    MASS_VALUES,
    NUMERICAL_EQUIVALENCE_TOLERANCE,
    audit_object_mass_history3,
    set_object_mass,
)


def test_mass_assignment_scales_moment() -> None:
    template = make_frozen_confirmation_templates()[0]
    env, _ = make_contact_friction_env(
        template,
        mode="medium_friction",
        resolution=96,
    )
    try:
        original_mass = float(env.block.mass)
        original_moment = float(env.block.moment)
        set_object_mass(env, 10.0)
        assert float(env.block.mass) == 10.0
        assert np.isclose(
            float(env.block.moment),
            original_moment * 10.0 / original_mass,
        )
    finally:
        env.close()


def test_current_pusht_mass_is_not_physically_identifiable() -> None:
    report = audit_object_mass_history3()
    assert report["mass_values"] == list(MASS_VALUES)
    assert report["templates"] == 8
    assert report["pusher_is_kinematic"] is True
    assert report["all_mass_trajectories_identical"] is True
    assert (
        report["maximum_history_state_gap"]
        <= NUMERICAL_EQUIVALENCE_TOLERANCE
    )
    assert report["maximum_true_future_state_gap"] == 0.0
    assert report["physically_identifiable"] is False
    assert report["generate_formal_data"] is False
    assert report["register_suite_component"] is False
