from __future__ import annotations

import numpy as np

from contextworld.evaluation.reacher_arm_mass_h3 import (
    ReacherArmMassSimulator,
    make_candidate,
)


def test_reacher_arm_mass_history3_pair_is_causal_and_identifiable() -> None:
    simulator = ReacherArmMassSimulator()
    try:
        result = None
        for index in range(24):
            candidate = make_candidate(
                split="test",
                index=index,
                catalog_seed=2026080201,
            )
            result = simulator.build_pair(candidate)
            if result is not None:
                break
        assert result is not None
        assert result["audit"]["passed"]
        assert np.array_equal(
            result["lighter"]["raw_actions"],
            result["heavier"]["raw_actions"],
        )
        assert np.array_equal(
            result["lighter"]["model_pixels"][2],
            result["heavier"]["model_pixels"][2],
        )
        assert not np.array_equal(
            result["lighter"]["model_pixels"][1],
            result["heavier"]["model_pixels"][1],
        )
        assert not np.array_equal(
            result["lighter"]["model_pixels"][3],
            result["heavier"]["model_pixels"][3],
        )
    finally:
        simulator.close()
