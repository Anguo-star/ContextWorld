#!/usr/bin/env python3
"""Evaluate the frozen Reacher arm-mass LeWM/PLDM training matrix."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, Path(__file__).resolve().parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from contextworld.benchmarks.reacher_arm_mass_icl_data import (  # noqa: E402
    DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG,
    load_reacher_arm_mass_icl_release,
)
from contextworld.benchmarks.reacher_arm_mass_icl_score import (  # noqa: E402
    score_reacher_arm_mass_icl_results,
)
import eval_pusht_contact_friction_h3_matrix as matrix  # noqa: E402


def main() -> None:
    matrix.DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = (
        DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG
    )
    matrix.load_contact_friction_icl_release = (
        load_reacher_arm_mass_icl_release
    )
    matrix.score_contact_friction_icl_results = (
        score_reacher_arm_mass_icl_results
    )
    matrix.EVALUATOR_MODULE = (
        "contextworld.benchmarks.reacher_arm_mass_icl_cli"
    )
    matrix.MODEL_NAME_PREFIX = "reacher_arm_mass"
    matrix.TRAINING_RECIPE_PREFIX = "reacher_arm_mass"
    matrix.VARIANTS = {
        "lewm": "mixed_frozen_image_paired_future_fit_1p00",
        "pldm": "mixed_pldm_joint",
    }
    matrix.EVALUATION_DESCRIPTION = __doc__
    matrix.main()


if __name__ == "__main__":
    main()
