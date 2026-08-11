#!/usr/bin/env python3
"""Evaluate the frozen motion-damping LeWM/PLDM training matrix."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, Path(__file__).resolve().parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from contextworld.benchmarks.motion_damping_icl_data import (
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    load_motion_damping_icl_release,
)
from contextworld.benchmarks.motion_damping_icl_score import (
    score_motion_damping_icl_results,
)
import eval_pusht_contact_friction_h3_matrix as matrix


def main() -> None:
    matrix.DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = (
        DEFAULT_MOTION_DAMPING_RELEASE_CONFIG
    )
    matrix.load_contact_friction_icl_release = load_motion_damping_icl_release
    matrix.score_contact_friction_icl_results = score_motion_damping_icl_results
    matrix.EVALUATOR_MODULE = "contextworld.benchmarks.motion_damping_icl_cli"
    matrix.MODEL_NAME_PREFIX = "motion_damping"
    matrix.TRAINING_RECIPE_PREFIX = "motion_damping"
    matrix.VARIANTS = {
        "lewm": "mixed_frozen_image_identifiable_future_native_0p09",
        "pldm": "mixed_pldm_identifiable_future_joint",
    }
    matrix.EVALUATION_DESCRIPTION = __doc__
    matrix.main()


if __name__ == "__main__":
    main()
