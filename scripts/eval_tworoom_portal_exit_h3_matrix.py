#!/usr/bin/env python3
"""Evaluate the frozen TwoRoom portal-exit LeWM/PLDM matrix."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, Path(__file__).resolve().parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from contextworld.benchmarks.portal_exit_icl_data import (  # noqa: E402
    DEFAULT_PORTAL_EXIT_RELEASE_CONFIG,
    load_portal_exit_icl_release,
)
from contextworld.benchmarks.portal_exit_icl_score import (  # noqa: E402
    score_portal_exit_icl_results,
)
import eval_pusht_contact_friction_h3_matrix as matrix  # noqa: E402


def main() -> None:
    matrix.DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = DEFAULT_PORTAL_EXIT_RELEASE_CONFIG
    matrix.load_contact_friction_icl_release = load_portal_exit_icl_release
    matrix.score_contact_friction_icl_results = score_portal_exit_icl_results
    matrix.EVALUATOR_MODULE = "contextworld.benchmarks.portal_exit_icl_cli"
    matrix.MODEL_NAME_PREFIX = "portal_exit"
    matrix.TRAINING_RECIPE_PREFIX = "portal_exit"
    matrix.EVALUATION_DESCRIPTION = __doc__
    matrix.VARIANTS = {
        "lewm": "mixed_frozen_image_native_0p09",
        "pldm": "mixed_pldm_joint",
    }
    matrix.main()


if __name__ == "__main__":
    main()
