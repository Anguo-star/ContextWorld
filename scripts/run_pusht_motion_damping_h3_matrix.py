#!/usr/bin/env python3
"""Run the motion-damping jobs permitted by the frozen release decision."""

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
from contextworld.paths import artifact_path
import run_pusht_contact_friction_h3_matrix as matrix


def main() -> None:
    matrix.DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = (
        DEFAULT_MOTION_DAMPING_RELEASE_CONFIG
    )
    matrix.load_contact_friction_icl_release = load_motion_damping_icl_release
    matrix.TRAINER = ROOT / "scripts/run_pusht_motion_damping_h3_train.py"
    matrix.DEFAULT_OUTPUT = artifact_path(
        "evaluation/history3/"
        "pusht_motion_damping_strict_causal_reference/reference_training"
    )
    matrix.MATRIX_DESCRIPTION = __doc__
    matrix.main()


if __name__ == "__main__":
    main()
