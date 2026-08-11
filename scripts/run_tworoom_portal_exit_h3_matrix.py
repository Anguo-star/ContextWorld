#!/usr/bin/env python3
"""Run the frozen 2-model x 3-seed TwoRoom portal-exit matrix."""

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
from contextworld.paths import artifact_path  # noqa: E402
import run_pusht_contact_friction_h3_matrix as matrix  # noqa: E402


def main() -> None:
    matrix.DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = DEFAULT_PORTAL_EXIT_RELEASE_CONFIG
    matrix.load_contact_friction_icl_release = load_portal_exit_icl_release
    matrix.TRAINER = ROOT / "scripts/run_tworoom_portal_exit_h3_train.py"
    matrix.DEFAULT_OUTPUT = artifact_path(
        "evaluation/history3/tworoom_portal_exit_h3_release_v1/reference_matrix"
    )
    matrix.MATRIX_DESCRIPTION = __doc__
    matrix.main()


if __name__ == "__main__":
    main()
