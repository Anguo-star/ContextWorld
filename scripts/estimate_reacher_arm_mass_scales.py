#!/usr/bin/env python3
"""Estimate frozen response scales from Reacher Arm Mass Training only."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, Path(__file__).resolve().parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from contextworld.benchmarks import reacher_arm_mass_icl_data as data  # noqa: E402
import estimate_pusht_contact_friction_scales as estimator  # noqa: E402
import run_reacher_arm_mass_h3_train as reacher  # noqa: E402


def main() -> None:
    estimator.DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = (
        data.DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG
    )
    estimator.load_contact_friction_icl_release = (
        data.load_reacher_arm_mass_icl_release
    )
    estimator.file_sha256 = data.file_sha256
    estimator.friction._read_lance_pairs = data._read_lance_pairs
    estimator.friction.DEFAULT_CHECKPOINT = (
        reacher.REACHER_ROOT
        / "ckpt/reacher_lewm/reacher_lewm_weights.ckpt"
    )
    estimator.friction.DEFAULT_ORIGINAL_H5 = (
        reacher.REACHER_ROOT / "reacher.h5"
    )
    estimator.pilot.original_action_stats = reacher._finite_action_stats
    estimator.main()


if __name__ == "__main__":
    main()
