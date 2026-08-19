#!/usr/bin/env python3
"""Estimate frozen response scales from Motion Damping Training only."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, Path(__file__).resolve().parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from contextworld.benchmarks import motion_damping_icl_data as data
import estimate_pusht_contact_friction_scales as estimator


def main() -> None:
    estimator.DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = (
        data.DEFAULT_MOTION_DAMPING_RELEASE_CONFIG
    )
    estimator.load_contact_friction_icl_release = (
        data.load_motion_damping_icl_release
    )
    estimator.file_sha256 = data.file_sha256
    estimator.friction._read_lance_pairs = data._read_lance_pairs
    estimator.main()


if __name__ == "__main__":
    main()
