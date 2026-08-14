#!/usr/bin/env python3
"""Freeze the OSMesa recovery Cube CEM retention authorization."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks import cube_original_task_retention_v2 as contract  # noqa: E402
from scripts import freeze_cube_original_task_retention_v1 as freezer  # noqa: E402


freezer.DEFAULT_CUBE_CEM_RETENTION_PREREG = (
    contract.DEFAULT_CUBE_CEM_RETENTION_V2_PREREG
)
freezer.load_cube_cem_retention_prereg = contract.load_cube_cem_retention_v2_prereg
freezer.collect_cube_cem_static_identities = (
    contract.collect_cube_cem_static_identities
)
freezer.expected_cube_cem_jobs = contract.expected_cube_cem_jobs
freezer.validate_cube_cem_query_catalog = contract.validate_cube_cem_query_catalog


if __name__ == "__main__":
    freezer.main()
