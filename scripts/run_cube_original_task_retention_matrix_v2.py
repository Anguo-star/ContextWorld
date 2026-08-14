#!/usr/bin/env python3
"""Run the OSMesa recovery Cube CEM retention matrix."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks import cube_original_task_retention_v2 as contract  # noqa: E402
from scripts import run_cube_original_task_retention_matrix as runner  # noqa: E402


runner.DEFAULT_CUBE_CEM_RETENTION_PREREG = (
    contract.DEFAULT_CUBE_CEM_RETENTION_V2_PREREG
)
runner.load_cube_cem_retention_prereg = contract.load_cube_cem_retention_v2_prereg
runner.build_cube_cem_retention_result = (
    contract.build_cube_cem_v2_retention_result
)
runner.validate_cube_cem_job_result = contract.validate_cube_cem_v2_job_result


if __name__ == "__main__":
    runner.main()
