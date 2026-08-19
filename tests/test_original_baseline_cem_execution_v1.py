from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _runner():
    path = ROOT / "scripts/run_contextworld_original_baseline_cem_v1.py"
    spec = importlib.util.spec_from_file_location(
        "test_contextworld_original_baseline_cem_execution_v1", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prereg() -> dict:
    path = ROOT / "configs/benchmark/contextworld_original_baseline_cem_prereg_v1.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["_identity_audit"] = {
        f"implementation.{name}": {
            "path": str((ROOT / row["path"]).resolve())
        }
        for name, row in payload["implementation"].items()
    }
    return payload


def test_frozen_execution_expands_to_17_unique_jobs_and_2100_episodes() -> None:
    runner = _runner()
    jobs = runner.build_jobs(_prereg())
    assert len(jobs) == 17
    assert sum(row["evaluations"] for row in jobs) == 2100
    assert len({row["job_id"] for row in jobs}) == 17
    assert len({row["output"] for row in jobs}) == 17
    assert sum(row["cell"] == ["tworoom", "lewm"] for row in jobs) == 6
    assert sum(row["cell"] == ["tworoom", "pldm"] for row in jobs) == 6
    assert all("--device" in row["argv"] for row in jobs)
    assert all(row["mujoco_gl"] in {"egl", "osmesa"} for row in jobs)


def test_job_cli_requires_explicit_gpu_but_list_does_not() -> None:
    runner = _runner()
    assert runner.parse_args(["--list"]).list is True
    with pytest.raises(SystemExit):
        runner.parse_args(["--job", "pusht_lewm"])
    parsed = runner.parse_args(
        ["--job", "pusht_lewm", "--gpu-index", "2"]
    )
    assert parsed.gpu_index == 2
