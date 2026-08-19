from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_portal_exit_retention_protocol_is_frozen_before_results() -> None:
    import yaml

    release = yaml.safe_load(
        (ROOT / "configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml").read_text()
    )
    protocol = release["scoring"]["original_task_retention"]
    assert protocol["eval_seeds"] == [42, 43, 44, 45, 46, 47]
    assert protocol["episodes_per_eval_seed"] == 50
    assert protocol["episodes_per_checkpoint"] == 300
    assert protocol["noninferiority_margin_successes"] == 15
    assert len(protocol["query_catalog"]["sha256"]) == 64
    assert len(protocol["query_data"]["sha256"]) == 64


def test_portal_exit_retention_scripts_parse() -> None:
    for relative in (
        "scripts/run_tworoom_portal_exit_original_task_cem.py",
        "scripts/aggregate_tworoom_portal_exit_original_task_retention.py",
        "scripts/aggregate_tworoom_portal_exit_reference_training.py",
        "scripts/eval_tworoom_portal_exit_h3_public_test.py",
    ):
        ast.parse((ROOT / relative).read_text(encoding="utf-8"))


def test_portal_exit_retention_runner_is_bounded_and_resumable() -> None:
    source = (
        ROOT / "scripts/run_tworoom_portal_exit_original_task_cem.py"
    ).read_text(encoding="utf-8")
    assert "_completed_result" in source
    assert '"OMP_NUM_THREADS": "1"' in source
    assert '"MKL_NUM_THREADS": "1"' in source
    assert "gpus[: len(wave)]" in source
