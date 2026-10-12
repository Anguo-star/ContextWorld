"""Aggregation contract for frozen native Public Test runs."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_current_reference_test as runner
from scripts import summarize_current_reference_test as summary


def _run(task: str, seed: int) -> dict:
    return {"id": f"{task}/lewm/frozen/s{seed}", "task": task,
            "family": "lewm", "regime": "frozen", "training_seed": seed,
            "checkpoint": f"/frozen/checkpoints/{task}/s{seed}.pt",
            "checkpoint_sha256": "a" * 64}


def _result(root: Path, run: dict, score: float, plan: Path, *, split: str = "test",
            checkpoint_sha: str | None = None, history_tokens: int = 3,
            adapter_id: str = "stable_worldmodel_lewm_contact_friction_v1") -> None:
    directory = root / run["id"]
    directory.mkdir(parents=True)
    path = directory / "test.json"
    path.write_text(json.dumps({"evaluation_split": split,
                                "result_kind": "public_test_offline_final_report_v1",
                                "task": run["task"], "adapter_spec": "lewm",
                                "model_name": f"lewm-frozen-s{run['training_seed']}",
                                "result": {"model": {
                                    "name": f"lewm-frozen-s{run['training_seed']}",
                                    "training_recipe": "frozen",
                                    "training_seed": run["training_seed"],
                                    "adapter": {"checkpoint": run["checkpoint"],
                                                "checkpoint_sha256": checkpoint_sha or run["checkpoint_sha256"],
                                                "adapter_id": adapter_id,
                                                "protocol": {"history_tokens": history_tokens}},
                                    "state_sha256_before": "same", "state_sha256_after": "same"},
                                    "metrics": {"correct_future_rate": score}}}))
    (directory / "receipt.json").write_text(json.dumps({
        "id": run["id"], "checkpoint_sha256": run["checkpoint_sha256"],
        "plan_sha256": hashlib.sha256(runner.canonical(json.loads(plan.read_text()))).hexdigest(),
        "returncode": 0, "result": str(path),
        "result_sha256": runner.sha256(path)}))


def test_complete_cell_uses_equal_seed_mean_and_sample_std(tmp_path: Path) -> None:
    runs = [_run("contact_friction", seed) for seed in (3072, 3073, 3074)]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": runner.SCHEMA, "evaluation_split": "test", "runs": runs}))
    result_root = tmp_path / "results"
    for run, score in zip(runs, (20, 50, 80)):
        _result(result_root, run, score, plan)
    data = summary.summarize(plan, result_root)
    cell = data["cells"][0]
    assert cell["status"] == "complete"
    assert cell["main"] == {"mean": 50.0, "std": 30.0, "n": 3}
    assert [r["main_score"] for r in cell["runs"]] == [20.0, 50.0, 80.0]
    output = tmp_path / "report"
    summary.write_outputs(data, output)
    with (output / "summary.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["main_mean"] == "50.0"
    assert "50.00 ± 30.00 (n=3/3)" in (output / "overview.md").read_text()


def test_incomplete_and_invalid_runs_are_not_rankable(tmp_path: Path) -> None:
    runs = [_run("contact_friction", seed) for seed in (3072, 3073, 3074)]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": runner.SCHEMA, "evaluation_split": "test", "runs": runs}))
    result_root = tmp_path / "results"
    _result(result_root, runs[0], 60, plan)
    _result(result_root, runs[1], 99, plan, split="development")
    data = summary.summarize(plan, result_root)
    cell = data["cells"][0]
    assert (cell["completed"], cell["failed"], cell["expected"]) == (1, 1, 3)
    assert cell["main"] is None
    assert cell["observed_main"] == {"mean": 60.0, "std": None, "n": 1}
    assert [r["status"] for r in cell["runs"]] == ["completed", "failed", "pending"]
    assert len(data["failures"]) == 1
    assert "待完成 (1/3)" in summary.render_markdown(data)
    assert "60.00" not in summary.render_markdown(data)
    assert "CEM remains" in summary.render_markdown(data)


def test_duplicate_frozen_run_rejected(tmp_path: Path) -> None:
    run = _run("contact_friction", 3072)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": runner.SCHEMA, "evaluation_split": "test",
                                "runs": [run, run]}))
    with pytest.raises(ValueError, match="duplicate run ID"):
        summary.summarize(plan, tmp_path / "results")


@pytest.mark.parametrize("mutation,reason", [
    ("plan", "receipt identity"),
    ("checkpoint", "checkpoint differs"),
    ("history", "native history mode"),
    ("projected", "native history mode"),
])
def test_identity_or_native_mode_mismatch_is_failed(
        tmp_path: Path, mutation: str, reason: str) -> None:
    run = _run("contact_friction", 3072)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": runner.SCHEMA, "evaluation_split": "test",
                                "runs": [run]}))
    root = tmp_path / "results"
    _result(root, run, 70, plan,
            checkpoint_sha="b" * 64 if mutation == "checkpoint" else None,
            history_tokens=2 if mutation == "history" else 3,
            adapter_id=("stable_worldmodel_lewm_h3_tail_projection_action_delay_v2"
                        if mutation == "projected" else
                        "stable_worldmodel_lewm_contact_friction_v1"))
    if mutation == "plan":
        changed = json.loads(plan.read_text())
        changed["extra_plan_field"] = "different frozen plan"
        plan.write_text(json.dumps(changed))
    cell = summary.summarize(plan, root)["cells"][0]
    assert cell["status"] == "incomplete"
    assert cell["main"] is None
    assert reason in cell["runs"][0]["error"]
