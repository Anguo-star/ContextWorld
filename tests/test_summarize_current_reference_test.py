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
    for run, score in zip(runs, (0.2, 0.5, 0.8)):
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


def test_native_fraction_is_rendered_as_percent(tmp_path: Path) -> None:
    run = _run("contact_friction", 3072)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": runner.SCHEMA, "evaluation_split": "test", "runs": [run]}))
    root = tmp_path / "results"
    _result(root, run, 0.974609, plan)
    data = summary.summarize(plan, root)
    assert data["cells"][0]["main"]["mean"] == pytest.approx(97.4609)
    assert "97.46 (n=1/1)" in summary.render_markdown(data)


def test_incomplete_and_invalid_runs_are_not_rankable(tmp_path: Path) -> None:
    runs = [_run("contact_friction", seed) for seed in (3072, 3073, 3074)]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": runner.SCHEMA, "evaluation_split": "test", "runs": runs}))
    result_root = tmp_path / "results"
    _result(result_root, runs[0], 0.6, plan)
    _result(result_root, runs[1], 0.99, plan, split="development")
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


def test_archive_relocates_receipts_and_rebuilds_report(tmp_path: Path) -> None:
    runs = [_run("contact_friction", seed) for seed in (3072, 3073)]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": runner.SCHEMA, "evaluation_split": "test", "runs": runs}))
    root = tmp_path / "results"
    for run, score in zip(runs, (0.2, 0.8)):
        _result(root, run, score, plan)
    archive = tmp_path / "archive"
    archived = summary.archive_complete(plan, root, archive, expected_runs=2)
    assert archived["completed_runs"] == 2
    assert archived["cells"][0]["main"]["mean"] == 50
    assert (archive / "overview.md").is_file()
    for run in runs:
        for name in ("test.json", "receipt.json"):
            assert (archive / "results" / run["id"] / name).read_bytes() == (
                root / run["id"] / name).read_bytes()
    assert summary.summarize(archive / "plan.json", archive / "results")["completed_runs"] == 2


def test_relocated_receipt_requires_exact_suffix_and_local_hash(tmp_path: Path) -> None:
    run = _run("contact_friction", 3072)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": runner.SCHEMA, "evaluation_split": "test", "runs": [run]}))
    root = tmp_path / "results"
    _result(root, run, 0.7, plan)
    receipt_path = root / run["id"] / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["result"] = "/elsewhere/wrong/test.json"
    receipt_path.write_text(json.dumps(receipt))
    assert summary.summarize(plan, root)["cells"][0]["runs"][0]["status"] == "failed"
    receipt["result"] = f"/elsewhere/{run['id']}/test.json"
    receipt["result_sha256"] = "b" * 64
    receipt_path.write_text(json.dumps(receipt))
    assert "hash differs" in summary.summarize(plan, root)["failures"][0]["error"]


def test_archive_refuses_incomplete_cohort(tmp_path: Path) -> None:
    runs = [_run("contact_friction", seed) for seed in (3072, 3073)]
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": runner.SCHEMA, "evaluation_split": "test", "runs": runs}))
    root = tmp_path / "results"
    _result(root, runs[0], 0.7, plan)
    archive = tmp_path / "archive"
    with pytest.raises(ValueError, match="every frozen Test run"):
        summary.archive_complete(plan, root, archive, expected_runs=2)
    assert not archive.exists()


def test_door_uses_native_score_audit_for_frozen_state(tmp_path: Path) -> None:
    run = _run("door", 3072)
    payload = {"evaluation_split": "test",
               "result_kind": "public_test_offline_final_report_v1",
               "task": "door", "adapter_spec": "lewm",
               "model_name": "lewm-frozen-s3072",
               "result": {"model": {"name": "lewm-frozen-s3072",
                                    "training_recipe": "frozen", "training_seed": 3072,
                                    "adapter": {"checkpoint": run["checkpoint"],
                                                "checkpoint_sha256": run["checkpoint_sha256"],
                                                "adapter_id": "stable_worldmodel_lewm_door_v1",
                                                "protocol": {"history_tokens": 3}}},
                          "score_audit": {"passed": True,
                                          "frozen_state_hash_before": "same",
                                          "frozen_state_hash_after": "same"}}}
    summary._validate_native_result(payload, run)
    payload["result"]["score_audit"]["frozen_state_hash_after"] = "changed"
    with pytest.raises(ValueError, match="model state changed"):
        summary._validate_native_result(payload, run)


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
    _result(root, run, 0.7, plan,
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


def test_public_table_requires_complete_results_and_detects_drift(tmp_path: Path) -> None:
    run = _run("contact_friction", 3072)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": runner.SCHEMA, "evaluation_split": "test", "runs": [run]}))
    root = tmp_path / "results"
    doc = tmp_path / "report.md"
    doc.write_text("Intro\n<!-- BEGIN CURRENT_REFERENCE_TEST -->\n<!-- END CURRENT_REFERENCE_TEST -->\nEnd\n")
    with pytest.raises(ValueError, match="every frozen Test run"):
        summary.sync_report(summary.summarize(plan, root), doc)
    _result(root, run, .75, plan)
    data = summary.summarize(plan, root)
    summary.sync_report(data, doc)
    summary.sync_report(data, doc, check=True)
    assert "75.00 (n=1)" in doc.read_text()
    assert doc.read_text().startswith("Intro\n") and doc.read_text().endswith("End\n")
    doc.write_text(doc.read_text().replace("75.00", "99.00"))
    with pytest.raises(ValueError, match="stale"):
        summary.sync_report(data, doc, check=True)


def test_archive_checkpoint_identity_uses_sha_not_local_mount_alias(tmp_path: Path) -> None:
    run = _run("contact_friction", 3072)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": runner.SCHEMA, "evaluation_split": "test", "runs": [run]}))
    root = tmp_path / "results"
    _result(root, run, .75, plan)
    result = root / run["id"] / "test.json"
    payload = json.loads(result.read_text())
    payload["result"]["model"]["adapter"]["checkpoint"] = "/original-machine/mount/weights.pt"
    result.write_text(json.dumps(payload))
    receipt_path = result.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["result_sha256"] = runner.sha256(result)
    receipt_path.write_text(json.dumps(receipt))
    assert summary.summarize(plan, root)["completed_runs"] == 1
