"""Small runtime-only compatibility and retry contracts."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import retry_current_reference_test as retry
from scripts import run_current_reference_test_compat as compat
from scripts.run_current_reference_test import SCHEMA, canonical, sha256


def test_damping_overlay_requires_matching_test_bytes_and_audit(tmp_path, monkeypatch):
    from contextworld.benchmarks import motion_damping_icl_data as data
    bundle = tmp_path / "bundle"
    overlay = tmp_path / "overlay"
    relative = Path("artifacts/synthesis/damping/validation.lance")
    for root in (bundle, overlay):
        table = root / relative
        table.mkdir(parents=True)
        (table / "part.bin").write_bytes(b"same Test table")
    (bundle / "task_registry.json").write_text(json.dumps({"components": [
        {"component_id": "motion_damping", "public_test_evaluation":
         {"payloads": [{"public_path": str(relative)}]}}]}))
    monkeypatch.setattr(data, "load_motion_damping_icl_release", lambda: {
        "data": {"artifact_tree": {"root": "artifacts/synthesis/damping"},
                 "lance_tables": {"validation": "validation.lance"}}})
    monkeypatch.setattr(data, "audit_motion_damping_icl_release", lambda **kwargs: {"passed": True})
    assert compat.verify_damping_overlay(bundle, overlay)["validation_files"] == 1
    (overlay / relative / "part.bin").write_bytes(b"different")
    with pytest.raises(ValueError, match="differs"):
        compat.verify_damping_overlay(bundle, overlay)


def test_retry_preserves_success_and_archives_failed_attempt(tmp_path, monkeypatch):
    runs = []
    for task in ("action_strength", "motion_damping"):
        runs.append({"id": f"{task}/lewm/scratch/s3072", "task": task,
                     "family": "lewm", "regime": "scratch", "training_seed": 3072,
                     "checkpoint": "/checkpoint.pt", "checkpoint_sha256": "a" * 64,
                     "stable_repo": "/stable", "stable_ref": "ref", "adapter": "lewm"})
    plan = {"schema": SCHEMA, "evaluation_split": "test", "benchmark_root": "/bundle",
            "runs": runs}
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical(plan))
    plan_sha = hashlib.sha256(canonical(plan)).hexdigest()
    root = tmp_path / "results"
    for run in runs:
        directory = root / run["id"]
        directory.mkdir(parents=True)
        result = directory / "test.json"
        if run["task"] == "action_strength":
            result.write_text("{}")
        (directory / "receipt.json").write_text(json.dumps({
            "id": run["id"], "plan_sha256": plan_sha,
            "checkpoint_sha256": run["checkpoint_sha256"],
            "returncode": 0 if result.exists() else 1,
            "result_sha256": sha256(result) if result.exists() else None}))
    success = root / runs[0]["id"] / "receipt.json"
    previous = success.read_bytes()
    selected = []
    monkeypatch.setattr(retry, "validate_plan", lambda plan, *, selected: None)
    def fake_run(command, **kwargs):
        selected.append(command)
        Path(command[command.index("--output") + 1]).write_text("{}")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(retry.subprocess, "run", fake_run)
    status = retry.execute(plan_path, root, num_shards=1, tasks={"motion_damping"},
                           damping_audit_root=tmp_path / "overlay")
    assert status["selected_ids"] == [runs[1]["id"]]
    assert success.read_bytes() == previous
    assert len(selected) == 1
    assert "scripts.run_current_reference_test_compat" in selected[0]
    directory = root / runs[1]["id"]
    assert (directory / "attempts/1/receipt.json").is_file()
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["result_sha256"] == sha256(directory / "test.json")
    assert receipt["wrapper_sha256"] == sha256(retry.WRAPPER)


def test_wrapper_scopes_artifact_env_without_changing_speed_scorer_root(tmp_path, monkeypatch):
    from contextworld.benchmarks import external_model_cli as cli
    observed = {}
    monkeypatch.setattr(compat, "_prepare_optional_flash_attention", lambda: False)
    monkeypatch.setattr(cli, "parse_args", lambda argv: SimpleNamespace(
        task="speed", benchmark_root=str(tmp_path)))
    monkeypatch.setattr(cli, "_scorer_keywords", lambda binding, args, *, repo_root:
                        {"repo_root": repo_root})
    def fake_main(argv):
        observed["environment"] = os.environ.get("CONTEXTWORLD_ARTIFACT_ROOT")
        observed["scorer_root"] = cli._scorer_keywords(SimpleNamespace(task="speed"),
                                                        None, repo_root=tmp_path)["repo_root"]
        return 0
    monkeypatch.setattr(cli, "main", fake_main)
    monkeypatch.setenv("CONTEXTWORLD_ARTIFACT_ROOT", "/old")
    artifact = tmp_path / "canonical"
    assert compat.main(["--artifact-root", str(artifact)]) == 0
    assert observed == {"environment": str(artifact), "scorer_root": tmp_path}
    assert os.environ["CONTEXTWORLD_ARTIFACT_ROOT"] == "/old"


def test_retry_refuses_ambiguous_existing_results(tmp_path):
    run = {"id": "speed/lewm/scratch/s3072", "checkpoint_sha256": "a" * 64}
    receipt = tmp_path / "receipt.json"
    result = tmp_path / "test.json"
    result.write_text("{}")
    receipt.write_text(json.dumps({"id": run["id"], "plan_sha256": "plan",
                                   "checkpoint_sha256": run["checkpoint_sha256"],
                                   "returncode": 0, "result_sha256": "wrong"}))
    with pytest.raises(ValueError, match="mismatched result"):
        retry._complete(receipt, result, run, "plan")
    receipt.write_text(json.dumps({"id": run["id"], "plan_sha256": "plan",
                                   "checkpoint_sha256": run["checkpoint_sha256"],
                                   "returncode": 1}))
    with pytest.raises(ValueError, match="unexpected result"):
        retry._complete(receipt, result, run, "plan")
