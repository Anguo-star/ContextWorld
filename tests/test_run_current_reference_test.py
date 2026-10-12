"""Safety checks for the current native public Test runner."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import run_current_reference_test as runner


def test_freeze_rejects_duplicate_or_incomplete_cohort(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    stable_repo = tmp_path / "stable"
    stable_repo.mkdir()
    model = {"id": "speed/lewm/scratch/s3072", "task": "speed", "family": "lewm",
             "regime": "scratch", "training_seed": 3072,
             "checkpoint": str(checkpoint), "checkpoint_sha256": runner.sha256(checkpoint),
             "stable_repo": str(stable_repo), "stable_ref": "abcdef"}
    models = tmp_path / "models.json"
    models.write_text(json.dumps([model, model]))
    with pytest.raises(ValueError, match="duplicate"):
        runner.freeze(models, tmp_path)
    models.write_text(json.dumps([model]))
    with pytest.raises(ValueError, match="136 runs"):
        runner.freeze(models, tmp_path)


def test_execute_refuses_changed_plan_before_test(tmp_path: Path) -> None:
    plan = {"schema": runner.SCHEMA, "evaluation_split": "test",
            "models_path": str(tmp_path / "models.json"),
            "benchmark_root": str(tmp_path), "runs": []}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    with patch.object(runner, "validate_plan", side_effect=ValueError("frozen cohort changed")), \
         patch.object(runner.subprocess, "run") as invoke:
        with pytest.raises(ValueError, match="frozen cohort"):
            runner.execute(plan_path, tmp_path / "out", "cuda")
        invoke.assert_not_called()


def test_bundle_pin_mismatch(tmp_path: Path) -> None:
    (tmp_path / "task_registry.json").write_text("registry")
    (tmp_path / "manifest.jsonl").write_text("manifest")
    pins = runner.pinned_files(tmp_path)
    (tmp_path / "task_registry.json").write_text("changed")
    assert runner.pinned_files(tmp_path) != pins


def test_shards_and_verified_resume(tmp_path: Path) -> None:
    runs = [{"id": f"speed/lewm/scratch/s{i}", "task": "speed", "family": "lewm",
             "regime": "scratch", "training_seed": i, "adapter": "lewm",
             "checkpoint": "weights.pt", "checkpoint_sha256": "hash",
             "stable_repo": "repo", "stable_ref": "ref"} for i in range(4)]
    plan = {"schema": runner.SCHEMA, "evaluation_split": "test",
            "models_path": "models.json", "benchmark_root": str(tmp_path), "runs": runs}
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(runner.canonical(plan))
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        Path(command[command.index("--output") + 1]).write_text("{}")
        return type("Result", (), {"returncode": 0})()

    with patch.object(runner, "validate_plan") as check, \
         patch.object(runner.subprocess, "run", side_effect=fake_run):
        runner.execute(plan_path, tmp_path / "out", "cuda:1", shard_index=1, num_shards=2)
        assert [x["id"] for x in check.call_args.kwargs["selected"]] == [runs[1]["id"], runs[3]["id"]]
        assert len(calls) == 2
        runner.execute(plan_path, tmp_path / "out", "cuda:1", shard_index=1, num_shards=2)
        assert len(calls) == 2
    receipt = json.loads((tmp_path / "out" / runs[1]["id"] / "receipt.json").read_text())
    assert receipt["plan_sha256"] == runner.hashlib.sha256(runner.canonical(plan)).hexdigest()
    (tmp_path / "out" / runs[1]["id"] / "test.json").write_text("changed")
    with patch.object(runner, "validate_plan"):
        with pytest.raises(ValueError, match="failed or ambiguous"):
            runner.execute(plan_path, tmp_path / "out", "cuda:1", shard_index=1, num_shards=2)
