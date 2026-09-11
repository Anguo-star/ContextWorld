"""The Public Test final-report orchestrator must be evidence, not memory.

The benchmark protocol is "Development-only selection, Test-only final
reporting".  These tests pin the properties that make the one-shot report
auditable without running any real evaluation: the admission gate refuses
uncleared units, existing results are not silently recomputed, one failing
unit does not sink the batch, and the receipt records enough identity
(checkpoint sha256, benchmark three-piece identity, output sha256) that
``verify`` can catch tampering later.

``subprocess.run`` is replaced with a scripted runner, so nothing here needs
a GPU, the benchmark data, or a real checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from contextworld.benchmarks import public_test_report_cli as cli


def test_v2_admission_rejects_headline_only_development_evidence(tmp_path: Path) -> None:
    """A caller's true flag cannot bypass missing anti-shortcut evidence."""
    checkpoint = make_checkpoint(tmp_path)
    source = tmp_path / "development.json"
    source.write_text(json.dumps({
        "evaluation_split": "development", "task": "action_delay",
        "result": {
            "model": {"training_seed": 3072, "adapter": {
                "checkpoint_sha256": sha256_bytes(checkpoint),
            }},
            "metrics": {"physical_group_macro_accuracy": 1.0},
        },
    }))
    raw = make_unit(component="action_delay", checkpoint=checkpoint,
                    output=tmp_path / "test.json")
    raw["admission"]["development_result"] = {
        "path": str(source), "sha256": sha256_bytes(source),
    }
    units = cli.parse_units({"schema_version": 2, "units": [raw]})
    assert units[0].admission_cleared is False
    assert units[0].reference_admission["decision"]["passed"] is False
    decision = cli.plan_unit(units[0], tmp_path, force=False)
    assert decision.status == cli.STATUS_SKIPPED_NOT_ADMITTED
    assert not decision.will_run


def test_v2_admission_rejects_changed_evidence_and_other_weights(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    source = tmp_path / "development.json"
    envelope = {
        "evaluation_split": "development", "task": "door",
        "result": {"model": {"training_seed": 3072, "adapter": {
            "checkpoint_sha256": "0" * 64,
        }}},
    }
    source.write_text(json.dumps(envelope))
    raw = make_unit(component="door", checkpoint=checkpoint, output=tmp_path / "test.json")
    raw["admission"]["development_result"] = {"path": str(source), "sha256": "0" * 64}
    manifest = {"schema_version": 2, "units": [raw]}
    with pytest.raises(ValueError, match="has changed"):
        cli.parse_units(manifest)
    raw["admission"]["development_result"]["sha256"] = sha256_bytes(source)
    with pytest.raises(ValueError, match="another checkpoint"):
        cli.parse_units(manifest)


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def benchmark_root(tmp_path: Path) -> Path:
    root = tmp_path / "ContextWorld-v1"
    root.mkdir()
    version_json = {
        "dataset_version": "1.0.0-test",
        "schema_version": 1,
        "public_test_included": True,
    }
    (root / "VERSION.json").write_text(json.dumps(version_json, indent=2))
    (root / "manifest.sha256").write_text(
        "aaaa1111  manifest.jsonl\nbbbb2222  task_registry.json\n"
    )
    (root / "task_registry.json").write_text(
        json.dumps({"tasks": ["speed", "door"]})
    )
    return root


def make_checkpoint(tmp_path: Path, name: str = "weights_epoch_10.pt") -> Path:
    checkpoint = tmp_path / "checkpoints" / name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"weights-for-{name}".encode())
    return checkpoint


def make_unit(
    *,
    component: str = "robot_arm_mass",
    checkpoint: Path,
    output: Path,
    cleared_development: bool = True,
    evidence: str = "development report 2026-09-07",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unit: dict[str, Any] = {
        "component": component,
        "family": "lewm",
        "training_seed": 3072,
        "checkpoint": str(checkpoint),
        "adapter": "lewm",
        "stablewm_repo": str(checkpoint.parent / ".stable-worldmodel"),
        "stablewm_ref": "34c55affcc8ec35c24b8a70c37421e90001ad264",
        "output": str(output),
        "admission": {
            "cleared_development": cleared_development,
            "evidence": evidence,
        },
    }
    if extra:
        unit.update(extra)
    return unit


def write_manifest(
    tmp_path: Path,
    benchmark_root: Path,
    units: list[dict[str, Any]],
    report_id: str = "lewm-2026-09-08",
) -> Path:
    manifest = tmp_path / "frozen_checkpoints.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "report_id": report_id,
        "benchmark_root": str(benchmark_root),
        "units": units,
    }, indent=2))
    return manifest


def write_result_output(
    output: Path, checkpoint: Path, *, component: str
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "result_kind": "public_test_offline_final_report_v1",
        "component": component,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_bytes(checkpoint),
    }))


class ScriptedRunner:
    """Stands in for ``subprocess.run`` without evaluating anything.

    Successful calls write a result JSON that records the checkpoint sha256,
    mirroring what the real entry points leave behind.  Components listed in
    ``failures`` return a nonzero exit and write no output.
    """

    def __init__(self, failures: set[str] = frozenset()) -> None:
        self.failures = set(failures)
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **_: Any) -> subprocess.CompletedProcess:
        self.calls.append(list(command))

        def option(name: str) -> str:
            return command[command.index(name) + 1]

        component = (
            option("--task") if "--task" in command
            else Path(option("--checkpoint")).stem
        )
        output = Path(option("--output"))
        checkpoint = Path(option("--checkpoint"))
        if component in self.failures:
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr=f"simulated crash for {component}"
            )
        write_result_output(output, checkpoint, component=component)
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ScriptedRunner:
    scripted = ScriptedRunner()
    monkeypatch.setattr(cli.subprocess, "run", scripted)
    return scripted


def read_receipt(receipt_path: Path) -> dict[str, Any]:
    return json.loads(receipt_path.read_text())


class TestPlan:
    def test_plan_lists_units_and_reasons_without_running_or_writing(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        checkpoint = make_checkpoint(tmp_path)
        output = tmp_path / "eval" / "robot_arm_mass" / "result.json"
        manifest = write_manifest(benchmark_root=benchmark_root, tmp_path=tmp_path, units=[
            make_unit(
                component="robot_arm_mass",
                checkpoint=checkpoint,
                output=output,
                cleared_development=True,
            ),
            make_unit(
                component="door",
                checkpoint=make_checkpoint(tmp_path, "door_weights.pt"),
                output=tmp_path / "eval" / "door" / "result.json",
                cleared_development=False,
                evidence="not yet cleared",
            ),
        ])

        exit_code = cli.main(["plan", "--manifest", str(manifest)])

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "robot_arm_mass/lewm/s3072" in out
        assert "run" in out
        assert "door" in out
        assert cli.STATUS_SKIPPED_NOT_ADMITTED in out
        assert "not yet cleared" in out
        # Planning must not evaluate anything or create any file.
        assert runner.calls == []
        assert not output.exists()
        assert list(tmp_path.glob("**/public_test_report*")) == []

    def test_plan_json_reports_structured_decisions(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        checkpoint = make_checkpoint(tmp_path)
        manifest = write_manifest(benchmark_root=benchmark_root, tmp_path=tmp_path, units=[
            make_unit(checkpoint=checkpoint,
                      output=tmp_path / "eval" / "speed" / "result.json"),
        ])

        assert cli.main(
            ["plan", "--manifest", str(manifest), "--json"]
        ) == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["report_id"] == "lewm-2026-09-08"
        assert plan["units"][0]["will_run"] is True
        assert plan["units"][0]["command"][:2] == [
            cli.sys.executable, "-m"
        ]


class TestAdmissionGate:
    def test_uncleared_unit_is_refused_and_recorded(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
    ) -> None:
        checkpoint = make_checkpoint(tmp_path)
        output = tmp_path / "eval" / "door" / "result.json"
        manifest = write_manifest(benchmark_root=benchmark_root, tmp_path=tmp_path, units=[
            make_unit(
                component="door",
                checkpoint=checkpoint,
                output=output,
                cleared_development=False,
                evidence="selection still in progress",
            ),
        ])
        receipt_path = tmp_path / "receipt.json"

        exit_code = cli.main(
            ["run", "--manifest", str(manifest), "--receipt", str(receipt_path)]
        )

        assert exit_code == 0
        assert runner.calls == []
        assert not output.exists()
        receipt = read_receipt(receipt_path)
        assert receipt["totals"] == {
            "planned": 1,
            "completed": 0,
            "skipped_not_admitted": 1,
            "skipped_already_current": 0,
            "failed": 0,
        }
        unit = receipt["units"][0]
        assert unit["status"] == cli.STATUS_SKIPPED_NOT_ADMITTED
        assert "selection still in progress" in unit["message"]
        assert unit["returncode"] is None
        assert unit["checkpoint_sha256_before"] is None

    def test_missing_admission_flag_counts_as_not_cleared(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
    ) -> None:
        """The gate defaults closed: absent evidence must never execute."""

        checkpoint = make_checkpoint(tmp_path)
        unit = make_unit(checkpoint=checkpoint,
                         output=tmp_path / "eval" / "door" / "result.json")
        del unit["admission"]["cleared_development"]
        manifest = write_manifest(benchmark_root=benchmark_root,
                                  tmp_path=tmp_path, units=[unit])

        exit_code = cli.main([
            "run", "--manifest", str(manifest),
            "--receipt", str(tmp_path / "receipt.json"),
        ])

        assert exit_code == 0
        assert runner.calls == []
        receipt = read_receipt(tmp_path / "receipt.json")
        assert receipt["totals"]["skipped_not_admitted"] == 1
        assert receipt["units"][0]["status"] == cli.STATUS_SKIPPED_NOT_ADMITTED


class TestIdempotency:
    def test_current_output_is_skipped_and_force_reruns(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
    ) -> None:
        checkpoint = make_checkpoint(tmp_path)
        output = tmp_path / "eval" / "robot_arm_mass" / "result.json"
        write_result_output(output, checkpoint, component="robot_arm_mass")
        manifest = write_manifest(benchmark_root=benchmark_root, tmp_path=tmp_path, units=[
            make_unit(checkpoint=checkpoint, output=output),
        ])
        receipt_path = tmp_path / "receipt.json"

        first = cli.main(
            ["run", "--manifest", str(manifest), "--receipt", str(receipt_path)]
        )

        assert first == 0
        assert runner.calls == []
        receipt = read_receipt(receipt_path)
        unit = receipt["units"][0]
        assert unit["status"] == cli.STATUS_SKIPPED_ALREADY_CURRENT
        assert unit["checkpoint_sha256_before"] == sha256_bytes(checkpoint)
        assert unit["checkpoint_sha256_after"] == sha256_bytes(checkpoint)
        assert unit["output_sha256"] == sha256_bytes(output)

        second = cli.main(
            ["run", "--manifest", str(manifest),
             "--receipt", str(receipt_path), "--force"]
        )

        assert second == 0
        assert len(runner.calls) == 1
        assert "--evaluation-split" in runner.calls[0]
        assert "test" in runner.calls[0]
        assert read_receipt(receipt_path)["units"][0]["status"] == (
            cli.STATUS_COMPLETED
        )

    def test_stale_output_is_rerun(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
    ) -> None:
        checkpoint = make_checkpoint(tmp_path)
        output = tmp_path / "eval" / "robot_arm_mass" / "result.json"
        output.parent.mkdir(parents=True)
        # Records the sha of a different checkpoint: not current any more.
        output.write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
        manifest = write_manifest(benchmark_root=benchmark_root, tmp_path=tmp_path, units=[
            make_unit(checkpoint=checkpoint, output=output),
        ])

        exit_code = cli.main([
            "run", "--manifest", str(manifest),
            "--receipt", str(tmp_path / "receipt.json"),
        ])

        assert exit_code == 0
        assert len(runner.calls) == 1
        assert read_receipt(tmp_path / "receipt.json")["units"][0][
            "status"
        ] == cli.STATUS_COMPLETED


class TestFailureIsolation:
    def test_failed_unit_does_not_abort_the_batch(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
    ) -> None:
        runner.failures = {"door"}
        units = []
        for component in ("robot_arm_mass", "door", "speed"):
            units.append(make_unit(
                component=component,
                checkpoint=make_checkpoint(tmp_path, f"{component}.pt"),
                output=tmp_path / "eval" / component / "result.json",
            ))
        manifest = write_manifest(benchmark_root=benchmark_root, tmp_path=tmp_path, units=units)
        receipt_path = tmp_path / "receipt.json"

        exit_code = cli.main(
            ["run", "--manifest", str(manifest), "--receipt", str(receipt_path)]
        )

        assert exit_code == 1
        assert len(runner.calls) == 3
        receipt = read_receipt(receipt_path)
        assert receipt["totals"] == {
            "planned": 3,
            "completed": 2,
            "skipped_not_admitted": 0,
            "skipped_already_current": 0,
            "failed": 1,
        }
        statuses = {unit["component"]: unit for unit in receipt["units"]}
        assert statuses["robot_arm_mass"]["status"] == cli.STATUS_COMPLETED
        assert statuses["speed"]["status"] == cli.STATUS_COMPLETED
        failed = statuses["door"]
        assert failed["status"] == cli.STATUS_FAILED
        assert failed["returncode"] == 1
        assert "simulated crash for door" in failed["message"]
        assert not (tmp_path / "eval" / "door" / "result.json").exists()

    def test_fail_fast_stops_after_first_failure(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
    ) -> None:
        runner.failures = {"robot_arm_mass"}
        units = [
            make_unit(
                component="robot_arm_mass",
                checkpoint=make_checkpoint(tmp_path, "a.pt"),
                output=tmp_path / "eval" / "robot_arm_mass" / "result.json",
            ),
            make_unit(
                component="door",
                checkpoint=make_checkpoint(tmp_path, "b.pt"),
                output=tmp_path / "eval" / "door" / "result.json",
            ),
        ]
        manifest = write_manifest(benchmark_root=benchmark_root, tmp_path=tmp_path, units=units)

        exit_code = cli.main([
            "run", "--manifest", str(manifest),
            "--receipt", str(tmp_path / "receipt.json"), "--fail-fast",
        ])

        assert exit_code == 1
        assert len(runner.calls) == 1


class TestReceiptIdentity:
    def test_receipt_pins_benchmark_identity(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
    ) -> None:
        checkpoint = make_checkpoint(tmp_path)
        manifest = write_manifest(benchmark_root=benchmark_root, tmp_path=tmp_path, units=[
            make_unit(
                checkpoint=checkpoint,
                output=tmp_path / "eval" / "robot_arm_mass" / "result.json",
            ),
        ])
        receipt_path = tmp_path / "receipt.json"

        cli.main(["run", "--manifest", str(manifest), "--receipt", str(receipt_path)])

        identity = read_receipt(receipt_path)["benchmark_identity"]
        assert identity["version_json"] == json.loads(
            (benchmark_root / "VERSION.json").read_text()
        )
        assert identity["manifest_sha256"] == (
            benchmark_root / "manifest.sha256"
        ).read_text()
        assert identity["task_registry_sha256"] == sha256_bytes(
            benchmark_root / "task_registry.json"
        )

    def test_receipt_records_checkpoint_and_command(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
    ) -> None:
        checkpoint = make_checkpoint(tmp_path)
        output = tmp_path / "eval" / "robot_arm_mass" / "result.json"
        manifest = write_manifest(benchmark_root=benchmark_root, tmp_path=tmp_path, units=[
            make_unit(checkpoint=checkpoint, output=output),
        ])
        receipt_path = tmp_path / "receipt.json"

        cli.main(["run", "--manifest", str(manifest), "--receipt", str(receipt_path)])

        unit = read_receipt(receipt_path)["units"][0]
        assert unit["checkpoint_sha256_before"] == sha256_bytes(checkpoint)
        assert unit["checkpoint_sha256_after"] == sha256_bytes(checkpoint)
        assert "unchanged" in unit["message"]
        assert unit["command"] == runner.calls[0]
        assert unit["output_sha256"] == sha256_bytes(output)
        assert unit["returncode"] == 0

    def test_default_receipt_path_lives_under_output_root(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
    ) -> None:
        checkpoint = make_checkpoint(tmp_path)
        output_root = tmp_path / "reports"
        output_root.mkdir()
        manifest = write_manifest(
            tmp_path=output_root,
            benchmark_root=benchmark_root,
            units=[make_unit(
                checkpoint=checkpoint,
                output=tmp_path / "eval" / "robot_arm_mass" / "result.json",
            )],
        )

        cli.main(["run", "--manifest", str(manifest), "--output-root", str(output_root)])

        expected = output_root / "public_test_report_lewm-2026-09-08.json"
        assert expected.exists()
        receipt = read_receipt(expected)
        assert receipt["report_id"] == "lewm-2026-09-08"
        assert receipt["schema_version"] == 1
        assert receipt["created_utc"].endswith("Z")


class TestVerify:
    def test_verify_detects_tampered_output(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        checkpoint = make_checkpoint(tmp_path)
        outputs = {}
        units = []
        for component in ("robot_arm_mass", "door"):
            output = tmp_path / "eval" / component / "result.json"
            outputs[component] = output
            units.append(make_unit(
                component=component,
                checkpoint=make_checkpoint(tmp_path, f"{component}.pt"),
                output=output,
            ))
        manifest = write_manifest(benchmark_root=benchmark_root, tmp_path=tmp_path, units=units)
        receipt_path = tmp_path / "receipt.json"
        cli.main(["run", "--manifest", str(manifest), "--receipt", str(receipt_path)])

        assert cli.main(["verify", "--receipt", str(receipt_path)]) == 0
        assert "all recorded outputs match" in capsys.readouterr().out

        outputs["door"].write_text(
            json.dumps({"checkpoint_sha256": "0" * 64, "tampered": True})
        )

        exit_code = cli.main(["verify", "--receipt", str(receipt_path)])

        assert exit_code == 1
        out = capsys.readouterr().out
        assert "door" in out
        assert "SHA256 MISMATCH" in out
        assert "robot_arm_mass: ok" in out

    def test_verify_detects_missing_output(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        checkpoint = make_checkpoint(tmp_path)
        output = tmp_path / "eval" / "robot_arm_mass" / "result.json"
        manifest = write_manifest(benchmark_root=benchmark_root, tmp_path=tmp_path, units=[
            make_unit(checkpoint=checkpoint, output=output),
        ])
        receipt_path = tmp_path / "receipt.json"
        cli.main(["run", "--manifest", str(manifest), "--receipt", str(receipt_path)])
        output.unlink()

        assert cli.main(["verify", "--receipt", str(receipt_path)]) == 1
        assert "MISSING" in capsys.readouterr().out


class TestComponentScorer:
    def test_declared_component_cli_is_invoked(
        self,
        tmp_path: Path,
        benchmark_root: Path,
        runner: ScriptedRunner,
    ) -> None:
        checkpoint = make_checkpoint(tmp_path)
        output = tmp_path / "eval" / "action_delay" / "result.json"
        manifest = write_manifest(benchmark_root=benchmark_root, tmp_path=tmp_path, units=[
            make_unit(
                component="action_delay",
                checkpoint=checkpoint,
                output=output,
                extra={"component_cli": "action_delay_icl_cli"},
            ),
        ])

        cli.main([
            "run", "--manifest", str(manifest),
            "--receipt", str(tmp_path / "receipt.json"),
        ])

        command = runner.calls[0]
        module = command[command.index("-m") + 1]
        assert module == "contextworld.benchmarks.action_delay_icl_cli"
        assert "eval" in command
        assert "--evaluation-split" not in command
        assert "--benchmark-root" not in command
