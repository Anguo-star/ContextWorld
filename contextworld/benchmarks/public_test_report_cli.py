"""One-shot Public Test final-report entry point.

The benchmark protocol is "Development-only selection, Test-only final
reporting": Development runs automatically with training, while Public Test is
run exactly once, after the method and its checkpoints are frozen.  This
command replaces the hand-written one-off evaluation scripts for that final
run: given a frozen checklist of checkpoints it plans, executes, and verifies
every reporting unit and leaves a receipt that answers "what ran, what did
not, and why" from artifacts instead of memory.

It is an orchestrator only and re-implements no scoring logic.  Each unit is
evaluated by an existing entry point invoked through ``subprocess``:

* the public-split generic entry
  ``python -m contextworld.benchmarks.external_model_cli --evaluation-split test ...``; or
* a component's frozen formal scorer
  ``python -m contextworld.benchmarks.<component>_icl_cli eval ...`` when the
  checklist declares one through the unit's ``component_cli`` field.

Hard guarantees enforced here
------------------------------

* Admission gate: any unit whose ``admission.cleared_development`` is not
  ``true`` is never executed and is recorded as ``skipped_not_admitted``.
  This keeps Test used for final reporting only, never for selection.
* Checkpoint identity: the checkpoint ``sha256`` is recorded before and after
  every executed unit; evaluation must not touch the weights.
* Data identity: the receipt pins the benchmark root's ``VERSION.json``
  content, ``manifest.sha256`` content, and ``task_registry.json`` digest.
* Idempotency: an existing output whose recorded checkpoint ``sha256`` matches
  the checklist's checkpoint is skipped as ``skipped_already_current``
  unless ``--force`` is given.
* Failures are per unit (returncode + error summary) and never abort the
  batch unless ``--fail-fast`` is given.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from contextworld.synthesis.manifest import write_json

SCHEMA_VERSION = 1
GENERIC_CLI_MODULE = "contextworld.benchmarks.external_model_cli"
COMPONENT_CLI_TEMPLATE = "contextworld.benchmarks.{suffix}"

STATUS_COMPLETED = "completed"
STATUS_SKIPPED_NOT_ADMITTED = "skipped_not_admitted"
STATUS_SKIPPED_ALREADY_CURRENT = "skipped_already_current"
STATUS_FAILED = "failed"

_ADMISSION_GATE_REASON = (
    "admission.cleared_development is not true; Public Test requires a unit "
    "cleared on Development first"
)
_ERROR_SUMMARY_LIMIT = 2000


def sha256_file(path: Path) -> str | None:
    """Hex digest of a file, or ``None`` when the file is absent/unreadable."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


@dataclass(frozen=True)
class ReportUnit:
    """One checklist entry: one checkpoint evaluated on one component."""

    component: str
    family: str
    training_seed: int
    checkpoint: Path
    adapter: str
    output: Path
    admission_cleared: bool
    admission_evidence: str
    stablewm_repo: Path | None = None
    stablewm_ref: str | None = None
    component_cli: str | None = None
    model_name: str | None = None
    training_recipe: str | None = None
    device: str | None = None
    batch_size: int | None = None
    reference_admission: dict[str, Any] | None = None

    def resolved_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        if self.training_seed is not None:
            return f"{self.component}_{self.family}_s{self.training_seed}"
        return f"{self.component}_{self.family}"


def load_report_manifest(path: Path) -> dict[str, Any]:
    """Read and validate a frozen checkpoint checklist."""

    try:
        manifest = json.loads(path.read_text())
    except OSError as error:
        raise ValueError(f"cannot read checklist {path}: {error}") from error
    except ValueError as error:
        raise ValueError(f"checklist {path} is not valid JSON: {error}") from error

    if not isinstance(manifest, dict):
        raise ValueError(f"checklist {path} must be a JSON object")
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError(
            f"checklist {path}: unsupported schema_version "
            f"{manifest.get('schema_version')!r}, expected 1 or 2"
        )
    for key in ("report_id", "benchmark_root", "units"):
        if not manifest.get(key):
            raise ValueError(f"checklist {path}: missing or empty {key!r}")
    if not isinstance(manifest["units"], list):
        raise ValueError(f"checklist {path}: 'units' must be a list")
    return manifest


def parse_unit(raw: Any, index: int) -> ReportUnit:
    """Validate one checklist unit and normalize it to :class:`ReportUnit`."""

    if not isinstance(raw, dict):
        raise ValueError(f"unit[{index}] must be a JSON object")
    for key in ("component", "family", "training_seed", "checkpoint",
                "adapter", "output", "admission"):
        if key not in raw:
            raise ValueError(f"unit[{index}] is missing required field {key!r}")

    seed = raw["training_seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"unit[{index}]: training_seed must be an integer")

    admission = raw["admission"]
    if not isinstance(admission, dict):
        raise ValueError(f"unit[{index}]: admission must be a JSON object")
    # The gate is deliberately tolerant: anything that is not the JSON literal
    # ``true`` (missing, null, false) counts as not cleared.  Refusing to
    # execute is the required behavior; rejecting the whole checklist is not.
    cleared = admission.get("cleared_development") is True
    evidence = admission.get("evidence", "")

    def absolute(key: str) -> Path:
        value = Path(raw[key])
        if not value.is_absolute():
            raise ValueError(f"unit[{index}]: {key} must be an absolute path")
        return value

    component_cli = raw.get("component_cli")
    if component_cli is not None and not isinstance(component_cli, str):
        raise ValueError(f"unit[{index}]: component_cli must be a string")

    batch_size = raw.get("batch_size")
    if batch_size is not None and (
        not isinstance(batch_size, int) or isinstance(batch_size, bool)
    ):
        raise ValueError(f"unit[{index}]: batch_size must be an integer")

    return ReportUnit(
        component=raw["component"],
        family=raw["family"],
        training_seed=seed,
        checkpoint=absolute("checkpoint"),
        adapter=raw["adapter"],
        output=absolute("output"),
        admission_cleared=cleared,
        admission_evidence=str(evidence),
        stablewm_repo=(
            Path(raw["stablewm_repo"]) if raw.get("stablewm_repo") else None
        ),
        stablewm_ref=raw.get("stablewm_ref"),
        component_cli=component_cli,
        model_name=raw.get("model_name"),
        training_recipe=raw.get("training_recipe"),
        device=raw.get("device"),
        batch_size=batch_size,
    )


def parse_units(manifest: dict[str, Any]) -> list[ReportUnit]:
    units = [parse_unit(raw, index) for index, raw in enumerate(manifest["units"])]
    if manifest.get("schema_version") != 2:
        return units  # Historical v1 checklists retain their declared semantics.
    return [
        _bind_reference_admission(unit, raw["admission"])
        for unit, raw in zip(units, manifest["units"], strict=True)
    ]


def _bind_reference_admission(unit: ReportUnit, admission: dict[str, Any]) -> ReportUnit:
    """V2 admission requires a hash-bound Development result for these weights."""
    from contextworld.benchmarks.reference_decision import reference_decision_for_result

    identity = admission.get("development_result")
    if not isinstance(identity, dict) or not identity.get("path") or not identity.get("sha256"):
        raise ValueError("schema v2 admission requires development_result path and sha256")
    source = Path(identity["path"])
    if not source.is_absolute() or sha256_file(source) != identity["sha256"]:
        raise ValueError("Development admission source is absent, relative, or has changed")
    envelope = json.loads(source.read_text())
    if envelope.get("evaluation_split") != "development":
        raise ValueError("Admission evidence must be a Development result")
    payload = envelope.get("result", envelope)
    if envelope.get("task", payload.get("bundle", {}).get("component_id")) != unit.component:
        raise ValueError("Development admission evidence names another component")
    model = payload.get("model", {})
    adapter = model.get("adapter", {})
    checkpoint_sha = sha256_file(unit.checkpoint)
    if not checkpoint_sha or adapter.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Development admission evidence names another checkpoint")
    if model.get("training_seed") != unit.training_seed:
        raise ValueError("Development admission evidence names another training seed")
    decision = reference_decision_for_result(unit.component, payload, split="development")
    receipt = {
        "development_result": {"path": str(source), "sha256": identity["sha256"]},
        "checkpoint_sha256": checkpoint_sha,
        "decision": decision,
    }
    return replace(
        unit,
        admission_cleared=unit.admission_cleared and decision["passed"] is True,
        admission_evidence=str(source),
        reference_admission=receipt,
    )


def benchmark_identity(benchmark_root: Path) -> dict[str, Any]:
    """Pin the identity of the data package the report was produced against."""

    try:
        version_json = json.loads((benchmark_root / "VERSION.json").read_text())
        manifest_sha256 = (benchmark_root / "manifest.sha256").read_text()
    except (OSError, ValueError) as error:
        raise ValueError(
            f"benchmark root {benchmark_root} is missing readable "
            f"VERSION.json/manifest.sha256: {error}"
        ) from error
    task_registry_sha256 = sha256_file(benchmark_root / "task_registry.json")
    if task_registry_sha256 is None:
        raise ValueError(
            f"benchmark root {benchmark_root} is missing task_registry.json"
        )
    return {
        "version_json": version_json,
        "manifest_sha256": manifest_sha256,
        "task_registry_sha256": task_registry_sha256,
    }


def build_command(unit: ReportUnit, benchmark_root: Path) -> list[str]:
    """Assemble the subprocess invocation for one unit.

    The generic entry evaluates the public Test split from the clean bundle.
    A declared ``component_cli`` switches to that component's frozen formal
    scorer, which reads its own pinned artifacts and takes no split or
    benchmark-root flags.
    """

    command = [sys.executable, "-m"]
    if unit.component_cli:
        command.append(
            COMPONENT_CLI_TEMPLATE.format(suffix=unit.component_cli)
        )
        command.append("eval")
    else:
        command.append(GENERIC_CLI_MODULE)
        command += ["--evaluation-split", "test"]
        command += ["--benchmark-root", str(benchmark_root)]
        command += ["--task", unit.component]
    command += [
        "--adapter", unit.adapter,
        "--checkpoint", str(unit.checkpoint),
        "--model-name", unit.resolved_model_name(),
    ]
    if unit.training_recipe is not None:
        command += ["--training-recipe", unit.training_recipe]
    if unit.training_seed is not None:
        command += ["--training-seed", str(unit.training_seed)]
    if unit.device is not None:
        command += ["--device", unit.device]
    if unit.batch_size is not None:
        command += ["--batch-size", str(unit.batch_size)]
    if unit.stablewm_repo is not None:
        command += ["--stablewm-repo", str(unit.stablewm_repo)]
    if unit.stablewm_ref is not None:
        command += ["--stablewm-ref", unit.stablewm_ref]
    command += ["--output", str(unit.output)]
    return command


def recorded_checkpoint_sha256(output: Path) -> str | None:
    """The checkpoint ``sha256`` an existing evaluation output records."""

    try:
        payload = json.loads(output.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    candidates = [payload.get("checkpoint_sha256")]
    for section in ("model", "provenance", "evaluation"):
        nested = payload.get(section)
        if isinstance(nested, dict):
            candidates.append(nested.get("checkpoint_sha256"))
    for value in candidates:
        if isinstance(value, str) and len(value) == 64:
            return value
    return None


@dataclass(frozen=True)
class PlanDecision:
    """What ``run`` would do with one unit, without evaluating anything."""

    unit: ReportUnit
    will_run: bool
    status: str | None
    reason: str
    command: list[str]


def plan_unit(unit: ReportUnit, benchmark_root: Path, *, force: bool) -> PlanDecision:
    if not unit.admission_cleared:
        reason = _ADMISSION_GATE_REASON
        if unit.admission_evidence:
            reason += f" (evidence: {unit.admission_evidence})"
        return PlanDecision(unit, False, STATUS_SKIPPED_NOT_ADMITTED, reason,
                            build_command(unit, benchmark_root))
    checkpoint_sha = sha256_file(unit.checkpoint)
    if checkpoint_sha is None:
        return PlanDecision(
            unit, False, STATUS_FAILED,
            f"checkpoint not found: {unit.checkpoint}",
            build_command(unit, benchmark_root),
        )
    if not force and unit.output.exists():
        recorded = recorded_checkpoint_sha256(unit.output)
        if recorded == checkpoint_sha:
            return PlanDecision(
                unit, False, STATUS_SKIPPED_ALREADY_CURRENT,
                f"output already exists and records checkpoint sha256 "
                f"{checkpoint_sha}",
                build_command(unit, benchmark_root),
            )
    return PlanDecision(unit, True, None, "admitted; no current output found",
                        build_command(unit, benchmark_root))


def _error_summary(result: subprocess.CompletedProcess) -> str:
    parts = [part.strip() for part in (result.stderr, result.stdout) if part]
    combined = "\n".join(parts)[-_ERROR_SUMMARY_LIMIT:]
    return combined or "<no output captured>"


def _base_entry(unit: ReportUnit) -> dict[str, Any]:
    return {
        "component": unit.component,
        "family": unit.family,
        "training_seed": unit.training_seed,
        "status": None,
        "output": str(unit.output),
        "output_sha256": None,
        "checkpoint_sha256_before": None,
        "checkpoint_sha256_after": None,
        "command": None,
        "returncode": None,
        "message": None,
        "reference_admission": unit.reference_admission,
    }


def _current_test_decision(unit: ReportUnit) -> dict[str, Any] | None:
    if unit.reference_admission is None:
        return None
    from contextworld.benchmarks.reference_decision import reference_decision_for_result

    envelope = json.loads(unit.output.read_text())
    return reference_decision_for_result(
        unit.component, envelope.get("result", envelope), split="test"
    )


def run_report(
    manifest_path: Path,
    receipt_path: Path,
    *,
    force: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Execute every admissible unit and write the receipt."""

    manifest = load_report_manifest(manifest_path)
    benchmark_root = Path(manifest["benchmark_root"])
    identity = benchmark_identity(benchmark_root)
    units = parse_units(manifest)

    entries: list[dict[str, Any]] = []
    totals = {
        "planned": len(units),
        "completed": 0,
        "skipped_not_admitted": 0,
        "skipped_already_current": 0,
        "failed": 0,
    }

    for position, unit in enumerate(units, start=1):
        entry = _base_entry(unit)
        entries.append(entry)
        decision = plan_unit(unit, benchmark_root, force=force)
        entry["command"] = decision.command

        if decision.status == STATUS_SKIPPED_NOT_ADMITTED:
            entry["status"] = STATUS_SKIPPED_NOT_ADMITTED
            entry["message"] = decision.reason
            totals[STATUS_SKIPPED_NOT_ADMITTED] += 1
            print(f"[{position}/{len(units)}] {unit.component}: "
                  f"{STATUS_SKIPPED_NOT_ADMITTED}")
            continue

        checkpoint_before = sha256_file(unit.checkpoint)
        if checkpoint_before is None:
            entry["status"] = STATUS_FAILED
            entry["message"] = decision.reason
            totals[STATUS_FAILED] += 1
            print(f"[{position}/{len(units)}] {unit.component}: failed")
            if fail_fast:
                print("fail-fast requested; stopping the batch")
                break
            continue

        if decision.status == STATUS_SKIPPED_ALREADY_CURRENT:
            output_sha = sha256_file(unit.output)
            entry.update(
                status=STATUS_SKIPPED_ALREADY_CURRENT,
                output_sha256=output_sha,
                checkpoint_sha256_before=checkpoint_before,
                checkpoint_sha256_after=checkpoint_before,
                message=decision.reason,
            )
            entry["reference_decision"] = _current_test_decision(unit)
            totals[STATUS_SKIPPED_ALREADY_CURRENT] += 1
            print(f"[{position}/{len(units)}] {unit.component}: "
                  f"{STATUS_SKIPPED_ALREADY_CURRENT}")
            continue

        unit.output.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(decision.command, capture_output=True, text=True)
        checkpoint_after = sha256_file(unit.checkpoint)
        output_sha = sha256_file(unit.output) if unit.output.exists() else None
        entry.update(
            checkpoint_sha256_before=checkpoint_before,
            checkpoint_sha256_after=checkpoint_after,
            output_sha256=output_sha,
            returncode=result.returncode,
        )

        if checkpoint_after != checkpoint_before:
            entry["status"] = STATUS_FAILED
            entry["message"] = (
                "checkpoint changed during evaluation "
                f"(before {checkpoint_before}, after {checkpoint_after})"
            )
            totals[STATUS_FAILED] += 1
            print(f"[{position}/{len(units)}] {unit.component}: failed")
            if fail_fast:
                print("fail-fast requested; stopping the batch")
                break
            continue

        if result.returncode == 0 and output_sha is not None:
            entry["reference_decision"] = _current_test_decision(unit)
            entry["status"] = STATUS_COMPLETED
            entry["message"] = (
                "evaluation finished; checkpoint unchanged after evaluation"
            )
            totals[STATUS_COMPLETED] += 1
            print(f"[{position}/{len(units)}] {unit.component}: completed")
            continue

        entry["status"] = STATUS_FAILED
        entry["message"] = (
            f"returncode={result.returncode}; {_error_summary(result)}"
            if output_sha is not None
            else f"returncode={result.returncode}; "
                 f"no output written; {_error_summary(result)}"
        )
        totals[STATUS_FAILED] += 1
        print(f"[{position}/{len(units)}] {unit.component}: failed")
        if fail_fast:
            print("fail-fast requested; stopping the batch")
            break

    receipt = {
        "schema_version": manifest["schema_version"],
        "report_id": manifest["report_id"],
        "created_utc": dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "benchmark_identity": identity,
        "totals": totals,
        "units": entries,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt_path, receipt)
    print(
        f"receipt: {receipt_path} "
        f"(completed={totals['completed']}, "
        f"skipped_not_admitted={totals[STATUS_SKIPPED_NOT_ADMITTED]}, "
        f"skipped_already_current={totals[STATUS_SKIPPED_ALREADY_CURRENT]}, "
        f"failed={totals['failed']})"
    )
    return receipt


def verify_receipt(receipt_path: Path) -> int:
    """Re-check every recorded output against the receipt; report differences."""

    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as error:
        print(f"verify: cannot read receipt {receipt_path}: {error}",
              file=sys.stderr)
        return 2

    mismatched = 0
    for entry in receipt.get("units", []):
        component = entry.get("component", "<unknown>")
        admission = entry.get("reference_admission")
        if admission:
            source = admission["development_result"]
            if sha256_file(Path(source["path"])) != source["sha256"]:
                print(f"{component}: Development admission evidence changed")
                mismatched += 1
        expected = entry.get("output_sha256")
        output = entry.get("output")
        if not expected or not output:
            print(f"{component}: no output recorded "
                  f"(status={entry.get('status')}); nothing to verify")
            continue
        output_path = Path(output)
        actual = sha256_file(output_path)
        if actual is None:
            print(f"{component}: MISSING {output}")
            mismatched += 1
        elif actual != expected:
            print(f"{component}: SHA256 MISMATCH for {output} "
                  f"(receipt {expected}, actual {actual})")
            mismatched += 1
        else:
            print(f"{component}: ok")

    if mismatched:
        print(f"verify: {mismatched} output(s) missing or modified")
        return 1
    print("verify: all recorded outputs match the receipt")
    return 0


def _default_receipt_path(output_root: Path, report_id: str) -> Path:
    return output_root / f"public_test_report_{report_id}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextworld-public-test-report",
        description=(
            "Plan, execute, and verify the one-shot Public Test final report "
            "from a frozen checkpoint checklist."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="list the units a report would execute and why; runs nothing",
    )
    plan_parser.add_argument("--manifest", type=Path, required=True,
                             help="frozen checkpoint checklist (JSON)")
    plan_parser.add_argument("--json", action="store_true",
                             help="print the plan as JSON")

    run_parser = subparsers.add_parser(
        "run", help="execute the Public Test evaluations and write a receipt")
    run_parser.add_argument("--manifest", type=Path, required=True,
                            help="frozen checkpoint checklist (JSON)")
    run_parser.add_argument("--receipt", type=Path,
                            help="receipt path (default: "
                                 "<output-root>/public_test_report_<id>.json)")
    run_parser.add_argument("--output-root", type=Path,
                            help="default receipt directory "
                                 "(default: the manifest's directory)")
    run_parser.add_argument("--force", action="store_true",
                            help="re-evaluate units whose output is current")
    run_parser.add_argument("--fail-fast", action="store_true",
                            help="stop the batch at the first failed unit")

    verify_parser = subparsers.add_parser(
        "verify", help="re-check recorded outputs against an existing receipt")
    verify_parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "plan":
        manifest_path: Path = args.manifest
        try:
            manifest = load_report_manifest(manifest_path)
            identity = benchmark_identity(Path(manifest["benchmark_root"]))
            units = parse_units(manifest)
        except ValueError as error:
            print(f"plan: {error}", file=sys.stderr)
            return 2
        benchmark_root = Path(manifest["benchmark_root"])
        decisions = [plan_unit(unit, benchmark_root, force=False)
                     for unit in units]
        if args.json:
            print(json.dumps({
                "report_id": manifest["report_id"],
                "benchmark_identity": identity,
                "units": [
                    {
                        "component": decision.unit.component,
                        "family": decision.unit.family,
                        "training_seed": decision.unit.training_seed,
                        "will_run": decision.will_run,
                        "status": decision.status,
                        "reason": decision.reason,
                        "command": decision.command,
                    }
                    for decision in decisions
                ],
            }, indent=2))
        else:
            print(f"report_id: {manifest['report_id']}")
            for position, decision in enumerate(decisions, start=1):
                outcome = "run" if decision.will_run else f"skip ({decision.status})"
                print(f"[{position}/{len(decisions)}] "
                      f"{decision.unit.component}/"
                      f"{decision.unit.family}/"
                      f"s{decision.unit.training_seed}: {outcome} — "
                      f"{decision.reason}")
        return 0

    if args.command == "run":
        manifest_path = args.manifest
        try:
            manifest = load_report_manifest(manifest_path)
            report_id = manifest["report_id"]
        except ValueError as error:
            print(f"run: {error}", file=sys.stderr)
            return 2
        output_root = args.output_root or manifest_path.parent
        receipt_path = args.receipt or _default_receipt_path(
            output_root, report_id)
        try:
            receipt = run_report(
                manifest_path, receipt_path,
                force=args.force, fail_fast=args.fail_fast,
            )
        except ValueError as error:
            print(f"run: {error}", file=sys.stderr)
            return 2
        return 1 if receipt["totals"]["failed"] else 0

    return verify_receipt(args.receipt)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
