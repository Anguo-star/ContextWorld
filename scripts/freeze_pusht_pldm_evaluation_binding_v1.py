#!/usr/bin/env python3
"""Freeze and consume a Development-only PLDM evaluation binding.

The training completion runner deliberately stops after the fixed checkpoint
while evaluator sources are still moving.  This additive stage binds the
finished checkpoint to the final evaluator implementation and the already
registered Loader Validation data.  It has no Public-Test or CEM command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for value in (ROOT, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_pusht_pldm_reference_completion_v1 as completion_runner  # noqa: E402


PUBLIC_CLOSED = dict(completion_runner.PUBLIC_CLOSED)
EVALUATOR_MODULES = dict(completion_runner.EVALUATOR_MODULES)
EVALUATOR_RUNTIME_FILES = {
    "contact_friction": (
        "contextworld/benchmarks/adapters.py",
        "contextworld/benchmarks/contact_friction_icl_cli.py",
        "contextworld/benchmarks/contact_friction_icl_data.py",
        "contextworld/benchmarks/contact_friction_icl_score.py",
        "contextworld/synthesis/stablewm.py",
    ),
    "motion_damping": (
        "contextworld/benchmarks/adapters.py",
        "contextworld/benchmarks/motion_damping_icl_cli.py",
        "contextworld/benchmarks/motion_damping_icl_data.py",
        "contextworld/benchmarks/motion_damping_icl_score.py",
        "contextworld/synthesis/stablewm.py",
    ),
}


def _resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite additive receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _record(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    **details: Any,
) -> None:
    checks[name] = {"passed": bool(passed), **details}


def _public_receipt(release: dict[str, Any]) -> dict[str, Any]:
    """Describe Public by contract strings only, without touching its path."""

    evaluation = release.get("evaluation", {})
    development = evaluation.get("development", {})
    policy = development.get("public_test", {})
    if {key: policy.get(key) for key in PUBLIC_CLOSED} != PUBLIC_CLOSED:
        raise RuntimeError("Source release no longer keeps Public Test closed")
    table_name = evaluation.get("lance_table")
    if not isinstance(table_name, str) or not table_name:
        raise RuntimeError("Source release is missing the Public table name")
    return {
        **PUBLIC_CLOSED,
        "table_name_only": table_name,
        "path_resolved": False,
        "path_statted": False,
        "path_walked": False,
        "path_hashed": False,
        "path_decoded": False,
        "accessed_by_binding": False,
        "accessed_by_evaluator": False,
    }


def _checkpoint_receipt(
    completion: dict[str, Any], pilot_root: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    execution_path = pilot_root / "pilot_execution_receipt.json"
    decision_path = pilot_root / "development_decision.json"
    execution = _load_json(execution_path)
    decision = _load_json(decision_path)
    if execution.get("training_exit_code") != 0:
        raise RuntimeError("Training execution receipt is not successful")
    if decision.get("status") != "trained_pending_evaluation_binding":
        raise RuntimeError("Pilot is not awaiting an evaluator binding")
    if execution.get("public_test", {}).get("accessed_by_this_runner") is not False:
        raise RuntimeError("Training receipt does not prove Public stayed closed")
    if execution.get("cem") != {"executed": False, "authorized": False}:
        raise RuntimeError("Training receipt does not prove CEM stayed closed")
    checkpoint = decision.get("checkpoint", {})
    checkpoint_path = _resolve(checkpoint.get("path", ""))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Fixed-step checkpoint is missing: {checkpoint_path}")
    observed_sha256 = _sha256(checkpoint_path)
    expected_name = (
        f"{completion['training']['recipe']}_step"
        f"{completion['training']['fixed_optimizer_step']}.pt"
    )
    if checkpoint_path.name != expected_name:
        raise RuntimeError(
            f"Unexpected fixed-step checkpoint name: {checkpoint_path.name}"
        )
    if checkpoint.get("sha256") != observed_sha256:
        raise RuntimeError("Fixed-step checkpoint SHA differs from training decision")
    return execution, decision, checkpoint_path


def _runtime_receipts(component: str) -> list[dict[str, Any]]:
    receipts = []
    for relative_path in EVALUATOR_RUNTIME_FILES[component]:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Evaluator runtime file is missing: {path}")
        receipts.append(
            {
                "path": relative_path,
                "sha256": _sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    receipts.append(
        {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
            "size_bytes": int(Path(__file__).resolve().stat().st_size),
            "role": "binding_and_development_evaluation_runner",
        }
    )
    return receipts


def _evaluation_commands(
    *,
    component: str,
    completion: dict[str, Any],
    release_path: Path,
    checkpoint_path: Path,
    pilot_root: Path,
    device: str,
    batch_size: int,
) -> dict[str, list[str]]:
    seed = int(completion["training"]["pilot_seed"])
    stablewm = completion["stable_worldmodel"]
    evaluation_path = pilot_root / "development_evaluation_v1.json"
    rescore_path = pilot_root / "development_rescore_v1.json"
    return {
        "evaluate": [
            sys.executable,
            "-m",
            EVALUATOR_MODULES[component],
            "--release-config",
            str(release_path),
            "eval-development",
            "--checkpoint",
            str(checkpoint_path),
            "--adapter",
            "pldm",
            "--model-name",
            f"{component}_pldm_completion_seed{seed}",
            "--training-recipe",
            str(completion["training"]["recipe"]),
            "--training-seed",
            str(seed),
            "--device",
            str(device),
            "--batch-size",
            str(int(batch_size)),
            "--stablewm-repo",
            str(Path(stablewm["worktree"]).expanduser().resolve()),
            "--stablewm-ref",
            str(stablewm["commit"]),
            "--output",
            str(evaluation_path),
        ],
        "rescore": [
            sys.executable,
            "-m",
            EVALUATOR_MODULES[component],
            "--release-config",
            str(release_path),
            "score-development",
            "--input",
            str(evaluation_path),
            "--output",
            str(rescore_path),
        ],
    }


def build_binding(
    *,
    completion_path: Path | str,
    pilot_root: Path | str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    completion_path, completion = completion_runner.load_completion(completion_path)
    release_path, release = completion_runner.load_source_release(completion)
    component = completion["scope"]["component"]
    pilot_root = _resolve(pilot_root)
    registered_root = _resolve(completion["artifacts"]["pilot_root"])
    if pilot_root != registered_root:
        raise RuntimeError("Pilot root differs from the preregistered additive root")

    checks: dict[str, dict[str, Any]] = {}
    preflight = completion_runner.preflight_payload(completion_path)
    _record(
        checks,
        "fresh_training_and_data_preflight",
        preflight.get("passed") is True,
        completion_sha256=preflight.get("completion", {}).get("sha256"),
        source_release_sha256=preflight.get("source_release", {}).get("sha256"),
        failed_checks=[
            name
            for name, row in preflight.get("checks", {}).items()
            if not row.get("passed")
        ],
    )
    execution, decision, checkpoint_path = _checkpoint_receipt(
        completion, pilot_root
    )
    checkpoint_sha256 = _sha256(checkpoint_path)
    _record(
        checks,
        "fixed_checkpoint_identity",
        decision.get("fixed_recipe") == completion["training"]["recipe"]
        and int(decision.get("fixed_optimizer_step", -1))
        == int(completion["training"]["fixed_optimizer_step"])
        and int(decision.get("pilot_seed", -1))
        == int(completion["training"]["pilot_seed"]),
        path=str(checkpoint_path),
        sha256=checkpoint_sha256,
        recipe=decision.get("fixed_recipe"),
        optimizer_step=decision.get("fixed_optimizer_step"),
        seed=decision.get("pilot_seed"),
    )

    expected_data = completion["source_release"]["data"]
    data_root = _resolve(expected_data["root"])
    manifest_path = data_root / "manifest.json"
    development_path = data_root / expected_data["development"]["lance_table"]
    manifest_sha256 = _sha256(manifest_path)
    development_sha256 = _directory_sha256(development_path)
    _record(
        checks,
        "loader_validation_identity",
        manifest_sha256 == expected_data["manifest_sha256"]
        and development_sha256
        == expected_data["development"]["lance_table_sha256"],
        manifest={"path": str(manifest_path), "sha256": manifest_sha256},
        loader_validation={
            "path": str(development_path),
            "sha256": development_sha256,
            "pair_count": int(expected_data["development"]["pair_count"]),
        },
    )
    public_test = _public_receipt(release)
    _record(
        checks,
        "public_test_closed_without_path_access",
        all(public_test[key] == value for key, value in PUBLIC_CLOSED.items())
        and not any(
            public_test[key]
            for key in (
                "path_resolved",
                "path_statted",
                "path_walked",
                "path_hashed",
                "path_decoded",
                "accessed_by_binding",
                "accessed_by_evaluator",
            )
        ),
        receipt=public_test,
    )
    runtime = _runtime_receipts(component)
    _record(
        checks,
        "evaluator_runtime_frozen",
        all(len(row["sha256"]) == 64 and row["size_bytes"] > 0 for row in runtime),
        files=runtime,
    )
    release_sha256 = _sha256(release_path)
    _record(
        checks,
        "current_release_full_identity",
        release.get("release_id") == completion["source_release"]["release_id"]
        and release_sha256
        == completion["source_release"][
            "release_config_sha256_observed_at_preregistration"
        ],
        path=str(release_path),
        release_id=release.get("release_id"),
        sha256=release_sha256,
    )
    commands = _evaluation_commands(
        component=component,
        completion=completion,
        release_path=release_path,
        checkpoint_path=checkpoint_path,
        pilot_root=pilot_root,
        device=device,
        batch_size=batch_size,
    )
    passed = all(row["passed"] for row in checks.values())
    return {
        "schema_version": 1,
        "binding_id": f"{completion['completion_id']}_evaluation_binding_v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "scope": "Development_only; Public Test and CEM remain closed",
        "component": component,
        "completion": {
            "path": str(completion_path),
            "sha256": _sha256(completion_path),
            "id": completion["completion_id"],
        },
        "source_release": {
            "path": str(release_path),
            "sha256": release_sha256,
            "release_id": release.get("release_id"),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "recipe": completion["training"]["recipe"],
            "optimizer_step": int(completion["training"]["fixed_optimizer_step"]),
            "seed": int(completion["training"]["pilot_seed"]),
        },
        "development_data": {
            "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
            "loader_validation": {
                "path": str(development_path),
                "sha256": development_sha256,
                "pair_count": int(expected_data["development"]["pair_count"]),
            },
        },
        "evaluator_runtime": runtime,
        "stable_worldmodel": completion["stable_worldmodel"],
        "commands": commands,
        "training_receipts": {
            "execution": str(pilot_root / "pilot_execution_receipt.json"),
            "pending_evaluation_decision": str(
                pilot_root / "development_decision.json"
            ),
            "training_exit_code": execution.get("training_exit_code"),
        },
        "public_test": public_test,
        "cem": {"executed": False, "authorized": False},
        "checks": checks,
    }


def validate_binding(
    binding_path: Path | str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    binding_path = _resolve(binding_path)
    binding = _load_json(binding_path)
    if binding.get("status") != "passed" or binding.get("passed") is not True:
        raise RuntimeError("Development evaluation requires a passed binding")
    completion_path, completion = completion_runner.load_completion(
        binding["completion"]["path"]
    )
    if _sha256(completion_path) != binding["completion"]["sha256"]:
        raise RuntimeError("Completion config changed after evaluator binding")
    release_path, release = completion_runner.load_source_release(completion)
    if _sha256(release_path) != binding["source_release"]["sha256"]:
        raise RuntimeError("Source release changed after evaluator binding")
    checkpoint_path = _resolve(binding["checkpoint"]["path"])
    if _sha256(checkpoint_path) != binding["checkpoint"]["sha256"]:
        raise RuntimeError("Checkpoint changed after evaluator binding")
    for receipt in binding["evaluator_runtime"]:
        path = _resolve(receipt["path"])
        if _sha256(path) != receipt["sha256"]:
            raise RuntimeError(f"Evaluator runtime changed after binding: {path}")
    manifest = binding["development_data"]["manifest"]
    if _sha256(_resolve(manifest["path"])) != manifest["sha256"]:
        raise RuntimeError("Development manifest changed after binding")
    development = binding["development_data"]["loader_validation"]
    if _directory_sha256(_resolve(development["path"])) != development["sha256"]:
        raise RuntimeError("Loader Validation table changed after binding")
    if _public_receipt(release) != binding["public_test"]:
        raise RuntimeError("Public closed-state contract changed after binding")
    completion_runner._pinned_stable_worldmodel(completion)
    return binding, completion, binding_path


def _run(command: list[str], log: Path) -> int:
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", "/tmp/contextworld-pldm-eval-mpl")
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
    return int(process.returncode)


def evaluate_binding(binding_path: Path | str) -> dict[str, Any]:
    binding, completion, binding_path = validate_binding(binding_path)
    pilot_root = binding_path.parent
    evaluation_path = pilot_root / "development_evaluation_v1.json"
    rescore_path = pilot_root / "development_rescore_v1.json"
    decision_path = pilot_root / "development_evaluation_decision_v1.json"
    for target in (evaluation_path, rescore_path, decision_path):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation artifact: {target}")
    evaluation_exit = _run(
        list(binding["commands"]["evaluate"]),
        pilot_root / "development_evaluation_v1.log",
    )
    if evaluation_exit:
        failure = {
            "schema_version": 1,
            "status": "development_evaluation_execution_failed",
            "binding": {"path": str(binding_path), "sha256": _sha256(binding_path)},
            "evaluation_exit_code": evaluation_exit,
            "public_test": binding["public_test"],
            "cem": {"executed": False, "authorized": False},
        }
        _write_json(decision_path, failure)
        raise RuntimeError("Development evaluator execution failed")
    rescore_exit = _run(
        list(binding["commands"]["rescore"]),
        pilot_root / "development_rescore_v1.log",
    )
    if rescore_exit:
        failure = {
            "schema_version": 1,
            "status": "development_rescore_execution_failed",
            "binding": {"path": str(binding_path), "sha256": _sha256(binding_path)},
            "rescore_exit_code": rescore_exit,
            "public_test": binding["public_test"],
            "cem": {"executed": False, "authorized": False},
        }
        _write_json(decision_path, failure)
        raise RuntimeError("Development independent rescore failed")

    evaluation = _load_json(evaluation_path)
    rescore = _load_json(rescore_path)
    expected_public = {
        key: binding["public_test"][key] for key in PUBLIC_CLOSED
    }
    if evaluation.get("public_test") != expected_public:
        raise RuntimeError("Development result does not prove Public stayed closed")
    checkpoint = evaluation.get("model", {}).get("checkpoint", {})
    adapter = evaluation.get("model", {}).get("adapter", {})
    stablewm_commit = completion["stable_worldmodel"]["commit"]
    identities_match = bool(
        checkpoint.get("sha256") == binding["checkpoint"]["sha256"]
        and adapter.get("checkpoint_sha256") == binding["checkpoint"]["sha256"]
        and adapter.get("stable_worldmodel_commit") == stablewm_commit
        and evaluation.get("contract", {}).get("release_config_sha256")
        == binding["source_release"]["sha256"]
        and evaluation.get("contract", {}).get("development_data_manifest_sha256")
        == binding["development_data"]["manifest"]["sha256"]
        and evaluation.get("contract", {}).get("development_lance_table_sha256")
        == binding["development_data"]["loader_validation"]["sha256"]
    )
    rescore_matches = bool(
        evaluation.get("metrics") == rescore.get("metrics")
        and evaluation.get("gate") == rescore.get("gate")
        and evaluation.get("records") == rescore.get("records")
    )
    passed = bool(
        identities_match
        and rescore_matches
        and evaluation.get("gate", {}).get("passed") is True
    )
    status = (
        "passed_development_recipe_frozen"
        if passed
        else "failed_development_current_protocol"
    )
    decision = {
        "schema_version": 1,
        "status": status,
        "binding": {"path": str(binding_path), "sha256": _sha256(binding_path)},
        "checkpoint": binding["checkpoint"],
        "development": {
            "evaluation": {"path": str(evaluation_path), "sha256": _sha256(evaluation_path)},
            "independent_rescore": {"path": str(rescore_path), "sha256": _sha256(rescore_path)},
            "identities_match_binding": identities_match,
            "independent_rescore_matches": rescore_matches,
            "metrics": evaluation.get("metrics"),
            "gate": evaluation.get("gate"),
        },
        "public_test": binding["public_test"],
        "cem": {"executed": False, "authorized": False},
        "next_stage": (
            {
                "recipe_frozen": True,
                "remaining_development_seeds_authorized": list(
                    completion["training"]["seeds"][1:]
                ),
                "public_test_authorized": False,
                "cem_authorized": False,
            }
            if passed
            else {
                "formal_failure_frozen": True,
                "remaining_development_seeds_authorized": False,
                "public_test_authorized": False,
                "cem_authorized": False,
            }
        ),
    }
    _write_json(decision_path, decision)
    if passed:
        _write_json(pilot_root / "recipe_freeze_v1.json", decision)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bind = commands.add_parser("bind")
    bind.add_argument("--completion-config", type=Path, required=True)
    bind.add_argument("--pilot-root", type=Path, required=True)
    bind.add_argument("--device", required=True)
    bind.add_argument("--batch-size", type=int, default=64)
    bind.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate-development")
    evaluate.add_argument("--binding", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "bind":
        payload = build_binding(
            completion_path=args.completion_config,
            pilot_root=args.pilot_root,
            device=args.device,
            batch_size=args.batch_size,
        )
        output = _resolve(args.output)
        _write_json(output, payload)
        print(json.dumps({"status": payload["status"], "output": str(output)}))
        if not payload["passed"]:
            raise SystemExit(1)
    else:
        payload = evaluate_binding(args.binding)
        print(json.dumps({"status": payload["status"]}))


if __name__ == "__main__":
    main()
