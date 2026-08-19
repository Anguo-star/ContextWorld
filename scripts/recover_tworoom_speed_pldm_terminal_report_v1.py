#!/usr/bin/env python3
"""Recover missing Speed PLDM terminal reports without resuming training.

The fixed-budget recovery runs reached optimizer step 12,840 and wrote their
native full-state checkpoints, model-only checkpoints, configurations, and
loss traces.  Their original processes then failed while hashing a source
file from a temporary Stable-WorldModel checkout, before the terminal JSON
reports were written.  This tool revalidates the already-written artifacts,
repeats the exact model load check, and writes only the missing report.  It
does not instantiate a Trainer, execute an optimizer, or access evaluation
data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from contextworld.synthesis.stablewm import _git_commit, load_stable_worldmodel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = (
    ROOT
    / "configs/benchmark/"
    "tworoom_speed_pldm_terminal_report_recovery_v1.yaml"
)
RECOVERY_ID = "tworoom_speed_pldm_terminal_report_recovery_v1"
COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
MODEL_ID = "H3_Speed_PLDM_ReferenceCompletion"
EXPECTED_SEEDS = (3072, 4096, 5120)
INITIAL_STEP = 10272
FINAL_STEP = 12840


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _repo_path(value: str | Path, *, label: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the repository") from error
    return resolved


def _logical(path: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the repository") from error


def _identity(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return {
        "path": _logical(path, label=label),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _require_identity(value: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
    if not (
        isinstance(value, Mapping)
        and isinstance(value.get("path"), str)
        and isinstance(value.get("sha256"), str)
        and isinstance(value.get("size_bytes"), int)
    ):
        raise ValueError(f"{label} must declare path, sha256, and size_bytes")
    path = _repo_path(str(value["path"]), label=label)
    observed = _identity(path, label=label)
    if observed != dict(value):
        raise RuntimeError(f"{label} identity changed: expected={value}, observed={observed}")
    return path, observed


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _entry(config: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    rows = config.get("runs")
    if not isinstance(rows, list) or tuple(int(row.get("seed", -1)) for row in rows) != EXPECTED_SEEDS:
        raise ValueError("Terminal-report recovery must declare all three seeds in order")
    matches = [row for row in rows if int(row["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"Missing terminal-report recovery entry for seed {seed}")
    return matches[0]


def _validate_config(path: Path, *, seed: int) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Any]]:
    config = _load_yaml(path)
    if not (
        config.get("schema_version") == 1
        and config.get("recovery_id") == RECOVERY_ID
        and config.get("completion_id") == COMPLETION_ID
        and config.get("status")
        == "preregistered_after_terminal_report_failure_before_report_recovery"
        and config.get("chronology")
        == {
            "fixed_budget_artifacts_present": True,
            "terminal_reports_present_at_registration": False,
            "completion_receipts_present_at_registration": False,
            "development_public_or_cem_executed_at_registration": False,
        }
        and config.get("scope")
        == {
            "training_or_optimizer_execution": False,
            "checkpoint_selection": False,
            "model_or_loss_trace_modification": False,
            "terminal_report_generation_only": True,
            "public_test_access": False,
            "evaluation_execution": False,
        }
    ):
        raise RuntimeError("Terminal-report recovery chronology or scope is invalid")
    frozen = config.get("frozen_inputs")
    if not isinstance(frozen, Mapping) or set(frozen) != {
        "completion_config",
        "interruption_recovery_preregistration",
        "training_script",
    }:
        raise ValueError("Terminal-report recovery frozen inputs are incomplete")
    frozen_identities = {
        name: _require_identity(specification, label=f"frozen_inputs.{name}")[1]
        for name, specification in frozen.items()
    }
    implementation = config.get("implementation")
    if not isinstance(implementation, Mapping) or set(implementation) != {"report_recovery"}:
        raise ValueError("Terminal-report recovery implementation is incomplete")
    implementation_path, implementation_identity = _require_identity(
        implementation["report_recovery"], label="implementation.report_recovery"
    )
    if implementation_path != Path(__file__).resolve():
        raise RuntimeError("Terminal-report recovery does not bind the active implementation")
    runtime = config.get("stable_worldmodel")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "worktree",
        "commit",
        "training_entry",
        "training_entry_sha256",
        "logger_entry",
        "logger_entry_expected_absent",
    }:
        raise ValueError("Terminal-report recovery Stable-WorldModel runtime is incomplete")
    stable_repo = Path(str(runtime["worktree"])).resolve()
    if _git_commit(stable_repo) != runtime["commit"]:
        raise RuntimeError("Terminal-report recovery Stable-WorldModel commit changed")
    training_entry = stable_repo / str(runtime["training_entry"])
    if not training_entry.is_file() or _sha256(training_entry) != runtime["training_entry_sha256"]:
        raise RuntimeError("Terminal-report recovery Stable-WorldModel training entry changed")
    logger_entry = stable_repo / str(runtime["logger_entry"])
    if runtime["logger_entry_expected_absent"] is not True or logger_entry.exists():
        raise RuntimeError("The missing offline-logger source boundary changed before report recovery")
    entry = _entry(config, seed)
    return config, entry, {
        "preregistration": _identity(path, label="terminal-report recovery preregistration"),
        "frozen_inputs": frozen_identities,
        "implementation": implementation_identity,
        "stable_repo": stable_repo,
    }


def _trace_audit(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    steps = [int(row["optimizer_step"]) for row in rows]
    if not rows or steps != sorted(set(steps)) or steps[-1] != FINAL_STEP:
        raise RuntimeError("Final loss trace is incomplete, duplicated, or out of order")
    if not any(step <= INITIAL_STEP for step in steps) or not any(step > INITIAL_STEP for step in steps):
        raise RuntimeError("Final loss trace does not span the interruption boundary")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "records": len(rows),
        "first_optimizer_step": steps[0],
        "last_optimizer_step": steps[-1],
        "passed": True,
    }


def _artifact_audit(entry: Mapping[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    import torch

    paths: dict[str, Path] = {}
    identities: dict[str, dict[str, Any]] = {}
    for name in (
        "final_checkpoint",
        "full_state_checkpoint",
        "checkpoint_config",
        "loss_trace",
        "preparation_receipt",
    ):
        paths[name], identities[name] = _require_identity(entry.get(name), label=f"seed {entry['seed']} {name}")
    full_state = torch.load(paths["full_state_checkpoint"], map_location="cpu", weights_only=False)
    required = {
        "state_dict",
        "optimizer_states",
        "lr_schedulers",
        "global_step",
        "epoch",
        "contextworld_rng_states_v1",
    }
    if not isinstance(full_state, dict) or not required.issubset(full_state):
        raise RuntimeError("Final native checkpoint lacks full trainer state")
    if not (
        int(full_state["global_step"]) == FINAL_STEP
        and int(full_state["epoch"]) == 5
        and full_state["optimizer_states"]
        and full_state["lr_schedulers"]
        and full_state["contextworld_rng_states_v1"]
    ):
        raise RuntimeError("Final native checkpoint does not prove the fixed optimizer budget")
    final_state = torch.load(paths["final_checkpoint"], map_location="cpu", weights_only=False)
    native_model_state = {
        key.removeprefix("model."): value
        for key, value in full_state["state_dict"].items()
        if key.startswith("model.")
    }
    tensor_equal = (
        isinstance(final_state, Mapping)
        and final_state.keys() == native_model_state.keys()
        and all(torch.equal(final_state[key], native_model_state[key]) for key in final_state)
    )
    if not tensor_equal:
        raise RuntimeError("Final model-only checkpoint differs from the final native trainer state")

    swm, stable_repo, commit = load_stable_worldmodel(
        ROOT,
        str(state["stable_repo"]),
        expected_ref=str(state["stable_commit"]),
    )
    reloaded = swm.wm.utils.load_pretrained(
        str(paths["final_checkpoint"]),
        cache_dir=str(paths["final_checkpoint"].parents[2]),
    )
    reloaded_state = reloaded.state_dict()
    reload_equal = (
        reloaded_state.keys() == final_state.keys()
        and all(torch.equal(reloaded_state[key].cpu(), final_state[key].cpu()) for key in final_state)
    )
    if not reload_equal:
        raise RuntimeError("Final model-only checkpoint does not exactly reload")
    trace = _trace_audit(paths["loss_trace"])
    preparation = _load_json(paths["preparation_receipt"])
    resume = preparation.get("resume_checkpoint")
    if not (
        preparation.get("passed") is True
        and preparation.get("seed") == int(entry["seed"])
        and isinstance(resume, Mapping)
        and resume.get("global_step") == INITIAL_STEP
        and resume.get("complete_epoch_boundary") is True
    ):
        raise RuntimeError("Preparation receipt does not prove the step-10,272 resume source")
    return {
        "paths": paths,
        "identities": identities,
        "final_state": final_state,
        "full_state": full_state,
        "trace": trace,
        "stable_repo": stable_repo,
        "stable_commit": commit,
        "save_load_exact": True,
        "native_model_state_exact": True,
    }


def build_report(preregistration: Path, *, seed: int) -> tuple[Path, dict[str, Any]]:
    config, entry, state = _validate_config(preregistration, seed=seed)
    state["stable_commit"] = config["stable_worldmodel"]["commit"]
    report_path = _repo_path(str(entry["report_output"]), label=f"seed {seed} report output")
    completion_path = _repo_path(
        str(entry["completion_receipt_output"]), label=f"seed {seed} completion receipt output"
    )
    if report_path.exists() or completion_path.exists():
        raise FileExistsError("Terminal report or completion receipt already exists")
    for value in config.get("evaluation_absence_paths", []):
        if _repo_path(str(value), label="evaluation absence path").exists():
            raise RuntimeError("Evaluation artifact exists before terminal-report recovery")
    audit = _artifact_audit(entry, state)
    paths = audit["paths"]
    runtime = config["stable_worldmodel"]
    full_state = audit["full_state"]
    report = {
        "schema_version": 1,
        "run_kind": "confirmation",
        "profile": "additive",
        "passed": True,
        "model_id": MODEL_ID,
        "run_name": str(entry["run_name"]),
        "stable_worldmodel": {
            "repo": str(audit["stable_repo"]),
            "commit": audit["stable_commit"],
            "training_entry": str(audit["stable_repo"] / runtime["training_entry"]),
            "training_entry_sha256": runtime["training_entry_sha256"],
            "logger_entry": str(audit["stable_repo"] / runtime["logger_entry"]),
            "logger_entry_sha256": None,
            "logger_entry_present": False,
        },
        "logger": {
            "backend": "none",
            "initialized": False,
            "terminal_report_recovery": True,
        },
        "model": {
            "class": "stable_worldmodel.wm.pldm.pldm.PLDM",
            "training_method": "pldm",
            "training_objective": "native_pldm",
            "parameters": sum(value.numel() for value in audit["final_state"].values()),
            "action_block": 5,
            "history_size": 3,
            "num_preds": 1,
        },
        "training": {
            "global_step": FINAL_STEP,
            "current_epoch": int(full_state["epoch"]),
            "max_epochs": 5,
            "initial_global_step": INITIAL_STEP,
            "initial_epoch": 4,
            "resumed_from_checkpoint": True,
            "restored_global_step": INITIAL_STEP,
            "restored_epoch": 4,
            "resume_policy": "required",
            "resume_weights_only": False,
            "training_complete": True,
            "expected_optimizer_steps": FINAL_STEP,
            "optimizer_states": len(full_state["optimizer_states"]),
            "lr_schedulers": len(full_state["lr_schedulers"]),
            "rng_state_rows": len(full_state["contextworld_rng_states_v1"]),
            "terminal_report_recovery_optimizer_steps": 0,
        },
        "artifacts": {
            "run_dir": str(paths["final_checkpoint"].parent),
            "pretrained": str(paths["final_checkpoint"]),
            "pretrained_sha256": _sha256(paths["final_checkpoint"]),
            "pretrained_config": str(paths["checkpoint_config"]),
            "pretrained_config_sha256": _sha256(paths["checkpoint_config"]),
            "full_state_checkpoint": audit["identities"]["full_state_checkpoint"],
            "loss_trace": audit["trace"],
        },
        "save_load_exact": audit["save_load_exact"],
        "terminal_report_recovery": {
            "recovery_id": RECOVERY_ID,
            "preregistration": state["preregistration"],
            "reason": "original_process_failed_after_fixed_budget_artifacts_before_terminal_report_write",
            "original_process_log_archived": False,
            "training_or_optimizer_execution": False,
            "model_or_loss_trace_modified": False,
            "native_model_state_exact": audit["native_model_state_exact"],
            "evaluation_executed": False,
            "public_test_accessed": False,
        },
    }
    return report_path, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "generate"))
    parser.add_argument("--seed", type=int, choices=EXPECTED_SEEDS, required=True)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output, report = build_report(args.preregistration.resolve(), seed=args.seed)
    if args.command == "generate":
        _write_json_exclusive(output, report)
    print(
        json.dumps(
            {
                "status": "validated_terminal_report_recovery_inputs"
                if args.command == "check"
                else "generated_terminal_report_recovery",
                "seed": args.seed,
                "output": _logical(output, label="terminal report output"),
                "optimizer_steps_executed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
