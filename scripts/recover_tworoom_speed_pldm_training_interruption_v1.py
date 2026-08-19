#!/usr/bin/env python3
"""Recover the interrupted fixed-budget Speed PLDM training runs.

The original jobs stopped after their fourth complete epoch.  Their native
Lightning checkpoints are therefore safe to resume, but each loss trace also
contains a short, uncommitted fifth-epoch tail.  This tool preserves that
entire trace byte-for-byte, restores the canonical trace to the checkpointed
prefix, and launches exactly one ``required`` full-state resume.

This is an execution-recovery tool.  It does not select a checkpoint, change
the training recipe, read Public Test payloads, or authorize evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ID = "tworoom_speed_pldm_training_interruption_recovery_v1"
COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
DEFAULT_PREREGISTRATION = (
    ROOT
    / "configs/benchmark/"
    "tworoom_speed_pldm_training_interruption_recovery_v1.yaml"
)
EXPECTED_SEEDS = (3072, 4096, 5120)
CHECKPOINT_STEP = 10272
FINAL_STEP = 12840
OPTIMIZER_STEPS_PER_EPOCH = 2568


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _repository_path(value: str | Path, *, label: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (ROOT / candidate).resolve()
    )
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


def _validate_repository_specification(
    specification: Any, *, label: str
) -> dict[str, Any]:
    if not (
        isinstance(specification, dict)
        and isinstance(specification.get("path"), str)
        and isinstance(specification.get("sha256"), str)
        and isinstance(specification.get("size_bytes"), int)
    ):
        raise ValueError(f"{label} needs path, sha256, and size_bytes")
    path = _repository_path(specification["path"], label=label)
    observed = _identity(path, label=label)
    expected = {
        "path": _logical(path, label=label),
        "sha256": specification["sha256"],
        "size_bytes": int(specification["size_bytes"]),
    }
    if observed != expected:
        raise RuntimeError(
            f"{label} identity changed: expected={expected}, observed={observed}"
        )
    return observed


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_bytes_exclusive(path, encoded)


def _replace_bytes_atomically(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.recovery.tmp")
    _write_bytes_exclusive(temporary, payload)
    try:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _trace_parts(payload: bytes) -> tuple[list[dict[str, Any]], bytes, bytes]:
    lines = payload.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise RuntimeError("Loss trace must be non-empty and newline terminated")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Invalid loss-trace JSON at line {index}") from error
        if not isinstance(row, dict) or not isinstance(
            row.get("optimizer_step"), int
        ):
            raise RuntimeError(f"Invalid optimizer step at trace line {index}")
        rows.append(row)
    steps = [int(row["optimizer_step"]) for row in rows]
    if steps != sorted(set(steps)):
        raise RuntimeError("Loss-trace steps are duplicate or out of order")
    prefix_lines = [
        line
        for line, row in zip(lines, rows, strict=True)
        if int(row["optimizer_step"]) <= CHECKPOINT_STEP
    ]
    tail_lines = [
        line
        for line, row in zip(lines, rows, strict=True)
        if int(row["optimizer_step"]) > CHECKPOINT_STEP
    ]
    return rows, b"".join(prefix_lines), b"".join(tail_lines)


def _bytes_identity(payload: bytes) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _entry_for_seed(preregistration: dict[str, Any], seed: int) -> dict[str, Any]:
    entries = preregistration.get("interrupted_runs")
    if not isinstance(entries, list):
        raise ValueError("Preregistration interrupted_runs must be a list")
    matches = [row for row in entries if row.get("seed") == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one interrupted run for seed {seed}")
    return matches[0]


def _validate_preregistration(
    path: Path, *, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    preregistration = _load_yaml(path)
    if not (
        preregistration.get("schema_version") == 1
        and preregistration.get("recovery_id") == RECOVERY_ID
        and preregistration.get("completion_id") == COMPLETION_ID
        and preregistration.get("status")
        == "preregistered_after_external_interruption_before_recovery"
        and preregistration.get("checkpoint_step") == CHECKPOINT_STEP
        and preregistration.get("final_optimizer_step") == FINAL_STEP
        and preregistration.get("optimizer_steps_per_epoch")
        == OPTIMIZER_STEPS_PER_EPOCH
    ):
        raise RuntimeError("Training-interruption recovery preregistration is invalid")
    scope = preregistration.get("scope")
    if scope != {
        "training_recipe_changed": False,
        "checkpoint_selected_by_metric": False,
        "public_test_accessed": False,
        "development_or_cem_executed": False,
        "full_state_required_resume_only": True,
        "uncommitted_trace_tail_preserved": True,
    }:
        raise RuntimeError("Recovery scope is not fail-closed")
    seeds = tuple(row.get("seed") for row in preregistration["interrupted_runs"])
    if seeds != EXPECTED_SEEDS or seed not in EXPECTED_SEEDS:
        raise RuntimeError(f"Unexpected recovery seed declaration: {seeds}")
    for label, specification in preregistration["frozen_inputs"].items():
        _validate_repository_specification(specification, label=label)
    implementation = preregistration.get("implementation")
    if not isinstance(implementation, dict):
        raise RuntimeError("Recovery implementation identities are missing")
    for label, specification in implementation.items():
        _validate_repository_specification(specification, label=label)
    runtime = preregistration.get("stable_worldmodel")
    if not isinstance(runtime, dict):
        raise RuntimeError("StableWM runtime declaration is missing")
    worktree = Path(runtime["worktree"]).resolve()
    from contextworld.synthesis.stablewm import _git_commit

    observed_ref = _git_commit(worktree)
    if observed_ref != runtime.get("commit"):
        raise RuntimeError(
            "StableWM commit changed: "
            f"expected={runtime.get('commit')}, observed={observed_ref}"
        )
    pldm_path = worktree / runtime["pldm_config"]
    if (
        not pldm_path.is_file()
        or _sha256(pldm_path) != runtime.get("pldm_config_sha256")
    ):
        raise RuntimeError("StableWM PLDM configuration identity changed")
    return preregistration, _entry_for_seed(preregistration, seed)


def _import_training_script(path: Path):
    specification = importlib.util.spec_from_file_location(
        "_speed_pldm_interruption_recovery_train", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot import training script: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _validate_checkpoint_pair(
    entry: dict[str, Any], *, training_module
) -> dict[str, Any]:
    last = _validate_repository_specification(
        entry.get("last_checkpoint"), label="last checkpoint"
    )
    state = _validate_repository_specification(
        entry.get("state_checkpoint"), label="state checkpoint"
    )
    if last["sha256"] != state["sha256"] or last["size_bytes"] != state["size_bytes"]:
        raise RuntimeError("last.ckpt and state.ckpt are not byte-identical")
    last_path = _repository_path(last["path"], label="last checkpoint")
    metadata = training_module._full_state_checkpoint_metadata(
        last_path,
        expected_optimizer_steps=FINAL_STEP,
        require_incomplete=True,
        expected_world_size=1,
        optimizer_steps_per_epoch=OPTIMIZER_STEPS_PER_EPOCH,
    )
    expected_metadata = {
        "global_step": CHECKPOINT_STEP,
        "epoch": 4,
        "optimizer_states": 1,
        "lr_schedulers": 1,
        "rng_ranks": [0],
        "complete_epoch_boundary": True,
        "optimizer_steps_per_epoch": OPTIMIZER_STEPS_PER_EPOCH,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Unsafe resume checkpoint metadata: {key}={metadata.get(key)!r}"
            )
    return metadata


def prepare_trace_recovery(entry: dict[str, Any]) -> dict[str, Any]:
    """Archive the full interrupted trace and restore its checkpointed prefix."""

    trace = _repository_path(entry["loss_trace"]["path"], label="loss trace")
    archive = _repository_path(entry["archive_path"], label="trace archive")
    original_expected = {
        "sha256": entry["loss_trace"]["sha256"],
        "size_bytes": int(entry["loss_trace"]["size_bytes"]),
    }
    prefix_expected = {
        "sha256": entry["canonical_prefix"]["sha256"],
        "size_bytes": int(entry["canonical_prefix"]["size_bytes"]),
    }
    current = trace.read_bytes()
    current_identity = _bytes_identity(current)

    if archive.exists():
        archived = archive.read_bytes()
        if _bytes_identity(archived) != original_expected:
            raise RuntimeError("Existing interrupted trace archive identity changed")
    elif current_identity == original_expected:
        _write_bytes_exclusive(archive, current)
        archived = current
    else:
        raise RuntimeError(
            "Loss trace is neither the preregistered interruption nor an "
            "already prepared canonical prefix"
        )

    archived_rows, prefix, tail = _trace_parts(archived)
    prefix_rows, _, prefix_tail = _trace_parts(prefix)
    tail_rows = [
        row for row in archived_rows if int(row["optimizer_step"]) > CHECKPOINT_STEP
    ]
    if prefix_tail:
        raise AssertionError("Internal prefix construction retained a tail")
    if (
        _bytes_identity(prefix) != prefix_expected
        or len(prefix_rows) != entry["canonical_prefix"]["rows"]
        or int(prefix_rows[-1]["optimizer_step"])
        != entry["canonical_prefix"]["last_optimizer_step"]
        or len(tail_rows) != entry["discarded_uncommitted_tail"]["rows"]
        or int(tail_rows[0]["optimizer_step"])
        != entry["discarded_uncommitted_tail"]["first_optimizer_step"]
        or int(tail_rows[-1]["optimizer_step"])
        != entry["discarded_uncommitted_tail"]["last_optimizer_step"]
        or not tail
    ):
        raise RuntimeError("Interrupted loss trace does not match preregistered rows")

    if current_identity == original_expected:
        _replace_bytes_atomically(trace, prefix)
    elif current_identity != prefix_expected:
        raise RuntimeError("Canonical loss trace changed after recovery preparation")
    if _bytes_identity(trace.read_bytes()) != prefix_expected:
        raise RuntimeError("Canonical loss-trace replacement did not persist")
    return {
        "original_trace": {
            "path": _logical(archive, label="trace archive"),
            **original_expected,
            "rows": len(archived_rows),
            "last_optimizer_step": int(archived_rows[-1]["optimizer_step"]),
        },
        "canonical_trace": {
            "path": _logical(trace, label="loss trace"),
            **prefix_expected,
            "rows": len(prefix_rows),
            "last_optimizer_step": int(prefix_rows[-1]["optimizer_step"]),
        },
        "excluded_uncommitted_tail": {
            "rows": len(tail_rows),
            "first_optimizer_step": int(tail_rows[0]["optimizer_step"]),
            "last_optimizer_step": int(tail_rows[-1]["optimizer_step"]),
            "preserved_in_original_trace": True,
        },
    }


def _forbidden_post_training_paths(entry: dict[str, Any]) -> list[Path]:
    return [
        _repository_path(value, label="post-training output")
        for value in entry.get("must_not_exist_before_resume", [])
    ]


def _preparation_receipt(
    preregistration_path: Path,
    entry: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    trace_recovery: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "completion_id": COMPLETION_ID,
        "seed": int(entry["seed"]),
        "status": "prepared_required_full_state_resume",
        "passed": True,
        "preregistration": _identity(
            preregistration_path, label="recovery preregistration"
        ),
        "scope": {
            "training_recipe_changed": False,
            "checkpoint_selected_by_metric": False,
            "public_test_accessed": False,
            "development_or_cem_executed": False,
        },
        "interruption": {
            "exact_exit_cause_available": False,
            "original_traceback_or_process_log_available": False,
            "causal_attribution_claimed": False,
        },
        "resume_checkpoint": checkpoint_metadata,
        "trace_recovery": trace_recovery,
        "next_action": {
            "resume_policy": "required",
            "initial_global_step": CHECKPOINT_STEP,
            "fixed_final_optimizer_step": FINAL_STEP,
            "remaining_optimizer_steps": FINAL_STEP - CHECKPOINT_STEP,
            "automatic_retry_after_failure": False,
        },
    }


def prepare(
    *, preregistration_path: Path, seed: int
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    preregistration, entry = _validate_preregistration(
        preregistration_path, seed=seed
    )
    training_path = _repository_path(
        preregistration["implementation"]["training_script"]["path"],
        label="training script",
    )
    training_module = _import_training_script(training_path)
    forbidden = [path for path in _forbidden_post_training_paths(entry) if path.exists()]
    if forbidden:
        raise RuntimeError(
            "Post-training or evaluation artifacts already exist before recovery: "
            + ", ".join(str(path) for path in forbidden)
        )
    checkpoint_metadata = _validate_checkpoint_pair(
        entry, training_module=training_module
    )
    trace_recovery = prepare_trace_recovery(entry)
    receipt_path = _repository_path(
        entry["preparation_receipt"], label="preparation receipt"
    )
    receipt = _preparation_receipt(
        preregistration_path, entry, checkpoint_metadata, trace_recovery
    )
    if receipt_path.exists():
        if _load_json(receipt_path) != receipt:
            raise RuntimeError("Existing preparation receipt does not match recovery")
    else:
        _write_json_exclusive(receipt_path, receipt)
    return preregistration, entry, training_module


def _training_arguments(
    preregistration: dict[str, Any], entry: dict[str, Any], training_module
) -> list[str]:
    completion = preregistration["training_invocation"]
    return [
        str(training_module.__file__),
        "--model-id",
        completion["model_id"],
        "--benchmark-config",
        str(
            _repository_path(
                preregistration["frozen_inputs"]["completion_config"]["path"],
                label="completion config",
            )
        ),
        "--run-name",
        entry["run_name"],
        "--profile",
        "additive",
        "--run-kind",
        "confirmation",
        "--resume-policy",
        "required",
        "--seed",
        str(entry["seed"]),
        "--data-split-seed",
        str(completion["data_split_seed"]),
        "--stablewm-repo",
        preregistration["stable_worldmodel"]["worktree"],
        "--stablewm-ref",
        preregistration["stable_worldmodel"]["commit"],
        "--original-h5",
        completion["original_h5"],
        "--output-root",
        str(_repository_path(completion["output_root"], label="training output root")),
        "--report",
        str(_repository_path(entry["training_report"], label="training report")),
        "--devices",
        "1",
        "--batch-size",
        "128",
        "--accumulate-grad-batches",
        "8",
        "--num-workers",
        "6",
        "--logger-backend",
        "none",
        "--initialization-checkpoint",
        str(
            _repository_path(
                preregistration["frozen_inputs"]["initialization_checkpoint"]["path"],
                label="initialization checkpoint",
            )
        ),
        "--initialization-checkpoint-sha256",
        preregistration["frozen_inputs"]["initialization_checkpoint"]["sha256"],
    ]


def _finalize(entry: dict[str, Any]) -> dict[str, Any]:
    report_path = _repository_path(entry["training_report"], label="training report")
    final_path = _repository_path(entry["final_checkpoint"], label="final checkpoint")
    trace_path = _repository_path(entry["loss_trace"]["path"], label="loss trace")
    report = _load_json(report_path)
    training = report.get("training")
    if not (
        report.get("passed") is True
        and report.get("run_name") == entry["run_name"]
        and isinstance(training, dict)
        and training.get("global_step") == FINAL_STEP
        and training.get("initial_global_step") == CHECKPOINT_STEP
        and training.get("restored_global_step") == CHECKPOINT_STEP
        and training.get("resumed_from_checkpoint") is True
        and training.get("resume_policy") == "required"
        and training.get("training_complete") is True
        and training.get("expected_optimizer_steps") == FINAL_STEP
    ):
        raise RuntimeError("Completed training report does not prove required resume")
    rows, _, _uncommitted_tail_partition = _trace_parts(trace_path.read_bytes())
    steps = [int(row["optimizer_step"]) for row in rows]
    if steps[-1] != FINAL_STEP or steps != sorted(set(steps)):
        raise RuntimeError("Final loss trace is incomplete or inconsistent")
    receipt = {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "completion_id": COMPLETION_ID,
        "seed": int(entry["seed"]),
        "status": "completed_fixed_budget_required_resume",
        "passed": True,
        "training_report": _identity(report_path, label="training report"),
        "final_checkpoint": _identity(final_path, label="final checkpoint"),
        "final_loss_trace": {
            **_identity(trace_path, label="loss trace"),
            "rows": len(rows),
            "first_optimizer_step": steps[0],
            "last_optimizer_step": steps[-1],
        },
        "resume_proof": {
            "initial_global_step": CHECKPOINT_STEP,
            "final_global_step": FINAL_STEP,
            "remaining_optimizer_steps_executed": FINAL_STEP - CHECKPOINT_STEP,
            "full_state_resume": True,
            "weights_only_resume": False,
        },
        "evaluation_executed": False,
        "public_test_accessed": False,
    }
    receipt_path = _repository_path(
        entry["completion_receipt"], label="completion receipt"
    )
    if receipt_path.exists():
        if _load_json(receipt_path) != receipt:
            raise RuntimeError("Existing completion receipt does not match artifacts")
    else:
        _write_json_exclusive(receipt_path, receipt)
    return receipt


def run_resume(
    *, preregistration_path: Path, seed: int, prepare_only: bool
) -> dict[str, Any]:
    preregistration, entry, training_module = prepare(
        preregistration_path=preregistration_path, seed=seed
    )
    if prepare_only:
        return {
            "status": "prepared_required_full_state_resume",
            "seed": seed,
            "training_started": False,
        }
    training_module.FORMAL_TOPOLOGIES[(1, 128, 8)] = "1gpu_x_b128_x_accum8"
    arguments = _training_arguments(preregistration, entry, training_module)
    previous = sys.argv
    try:
        sys.argv = arguments
        result = training_module.run(training_module.parse_args())
    finally:
        sys.argv = previous
    if not isinstance(result, dict) or result.get("passed") is not True:
        raise RuntimeError("Required-resume training did not complete")
    return _finalize(entry)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive an interrupted Speed PLDM trace and perform one "
            "full-state required resume to the frozen optimizer budget."
        )
    )
    parser.add_argument("--seed", type=int, choices=EXPECTED_SEEDS, required=True)
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    preregistration_path = args.preregistration.resolve()
    if args.prepare_only and args.finalize_only:
        raise ValueError("--prepare-only and --finalize-only are mutually exclusive")
    if args.finalize_only:
        preregistration, entry = _validate_preregistration(
            preregistration_path, seed=args.seed
        )
        del preregistration
        result = _finalize(entry)
    else:
        result = run_resume(
            preregistration_path=preregistration_path,
            seed=args.seed,
            prepare_only=args.prepare_only,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
