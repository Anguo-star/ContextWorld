#!/usr/bin/env python3
"""Freeze and audit the post-recovery, pre-evaluation Speed PLDM disclosure.

The three Speed PLDM jobs were resumed after an externally observed
interruption.  This program is deliberately narrower than an evaluator: it
does not construct a dataset, instantiate a model, open a Public split, or
run CEM.  Once all three required resumes have completed, it records the
immutable evidence needed before the Development gate may begin.

The disclosure makes two kinds of statements explicit:

* it asserts the recorded full trainer-state resume and the fixed recipe and
  optimizer budget; and
* it does *not* assert bitwise equivalence to a hypothetical uninterrupted
  run for worker RNG, samples, batches, loss values, or parameter tensors.

Use ``generate`` exactly once after all three completion receipts exist.
Use ``audit`` later to re-check the immutable chain without requiring that
the Development namespace remains empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
DISCLOSURE_ID = "tworoom_speed_pldm_training_interruption_execution_disclosure_v1"
RECOVERY_ID = "tworoom_speed_pldm_training_interruption_recovery_v1"
TERMINAL_REPORT_RECOVERY_ID = "tworoom_speed_pldm_terminal_report_recovery_v1"
COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
EXPECTED_SEEDS = (3072, 4096, 5120)
CHECKPOINT_STEP = 10272
FINAL_STEP = 12840
OPTIMIZER_STEPS_PER_EPOCH = 2568
DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_speed_pldm_training_interruption_execution_disclosure_v1.yaml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "attempts/training_interruption_recovery_v1/execution_disclosure_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_local(value: str | Path, *, label: str) -> Path:
    raw = Path(value).expanduser()
    path = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the ContextWorld checkout") from error
    return path


def _logical(path: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the ContextWorld checkout") from error


def _identity(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return {
        "path": _logical(path, label=label),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _external_identity(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return payload


def _same_identity(value: Any, expected: Mapping[str, Any]) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("path") == expected.get("path")
        and value.get("sha256") == expected.get("sha256")
        and value.get("size_bytes") == expected.get("size_bytes")
    )


def _valid_checkpoint_archive_copy(
    archived: Mapping[str, Any],
    source_last: Mapping[str, Any],
    source_state: Mapping[str, Any],
    *,
    expected_archive_path: str,
) -> bool:
    """Match an archived copy by its own path and the source file's bytes."""

    return bool(
        archived.get("path") == expected_archive_path
        and archived.get("sha256")
        == source_last.get("sha256")
        == source_state.get("sha256")
        and archived.get("size_bytes")
        == source_last.get("size_bytes")
        == source_state.get("size_bytes")
    )


def _valid_trace_archive_copy(
    archived: Mapping[str, Any],
    source_trace: Mapping[str, Any],
    canonical_prefix: Mapping[str, Any],
    excluded_tail: Mapping[str, Any],
    *,
    expected_archive_path: str,
) -> bool:
    """Validate a relocated interrupted trace without conflating its paths."""

    return bool(
        archived.get("path") == expected_archive_path
        and archived.get("sha256") == source_trace.get("sha256")
        and archived.get("size_bytes") == source_trace.get("size_bytes")
        and archived.get("rows")
        == int(canonical_prefix.get("rows", -1))
        + int(excluded_tail.get("rows", -1))
        and archived.get("last_optimizer_step")
        == excluded_tail.get("last_optimizer_step")
    )


def _require_local_identity(
    specification: Any, *, label: str
) -> tuple[Path, dict[str, Any]]:
    if not (
        isinstance(specification, Mapping)
        and isinstance(specification.get("path"), str)
        and isinstance(specification.get("sha256"), str)
        and isinstance(specification.get("size_bytes"), int)
    ):
        raise ValueError(f"{label} needs path, sha256, and size_bytes")
    path = _resolve_local(str(specification["path"]), label=label)
    observed = _identity(path, label=label)
    if not _same_identity(observed, specification):
        raise RuntimeError(
            f"{label} identity drifted: expected={dict(specification)}, observed={observed}"
        )
    return path, observed


def _git_head(worktree: Path) -> str:
    marker = worktree / ".git"
    if marker.is_file():
        text = marker.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir: "):
            raise RuntimeError(f"Unsupported git pointer: {marker}")
        gitdir = Path(text[len("gitdir: ") :]).expanduser()
    elif marker.is_dir():
        gitdir = marker
    else:
        raise FileNotFoundError(f"StableWM worktree has no .git: {worktree}")
    head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head[len("ref: ") :]
        target = gitdir / ref
        if not target.is_file():
            common_file = gitdir / "commondir"
            if not common_file.is_file():
                raise RuntimeError(f"Cannot resolve StableWM ref: {ref}")
            target = (gitdir / common_file.read_text(encoding="utf-8").strip() / ref).resolve()
        head = target.read_text(encoding="utf-8").strip()
    if len(head) != 40:
        raise RuntimeError("StableWM HEAD is not a full commit SHA")
    return head


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _assert_output(config: Mapping[str, Any], output: Path) -> Path:
    outputs = config.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get("disclosure"), str):
        raise ValueError("Execution-disclosure config lacks outputs.disclosure")
    expected = _resolve_local(str(outputs["disclosure"]), label="disclosure output")
    if expected != DEFAULT_OUTPUT.resolve():
        raise ValueError("Execution-disclosure output is not in its dedicated namespace")
    actual = _resolve_local(output, label="disclosure output")
    if actual != expected:
        raise ValueError(
            "Execution disclosure output must equal its preregistered destination "
            f"{_logical(expected, label='disclosure output')}"
        )
    return expected


def _assert_exact(value: Any, expected: Any, *, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} differs from the registered execution-disclosure contract")


def _validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = _resolve_local(config_path, label="execution-disclosure config")
    config = _load_yaml(config_path)
    _assert_exact(config.get("schema_version"), 1, label="schema version")
    _assert_exact(config.get("execution_disclosure_id"), DISCLOSURE_ID, label="disclosure id")
    _assert_exact(config.get("completion_id"), COMPLETION_ID, label="completion id")
    _assert_exact(
        config.get("status"),
        "registered_after_terminal_report_failure_before_report_recovery_and_execution_disclosure",
        label="status",
    )
    _assert_exact(
        config.get("chronology"),
        {
            "registered_after_external_interruption": True,
            "registered_after_recovery_preparation": True,
            "registered_after_terminal_report_failure": True,
            "registered_after_terminal_report_recovery_preregistration": True,
            "not_preregistered_before_original_training": True,
            "terminal_reports_generated_at_registration": False,
            "development_executed_at_registration": False,
            "public_or_cem_executed_at_registration": False,
        },
        label="chronology",
    )
    _assert_exact(
        config.get("scope"),
        {
            "post_training_execution_disclosure_only": True,
            "terminal_report_recovery_bound": True,
            "training_recipe_changed": False,
            "checkpoint_selected_by_metric": False,
            "development_or_cem_executed": False,
            "public_test_accessed": False,
            "evaluation_authorized": False,
        },
        label="scope",
    )
    inputs = config.get("frozen_inputs")
    expected_inputs = {
        "recovery_preregistration",
        "resume_source_checkpoint_archive",
        "completion_config",
        "terminal_report_recovery_preregistration",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_inputs:
        raise ValueError("Execution-disclosure config has incomplete frozen inputs")
    input_identities = {
        name: _require_local_identity(specification, label=f"frozen_inputs.{name}")[1]
        for name, specification in inputs.items()
    }
    implementation = config.get("implementation")
    if not isinstance(implementation, Mapping) or set(implementation) != {
        "execution_disclosure_freezer"
    }:
        raise ValueError("Execution-disclosure implementation identities are incomplete")
    implementation_identities = {
        name: _require_local_identity(specification, label=f"implementation.{name}")[1]
        for name, specification in implementation.items()
    }
    expected_self = _identity(Path(__file__).resolve(), label="execution disclosure freezer")
    if implementation_identities["execution_disclosure_freezer"] != expected_self:
        raise RuntimeError("Execution-disclosure freezer is not the registered implementation")

    absence = config.get("pre_evaluation_absence_paths")
    if not isinstance(absence, list) or not absence or any(not isinstance(item, str) for item in absence):
        raise ValueError("Execution-disclosure config lacks concrete pre-evaluation absence paths")
    resolved_absence = [
        _resolve_local(value, label="pre-evaluation absence path") for value in absence
    ]
    if len(set(resolved_absence)) != len(resolved_absence):
        raise ValueError("Execution-disclosure absence paths are duplicated")

    _assert_exact(
        config.get("assertion_boundary"),
        {
            "asserted": {
                "full_trainer_state_resume_from_step_10272": True,
                "fixed_training_recipe_and_optimizer_budget": True,
                "terminal_reports_recovered_with_zero_optimizer_steps": True,
            },
            "not_asserted": {
                "worker_rng_bitwise_equivalence": False,
                "sample_order_bitwise_equivalence": False,
                "batch_composition_bitwise_equivalence": False,
                "loss_trace_bitwise_equivalence_after_resume": False,
                "parameter_tensor_bitwise_equivalence_to_an_uninterrupted_counterfactual": False,
            },
        },
        label="assertion boundary",
    )
    return config, {
        "config": _identity(config_path, label="execution-disclosure config"),
        "frozen_inputs": input_identities,
        "implementation": implementation_identities,
        "absence_paths": resolved_absence,
    }


def _validate_recovery_preregistration(
    prereg_path: Path,
    prereg_identity: Mapping[str, Any],
    completion_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    prereg = _load_yaml(prereg_path)
    if not (
        prereg.get("schema_version") == 1
        and prereg.get("recovery_id") == RECOVERY_ID
        and prereg.get("completion_id") == COMPLETION_ID
        and prereg.get("status")
        == "preregistered_after_external_interruption_before_recovery"
        and prereg.get("checkpoint_step") == CHECKPOINT_STEP
        and prereg.get("final_optimizer_step") == FINAL_STEP
        and prereg.get("optimizer_steps_per_epoch") == OPTIMIZER_STEPS_PER_EPOCH
    ):
        raise RuntimeError("Recovery preregistration is not the registered interruption contract")
    _assert_exact(
        prereg.get("scope"),
        {
            "training_recipe_changed": False,
            "checkpoint_selected_by_metric": False,
            "public_test_accessed": False,
            "development_or_cem_executed": False,
            "full_state_required_resume_only": True,
            "uncommitted_trace_tail_preserved": True,
        },
        label="recovery preregistration scope",
    )
    frozen = prereg.get("frozen_inputs")
    if not isinstance(frozen, Mapping) or set(frozen) != {
        "completion_config",
        "initialization_checkpoint",
    }:
        raise RuntimeError("Recovery preregistration frozen inputs are incomplete")
    completion_path, observed_completion = _require_local_identity(
        frozen["completion_config"], label="recovery completion config"
    )
    if observed_completion != completion_identity:
        raise RuntimeError("Recovery and disclosure completion-config identities disagree")
    _require_local_identity(
        frozen["initialization_checkpoint"], label="recovery initialization checkpoint"
    )
    implementation = prereg.get("implementation")
    expected_implementation = {"training_script", "recovery_wrapper", "stablewm_loader"}
    if not isinstance(implementation, Mapping) or set(implementation) != expected_implementation:
        raise RuntimeError("Recovery preregistration implementation identities are incomplete")
    implementation_identities = {
        name: _require_local_identity(specification, label=f"recovery implementation.{name}")[1]
        for name, specification in implementation.items()
    }
    runtime = prereg.get("stable_worldmodel")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "worktree",
        "commit",
        "pldm_config",
        "pldm_config_sha256",
    }:
        raise RuntimeError("Recovery preregistration StableWM runtime is incomplete")
    worktree = Path(str(runtime["worktree"])).resolve()
    observed_commit = _git_head(worktree)
    pldm_config = worktree / str(runtime["pldm_config"])
    observed_pldm = _external_identity(pldm_config, label="StableWM PLDM config")
    if not (
        observed_commit == runtime["commit"]
        and observed_pldm["sha256"] == runtime["pldm_config_sha256"]
    ):
        raise RuntimeError("Pinned StableWM runtime drifted")
    invocation = prereg.get("training_invocation")
    if not isinstance(invocation, Mapping) or not (
        invocation.get("model_id") == "H3_Speed_PLDM_ReferenceCompletion"
        and invocation.get("data_split_seed") == 3072
        and invocation.get("devices") == 1
        and invocation.get("batch_size_per_device") == 128
        and invocation.get("gradient_accumulation_steps") == 8
        and invocation.get("num_workers") == 6
        and invocation.get("logger_backend") == "none"
        and invocation.get("resume_policy") == "required"
    ):
        raise RuntimeError("Recovery invocation does not preserve the fixed recipe")
    rows = prereg.get("interrupted_runs")
    if not isinstance(rows, list) or tuple(int(row.get("seed", -1)) for row in rows) != EXPECTED_SEEDS:
        raise RuntimeError("Recovery preregistration has the wrong interrupted seed set")
    entries = {int(row["seed"]): row for row in rows}
    return prereg, entries, {
        "recovery_preregistration": dict(prereg_identity),
        "completion_config": dict(observed_completion),
        "implementation": implementation_identities,
        "stable_worldmodel": {
            "worktree": str(worktree),
            "expected_commit": str(runtime["commit"]),
            "observed_commit": observed_commit,
            "pldm_config": observed_pldm,
        },
        "training_invocation": {
            "model_id": invocation["model_id"],
            "data_split_seed": int(invocation["data_split_seed"]),
            "devices": int(invocation["devices"]),
            "batch_size_per_device": int(invocation["batch_size_per_device"]),
            "gradient_accumulation_steps": int(invocation["gradient_accumulation_steps"]),
            "num_workers": int(invocation["num_workers"]),
            "logger_backend": invocation["logger_backend"],
            "resume_policy": invocation["resume_policy"],
        },
    }


def _validate_terminal_report_recovery_preregistration(
    *,
    path: Path,
    identity: Mapping[str, Any],
    interruption_recovery_identity: Mapping[str, Any],
    completion_identity: Mapping[str, Any],
    interruption_entries: Mapping[int, Mapping[str, Any]],
    interruption_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the zero-optimizer terminal-report recovery into the disclosure."""

    payload = _load_yaml(path)
    if not (
        payload.get("schema_version") == 1
        and payload.get("recovery_id") == TERMINAL_REPORT_RECOVERY_ID
        and payload.get("completion_id") == COMPLETION_ID
        and payload.get("status")
        == "preregistered_after_terminal_report_failure_before_report_recovery"
        and payload.get("chronology")
        == {
            "fixed_budget_artifacts_present": True,
            "terminal_reports_present_at_registration": False,
            "completion_receipts_present_at_registration": False,
            "development_public_or_cem_executed_at_registration": False,
        }
        and payload.get("scope")
        == {
            "training_or_optimizer_execution": False,
            "checkpoint_selection": False,
            "model_or_loss_trace_modification": False,
            "terminal_report_generation_only": True,
            "public_test_access": False,
            "evaluation_execution": False,
        }
    ):
        raise RuntimeError("Terminal-report recovery preregistration is invalid")
    frozen = payload.get("frozen_inputs")
    if not isinstance(frozen, Mapping) or set(frozen) != {
        "completion_config",
        "interruption_recovery_preregistration",
        "training_script",
    }:
        raise RuntimeError("Terminal-report recovery frozen inputs are incomplete")
    _completion_path, observed_completion = _require_local_identity(
        frozen["completion_config"], label="terminal recovery completion config"
    )
    _interruption_path, observed_interruption = _require_local_identity(
        frozen["interruption_recovery_preregistration"],
        label="terminal recovery interruption preregistration",
    )
    _training_path, observed_training = _require_local_identity(
        frozen["training_script"], label="terminal recovery training script"
    )
    if (
        observed_completion != completion_identity
        or observed_interruption != interruption_recovery_identity
        or observed_training != interruption_state["implementation"]["training_script"]
    ):
        raise RuntimeError("Terminal-report recovery lineage disagrees with the interruption recovery")
    implementation = payload.get("implementation")
    if not isinstance(implementation, Mapping) or set(implementation) != {"report_recovery"}:
        raise RuntimeError("Terminal-report recovery implementation identity is incomplete")
    _tool_path, tool_identity = _require_local_identity(
        implementation["report_recovery"], label="terminal report recovery tool"
    )
    runtime = payload.get("stable_worldmodel")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("Terminal-report recovery runtime is missing")
    worktree = Path(str(runtime.get("worktree", ""))).resolve()
    training_entry = worktree / str(runtime.get("training_entry", ""))
    logger_entry = worktree / str(runtime.get("logger_entry", ""))
    if not (
        _git_head(worktree) == runtime.get("commit")
        and training_entry.is_file()
        and _sha256(training_entry) == runtime.get("training_entry_sha256")
        and runtime.get("logger_entry_expected_absent") is True
        and not logger_entry.exists()
    ):
        raise RuntimeError("Terminal-report recovery runtime boundary changed")
    rows = payload.get("runs")
    if not isinstance(rows, list) or tuple(int(row.get("seed", -1)) for row in rows) != EXPECTED_SEEDS:
        raise RuntimeError("Terminal-report recovery has the wrong seed set")
    registered_runs = []
    for row in rows:
        seed = int(row["seed"])
        interruption = interruption_entries[seed]
        if not (
            row.get("run_name") == interruption.get("run_name")
            and row.get("report_output") == interruption.get("training_report")
            and row.get("completion_receipt_output") == interruption.get("completion_receipt")
            and isinstance(row.get("final_checkpoint"), Mapping)
            and row["final_checkpoint"].get("path") == interruption.get("final_checkpoint")
            and isinstance(row.get("full_state_checkpoint"), Mapping)
            and row["full_state_checkpoint"].get("path")
            == interruption.get("state_checkpoint", {}).get("path")
            and isinstance(row.get("preparation_receipt"), Mapping)
            and row["preparation_receipt"].get("path")
            == interruption.get("preparation_receipt")
        ):
            raise RuntimeError(f"Terminal-report recovery paths disagree for seed {seed}")
        artifacts = {}
        for name in (
            "final_checkpoint",
            "full_state_checkpoint",
            "checkpoint_config",
            "loss_trace",
            "preparation_receipt",
        ):
            _artifact_path, artifacts[name] = _require_local_identity(
                row[name], label=f"terminal recovery seed {seed} {name}"
            )
        registered_runs.append({"seed": seed, "artifacts": artifacts})
    return {
        "preregistration": dict(identity),
        "implementation": tool_identity,
        "runs": registered_runs,
        "training_or_optimizer_execution_authorized": False,
    }


def _validate_preparation_receipt(
    *,
    path: Path,
    identity: Mapping[str, Any],
    seed: int,
    prereg_identity: Mapping[str, Any],
    entry: Mapping[str, Any],
    archive_identity: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _load_json(path)
    checkpoint = payload.get("resume_checkpoint")
    trace = payload.get("trace_recovery")
    if not (
        payload.get("schema_version") == 1
        and payload.get("recovery_id") == RECOVERY_ID
        and payload.get("completion_id") == COMPLETION_ID
        and payload.get("seed") == seed
        and payload.get("status") == "prepared_required_full_state_resume"
        and payload.get("passed") is True
        and payload.get("preregistration") == prereg_identity
        and payload.get("scope")
        == {
            "training_recipe_changed": False,
            "checkpoint_selected_by_metric": False,
            "public_test_accessed": False,
            "development_or_cem_executed": False,
        }
        and isinstance(checkpoint, Mapping)
        and checkpoint.get("global_step") == CHECKPOINT_STEP
        and checkpoint.get("epoch") == 4
        and checkpoint.get("optimizer_states") == 1
        and checkpoint.get("lr_schedulers") == 1
        and checkpoint.get("rng_ranks") == [0]
        and checkpoint.get("complete_epoch_boundary") is True
        and checkpoint.get("optimizer_steps_per_epoch") == OPTIMIZER_STEPS_PER_EPOCH
        and checkpoint.get("sha256") == archive_identity["sha256"]
        and isinstance(trace, Mapping)
        and isinstance(trace.get("original_trace"), Mapping)
        and isinstance(trace.get("canonical_trace"), Mapping)
        and isinstance(trace.get("excluded_uncommitted_tail"), Mapping)
    ):
        raise RuntimeError(f"Preparation receipt does not prove the required resume for seed {seed}")
    original_trace = trace["original_trace"]
    expected_original = entry.get("loss_trace")
    expected_archive_path = entry.get("archive_path")
    expected_tail = entry.get("discarded_uncommitted_tail")
    expected_prefix = entry.get("canonical_prefix")
    if not (
        isinstance(expected_original, Mapping)
        and isinstance(expected_archive_path, str)
        and isinstance(expected_tail, Mapping)
        and isinstance(expected_prefix, Mapping)
        and _valid_trace_archive_copy(
            original_trace,
            expected_original,
            expected_prefix,
            expected_tail,
            expected_archive_path=expected_archive_path,
        )
    ):
        raise RuntimeError(f"Preparation receipt original trace mismatches preregistration for seed {seed}")
    original_path = _resolve_local(str(original_trace["path"]), label=f"seed {seed} interrupted trace archive")
    observed_original = _identity(original_path, label=f"seed {seed} interrupted trace archive")
    if not _same_identity(observed_original, original_trace):
        raise RuntimeError(f"Interrupted trace archive identity drifted for seed {seed}")
    canonical = trace["canonical_trace"]
    if not isinstance(expected_prefix, Mapping) or not (
        canonical.get("sha256") == expected_prefix.get("sha256")
        and canonical.get("size_bytes") == expected_prefix.get("size_bytes")
        and canonical.get("rows") == expected_prefix.get("rows")
        and canonical.get("last_optimizer_step") == expected_prefix.get("last_optimizer_step")
    ):
        raise RuntimeError(f"Preparation receipt canonical trace mismatches preregistration for seed {seed}")
    if trace["excluded_uncommitted_tail"] != {
        **dict(expected_tail),
        "preserved_in_original_trace": True,
    }:
        raise RuntimeError(f"Preparation receipt uncommitted-tail record drifted for seed {seed}")
    return {
        "receipt": dict(identity),
        "resume_checkpoint_archive": dict(archive_identity),
        "interrupted_trace_archive": observed_original,
        "resume_checkpoint_metadata": {
            key: checkpoint[key]
            for key in (
                "global_step",
                "epoch",
                "optimizer_states",
                "lr_schedulers",
                "rng_ranks",
                "complete_epoch_boundary",
                "optimizer_steps_per_epoch",
            )
        },
    }


def _validate_completion_receipt(
    *,
    path: Path,
    identity: Mapping[str, Any],
    seed: int,
    entry: Mapping[str, Any],
    terminal_report_recovery_identity: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _load_json(path)
    report_path = _resolve_local(str(entry["training_report"]), label=f"seed {seed} training report")
    report_identity = _identity(report_path, label=f"seed {seed} training report")
    report = _load_json(report_path)
    training = report.get("training")
    report_recovery = report.get("terminal_report_recovery")
    if not (
        report.get("schema_version") == 1
        and report.get("passed") is True
        and report.get("run_name") == entry["run_name"]
        and isinstance(training, Mapping)
        and training.get("initial_global_step") == CHECKPOINT_STEP
        and training.get("restored_global_step") == CHECKPOINT_STEP
        and training.get("global_step") == FINAL_STEP
        and training.get("expected_optimizer_steps") == FINAL_STEP
        and training.get("resumed_from_checkpoint") is True
        and training.get("resume_policy") == "required"
        and training.get("training_complete") is True
        and training.get("terminal_report_recovery_optimizer_steps") == 0
        and isinstance(report_recovery, Mapping)
        and report_recovery.get("recovery_id") == TERMINAL_REPORT_RECOVERY_ID
        and report_recovery.get("preregistration")
        == terminal_report_recovery_identity
        and report_recovery.get("reason")
        == "original_process_failed_after_fixed_budget_artifacts_before_terminal_report_write"
        and report_recovery.get("original_process_log_archived") is False
        and report_recovery.get("training_or_optimizer_execution") is False
        and report_recovery.get("model_or_loss_trace_modified") is False
        and report_recovery.get("native_model_state_exact") is True
        and report_recovery.get("evaluation_executed") is False
        and report_recovery.get("public_test_accessed") is False
    ):
        raise RuntimeError(f"Final training report does not prove required resume for seed {seed}")
    final_checkpoint_path = _resolve_local(str(entry["final_checkpoint"]), label=f"seed {seed} final checkpoint")
    final_checkpoint = _identity(final_checkpoint_path, label=f"seed {seed} final checkpoint")
    final_trace_path = _resolve_local(str(entry["loss_trace"]["path"]), label=f"seed {seed} final loss trace")
    final_trace = _identity(final_trace_path, label=f"seed {seed} final loss trace")
    if not (
        payload.get("schema_version") == 1
        and payload.get("recovery_id") == RECOVERY_ID
        and payload.get("completion_id") == COMPLETION_ID
        and payload.get("seed") == seed
        and payload.get("status") == "completed_fixed_budget_required_resume"
        and payload.get("passed") is True
        and payload.get("training_report") == report_identity
        and payload.get("final_checkpoint") == final_checkpoint
        and isinstance(payload.get("final_loss_trace"), Mapping)
        and _same_identity(payload["final_loss_trace"], final_trace)
        and payload["final_loss_trace"].get("last_optimizer_step") == FINAL_STEP
        and payload.get("resume_proof")
        == {
            "initial_global_step": CHECKPOINT_STEP,
            "final_global_step": FINAL_STEP,
            "remaining_optimizer_steps_executed": FINAL_STEP - CHECKPOINT_STEP,
            "full_state_resume": True,
            "weights_only_resume": False,
        }
        and payload.get("evaluation_executed") is False
        and payload.get("public_test_accessed") is False
    ):
        raise RuntimeError(f"Completion receipt is invalid for seed {seed}")
    return {
        "completion_receipt": dict(identity),
        "training_report": report_identity,
        "final_checkpoint": final_checkpoint,
        "final_loss_trace": final_trace,
    }


def _validated_evidence(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate all execution evidence without requiring current path absence."""

    config, config_state = _validate_config(config_path)
    prereg_path = _resolve_local(
        config_state["frozen_inputs"]["recovery_preregistration"]["path"],
        label="recovery preregistration",
    )
    prereg, entries, recovery = _validate_recovery_preregistration(
        prereg_path,
        config_state["frozen_inputs"]["recovery_preregistration"],
        config_state["frozen_inputs"]["completion_config"],
    )
    terminal_path = _resolve_local(
        config_state["frozen_inputs"]["terminal_report_recovery_preregistration"]["path"],
        label="terminal-report recovery preregistration",
    )
    terminal_recovery = _validate_terminal_report_recovery_preregistration(
        path=terminal_path,
        identity=config_state["frozen_inputs"]["terminal_report_recovery_preregistration"],
        interruption_recovery_identity=recovery["recovery_preregistration"],
        completion_identity=recovery["completion_config"],
        interruption_entries=entries,
        interruption_state=recovery,
    )
    archive_path = _resolve_local(
        config_state["frozen_inputs"]["resume_source_checkpoint_archive"]["path"],
        label="resume-source checkpoint archive receipt",
    )
    archive_identity = config_state["frozen_inputs"]["resume_source_checkpoint_archive"]
    archive = _load_json(archive_path)
    if not (
        archive.get("schema_version") == 1
        and archive.get("archive_id")
        == "tworoom_speed_pldm_resume_source_checkpoint_archive_v1"
        and archive.get("completion_id") == COMPLETION_ID
        and archive.get("status")
        == "archived_during_required_resume_before_terminal_checkpoint_write"
        and archive.get("recovery_preregistration") == recovery["recovery_preregistration"]
        and archive.get("scope")
        == {
            "checkpoint_selection_performed": False,
            "public_test_accessed": False,
            "training_artifacts_modified": False,
            "training_recipe_changed": False,
        }
    ):
        raise RuntimeError("Resume-source checkpoint archive receipt is invalid")
    archive_rows = archive.get("entries")
    if not isinstance(archive_rows, list) or tuple(int(row.get("seed", -1)) for row in archive_rows) != EXPECTED_SEEDS:
        raise RuntimeError("Resume-source checkpoint archive has the wrong seed set")
    archive_by_seed = {int(row["seed"]): row for row in archive_rows}

    completed: list[dict[str, Any]] = []
    resume_sources: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        entry = entries[seed]
        archive_row = archive_by_seed[seed]
        archived_checkpoint_path, archived_checkpoint = _require_local_identity(
            archive_row.get("archive"), label=f"seed {seed} resume-source checkpoint archive"
        )
        del archived_checkpoint_path
        expected_checkpoint = entry.get("last_checkpoint")
        expected_state = entry.get("state_checkpoint")
        expected_archive_path = (
            "artifacts/evaluation/history3/"
            "tworoom_speed_pldm_reference_completion_v1/attempts/"
            f"training_interruption_recovery_v1/seed_{seed}/"
            "resume_source_step_10272.ckpt"
        )
        if not (
            isinstance(expected_checkpoint, Mapping)
            and isinstance(expected_state, Mapping)
            and _valid_checkpoint_archive_copy(
                archived_checkpoint,
                expected_checkpoint,
                expected_state,
                expected_archive_path=expected_archive_path,
            )
            and archive_row.get("source_identity_matched_at_copy") is True
        ):
            raise RuntimeError(f"Resume-source archive does not match step 10272 for seed {seed}")
        preparation_path, preparation_identity = _require_local_identity(
            archive_row.get("preparation_receipt"), label=f"seed {seed} preparation receipt"
        )
        if str(entry.get("preparation_receipt")) != preparation_identity["path"]:
            raise RuntimeError(f"Resume-source archive references the wrong preparation receipt for seed {seed}")
        prepared = _validate_preparation_receipt(
            path=preparation_path,
            identity=preparation_identity,
            seed=seed,
            prereg_identity=recovery["recovery_preregistration"],
            entry=entry,
            archive_identity=archived_checkpoint,
        )
        receipt_path = _resolve_local(
            str(entry.get("completion_receipt")),
            label=f"seed {seed} completion receipt",
        )
        receipt_identity = _identity(receipt_path, label=f"seed {seed} completion receipt")
        completed_evidence = _validate_completion_receipt(
            path=receipt_path,
            identity=receipt_identity,
            seed=seed,
            entry=entry,
            terminal_report_recovery_identity=terminal_recovery["preregistration"],
        )
        completed.append({"seed": seed, **completed_evidence})
        resume_sources.append({"seed": seed, **prepared})

    evidence = {
        "disclosure_config": config_state["config"],
        "recovery_preregistration": recovery["recovery_preregistration"],
        "completion_config": recovery["completion_config"],
        "implementation": recovery["implementation"],
        "runtime": {
            "stable_worldmodel": recovery["stable_worldmodel"],
            "training_invocation": recovery["training_invocation"],
        },
        "resume_source_checkpoint_archive": archive_identity,
        "terminal_report_recovery": terminal_recovery,
        "resume_sources": resume_sources,
        "completion_receipts": completed,
    }
    return config, config_state, evidence


def _absence_snapshot(paths: list[Path]) -> dict[str, Any]:
    rows = [
        {"path": _logical(path, label="pre-evaluation absence path"), "exists": path.exists()}
        for path in paths
    ]
    if any(row["exists"] for row in rows):
        existing = [row["path"] for row in rows if row["exists"]]
        raise RuntimeError(
            "Development/Public/CEM or binding artifacts already exist before execution disclosure: "
            + ", ".join(existing)
        )
    return {"all_absent": True, "paths": rows}


def build_disclosure(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Build, but do not write, the one immutable execution disclosure."""

    config, config_state, evidence = _validated_evidence(config_path)
    absence = _absence_snapshot(config_state["absence_paths"])
    return {
        "schema_version": 1,
        "execution_disclosure_id": DISCLOSURE_ID,
        "completion_id": COMPLETION_ID,
        "status": "frozen_post_training_pre_evaluation_execution_disclosure",
        "passed": True,
        "disclosure_config": evidence["disclosure_config"],
        "recovery_preregistration": evidence["recovery_preregistration"],
        "completion_config": evidence["completion_config"],
        "implementation": evidence["implementation"],
        "runtime": evidence["runtime"],
        "resume_source_checkpoint_archive": evidence["resume_source_checkpoint_archive"],
        "terminal_report_recovery": evidence["terminal_report_recovery"],
        "resume_sources": evidence["resume_sources"],
        "completion_receipts": evidence["completion_receipts"],
        "assertion_boundary": config["assertion_boundary"],
        "pre_evaluation_absence": absence,
        "evaluation_executed": False,
        "public_test_accessed": False,
        "cem_executed": False,
    }


def _receipt_evidence(receipt: Mapping[str, Any]) -> dict[str, Any]:
    completion = receipt.get("completion_receipts")
    if not isinstance(completion, list) or tuple(
        int(row.get("seed", -1)) for row in completion if isinstance(row, Mapping)
    ) != EXPECTED_SEEDS:
        raise RuntimeError("Execution disclosure has the wrong completion-receipt seed set")
    return {
        "receipt": None,
        "completion_receipts": [
            {"seed": int(row["seed"]), "receipt": dict(row["completion_receipt"])}
            for row in completion
        ],
        "training_reports": [
            {"seed": int(row["seed"]), "report": dict(row["training_report"])}
            for row in completion
        ],
    }


def audit_disclosure(
    *,
    config_path: Path = DEFAULT_CONFIG,
    disclosure_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Revalidate a frozen disclosure and return compact downstream evidence.

    This deliberately does not require the declared absence paths to remain
    absent.  Once the disclosure is frozen, Development receipts are expected
    to appear later; the receipt's immutable snapshot proves the earlier
    boundary instead.
    """

    config, _config_state, evidence = _validated_evidence(config_path)
    path = _assert_output(config, disclosure_path)
    receipt_identity = _identity(path, label="execution disclosure")
    receipt = _load_json(path)
    expected = {
        "schema_version": 1,
        "execution_disclosure_id": DISCLOSURE_ID,
        "completion_id": COMPLETION_ID,
        "status": "frozen_post_training_pre_evaluation_execution_disclosure",
        "passed": True,
        "disclosure_config": evidence["disclosure_config"],
        "recovery_preregistration": evidence["recovery_preregistration"],
        "completion_config": evidence["completion_config"],
        "implementation": evidence["implementation"],
        "runtime": evidence["runtime"],
        "resume_source_checkpoint_archive": evidence["resume_source_checkpoint_archive"],
        "terminal_report_recovery": evidence["terminal_report_recovery"],
        "resume_sources": evidence["resume_sources"],
        "completion_receipts": evidence["completion_receipts"],
        "assertion_boundary": config["assertion_boundary"],
        "evaluation_executed": False,
        "public_test_accessed": False,
        "cem_executed": False,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Execution disclosure does not match the immutable recovery evidence")
    absence = receipt.get("pre_evaluation_absence")
    if not (
        isinstance(absence, Mapping)
        and absence.get("all_absent") is True
        and isinstance(absence.get("paths"), list)
        and len(absence["paths"]) == len(config["pre_evaluation_absence_paths"])
        and all(
            isinstance(row, Mapping) and row.get("exists") is False
            for row in absence["paths"]
        )
    ):
        raise RuntimeError("Execution disclosure does not prove a pre-evaluation absence boundary")
    result = _receipt_evidence(receipt)
    result["receipt"] = receipt_identity
    return result


def generate(config_path: Path = DEFAULT_CONFIG, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    config, _config_state = _validate_config(config_path)
    destination = _assert_output(config, output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable execution disclosure: {destination}")
    receipt = build_disclosure(config_path)
    _write_json_exclusive(destination, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "audit"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "generate":
        receipt = generate(args.config, args.output)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "output": _logical(_assert_output(_load_yaml(args.config), args.output), label="execution disclosure"),
                    "seeds": list(EXPECTED_SEEDS),
                },
                sort_keys=True,
            )
        )
        return
    evidence = audit_disclosure(config_path=args.config, disclosure_path=args.output)
    print(
        json.dumps(
            {
                "status": "passed_execution_disclosure_audit",
                "receipt": evidence["receipt"],
                "completion_receipts": evidence["completion_receipts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
