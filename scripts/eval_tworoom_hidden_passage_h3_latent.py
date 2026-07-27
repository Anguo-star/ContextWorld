#!/usr/bin/env python3
"""Score a checkpoint on frozen History-3 hidden-passage true futures."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMAdapter,
    StableWorldModelPLDMAdapter,
)
from contextworld.evaluation.hidden_passage_h3_data import (
    TRAINING_RUN_LOCK_PROTOCOL,
    hidden_passage_training_run_lock,
    hidden_passage_training_run_lock_path,
)
from contextworld.evaluation.hidden_passage_validation import (
    canonical_sha256,
    file_sha256,
    load_validation_assets,
    score_validation_assets,
    summarize_validation_records,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.training.tworoom_data import (
    _load_formal_passage_build_report,
    hidden_passage_training_release_root,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_validation_v2.yaml"
)
AUDIT_SCHEDULING_CONTRACT = {
    "policy": "sibling_exclusive_flock",
    "maximum_concurrency": 1,
    "scope": "per_rank_full_audit_and_fit_start_storage_revalidation",
    "lock_protocol": "contextworld.hidden_passage_h3.audit_scheduling_lock.v1",
    "lock_order": "release_shared_then_audit_exclusive",
    "collective_holds_lock": False,
    "topology_scope": "single_node_8gpu",
    "concurrent_training_runs_per_release": 1,
}
PARALLEL_AUDIT_SCHEDULING_CONTRACT = {
    "policy": "sibling_shared_flock",
    "maximum_concurrency": 8,
    "scope": "per_rank_full_audit_and_fit_start_storage_revalidation",
    "lock_protocol": (
        "contextworld.hidden_passage_h3.audit_scheduling_lock.v2"
    ),
    "lock_order": "release_shared_then_audit_shared",
    "collective_holds_lock": False,
    "topology_scope": "single_node_8gpu",
    "concurrent_training_runs_per_release": 1,
}
AUDIT_SCHEDULING_CONTRACTS = (
    AUDIT_SCHEDULING_CONTRACT,
    PARALLEL_AUDIT_SCHEDULING_CONTRACT,
)
PARALLEL_RANK_CPU_AFFINITY_CONTRACT = {
    "policy": "local_rank_disjoint_contiguous_from_zero",
    "cpus_per_rank": 8,
    "expected_world_size": 8,
    "scope": "full_rank_process",
    "apply_before_stableworldmodel_and_lance_import": True,
}
TRAINING_RUN_EXCLUSIVITY_CONTRACT = {
    "protocol": TRAINING_RUN_LOCK_PROTOCOL,
    "policy": "one_root_training_run_per_release",
    "maximum_concurrency": 1,
    "blocking": False,
    "scope": "full_root_training_or_preflight_call",
    "held_through_report_snapshot": True,
    "nonzero_rank_admission": "direct_parent_holds_root_training_lock",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _checkpoint_protocol(checkpoint_config: dict) -> dict[str, int]:
    return {
        "history_size": int(checkpoint_config["wm"]["history_size"]),
        "num_preds": int(checkpoint_config["wm"]["num_preds"]),
        "frameskip": int(
            checkpoint_config["data"]["dataset"]["frameskip"]
        ),
        "num_steps": int(
            checkpoint_config["data"]["dataset"]["num_steps"]
        ),
        "action_encoder_input_dim": int(
            checkpoint_config["model"]["action_encoder"]["input_dim"]
        ),
    }


def _audit_group_training_artifact_hashes(
    training_config: dict,
) -> dict:
    expected_groups = (
        "passage_passable",
        "passage_blocked",
        "passage_mixed",
    )
    result = {}
    for group in expected_groups:
        catalog = resolve_contextworld_path(
            training_config["data"]["catalogs"][group],
            repo_root=ROOT,
        )
        artifact_root = catalog.parent.parent
        paths = {
            "catalog": catalog,
            "manifest": (
                artifact_root
                / "manifests"
                / f"{catalog.stem}.jsonl"
            ),
            "synthesis_report": (
                artifact_root
                / "reports"
                / f"{catalog.stem}.json"
            ),
        }
        quality = training_config["data_quality"]["groups"][group]
        expected = {
            "catalog": quality.get("required_catalog_sha256"),
            "manifest": quality.get("required_manifest_sha256"),
            "synthesis_report": quality.get(
                "required_synthesis_report_sha256"
            ),
        }
        _require(
            all(
                isinstance(value, str) and len(value) == 64
                for value in expected.values()
            ),
            f"{group} does not freeze all three formal artifact hashes",
        )
        observed = {}
        for name, path in paths.items():
            _require(path.is_file(), f"Missing frozen {group} {name}: {path}")
            observed[name] = file_sha256(path)
            _require(
                observed[name] == expected[name],
                f"Frozen {group} {name} hash mismatch",
            )
        result[group] = {
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": observed,
        }
    return result


def _sha256_values(value) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                "sha256" in str(key).lower()
                and isinstance(child, str)
                and len(child) == 64
            ):
                result.add(child)
            result.update(_sha256_values(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_sha256_values(child))
    return result


def _audit_formal_build_report(
    *,
    training_config: dict,
    checkpoint_data: dict,
    training_plan: dict,
    formal_artifact_hashes: dict,
) -> dict:
    specification = training_config.get("data", {}).get(
        "formal_build_report"
    )
    _require(
        isinstance(specification, dict),
        "Passage training config has no frozen formal build report",
    )
    authoritative_audit = _load_formal_passage_build_report(
        training_config,
        repo_root=ROOT,
    )
    report_path = Path(authoritative_audit["path"])
    _require(
        report_path.is_file(),
        f"Missing formal build report: {report_path}",
    )
    report_sha256 = file_sha256(report_path)
    _require(
        isinstance(specification.get("sha256"), str)
        and len(specification["sha256"]) == 64
        and report_sha256 == specification["sha256"],
        "Formal build report hash is pending or mismatched",
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        payload.get("passed") is True
        and payload.get("status") == "passed"
        and payload.get("benchmark") == specification.get("benchmark")
        and payload.get("scale") == specification.get("scale") == "formal",
        "Formal build report identity/status/scale failed",
    )
    checks = payload.get("checks")
    _require(
        isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values()),
        "Formal build report checks did not all pass",
    )
    # The immutable build report records source fields as
    # ``artifacts_by_group`` and ``physical_{shards,episodes,rows}``.
    # Training's authoritative loader validates those source fields and
    # derives the normalized receipt embedded in every checkpoint.
    active_artifacts = authoritative_audit.get("active_artifacts")
    physical_counts = authoritative_audit.get("physical_counts")
    _require(
        isinstance(active_artifacts, dict)
        and bool(active_artifacts)
        and isinstance(physical_counts, dict)
        and bool(physical_counts),
        "Formal build-report audit lacks derived artifacts or physical "
        "counts",
    )
    expected_hashes = {
        sha256
        for group in formal_artifact_hashes.values()
        for sha256 in group["sha256"].values()
    }
    _require(
        expected_hashes == _sha256_values(active_artifacts),
        "Formal build report does not bind every active training artifact "
        "hash",
    )

    embedded = checkpoint_data.get("formal_build_report_audit")
    _require(
        isinstance(embedded, dict)
        and canonical_sha256(embedded)
        == canonical_sha256(authoritative_audit),
        "Checkpoint does not embed the passed formal build-report audit",
    )
    gate = training_plan.get("formal_build_report_gate")
    _require(
        isinstance(gate, dict)
        and gate.get("required") is True
        and gate.get("passed") is True
        and Path(gate.get("path", "")).resolve() == report_path
        and gate.get("sha256") == report_sha256,
        "Training plan formal build-report gate did not pass",
    )
    return {
        "path": str(report_path),
        "sha256": report_sha256,
        "benchmark": payload["benchmark"],
        "scale": payload["scale"],
        "status": payload["status"],
        "checks": checks,
        "active_artifacts_sha256": canonical_sha256(active_artifacts),
        "physical_counts": physical_counts,
        "passed": True,
    }


def _audit_passage_ddp_timeout(
    *,
    runtime_contract: dict,
    checkpoint_contract: dict,
) -> dict:
    _require(
        isinstance(runtime_contract, dict)
        and canonical_sha256(runtime_contract)
        == canonical_sha256(checkpoint_contract)
        and runtime_contract.get("rendezvous_timeout_seconds_declared")
        == 7200
        and runtime_contract.get("rendezvous_timeout_scope")
        == "passage_multi_gpu_only"
        and runtime_contract.get("rendezvous_timeout_seconds_applied")
        == 7200
        and runtime_contract.get("rendezvous_timeout_override_applied")
        is True
        and runtime_contract.get("transport_configuration")
        == "framework_defaults_with_frozen_rendezvous_timeout"
        and runtime_contract.get("transport_overrides_applied") is False
        and runtime_contract.get("audit_scheduling")
        in AUDIT_SCHEDULING_CONTRACTS
        and runtime_contract.get("training_run_exclusivity")
        == TRAINING_RUN_EXCLUSIVITY_CONTRACT,
        "Passage formal DDP rendezvous timeout provenance failed",
    )
    audit_scheduling = runtime_contract["audit_scheduling"]
    rank_cpu_affinity = runtime_contract.get("rank_cpu_affinity")
    local_affinity = runtime_contract.get("local_rank_cpu_affinity")
    if audit_scheduling == PARALLEL_AUDIT_SCHEDULING_CONTRACT:
        _require(
            rank_cpu_affinity
            == PARALLEL_RANK_CPU_AFFINITY_CONTRACT
            and isinstance(local_affinity, dict)
            and local_affinity.get("passed") is True
            and int(local_affinity.get("local_rank", -1)) == 0
            and int(local_affinity.get("cpus_per_rank", -1)) == 8
            and local_affinity.get("cpu_ids") == list(range(8))
            and local_affinity.get(
                "applied_before_stableworldmodel_and_lance_import"
            )
            is True,
            "Parallel passage audit lacks the rank-0 CPU-affinity receipt",
        )
    else:
        _require(
            rank_cpu_affinity is None and local_affinity is None,
            "Serial passage audit must not retain parallel CPU affinity",
        )
    return {
        "rendezvous_timeout_seconds": 7200,
        "scope": "passage_multi_gpu_only",
        "audit_scheduling": audit_scheduling,
        "rank_cpu_affinity": rank_cpu_affinity,
        "passed": True,
    }


def _audit_training_run_exclusivity(
    *,
    report: dict | None,
    checkpoint_data: dict,
    release_root: Path,
    verify_lock_available: bool,
) -> dict:
    report_lock = (
        report.get("distributed_passage_audit", {}).get(
            "training_run_exclusivity"
        )
        if report is not None
        else None
    )
    checkpoint_lock = checkpoint_data.get(
        "distributed_passage_audit", {}
    ).get("training_run_exclusivity")
    _require(
        isinstance(checkpoint_lock, dict)
        and (
            report is None
            or (
                isinstance(report_lock, dict)
                and canonical_sha256(report_lock)
                == canonical_sha256(checkpoint_lock)
            )
        ),
        "Training-run exclusivity differs between report and checkpoint",
    )
    audited_lock = checkpoint_lock
    expected_path = hidden_passage_training_run_lock_path(release_root)
    wait_seconds = audited_lock.get("wait_seconds")
    holder_pid = audited_lock.get("holder_pid")
    _require(
        audited_lock.get("protocol") == TRAINING_RUN_LOCK_PROTOCOL
        and audited_lock.get("policy")
        == "one_root_training_run_per_release"
        and int(audited_lock.get("maximum_concurrency", -1)) == 1
        and audited_lock.get("blocking") is False
        and audited_lock.get("acquired") is True
        and isinstance(wait_seconds, (int, float))
        and float(wait_seconds) >= 0.0
        and audited_lock.get("hold_seconds") is None
        and audited_lock.get("path_identity_verified") is True
        and audited_lock.get(
            "path_identity_verified_after_acquire"
        )
        is True
        and audited_lock.get("descriptor_inheritable") is False
        and audited_lock.get("fork_child_close_registered") is True
        and isinstance(holder_pid, int)
        and holder_pid > 1
        and audited_lock.get("holder_pid_written") is True
        and audited_lock.get("released") is False
        and audited_lock.get("held_through_report_snapshot") is True
        and Path(str(audited_lock.get("path", ""))) == expected_path,
        "Training report/checkpoint lacks the frozen root-run lock receipt",
    )
    availability_receipt = None
    if verify_lock_available:
        try:
            with hidden_passage_training_run_lock(
                release_root
            ) as available:
                availability_receipt = dict(available)
        except BlockingIOError as exc:
            raise ValueError(
                "Training-run lock is still held; scoring cannot start "
                f"while training is active: {expected_path}"
            ) from exc
        availability_receipt = (
            _normalize_training_run_availability_receipt(
                dict(available),
                expected_path=expected_path,
            )
        )
    return {
        "protocol": TRAINING_RUN_LOCK_PROTOCOL,
        "policy": "one_root_training_run_per_release",
        "maximum_concurrency": 1,
        "receipt_sha256": canonical_sha256(audited_lock),
        "report_snapshot_held": True,
        "available_for_scoring": (
            True if verify_lock_available else None
        ),
        "availability_receipt": availability_receipt,
        "passed": True,
    }


def _normalize_training_run_availability_receipt(
    receipt: dict,
    *,
    expected_path: Path,
) -> dict:
    """Keep stable proof of a real probe, without PID/timing telemetry."""

    holder_pid = receipt.get("holder_pid")
    wait_seconds = receipt.get("wait_seconds")
    hold_seconds = receipt.get("hold_seconds")
    _require(
        receipt.get("protocol") == TRAINING_RUN_LOCK_PROTOCOL
        and receipt.get("policy")
        == "one_root_training_run_per_release"
        and int(receipt.get("maximum_concurrency", -1)) == 1
        and receipt.get("blocking") is False
        and receipt.get("acquired") is True
        and receipt.get("released") is True
        and receipt.get("path_identity_verified") is True
        and receipt.get("path_identity_verified_after_acquire") is True
        and receipt.get("descriptor_inheritable") is False
        and receipt.get("fork_child_close_registered") is True
        and receipt.get("holder_pid_written") is True
        and Path(str(receipt.get("path", ""))) == expected_path,
        "Training-run availability probe did not release cleanly",
    )
    _require(
        type(holder_pid) is int and holder_pid == os.getpid(),
        "Training-run availability probe was not held by this process",
    )
    _require(
        isinstance(wait_seconds, (int, float))
        and not isinstance(wait_seconds, bool)
        and math.isfinite(float(wait_seconds))
        and float(wait_seconds) >= 0.0,
        "Training-run availability probe wait time is invalid",
    )
    _require(
        isinstance(hold_seconds, (int, float))
        and not isinstance(hold_seconds, bool)
        and math.isfinite(float(hold_seconds))
        and float(hold_seconds) >= 0.0,
        "Training-run availability probe hold time is invalid",
    )
    normalized = dict(receipt)
    for field in ("holder_pid", "wait_seconds", "hold_seconds"):
        del normalized[field]
    normalized.update(
        {
            "holder_pid_is_current_process": True,
            "wait_seconds_nonnegative_and_finite": True,
            "hold_seconds_nonnegative_and_finite": True,
        }
    )
    return normalized


def _audit_distributed_passage_training(
    *,
    report: dict,
    checkpoint_data: dict,
    world_size: int,
    expected_audit_scheduling: dict = AUDIT_SCHEDULING_CONTRACT,
) -> dict:
    """Require the safe per-rank full-audit fallback before scoring."""

    audit = report.get("distributed_passage_audit")
    _require(
        isinstance(audit, dict)
        and audit.get("required") is True
        and audit.get("passed") is True
        and audit.get("optimization")
        == "disabled_per_rank_full_audit"
        and audit.get("every_rank_executed_full_logical_audit") is True
        and int(audit.get("full_logical_audit_execution_count", -1))
        == world_size
        and int(audit.get("storage_revalidation_execution_count", -1))
        == world_size
        and int(audit.get("attested_view_rank_count", -1)) == 0
        and audit.get("internal_attestation_used") is False,
        "Passage report did not use the safe per-rank full-audit fallback",
    )
    _require(
        int(audit.get("expected_rank_receipts", -1)) == world_size
        and audit.get("rank_coverage_passed") is True
        and audit.get("all_ranks_accepted_before_first_batch") is True
        and int(
            audit.get("training_dataloader_reads_before_release", -1)
        )
        == 0
        and int(
            audit.get("validation_dataloader_reads_before_release", -1)
        )
        == 0,
        "Passage Dataset release coverage/read-count contract failed",
    )
    receipts = audit.get("rank_receipts_before_first_batch")
    _require(
        isinstance(receipts, list)
        and len(receipts) == world_size
        and [int(row.get("rank", -1)) for row in receipts]
        == list(range(world_size)),
        "Passage report has incomplete per-rank audit receipts",
    )
    receipt_fields = {
        "passed": 1,
        "full_logical_audit_count": 1,
        "storage_revalidation_count": 1,
        "train_pre_release_calls": 0,
        "train_pre_release_items": 0,
        "val_pre_release_calls": 0,
        "val_pre_release_items": 0,
        "internal_environment_clean": 1,
        "full_audit_lock_acquired": 1,
        "full_audit_lock_released": 1,
        "full_audit_release_shared_held": 1,
        "full_audit_collective_unlocked": 1,
        "full_audit_path_identity_verified": 1,
        "full_audit_path_identity_verified_after_acquire": 1,
        "full_audit_descriptor_noninheritable": 1,
        "full_audit_fork_child_close_registered": 1,
        "full_audit_sample_contract_reads": 8,
        "revalidation_lock_acquired": 1,
        "revalidation_lock_released": 1,
        "revalidation_release_shared_held": 1,
        "revalidation_collective_unlocked": 1,
        "revalidation_path_identity_verified": 1,
        "revalidation_path_identity_verified_after_acquire": 1,
        "revalidation_descriptor_noninheritable": 1,
        "revalidation_fork_child_close_registered": 1,
    }
    _require(
        all(
            all(int(row.get(name, -1)) == expected for name, expected in (
                receipt_fields.items()
            ))
            for row in receipts
        ),
        "At least one passage rank lacks a clean full-audit receipt",
    )
    _require(
        all(
            int(row.get("full_audit_wait_milliseconds", -1)) >= 0
            and int(row.get("full_audit_hold_milliseconds", -1)) >= 0
            and int(row.get("revalidation_wait_milliseconds", -1)) >= 0
            and int(row.get("revalidation_hold_milliseconds", -1)) >= 0
            for row in receipts
        )
        and audit.get("audit_scheduling_contract")
        == expected_audit_scheduling,
        "Passage audit scheduling timing/contract is incomplete",
    )
    if expected_audit_scheduling == PARALLEL_AUDIT_SCHEDULING_CONTRACT:
        affinity_audit = report.get(
            "pre_batch_storage_revalidation",
            {},
        )
        _require(
            affinity_audit.get("rank_cpu_affinity_contract")
            == PARALLEL_RANK_CPU_AFFINITY_CONTRACT
            and affinity_audit.get(
                "all_rank_cpu_affinity_checked_before_consensus"
            )
            is True,
            "Parallel passage audit did not enforce per-rank CPU affinity",
        )
    sample_count = int(
        report.get("sample_contract", {}).get("sample_count", -1)
    )
    _require(
        sample_count > 0
        and int(
            audit.get("preflight_sample_contract_reads_per_rank", -1)
        )
        == sample_count
        and audit.get(
            "preflight_sample_contract_reads_are_not_training_reads"
        )
        is True,
        "Passage preflight sample reads are not separated from training reads",
    )
    train_gate = audit.get("train_gate_final")
    validation_gate = audit.get("validation_gate_final")
    _require(
        isinstance(train_gate, dict)
        and train_gate.get("released") is True
        and int(train_gate.get("pre_release_calls", -1)) == 0
        and int(train_gate.get("pre_release_items", -1)) == 0
        and int(train_gate.get("post_release_items", 0)) > 0
        and isinstance(validation_gate, dict)
        and validation_gate.get("released") is True
        and int(validation_gate.get("pre_release_calls", -1)) == 0
        and int(validation_gate.get("pre_release_items", -1)) == 0,
        "Passage Dataset gates were not released safely",
    )
    adapter_audit = checkpoint_data.get("distributed_passage_audit")
    _require(
        isinstance(adapter_audit, dict)
        and adapter_audit.get("required") is True
        and adapter_audit.get("passed") is True
        and adapter_audit.get("optimization")
        == "disabled_per_rank_full_audit"
        and adapter_audit.get("process_mode") == "full"
        and adapter_audit.get(
            "full_logical_audit_executed_in_this_process"
        )
        is True
        and adapter_audit.get("rank0_attestation_required") is False
        and adapter_audit.get("rank0_attestation_used") is False,
        "Checkpoint data adapter does not prove a local full logical audit",
    )
    pre_batch = report.get("pre_batch_storage_revalidation")
    _require(
        isinstance(pre_batch, dict)
        and pre_batch.get("passed") is True
        and pre_batch.get("optimization")
        == "disabled_per_rank_full_audit"
        and pre_batch.get("rank_coverage_passed") is True
        and pre_batch.get("gates_opened_only_after_consensus") is True
        and int(
            pre_batch.get("training_dataloader_reads_before_release", -1)
        )
        == 0
        and int(
            pre_batch.get(
                "validation_dataloader_reads_before_release",
                -1,
            )
        )
        == 0
        and canonical_sha256(pre_batch.get("rank_receipts"))
        == canonical_sha256(receipts),
        "Pre-batch passage storage revalidation provenance failed",
    )
    forbidden = {
        "launch_id",
        "rank0_attestation_sha256",
        "secret",
        "secret_path",
    }
    _require(
        not (forbidden & set(audit)),
        "Passage report leaked or retained internal attestation state",
    )
    return {
        "optimization": audit["optimization"],
        "world_size": world_size,
        "rank_receipts_sha256": canonical_sha256(receipts),
        "training_dataloader_reads_before_release": 0,
        "validation_dataloader_reads_before_release": 0,
        "passed": True,
    }


def validate_training_provenance(
    *,
    config: dict,
    model_id: str,
    training_seed: int,
    checkpoint: Path,
    training_report: Path,
) -> dict:
    """Validate the completed training run before any model scoring."""

    checkpoint = Path(checkpoint).resolve()
    training_report = Path(training_report).resolve()
    checkpoint_config_path = checkpoint.parent / "config.json"
    _require(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
    _require(
        checkpoint_config_path.is_file(),
        f"Missing checkpoint config: {checkpoint_config_path}",
    )
    _require(
        training_report.is_file(),
        f"Missing training report: {training_report}",
    )
    checkpoint_sha256 = file_sha256(checkpoint)
    checkpoint_config_sha256 = file_sha256(checkpoint_config_path)
    training_report_sha256 = file_sha256(training_report)
    checkpoint_config = json.loads(
        checkpoint_config_path.read_text(encoding="utf-8")
    )
    report = json.loads(training_report.read_text(encoding="utf-8"))
    protocol = _checkpoint_protocol(checkpoint_config)
    _require(
        protocol
        == {
            "history_size": 3,
            "num_preds": 1,
            "frameskip": 5,
            "num_steps": 4,
            "action_encoder_input_dim": 10,
        },
        f"Checkpoint is not a History-3/action-block-5 StableWM: {protocol}",
    )

    context = checkpoint_config.get("contextworld_benchmark", {})
    expected_training_model_id = str(
        config["comparison"]["checkpoint_training_model_id"][model_id]
    )
    _require(
        str(context.get("model_id")) == expected_training_model_id,
        "Checkpoint training model identity mismatch",
    )
    plan = context.get("training_plan", {})
    _require(
        int(plan.get("training_seed", -1)) == int(training_seed),
        "Checkpoint training seed mismatch",
    )
    _require(
        report.get("passed") is True
        and report.get("run_kind") == "confirmation"
        and report.get("save_load_exact") is True,
        "Training report is not a completed exact-save confirmation",
    )
    _require(
        report.get("model_id") == expected_training_model_id,
        "Training report model identity mismatch",
    )
    _require(
        report.get("stable_worldmodel", {}).get("commit")
        == config["stable_worldmodel"]["commit"],
        "Training report Stable-WorldModel commit mismatch",
    )
    training = report.get("training", {})
    _require(
        training.get("training_complete") is True,
        "Training report is incomplete",
    )
    _require(
        int(training.get("seed_before_model_initialization", -1))
        == int(training_seed),
        "Training report seed mismatch",
    )
    _require(
        canonical_sha256(training.get("plan"))
        == canonical_sha256(plan),
        "Training report and checkpoint embed different training plans",
    )
    artifacts = report.get("artifacts", {})
    _require(
        Path(artifacts.get("pretrained", "")).resolve() == checkpoint,
        "Training report checkpoint path mismatch",
    )
    _require(
        artifacts.get("pretrained_sha256") == checkpoint_sha256,
        "Training report checkpoint hash mismatch",
    )
    _require(
        Path(artifacts.get("pretrained_config", "")).resolve()
        == checkpoint_config_path,
        "Training report checkpoint-config path mismatch",
    )
    _require(
        artifacts.get("pretrained_config_sha256")
        == checkpoint_config_sha256,
        "Training report checkpoint-config hash mismatch",
    )
    _require(
        canonical_sha256(report.get("data"))
        == canonical_sha256(context.get("data")),
        "Training report and checkpoint embed different data provenance",
    )

    if model_id == "H3_Original_LEWM":
        contract = config["training_provenance"]["original_baseline"]
        expected_report = resolve_contextworld_path(
            contract["training_report"],
            repo_root=ROOT,
        )
        _require(
            training_report == expected_report
            and training_report_sha256
            == contract["training_report_sha256"],
            "Original baseline must use the frozen baseline training report",
        )
        _require(
            checkpoint_sha256 == contract["checkpoint_sha256"]
            and checkpoint_config_sha256
            == contract["checkpoint_config_sha256"],
            "Original baseline checkpoint identity mismatch",
        )
        expected_steps = int(contract["optimizer_steps"])
        _require(
            report.get("profile") == contract["profile"]
            and context.get("profile") == contract["profile"],
            "Original baseline profile mismatch",
        )
        _require(
            int(training.get("global_step", -1)) == expected_steps
            and int(training.get("expected_optimizer_steps", -1))
            == expected_steps
            and int(plan.get("optimizer_steps_total", -1))
            == expected_steps,
            "Original baseline optimizer-step contract failed",
        )
        _require(
            int(plan.get("total_global_sample_draws", -1))
            == int(contract["total_logical_draws"]),
            "Original baseline logical-draw contract failed",
        )
        topology = contract["topology"]
        _require(
            int(training.get("world_size", -1))
            == int(topology["world_size"])
            and int(training.get("devices", -1))
            == int(topology["devices"])
            and int(training.get("batch_size_per_device", -1))
            == int(topology["batch_size_per_device"])
            and int(
                training.get(
                    "adapter_gradient_accumulation_steps",
                    -1,
                )
            )
            == int(topology["accumulation_steps"])
            and plan.get("execution_topology")
            == topology["execution_topology"],
            "Original baseline execution topology mismatch",
        )
        data = context["data"]
        _require(
            data.get("group_weights") == {"original": 1.0}
            and set(data.get("groups", {})) == {"original"},
            "Original baseline is not original-data-only",
        )
        _require(
            context.get("initialization_checkpoint") in (None, {})
            and report.get("initialization_checkpoint") in (None, {}),
            "Original baseline must not use a model initialization checkpoint",
        )
        detail = {
            "contract": "frozen_original_baseline",
            "formal_training_artifact_hashes": {},
            "training_exclusion": None,
            "initialization_checkpoint": None,
        }
    else:
        by_model = config["training_provenance"].get(
            "passage_formal_by_model",
            {},
        )
        contract = by_model.get(
            model_id,
            config["training_provenance"].get("passage_formal"),
        )
        _require(
            isinstance(contract, dict),
            f"No passage-formal provenance contract for {model_id}",
        )
        configured_groups = config["comparison"].get(
            "checkpoint_training_group",
            {},
        )
        if model_id in configured_groups:
            expected_group = str(configured_groups[model_id])
        else:
            expected_group = {
                "H3_Passage_PassableOnly": "passage_passable",
                "H3_Passage_BlockedOnly": "passage_blocked",
                "H3_Passage_MixedRules": "passage_mixed",
            }[model_id]
        _require(
            report.get("profile") == contract["profile"]
            and context.get("profile") == contract["profile"],
            "Pilot/smoke checkpoint is forbidden; passage_formal is required",
        )
        expected_steps = int(contract["optimizer_steps"])
        _require(
            int(training.get("global_step", -1)) == expected_steps
            and int(training.get("expected_optimizer_steps", -1))
            == expected_steps
            and int(plan.get("optimizer_steps_total", -1))
            == expected_steps,
            "Passage formal optimizer-step contract failed",
        )
        _require(
            int(plan.get("total_global_sample_draws", -1))
            == int(contract["total_logical_draws"])
            and int(plan.get("global_batch_size", -1))
            == int(contract["effective_global_batch"]),
            "Passage formal draw/global-batch contract failed",
        )
        topology = contract["topology"]
        _require(
            int(training.get("world_size", -1))
            == int(topology["world_size"])
            and int(training.get("devices", -1))
            == int(topology["devices"])
            and int(training.get("batch_size_per_device", -1))
            == int(topology["batch_size_per_device"])
            and int(
                training.get(
                    "adapter_gradient_accumulation_steps",
                    -1,
                )
            )
            == int(topology["accumulation_steps"])
            and int(plan.get("devices", -1)) == int(topology["devices"])
            and int(plan.get("batch_size_per_device", -1))
            == int(topology["batch_size_per_device"])
            and int(
                plan.get(
                    "adapter_gradient_accumulation_steps",
                    -1,
                )
            )
            == int(topology["accumulation_steps"])
            and plan.get("execution_topology")
            == topology["execution_topology"],
            "Passage formal execution topology mismatch",
        )
        runtime_distributed = training.get(
            "distributed_execution_contract"
        )
        checkpoint_distributed = context.get(
            "distributed_execution_contract"
        )
        rendezvous_timeout_audit = _audit_passage_ddp_timeout(
            runtime_contract=runtime_distributed,
            checkpoint_contract=checkpoint_distributed,
        )
        distributed_audit = _audit_distributed_passage_training(
            report=report,
            checkpoint_data=context["data"],
            world_size=int(training["world_size"]),
            expected_audit_scheduling=runtime_distributed[
                "audit_scheduling"
            ],
        )
        data = context["data"]
        scope = data.get("training_data_scope", {})
        _require(
            scope.get("synthetic_only") is True
            and scope.get("original_samples_included") is False,
            "Passage formal training must be synthetic-only",
        )
        _require(
            data.get("group_weights") == {expected_group: 1.0}
            and set(data.get("groups", {})) == {expected_group},
            "Passage formal checkpoint does not use its one registered group",
        )

        training_config_path = resolve_contextworld_path(
            contract["training_benchmark_config"],
            repo_root=ROOT,
        )
        _require(
            training_config_path.is_file(),
            f"Missing passage training benchmark: {training_config_path}",
        )
        _require(
            Path(context.get("benchmark_config", "")).resolve()
            == training_config_path,
            "Checkpoint embeds a different passage training benchmark",
        )
        training_config = yaml.safe_load(
            training_config_path.read_text(encoding="utf-8")
        )
        expected_frozen_modules = list(
            contract.get("frozen_model_modules", [])
        )
        declared_frozen_modules = list(
            training_config.get("training_protocol", {}).get(
                "frozen_model_modules",
                [],
            )
        )
        _require(
            declared_frozen_modules == expected_frozen_modules,
            "Training config representation-freeze contract mismatch",
        )
        checkpoint_freeze = context.get("frozen_model_modules", {})
        report_freeze = report.get("frozen_model_modules", {})
        if expected_frozen_modules:
            _require(
                expected_frozen_modules == ["encoder", "projector"],
                "Only the audited encoder/projector freeze is supported",
            )
            _require(
                checkpoint_freeze.get("configured") is True
                and checkpoint_freeze.get("modules")
                == expected_frozen_modules
                and checkpoint_freeze.get("force_eval_mode") is True,
                "Checkpoint config does not declare the frozen "
                "encoder/projector contract",
            )
            _require(
                report_freeze.get("configured") is True
                and report_freeze.get("applied") is True
                and report_freeze.get("passed") is True
                and report_freeze.get("modules")
                == expected_frozen_modules
                and report_freeze.get("force_eval_mode") is True,
                "Training report does not contain a passed representation "
                "freeze audit",
            )
            initial_hashes = report_freeze.get(
                "initial_state_sha256",
                {},
            )
            final_hashes = report_freeze.get(
                "final_state_sha256",
                {},
            )
            unchanged = report_freeze.get("state_unchanged", {})
            _require(
                set(initial_hashes) == set(expected_frozen_modules)
                and initial_hashes == final_hashes
                and all(
                    unchanged.get(module) is True
                    for module in expected_frozen_modules
                ),
                "Frozen encoder/projector state changed during training",
            )
        else:
            _require(
                checkpoint_freeze.get("configured") in (None, False)
                and report_freeze.get("configured") in (None, False),
                "Unexpected representation freeze in an unfrozen recipe",
            )
        release_root = hidden_passage_training_release_root(
            training_config_path,
            repo_root=ROOT,
            model_id=expected_training_model_id,
        )
        _require(
            release_root is not None,
            "Passage training config does not resolve a sealed release",
        )
        training_run_exclusivity_audit = (
            _audit_training_run_exclusivity(
                report=report,
                checkpoint_data=data,
                release_root=release_root,
                verify_lock_available=True,
            )
        )
        formal_hashes = _audit_group_training_artifact_hashes(
            training_config
        )
        formal_build_report = _audit_formal_build_report(
            training_config=training_config,
            checkpoint_data=data,
            training_plan=plan,
            formal_artifact_hashes=formal_hashes,
        )
        quality = data["groups"][expected_group]
        expected_selected_hashes = formal_hashes[expected_group]["sha256"]
        split_audit = quality["catalog_split_audit"]
        _require(
            split_audit.get("required_artifact_hashes")
            == expected_selected_hashes
            and {
                "catalog": split_audit.get("catalog_sha256"),
                "manifest": split_audit.get("manifest_sha256"),
                "synthesis_report": split_audit.get(
                    "synthesis_report_sha256"
                ),
            }
            == expected_selected_hashes,
            "Checkpoint does not embed the selected formal artifact hashes",
        )
        quality_gate = plan.get("data_quality_gates", {}).get(
            expected_group,
            {},
        )
        _require(
            quality_gate.get("passed") is True
            and quality_gate.get("gates", {}).get(
                "formal_artifact_hashes_frozen"
            )
            is True,
            "Checkpoint formal data-quality hash gate did not pass",
        )

        exclusion_spec = training_config.get("data", {}).get(
            "training_exclusion_manifest"
        )
        _require(
            isinstance(exclusion_spec, dict),
            "Passage formal training has no frozen V2 exclusion manifest",
        )
        exclusion_path = resolve_contextworld_path(
            exclusion_spec["path"],
            repo_root=ROOT,
        )
        _require(
            exclusion_path.is_file()
            and file_sha256(exclusion_path) == exclusion_spec["sha256"],
            "V2 training exclusion manifest path/hash mismatch",
        )
        exclusion_payload = json.loads(
            exclusion_path.read_text(encoding="utf-8")
        )
        _require(
            exclusion_payload.get("benchmark")
            == contract["validation_exclusion_benchmark"]
            and exclusion_payload.get("content_manifest_sha256")
            == exclusion_spec["content_sha256"],
            "Training exclusion is not the frozen Validation V2 manifest",
        )
        exclusion_audit = data.get("training_exclusion_audit", {})
        _require(
            exclusion_audit.get("passed") is True
            and exclusion_audit.get("sha256")
            == exclusion_spec["sha256"]
            and exclusion_audit.get("content_sha256")
            == exclusion_spec["content_sha256"],
            "Checkpoint does not embed the passed V2 exclusion audit",
        )

        expected_init = contract["initialization_checkpoint"]
        config_init = context.get("initialization_checkpoint", {})
        report_init = report.get("initialization_checkpoint", {})
        for observed in (config_init, report_init):
            _require(
                observed.get("sha256") == expected_init["sha256"]
                and observed.get("config_sha256")
                == expected_init["config_sha256"]
                and observed.get("role") == expected_init["role"],
                "Passage model initialization identity mismatch",
            )
        _require(
            report_init.get("configured") is True
            and report_init.get("applied") is True
            and report_init.get("state_exact") is True
            and report_init.get("source_state_sha256")
            == report_init.get("initialized_state_sha256"),
            "Passage initialization was not applied exactly",
        )
        detail = {
            "contract": "passage_formal_synthetic_only",
            "training_benchmark_config": str(training_config_path),
            "training_benchmark_config_sha256": file_sha256(
                training_config_path
            ),
            "formal_training_artifact_hashes": formal_hashes,
            "formal_build_report": formal_build_report,
            "training_exclusion": {
                "path": str(exclusion_path),
                "sha256": exclusion_spec["sha256"],
                "content_sha256": exclusion_spec["content_sha256"],
            },
            "initialization_checkpoint": expected_init,
            "distributed_passage_audit": distributed_audit,
            "distributed_execution_contract": runtime_distributed,
            "rendezvous_timeout_audit": rendezvous_timeout_audit,
            "training_run_exclusivity_audit": (
                training_run_exclusivity_audit
            ),
            "frozen_model_modules": (
                report_freeze if expected_frozen_modules else None
            ),
        }

    return {
        "passed": True,
        "model_id": model_id,
        "training_seed": int(training_seed),
        "training_model_id": expected_training_model_id,
        "training_report": str(training_report),
        "training_report_sha256": training_report_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_config": str(checkpoint_config_path),
        "checkpoint_config_sha256": checkpoint_config_sha256,
        "checkpoint_protocol": protocol,
        "stable_worldmodel_commit": config["stable_worldmodel"]["commit"],
        **detail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one StableWM checkpoint on the offline 2-future x "
            "3-history hidden-passage Validation matrix"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--model-slug")
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--stablewm-repo")
    parser.add_argument("--stablewm-ref")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    allowed_statuses = {
        "preregistered_before_independent_catalog_generation_and_scoring",
        "diagnostic_frozen_before_catalog_generation_and_scoring",
        "preregistered_pldm_objective_control_before_scoring",
        (
            "aligned_to_preexisting_rule_switch_v2_before_"
            "final_pldm_attribution"
        ),
        (
            "frozen_after_v1_train_seen_diagnostic_before_"
            "unseen_validation_scoring"
        ),
        "preregistered_after_train_seen_diagnostic_before_unseen_scoring",
    }
    if config.get("status") not in allowed_statuses:
        raise ValueError(
            "Validation/diagnostic config is not frozen before scoring"
        )
    required_results = {
        str(model_id): tuple(map(int, seeds))
        for model_id, seeds in config["comparison"][
            "required_results"
        ].items()
    }
    if args.model_id not in required_results:
        raise ValueError(
            f"Unknown model-id {args.model_id!r}; "
            f"expected one of {sorted(required_results)}"
        )
    if int(args.training_seed) not in required_results[args.model_id]:
        raise ValueError(
            f"Training seed {args.training_seed} is not registered for "
            f"{args.model_id}: {required_results[args.model_id]}"
        )
    model_slug = args.model_slug or (
        f"{args.model_id}_s{int(args.training_seed)}"
    )
    catalog = resolve_contextworld_path(
        args.catalog or config["artifacts"]["catalog"],
        repo_root=ROOT,
    )
    checkpoint = resolve_contextworld_path(args.checkpoint, repo_root=ROOT)
    training_report = resolve_contextworld_path(
        args.training_report,
        repo_root=ROOT,
    )
    training_provenance = validate_training_provenance(
        config=config,
        model_id=args.model_id,
        training_seed=int(args.training_seed),
        checkpoint=checkpoint,
        training_report=training_report,
    )
    checkpoint_config_path = Path(
        training_provenance["checkpoint_config"]
    )
    checkpoint_protocol = training_provenance["checkpoint_protocol"]
    normalizer = resolve_contextworld_path(
        args.normalizer or config["adapter"]["normalizer"],
        repo_root=ROOT,
    )
    output = resolve_contextworld_path(args.output, repo_root=ROOT)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    batch_size = int(
        args.batch_size or config["evaluation"]["batch_size"]
    )
    stablewm_repo = str(
        args.stablewm_repo or config["stable_worldmodel"]["repo"]
    )
    stablewm_ref = str(
        args.stablewm_ref or config["stable_worldmodel"]["commit"]
    )
    if stablewm_repo != str(config["stable_worldmodel"]["repo"]):
        raise ValueError(
            "Formal Validation forbids overriding stable_worldmodel.repo"
        )
    if stablewm_ref != str(config["stable_worldmodel"]["commit"]):
        raise ValueError(
            "Formal Validation forbids overriding stable_worldmodel.commit"
        )
    normalizer_sha256 = file_sha256(normalizer)
    if normalizer_sha256 != str(config["adapter"]["normalizer_sha256"]):
        raise ValueError(
            "Frozen normalizer hash mismatch: "
            f"{normalizer_sha256} != "
            f"{config['adapter']['normalizer_sha256']}"
        )
    catalog_payload = json.loads(catalog.read_text(encoding="utf-8"))
    if catalog_payload.get("benchmark") != config["benchmark"]:
        raise ValueError(
            "Catalog benchmark differs from the Validation config"
        )

    # Asset loading and every call below use only frozen arrays.  No simulator
    # is instantiated during checkpoint scoring.
    assets, data_audit = load_validation_assets(
        catalog,
        repo_root=ROOT,
    )
    adapter_name = str(
        config.get("adapter", {}).get(
            "implementation",
            "StableWorldModelLeWMAdapter",
        )
    )
    adapter_classes = {
        "StableWorldModelLeWMAdapter": StableWorldModelLeWMAdapter,
        "StableWorldModelPLDMAdapter": StableWorldModelPLDMAdapter,
    }
    _require(
        adapter_name in adapter_classes,
        f"Unsupported StableWM adapter: {adapter_name}",
    )
    adapter = adapter_classes[adapter_name].from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=stablewm_repo,
        stablewm_ref=stablewm_ref,
        device=args.device,
    )
    scored = score_validation_assets(
        adapter,
        assets,
        batch_size=batch_size,
    )
    evaluation = config["evaluation"]
    summary = summarize_validation_records(
        scored["records"],
        eval_seeds=evaluation["eval_seeds"],
        unique_queries_per_seed=int(
            evaluation["unique_queries_per_seed"]
        ),
        gates=config["gates"],
    )
    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "model_id": args.model_id,
        "training_seed": int(args.training_seed),
        "model_slug": model_slug,
        "model": adapter.metadata,
        "identity": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "catalog": str(catalog),
            "catalog_sha256": file_sha256(catalog),
            "checkpoint": str(checkpoint),
            "checkpoint_config": str(checkpoint_config_path),
            "checkpoint_config_sha256": file_sha256(
                checkpoint_config_path
            ),
            "checkpoint_protocol": checkpoint_protocol,
            "training_report": str(training_report),
            "training_report_sha256": training_provenance[
                "training_report_sha256"
            ],
            "normalizer": str(normalizer),
            "normalizer_sha256": normalizer_sha256,
        },
        "data_audit": data_audit,
        "training_provenance": training_provenance,
        "score_audit": scored["score_audit"],
        "summary": summary,
        "records": scored["records"],
    }
    write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "model_id": args.model_id,
                "training_seed": int(args.training_seed),
                "model_slug": model_slug,
                "records": len(scored["records"]),
                "decision": summary["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
