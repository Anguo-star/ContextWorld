#!/usr/bin/env python3
"""Fail-closed pre-Public binding freezer for the Speed PLDM completion.

The freezer reads completed training artifacts, frozen source identities and a
CPU-loaded checkpoint only.  It deliberately does not instantiate a Speed
ICL dataset or call a planner, so the Public boundary remains closed until
this receipt passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from contextworld.benchmarks.adapters import StableWorldModelPLDMAdapter
from contextworld.benchmarks.speed_pldm_infrastructure_development import (
    DEVELOPMENT_ID,
    DEVELOPMENT_SCOPE,
    EXPECTED_SEEDS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINDING = ROOT / "configs/benchmark/tworoom_speed_pldm_evaluation_binding_v1.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "evaluation_binding_v1/evaluation_binding_receipt.json"
)
COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
CLAIM_LEVEL = "behavioral_trained_reference_only"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _logical(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


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


def _git_head(worktree: Path) -> str:
    pointer = worktree / ".git"
    if pointer.is_file():
        text = pointer.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir: "):
            raise RuntimeError(f"Unsupported git pointer: {pointer}")
        gitdir = Path(text[len("gitdir: ") :]).expanduser()
    elif pointer.is_dir():
        gitdir = pointer
    else:
        raise FileNotFoundError(f"Missing .git under {worktree}")
    head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head[len("ref: ") :]
        target = gitdir / ref
        if not target.is_file():
            common = (gitdir / "commondir").read_text(encoding="utf-8").strip()
            target = (gitdir / common / ref).resolve()
        head = target.read_text(encoding="utf-8").strip()
    if len(head) != 40:
        raise RuntimeError(f"Unable to resolve git HEAD for {worktree}")
    return head


def _file_check(
    checks: dict[str, dict[str, Any]], name: str, specification: dict[str, Any]
) -> Path:
    path = _resolve(specification["path"])
    observed = _sha256(path) if path.is_file() else None
    checks[name] = {
        "passed": observed == specification["sha256"],
        "path": _logical(path),
        "expected_sha256": specification["sha256"],
        "observed_sha256": observed,
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
    }
    return path


def _same_specification(value: Any, expected: dict[str, Any]) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("path") == expected.get("path")
        and value.get("sha256") == expected.get("sha256")
    )


def _claim_scope() -> dict[str, Any]:
    return {
        "paired_single_speed_control_available": False,
        "training_attribution_claim": False,
        "public_test_reopened": False,
        "claim_level": CLAIM_LEVEL,
    }


def _validated_behavioral_claim_boundary(
    *, binding: dict[str, Any], checks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Freeze the no-attribution claim boundary before Public access exists."""

    specification = binding.get("behavioral_claim_boundary")
    if not isinstance(specification, dict):
        raise ValueError("Binding lacks behavioral_claim_boundary")
    path = _file_check(checks, "behavioral_claim_boundary", specification)
    identity = {
        "path": _logical(path),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
    }
    payload = _load_yaml(path)
    chronology = payload.get("chronology")
    claim = payload.get("claim_boundary")
    conditional = payload.get("conditional_evaluation")
    mutation = payload.get("mutation_boundary")
    frozen = payload.get("frozen_inputs")
    scorer = binding.get("evaluator_sources", {}).get("speed_icl_score")
    contract = bool(
        payload.get("schema_version") == 1
        and payload.get("amendment_id")
        == "tworoom_speed_pldm_behavioral_claim_boundary_v1"
        and payload.get("completion_id") == COMPLETION_ID
        and payload.get("release_id") == binding.get("release", {}).get("release_id")
        and payload.get("status")
        == "preregistered_during_fixed_training_before_development_or_public_evaluation"
        and chronology
        == {
            "fixed_training_already_running": True,
            "development_evaluation_started": False,
            "public_test_opened": False,
            "checkpoint_selection_changed": False,
            "training_budget_changed": False,
        }
        and isinstance(frozen, dict)
        and _same_specification(frozen.get("completion_config"), binding.get("completion", {}))
        and _same_specification(frozen.get("speed_release"), binding.get("release", {}))
        and _same_specification(frozen.get("public_scorer"), scorer)
        and isinstance(claim, dict)
        and claim.get("paired_single_speed_pldm_controls_trained") is False
        and claim.get("training_attribution_claim_authorized") is False
        and claim.get("training_attributed_speed_icl_claim_authorized") is False
        and claim.get("three_seed_behavioral_reference_authorized") is True
        and claim.get("comparison_with_lewm_single_speed_controls_authorized")
        is False
        and claim.get("cross_architecture_raw_latent_loss_comparison_authorized")
        is False
        and claim.get("method_name_must_identify_pldm") is True
        and claim.get("scoreboard_evidence_scope_if_reported") == "behavioral"
        and isinstance(claim.get("behavioral_reference_requirement"), str)
        and claim["behavioral_reference_requirement"].strip()
        and conditional
        == {
            "development_must_precede_public": True,
            "public_test_authorized_by_this_record": False,
            "public_test_requires_separate_passed_binding": True,
            "if_three_seed_public_behavioral_gate_passes": {
                "action_planning_cem_may_be_separately_authorized": True,
                "original_tworoom_retention_cem_may_be_separately_authorized": True,
            },
            "if_any_public_behavioral_gate_fails": {
                "cem_authorized": False,
                "cem_executed": False,
                "terminal_stop_receipt_required": True,
            },
        }
        and mutation
        == {
            "training_or_checkpoint_selection_authorized": False,
            "optimizer_step_or_batch_change_authorized": False,
            "raw_data_mutation_authorized": False,
            "score_or_threshold_change_authorized": False,
            "public_test_access_authorized": False,
        }
    )
    _record(
        checks,
        "behavioral_claim_boundary_contract",
        contract,
        claim_scope=_claim_scope(),
        boundary=payload,
    )
    return identity


def _record(checks: dict[str, dict[str, Any]], name: str, passed: bool, **data: Any) -> None:
    checks[name] = {"passed": bool(passed), **data}


def _validate_prepublic_cem_protocol(
    *, binding: dict[str, Any], checks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Require the full CEM implementation/decision closure before Public ICL.

    This routine is intentionally file-identity-only.  It does not validate a
    catalog through a simulator, create a planner, or read any Public payload.
    Its job is to make later positive-branch code selection impossible.
    """

    protocol = binding.get("cem_protocol")
    evaluator_sources = binding.get("evaluator_sources")
    if not isinstance(protocol, Mapping) or not isinstance(evaluator_sources, Mapping):
        raise ValueError("Binding lacks the pre-Public CEM authority")
    source_identities = protocol.get("source_identities")
    implementation = protocol.get("implementation")
    tracks = protocol.get("tracks")
    outputs = protocol.get("outputs")
    expected_sources = {
        "preregistration",
        "source_protocol",
        "aggregate_preregistration",
        "retention_noninferiority_protocol",
        "action_catalog",
        "action_catalog_validator",
        "action_episode_oracle",
        "action_runner_core",
        "speed_cli",
        "speed_score",
        "retention_catalog",
        "retention_catalog_builder",
        "retention_episode_oracle",
        "retention_runner_core",
        "retention_frozen_baseline_wrapper",
        "implementation_formal_runner",
        "implementation_aggregate_freezer",
        "implementation_binding_freezer",
        "implementation_adapter_boundary",
        "implementation_development_contract",
        "implementation_paired_retention_comparator",
    }
    source_contract = bool(
        isinstance(source_identities, Mapping)
        and set(source_identities) == expected_sources
        and all(
            evaluator_sources.get(f"cem_{name}") == source_identities[name]
            for name in expected_sources
        )
    )
    _record(
        checks,
        "prepublic_cem_source_closure",
        source_contract,
        source_names=sorted(source_identities) if isinstance(source_identities, Mapping) else None,
    )
    if not source_contract:
        raise ValueError("Binding CEM source closure is incomplete or rebinding a source")
    for name in sorted(expected_sources):
        _file_check(checks, f"prepublic_cem_source_{name}", source_identities[name])

    action = tracks.get("action_planning_cem") if isinstance(tracks, Mapping) else None
    retention = tracks.get("original_task_retention_cem") if isinstance(tracks, Mapping) else None
    semantic_contract = bool(
        protocol.get("cem_preregistration_id") == "tworoom_speed_pldm_cem_prereg_v1"
        and protocol.get("status") == "frozen_prepublic_cem_execution_and_decision_authority"
        and protocol.get("completion") == binding.get("completion")
        and protocol.get("release") == binding.get("release")
        and protocol.get("behavioral_claim_boundary")
        == binding.get("behavioral_claim_boundary")
        and protocol.get("normalizer") == binding.get("normalizer")
        and protocol.get("stable_worldmodel") == binding.get("stable_worldmodel")
        and protocol.get("preregistration") == source_identities["preregistration"]
        and isinstance(implementation, Mapping)
        and implementation
        == {
            "formal_runner": source_identities["implementation_formal_runner"],
            "aggregate_freezer": source_identities["implementation_aggregate_freezer"],
            "binding_freezer": source_identities["implementation_binding_freezer"],
            "adapter_boundary": source_identities["implementation_adapter_boundary"],
            "development_contract": source_identities["implementation_development_contract"],
            "paired_retention_comparator": source_identities[
                "implementation_paired_retention_comparator"
            ],
        }
        and isinstance(action, Mapping)
        and action.get("evaluation_kind") == "action_planning_cem"
        and action.get("source", {}).get("catalog") == source_identities["action_catalog"]
        and action.get("source", {}).get("catalog_validator")
        == source_identities["action_catalog_validator"]
        and action.get("source", {}).get("episode_oracle")
        == source_identities["action_episode_oracle"]
        and action.get("source", {}).get("runner_core")
        == source_identities["action_runner_core"]
        and action.get("protocol")
        == {
            "eval_budget_raw_steps": 100,
            "deadline_budgets_raw_steps": [50, 75, 100],
            "horizon_action_blocks": 10,
            "receding_horizon_action_blocks": 5,
            "cem_samples": 300,
            "cem_iterations": 30,
            "cem_topk": 30,
            "cem_var_scale": 1.0,
        }
        and action.get("metric", {}).get("result_semantics")
        == "EXECUTED_VALID_DESCRIPTIVE"
        and action.get("metric", {}).get("performance_threshold") is None
        and action.get("metric", {}).get("pass_threshold") is None
        and isinstance(retention, Mapping)
        and retention.get("evaluation_kind") == "original_task_retention_cem"
        and retention.get("source", {}).get("catalog") == source_identities["retention_catalog"]
        and retention.get("source", {}).get("catalog_builder")
        == source_identities["retention_catalog_builder"]
        and retention.get("source", {}).get("episode_oracle")
        == source_identities["retention_episode_oracle"]
        and retention.get("source", {}).get("runner_core")
        == source_identities["retention_runner_core"]
        and retention.get("source", {}).get("frozen_baseline_wrapper")
        == source_identities["retention_frozen_baseline_wrapper"]
        and retention.get("protocol")
        == {
            "eval_budget_raw_steps": 50,
            "horizon_action_blocks": 5,
            "receding_horizon_action_blocks": 5,
            "cem_samples": 300,
            "cem_iterations": 30,
            "cem_topk": 30,
        }
        and retention.get("metric", {}).get("paired_noninferiority", {}).get(
            "success_rate_delta_lower_bound"
        )
        == -0.05
        and retention.get("metric", {}).get("paired_noninferiority", {}).get(
            "final_distance_delta_upper_bound_px"
        )
        == 5.0
        and retention.get("metric", {}).get("paired_noninferiority", {}).get(
            "stratum_definition"
        )
        == "room_relation"
        and retention.get("metric", {}).get("paired_noninferiority", {}).get(
            "require_no_solvable_room_relation_stratum_collapse"
        )
        is True
        and isinstance(outputs, Mapping)
        and outputs.get("cem_binding")
        == "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1/formal_icl_v1/cem_binding_v1.json"
        and outputs.get("action_planning", {}).get("receipts") == "seed_{training_seed}.jsonl"
        and outputs.get("original_task_retention", {}).get("receipts")
        == "seed_{training_seed}.jsonl"
        and protocol.get("authority")
        == {
            "all_source_identities_frozen_before_public_icl": True,
            "post_icl_cem_binding_may_only_validate_and_rebind_this_closure": True,
            "action_planning_outcomes_are_descriptive_not_a_model_gate": True,
            "retention_pass_fail_uses_only_frozen_paired_noninferiority": True,
        }
    )
    _record(checks, "prepublic_cem_semantic_contract", semantic_contract)
    if not semantic_contract:
        raise ValueError("Binding CEM authority has an invalid execution or decision contract")
    return dict(protocol)


def _strict_checkpoint_check(
    *,
    binding: dict[str, Any],
    entry: dict[str, Any],
    checks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    seed = int(entry["seed"])
    checkpoint = _file_check(checks, f"checkpoint_{seed}", entry["checkpoint"])
    config = _file_check(checks, f"checkpoint_config_{seed}", entry["config"])
    report_path = _file_check(checks, f"training_report_{seed}", entry["training_report"])
    trace_path = _file_check(checks, f"loss_trace_{seed}", entry["loss_trace"])
    preflight_path = _file_check(checks, f"preflight_{seed}", entry["preflight"])
    report = _load_json(report_path)
    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    preflight = _load_json(preflight_path)
    training = report.get("training", {})
    artifacts = report.get("artifacts", {})
    fixed_steps = int(binding["completion"]["fixed_optimizer_steps"])
    report_contract = bool(
        report.get("schema_version") == 1
        and report.get("passed") is True
        and report.get("run_kind") == "confirmation"
        and report.get("profile") == "additive"
        and report.get("model_id") == binding["completion"]["model_id"]
        and report.get("run_name") == entry["run_name"]
        and report.get("model", {}).get("training_method") == "pldm"
        and report.get("model", {}).get("history_size") == 3
        and report.get("model", {}).get("action_block") == 5
        and training.get("training_complete") is True
        and int(training.get("global_step", -1)) == fixed_steps
        and int(training.get("expected_optimizer_steps", -1)) == fixed_steps
        and artifacts.get("pretrained") == str(checkpoint)
        and artifacts.get("pretrained_sha256") == entry["checkpoint"]["sha256"]
        and artifacts.get("pretrained_config") == str(config)
        and artifacts.get("pretrained_config_sha256") == entry["config"]["sha256"]
        and artifacts.get("loss_trace", {}).get("sha256") == entry["loss_trace"]["sha256"]
        and int(artifacts.get("loss_trace", {}).get("last_optimizer_step", -1))
        == fixed_steps
        and bool(report.get("save_load_exact"))
    )
    _record(
        checks,
        f"fixed_training_report_{seed}",
        report_contract,
        report=report,
    )
    trace_contract = bool(
        trace_rows
        and int(trace_rows[-1].get("optimizer_step", -1)) == fixed_steps
        and [int(row["optimizer_step"]) for row in trace_rows]
        == sorted({int(row["optimizer_step"]) for row in trace_rows})
    )
    _record(
        checks,
        f"fixed_loss_trace_{seed}",
        trace_contract,
        records=len(trace_rows),
        final_optimizer_step=(
            int(trace_rows[-1].get("optimizer_step", -1)) if trace_rows else None
        ),
    )
    preflight_contract = bool(
        preflight.get("completion_id") == COMPLETION_ID
        and preflight.get("status") == "passed"
        and preflight.get("seed") == seed
        and preflight.get("training_started") is True
        and preflight.get("training_completed") is True
        and preflight.get("training_failed") is not True
        and preflight.get("strict_load", {}).get("model_state_sha256")
        == binding["completion"]["initial_model_state_sha256"]
    )
    _record(
        checks,
        f"training_preflight_{seed}",
        preflight_contract,
        preflight=preflight,
    )
    runtime = binding["stable_worldmodel"]
    adapter = StableWorldModelPLDMAdapter.from_checkpoint(
        checkpoint,
        normalizer=_resolve(binding["normalizer"]["path"]),
        repo_root=ROOT,
        stablewm_repo=str(runtime["worktree"]),
        stablewm_ref=str(runtime["expected_ref"]),
        device="cpu",
    )
    observed_state = adapter.frozen_state_hash()
    metadata = adapter.metadata
    strict_contract = bool(
        observed_state == entry["checkpoint"]["model_state_sha256"]
        and metadata.get("checkpoint_sha256") == entry["checkpoint"]["sha256"]
        and metadata.get("stable_worldmodel_commit") == runtime["expected_ref"]
        and metadata.get("adapter_id") == "stable_worldmodel_pldm_v1"
        and metadata.get("protocol", {}).get("history_tokens") == 3
        and metadata.get("protocol", {}).get("action_block_raw_steps") == 5
        and metadata.get("protocol", {}).get("action_dim") == 2
        and metadata.get("protocol", {}).get("future_action_blocks") >= 5
    )
    _record(
        checks,
        f"strict_cpu_checkpoint_load_{seed}",
        strict_contract,
        model_state_sha256=observed_state,
        metadata=metadata,
    )
    return {
        "seed": seed,
        "checkpoint": entry["checkpoint"],
        "config": entry["config"],
        "training_report": entry["training_report"],
        "loss_trace": entry["loss_trace"],
        "preflight": entry["preflight"],
        "strict_load": {"state_sha256": observed_state, "metadata": metadata},
    }


def _validate_development_evidence(
    *, binding: dict[str, Any], checks: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Require all three no-score Development receipts before Public binding.

    The binding freezer still does not open Public payloads.  It merely verifies
    that the frozen manifest and all immutable per-seed readiness receipts were
    completed against the final fixed-step checkpoint set.
    """

    development = binding.get("development")
    if not isinstance(development, dict) or set(development) != {
        "config",
        "manifest",
        "receipts",
    }:
        raise ValueError("Binding lacks a complete Development evidence declaration")
    config_path = _file_check(checks, "development_config", development["config"])
    manifest_path = _file_check(checks, "development_manifest", development["manifest"])
    development_config = _load_yaml(config_path)
    manifest = _load_json(manifest_path)
    config_identity = {
        "path": _logical(config_path),
        "sha256": _sha256(config_path),
        "size_bytes": int(config_path.stat().st_size),
    }
    manifest_identity = {
        "path": _logical(manifest_path),
        "sha256": _sha256(manifest_path),
        "size_bytes": int(manifest_path.stat().st_size),
    }
    manifest_contract = bool(
        development_config.get("development_id") == DEVELOPMENT_ID
        and development_config.get("completion_id") == COMPLETION_ID
        and development_config.get("scope") == DEVELOPMENT_SCOPE
        and manifest.get("schema_version") == 1
        and manifest.get("development_id") == DEVELOPMENT_ID
        and manifest.get("completion_id") == COMPLETION_ID
        and manifest.get("status") == "frozen_prepublic_development_manifest"
        and manifest.get("passed") is True
        and manifest.get("scope") == DEVELOPMENT_SCOPE
        and manifest.get("development_config") == config_identity
        and manifest.get("public_payload_accessed") is False
        and manifest.get("formal_public_or_cem_artifacts_present") is False
        and manifest.get("coverage", {}).get("validation_scenarios") == 96
        and manifest.get("coverage", {}).get("total_samples") == 384
        and manifest.get("coverage", {}).get("all_actual_indices_unique_per_scenario")
        is True
        and manifest.get("coverage", {}).get("all_source_spans_continuous")
        is True
    )
    _record(
        checks,
        "development_manifest_prepublic_contract",
        manifest_contract,
        development_scope=DEVELOPMENT_SCOPE,
        manifest=manifest_identity,
    )

    rows = development["receipts"]
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_SEEDS):
        raise ValueError("Binding requires exactly three Development receipts")
    if tuple(sorted(int(row.get("seed", -1)) for row in rows if isinstance(row, dict))) != EXPECTED_SEEDS:
        raise ValueError("Binding Development receipts use the wrong seed set")
    checkpoint_by_seed = {
        int(row["seed"]): row for row in binding.get("checkpoints", []) if isinstance(row, dict)
    }
    receipts = []
    for row in sorted(rows, key=lambda item: int(item["seed"])):
        if not isinstance(row, dict) or set(row) != {"seed", "receipt"}:
            raise ValueError("Development receipt declaration fields are invalid")
        seed = int(row["seed"])
        checkpoint_entry = checkpoint_by_seed.get(seed)
        if checkpoint_entry is None:
            raise ValueError(f"Development receipt seed is absent from checkpoints: {seed}")
        expected_checkpoint_path = _resolve(checkpoint_entry["checkpoint"]["path"])
        expected_checkpoint_identity = {
            "path": _logical(expected_checkpoint_path),
            "sha256": _sha256(expected_checkpoint_path),
            "size_bytes": int(expected_checkpoint_path.stat().st_size),
        }
        receipt_path = _file_check(checks, f"development_receipt_{seed}", row["receipt"])
        receipt_identity = {
            "path": _logical(receipt_path),
            "sha256": _sha256(receipt_path),
            "size_bytes": int(receipt_path.stat().st_size),
        }
        receipt = _load_json(receipt_path)
        receipt_checks = receipt.get("checks", {})
        expected_state = checkpoint_entry["checkpoint"].get("model_state_sha256")
        receipt_contract = bool(
            receipt.get("schema_version") == 1
            and receipt.get("development_id") == DEVELOPMENT_ID
            and receipt.get("completion_id") == COMPLETION_ID
            and int(receipt.get("seed", -1)) == seed
            and receipt.get("status") == "passed_infrastructure_readiness"
            and receipt.get("passed") is True
            and receipt.get("scope") == DEVELOPMENT_SCOPE
            and receipt.get("development_config") == config_identity
            and receipt.get("development_manifest") == manifest_identity
            and receipt.get("checkpoint") == expected_checkpoint_identity
            and receipt.get("checkpoint_model_state_sha256") == expected_state
            and isinstance(receipt_checks, dict)
            and all(
                receipt_checks.get(name, {}).get("passed") is True
                for name in (
                    "strict_native_checkpoint_load",
                    "complete_heldout_manifest_coverage",
                    "prefix_autoregressive_geometry",
                    "native_future_latent_mse_finiteness",
                    "frozen_weight_audit",
                    "public_boundary",
                )
            )
            and receipt_checks.get("complete_heldout_manifest_coverage", {}).get("samples")
            == 384
            and receipt_checks.get("complete_heldout_manifest_coverage", {}).get("scenarios")
            == 96
            and receipt_checks.get("native_future_latent_mse_finiteness", {}).get(
                "mse_value_withheld_not_a_score"
            )
            is True
            and receipt_checks.get("frozen_weight_audit", {}).get("state_hash_before")
            == expected_state
            and receipt_checks.get("frozen_weight_audit", {}).get("state_hash_after")
            == expected_state
            and receipt_checks.get("public_boundary", {}).get("public_payload_accessed")
            is False
            and receipt_checks.get("public_boundary", {}).get("checkpoint_selection")
            is False
            and receipt_checks.get("public_boundary", {}).get("scoreboard_score_emitted")
            is False
        )
        _record(
            checks,
            f"development_receipt_contract_{seed}",
            receipt_contract,
            receipt=receipt_identity,
            development_scope=DEVELOPMENT_SCOPE,
        )
        receipts.append({"seed": seed, "receipt": receipt_identity})
    return {
        "config": config_identity,
        "manifest": manifest_identity,
        "receipts": receipts,
    }


def build_receipt(binding_path: Path) -> dict[str, Any]:
    binding = _load_yaml(binding_path)
    checks: dict[str, dict[str, Any]] = {}
    checkpoint_entries = binding.get("checkpoints")
    checkpoint_seeds = (
        [entry.get("seed") for entry in checkpoint_entries if isinstance(entry, dict)]
        if isinstance(checkpoint_entries, list)
        else []
    )
    _record(
        checks,
        "binding_shape",
        bool(
            binding.get("schema_version") == 1
            and binding.get("binding_id")
            == "tworoom_speed_pldm_evaluation_binding_v1"
            and binding.get("status")
            == "preregistered_after_training_before_formal_public_evaluation"
            and binding.get("completion", {}).get("completion_id") == COMPLETION_ID
            and binding.get("completion", {}).get("training_seeds")
            == [3072, 4096, 5120]
            and isinstance(checkpoint_entries, list)
            and len(checkpoint_entries) == 3
            and len(checkpoint_seeds) == 3
            and set(checkpoint_seeds) == {3072, 4096, 5120}
            and isinstance(binding.get("development"), dict)
        ),
        checkpoint_seeds=checkpoint_seeds,
    )
    completion_path = _file_check(checks, "completion_config", binding["completion"])
    normalizer_path = _file_check(checks, "normalizer", binding["normalizer"])
    completion = _load_yaml(completion_path)
    _record(
        checks,
        "completion_identity",
        bool(
            completion.get("completion_id") == COMPLETION_ID
            and completion.get("training", {}).get("seeds")
            == binding["completion"]["training_seeds"]
            and int(completion.get("training", {}).get("optimizer_steps", -1))
            == int(binding["completion"]["fixed_optimizer_steps"])
            and completion.get("training", {}).get("model_id")
            == binding["completion"]["model_id"]
        ),
    )
    development = _validate_development_evidence(binding=binding, checks=checks)
    # A release configuration is declarative metadata only at this point; no
    # Public payload is opened until the passed binding receipt is later used
    # by the formal evaluator.
    release_path = _file_check(checks, "release_config", binding["release"])
    release = _load_yaml(release_path)
    runtime = binding["stable_worldmodel"]
    runtime_root = _resolve(runtime["worktree"])
    pldm_config = runtime_root / runtime["pldm_config"]
    _record(
        checks,
        "pinned_stable_worldmodel",
        bool(
            _git_head(runtime_root) == runtime["expected_ref"]
            and _sha256(pldm_config) == runtime["pldm_config_sha256"]
        ),
        worktree=str(runtime_root),
        expected_ref=runtime["expected_ref"],
        observed_ref=_git_head(runtime_root),
        pldm_config=str(pldm_config),
        pldm_config_sha256=_sha256(pldm_config),
    )
    _record(
        checks,
        "release_identity_and_public_declaration",
        bool(
            release.get("release_id") == binding["release"]["release_id"]
            and release.get("scope", {}).get("public_test_included") is True
            and release.get("scope", {}).get("sealed_test_included") is False
            and release.get("runtime", {}).get("stable_worldmodel", {}).get(
                "expected_ref"
            )
            == runtime["expected_ref"]
            and release.get("scope", {}).get("public_tracks")
            == binding["formal_icl"]["tracks"]
        ),
        public_payload_accessed=False,
    )
    _record(
        checks,
        "normalizer_identity_matches_release",
        bool(
            normalizer_path.is_file()
            and release.get("evaluation", {}).get("normalizer_sha256")
            == binding["normalizer"]["sha256"]
        ),
    )
    cem_protocol = _validate_prepublic_cem_protocol(binding=binding, checks=checks)
    for name, specification in binding["evaluator_sources"].items():
        _file_check(checks, f"evaluator_source_{name}", specification)
    behavioral_claim_boundary = _validated_behavioral_claim_boundary(
        binding=binding, checks=checks
    )
    checkpoint_receipts = [
        _strict_checkpoint_check(binding=binding, entry=entry, checks=checks)
        for entry in binding["checkpoints"]
    ]
    formal_paths = [
        _resolve(binding["artifacts"]["formal_icl_root"]),
        _resolve(binding["artifacts"]["action_planning_root"]),
        _resolve(binding["artifacts"]["retention_root"]),
    ]
    _record(
        checks,
        "formal_evaluation_not_started_before_binding",
        not any(path.exists() for path in formal_paths),
        paths=[_logical(path) for path in formal_paths],
    )
    passed = all(row["passed"] for row in checks.values())
    return {
        "schema_version": 1,
        "binding_id": binding["binding_id"],
        "status": (
            "passed_evaluation_binding_freeze"
            if passed
            else "failed_evaluation_binding_freeze"
        ),
        "passed": passed,
        "binding": {"path": _logical(binding_path), "sha256": _sha256(binding_path)},
        "completion": binding["completion"],
        "release": binding["release"],
        "development": development,
        "behavioral_claim_boundary": behavioral_claim_boundary,
        "cem_protocol": cem_protocol,
        "claim_boundary": _claim_scope(),
        "stable_worldmodel": binding["stable_worldmodel"],
        "public_test": {
            "accessed_by_binding": False,
            "scored_by_binding": False,
            "declarative_tracks": binding["formal_icl"]["tracks"],
        },
        "checkpoints": checkpoint_receipts,
        "checks": checks,
        "next_stage": (
            {
                "formal_public_icl_authorized": True,
                "action_planning_cem_authorized_after_three_seed_icl_gate": True,
                "original_tworoom_retention_cem_authorized_after_three_seed_icl_gate": True,
            }
            if passed
            else {
                "formal_public_icl_authorized": False,
                "action_planning_cem_authorized": False,
                "original_tworoom_retention_cem_authorized": False,
            }
        ),
    }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
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


def _assert_output(path: Path) -> Path:
    expected = DEFAULT_OUTPUT.resolve()
    actual = path.resolve()
    if actual != expected:
        raise ValueError(
            "Binding output must equal its dedicated destination "
            f"{_logical(expected)}"
        )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    binding_path = _resolve(args.binding)
    output = _assert_output(_resolve(args.output))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite binding receipt: {output}")
    receipt = build_receipt(binding_path)
    _write_exclusive(output, receipt)
    print(json.dumps({"status": receipt["status"], "output": _logical(output)}))
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
