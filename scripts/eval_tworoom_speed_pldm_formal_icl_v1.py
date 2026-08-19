#!/usr/bin/env python3
"""Run one bound Public ICL evaluation for the Speed PLDM completion.

The generic Speed scorer remains unchanged.  This narrow launcher adds only
completion-specific binding checks, an x-exclusive output policy, and the
pre-registered one-seed gate used to authorize (but not execute) later CEM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from contextworld.benchmarks.adapters import StableWorldModelPLDMAdapter
from contextworld.benchmarks.speed_pldm_infrastructure_development import (
    DEVELOPMENT_ID,
    DEVELOPMENT_SCOPE,
    EXPECTED_SEEDS,
)
from contextworld.benchmarks.speed_icl_data import load_speed_icl_release
from contextworld.benchmarks.speed_icl_score import evaluate_speed_icl_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINDING = (
    ROOT / "configs/benchmark/tworoom_speed_pldm_evaluation_binding_v1.yaml"
)
DEFAULT_BINDING_RECEIPT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "evaluation_binding_v1/evaluation_binding_receipt.json"
)
COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
PRIMARY_TRACK = "unseen_interpolation"
PRIMARY_HORIZON = "1"
CLAIM_LEVEL = "behavioral_trained_reference_only"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _logical(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"Path must remain inside repository: {path}") from error


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
        raise ValueError(f"Expected JSON mapping: {path}")
    return payload


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


def _identity(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = _sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"Frozen input changed: {path}; expected={expected_sha256}, "
            f"observed={observed}"
        )
    return {
        "path": _logical(path),
        "sha256": observed,
        "size_bytes": int(path.stat().st_size),
    }


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


def _validate_behavioral_claim_boundary(binding: dict[str, Any]) -> dict[str, Any]:
    """Validate the pre-Public boundary before loading a release or adapter."""

    specification = binding.get("behavioral_claim_boundary")
    if not isinstance(specification, dict):
        raise RuntimeError("Speed binding lacks a behavioral claim-boundary identity")
    path = _resolve(specification.get("path", ""))
    identity = _identity(path, str(specification.get("sha256", "")))
    payload = _load_yaml(path)
    chronology = payload.get("chronology")
    claim = payload.get("claim_boundary")
    conditional = payload.get("conditional_evaluation")
    mutation = payload.get("mutation_boundary")
    frozen = payload.get("frozen_inputs")
    scorer = binding.get("evaluator_sources", {}).get("speed_icl_score")
    if not (
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
    ):
        raise RuntimeError("Speed behavioral claim-boundary contract is invalid")
    return identity


def _entry(binding: dict[str, Any], seed: int) -> dict[str, Any]:
    matches = [row for row in binding["checkpoints"] if int(row["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"Seed {seed} is not registered exactly once")
    return matches[0]


def _validate_development_gate(
    binding: dict[str, Any], binding_receipt: dict[str, Any]
) -> dict[str, Any]:
    """Reject Public evaluation unless all no-score Development gates passed."""

    declared = binding.get("development")
    frozen = binding_receipt.get("development")
    if not (
        isinstance(declared, dict)
        and set(declared) == {"config", "manifest", "receipts"}
        and isinstance(frozen, dict)
        and set(frozen) == {"config", "manifest", "receipts"}
    ):
        raise RuntimeError("Speed binding lacks frozen Development evidence")
    config = _identity(
        _resolve(declared["config"].get("path", "")),
        str(declared["config"].get("sha256", "")),
    )
    manifest = _identity(
        _resolve(declared["manifest"].get("path", "")),
        str(declared["manifest"].get("sha256", "")),
    )
    if not (frozen.get("config") == config and frozen.get("manifest") == manifest):
        raise RuntimeError("Speed binding Development identities drifted")
    config_payload = _load_yaml(_resolve(config["path"]))
    manifest_payload = _load_json(_resolve(manifest["path"]))
    if not (
        config_payload.get("development_id") == DEVELOPMENT_ID
        and config_payload.get("completion_id") == COMPLETION_ID
        and config_payload.get("scope") == DEVELOPMENT_SCOPE
        and manifest_payload.get("development_id") == DEVELOPMENT_ID
        and manifest_payload.get("completion_id") == COMPLETION_ID
        and manifest_payload.get("status") == "frozen_prepublic_development_manifest"
        and manifest_payload.get("passed") is True
        and manifest_payload.get("scope") == DEVELOPMENT_SCOPE
        and manifest_payload.get("development_config") == config
        and manifest_payload.get("public_payload_accessed") is False
    ):
        raise RuntimeError("Speed Development manifest is not a valid no-score gate")
    receipt_rows = declared["receipts"]
    frozen_rows = frozen["receipts"]
    if not (
        isinstance(receipt_rows, list)
        and isinstance(frozen_rows, list)
        and tuple(sorted(int(row.get("seed", -1)) for row in receipt_rows))
        == EXPECTED_SEEDS
        and tuple(sorted(int(row.get("seed", -1)) for row in frozen_rows))
        == EXPECTED_SEEDS
    ):
        raise RuntimeError("Speed Development evidence lacks the exact seed set")
    observed_rows = []
    checkpoints_by_seed = {
        int(item["seed"]): item
        for item in binding.get("checkpoints", [])
        if isinstance(item, dict)
    }
    for row in sorted(receipt_rows, key=lambda item: int(item["seed"])):
        if not isinstance(row, dict) or set(row) != {"seed", "receipt"}:
            raise RuntimeError("Speed Development receipt declaration is invalid")
        seed = int(row["seed"])
        identity = _identity(
            _resolve(row["receipt"].get("path", "")),
            str(row["receipt"].get("sha256", "")),
        )
        frozen_match = [item for item in frozen_rows if int(item.get("seed", -1)) == seed]
        if len(frozen_match) != 1 or frozen_match[0].get("receipt") != identity:
            raise RuntimeError("Speed Development receipt identity drifted")
        receipt = _load_json(_resolve(identity["path"]))
        checks = receipt.get("checks", {})
        checkpoint = checkpoints_by_seed.get(seed)
        if checkpoint is None:
            raise RuntimeError("Speed binding checkpoint set is incomplete")
        expected_checkpoint = _identity(
            _resolve(checkpoint["checkpoint"].get("path", "")),
            str(checkpoint["checkpoint"].get("sha256", "")),
        )
        if not (
            receipt.get("development_id") == DEVELOPMENT_ID
            and receipt.get("completion_id") == COMPLETION_ID
            and int(receipt.get("seed", -1)) == seed
            and receipt.get("status") == "passed_infrastructure_readiness"
            and receipt.get("passed") is True
            and receipt.get("scope") == DEVELOPMENT_SCOPE
            and receipt.get("development_config") == config
            and receipt.get("development_manifest") == manifest
            and receipt.get("checkpoint") == expected_checkpoint
            and receipt.get("checkpoint_model_state_sha256")
            == checkpoint["checkpoint"].get("model_state_sha256")
            and isinstance(checks, dict)
            and all(
                checks.get(name, {}).get("passed") is True
                for name in (
                    "strict_native_checkpoint_load",
                    "complete_heldout_manifest_coverage",
                    "prefix_autoregressive_geometry",
                    "native_future_latent_mse_finiteness",
                    "frozen_weight_audit",
                    "public_boundary",
                )
            )
            and checks.get("public_boundary", {}).get("public_payload_accessed")
            is False
            and checks.get("public_boundary", {}).get("checkpoint_selection")
            is False
        ):
            raise RuntimeError("Speed Development receipt does not pass the no-score contract")
        observed_rows.append({"seed": seed, "receipt": identity})
    if binding_receipt.get("checks", {}).get("development_manifest_prepublic_contract", {}).get("passed") is not True:
        raise RuntimeError("Binding receipt did not pass the Development manifest contract")
    if any(
        binding_receipt.get("checks", {}).get(
            f"development_receipt_contract_{seed}", {}
        ).get("passed")
        is not True
        for seed in EXPECTED_SEEDS
    ):
        raise RuntimeError("Binding receipt did not pass every Development receipt")
    return {"config": config, "manifest": manifest, "receipts": observed_rows}


def _expected_output(binding: dict[str, Any], seed: int) -> Path:
    root = _resolve(binding["artifacts"]["formal_icl_root"])
    expected = root / f"seed_{seed}.json"
    try:
        expected.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Formal ICL output escapes its dedicated namespace") from error
    return expected.resolve()


def _validate_binding(
    binding_path: Path,
    receipt_path: Path,
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    binding = _load_yaml(binding_path)
    receipt = _load_json(receipt_path)
    if not (
        binding.get("schema_version") == 1
        and binding.get("binding_id") == "tworoom_speed_pldm_evaluation_binding_v1"
        and binding.get("status")
        == "preregistered_after_training_before_formal_public_evaluation"
        and binding.get("completion", {}).get("completion_id") == COMPLETION_ID
        and receipt.get("status") == "passed_evaluation_binding_freeze"
        and receipt.get("passed") is True
        and receipt.get("binding", {}).get("sha256") == _sha256(binding_path)
    ):
        raise RuntimeError("Speed formal Public ICL requires a passed binding freeze")
    for name, specification in binding["evaluator_sources"].items():
        source = _resolve(specification["path"])
        _identity(source, specification["sha256"])
    cem_protocol = binding.get("cem_protocol")
    receipt_cem_protocol = receipt.get("cem_protocol")
    required_cem_sources = {
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
    if not (
        isinstance(cem_protocol, dict)
        and receipt_cem_protocol == cem_protocol
        and cem_protocol.get("status")
        == "frozen_prepublic_cem_execution_and_decision_authority"
        and isinstance(cem_protocol.get("source_identities"), dict)
        and set(cem_protocol["source_identities"]) == required_cem_sources
        and all(
            binding["evaluator_sources"].get(f"cem_{name}")
            == cem_protocol["source_identities"][name]
            for name in required_cem_sources
        )
        and cem_protocol.get("authority", {}).get(
            "all_source_identities_frozen_before_public_icl"
        )
        is True
    ):
        raise RuntimeError("Speed formal ICL binding lacks a frozen pre-Public CEM closure")
    development = _validate_development_gate(binding, receipt)
    boundary_identity = _validate_behavioral_claim_boundary(binding)
    if not (
        receipt.get("behavioral_claim_boundary") == boundary_identity
        and receipt.get("claim_boundary") == _claim_scope()
    ):
        raise RuntimeError("Passed Speed binding does not preserve the claim boundary")
    entry = _entry(binding, seed)
    _identity(_resolve(entry["checkpoint"]["path"]), entry["checkpoint"]["sha256"])
    binding["_validated_behavioral_claim_boundary"] = boundary_identity
    binding["_validated_development"] = development
    return binding, receipt, entry


def _completion_gate(payload: dict[str, Any]) -> dict[str, Any]:
    track = payload.get("tracks", {}).get(PRIMARY_TRACK, {})
    horizon = track.get("horizons", {}).get(PRIMARY_HORIZON, {})
    value = horizon.get("formal_within_checkpoint_pass")
    if not isinstance(value, bool):
        raise RuntimeError("Formal Speed ICL primary gate was not computed")
    return {
        "id": "unseen_in_range_one_step_strict_history_accuracy",
        "track": PRIMARY_TRACK,
        "horizon_action_blocks": int(PRIMARY_HORIZON),
        "metric_path": (
            "tracks.unseen_interpolation.horizons.1."
            "reference_speed_balanced_strict_query_win_rate_vs_every_other"
        ),
        "value": float(
            horizon["reference_speed_balanced_strict_query_win_rate_vs_every_other"]
        ),
        "passed": value,
    }


def evaluate(
    *,
    binding_path: Path,
    receipt_path: Path,
    seed: int,
    device: str,
    output: Path,
) -> dict[str, Any]:
    binding, binding_receipt, entry = _validate_binding(
        binding_path, receipt_path, seed=seed
    )
    expected_output = _expected_output(binding, seed)
    if output.resolve() != expected_output:
        raise ValueError(
            "Formal ICL output must equal its preregistered destination "
            f"{_logical(expected_output)}"
        )
    planned_paths = [
        _resolve(binding["artifacts"]["action_planning_root"]),
        _resolve(binding["artifacts"]["retention_root"]),
    ]
    if any(path.exists() for path in planned_paths):
        raise RuntimeError("CEM artifacts exist before the three-seed ICL decision")
    release_path = _resolve(binding["release"]["path"])
    _identity(release_path, binding["release"]["sha256"])
    release = load_speed_icl_release(release_path)
    runtime = binding["stable_worldmodel"]
    normalizer = _resolve(release["evaluation"]["normalizer"])
    checkpoint = _resolve(entry["checkpoint"]["path"])
    adapter = StableWorldModelPLDMAdapter.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=str(runtime["worktree"]),
        stablewm_ref=str(runtime["expected_ref"]),
        device=device,
    )
    observed_state = adapter.frozen_state_hash()
    if observed_state != entry["checkpoint"]["model_state_sha256"]:
        raise RuntimeError("Strict-loaded checkpoint state differs from binding")
    payload = evaluate_speed_icl_model(
        adapter=adapter,
        model_name=f"speed_pldm_reference_completion_seed{seed}",
        training_role="multi_speed_target",
        training_seed=seed,
        release_config=release_path,
        repo_root=ROOT,
        tracks=list(release["scope"]["public_tracks"]),
        eval_seeds=[int(value) for value in release["evaluation"]["eval_seeds"]],
        limit_per_reference_speed_per_seed=None,
        encode_batch_size=int(binding["formal_icl"]["encode_batch_size"]),
        rollout_batch_size=int(binding["formal_icl"]["rollout_batch_size"]),
        bundle_batch_size=int(binding["formal_icl"]["bundle_batch_size"]),
        include_records=True,
    )
    gate = _completion_gate(payload)
    if not (
        payload.get("status") == "passed"
        and payload.get("full_protocol") is True
        and payload.get("frozen_weight_audit", {}).get("passed") is True
        and payload.get("model", {}).get("checkpoint_sha256")
        == entry["checkpoint"]["sha256"]
        and payload.get("frozen_weight_audit", {}).get("state_hash_before")
        == entry["checkpoint"]["model_state_sha256"]
        and payload.get("frozen_weight_audit", {}).get("state_hash_after")
        == entry["checkpoint"]["model_state_sha256"]
    ):
        raise RuntimeError("Formal Speed ICL evaluator did not meet its frozen contract")
    payload["completion_evaluation"] = {
        "completion_id": COMPLETION_ID,
        "evaluation_id": "tworoom_speed_pldm_formal_icl_v1",
        "binding": _identity(binding_path, _sha256(binding_path)),
        "binding_receipt": _identity(receipt_path, _sha256(receipt_path)),
        "checkpoint": _identity(checkpoint, entry["checkpoint"]["sha256"]),
        "checkpoint_model_state_sha256": observed_state,
        "development": binding["_validated_development"],
        "behavioral_claim_boundary": binding["_validated_behavioral_claim_boundary"],
        "primary_gate": gate,
        "scope": {
            "public_icl_evaluated": True,
            "action_planning_cem_executed": False,
            "original_tworoom_retention_cem_executed": False,
            "checkpoint_selection_performed": False,
            **_claim_scope(),
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument(
        "--binding-receipt", type=Path, default=DEFAULT_BINDING_RECEIPT
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binding_path = _resolve(args.binding)
    receipt_path = _resolve(args.binding_receipt)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite formal ICL output: {output}")
    payload = evaluate(
        binding_path=binding_path,
        receipt_path=receipt_path,
        seed=int(args.seed),
        device=str(args.device),
        output=output,
    )
    _write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": _logical(output),
                "primary_gate": payload["completion_evaluation"]["primary_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
