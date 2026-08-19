#!/usr/bin/env python3
"""Freeze Speed's post-raw-Public, pre-recovery preregistration.

This is deliberately a one-way bridge: it is allowed to read the three
already-written Public ICL receipts, but it cannot run recovery, rewrite a raw
receipt, choose a checkpoint, or enter either CEM namespace.  The resulting
YAML pins every raw input before the independent recovery script starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from contextworld.benchmarks.speed_pldm_infrastructure_development import (
    COMPLETION_ID,
    DEVELOPMENT_ID,
    DEVELOPMENT_SCOPE,
    EXPECTED_SEEDS,
    identity,
    logical_path,
    resolve_local_output,
    resolve_source,
    root,
)


ROOT = root()
BINDING_CONFIG = ROOT / "configs/benchmark/tworoom_speed_pldm_evaluation_binding_v1.yaml"
BINDING_RECEIPT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "evaluation_binding_v1/evaluation_binding_receipt.json"
)
DEFAULT_OUTPUT = ROOT / "configs/benchmark/tworoom_speed_pldm_formal_icl_recovery_v1.yaml"
FORMAL_ROOT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_icl_v1"
)
RECOVERY_ROOT = FORMAL_ROOT / "recovery_v1"
AGGREGATE_OUTPUT = FORMAL_ROOT / "three_seed_aggregate.json"
PLANNED_CEM_ROOTS = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_action_planning_cem_v1",
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_original_tworoom_retention_cem_v1",
)
RECOVERY_ID = "tworoom_speed_pldm_reference_completion_recovery_v1"


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


def _identity(path: Path) -> dict[str, Any]:
    return identity(path, repo_root=ROOT)


def _source(path: str) -> dict[str, Any]:
    return _identity(resolve_source(path, repo_root=ROOT))


def _same_path_sha(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    left_path = left.get("path")
    right_path = right.get("path")
    if not isinstance(left_path, str) or not isinstance(right_path, str):
        return False
    try:
        same_file = resolve_source(left_path, repo_root=ROOT) == resolve_source(
            right_path, repo_root=ROOT
        )
    except (TypeError, ValueError):
        return False
    return bool(same_file and left.get("sha256") == right.get("sha256"))


def _write_yaml_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(
                payload,
                stream,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _assert_output(path: Path) -> Path:
    expected = DEFAULT_OUTPUT.resolve()
    actual = resolve_local_output(path, repo_root=ROOT)
    if actual != expected:
        raise ValueError(
            "Recovery preregistration output must equal its dedicated destination "
            f"{logical_path(expected, repo_root=ROOT)}"
        )
    return actual


def _development_chain(binding: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    declared = binding.get("development")
    frozen = receipt.get("development")
    if not (
        isinstance(declared, dict)
        and set(declared) == {"config", "manifest", "receipts"}
        and isinstance(frozen, dict)
        and set(frozen) == {"config", "manifest", "receipts"}
    ):
        raise RuntimeError("Binding lacks a complete Speed Development chain")
    config_path = resolve_source(declared["config"].get("path", ""), repo_root=ROOT)
    manifest_path = resolve_source(declared["manifest"].get("path", ""), repo_root=ROOT)
    config = _identity(config_path)
    manifest = _identity(manifest_path)
    if not (frozen.get("config") == config and frozen.get("manifest") == manifest):
        raise RuntimeError("Binding receipt Development config/manifest identities drifted")
    config_payload = _load_yaml(config_path)
    manifest_payload = _load_json(manifest_path)
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
        and manifest_payload.get("formal_public_or_cem_artifacts_present") is False
    ):
        raise RuntimeError("Binding's Development config/manifest contract is invalid")
    rows = declared["receipts"]
    frozen_rows = frozen["receipts"]
    if not (
        isinstance(rows, list)
        and isinstance(frozen_rows, list)
        and tuple(sorted(int(row.get("seed", -1)) for row in rows)) == EXPECTED_SEEDS
        and tuple(sorted(int(row.get("seed", -1)) for row in frozen_rows))
        == EXPECTED_SEEDS
    ):
        raise RuntimeError("Binding Development receipt seed set is invalid")
    checkpoints = {
        int(row["seed"]): row
        for row in binding.get("checkpoints", [])
        if isinstance(row, dict) and isinstance(row.get("seed"), int)
    }
    if set(checkpoints) != set(EXPECTED_SEEDS):
        raise RuntimeError("Binding checkpoint set is invalid")
    observed = []
    for row in sorted(rows, key=lambda item: int(item["seed"])):
        if not isinstance(row, dict) or set(row) != {"seed", "receipt"}:
            raise RuntimeError("Binding Development receipt declaration is invalid")
        seed = int(row["seed"])
        receipt_path = resolve_source(row["receipt"].get("path", ""), repo_root=ROOT)
        receipt_identity = _identity(receipt_path)
        matching = [item for item in frozen_rows if int(item.get("seed", -1)) == seed]
        if len(matching) != 1 or matching[0].get("receipt") != receipt_identity:
            raise RuntimeError("Binding receipt does not preserve a Development receipt")
        payload = _load_json(receipt_path)
        checks = payload.get("checks", {})
        checkpoint = checkpoints[seed]["checkpoint"]
        if not (
            payload.get("development_id") == DEVELOPMENT_ID
            and payload.get("completion_id") == COMPLETION_ID
            and payload.get("seed") == seed
            and payload.get("status") == "passed_infrastructure_readiness"
            and payload.get("passed") is True
            and payload.get("scope") == DEVELOPMENT_SCOPE
            and payload.get("development_config") == config
            and payload.get("development_manifest") == manifest
            and _same_path_sha(payload.get("checkpoint"), checkpoint)
            and payload.get("checkpoint_model_state_sha256")
            == checkpoint.get("model_state_sha256")
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
            and checks.get("public_boundary", {}).get("public_payload_accessed") is False
            and checks.get("public_boundary", {}).get("checkpoint_selection") is False
            and checks.get("public_boundary", {}).get("scoreboard_score_emitted") is False
        ):
            raise RuntimeError(f"Development receipt is invalid for seed {seed}")
        observed.append({"seed": seed, "receipt": receipt_identity})
    if any(
        receipt.get("checks", {}).get(name, {}).get("passed") is not True
        for name in (
            "development_manifest_prepublic_contract",
            *(f"development_receipt_contract_{seed}" for seed in EXPECTED_SEEDS),
        )
    ):
        raise RuntimeError("Binding receipt did not pass the complete Development gate")
    return {"config": config, "manifest": manifest, "receipts": observed}


def build_preregistration(
    *, binding_config_path: Path = BINDING_CONFIG, binding_receipt_path: Path = BINDING_RECEIPT
) -> dict[str, Any]:
    binding_config_path = resolve_source(binding_config_path, repo_root=ROOT)
    binding_receipt_path = resolve_source(binding_receipt_path, repo_root=ROOT)
    binding = _load_yaml(binding_config_path)
    binding_receipt = _load_json(binding_receipt_path)
    binding_identity = _identity(binding_config_path)
    receipt_identity = _identity(binding_receipt_path)
    if not (
        binding.get("schema_version") == 1
        and binding.get("binding_id") == "tworoom_speed_pldm_evaluation_binding_v1"
        and binding.get("status")
        == "preregistered_after_training_before_formal_public_evaluation"
        and binding.get("completion", {}).get("completion_id") == COMPLETION_ID
        and binding_receipt.get("status") == "passed_evaluation_binding_freeze"
        and binding_receipt.get("passed") is True
        and _same_path_sha(binding_receipt.get("binding"), binding_identity)
    ):
        raise RuntimeError("Passed Speed evaluation binding is not intact")
    development = _development_chain(binding, binding_receipt)
    completion_path = resolve_source(binding["completion"].get("path", ""), repo_root=ROOT)
    release_path = resolve_source(binding["release"].get("path", ""), repo_root=ROOT)
    boundary_path = resolve_source(
        binding["behavioral_claim_boundary"].get("path", ""), repo_root=ROOT
    )
    completion_identity = _identity(completion_path)
    release_identity = _identity(release_path)
    boundary_identity = _identity(boundary_path)
    completion = _load_yaml(completion_path)
    release = _load_yaml(release_path)
    if not (
        completion.get("completion_id") == COMPLETION_ID
        and completion.get("training", {}).get("seeds") == list(EXPECTED_SEEDS)
        and release.get("release_id") == binding["release"].get("release_id")
        and _same_path_sha(binding["completion"], completion_identity)
        and _same_path_sha(binding["release"], release_identity)
        and _same_path_sha(binding["behavioral_claim_boundary"], boundary_identity)
    ):
        raise RuntimeError("Binding completion/release/boundary inputs drifted")
    if RECOVERY_ROOT.exists() or AGGREGATE_OUTPUT.exists() or any(
        path.exists() for path in PLANNED_CEM_ROOTS
    ):
        raise RuntimeError("Recovery/CEM artifacts exist before raw-Public recovery preregistration")

    checkpoints = {
        int(row["seed"]): row
        for row in binding.get("checkpoints", [])
        if isinstance(row, dict) and isinstance(row.get("seed"), int)
    }
    if set(checkpoints) != set(EXPECTED_SEEDS):
        raise RuntimeError("Binding checkpoint seeds are incomplete")
    raw_entries = []
    for seed in EXPECTED_SEEDS:
        raw_path = FORMAL_ROOT / f"seed_{seed}.json"
        raw_identity = _identity(raw_path)
        raw = _load_json(raw_path)
        checkpoint = checkpoints[seed]["checkpoint"]
        state = checkpoint.get("model_state_sha256")
        completion_evaluation = raw.get("completion_evaluation", {})
        gate = completion_evaluation.get("primary_gate", {})
        scope = {
            "public_icl_evaluated": True,
            "action_planning_cem_executed": False,
            "original_tworoom_retention_cem_executed": False,
            "checkpoint_selection_performed": False,
            "paired_single_speed_control_available": False,
            "training_attribution_claim": False,
            "public_test_reopened": False,
            "claim_level": "behavioral_trained_reference_only",
        }
        if not (
            raw.get("schema_version") == 1
            and raw.get("benchmark") == release.get("release_id")
            and raw.get("submission_kind") == "single_model"
            and raw.get("status") == "passed"
            and raw.get("full_protocol") is True
            and _same_path_sha(raw.get("release_config"), release_identity)
            and raw.get("model", {}).get("training_seed") == seed
            and raw.get("model", {}).get("training_role") == "multi_speed_target"
            and raw.get("model", {}).get("checkpoint_sha256") == checkpoint.get("sha256")
            and raw.get("frozen_weight_audit", {}).get("state_hash_before") == state
            and raw.get("frozen_weight_audit", {}).get("state_hash_after") == state
            and raw.get("frozen_weight_audit", {}).get("passed") is True
            and completion_evaluation.get("completion_id") == COMPLETION_ID
            and _same_path_sha(completion_evaluation.get("binding"), binding_identity)
            and _same_path_sha(completion_evaluation.get("binding_receipt"), receipt_identity)
            and _same_path_sha(completion_evaluation.get("checkpoint"), checkpoint)
            and completion_evaluation.get("checkpoint_model_state_sha256") == state
            and completion_evaluation.get("development") == development
            and completion_evaluation.get("scope") == scope
            and gate.get("id") == "unseen_in_range_one_step_strict_history_accuracy"
            and type(gate.get("passed")) is bool
            and set(raw.get("tracks", {})) == set(release.get("scope", {}).get("public_tracks", []))
        ):
            raise RuntimeError(f"Raw Public ICL contract is invalid for seed {seed}")
        raw_entries.append(
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint["sha256"],
                "model_state_sha256": state,
                "raw_result": raw_identity,
                "raw_gate_passed": gate["passed"],
                "recovery_receipt": {
                    "path": logical_path(RECOVERY_ROOT / f"seed_{seed}.json", repo_root=ROOT)
                },
            }
        )
    scorer = _source("contextworld/benchmarks/speed_icl_score.py")
    if not _same_path_sha(
        binding.get("evaluator_sources", {}).get("speed_icl_score"), scorer
    ):
        raise RuntimeError("Binding's frozen Speed scorer identity drifted")
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "completion_id": COMPLETION_ID,
        "release_id": release["release_id"],
        "status": "preregistered_after_raw_public_icl_before_recovery",
        "frozen_inputs": {
            "completion_config": completion_identity,
            "evaluation_binding_config": binding_identity,
            "evaluation_binding_receipt": receipt_identity,
            "release_config": release_identity,
            "behavioral_claim_boundary": boundary_identity,
            "development_config": development["config"],
            "development_manifest": development["manifest"],
            **{
                f"development_receipt_{row['seed']}": row["receipt"]
                for row in development["receipts"]
            },
        },
        "implementation": {
            "frozen_icl_scorer": scorer,
            "recovery_launcher": _source(
                "scripts/recover_tworoom_speed_pldm_formal_icl_v1.py"
            ),
            "formal_icl_evaluator": _source(
                "scripts/eval_tworoom_speed_pldm_formal_icl_v1.py"
            ),
            "development_contract": _source(
                "contextworld/benchmarks/speed_pldm_infrastructure_development.py"
            ),
        },
        "raw_public_icl": {"checkpoints": raw_entries},
        "outputs": {
            "root": logical_path(RECOVERY_ROOT, repo_root=ROOT),
            "aggregate": {
                "path": logical_path(AGGREGATE_OUTPUT, repo_root=ROOT),
                "content_sha256_not_embedded_to_avoid_self_reference": True,
            },
        },
        "scope": {
            "raw_public_icl_already_completed": True,
            "recovery_only": True,
            "model_or_environment_execution_authorized": False,
            "checkpoint_selection_authorized": False,
            "cem_authorized_by_this_record": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, default=BINDING_CONFIG)
    parser.add_argument("--binding-receipt", type=Path, default=BINDING_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = _assert_output(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite recovery preregistration: {output}")
    payload = build_preregistration(
        binding_config_path=args.binding, binding_receipt_path=args.binding_receipt
    )
    _write_yaml_exclusive(output, payload)
    print(
        json.dumps(
            {
                "recovery_id": payload["recovery_id"],
                "status": payload["status"],
                "output": logical_path(output, repo_root=ROOT),
                "cem_executed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
