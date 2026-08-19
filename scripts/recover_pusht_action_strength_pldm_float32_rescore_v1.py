#!/usr/bin/env python3
"""Narrow, additive float32 recovery for frozen ActionStrength PLDM ICL.

The already-frozen generic scorer rebuilds JSON MSE values as float64, while
the evaluator originally aggregated those arrays as float32.  This launcher
does not alter that scorer or rerun a model.  It validates the preregistered
three raw Public receipts and reconstructs them with the original float32
aggregation semantics into exclusive additive receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any

import yaml

from contextworld.benchmarks.action_strength_icl_data import (
    load_action_strength_icl_release,
)
from contextworld.benchmarks.action_strength_rescore_recovery import (
    _latent_comparison,
    _records,
    _reconstruct_metrics,
    _scalar_comparison,
)
from contextworld.benchmarks.action_strength_icl_score import (
    _prediction_contract_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "configs/benchmark/pusht_action_strength_pldm_float32_rescore_recovery_v1.yaml"
RECOVERY_NAMESPACE = Path(
    "artifacts/evaluation/history3/"
    "pusht_action_strength_pldm_reference_completion_v1/"
    "formal_icl_v1/float32_rescore_recovery_v1"
)


def _resolve(value: str | Path, *, label: str = "path") -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the repository") from error
    return resolved


def _logical_path(path: Path, *, label: str = "path") -> str:
    """Return the canonical repository-relative spelling for a path."""

    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the repository") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
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


def _file_matches(specification: dict[str, Any]) -> tuple[Path, bool, str | None]:
    if not isinstance(specification, dict) or not isinstance(
        specification.get("path"), str
    ) or not isinstance(specification.get("sha256"), str):
        raise ValueError("Every frozen file specification needs path and sha256")
    path = _resolve(specification["path"], label="frozen input")
    observed = _sha256(path) if path.is_file() else None
    return path, observed == specification["sha256"], observed


def _identity(specification: dict[str, Any]) -> dict[str, Any]:
    path, matched, observed = _file_matches(specification)
    return {
        "path": _logical_path(path, label="frozen input"),
        "expected_sha256": specification["sha256"],
        "observed_sha256": observed,
        "matched": matched,
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
    }


def _path_identity(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    return {
        "path": _logical_path(path, label=label),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _matched_identities(
    specifications: dict[str, Any], *, category: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(specifications, dict) or not specifications:
        raise ValueError(f"{category} must be a non-empty mapping")
    identities: dict[str, dict[str, Any]] = {}
    for name, specification in specifications.items():
        if not isinstance(name, str):
            raise ValueError(f"{category} names must be strings")
        identity = _identity(specification)
        if not identity["matched"]:
            raise RuntimeError(
                f"{category} changed: {name}; observed={identity['observed_sha256']}"
            )
        identities[name] = identity
    return identities


def _checkpoint_entries(prereg: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = prereg.get("raw_public_icl", {}).get("checkpoints", [])
    entries = {int(row["seed"]): row for row in rows}
    if tuple(sorted(entries)) != (13313, 13314, 13315):
        raise ValueError("Recovery must preregister exactly seeds 13313/13314/13315")
    if len(entries) != len(rows):
        raise ValueError("Recovery seeds must be unique")
    for seed, entry in entries.items():
        checkpoint = entry.get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("sha256") != entry.get("checkpoint_sha256")
            or not isinstance(entry.get("model_state_sha256"), str)
            or not isinstance(entry.get("raw_gate_passed"), bool)
        ):
            raise ValueError(f"Recovery seed {seed} lacks an exact checkpoint identity")
        identity = _identity(checkpoint)
        if not identity["matched"]:
            raise RuntimeError(
                f"Recovery checkpoint changed: seed {seed}; "
                f"observed={identity['observed_sha256']}"
            )
    return entries


def _expected_recovery_output(
    prereg: dict[str, Any], entries: dict[int, dict[str, Any]], seed: int
) -> tuple[Path, str]:
    if seed not in entries:
        raise ValueError(f"Seed {seed} is not preregistered")
    outputs = prereg.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Recovery preregistration must declare outputs")
    root = _resolve(outputs["root"], label="recovery output root")
    if _logical_path(root, label="recovery output root") != RECOVERY_NAMESPACE.as_posix():
        raise ValueError("Recovery output root is not the dedicated namespace")
    specification = entries[seed].get("recovery_receipt")
    if not isinstance(specification, dict) or not isinstance(specification.get("path"), str):
        raise ValueError(f"Seed {seed} lacks a preregistered recovery receipt path")
    expected = _resolve(specification["path"], label="recovery output")
    try:
        expected.relative_to(root)
    except ValueError as error:
        raise ValueError("Recovery output must be below its dedicated namespace") from error
    if expected.suffix != ".json":
        raise ValueError("Recovery output must be a JSON receipt")
    return expected, _logical_path(expected, label="recovery output")


def _expected_aggregate_output(prereg: dict[str, Any]) -> tuple[Path, str]:
    outputs = prereg.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("aggregate"), dict):
        raise ValueError("Recovery preregistration must declare aggregate output")
    root = _resolve(outputs["root"], label="recovery output root")
    if _logical_path(root, label="recovery output root") != RECOVERY_NAMESPACE.as_posix():
        raise ValueError("Recovery output root is not the dedicated namespace")
    aggregate = _resolve(outputs["aggregate"]["path"], label="aggregate output")
    try:
        aggregate.relative_to(root)
    except ValueError as error:
        raise ValueError("Aggregate output must be below its dedicated namespace") from error
    if aggregate.suffix != ".json":
        raise ValueError("Aggregate output must be a JSON receipt")
    return aggregate, _logical_path(aggregate, label="aggregate output")


def _assert_output(actual: Path, expected: Path, *, label: str) -> tuple[Path, str]:
    actual = _resolve(actual, label=label)
    if actual != expected:
        raise ValueError(
            f"{label} must equal its preregistered exclusive destination "
            f"{_logical_path(expected, label=label)}, got "
            f"{_logical_path(actual, label=label)}"
        )
    return actual, _logical_path(actual, label=label)


def _input_snapshot(
    prereg_path: Path,
    prereg: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Hash every input read by a recovery before or after reconstruction."""

    return {
        "preregistration": _path_identity(prereg_path, label="preregistration"),
        "frozen_inputs": {
            name: _identity(specification)
            for name, specification in prereg["frozen_inputs"].items()
        },
        "implementation": {
            name: _identity(specification)
            for name, specification in prereg["implementation"].items()
        },
        "checkpoint": _identity(entry["checkpoint"]),
        "raw_public_result": _identity(entry["raw_result"]),
    }


def _validate_prereg(prereg_path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    prereg_path = _resolve(prereg_path, label="preregistration")
    prereg = _load_yaml(prereg_path)
    if (
        prereg.get("schema_version") != 1
        or prereg.get("recovery_id")
        != "pusht_action_strength_pldm_float32_rescore_recovery_v1"
        or prereg.get("status")
        != "preregistered_after_raw_public_icl_before_float32_recovery"
    ):
        raise ValueError("Unexpected float32 recovery preregistration")
    if not isinstance(prereg.get("completion_id"), str) or not isinstance(
        prereg.get("release_id"), str
    ):
        raise ValueError("Recovery preregistration lacks completion or release identity")
    required_frozen = {
        "completion_config",
        "evaluation_binding_config",
        "evaluation_binding_receipt",
        "release_config",
    }
    if not required_frozen.issubset(prereg.get("frozen_inputs", {})):
        raise ValueError("Recovery preregistration lacks required frozen inputs")
    _matched_identities(prereg["frozen_inputs"], category="Frozen recovery input")
    # Implementation hashes are evidence, not informational metadata.  A
    # drifted recovery module/scorer must fail before any receipt is built.
    _matched_identities(prereg.get("implementation"), category="Implementation")
    entries = _checkpoint_entries(prereg)
    expected_paths = set()
    for seed in entries:
        expected, _ = _expected_recovery_output(prereg, entries, seed)
        if expected in expected_paths:
            raise ValueError("Recovery receipt destinations must be unique")
        expected_paths.add(expected)
    aggregate_path, _ = _expected_aggregate_output(prereg)
    if aggregate_path in expected_paths:
        raise ValueError("Aggregate output cannot overlap a seed recovery receipt")
    binding = _load_json(
        _resolve(
            prereg["frozen_inputs"]["evaluation_binding_receipt"]["path"],
            label="evaluation binding receipt",
        )
    )
    if (
        binding.get("status") != "passed_evaluation_binding_freeze"
        or binding.get("passed") is not True
        or binding.get("binding", {}).get("sha256")
        != prereg["frozen_inputs"]["evaluation_binding_config"]["sha256"]
    ):
        raise RuntimeError("Passed evaluation binding identity is not intact")
    return prereg, entries


def _validate_raw(
    *,
    raw: dict[str, Any],
    entry: dict[str, Any],
    release: dict[str, Any],
    prereg: dict[str, Any],
) -> None:
    adapter = raw.get("model", {}).get("adapter", {})
    expected_release = {
        "release_id": release["release_id"],
        "release_config_sha256_at_evaluation": prereg["frozen_inputs"]["release_config"]["sha256"],
        "prediction_contract_sha256": _prediction_contract_sha256(release),
        "training_manifest_sha256": release["training"]["manifest_sha256"],
        "confirmation_manifest_sha256": release["evaluation"]["manifest_sha256"],
        "sealed_test_included": False,
    }
    if not (
        raw.get("schema_version") == 1
        and raw.get("benchmark") == "pusht_history3_action_strength_icl_v1"
        and raw.get("submission_kind") == "single_checkpoint"
        and raw.get("status") == "completed"
        and raw.get("release") == expected_release
        and raw.get("model", {}).get("training_seed") == entry["seed"]
        and raw.get("model", {}).get("training_recipe") == "mixed_pldm_joint"
        and raw.get("model", {}).get("state_sha256_before")
        == entry["model_state_sha256"]
        and raw.get("model", {}).get("state_sha256_after")
        == entry["model_state_sha256"]
        and adapter.get("adapter_id") == "stable_worldmodel_pldm_action_strength_v1"
        and adapter.get("checkpoint_sha256") == entry["checkpoint_sha256"]
        and adapter.get("stable_worldmodel_commit")
        == prereg["stable_worldmodel"]["expected_ref"]
        and len(raw.get("records", [])) == 256
        and raw.get("data", {}).get("pair_count") == 256
        and raw.get("data", {}).get("condition_count") == 512
        and raw.get("gate", {}).get("passed") == entry["raw_gate_passed"]
    ):
        raise RuntimeError(f"Raw Public ICL contract mismatch for seed {entry['seed']}")


def recover_one(
    prereg_path: Path,
    seed: int,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    prereg_path = _resolve(prereg_path, label="preregistration")
    prereg, entries = _validate_prereg(prereg_path)
    if seed not in entries:
        raise ValueError(f"Seed {seed} is not preregistered")
    entry = entries[seed]
    expected_output, expected_output_logical = _expected_recovery_output(
        prereg, entries, seed
    )
    _assert_output(
        output_path if output_path is not None else expected_output,
        expected_output,
        label="recovery output",
    )
    before = _input_snapshot(prereg_path, prereg, entry)
    raw_path, matched, observed = _file_matches(entry["raw_result"])
    if not matched:
        raise RuntimeError(
            f"Raw result changed for seed {seed}: expected {entry['raw_result']['sha256']}, got {observed}"
        )
    release_path = _resolve(
        prereg["frozen_inputs"]["release_config"]["path"], label="release config"
    )
    release = load_action_strength_icl_release(release_path)
    raw = _load_json(raw_path)
    _validate_raw(raw=raw, entry=entry, release=release, prereg=prereg)
    records = _records(raw.get("records"))
    metrics, gate = _reconstruct_metrics(records, release=release)
    stored_metrics = raw.get("metrics")
    if not isinstance(stored_metrics, dict):
        raise RuntimeError("Raw metrics are missing")
    scalar = _scalar_comparison(stored_metrics, metrics)
    stored_latent = stored_metrics.get("latent_response")
    if not isinstance(stored_latent, dict):
        raise RuntimeError("Raw latent response is missing")
    latent = _latent_comparison(stored_latent, metrics["latent_response"])
    exact = bool(
        scalar["all_float64_json_bitwise_equal"]
        and scalar["all_float32_loss_aggregates_bitwise_equal"]
        and latent["paired_latent_response_summaries_close"]
        and latent["all_float64_json_bitwise_equal"]
        and gate == raw.get("gate")
    )
    if not exact:
        raise RuntimeError("Float32 reconstruction did not reproduce raw Public result")
    after = _input_snapshot(prereg_path, prereg, entry)
    inputs_unchanged = before == after
    if not inputs_unchanged:
        raise RuntimeError("A frozen recovery input changed while it was rescored")
    return {
        "schema_version": 1,
        "recovery_id": prereg["recovery_id"],
        "completion_id": prereg["completion_id"],
        "status": "completed",
        "seed": seed,
        "preregistration": before["preregistration"],
        "output_policy": {
            "namespace": RECOVERY_NAMESPACE.as_posix(),
            "exclusive_create_required": True,
            "overwrite_permitted": False,
        },
        "output": {
            "path": expected_output_logical,
            "content_sha256_not_embedded_to_avoid_self_reference": True,
        },
        "bindings": {
            "evaluation_binding_config": before["frozen_inputs"][
                "evaluation_binding_config"
            ],
            "evaluation_binding_receipt": before["frozen_inputs"][
                "evaluation_binding_receipt"
            ],
            "release_config": before["frozen_inputs"]["release_config"],
            "raw_public_result": before["raw_public_result"],
            "checkpoint": before["checkpoint"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "implementation": before["implementation"],
        },
        "scope": {
            "model_evaluation_rerun_performed": False,
            "raw_public_result_rewritten": False,
            "frozen_generic_scorer_modified": False,
            "float32_mse_aggregation_only": True,
            "public_test_reopened": False,
        },
        "reconstruction": {
            "mse_record_dtype": "float32",
            "metrics": metrics,
            "gate": gate,
        },
        "input_integrity": {
            "all_frozen_inputs_unchanged_during_recovery": inputs_unchanged,
            "identities_before_recovery_read": before,
            "identities_after_recovery_read": after,
        },
        "verification": {
            "passed": True,
            "scalar_metrics": scalar,
            "latent_metrics": latent,
            "float64_scalar_json_bitwise_equal": scalar[
                "all_float64_json_bitwise_equal"
            ],
            "float32_scalar_aggregates_bitwise_equal": scalar[
                "all_float32_loss_aggregates_bitwise_equal"
            ],
            "latent_summary_close": latent[
                "paired_latent_response_summaries_close"
            ],
            "latent_summary_float64_json_bitwise_equal": latent[
                "all_float64_json_bitwise_equal"
            ],
            "gate_exact_equal": gate == raw.get("gate"),
            "stored_model_gate_passed": raw["gate"]["passed"],
            "recomputed_model_gate_passed": gate["passed"],
        },
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _identity_matches_spec(identity: Any, specification: dict[str, Any]) -> bool:
    expected_path, matched, observed = _file_matches(specification)
    return bool(
        matched
        and isinstance(identity, dict)
        and identity.get("path")
        == _logical_path(expected_path, label="frozen input")
        and identity.get("expected_sha256") == specification["sha256"]
        and identity.get("observed_sha256") == observed
        and identity.get("matched") is True
        and identity.get("size_bytes") == expected_path.stat().st_size
    )


def _validate_recovery_receipt(
    *,
    receipt: dict[str, Any],
    receipt_path: Path,
    prereg_path: Path,
    prereg: dict[str, Any],
    entries: dict[int, dict[str, Any]],
) -> int:
    try:
        seed = int(receipt["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Recovery receipt lacks a valid seed") from error
    expected_path, expected_logical = _expected_recovery_output(prereg, entries, seed)
    if receipt_path != expected_path:
        raise RuntimeError(f"Recovery seed {seed} was supplied from a noncanonical path")
    entry = entries[seed]
    current_prereg = _path_identity(prereg_path, label="preregistration")
    bindings = receipt.get("bindings")
    implementation = bindings.get("implementation") if isinstance(bindings, dict) else None
    verification = receipt.get("verification")
    input_integrity = receipt.get("input_integrity")
    expected_implementation = prereg["implementation"]
    implementation_intact = bool(
        isinstance(implementation, dict)
        and set(implementation) == set(expected_implementation)
        and all(
            _identity_matches_spec(implementation[name], specification)
            for name, specification in expected_implementation.items()
        )
    )
    binding_intact = bool(
        isinstance(bindings, dict)
        and all(
            _identity_matches_spec(bindings.get(name), prereg["frozen_inputs"][name])
            for name in (
                "evaluation_binding_config",
                "evaluation_binding_receipt",
                "release_config",
            )
        )
        and _identity_matches_spec(bindings.get("raw_public_result"), entry["raw_result"])
        and _identity_matches_spec(bindings.get("checkpoint"), entry["checkpoint"])
        and bindings.get("checkpoint_sha256") == entry["checkpoint_sha256"]
    )
    verification_intact = bool(
        isinstance(verification, dict)
        and verification.get("passed") is True
        and verification.get("float64_scalar_json_bitwise_equal") is True
        and verification.get("float32_scalar_aggregates_bitwise_equal") is True
        and verification.get("latent_summary_close") is True
        and verification.get("latent_summary_float64_json_bitwise_equal") is True
        and verification.get("gate_exact_equal") is True
    )
    if not (
        receipt.get("schema_version") == 1
        and receipt.get("recovery_id") == prereg["recovery_id"]
        and receipt.get("completion_id") == prereg["completion_id"]
        and receipt.get("status") == "completed"
        and receipt.get("preregistration") == current_prereg
        and receipt.get("output", {}).get("path") == expected_logical
        and receipt.get("output_policy", {}).get("namespace")
        == RECOVERY_NAMESPACE.as_posix()
        and receipt.get("output_policy", {}).get("exclusive_create_required") is True
        and receipt.get("scope", {}).get("model_evaluation_rerun_performed") is False
        and receipt.get("scope", {}).get("raw_public_result_rewritten") is False
        and receipt.get("scope", {}).get("frozen_generic_scorer_modified") is False
        and receipt.get("scope", {}).get("public_test_reopened") is False
        and receipt.get("scope", {}).get("float32_mse_aggregation_only") is True
        and isinstance(input_integrity, dict)
        and input_integrity.get("all_frozen_inputs_unchanged_during_recovery") is True
        and binding_intact
        and implementation_intact
        and verification_intact
    ):
        raise RuntimeError(f"Recovery receipt is not an intact frozen result: seed {seed}")
    return seed


def _aggregate_snapshot(
    prereg_path: Path,
    prereg: dict[str, Any],
    receipt_paths: dict[int, Path],
) -> dict[str, Any]:
    return {
        "preregistration": _path_identity(prereg_path, label="preregistration"),
        "frozen_inputs": {
            name: _identity(specification)
            for name, specification in prereg["frozen_inputs"].items()
        },
        "implementation": {
            name: _identity(specification)
            for name, specification in prereg["implementation"].items()
        },
        "recovery_receipts": {
            str(seed): _path_identity(path, label="recovery receipt")
            for seed, path in sorted(receipt_paths.items())
        },
    }


def aggregate(
    prereg_path: Path,
    recovery_paths: list[Path],
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    prereg, entries = _validate_prereg(prereg_path)
    prereg_path = _resolve(prereg_path, label="preregistration")
    expected_output, expected_output_logical = _expected_aggregate_output(prereg)
    _assert_output(
        output_path if output_path is not None else expected_output,
        expected_output,
        label="aggregate output",
    )
    if len(recovery_paths) != 3:
        raise ValueError("Aggregate requires exactly three recovery receipts")
    canonical_paths = [_resolve(path, label="recovery receipt") for path in recovery_paths]
    if len(set(canonical_paths)) != len(canonical_paths):
        raise ValueError("Aggregate recovery receipt paths must be unique")
    recovered: dict[int, tuple[dict[str, Any], Path]] = {}
    for path in canonical_paths:
        receipt = _load_json(path)
        seed = _validate_recovery_receipt(
            receipt=receipt,
            receipt_path=path,
            prereg_path=prereg_path,
            prereg=prereg,
            entries=entries,
        )
        if seed in recovered:
            raise ValueError("Aggregate recovery receipt seeds must be unique")
        recovered[seed] = (receipt, path)
    if tuple(sorted(recovered)) != (13313, 13314, 13315) or len(recovered) != 3:
        raise ValueError("Aggregate must contain the three preregistered seeds exactly once")
    before = _aggregate_snapshot(
        prereg_path, prereg, {seed: path for seed, (_, path) in recovered.items()}
    )
    metric_names = (
        "correct_future_rate",
        "correct_history_rate",
        "rule_switch_rate",
        "worst_strength_correct_future_rate",
        "joint_icl_pair_success_rate",
    )
    checkpoints = []
    for seed in sorted(recovered):
        receipt, receipt_path = recovered[seed]
        metrics = receipt["reconstruction"]["metrics"]
        checkpoints.append(
            {
                "recovery_receipt": {
                    **_path_identity(receipt_path, label="recovery receipt"),
                },
                "checkpoint_sha256": entries[seed]["checkpoint_sha256"],
                "training_seed": seed,
                "value": metrics["correct_future_rate"],
                **{name: metrics[name] for name in metric_names},
                "passed": bool(receipt["reconstruction"]["gate"]["passed"]),
            }
        )
    passed = all(row["passed"] for row in checkpoints)
    after = _aggregate_snapshot(
        prereg_path, prereg, {seed: path for seed, (_, path) in recovered.items()}
    )
    inputs_unchanged = before == after
    if not inputs_unchanged:
        raise RuntimeError("A recovery aggregate input changed while it was read")
    return {
        "schema_version": 1,
        "recovery_id": prereg["recovery_id"],
        "completion_id": prereg["completion_id"],
        "status": "completed",
        "submission_kind": "three_seed_method_float32_recovery",
        "evaluation_kind": "public_icl_float32_recovery_aggregate",
        "preregistration": before["preregistration"],
        "output_policy": {
            "namespace": RECOVERY_NAMESPACE.as_posix(),
            "exclusive_create_required": True,
            "overwrite_permitted": False,
        },
        "output": {
            "path": expected_output_logical,
            "content_sha256_not_embedded_to_avoid_self_reference": True,
        },
        "release_id": prereg["release_id"],
        "metric": {
            "id": "correct_real_next_state_choice_accuracy",
            "label": "真实下一状态选择正确率",
        },
        "checkpoints": checkpoints,
        "aggregate": {name: _stats([row[name] for row in checkpoints]) for name in metric_names},
        "decision": {
            "passed": passed,
            "formal_evaluation_completed": True,
            "formal_method_claim": passed,
            "reason": "all_three_training_seeds_passed" if passed else "one_or_more_training_seeds_failed",
        },
        "cem": {
            "authorized": passed,
            "executed": False,
            "reason": (
                "authorized_only_after_recovered_three_seed_icl_gate"
                if passed
                else "not_authorized_because_recovered_three_seed_icl_gate_failed"
            ),
        },
        "input_integrity": {
            "all_aggregate_inputs_unchanged_during_read": inputs_unchanged,
            "identities_before_aggregate_read": before,
            "identities_after_aggregate_read": after,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    commands = parser.add_subparsers(dest="command", required=True)
    recover = commands.add_parser("recover")
    recover.add_argument("--seed", type=int, required=True)
    recover.add_argument("--output", type=Path, required=True)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--input", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preregistration = _resolve(args.preregistration, label="preregistration")
    output = _resolve(args.output, label="output")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite recovery output: {output}")
    if args.command == "recover":
        payload = recover_one(
            preregistration,
            int(args.seed),
            output_path=output,
        )
    else:
        payload = aggregate(
            preregistration,
            [_resolve(path, label="recovery receipt") for path in args.input],
            output_path=output,
        )
    _write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": payload["output"]["path"],
                "decision": payload.get("decision"),
            }
        )
    )


if __name__ == "__main__":
    main()
