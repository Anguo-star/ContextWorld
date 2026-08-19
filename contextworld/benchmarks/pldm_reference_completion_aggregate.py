"""Fail-closed finalization for the four PLDM reference completions.

This module is intentionally an accounting and identity-validation layer.  It
does not train a model, open an evaluation dataset, or execute a planner.  It
only consumes already-frozen JSON receipts.  In particular, the two
Development-only PushT failures remain aggregate evidence but can never become
public-scoreboard rows.

The historical eleven-row Suite v2 files are inputs with pinned identities.
The only scoreboard files this module can create live in the additive v1
namespace.  The aggregate freeze itself uses the separately preregistered
``configs/...aggregate_results_freeze_v1.json`` path required by the inactive
integrity-reseal-v2 contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import yaml

from contextworld.benchmarks.public_score import make_public_scoreboard_from_spec
from contextworld.benchmarks.speed_pldm_infrastructure_development import (
    COMPLETION_ID as SPEED_DEVELOPMENT_COMPLETION_ID,
    DEVELOPMENT_ID as SPEED_DEVELOPMENT_ID,
    DEVELOPMENT_SCOPE as SPEED_DEVELOPMENT_SCOPE,
    EXPECTED_SEEDS as SPEED_DEVELOPMENT_SEEDS,
)
from contextworld.paths import repository_root, resolve_contextworld_path


AGGREGATE_ID = "contextworld_pldm_reference_completion_aggregate_results_freeze_v1"
AGGREGATE_CONFIG = (
    repository_root()
    / "configs/benchmark/contextworld_pldm_reference_completion_aggregate_prereg_v1.yaml"
)
FORMAL_COMPONENTS = ("speed", "action_strength")
DEVELOPMENT_ONLY_COMPONENTS = ("contact_friction", "motion_damping")
COMPONENTS = (*FORMAL_COMPONENTS, *DEVELOPMENT_ONLY_COMPONENTS)

# ActionStrength has an independently preregistered float32 recovery protocol.
# Its receipt layout is intentionally checked separately from the forthcoming
# TwoRoom Speed layout; a generic "some fields look true" fallback would make
# the aggregate accept evidence from the wrong recovery procedure.
ACTION_STRENGTH_RECOVERY_ID = "pusht_action_strength_pldm_float32_rescore_recovery_v1"
ACTION_STRENGTH_RECOVERY_PREREGISTRATION = (
    "configs/benchmark/pusht_action_strength_pldm_float32_rescore_recovery_v1.yaml"
)
ACTION_STRENGTH_RECOVERY_NAMESPACE = (
    "artifacts/evaluation/history3/pusht_action_strength_pldm_reference_completion_v1/"
    "formal_icl_v1/float32_rescore_recovery_v1"
)
ACTION_STRENGTH_SCORE_CONSISTENCY_AMENDMENT = (
    "configs/benchmark/pusht_action_strength_score_float32_consistency_amendment_v1.yaml"
)
SPEED_RECOVERY_ID = "tworoom_speed_pldm_reference_completion_recovery_v1"


class CompletionAggregateBlocked(ValueError):
    """Raised when a required completion receipt is missing or inconsistent."""

    def __init__(self, blockers: list[str]) -> None:
        self.blockers = blockers
        super().__init__("; ".join(blockers))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _identity(logical_path: str, *, repo_root: Path) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(logical_path)
    return {
        "path": logical_path,
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _serialized_identity(logical_path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    data = _json_bytes(payload)
    return {
        "path": logical_path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }


def _new_output_path(logical_path: str, *, repo_root: Path) -> Path:
    """Resolve a preregistered *new* output inside this checkout only.

    Inputs use :func:`resolve_contextworld_path`, which may correctly locate a
    canonical historical artifact outside the checkout.  That behavior is not
    safe for an absent output: it would create a new addendum in an external
    data root.  Finalization therefore resolves each new destination
    lexically below ``repo_root`` and rejects every escape before opening it.
    """

    path = Path(logical_path)
    if (
        path.is_absolute()
        or not logical_path
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("new finalization output must be a non-empty relative path")
    root = repo_root.resolve()
    candidate = root.joinpath(*path.parts)

    # New release outputs are local commit materials, not logical artifact
    # references.  Do not let an existing in-tree symlink redirect either a
    # write or a later validation to an external artifact root.
    current = root
    for part in path.parts:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("new finalization output traverses a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("new finalization output escaped repo_root") from error
    return resolved


def _local_output_identity(
    logical_path: str, *, repo_root: Path
) -> dict[str, Any]:
    """Identify a new finalization output without artifact-root fallback.

    Historical inputs may legitimately resolve through
    ``CONTEXTWORLD_ARTIFACT_ROOT``.  The aggregate, addendum specification,
    addendum scoreboard, and resolution decision are newly created commit
    materials and must instead be regular, non-symlink files in this
    checkout.
    """

    path = _new_output_path(logical_path, repo_root=repo_root)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"local finalization output is missing: {logical_path}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(
            f"local finalization output is not a regular file: {logical_path}"
        )
    return {
        "path": logical_path,
        "sha256": _sha256(path),
        "size_bytes": int(metadata.st_size),
    }


def _read_local_output_json(
    logical_path: str, *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one local, no-symlink finalization JSON output."""

    identity = _local_output_identity(logical_path, repo_root=repo_root)
    path = _new_output_path(logical_path, repo_root=repo_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"local finalization output is not valid JSON: {logical_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            f"local finalization output must contain a JSON object: {logical_path}"
        )
    return payload, identity


def _read_json(logical_path: str, *, repo_root: Path) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(logical_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{logical_path} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{logical_path} must contain a JSON object")
    return payload


def _read_yaml(logical_path: str, *, repo_root: Path) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(logical_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{logical_path} must contain a YAML mapping")
    return payload


def _mapping(value: Any, *, label: str, keys: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if keys is not None and set(value) != keys:
        raise ValueError(f"{label} keys are invalid")
    return value


def _identity_mapping(value: Any, *, label: str, repo_root: Path) -> dict[str, Any]:
    expected = _mapping(value, label=label, keys={"path", "sha256", "size_bytes"})
    path = expected.get("path")
    if not isinstance(path, str) or not path.startswith("artifacts/"):
        raise ValueError(f"{label} path must be an artifacts-relative path")
    actual = _identity(path, repo_root=repo_root)
    if actual != expected:
        raise ValueError(f"{label} identity drifted")
    return actual


def _file_identity_mapping(value: Any, *, label: str, repo_root: Path) -> dict[str, Any]:
    """Validate an exact identity for either a repo or artifact path."""

    expected = _mapping(value, label=label, keys={"path", "sha256", "size_bytes"})
    path = expected.get("path")
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise ValueError(f"{label} path must be a non-empty logical path")
    actual = _identity(path, repo_root=repo_root)
    if actual != expected:
        raise ValueError(f"{label} identity drifted")
    return actual


def _speed_development_declaration(
    formal: Mapping[str, Any], *, expected_seeds: list[int]
) -> dict[str, Any]:
    """Validate the static locations of Speed's pre-Public readiness chain.

    This declaration lives in the aggregate preregistration, while the
    identities themselves are created only after fixed training completes.
    Keeping those two pieces separate avoids inventing output hashes before
    the no-score Development stage has actually run.
    """

    if tuple(expected_seeds) != SPEED_DEVELOPMENT_SEEDS:
        raise ValueError("Speed completion does not use the registered Development seed set")
    value = _mapping(formal.get("development"), label="Speed Development declaration")
    if set(value) != {"config", "manifest", "receipts"}:
        raise ValueError("Speed Development declaration keys are invalid")
    config = value.get("config")
    manifest = value.get("manifest")
    if not all(
        isinstance(path, str) and path and not Path(path).is_absolute()
        for path in (config, manifest)
    ):
        raise ValueError("Speed Development config/manifest paths are invalid")
    rows = value.get("receipts")
    if not isinstance(rows, list) or len(rows) != len(SPEED_DEVELOPMENT_SEEDS):
        raise ValueError("Speed Development declaration requires three receipts")
    receipts: list[dict[str, Any]] = []
    for row in rows:
        entry = _mapping(row, label="Speed Development receipt declaration")
        if set(entry) != {"seed", "path"}:
            raise ValueError("Speed Development receipt declaration keys are invalid")
        seed = _integer(entry.get("seed"), label="Speed Development receipt seed")
        path = entry.get("path")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError("Speed Development receipt path is invalid")
        receipts.append({"seed": seed, "path": path})
    if tuple(sorted(row["seed"] for row in receipts)) != SPEED_DEVELOPMENT_SEEDS:
        raise ValueError("Speed Development receipt declaration uses the wrong seed set")
    return {
        "config": config,
        "manifest": manifest,
        "receipts": sorted(receipts, key=lambda row: row["seed"]),
    }


def _validated_speed_development_chain(
    *,
    formal: Mapping[str, Any],
    aggregate_value: Any,
    completion_id: str,
    expected_seeds: list[int],
    repo_root: Path,
) -> dict[str, Any]:
    """Validate and canonically return Speed's immutable no-score chain.

    The final aggregate/CEM authorizer must not merely trust a boolean that
    says Development passed.  It reads the separately frozen config, manifest
    and all three receipts, verifies their identities and contracts, and ties
    each receipt to the matching final checkpoint hash.
    """

    if completion_id != SPEED_DEVELOPMENT_COMPLETION_ID:
        raise ValueError("Speed Development chain is attached to the wrong completion")
    declared = _speed_development_declaration(formal, expected_seeds=expected_seeds)
    value = _mapping(aggregate_value, label="Speed Public ICL Development chain")
    if set(value) != {"config", "manifest", "receipts"}:
        raise ValueError("Speed Public ICL aggregate lacks its Development chain")

    config = _file_identity_mapping(
        value.get("config"), label="Speed Development config", repo_root=repo_root
    )
    manifest = _file_identity_mapping(
        value.get("manifest"), label="Speed Development manifest", repo_root=repo_root
    )
    if not (
        _same_path(config["path"], declared["config"], repo_root=repo_root)
        and _same_path(manifest["path"], declared["manifest"], repo_root=repo_root)
    ):
        raise ValueError("Speed Public ICL aggregate binds an undeclared Development chain")

    config_payload = _read_yaml(config["path"], repo_root=repo_root)
    manifest_payload = _read_json(manifest["path"], repo_root=repo_root)
    if not (
        config_payload.get("development_id") == SPEED_DEVELOPMENT_ID
        and config_payload.get("completion_id") == completion_id
        and config_payload.get("scope") == SPEED_DEVELOPMENT_SCOPE
        and manifest_payload.get("schema_version") == 1
        and manifest_payload.get("development_id") == SPEED_DEVELOPMENT_ID
        and manifest_payload.get("completion_id") == completion_id
        and manifest_payload.get("status") == "frozen_prepublic_development_manifest"
        and manifest_payload.get("passed") is True
        and manifest_payload.get("scope") == SPEED_DEVELOPMENT_SCOPE
        and manifest_payload.get("development_config") == config
        and manifest_payload.get("public_payload_accessed") is False
        and manifest_payload.get("formal_public_or_cem_artifacts_present") is False
        and manifest_payload.get("coverage", {}).get("validation_scenarios") == 96
        and manifest_payload.get("coverage", {}).get("total_samples") == 384
        and manifest_payload.get("coverage", {}).get("all_actual_indices_unique_per_scenario")
        is True
        and manifest_payload.get("coverage", {}).get("all_source_spans_continuous")
        is True
    ):
        raise ValueError("Speed Development config/manifest contract is invalid")

    rows = value.get("receipts")
    if not isinstance(rows, list) or len(rows) != len(SPEED_DEVELOPMENT_SEEDS):
        raise ValueError("Speed Public ICL aggregate lacks all Development receipts")
    declared_paths = {row["seed"]: row["path"] for row in declared["receipts"]}
    observed: list[dict[str, Any]] = []
    checkpoint_sha256_by_seed: dict[int, str] = {}
    for row in rows:
        entry = _mapping(row, label="Speed Public ICL Development receipt")
        if set(entry) != {"seed", "receipt"}:
            raise ValueError("Speed Public ICL Development receipt keys are invalid")
        seed = _integer(entry.get("seed"), label="Speed Public ICL Development receipt seed")
        if seed not in declared_paths:
            raise ValueError("Speed Public ICL Development receipt has an unexpected seed")
        receipt_identity = _file_identity_mapping(
            entry.get("receipt"),
            label=f"Speed Development receipt {seed}",
            repo_root=repo_root,
        )
        if not _same_path(
            receipt_identity["path"], declared_paths[seed], repo_root=repo_root
        ):
            raise ValueError("Speed Public ICL aggregate binds an undeclared Development receipt")
        receipt = _read_json(receipt_identity["path"], repo_root=repo_root)
        checks = _mapping(receipt.get("checks"), label=f"Speed Development receipt {seed} checks")
        checkpoint = _mapping(
            receipt.get("checkpoint"), label=f"Speed Development receipt {seed} checkpoint"
        )
        checkpoint_sha256 = checkpoint.get("sha256")
        state = receipt.get("checkpoint_model_state_sha256")
        if not (
            receipt.get("schema_version") == 1
            and receipt.get("development_id") == SPEED_DEVELOPMENT_ID
            and receipt.get("completion_id") == completion_id
            and _integer(receipt.get("seed"), label="Speed Development receipt seed") == seed
            and receipt.get("status") == "passed_infrastructure_readiness"
            and receipt.get("passed") is True
            and receipt.get("scope") == SPEED_DEVELOPMENT_SCOPE
            and receipt.get("development_config") == config
            and receipt.get("development_manifest") == manifest
            and isinstance(checkpoint_sha256, str)
            and len(checkpoint_sha256) == 64
            and isinstance(state, str)
            and len(state) == 64
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
            and checks.get("complete_heldout_manifest_coverage", {}).get("samples") == 384
            and checks.get("complete_heldout_manifest_coverage", {}).get("scenarios") == 96
            and checks.get("native_future_latent_mse_finiteness", {}).get(
                "mse_value_withheld_not_a_score"
            )
            is True
            and checks.get("frozen_weight_audit", {}).get("state_hash_before") == state
            and checks.get("frozen_weight_audit", {}).get("state_hash_after") == state
            and checks.get("public_boundary", {}).get("public_payload_accessed") is False
            and checks.get("public_boundary", {}).get("checkpoint_selection") is False
            and checks.get("public_boundary", {}).get("scoreboard_score_emitted") is False
        ):
            raise ValueError(f"Speed Development receipt {seed} contract is invalid")
        observed.append({"seed": seed, "receipt": receipt_identity})
        checkpoint_sha256_by_seed[seed] = checkpoint_sha256
    if tuple(sorted(row["seed"] for row in observed)) != SPEED_DEVELOPMENT_SEEDS:
        raise ValueError("Speed Public ICL aggregate Development receipt seeds are incomplete")
    return {
        "chain": {
            "config": config,
            "manifest": manifest,
            "receipts": sorted(observed, key=lambda row: row["seed"]),
        },
        "checkpoint_sha256_by_seed": checkpoint_sha256_by_seed,
    }


def _same_path(left: str, right: str, *, repo_root: Path) -> bool:
    return resolve_contextworld_path(left, repo_root=repo_root) == resolve_contextworld_path(
        right, repo_root=repo_root
    )


def _at_path(payload: Mapping[str, Any], dotted_path: str, *, label: str) -> Any:
    if not dotted_path or dotted_path.startswith(".") or dotted_path.endswith("."):
        raise ValueError(f"{label} JSON path is invalid")
    value: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"{label} is missing JSON path {dotted_path!r}")
        value = value[key]
    return value


def _fraction(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a numeric fraction")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _model_family(completion: Mapping[str, Any]) -> str:
    scope = completion.get("scope")
    training = completion.get("training")
    scope_value = scope.get("model_family") if isinstance(scope, Mapping) else None
    training_value = training.get("model_family") if isinstance(training, Mapping) else None
    return str(scope_value or training_value or "").lower()


def _completion_release_path(completion: Mapping[str, Any]) -> str:
    evaluation = _mapping(completion.get("evaluation"), label="completion evaluation")
    icl = _mapping(evaluation.get("icl"), label="completion ICL evaluation")
    path = icl.get("release_config")
    if not isinstance(path, str) or not path:
        raise ValueError("completion ICL release_config is missing")
    return path


def _expected_training_seeds(completion: Mapping[str, Any]) -> list[int]:
    training = _mapping(completion.get("training"), label="completion training")
    raw = training.get("seeds")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("completion must preregister exactly three training seeds")
    values = [_integer(value, label="completion training seed") for value in raw]
    if len(set(values)) != 3:
        raise ValueError("completion training seeds must be distinct")
    return sorted(values)


def _load_completion(
    specification: Mapping[str, Any], *, repo_root: Path, require_icl_release: bool = True
) -> tuple[dict[str, Any], dict[str, Any], str | None, list[int]]:
    path = specification.get("completion_config")
    completion_id = specification.get("completion_id")
    if not isinstance(path, str) or not isinstance(completion_id, str):
        raise ValueError("completion input is missing its config or id")
    completion = _read_yaml(path, repo_root=repo_root)
    if completion.get("completion_id") != completion_id or _model_family(completion) != "pldm":
        raise ValueError(f"completion {completion_id} is not a PLDM completion")
    release_path = _completion_release_path(completion) if require_icl_release else None
    return completion, _identity(path, repo_root=repo_root), release_path, _expected_training_seeds(completion)


def load_completion_aggregate_preregistration(
    path: Path | str = AGGREGATE_CONFIG,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load the immutable finalization plan without consuming any results."""

    root = (repo_root or repository_root()).resolve()
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_top = {
        "schema_version",
        "aggregate_id",
        "status",
        "release_id",
        "created_utc",
        "scope",
        "historical_base",
        "original_baseline_cem",
        "completion_inputs",
        "formal_evidence_contract",
        "scoreboard_addendum",
        "outputs",
        "membership_authority",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_top
        or payload.get("schema_version") != 1
        or payload.get("aggregate_id") != AGGREGATE_ID
        or payload.get("status") != "preregistered_pending_four_final_outcomes"
        or payload.get("release_id") != "contextworld_icl_benchmark_suite_v2"
    ):
        raise ValueError("PLDM completion aggregate preregistration is invalid")

    scope = _mapping(payload.get("scope"), label="aggregate scope")
    if (
        scope.get("components_in_final_aggregate") != list(COMPONENTS)
        or scope.get("public_scoreboard_components") != list(FORMAL_COMPONENTS)
        or scope.get("development_only_components") != list(DEVELOPMENT_ONLY_COMPONENTS)
        or scope.get("formal_public_result_required_for_scoreboard_row") is not True
        or scope.get("public_result_added_regardless_of_pass_or_fail") is not True
        or scope.get("public_icl_three_seed_result_required_for_every_formal_component")
        is not True
        or scope.get("cem_required_only_after_three_seed_public_icl_pass") is not True
        or scope.get("nonpassing_public_icl_requires_cem_not_authorized_proof")
        is not True
        or scope.get("development_only_result_must_not_add_public_row") is not True
    ):
        raise ValueError("aggregate scope is incomplete")

    historical = _mapping(payload.get("historical_base"), label="historical base")
    if historical.get("formal_reference_rows") != 11:
        raise ValueError("historical base must contain eleven formal rows")
    for label in ("specification", "scoreboard"):
        entry = _mapping(historical.get(label), label=f"historical base {label}")
        if set(entry) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"historical base {label} identity is invalid")

    baseline = _mapping(payload.get("original_baseline_cem"), label="original baseline CEM")
    if (
        baseline.get("path")
        != "configs/benchmark/contextworld_original_baseline_cem_results_freeze_v1.json"
        or baseline.get("freeze_id")
        != "contextworld_original_baseline_cem_results_freeze_v1"
        or baseline.get("required_status")
        != "frozen_after_completed_descriptive_matrix"
    ):
        raise ValueError("original baseline CEM contract is invalid")
    mappings = _mapping(baseline.get("score_mapping"), label="baseline score mapping")
    if mappings != {
        "speed": {"environment": "tworoom", "family": "pldm"},
        "action_strength": {"environment": "pusht", "family": "pldm"},
    }:
        raise ValueError("baseline score mapping is invalid")

    inputs = _mapping(payload.get("completion_inputs"), label="completion inputs")
    if tuple(inputs) != COMPONENTS:
        raise ValueError("completion inputs are incomplete or unordered")
    for component in FORMAL_COMPONENTS:
        entry = _mapping(inputs.get(component), label=f"{component} completion input")
        expected_keys = {
            "completion_config",
            "completion_id",
            "component_id",
            "component_name",
            "formal_result",
            "retention_metric",
        }
        if set(entry) != expected_keys or entry.get("component_id") != component:
            raise ValueError(f"{component} formal completion contract is invalid")
        metric = _mapping(entry.get("retention_metric"), label=f"{component} retention metric")
        if set(metric) != {"id", "label"}:
            raise ValueError(f"{component} retention metric contract is invalid")
        formal_result = _mapping(entry.get("formal_result"), label=f"{component} formal result")
        expected_formal_result_keys = {
            "public_icl_aggregate",
            "raw_public_results_root",
            "recovery_root",
            "cem_not_authorized_stop",
            "action_planning_aggregate",
            "original_task_retention_aggregate",
            "primary_metric",
            "aggregate_metric_key",
            "method_name",
        }
        if component == "speed":
            expected_formal_result_keys.add("behavioral_claim_boundary")
            expected_formal_result_keys.add("development")
            expected_formal_result_keys.add("cem_binding")
        if set(formal_result) != expected_formal_result_keys:
            raise ValueError(f"{component} formal-result paths are invalid")
        if component == "speed":
            # This is path-only preregistration.  The generated manifest and
            # receipts are independently identity-validated only after the
            # fixed training run and pre-Public Development gate finish.
            _speed_development_declaration(
                formal_result, expected_seeds=list(SPEED_DEVELOPMENT_SEEDS)
            )
        _metric_definition(formal_result.get("primary_metric"), label=f"{component} primary metric")
        if (
            not isinstance(formal_result.get("aggregate_metric_key"), str)
            or not formal_result["aggregate_metric_key"].strip()
            or not isinstance(formal_result.get("method_name"), str)
            or not formal_result["method_name"].strip()
        ):
            raise ValueError(f"{component} formal-result reader metadata is invalid")
    for component in DEVELOPMENT_ONLY_COMPONENTS:
        entry = _mapping(inputs.get(component), label=f"{component} completion input")
        expected_keys = {
            "completion_config",
            "completion_id",
            "component_id",
            "development_decision",
            "evaluation_binding",
            "development_evaluation",
            "development_rescore",
            "required_terminal_status",
        }
        if set(entry) != expected_keys or entry.get("component_id") != component:
            raise ValueError(f"{component} Development-only completion contract is invalid")

    contract = _mapping(payload.get("formal_evidence_contract"), label="formal evidence contract")
    if (
        contract.get("public_icl_aggregate_schema_version") != 1
        or contract.get("required_training_seeds") != 3
        or not isinstance(contract.get("public_icl_requirements"), list)
        or not isinstance(contract.get("if_all_three_public_icl_gates_pass"), list)
        or not isinstance(contract.get("if_any_public_icl_gate_fails"), list)
    ):
        raise ValueError("formal evidence contract is invalid")

    addendum = _mapping(payload.get("scoreboard_addendum"), label="scoreboard addendum")
    expected_addendum = {
        "preregistration": "configs/benchmark/contextworld_icl_suite_v2_scoreboard_addendum_prereg_v1.yaml",
        "decision": "configs/benchmark/contextworld_icl_suite_v2_scoreboard_addendum_decision_v1.json",
        "decision_id": "contextworld_icl_suite_v2_scoreboard_addendum_decision_v1",
        "output_namespace": "artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1",
        "specification": "artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1/public_scoreboard_spec.json",
        "scoreboard": "artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1/public_scoreboard.json",
    }
    if addendum != expected_addendum:
        raise ValueError("scoreboard addendum output contract is invalid")
    outputs = _mapping(payload.get("outputs"), label="aggregate outputs")
    if outputs != {
        "aggregate_freeze": "configs/benchmark/contextworld_pldm_reference_completion_aggregate_results_freeze_v1.json",
        "only_scoreboard_outputs_may_use_addendum_namespace": True,
        "exclusive_creation_required": True,
        "overwrite_authorized": False,
    }:
        raise ValueError("aggregate output contract is invalid")
    authority = _mapping(payload.get("membership_authority"), label="aggregate authority")
    if authority != {
        "config_alone_grants_membership": False,
        "aggregate_alone_grants_membership": False,
        "suite_default_or_activation_switched_by_this_record": False,
        "integrity_reseal_v2_decision_required_for_activation": True,
    }:
        raise ValueError("aggregate membership authority is invalid")
    return {**payload, "_config_path": str(config_path)}


def _validate_historical_base(
    preregistration: Mapping[str, Any], *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    historical = _mapping(preregistration.get("historical_base"), label="historical base")
    specification_identity = _file_identity_mapping(
        historical["specification"], label="historical specification", repo_root=repo_root
    )
    scoreboard_identity = _file_identity_mapping(
        historical["scoreboard"], label="historical scoreboard", repo_root=repo_root
    )
    specification = _read_json(specification_identity["path"], repo_root=repo_root)
    scoreboard = _read_json(scoreboard_identity["path"], repo_root=repo_root)
    rows = specification.get("components")
    result_rows = scoreboard.get("component_results")
    if (
        specification.get("schema_version") != 1
        or specification.get("result_kind") != "contextworld_public_scoreboard_spec"
        or not isinstance(rows, list)
        or len(rows) != 11
        or scoreboard.get("schema_version") != 1
        or scoreboard.get("result_kind") != "contextworld_public_scoreboard"
        or not isinstance(result_rows, list)
        or len(result_rows) != 11
    ):
        raise ValueError("historical base scoreboard is not the frozen eleven-row table")
    recomputed = make_public_scoreboard_from_spec(specification)
    if recomputed != scoreboard:
        raise ValueError("historical base scoreboard does not match its specification")
    return specification, scoreboard, specification_identity, scoreboard_identity


def _validate_baseline_cem(
    preregistration: Mapping[str, Any], *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, float]]:
    specification = _mapping(preregistration.get("original_baseline_cem"), label="original baseline CEM")
    path = str(specification["path"])
    payload = _read_json(path, repo_root=repo_root)
    if (
        payload.get("schema_version") != 1
        or payload.get("freeze_id") != specification["freeze_id"]
        or payload.get("status") != specification["required_status"]
    ):
        raise ValueError("original baseline CEM freeze is invalid")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError("original baseline CEM cells are missing")
    mapping = _mapping(specification.get("score_mapping"), label="baseline score mapping")
    values: dict[str, float] = {}
    for component, target in mapping.items():
        matches = [
            row
            for row in cells
            if isinstance(row, Mapping)
            and row.get("environment") == target["environment"]
            and row.get("family") == target["family"]
        ]
        if len(matches) != 1:
            raise ValueError(f"original baseline CEM has no unique {component} PLDM cell")
        cell = matches[0]
        successes = _integer(cell.get("success_count"), label=f"{component} baseline successes")
        evaluations = _integer(cell.get("evaluation_count"), label=f"{component} baseline evaluations")
        rate = _fraction(cell.get("success_rate"), label=f"{component} baseline rate")
        if evaluations <= 0 or not math.isclose(rate, successes / evaluations, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{component} baseline CEM rate is inconsistent")
        values[component] = rate
    return _identity(path, repo_root=repo_root), values


def _validate_addendum_preregistration(
    preregistration: Mapping[str, Any],
    *,
    base_specification_identity: Mapping[str, Any],
    base_scoreboard_identity: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    specification = _mapping(preregistration.get("scoreboard_addendum"), label="scoreboard addendum")
    path = str(specification["preregistration"])
    payload = _read_yaml(path, repo_root=repo_root)
    addendum = _mapping(payload.get("scoreboard_addendum"), label="addendum preregistration")
    expected_outputs = specification
    declared_outputs = _mapping(addendum.get("additive_output_namespace"), label="addendum output namespace")
    if (
        addendum.get("record_id") != "contextworld_icl_suite_v2_scoreboard_addendum_prereg_v1"
        or addendum.get("status") != "pending_final_pldm_completion_aggregate"
        or addendum.get("base_formal_reference_rows") != 11
        or declared_outputs.get("root") != expected_outputs["output_namespace"]
        or declared_outputs.get("specification") != expected_outputs["specification"]
        or declared_outputs.get("scoreboard") != expected_outputs["scoreboard"]
    ):
        raise ValueError("scoreboard addendum preregistration is invalid")
    resolution = _mapping(addendum.get("final_resolution_decision"), label="addendum resolution")
    if resolution != {
        "path": expected_outputs["decision"],
        "decision_id": expected_outputs["decision_id"],
    }:
        raise ValueError("scoreboard addendum resolution contract is invalid")
    historical = _mapping(addendum.get("historical_base_evidence"), label="addendum historical base")
    if historical.get("specification") != dict(base_specification_identity) or historical.get(
        "scoreboard"
    ) != dict(base_scoreboard_identity):
        raise ValueError("scoreboard addendum historical identities drifted")
    return _identity(path, repo_root=repo_root)


def _metric_definition(value: Any, *, label: str) -> dict[str, str]:
    definition = _mapping(
        value,
        label=label,
        keys={"id", "label", "value_path", "gate_path", "seed_path"},
    )
    result: dict[str, str] = {}
    for key in definition:
        raw = definition[key]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{label} {key} must be non-empty text")
        result[key] = raw
    return result


def _reference_identity(
    value: Any,
    *,
    label: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Validate a receipt reference that may use ``sha256`` or ``observed_sha256``.

    Runner receipts often retain an absolute execution path.  The finalizer
    resolves that path only to verify the byte identity and serializes the
    original reference string rather than inventing a new location.
    """

    reference = _mapping(value, label=label)
    path = reference.get("path")
    expected_hash = reference.get("sha256", reference.get("observed_sha256"))
    if not isinstance(path, str) or not path or not isinstance(expected_hash, str):
        raise ValueError(f"{label} must carry path and SHA-256")
    resolved = resolve_contextworld_path(path, repo_root=repo_root)
    if not resolved.is_file():
        raise FileNotFoundError(path)
    actual_hash = _sha256(resolved)
    if actual_hash != expected_hash:
        raise ValueError(f"{label} identity drifted")
    return {"path": path, "sha256": actual_hash, "size_bytes": int(resolved.stat().st_size)}


def _action_identity_matches_spec(
    value: Any,
    specification: Any,
    *,
    label: str,
    repo_root: Path,
) -> bool:
    """Validate the explicit expected/observed identity layout of Action recovery."""

    observed = _mapping(value, label=label)
    expected = _mapping(specification, label=f"expected {label}")
    expected_path = expected.get("path")
    expected_hash = expected.get("sha256")
    if not isinstance(expected_path, str) or not isinstance(expected_hash, str):
        raise ValueError(f"expected {label} is missing path or SHA-256")
    return bool(
        observed.get("path") == expected_path
        and observed.get("expected_sha256") == expected_hash
        and observed.get("observed_sha256") == expected_hash
        and observed.get("matched") is True
        and isinstance(observed.get("size_bytes"), int)
        and observed["size_bytes"] >= 0
    )


def _historical_reference_matches_spec(value: Any, specification: Any, *, label: str) -> bool:
    """Compare a historical path/SHA receipt with its frozen specification.

    Unlike a live identity check, this must not re-hash a file intentionally
    superseded by the post-freeze scorer amendment.
    """

    observed = _mapping(value, label=label)
    expected = _mapping(specification, label=f"expected {label}")
    return bool(
        observed.get("path") == expected.get("path")
        and observed.get("sha256") == expected.get("sha256")
        and (
            observed.get("size_bytes") is None
            or (
                isinstance(observed.get("size_bytes"), int)
                and observed["size_bytes"] >= 0
            )
        )
    )


def _action_recovery_preregistration(
    *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    """Load the one frozen ActionStrength float32-recovery protocol exactly."""

    identity = _identity(ACTION_STRENGTH_RECOVERY_PREREGISTRATION, repo_root=repo_root)
    preregistration = _read_yaml(ACTION_STRENGTH_RECOVERY_PREREGISTRATION, repo_root=repo_root)
    if (
        preregistration.get("schema_version") != 1
        or preregistration.get("recovery_id") != ACTION_STRENGTH_RECOVERY_ID
        or preregistration.get("completion_id")
        != "pusht_action_strength_pldm_reference_completion_v1"
        or preregistration.get("release_id")
        != "contextworld_pusht_action_strength_icl_history3_v1"
    ):
        raise ValueError("ActionStrength float32-recovery preregistration is invalid")
    frozen_inputs = _mapping(
        preregistration.get("frozen_inputs"), label="ActionStrength recovery frozen inputs"
    )
    implementation = _mapping(
        preregistration.get("implementation"), label="ActionStrength recovery implementation"
    )
    if set(frozen_inputs) != {
        "completion_config",
        "evaluation_binding_config",
        "evaluation_binding_receipt",
        "release_config",
    } or set(implementation) != {
        "recovery_launcher",
        "frozen_float32_reconstruction",
        "frozen_icl_scorer",
        "release_loader",
        "paired_latent_metric",
        "frozen_icl_cli",
    }:
        raise ValueError("ActionStrength float32-recovery preregistration inputs are incomplete")
    outputs = _mapping(preregistration.get("outputs"), label="ActionStrength recovery outputs")
    if (
        outputs.get("root") != ACTION_STRENGTH_RECOVERY_NAMESPACE
        or outputs.get("exclusive_creation_required") is not True
        or outputs.get("overwrite_authorized") is not False
    ):
        raise ValueError("ActionStrength float32-recovery outputs are invalid")
    raw_public_icl = _mapping(
        preregistration.get("raw_public_icl"), label="ActionStrength recovery raw Public ICL"
    )
    raw_entries = raw_public_icl.get("checkpoints")
    if not isinstance(raw_entries, list) or len(raw_entries) != 3:
        raise ValueError("ActionStrength float32-recovery must preregister three checkpoints")
    entries: dict[int, dict[str, Any]] = {}
    for entry in raw_entries:
        item = _mapping(entry, label="ActionStrength recovery checkpoint")
        seed = _integer(item.get("seed"), label="ActionStrength recovery checkpoint seed")
        if (
            seed in entries
            or not isinstance(item.get("checkpoint_sha256"), str)
            or type(item.get("raw_gate_passed")) is not bool
            or not isinstance(_mapping(item.get("recovery_receipt"), label="ActionStrength recovery output").get("path"), str)
        ):
            raise ValueError("ActionStrength float32-recovery checkpoint is invalid")
        entries[seed] = item
    if tuple(sorted(entries)) != (13313, 13314, 13315):
        raise ValueError("ActionStrength float32-recovery seeds are invalid")
    return preregistration, identity, entries


def _action_strength_post_freeze_maintenance_audit(
    *,
    preregistration: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Bind the scorer maintenance amendment without rewriting frozen history.

    The float32 recovery was executed against an earlier frozen release/scorer
    identity.  The maintenance amendment records that historical identity and
    the narrow post-freeze scorer correction.  A finalizer must therefore
    validate both timelines separately rather than falsely treating a valid
    maintenance edit as evidence drift in the completed experiment.
    """

    amendment_identity = _identity(ACTION_STRENGTH_SCORE_CONSISTENCY_AMENDMENT, repo_root=repo_root)
    amendment = _read_yaml(ACTION_STRENGTH_SCORE_CONSISTENCY_AMENDMENT, repo_root=repo_root)
    expected_inputs = _mapping(
        preregistration.get("frozen_inputs"), label="ActionStrength recovery frozen inputs"
    )
    expected_implementation = _mapping(
        preregistration.get("implementation"), label="ActionStrength recovery implementation"
    )
    pre_change = _mapping(amendment.get("pre_change_identity"), label="ActionStrength amendment pre-change identity")
    purpose = _mapping(amendment.get("purpose"), label="ActionStrength amendment purpose")
    required_behavior = _mapping(
        amendment.get("required_post_change_behavior"), label="ActionStrength amendment required behavior"
    )
    if (
        amendment.get("schema_version") != 1
        or amendment.get("amendment_id")
        != "pusht_action_strength_score_float32_consistency_amendment_v1"
        or amendment.get("release_id") != preregistration.get("release_id")
        or amendment.get("status")
        != "preregistered_after_formal_result_freeze_before_public_scorer_fix"
        or pre_change.get("release_config") != expected_inputs["release_config"]
        or pre_change.get("scorer") != expected_implementation["frozen_icl_scorer"]
        or purpose.get("scientific_effect") != "none"
        or purpose.get("gate_or_threshold_change_authorized") is not False
        or purpose.get("raw_result_rewrite_authorized") is not False
        or purpose.get("model_evaluation_rerun_authorized") is not False
        or purpose.get("checkpoint_selection_authorized") is not False
        or purpose.get("public_test_reopen_authorized") is not False
        or required_behavior.get("historical_receipts_must_remain_byte_identical")
        is not True
        or required_behavior.get("reconstructed_gate_matches_independent_recovery_bit_for_bit")
        is not True
        or required_behavior.get("cem_authorization_must_remain_false") is not True
    ):
        raise ValueError("ActionStrength scorer-maintenance amendment is invalid")
    release_path = str(expected_inputs["release_config"]["path"])
    current_release = _read_yaml(release_path, repo_root=repo_root)
    current_release_identity = _identity(release_path, repo_root=repo_root)
    current_scorer = _mapping(
        _mapping(current_release.get("identity"), label="current ActionStrength release identity").get("score_api"),
        label="current ActionStrength release scorer identity",
    )
    scorer_path = current_scorer.get("path")
    scorer_hash = current_scorer.get("sha256")
    if (
        current_release.get("schema_version") != 1
        or current_release.get("release_id") != preregistration.get("release_id")
        or not isinstance(scorer_path, str)
        or not isinstance(scorer_hash, str)
        or _identity(scorer_path, repo_root=repo_root)["sha256"] != scorer_hash
        or scorer_hash == expected_implementation["frozen_icl_scorer"]["sha256"]
    ):
        raise ValueError("current ActionStrength release/scorer audit is invalid")
    return {
        "score_consistency_amendment": amendment_identity,
        "current_release_config": current_release_identity,
        "current_release_scorer": _identity(scorer_path, repo_root=repo_root),
        "frozen_release_config_sha256": expected_inputs["release_config"]["sha256"],
        "frozen_scorer_sha256": expected_implementation["frozen_icl_scorer"]["sha256"],
    }


def _action_snapshot_is_intact(
    snapshot: Any,
    *,
    preregistration_identity: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    entry: Mapping[str, Any],
    label: str,
    repo_root: Path,
) -> bool:
    value = _mapping(snapshot, label=label)
    if set(value) != {
        "preregistration",
        "frozen_inputs",
        "implementation",
        "checkpoint",
        "raw_public_result",
    }:
        return False
    prereg_reference = _reference_identity(
        value.get("preregistration"), label=f"{label} preregistration", repo_root=repo_root
    )
    if prereg_reference != dict(preregistration_identity):
        return False
    frozen_inputs = _mapping(value.get("frozen_inputs"), label=f"{label} frozen inputs")
    implementation = _mapping(value.get("implementation"), label=f"{label} implementation")
    expected_inputs = _mapping(preregistration.get("frozen_inputs"), label="ActionStrength frozen inputs")
    expected_implementation = _mapping(
        preregistration.get("implementation"), label="ActionStrength implementation"
    )
    if set(frozen_inputs) != set(expected_inputs) or set(implementation) != set(expected_implementation):
        return False
    return all(
        _action_identity_matches_spec(
            frozen_inputs[name], expected_inputs[name], label=f"{label} frozen input {name}", repo_root=repo_root
        )
        for name in expected_inputs
    ) and all(
        _action_identity_matches_spec(
            implementation[name], expected_implementation[name], label=f"{label} implementation {name}", repo_root=repo_root
        )
        for name in expected_implementation
    ) and _action_identity_matches_spec(
        value.get("checkpoint"), entry.get("checkpoint"), label=f"{label} checkpoint", repo_root=repo_root
    ) and _action_identity_matches_spec(
        value.get("raw_public_result"), entry.get("raw_result"), label=f"{label} raw Public result", repo_root=repo_root
    )


def _matches_raw_public_result_contract(
    payload: Mapping[str, Any],
    *,
    component: str,
    expected_release_id: str,
) -> bool:
    """Check the release/result envelope shared by raw ICL readers.

    Speed predates the generic component receipt shape.  Keep its explicit
    ``benchmark``/``release_config`` contract here so the conditional CEM
    branch probe and the final result reader cannot disagree about whether a
    frozen receipt is usable.
    """

    if component == "speed":
        release = _mapping(
            payload.get("release_config"), label="Speed raw Public ICL release"
        )
        return bool(
            payload.get("schema_version") == 1
            and payload.get("benchmark") == expected_release_id
            and payload.get("submission_kind") == "single_model"
            and payload.get("status") == "passed"
            and payload.get("full_protocol") is True
            and isinstance(release.get("path"), str)
            and isinstance(release.get("sha256"), str)
        )
    else:
        release = _mapping(
            payload.get("release"), label=f"{component} raw Public ICL release"
        )
        return bool(
            payload.get("schema_version") == 1
            and release.get("release_id") == expected_release_id
            and isinstance(payload.get("status"), str)
            and payload.get("status") in {"completed", "passed"}
        )


def _raw_public_result(
    *,
    component: str,
    raw_path: str,
    primary_metric: Mapping[str, str],
    expected_seed: int,
    expected_release_id: str,
    expected_checkpoint_sha256: str,
    expected_speed_development: Mapping[str, Any] | None = None,
    repo_root: Path,
) -> dict[str, Any]:
    identity = _identity(raw_path, repo_root=repo_root)
    payload = _read_json(raw_path, repo_root=repo_root)
    result_contract = _matches_raw_public_result_contract(
        payload,
        component=component,
        expected_release_id=expected_release_id,
    )
    if not result_contract:
        raise ValueError(f"{component} raw Public ICL result is not a completed release result")
    if component == "speed":
        if expected_speed_development is None:
            raise ValueError("Speed raw Public ICL validation lacks Development evidence")
        completion_evaluation = _mapping(
            payload.get("completion_evaluation"),
            label="Speed raw Public ICL completion evaluation",
        )
        if completion_evaluation.get("development") != dict(expected_speed_development):
            raise ValueError("Speed raw Public ICL receipt does not bind the Development chain")
    seed = _integer(
        _at_path(payload, primary_metric["seed_path"], label=f"{component} raw Public ICL"),
        label=f"{component} raw Public ICL training seed",
    )
    value = _fraction(
        _at_path(payload, primary_metric["value_path"], label=f"{component} raw Public ICL"),
        label=f"{component} raw Public ICL metric",
    )
    passed = _at_path(payload, primary_metric["gate_path"], label=f"{component} raw Public ICL")
    checkpoint_hash = _at_path(
        payload,
        (
            "model.checkpoint_sha256"
            if component == "speed"
            else "model.adapter.checkpoint_sha256"
        ),
        label=f"{component} raw Public ICL",
    )
    if (
        seed != expected_seed
        or type(passed) is not bool
        or checkpoint_hash != expected_checkpoint_sha256
    ):
        raise ValueError(f"{component} raw Public ICL result does not bind seed/checkpoint")
    return {
        "training_seed": seed,
        "value": value,
        "passed": passed,
        "source": identity,
    }


def _validate_action_strength_recovery_receipt(
    *,
    receipt_path: str,
    expected_raw_path: str,
    expected_raw_identity: Mapping[str, Any],
    expected_seed: int,
    expected_checkpoint_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Validate one ActionStrength float32 receipt against its frozen protocol."""

    preregistration, preregistration_identity, entries = _action_recovery_preregistration(
        repo_root=repo_root
    )
    entry = entries.get(expected_seed)
    if entry is None:
        raise ValueError("ActionStrength receipt seed is not preregistered")
    receipt_identity = _identity(receipt_path, repo_root=repo_root)
    payload = _read_json(receipt_path, repo_root=repo_root)
    expected_output_path = _mapping(
        entry.get("recovery_receipt"), label="ActionStrength recovery receipt output"
    ).get("path")
    if not isinstance(expected_output_path, str) or not _same_path(
        expected_output_path, receipt_path, repo_root=repo_root
    ):
        raise ValueError("ActionStrength recovery receipt is outside its preregistered destination")
    expected_raw = _reference_identity(
        entry.get("raw_result"), label="ActionStrength preregistered raw Public result", repo_root=repo_root
    )
    if (
        not _same_path(expected_raw["path"], expected_raw_path, repo_root=repo_root)
        or expected_raw != dict(expected_raw_identity)
        or entry.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        raise ValueError("ActionStrength recovery preregistration drifted from the formal aggregate")
    raw_payload = _read_json(expected_raw_path, repo_root=repo_root)
    raw_gate = _mapping(raw_payload.get("gate"), label="ActionStrength raw Public ICL gate")
    if raw_gate.get("passed") is not entry.get("raw_gate_passed"):
        raise ValueError("ActionStrength recovery preregistration does not match the raw Public ICL gate")
    bindings = _mapping(payload.get("bindings"), label="ActionStrength recovery bindings")
    expected_inputs = _mapping(
        preregistration.get("frozen_inputs"), label="ActionStrength recovery frozen inputs"
    )
    expected_implementation = _mapping(
        preregistration.get("implementation"), label="ActionStrength recovery implementation"
    )
    implementation = _mapping(bindings.get("implementation"), label="ActionStrength recovery implementation bindings")
    binding_intact = bool(
        set(bindings)
        == {
            "evaluation_binding_config",
            "evaluation_binding_receipt",
            "release_config",
            "raw_public_result",
            "checkpoint",
            "checkpoint_sha256",
            "implementation",
        }
        and all(
            _action_identity_matches_spec(
                bindings.get(name), expected_inputs[name], label=f"ActionStrength recovery {name}", repo_root=repo_root
            )
            for name in (
                "evaluation_binding_config",
                "evaluation_binding_receipt",
                "release_config",
            )
        )
        and _action_identity_matches_spec(
            bindings.get("raw_public_result"), entry.get("raw_result"), label="ActionStrength recovery raw Public result", repo_root=repo_root
        )
        and _action_identity_matches_spec(
            bindings.get("checkpoint"), entry.get("checkpoint"), label="ActionStrength recovery checkpoint", repo_root=repo_root
        )
        and bindings.get("checkpoint_sha256") == expected_checkpoint_sha256
        and set(implementation) == set(expected_implementation)
        and all(
            _action_identity_matches_spec(
                implementation.get(name), expected_implementation[name], label=f"ActionStrength recovery implementation {name}", repo_root=repo_root
            )
            for name in expected_implementation
        )
    )
    prereg_reference = _reference_identity(
        payload.get("preregistration"), label="ActionStrength recovery preregistration", repo_root=repo_root
    )
    output = _mapping(payload.get("output"), label="ActionStrength recovery output")
    output_policy = _mapping(payload.get("output_policy"), label="ActionStrength recovery output policy")
    scope = _mapping(payload.get("scope"), label="ActionStrength recovery scope")
    reconstruction = _mapping(payload.get("reconstruction"), label="ActionStrength recovery reconstruction")
    reconstruction_metrics = _mapping(
        reconstruction.get("metrics"), label="ActionStrength recovery reconstructed metrics"
    )
    verification = _mapping(payload.get("verification"), label="ActionStrength recovery verification")
    scalar_metrics = _mapping(
        verification.get("scalar_metrics"), label="ActionStrength recovery scalar verification"
    )
    latent_metrics = _mapping(
        verification.get("latent_metrics"), label="ActionStrength recovery latent verification"
    )
    input_integrity = _mapping(payload.get("input_integrity"), label="ActionStrength recovery input integrity")
    expected_metric = _at_path(
        raw_payload, "metrics.correct_future_rate", label="ActionStrength raw Public ICL"
    )
    verification_intact = bool(
        verification.get("passed") is True
        and verification.get("float64_scalar_json_bitwise_equal") is True
        and verification.get("float32_scalar_aggregates_bitwise_equal") is True
        and verification.get("latent_summary_close") is True
        and verification.get("latent_summary_float64_json_bitwise_equal") is True
        and verification.get("gate_exact_equal") is True
        and verification.get("stored_model_gate_passed") is raw_gate.get("passed")
        and verification.get("recomputed_model_gate_passed") is raw_gate.get("passed")
        and scalar_metrics.get("all_float64_json_bitwise_equal") is True
        and scalar_metrics.get("all_float32_loss_aggregates_bitwise_equal") is True
        and latent_metrics.get("paired_latent_response_summaries_close") is True
        and latent_metrics.get("all_float64_json_bitwise_equal") is True
    )
    if not (
        payload.get("schema_version") == 1
        and payload.get("recovery_id") == ACTION_STRENGTH_RECOVERY_ID
        and payload.get("completion_id") == preregistration["completion_id"]
        and payload.get("status") == "completed"
        and _integer(payload.get("seed"), label="ActionStrength recovery seed") == expected_seed
        and prereg_reference == preregistration_identity
        and output.get("path") == expected_output_path
        and output.get("content_sha256_not_embedded_to_avoid_self_reference") is True
        and output_policy
        == {
            "namespace": ACTION_STRENGTH_RECOVERY_NAMESPACE,
            "exclusive_create_required": True,
            "overwrite_permitted": False,
        }
        and scope
        == {
            "model_evaluation_rerun_performed": False,
            "raw_public_result_rewritten": False,
            "frozen_generic_scorer_modified": False,
            "float32_mse_aggregation_only": True,
            "public_test_reopened": False,
        }
        and reconstruction.get("mse_record_dtype") == "float32"
        and reconstruction.get("gate") == raw_gate
        and math.isclose(
            _fraction(reconstruction_metrics.get("correct_future_rate"), label="ActionStrength reconstructed metric"),
            _fraction(expected_metric, label="ActionStrength raw Public ICL metric"),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and input_integrity.get("all_frozen_inputs_unchanged_during_recovery") is True
        and _action_snapshot_is_intact(
            input_integrity.get("identities_before_recovery_read"),
            preregistration_identity=preregistration_identity,
            preregistration=preregistration,
            entry=entry,
            label="ActionStrength recovery identities before read",
            repo_root=repo_root,
        )
        and input_integrity.get("identities_after_recovery_read")
        == input_integrity.get("identities_before_recovery_read")
        and binding_intact
        and verification_intact
    ):
        raise ValueError("ActionStrength recovery receipt is not an intact frozen float32 recovery")
    return receipt_identity


def _validate_recovery_receipt(
    *,
    component: str,
    receipt_path: str,
    expected_raw_path: str,
    expected_raw_identity: Mapping[str, Any],
    expected_seed: int,
    expected_checkpoint_sha256: str,
    completion_id: str,
    release_id: str,
    expected_behavioral_claim_boundary: Mapping[str, Any] | None,
    expected_speed_development: Mapping[str, Any] | None = None,
    repo_root: Path,
) -> dict[str, Any]:
    if component == "action_strength":
        return _validate_action_strength_recovery_receipt(
            receipt_path=receipt_path,
            expected_raw_path=expected_raw_path,
            expected_raw_identity=expected_raw_identity,
            expected_seed=expected_seed,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            repo_root=repo_root,
        )
    if component != "speed":
        raise ValueError(f"No receipt schema is registered for {component}")
    if expected_behavioral_claim_boundary is None:
        raise ValueError("Speed recovery is missing its behavioral claim boundary")
    if expected_speed_development is None:
        raise ValueError("Speed recovery is missing the Development chain")
    identity = _identity(receipt_path, repo_root=repo_root)
    payload = _read_json(receipt_path, repo_root=repo_root)
    if (
        payload.get("schema_version") != 1
        or payload.get("recovery_id") != SPEED_RECOVERY_ID
        or payload.get("completion_id") != completion_id
        or payload.get("release_id") != release_id
        or payload.get("status") != "completed"
    ):
        raise ValueError("Speed recovery receipt schema is invalid")
    bindings = _mapping(payload.get("bindings"), label=f"{component} recovery bindings")
    raw_reference = _reference_identity(
        bindings.get("raw_public_result"),
        label=f"{component} recovery raw Public result",
        repo_root=repo_root,
    )
    if (
        not _same_path(raw_reference["path"], expected_raw_path, repo_root=repo_root)
        or raw_reference["sha256"] != expected_raw_identity["sha256"]
    ):
        raise ValueError(f"{component} recovery receipt binds a different raw Public result")
    for key in ("evaluation_binding_config", "evaluation_binding_receipt"):
        _reference_identity(bindings.get(key), label=f"{component} recovery {key}", repo_root=repo_root)
    if payload.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError(f"{component} recovery receipt binds a different checkpoint")
    if _integer(payload.get("training_seed"), label=f"{component} recovery seed") != expected_seed:
        raise ValueError(f"{component} recovery receipt binds a different seed")
    verification = _mapping(payload.get("verification"), label=f"{component} recovery verification")
    input_integrity = _mapping(payload.get("input_integrity"), label="Speed recovery input integrity")
    if (
        verification.get("passed") is not True
        or verification.get("gate_exact_equal") is not True
        or verification.get("all_track_horizon_metrics_exact") is not True
        or verification.get("latent_summary_close") is not True
        or verification.get("float32_scalar_aggregation_applicable") is not False
        or verification.get("float32_scalar_aggregates_bitwise_equal") is not None
        or input_integrity.get("all_frozen_inputs_unchanged_during_recovery") is not True
        or input_integrity.get("identities_after_recovery_read")
        != input_integrity.get("identities_before_recovery_read")
        or payload.get("development") != dict(expected_speed_development)
        or bindings.get("development") != dict(expected_speed_development)
    ):
        raise ValueError(f"{component} recovery receipt lacks its required independent checks")
    output = _mapping(payload.get("output"), label="Speed recovery output")
    if not isinstance(output.get("path"), str) or not _same_path(
        output["path"], receipt_path, repo_root=repo_root
    ):
        raise ValueError("Speed recovery receipt output path is not canonical")
    boundary = _file_identity_mapping(
        payload.get("behavioral_claim_boundary"),
        label="Speed recovery behavioral claim boundary",
        repo_root=repo_root,
    )
    if (
        boundary != dict(expected_behavioral_claim_boundary)
        or _mapping(payload.get("claim_boundary"), label="Speed recovery claim boundary")
        != {
            "paired_single_speed_control_available": False,
            "training_attribution_claim": False,
            "public_test_reopened": False,
            "claim_level": "behavioral_trained_reference_only",
        }
    ):
        raise ValueError("Speed recovery receipt claim boundary is invalid")
    return identity


def _public_icl_from_recovery(
    component: str,
    specification: Mapping[str, Any],
    *,
    completion_id: str,
    release_id: str,
    expected_seeds: list[int],
    repo_root: Path,
) -> dict[str, Any]:
    formal = _mapping(specification.get("formal_result"), label=f"{component} formal result")
    primary_metric = _metric_definition(formal.get("primary_metric"), label=f"{component} primary metric")
    behavioral_claim_boundary = (
        _identity(str(formal["behavioral_claim_boundary"]), repo_root=repo_root)
        if component == "speed"
        else None
    )
    aggregate_path = str(formal["public_icl_aggregate"])
    aggregate_identity = _identity(aggregate_path, repo_root=repo_root)
    aggregate = _read_json(aggregate_path, repo_root=repo_root)
    action_preregistration: dict[str, Any] | None = None
    action_preregistration_identity: dict[str, Any] | None = None
    action_maintenance: dict[str, Any] | None = None
    speed_development: dict[str, Any] | None = None
    if component == "action_strength":
        (
            action_preregistration,
            action_preregistration_identity,
            _action_entries,
        ) = _action_recovery_preregistration(repo_root=repo_root)
        aggregate_preregistration = _reference_identity(
            aggregate.get("preregistration"),
            label="ActionStrength Public ICL aggregate preregistration",
            repo_root=repo_root,
        )
        aggregate_output = _mapping(
            aggregate.get("output"), label="ActionStrength Public ICL aggregate output"
        )
        aggregate_output_policy = _mapping(
            aggregate.get("output_policy"), label="ActionStrength Public ICL aggregate output policy"
        )
        aggregate_metric = _mapping(
            aggregate.get("metric"), label="ActionStrength Public ICL aggregate metric"
        )
        if (
            aggregate.get("schema_version") != 1
            or aggregate.get("recovery_id") != ACTION_STRENGTH_RECOVERY_ID
            or aggregate.get("completion_id") != completion_id
            or aggregate.get("release_id") != release_id
            or aggregate.get("status") != "completed"
            or aggregate.get("submission_kind") != "three_seed_method_float32_recovery"
            or aggregate.get("evaluation_kind") != "public_icl_float32_recovery_aggregate"
            or aggregate_preregistration != action_preregistration_identity
            or aggregate_output.get("path") != aggregate_path
            or aggregate_output.get("content_sha256_not_embedded_to_avoid_self_reference")
            is not True
            or aggregate_output_policy
            != {
                "namespace": ACTION_STRENGTH_RECOVERY_NAMESPACE,
                "exclusive_create_required": True,
                "overwrite_permitted": False,
            }
            or aggregate_metric != {"id": primary_metric["id"], "label": primary_metric["label"]}
        ):
            raise ValueError("ActionStrength three-seed Public ICL aggregate is invalid")
        action_maintenance = _action_strength_post_freeze_maintenance_audit(
            preregistration=action_preregistration, repo_root=repo_root
        )
    elif component == "speed":
        metric = _mapping(aggregate.get("metric"), label="Speed Public ICL aggregate metric")
        output = _mapping(aggregate.get("output"), label="Speed Public ICL aggregate output")
        speed_development = _validated_speed_development_chain(
            formal=formal,
            aggregate_value=aggregate.get("development"),
            completion_id=completion_id,
            expected_seeds=expected_seeds,
            repo_root=repo_root,
        )
        aggregate_boundary = _file_identity_mapping(
            aggregate.get("behavioral_claim_boundary"),
            label="Speed Public ICL aggregate behavioral claim boundary",
            repo_root=repo_root,
        )
        if (
            aggregate.get("schema_version") != 1
            or aggregate.get("recovery_id") != SPEED_RECOVERY_ID
            or aggregate.get("completion_id") != completion_id
            or aggregate.get("release_id") != release_id
            or aggregate.get("status") != "completed"
            or aggregate.get("submission_kind") != "three_seed_method_recovery"
            or aggregate.get("evaluation_kind") != "public_icl_recovery_aggregate"
            or metric != {"id": primary_metric["id"], "label": primary_metric["label"]}
            or output.get("path") != aggregate_path
            or aggregate_boundary != behavioral_claim_boundary
            or _mapping(aggregate.get("claim_boundary"), label="Speed Public ICL aggregate claim boundary")
            != {
                "paired_single_speed_control_available": False,
                "training_attribution_claim": False,
                "public_test_reopened": False,
                "claim_level": "behavioral_trained_reference_only",
            }
        ):
            raise ValueError("Speed three-seed Public ICL aggregate is invalid")
    else:
        raise ValueError(f"No formal recovery aggregate schema is registered for {component}")
    aggregate_completion_id = aggregate.get("completion_id")
    if aggregate_completion_id is not None and aggregate_completion_id != completion_id:
        raise ValueError(f"{component} Public ICL aggregate binds a different completion")
    checkpoints = aggregate.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        raise ValueError(f"{component} Public ICL aggregate must contain three checkpoints")
    raw_root = str(formal["raw_public_results_root"])
    recovery_root = str(formal["recovery_root"])
    aggregate_key = str(formal["aggregate_metric_key"])
    rows = []
    for checkpoint in checkpoints:
        row = _mapping(checkpoint, label=f"{component} aggregate checkpoint")
        seed = _integer(row.get("training_seed"), label=f"{component} aggregate seed")
        checkpoint_hash = row.get("checkpoint_sha256")
        if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
            raise ValueError(f"{component} aggregate checkpoint SHA is invalid")
        if (
            component == "speed"
            and speed_development is not None
            and speed_development["checkpoint_sha256_by_seed"].get(seed) != checkpoint_hash
        ):
            raise ValueError(
                "Speed Public ICL aggregate checkpoint does not match its Development receipt"
            )
        raw_path = str(Path(raw_root) / f"seed_{seed}.json")
        raw = _raw_public_result(
            component=component,
            raw_path=raw_path,
            primary_metric=primary_metric,
            expected_seed=seed,
            expected_release_id=release_id,
            expected_checkpoint_sha256=checkpoint_hash,
            expected_speed_development=(
                speed_development["chain"] if speed_development else None
            ),
            repo_root=repo_root,
        )
        recovered = _validate_recovery_receipt(
            component=component,
            receipt_path=str(Path(recovery_root) / f"seed_{seed}.json"),
            expected_raw_path=raw_path,
            expected_raw_identity=raw["source"],
            expected_seed=seed,
            expected_checkpoint_sha256=checkpoint_hash,
            completion_id=completion_id,
            release_id=release_id,
            expected_behavioral_claim_boundary=behavioral_claim_boundary,
            expected_speed_development=(
                speed_development["chain"] if speed_development else None
            ),
            repo_root=repo_root,
        )
        aggregate_recovery = _reference_identity(
            row.get("recovery_receipt"),
            label=f"{component} aggregate recovery receipt",
            repo_root=repo_root,
        )
        if not _same_path(aggregate_recovery["path"], str(Path(recovery_root) / f"seed_{seed}.json"), repo_root=repo_root):
            raise ValueError(f"{component} aggregate checkpoint binds the wrong recovery receipt")
        aggregate_value = _fraction(row.get(aggregate_key), label=f"{component} aggregate primary metric")
        if type(row.get("passed")) is not bool or not math.isclose(
            aggregate_value, raw["value"], rel_tol=0.0, abs_tol=1e-15
        ) or row["passed"] is not raw["passed"]:
            raise ValueError(f"{component} aggregate values do not match independently recovered raw Public ICL")
        rows.append({**raw, "recovery_receipt": recovered})
    seeds = [row["training_seed"] for row in rows]
    if len(set(seeds)) != 3 or set(seeds) != set(expected_seeds):
        raise ValueError(f"{component} Public ICL aggregate does not cover preregistered seeds")
    rows.sort(key=lambda row: row["training_seed"])
    decision = _mapping(aggregate.get("decision"), label=f"{component} aggregate decision")
    ability_passed = all(row["passed"] for row in rows)
    decision_valid = bool(
        decision.get("formal_evaluation_completed") is True
        and decision.get("passed") is ability_passed
        and isinstance(decision.get("reason"), str)
        and decision["reason"].strip()
    )
    if component == "speed":
        # A three-seed behavioral result may authorize downstream CEM, but
        # without paired single-speed PLDM controls it is not a causal
        # training-attribution claim.
        decision_valid = decision_valid and decision.get("formal_method_claim") is False
    else:
        decision_valid = decision_valid and decision.get("formal_method_claim") is ability_passed
    if not decision_valid:
        raise ValueError(f"{component} Public ICL aggregate decision is inconsistent")
    cem = _mapping(aggregate.get("cem"), label=f"{component} aggregate CEM state")
    # The Public ICL aggregate is immutable evidence for the authorization
    # decision.  It is produced before either CEM run begins, so an ICL pass
    # authorizes CEM but cannot itself claim that CEM has executed.
    expected_cem_state = {"authorized": ability_passed, "executed": False}
    if (
        cem.get("authorized") is not ability_passed
        or cem.get("executed") is not False
        or not isinstance(cem.get("reason"), str)
        or not cem["reason"].strip()
    ):
        raise ValueError(
            f"{component} Public ICL freeze must record {expected_cem_state}; "
            "CEM execution is a later, separately frozen stage"
        )
    if component == "action_strength":
        assert action_preregistration is not None
        assert action_preregistration_identity is not None
        assert action_maintenance is not None
        aggregate_input_integrity = _mapping(
            aggregate.get("input_integrity"), label="ActionStrength aggregate input integrity"
        )
        before = _mapping(
            aggregate_input_integrity.get("identities_before_aggregate_read"),
            label="ActionStrength aggregate identities before read",
        )
        expected_inputs = _mapping(
            action_preregistration.get("frozen_inputs"), label="ActionStrength frozen inputs"
        )
        expected_implementation = _mapping(
            action_preregistration.get("implementation"), label="ActionStrength implementation"
        )
        frozen_inputs = _mapping(before.get("frozen_inputs"), label="ActionStrength aggregate frozen inputs")
        implementation = _mapping(before.get("implementation"), label="ActionStrength aggregate implementation")
        receipt_references = _mapping(
            before.get("recovery_receipts"), label="ActionStrength aggregate recovery receipts"
        )
        prereg_reference = _reference_identity(
            before.get("preregistration"),
            label="ActionStrength aggregate preregistration identity",
            repo_root=repo_root,
        )
        expected_receipt_references = {
            str(row["training_seed"]): row["recovery_receipt"] for row in rows
        }
        if not (
            aggregate_input_integrity.get("all_aggregate_inputs_unchanged_during_read") is True
            and aggregate_input_integrity.get("identities_after_aggregate_read") == before
            and prereg_reference == action_preregistration_identity
            and set(frozen_inputs) == set(expected_inputs)
            and set(implementation) == set(expected_implementation)
            and all(
                _action_identity_matches_spec(
                    frozen_inputs[name], expected_inputs[name], label=f"ActionStrength aggregate frozen input {name}", repo_root=repo_root
                )
                for name in expected_inputs
            )
            and all(
                _action_identity_matches_spec(
                    implementation[name], expected_implementation[name], label=f"ActionStrength aggregate implementation {name}", repo_root=repo_root
                )
                for name in expected_implementation
            )
            and receipt_references == expected_receipt_references
        ):
            raise ValueError("ActionStrength aggregate input-integrity proof is invalid")
        amendment = _read_yaml(ACTION_STRENGTH_SCORE_CONSISTENCY_AMENDMENT, repo_root=repo_root)
        formal_evidence = _mapping(
            amendment.get("frozen_formal_evidence"), label="ActionStrength amendment formal evidence"
        )
        # The list entries are converted explicitly so malformed or duplicate
        # amendment seeds cannot silently overwrite each other.
        raw_entries = formal_evidence.get("raw_public_icl")
        recovery_entries = _mapping(
            formal_evidence.get("independent_float32_recovery"), label="ActionStrength amendment recovery"
        ).get("seed_receipts")
        if not isinstance(raw_entries, list) or not isinstance(recovery_entries, list):
            raise ValueError("ActionStrength amendment formal evidence is incomplete")
        raw_by_seed = {
            _integer(item.get("seed"), label="ActionStrength amendment raw seed"): item
            for item in raw_entries
            if isinstance(item, Mapping)
        }
        recovery_by_seed = {
            _integer(item.get("seed"), label="ActionStrength amendment recovery seed"): item
            for item in recovery_entries
            if isinstance(item, Mapping)
        }
        if (
            set(raw_by_seed) != set(expected_seeds)
            or set(recovery_by_seed) != set(expected_seeds)
            or len(raw_by_seed) != len(raw_entries)
            or len(recovery_by_seed) != len(recovery_entries)
            or any(
                _reference_identity(raw_by_seed[seed], label="ActionStrength amendment raw receipt", repo_root=repo_root)
                != row["source"]
                or _reference_identity(recovery_by_seed[seed], label="ActionStrength amendment recovery receipt", repo_root=repo_root)
                != row["recovery_receipt"]
                for seed, row in ((row["training_seed"], row) for row in rows)
            )
            or _reference_identity(
                _mapping(formal_evidence.get("independent_float32_recovery"), label="ActionStrength amendment recovery").get("aggregate"),
                label="ActionStrength amendment aggregate",
                repo_root=repo_root,
            )
            != aggregate_identity
        ):
            raise ValueError("ActionStrength amendment does not bind the frozen formal evidence")
    elif component == "speed":
        aggregate_input_integrity = _mapping(
            aggregate.get("input_integrity"), label="Speed aggregate input integrity"
        )
        if (
            aggregate_input_integrity.get("all_aggregate_inputs_unchanged_during_read")
            is not True
            or aggregate_input_integrity.get("identities_after_aggregate_read")
            != aggregate_input_integrity.get("identities_before_aggregate_read")
        ):
            raise ValueError("Speed aggregate input-integrity proof is invalid")
    return {
        "metric": primary_metric,
        "records": rows,
        "aggregate": aggregate_identity,
        "cem": dict(expected_cem_state),
        "ability_passed": ability_passed,
        **({"post_freeze_maintenance": action_maintenance} if action_maintenance else {}),
        **({"development": speed_development["chain"]} if speed_development else {}),
    }


def _speed_positive_cem_binding(
    *,
    path: str,
    completion_id: str,
    public_icl: Mapping[str, Any],
    expected_development: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deeply validate the one post-3/3 CEM bridge for Speed.

    The generic finalizer deliberately delegates ledger/schema details to the
    CEM freezer implementation, but it independently binds the CEM bridge to
    this exact recovered Public-ICL chain.  Thus a result from another
    successful three-seed run cannot be substituted merely because its scalar
    summary looks plausible.
    """

    from scripts import freeze_tworoom_speed_pldm_cem_aggregate_v1 as speed_cem

    binding_identity = _identity(path, repo_root=repo_root)
    binding_path = resolve_contextworld_path(path, repo_root=repo_root)
    binding = speed_cem.runner._load_json(binding_path)
    if not (
        binding.get("schema_version") == 1
        and binding.get("cem_binding_id") == "tworoom_speed_pldm_cem_binding_v1"
        and binding.get("completion_id") == completion_id
        and binding.get("status")
        == "frozen_after_passed_three_seed_public_icl_before_cem"
        and binding.get("passed") is True
        and binding.get("cem") == {"authorized": True, "executed": False}
        and binding.get("frozen_chain", {}).get("public_icl_aggregate")
        == public_icl.get("aggregate")
        and binding.get("frozen_chain", {}).get("development")
        == dict(expected_development)
        and binding.get("input_integrity", {}).get(
            "all_frozen_inputs_unchanged_during_binding"
        )
        is True
        and binding.get("input_integrity", {}).get("identities_after_binding")
        == binding.get("input_integrity", {}).get("identities_before_binding")
    ):
        raise ValueError("Speed CEM binding is not tied to this passed Public-ICL chain")
    # Do not rely solely on the runner's later preflight: the finalizer also
    # proves that the positive binding preserved the exact CEM authority that
    # was sealed into both evaluation-binding materials *before* Public ICL.
    # This prevents a post-3/3 binding from replacing an otherwise valid
    # planner/catalog/criterion closure with a newly selected one.
    frozen_chain = _mapping(binding.get("frozen_chain"), label="Speed CEM frozen chain")
    evaluation_binding_identity = _file_identity_mapping(
        frozen_chain.get("evaluation_binding_config"),
        label="Speed CEM evaluation-binding config",
        repo_root=repo_root,
    )
    evaluation_receipt_identity = _file_identity_mapping(
        frozen_chain.get("evaluation_binding_receipt"),
        label="Speed CEM evaluation-binding receipt",
        repo_root=repo_root,
    )
    prepublic_authority = _mapping(
        frozen_chain.get("prepublic_cem_authority"),
        label="Speed CEM pre-Public authority",
    )
    evaluation_binding = _read_yaml(
        evaluation_binding_identity["path"], repo_root=repo_root
    )
    evaluation_receipt = _read_json(
        evaluation_receipt_identity["path"], repo_root=repo_root
    )
    if not (
        prepublic_authority == evaluation_binding.get("cem_protocol")
        and prepublic_authority == evaluation_receipt.get("cem_protocol")
        and binding.get("preregistration") == prepublic_authority.get("preregistration")
    ):
        raise ValueError("Speed CEM binding is not rooted in one pre-Public CEM authority")
    expected_raw = {
        int(row["training_seed"]): row["source"] for row in public_icl.get("records", [])
    }
    expected_recovery = {
        int(row["training_seed"]): row["recovery_receipt"]
        for row in public_icl.get("records", [])
    }
    raw = {
        int(row["seed"]): row["receipt"]
        for row in binding["frozen_chain"].get("raw_public_icl", [])
        if isinstance(row, Mapping)
    }
    recovery = {
        int(row["seed"]): row["receipt"]
        for row in binding["frozen_chain"].get("recovery_receipts", [])
        if isinstance(row, Mapping)
    }
    if raw != expected_raw or recovery != expected_recovery:
        raise ValueError("Speed CEM binding does not preserve all raw/recovery receipt identities")
    checkpoints = binding["frozen_chain"].get("checkpoints")
    if not isinstance(checkpoints, list) or {
        int(row.get("seed", -1)) for row in checkpoints if isinstance(row, Mapping)
    } != set(SPEED_DEVELOPMENT_SEEDS):
        raise ValueError("Speed CEM binding does not preserve all three fixed checkpoints")
    # Re-run the CEM runner's no-inference binding preflight for each seed.
    # It rehashes the CEM source closure, runtime, normalizer, catalogs and
    # raw/recovery chain selected before Public ICL; it does not execute CEM.
    for track in ("action_planning_cem", "original_task_retention_cem"):
        for seed in SPEED_DEVELOPMENT_SEEDS:
            validated, _track, _checkpoint, _output, _work = speed_cem.runner._validate_binding(
                binding_path, track=track, seed=seed
            )
            if validated != binding:
                raise ValueError("Speed CEM binding changed during no-inference preflight")
    return binding_identity, binding


def _speed_cem_aggregate(
    *,
    path: str,
    completion_id: str,
    expected_kind: str,
    expected_seeds: list[int],
    expected_development: Mapping[str, Any],
    speed_cem_binding: tuple[Mapping[str, Any], Mapping[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    """Validate Speed's ledger-backed descriptive or paired-retention result."""

    from scripts import freeze_tworoom_speed_pldm_cem_aggregate_v1 as speed_cem

    binding_identity, binding = speed_cem_binding
    identity = _identity(path, repo_root=repo_root)
    payload = _read_json(path, repo_root=repo_root)
    metric = _mapping(payload.get("metric"), label=f"Speed {expected_kind} metric")
    if not (
        payload.get("schema_version") == 1
        and payload.get("completion_id") == completion_id
        and payload.get("cem_binding_id") == "tworoom_speed_pldm_cem_binding_v1"
        and payload.get("binding") == dict(binding_identity)
        and payload.get("evaluation_kind") == expected_kind
        and payload.get("development") == dict(expected_development)
        and set(metric) == {"id", "label"}
        and payload.get("output", {}).get("path") == path
        and payload.get("output", {}).get("content_sha256_not_embedded_to_avoid_self_reference")
        is True
        and payload.get("input_integrity", {}).get(
            "all_bound_inputs_unchanged_during_aggregate_read"
        )
        is True
        and payload.get("input_integrity", {}).get("identities_after_aggregate_read")
        == payload.get("input_integrity", {}).get("identities_before_aggregate_read")
    ):
        raise ValueError(f"Speed {expected_kind} aggregate is not an intact ledger freeze")
    track = binding.get("tracks", {}).get(expected_kind)
    if not isinstance(track, Mapping):
        raise ValueError(f"Speed CEM binding lacks {expected_kind}")
    output_path = resolve_contextworld_path(path, repo_root=repo_root)
    expected_output = speed_cem._aggregate_output(track)
    if output_path.resolve() != expected_output.resolve():
        raise ValueError(f"Speed {expected_kind} aggregate output is not canonical")
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(expected_seeds):
        raise ValueError(f"Speed {expected_kind} aggregate lacks three checkpoint receipts")
    by_seed = {
        _integer(row.get("training_seed"), label=f"Speed {expected_kind} seed"): row
        for row in checkpoints
        if isinstance(row, Mapping)
    }
    if set(by_seed) != set(expected_seeds) or len(by_seed) != len(checkpoints):
        raise ValueError(f"Speed {expected_kind} aggregate seed closure is invalid")
    ledger_items = payload["input_integrity"].get("ledgers")
    if not isinstance(ledger_items, list) or len(ledger_items) != len(expected_seeds):
        raise ValueError(f"Speed {expected_kind} aggregate does not list all ledger inputs")
    ledger_sources = {
        _integer(item.get("training_seed"), label="Speed CEM ledger seed"): _identity_mapping(
            item.get("source"), label="Speed CEM ledger source", repo_root=repo_root
        )
        for item in ledger_items
        if isinstance(item, Mapping)
    }
    if set(ledger_sources) != set(expected_seeds) or len(ledger_sources) != len(ledger_items):
        raise ValueError(f"Speed {expected_kind} aggregate ledger source closure is invalid")

    validated_ledgers: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for seed in expected_seeds:
        terminal, ledger_identity = speed_cem._ledger_rows(
            binding=binding,
            binding_identity=binding_identity,
            track_name=expected_kind,
            track=track,
            seed=seed,
        )
        item = _mapping(by_seed[seed], label=f"Speed {expected_kind} checkpoint {seed}")
        if item.get("source") != ledger_identity or ledger_sources[seed] != ledger_identity:
            raise ValueError(f"Speed {expected_kind} aggregate source does not bind seed {seed} ledger")
        validated_ledgers.append((terminal, ledger_identity))

    if expected_kind == "action_planning_cem":
        if not (
            payload.get("status") == "completed_executed_valid_descriptive"
            and payload.get("result_semantics") == "EXECUTED_VALID_DESCRIPTIVE"
            and metric.get("id") == track.get("metric", {}).get("id")
            and payload.get("decision")
            == {
                "execution_valid": True,
                "model_performance_gate": None,
                "retention_result": "NOT_APPLICABLE",
                "result": "EXECUTED_VALID_DESCRIPTIVE",
            }
        ):
            raise ValueError("Speed action-planning CEM improperly declares a performance PASS/FAIL")
        rows = []
        for terminal, ledger_identity in validated_ledgers:
            speed_cem._expected_action_records(track, terminal["records"])
            item = by_seed[int(terminal["training_seed"])]
            aggregate = _mapping(terminal.get("aggregate"), label="Speed action CEM ledger aggregate")
            if not (
                set(item) == {"training_seed", "value", "execution_valid", "source"}
                and item.get("execution_valid") is True
                and math.isclose(
                    _fraction(item.get("value"), label="Speed action CEM value"),
                    _fraction(aggregate.get("success_rate"), label="Speed action CEM ledger value"),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise ValueError("Speed action CEM checkpoint is not descriptive execution evidence")
            rows.append(
                {
                    "training_seed": int(terminal["training_seed"]),
                    "value": float(item["value"]),
                    "execution_valid": True,
                    "source": ledger_identity,
                }
            )
        rows.sort(key=lambda row: row["training_seed"])
        return {
            "metric": dict(metric),
            "records": rows,
            "aggregate": identity,
            "decision": dict(payload["decision"]),
            "result_semantics": "EXECUTED_VALID_DESCRIPTIVE",
        }

    if expected_kind != "original_task_retention_cem":
        raise ValueError(f"Unknown Speed CEM kind: {expected_kind}")
    criteria = track.get("metric", {}).get("paired_noninferiority")
    if not isinstance(criteria, Mapping):
        raise ValueError("Speed retention CEM binding lacks paired criteria")
    if not (
        payload.get("status") == "completed_paired_retention_evaluation"
        and payload.get("result_semantics") == "PAIRED_NONINFERIORITY_RETENTION"
        and metric.get("id") == track.get("metric", {}).get("id")
        and payload.get("paired_baseline") == track.get("paired_baseline")
    ):
        raise ValueError("Speed retention CEM aggregate schema is invalid")
    rows = []
    for terminal, ledger_identity in validated_ledgers:
        baseline, candidate = speed_cem._expected_retention_records(track, terminal["records"])
        comparison = speed_cem._paired_retention_result(
            baseline=baseline, candidate=candidate, criteria=criteria
        )
        item = by_seed[int(terminal["training_seed"])]
        aggregate = _mapping(terminal.get("aggregate"), label="Speed retention CEM ledger aggregate")
        if not (
            set(item)
            == {"training_seed", "value", "passed", "source", "paired_noninferiority"}
            and type(item.get("passed")) is bool
            and item.get("passed") is comparison["passed"]
            and item.get("paired_noninferiority") == comparison
            and math.isclose(
                _fraction(item.get("value"), label="Speed retention CEM value"),
                _fraction(aggregate.get("success_rate"), label="Speed retention CEM ledger value"),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("Speed retention CEM checkpoint is not an exact paired comparison")
        rows.append(
            {
                "training_seed": int(terminal["training_seed"]),
                "value": float(item["value"]),
                "passed": bool(item["passed"]),
                "source": ledger_identity,
                "paired_noninferiority": comparison,
            }
        )
    rows.sort(key=lambda row: row["training_seed"])
    all_passed = all(row["passed"] for row in rows)
    expected_decision = {
        "all_training_seeds_passed": all_passed,
        "passed": all_passed,
        "result": "PASS" if all_passed else "FAIL",
        "criterion": "all_three_fixed_checkpoints_must_pass_paired_noninferiority",
    }
    if payload.get("decision") != expected_decision:
        raise ValueError("Speed retention CEM final decision is not the all-three paired gate")
    return {
        "metric": dict(metric),
        "records": rows,
        "aggregate": identity,
        "decision": expected_decision,
        "result_semantics": "PAIRED_NONINFERIORITY_RETENTION",
    }


def _cem_aggregate(
    *,
    component: str,
    path: str,
    completion_id: str,
    expected_kind: str,
    expected_seeds: list[int],
    expected_development: Mapping[str, Any] | None = None,
    speed_cem_binding: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
    repo_root: Path,
) -> dict[str, Any]:
    """Read a compact, one-use CEM aggregate emitted after a positive ICL gate."""

    if component == "speed":
        if expected_development is None or speed_cem_binding is None:
            raise ValueError("Speed CEM aggregate needs its passed Development and CEM binding chains")
        return _speed_cem_aggregate(
            path=path,
            completion_id=completion_id,
            expected_kind=expected_kind,
            expected_seeds=expected_seeds,
            expected_development=expected_development,
            speed_cem_binding=speed_cem_binding,
            repo_root=repo_root,
        )

    identity = _identity(path, repo_root=repo_root)
    payload = _read_json(path, repo_root=repo_root)
    metric = _mapping(payload.get("metric"), label=f"{component} {expected_kind} metric")
    if (
        payload.get("schema_version") != 1
        or payload.get("completion_id") != completion_id
        or payload.get("evaluation_kind") != expected_kind
        or set(metric) != {"id", "label"}
    ):
        raise ValueError(f"{component} {expected_kind} aggregate is invalid")
    if expected_development is not None:
        raise ValueError("Only Speed CEM aggregates may carry the Speed Development chain")
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        raise ValueError(f"{component} {expected_kind} aggregate must contain three checkpoints")
    rows = []
    for row in checkpoints:
        item = _mapping(row, label=f"{component} {expected_kind} checkpoint")
        if set(item) != {"training_seed", "value", "passed", "source"}:
            raise ValueError(f"{component} {expected_kind} checkpoint keys are invalid")
        source = _identity_mapping(
            item["source"], label=f"{component} {expected_kind} source", repo_root=repo_root
        )
        if type(item["passed"]) is not bool:
            raise ValueError(f"{component} {expected_kind} gate must be boolean")
        rows.append(
            {
                "training_seed": _integer(item["training_seed"], label=f"{component} {expected_kind} seed"),
                "value": _fraction(item["value"], label=f"{component} {expected_kind} value"),
                "passed": item["passed"],
                "source": source,
            }
        )
    if {row["training_seed"] for row in rows} != set(expected_seeds):
        raise ValueError(f"{component} {expected_kind} aggregate does not cover preregistered seeds")
    decision = _mapping(payload.get("decision"), label=f"{component} {expected_kind} decision")
    if type(decision.get("passed")) is not bool or decision["passed"] is not all(
        row["passed"] for row in rows
    ):
        raise ValueError(f"{component} {expected_kind} decision is invalid")
    rows.sort(key=lambda row: row["training_seed"])
    return {"metric": dict(metric), "records": rows, "aggregate": identity, "decision": dict(decision)}


def _cem_not_authorized_stop(
    *,
    component: str,
    path: str,
    completion_id: str,
    public_icl_identity: Mapping[str, Any],
    ability_passed: bool,
    passed_checkpoints: int,
    gate_by_seed: Mapping[int, bool],
    expected_development: Mapping[str, Any] | None = None,
    repo_root: Path,
) -> dict[str, Any]:
    if ability_passed:
        raise ValueError(f"{component} cannot use a CEM-stop receipt after a 3/3 ICL pass")
    identity = _identity(path, repo_root=repo_root)
    payload = _read_json(path, repo_root=repo_root)
    if component == "action_strength":
        preregistration, _preregistration_identity, _entries = _action_recovery_preregistration(
            repo_root=repo_root
        )
        frozen_inputs = _mapping(
            preregistration.get("frozen_inputs"), label="ActionStrength recovery frozen inputs"
        )
        cem = _mapping(payload.get("cem"), label="ActionStrength CEM-stop CEM state")
        action_planning = _mapping(
            cem.get("action_planning_cem"), label="ActionStrength CEM-stop action-planning state"
        )
        retention = _mapping(
            cem.get("original_pusht_retention_cem"), label="ActionStrength CEM-stop retention state"
        )
        public = _mapping(payload.get("public_icl"), label="ActionStrength CEM-stop Public ICL")
        expected_gate_map = {str(seed): passed for seed, passed in sorted(gate_by_seed.items())}
        output = _mapping(payload.get("output"), label="ActionStrength CEM-stop output")
        scope = _mapping(payload.get("scope"), label="ActionStrength CEM-stop scope")
        binding = _mapping(payload.get("evaluation_binding"), label="ActionStrength CEM-stop binding")
        release_reference = payload.get("release_config")
        recovery_preregistration = payload.get("recovery_preregistration")
        aggregate_reference = _reference_identity(
            payload.get("public_icl_aggregate"),
            label="ActionStrength CEM-stop Public ICL aggregate",
            repo_root=repo_root,
        )
        integrity = _mapping(payload.get("input_integrity"), label="ActionStrength CEM-stop input integrity")
        before = _mapping(
            integrity.get("identities_before_stop_freeze"),
            label="ActionStrength CEM-stop identities before freeze",
        )
        if not (
            payload.get("schema_version") == 1
            and payload.get("completion_id") == completion_id
            and payload.get("status")
            == "frozen_cem_not_authorized_after_failed_three_seed_public_icl"
            and output.get("path") == path
            and output.get("content_sha256_not_embedded_to_avoid_self_reference") is True
            and cem.get("authorized") is False
            and cem.get("executed") is False
            and action_planning == {"authorized": False, "executed": False}
            and retention == {"authorized": False, "executed": False}
            and isinstance(cem.get("reason"), str)
            and cem["reason"].strip()
            and public.get("passed") is False
            and public.get("passed_checkpoints") == passed_checkpoints
            and public.get("evaluated_checkpoints") == 3
            and public.get("raw_public_gate_passed") == expected_gate_map
            and public.get("float32_recovered_gate_passed") == expected_gate_map
            and _same_path(aggregate_reference["path"], public_icl_identity["path"], repo_root=repo_root)
            and aggregate_reference["sha256"] == public_icl_identity["sha256"]
            and _historical_reference_matches_spec(
                release_reference, frozen_inputs["release_config"], label="ActionStrength CEM-stop release config"
            )
            and _historical_reference_matches_spec(
                recovery_preregistration,
                {"path": ACTION_STRENGTH_RECOVERY_PREREGISTRATION, "sha256": _identity(ACTION_STRENGTH_RECOVERY_PREREGISTRATION, repo_root=repo_root)["sha256"]},
                label="ActionStrength CEM-stop recovery preregistration",
            )
            and _historical_reference_matches_spec(
                binding.get("config"), frozen_inputs["evaluation_binding_config"], label="ActionStrength CEM-stop binding config"
            )
            and _historical_reference_matches_spec(
                binding.get("receipt"), frozen_inputs["evaluation_binding_receipt"], label="ActionStrength CEM-stop binding receipt"
            )
            and scope
            == {
                "model_evaluation_rerun_performed": False,
                "public_test_reopened": False,
                "action_planning_cem_executed": False,
                "original_pusht_retention_cem_executed": False,
                "checkpoint_selection_performed": False,
            }
            and integrity.get("all_frozen_inputs_unchanged_during_stop_freeze") is True
            and integrity.get("identities_after_stop_freeze") == before
        ):
            raise ValueError("ActionStrength CEM-not-authorized stop receipt is invalid")
        amendment = _read_yaml(ACTION_STRENGTH_SCORE_CONSISTENCY_AMENDMENT, repo_root=repo_root)
        formal_evidence = _mapping(
            amendment.get("frozen_formal_evidence"), label="ActionStrength amendment formal evidence"
        )
        amendment_stop = _reference_identity(
            _mapping(formal_evidence.get("independent_float32_recovery"), label="ActionStrength amendment recovery").get("cem_stop"),
            label="ActionStrength amendment CEM stop",
            repo_root=repo_root,
        )
        if amendment_stop != identity:
            raise ValueError("ActionStrength amendment does not bind the frozen CEM-stop receipt")
        return identity
    if (
        payload.get("schema_version") != 1
        or payload.get("completion_id") != completion_id
        or _mapping(payload.get("cem"), label=f"{component} CEM-stop CEM state")
        != {"authorized": False, "executed": False}
    ):
        raise ValueError(f"{component} CEM-not-authorized stop receipt is invalid")
    if component == "speed":
        if expected_development is None or payload.get("development") != dict(
            expected_development
        ):
            raise ValueError(
                "Speed CEM-not-authorized stop does not bind the passed Development chain"
            )
    elif expected_development is not None:
        raise ValueError("Only Speed CEM stops may carry the Speed Development chain")
    public = _mapping(payload.get("public_icl"), label=f"{component} CEM-stop Public ICL")
    if not 0 <= passed_checkpoints < 3:
        raise ValueError(f"{component} CEM-stop must follow a non-passing three-seed ICL outcome")
    if (
        public.get("passed") is not False
        or public.get("passed_checkpoints") != passed_checkpoints
        or public.get("evaluated_checkpoints") != 3
    ):
        raise ValueError(
            f"{component} CEM-stop receipt does not prove the exact "
            f"{passed_checkpoints}/3 Public ICL outcome"
        )
    aggregate_reference = _reference_identity(
        payload.get("public_icl_aggregate"),
        label=f"{component} CEM-stop Public ICL aggregate",
        repo_root=repo_root,
    )
    if (
        not _same_path(aggregate_reference["path"], public_icl_identity["path"], repo_root=repo_root)
        or aggregate_reference["sha256"] != public_icl_identity["sha256"]
    ):
        raise ValueError(f"{component} CEM-stop receipt binds a different Public ICL aggregate")
    return identity


def _speed_behavioral_claim_boundary(
    *,
    path: str,
    completion_id: str,
    release_id: str,
    completion_identity: Mapping[str, Any],
    release_identity: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Validate the pre-evaluation boundary on what a Speed row may claim."""

    identity = _identity(path, repo_root=repo_root)
    payload = _read_yaml(path, repo_root=repo_root)
    chronology = _mapping(payload.get("chronology"), label="Speed behavioral-boundary chronology")
    frozen_inputs = _mapping(payload.get("frozen_inputs"), label="Speed behavioral-boundary inputs")
    boundary = _mapping(payload.get("claim_boundary"), label="Speed behavioral claim boundary")
    conditional = _mapping(payload.get("conditional_evaluation"), label="Speed behavioral conditional evaluation")
    mutation = _mapping(payload.get("mutation_boundary"), label="Speed behavioral mutation boundary")
    if (
        payload.get("schema_version") != 1
        or payload.get("amendment_id") != "tworoom_speed_pldm_behavioral_claim_boundary_v1"
        or payload.get("completion_id") != completion_id
        or payload.get("release_id") != release_id
        or payload.get("status")
        != "preregistered_during_fixed_training_before_development_or_public_evaluation"
        or chronology
        != {
            "fixed_training_already_running": True,
            "development_evaluation_started": False,
            "public_test_opened": False,
            "checkpoint_selection_changed": False,
            "training_budget_changed": False,
        }
        or not _historical_reference_matches_spec(
            frozen_inputs.get("completion_config"), completion_identity, label="Speed behavioral completion config"
        )
        or not _historical_reference_matches_spec(
            frozen_inputs.get("speed_release"), release_identity, label="Speed behavioral release config"
        )
        or not isinstance(_mapping(frozen_inputs.get("public_scorer"), label="Speed behavioral scorer").get("path"), str)
        or _reference_identity(
            frozen_inputs.get("public_scorer"), label="Speed behavioral scorer", repo_root=repo_root
        )["sha256"]
        != _mapping(frozen_inputs.get("public_scorer"), label="Speed behavioral scorer")["sha256"]
        or boundary.get("paired_single_speed_pldm_controls_trained") is not False
        or boundary.get("training_attribution_claim_authorized") is not False
        or boundary.get("training_attributed_speed_icl_claim_authorized") is not False
        or boundary.get("three_seed_behavioral_reference_authorized") is not True
        or boundary.get("scoreboard_evidence_scope_if_reported") != "behavioral"
        or boundary.get("method_name_must_identify_pldm") is not True
        or conditional.get("development_must_precede_public") is not True
        or conditional.get("public_test_authorized_by_this_record") is not False
        or conditional.get("public_test_requires_separate_passed_binding") is not True
        or _mapping(
            conditional.get("if_three_seed_public_behavioral_gate_passes"),
            label="Speed behavioral positive branch",
        )
        != {
            "action_planning_cem_may_be_separately_authorized": True,
            "original_tworoom_retention_cem_may_be_separately_authorized": True,
        }
        or _mapping(
            conditional.get("if_any_public_behavioral_gate_fails"),
            label="Speed behavioral negative branch",
        )
        != {
            "cem_authorized": False,
            "cem_executed": False,
            "terminal_stop_receipt_required": True,
        }
        or any(value is not False for value in mutation.values())
    ):
        raise ValueError("Speed behavioral claim-boundary record is invalid")
    return identity


def _formal_completion_material(
    component: str,
    specification: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    completion, completion_identity, release_path, expected_seeds = _load_completion(
        specification, repo_root=repo_root
    )
    release = _read_yaml(release_path, repo_root=repo_root)
    release_identity = _identity(release_path, repo_root=repo_root)
    release_id = release.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise ValueError(f"{component} release config is invalid")
    formal = _mapping(specification.get("formal_result"), label=f"{component} formal result")
    behavioral_claim_boundary = None
    if component == "speed":
        behavioral_claim_boundary = _speed_behavioral_claim_boundary(
            path=str(formal["behavioral_claim_boundary"]),
            completion_id=str(specification["completion_id"]),
            release_id=release_id,
            completion_identity=completion_identity,
            release_identity=release_identity,
            repo_root=repo_root,
        )
    public_icl = _public_icl_from_recovery(
        component,
        specification,
        completion_id=str(specification["completion_id"]),
        release_id=release_id,
        expected_seeds=expected_seeds,
        repo_root=repo_root,
    )
    if public_icl["ability_passed"]:
        if public_icl["cem"] != {"authorized": True, "executed": False}:
            raise ValueError(f"{component} 3/3 Public ICL pass lacks a pre-CEM authorization gate")
        speed_cem_binding = (
            _speed_positive_cem_binding(
                path=str(formal["cem_binding"]),
                completion_id=str(specification["completion_id"]),
                public_icl=public_icl,
                expected_development=_mapping(
                    public_icl.get("development"), label="Speed passed Development chain"
                ),
                repo_root=repo_root,
            )
            if component == "speed"
            else None
        )
        planning = _cem_aggregate(
            component=component,
            path=str(formal["action_planning_aggregate"]),
            completion_id=str(specification["completion_id"]),
            expected_kind="action_planning_cem",
            expected_seeds=expected_seeds,
            expected_development=(
                public_icl.get("development") if component == "speed" else None
            ),
            speed_cem_binding=speed_cem_binding,
            repo_root=repo_root,
        )
        retention = _cem_aggregate(
            component=component,
            path=str(formal["original_task_retention_aggregate"]),
            completion_id=str(specification["completion_id"]),
            expected_kind="original_task_retention_cem",
            expected_seeds=expected_seeds,
            expected_development=(
                public_icl.get("development") if component == "speed" else None
            ),
            speed_cem_binding=speed_cem_binding,
            repo_root=repo_root,
        )
        stop = None
        cem_finalization = {
            "authorized": True,
            "executed": True,
            "action_planning_aggregate": planning["aggregate"],
            "original_task_retention_aggregate": retention["aggregate"],
            **(
                {"cem_binding": speed_cem_binding[0]}
                if speed_cem_binding is not None
                else {}
            ),
        }
        outcome = "passed_public_icl_cem_completed"
    else:
        if public_icl["cem"] != {"authorized": False, "executed": False}:
            raise ValueError(f"{component} nonpassing Public ICL must keep CEM unexecuted")
        stop = _cem_not_authorized_stop(
            component=component,
            path=str(formal["cem_not_authorized_stop"]),
            completion_id=str(specification["completion_id"]),
            public_icl_identity=public_icl["aggregate"],
            ability_passed=False,
            passed_checkpoints=sum(row["passed"] for row in public_icl["records"]),
            gate_by_seed={
                int(row["training_seed"]): bool(row["passed"])
                for row in public_icl["records"]
            },
            expected_development=(
                public_icl.get("development") if component == "speed" else None
            ),
            repo_root=repo_root,
        )
        planning = None
        retention = None
        cem_finalization = {
            "authorized": False,
            "executed": False,
            "not_authorized_stop": stop,
        }
        outcome = "failed_public_icl_cem_not_authorized"
    reader = {
        "component_id": specification["component_id"],
        "component_name": specification["component_name"],
        "method_name": formal["method_name"],
        "evidence_scope": "behavioral",
        "primary_metric": public_icl["metric"],
    }
    training_attribution = {
        "claim": False,
        "paired_training_controls_available": False,
        "reason": (
            "该参考完成没有预注册的配对训练对照；Public ICL 行仅报告行为结果，"
            "不把表现归因于合成训练因素。"
        ),
    }
    return {
        "completion_id": specification["completion_id"],
        "finalized": True,
        "outcome": outcome,
        "public_scoreboard_row_included": True,
        "completion_config": completion_identity,
        "release_config": release_identity,
        "formal_public_icl": public_icl,
        "action_planning_cem": planning,
        "original_task_retention_cem": retention,
        "cem_not_authorized_stop": stop,
        "cem_finalization": cem_finalization,
        "training_attribution": training_attribution,
        **({"behavioral_claim_boundary": behavioral_claim_boundary} if behavioral_claim_boundary else {}),
        "reader_result": reader,
    }


def _development_only_material(
    component: str,
    specification: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    completion, completion_identity, _release_path, expected_seeds = _load_completion(
        specification, repo_root=repo_root, require_icl_release=False
    )
    pilot_seed = _mapping(completion.get("training"), label=f"{component} training").get(
        "pilot_seed"
    )
    if _integer(pilot_seed, label=f"{component} pilot seed") not in expected_seeds:
        raise ValueError(f"{component} pilot seed is not preregistered")
    identities = {
        name: _identity(str(specification[name]), repo_root=repo_root)
        for name in (
            "development_decision",
            "evaluation_binding",
            "development_evaluation",
            "development_rescore",
        )
    }
    decision = _read_json(identities["development_decision"]["path"], repo_root=repo_root)
    binding = _read_json(identities["evaluation_binding"]["path"], repo_root=repo_root)
    evaluation = _read_json(identities["development_evaluation"]["path"], repo_root=repo_root)
    rescore = _read_json(identities["development_rescore"]["path"], repo_root=repo_root)
    if (
        decision.get("schema_version") != 1
        or decision.get("status") != specification["required_terminal_status"]
        or decision.get("cem") != {"authorized": False, "executed": False}
        or decision.get("next_stage", {}).get("formal_failure_frozen") is not True
        or decision.get("next_stage", {}).get("public_test_authorized") is not False
        or decision.get("next_stage", {}).get("cem_authorized") is not False
        or decision.get("development", {}).get("gate", {}).get("passed") is not False
        or decision.get("development", {}).get("identities_match_binding") is not True
        or decision.get("development", {}).get("independent_rescore_matches") is not True
    ):
        raise ValueError(f"{component} Development-only terminal decision is invalid")
    public = decision.get("public_test")
    expected_public_false = {
        "opened",
        "read",
        "hashed",
        "scored",
        "accessed_by_binding",
        "accessed_by_evaluator",
        "path_decoded",
        "path_hashed",
        "path_resolved",
        "path_statted",
        "path_walked",
    }
    if not isinstance(public, Mapping) or any(public.get(name) is not False for name in expected_public_false):
        raise ValueError(f"{component} Development-only decision touched Public Test")
    bound_completion = _mapping(binding.get("completion"), label=f"{component} binding completion")
    if (
        binding.get("schema_version") != 1
        or binding.get("component") != component
        or binding.get("status") != "passed"
        or binding.get("passed") is not True
        or bound_completion.get("id") != specification["completion_id"]
        or bound_completion.get("sha256") != completion_identity["sha256"]
        or binding.get("cem") != {"authorized": False, "executed": False}
    ):
        raise ValueError(f"{component} Development-only binding is invalid")
    binding_public = binding.get("public_test")
    if not isinstance(binding_public, Mapping) or any(
        binding_public.get(name) is not False for name in expected_public_false
    ):
        raise ValueError(f"{component} Development-only binding touched Public Test")
    for label, payload in (("evaluation", evaluation), ("rescore", rescore)):
        if (
            payload.get("schema_version") != 1
            or payload.get("claim_scope") != "Development_only_not_Public_or_release"
            or payload.get("data", {}).get("split") != "Development"
            or payload.get("gate", {}).get("passed") is not False
            or payload.get("public_test", {}).get("opened") is not False
            or payload.get("public_test", {}).get("read") is not False
            or payload.get("public_test", {}).get("scored") is not False
        ):
            raise ValueError(f"{component} Development-only {label} receipt is invalid")
    if identities["development_evaluation"]["sha256"] != identities["development_rescore"]["sha256"]:
        raise ValueError(f"{component} Development evaluation and independent rescore differ")
    decision_binding = _mapping(decision.get("binding"), label=f"{component} decision binding")
    decision_evaluation = _mapping(
        _mapping(decision.get("development"), label=f"{component} decision development").get(
            "evaluation"
        ),
        label=f"{component} decision evaluation",
    )
    decision_rescore = _mapping(
        _mapping(decision.get("development"), label=f"{component} decision development").get(
            "independent_rescore"
        ),
        label=f"{component} decision rescore",
    )
    for label, declared, actual in (
        ("binding", decision_binding, identities["evaluation_binding"]),
        ("evaluation", decision_evaluation, identities["development_evaluation"]),
        ("rescore", decision_rescore, identities["development_rescore"]),
    ):
        if (
            not isinstance(declared.get("path"), str)
            or not _same_path(str(declared["path"]), actual["path"], repo_root=repo_root)
            or declared.get("sha256") != actual["sha256"]
        ):
            raise ValueError(f"{component} terminal decision does not bind its {label} receipt")
    return {
        "completion_id": specification["completion_id"],
        "finalized": True,
        "outcome": specification["required_terminal_status"],
        "public_scoreboard_row_included": False,
        "completion_config": completion_identity,
        "development_decision": identities["development_decision"],
        "evaluation_binding": identities["evaluation_binding"],
        "development_evaluation": identities["development_evaluation"],
        "development_rescore": identities["development_rescore"],
        "public_test_accessed": False,
        "cem_executed": False,
    }


def _row_from_formal_material(
    component: str,
    material: Mapping[str, Any],
    *,
    baseline_value: float,
) -> dict[str, Any]:
    reader = _mapping(material.get("reader_result"), label=f"{component} reader result")
    public_icl = _mapping(material.get("formal_public_icl"), label=f"{component} Public ICL")
    values = [float(row["value"]) for row in public_icl["records"]]
    gate_passes = [bool(row["passed"]) for row in public_icl["records"]]
    primary_metric = _mapping(reader.get("primary_metric"), label=f"{component} primary metric")
    row: dict[str, Any] = {
        "component_id": reader["component_id"],
        "component_name": reader["component_name"],
        "method_name": reader["method_name"],
        "primary_metric": {
            "id": primary_metric["id"],
            "label": primary_metric["label"],
            "per_seed_values": values,
        },
        "per_seed_gate_passes": gate_passes,
        "ability_passed": all(gate_passes),
        "required_training_seeds": 3,
        "evidence_scope": reader["evidence_scope"],
    }
    retention_value = material.get("original_task_retention_cem")
    if retention_value is None:
        row["original_task_retention"] = {
            "result": "NOT_EVALUATED",
            "reason": (
                "该方法未通过三枚 Public ICL 检查；按预注册规则，"
                "原任务 CEM 未获授权且未执行。"
            ),
        }
    else:
        retention = _mapping(retention_value, label=f"{component} retention")
        retention_values = [float(entry["value"]) for entry in retention["records"]]
        retention_passes = [bool(entry["passed"]) for entry in retention["records"]]
        retention_metric = _mapping(retention.get("metric"), label=f"{component} retention metric")
        row["original_task_retention"] = {
            "result": "PASS" if all(retention_passes) else "FAIL",
            "metric_id": retention_metric["id"],
            "metric_label": retention_metric["label"],
            "per_seed_values": retention_values,
            "baseline_value": baseline_value,
        }
    return row


def _probe_public_icl_branch(
    component: str,
    specification: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[bool | None, str | None]:
    """Determine the conditional CEM branch from the three raw ICL receipts.

    This deliberately reads only the immutable raw Public ICL outputs.  The
    recovery aggregate is still independently validated later.  Separating
    the probe lets check-only report the correct *conditional* missing input
    without pretending that both the CEM and CEM-stop branches are required.
    """

    try:
        completion, _completion_identity, release_path, seeds = _load_completion(
            specification, repo_root=repo_root
        )
        if release_path is None:
            raise ValueError("formal completion has no ICL release config")
        release = _read_yaml(release_path, repo_root=repo_root)
        release_id = release.get("release_id")
        if not isinstance(release_id, str) or not release_id:
            raise ValueError("release config has no release_id")
        formal = _mapping(specification.get("formal_result"), label=f"{component} formal result")
        metric = _metric_definition(formal.get("primary_metric"), label=f"{component} primary metric")
        raw_root = str(formal["raw_public_results_root"])
        gates: list[bool] = []
        for seed in seeds:
            raw_path = str(Path(raw_root) / f"seed_{seed}.json")
            if not resolve_contextworld_path(raw_path, repo_root=repo_root).is_file():
                return None, None
            payload = _read_json(raw_path, repo_root=repo_root)
            if not _matches_raw_public_result_contract(
                payload,
                component=component,
                expected_release_id=release_id,
            ):
                raise ValueError("raw Public ICL receipt is not a completed result for this release")
            observed_seed = _integer(
                _at_path(payload, metric["seed_path"], label=f"{component} raw Public ICL"),
                label=f"{component} raw Public ICL training seed",
            )
            _fraction(
                _at_path(payload, metric["value_path"], label=f"{component} raw Public ICL"),
                label=f"{component} raw Public ICL metric",
            )
            gate = _at_path(payload, metric["gate_path"], label=f"{component} raw Public ICL")
            if observed_seed != seed or type(gate) is not bool:
                raise ValueError("raw Public ICL receipt does not bind its preregistered seed and gate")
            gates.append(gate)
        return all(gates), None
    except (FileNotFoundError, ValueError) as error:
        return None, str(error)


def _required_input_paths(
    preregistration: Mapping[str, Any], *, repo_root: Path
) -> tuple[list[str], list[str]]:
    """List unconditional and currently selected conditional inputs.

    A CEM path is required only after the three raw Public ICL gates establish
    a 3/3 pass.  Otherwise the stop receipt is required.  When the raw gates
    are unavailable, the audit reports that the branch is unresolved rather
    than inventing requirements for both mutually exclusive branches.
    """

    historical = _mapping(preregistration.get("historical_base"), label="historical base")
    baseline = _mapping(preregistration.get("original_baseline_cem"), label="baseline")
    addendum = _mapping(preregistration.get("scoreboard_addendum"), label="addendum")
    inputs = _mapping(preregistration.get("completion_inputs"), label="completion inputs")
    paths: list[str] = [
        str(historical["specification"]["path"]),
        str(historical["scoreboard"]["path"]),
        str(baseline["path"]),
        str(addendum["preregistration"]),
    ]
    blockers: list[str] = []
    for component in COMPONENTS:
        specification = _mapping(inputs[component], label=f"{component} input")
        paths.append(str(specification["completion_config"]))
        if component in FORMAL_COMPONENTS:
            formal = _mapping(specification["formal_result"], label=f"{component} formal result")
            paths.append(str(formal["public_icl_aggregate"]))
            if component == "speed":
                paths.append(str(formal["behavioral_claim_boundary"]))
            raw_root = str(formal["raw_public_results_root"])
            recovery_root = str(formal["recovery_root"])
            completion_path = str(specification["completion_config"])
            if resolve_contextworld_path(completion_path, repo_root=repo_root).is_file():
                try:
                    completion = _read_yaml(completion_path, repo_root=repo_root)
                    seeds = _expected_training_seeds(completion)
                    if component == "speed":
                        development = _speed_development_declaration(
                            formal, expected_seeds=seeds
                        )
                        paths.extend(
                            (
                                development["config"],
                                development["manifest"],
                                *(row["path"] for row in development["receipts"]),
                            )
                        )
                except ValueError as error:
                    blockers.append(f"{component}: cannot read preregistered training seeds: {error}")
                    seeds = []
                for seed in seeds:
                    paths.append(str(Path(raw_root) / f"seed_{seed}.json"))
                    paths.append(str(Path(recovery_root) / f"seed_{seed}.json"))
                branch, branch_error = _probe_public_icl_branch(
                    component, specification, repo_root=repo_root
                )
                if branch is True:
                    paths.extend(
                        (
                            str(formal["cem_binding"]),
                            str(formal["action_planning_aggregate"]),
                            str(formal["original_task_retention_aggregate"]),
                        )
                    )
                elif branch is False:
                    paths.append(str(formal["cem_not_authorized_stop"]))
                else:
                    detail = f": {branch_error}" if branch_error else ""
                    blockers.append(
                        f"{component}: conditional CEM branch is unresolved until all three raw Public ICL gates are available and valid{detail}"
                    )
            else:
                blockers.append(
                    f"{component}: conditional CEM branch is unresolved until its completion config is available"
                )
        else:
            paths.extend(
                str(specification[name])
                for name in (
                    "development_decision",
                    "evaluation_binding",
                    "development_evaluation",
                    "development_rescore",
                )
            )
    return paths, blockers


def audit_completion_aggregate_readiness(
    *,
    aggregate_config: Path | str = AGGREGATE_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Report every presently knowable blocker without creating any output."""

    root = (repo_root or repository_root()).resolve()
    try:
        preregistration = load_completion_aggregate_preregistration(
            aggregate_config, repo_root=root
        )
    except (FileNotFoundError, ValueError) as error:
        return {
            "schema_version": 1,
            "aggregate_id": AGGREGATE_ID,
            "status": "blocked_pending_four_final_outcomes",
            "ready": False,
            "blockers": [str(error)],
            "outputs_created": False,
        }
    required_paths, branch_blockers = _required_input_paths(preregistration, repo_root=root)
    blockers = [
        f"missing required evidence: {path}"
        for path in required_paths
        if not resolve_contextworld_path(path, repo_root=root).is_file()
    ]
    blockers.extend(branch_blockers)
    validators = [
        ("historical base", lambda: _validate_historical_base(preregistration, repo_root=root)),
        ("original baseline CEM", lambda: _validate_baseline_cem(preregistration, repo_root=root)),
    ]
    for component in FORMAL_COMPONENTS:
        specification = preregistration["completion_inputs"][component]
        validators.append(
            (
                component,
                lambda component=component, specification=specification: _formal_completion_material(
                    component, specification, repo_root=root
                ),
            )
        )
    for component in DEVELOPMENT_ONLY_COMPONENTS:
        specification = preregistration["completion_inputs"][component]
        validators.append(
            (
                component,
                lambda component=component, specification=specification: _development_only_material(
                    component, specification, repo_root=root
                ),
            )
        )
    # The addendum preregistration depends on the pinned base identities, so
    # assess it last and report a concrete error rather than trusting its text.
    try:
        _, _, base_specification_identity, base_scoreboard_identity = _validate_historical_base(
            preregistration, repo_root=root
        )
        validators.append(
            (
                "scoreboard addendum preregistration",
                lambda: _validate_addendum_preregistration(
                    preregistration,
                    base_specification_identity=base_specification_identity,
                    base_scoreboard_identity=base_scoreboard_identity,
                    repo_root=root,
                ),
            )
        )
    except (FileNotFoundError, ValueError) as error:
        blockers.append(f"historical base: {error}")
    for label, validator in validators:
        try:
            validator()
        except (FileNotFoundError, ValueError) as error:
            blockers.append(f"{label}: {error}")
    return {
        "schema_version": 1,
        "aggregate_id": AGGREGATE_ID,
        "status": (
            "ready_for_explicit_exclusive_finalization"
            if not blockers
            else "blocked_pending_four_final_outcomes"
        ),
        "ready": not blockers,
        "blockers": sorted(set(blockers)),
        "outputs_created": False,
    }


def build_completion_aggregate_and_scoreboard(
    *,
    aggregate_config: Path | str = AGGREGATE_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build final JSON objects in memory after independently checking inputs."""

    root = (repo_root or repository_root()).resolve()
    audit = audit_completion_aggregate_readiness(
        aggregate_config=aggregate_config, repo_root=root
    )
    if not audit["ready"]:
        raise CompletionAggregateBlocked(list(audit["blockers"]))
    preregistration = load_completion_aggregate_preregistration(
        aggregate_config, repo_root=root
    )
    base_specification, base_scoreboard, base_spec_identity, base_scoreboard_identity = (
        _validate_historical_base(preregistration, repo_root=root)
    )
    baseline_identity, baseline_values = _validate_baseline_cem(preregistration, repo_root=root)
    addendum_prereg_identity = _validate_addendum_preregistration(
        preregistration,
        base_specification_identity=base_spec_identity,
        base_scoreboard_identity=base_scoreboard_identity,
        repo_root=root,
    )
    inputs = preregistration["completion_inputs"]
    formal_materials = {
        component: _formal_completion_material(component, inputs[component], repo_root=root)
        for component in FORMAL_COMPONENTS
    }
    development_materials = {
        component: _development_only_material(component, inputs[component], repo_root=root)
        for component in DEVELOPMENT_ONLY_COMPONENTS
    }
    rows = [
        _row_from_formal_material(
            component,
            formal_materials[component],
            baseline_value=baseline_values[component],
        )
        for component in FORMAL_COMPONENTS
    ]
    spec = copy.deepcopy(base_specification)
    spec["components"] = [*spec["components"], *rows]
    # The specification retains its historical eleven-row prefix.  The public
    # scoreboard itself is the canonical full scorer output, which sorts by
    # component/method; therefore historical rows are checked by identity,
    # not by their incidental serialized position.
    fully_scored = make_public_scoreboard_from_spec(spec)
    scoreboard = fully_scored
    historical_by_identity = {
        (row["component_id"], row["method_name"]): row
        for row in base_scoreboard["component_results"]
    }
    observed_by_identity = {
        (row["component_id"], row["method_name"]): row
        for row in scoreboard["component_results"]
    }
    if (
        len(observed_by_identity) != len(scoreboard["component_results"])
        or any(observed_by_identity.get(identity) != row for identity, row in historical_by_identity.items())
        or scoreboard != make_public_scoreboard_from_spec(spec)
    ):
        raise ValueError("addendum scoreboard does not independently reproduce its specification")
    aggregate_path = preregistration["outputs"]["aggregate_freeze"]
    config_path = Path(preregistration["_config_path"]).resolve()
    try:
        config_logical_path = str(config_path.relative_to(root))
    except ValueError as error:
        raise ValueError("aggregate preregistration must live under repo_root") from error
    aggregate = {
        "schema_version": 1,
        "freeze_id": AGGREGATE_ID,
        "status": "frozen_after_all_four_pldm_reference_completion_outcomes",
        "all_four_completion_outcomes_finalized": True,
        "aggregate_preregistration": _identity(config_logical_path, repo_root=root),
        "finalizer_implementation": _identity(
            "contextworld/benchmarks/pldm_reference_completion_aggregate.py",
            repo_root=root,
        ),
        "historical_base": {
            "specification": base_spec_identity,
            "scoreboard": base_scoreboard_identity,
            "formal_reference_rows_before": 11,
        },
        "original_baseline_cem": baseline_identity,
        "completion_results": {
            **formal_materials,
            **development_materials,
        },
        "public_scoreboard_addendum": {
            "preregistration": addendum_prereg_identity,
            "formal_reference_rows_before": 11,
            "formal_reference_rows_added": len(rows),
            "formal_reference_rows_after": len(spec["components"]),
            "components_added": list(FORMAL_COMPONENTS),
            "development_only_components_not_added": list(DEVELOPMENT_ONLY_COMPONENTS),
            "specification_path": preregistration["scoreboard_addendum"]["specification"],
            "scoreboard_path": preregistration["scoreboard_addendum"]["scoreboard"],
        },
        "claims": {
            "historical_eleven_row_scoreboard_rewritten": False,
            "development_only_results_added_to_public_scoreboard": False,
            "speed_and_action_strength_rows_derived_from_formal_public_results": True,
            "public_rows_added_regardless_of_icl_pass_fail": True,
            "training_or_checkpoint_selection_performed": False,
            "public_test_rerun_performed_by_finalizer": False,
            "suite_default_or_activation_switched_by_finalizer": False,
        },
    }
    if aggregate["freeze_id"] != AGGREGATE_ID or aggregate_path != preregistration["outputs"]["aggregate_freeze"]:
        raise AssertionError("aggregate output contract drifted")
    return {
        "aggregate": aggregate,
        "scoreboard_specification": spec,
        "scoreboard": scoreboard,
        "rows_added": rows,
        "base_specification_identity": base_spec_identity,
        "base_scoreboard_identity": base_scoreboard_identity,
        "preregistration": preregistration,
    }


def _resolution_decision(
    *,
    preregistration: Mapping[str, Any],
    base_specification_identity: Mapping[str, Any],
    base_scoreboard_identity: Mapping[str, Any],
    aggregate_identity: Mapping[str, Any],
    addendum_specification_identity: Mapping[str, Any],
    addendum_scoreboard_identity: Mapping[str, Any],
    rows_added: int,
) -> dict[str, Any]:
    addendum = preregistration["scoreboard_addendum"]
    formal_rows = 11 + int(rows_added)
    return {
        "schema_version": 1,
        "decision_id": addendum["decision_id"],
        "status": "additive_scoreboard_extension_authorized",
        "passed": True,
        "scoreboard_extension_authorized": True,
        "formal_reference_rows": formal_rows,
        "formal_reference_rows_added": int(rows_added),
        "historical_base_specification": dict(base_specification_identity),
        "historical_base_scoreboard": dict(base_scoreboard_identity),
        "addendum_specification": dict(addendum_specification_identity),
        "addendum_scoreboard": dict(addendum_scoreboard_identity),
        "final_pldm_completion_aggregate_results_freeze": dict(aggregate_identity),
        "claims": {
            "historical_base_remained_byte_identical": True,
            "row_count_derived_from_final_formal_evidence": True,
            "development_only_results_added_to_public_scoreboard": False,
        },
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_completion_aggregate_and_scoreboard(
    *,
    aggregate_config: Path | str = AGGREGATE_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Exclusively create the aggregate and new addendum scoreboard files.

    This is intentionally separate from check-only and in-memory construction.
    It never overwrites an output and never writes to the historical base
    namespace.
    """

    root = (repo_root or repository_root()).resolve()
    bundle = build_completion_aggregate_and_scoreboard(
        aggregate_config=aggregate_config, repo_root=root
    )
    preregistration = bundle["preregistration"]
    addendum = preregistration["scoreboard_addendum"]
    aggregate_path = _new_output_path(preregistration["outputs"]["aggregate_freeze"], repo_root=root)
    specification_path = _new_output_path(addendum["specification"], repo_root=root)
    scoreboard_path = _new_output_path(addendum["scoreboard"], repo_root=root)
    decision_path = _new_output_path(addendum["decision"], repo_root=root)
    namespace = _new_output_path(addendum["output_namespace"], repo_root=root)
    if specification_path.parent != namespace or scoreboard_path.parent != namespace:
        raise ValueError("scoreboard output escaped the preregistered addendum namespace")
    protected_base = {
        resolve_contextworld_path(bundle["base_specification_identity"]["path"], repo_root=root),
        resolve_contextworld_path(bundle["base_scoreboard_identity"]["path"], repo_root=root),
    }
    targets = [aggregate_path, specification_path, scoreboard_path, decision_path]
    if any(path in protected_base for path in targets):
        raise ValueError("finalizer may not target the historical scoreboard")
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite finalization output: {existing}")
    aggregate_identity = _serialized_identity(
        preregistration["outputs"]["aggregate_freeze"], bundle["aggregate"]
    )
    specification_identity = _serialized_identity(addendum["specification"], bundle["scoreboard_specification"])
    scoreboard_identity = _serialized_identity(addendum["scoreboard"], bundle["scoreboard"])
    decision = _resolution_decision(
        preregistration=preregistration,
        base_specification_identity=bundle["base_specification_identity"],
        base_scoreboard_identity=bundle["base_scoreboard_identity"],
        aggregate_identity=aggregate_identity,
        addendum_specification_identity=specification_identity,
        addendum_scoreboard_identity=scoreboard_identity,
        rows_added=len(bundle["rows_added"]),
    )
    _write_json_exclusive(aggregate_path, bundle["aggregate"])
    _write_json_exclusive(specification_path, bundle["scoreboard_specification"])
    _write_json_exclusive(scoreboard_path, bundle["scoreboard"])
    _write_json_exclusive(decision_path, decision)
    return {
        "schema_version": 1,
        "aggregate_id": AGGREGATE_ID,
        "status": "created_exclusive_finalization_outputs",
        "aggregate_freeze": aggregate_identity,
        "addendum_specification": specification_identity,
        "addendum_scoreboard": scoreboard_identity,
        "scoreboard_resolution_decision": _identity(addendum["decision"], repo_root=root),
        "formal_reference_rows": 11 + len(bundle["rows_added"]),
        "formal_reference_rows_added": len(bundle["rows_added"]),
    }


def _validated_public_addendum_summary(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """State the fixed public-row boundary for the final v1 addendum.

    The exact-file comparison below is the primary proof.  This small
    summary makes the release boundary explicit to the successor reseal: the
    failed ActionStrength Public result remains a formal row, whereas the two
    Development-only PushT outcomes remain evidence only.  Speed is a
    behavioral result because the preregistered paired training controls do
    not exist.
    """

    rows = bundle.get("rows_added")
    aggregate = bundle.get("aggregate")
    if not isinstance(rows, list) or not isinstance(aggregate, Mapping):
        raise ValueError("rebuilt aggregate has no public-addendum rows")
    if len(rows) != len(FORMAL_COMPONENTS):
        raise ValueError("rebuilt aggregate has the wrong formal-row count")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("rebuilt aggregate contains an invalid formal row")
    components_added = [row.get("component_id") for row in rows]
    if components_added != list(FORMAL_COMPONENTS):
        raise ValueError("rebuilt aggregate formal rows are not canonical")
    if any(component in DEVELOPMENT_ONLY_COMPONENTS for component in components_added):
        raise ValueError("Development-only evidence entered the public addendum")

    completion_results = aggregate.get("completion_results")
    if not isinstance(completion_results, Mapping):
        raise ValueError("rebuilt aggregate has no completion results")
    speed = completion_results.get("speed")
    action_strength = completion_results.get("action_strength")
    if not isinstance(speed, Mapping) or not isinstance(action_strength, Mapping):
        raise ValueError("rebuilt aggregate formal completion records are missing")
    speed_row = rows[0]
    speed_reader = speed.get("reader_result")
    speed_attribution = speed.get("training_attribution")
    if (
        speed_row.get("evidence_scope") != "behavioral"
        or not isinstance(speed_reader, Mapping)
        or speed_reader.get("evidence_scope") != "behavioral"
        or not isinstance(speed_attribution, Mapping)
        or speed_attribution.get("claim") is not False
        or speed_attribution.get("paired_training_controls_available")
        is not False
    ):
        raise ValueError("Speed formal row exceeds its behavioral claim boundary")

    action_row = rows[1]
    action_retention = action_row.get("original_task_retention")
    if (
        action_row.get("ability_passed") is not False
        or not isinstance(action_retention, Mapping)
        or action_retention.get("result") != "NOT_EVALUATED"
    ):
        raise ValueError(
            "failed ActionStrength Public result is not retained as the "
            "required formal non-CEM row"
        )

    addendum = aggregate.get("public_scoreboard_addendum")
    if (
        not isinstance(addendum, Mapping)
        or addendum.get("components_added") != list(FORMAL_COMPONENTS)
        or addendum.get("development_only_components_not_added")
        != list(DEVELOPMENT_ONLY_COMPONENTS)
        or addendum.get("formal_reference_rows_added") != len(FORMAL_COMPONENTS)
        or addendum.get("formal_reference_rows_after") != 11 + len(FORMAL_COMPONENTS)
    ):
        raise ValueError("rebuilt aggregate public-addendum boundary drifted")
    return {
        "formal_reference_rows": 11 + len(FORMAL_COMPONENTS),
        "formal_reference_rows_added": len(FORMAL_COMPONENTS),
        "components_added": list(FORMAL_COMPONENTS),
        "development_only_components_not_added": list(
            DEVELOPMENT_ONLY_COMPONENTS
        ),
        "speed_evidence_scope": "behavioral",
        "speed_training_attribution_claim": False,
        "action_strength_formal_row_included": True,
        "action_strength_ability_passed": False,
    }


def validate_written_completion_aggregate_and_scoreboard(
    *,
    aggregate_config: Path | str = AGGREGATE_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Rebuild expected outputs and reject drift after exclusive creation."""

    root = (repo_root or repository_root()).resolve()
    bundle = build_completion_aggregate_and_scoreboard(
        aggregate_config=aggregate_config, repo_root=root
    )
    preregistration = bundle["preregistration"]
    addendum = preregistration["scoreboard_addendum"]
    aggregate_path = preregistration["outputs"]["aggregate_freeze"]
    expected_aggregate = bundle["aggregate"]
    actual_aggregate, aggregate_identity = _read_local_output_json(
        aggregate_path, repo_root=root
    )
    if actual_aggregate != expected_aggregate:
        raise ValueError("aggregate freeze does not match independently rebuilt evidence")
    actual_specification, specification_identity = _read_local_output_json(
        addendum["specification"], repo_root=root
    )
    actual_scoreboard, scoreboard_identity = _read_local_output_json(
        addendum["scoreboard"], repo_root=root
    )
    if (
        actual_specification != bundle["scoreboard_specification"]
        or actual_scoreboard != bundle["scoreboard"]
    ):
        raise ValueError("addendum scoreboard output does not match independently rebuilt rows")
    decision, decision_identity = _read_local_output_json(
        addendum["decision"], repo_root=root
    )
    expected_decision = _resolution_decision(
        preregistration=preregistration,
        base_specification_identity=bundle["base_specification_identity"],
        base_scoreboard_identity=bundle["base_scoreboard_identity"],
        aggregate_identity=aggregate_identity,
        addendum_specification_identity=specification_identity,
        addendum_scoreboard_identity=scoreboard_identity,
        rows_added=len(bundle["rows_added"]),
    )
    if decision != expected_decision:
        raise ValueError("scoreboard addendum decision does not bind current outputs")
    summary = _validated_public_addendum_summary(bundle)
    return {
        "passed": True,
        "aggregate_id": AGGREGATE_ID,
        **summary,
        "local_outputs": {
            "aggregate_freeze": aggregate_identity,
            "addendum_specification": specification_identity,
            "addendum_scoreboard": scoreboard_identity,
            "scoreboard_resolution_decision": decision_identity,
        },
    }


__all__ = [
    "AGGREGATE_CONFIG",
    "AGGREGATE_ID",
    "CompletionAggregateBlocked",
    "audit_completion_aggregate_readiness",
    "build_completion_aggregate_and_scoreboard",
    "load_completion_aggregate_preregistration",
    "validate_written_completion_aggregate_and_scoreboard",
    "write_completion_aggregate_and_scoreboard",
]
