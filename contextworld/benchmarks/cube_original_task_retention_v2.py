"""OSMesa recovery contract for frozen original-Cube CEM retention."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from contextworld.benchmarks import cube_original_task_retention as v1
from contextworld.paths import repository_root


CUBE_CEM_RETENTION_V2_ID = (
    "contextworld_cube_gripper_carry_h3_v4r1_cem_retention_v2"
)
CUBE_CEM_RETENTION_V2_PROTOCOL = (
    "cube_gripper_carry_rule_history3_v4r1_original_task_cem_retention_v2"
)
DEFAULT_CUBE_CEM_RETENTION_V2_PREREG = (
    repository_root()
    / "configs/benchmark/cube_gripper_carry_h3_v4r1_cem_retention_prereg_v2.yaml"
)

EVAL_SEEDS = v1.EVAL_SEEDS
TRAINING_SEEDS = v1.TRAINING_SEEDS
QUERIES_PER_EVAL_SEED = v1.QUERIES_PER_EVAL_SEED
TOTAL_QUERIES = v1.TOTAL_QUERIES
NONINFERIORITY_MARGIN_SUCCESSES = v1.NONINFERIORITY_MARGIN_SUCCESSES
closed_public_contract = v1.closed_public_contract
file_sha256 = v1.file_sha256
resolve_declared_path = v1.resolve_declared_path
expected_cube_cem_jobs = v1.expected_cube_cem_jobs
expected_cube_cem_result_directory = v1.expected_cube_cem_result_directory
collect_cube_cem_static_identities = v1.collect_cube_cem_static_identities
validate_cube_cem_query_catalog = v1.validate_cube_cem_query_catalog
paired_cube_cem_noninferiority = v1.paired_cube_cem_noninferiority


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    return value


def _validate_declared_artifact(
    entry: Mapping[str, Any], *, repo_root: Path, field: str
) -> dict[str, Any]:
    path = resolve_declared_path(str(entry.get("path", "")), repo_root=repo_root)
    return v1._declared_identity(entry, path=path, field=field)


def _validate_v1_failure_receipt(
    payload: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    predecessor = _mapping(payload.get("predecessor"), field="predecessor")
    if set(predecessor) != {
        "preregistration",
        "freeze_receipt",
        "query_catalog",
        "matrix_request",
        "matrix_report",
        "failure_receipt",
    }:
        raise ValueError("Cube CEM v2 predecessor evidence set drifted")
    observed = {
        name: _validate_declared_artifact(
            _mapping(entry, field=f"predecessor.{name}"),
            repo_root=repo_root,
            field=f"predecessor.{name}",
        )
        for name, entry in predecessor.items()
    }
    failure_path = Path(observed["failure_receipt"]["path"])
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if (
        failure.get("schema_version") != 1
        or failure.get("receipt_id")
        != "cube_gripper_carry_h3_v4r1_cem_retention_v1_egl_failure"
        or failure.get("status")
        != "archived_zero_episode_infrastructure_failure"
        or failure.get("classification")
        != "render_backend_initialization_failure_not_model_result"
        or failure.get("execution", {}).get("cem_episodes_completed") != 0
        or failure.get("execution", {}).get("world_evaluate_calls") != 0
        or failure.get("execution", {}).get("aggregate_reports_created") != 0
        or failure.get("scientific_interpretation", {}).get(
            "retention_pass_or_fail_observed"
        )
        is not False
        or failure.get("public_test") != closed_public_contract()
    ):
        raise RuntimeError("Cube CEM v1 failure receipt is not recovery-eligible")
    chain = failure.get("authorization_chain", {})
    for name in (
        "preregistration",
        "freeze_receipt",
        "query_catalog",
        "matrix_request",
        "matrix_report",
    ):
        if chain.get(name, {}).get("sha256") != observed[name]["sha256"]:
            raise RuntimeError(f"Cube CEM v1 {name} evidence drifted")
    return observed


def _build_effective_prereg(
    overlay: Mapping[str, Any], *, overlay_path: Path, repo_root: Path
) -> dict[str, Any]:
    if (
        overlay.get("schema_version") != 1
        or overlay.get("preregistration_id") != CUBE_CEM_RETENTION_V2_ID
        or overlay.get("protocol_id") != CUBE_CEM_RETENTION_V2_PROTOCOL
        or overlay.get("status")
        != "preregistered_before_original_task_retention_recovery"
        or overlay.get("phase")
        != "post_development_original_task_retention_recovery_only"
    ):
        raise ValueError("Cube CEM v2 recovery overlay identity drifted")
    predecessor = _mapping(overlay.get("predecessor"), field="predecessor")
    base_entry = _mapping(
        predecessor.get("preregistration"), field="predecessor.preregistration"
    )
    base_identity = _validate_declared_artifact(
        base_entry, repo_root=repo_root, field="predecessor.preregistration"
    )
    base_path = Path(base_identity["path"])
    base = v1.load_cube_cem_retention_prereg(
        base_path, require_freeze=False, repo_root=repo_root
    )
    predecessor_observed = _validate_v1_failure_receipt(
        overlay, repo_root=repo_root
    )

    recovery = _mapping(overlay.get("recovery"), field="recovery")
    allowed_change = _mapping(
        recovery.get("only_authorized_change"),
        field="recovery.only_authorized_change",
    )
    if (
        recovery.get("classification")
        != "render_backend_initialization_failure_not_scientific_failure"
        or recovery.get("prior_cem_episodes_completed") != 0
        or recovery.get("prior_world_evaluate_calls") != 0
        or allowed_change
        != {
            "field": "evaluation.mujoco_gl",
            "before": "egl",
            "after": "osmesa",
            "scientific_query_or_cem_parameter_changed": False,
        }
        or recovery.get("data_changed") is not False
        or recovery.get("query_catalog_changed") is not False
        or recovery.get("checkpoint_changed") is not False
        or recovery.get("evaluation_seed_changed") is not False
        or recovery.get("cem_parameter_changed") is not False
        or recovery.get("noninferiority_gate_changed") is not False
        or recovery.get("public_test_access_changed") is not False
    ):
        raise ValueError("Cube CEM v2 recovery scope drifted")
    evaluation_override = _mapping(
        overlay.get("evaluation_override"), field="evaluation_override"
    )
    if evaluation_override != {
        "mujoco_gl": "osmesa",
        "environment_preflight_required": True,
        "environment_preflight_num_envs": 1,
        "environment_preflight_world_evaluate_called": False,
        "environment_preflight_cem_episodes_consumed": 0,
    }:
        raise ValueError("Cube CEM v2 environment preflight drifted")
    planned = _mapping(overlay.get("planned_artifacts"), field="planned_artifacts")
    if (
        set(planned)
        != {
            "freeze_receipt",
            "query_catalog",
            "retention_root",
            "retention_decision",
        }
        or len(set(str(value) for value in planned.values())) != 4
        or any("v2" not in str(value) for value in planned.values())
        or dict(_mapping(overlay.get("public_test"), field="public_test"))
        != closed_public_contract()
    ):
        raise ValueError("Cube CEM v2 output or Public contract drifted")

    effective = copy.deepcopy(
        {key: value for key, value in base.items() if not str(key).startswith("_")}
    )
    for field in (
        "preregistration_id",
        "protocol_id",
        "status",
        "date",
        "phase",
        "claim_limit",
    ):
        effective[field] = overlay[field]
    effective["predecessor"] = copy.deepcopy(dict(predecessor))
    effective["predecessor_observed"] = predecessor_observed
    effective["recovery"] = copy.deepcopy(dict(recovery))
    effective["evaluation"].update(dict(evaluation_override))
    effective["identity"].update(
        copy.deepcopy(dict(_mapping(overlay.get("identity"), field="identity")))
    )
    effective["planned_artifacts"] = copy.deepcopy(dict(planned))
    effective["public_test"] = closed_public_contract()

    normalized = copy.deepcopy(effective)
    normalized["preregistration_id"] = v1.CUBE_CEM_RETENTION_ID
    normalized["protocol_id"] = v1.CUBE_CEM_RETENTION_PROTOCOL
    normalized["status"] = "preregistered_before_original_task_retention"
    normalized["phase"] = "post_development_original_task_retention_only"
    normalized["evaluation"]["mujoco_gl"] = "egl"
    v1._validate_preregistration(normalized)
    if (
        effective["evaluation"]["mujoco_gl"] != "osmesa"
        or effective["scope"]["passing_families"] != ["lewm"]
        or effective["authorization"] != base["authorization"]
        or effective["data"] != base["data"]
        or effective["runtime"] != base["runtime"]
        or effective["public_test"] != base["public_test"]
        or predecessor_observed["query_catalog"]["sha256"]
        != recovery.get("frozen_query_catalog_sha256")
    ):
        raise ValueError("Cube CEM v2 scientific contract drifted")
    effective["_config_path"] = str(overlay_path)
    effective["_base_preregistration_path"] = str(base_path)
    return effective


def _validate_freeze_receipt(
    prereg: dict[str, Any], *, prereg_path: Path, repo_root: Path
) -> dict[str, Any]:
    freeze_path = resolve_declared_path(
        prereg["planned_artifacts"]["freeze_receipt"], repo_root=repo_root
    )
    if not freeze_path.is_file() or freeze_path.is_symlink():
        raise FileNotFoundError(f"Missing Cube CEM v2 freeze receipt: {freeze_path}")
    receipt = json.loads(freeze_path.read_text(encoding="utf-8"))
    config_identity = v1._stable_file_identity(prereg_path)
    static = collect_cube_cem_static_identities(prereg, repo_root=repo_root)
    query_path = resolve_declared_path(
        prereg["planned_artifacts"]["query_catalog"], repo_root=repo_root
    )
    query = validate_cube_cem_query_catalog(
        prereg, path=query_path, expected_identity=receipt.get("query_catalog")
    )
    authorized_jobs = [
        {
            "kind": row["kind"],
            "model_family": row["model_family"],
            "model_name": row["model_name"],
            **(
                {"training_seed": int(row["training_seed"])}
                if row["kind"] == "candidate"
                else {}
            ),
        }
        for row in expected_cube_cem_jobs(prereg)
    ]
    model_preflight = _mapping(
        receipt.get("model_preflight"), field="freeze.model_preflight"
    )
    preflight_rows = _sequence(
        model_preflight.get("models"), field="freeze.model_preflight.models"
    )
    expected_rows = [
        {
            "model": job["model_name"],
            "checkpoint_sha256": job["checkpoint_identity"]["sha256"],
            "config_sha256": job["config_identity"]["sha256"],
        }
        for job in expected_cube_cem_jobs(prereg)
    ]
    observed_rows = [
        {
            "model": row.get("model"),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "config_sha256": row.get("config_sha256"),
        }
        for row in preflight_rows
        if isinstance(row, Mapping)
    ]
    stable = prereg["runtime"]["stable_worldmodel"]
    expected_runtime = {
        "root": str(Path(stable["repo"]).resolve()),
        "commit": stable["expected_ref"],
        "clean": True,
    }
    expected_environment = {
        "mujoco_gl": "osmesa",
        "world_constructed": True,
        "num_envs": 1,
        "world_evaluate_called": False,
        "cem_episodes_consumed": 0,
    }
    authorization = _mapping(
        receipt.get("authorization"), field="freeze.authorization"
    )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "frozen_authorized"
        or receipt.get("preregistration_id") != CUBE_CEM_RETENTION_V2_ID
        or receipt.get("preregistration") != config_identity
        or receipt.get("static_identities") != static
        or model_preflight.get("runtime") != expected_runtime
        or model_preflight.get("environment_preflight") != expected_environment
        or observed_rows != expected_rows
        or any(
            not isinstance(row, Mapping)
            or row.get("strict_load") is not True
            or int(row.get("parameter_count", -1))
            != int(prereg["evaluation"]["lewm_parameter_count"])
            for row in preflight_rows
        )
        or authorization.get("jobs") != authorized_jobs
        or int(authorization.get("jobs_count", -1)) != 4
        or int(authorization.get("episodes_per_job", -1)) != TOTAL_QUERIES
        or int(authorization.get("total_cem_episodes", -1)) != TOTAL_QUERIES * 4
        or authorization.get("baseline_and_candidates_share_frozen_queries")
        is not True
        or int(authorization.get("noninferiority_margin_successes", -1))
        != NONINFERIORITY_MARGIN_SUCCESSES
        or receipt.get("query_catalog", {}).get("sha256")
        != prereg["recovery"]["frozen_query_catalog_sha256"]
        or receipt.get("public_test") != closed_public_contract()
    ):
        raise RuntimeError("Cube CEM v2 freeze receipt drifted")
    return {
        **prereg,
        "_freeze_receipt": receipt,
        "_freeze_receipt_path": str(freeze_path),
        "_query_catalog": query["payload"],
        "_query_catalog_path": str(query_path),
    }


def load_cube_cem_retention_v2_prereg(
    path: Path | str = DEFAULT_CUBE_CEM_RETENTION_V2_PREREG,
    *,
    require_freeze: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    overlay = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(overlay, dict):
        raise ValueError("Cube CEM v2 recovery preregistration must be a mapping")
    root = (repo_root or repository_root()).resolve()
    effective = _build_effective_prereg(
        overlay, overlay_path=config_path, repo_root=root
    )
    if not require_freeze:
        return effective
    return _validate_freeze_receipt(
        effective, prereg_path=config_path, repo_root=root
    )


def validate_cube_cem_v2_job_result(
    prereg: Mapping[str, Any],
    *,
    model_name: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    validated = v1.validate_cube_cem_job_result(
        prereg, model_name=model_name, repo_root=repo_root
    )
    root = resolve_declared_path(
        prereg["planned_artifacts"]["retention_root"], repo_root=repo_root
    )
    log_path = root / "logs" / f"{model_name}.log"
    log_identity = v1._stable_file_identity(log_path)
    text = log_path.read_text(encoding="utf-8")
    if (
        "[contextworld] MUJOCO_GL=osmesa" not in text
        or "MUJOCO_GL=egl" in text
        or "Cannot initialize a headless EGL display" in text
    ):
        raise RuntimeError(f"Cube CEM v2 render backend drifted for {model_name}")
    return {
        **validated,
        "render_backend": "osmesa",
        "log_identity": log_identity,
    }


def build_cube_cem_v2_retention_result(
    prereg: Mapping[str, Any],
    *,
    validated: Sequence[Mapping[str, Any]],
    matrix_report_path: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    payload = v1.build_cube_cem_retention_result(
        prereg,
        validated=validated,
        matrix_report_path=matrix_report_path,
        repo_root=repo_root,
    )
    payload["preregistration_id"] = CUBE_CEM_RETENTION_V2_ID
    payload["render_backend"] = {
        "mujoco_gl": "osmesa",
        "same_for_baseline_and_all_candidates": True,
        "preflighted_before_freeze": True,
        "job_logs": {
            str(row["model_name"]): row["log_identity"] for row in validated
        },
    }
    payload["recovery"] = {
        "predecessor_preregistration_id": v1.CUBE_CEM_RETENTION_ID,
        "predecessor_failure": "zero_episode_egl_initialization_failure",
        "scientific_query_or_cem_parameter_changed": False,
        "query_catalog_sha256": prereg["recovery"][
            "frozen_query_catalog_sha256"
        ],
    }
    return payload


def validate_cube_cem_v2_retention_result(
    prereg: Mapping[str, Any],
    *,
    result_path: Path,
    matrix_report_path: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    expected_root = resolve_declared_path(
        prereg["planned_artifacts"]["retention_root"], repo_root=repo_root
    )
    if (
        result_path.resolve() != expected_root / "retention_result.json"
        or matrix_report_path.resolve() != expected_root / "matrix_report.json"
    ):
        raise RuntimeError("Cube CEM v2 retention result path drifted")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validated = [
        validate_cube_cem_v2_job_result(
            prereg, model_name=job["model_name"], repo_root=repo_root
        )
        for job in expected_cube_cem_jobs(prereg)
    ]
    expected = build_cube_cem_v2_retention_result(
        prereg,
        validated=validated,
        matrix_report_path=matrix_report_path,
        repo_root=repo_root,
    )
    if payload != expected:
        raise RuntimeError("Cube CEM v2 retention aggregate drifted")
    return {"payload": payload, "validated_jobs": validated}


__all__ = [
    "CUBE_CEM_RETENTION_V2_ID",
    "CUBE_CEM_RETENTION_V2_PROTOCOL",
    "DEFAULT_CUBE_CEM_RETENTION_V2_PREREG",
    "EVAL_SEEDS",
    "NONINFERIORITY_MARGIN_SUCCESSES",
    "QUERIES_PER_EVAL_SEED",
    "TOTAL_QUERIES",
    "TRAINING_SEEDS",
    "build_cube_cem_v2_retention_result",
    "closed_public_contract",
    "collect_cube_cem_static_identities",
    "expected_cube_cem_jobs",
    "expected_cube_cem_result_directory",
    "file_sha256",
    "load_cube_cem_retention_v2_prereg",
    "paired_cube_cem_noninferiority",
    "resolve_declared_path",
    "validate_cube_cem_query_catalog",
    "validate_cube_cem_v2_job_result",
    "validate_cube_cem_v2_retention_result",
]
