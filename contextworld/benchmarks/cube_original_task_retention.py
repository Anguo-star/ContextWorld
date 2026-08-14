"""Frozen post-Development original-Cube CEM retention contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import yaml

from contextworld.paths import repository_root, resolve_contextworld_path


CUBE_CEM_RETENTION_ID = "contextworld_cube_gripper_carry_h3_v4r1_cem_retention_v1"
CUBE_CEM_RETENTION_PROTOCOL = (
    "cube_gripper_carry_rule_history3_v4r1_original_task_cem_retention_v1"
)
DEFAULT_CUBE_CEM_RETENTION_PREREG = (
    repository_root()
    / "configs/benchmark/cube_gripper_carry_h3_v4r1_cem_retention_prereg_v1.yaml"
)
EVAL_SEEDS = (42, 43, 44)
TRAINING_SEEDS = (17321, 17322, 17323)
QUERIES_PER_EVAL_SEED = 100
TOTAL_QUERIES = len(EVAL_SEEDS) * QUERIES_PER_EVAL_SEED
NONINFERIORITY_MARGIN_SUCCESSES = 15


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    return value


def closed_public_contract() -> dict[str, Any]:
    return {
        "access_status": "closed_not_read_not_scored",
        "generated": False,
        "opened": False,
        "read": False,
        "hashed": False,
        "scored": False,
        "validation_lance_access_allowed": False,
    }


def _stable_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(f"Expected a regular non-symlink file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _declared_identity(
    entry: Mapping[str, Any], *, path: Path, field: str
) -> dict[str, Any]:
    observed = _stable_file_identity(path)
    if (
        entry.get("sha256") != observed["sha256"]
        or int(entry.get("size_bytes", -1)) != observed["size_bytes"]
    ):
        raise RuntimeError(f"{field} identity drifted: {path}")
    return observed


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            f"--git-dir={root / '.git'}",
            f"--work-tree={root}",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _validate_preregistration(payload: Mapping[str, Any]) -> None:
    scope = _mapping(payload.get("scope"), field="scope")
    evaluation = _mapping(payload.get("evaluation"), field="evaluation")
    authorization = _mapping(payload.get("authorization"), field="authorization")
    runtime = _mapping(payload.get("runtime"), field="runtime")
    data = _mapping(payload.get("data"), field="data")
    planned = _mapping(payload.get("planned_artifacts"), field="planned_artifacts")
    public = _mapping(payload.get("public_test"), field="public_test")
    candidates = _sequence(authorization.get("candidates"), field="candidates")
    baseline = _mapping(authorization.get("baseline"), field="baseline")

    candidate_seeds = tuple(
        int(_mapping(row, field="candidate").get("training_seed", -1))
        for row in candidates
    )
    candidate_families = tuple(
        _mapping(row, field="candidate").get("model_family") for row in candidates
    )
    candidate_names = tuple(
        _mapping(row, field="candidate").get("model_name") for row in candidates
    )
    output_values = tuple(str(value) for value in planned.values())
    if (
        payload.get("schema_version") != 1
        or payload.get("preregistration_id") != CUBE_CEM_RETENTION_ID
        or payload.get("protocol_id") != CUBE_CEM_RETENTION_PROTOCOL
        or payload.get("status") != "preregistered_before_original_task_retention"
        or payload.get("phase") != "post_development_original_task_retention_only"
        or scope.get("environment") != "Cube"
        or scope.get("history_tokens") != 3
        or scope.get("raw_action_dim") != 5
        or scope.get("flattened_action_input_dim") != 25
        or scope.get("passing_families") != ["lewm"]
        or scope.get("public_test_included") is not False
        or baseline.get("model_family") != "lewm"
        or baseline.get("model_name") != "baseline_lewm"
        or candidate_seeds != TRAINING_SEEDS
        or candidate_families != ("lewm", "lewm", "lewm")
        or candidate_names != tuple(f"lewm_seed{seed}" for seed in TRAINING_SEEDS)
        or tuple(int(value) for value in evaluation.get("eval_seeds", ()))
        != EVAL_SEEDS
        or int(evaluation.get("queries_per_eval_seed", -1))
        != QUERIES_PER_EVAL_SEED
        or int(evaluation.get("episodes_per_checkpoint", -1)) != TOTAL_QUERIES
        or int(evaluation.get("noninferiority_margin_successes", -1))
        != NONINFERIORITY_MARGIN_SUCCESSES
        or int(evaluation.get("lewm_parameter_count", -1)) != 18_034_628
        or evaluation.get("baseline_and_candidates_share_frozen_queries") is not True
        or evaluation.get("videos_written") is not False
        or evaluation.get("public_test_opened") is not False
        or runtime.get("stable_worldmodel", {}).get("expected_ref")
        != "875e607fc08aa72eacb94d5d178127804134cc06"
        or data.get("original_h5", {}).get("expected_identity", {}).get("action_dim")
        != 5
        or dict(public) != closed_public_contract()
        or len(output_values) != len(set(output_values))
        or set(planned)
        != {
            "freeze_receipt",
            "query_catalog",
            "retention_root",
            "retention_decision",
        }
    ):
        raise ValueError("Cube CEM retention preregistration contract drifted")


def expected_cube_cem_jobs(prereg: Mapping[str, Any]) -> list[dict[str, Any]]:
    authorization = prereg["authorization"]
    return [dict(authorization["baseline"])] + [
        dict(value) for value in authorization["candidates"]
    ]


def resolve_declared_path(
    value: str | Path, *, repo_root: Path | None = None
) -> Path:
    return resolve_contextworld_path(
        value, repo_root=(repo_root or repository_root()).resolve()
    )


def _validate_prior_development(
    prereg: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    declared = _mapping(
        prereg.get("prior_development"), field="prior_development"
    )
    observed: dict[str, Any] = {}
    for name, raw_entry in declared.items():
        entry = _mapping(raw_entry, field=f"prior_development.{name}")
        path = resolve_declared_path(str(entry.get("path", "")), repo_root=repo_root)
        observed[name] = _declared_identity(
            entry, path=path, field=f"prior_development.{name}"
        )
    decision_path = Path(observed["development_decision"]["path"])
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("schema_version") != 1
        or decision.get("decision_id")
        != "cube_gripper_carry_h3_v4r1_reference_development_v3"
        or decision.get("status") != "passed_development"
        or decision.get("passing_families") != ["lewm"]
        or decision.get("claims", {}).get("reference_development_passed") is not True
        or decision.get("claims", {}).get("original_task_retention_claim_allowed")
        is not False
        or decision.get("public_test", {}).get("opened") is not False
        or decision.get("public_test", {}).get("read") is not False
        or decision.get("public_test", {}).get("hashed") is not False
        or decision.get("public_test", {}).get("scored") is not False
    ):
        raise RuntimeError("Prior Cube Development decision is not an eligible gate")
    return observed


def collect_cube_cem_static_identities(
    prereg: Mapping[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    identities: dict[str, Any] = {}
    for name, raw_entry in _mapping(prereg.get("identity"), field="identity").items():
        entry = _mapping(raw_entry, field=f"identity.{name}")
        path = resolve_declared_path(str(entry.get("path", "")), repo_root=root)
        identities[name] = _declared_identity(
            entry, path=path, field=f"identity.{name}"
        )

    stable = _mapping(
        _mapping(prereg.get("runtime"), field="runtime").get("stable_worldmodel"),
        field="runtime.stable_worldmodel",
    )
    stable_root = Path(str(stable.get("repo", ""))).expanduser().resolve()
    if not (stable_root / ".git").exists():
        raise FileNotFoundError(f"Missing Stable-WorldModel git metadata: {stable_root}")
    commit = _git(stable_root, "rev-parse", "HEAD").stdout.strip()
    status = _git(
        stable_root, "status", "--porcelain", "--untracked-files=all"
    ).stdout
    if commit != stable.get("expected_ref") or status:
        raise RuntimeError("Pinned Stable-WorldModel runtime drifted or is dirty")
    runtime_files: dict[str, Any] = {}
    for name, raw_entry in _mapping(
        stable.get("required_files"), field="runtime.required_files"
    ).items():
        entry = _mapping(raw_entry, field=f"runtime.required_files.{name}")
        path = (stable_root / str(entry.get("path", ""))).resolve()
        runtime_files[name] = _declared_identity(
            entry, path=path, field=f"runtime.required_files.{name}"
        )

    original_h5 = _mapping(
        _mapping(prereg.get("data"), field="data").get("original_h5"),
        field="data.original_h5",
    )
    dataset_path = Path(str(original_h5.get("path", ""))).expanduser().resolve()
    dataset_identity = _declared_identity(
        _mapping(original_h5.get("expected_identity"), field="dataset_identity"),
        path=dataset_path,
        field="data.original_h5",
    )
    dataset_identity["row_count"] = int(
        original_h5["expected_identity"]["row_count"]
    )
    dataset_identity["episode_count"] = int(
        original_h5["expected_identity"]["episode_count"]
    )
    dataset_identity["action_dim"] = int(
        original_h5["expected_identity"]["action_dim"]
    )

    jobs: list[dict[str, Any]] = []
    for index, raw_job in enumerate(expected_cube_cem_jobs(prereg)):
        job = _mapping(raw_job, field=f"authorization.jobs[{index}]")
        checkpoint = Path(str(job.get("checkpoint", ""))).expanduser().resolve()
        config = Path(str(job.get("config", ""))).expanduser().resolve()
        checkpoint_identity = _declared_identity(
            _mapping(job.get("checkpoint_identity"), field="checkpoint_identity"),
            path=checkpoint,
            field=f"authorization.jobs[{index}].checkpoint",
        )
        config_identity = _declared_identity(
            _mapping(job.get("config_identity"), field="config_identity"),
            path=config,
            field=f"authorization.jobs[{index}].config",
        )
        observed_job = {
            "kind": job.get("kind"),
            "model_family": job.get("model_family"),
            "model_name": job.get("model_name"),
            "checkpoint": checkpoint_identity,
            "config": config_identity,
        }
        if job.get("kind") == "candidate":
            report = Path(str(job.get("training_report", ""))).expanduser().resolve()
            report_identity = _declared_identity(
                _mapping(
                    job.get("training_report_identity"),
                    field="training_report_identity",
                ),
                path=report,
                field=f"authorization.jobs[{index}].training_report",
            )
            observed_job["training_seed"] = int(job.get("training_seed", -1))
            observed_job["training_report"] = report_identity
        jobs.append(observed_job)

    return {
        "identity": identities,
        "prior_development": _validate_prior_development(prereg, repo_root=root),
        "runtime": {
            "root": str(stable_root),
            "commit": commit,
            "clean": True,
            "required_files": runtime_files,
        },
        "dataset": dataset_identity,
        "jobs": jobs,
    }


def _validate_query_catalog(
    prereg: Mapping[str, Any],
    *,
    path: Path,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = _stable_file_identity(path)
    if expected_identity is not None and dict(expected_identity) != identity:
        raise RuntimeError("Frozen Cube CEM query catalog identity drifted")
    payload = json.loads(path.read_text(encoding="utf-8"))
    selection = payload.get("selection", {})
    expected_selection = {
        "algorithm": "numpy_default_rng_choice_sorted_valid_rows",
        "historical_final_index_exclusion": True,
        "goal_offset_steps": 25,
        "eval_seeds": list(EVAL_SEEDS),
        "queries_per_seed": QUERIES_PER_EVAL_SEED,
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("task") != "cube"
        or selection != expected_selection
        or set(payload.get("queries", {})) != {str(seed) for seed in EVAL_SEEDS}
        or payload.get("runtime", {}).get("commit")
        != prereg["runtime"]["stable_worldmodel"]["expected_ref"]
        or payload.get("runtime", {}).get("clean") is not True
        or payload.get("dataset", {}).get("sha256")
        != prereg["data"]["original_h5"]["expected_identity"]["sha256"]
    ):
        raise RuntimeError("Frozen Cube CEM query catalog content drifted")
    for seed in EVAL_SEEDS:
        row = payload["queries"][str(seed)]
        indices = row.get("row_indices", [])
        episodes = row.get("episode_indices", [])
        starts = row.get("start_steps", [])
        if (
            len(indices) != QUERIES_PER_EVAL_SEED
            or len(set(indices)) != QUERIES_PER_EVAL_SEED
            or indices != sorted(indices)
            or len(episodes) != QUERIES_PER_EVAL_SEED
            or len(starts) != QUERIES_PER_EVAL_SEED
        ):
            raise RuntimeError(f"Frozen Cube CEM query row drifted for seed {seed}")
    return {"identity": identity, "payload": payload}


def validate_cube_cem_query_catalog(
    prereg: Mapping[str, Any],
    *,
    path: Path,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the frozen shared query catalog without opening Public data."""

    return _validate_query_catalog(
        prereg, path=path, expected_identity=expected_identity
    )


def _validate_freeze_receipt(
    prereg: dict[str, Any], *, prereg_path: Path, repo_root: Path
) -> dict[str, Any]:
    freeze_path = resolve_declared_path(
        prereg["planned_artifacts"]["freeze_receipt"], repo_root=repo_root
    )
    if not freeze_path.is_file() or freeze_path.is_symlink():
        raise FileNotFoundError(f"Missing Cube CEM retention freeze: {freeze_path}")
    receipt = json.loads(freeze_path.read_text(encoding="utf-8"))
    config_identity = _stable_file_identity(prereg_path)
    static = collect_cube_cem_static_identities(prereg, repo_root=repo_root)
    query_path = resolve_declared_path(
        prereg["planned_artifacts"]["query_catalog"], repo_root=repo_root
    )
    query = _validate_query_catalog(
        prereg,
        path=query_path,
        expected_identity=receipt.get("query_catalog"),
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
    expected_preflight_rows = [
        {
            "model": job["model_name"],
            "checkpoint_sha256": job["checkpoint_identity"]["sha256"],
            "config_sha256": job["config_identity"]["sha256"],
        }
        for job in expected_cube_cem_jobs(prereg)
    ]
    observed_preflight_rows = [
        {
            "model": row.get("model"),
            "checkpoint_sha256": row.get("checkpoint_sha256"),
            "config_sha256": row.get("config_sha256"),
        }
        for row in preflight_rows
        if isinstance(row, Mapping)
    ]
    expected_preflight_runtime = {
        "root": str(
            Path(prereg["runtime"]["stable_worldmodel"]["repo"]).resolve()
        ),
        "commit": prereg["runtime"]["stable_worldmodel"]["expected_ref"],
        "clean": True,
    }
    authorization = _mapping(
        receipt.get("authorization"), field="freeze.authorization"
    )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "frozen_authorized"
        or receipt.get("preregistration_id") != CUBE_CEM_RETENTION_ID
        or receipt.get("preregistration") != config_identity
        or receipt.get("static_identities") != static
        or model_preflight.get("runtime") != expected_preflight_runtime
        or observed_preflight_rows != expected_preflight_rows
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
        or receipt.get("public_test") != closed_public_contract()
    ):
        raise RuntimeError("Cube CEM retention freeze receipt drifted")
    return {
        **prereg,
        "_freeze_receipt": receipt,
        "_freeze_receipt_path": str(freeze_path),
        "_query_catalog": query["payload"],
        "_query_catalog_path": str(query_path),
    }


def load_cube_cem_retention_prereg(
    path: Path | str = DEFAULT_CUBE_CEM_RETENTION_PREREG,
    *,
    require_freeze: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Cube CEM retention preregistration must be a mapping")
    _validate_preregistration(payload)
    result = {**payload, "_config_path": str(config_path)}
    if not require_freeze:
        return result
    return _validate_freeze_receipt(
        result,
        prereg_path=config_path,
        repo_root=(repo_root or repository_root()).resolve(),
    )


def expected_cube_cem_result_directory(
    prereg: Mapping[str, Any], *, model_name: str, repo_root: Path | None = None
) -> Path:
    names = {row["model_name"] for row in expected_cube_cem_jobs(prereg)}
    if model_name not in names:
        raise ValueError(f"Unknown Cube CEM model name: {model_name}")
    root = resolve_declared_path(
        prereg["planned_artifacts"]["retention_root"], repo_root=repo_root
    )
    return root / "results" / model_name


def _expected_protocol(prereg: Mapping[str, Any]) -> dict[str, Any]:
    stable_root = Path(prereg["runtime"]["stable_worldmodel"]["repo"]).resolve()
    source = stable_root / "scripts/plan/config/cube.yaml"
    dataset = Path(prereg["data"]["original_h5"]["path"]).resolve()
    expected = prereg["data"]["original_h5"]["expected_identity"]
    evaluation = prereg["evaluation"]
    return {
        "source": str(source),
        "source_sha256": file_sha256(source),
        "dataset": str(dataset),
        "dataset_size_bytes": int(expected["size_bytes"]),
        "dataset_sha256": expected["sha256"],
        "num_eval_per_seed": QUERIES_PER_EVAL_SEED,
        "eval_seeds": list(EVAL_SEEDS),
        "goal_offset_steps": 25,
        "eval_budget": int(evaluation["eval_budget"]),
        "history_len": 3,
        "horizon": int(evaluation["horizon"]),
        "receding_horizon": int(evaluation["receding_horizon"]),
        "action_block": int(evaluation["action_block"]),
        "cem_samples": int(evaluation["cem_samples"]),
        "cem_iterations": int(evaluation["cem_iterations"]),
        "cem_topk": int(evaluation["cem_topk"]),
        "videos_written": False,
    }


def validate_cube_cem_job_result(
    prereg: Mapping[str, Any],
    *,
    model_name: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    expected_job = next(
        row
        for row in expected_cube_cem_jobs(prereg)
        if row["model_name"] == model_name
    )
    result_dir = expected_cube_cem_result_directory(
        prereg, model_name=model_name, repo_root=repo_root
    )
    report_path = result_dir / "aggregate.json"
    catalog_path = result_dir / "query_catalog.json"
    report_identity = _stable_file_identity(report_path)
    catalog_identity = _stable_file_identity(catalog_path)
    frozen_catalog_path = Path(str(prereg["_query_catalog_path"])).resolve()
    if catalog_path.read_bytes() != frozen_catalog_path.read_bytes():
        raise RuntimeError(f"Cube CEM query catalog differs for {model_name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if len(report.get("models", ())) != 1:
        raise RuntimeError(f"Cube CEM result must contain one model: {model_name}")
    model = report["models"][0]
    expected_checkpoint = Path(expected_job["checkpoint"]).resolve()
    expected_config = Path(expected_job["config"]).resolve()
    seeds = model.get("seeds", [])
    expected_runtime = {
        "root": str(Path(prereg["runtime"]["stable_worldmodel"]["repo"]).resolve()),
        "commit": prereg["runtime"]["stable_worldmodel"]["expected_ref"],
        "clean": True,
    }
    if (
        report.get("schema_version") != 1
        or report.get("status") != "standard_original_task_real_environment_cem"
        or report.get("task") != "cube"
        or report.get("runtime") != expected_runtime
        or report.get("protocol") != _expected_protocol(prereg)
        or report.get("query_catalog", {}).get("sha256")
        != prereg["_freeze_receipt"]["query_catalog"]["sha256"]
        or report.get("public_test", {}).get("opened") is not False
        or report.get("public_test", {}).get("read") is not False
        or report.get("public_test", {}).get("hashed") is not False
        or report.get("public_test", {}).get("scored") is not False
        or model.get("model") != model_name
        or Path(str(model.get("checkpoint", ""))).resolve() != expected_checkpoint
        or model.get("checkpoint_sha256")
        != expected_job["checkpoint_identity"]["sha256"]
        or Path(str(model.get("config", ""))).resolve() != expected_config
        or model.get("config_sha256") != expected_job["config_identity"]["sha256"]
        or [int(row.get("eval_seed", -1)) for row in seeds] != list(EVAL_SEEDS)
    ):
        raise RuntimeError(f"Cube CEM result provenance drifted for {model_name}")
    total_successes = 0
    all_outcomes: list[bool] = []
    for seed, row in zip(EVAL_SEEDS, seeds, strict=True):
        outcomes = row.get("episode_successes")
        if (
            not isinstance(outcomes, list)
            or len(outcomes) != QUERIES_PER_EVAL_SEED
            or any(not isinstance(value, bool) for value in outcomes)
            or int(row.get("query_count", -1)) != QUERIES_PER_EVAL_SEED
            or int(row.get("success_count", -1)) != sum(outcomes)
            or int(row.get("eval_seed", -1)) != seed
        ):
            raise RuntimeError(f"Cube CEM outcomes drifted for {model_name}/seed{seed}")
        total_successes += sum(outcomes)
        all_outcomes.extend(outcomes)
    aggregate = model.get("aggregate", {})
    if (
        int(aggregate.get("evaluation_count", -1)) != TOTAL_QUERIES
        or int(aggregate.get("success_count", -1)) != total_successes
        or float(aggregate.get("success_rate", -1.0))
        != total_successes / TOTAL_QUERIES
    ):
        raise RuntimeError(f"Cube CEM aggregate drifted for {model_name}")
    return {
        "model_name": model_name,
        "kind": expected_job["kind"],
        "training_seed": expected_job.get("training_seed"),
        "checkpoint": expected_checkpoint,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "report": report_path,
        "report_identity": report_identity,
        "catalog_identity": catalog_identity,
        "report_payload": report,
        "model_payload": model,
        "success_count": total_successes,
        "outcomes": all_outcomes,
    }


def paired_cube_cem_noninferiority(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    training_seed: int,
) -> dict[str, Any]:
    """Build one paired 300-query success-count comparison."""

    baseline_seed_rows = {
        int(row["eval_seed"]): row for row in baseline["model_payload"]["seeds"]
    }
    candidate_seed_rows = {
        int(row["eval_seed"]): row for row in candidate["model_payload"]["seeds"]
    }
    if set(baseline_seed_rows) != set(EVAL_SEEDS) or set(candidate_seed_rows) != set(
        EVAL_SEEDS
    ):
        raise RuntimeError("Cube CEM paired comparison seed set drifted")
    by_eval_seed = []
    for eval_seed in EVAL_SEEDS:
        base_outcomes = baseline_seed_rows[eval_seed]["episode_successes"]
        new_outcomes = candidate_seed_rows[eval_seed]["episode_successes"]
        if (
            len(base_outcomes) != QUERIES_PER_EVAL_SEED
            or len(new_outcomes) != QUERIES_PER_EVAL_SEED
        ):
            raise RuntimeError("Cube CEM paired comparison query count drifted")
        by_eval_seed.append(
            {
                "eval_seed": eval_seed,
                "query_count": QUERIES_PER_EVAL_SEED,
                "baseline_successes": sum(base_outcomes),
                "candidate_successes": sum(new_outcomes),
                "success_delta": sum(new_outcomes) - sum(base_outcomes),
                "regressions": sum(
                    before and not after
                    for before, after in zip(base_outcomes, new_outcomes)
                ),
                "improvements": sum(
                    not before and after
                    for before, after in zip(base_outcomes, new_outcomes)
                ),
            }
        )
    candidate_total = int(candidate["success_count"])
    baseline_total = int(baseline["success_count"])
    if baseline_total != sum(row["baseline_successes"] for row in by_eval_seed):
        raise RuntimeError("Cube CEM baseline success total drifted")
    if candidate_total != sum(row["candidate_successes"] for row in by_eval_seed):
        raise RuntimeError("Cube CEM candidate success total drifted")
    return {
        "model_name": candidate["model_name"],
        "model_family": "lewm",
        "training_seed": int(training_seed),
        "checkpoint": str(candidate["checkpoint"]),
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "evaluation_count": TOTAL_QUERIES,
        "baseline_successes": baseline_total,
        "candidate_successes": candidate_total,
        "success_delta": candidate_total - baseline_total,
        "noninferiority_margin_successes": NONINFERIORITY_MARGIN_SUCCESSES,
        "passed": candidate_total
        >= baseline_total - NONINFERIORITY_MARGIN_SUCCESSES,
        "by_eval_seed": by_eval_seed,
        "source_result": candidate["report_identity"],
    }


def build_cube_cem_retention_result(
    prereg: Mapping[str, Any],
    *,
    validated: Sequence[Mapping[str, Any]],
    matrix_report_path: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    by_name = {row["model_name"]: row for row in validated}
    jobs = expected_cube_cem_jobs(prereg)
    if set(by_name) != {row["model_name"] for row in jobs}:
        raise RuntimeError("Cube CEM validated result set is incomplete")
    baseline = by_name["baseline_lewm"]
    comparisons = []
    for seed in TRAINING_SEEDS:
        candidate = by_name[f"lewm_seed{seed}"]
        comparisons.append(
            paired_cube_cem_noninferiority(
                baseline, candidate, training_seed=seed
            )
        )
    config_path = Path(str(prereg["_config_path"])).resolve()
    freeze_path = Path(str(prereg["_freeze_receipt_path"])).resolve()
    decision_entry = prereg["prior_development"]["development_decision"]
    decision_path = resolve_declared_path(decision_entry["path"], repo_root=root)
    query_path = Path(str(prereg["_query_catalog_path"])).resolve()
    return {
        "schema_version": 1,
        "status": "completed",
        "preregistration_id": CUBE_CEM_RETENTION_ID,
        "task": "cube_original_task_retention",
        "protocol": _expected_protocol(prereg),
        "authorization_chain": {
            "preregistration": _stable_file_identity(config_path),
            "freeze_receipt": _stable_file_identity(freeze_path),
            "development_decision": _stable_file_identity(decision_path),
            "query_catalog": _stable_file_identity(query_path),
            "matrix_report": _stable_file_identity(matrix_report_path),
        },
        "query_catalog": {
            **_stable_file_identity(query_path),
            "identical_across_all_results": all(
                row["catalog_identity"]["sha256"]
                == file_sha256(query_path)
                for row in validated
            ),
        },
        "baseline": {
            "model_name": baseline["model_name"],
            "checkpoint": str(baseline["checkpoint"]),
            "checkpoint_sha256": baseline["checkpoint_sha256"],
            "success_count": baseline["success_count"],
            "evaluation_count": TOTAL_QUERIES,
            "source_result": baseline["report_identity"],
        },
        "comparisons": comparisons,
        "passed": bool(comparisons and all(row["passed"] for row in comparisons)),
        "public_test": closed_public_contract(),
    }


def validate_cube_cem_retention_result(
    prereg: Mapping[str, Any],
    *,
    result_path: Path,
    matrix_report_path: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    expected_root = resolve_declared_path(
        prereg["planned_artifacts"]["retention_root"], repo_root=repo_root
    )
    expected_result = expected_root / "retention_result.json"
    expected_matrix = expected_root / "matrix_report.json"
    if result_path.resolve() != expected_result or matrix_report_path.resolve() != expected_matrix:
        raise RuntimeError("Cube CEM retention result path drifted")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    validated = [
        validate_cube_cem_job_result(
            prereg, model_name=job["model_name"], repo_root=repo_root
        )
        for job in expected_cube_cem_jobs(prereg)
    ]
    expected = build_cube_cem_retention_result(
        prereg,
        validated=validated,
        matrix_report_path=matrix_report_path,
        repo_root=repo_root,
    )
    if payload != expected:
        raise RuntimeError("Cube CEM retention aggregate drifted")
    return {"payload": payload, "validated_jobs": validated}


__all__ = [
    "CUBE_CEM_RETENTION_ID",
    "CUBE_CEM_RETENTION_PROTOCOL",
    "DEFAULT_CUBE_CEM_RETENTION_PREREG",
    "EVAL_SEEDS",
    "NONINFERIORITY_MARGIN_SUCCESSES",
    "QUERIES_PER_EVAL_SEED",
    "TOTAL_QUERIES",
    "TRAINING_SEEDS",
    "build_cube_cem_retention_result",
    "closed_public_contract",
    "collect_cube_cem_static_identities",
    "expected_cube_cem_jobs",
    "expected_cube_cem_result_directory",
    "file_sha256",
    "load_cube_cem_retention_prereg",
    "paired_cube_cem_noninferiority",
    "resolve_declared_path",
    "validate_cube_cem_job_result",
    "validate_cube_cem_query_catalog",
    "validate_cube_cem_retention_result",
]
