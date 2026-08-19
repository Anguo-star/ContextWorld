#!/usr/bin/env python3
"""Freeze the omitted report-all CEM cells before any amended execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from contextworld.paths import repository_root, resolve_contextworld_path


ROOT = repository_root()
DEFAULT_PREREGISTRATION = (
    ROOT
    / "configs/benchmark/contextworld_complete_reference_comparison_execution_amendment_v2.yaml"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return resolve_contextworld_path(candidate, repo_root=ROOT)


def resolve_output(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT / candidate).resolve()


def validate_file(specification: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = resolve_input(str(specification["path"]))
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label}: {path}")
    observed = {
        "path": str(specification["path"]),
        "resolved_path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if observed["sha256"] != str(specification["sha256"]):
        raise RuntimeError(f"{label} SHA-256 drifted")
    if observed["size_bytes"] != int(specification["size_bytes"]):
        raise RuntimeError(f"{label} size drifted")
    return observed


def validate_dataset(
    specification: Mapping[str, Any],
    *,
    label: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    path = resolve_input(str(specification["path"]))
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label}: {path}")
    if path.stat().st_size != int(specification["size_bytes"]):
        raise RuntimeError(f"{label} size drifted")
    frozen = audit["datasets"][label]
    for field in ("path", "sha256", "size_bytes"):
        if str(frozen[field]) != str(specification[field]):
            raise RuntimeError(f"{label} differs from the bound identity audit")
    if frozen.get("content_hash_checked") is not True:
        raise RuntimeError(f"{label} lacks a full-file identity audit")
    return {
        "path": str(specification["path"]),
        "resolved_path": str(path),
        "sha256": str(specification["sha256"]),
        "size_bytes": path.stat().st_size,
        "content_hash_authority": "bound_full_file_identity_audit",
        "rehash_during_amendment_freeze": False,
    }


def worktree_commit(root: Path) -> str:
    dot_git = root / ".git"
    if dot_git.is_file():
        pointer = dot_git.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir: "):
            raise RuntimeError(f"Invalid worktree git pointer: {root}")
        git_dir = Path(pointer.removeprefix("gitdir: ")).resolve()
    elif dot_git.is_dir():
        git_dir = dot_git.resolve()
    else:
        raise FileNotFoundError(f"Git metadata is missing: {root}")
    return subprocess.run(
        [
            "git",
            f"--git-dir={git_dir}",
            f"--work-tree={root}",
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_runtime(
    specification: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    root = Path(str(specification["root"])).expanduser().resolve()
    commit = worktree_commit(root)
    if commit != str(specification["commit"]):
        raise RuntimeError(f"{label} Stable-WorldModel commit drifted")
    observed: dict[str, Any] = {"root": str(root), "commit": commit}
    for name in ("cem_solver", "policy"):
        if name in specification:
            relative = specification[name]
            observed[name] = validate_file(
                {**relative, "path": str(root / relative["path"])},
                label=f"{label}.{name}",
            )
    return observed


def validate_checkpoint(
    specification: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    checkpoint = validate_file(specification, label=f"{label} checkpoint")
    config_path = Path(checkpoint["resolved_path"]).parent / "config.json"
    config_specification = {
        "path": str(config_path),
        "sha256": specification["config_sha256"],
        "size_bytes": specification["config_size_bytes"],
    }
    config = validate_file(config_specification, label=f"{label} config")
    return {
        "seed": int(specification["seed"]),
        "checkpoint": checkpoint,
        "config": config,
        "output": str(specification["output"]),
    }


def aggregate_counts(path: Path) -> tuple[int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model")
    if not isinstance(model, Mapping) or not isinstance(model.get("aggregate"), Mapping):
        raise RuntimeError(f"Unsupported aggregate schema: {path}")
    aggregate = model["aggregate"]
    return int(aggregate["success_count"]), int(aggregate["evaluation_count"])


def freeze(preregistration_path: Path, output: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.expanduser().resolve()
    preregistration = yaml.safe_load(preregistration_path.read_text(encoding="utf-8"))
    if (
        preregistration.get("schema_version") != 1
        or preregistration.get("amendment_id")
        != "contextworld_complete_reference_comparison_execution_amendment_v2"
        or preregistration.get("status")
        != "preregistered_before_amended_cem_execution"
    ):
        raise ValueError("Unexpected complete-comparison amendment")

    policy = preregistration["report_all_policy"]
    required_true = (
        "every_frozen_comparator_runs_icl_and_cem",
        "threshold_controls_verdict_only",
        "failed_results_remain_visible",
        "family_matched_original_baseline_required",
        "no_result_based_checkpoint_selection",
        "no_retraining_or_finetuning",
        "no_threshold_or_recipe_changes",
        "no_result_based_retry",
    )
    if not all(policy.get(field) is True for field in required_true):
        raise RuntimeError("Report-all policy is incomplete")
    if policy.get("threshold_controls_execution") is not False:
        raise RuntimeError("Thresholds must not control execution")

    expected_output = resolve_output(preregistration["outputs"]["amendment_freeze_receipt"])
    output = output.expanduser().resolve()
    if output != expected_output:
        raise ValueError("Amendment receipt path differs from preregistration")
    if output.exists():
        raise FileExistsError(f"Amendment receipt already exists: {output}")
    for key in ("action_strength_root", "reacher_arm_mass_root", "portal_exit_root"):
        target = resolve_output(preregistration["outputs"][key])
        if target.exists():
            raise FileExistsError(f"Amended CEM namespace already exists: {target}")

    base_preregistration = validate_file(
        preregistration["base_freeze"]["preregistration"],
        label="base preregistration",
    )
    base_receipt = validate_file(
        preregistration["base_freeze"]["receipt"], label="base freeze receipt"
    )
    base_payload = json.loads(
        Path(base_receipt["resolved_path"]).read_text(encoding="utf-8")
    )
    if base_payload.get("status") != preregistration["base_freeze"]["status_required"]:
        raise RuntimeError("Base freeze status drifted")
    if base_payload.get("public_access", {}).get("scores_observed_during_freeze") is not False:
        raise RuntimeError("Base freeze did not precede score access")

    scoreboard = validate_file(
        preregistration["historical_scoreboard"], label="historical scoreboard"
    )
    scoreboard_payload = json.loads(
        Path(scoreboard["resolved_path"]).read_text(encoding="utf-8")
    )
    rows = scoreboard_payload.get("component_results", [])
    if len(rows) != int(preregistration["historical_scoreboard"]["row_count"]):
        raise RuntimeError("Historical scoreboard row count drifted")
    missing = {
        (row.get("component_id"), row.get("method_name"))
        for row in rows
        if row.get("original_task_retention", {}).get("result") == "NOT_EVALUATED"
    }
    if len(missing) != 3:
        raise RuntimeError("Historical scoreboard no longer has exactly three missing CEM rows")

    implementation = {
        name: validate_file(specification, label=f"implementation.{name}")
        for name, specification in preregistration["implementation"].items()
    }
    runtimes = {
        name: validate_runtime(specification, label=name)
        for name, specification in preregistration["runtimes"].items()
    }

    baseline_section = preregistration["baseline_evidence"]
    baseline_evidence = {
        "result_freeze": validate_file(
            baseline_section["result_freeze"], label="baseline result freeze"
        ),
        "matrix_summary": validate_file(
            baseline_section["matrix_summary"], label="baseline matrix summary"
        ),
    }
    for name in ("pusht_pldm", "reacher_pldm"):
        specification = baseline_section[name]
        result = validate_file(specification["result"], label=f"{name} baseline")
        successes, evaluations = aggregate_counts(Path(result["resolved_path"]))
        if successes != int(specification["successes"]) or evaluations != int(
            specification["evaluations"]
        ):
            raise RuntimeError(f"{name} baseline outcome drifted")
        baseline_evidence[name] = {
            "result": result,
            "successes": successes,
            "evaluations": evaluations,
            "threshold_successes": int(specification["threshold_successes"]),
        }
    portal_receipts = []
    for specification in baseline_section["tworoom_pldm"]["receipts"]:
        identity = validate_file(
            specification, label=f"tworoom PLDM baseline seed{specification['eval_seed']}"
        )
        payload = json.loads(Path(identity["resolved_path"]).read_text(encoding="utf-8"))
        aggregate = payload.get("aggregate", {})
        if (
            int(aggregate.get("successes", -1)) != int(specification["successes"])
            or int(aggregate.get("evaluations", -1))
            != int(specification["evaluations"])
            or int(payload.get("protocol", {}).get("eval_seed", -1))
            != int(specification["eval_seed"])
        ):
            raise RuntimeError("TwoRoom PLDM baseline receipt outcome drifted")
        portal_receipts.append(
            {
                **identity,
                "eval_seed": int(specification["eval_seed"]),
                "successes": int(specification["successes"]),
                "evaluations": int(specification["evaluations"]),
            }
        )
    if sum(row["successes"] for row in portal_receipts) != int(
        baseline_section["tworoom_pldm"]["successes"]
    ):
        raise RuntimeError("TwoRoom PLDM baseline total drifted")
    baseline_evidence["tworoom_pldm"] = {
        "receipts": portal_receipts,
        "successes": int(baseline_section["tworoom_pldm"]["successes"]),
        "evaluations": int(baseline_section["tworoom_pldm"]["evaluations"]),
        "threshold_successes": int(
            baseline_section["tworoom_pldm"]["threshold_successes"]
        ),
    }

    input_section = preregistration["frozen_inputs"]
    audit_identity = validate_file(input_section["identity_audit"], label="input identity audit")
    audit = json.loads(Path(audit_identity["resolved_path"]).read_text(encoding="utf-8"))
    frozen_inputs: dict[str, Any] = {"identity_audit": audit_identity}
    for environment in ("pusht", "reacher"):
        specification = input_section[environment]
        frozen_inputs[environment] = {
            "plan_config": validate_file(
                specification["plan_config"], label=f"{environment} plan config"
            ),
            "catalog": validate_file(
                specification["catalog"], label=f"{environment} catalog"
            ),
            "dataset": validate_dataset(
                specification["dataset"], label=environment, audit=audit
            ),
            "eval_seeds": list(specification["eval_seeds"]),
            "queries_per_seed": int(specification["queries_per_seed"]),
        }
    tworoom = input_section["tworoom"]
    frozen_inputs["tworoom"] = {
        "catalog": validate_file(tworoom["catalog"], label="tworoom catalog"),
        "dataset": validate_file(tworoom["dataset"], label="tworoom dataset"),
        "normalizer": validate_file(
            tworoom["normalizer"], label="tworoom normalizer"
        ),
        "eval_seeds": list(tworoom["eval_seeds"]),
        "queries_per_seed": int(tworoom["queries_per_seed"]),
    }

    execution_cells: dict[str, Any] = {}
    checkpoint_count = 0
    for cell_name, cell in preregistration["execution_cells"].items():
        cell_receipt: dict[str, Any] = {
            "release_config": validate_file(
                cell["release_config"], label=f"{cell_name} release config"
            ),
            "task": str(cell["task"]),
            "mujoco_gl": str(cell["mujoco_gl"]),
            "baseline": str(cell["baseline"]),
            "checkpoints": [],
        }
        if "superseded_stop_receipt" in cell:
            stop = validate_file(
                cell["superseded_stop_receipt"],
                label=f"{cell_name} superseded stop receipt",
            )
            stop_payload = json.loads(Path(stop["resolved_path"]).read_text(encoding="utf-8"))
            if stop_payload.get("cem", {}).get("executed") is not False:
                raise RuntimeError("Historical ActionStrength stop receipt is not a zero-run stop")
            cell_receipt["superseded_stop_receipt"] = stop
        for checkpoint in cell["checkpoints"]:
            target = resolve_output(checkpoint["output"])
            if target.exists():
                raise FileExistsError(f"Candidate output already exists: {target}")
            cell_receipt["checkpoints"].append(
                validate_checkpoint(
                    checkpoint, label=f"{cell_name}/seed{checkpoint['seed']}"
                )
            )
            checkpoint_count += 1
        execution_cells[cell_name] = cell_receipt
    if checkpoint_count != 9:
        raise RuntimeError("Amendment must freeze exactly nine candidate checkpoints")

    preregistration_identity = {
        "path": str(preregistration_path.relative_to(ROOT)),
        "sha256": file_sha256(preregistration_path),
        "size_bytes": preregistration_path.stat().st_size,
    }
    receipt = {
        "schema_version": 1,
        "freeze_id": "contextworld_complete_reference_comparison_execution_amendment_freeze_v2",
        "status": "frozen_before_amended_cem_execution",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": preregistration_identity,
        "report_all_policy": dict(policy),
        "base_freeze": {
            "preregistration": base_preregistration,
            "receipt": base_receipt,
        },
        "historical_scoreboard": scoreboard,
        "implementation": implementation,
        "runtimes": runtimes,
        "baseline_evidence": baseline_evidence,
        "frozen_inputs": frozen_inputs,
        "execution_cells": execution_cells,
        "decision_rules": preregistration["decision_rules"],
        "execution_budget": preregistration["execution_budget"],
        "public_access": {
            "supplemental_public_scores_observed_before_amendment": False,
            "candidate_cem_episodes_consumed_before_amendment": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = freeze(args.prereg, args.output)
    print(
        json.dumps(
            {"status": receipt["status"], "output": str(args.output)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
