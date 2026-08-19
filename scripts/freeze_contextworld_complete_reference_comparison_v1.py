#!/usr/bin/env python3
"""Freeze the report-all ContextWorld reference-comparison addendum."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

import yaml

from contextworld.paths import repository_root, resolve_contextworld_path


ROOT = repository_root()
DEFAULT_PREREGISTRATION = (
    ROOT
    / "configs/benchmark/contextworld_complete_reference_comparison_prereg_v1.yaml"
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


def checkpoint_specs(preregistration: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for component, section in preregistration["icl_checkpoints"].items():
        for checkpoint in section["checkpoints"]:
            yield f"{component}/{checkpoint['family']}/seed{checkpoint['seed']}", checkpoint


def validate_checkpoint(label: str, specification: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = validate_file(specification, label=f"{label} checkpoint")
    config_path = Path(checkpoint["resolved_path"]).parent / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise FileNotFoundError(f"{label} config: {config_path}")
    config = {
        "path": str(config_path),
        "sha256": file_sha256(config_path),
        "size_bytes": config_path.stat().st_size,
    }
    if config["sha256"] != str(specification["config_sha256"]):
        raise RuntimeError(f"{label} config SHA-256 drifted")
    if config["size_bytes"] != int(specification["config_size_bytes"]):
        raise RuntimeError(f"{label} config size drifted")
    return {"checkpoint": checkpoint, "config": config}


def validate_dataset(specification: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path = resolve_input(str(specification["path"]))
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label}: {path}")
    if path.stat().st_size != int(specification["size_bytes"]):
        raise RuntimeError(f"{label} size drifted")
    return {
        "path": str(specification["path"]),
        "resolved_path": str(path),
        "sha256": str(specification["sha256"]),
        "size_bytes": path.stat().st_size,
        "content_hash_authority": "previously_frozen_full_file_identity",
        "rehash_during_addendum_freeze": False,
    }


def aggregate_successes(path: Path) -> tuple[str, int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = payload.get("model")
    if isinstance(model, Mapping):
        aggregate = model.get("aggregate")
        family = str(model.get("family", model.get("model", "")))
    else:
        models = payload.get("models")
        if not isinstance(models, list) or len(models) != 1:
            raise RuntimeError(f"Unsupported CEM result schema: {path}")
        aggregate = models[0].get("aggregate")
        family = str(models[0].get("model", ""))
    if not isinstance(aggregate, Mapping):
        raise RuntimeError(f"CEM aggregate is missing: {path}")
    return family, int(aggregate["success_count"]), int(aggregate["evaluation_count"])


def freeze(preregistration_path: Path, output: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.expanduser().resolve()
    preregistration = yaml.safe_load(preregistration_path.read_text(encoding="utf-8"))
    if (
        preregistration.get("schema_version") != 1
        or preregistration.get("preregistration_id")
        != "contextworld_complete_reference_comparison_v1"
        or preregistration.get("status")
        != "preregistered_before_supplemental_public_scoring"
    ):
        raise ValueError("Unexpected complete-comparison preregistration")
    authorization = preregistration["authorization"]
    if not all(
        authorization.get(field) is True
        for field in (
            "public_scoring_authorized",
            "cem_scoring_authorized",
            "failed_models_must_remain_visible",
            "no_result_based_checkpoint_selection",
            "no_retraining_after_public_scores",
            "no_threshold_or_recipe_changes",
        )
    ):
        raise RuntimeError("Complete-comparison authorization is incomplete")

    output = output.expanduser().resolve()
    expected_output = (ROOT / preregistration["outputs"]["freeze_receipt"]).resolve()
    if output != expected_output:
        raise ValueError("Freeze receipt path differs from preregistration")
    comparison_root = (ROOT / preregistration["outputs"]["root"]).resolve()
    if comparison_root.exists():
        raise FileExistsError(f"Comparison namespace already exists: {comparison_root}")

    runtime_root = Path(preregistration["runtime"]["stable_worldmodel"]["path"])
    git_pointer = (runtime_root / ".git").read_text(encoding="utf-8").strip()
    if not git_pointer.startswith("gitdir: "):
        raise RuntimeError("Stable-WorldModel worktree git pointer is invalid")
    git_directory = Path(git_pointer.removeprefix("gitdir: ")).resolve()
    observed_commit = subprocess.run(
        [
            "git",
            f"--git-dir={git_directory}",
            f"--work-tree={runtime_root}",
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_commit = preregistration["runtime"]["stable_worldmodel"]["commit"]
    if observed_commit != expected_commit:
        raise RuntimeError("Stable-WorldModel commit drifted")

    implementation = {
        name: validate_file(specification, label=f"implementation.{name}")
        for name, specification in preregistration["implementation"].items()
    }
    releases: dict[str, Any] = {}
    for component, section in preregistration["frozen_data"].items():
        releases[component] = {
            "release_config": validate_file(
                section["release_config"], label=f"{component} release config"
            ),
            "public_table_contract": dict(section["public_table"]),
            "public_table_opened_during_freeze": False,
        }
        if "public_success" in section:
            releases[component]["public_success"] = validate_file(
                section["public_success"], label=f"{component} Public success marker"
            )

    checkpoints = {
        label: validate_checkpoint(label, specification)
        for label, specification in checkpoint_specs(preregistration)
    }
    if len(checkpoints) != 15:
        raise RuntimeError("Complete comparison must freeze exactly 15 checkpoints")

    cem_inputs: dict[str, Any] = {}
    for environment, section in preregistration["cem_inputs"].items():
        cem_inputs[environment] = {
            "plan_config": validate_file(
                section["plan_config"], label=f"{environment} plan config"
            ),
            "dataset": validate_dataset(
                section["dataset"], label=f"{environment} dataset"
            ),
            "query_catalog": validate_file(
                section["query_catalog"], label=f"{environment} query catalog"
            ),
        }
        if "dataset_identity_audit" in section:
            cem_inputs[environment]["dataset_identity_audit"] = validate_file(
                section["dataset_identity_audit"],
                label=f"{environment} dataset identity audit",
            )

    reused: list[dict[str, Any]] = []
    for row in preregistration["reused_motion_damping_cem"]["results"]:
        identity = validate_file(row, label=f"motion CEM {row['family']} {row['seed']}")
        _, successes, evaluations = aggregate_successes(Path(identity["resolved_path"]))
        if successes != int(row["successes"]) or evaluations != int(row["evaluations"]):
            raise RuntimeError("Frozen Motion Damping CEM outcome drifted")
        reused.append({**identity, "family": row["family"], "seed": row["seed"], "successes": successes, "evaluations": evaluations})

    preregistration_identity = {
        "path": str(preregistration_path.relative_to(ROOT)),
        "sha256": file_sha256(preregistration_path),
        "size_bytes": preregistration_path.stat().st_size,
    }
    receipt = {
        "schema_version": 1,
        "freeze_id": "contextworld_complete_reference_comparison_freeze_v1",
        "status": "frozen_before_supplemental_public_scoring",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": preregistration_identity,
        "authorization": dict(authorization),
        "stable_worldmodel": {"path": str(runtime_root), "commit": observed_commit},
        "implementation": implementation,
        "releases": releases,
        "checkpoints": checkpoints,
        "cem_inputs": cem_inputs,
        "reused_motion_damping_cem": reused,
        "execution_counts": {
            "icl_checkpoints": 15,
            "new_contact_friction_cem_checkpoints": 6,
            "reused_motion_damping_cem_checkpoints": 6,
            "new_cube_pldm_cem_checkpoints": 3,
        },
        "public_access": {
            "started_during_freeze": False,
            "tables_opened_during_freeze": False,
            "scores_observed_during_freeze": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=False)
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
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
