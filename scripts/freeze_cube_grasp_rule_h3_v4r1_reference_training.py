#!/usr/bin/env python3
"""Freeze the Cube v4r1 Development-only LeWM/PLDM training authorization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import h5py


ROOT = Path(__file__).resolve().parents[1]

from contextworld.benchmarks.cube_grasp_rule_reference_training import (  # noqa: E402
    AUTHORIZED_SPLITS,
    CUBE_REFERENCE_TRAINING_ID,
    CUBE_REFERENCE_TRAINING_PROTOCOL,
    DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
    FREEZE_STATUS,
    cube_reference_data_tree_identity,
    cube_reference_infrastructure_recovery_identity,
    file_sha256,
    load_cube_reference_training_prereg,
    resolve_cube_reference_initial_checkpoint,
    resolve_cube_reference_training_input,
)
from contextworld.paths import portable_contextworld_path, resolve_contextworld_path  # noqa: E402


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _stable_stat(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "inode": int(value.st_ino),
        "device": int(value.st_dev),
    }


def _hash_stable_file(path: Path) -> tuple[str, dict[str, int]]:
    before = _stable_stat(path)
    digest = file_sha256(path)
    after = _stable_stat(path)
    if before != after:
        raise RuntimeError(f"Input changed while it was being hashed: {path}")
    return digest, after


def _resolve(value: str, *, repo_root: Path) -> Path:
    return resolve_contextworld_path(value, repo_root=repo_root)


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _verify_declared_file(
    entry: Mapping[str, Any], *, repo_root: Path, name: str
) -> dict[str, Any]:
    path = _resolve(str(entry["path"]), repo_root=repo_root)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{name}: missing regular file {path}")
    digest, stat = _hash_stable_file(path)
    if digest != str(entry["sha256"]):
        raise RuntimeError(f"{name}: SHA256 mismatch")
    if stat["size_bytes"] != int(entry["size_bytes"]):
        raise RuntimeError(f"{name}: size mismatch")
    return {
        "path": str(entry["path"]),
        "sha256": digest,
        "size_bytes": stat["size_bytes"],
    }


def _verify_source_h5(path: Path, specification: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing Cube H5: {path}")
    expected = _mapping(specification.get("expected_identity"), field="original_h5.expected_identity")
    stat_before = _stable_stat(path)
    if stat_before["size_bytes"] != int(expected["size_bytes"]):
        raise RuntimeError("Upstream Cube H5 size mismatch")
    with h5py.File(path, "r", swmr=True) as handle:
        action_shape = tuple(int(value) for value in handle["action"].shape)
        episode_count = int(handle["ep_len"].shape[0])
    if action_shape != (int(expected["row_count"]), 5):
        raise RuntimeError("Upstream Cube H5 must contain rows x five actions")
    if episode_count != int(expected["episode_count"]):
        raise RuntimeError("Upstream Cube H5 episode count mismatch")
    digest, stat_after = _hash_stable_file(path)
    if digest != str(expected["sha256"]):
        raise RuntimeError("Upstream Cube H5 SHA256 mismatch")
    if stat_before != stat_after:
        raise RuntimeError("Upstream Cube H5 changed during verification")
    return {
        "symbol": specification["source_symbol"],
        "path_recorded": False,
        "sha256": digest,
        "size_bytes": stat_after["size_bytes"],
        "row_count": action_shape[0],
        "action_dim": action_shape[1],
        "episode_count": episode_count,
        "stable_stat": stat_after,
    }


def _verify_original_lance(
    path: Path, specification: Mapping[str, Any]
) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise FileNotFoundError(f"Missing original Cube Lance directory: {path}")
    expected = _mapping(
        specification.get("expected_identity"),
        field="original_lance.expected_identity",
    )
    expected_files = _mapping(
        expected.get("files"), field="original_lance.expected_identity.files"
    )
    observed_paths = {
        child.relative_to(path).as_posix(): child
        for child in path.rglob("*")
        if child.is_file()
    }
    if set(observed_paths) != set(expected_files):
        raise RuntimeError("Original Cube Lance file set drifted")
    observed_files: dict[str, Any] = {}
    for relative, entry in expected_files.items():
        child = observed_paths[relative]
        if child.is_symlink():
            raise RuntimeError("Original Cube Lance must not contain symlinks")
        digest, stat = _hash_stable_file(child)
        if (
            digest != str(entry["sha256"])
            or stat["size_bytes"] != int(entry["size_bytes"])
        ):
            raise RuntimeError(f"Original Cube Lance file drifted: {relative}")
        observed_files[relative] = {
            "sha256": digest,
            "size_bytes": stat["size_bytes"],
            "stable_stat": stat,
        }
    import lance

    dataset = lance.dataset(path)
    row_count = int(dataset.count_rows())
    if row_count != int(expected["row_count"]):
        raise RuntimeError("Original Cube Lance row count mismatch")
    action = dataset.schema.field("action").type
    if getattr(action, "list_size", None) != 5:
        raise RuntimeError("Original Cube Lance action column is not five-dimensional")
    return {
        "symbol": specification["source_symbol"],
        "path_recorded": False,
        "row_count": row_count,
        "action_dim": 5,
        "file_count": len(observed_files),
        "bytes": sum(value["size_bytes"] for value in observed_files.values()),
        "files": observed_files,
    }


def _verify_checkpoint(
    path: Path, specification: Mapping[str, Any], *, family: str
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing {family} initialization checkpoint: {path}")
    digest, stat = _hash_stable_file(path)
    if (
        digest != str(specification["sha256"])
        or stat["size_bytes"] != int(specification["bytes"])
    ):
        raise RuntimeError(f"Cube {family} initialization checkpoint drifted")
    return {
        "source_symbol": specification["source_symbol"],
        "path_recorded": False,
        "sha256": digest,
        "size_bytes": stat["size_bytes"],
        "stable_stat": stat,
    }


def _verify_stable_worldmodel_runtime(prereg: Mapping[str, Any]) -> dict[str, Any]:
    specification = prereg["runtime"]["stable_worldmodel"]
    repo = Path(str(specification["repo"])).expanduser()
    if not repo.is_absolute():
        repo = (ROOT / repo).resolve()
    if not repo.is_dir() or repo.is_symlink():
        raise FileNotFoundError(f"Pinned Stable-WorldModel repo missing: {repo}")
    environment = os.environ.copy()
    environment["SUDO_UID"] = str(repo.stat().st_uid)
    commit = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    ).stdout.strip()
    if commit != str(specification["expected_ref"]):
        raise RuntimeError("Stable-WorldModel runtime commit drifted")
    dirty = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "status", "--porcelain"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    ).stdout
    if dirty:
        raise RuntimeError("Pinned Stable-WorldModel runtime is dirty")
    files = {}
    for name, entry in specification["required_files"].items():
        path = repo / str(entry["path"])
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Stable-WorldModel runtime file missing: {path}")
        digest, stat = _hash_stable_file(path)
        if (
            digest != str(entry["sha256"])
            or stat["size_bytes"] != int(entry["size_bytes"])
        ):
            raise RuntimeError(f"Stable-WorldModel runtime file drifted: {name}")
        files[name] = {
            "path": str(entry["path"]),
            "sha256": digest,
            "size_bytes": stat["size_bytes"],
        }
    return {
        "path": str(repo),
        "commit": commit,
        "clean_worktree": True,
        "required_files": files,
    }


def _verify_checkpoint_runtime_compatibility(
    *,
    stable_repo: Path,
    lewm_checkpoint: Path,
    pldm_checkpoint: Path,
) -> dict[str, Any]:
    code = """
import json
from pathlib import Path
import sys
import torch
import scripts.run_cube_grasp_rule_h3_train as cube
cube._install_cube_action_dimensions()
cube._install_cube_diagnostic_name()
variant = "mixed_frozen_image_paired_future_fit_1p00"
cube.trainer.mixed.VARIANT_WEIGHTS[variant] = (
    "paired_future_fit", 1.0, "paired_future_fit"
)
models = {}
synthetic_evaluation = {
    "low_pixels": torch.zeros((1, 4, 3, 224, 224), dtype=torch.uint8),
    "high_pixels": torch.ones((1, 4, 3, 224, 224), dtype=torch.uint8),
    "action": torch.zeros((1, 4, 25), dtype=torch.float32),
    "low_states": torch.zeros((1, 4, 5), dtype=torch.float32),
    "high_states": torch.ones((1, 4, 5), dtype=torch.float32),
}
for family, checkpoint, recipe in (
    ("lewm", Path(sys.argv[1]), variant),
    ("pldm", Path(sys.argv[2]), "mixed_pldm_joint"),
):
    model, receipt = cube.trainer.mixed.load_model_for_variant(
        checkpoint, variant=recipe, device=torch.device("cpu")
    )
    forward_receipt = cube.trainer.pilot.evaluate_model(
        model,
        synthetic_evaluation,
        device=torch.device("cpu"),
        batch_size=1,
    )
    if (
        forward_receipt.get("pair_count") != 1
        or forward_receipt.get("decision_count") != 2
        or "physical_future_cube_height_gap_m" not in forward_receipt
    ):
        raise RuntimeError(f"Cube {family} synthetic CPU forward preflight failed")
    models[family] = {
        "strict_state_dict_load": receipt["strict_state_dict_load"],
        "loaded_model_config": receipt["loaded_model_config"],
        "action_input_dim": int(model.action_encoder.input_dim),
        "model_state_sha256": receipt["model_state_sha256"],
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "synthetic_cpu_forward_preflight": True,
    }
loss_compatibility = cube._PINNED_LOSS_COMPATIBILITY
if loss_compatibility is None:
    raise RuntimeError("Cube pinned loss compatibility was not installed")
conditional = cube.trainer.mixed.ConditionalSIGReg(
    knots=17,
    num_proj=1024,
    randomize_pair_orientation=True,
    include_unpaired=False,
    complete_haar_population=False,
).to(torch.device("cpu"))
loss_compatibility = {
    **loss_compatibility,
    "shared_engine_constructor_preflight": True,
    "constructed_class": type(conditional).__name__,
    "include_unpaired": conditional.contextworld_include_unpaired,
    "complete_haar_population": (
        conditional.contextworld_complete_haar_population
    ),
}
print(
    "CUBE_COMPAT="
    + json.dumps(
        {"models": models, "pinned_loss_compatibility": loss_compatibility},
        sort_keys=True,
    )
)
"""
    environment = os.environ.copy()
    environment.update(
        {
            "CONTEXTWORLD_STABLE_WORLDMODEL_REPO": str(stable_repo),
            "MPLCONFIGDIR": "/tmp/contextworld-cube-v4r1-freeze-mpl",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(lewm_checkpoint),
            str(pldm_checkpoint),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    marker = "CUBE_COMPAT="
    matches = [
        line[len(marker) :]
        for line in completed.stdout.splitlines()
        if line.startswith(marker)
    ]
    if len(matches) != 1:
        raise RuntimeError("Cube checkpoint/runtime compatibility receipt is missing")
    result = json.loads(matches[0])
    models = result.get("models", {})
    loss_compatibility = result.get("pinned_loss_compatibility", {})
    if (
        set(models) != {"lewm", "pldm"}
        or any(
            row.get("strict_state_dict_load") is not True
            or int(row.get("action_input_dim", -1)) != 25
            or row.get("loaded_model_config") != family
            or not isinstance(row.get("model_state_sha256"), str)
            or len(row["model_state_sha256"]) != 64
            or int(row.get("parameter_count", 0)) <= 0
            or row.get("synthetic_cpu_forward_preflight") is not True
            for family, row in models.items()
        )
        or loss_compatibility
        != {
            "conditional_sigreg_constructor_adapter_installed": True,
            "conditional_sigreg_missing_keywords": [
                "include_unpaired",
                "complete_haar_population",
            ],
            "conditional_sigreg_false_only": True,
            "unavailable_eager_diagnostic_sentinels": [
                "DynamicsResponseSIGReg",
                "GroupBalancedSIGReg",
                "ScaleCalibratedConditionalSIGReg",
            ],
            "shared_engine_constructor_preflight": True,
            "constructed_class": "PinnedConditionalSIGReg",
            "include_unpaired": False,
            "complete_haar_population": False,
        }
    ):
        raise RuntimeError(
            "Cube checkpoint/runtime or pinned-loss compatibility contract failed"
        )
    return result


def _validate_data_decision(
    decision: Mapping[str, Any], *, prereg: Mapping[str, Any]
) -> None:
    if (
        decision.get("status") != "passed_development"
        or decision.get("scope") != "data_readiness_only_not_reference_model_or_public"
        or decision.get("protocol_id") != prereg["data"]["protocol"]
    ):
        raise RuntimeError("Cube v4r1 data-readiness decision is not eligible")
    claims = _mapping(decision.get("claims"), field="data_decision.claims")
    if (
        claims.get("data_readiness_passed") is not True
        or claims.get("positive_reference_model_claim_allowed") is not False
        or claims.get("public_test_claim_allowed") is not False
        or claims.get("release_claim_allowed") is not False
    ):
        raise RuntimeError("Cube data-readiness claim boundary drifted")
    phase = _mapping(
        decision.get("reference_model_phase"),
        field="data_decision.reference_model_phase",
    )
    if (
        phase.get("trainer_invoked") is not False
        or int(phase.get("optimizer_steps_run", -1)) != 0
        or phase.get("checkpoints_created") is not False
        or phase.get("lewm_or_pldm_development_scoring_run") is not False
        or phase.get("public_test_model_scoring_opened") is not False
    ):
        raise RuntimeError("Cube model phase was not clean before training freeze")
    public = _mapping(decision.get("public_test"), field="data_decision.public_test")
    if public != {
        "access_status": "closed_not_read_not_scored",
        "generated": False,
        "hashed": False,
        "opened": False,
        "read": False,
        "scored": False,
    }:
        raise RuntimeError("Cube Public Test was not closed at data decision")


def freeze(
    *,
    prereg_path: Path,
    source_h5: Path,
    original_lance: Path,
    lewm_checkpoint: Path,
    pldm_checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite freeze receipt: {output}")
    prereg = load_cube_reference_training_prereg(
        prereg_path,
        require_freeze=False,
        repo_root=ROOT,
    )
    expected_inputs = {
        "source_h5": resolve_cube_reference_training_input(
            prereg, "original_h5", repo_root=ROOT
        ),
        "original_lance": resolve_cube_reference_training_input(
            prereg, "original_lance", repo_root=ROOT
        ),
        "lewm_checkpoint": resolve_cube_reference_initial_checkpoint(
            prereg, "lewm", repo_root=ROOT
        ),
        "pldm_checkpoint": resolve_cube_reference_initial_checkpoint(
            prereg, "pldm", repo_root=ROOT
        ),
    }
    supplied_inputs = {
        "source_h5": _absolute_without_resolving_symlinks(source_h5),
        "original_lance": _absolute_without_resolving_symlinks(original_lance),
        "lewm_checkpoint": _absolute_without_resolving_symlinks(lewm_checkpoint),
        "pldm_checkpoint": _absolute_without_resolving_symlinks(pldm_checkpoint),
    }
    if supplied_inputs != expected_inputs:
        raise RuntimeError(
            "Cube v3 freezer inputs do not match the shared-resolver aliases"
        )
    planned_output = _resolve(
        str(prereg["planned_artifacts"]["freeze_receipt"]), repo_root=ROOT
    )
    if output.resolve() != planned_output:
        raise RuntimeError("Freeze receipt output does not match preregistration")
    training_root = _resolve(
        str(prereg["planned_artifacts"]["training_root"]), repo_root=ROOT
    )
    score_root = _resolve(
        str(prereg["planned_artifacts"]["development_score_root"]),
        repo_root=ROOT,
    )
    if training_root.exists() or score_root.exists():
        raise RuntimeError("Cube v4r1 model output exists before training freeze")

    recovery = cube_reference_infrastructure_recovery_identity(
        prereg, repo_root=ROOT
    )

    identity = {
        name: _verify_declared_file(entry, repo_root=ROOT, name=f"identity.{name}")
        for name, entry in prereg["identity"].items()
    }
    evidence = {
        name: _verify_declared_file(
            entry, repo_root=ROOT, name=f"data.artifacts.{name}"
        )
        for name, entry in prereg["data"]["artifacts"].items()
    }
    manifest = json.loads(
        _resolve(evidence["manifest"]["path"], repo_root=ROOT).read_text(
            encoding="utf-8"
        )
    )
    if (
        manifest.get("active_splits") != list(AUTHORIZED_SPLITS)
        or manifest.get("build_passed") is not True
        or manifest.get("public_test_generated") is not False
        or manifest.get("public_test_opened") is not False
        or set(manifest.get("splits", {})) & {"validation", "public", "test"}
    ):
        raise RuntimeError("Cube v4r1 manifest does not preserve Public closure")
    data_root = _resolve(
        str(prereg["data"]["artifact_tree"]["root"]), repo_root=ROOT
    )
    if (data_root / "validation.lance").exists():
        raise RuntimeError("Cube v4r1 data root unexpectedly contains Public validation")
    decision_path = _resolve(
        evidence["data_readiness_decision"]["path"], repo_root=ROOT
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    _validate_data_decision(decision, prereg=prereg)
    data_tree = cube_reference_data_tree_identity(prereg, repo_root=ROOT)

    upstream = prereg["training"]["upstream"]
    source_receipt = _verify_source_h5(
        _absolute_without_resolving_symlinks(source_h5),
        upstream["original_h5"],
    )
    lance_receipt = _verify_original_lance(
        _absolute_without_resolving_symlinks(original_lance),
        upstream["original_lance"],
    )
    checkpoints = prereg["training"]["reference_matrix"]["initial_checkpoints"]
    checkpoint_receipts = {
        "lewm": _verify_checkpoint(
            _absolute_without_resolving_symlinks(lewm_checkpoint),
            checkpoints["lewm"],
            family="lewm",
        ),
        "pldm": _verify_checkpoint(
            _absolute_without_resolving_symlinks(pldm_checkpoint),
            checkpoints["pldm"],
            family="pldm",
        ),
    }
    stable_runtime = _verify_stable_worldmodel_runtime(prereg)
    checkpoint_runtime = _verify_checkpoint_runtime_compatibility(
        stable_repo=Path(stable_runtime["path"]),
        lewm_checkpoint=_absolute_without_resolving_symlinks(lewm_checkpoint),
        pldm_checkpoint=_absolute_without_resolving_symlinks(pldm_checkpoint),
    )

    prereg_hash, prereg_stat = _hash_stable_file(prereg_path)
    matrix = prereg["training"]["reference_matrix"]
    jobs = [
        {"model": model, "seed": int(seed)}
        for model in ("lewm", "pldm")
        for seed in matrix["training_seeds"]
    ]
    receipt = {
        "schema_version": 1,
        "preregistration_id": CUBE_REFERENCE_TRAINING_ID,
        "protocol_id": CUBE_REFERENCE_TRAINING_PROTOCOL,
        "status": FREEZE_STATUS,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": True,
        "training_and_development_scoring_authorized": True,
        "preregistration": {
            "path": portable_contextworld_path(prereg_path, repo_root=ROOT),
            "sha256": prereg_hash,
            "size_bytes": prereg_stat["size_bytes"],
        },
        "identity": identity,
        "infrastructure_recovery": recovery,
        "data_evidence": evidence,
        "data_tree": data_tree,
        "inputs": {
            "original_h5": source_receipt,
            "original_lance": lance_receipt,
            "initial_checkpoints": checkpoint_receipts,
        },
        "runtime": {
            "stable_worldmodel": stable_runtime,
            "checkpoint_compatibility": checkpoint_runtime,
        },
        "authorization": {
            "jobs": jobs,
            "optimizer_steps_per_job": int(matrix["common"]["optimizer_steps"]),
            "total_optimizer_steps_authorized": len(jobs)
            * int(matrix["common"]["optimizer_steps"]),
            "authorized_splits": list(AUTHORIZED_SPLITS),
            "development_scoring_authorized": True,
            "public_model_scoring_authorized": False,
            "original_task_retention_authorized": False,
            "recipe_or_threshold_changes_authorized": False,
        },
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "generated": False,
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
            "validation_lance_access_allowed": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_CUBE_REFERENCE_TRAINING_PREREG)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--original-lance", type=Path, required=True)
    parser.add_argument("--lewm-checkpoint", type=Path, required=True)
    parser.add_argument("--pldm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = freeze(
        prereg_path=args.prereg.expanduser().resolve(),
        source_h5=_absolute_without_resolving_symlinks(args.source_h5),
        original_lance=_absolute_without_resolving_symlinks(args.original_lance),
        lewm_checkpoint=_absolute_without_resolving_symlinks(args.lewm_checkpoint),
        pldm_checkpoint=_absolute_without_resolving_symlinks(args.pldm_checkpoint),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(args.output),
                "preregistration_sha256": receipt["preregistration"]["sha256"],
                "authorized_jobs": len(receipt["authorization"]["jobs"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
