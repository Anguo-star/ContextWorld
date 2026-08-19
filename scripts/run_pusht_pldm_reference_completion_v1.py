#!/usr/bin/env python3
"""Development-only PLDM completion runner for current PushT release data.

This intentionally has no Public-Test or CEM command.  It materializes a
temporary training overlay from an immutable source release, trains one fixed
PLDM pilot, and invokes only the explicitly separate ``eval-development`` and
``score-development`` paths.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import types
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for value in (ROOT, SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))


PUBLIC_CLOSED = {
    "access_status": "closed_not_read_not_scored",
    "opened": False,
    "read": False,
    "hashed": False,
    "scored": False,
}
EXPECTED_RECIPES = {
    "contact_friction": "mixed_pldm_joint",
    "motion_damping": "mixed_pldm_identifiable_future_joint",
}
EVALUATOR_MODULES = {
    "contact_friction": "contextworld.benchmarks.contact_friction_icl_cli",
    "motion_damping": "contextworld.benchmarks.motion_damping_icl_cli",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _path_size(path: Path) -> int | None:
    if path.is_file():
        return int(path.stat().st_size)
    if path.is_dir():
        return sum(int(child.stat().st_size) for child in path.rglob("*") if child.is_file())
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def load_completion(path: Path | str) -> tuple[Path, dict[str, Any]]:
    config_path = _resolve(path)
    completion = _load_yaml(config_path)
    if completion.get("schema_version") != 1:
        raise ValueError("Completion config must use schema_version=1")
    if completion.get("status") != "preregistered_development_only":
        raise ValueError("Completion config is not Development-only")
    component = completion.get("scope", {}).get("component")
    if component not in EXPECTED_RECIPES:
        raise ValueError(f"Unsupported completion component: {component!r}")
    if completion.get("training", {}).get("model_family") != "PLDM":
        raise ValueError("Completion is restricted to PLDM")
    if completion["training"].get("recipe") != EXPECTED_RECIPES[component]:
        raise ValueError("Completion recipe does not match the registered PLDM recipe")
    seeds = tuple(int(value) for value in completion["training"].get("seeds", ()))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Completion must register exactly three distinct seeds")
    if int(completion["training"].get("pilot_seed", -1)) != seeds[0]:
        raise ValueError("The first registered seed must be the fixed pilot seed")
    public_policy = completion["source_release"]["data"]["public_test"]
    if {key: public_policy.get(key) for key in PUBLIC_CLOSED} != PUBLIC_CLOSED:
        raise ValueError("Completion config must keep Public Test closed")
    return config_path, completion


def load_source_release(completion: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(completion["source_release"]["config_path"])
    release = _load_yaml(path)
    if release.get("release_id") != completion["source_release"]["release_id"]:
        raise ValueError("Source release ID differs from preregistration")
    return path, release


def _record(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    **details: Any,
) -> None:
    checks[name] = {"passed": bool(passed), **details}


def _worktree_head(worktree: Path) -> tuple[str | None, str | None]:
    """Read detached-worktree HEAD without changing Git safe-directory state."""

    pointer = worktree / ".git"
    if not pointer.is_file():
        return None, None
    text = pointer.read_text(encoding="utf-8").strip()
    prefix = "gitdir: "
    if not text.startswith(prefix):
        return None, None
    gitdir = Path(text[len(prefix) :]).expanduser()
    head = gitdir / "HEAD"
    return (head.read_text(encoding="utf-8").strip() if head.is_file() else None), str(gitdir)


def _pinned_stable_worldmodel(
    completion: dict[str, Any],
    checks: dict[str, dict[str, Any]] | None = None,
) -> Path:
    specification = completion.get("stable_worldmodel", {})
    worktree = Path(specification.get("worktree", "")).expanduser().resolve()
    expected_commit = str(specification.get("commit", ""))
    observed_commit, gitdir = _worktree_head(worktree)
    passed = bool(
        worktree.is_dir()
        and observed_commit == expected_commit
        and not specification.get("require_clean_worktree", False)
    )
    details = {
        "worktree": str(worktree),
        "expected_commit": expected_commit,
        "observed_commit": observed_commit,
        "gitdir": gitdir,
        "cleanliness_policy": "exact_pinned_runtime_file_hashes_only",
    }
    if checks is not None:
        _record(checks, "pinned_stable_worldmodel_worktree", passed, **details)
    elif not passed:
        raise RuntimeError(f"Pinned Stable-WorldModel identity failed: {details}")
    for index, file_specification in enumerate(specification.get("files", ())):
        path = worktree / str(file_specification["relative_path"])
        observed = _sha256(path) if path.is_file() else None
        file_passed = observed == file_specification["sha256"]
        if checks is not None:
            _record(
                checks,
                f"pinned_stable_worldmodel_file_{index}",
                file_passed,
                path=str(path),
                expected_sha256=file_specification["sha256"],
                observed_sha256=observed,
            )
        elif not file_passed:
            raise RuntimeError(
                "Pinned Stable-WorldModel file identity failed: "
                f"{path}"
            )
    if checks is None and not passed:
        raise AssertionError("unreachable")
    return worktree


def _load_pinned_module(name: str, path: Path, worktree: Path) -> Any:
    """Load a source-identical ContextWorld helper with a pinned WM root.

    The shared historical trainer hard-codes a sibling checkout.  This loader
    changes that one import-root assignment only in memory, leaving the frozen
    source file and dirty sibling worktree untouched.
    """

    source = path.read_text(encoding="utf-8")
    replacements = (
        'STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / "stable-worldmodel"',
        'STABLE_WORLD_MODEL_ROOT = ROOT.parent / "stable-worldmodel"',
    )
    replacement = f"STABLE_WORLD_MODEL_ROOT = Path({str(worktree)!r}).resolve()"
    replaced = False
    for candidate in replacements:
        if candidate in source:
            source = source.replace(candidate, replacement, 1)
            replaced = True
            break
    if not replaced:
        raise RuntimeError(f"Could not bind pinned worktree in {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def _load_pinned_training_stack(worktree: Path) -> tuple[Any, Any, Any]:
    """Load pilot/mixed/trainer modules with no dependency on the dirty sibling."""

    dirty_sibling = str((ROOT.parent / "stable-worldmodel").resolve())
    sys.path[:] = [
        str(worktree),
        str(ROOT),
        str(SCRIPTS),
        *[value for value in sys.path if Path(value or ".").resolve() != Path(dirty_sibling)],
    ]
    for name in list(sys.modules):
        if name in {
            "run_pusht_hidden_actuation_pilot",
            "run_pusht_hidden_actuation_mixed",
            "run_pusht_contact_friction_h3_train",
        } or name.startswith("stable_worldmodel") or name.startswith("stable_pretraining"):
            sys.modules.pop(name, None)
    pilot = _load_pinned_module(
        "run_pusht_hidden_actuation_pilot",
        SCRIPTS / "run_pusht_hidden_actuation_pilot.py",
        worktree,
    )
    mixed = _load_pinned_module(
        "run_pusht_hidden_actuation_mixed",
        SCRIPTS / "run_pusht_hidden_actuation_mixed.py",
        worktree,
    )
    trainer = _load_pinned_module(
        "run_pusht_contact_friction_h3_train",
        SCRIPTS / "run_pusht_contact_friction_h3_train.py",
        worktree,
    )
    return pilot, mixed, trainer


def _strict_pldm_load(checkpoint: Path, worktree: Path) -> dict[str, Any]:
    """Instantiate the pinned PLDM architecture and strict-load its state."""

    for name in list(sys.modules):
        if name.startswith("stable_worldmodel"):
            sys.modules.pop(name, None)
    sys.path.insert(0, str(worktree))
    import hydra
    import torch
    from omegaconf import OmegaConf, open_dict

    configuration = OmegaConf.load(worktree / "scripts/train/config/pldm.yaml")
    with open_dict(configuration):
        configuration.model.action_encoder.input_dim = 10
    model = hydra.utils.instantiate(configuration.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload.get("state_dict", payload)
    state = {
        name[len("model.") :]: value
        for name, value in source.items()
        if name.startswith("model.")
    }
    if not state:
        state = dict(source)
    model.load_state_dict(state, strict=True)
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return {
        "path": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "model_state_sha256": digest.hexdigest(),
        "loaded_model_config": "pldm",
        "strict_state_dict_load": True,
        "stable_worldmodel_worktree": str(worktree),
    }


def preflight_payload(config_path: Path | str) -> dict[str, Any]:
    """Validate only current Training/Development inputs and PLDM strict load.

    This routine intentionally never resolves, stats, walks, hashes, decodes,
    or opens the source release's ``validation.lance`` path.
    """

    completion_path, completion = load_completion(config_path)
    release_path, release = load_source_release(completion)
    component = completion["scope"]["component"]
    expected_data = completion["source_release"]["data"]
    checks: dict[str, dict[str, Any]] = {}

    observed_release_config_sha256 = _sha256(release_path)
    _record(
        checks,
        "source_release_identity",
        release.get("release_id") == completion["source_release"]["release_id"]
        and observed_release_config_sha256
        == completion["source_release"].get(
            "release_config_sha256_observed_at_preregistration"
        ),
        observed_release_id=release.get("release_id"),
        expected_release_id=completion["source_release"]["release_id"],
        observed_config_sha256=observed_release_config_sha256,
        preregistration_observed_config_sha256=completion["source_release"].get(
            "release_config_sha256_observed_at_preregistration"
        ),
    )
    data = release.get("data", {})
    evaluation = release.get("evaluation", {})
    development = evaluation.get("development", {})
    _record(
        checks,
        "registered_current_data_identity",
        data.get("protocol") == expected_data["protocol"]
        and data.get("artifact_tree", {}).get("root") == expected_data["root"]
        and data.get("manifest_sha256") == expected_data["manifest_sha256"]
        and data.get("lance_tables", {}).get("train")
        == expected_data["train"]["lance_table"]
        and int(data.get("pair_counts", {}).get("train", -1))
        == int(expected_data["train"]["pair_count"])
        and development.get("split") == expected_data["development"]["split"]
        and development.get("lance_table")
        == expected_data["development"]["lance_table"]
        and development.get("lance_table_sha256")
        == expected_data["development"]["lance_table_sha256"]
        and int(development.get("pair_count", -1))
        == int(expected_data["development"]["pair_count"]),
        observed_protocol=data.get("protocol"),
        observed_root=data.get("artifact_tree", {}).get("root"),
        observed_manifest_sha256=data.get("manifest_sha256"),
        observed_development_table=development.get("lance_table"),
        observed_development_table_sha256=development.get("lance_table_sha256"),
    )
    source_public = development.get("public_test")
    _record(
        checks,
        "public_test_closed_without_access",
        {key: source_public.get(key) for key in PUBLIC_CLOSED}
        == PUBLIC_CLOSED
        and evaluation.get("lance_table")
        == expected_data["public_test"]["lance_table_name_only"],
        source_release_development_public_policy=source_public,
        public_table_name_only=evaluation.get("lance_table"),
        public_table_read=False,
        public_table_hashed=False,
        public_table_decoded=False,
    )

    data_root = _resolve(expected_data["root"])
    manifest_path = data_root / "manifest.json"
    development_path = data_root / expected_data["development"]["lance_table"]
    observed_manifest = _sha256(manifest_path) if manifest_path.is_file() else None
    observed_development = (
        _directory_sha256(development_path) if development_path.is_dir() else None
    )
    _record(
        checks,
        "current_manifest_sha256",
        observed_manifest == expected_data["manifest_sha256"],
        path=str(manifest_path),
        expected_sha256=expected_data["manifest_sha256"],
        observed_sha256=observed_manifest,
    )
    _record(
        checks,
        "current_development_table_sha256",
        observed_development == expected_data["development"]["lance_table_sha256"],
        path=str(development_path),
        expected_sha256=expected_data["development"]["lance_table_sha256"],
        observed_sha256=observed_development,
    )

    training = completion["training"]
    common = release.get("training", {}).get("reference_matrix", {}).get("common", {})
    expected_common = {
        "optimizer_steps": training["fixed_optimizer_step"],
        "fixed_checkpoint_step": training["fixed_optimizer_step"],
        "batch_size": training["fixed_batch_size"],
        "original_pusht_samples_per_batch": training[
            "original_pusht_samples_per_batch"
        ],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "gradient_clip_norm": training["gradient_clip_norm"],
        "checkpoint_selection": training["checkpoint_selection"],
        "loader_validation_monitor_steps": training["development_monitor_steps"],
    }
    component_batch_key = (
        "contact_friction_samples_per_batch"
        if component == "contact_friction"
        else "motion_damping_samples_per_batch"
    )
    expected_common[component_batch_key] = training[component_batch_key]
    _record(
        checks,
        "fixed_recipe_and_training_budget",
        all(common.get(key) == value for key, value in expected_common.items()),
        expected=expected_common,
        observed={key: common.get(key) for key in expected_common},
        recipe=training["recipe"],
        pilot_seed=training["pilot_seed"],
    )

    for input_name, specification in training["upstream"].items():
        input_path = _resolve(specification["path"])
        observed_size = _path_size(input_path)
        _record(
            checks,
            f"upstream_{input_name}_available",
            input_path.exists()
            and observed_size == int(specification["size_bytes"]),
            path=str(input_path),
            expected_size_bytes=int(specification["size_bytes"]),
            observed_size_bytes=observed_size,
        )

    initialization = completion["initialization"]
    checkpoint = _resolve(initialization["path"])
    source_config = _resolve(initialization["source_training_config"])
    observed_checkpoint = _sha256(checkpoint) if checkpoint.is_file() else None
    observed_source_config = _sha256(source_config) if source_config.is_file() else None
    _record(
        checks,
        "original_pldm_initialization_identity",
        checkpoint.is_file()
        and checkpoint.stat().st_size == int(initialization["size_bytes"])
        and observed_checkpoint == initialization["sha256"]
        and observed_source_config == initialization["source_training_config_sha256"],
        checkpoint=str(checkpoint),
        observed_checkpoint_sha256=observed_checkpoint,
        expected_checkpoint_sha256=initialization["sha256"],
        observed_source_config_sha256=observed_source_config,
        expected_source_config_sha256=initialization["source_training_config_sha256"],
    )
    for index, specification in enumerate(completion["runtime_identity"]):
        path = _resolve(specification["path"])
        observed = _sha256(path) if path.is_file() else None
        _record(
            checks,
            f"runtime_identity_{index}",
            observed == specification["sha256"],
            path=str(path),
            expected_sha256=specification["sha256"],
            observed_sha256=observed,
        )

    pinned_worktree = _pinned_stable_worldmodel(completion, checks)

    strict_receipt: dict[str, Any]
    try:
        strict_receipt = _strict_pldm_load(checkpoint, pinned_worktree)
        strict_passed = bool(
            strict_receipt.get("strict_state_dict_load") is True
            and strict_receipt.get("loaded_model_config")
            == initialization["required_model_config"]
            and strict_receipt.get("model_state_sha256")
            == initialization["expected_loaded_model_state_sha256"]
        )
    except Exception as error:  # receipt is evidence, not a silent fallback
        strict_receipt = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        strict_passed = False
    _record(
        checks,
        "original_pldm_strict_state_dict_load",
        strict_passed,
        required=bool(initialization["strict_state_dict_load_required"]),
        receipt=strict_receipt,
    )
    return {
        "schema_version": 1,
        "status": "passed" if all(row["passed"] for row in checks.values()) else "failed",
        "completion": {
            "id": completion["completion_id"],
            "path": str(completion_path),
            "sha256": _sha256(completion_path),
            "component": component,
        },
        "source_release": {
            "path": str(release_path),
            "sha256": _sha256(release_path),
            "release_id": release.get("release_id"),
        },
        "scope": "Development_only; Public Test was not accessed",
        "checks": checks,
        "passed": all(row["passed"] for row in checks.values()),
    }


def build_training_overlay(
    completion: dict[str, Any], source_release: dict[str, Any]
) -> dict[str, Any]:
    """Create a disposable training-only overlay without touching the release."""

    overlay = copy.deepcopy(source_release)
    training = completion["training"]
    initialization = completion["initialization"]
    matrix = overlay["training"]["reference_matrix"]
    common = matrix["common"]
    matrix.update(
        {
            "status": "planned_not_executed",
            "training_seeds": list(training["seeds"]),
            "completed_development_seeds": [],
            "remaining_seeds_run": False,
            "public_model_scoring_opened": False,
            "reported_endpoint": {
                "model_family": "PLDM",
                "recipe": training["recipe"],
                "training_seed": int(training["pilot_seed"]),
                "optimizer_step": int(training["fixed_optimizer_step"]),
            },
        }
    )
    common.update(
        {
            "optimizer_steps": int(training["fixed_optimizer_step"]),
            "fixed_checkpoint_step": int(training["fixed_optimizer_step"]),
            "batch_size": int(training["fixed_batch_size"]),
            "original_pusht_samples_per_batch": int(
                training["original_pusht_samples_per_batch"]
            ),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "gradient_clip_norm": float(training["gradient_clip_norm"]),
            "checkpoint_selection": training["checkpoint_selection"],
            "loader_validation_monitor_steps": list(
                training["development_monitor_steps"]
            ),
        }
    )
    component = completion["scope"]["component"]
    component_batch_key = (
        "contact_friction_samples_per_batch"
        if component == "contact_friction"
        else "motion_damping_samples_per_batch"
    )
    common[component_batch_key] = int(training[component_batch_key])
    overlay["training"]["upstream"] = {
        "original_h5": copy.deepcopy(training["upstream"]["original_h5"]),
        "original_lance": copy.deepcopy(training["upstream"]["original_lance"]),
        "initialization": {
            "path": initialization["path"],
            "sha256": initialization["sha256"],
            "bytes": int(initialization["size_bytes"]),
            "role": "canonical_original_pusht_pldm_initialization",
        },
    }
    return overlay


def _configure_component_trainer(component: str, worktree: Path) -> Any:
    """Use the shared trainer while preserving a Development-only MD path."""

    _, _, trainer = _load_pinned_training_stack(worktree)

    if component == "contact_friction":
        return trainer
    from contextworld.benchmarks.motion_damping_icl_data import (
        DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
        _read_lance_pairs,
        load_motion_damping_icl_release,
    )
    import run_pusht_motion_damping_h3_train as motion

    # Do not call audit_motion_damping_icl_release here: its non-full audit
    # counts every split, including Public Test. The completion preflight and
    # subsequent Development evaluator deliberately bind only train/dev data.
    trainer.DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = (
        DEFAULT_MOTION_DAMPING_RELEASE_CONFIG
    )
    trainer.load_contact_friction_icl_release = load_motion_damping_icl_release
    trainer._read_lance_pairs = _read_lance_pairs
    trainer.CAPABILITY_SLUG = "motion_damping"
    trainer.CAPABILITY_DISPLAY = "motion-damping"
    trainer.HIDDEN_FIELD = "hidden_motion_damping"
    trainer.TRAINER_DESCRIPTION = "Development-only motion-damping PLDM completion"
    trainer.mixed.VARIANT_WEIGHTS[motion.PLDM_REFERENCE_VARIANT] = (
        "pldm",
        1.0,
        "identifiable_future_only",
    )
    trainer.DIAGNOSTIC_VARIANTS["pldm"].add(motion.PLDM_REFERENCE_VARIANT)
    trainer.MODEL_VARIANTS = {
        "lewm": motion.LEWM_REFERENCE_VARIANT,
        "pldm": motion.PLDM_REFERENCE_VARIANT,
    }
    return trainer


def internal_train(
    completion_path: Path | str,
    component: str,
    trainer_args: list[str],
) -> None:
    _, completion = load_completion(completion_path)
    if completion["scope"]["component"] != component:
        raise ValueError("Internal trainer component disagrees with completion config")
    worktree = _pinned_stable_worldmodel(completion)
    trainer = _configure_component_trainer(component, worktree)
    saved_argv = sys.argv
    try:
        sys.argv = [str(ROOT / "scripts/run_pusht_contact_friction_h3_train.py"), *trainer_args]
        trainer.main()
    finally:
        sys.argv = saved_argv


def _run(command: list[str], *, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", "/tmp/contextworld-pldm-mpl")
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
    return int(process.returncode)


def _load_preflight(path: Path, completion_path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("passed"):
        raise RuntimeError("Pilot requires a passed preflight")
    if payload.get("completion", {}).get("sha256") != _sha256(completion_path):
        raise RuntimeError("Preflight belongs to a different completion config")
    return payload


def pilot(
    *,
    completion_path: Path | str,
    preflight_path: Path | str,
    output: Path | str,
    device: str,
    num_workers: int,
    eval_batch_size: int,
    training_only: bool = False,
) -> dict[str, Any]:
    completion_path, completion = load_completion(completion_path)
    release_path, source_release = load_source_release(completion)
    component = completion["scope"]["component"]
    preflight_path = _resolve(preflight_path)
    _load_preflight(preflight_path, completion_path)
    target = _resolve(output)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {target}")
    recheck = preflight_payload(completion_path)
    if not recheck["passed"]:
        raise RuntimeError("Current preflight recheck failed; training is not allowed")
    target.mkdir(parents=True)
    _write_json(target / "preflight_recheck.json", recheck)
    training = completion["training"]
    recipe = str(training["recipe"])
    seed = int(training["pilot_seed"])
    training_output = target / "training"
    with tempfile.TemporaryDirectory(prefix="contextworld-pldm-completion-") as directory:
        overlay_path = Path(directory) / "training_overlay.yaml"
        overlay = build_training_overlay(completion, source_release)
        overlay_path.write_text(
            yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8"
        )
        train_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "internal-train",
            "--component",
            component,
            "--completion-config",
            str(completion_path),
            "--",
            "--release-config",
            str(overlay_path),
            "--model",
            "pldm",
            "--seed",
            str(seed),
            "--output",
            str(training_output),
            "--device",
            device,
            "--num-workers",
            str(num_workers),
            "--eval-batch-size",
            str(eval_batch_size),
        ]
        training_exit = _run(train_command, log=target / "training.log")
    receipt = {
        "schema_version": 1,
        "completion_id": completion["completion_id"],
        "component": component,
        "source_release": {"path": str(release_path), "sha256": _sha256(release_path)},
        "completion_config": {"path": str(completion_path), "sha256": _sha256(completion_path)},
        "preflight": str(preflight_path),
        "training_command": train_command,
        "training_exit_code": training_exit,
        "public_test": {**PUBLIC_CLOSED, "accessed_by_this_runner": False},
        "cem": {"executed": False, "authorized": False},
    }
    _write_json(target / "pilot_execution_receipt.json", receipt)
    if training_exit:
        decision = {
            **receipt,
            "status": "training_execution_failed_before_development_gate",
            "public_test": {**PUBLIC_CLOSED, "accessed_by_this_runner": False},
            "next_action": "repair_execution_only; Public and CEM remain closed",
        }
        _write_json(target / "development_decision.json", decision)
        raise RuntimeError(f"Pilot training failed; see {target / 'training.log'}")

    checkpoint = training_output / f"{recipe}_step{training['fixed_optimizer_step']}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing fixed-step checkpoint: {checkpoint}")
    if training_only:
        decision = {
            **receipt,
            "status": "trained_pending_evaluation_binding",
            "fixed_recipe": recipe,
            "fixed_optimizer_step": int(training["fixed_optimizer_step"]),
            "pilot_seed": seed,
            "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            "public_test": {**PUBLIC_CLOSED, "accessed_by_this_runner": False},
            "cem": {"executed": False, "authorized": False},
            "next_stage": {
                "evaluation_binding_required": True,
                "reason": completion.get("evaluation_binding", {}).get("status"),
                "public_test_authorized": False,
                "cem_authorized": False,
            },
        }
        _write_json(target / "development_decision.json", decision)
        return decision
    evaluation_path = target / "development_evaluation.json"
    pinned_worktree = _pinned_stable_worldmodel(completion)
    pinned_commit = completion["stable_worldmodel"]["commit"]
    evaluation_command = [
        sys.executable,
        "-m",
        EVALUATOR_MODULES[component],
        "--release-config",
        str(release_path),
        "eval-development",
        "--checkpoint",
        str(checkpoint),
        "--adapter",
        "pldm",
        "--model-name",
        f"{component}_pldm_completion_seed{seed}",
        "--training-recipe",
        recipe,
        "--training-seed",
        str(seed),
        "--device",
        device,
        "--batch-size",
        str(eval_batch_size),
        "--stablewm-repo",
        str(pinned_worktree),
        "--stablewm-ref",
        str(pinned_commit),
        "--output",
        str(evaluation_path),
    ]
    evaluation_exit = _run(evaluation_command, log=target / "development_evaluation.log")
    if evaluation_exit:
        raise RuntimeError(f"Development evaluation failed; see {target / 'development_evaluation.log'}")
    rescore_path = target / "development_rescore.json"
    rescore_command = [
        sys.executable,
        "-m",
        EVALUATOR_MODULES[component],
        "--release-config",
        str(release_path),
        "score-development",
        "--input",
        str(evaluation_path),
        "--output",
        str(rescore_path),
    ]
    rescore_exit = _run(rescore_command, log=target / "development_rescore.log")
    if rescore_exit:
        raise RuntimeError(f"Development rescore failed; see {target / 'development_rescore.log'}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    rescore = json.loads(rescore_path.read_text(encoding="utf-8"))
    gate_match = (
        evaluation.get("gate") == rescore.get("gate")
        and evaluation.get("metrics") == rescore.get("metrics")
    )
    passed = bool(evaluation.get("gate", {}).get("passed") and gate_match)
    status = (
        "passed_development_recipe_frozen"
        if passed
        else "failed_development_current_protocol"
    )
    decision = {
        **receipt,
        "status": status,
        "fixed_recipe": recipe,
        "fixed_optimizer_step": int(training["fixed_optimizer_step"]),
        "pilot_seed": seed,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "development": {
            "evaluation": str(evaluation_path),
            "rescore": str(rescore_path),
            "gate": evaluation.get("gate"),
            "metrics": evaluation.get("metrics"),
            "independent_rescore_matches": gate_match,
        },
        "public_test": {**PUBLIC_CLOSED, "accessed_by_this_runner": False},
        "cem": {"executed": False, "authorized": False},
        "next_stage": (
            {
                "recipe_frozen": True,
                "remaining_development_seeds_authorized": list(training["seeds"][1:]),
                "public_test_authorized": False,
                "cem_authorized": False,
                "condition": "run the two remaining frozen-recipe Development seeds first",
            }
            if passed
            else {
                "formal_failure_frozen": True,
                "remaining_development_seeds_authorized": False,
                "public_test_authorized": False,
                "cem_authorized": False,
            }
        ),
    }
    _write_json(target / "development_decision.json", decision)
    if passed:
        _write_json(target / "recipe_freeze.json", decision)
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--completion-config", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    pilot_parser = commands.add_parser("pilot")
    pilot_parser.add_argument("--completion-config", type=Path, required=True)
    pilot_parser.add_argument("--preflight", type=Path, required=True)
    pilot_parser.add_argument("--output", type=Path, required=True)
    pilot_parser.add_argument("--device", required=True)
    pilot_parser.add_argument("--num-workers", type=int, default=8)
    pilot_parser.add_argument("--eval-batch-size", type=int, default=64)
    pilot_parser.add_argument("--training-only", action="store_true")
    internal = commands.add_parser("internal-train")
    internal.add_argument("--component", choices=tuple(EXPECTED_RECIPES), required=True)
    internal.add_argument("--completion-config", type=Path, required=True)
    internal.add_argument("trainer_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        payload = preflight_payload(args.completion_config)
        target = _resolve(args.output)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite preflight: {target}")
        _write_json(target, payload)
        print(json.dumps({"passed": payload["passed"], "output": str(target)}))
        if not payload["passed"]:
            raise SystemExit(1)
    elif args.command == "pilot":
        payload = pilot(
            completion_path=args.completion_config,
            preflight_path=args.preflight,
            output=args.output,
            device=args.device,
            num_workers=args.num_workers,
            eval_batch_size=args.eval_batch_size,
            training_only=args.training_only,
        )
        print(json.dumps({"status": payload["status"], "output": str(_resolve(args.output))}))
    else:
        trainer_args = list(args.trainer_args)
        if trainer_args and trainer_args[0] == "--":
            trainer_args = trainer_args[1:]
        internal_train(args.completion_config, args.component, trainer_args)


if __name__ == "__main__":
    main()
