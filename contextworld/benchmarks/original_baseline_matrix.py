from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from contextworld.paths import repository_root


DEFAULT_ORIGINAL_BASELINE_PREREG = Path(
    "configs/benchmark/contextworld_original_baseline_completion_prereg_v1.yaml"
)
DEFAULT_ORIGINAL_BASELINE_FREEZE = Path(
    "configs/benchmark/contextworld_original_baseline_completion_freeze_v1.json"
)

EXPECTED_ENVIRONMENTS = {"tworoom", "pusht", "reacher", "cube"}
EXPECTED_FAMILIES = {"lewm", "pldm"}
EXPECTED_CAPABILITIES = {
    "contextworld-speed",
    "contextworld-door",
    "contextworld-action-delay",
    "contextworld-action-strength",
    "contextworld-contact-friction",
    "contextworld-motion-damping",
    "contextworld-reacher-arm-mass",
    "contextworld-portal-exit",
    "contextworld-cube-gripper-carry",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, *, logical_path: str | None = None) -> dict[str, Any]:
    return {
        "path": logical_path or path.as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _resolve(path: str, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def _check_declared_identity(
    entry: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> dict[str, Any]:
    path = _resolve(str(entry.get("path", "")), repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = _identity(path, logical_path=str(entry["path"]))
    expected = {
        "path": str(entry.get("path")),
        "sha256": str(entry.get("sha256")),
        "size_bytes": int(entry.get("size_bytes", -1)),
    }
    if observed != expected:
        raise RuntimeError(
            f"{label} identity mismatch: expected {expected}, got {observed}"
        )
    return observed


def load_original_baseline_prereg(
    path: Path | str = DEFAULT_ORIGINAL_BASELINE_PREREG,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    config_path = _resolve(str(path), repo_root=root).resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prereg = dict(_mapping(document, label="original baseline preregistration"))

    if prereg.get("schema_version") != 1:
        raise ValueError("original baseline preregistration schema_version must be 1")
    if prereg.get("preregistration_id") != "contextworld_original_baseline_completion_v1":
        raise ValueError("unexpected original baseline preregistration_id")
    if prereg.get("status") != "frozen_before_new_baseline_scoring":
        raise ValueError("original baseline preregistration is not frozen")

    authority = _mapping(prereg.get("authority"), label="authority")
    required_authority = {
        "training_authorized": False,
        "checkpoint_selection_authorized": False,
        "formal_reference_rerun": False,
        "formal_scoreboard_mutation": False,
        "post_release_descriptive_baseline_scoring": True,
    }
    for key, expected in required_authority.items():
        if authority.get(key) is not expected:
            raise ValueError(f"authority.{key} must be {expected!r}")

    checkpoints = _sequence(prereg.get("checkpoints"), label="checkpoints")
    checkpoint_ids: set[str] = set()
    checkpoint_pairs: set[tuple[str, str]] = set()
    checkpoint_pair_by_id: dict[str, tuple[str, str]] = {}
    for raw in checkpoints:
        checkpoint = _mapping(raw, label="checkpoint")
        checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
        pair = (str(checkpoint.get("environment", "")), str(checkpoint.get("family", "")))
        if checkpoint_id in checkpoint_ids or pair in checkpoint_pairs:
            raise ValueError("checkpoint ids and environment/family pairs must be unique")
        checkpoint_ids.add(checkpoint_id)
        checkpoint_pairs.add(pair)
        checkpoint_pair_by_id[checkpoint_id] = pair
        for identity_name in ("weights", "training_config"):
            identity = _mapping(checkpoint.get(identity_name), label=f"{checkpoint_id}.{identity_name}")
            if len(str(identity.get("sha256", ""))) != 64:
                raise ValueError(f"{checkpoint_id}.{identity_name} has invalid SHA-256")
            if int(identity.get("size_bytes", 0)) <= 0:
                raise ValueError(f"{checkpoint_id}.{identity_name} has invalid size")
        if checkpoint.get("contextworld_capability_training_used") is not False:
            raise ValueError(f"{checkpoint_id} is not frozen as an original-only baseline")
    expected_pairs = {
        (environment, family)
        for environment in EXPECTED_ENVIRONMENTS
        for family in EXPECTED_FAMILIES
    }
    if checkpoint_pairs != expected_pairs:
        raise ValueError(f"checkpoint matrix mismatch: {checkpoint_pairs}")

    components = _sequence(prereg.get("components"), label="components")
    component_ids: set[str] = set()
    component_environment: dict[str, str] = {}
    for raw in components:
        component = _mapping(raw, label="component")
        component_id = str(component.get("capability_id", ""))
        if component_id in component_ids:
            raise ValueError(f"duplicate component: {component_id}")
        component_ids.add(component_id)
        component_environment[component_id] = str(component.get("environment", ""))
        release = _mapping(component.get("release_config"), label=f"{component_id}.release_config")
        if len(str(release.get("sha256", ""))) != 64 or int(release.get("size_bytes", 0)) <= 0:
            raise ValueError(f"invalid release identity for {component_id}")
    if component_ids != EXPECTED_CAPABILITIES:
        raise ValueError(f"component set mismatch: {component_ids}")

    cells = _sequence(prereg.get("icl_cells"), label="icl_cells")
    cell_pairs: set[tuple[str, str]] = set()
    for raw in cells:
        cell = _mapping(raw, label="ICL cell")
        pair = (str(cell.get("capability_id", "")), str(cell.get("family", "")))
        if pair in cell_pairs:
            raise ValueError(f"duplicate ICL cell: {pair}")
        cell_pairs.add(pair)
        checkpoint_id = str(cell.get("checkpoint_id", ""))
        if checkpoint_id not in checkpoint_ids:
            raise ValueError(f"unknown checkpoint in ICL cell {pair}")
        expected_checkpoint_pair = (component_environment[pair[0]], pair[1])
        if checkpoint_pair_by_id[checkpoint_id] != expected_checkpoint_pair:
            raise ValueError(
                f"ICL cell {pair} uses {checkpoint_id}, expected "
                f"{expected_checkpoint_pair}"
            )
        output = str(cell.get("output", ""))
        if not output.startswith("artifacts/evaluation/original_baseline_matrix_v1/"):
            raise ValueError(f"ICL output escapes additive namespace: {output}")
        if cell.get("formal_scoreboard_eligible") is not False:
            raise ValueError(f"ICL cell {pair} must be descriptive-only")
    expected_cells = {
        (capability, family)
        for capability in EXPECTED_CAPABILITIES
        for family in EXPECTED_FAMILIES
    }
    if cell_pairs != expected_cells:
        raise ValueError(f"ICL matrix mismatch: {cell_pairs}")

    cem_followup = _mapping(prereg.get("cem_followup"), label="cem_followup")
    if (
        cem_followup.get("authorized_by_this_preregistration") is not False
        or cem_followup.get("uniform_environment_level_reuse_allowed") is not False
        or cem_followup.get("component_specific_protocol_identity_required") is not True
    ):
        raise ValueError(
            "CEM must remain outside this ICL preregistration until each "
            "component-specific query/runtime identity is frozen"
        )

    action_delay_cells = [
        cell
        for cell in cells
        if cell["capability_id"] == "contextworld-action-delay"
    ]
    if any(
        cell.get("history_adapter") != "frozen_history7_inference_from_h3_checkpoint"
        or cell.get("native_history7_checkpoint") is not False
        for cell in action_delay_cells
    ):
        raise ValueError("Action Delay baseline must disclose the H3-to-H7 inference adapter")

    prereg["_config_path"] = config_path.as_posix()
    return prereg


def audit_original_baseline_prereg(
    prereg_path: Path | str = DEFAULT_ORIGINAL_BASELINE_PREREG,
    *,
    freeze_path: Path | str = DEFAULT_ORIGINAL_BASELINE_FREEZE,
    repo_root: Path | None = None,
    verify_local_checkpoints: bool = False,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    prereg = load_original_baseline_prereg(prereg_path, repo_root=root)
    config_path = Path(prereg["_config_path"])

    checked: list[dict[str, Any]] = []
    for component in prereg["components"]:
        checked.append(
            _check_declared_identity(
                component["release_config"],
                repo_root=root,
                label=f"{component['capability_id']} release config",
            )
        )
    for implementation in prereg["implementation"]:
        checked.append(
            _check_declared_identity(
                implementation,
                repo_root=root,
                label=f"implementation {implementation['path']}",
            )
        )
    for checkpoint in prereg["checkpoints"]:
        config = checkpoint["training_config"]
        config_path_declared = _resolve(str(config["path"]), repo_root=root)
        if config_path_declared.is_relative_to(root) or verify_local_checkpoints:
            checked.append(
                _check_declared_identity(
                    config,
                    repo_root=root,
                    label=f"{checkpoint['checkpoint_id']} training config",
                )
            )
        if verify_local_checkpoints:
            checked.append(
                _check_declared_identity(
                    checkpoint["weights"],
                    repo_root=root,
                    label=f"{checkpoint['checkpoint_id']} weights",
                )
            )

    freeze_file = _resolve(str(freeze_path), repo_root=root).resolve()
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))
    expected_prereg = _identity(
        config_path,
        logical_path=str(Path(prereg_path).as_posix()),
    )
    if freeze.get("preregistration") != expected_prereg:
        raise RuntimeError("freeze receipt does not bind the current preregistration")
    if freeze.get("status") != "frozen_before_new_baseline_scoring":
        raise RuntimeError("freeze receipt status mismatch")

    return {
        "schema_version": 1,
        "audit_id": "contextworld_original_baseline_completion_v1",
        "status": "passed",
        "preregistration": expected_prereg,
        "freeze_receipt": _identity(
            freeze_file,
            logical_path=str(Path(freeze_path).as_posix()),
        ),
        "counts": {
            "canonical_checkpoints": len(prereg["checkpoints"]),
            "icl_cells": len(prereg["icl_cells"]),
            "authorized_cem_jobs": 0,
            "verified_files": len(checked),
        },
        "local_checkpoint_weights_verified": bool(verify_local_checkpoints),
        "verified_identities": checked,
    }


__all__ = [
    "DEFAULT_ORIGINAL_BASELINE_FREEZE",
    "DEFAULT_ORIGINAL_BASELINE_PREREG",
    "audit_original_baseline_prereg",
    "load_original_baseline_prereg",
]
