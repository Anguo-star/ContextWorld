"""Static audit for the blocked original-baseline CEM completion draft.

This module deliberately has no CEM, torch, or environment imports.  It is
only allowed to describe and audit a future component-bound CEM matrix.  In
particular, it must not turn historical CEM receipts into new results or emit
an execution authorization.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from contextworld.benchmarks.original_baseline_matrix import (
    load_original_baseline_prereg,
)
from contextworld.paths import repository_root


DEFAULT_ORIGINAL_BASELINE_CEM_COMPLETION_DRAFT = Path(
    "configs/benchmark/"
    "contextworld_original_baseline_cem_completion_prereg_draft_v1.yaml"
)
DEFAULT_ORIGINAL_BASELINE_CEM_COMPLETION_BLOCKED_AUDIT = Path(
    "artifacts/evaluation/original_baseline_cem_completion_v1/"
    "blocked_audit.json"
)

PREREGISTRATION_ID = "contextworld_original_baseline_cem_completion_draft_v1"
EXPECTED_COMPONENTS = (
    "contextworld-speed",
    "contextworld-door",
    "contextworld-action-delay",
    "contextworld-action-strength",
    "contextworld-reacher-arm-mass",
    "contextworld-portal-exit",
    "contextworld-cube-gripper-carry",
)
EXPECTED_EXCLUDED_COMPONENTS = (
    "contextworld-contact-friction",
    "contextworld-motion-damping",
)
EXPECTED_UNIT_COUNTS = {
    "contextworld-speed": 72,
    "contextworld-door": 12,
    "contextworld-action-delay": 12,
    "contextworld-action-strength": 12,
    "contextworld-reacher-arm-mass": 6,
    "contextworld-portal-exit": 12,
    "contextworld-cube-gripper-carry": 6,
}
EXPECTED_FAMILIES = ("lewm", "pldm")
EXPECTED_CELL_CHECKPOINTS = {
    ("contextworld-speed", "lewm"): "tworoom_lewm_original",
    ("contextworld-speed", "pldm"): "tworoom_pldm_original",
    ("contextworld-door", "lewm"): "tworoom_lewm_original",
    ("contextworld-door", "pldm"): "tworoom_pldm_original",
    ("contextworld-action-delay", "lewm"): "tworoom_lewm_original",
    ("contextworld-action-delay", "pldm"): "tworoom_pldm_original",
    ("contextworld-action-strength", "lewm"): "pusht_lewm_original",
    ("contextworld-action-strength", "pldm"): "pusht_pldm_original",
    ("contextworld-reacher-arm-mass", "lewm"): "reacher_lewm_original",
    ("contextworld-reacher-arm-mass", "pldm"): "reacher_pldm_original",
    ("contextworld-portal-exit", "lewm"): "tworoom_lewm_original",
    ("contextworld-portal-exit", "pldm"): "tworoom_pldm_original",
    ("contextworld-cube-gripper-carry", "lewm"): "cube_lewm_original",
    ("contextworld-cube-gripper-carry", "pldm"): "cube_pldm_original",
}
EXPECTED_BLOCKER_KINDS = {
    "contextworld-speed": "runner_config_catalog_identity_unclosed",
    "contextworld-door": "original_heldout_catalog_missing",
    "contextworld-action-delay": "h3_checkpoint_h7_runner_incompatible",
    "contextworld-action-strength": "runtime_query_resampling_without_pre_frozen_catalog",
    "contextworld-reacher-arm-mass": "runtime_query_resampling_without_pre_frozen_catalog",
    "contextworld-portal-exit": "original_heldout_catalog_missing",
    "contextworld-cube-gripper-carry": "legacy_catalog_not_verifiable_in_current_tree",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, *, logical_path: str) -> dict[str, Any]:
    return {
        "path": logical_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    return value


def _resolve(raw_path: str, *, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else repo_root / path


def _verify_declared_identity(
    entry: Mapping[str, Any], *, repo_root: Path, label: str
) -> dict[str, Any]:
    raw_path = str(entry.get("path", ""))
    if not raw_path:
        raise ValueError(f"{label}.path is required")
    path = _resolve(raw_path, repo_root=repo_root)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    observed = _identity(path, logical_path=raw_path)
    expected = {
        "path": raw_path,
        "sha256": str(entry.get("sha256", "")),
        "size_bytes": int(entry.get("size_bytes", -1)),
    }
    if observed != expected:
        raise RuntimeError(
            f"{label} identity mismatch: expected {expected}, got {observed}"
        )
    return observed


def _identity_entries(
    component: Mapping[str, Any], *, component_id: str
) -> list[tuple[str, Mapping[str, Any]]]:
    identities = _mapping(
        component.get("source_identities"), field=f"{component_id}.source_identities"
    )
    release = _mapping(
        identities.get("release_config"),
        field=f"{component_id}.source_identities.release_config",
    )
    rows: list[tuple[str, Mapping[str, Any]]] = [("release_config", release)]
    for group in ("runner_sources", "query_catalog_sources"):
        values = _sequence(
            identities.get(group), field=f"{component_id}.source_identities.{group}"
        )
        if not values:
            raise ValueError(f"{component_id}.{group} must be non-empty")
        for index, value in enumerate(values):
            rows.append(
                (
                    f"{group}[{index}]",
                    _mapping(value, field=f"{component_id}.{group}[{index}]"),
                )
            )
    return rows


def _validate_planning(component: Mapping[str, Any], *, component_id: str) -> None:
    planning = _mapping(component.get("planning"), field=f"{component_id}.planning")
    required_integer_fields = (
        "execution_budget_raw_steps",
        "horizon_action_blocks",
        "receding_horizon_action_blocks",
        "cem_candidates",
        "cem_iterations",
        "cem_topk",
        "action_block_raw_steps",
    )
    for field in required_integer_fields:
        if int(planning.get(field, 0)) <= 0:
            raise ValueError(f"{component_id}.planning.{field} must be positive")
    if int(planning["horizon_action_blocks"]) * int(
        planning["action_block_raw_steps"]
    ) > int(planning["execution_budget_raw_steps"]):
        raise ValueError(f"{component_id} planning horizon exceeds execution budget")


def _validate_component(component: Mapping[str, Any]) -> None:
    component_id = str(component.get("capability_id", ""))
    if component_id not in EXPECTED_COMPONENTS:
        raise ValueError(f"unexpected CEM component: {component_id}")
    if component.get("execution_status") != "blocked_before_execution":
        raise ValueError(f"{component_id} must remain blocked_before_execution")
    if component.get("cem_execution_authorized") is not False:
        raise ValueError(f"{component_id} must not authorize CEM execution")
    if component.get("executable_argv") is not None:
        raise ValueError(f"{component_id} must not contain an executable command")
    blockers = _sequence(component.get("blockers"), field=f"{component_id}.blockers")
    if not blockers or not all(isinstance(value, str) and value for value in blockers):
        raise ValueError(f"{component_id} needs explicit non-empty blockers")
    blocker_evidence = _mapping(
        component.get("blocker_evidence"), field=f"{component_id}.blocker_evidence"
    )
    if blocker_evidence.get("kind") != EXPECTED_BLOCKER_KINDS[component_id]:
        raise ValueError(f"{component_id} blocker evidence kind drifted")
    if int(blocker_evidence.get("affected_component_bound_cells", -1)) != 2:
        raise ValueError(f"{component_id} blocker must cover the LeWM/PLDM pair")
    if blocker_evidence.get("checkpoint_missing_is_a_blocker") is not False:
        raise ValueError(f"{component_id} must not describe a missing checkpoint")
    expected_units = EXPECTED_UNIT_COUNTS[component_id]
    if int(component.get("planned_atomic_units", -1)) != expected_units:
        raise ValueError(
            f"{component_id} planned units must be {expected_units}, not "
            f"{component.get('planned_atomic_units')!r}"
        )
    families = tuple(
        str(value)
        for value in _sequence(
            component.get("families"), field=f"{component_id}.families"
        )
    )
    if families != EXPECTED_FAMILIES:
        raise ValueError(f"{component_id} must retain the LeWM/PLDM pair")
    unit_axes = _mapping(component.get("unit_axes"), field=f"{component_id}.unit_axes")
    seeds = tuple(
        int(value)
        for value in _sequence(
            unit_axes.get("eval_seeds"),
            field=f"{component_id}.unit_axes.eval_seeds",
        )
    )
    conditions = _sequence(
        unit_axes.get("conditions"), field=f"{component_id}.unit_axes.conditions"
    )
    if not seeds or len(seeds) != len(set(seeds)) or not conditions:
        raise ValueError(f"{component_id} has invalid atomic-unit axes")
    if len(families) * len(seeds) * len(conditions) != expected_units:
        raise ValueError(f"{component_id} unit axes do not yield {expected_units} units")
    if not all(isinstance(value, Mapping) for value in conditions):
        raise ValueError(f"{component_id}.unit_axes.conditions must contain mappings")
    _validate_planning(component, component_id=component_id)
    runtime = _mapping(component.get("stable_worldmodel"), field=f"{component_id}.stable_worldmodel")
    expected_ref = str(runtime.get("expected_commit", ""))
    if len(expected_ref) != 40:
        raise ValueError(f"{component_id} needs a 40-character Stable-WorldModel pin")
    if runtime.get("clean_checkout_required") is not True:
        raise ValueError(f"{component_id} requires a clean Stable-WorldModel checkout")
    _identity_entries(component, component_id=component_id)


def _canonical_checkpoint_ids(base: Mapping[str, Any]) -> set[str]:
    checkpoints = _sequence(base.get("checkpoints"), field="base.checkpoints")
    return {
        str(_mapping(row, field="base.checkpoint").get("checkpoint_id", ""))
        for row in checkpoints
    }


def load_original_baseline_cem_completion_draft(
    path: Path | str = DEFAULT_ORIGINAL_BASELINE_CEM_COMPLETION_DRAFT,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load and validate the non-executable, blocked CEM completion draft."""

    root = (repo_root or repository_root()).resolve()
    config_path = _resolve(str(path), repo_root=root).resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    draft = dict(_mapping(document, field="CEM completion draft"))
    if draft.get("schema_version") != 1:
        raise ValueError("CEM completion draft schema_version must be 1")
    if draft.get("preregistration_id") != PREREGISTRATION_ID:
        raise ValueError("unexpected CEM completion draft preregistration_id")
    if draft.get("status") != "draft_blocked":
        raise ValueError("CEM completion draft must remain blocked")
    if draft.get("freeze_generated") is not False:
        raise ValueError("a blocked draft must not carry a freeze receipt")

    scientific_status = _mapping(
        draft.get("scientific_status"), field="scientific_status"
    )
    if (
        scientific_status.get("atomic_unit_count_kind")
        != "planning_estimate_only"
        or scientific_status.get("atomic_unit_count_is_not_cli_process_count")
        is not True
        or int(scientific_status.get("planned_atomic_execution_units", -1))
        != sum(EXPECTED_UNIT_COUNTS.values())
    ):
        raise ValueError("132 atomic units must remain a planning-only estimate")

    commands = _sequence(draft.get("commands"), field="commands")
    if commands:
        raise ValueError("a blocked draft must not materialize commands")

    authority = _mapping(draft.get("authority"), field="authority")
    required_authority = {
        "cem_execution_authorized": False,
        "training_authorized": False,
        "finetuning_authorized": False,
        "checkpoint_selection_authorized": False,
        "result_based_retry_or_checkpoint_swap_authorized": False,
        "formal_scoreboard_mutation": False,
        "legacy_cem_result_reuse_authorized": False,
    }
    for field, expected in required_authority.items():
        if authority.get(field) is not expected:
            raise ValueError(f"authority.{field} must be {expected!r}")

    binding = _mapping(
        draft.get("canonical_checkpoint_source"), field="canonical_checkpoint_source"
    )
    _verify_declared_identity(binding, repo_root=root, label="canonical checkpoint source")
    base = load_original_baseline_prereg(
        str(binding["path"]), repo_root=root
    )
    expected_checkpoints = _canonical_checkpoint_ids(base)
    declared_checkpoints = {
        str(value)
        for value in _sequence(
            draft.get("canonical_checkpoint_ids"), field="canonical_checkpoint_ids"
        )
    }
    if len(expected_checkpoints) != 8 or declared_checkpoints != expected_checkpoints:
        raise ValueError("draft must use exactly the eight canonical base checkpoints")

    availability = _mapping(
        draft.get("checkpoint_availability"), field="checkpoint_availability"
    )
    if (
        availability.get("status") != "complete"
        or int(availability.get("canonical_checkpoint_count", -1)) != 8
        or availability.get("checkpoint_missing_is_a_blocker") is not False
    ):
        raise ValueError("checkpoint availability must state that all eight are present")

    planned_units = _mapping(
        draft.get("planned_atomic_execution_units"),
        field="planned_atomic_execution_units",
    )
    if (
        planned_units.get("classification") != "estimate_planned_only"
        or int(planned_units.get("total", -1)) != sum(EXPECTED_UNIT_COUNTS.values())
    ):
        raise ValueError("planned atomic unit total must remain a 132-unit estimate")
    declared_unit_counts = _mapping(
        planned_units.get("by_component"),
        field="planned_atomic_execution_units.by_component",
    )
    if {
        str(key): int(value) for key, value in declared_unit_counts.items()
    } != EXPECTED_UNIT_COUNTS:
        raise ValueError("planned atomic unit breakdown drifted")

    components = _sequence(draft.get("component_protocols"), field="component_protocols")
    component_ids = [
        str(_mapping(value, field="component").get("capability_id", ""))
        for value in components
    ]
    if tuple(component_ids) != EXPECTED_COMPONENTS or len(set(component_ids)) != len(
        component_ids
    ):
        raise ValueError("draft component set/order drifted")
    for component in components:
        _validate_component(_mapping(component, field="component"))

    cells = _sequence(draft.get("cem_cells"), field="cem_cells")
    observed_pairs: set[tuple[str, str]] = set()
    component_by_id = {
        str(component["capability_id"]): component for component in components
    }
    for raw_cell in cells:
        cell = _mapping(raw_cell, field="cem_cell")
        component_id = str(cell.get("capability_id", ""))
        family = str(cell.get("family", ""))
        pair = (component_id, family)
        if pair in observed_pairs:
            raise ValueError(f"duplicate CEM cell: {pair}")
        observed_pairs.add(pair)
        if component_id not in component_by_id or family not in EXPECTED_FAMILIES:
            raise ValueError(f"unexpected CEM cell: {pair}")
        if cell.get("execution_status") != "blocked_before_execution":
            raise ValueError(f"{pair} is not blocked")
        if cell.get("cem_execution_authorized") is not False:
            raise ValueError(f"{pair} authorizes CEM execution")
        if cell.get("result_based_retry") is not False:
            raise ValueError(f"{pair} authorizes a result-based retry")
        if cell.get("formal_scoreboard_eligible") is not False:
            raise ValueError(f"{pair} is scoreboard eligible")
        if cell.get("legacy_cem_result_reused") is not False:
            raise ValueError(f"{pair} reuses a legacy CEM result")
        checkpoint_id = str(cell.get("checkpoint_id", ""))
        if checkpoint_id not in declared_checkpoints:
            raise ValueError(f"{pair} does not use a canonical checkpoint")
        if checkpoint_id != EXPECTED_CELL_CHECKPOINTS[pair]:
            raise ValueError(
                f"{pair} uses {checkpoint_id}, expected "
                f"{EXPECTED_CELL_CHECKPOINTS[pair]}"
            )
        if not str(cell.get("output", "")).startswith(
            "artifacts/evaluation/original_baseline_cem_completion_v1/"
        ):
            raise ValueError(f"{pair} output escapes the additive namespace")
    expected_pairs = {
        (component, family)
        for component in EXPECTED_COMPONENTS
        for family in EXPECTED_FAMILIES
    }
    if observed_pairs != expected_pairs:
        raise ValueError("draft must contain exactly 14 component-bound CEM cells")

    excluded = _sequence(draft.get("excluded_cem_cells"), field="excluded_cem_cells")
    excluded_pairs = {
        (
            str(
                _mapping(row, field="excluded_cem_cell").get(
                    "capability_id", ""
                )
            ),
            str(_mapping(row, field="excluded_cem_cell").get("family", "")),
        )
        for row in excluded
    }
    expected_excluded = {
        (component, family)
        for component in EXPECTED_EXCLUDED_COMPONENTS
        for family in EXPECTED_FAMILIES
    }
    if excluded_pairs != expected_excluded or len(excluded) != 4:
        raise ValueError("the four development/public-closed cells must be explicit")
    for row in excluded:
        cell = _mapping(row, field="excluded_cem_cell")
        if cell.get("execution_status") != "excluded_development_public_closed":
            raise ValueError("excluded cell status drifted")
        if cell.get("cem_execution_authorized") is not False:
            raise ValueError("excluded cells must not authorize CEM")

    legacy = _mapping(draft.get("legacy_cem_disclosure"), field="legacy_cem_disclosure")
    if (
        legacy.get("any_legacy_cem_promoted_to_new_result") is not False
        or legacy.get("cube_lewm_closed_loop") != "legacy_candidate_only_not_reused"
    ):
        raise ValueError("legacy CEM disclosure drifted")

    publication = _mapping(draft.get("publication"), field="publication")
    if (
        publication.get("publish_success_receipts") is not True
        or publication.get("publish_failure_receipts") is not True
        or publication.get("formal_scoreboard_mutation") is not False
    ):
        raise ValueError("publication contract drifted")

    draft["_config_path"] = config_path.as_posix()
    return draft


def _runtime_observation(component: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    runtime = _mapping(component["stable_worldmodel"], field="stable_worldmodel")
    raw_root = str(runtime.get("audit_checkout", ""))
    checkout = _resolve(raw_root, repo_root=repo_root)
    expected_commit = str(runtime["expected_commit"])
    result: dict[str, Any] = {
        "audit_checkout": raw_root,
        "expected_commit": expected_commit,
        "checked": False,
        "commit_matches_expected": False,
        "clean": False,
    }
    if not checkout.is_dir() or not (checkout / ".git").exists():
        result["blocker"] = "stable_worldmodel_checkout_missing"
        return result
    try:
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        result["blocker"] = f"stable_worldmodel_identity_unreadable:{type(error).__name__}"
        return result
    result.update(
        {
            "checked": True,
            "observed_commit": head,
            "commit_matches_expected": head == expected_commit,
            "clean": not bool(status.strip()),
        }
    )
    if not result["commit_matches_expected"]:
        result["blocker"] = "stable_worldmodel_commit_mismatch"
    elif not result["clean"]:
        result["blocker"] = "stable_worldmodel_checkout_dirty"
    return result


def enumerate_atomic_execution_units(
    draft: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Materialize the 132 *non-executable* units for audit transparency."""

    units: list[dict[str, Any]] = []
    for component in draft["component_protocols"]:
        component_id = str(component["capability_id"])
        axes = component["unit_axes"]
        for family in component["families"]:
            for condition in axes["conditions"]:
                condition_id = str(condition["condition_id"])
                for seed in axes["eval_seeds"]:
                    unit_id = f"{component_id}/{family}/{condition_id}/seed{int(seed)}"
                    units.append(
                        {
                            "unit_id": unit_id,
                            "capability_id": component_id,
                            "family": str(family),
                            "checkpoint_id": next(
                                str(cell["checkpoint_id"])
                                for cell in draft["cem_cells"]
                                if cell["capability_id"] == component_id
                                and cell["family"] == family
                            ),
                            "condition": dict(condition),
                            "eval_seed": int(seed),
                            "execution_status": "blocked_before_execution",
                            "cem_execution_authorized": False,
                            "command": None,
                            "result_status": "not_started",
                            "planned_output": (
                                "artifacts/evaluation/original_baseline_cem_completion_v1/"
                                f"units/{component_id}/{family}/{condition_id}/"
                                f"seed{int(seed)}.json"
                            ),
                        }
                    )
    if len(units) != sum(EXPECTED_UNIT_COUNTS.values()):
        raise AssertionError("atomic execution-unit count drifted")
    return units


def audit_original_baseline_cem_completion_draft(
    path: Path | str = DEFAULT_ORIGINAL_BASELINE_CEM_COMPLETION_DRAFT,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Verify static identities and emit a blocked, non-scoring audit record."""

    root = (repo_root or repository_root()).resolve()
    draft = load_original_baseline_cem_completion_draft(path, repo_root=root)
    config_path = Path(draft["_config_path"])
    checked: list[dict[str, Any]] = []
    runtime: dict[str, dict[str, Any]] = {}
    for component in draft["component_protocols"]:
        component_id = str(component["capability_id"])
        for label, identity in _identity_entries(component, component_id=component_id):
            checked.append(
                {
                    "component_id": component_id,
                    "role": label,
                    **_verify_declared_identity(
                        identity,
                        repo_root=root,
                        label=f"{component_id}.{label}",
                    ),
                }
            )
        runtime[component_id] = _runtime_observation(component, repo_root=root)

    units = enumerate_atomic_execution_units(draft)
    component_counts = {
        component_id: sum(
            unit["capability_id"] == component_id for unit in units
        )
        for component_id in EXPECTED_COMPONENTS
    }
    if component_counts != EXPECTED_UNIT_COUNTS:
        raise AssertionError("component atomic-unit counts drifted")
    if any(unit["command"] is not None for unit in units):
        raise AssertionError("blocked audit emitted a command")

    logical_path = str(Path(path).as_posix())
    return {
        "schema_version": 1,
        "audit_id": "contextworld_original_baseline_cem_completion_blocked_audit_v1",
        "status": "draft_blocked",
        "preregistration": _identity(config_path, logical_path=logical_path),
        "freeze_generated": False,
        "cem_execution_started": False,
        "cem_gpu_execution_started": False,
        "authorized_commands": [],
        "formal_scoreboard_mutated": False,
        "legacy_cem_result_reused": False,
        "counts": {
            "canonical_checkpoints": 8,
            "included_components": len(EXPECTED_COMPONENTS),
            "included_component_bound_cells": 14,
            "explicitly_excluded_development_public_closed_cells": 4,
            "planned_atomic_execution_units": len(units),
            "blocked_atomic_execution_units": len(units),
            "executable_atomic_execution_units": 0,
            "verified_static_identities": len(checked),
        },
        "component_unit_counts": component_counts,
        "checkpoint_availability": {
            "status": draft["checkpoint_availability"]["status"],
            "canonical_checkpoint_count": draft["checkpoint_availability"][
                "canonical_checkpoint_count"
            ],
            "checkpoint_missing_is_a_blocker": draft["checkpoint_availability"][
                "checkpoint_missing_is_a_blocker"
            ],
        },
        "planned_atomic_execution_units": {
            "classification": draft["planned_atomic_execution_units"]["classification"],
            "total": draft["planned_atomic_execution_units"]["total"],
            "by_component": draft["planned_atomic_execution_units"]["by_component"],
        },
        "component_blockers": {
            str(component["capability_id"]): list(component["blockers"])
            for component in draft["component_protocols"]
        },
        "component_blocker_evidence": {
            str(component["capability_id"]): dict(component["blocker_evidence"])
            for component in draft["component_protocols"]
        },
        "runtime_observations": runtime,
        "static_identity_checks": checked,
        "blocker_policy": {
            "all_cells_remain_non_executable": True,
            "new_freeze_prohibited_until_every_execution_identity_is_closed": True,
            "no_command_is_materialized_for_a_blocked_cell": True,
            "failures_must_be_published_if_execution_is_later_authorized": True,
        },
        "atomic_execution_units": units,
    }


def write_blocked_audit(
    output: Path | str = DEFAULT_ORIGINAL_BASELINE_CEM_COMPLETION_BLOCKED_AUDIT,
    *,
    prereg_path: Path | str = DEFAULT_ORIGINAL_BASELINE_CEM_COMPLETION_DRAFT,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Write an additive blocked audit; this is intentionally not a freeze."""

    root = (repo_root or repository_root()).resolve()
    payload = audit_original_baseline_cem_completion_draft(
        prereg_path, repo_root=root
    )
    target = _resolve(str(output), repo_root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DEFAULT_ORIGINAL_BASELINE_CEM_COMPLETION_BLOCKED_AUDIT",
    "DEFAULT_ORIGINAL_BASELINE_CEM_COMPLETION_DRAFT",
    "EXPECTED_COMPONENTS",
    "EXPECTED_EXCLUDED_COMPONENTS",
    "EXPECTED_UNIT_COUNTS",
    "audit_original_baseline_cem_completion_draft",
    "enumerate_atomic_execution_units",
    "load_original_baseline_cem_completion_draft",
    "write_blocked_audit",
]
