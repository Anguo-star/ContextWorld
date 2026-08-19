"""Audit the immutable ContextWorld original-baseline result archive.

The original baseline matrix was derived against the release identities that
were current when its results were frozen.  Later maintenance may legitimately
change a live release file without changing those historical result receipts.
This auditor therefore verifies the frozen archive and every identity it
declares; it deliberately does not re-derive the archive from mutable live
release configurations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from contextworld.paths import repository_root, resolve_contextworld_path


DEFAULT_ORIGINAL_BASELINE_RESULTS_FREEZE = Path(
    "configs/benchmark/contextworld_original_baseline_matrix_results_freeze_v1.json"
)
FROZEN_RESULTS_FREEZE_SHA256 = (
    "ec72baaf04db9b084b1047e0ba9d6eaff3b13215902574d7e9a2b131de08eef5"
)
FROZEN_RESULTS_FREEZE_SIZE_BYTES = 4731

EXPECTED_CAPABILITIES = frozenset(
    {
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
)
EXPECTED_FAMILIES = frozenset({"lewm", "pldm"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve(logical_path: str, *, root: Path) -> Path:
    local = (root / logical_path).resolve()
    if local.is_file():
        return local
    return resolve_contextworld_path(logical_path, repo_root=root)


def _identity_mappings(value: Any) -> Iterator[Mapping[str, Any]]:
    """Yield all nested ``path``/``sha256`` identity declarations."""

    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(
            value.get("sha256"), str
        ):
            yield value
        for child in value.values():
            yield from _identity_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _identity_mappings(child)


def _verify_identity(declaration: Mapping[str, Any], *, root: Path) -> None:
    logical_path = str(declaration["path"])
    path = _resolve(logical_path, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"Archived evidence is missing: {logical_path}")
    observed_sha = _sha256(path)
    expected_sha = str(declaration["sha256"])
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"Archived evidence SHA-256 mismatch for {logical_path}: "
            f"{observed_sha} != {expected_sha}"
        )
    if "size_bytes" in declaration:
        expected_size = int(declaration["size_bytes"])
        observed_size = path.stat().st_size
        if observed_size != expected_size:
            raise RuntimeError(
                f"Archived evidence size mismatch for {logical_path}: "
                f"{observed_size} != {expected_size}"
            )


def validate_archived_original_baseline_summary(
    summary: Mapping[str, Any],
) -> dict[str, int]:
    """Validate the scientific scope and shape of the frozen 18-cell matrix."""

    if summary.get("schema_version") != 1:
        raise ValueError("Unsupported original-baseline archive schema")
    if summary.get("matrix_id") != "contextworld_original_baseline_matrix_v1":
        raise ValueError("Unexpected original-baseline matrix identity")
    if summary.get("status") != "completed":
        raise ValueError("Original-baseline archive is not complete")
    required_false = (
        "formal_scoreboard_mutated",
        "training_performed",
        "checkpoint_selection_performed",
        "cross_model_raw_latent_comparison_permitted",
    )
    if any(summary.get(field) is not False for field in required_false):
        raise ValueError("Original-baseline archive exceeds its descriptive scope")
    if summary.get("claim_scope") != (
        "post_release_single_checkpoint_descriptive_only"
    ):
        raise ValueError("Original-baseline claim scope changed")

    cells = summary.get("cells")
    if not isinstance(cells, list) or len(cells) != 18:
        raise ValueError("Original-baseline archive must contain exactly 18 cells")
    pairs: set[tuple[str, str]] = set()
    checkpoints: set[str] = set()
    passes = 0
    rescores = 0
    for index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            raise ValueError(f"Cell {index} is not an object")
        capability = str(cell.get("capability_id", ""))
        family = str(cell.get("family", ""))
        if capability not in EXPECTED_CAPABILITIES or family not in EXPECTED_FAMILIES:
            raise ValueError(f"Unexpected matrix cell: {(capability, family)}")
        pair = (capability, family)
        if pair in pairs:
            raise ValueError(f"Duplicate matrix cell: {pair}")
        pairs.add(pair)
        if cell.get("formal_scoreboard_eligible") is not False:
            raise ValueError(f"Archived descriptive cell became scoreboard-eligible: {pair}")
        checkpoint_id = str(cell.get("checkpoint_id", ""))
        checkpoint_sha = str(cell.get("checkpoint_sha256", ""))
        if not checkpoint_id or len(checkpoint_sha) != 64:
            raise ValueError(f"Cell does not bind a checkpoint: {pair}")
        checkpoints.add(checkpoint_id)
        metric = cell.get("metric")
        if not isinstance(metric, Mapping) or not isinstance(metric.get("gate"), Mapping):
            raise ValueError(f"Cell metric is malformed: {pair}")
        passes += metric["gate"].get("passed") is True
        rescores += cell.get("rescore_evidence") is not None

    expected_pairs = {
        (capability, family)
        for capability in EXPECTED_CAPABILITIES
        for family in EXPECTED_FAMILIES
    }
    if pairs != expected_pairs:
        raise ValueError("Original-baseline capability/family matrix is incomplete")
    derived = {
        "canonical_checkpoints": len(checkpoints),
        "capabilities": len({capability for capability, _ in pairs}),
        "icl_cells": len(cells),
        "completed_cells": len(cells),
        "passing_single_checkpoint_gates": passes,
        "formal_scoreboard_eligible_cells": 0,
        "authorized_cem_jobs": 0,
        "cells_with_rescore_evidence": rescores,
        "cells_without_rescore_entrypoint": len(cells) - rescores,
    }
    expected_counts = summary.get("counts")
    if not isinstance(expected_counts, Mapping) or dict(expected_counts) != derived:
        raise ValueError(
            f"Original-baseline count summary changed: {expected_counts} != {derived}"
        )
    if derived != {
        "canonical_checkpoints": 8,
        "capabilities": 9,
        "icl_cells": 18,
        "completed_cells": 18,
        "passing_single_checkpoint_gates": 1,
        "formal_scoreboard_eligible_cells": 0,
        "authorized_cem_jobs": 0,
        "cells_with_rescore_evidence": 12,
        "cells_without_rescore_entrypoint": 6,
    }:
        raise ValueError("Original-baseline frozen matrix invariants changed")
    cem = summary.get("cem_followup")
    if not isinstance(cem, Mapping) or cem.get("status") != (
        "pending_separate_component_bound_preregistration"
    ):
        raise ValueError("The archived ICL matrix unexpectedly authorizes CEM")
    return derived


def audit_archived_original_baseline_matrix(
    *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Verify the frozen archive without consulting mutable live releases."""

    root = (repo_root or repository_root()).resolve()
    freeze_path = (root / DEFAULT_ORIGINAL_BASELINE_RESULTS_FREEZE).resolve()
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)
    if (
        _sha256(freeze_path) != FROZEN_RESULTS_FREEZE_SHA256
        or freeze_path.stat().st_size != FROZEN_RESULTS_FREEZE_SIZE_BYTES
    ):
        raise RuntimeError("Original-baseline result-freeze identity changed")
    freeze = _load_json(freeze_path)
    if (
        freeze.get("schema_version") != 1
        or freeze.get("freeze_id")
        != "contextworld_original_baseline_matrix_results_freeze_v1"
        or freeze.get("status") != "frozen_after_derived_summary"
    ):
        raise ValueError("Original-baseline result freeze is malformed")
    scientific = freeze.get("scientific_status")
    if not isinstance(scientific, Mapping) or any(
        scientific.get(field) is not False
        for field in (
            "blind_preregistration_claimed",
            "formal_scoreboard_mutated",
            "training_performed",
            "checkpoint_selection_performed",
            "cem_authorized",
        )
    ):
        raise ValueError("Original-baseline result freeze exceeds its authority")

    matrix_declaration = freeze.get("matrix_summary")
    if not isinstance(matrix_declaration, Mapping):
        raise ValueError("Result freeze does not identify its matrix summary")
    _verify_identity(matrix_declaration, root=root)
    summary_path = _resolve(str(matrix_declaration["path"]), root=root)
    summary = _load_json(summary_path)
    counts = validate_archived_original_baseline_summary(summary)

    # Validate every evidence identity retained by either archival layer.  This
    # includes all 18 raw receipts, 12 rescore receipts, recovery receipts,
    # checkpoint audit, preregistration, and frozen derivation sources.
    declarations = list(_identity_mappings(freeze)) + list(
        _identity_mappings(summary)
    )
    unique: dict[tuple[str, str, int | None], Mapping[str, Any]] = {}
    for declaration in declarations:
        key = (
            str(declaration["path"]),
            str(declaration["sha256"]),
            int(declaration["size_bytes"])
            if "size_bytes" in declaration
            else None,
        )
        unique[key] = declaration
    for declaration in unique.values():
        _verify_identity(declaration, root=root)

    for section in ("base_protocol", "derivation_implementation"):
        if freeze.get(section) != summary.get(section):
            raise RuntimeError(f"Freeze/summary {section} declarations disagree")
    return {
        "schema_version": 1,
        "audit_id": "contextworld_original_baseline_archive_audit_v1",
        "status": "passed",
        "archive_scope": "immutable_frozen_results_only",
        "live_release_rederivation_performed": False,
        "result_freeze": {
            "path": DEFAULT_ORIGINAL_BASELINE_RESULTS_FREEZE.as_posix(),
            "sha256": FROZEN_RESULTS_FREEZE_SHA256,
            "size_bytes": FROZEN_RESULTS_FREEZE_SIZE_BYTES,
        },
        "verified_identity_declarations": len(unique),
        "counts": counts,
    }


__all__ = [
    "DEFAULT_ORIGINAL_BASELINE_RESULTS_FREEZE",
    "FROZEN_RESULTS_FREEZE_SHA256",
    "audit_archived_original_baseline_matrix",
    "validate_archived_original_baseline_summary",
]
