#!/usr/bin/env python3
"""Receipt-only recovery for the original-baseline CEM matrix finalizer.

The frozen standard runner writes ``query_count`` in each per-seed row.  The
original finalizer accepted only the equivalent ``evaluation_count`` and
``evaluations`` spellings, so it stopped before creating ``matrix_summary``.
This additive recovery admits that one exact alias and delegates every other
check to the frozen original finalizer.  It never launches CEM or modifies a
receipt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

import scripts.finalize_contextworld_original_baseline_cem_v1 as original
from contextworld.paths import repository_root, resolve_contextworld_path


ROOT = repository_root()
RECOVERY_ID = "contextworld_original_baseline_cem_finalizer_recovery_v1"
RECOVERY_PREREG = Path(
    "configs/benchmark/"
    "contextworld_original_baseline_cem_finalizer_recovery_prereg_v1.yaml"
)
RECOVERY_FREEZE = Path(
    "configs/benchmark/"
    "contextworld_original_baseline_cem_finalizer_recovery_freeze_v1.json"
)
EXPECTED_FAILURE = "pusht lewm seed 42.evaluation_count is missing"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(logical_path: str | Path, *, root: Path) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(logical_path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _require_identity(
    declared: Any, logical_path: str | Path, *, root: Path, label: str
) -> dict[str, Any]:
    observed = _identity(logical_path, root=root)
    if not isinstance(declared, Mapping) or dict(declared) != observed:
        raise RuntimeError(f"{label} identity drifted")
    return observed


def _validate_authorization(*, root: Path) -> dict[str, Any]:
    prereg_path = resolve_contextworld_path(RECOVERY_PREREG, repo_root=root)
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    if (
        not isinstance(prereg, dict)
        or prereg.get("schema_version") != 1
        or prereg.get("recovery_id") != RECOVERY_ID
        or prereg.get("status")
        != "preregistered_after_finalizer_failure_before_summary_write"
        or prereg.get("observed_failure", {}).get("message") != EXPECTED_FAILURE
        or prereg.get("scope", {}).get("cem_episode_execution_authorized")
        is not False
        or prereg.get("scope", {}).get("receipt_mutation_authorized") is not False
        or prereg.get("completion_contract", {}).get("cem_episodes_rerun_by_recovery")
        != 0
    ):
        raise RuntimeError("CEM finalizer recovery preregistration is invalid")

    authority = prereg.get("original_authority")
    if not isinstance(authority, Mapping):
        raise RuntimeError("CEM finalizer recovery original authority is invalid")
    authority_paths = {
        "preregistration": "configs/benchmark/contextworld_original_baseline_cem_prereg_v1.yaml",
        "pre_execution_freeze": "configs/benchmark/contextworld_original_baseline_cem_freeze_v1.json",
        "original_finalizer": "scripts/finalize_contextworld_original_baseline_cem_v1.py",
    }
    observed_authority = {
        name: _require_identity(
            authority.get(name), logical_path, root=root, label=name
        )
        for name, logical_path in authority_paths.items()
    }

    compatibility = prereg.get("authorized_compatibility_change")
    if (
        not isinstance(compatibility, Mapping)
        or compatibility.get("additive_field") != "query_count"
        or compatibility.get("semantics") != "exact_evaluation_count_alias"
        or compatibility.get("conflicting_aliases_must_fail") is not True
        or compatibility.get("all_other_finalizer_checks_unchanged") is not True
    ):
        raise RuntimeError("CEM finalizer recovery compatibility scope is invalid")

    freeze_path = resolve_contextworld_path(RECOVERY_FREEZE, repo_root=root)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        not isinstance(freeze, dict)
        or freeze.get("schema_version") != 1
        or freeze.get("freeze_id")
        != "contextworld_original_baseline_cem_finalizer_recovery_freeze_v1"
        or freeze.get("recovery_id") != RECOVERY_ID
        or freeze.get("status") != "frozen_before_recovery_summary_write"
        or freeze.get("result_summary_absent_before_freeze") is not True
        or freeze.get("cem_episodes_rerun_before_freeze") != 0
    ):
        raise RuntimeError("CEM finalizer recovery authorization freeze is invalid")

    prereg_identity = _require_identity(
        freeze.get("recovery_preregistration"),
        RECOVERY_PREREG,
        root=root,
        label="recovery preregistration",
    )
    implementation_identity = _require_identity(
        freeze.get("recovery_implementation"),
        Path("scripts") / Path(__file__).name,
        root=root,
        label="recovery implementation",
    )
    if freeze.get("original_authority") != observed_authority:
        raise RuntimeError("CEM finalizer recovery freeze authority drifted")

    summary_logical = prereg["scope"]["result_summary"]["path"]
    summary_path = resolve_contextworld_path(summary_logical, repo_root=root)
    if summary_path.exists() or summary_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite CEM matrix summary: {summary_path}")
    return {
        "recovery_id": RECOVERY_ID,
        "status": "authorized_receipt_only_finalization_recovery",
        "authorization_freeze": _identity(RECOVERY_FREEZE, root=root),
        "recovery_preregistration": prereg_identity,
        "recovery_implementation": implementation_identity,
        "original_failure": {
            "exception_type": "FinalizationError",
            "message": EXPECTED_FAILURE,
            "result_summary_written": False,
        },
        "compatibility_change": {
            "added_evaluation_count_alias": "query_count",
            "conflicting_aliases_rejected": True,
            "all_other_checks_delegated_to_frozen_original_finalizer": True,
        },
        "cem_episodes_rerun": 0,
        "receipts_modified": False,
        "formal_scoreboard_mutated": False,
    }


_ORIGINAL_VALIDATE_AGGREGATE = original._validate_aggregate


def _validate_aggregate_with_query_count(
    aggregate: Any,
    *,
    successes: int,
    evaluations: int,
    label: str,
) -> dict[str, Any]:
    row = original._mapping(aggregate, label=label)
    if "query_count" not in row:
        return _ORIGINAL_VALIDATE_AGGREGATE(
            row, successes=successes, evaluations=evaluations, label=label
        )
    try:
        query_count = int(row["query_count"])
    except (TypeError, ValueError) as error:
        raise original.FinalizationError(f"{label}.query_count is invalid") from error
    for alias in ("evaluation_count", "evaluations"):
        if alias in row and int(row[alias]) != query_count:
            raise original.FinalizationError(
                f"{label} evaluation-count aliases conflict"
            )
    normalized = dict(row)
    normalized.setdefault("evaluation_count", query_count)
    return _ORIGINAL_VALIDATE_AGGREGATE(
        normalized, successes=successes, evaluations=evaluations, label=label
    )


def finalize(
    *,
    prereg_path: Path = original.DEFAULT_PREREG,
    results_root: Path = original.DEFAULT_RESULTS_ROOT,
    cube_lewm_aggregate: Path = original.DEFAULT_CUBE_LEWM_AGGREGATE,
    cube_lewm_freeze_receipt: Path = original.DEFAULT_CUBE_LEWM_FREEZE_RECEIPT,
    output: Path | None = None,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Run the frozen finalizer with only the authorized count alias added."""

    root = repo_root.expanduser().resolve()
    recovery_evidence = _validate_authorization(root=root)
    original_validator = original._validate_aggregate
    original_writer = original._write_exclusive
    original_file = original.__file__

    def _write_with_recovery(path: Path, payload: dict[str, Any]) -> None:
        if "finalization_recovery" in payload:
            raise original.FinalizationError("CEM summary recovery evidence duplicated")
        payload["finalization_recovery"] = recovery_evidence
        original_writer(path, payload)

    original._validate_aggregate = _validate_aggregate_with_query_count
    original._write_exclusive = _write_with_recovery
    original.__file__ = str(Path(__file__).resolve())
    try:
        return original.finalize(
            prereg_path=prereg_path,
            results_root=results_root,
            cube_lewm_aggregate=cube_lewm_aggregate,
            cube_lewm_freeze_receipt=cube_lewm_freeze_receipt,
            output=output,
            repo_root=root,
        )
    finally:
        original._validate_aggregate = original_validator
        original._write_exclusive = original_writer
        original.__file__ = original_file


def main(argv: Sequence[str] | None = None) -> None:
    args = original.parse_args(argv)
    summary = finalize(
        prereg_path=args.prereg,
        results_root=args.results_root,
        cube_lewm_aggregate=args.cube_lewm_aggregate,
        cube_lewm_freeze_receipt=args.cube_lewm_freeze_receipt,
        output=args.output,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "counts": summary["counts"],
                "recovery_id": summary["finalization_recovery"]["recovery_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
