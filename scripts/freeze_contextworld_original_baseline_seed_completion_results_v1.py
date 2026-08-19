#!/usr/bin/env python3
"""Freeze the three-seed original-baseline family results.

Mirrors ``contextworld_original_baseline_cem_results_freeze_v1.json`` one level
up: after the finalizer has written ``family_summary.json`` and the v2 builder
has written ``complete_comparison_v2.json``, this emits the results freeze that
pins the eight baseline families (mean / sample std over training seeds
3072/3073/3074) together with every identity that produced them.

The freeze is derived, never transcribed: each recorded SHA-256 is hashed from
the file at freeze time, and each family row is copied out of the finalizer's
own summary.  Descriptive result freeze only -- no formal-scoreboard mutation,
no cross-environment average, no threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from contextworld.paths import repository_root, resolve_contextworld_path


ROOT = repository_root()
FREEZE_ID = "contextworld_original_baseline_seed_completion_results_freeze_v1"
FAMILY_SUMMARY_ID = "contextworld_original_baseline_seed_completion_family_summary_v1"
COMPARISON_V2_ID = "contextworld_complete_reference_comparison_v2"
TRAINING_SEEDS = (3072, 3073, 3074)

DEFAULT_FAMILY_SUMMARY = Path(
    "artifacts/evaluation/original_baseline_seed_completion_v1/family_summary.json"
)
DEFAULT_COMPARISON_V2 = Path(
    "artifacts/evaluation/complete_reference_comparison_v1/complete_comparison_v2.json"
)
DEFAULT_OUTPUT = Path(
    "configs/benchmark/contextworld_original_baseline_seed_completion_results_freeze_v1.json"
)
AUTHORITY_FILES = {
    "preregistration": Path(
        "configs/benchmark/contextworld_original_baseline_seed_completion_prereg_v1.yaml"
    ),
    "freeze": Path(
        "configs/benchmark/contextworld_original_baseline_seed_completion_freeze_v1.json"
    ),
    "execution_preflight": Path(
        "configs/benchmark/contextworld_original_baseline_seed_completion_preflight_v1.json"
    ),
    "infrastructure_relaunch_recovery": Path(
        "configs/benchmark/original_baseline_seed_completion_tworoom_pldm_seed3074_"
        "eval43_infra_relaunch_recovery_v1.yaml"
    ),
}
IMPLEMENTATION_FILES = {
    "launcher": Path("scripts/run_contextworld_original_baseline_seed_completion_v1.py"),
    "finalizer": Path("scripts/finalize_contextworld_original_baseline_seed_completion_v1.py"),
    "comparison_v2_builder": Path("scripts/build_contextworld_complete_comparison_v2.py"),
    "results_freezer": Path(
        "scripts/freeze_contextworld_original_baseline_seed_completion_results_v1.py"
    ),
}
PARENT_RESULTS_FREEZE = Path(
    "configs/benchmark/contextworld_original_baseline_cem_results_freeze_v1.json"
)


class FreezeError(RuntimeError):
    """Raised when the inputs are not the completed, finalized seed-completion set."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path, *, root: Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return resolve_contextworld_path(value, repo_root=root)


def _identity(path: Path, *, label: str, repo_path: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return {
        "path": repo_path if repo_path is not None else str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FreezeError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise FreezeError(f"{label} must be a mapping")
    return payload


def freeze(
    *,
    frozen_date_utc: str,
    family_summary_path: Path = DEFAULT_FAMILY_SUMMARY,
    comparison_v2_path: Path = DEFAULT_COMPARISON_V2,
    output: Path = DEFAULT_OUTPUT,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    summary_file = _resolve(family_summary_path, root=root)
    summary_identity = _identity(
        summary_file, label="family summary", repo_path=str(family_summary_path)
    )
    summary = _read_json(summary_file, label="family summary")
    if summary.get("summary_id") != FAMILY_SUMMARY_ID:
        raise FreezeError("family summary id drifted")
    if summary.get("status") != "completed_descriptive_original_environment_baseline_families":
        raise FreezeError("family summary is not a completed descriptive result")

    families_raw = summary.get("families")
    if not isinstance(families_raw, list) or len(families_raw) != 8:
        raise FreezeError("family summary must contain exactly eight families")

    families: list[dict[str, Any]] = []
    for raw in families_raw:
        if not isinstance(raw, Mapping):
            raise FreezeError("family summary rows must be mappings")
        members = raw["members"]
        if tuple(int(member["training_seed"]) for member in members) != TRAINING_SEEDS:
            raise FreezeError(
                f"family {raw['environment']}/{raw['family']} does not close 3072/3073/3074"
            )
        statistics = raw["statistics"]
        families.append(
            {
                "environment": str(raw["environment"]),
                "family": str(raw["family"]),
                "members": [
                    {
                        "training_seed": int(member["training_seed"]),
                        # Only newly executed cells carry a cell_id; members
                        # reused from the parent matrix are identified by their
                        # provenance instead.
                        **(
                            {"cell_id": str(member["cell_id"])}
                            if "cell_id" in member
                            else {}
                        ),
                        "success_count": int(member["success_count"]),
                        "evaluation_count": int(member["evaluation_count"]),
                        "success_rate": float(member["success_rate"]),
                        "provenance": str(member["provenance"]),
                    }
                    for member in members
                ],
                "mean_success_rate": float(statistics["mean"]),
                "sample_std": float(statistics["sample_std"]),
                "sample_variance": float(statistics["sample_variance"]),
                "minimum_success_rate": float(statistics["minimum"]),
                "maximum_success_rate": float(statistics["maximum"]),
                "n_training_seeds": int(statistics["n_training_seeds"]),
                "lineage_notes": [dict(note) for note in raw.get("lineage_notes", [])],
            }
        )

    comparison_file = _resolve(comparison_v2_path, root=root)
    comparison_identity = _identity(
        comparison_file, label="complete comparison v2", repo_path=str(comparison_v2_path)
    )
    comparison = _read_json(comparison_file, label="complete comparison v2")
    if comparison.get("comparison_id") != COMPARISON_V2_ID:
        raise FreezeError("complete comparison v2 id drifted")
    if int(comparison.get("row_count", -1)) != 18:
        raise FreezeError("complete comparison v2 row count drifted")
    derived = comparison.get("derived_from_v1")
    if not isinstance(derived, Mapping) or derived.get("v1_file_unmodified") is not True:
        raise FreezeError("complete comparison v2 does not record an unmodified v1")
    baseline_binding = comparison.get("baseline_family_summary")
    if (
        not isinstance(baseline_binding, Mapping)
        or baseline_binding.get("sha256") != summary_identity["sha256"]
    ):
        raise FreezeError("complete comparison v2 is bound to a different family summary")

    authority = {
        name: _identity(_resolve(path, root=root), label=name, repo_path=str(path))
        for name, path in AUTHORITY_FILES.items()
    }
    implementation = {
        name: _identity(_resolve(path, root=root), label=name, repo_path=str(path))
        for name, path in IMPLEMENTATION_FILES.items()
    }
    parent_freeze = _identity(
        _resolve(PARENT_RESULTS_FREEZE, root=root),
        label="parent results freeze",
        repo_path=str(PARENT_RESULTS_FREEZE),
    )

    document = {
        "schema_version": 1,
        "freeze_id": FREEZE_ID,
        "status": "frozen_after_completed_three_training_seed_baseline_families",
        "frozen_date_utc": frozen_date_utc,
        "scientific_status": {
            "kind": "post_release_descriptive_original_environment_baseline_family_freeze",
            "formal_scoreboard_mutated": False,
            "cross_environment_average_authorized": False,
            "cross_environment_average_reported": False,
            "pass_fail_threshold": None,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "public_test_accessed": False,
            "result_based_retry_performed": False,
        },
        "family_summary": {
            **summary_identity,
            "summary_id": FAMILY_SUMMARY_ID,
            "status": summary["status"],
            **{key: value for key, value in dict(summary.get("counts", {})).items()},
        },
        "families": families,
        "downstream": {
            "complete_comparison_v2": {**comparison_identity, "comparison_id": COMPARISON_V2_ID},
            "complete_comparison_v1_unmodified": True,
            "historical_scoreboard_unmodified": True,
        },
        "carried_membership": {
            "parent_results_freeze": parent_freeze,
            "reused_member_cells": int(summary["counts"]["reused_member_cells"]),
            "lineage_note_cells": int(summary["counts"]["lineage_note_cells"]),
            "cem_episodes_rerun_for_reused_cells": 0,
        },
        "execution_disclosures": [
            dict(row) for row in summary.get("execution_disclosures", [])
        ],
        "pre_execution_authority": authority,
        "implementation": implementation,
    }

    target = _resolve(output, root=root) if Path(output).is_absolute() else root / output
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite results freeze: {target}")
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        stream.flush()
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--frozen-date-utc",
        required=True,
        help="UTC date of the freeze, YYYY-MM-DD (recorded verbatim)",
    )
    parser.add_argument("--family-summary", type=Path, default=DEFAULT_FAMILY_SUMMARY)
    parser.add_argument("--comparison-v2", type=Path, default=DEFAULT_COMPARISON_V2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    document = freeze(
        frozen_date_utc=args.frozen_date_utc,
        family_summary_path=args.family_summary,
        comparison_v2_path=args.comparison_v2,
        output=args.output,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "freeze_id": document["freeze_id"],
                "families": [
                    {
                        "environment": family["environment"],
                        "family": family["family"],
                        "mean": family["mean_success_rate"],
                        "sample_std": family["sample_std"],
                    }
                    for family in document["families"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
