#!/usr/bin/env python3
"""Build the v2 complete-comparison table with three-seed baseline columns.

Preregistered by ``contextworld_original_baseline_seed_completion_prereg_v1``
(``downstream_use.complete_comparison_v2``): a NEW file
``complete_comparison_v2.json`` that carries every one of the 18 v1 rows
verbatim and adds, inside each row's ``original_task_cem``, an additive
``family_baseline_v2`` block reporting the family-matched original-recipe
baseline as mean +/- sample std across training seeds 3072/3073/3074 (from the
frozen ``family_summary.json``).  The v1 file, the 13-row historical
scoreboard, and every prior receipt remain byte-identical; the v1
single-checkpoint ``family_baseline_rate`` value stays visible inside each row
as the historical comparator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from contextworld.paths import repository_root, resolve_contextworld_path


ROOT = repository_root()
DEFAULT_V1 = Path(
    "artifacts/evaluation/complete_reference_comparison_v1/complete_comparison.json"
)
# Frozen at v1 publication (see COMPLETE_COMPARISON_REPORT.md, line 3).
EXPECTED_V1_SHA256 = "025085bf6f1f68e8c736ca1ef0e0bab4a0a761ad2df62b385f1a623d17d7d39f"
DEFAULT_FAMILY_SUMMARY = Path(
    "artifacts/evaluation/original_baseline_seed_completion_v1/family_summary.json"
)
EXPECTED_FAMILY_SUMMARY_ID = (
    "contextworld_original_baseline_seed_completion_family_summary_v1"
)
DEFAULT_SEED_COMPLETION_PREREG = Path(
    "configs/benchmark/contextworld_original_baseline_seed_completion_prereg_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "artifacts/evaluation/complete_reference_comparison_v1/complete_comparison_v2.json"
)

COMPONENT_ENVIRONMENT = {
    "speed": "tworoom",
    "door": "tworoom",
    "action_delay": "tworoom",
    "portal_exit": "tworoom",
    "action_strength": "pusht",
    "contact_friction": "pusht",
    "motion_damping": "pusht",
    "robot_arm_mass": "reacher",
    "cube_gripper_carry": "cube",
}


class ComparisonBuildError(RuntimeError):
    """Raised when an input is not the frozen v1/family-summary pair."""


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


def _file_identity(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ComparisonBuildError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise ComparisonBuildError(f"{label} must be a mapping")
    return dict(payload)


def build(
    *,
    v1_path: Path = DEFAULT_V1,
    family_summary_path: Path = DEFAULT_FAMILY_SUMMARY,
    seed_completion_prereg_path: Path = DEFAULT_SEED_COMPLETION_PREREG,
    output: Path = DEFAULT_OUTPUT,
    expected_v1_sha256: str = EXPECTED_V1_SHA256,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    v1_file = _resolve(v1_path, root=root)
    v1_identity = _file_identity(v1_file, label="complete_comparison v1")
    if v1_identity["sha256"] != expected_v1_sha256:
        raise ComparisonBuildError(
            "complete_comparison.json does not match its frozen v1 identity"
        )
    v1 = _read_json(v1_file, label="complete_comparison v1")
    if (
        v1.get("comparison_id") != "contextworld_complete_reference_comparison_v1"
        or int(v1.get("row_count", -1)) != 18
    ):
        raise ComparisonBuildError("v1 comparison id or row count drifted")
    rows = v1.get("rows")
    if not isinstance(rows, list) or len(rows) != 18:
        raise ComparisonBuildError("v1 must contain exactly 18 rows")

    summary_file = _resolve(family_summary_path, root=root)
    summary_identity = _file_identity(summary_file, label="family summary")
    summary = _read_json(summary_file, label="family summary")
    if summary.get("summary_id") != EXPECTED_FAMILY_SUMMARY_ID:
        raise ComparisonBuildError("family summary id drifted")
    families: dict[tuple[str, str], Mapping[str, Any]] = {}
    for family in summary.get("families", ()):
        if not isinstance(family, Mapping):
            raise ComparisonBuildError("family summary families must be mappings")
        families[(str(family["environment"]), str(family["family"]))] = family
    if len(families) != 8:
        raise ComparisonBuildError("family summary must contain exactly eight families")

    prereg_identity = _file_identity(
        _resolve(seed_completion_prereg_path, root=root),
        label="seed-completion preregistration",
    )

    v2_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ComparisonBuildError(f"v1 row {index} must be a mapping")
        row = copy.deepcopy(dict(raw))
        component = str(row.get("component_id", ""))
        environment = COMPONENT_ENVIRONMENT.get(component)
        if environment is None:
            raise ComparisonBuildError(f"unknown component in v1 row {index}: {component}")
        family_key = str(row.get("family", "")).lower()
        family = families.get((environment, family_key))
        if family is None:
            raise ComparisonBuildError(
                f"family summary is missing {environment}/{family_key} for row {index}"
            )
        cem = row.get("original_task_cem")
        if not isinstance(cem, dict):
            raise ComparisonBuildError(f"v1 row {index} original_task_cem is missing")
        if "family_baseline_v2" in cem:
            raise ComparisonBuildError(f"v1 row {index} already carries family_baseline_v2")
        statistics = family["statistics"]
        cem["family_baseline_v2"] = {
            "environment": environment,
            "per_training_seed": [
                {
                    "training_seed": int(member["training_seed"]),
                    "success_count": int(member["success_count"]),
                    "evaluation_count": int(member["evaluation_count"]),
                    "success_rate": float(member["success_rate"]),
                    "provenance": str(member["provenance"]),
                }
                for member in family["members"]
            ],
            "mean": float(statistics["mean"]),
            "sample_std": float(statistics["sample_std"]),
            "sample_variance": float(statistics["sample_variance"]),
            "minimum": float(statistics["minimum"]),
            "maximum": float(statistics["maximum"]),
            "historical_family_baseline_rate_v1": cem.get("family_baseline_rate"),
            "lineage_notes": copy.deepcopy(list(family.get("lineage_notes", []))),
            "source": {
                "path": summary_identity["path"],
                "sha256": summary_identity["sha256"],
            },
        }
        v2_rows.append(row)

    document = {
        "schema_version": 1,
        "comparison_id": "contextworld_complete_reference_comparison_v2",
        "result_kind": "contextworld_complete_reference_comparison_v2",
        "derived_from_v1": {
            **v1_identity,
            "comparison_id": "contextworld_complete_reference_comparison_v1",
            "v1_file_unmodified": True,
            "rows_carried_verbatim_plus_additive_family_baseline_v2": True,
        },
        "baseline_family_summary": {
            **summary_identity,
            "summary_id": EXPECTED_FAMILY_SUMMARY_ID,
        },
        "seed_completion_preregistration": prereg_identity,
        "claim_boundary": {
            **dict(v1.get("claim_boundary", {})),
            "historical_scoreboard_rows_unchanged": True,
            "v1_comparison_file_unmodified": True,
            "baseline_columns_reported_as_three_training_seed_statistics": True,
        },
        "historical_scoreboard": copy.deepcopy(v1.get("historical_scoreboard")),
        "report_all_policy": copy.deepcopy(v1.get("report_all_policy")),
        "row_count": 18,
        "rows": v2_rows,
    }

    if Path(output) == DEFAULT_OUTPUT:
        # Anchor the new file next to the frozen v1 comparison; a plain
        # resolve of the not-yet-existing artifacts path would fall back to
        # the data tree instead of the directory that actually holds v1.
        target = v1_file.parent / DEFAULT_OUTPUT.name
    else:
        target = _resolve(output, root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite v2 comparison: {target}")
    encoded = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--v1", type=Path, default=DEFAULT_V1)
    parser.add_argument("--family-summary", type=Path, default=DEFAULT_FAMILY_SUMMARY)
    parser.add_argument(
        "--seed-completion-prereg", type=Path, default=DEFAULT_SEED_COMPLETION_PREREG
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    document = build(
        v1_path=args.v1,
        family_summary_path=args.family_summary,
        seed_completion_prereg_path=args.seed_completion_prereg,
        output=args.output,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "comparison_id": document["comparison_id"],
                "row_count": document["row_count"],
                "baselines": sorted(
                    {
                        (
                            row["original_task_cem"]["family_baseline_v2"]["environment"],
                            row["family"],
                        )
                        for row in document["rows"]
                    }
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
