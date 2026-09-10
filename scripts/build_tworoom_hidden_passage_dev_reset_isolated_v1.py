#!/usr/bin/env python3
"""Build the reset-isolated History-3 hidden-passage Development catalog.

Second Development replica of the frozen Public Test validation protocol
(``tworoom_hidden_passage_h3_validation_v2``), replacing
``tworoom_hidden_passage_h3_dev_structural_parity_v1`` under the S3
conservative reading: no Development reset tuple may coincide with any Public
Test reset tuple, at *candidate* level.

Why a new entry point: the v1 (loader_val-only) pool cannot satisfy that
requirement — of its 320 candidates, 160 share a reset tuple with the Public
Test candidate templates (149 with the tuples Test actually selected),
leaving only 171 (84 left-to-right / 87 right-to-left), below the 150
candidates per direction the 6 x 2 x 25 stratification needs.  This build
draws instead from the complete frozen formal-scale training geometry
(train 96 + loader_val 16 = 112 door positions, all disjoint from the 42
eval-only Test doors) and drops every template whose reset tuple appears
among the 286 unique Public Test candidate reset tuples, leaving 1120
candidates (560 per direction).

The frozen ``hidden_passage_validation`` module (byte-pinned by
``tworoom_door_icl_release_v1.yaml``) is imported and executed verbatim:
``build_validation_catalog`` and ``select_validation_assignments`` run
unmodified, including the query-pixel dedup, the
``seeded_disjoint_without_replacement_query_sets`` selection, the physics
gates, and every frozen build check.  Only the candidate-pool function
``candidate_templates`` is swapped at runtime for the reset-isolated union
pool below; the swap and the pool composition are recorded in the build
report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation import hidden_passage_validation as validation
from contextworld.evaluation.hidden_passage import make_templates
from contextworld.evaluation.hidden_passage_h3_data import (
    door_splits_for_scale,
    templates_for_door,
)
from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.config import load_config
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_dev_reset_isolated_v1.yaml"
)


def _test_candidate_reset_tuples(
    test_config_path: Path,
) -> tuple[set[tuple[float, ...]], dict[str, Any]]:
    """Reset tuples over ALL frozen Public Test candidate templates."""

    test_config = load_config(test_config_path)
    gen = test_config["data"]["generation"]
    templates = make_templates(
        door_positions=[int(v) for v in gen["candidate_door_positions"]],
        directions=tuple(gen["directions"]),
        doorway_offsets_px=[float(v) for v in gen["doorway_offsets_px"]],
        catalog_seed=int(gen["catalog_seed"]),
    )
    tuples = {tuple(map(float, t.reset_state)) for t in templates}
    return tuples, {
        "test_config": str(test_config_path),
        "test_candidate_templates": len(templates),
        "test_candidate_reset_tuples": len(tuples),
        "test_eval_only_doors": [
            int(v) for v in gen["candidate_door_positions"]
        ],
    }


def _union_pool(
    generation: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """The reset-isolated train ∪ loader_val w00 candidate pool."""

    training_config = load_config(
        resolve_contextworld_path(
            generation["training_data_config"], repo_root=ROOT
        )
    )
    splits = door_splits_for_scale(
        training_config, str(generation["training_scale"])
    )
    split_positions = {
        "train": sorted(map(int, splits.train)),
        "loader_val": sorted(map(int, splits.val)),
    }
    declared_splits = [
        str(value) for value in generation["training_splits"]
    ]
    if sorted(declared_splits) != sorted(split_positions):
        raise ValueError("training_splits must be exactly train ∪ loader_val")
    doors = sorted(
        {
            door
            for name in declared_splits
            for door in split_positions[name]
        }
    )
    declared = sorted(int(v) for v in generation["candidate_door_positions"])
    if declared != doors:
        raise ValueError(
            "candidate_door_positions must equal the complete frozen "
            "train ∪ loader_val union"
        )

    geometry_filter = generation["training_geometry_filter"]
    allowed_wall = {
        int(value) for value in geometry_filter["wall_distance_indices"]
    }
    allowed_offset = {
        int(value)
        for value in geometry_filter["doorway_offset_indices"]
    }

    def selected_geometry(template: Any) -> bool:
        parts = template.template_id.rsplit("-", 2)
        wall_slug, offset_slug = parts[-2:]
        return (
            int(wall_slug[1:]) in allowed_wall
            and int(offset_slug[1:]) in allowed_offset
        )

    raw: list[Any] = []
    for door in doors:
        raw.extend(
            template
            for template in templates_for_door(
                training_config,
                scale=str(generation["training_scale"]),
                door_position=door,
            )
            if selected_geometry(template)
        )
    if len({t.template_id for t in raw}) != len(raw):
        raise ValueError("Union pool contains duplicate template ids")

    isolation = generation["reset_tuple_isolation"]
    test_tuples, test_summary = _test_candidate_reset_tuples(
        resolve_contextworld_path(isolation["test_config"], repo_root=ROOT)
    )
    if len(test_tuples) != int(isolation["expected_test_candidate_reset_tuples"]):
        raise ValueError(
            "Public Test candidate reset-tuple count changed: "
            f"{len(test_tuples)}"
        )
    kept = [
        template
        for template in raw
        if tuple(map(float, template.reset_state)) not in test_tuples
    ]
    blocked = len(raw) - len(kept)
    if len(kept) != int(generation["candidate_templates"]):
        raise ValueError(
            "Reset-isolated union pool size disagrees with the frozen "
            f"config: observed={len(kept)} "
            f"expected={int(generation['candidate_templates'])}"
        )
    by_direction = Counter(t.direction for t in kept)
    by_split = Counter(
        "loader_val"
        if int(t.door_position) in set(split_positions["loader_val"])
        else "train"
        for t in kept
    )
    composition = {
        "doors": {
            "train": split_positions["train"],
            "loader_val": split_positions["loader_val"],
        },
        "raw_union_candidates": len(raw),
        "blocked_by_test_reset_tuple": blocked,
        "kept_candidates": len(kept),
        "kept_by_direction": dict(sorted(by_direction.items())),
        "kept_by_split": dict(sorted(by_split.items())),
        "test_reset_tuple_isolation": test_summary,
    }
    return kept, composition


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the 300-query reset-isolated History-3 hidden-passage "
            "Development catalog (train ∪ loader_val pool, no reset tuple "
            "shared with any Public Test candidate)"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Replace output owned by this same benchmark configuration",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    if config.get("status") != "diagnostic_frozen_before_catalog_generation_and_scoring":
        raise ValueError("Development config is not frozen before scoring")
    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    if stable_commit != str(config["stable_worldmodel"]["commit"]):
        raise RuntimeError(
            "Stable-WorldModel commit mismatch: "
            f"{stable_commit} != {config['stable_worldmodel']['commit']}"
        )

    configured_output = (
        args.output_root
        if args.output_root is not None
        else Path(config["artifacts"]["output_root"])
    )
    output_root = resolve_contextworld_path(
        configured_output, repo_root=ROOT
    )
    if output_root.exists():
        if not args.refresh_existing:
            raise FileExistsError(
                f"Refusing to overwrite Development output {output_root}"
            )
        prior = json.loads(
            (output_root / "build_report.json").read_text(encoding="utf-8")
        )
        if prior.get("benchmark") != config["benchmark"]:
            raise ValueError(
                "Existing output belongs to another benchmark: "
                f"{prior.get('benchmark')!r}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    generation = config["data"]["generation"]
    pool, composition = _union_pool(generation)

    frozen_candidate_templates = validation.candidate_templates

    def reset_isolated_candidates(
        builder_config: dict[str, Any],
        *,
        repo_root: Path | None = None,
    ) -> list[Any]:
        if builder_config is not config:
            raise RuntimeError(
                "The reset-isolated pool builder received a foreign config"
            )
        if len({t.template_id for t in pool}) != len(pool):
            raise ValueError("Candidate template IDs are not unique")
        return list(pool)

    # The frozen selection/catalog pipeline runs verbatim against the
    # reset-isolated pool: only the pool builder is swapped.
    validation.candidate_templates = reset_isolated_candidates
    try:
        catalog, report = validation.build_validation_catalog(
            config=config,
            repo_root=ROOT,
            output_root=output_root,
        )
    finally:
        validation.candidate_templates = frozen_candidate_templates

    catalog_path = output_root / "catalog.json"
    exclusion_path = output_root / "training_exclusion_manifest.json"
    report_path = output_root / "build_report.json"
    write_json(catalog_path, catalog)
    write_json(
        exclusion_path,
        {
            "schema_version": 1,
            "benchmark": config["benchmark"],
            "purpose": (
                "identity manifest for frozen training-seen diagnostic "
                "queries; not a training exclusion"
            ),
            "query_domain": "training_seen",
            "eval_only_door_positions": [
                int(value)
                for value in generation["eval_only_door_positions"]
            ],
            "query_records": catalog["training_exclusion_manifest"],
            "query_count": len(catalog["training_exclusion_manifest"]),
            "content_manifest_sha256": catalog[
                "content_manifest_sha256"
            ],
        },
    )

    # ---- S3 conservative isolation checks (beyond the frozen build checks)
    isolation_spec = generation["reset_tuple_isolation"]
    test_catalog_path = Path(isolation_spec["test_catalog"])
    test_catalog = json.loads(test_catalog_path.read_text(encoding="utf-8"))
    test_selected_tuples = {
        tuple(map(float, b["template"]["reset_state"]))
        for b in test_catalog["bundles"]
    }
    dev_tuples = {
        tuple(map(float, b["template"]["reset_state"]))
        for b in catalog["bundles"]
    }
    test_doors = {
        int(b["template"]["door_position"]) for b in test_catalog["bundles"]
    }
    dev_doors = {
        int(b["template"]["door_position"]) for b in catalog["bundles"]
    }
    test_template_ids = {
        b["template"]["template_id"] for b in test_catalog["bundles"]
    }
    dev_template_ids = {
        b["template"]["template_id"] for b in catalog["bundles"]
    }
    test_query_hashes = {
        b["query_pixels_sha256"] for b in test_catalog["bundles"]
    }
    dev_query_hashes = {
        b["query_pixels_sha256"] for b in catalog["bundles"]
    }
    test_candidate_tuples, _ = _test_candidate_reset_tuples(
        resolve_contextworld_path(isolation_spec["test_config"], repo_root=ROOT)
    )
    selected_by_split = Counter(
        "loader_val"
        if int(b["template"]["door_position"])
        in set(composition["doors"]["loader_val"])
        else "train"
        for b in catalog["bundles"]
    )
    isolation_checks = {
        "dev_reset_tuples_disjoint_from_test_candidates": not (
            dev_tuples & test_candidate_tuples
        ),
        "dev_reset_tuples_disjoint_from_test_selected": not (
            dev_tuples & test_selected_tuples
        ),
        "dev_doors_disjoint_from_test_doors": not (dev_doors & test_doors),
        "dev_template_ids_disjoint_from_test": not (
            dev_template_ids & test_template_ids
        ),
        "dev_query_pixel_hashes_disjoint_from_test": not (
            dev_query_hashes & test_query_hashes
        ),
    }
    if not all(isolation_checks.values()):
        raise RuntimeError(
            "Reset-isolated Development build failed isolation: "
            f"{[k for k, v in isolation_checks.items() if not v]}"
        )
    report["reset_isolation"] = {
        "pool_composition": composition,
        "selected_queries_by_split": dict(sorted(selected_by_split.items())),
        "checks": isolation_checks,
        "counts": {
            "dev_selected_reset_tuples": len(dev_tuples),
            "test_selected_reset_tuples": len(test_selected_tuples),
            "test_candidate_reset_tuples": len(test_candidate_tuples),
        },
    }
    report["identity"] = {
        "config": portable_contextworld_path(config_path, repo_root=ROOT),
        "config_sha256": validation.file_sha256(config_path),
        "stable_worldmodel_repo": str(stable_repo),
        "stable_worldmodel_commit": stable_commit,
        "catalog": portable_contextworld_path(catalog_path, repo_root=ROOT),
        "catalog_sha256": validation.file_sha256(catalog_path),
        "training_exclusion_manifest": portable_contextworld_path(
            exclusion_path, repo_root=ROOT
        ),
        "training_exclusion_manifest_sha256": validation.file_sha256(
            exclusion_path
        ),
        "frozen_pool_builder_swapped_for": (
            "reset_isolated_train_union_loader_val_pool_v1"
        ),
    }
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "status": report["status"],
                "pool_composition": {
                    k: v
                    for k, v in composition.items()
                    if k != "test_reset_tuple_isolation"
                },
                "selected_queries_by_split": dict(selected_by_split),
                "isolation_checks": isolation_checks,
                "content_manifest_sha256": report["content_manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
