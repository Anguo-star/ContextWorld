from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contextworld.benchmarks.contact_friction_icl_data import (
    _pusht_physics_state_max_abs_gap,
)
from contextworld.evaluation.pusht_contact_friction_h3 import (
    make_stratified_contact_friction_training_template,
    stratified_contact_friction_training_coordinates,
)
from scripts.build_pusht_contact_friction_h3_data import (
    directory_file_receipts,
    directory_sha256,
    reuse_evaluation_split,
    training_coverage_comparison,
)


def test_8192_training_pairs_balance_every_registered_stratum() -> None:
    counts: dict[tuple[int, int, int, int], int] = {}
    for pair_index in range(8192):
        row = stratified_contact_friction_training_coordinates(pair_index)
        key = (
            row["family_id"],
            row["angle_bin"],
            row["translation_x_bin"],
            row["translation_y_bin"],
        )
        counts[key] = counts.get(key, 0) + 1

    assert len(counts) == 2 * 16 * 8 * 8
    assert set(counts.values()) == {4}


def test_serialized_physics_gap_wraps_body_angles_at_pi_boundary() -> None:
    left = np.zeros((2, 12), dtype=np.float32)
    right = np.zeros((2, 12), dtype=np.float32)
    left[:, 4] = np.pi
    right[:, 4] = -np.pi
    left[:, 10] = -np.pi
    right[:, 10] = np.pi
    assert _pusht_physics_state_max_abs_gap(left, right) < 1.0e-6

    right[1, 6] = 2.0e-4
    assert np.isclose(
        _pusht_physics_state_max_abs_gap(left, right),
        2.0e-4,
        atol=1.0e-8,
        rtol=0.0,
    )


def test_stratified_template_is_deterministic_and_retry_specific() -> None:
    first = make_stratified_contact_friction_training_template(
        pair_index=73,
        attempt_index=0,
        catalog_seed=20260801,
    )
    replay = make_stratified_contact_friction_training_template(
        pair_index=73,
        attempt_index=0,
        catalog_seed=20260801,
    )
    retry = make_stratified_contact_friction_training_template(
        pair_index=73,
        attempt_index=1,
        catalog_seed=20260801,
    )

    assert first == replay
    assert first.template_id != retry.template_id
    assert first.strict_family_id == retry.strict_family_id


def test_evaluation_reuse_preserves_every_file_byte_for_byte(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    table = source / "validation.lance"
    table.mkdir(parents=True)
    (table / "data.bin").write_bytes(b"frozen-public-bytes")
    (table / "metadata.json").write_text("{}\n", encoding="utf-8")
    table_hash = directory_sha256(table)
    manifest = {
        "splits": {
            "validation": {
                "pair_count": 256,
                "table_path": "validation.lance",
                "table_sha256": table_hash,
            }
        }
    }
    (source / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    destination = tmp_path / "destination"
    destination.mkdir()

    report = reuse_evaluation_split(
        source_root=source,
        destination_root=destination,
        split="validation",
        expected_pairs=256,
    )

    assert report["frozen_split_reuse"]["passed"] is True
    assert report["frozen_split_reuse"]["model_visible_bytes_preserved"]
    assert directory_file_receipts(table) == directory_file_receipts(
        destination / "validation.lance"
    )
    assert directory_sha256(destination / "validation.lance") == table_hash


def test_training_coverage_requires_four_balanced_stratum_cycles() -> None:
    source = {
        "pair_count": 2048,
        "strict_family_counts": {"0": 1024, "1": 1024},
        "orientation_bin_counts": {str(index): 256 for index in range(8)},
        "position_bin_counts": {"p0": 1024, "p1": 1024},
        "query_hashes": ["old-query"],
        "template_ids": ["old-template"],
    }
    expanded = {
        "pair_count": 8192,
        "strict_family_counts": {"0": 4096, "1": 4096},
        "orientation_bin_counts": {str(index): 1024 for index in range(8)},
        "position_bin_counts": {"p0": 4096, "p1": 4096},
        "query_hashes": ["new-query"],
        "template_ids": ["new-template"],
        "stratified_training": True,
        "training_stratum_count": 2048,
        "training_stratum_minimum_count": 4,
        "training_stratum_maximum_count": 4,
        "training_strata_balanced": True,
    }

    report = training_coverage_comparison(
        expanded=expanded,
        source=source,
    )

    assert report["passed"] is True
    assert report["pair_count_multiplier"] == 4.0
    assert report["stratification"]["observed_strata"] == 2048
