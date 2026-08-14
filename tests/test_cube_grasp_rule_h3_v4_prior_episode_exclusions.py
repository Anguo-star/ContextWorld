from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.freeze_cube_grasp_rule_h3_v4_prior_episode_exclusions as freeze


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _build_report(path: Path, train: list[int], development: list[int]) -> None:
    _write_json(
        path,
        {
            "protocol": freeze.V3_PROTOCOL,
            "passed": True,
            "active_splits": list(freeze.ACTIVE_SPLITS),
            "splits": {
                "train": {
                    "passed": True,
                    "pair_count": len(train),
                    "source_episodes": train,
                },
                "loader_validation": {
                    "passed": True,
                    "pair_count": len(development),
                    "source_episodes": development,
                },
            },
        },
    )


def test_episode_digest_is_sorted_unique_and_newline_terminated() -> None:
    expected = hashlib.sha256(b"2\n5\n9\n").hexdigest()
    assert freeze.episode_ids_sha256([9, 2, 5, 2]) == expected


def test_report_reader_rejects_cross_split_episode_reuse(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    _build_report(report, [1, 2], [2, 3])
    with pytest.raises(RuntimeError, match="crosses splits"):
        freeze._formal_or_smoke_episodes(report, expected_pair_count=4)


def test_freeze_unions_formal_and_pilot_and_requires_smoke_subset(
    tmp_path: Path,
) -> None:
    formal = tmp_path / "formal.json"
    smoke = tmp_path / "smoke.json"
    _build_report(formal, list(range(2048)), list(range(2048, 2304)))
    _build_report(smoke, list(range(32)), list(range(2048, 2056)))

    pilot = tmp_path / "pilot.json"
    pilot_rows = [
        {"source_episode": value, "candidate_id": f"pilot-{value}"}
        for value in range(3000, 3016)
    ]
    _write_json(
        pilot,
        {
            "schema_version": 1,
            "role": "nonformal_v4_design_feasibility_not_a_frozen_gate",
            "scope": {
                "sample_count": 16,
                "new_scene_and_selection_seed": 7,
                "public_test_opened_read_hashed_or_scored": False,
                "reference_model_training_or_scoring": False,
            },
            "design": {"couplings_n": [0.3, 0.4]},
            "variants": {
                "a": {"rows_without_feature_vectors": pilot_rows},
                "b": {"rows_without_feature_vectors": pilot_rows},
            },
        },
    )
    diagnostic = tmp_path / "diagnostic.json"
    _write_json(
        diagnostic,
        {
            "status": "completed_exploratory_diagnostic",
            "diagnostic_id": "fixture",
            "scope": {
                "old_development_reused_for_exploratory_design": True,
                "reference_model_training_or_scoring": False,
                "public_test": {
                    "opened": False,
                    "read": False,
                    "hashed": False,
                    "scored": False,
                },
            },
        },
    )
    v3_freeze = tmp_path / "v3-freeze.json"
    _write_json(
        v3_freeze,
        {
            "protocol_id": freeze.V3_PROTOCOL,
            "checks_passed": True,
            "source_h5": {
                "symbol": "source",
                "sha256": "a" * 64,
                "size_bytes": 1,
                "row_count": 2,
                "episode_count": 4000,
            },
        },
    )
    output = tmp_path / "receipt.json"
    receipt = freeze.freeze(
        formal_v3_report=formal,
        v3_smoke_reports=[smoke],
        coupling_pilot=pilot,
        exploratory_diagnostic=diagnostic,
        v3_freeze_receipt=v3_freeze,
        output=output,
    )
    assert receipt["excluded_source_episode_count"] == 2320
    assert receipt["excluded_source_episodes"][:2] == [0, 1]
    assert receipt["excluded_source_episodes"][-2:] == [3014, 3015]
    assert receipt["components"][
        "smoke_source_episodes_are_subsets_of_formal_v3"
    ]
    assert receipt["public_test"]["read"] is False
    with pytest.raises(FileExistsError):
        freeze.freeze(
            formal_v3_report=formal,
            v3_smoke_reports=[smoke],
            coupling_pilot=pilot,
            exploratory_diagnostic=diagnostic,
            v3_freeze_receipt=v3_freeze,
            output=output,
        )


def test_cli_requires_all_non_smoke_inputs() -> None:
    with pytest.raises(SystemExit):
        freeze.parse_args([])
