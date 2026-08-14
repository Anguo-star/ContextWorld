from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from contextworld.evaluation.cube_grasp_rule_h3_v4 import (
    V4_ACTION_ANCHORS,
    V4_ANCHOR_GRIPPER_ACTIONS,
    V4_ANCHOR_PROFILES,
    V4_FORMAL_CATALOG_INDEX_OFFSET,
    V4R1_FORMAL_CATALOG_INDEX_OFFSET,
)
from scripts import audit_cube_grasp_rule_h3_v3_action_support as v3_audit
from scripts import audit_cube_grasp_rule_h3_v4_action_support as audit


_NEAREST = {
    "endpoint4": 0.28668928146362305,
    "plateau": 0.2924521565437317,
    "ramp4": 0.29142388701438904,
    "front_hold": 0.29634425044059753,
}
_SUPPORT_COUNTS = {
    "endpoint4": 19756,
    "plateau": 18459,
    "ramp4": 9857,
    "front_hold": 8857,
}
_METRIC_MEAN = np.asarray(
    [
        0.010884696617722511,
        -0.003141433000564575,
        0.002646582666784525,
        0.00042392866453155875,
        0.1592525690793991,
    ],
    dtype=np.float32,
)
_METRIC_STD = np.asarray(
    [
        0.28941982984542847,
        0.393716961145401,
        0.6431365013122559,
        0.3928016126155853,
        0.2503073513507843,
    ],
    dtype=np.float32,
)


def _fake_original_h5_stats(_path: Path | str) -> audit.OriginalH5ActionStats:
    return audit.OriginalH5ActionStats(
        file_size_bytes=40_200_000,
        dataset="action",
        dataset_shape=(2_010_000, 5),
        dataset_dtype="float32",
        dataset_chunks=(1_000, 5),
        total_rows=2_010_000,
        finite_action_rows=2_000_000,
        excluded_nonfinite_rows=10_000,
        population_mean_float64=tuple(float(value) for value in _METRIC_MEAN),
        population_std_float64=tuple(float(value) for value in _METRIC_STD),
        metric_mean_float32=tuple(float(value) for value in _METRIC_MEAN),
        metric_std_population_float32=tuple(
            float(value) for value in _METRIC_STD
        ),
        gripper_maximum=0.9075843095779419,
        action_dataset_sha256=audit.FROZEN_ORIGINAL_H5_ACTION_DATASET_SHA256,
        finite_action_content_sha256=(
            audit.FROZEN_ORIGINAL_H5_FINITE_ACTION_CONTENT_SHA256
        ),
        population_std_float64_sha256="c" * 64,
        metric_std_population_float32_sha256=(
            audit.FROZEN_ORIGINAL_H5_METRIC_STD_POPULATION_FLOAT32_SHA256
        ),
    )


def _anchor_evidence_payload(
    *,
    nearest_override: float | None = None,
    data_read: list[str] | None = None,
) -> dict[str, object]:
    templates = {
        f"fixture_{anchor_id}": {
            "action_sequence": (
                "z=[p,-p,p], gripper=constant, xyz_rot_other_axes=0"
            ),
            "profile": [
                float(value) for value in V4_ANCHOR_PROFILES[anchor_id]
            ],
            "gripper": float(V4_ANCHOR_GRIPPER_ACTIONS[anchor_id]),
            "support_nearest_nrmse": (
                _NEAREST[anchor_id]
                if nearest_override is None
                else nearest_override
            ),
            "support_count_nrmse_le_0p5": _SUPPORT_COUNTS[anchor_id],
        }
        for anchor_id in V4_ACTION_ANCHORS
    }
    templates["v2_control"] = {
        "profile": [1.0, -1.0, 0.0, 0.0, 0.0],
        "gripper": 1.0,
        "support_nearest_nrmse": 0.9893301129341125,
        "support_count_nrmse_le_0p5": 0,
    }
    return {
        "all_causal_audits_passed": True,
        "scope": {
            "data_read": data_read
            if data_read is not None
            else [
                "original cube_single_expert.h5",
                "formal Cube Training train.lance source_row metadata only",
            ],
            "explicitly_not_read": [
                "loader_validation.lance",
                "validation.lance / Public Test",
            ],
        },
        "support_definition": {
            "joint": (
                "nearest standardized RMSE over all 75 values in every "
                "finite same-episode 15-step H5 action window"
            ),
            "same_episode_windows": 1_860_000,
        },
        "templates": templates,
    }


def _variant(
    *,
    height: float,
    rgb_median: float,
    feature_median: float,
) -> dict[str, object]:
    return {
        "pair_count": 16,
        "all_initial_pixels_equal": True,
        "all_query_pixels_equal": True,
        "all_actions_equal": True,
        "all_continuous": True,
        "all_exact_profile_constraints": True,
        "all_pair_checks_passed": True,
        "maximum_query_simulator_state_gap": 7.0e-17,
        "history_height_gap_m": {
            "minimum": height,
            "median": height,
        },
        "history_changed_rgb_values": {
            "minimum": rgb_median - 100.0,
            "median": rgb_median,
        },
        "feature_effect_rms": {
            "minimum": feature_median - 0.1,
            "median": feature_median,
        },
    }


def _coupling_evidence_payload(
    *, public_opened: bool = False
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": "nonformal_v4_design_feasibility_not_a_frozen_gate",
        "scope": {
            "history": 3,
            "new_scene_and_selection_seed": 2026081207,
            "public_test_opened_read_hashed_or_scored": public_opened,
            "reference_model_training_or_scoring": False,
            "repository_modified": False,
            "sample_count": 16,
            "source": "new Training H5 eligible episodes",
        },
        "design": {
            "only_variable": (
                "can_hold vertical force coupling in newtons per normalized "
                "z command"
            ),
            "couplings_n": [0.30, 0.40, 0.45, 0.50],
            "unchanged": [
                "v3 four anchor action profiles",
                "paired x0",
                "bitwise paired actions",
                "continuous [p,-p,p] trajectory",
                "sum(p)=0, p[-1]=0, moment(p)=1",
            ],
        },
        "variants": {
            "coupling_0.30_n": _variant(
                height=0.009450605,
                rgb_median=349.0,
                feature_median=0.6614,
            ),
            "coupling_0.40_n": _variant(
                height=0.012600806,
                rgb_median=431.0,
                feature_median=0.8854,
            ),
        },
    }


def _write_json(
    tmp_path: Path, name: str, payload: dict[str, object]
) -> tuple[Path, str]:
    path = tmp_path / name
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _failed_attempt_payload() -> dict[str, object]:
    values = sorted(
        hashlib.sha256(f"failed-v4-profile:{index}".encode()).hexdigest()
        for index in range(2048)
    )
    return {
        "schema_version": 1,
        "protocol_id": "cube_gripper_carry_rule_history3_development_v4",
        "status": "infrastructure_failed_immutable_attempt",
        "formal_build_attempt_consumed": True,
        "checks_passed": True,
        "build_passed": False,
        "scope": {
            "public_test": {
                "access_status": "closed_not_read_not_scored",
                "opened": False,
                "read": False,
                "hashed": False,
                "scored": False,
            },
            "reference_model_training_or_scoring": False,
            "optimizer_steps": 0,
        },
        "failed_attempt_content": {
            "split": "train",
            "pair_count": 2048,
            "catalog_index_start_inclusive": 1_000_000,
            "catalog_index_stop_exclusive": 1_002_048,
            "prior_content_exclusions": {
                "action_profile_ids": {
                    "values": values,
                    "count": len(values),
                    "sha256": audit._canonical_content_digest(
                        values, field_name="action_profile_ids"
                    ),
                }
            },
        },
    }


def _install_fixture_evidence_hashes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    anchor_sha256: str,
    coupling_sha256: str,
    failed_sha256: str,
) -> None:
    monkeypatch.setattr(
        audit,
        "FROZEN_V3_ANCHOR_SUPPORT_EVIDENCE_SHA256",
        anchor_sha256,
    )
    monkeypatch.setattr(
        audit,
        "FROZEN_V4_COUPLING_FEASIBILITY_SHA256",
        coupling_sha256,
    )
    monkeypatch.setattr(
        audit,
        "FROZEN_V4_FAILED_FORMAL_ATTEMPT_RECEIPT_SHA256",
        failed_sha256,
    )


def test_v4_reuses_exact_v3_source_and_action_identities() -> None:
    assert audit.FROZEN_ORIGINAL_H5_FILE_SHA256 == (
        "0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625"
    )
    assert audit.FROZEN_ORIGINAL_H5_ACTION_DATASET_SHA256 == (
        v3_audit.FROZEN_ORIGINAL_H5_ACTION_DATASET_SHA256
    )
    assert audit.FROZEN_ORIGINAL_H5_FINITE_ACTION_CONTENT_SHA256 == (
        v3_audit.FROZEN_ORIGINAL_H5_FINITE_ACTION_CONTENT_SHA256
    )
    assert audit.FROZEN_ORIGINAL_H5_METRIC_STD_POPULATION_FLOAT32_SHA256 == (
        v3_audit.FROZEN_ORIGINAL_H5_METRIC_STD_POPULATION_FLOAT32_SHA256
    )
    assert audit.FROZEN_V3_ANCHOR_SUPPORT_EVIDENCE_SHA256 == (
        "20dd1f7f629f569719886360c6ffca004a44df17e8632e55f7d37a1c400ed055"
    )
    assert audit.FROZEN_V4_COUPLING_FEASIBILITY_SHA256 == (
        "b9050fb203904bbc0dc8aec2c32e5b950567b1014cb91fb923338f1979cacad7"
    )


def test_standardized_joint_metric_remains_the_v3_all_75_value_metric() -> None:
    left = np.zeros((15, 5), dtype=np.float64)
    right = left.copy()
    right[:, 2] = _METRIC_STD[2]
    expected = np.sqrt(15.0 / 75.0)
    assert audit.standardized_joint_15_step_nrmse(
        left,
        right,
        action_std_population=_METRIC_STD,
    ) == pytest.approx(expected)
    assert audit.standardized_joint_15_step_nrmse is (
        v3_audit.standardized_joint_15_step_nrmse
    )


def test_original_h5_action_fixture_is_streamed_without_other_datasets(
    tmp_path: Path,
) -> None:
    actions = np.asarray(
        [
            [-1.0, -2.0, -3.0, -4.0, -0.5],
            [0.0, 1.0, 2.0, 3.0, 0.25],
            [1.0, 2.0, 4.0, 6.0, 0.75],
            [2.0, 4.0, 8.0, 9.0, 0.5],
            [np.nan, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, np.inf, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    source = tmp_path / "original_fixture.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("action", data=actions, chunks=(2, 5))
        handle.create_dataset("public_pixels_must_not_be_read", data=[1])

    stats = audit.compute_original_h5_action_population_stats(
        source, chunk_rows=2
    )
    finite = actions[np.isfinite(actions).all(axis=1)].astype(np.float64)
    assert stats.total_rows == 6
    assert stats.finite_action_rows == 4
    assert stats.excluded_nonfinite_rows == 2
    assert stats.population_mean_float64 == pytest.approx(
        np.mean(finite, axis=0)
    )
    assert stats.population_std_float64 == pytest.approx(
        np.std(finite, axis=0, ddof=0)
    )
    assert stats.gripper_maximum == pytest.approx(0.75)


def test_coupling_feasibility_is_only_a_0_40_selection_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write_json(
        tmp_path,
        "coupling_feasibility.json",
        _coupling_evidence_payload(),
    )
    monkeypatch.setattr(
        audit, "FROZEN_V4_COUPLING_FEASIBILITY_SHA256", digest
    )

    receipt = audit._load_coupling_feasibility(
        path, expected_sha256=digest
    )

    assert receipt["passed"]
    assert receipt["selected_vertical_force_coupling_n"] == 0.40
    assert receipt["scientific_role"] == (
        "select_v4_vertical_force_coupling_only"
    )
    assert not receipt[
        "action_support_distance_or_window_count_contribution"
    ]
    assert receipt["selected_history_height_gap_m_minimum"] == pytest.approx(
        0.012600806
    )


def test_formal_audit_enumerates_all_4608_v4r1_candidate_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor_path, anchor_sha = _write_json(
        tmp_path,
        "action_template_feasibility_input.json",
        _anchor_evidence_payload(),
    )
    coupling_path, coupling_sha = _write_json(
        tmp_path,
        "coupling_feasibility.json",
        _coupling_evidence_payload(),
    )
    failed_path, failed_sha = _write_json(
        tmp_path,
        "failed_formal_attempt_receipt.json",
        _failed_attempt_payload(),
    )
    _install_fixture_evidence_hashes(
        monkeypatch,
        anchor_sha256=anchor_sha,
        coupling_sha256=coupling_sha,
        failed_sha256=failed_sha,
    )
    monkeypatch.setattr(
        audit,
        "compute_original_h5_action_population_stats",
        _fake_original_h5_stats,
    )

    result = audit.audit_action_support(
        anchor_path,
        feasibility_evidence_sha256=anchor_sha,
        coupling_feasibility=coupling_path,
        coupling_feasibility_sha256=coupling_sha,
        failed_formal_attempt_receipt=failed_path,
        failed_formal_attempt_receipt_sha256=failed_sha,
        original_h5=tmp_path / "original.h5",
    )

    assert result["passed"]
    assert result["status"] == "passed"
    assert result["protocol"] == (
        "cube_gripper_carry_rule_history3_development_v4"
    )
    assert result["scope"]["profile_split_seeds"] == {
        "train": 2026081201,
        "loader_validation": 2026081202,
    }
    assert result["scope"]["total_concrete_profiles"] == 4608
    namespace = result["scope"]["formal_catalog_namespace"]
    assert V4_FORMAL_CATALOG_INDEX_OFFSET == 1_000_000
    assert V4R1_FORMAL_CATALOG_INDEX_OFFSET == 2_000_000
    assert namespace == {
        "catalog_index_offset": 2_000_000,
        "local_index_policy": audit.FORMAL_CATALOG_LOCAL_INDEX_POLICY,
        "catalog_index_formula": (
            "FORMAL_CATALOG_INDEX_OFFSET + local_index"
        ),
        "offset_positive": True,
        "offset_modulo_anchor_count": 0,
        "prior_catalog_namespaces_excluded": [
            {"start_inclusive": 0, "stop_exclusive": 2},
            {"start_inclusive": 1_000_000, "stop_exclusive": 1_002_048},
        ],
        "per_split_ranges": {
            "train": {
                "local_index_start_inclusive": 0,
                "local_index_stop_exclusive": 4096,
                "catalog_index_start_inclusive": 2_000_000,
                "catalog_index_stop_exclusive": 2_004_096,
            },
            "loader_validation": {
                "local_index_start_inclusive": 0,
                "local_index_stop_exclusive": 512,
                "catalog_index_start_inclusive": 2_000_000,
                "catalog_index_stop_exclusive": 2_000_512,
            },
        },
    }
    assert result["scope"]["lance_tables_opened"] == []
    assert result["scope"]["public_test_inputs"] == []
    assert not result["scope"]["public_test_opened"]
    assert result["splits"]["train"]["profile_count"] == 4096
    assert result["splits"]["loader_validation"]["profile_count"] == 512
    assert result["splits"]["train"]["local_index_range"] == {
        "minimum": 0,
        "maximum": 4095,
    }
    assert result["splits"]["train"]["catalog_index_range"] == {
        "minimum": 2_000_000,
        "maximum": 2_004_095,
    }
    assert result["splits"]["loader_validation"][
        "catalog_index_range"
    ] == {"minimum": 2_000_000, "maximum": 2_000_511}
    assert result["splits"]["train"]["action_anchor_counts"] == {
        anchor: 1024 for anchor in V4_ACTION_ANCHORS
    }
    assert result["splits"]["loader_validation"]["action_anchor_counts"] == {
        anchor: 128 for anchor in V4_ACTION_ANCHORS
    }
    assert result["cross_split"]["profile_content_overlap"]["count"] == 0
    assert result["cross_split"]["anchor_family_overlap"]["count"] == 4
    assert result["failed_v4_attempt_exclusion"]["overlap_count"] == 0
    assert result["failed_v4_attempt_exclusion"]["passed"]
    assert result["overall"]["profile_count"] == 4608
    assert result["overall"]["unique_profile_count"] == 4608
    assert result["overall"]["conservatively_supported_profile_count"] == 4608
    assert result["checks"][
        "formal_catalog_offset_positive_and_four_aligned"
    ]
    assert result["checks"][
        "formal_catalog_indices_equal_offset_plus_local_index"
    ]
    assert result["checks"][
        "formal_catalog_indices_disjoint_from_preformal_zero_and_one"
    ]
    assert result["overall"]["maxima"][
        "conservative_original_h5_nrmse_upper_bound"
    ] <= 0.5
    assert result["overall"]["maxima"]["absolute_probe_sum_residual"] == 0.0
    assert result["overall"]["maxima"]["absolute_probe_last_residual"] == 0.0
    assert result["overall"]["maxima"]["absolute_probe_moment_residual"] == 0.0
    assert result["overall"]["maxima"]["action_absolute_maximum"] <= 1.0
    assert result["overall"]["maxima"]["gripper_action"] <= (
        result["gates"]["original_h5_gripper_maximum"]
    )
    assert result["support_counts"][
        "anchor_h5_windows_at_or_below_nrmse_0p5"
    ] == _SUPPORT_COUNTS
    coupling_receipt = result["evidence_roles"]["coupling_selection"]
    anchor_receipt = result["evidence_roles"][
        "anchor_original_h5_support"
    ]
    assert not coupling_receipt[
        "action_support_distance_or_window_count_contribution"
    ]
    assert anchor_receipt["reused_from_v3"]
    assert "unchanged" in anchor_receipt["reuse_is_valid_because"]
    for anchor in V4_ACTION_ANCHORS:
        assert result["anchors"][anchor]["profile_count"] == 1152


def test_canonical_anchor_sha_is_required_before_profile_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor_path, _ = _write_json(
        tmp_path,
        "anchor.json",
        _anchor_evidence_payload(),
    )

    def forbidden_factory(*, split: str, catalog_index: int) -> object:
        raise AssertionError((split, catalog_index))

    monkeypatch.setattr(audit, "make_v4_action_profile", forbidden_factory)
    with pytest.raises(ValueError, match="must equal the frozen canonical"):
        audit.audit_action_support(
            anchor_path,
            feasibility_evidence_sha256="0" * 64,
            coupling_feasibility=tmp_path / "must_not_be_read.json",
            failed_formal_attempt_receipt=(
                tmp_path / "must_not_be_read_failed.json"
            ),
            original_h5=tmp_path / "must_not_be_read.h5",
        )


def test_coupling_evidence_public_contamination_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write_json(
        tmp_path,
        "coupling_public.json",
        _coupling_evidence_payload(public_opened=True),
    )
    monkeypatch.setattr(
        audit, "FROZEN_V4_COUPLING_FEASIBILITY_SHA256", digest
    )
    with pytest.raises(ValueError, match="selection contract"):
        audit._load_coupling_feasibility(path, expected_sha256=digest)


@pytest.mark.parametrize(
    "contaminated_read",
    [
        "loader_validation.lance",
        "validation.lance",
        "decoded Public Test profiles",
    ],
)
def test_anchor_evidence_public_or_development_reads_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contaminated_read: str,
) -> None:
    path, digest = _write_json(
        tmp_path,
        "anchor_contaminated.json",
        _anchor_evidence_payload(data_read=["original H5", contaminated_read]),
    )
    monkeypatch.setattr(
        audit, "FROZEN_V3_ANCHOR_SUPPORT_EVIDENCE_SHA256", digest
    )
    with pytest.raises(ValueError, match="Development/Public data reads"):
        audit._load_anchor_support_evidence(path, expected_sha256=digest)


def test_conservative_triangle_upper_bound_remains_a_hard_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor_path, anchor_sha = _write_json(
        tmp_path,
        "anchor_far.json",
        _anchor_evidence_payload(nearest_override=0.5),
    )
    coupling_path, coupling_sha = _write_json(
        tmp_path,
        "coupling.json",
        _coupling_evidence_payload(),
    )
    failed_path, failed_sha = _write_json(
        tmp_path,
        "failed_formal_attempt_receipt.json",
        _failed_attempt_payload(),
    )
    _install_fixture_evidence_hashes(
        monkeypatch,
        anchor_sha256=anchor_sha,
        coupling_sha256=coupling_sha,
        failed_sha256=failed_sha,
    )
    monkeypatch.setattr(
        audit,
        "compute_original_h5_action_population_stats",
        _fake_original_h5_stats,
    )

    result = audit.audit_action_support(
        anchor_path,
        coupling_feasibility=coupling_path,
        failed_formal_attempt_receipt=failed_path,
        original_h5=tmp_path / "original.h5",
    )

    assert not result["passed"]
    assert result["status"] == "failed"
    assert result["overall"]["conservatively_supported_profile_count"] < 4608
    assert result["checks"][
        "all_profile_invariants_and_action_bounds_passed"
    ]
    assert not result["checks"][
        "all_profiles_conservatively_within_original_h5_support_gate"
    ]


@pytest.mark.parametrize(
    "option",
    [
        "--validation",
        "--validation-lance",
        "--validation-pairs",
        "--public-test",
        "--public-test-pairs",
        "--test-pairs",
    ],
)
def test_cli_rejects_all_validation_and_public_input_options(option: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        audit.parse_args(
            [
                "--feasibility-evidence",
                "anchor.json",
                "--coupling-feasibility",
                "coupling.json",
                "--failed-formal-attempt-receipt",
                "failed.json",
                "--original-h5",
                "original.h5",
                "--output",
                "audit.json",
                option,
                "anything",
            ]
        )


@pytest.mark.parametrize(
    "component",
    ("validation", "validation.lance", "public", "public_test", "public-test"),
)
def test_input_path_gate_rejects_public_components_before_read(
    tmp_path: Path, component: str
) -> None:
    path = tmp_path / component / "evidence.json"
    with pytest.raises(ValueError, match="forbidden Public path component"):
        audit._validated_nonpublic_input_path(path, field="fixture")


def test_input_path_gate_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        audit._validated_nonpublic_input_path(alias, field="fixture")


def test_cli_uses_exclusive_create_for_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("sentinel\n", encoding="utf-8")

    def fake_audit(
        feasibility_evidence: Path,
        *,
        coupling_feasibility: Path,
        failed_formal_attempt_receipt: Path,
        original_h5: Path,
        feasibility_evidence_sha256: str | None,
        coupling_feasibility_sha256: str | None,
        failed_formal_attempt_receipt_sha256: str | None,
    ) -> dict[str, object]:
        del (
            feasibility_evidence,
            coupling_feasibility,
            failed_formal_attempt_receipt,
            original_h5,
            feasibility_evidence_sha256,
            coupling_feasibility_sha256,
            failed_formal_attempt_receipt_sha256,
        )
        return {"passed": True}

    monkeypatch.setattr(audit, "audit_action_support", fake_audit)
    with pytest.raises(FileExistsError):
        audit.main(
            [
                "--feasibility-evidence",
                str(tmp_path / "anchor.json"),
                "--coupling-feasibility",
                str(tmp_path / "coupling.json"),
                "--failed-formal-attempt-receipt",
                str(tmp_path / "failed.json"),
                "--original-h5",
                str(tmp_path / "original.h5"),
                "--output",
                str(output),
            ]
        )
    assert output.read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.parametrize(
    "field",
    [
        "action_dataset_sha256",
        "finite_action_content_sha256",
        "metric_std_population_float32_sha256",
    ],
)
def test_original_h5_action_hashes_remain_hard_gates(field: str) -> None:
    valid = _fake_original_h5_stats("unused.h5")
    assert valid.passed
    corrupted = replace(valid, **{field: "0" * 64})
    assert not corrupted.passed
