from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from contextworld.evaluation.cube_grasp_rule_h3_v3 import (
    V3_ACTION_ANCHORS,
    V3_ANCHOR_GRIPPER_ACTIONS,
    V3_ANCHOR_PROFILES,
)
from scripts import audit_cube_grasp_rule_h3_v3_action_support as audit


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


def _install_fake_original_h5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "compute_original_h5_action_population_stats",
        _fake_original_h5_stats,
    )


def _evidence_payload(
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
                float(value) for value in V3_ANCHOR_PROFILES[anchor_id]
            ],
            "gripper": float(V3_ANCHOR_GRIPPER_ACTIONS[anchor_id]),
            "support_nearest_nrmse": (
                _NEAREST[anchor_id]
                if nearest_override is None
                else nearest_override
            ),
            "support_count_nrmse_le_0p5": _SUPPORT_COUNTS[anchor_id],
        }
        for anchor_id in V3_ACTION_ANCHORS
    }
    # An unrelated v2 control row is allowed; anchors are matched by their
    # numeric profile and gripper content rather than by JSON key or ordering.
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
                "nearest standardized RMSE over all 75 values in every finite "
                "same-episode 15-step H5 action window"
            ),
            "same_episode_windows": 1_860_000,
        },
        "templates": templates,
    }


def _write_evidence(
    tmp_path: Path, payload: dict[str, object]
) -> tuple[Path, str]:
    path = tmp_path / "frozen_feasibility_evidence.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def test_standardized_joint_metric_uses_all_75_values() -> None:
    left = np.zeros((15, 5), dtype=np.float64)
    right = left.copy()
    right[:, 2] = _METRIC_STD[2]

    # Fifteen of 75 standardized values differ by exactly one.
    assert audit.standardized_joint_15_step_nrmse(
        left,
        right,
        action_std_population=_METRIC_STD,
    ) == pytest.approx(np.sqrt(15.0 / 75.0))
    with pytest.raises(ValueError, match=r"\[15,5\]"):
        audit.standardized_joint_15_step_nrmse(
            np.zeros((14, 5)),
            np.zeros((14, 5)),
            action_std_population=_METRIC_STD,
        )


def test_original_h5_population_stats_are_streamed_and_hashed(
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

    stats = audit.compute_original_h5_action_population_stats(
        source, chunk_rows=2
    )

    finite = actions[np.isfinite(actions).all(axis=1)].astype(np.float64)
    expected_mean = np.mean(finite, axis=0, dtype=np.float64)
    expected_std = np.std(finite, axis=0, dtype=np.float64, ddof=0)
    assert stats.total_rows == 6
    assert stats.finite_action_rows == 4
    assert stats.excluded_nonfinite_rows == 2
    assert stats.population_mean_float64 == pytest.approx(expected_mean)
    assert stats.population_std_float64 == pytest.approx(expected_std)
    assert stats.metric_std_population_float32 == pytest.approx(
        expected_std.astype(np.float32), rel=0.0, abs=0.0
    )
    assert stats.gripper_maximum == pytest.approx(0.75)
    assert stats.action_dataset_sha256 == hashlib.sha256(
        np.ascontiguousarray(actions, dtype="<f4").tobytes(order="C")
    ).hexdigest()
    assert stats.finite_action_content_sha256 == hashlib.sha256(
        np.ascontiguousarray(finite, dtype="<f4").tobytes(order="C")
    ).hexdigest()
    assert stats.metric_std_population_float32_sha256 == hashlib.sha256(
        np.ascontiguousarray(expected_std, dtype="<f4").tobytes(order="C")
    ).hexdigest()
    assert stats.passed is False


@pytest.mark.parametrize(
    "field",
    [
        "action_dataset_sha256",
        "finite_action_content_sha256",
        "metric_std_population_float32_sha256",
    ],
)
def test_frozen_original_h5_hashes_are_hard_gates(field: str) -> None:
    valid = _fake_original_h5_stats("unused.h5")
    assert valid.passed is True
    corrupted = replace(valid, **{field: "0" * 64})
    assert corrupted.passed is False
    assert any(
        "sha256_matches_frozen" in name and not passed
        for name, passed in corrupted.checks.items()
    )


def test_formal_audit_enumerates_frozen_profiles_and_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_original_h5(monkeypatch)
    evidence_path, evidence_sha256 = _write_evidence(
        tmp_path, _evidence_payload()
    )

    result = audit.audit_action_support(
        evidence_path,
        feasibility_evidence_sha256=evidence_sha256,
        original_h5=tmp_path / "original.h5",
    )

    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["scope"]["active_splits"] == [
        "train",
        "loader_validation",
    ]
    assert result["scope"]["total_concrete_profiles"] == 2304
    assert result["scope"]["lance_tables_opened"] == []
    assert result["scope"]["public_test_inputs"] == []
    assert result["scope"]["public_test_opened"] is False
    assert result["original_h5_action"]["source_symbol"] == (
        "upstream_cube_single_expert_h5"
    )
    assert result["original_h5_action"]["path_recorded"] is False
    assert "path" not in result["original_h5_action"]
    assert result["feasibility_evidence"]["source_symbol"] == (
        "frozen_action_template_feasibility_json"
    )
    assert result["feasibility_evidence"]["path_recorded"] is False
    assert "path" not in result["feasibility_evidence"]
    assert result["metric"]["action_std_population"] == pytest.approx(
        _METRIC_STD
    )
    assert result["metric"]["action_std_population_sha256"] == (
        audit.FROZEN_ORIGINAL_H5_METRIC_STD_POPULATION_FLOAT32_SHA256
    )
    assert result["splits"]["train"]["profile_count"] == 2048
    assert result["splits"]["loader_validation"]["profile_count"] == 256
    assert result["splits"]["train"]["action_anchor_counts"] == {
        anchor_id: 512 for anchor_id in V3_ACTION_ANCHORS
    }
    assert result["splits"]["loader_validation"][
        "action_anchor_counts"
    ] == {anchor_id: 64 for anchor_id in V3_ACTION_ANCHORS}
    assert result["cross_split"]["profile_content_overlap"]["count"] == 0
    assert result["cross_split"]["anchor_family_overlap"]["count"] == 4
    assert result["overall"]["profile_count"] == 2304
    assert result["overall"]["conservatively_supported_profile_count"] == 2304
    assert (
        result["overall"]["maxima"]
        ["conservative_original_h5_nrmse_upper_bound"]
        <= 0.5
    )
    assert result["support_counts"][
        "anchor_h5_windows_at_or_below_nrmse_0p5"
    ] == _SUPPORT_COUNTS
    for anchor_id in V3_ACTION_ANCHORS:
        assert result["anchors"][anchor_id]["profile_count"] == 576
        assert (
            result["anchors"][anchor_id]["maxima"]
            ["conservative_original_h5_nrmse_upper_bound"]
            <= 0.5
        )


def test_evidence_sha_mismatch_is_rejected_before_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path, _ = _write_evidence(tmp_path, _evidence_payload())

    def forbidden_factory(*, split: str, catalog_index: int) -> object:
        raise AssertionError((split, catalog_index))

    monkeypatch.setattr(audit, "make_v3_action_profile", forbidden_factory)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit.audit_action_support(
            evidence_path,
            feasibility_evidence_sha256="0" * 64,
            original_h5=tmp_path / "must_not_be_opened.h5",
        )


@pytest.mark.parametrize(
    "contaminated_read",
    [
        "loader_validation.lance",
        "validation.lance",
        "decoded Public Test profiles",
    ],
)
def test_evidence_declaring_development_or_public_reads_is_rejected(
    tmp_path: Path, contaminated_read: str
) -> None:
    evidence_path, evidence_sha256 = _write_evidence(
        tmp_path,
        _evidence_payload(data_read=["original cube H5", contaminated_read]),
    )

    with pytest.raises(ValueError, match="Development/Public data reads"):
        audit.audit_action_support(
            evidence_path,
            feasibility_evidence_sha256=evidence_sha256,
            original_h5=tmp_path / "must_not_be_opened.h5",
        )


def test_triangle_upper_bound_is_a_hard_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_original_h5(monkeypatch)
    evidence_path, evidence_sha256 = _write_evidence(
        tmp_path, _evidence_payload(nearest_override=0.5)
    )

    result = audit.audit_action_support(
        evidence_path,
        feasibility_evidence_sha256=evidence_sha256,
        original_h5=tmp_path / "original.h5",
    )

    assert result["status"] == "failed"
    assert result["passed"] is False
    assert result["overall"]["conservatively_supported_profile_count"] < 2304
    assert result["checks"][
        "all_profile_invariants_and_action_bounds_passed"
    ]
    assert not result["checks"][
        "all_profiles_conservatively_within_original_h5_support_gate"
    ]


@pytest.mark.parametrize(
    "option",
    [
        "--validation-lance",
        "--validation-pairs",
        "--public-test",
        "--public-test-pairs",
        "--test-pairs",
    ],
)
def test_cli_rejects_every_public_input_option(option: str) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        audit.parse_args(
            [
                "--feasibility-evidence",
                "evidence.json",
                "--feasibility-evidence-sha256",
                "0" * 64,
                "--original-h5",
                "original.h5",
                "--output",
                "audit.json",
                option,
                "anything",
            ]
        )


def test_cli_refuses_to_overwrite_existing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("sentinel\n", encoding="utf-8")

    def fake_audit(
        feasibility_evidence: Path,
        *,
        feasibility_evidence_sha256: str,
        original_h5: Path,
    ) -> dict[str, object]:
        del feasibility_evidence, feasibility_evidence_sha256, original_h5
        return {"passed": True}

    monkeypatch.setattr(audit, "audit_action_support", fake_audit)
    with pytest.raises(FileExistsError):
        audit.main(
            [
                "--feasibility-evidence",
                str(tmp_path / "evidence.json"),
                "--feasibility-evidence-sha256",
                "0" * 64,
                "--original-h5",
                str(tmp_path / "original.h5"),
                "--output",
                str(output),
            ]
        )
    assert output.read_text(encoding="utf-8") == "sentinel\n"
