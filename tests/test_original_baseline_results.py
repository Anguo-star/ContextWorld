from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextworld.benchmarks.original_baseline_results import (
    extract_cell_metric,
)
from contextworld.benchmarks.original_baseline_archive import (
    audit_archived_original_baseline_matrix,
)
from contextworld.paths import repository_root, resolve_contextworld_path


def _speed_horizon(value: float, passed: bool) -> dict[str, object]:
    return {
        "reference_speed_balanced_matching_to_other_loss_ratio": 0.9,
        "reference_speed_balanced_query_win_rate_vs_other_mean": 0.8,
        "reference_speed_balanced_strict_query_win_rate_vs_every_other": value,
        "formal_within_checkpoint_pass": passed,
    }


def test_extract_speed_metric_preserves_tracks_and_horizons() -> None:
    payload = {
        "tracks": {
            track: {
                "horizons": {
                    horizon: _speed_horizon(
                        0.61 if track == "unseen_interpolation" and horizon == "1" else 0.55,
                        not (track == "seen_for_multi" and horizon == "5"),
                    )
                    for horizon in ("1", "2", "3", "5")
                }
            }
            for track in ("seen_for_multi", "unseen_interpolation")
        }
    }

    metric = extract_cell_metric("contextworld-speed", payload)

    assert metric["name"].startswith("unseen_interpolation_h1")
    assert metric["value"] == pytest.approx(0.61)
    assert metric["gate"]["passed"] is False
    assert (
        metric["reader_metrics"]["seen_for_multi"]["5"][
            "within_checkpoint_passed"
        ]
        is False
    )


def test_extract_specialized_and_paired_metrics() -> None:
    door = extract_cell_metric(
        "contextworld-door",
        {
            "summary": {
                "overall": {
                    "same_history_two_target_accuracy": 0.5,
                    "matching_vs_opposite_history_win_rate": 0.51,
                    "strict_win_rate": 0.48,
                }
            },
            "formal_checkpoint_passed": False,
        },
    )
    delay = extract_cell_metric(
        "contextworld-action-delay",
        {
            "core_h1": {
                "physical_group_macro_accuracy": 1 / 6,
                "minimum_physical_group_accuracy": 0.0,
                "paired_query_bootstrap_95_percent_interval": {"lower": 1 / 6},
            },
            "gate": {"passed": False},
        },
    )
    paired = extract_cell_metric(
        "contextworld-cube-gripper-carry",
        {
            "metrics": {
                "correct_future_rate": 0.51,
                "correct_history_rate": 0.75,
                "context_switch_rate": 0.99,
            },
            "gate": {"passed": False},
        },
    )

    assert door["value"] == 0.5 and door["gate"]["passed"] is False
    assert delay["value"] == pytest.approx(1 / 6)
    assert delay["reader_metrics"]["minimum_physical_group_accuracy"] == 0.0
    assert paired["value"] == 0.51
    assert paired["reader_metrics"]["context_switch_rate"] == 0.99


_ROOT = repository_root()
_EXTERNAL_PLDM = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/"
    "lewm-tworooms/ckpt/tworoom_pldm_baseline/"
    "tworoom_pldm_baseline_weights.ckpt"
)
_LOCAL_MATRIX_READY = (
    (_ROOT / "artifacts/evaluation/original_baseline_matrix_v1/")
    .joinpath("contextworld-action-delay/recovery_v1/lewm.json")
    .is_file()
    and _EXTERNAL_PLDM.is_file()
)


@pytest.mark.skipif(
    not _LOCAL_MATRIX_READY,
    reason="complete local original-baseline evidence is not installed",
)
def test_complete_local_matrix_has_exactly_eighteen_descriptive_cells() -> None:
    audit = audit_archived_original_baseline_matrix(repo_root=_ROOT)
    assert audit["status"] == "passed"
    summary_path = resolve_contextworld_path(
        "artifacts/evaluation/original_baseline_matrix_v1/matrix_summary.json",
        repo_root=_ROOT,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "completed"
    assert summary["counts"]["canonical_checkpoints"] == 8
    assert summary["counts"]["icl_cells"] == 18
    assert summary["counts"]["formal_scoreboard_eligible_cells"] == 0
    assert summary["counts"]["cells_with_rescore_evidence"] == 12
    assert summary["counts"]["cells_without_rescore_entrypoint"] == 6
    assert len(summary["cells"]) == 18
    assert all(row["formal_scoreboard_eligible"] is False for row in summary["cells"])
    delay = [
        row
        for row in summary["cells"]
        if row["capability_id"] == "contextworld-action-delay"
    ]
    assert {row["family"] for row in delay} == {"lewm", "pldm"}
    assert all(row["history_adapter"] == "h3_tail_projection" for row in delay)
    assert all(row["metric"]["value"] == pytest.approx(1 / 6) for row in delay)
    assert all(row["original_attempt"]["status"] == "failed_before_predictions" for row in delay)
    portal = [
        row
        for row in summary["cells"]
        if row["capability_id"] == "contextworld-portal-exit"
    ]
    assert all(
        row["rescore_evidence"]["kind"] == "float32_exact_rescore_recovery"
        for row in portal
    )
    assert all(row["rescore_evidence"]["status"] == "verified" for row in portal)
