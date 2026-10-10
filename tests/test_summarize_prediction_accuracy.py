from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_prediction_accuracy as summary  # noqa: E402


def _tworoom_scene_arrays():
    truth = np.zeros((2, 1, 2, 2), dtype=np.float64)
    truth[1, ..., 0] = 10.0
    perfect = truth.copy()
    return truth, perfect


def test_same_matched_prediction_for_every_condition_has_zero_history_gain():
    truth = np.zeros((2, 1, 2, 2), dtype=np.float64)
    truth[1, ..., 0] = 10.0
    same_prediction = np.zeros_like(truth)
    same_prediction[..., 1] = 1.0
    objects = summary.summarize_scene(
        truth,
        same_prediction,
        truth,
        "tworoom",
        [0.5, 1.5, 3.0],
    )

    aggregate = summary.aggregate(
        [{"source_group": "source-1", "objects": objects}], "object"
    )

    assert aggregate["matched"]["score"] == aggregate["wrong"]["score"]
    assert aggregate["history_gain"] == 0.0
    assert aggregate["history_gain_ci95"] == [0.0, 0.0]


def test_perfect_matched_prediction_scores_one_hundred_and_wrong_history_scores_lower():
    truth, perfect = _tworoom_scene_arrays()
    objects = summary.summarize_scene(
        truth,
        perfect,
        perfect,
        "tworoom",
        [1.0, 5.0, 15.0],
    )

    assert objects["object"]["matched"]["score"] == 100.0
    assert objects["object"]["wrong"]["score"] < 100.0


def test_shared_bias_preserves_condition_response_but_lowers_absolute_accuracy():
    truth, perfect = _tworoom_scene_arrays()
    biased = truth.copy()
    biased[..., 1] += 3.0
    perfect_objects = summary.summarize_scene(
        truth, perfect, perfect, "tworoom", [1.0, 2.0, 4.0, 5.0]
    )
    biased_objects = summary.summarize_scene(
        truth, biased, perfect, "tworoom", [1.0, 2.0, 4.0, 5.0]
    )

    # The condition response is exactly preserved by the shared offset.
    np.testing.assert_array_equal(
        np.diff(biased, axis=0), np.diff(truth, axis=0)
    )
    assert perfect_objects["object"]["matched"]["score"] == 100.0
    assert biased_objects["object"]["matched"]["score"] < 100.0


def _write_build_unit(
    root: Path,
    identifier: str,
    *,
    scene_id: str,
    source_group: str,
) -> dict[str, object]:
    model_manifest = root / f"{identifier.replace('/', '_')}_models.json"
    model_manifest.write_text(
        json.dumps([{"id": identifier, "checkpoint_sha256": f"checkpoint-{identifier}"}]),
        encoding="utf-8",
    )
    physical = root / f"{identifier.replace('/', '_')}_physical.json"
    physical.write_text(
        json.dumps(
            {"query_metrics": [{"scene_id": scene_id, "source_group": source_group}]}
        ),
        encoding="utf-8",
    )
    truth, matched = _tworoom_scene_arrays()
    output = root / "readout" / identifier
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "scene.npz",
        truth=truth,
        matched=matched,
        calibration=truth,
        scene_id=np.asarray(scene_id),
        source_group=np.asarray(source_group),
        fold=np.asarray(0),
    )
    return {
        "id": identifier,
        "models_manifest": str(model_manifest),
        "source_physical_json": str(physical),
        "suite": "fixture",
        "panel_identity": {"manifest_sha256": "same-panel"},
    }


@pytest.mark.parametrize(
    "scratch_scene_id,scratch_source_group,error_message",
    [
        ("scene-scratch", "source-1", "Unpaired training comparison"),
        ("scene-original", "source-scratch", "Unpaired source groups"),
    ],
)
def test_stage_comparison_rejects_unpaired_scenes_or_sources(
    tmp_path, scratch_scene_id, scratch_source_group, error_message
):
    root = tmp_path
    (root / "readout").mkdir()
    (root / "protocol.json").write_text(
        json.dumps(
            {
                "thresholds": {"tworoom": [1, 2, 3, 4, 5]},
                "primary_object": {"speed": "object"},
            }
        ),
        encoding="utf-8",
    )
    units = [
        _write_build_unit(
            root,
            "speed/lewm/original/s1",
            scene_id="scene-original",
            source_group="source-1",
        ),
        _write_build_unit(
            root,
            "speed/lewm/scratch/s1",
            scene_id=scratch_scene_id,
            source_group=scratch_source_group,
        ),
    ]
    (root / "readout" / "run_manifest.json").write_text(
        json.dumps({"units": units}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=error_message):
        summary.build(root)
