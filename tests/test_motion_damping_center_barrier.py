from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from contextworld.benchmarks.motion_damping_icl_data import file_sha256
from scripts import estimate_pusht_motion_damping_barrier_scales as scales
from scripts import run_pusht_motion_damping_center_barrier as runner


def test_motion_candidates_are_exactly_the_frozen_b1_b2_pair() -> None:
    assert runner.CANDIDATES == {
        "mixed_frozen_image_history_center_barrier_b1": {
            "candidate": "B1",
            "center_weight": 1.0,
        },
        "mixed_frozen_image_history_center_barrier_b2": {
            "candidate": "B2",
            "center_weight": 2.0,
        },
    }
    assert runner.SELECTION_SEED == 14321
    assert runner.OPTIMIZER_STEPS == 8192


def test_motion_wrapper_is_bound_to_frozen_shared_code() -> None:
    shared = runner.SCRIPT_ROOT / "run_pusht_hidden_actuation_mixed.py"
    assert file_sha256(shared) == runner.SHARED_RUNNER_SHA256
    loss_module = (
        runner.ROOT
        / "contextworld/training/paired_prediction_geometry.py"
    )
    assert loss_module.is_file()
    assert file_sha256(loss_module) == runner.EXPECTED_LOSS_MODULE_SHA256
    assert file_sha256(scales.LOSS_MODULE_PATH) == (
        scales.EXPECTED_LOSS_MODULE_SHA256
    )


def test_motion_candidates_register_complete_twin_batching() -> None:
    for variant in runner.CANDIDATES:
        runner._register_candidate(variant)
        assert variant in runner.motion.TWIN_GROUP_VARIANTS
        assert variant in runner.motion.trainer.DIAGNOSTIC_VARIANTS["lewm"]
        assert variant in runner.mixed.FROZEN_IMAGE_VARIANTS
        assert runner.mixed.VARIANT_WEIGHTS[variant][0] == (
            "paired_future_matching"
        )


def test_motion_train_hook_installs_complete_twin_stream() -> None:
    variant = next(iter(runner.CANDIDATES))
    runner._register_candidate(variant)
    observed = {}
    sentinel_stream = object()

    def train_variant(**kwargs):
        observed["stream"] = fake_mixed.pilot.PairedBatchStream
        return {"batch": {}}

    fake_mixed = SimpleNamespace(
        train_variant=train_variant,
        pilot=SimpleNamespace(PairedBatchStream=sentinel_stream),
        load_model_for_variant=object(),
    )
    fake_trainer = SimpleNamespace(mixed=fake_mixed)
    runner.motion._install_complete_twin_batching(fake_trainer)
    result = fake_mixed.train_variant(variant=variant)
    assert observed["stream"] is runner.motion.CompleteTwinPairedBatchStream
    assert fake_mixed.pilot.PairedBatchStream is sentinel_stream
    assert result["batch"]["motion_damping_twin_grouping"]["enabled"]


@pytest.mark.parametrize(
    ("seed", "steps"),
    [(14322, 8192), (14321, 1024)],
)
def test_motion_runner_rejects_nonfrozen_seed_or_steps(
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    steps: int,
) -> None:
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "runner",
            "--variant",
            next(iter(runner.CANDIDATES)),
            "--barrier-scales",
            "scales.json",
            "--data-root",
            "data",
            "--output",
            "output",
            "--model",
            "lewm",
            "--seed",
            str(seed),
            "--optimizer-steps",
            str(steps),
        ],
    )
    with pytest.raises(ValueError):
        runner._runner_args()


def test_motion_runner_uses_formal_data_root_and_step_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "runner",
            "--variant",
            next(iter(runner.CANDIDATES)),
            "--barrier-scales",
            "scales.json",
            "--data-root",
            str(scales.DEFAULT_DATA_ROOT),
            "--output",
            "output",
            "--release-config",
            str(runner.DEFAULT_MOTION_DAMPING_RELEASE_CONFIG),
            "--model",
            "lewm",
            "--seed",
            "14321",
            "--optimizer-steps",
            "8192",
        ],
    )
    parsed, forwarded = runner._runner_args()
    assert parsed.seed == runner.SELECTION_SEED
    assert parsed.optimizer_steps == runner.OPTIMIZER_STEPS
    assert "--data-root" not in forwarded
    assert "--optimizer-steps" not in forwarded
    assert "--release-config" in forwarded
    assert "--seed" in forwarded


def test_motion_scale_freeze_is_bound_to_v4_training() -> None:
    manifest = json.loads(
        (scales.DEFAULT_DATA_ROOT / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert scales.DEFAULT_DATA_ROOT.name == (
        "pusht_motion_damping_h3_release_v4"
    )
    assert manifest["passed"] is True
    assert manifest["pair_counts"]["train"] == scales.EXPECTED_TRAINING_PAIRS
    assert file_sha256(scales.DEFAULT_DATA_ROOT / "manifest.json") == (
        scales.EXPECTED_MANIFEST_SHA256
    )
    assert scales.EXPECTED_MANIFEST_SHA256 == (
        runner.EXPECTED_MANIFEST_SHA256
    )
    assert manifest["splits"]["train"]["passed"] is True
