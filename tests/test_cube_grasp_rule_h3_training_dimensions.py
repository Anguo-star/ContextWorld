from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

import scripts.run_cube_grasp_rule_h3_train as cube_train


def test_pinned_loss_compatibility_accepts_only_unused_false_path() -> None:
    class LegacyConditionalSIGReg(torch.nn.Module):
        def __init__(
            self,
            knots: int = 17,
            num_proj: int = 1024,
            randomize_pair_orientation: bool = True,
        ) -> None:
            super().__init__()
            self.knots = knots
            self.num_proj = num_proj
            self.randomize_pair_orientation = randomize_pair_orientation

    stable_loss = SimpleNamespace(ConditionalSIGReg=LegacyConditionalSIGReg)
    receipt = cube_train._install_pinned_loss_compatibility(stable_loss)

    adapted = stable_loss.ConditionalSIGReg(
        knots=9,
        num_proj=32,
        randomize_pair_orientation=False,
        include_unpaired=False,
        complete_haar_population=False,
    )
    assert adapted.knots == 9
    assert adapted.num_proj == 32
    assert adapted.randomize_pair_orientation is False
    assert receipt["conditional_sigreg_constructor_adapter_installed"] is True
    assert receipt["conditional_sigreg_missing_keywords"] == [
        "include_unpaired",
        "complete_haar_population",
    ]
    assert receipt["unavailable_eager_diagnostic_sentinels"] == [
        "DynamicsResponseSIGReg",
        "GroupBalancedSIGReg",
        "ScaleCalibratedConditionalSIGReg",
    ]
    with pytest.raises(RuntimeError, match="false/false constructor path"):
        stable_loss.ConditionalSIGReg(include_unpaired=True)
    with pytest.raises(RuntimeError, match="unavailable in Cube's pinned"):
        stable_loss.GroupBalancedSIGReg()(torch.zeros(1))


def test_cube_installs_five_by_five_action_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cube_train.trainer, "ACTION_INPUT_DIM", 10)
    monkeypatch.setattr(cube_train.trainer.pilot, "ACTION_DIM", 2)
    monkeypatch.setattr(cube_train.trainer.pilot, "ACTION_INPUT_DIM", 10)
    monkeypatch.setattr(cube_train.trainer.mixed, "ACTION_INPUT_DIM", 10)

    cube_train._install_cube_action_dimensions()

    assert cube_train.CUBE_RAW_ACTION_DIM == 5
    assert cube_train.CUBE_ACTION_BLOCK_STEPS == 5
    assert cube_train.CUBE_ACTION_INPUT_DIM == 25
    assert cube_train.trainer.ACTION_INPUT_DIM == 25
    assert cube_train.trainer.pilot.ACTION_DIM == 5
    assert cube_train.trainer.pilot.ACTION_INPUT_DIM == 25
    assert cube_train.trainer.mixed.ACTION_INPUT_DIM == 25


def test_materialized_split_observes_installed_cube_dimension() -> None:
    cube_train._install_cube_action_dimensions()
    split = cube_train.trainer.pilot.MaterializedSplit(
        pixels=torch.zeros((2, 4, 3, 224, 224), dtype=torch.uint8),
        action=torch.zeros((2, 4, 25), dtype=torch.float32),
        pair_count=1,
    )
    assert split.action.shape == (2, 4, 25)

    with pytest.raises(ValueError, match="Unexpected action shape"):
        cube_train.trainer.pilot.MaterializedSplit(
            pixels=split.pixels,
            action=torch.zeros((2, 4, 10), dtype=torch.float32),
            pair_count=1,
        )


def test_cube_action_normalizer_rejects_two_axis_source(tmp_path: Path) -> None:
    source = tmp_path / "two_axis.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("action", data=np.zeros((8, 2), dtype=np.float32))

    with pytest.raises(ValueError, match=r"shape \(rows, 5\)"):
        cube_train._finite_action_stats(source)


def test_cube_action_normalizer_preserves_all_five_axes(tmp_path: Path) -> None:
    source = tmp_path / "cube.h5"
    actions = np.arange(40, dtype=np.float32).reshape(8, 5)
    with h5py.File(source, "w") as handle:
        handle.create_dataset("action", data=actions)

    stats = cube_train._finite_action_stats(source)

    assert stats["count"] == 8
    assert stats["mean"].shape == (5,)
    assert stats["std"].shape == (5,)
    np.testing.assert_allclose(stats["mean"], actions.mean(axis=0))
    np.testing.assert_allclose(stats["std"], actions.std(axis=0))


def test_cube_lance_wrapper_rejects_non_five_by_five_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = SimpleNamespace(
        raw_action_blocks=np.zeros((2, 4, 5, 2), dtype=np.float32)
    )
    monkeypatch.setattr(cube_train, "_read_lance_pairs", lambda *args, **kwargs: arrays)

    with pytest.raises(ValueError, match="five raw steps x five Cube action axes"):
        cube_train._read_cube_lance_pairs(
            Path("unused.lance"),
            expected_pairs=2,
            expected_split="train",
        )


def test_cube_main_binds_dimensions_before_shared_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int] = {}

    def shared_main() -> None:
        observed.update(
            trainer=cube_train.trainer.ACTION_INPUT_DIM,
            pilot_raw=cube_train.trainer.pilot.ACTION_DIM,
            pilot_flat=cube_train.trainer.pilot.ACTION_INPUT_DIM,
            mixed=cube_train.trainer.mixed.ACTION_INPUT_DIM,
        )

    monkeypatch.setattr(cube_train, "_install_cube_diagnostic_name", lambda: None)
    monkeypatch.setattr(cube_train, "_PINNED_STABLE_RUNTIME", Path("/pinned"))
    monkeypatch.setattr(cube_train.trainer, "main", shared_main)
    monkeypatch.setattr(cube_train.trainer, "ACTION_INPUT_DIM", 10)
    monkeypatch.setattr(cube_train.trainer.pilot, "ACTION_DIM", 2)
    monkeypatch.setattr(cube_train.trainer.pilot, "ACTION_INPUT_DIM", 10)
    monkeypatch.setattr(cube_train.trainer.mixed, "ACTION_INPUT_DIM", 10)

    cube_train.main()

    assert observed == {
        "trainer": 25,
        "pilot_raw": 5,
        "pilot_flat": 25,
        "mixed": 25,
    }
