from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import contextworld.evaluation.protocol as protocol

from contextworld.evaluation.protocol import (
    ColumnStandardizer,
    allocate_scenario_evaluations,
    load_catalog_regime,
    load_pretrained_cost_model,
    select_episode_balanced_starts,
)
from contextworld.paths import artifact_path


def test_episode_balanced_starts_do_not_weight_long_episodes() -> None:
    starts = select_episode_balanced_starts(
        np.asarray([11, 1000, 15, 4]),
        goal_offset=10,
        count=3,
        seed=9,
    )

    assert sorted(starts.episodes) == [0, 1, 2]
    assert len(set(starts.episodes)) == 3
    assert all(
        0 <= step <= np.asarray([11, 1000, 15, 4])[episode] - 11
        for episode, step in zip(starts.episodes, starts.steps)
    )


def test_episode_balanced_starts_reuse_only_after_full_round() -> None:
    starts = select_episode_balanced_starts(
        np.asarray([20, 20, 20]),
        goal_offset=10,
        count=8,
        seed=42,
    )

    assert len(starts.episodes) == 8
    assert len(set(zip(starts.episodes, starts.steps))) == 8
    assert all(type(value) is int for value in starts.episodes)
    assert all(type(value) is int for value in starts.steps)
    assert sorted(starts.episodes[:3]) == [0, 1, 2]
    assert sorted(starts.episodes[3:6]) == [0, 1, 2]
    counts = np.bincount(starts.episodes, minlength=3)
    assert counts.max() - counts.min() <= 1


def test_fixed_total_evaluation_budget_is_scenario_balanced() -> None:
    counts = allocate_scenario_evaluations(
        scenario_count=8,
        total_evaluations=100,
        seed=42,
    )

    assert sum(counts) == 100
    assert sorted(counts) == [12, 12, 12, 12, 13, 13, 13, 13]
    assert counts == allocate_scenario_evaluations(
        scenario_count=8,
        total_evaluations=100,
        seed=42,
    )


def test_column_standardizer_roundtrip() -> None:
    transform = ColumnStandardizer(
        mean=np.asarray([[2.0, -1.0]]),
        std=np.asarray([[0.5, 4.0]]),
    )
    values = np.asarray([[3.0, 7.0], [1.0, -5.0]])

    assert np.allclose(
        transform.inverse_transform(transform.transform(values)), values
    )


def test_catalog_regime_paths_resolve_from_repo_root() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = load_catalog_regime(
        artifact_path(
            "synthesis/catalogs/tworoom_speed_pixel_v2.json", repo_root=root
        ),
        "test_interp",
        repo_root=root,
    )

    assert len(paths) == 16
    assert all(path.is_absolute() and path.is_dir() for path in paths)


def test_native_checkpoint_loader_accepts_current_dynamics_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")

    class Dynamics:
        def encode(self):
            pass

        def rollout(self):
            pass

    model = Dynamics()
    calls: list[str] = []
    monkeypatch.setattr(
        protocol,
        "_prepare_optional_flash_attention",
        lambda: calls.append("flash-fallback") or False,
    )
    stable_worldmodel = SimpleNamespace(
        wm=SimpleNamespace(
            utils=SimpleNamespace(
                load_pretrained=lambda *args, **kwargs: (
                    calls.append("load-pretrained") or model
                )
            )
        )
    )

    assert (
        load_pretrained_cost_model(
            checkpoint,
            stable_worldmodel,
            cache_dir=tmp_path / "cache",
        )
        is model
    )
    assert calls == ["flash-fallback", "load-pretrained"]


def test_native_checkpoint_loader_rejects_non_model_surface(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    stable_worldmodel = SimpleNamespace(
        wm=SimpleNamespace(
            utils=SimpleNamespace(load_pretrained=lambda *args, **kwargs: object())
        )
    )

    with pytest.raises(RuntimeError, match="neither the legacy get_cost API"):
        load_pretrained_cost_model(
            checkpoint,
            stable_worldmodel,
            cache_dir=tmp_path / "cache",
        )
