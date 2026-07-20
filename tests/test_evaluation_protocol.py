from pathlib import Path

import numpy as np

from contextworld.evaluation.protocol import (
    ColumnStandardizer,
    allocate_scenario_evaluations,
    load_catalog_regime,
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
