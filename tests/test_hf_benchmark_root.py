"""A downloaded snapshot and the local HF candidate use one path contract."""

from pathlib import Path

import pytest

from contextworld.training.benchmark_root import resolve_benchmark_root


def test_snapshot_is_resolved_under_external_data_root(tmp_path: Path) -> None:
    installed = tmp_path / "data/ContextWorld-v3-hf"
    staged = tmp_path / "repo/artifacts/releases/ContextWorld-v3-hf"
    installed.mkdir(parents=True)
    staged.mkdir(parents=True)
    assert resolve_benchmark_root(None, installed.parent) == installed


def test_explicit_snapshot_name_and_incomplete_selection_are_authoritative(tmp_path: Path) -> None:
    staged = tmp_path / "artifacts/releases/ContextWorld-v3-hf"
    staged.mkdir(parents=True)
    selected = tmp_path / "downloaded-at-pinned-revision"
    assert resolve_benchmark_root(selected, staged.parent) == selected


def test_in_checkout_candidate_is_not_an_automatic_fallback(tmp_path: Path, monkeypatch) -> None:
    staged = tmp_path / "repo/artifacts/releases/ContextWorld-v3-hf"
    staged.mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "repo")
    with pytest.raises(ValueError, match="CONTEXTWORLD_BENCHMARK_ROOT"):
        resolve_benchmark_root(None, None)


def test_legacy_bundle_is_never_an_automatic_fallback(tmp_path: Path) -> None:
    legacy = tmp_path / "data/ContextWorld-v1"
    legacy.mkdir(parents=True)
    selected = resolve_benchmark_root(None, legacy.parent)
    assert selected == legacy.parent / "ContextWorld-v3-hf"
    assert not selected.exists()


@pytest.mark.parametrize("explicit,dataset_root", [("relative", None), (None, "relative")])
def test_relative_operator_paths_fail(explicit, dataset_root, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        resolve_benchmark_root(explicit, dataset_root)


def test_missing_configuration_requires_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CONTEXTWORLD_BENCHMARK_ROOT"):
        resolve_benchmark_root(None, None)
