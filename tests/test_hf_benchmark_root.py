"""A downloaded snapshot and the local HF candidate use one path contract."""

from pathlib import Path

import pytest

from contextworld.training.benchmark_root import resolve_benchmark_root


def test_installed_snapshot_precedes_local_candidate(tmp_path: Path) -> None:
    installed = tmp_path / "data/ContextWorld-v3-hf"
    staged = tmp_path / "repo/artifacts/releases/ContextWorld-v3-hf"
    installed.mkdir(parents=True)
    staged.mkdir(parents=True)
    assert resolve_benchmark_root(None, installed.parent, checkout_root=tmp_path / "repo") == installed


def test_explicit_snapshot_name_and_incomplete_selection_are_authoritative(tmp_path: Path) -> None:
    staged = tmp_path / "artifacts/releases/ContextWorld-v3-hf"
    staged.mkdir(parents=True)
    selected = tmp_path / "downloaded-at-pinned-revision"
    assert resolve_benchmark_root(selected, None, checkout_root=tmp_path) == selected


def test_local_candidate_replaces_legacy_default(tmp_path: Path) -> None:
    legacy = tmp_path / "data/ContextWorld-v1"
    legacy.mkdir(parents=True)
    staged = tmp_path / "repo/artifacts/releases/ContextWorld-v3-hf"
    staged.mkdir(parents=True)
    assert resolve_benchmark_root(None, legacy.parent, checkout_root=tmp_path / "repo") == staged


def test_legacy_bundle_is_never_an_automatic_fallback(tmp_path: Path) -> None:
    legacy = tmp_path / "data/ContextWorld-v1"
    legacy.mkdir(parents=True)
    selected = resolve_benchmark_root(None, legacy.parent, checkout_root=tmp_path / "repo")
    assert selected == legacy.parent / "ContextWorld-v3-hf"
    assert not selected.exists()


@pytest.mark.parametrize("explicit,dataset_root", [("relative", None), (None, "relative")])
def test_relative_operator_paths_fail(explicit, dataset_root, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        resolve_benchmark_root(explicit, dataset_root, checkout_root=tmp_path)


def test_missing_configuration_requires_snapshot(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CONTEXTWORLD_BENCHMARK_ROOT"):
        resolve_benchmark_root(None, None, checkout_root=tmp_path)
