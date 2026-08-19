from pathlib import Path

from contextworld.paths import (
    artifact_root,
    portable_contextworld_path,
    resolve_contextworld_path,
)


def test_default_artifact_root_is_outside_code_repository(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXTWORLD_ARTIFACT_ROOT", raising=False)
    repo = Path(__file__).resolve().parents[1]

    assert artifact_root(repo) == (
        repo.resolve().parents[1] / "data/world_model/context_world"
    ).resolve()
    assert repo.resolve() not in artifact_root(repo).parents


def test_logical_artifact_reference_round_trips(monkeypatch, tmp_path) -> None:
    repo = tmp_path / "ag_data/code/ContextWorld"
    external = tmp_path / "external/context_world"
    repo.mkdir(parents=True)
    monkeypatch.setenv("CONTEXTWORLD_ARTIFACT_ROOT", str(external))

    resolved = resolve_contextworld_path(
        "artifacts/evaluation/history3/result.json", repo_root=repo
    )

    assert resolved == external / "evaluation/history3/result.json"
    assert (
        portable_contextworld_path(resolved, repo_root=repo)
        == "artifacts/evaluation/history3/result.json"
    )


def test_existing_bundled_artifact_precedes_external_root(
    monkeypatch,
    tmp_path,
) -> None:
    repo = tmp_path / "ag_data/code/ContextWorld"
    bundled = repo / "artifacts/synthesis/component/manifest.json"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("{}\n", encoding="utf-8")
    external = tmp_path / "external/context_world"
    monkeypatch.setenv("CONTEXTWORLD_ARTIFACT_ROOT", str(external))

    resolved = resolve_contextworld_path(
        "artifacts/synthesis/component/manifest.json",
        repo_root=repo,
    )

    assert resolved == bundled
