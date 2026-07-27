from __future__ import annotations

from pathlib import Path

from contextworld.benchmarks import suite_data
from contextworld.benchmarks.suite_data import (
    COMPONENT_IDS,
    SUITE_RELEASE_ID,
    export_icl_suite_artifacts,
    load_icl_suite_release,
)


def test_default_suite_registers_speed_and_door() -> None:
    suite = load_icl_suite_release()
    assert suite["release_id"] == SUITE_RELEASE_ID
    assert tuple(suite["components"]) == COMPONENT_IDS
    assert suite["bundle"]["top_level_entries"] == [
        "README.md",
        "benchmark",
    ]
    assert suite["scope"]["sealed_test_included"] is False


def test_integrated_export_has_only_readme_and_benchmark(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    destination = tmp_path / "bundle"
    (repo / "configs/benchmark").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "docs/ContextWorld_ICL_Benchmark.md").write_text(
        "# Unified benchmark\n",
        encoding="utf-8",
    )
    suite_config = repo / "configs/benchmark/suite.yaml"
    speed_config = repo / "configs/benchmark/speed.yaml"
    door_config = repo / "configs/benchmark/door.yaml"
    suite_config.write_text("suite\n", encoding="utf-8")
    speed_config.write_text("speed\n", encoding="utf-8")
    door_config.write_text("door\n", encoding="utf-8")

    directory_paths = (
        "synthesis/speed",
        "evaluation/history3/speed_multistep_extrap_v5/catalogs",
        "evaluation/history3/speed_multistep_extrap_v5/payloads",
        "evaluation/history3/speed_isolated_v2/catalogs",
        "evaluation/history3/speed_isolated_v2/payloads",
        "synthesis/door",
        "evaluation/door",
        "evaluation/door-reference",
    )
    for relative in directory_paths:
        path = artifacts / relative
        path.mkdir(parents=True)
        (path / "payload.bin").write_bytes(relative.encode("utf-8"))
    file_paths = (
        "synthesis/speed.json",
        "synthesis/speed.jsonl",
        "synthesis/speed-report.json",
        "splits/normalizer.json",
        "training/init.pt",
        "training/config.json",
    )
    for relative in file_paths:
        path = artifacts / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    original_h5 = tmp_path / "tworoom.h5"
    original_h5.write_bytes(b"original")

    suite = {
        "release_id": SUITE_RELEASE_ID,
        "_config_path": str(suite_config),
        "components": {
            "speed": {"release_config": str(speed_config)},
            "door": {"release_config": str(door_config)},
        },
        "repository": {
            "public_document": {
                "path": "docs/ContextWorld_ICL_Benchmark.md"
            }
        },
        "distribution": {},
    }
    speed = {
        "training": {
            "synthetic": {
                "single": {
                    "data_root": "artifacts/synthesis/speed",
                    "catalog": "artifacts/synthesis/speed.json",
                    "manifest": "artifacts/synthesis/speed.jsonl",
                    "report": "artifacts/synthesis/speed-report.json",
                }
            },
            "original": {"source": "upstream", "license": "MIT"},
        },
        "evaluation": {"normalizer": "artifacts/splits/normalizer.json"},
        "planning": {"enabled": True},
    }
    door = {
        "training": {
            "artifact_tree": {"root": "artifacts/synthesis/door"},
            "initialization": {
                "checkpoint": "artifacts/training/init.pt",
                "checkpoint_config": "artifacts/training/config.json",
            },
        },
        "evaluation": {
            "artifact_tree": {"root": "artifacts/evaluation/door"},
            "normalizer": "artifacts/splits/normalizer.json",
        },
        "reference_results": {
            "reference": {
                "root": "artifacts/evaluation/door-reference"
            }
        },
    }

    original_resolver = suite_data.resolve_contextworld_path

    def fake_resolve(value, *, repo_root=None):
        path = Path(value)
        if path.parts and path.parts[0] == "artifacts":
            return artifacts.joinpath(*path.parts[1:])
        return original_resolver(value, repo_root=repo_root)

    monkeypatch.setattr(suite_data, "load_icl_suite_release", lambda *a, **k: suite)
    monkeypatch.setattr(suite_data, "load_speed_icl_release", lambda *a, **k: speed)
    monkeypatch.setattr(suite_data, "load_door_icl_release", lambda *a, **k: door)
    monkeypatch.setattr(suite_data, "resolve_contextworld_path", fake_resolve)
    monkeypatch.setattr(
        suite_data,
        "resolve_original_h5",
        lambda *a, **k: original_h5,
    )

    result = export_icl_suite_artifacts(
        destination,
        repo_root=repo,
        mode="copy",
    )
    assert sorted(path.name for path in destination.iterdir()) == [
        "README.md",
        "benchmark",
    ]
    assert (destination / "benchmark/suite.yaml").is_file()
    assert (destination / "benchmark/releases/speed.yaml").is_file()
    assert (destination / "benchmark/releases/door.yaml").is_file()
    assert (
        destination / "benchmark/upstream/lewm-tworooms/tworoom.h5"
    ).is_file()
    assert result["components"] == ["speed", "door"]
    assert result["includes_upstream_original_h5"] is True
