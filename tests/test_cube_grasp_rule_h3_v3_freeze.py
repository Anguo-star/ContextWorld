from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import scripts.freeze_cube_grasp_rule_h3_v3_development as freeze


def test_placeholder_scan_is_recursive() -> None:
    assert freeze._contains_placeholder(
        {"a": [1, {"b": freeze.PLACEHOLDER}]}
    )
    assert not freeze._contains_placeholder({"a": [1, {"b": "frozen"}]})


def test_declared_path_resolution_keeps_artifacts_in_supplied_root(
    tmp_path: Path,
) -> None:
    resolved = freeze._resolve_declared_path(
        "artifacts/evaluation/example.json", artifact_root=tmp_path
    )
    assert resolved == tmp_path / "evaluation/example.json"


def test_declared_path_resolution_prefers_existing_bundled_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    artifact_root = tmp_path / "external"
    relative = Path("artifacts/evaluation/frozen.json")
    bundled = repo / relative
    bundled.parent.mkdir(parents=True)
    bundled.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(freeze, "ROOT", repo)
    assert freeze._resolve_declared_path(
        relative.as_posix(), artifact_root=artifact_root
    ) == bundled.resolve()


def test_identity_verifier_requires_exact_hash(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"cube-v3")
    digest = hashlib.sha256(b"cube-v3").hexdigest()
    entries = {"value": {"path": str(target), "sha256": digest}}
    observed = freeze._verify_identity_entries(
        entries, artifact_root=tmp_path
    )
    assert observed["value"]["sha256"] == digest

    entries["value"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        freeze._verify_identity_entries(entries, artifact_root=tmp_path)


def test_freeze_cli_requires_explicit_artifact_root_source_and_output() -> None:
    with pytest.raises(SystemExit):
        freeze.parse_args([])
