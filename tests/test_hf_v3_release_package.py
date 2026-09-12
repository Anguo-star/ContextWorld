"""Release packaging preserves pinned bytes and current split contracts."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hf_v3_release", ROOT / "scripts/prepare_contextworld_hf_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def write(root, path, content):
    data = content if isinstance(content, bytes) else content.encode()
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def seal(root, rows):
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (root / "manifest.jsonl").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (root / "manifest.sha256").write_text(f"{digest}  manifest.jsonl\n")
    return digest


@pytest.fixture
def source(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    for name in release.LEGAL_FILES:
        write(repo, name, "fixture legal text\n")
    baseline = write(repo, release.BASELINE_REL, '{"freeze":"fixture-v3"}\n')
    write(repo, release.TEMPLATE_REL, (ROOT / release.TEMPLATE_REL).read_text())
    bundles = []
    pins = {"baseline": baseline["sha256"]}
    for label, name in (("dev", release.DEV_BUNDLE), ("test", release.TEST_BUNDLE)):
        root = tmp_path / name
        root.mkdir()
        rows, components = [], []
        for task in ("a", "b"):
            payloads = []
            splits = ("training", "development", "development_legacy") if label == "dev" else ("training", "development", "test")
            for split in splits:
                prefix = f"components/{task}/v1/{split}/data.lance"
                if split == "test":
                    prefix = f"artifacts/evaluation/{task}/validation.lance"
                path = prefix + "/data/member.lance"
                row = write(root, path, f"{label}:{task}:{split}")
                rows.append({**row, "role": "dataset_payload", "component": task, "split": split})
                # Payload IDs deliberately repeat across tasks AND splits.
                payloads.append({"payload_id": "data", "split": split, "public_path": prefix,
                                 "members": [prefix], "file_count": 1})
            component = {"component_id": task, "dataset_id": task, "environment": "Fixture",
                         "history_length": 3, "action_dimension": 2, "frameskip": 5,
                         "release_config": "fixture.yaml", "release_config_sha256": "a" * 64,
                         "payloads": payloads, "development_evaluation": {
                             "reader_id": "current" if label == "dev" else "obsolete",
                             "selection": {"fixture": label}, "normalizer_path": "normalizers/fixture.json"}}
            if label == "test":
                component["public_test_evaluation"] = {"split": "test", "artifact_root": f"artifacts/evaluation/{task}"}
            components.append(component)
        registry = {"schema_version": 1, "components": components, "public_test": {
            "included": label == "test", "policy": "public_offline_final_reporting"}}
        rows.append({**write(root, "normalizers/fixture.json", '{"normalizer":"frozen"}'), "role": "release_metadata"})
        rows.append({**write(root, "components/a/v1/development/registry_contract.json", '{"selection":"frozen"}'),
                     "role": "release_metadata"})
        reg = write(root, "task_registry.json", json.dumps(registry))
        rows.append({**reg, "role": "release_metadata"})
        pins[label + "_registry"] = reg["sha256"]
        pins[label + "_manifest"] = seal(root, rows)
        bundles.append(root)
    monkeypatch.setattr(release, "REPO_ROOT", repo)
    monkeypatch.setattr(release, "REQUIRED_COMPONENTS", 2)
    monkeypatch.setattr(release, "PINS", pins)
    return argparse.Namespace(development_root=bundles[0], test_root=bundles[1], output=tmp_path / "output", workers=2)


def manifest(root):
    return [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("direct_write", [False, True])
def test_build_merges_current_dev_and_test_without_rewriting_sources(source, direct_write):
    source.direct_write = direct_write
    before = {p: p.read_bytes() for root in (source.development_root, source.test_root) for p in root.rglob("*") if p.is_file()}
    result = release.cmd_execute(source)
    assert result["payload_files"] == 6
    assert release.verify_package(source.output, 2)["status"] == "passed"
    registry = json.loads((source.output / "task_registry.json").read_text())
    for comp in registry["components"]:
        assert comp["development_evaluation"]["reader_id"] == "current"
        assert {p["split"] for p in comp["payloads"]} == {"training", "development", "test"}
    for row in manifest(source.output):
        if row["role"] == "dataset_payload":
            expected_root = source.test_root if row["split"] == "test" else source.development_root
            assert (source.output / row["path"]).read_bytes() == (expected_root / row["path"]).read_bytes()
            assert row["source_bundle"] == expected_root.name
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not list((source.output / "components").rglob("development_legacy"))
    assert "{{" not in (source.output / "README.md").read_text()


def test_downloaded_layout_verifies_without_original_source_directories(source, tmp_path):
    release.cmd_execute(source)
    downloaded = tmp_path / "external-user/downloaded-snapshot"
    shutil.copytree(source.output, downloaded)
    for root in (source.development_root, source.test_root, source.output):
        shutil.rmtree(root)
    assert release.verify_package(downloaded, 2)["status"] == "passed"


def test_refuses_existing_output(source):
    source.output.mkdir()
    (source.output / "sentinel").write_text("retain")
    with pytest.raises(release.ReleaseError, match="already exists"):
        release.cmd_execute(source)
    assert (source.output / "sentinel").read_text() == "retain"


def test_source_tampering_aborts_and_removes_own_staging(source):
    row = next(r for r in manifest(source.development_root) if r.get("split") == "training")
    path = source.development_root / row["path"]
    path.write_bytes(b"x" * row["bytes"])
    with pytest.raises(release.ReleaseError, match="sha256 mismatch copying"):
        release.cmd_execute(source)
    assert not source.output.exists()
    assert not list(source.output.parent.glob(".output.staging-*"))


def test_rejects_wrong_pin_and_missing_sidecar(source):
    sidecar = source.development_root / "manifest.sha256"
    data = sidecar.read_bytes()
    sidecar.unlink()
    with pytest.raises(release.ReleaseError, match="Missing regular file"):
        release.build_plan(source)
    sidecar.write_bytes(data)
    (source.development_root / "task_registry.json").write_text("{}")
    with pytest.raises(release.ReleaseError, match="pin mismatch"):
        release.build_plan(source)


@pytest.mark.parametrize("kind", ["bytes", "unlisted", "missing", "receipt", "symlink", "registry"])
def test_verify_detects_corrupted_candidate(source, kind):
    release.cmd_execute(source)
    rows = manifest(source.output)
    row = next(r for r in rows if r["role"] == "dataset_payload")
    path = source.output / row["path"]
    if kind == "bytes":
        path.write_bytes(b"z" * row["bytes"])
    elif kind == "unlisted":
        (source.output / "extra.bin").write_bytes(b"extra")
    elif kind == "missing":
        path.unlink()
    elif kind == "receipt":
        (source.output / "manifest.sha256").write_text("0" * 64 + "  manifest.jsonl\n")
    elif kind == "symlink":
        path.unlink()
        path.symlink_to(source.development_root / row["path"])
    elif kind == "registry":
        reg_path = source.output / "task_registry.json"
        registry = json.loads(reg_path.read_text())
        registry["components"][0]["development_evaluation"]["reader_id"] = "obsolete"
        changed = write(source.output, "task_registry.json", json.dumps(registry))
        rows = [{**r, **changed} if r["path"] == "task_registry.json" else r for r in rows]
        seal(source.output, rows)
    with pytest.raises(release.ReleaseError):
        release.verify_package(source.output, 2)


def test_resealed_omission_cannot_drop_frozen_data(source):
    release.cmd_execute(source)
    rows = manifest(source.output)
    removed = next(r for r in rows if r["role"] == "dataset_payload")
    (source.output / removed["path"]).unlink()
    seal(source.output, [r for r in rows if r != removed])
    with pytest.raises(release.ReleaseError, match="coverage differs"):
        release.verify_package(source.output, 2)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../b", "a\\b", "a//b", "."])
def test_rejects_unsafe_paths(path):
    assert not release.is_safe_rel(path)
