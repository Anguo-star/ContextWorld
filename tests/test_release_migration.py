from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import lance
import pyarrow as pa
import pytest

import contextworld.release_migration as release_migration

from contextworld.release_migration import (
    LANCE_SPLITS,
    METADATA_FILES,
    PORTABILITY_RECEIPT,
    absolute_json_path_audit,
    canonical_json_sha256,
    commit_prepared_migration,
    file_sha256,
    lance_table_identity,
    hardlink_clone_release,
    prepare_release_migration,
)


def _write_lance(path: Path, *, split_index: int) -> None:
    table = pa.table(
        {
            "episode_idx": pa.array([0, 0], type=pa.int32()),
            "step_idx": pa.array([0, 1], type=pa.int32()),
            "pixels": pa.array(
                [
                    bytes([split_index, 0, 1]),
                    bytes([split_index, 2, 3]),
                ],
                type=pa.binary(),
            ),
            "action": pa.array(
                [[0.1, 0.2], [0.3, 0.4]],
                type=pa.list_(pa.float32(), 2),
            ),
        }
    )
    lance.write_dataset(table, str(path), mode="create")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fake_release(root: Path, predecessor: Path) -> None:
    root.mkdir()
    table_hashes = {}
    for index, split in enumerate(LANCE_SPLITS):
        table_path = root / f"{split}.lance"
        _write_lance(table_path, split_index=index)
        table_hashes[split] = lance_table_identity(table_path)[
            "tree_sha256"
        ]

    predecessor_manifest = "a" * 64
    request = {
        "protocol": "test_portability",
        "pair_counts": {split: 1 for split in LANCE_SPLITS},
        "evaluation_reuse_source": {
            "root": str(predecessor),
            "manifest_sha256": predecessor_manifest,
        },
    }
    manifest = {
        **request,
        "request_sha256": "0" * 64,
        "splits": {
            split: {
                "table_path": f"{split}.lance",
                "pair_count": 1,
                "frozen_split_reuse": {
                    "source_root": str(predecessor),
                    "source_manifest_sha256": predecessor_manifest,
                    "source_table_sha256": table_hashes[split],
                    "destination_table_sha256": table_hashes[split],
                    "passed": True,
                },
                "passed": True,
            }
            for split in LANCE_SPLITS
        },
        "passed": True,
    }
    _write_json(root / "request.json", request)
    _write_json(root / "manifest.json", manifest)
    _write_json(
        root / "build_report.json",
        {
            "root": str(root),
            "manifest_sha256": file_sha256(root / "manifest.json"),
            "source_root": str(predecessor),
            "passed": True,
        },
    )
    _write_json(
        root / "strict_causal_audit.json",
        {"release": str(root), "passed": True},
    )


def test_prepare_is_metadata_only_and_does_not_need_predecessor(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release-v4"
    predecessor = tmp_path / "missing-release-v3"
    _fake_release(release, predecessor)
    before_metadata = {
        name: file_sha256(release / name) for name in METADATA_FILES
    }
    before_lance = {
        split: lance_table_identity(release / f"{split}.lance")
        for split in LANCE_SPLITS
    }
    before_inodes = {
        path.relative_to(release).as_posix(): (
            path.stat().st_dev,
            path.stat().st_ino,
        )
        for split in LANCE_SPLITS
        for path in (release / f"{split}.lance").rglob("*")
        if path.is_file()
    }

    result = prepare_release_migration(
        release_root=release,
        frozen_predecessor_root=predecessor,
    )
    staging = Path(result["staging_root"])

    assert result["passed"] is True
    assert not predecessor.exists()
    assert absolute_json_path_audit(staging)["absolute_path_count"] == 0
    assert {
        name: file_sha256(release / name) for name in METADATA_FILES
    } == before_metadata
    assert not list(staging.glob("*.lance"))
    for split in LANCE_SPLITS:
        source = release / f"{split}.lance"
        assert lance_table_identity(source) == before_lance[split]
    after_inodes = {
        path.relative_to(release).as_posix(): (
            path.stat().st_dev,
            path.stat().st_ino,
        )
        for split in LANCE_SPLITS
        for path in (release / f"{split}.lance").rglob("*")
        if path.is_file()
    }
    assert after_inodes == before_inodes

    request = json.loads((staging / "request.json").read_text())
    manifest = json.loads((staging / "manifest.json").read_text())
    report = json.loads((staging / "build_report.json").read_text())
    receipt = json.loads((staging / PORTABILITY_RECEIPT).read_text())
    reference = request["evaluation_reuse_source"]
    assert reference["source"] == "frozen_predecessor"
    assert reference["manifest_sha256"] == "a" * 64
    assert set(reference["table_sha256"]) == {
        "loader_validation",
        "validation",
    }
    assert manifest["request_sha256"] == canonical_json_sha256(request)
    assert report["root"] == "."
    assert report["manifest_sha256"] == file_sha256(
        staging / "manifest.json"
    )
    assert receipt["passed"] is True
    assert receipt["lance_storage_contract"] == {
        "mode": "in_place_read_only",
        "directories_moved": False,
        "files_rewritten": False,
        "passed": True,
    }
    assert receipt["metadata_exchange_order"] == [
        "request.json",
        "manifest.json",
        "build_report.json",
        "strict_causal_audit.json",
    ]
    assert all(
        receipt["lance_tables"][split]["identical"]
        for split in LANCE_SPLITS
    )


def test_commit_uses_atomic_replace_and_keeps_rollback_backup(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release-v3"
    predecessor = tmp_path / "missing-release-v2"
    _fake_release(release, predecessor)
    original_manifest = file_sha256(release / "manifest.json")
    lance_inodes = {
        path.relative_to(release).as_posix(): (
            path.stat().st_dev,
            path.stat().st_ino,
        )
        for split in LANCE_SPLITS
        for path in (release / f"{split}.lance").rglob("*")
        if path.is_file()
    }
    preparation = prepare_release_migration(
        release_root=release,
        frozen_predecessor_root=predecessor,
    )
    staged_manifest = preparation["receipt"]["metadata_sha256"][
        "manifest.json"
    ]["after"]

    result = commit_prepared_migration(release_root=release)
    backup = Path(result["backup_root"])

    assert result["passed"] is True
    assert result["old_manifest_sha256"] == original_manifest
    assert result["new_manifest_sha256"] == staged_manifest
    assert backup.is_dir()
    assert (release / PORTABILITY_RECEIPT).is_file()
    assert not (backup / PORTABILITY_RECEIPT).exists()
    assert file_sha256(backup / "manifest.json") == original_manifest
    assert (backup / "strict_causal_audit.json").is_file()
    assert absolute_json_path_audit(release)["passed"] is True
    assert absolute_json_path_audit(backup)["passed"] is False
    assert not list(backup.glob("*.lance"))
    assert {
        path.relative_to(release).as_posix(): (
            path.stat().st_dev,
            path.stat().st_ino,
        )
        for split in LANCE_SPLITS
        for path in (release / f"{split}.lance").rglob("*")
        if path.is_file()
    } == lance_inodes
    with pytest.raises(FileExistsError, match="already has a portability"):
        commit_prepared_migration(release_root=release)


def test_replace_failure_rolls_back_in_reverse_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = tmp_path / "release-v3"
    predecessor = tmp_path / "missing-release-v2"
    _fake_release(release, predecessor)
    preparation = prepare_release_migration(
        release_root=release,
        frozen_predecessor_root=predecessor,
    )
    staging = Path(preparation["staging_root"])
    original = {
        name: file_sha256(release / name)
        for name in preparation["receipt"]["metadata_exchange_order"]
    }
    real_replace = release_migration._atomic_replace
    calls = 0

    def fail_second_replace(staged: Path, active: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected replace failure")
        real_replace(staged, active)

    monkeypatch.setattr(
        release_migration,
        "_atomic_replace",
        fail_second_replace,
    )
    with pytest.raises(OSError, match="injected replace failure"):
        commit_prepared_migration(release_root=release)

    assert {
        name: file_sha256(release / name) for name in original
    } == original
    assert not (release / PORTABILITY_RECEIPT).exists()
    assert (staging / PORTABILITY_RECEIPT).is_file()


def test_failed_hardlink_clone_removes_partial_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"payload")
    staging = tmp_path / "staging"

    def blocked_link(*args, **kwargs):
        raise PermissionError("hardlinks disabled")

    monkeypatch.setattr(os, "link", blocked_link)
    with pytest.raises(shutil.Error):
        hardlink_clone_release(source, staging)
    assert not staging.exists()


def _dynamic_release(
    root: Path,
    *,
    tables: tuple[str, ...],
    external_path: Path,
    extra_metadata: bool = False,
) -> None:
    root.mkdir()
    for index, name in enumerate(tables):
        _write_lance(root / f"{name}.lance", split_index=index)
    request = {
        "protocol": "dynamic_portability",
        "source": {
            "path": str(external_path),
            "size_bytes": 123,
        },
        "pair_counts": {name: 1 for name in tables},
    }
    manifest = {
        **request,
        "request_sha256": "0" * 64,
        "splits": {
            name: {
                "table_path": f"{name}.lance",
                "pair_count": 1,
            }
            for name in tables
        },
        "passed": True,
    }
    _write_json(root / "request.json", request)
    _write_json(root / "manifest.json", manifest)
    _write_json(
        root / "build_report.json",
        {
            "root": "/tmp/historical-build-output",
            "manifest_sha256": file_sha256(root / "manifest.json"),
            "passed": True,
        },
    )
    if extra_metadata:
        _write_json(
            root / "distribution_audit.json",
            {
                "strict_manifest": str(root / "manifest.json"),
                "strict_manifest_sha256": file_sha256(
                    root / "manifest.json"
                ),
                "passed": True,
            },
        )


def test_two_table_release_uses_semantic_source_receipt(
    tmp_path: Path,
) -> None:
    release = tmp_path / "training-release"
    external = tmp_path / "upstream" / "source.h5"
    tables = {"train": "train.lance", "validation": "validation.lance"}
    _dynamic_release(
        release,
        tables=tuple(tables),
        external_path=external,
        extra_metadata=True,
    )
    before_lance = {
        name: lance_table_identity(release / relative)
        for name, relative in tables.items()
    }
    source_sha = "b" * 64
    prepared = prepare_release_migration(
        release_root=release,
        lance_tables=tables,
        semantic_sources={
            external: {
                "symbol": "upstream_source_h5",
                "digest_role": "file_sha256",
                "sha256": source_sha,
            }
        },
    )
    staging = Path(prepared["staging_root"])
    request = json.loads((staging / "request.json").read_text())
    audit = json.loads((staging / "distribution_audit.json").read_text())
    receipt = json.loads((staging / PORTABILITY_RECEIPT).read_text())

    assert request["source"]["source"] == "upstream_source_h5"
    assert request["source"]["file_sha256"] == source_sha
    assert "path" not in request["source"]
    assert receipt["semantic_sources"] == {
        "upstream_source_h5": {"file_sha256": source_sha}
    }
    assert set(receipt["lance_tables"]) == set(tables)
    assert {
        name: receipt["lance_tables"][name]["relative_path"]
        for name in tables
    } == tables
    assert audit["strict_manifest"] == "manifest.json"
    assert audit["strict_manifest_sha256"] == file_sha256(
        staging / "manifest.json"
    )
    assert absolute_json_path_audit(staging)["passed"] is True

    result = commit_prepared_migration(release_root=release)
    assert result["passed"] is True
    assert {
        name: lance_table_identity(release / relative)
        for name, relative in tables.items()
    } == before_lance


def test_one_table_release_updates_semantic_manifest_identity(
    tmp_path: Path,
) -> None:
    release = tmp_path / "public-test"
    training_root = tmp_path / "historical-training-root"
    tables = {"validation": "validation.lance"}
    _dynamic_release(
        release,
        tables=tuple(tables),
        external_path=training_root,
    )
    for name in ("request.json", "manifest.json"):
        path = release / name
        payload = json.loads(path.read_text())
        payload["prior_training"] = {
            "root": str(training_root),
            "manifest_sha256": "1" * 64,
        }
        # The generic fixture's source path is the same external root.  Keep
        # only the release reference so one symbol has one digest role.
        payload.pop("source")
        _write_json(path, payload)
    training_manifest_sha = "c" * 64
    prepared = prepare_release_migration(
        release_root=release,
        lance_tables=tables,
        semantic_sources={
            training_root: {
                "symbol": "action_strength_training_release",
                "digest_role": "manifest_sha256",
                "sha256": training_manifest_sha,
            }
        },
    )
    staging = Path(prepared["staging_root"])
    request = json.loads((staging / "request.json").read_text())
    reference = request["prior_training"]
    assert reference == {
        "source": "action_strength_training_release",
        "manifest_sha256": training_manifest_sha,
    }
    assert set(prepared["receipt"]["lance_tables"]) == {"validation"}
    assert absolute_json_path_audit(staging)["passed"] is True
    assert commit_prepared_migration(release_root=release)["passed"] is True
