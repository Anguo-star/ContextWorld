from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from contextworld.release_metadata import (
    FROZEN_PREDECESSOR,
    NonPortableMetadataPathError,
    frozen_predecessor_reference,
    portable_release_metadata,
    portable_release_path,
    write_portable_release_json,
)


def test_nested_release_and_predecessor_paths_are_portable_without_io(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "does-not-exist" / "release-v4"
    predecessor_root = tmp_path / "also-missing" / "release-v3"
    payload = {
        "root": str(release_root),
        "nested": [
            {"manifest": release_root / "manifest.json"},
            {
                "source_root": str(predecessor_root),
                "source_table": str(
                    predecessor_root / "validation.lance"
                ),
            },
            (release_root / "train.lance", "ordinary-label"),
        ],
    }
    original = copy.deepcopy(payload)

    result = portable_release_metadata(
        payload,
        release_root=release_root,
        frozen_predecessor_root=predecessor_root,
    )

    assert result == {
        "root": ".",
        "nested": [
            {"manifest": "manifest.json"},
            {
                "source_root": FROZEN_PREDECESSOR,
                "source_table": FROZEN_PREDECESSOR,
            },
            ["train.lance", "ordinary-label"],
        ],
    }
    assert payload == original
    assert not release_root.exists()
    assert not predecessor_root.exists()


def test_current_root_and_children_have_release_local_names(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "release"
    assert portable_release_path(
        release_root,
        release_root=release_root,
    ) == "."
    assert portable_release_path(
        release_root / "loader_validation.lance" / "data.lance",
        release_root=release_root,
    ) == "loader_validation.lance/data.lance"
    assert portable_release_path(
        "already/portable.json",
        release_root=release_root,
    ) == "already/portable.json"


def test_predecessor_reference_contains_only_symbol_and_hashes() -> None:
    receipt = frozen_predecessor_reference(
        manifest_sha256="A" * 64,
        table_sha256={
            "validation": "b" * 64,
            "loader_validation": "C" * 64,
        },
    )
    assert receipt == {
        "source": FROZEN_PREDECESSOR,
        "manifest_sha256": "a" * 64,
        "table_sha256": {
            "loader_validation": "c" * 64,
            "validation": "b" * 64,
        },
    }
    assert "/tmp/" not in json.dumps(receipt, sort_keys=True)


def test_unknown_external_absolute_path_is_rejected_with_location(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "release"
    predecessor_root = tmp_path / "predecessor"
    external = tmp_path / "unregistered-source" / "input.h5"

    with pytest.raises(
        NonPortableMetadataPathError,
        match=r"\$\.nested\[0\]\.source",
    ):
        portable_release_metadata(
            {"nested": [{"source": str(external)}]},
            release_root=release_root,
            frozen_predecessor_root=predecessor_root,
        )


def test_atomic_json_writer_sanitizes_metadata_only(tmp_path: Path) -> None:
    release_root = tmp_path / "release"
    predecessor_root = tmp_path / "predecessor"
    destination = tmp_path / "staging" / "manifest.json"
    payload = {
        "root": str(release_root),
        "frozen": str(predecessor_root / "validation.lance"),
        "rows": 123,
    }

    written = write_portable_release_json(
        destination,
        payload,
        release_root=release_root,
        frozen_predecessor_root=predecessor_root,
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == written
    assert written == {
        "root": ".",
        "frozen": FROZEN_PREDECESSOR,
        "rows": 123,
    }
    assert sorted(path.name for path in destination.parent.iterdir()) == [
        "manifest.json"
    ]


@pytest.mark.parametrize(
    "manifest_sha256,table_sha256",
    [
        ("short", {"validation": "b" * 64}),
        ("a" * 64, {"validation": "not-a-digest"}),
        ("a" * 64, {}),
    ],
)
def test_predecessor_reference_rejects_incomplete_receipts(
    manifest_sha256: str,
    table_sha256: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        frozen_predecessor_reference(
            manifest_sha256=manifest_sha256,
            table_sha256=table_sha256,
        )
