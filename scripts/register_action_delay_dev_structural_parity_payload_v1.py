#!/usr/bin/env python3
# Register the structural-parity Development Action Delay payload (1.0.3-rc1).
# 
# Third Development amendment of the ContextWorld-v1 staging package.  The
# action_delay Development payload moves from the legacy 66-table
# lexicographic selection (30 queries, no eval-seed stratification,
# source synthesis/action_delay_h7_core_training_v3/tables/val) to the
# Public-Test-structured payload built from
# configs/benchmark/tworoom_action_delay_h7_dev_structural_parity_v1.yaml:
# 300 queries, six eval seeds 52-57, 25/25 rooms and directions per seed,
# all eleven delays paired over every query, start-tuple intersection with
# the Public Test catalog equal to zero (hard-gate audited by
# scripts/audit_tworoom_action_delay_h7_dev_disjointness_v1.py).
# 
# What this run changes (and nothing else):
# * components/tworoom-action-delay/v1/development/full is moved to
#   components/tworoom-action-delay/v1/development_legacy_v1 (kept, not
#   deleted) and the new 66 ad-h7-paired-val tables are installed at the
#   original path so the reader_id and payload_id stay stable.
# * task_registry.json: the action_delay split=development payload_id=full
#   entry and its development_evaluation selection/binding are rewritten;
#   the coarse payloads and every other component are untouched.
# * manifest.jsonl: the 198 legacy payload rows move to the legacy path
#   under split=development_legacy; rows for the new payload are added;
#   the receipt is recomputed.
# * VERSION.json: 1.0.2-rc1 -> 1.0.3-rc1 plus an appended amendment record.
# Post-conditions assert every other component, both action_delay training
# payloads, the coarse development payload, and ContextWorld-v1-full stay
# byte-identical, then resolve_development_payload is run for all nine
# components.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_VERSION_BEFORE = "1.0.2-rc1"
DATASET_VERSION_AFTER = "1.0.3-rc1"
PREVIOUS_AMENDMENT_IDS = [
    "development_structural_parity_v1",
    "door_dev_payload_loader_val_revert_v1",
]
AMENDMENT_ID = "action_delay_dev_structural_parity_v1"
COMPONENT = "action_delay"
DATASET_DIR = "components/tworoom-action-delay/v1"
PAYLOAD_PREFIX = DATASET_DIR + "/development/full"
LEGACY_PREFIX = DATASET_DIR + "/development_legacy_v1"
PROVENANCE_SOURCE = "evaluation/history7/action_delay_dev_structural_parity_lance_v1"
SUPERSEDED_PROVENANCE = "synthesis/action_delay_h7_core_training_v3/tables/val"
EXPECTED_LEGACY = {
    "file_count": 198,
    "lance_table_count": 66,
    "total_bytes": 842373332,
}
NEW_SELECTION = {
    "contrasts": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "delay_values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "method": "test_structured_dev_catalog_exclusion_v1",
    "pairs_per_contrast_per_profile": 50,
    "profile_eval_seed_mapping": {
        "p000": 52,
        "p001": 53,
        "p002": 54,
        "p003": 55,
        "p004": 56,
        "p005": 57,
    },
    "profiles": 6,
    "reference_condition": 0,
    "selected_pair_count": 3000,
    "eval_seeds": [52, 53, 54, 55, 56, 57],
    "unique_queries": 300,
    "unique_queries_per_eval_seed": 50,
    "rooms_per_seed": {"left": 25, "right": 25},
    "directions_per_seed": {"up": 25, "down": 25},
    "independent_queries_per_delay": 300,
    "start_tuple_disjointness": (
        "intersection of the 300 (room, direction, x, y) start tuples with "
        "the Public Test catalog is empty; audited by "
        "scripts/audit_tworoom_action_delay_h7_dev_disjointness_v1.py"
    ),
}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _json_line(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.amend-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_row(
    path: Path,
    root: Path,
    *,
    role: str,
    component: str | None = None,
    split: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "role": role,
    }
    if component is not None:
        row["component"] = component
    if split is not None:
        row["split"] = split
    if source is not None:
        row["source_logical_path"] = source
    return row


def _verify_receipt(root: Path) -> str:
    receipt = (root / "manifest.sha256").read_text(encoding="utf-8")
    fields = receipt.strip().split()
    if len(fields) != 2 or fields[1] != "manifest.jsonl":
        raise RuntimeError(f"Malformed manifest receipt: {receipt!r}")
    observed = _sha256(root / "manifest.jsonl")
    if observed != fields[0]:
        raise RuntimeError(f"Manifest receipt mismatch: {fields[0]} != {observed}")
    return observed


def _sanitized_contract(contract: dict[str, Any]) -> dict[str, Any]:
    episode_layout = {
        key: value
        for key, value in contract["episode_layout"].items()
    }
    selection = dict(contract.get("selection", {}))
    sanitized = {
        "schema_version": 1,
        "benchmark": contract["benchmark"],
        "component": COMPONENT,
        "payload_kind": contract["payload_kind"],
        "reader_id": contract["reader_id"],
        "episode_layout": episode_layout,
        "lance_table_count": contract["lance_table_count"],
        "selection": selection,
        "stable_worldmodel_commit": contract["stable_worldmodel_commit"],
    }
    if "source_catalog" in contract:
        sanitized["source_catalog_sha256"] = contract["source_catalog"].get(
            "sha256"
        )
    text = json.dumps(sanitized, ensure_ascii=False)
    for marker in ("/opt/huawei/", "/root/", "explorer-env"):
        if marker in text:
            raise RuntimeError(
                f"Sanitized contract still contains private path marker {marker!r}"
            )
    return sanitized

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install the structural-parity Development Action Delay payload "
            "(amendment 1.0.3-rc1)"
        ),
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--development-root",
        type=Path,
        required=True,
        help="Artifact root holding data/ and the registry contract of the "
        "new payload",
    )
    parser.add_argument(
        "--full-bundle-root",
        type=Path,
        default=Path(
            "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
            "ContextWorld-v1-full"
        ),
        help="Read-only sentinel: its manifest digest is asserted unchanged",
    )
    args = parser.parse_args()

    started = time.monotonic()
    root = args.bundle_root.resolve()
    source = args.development_root.resolve()
    data = source / "data"
    contract_path = source / "development_registry_contract.json"
    if not data.is_dir() or not contract_path.is_file():
        raise FileNotFoundError(f"New action delay payload incomplete: {source}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("payload_kind") != "development_structural_parity_v1":
        raise RuntimeError("Registry contract is not the structural-parity kind")
    if contract.get("reader_id") != "tworoom_action_delay_paired_collection_v1":
        raise RuntimeError("Registry contract reader id changed")
    new_tables = sorted(p for p in data.iterdir() if p.suffix == ".lance")
    declared_table_count = int(contract["lance_table_count"])
    if len(new_tables) != declared_table_count:
        raise RuntimeError(
            f"table count {len(new_tables)} disagrees with the contract "
            f"{declared_table_count}",
        )

    manifest_before = _verify_receipt(root)
    full_digest_before = _sha256(args.full_bundle_root / "manifest.jsonl")
    version = json.loads((root / "VERSION.json").read_text(encoding="utf-8"))
    if version["dataset_version"] != DATASET_VERSION_BEFORE:
        version_label = str(version["dataset_version"])
        raise RuntimeError(f"Unexpected dataset_version: {version_label}")
    if [row.get("amendment_id") for row in version.get("amendments", [])] != (
        PREVIOUS_AMENDMENT_IDS
    ):
        observed_amendments = [
            row.get("amendment_id")
            for row in version.get("amendments", [])
        ]
        raise RuntimeError(
            "Expected exactly the two prior amendments, found "
            f"{observed_amendments}",
        )
    registry = json.loads((root / "task_registry.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    matches = [
        entry for entry in registry["components"]
        if entry.get("component_id") == COMPONENT
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Registry component lookup failed: {COMPONENT}")
    entry = matches[0]
    payload = None
    for candidate in entry["payloads"]:
        if (
            candidate.get("split") == "development"
            and candidate.get("payload_id") == "full"
        ):
            payload = candidate
            break
    if payload is None:
        raise RuntimeError("No development/full payload for action_delay")
    for key, expected in EXPECTED_LEGACY.items():
        if int(payload[key]) != expected:
            raise RuntimeError(
                f"Legacy payload {key} is {payload[key]}, expected {expected}"
            )
    if payload["provenance"]["source_logical_path"] != SUPERSEDED_PROVENANCE:
        raise RuntimeError("Installed payload provenance is unexpected")
    if int(entry["development_evaluation"]["selection"]["pairs_per_contrast_per_profile"]) != 5:
        raise RuntimeError("Installed selection contract is not the legacy 5-pair one")

    component_dir = root / DATASET_DIR
    development = component_dir / "development"
    legacy = component_dir / "development_legacy_v1"
    old_data = development / "full"
    if legacy.exists() or not old_data.is_dir():
        raise RuntimeError(
            "action_delay: refusing to amend a tree not in the expected "
            "1.0.2-rc1 layout"
        )
    training_before = _tree_snapshot(component_dir / "training")
    coarse_before = _tree_snapshot(development / "coarse")
    others_before = {
        row["path"]: row["sha256"]
        for row in rows
        if row.get("role") == "dataset_payload"
        and row.get("component") != COMPONENT
    }
    legacy_rows_before = [
        row
        for row in rows
        if row.get("role") == "dataset_payload"
        and row.get("split") == "development_legacy"
    ]

    old_files = sum(1 for path in old_data.rglob("*") if path.is_file())
    old_tables = sorted(p for p in old_data.iterdir() if p.suffix == ".lance")
    old_bytes = sum(
        row["bytes"]
        for row in rows
        if row.get("role") == "dataset_payload"
        and row.get("component") == COMPONENT
        and row.get("split") == "development"
        and str(row["path"]).startswith(PAYLOAD_PREFIX + "/")
    )
    if (
        len(old_tables) != EXPECTED_LEGACY["lance_table_count"]
        or old_files != EXPECTED_LEGACY["file_count"]
        or old_bytes != EXPECTED_LEGACY["total_bytes"]
    ):
        raise RuntimeError(
            f"Legacy action delay payload disagrees: tables={len(old_tables)} "
            f"files={old_files} bytes={old_bytes}"
        )

    # The bundle lives on NFS, which refuses this directory rename
    # (os.rename -> EPERM).  Copy the legacy payload to its archive path,
    # verify the copy matches file-for-file and byte-for-byte, and only then
    # remove the original.  The data therefore exists in both places before
    # anything is deleted, so an interruption cannot lose the legacy payload.
    shutil.copytree(old_data, legacy)
    copied_files = sorted(
        str(q.relative_to(legacy)) for q in legacy.rglob("*") if q.is_file()
    )
    source_files = sorted(
        str(q.relative_to(old_data)) for q in old_data.rglob("*") if q.is_file()
    )
    if copied_files != source_files:
        raise RuntimeError(
            "Legacy payload copy is missing files: "
            f"{sorted(set(source_files) - set(copied_files))[:5]}"
        )
    copied_bytes = sum(q.stat().st_size for q in legacy.rglob("*") if q.is_file())
    if copied_bytes != old_bytes:
        raise RuntimeError(
            f"Legacy payload copy is {copied_bytes} bytes, expected {old_bytes}"
        )
    for relative in source_files:
        if _sha256(old_data / relative) != _sha256(legacy / relative):
            raise RuntimeError(f"Legacy payload copy differs at {relative}")
    shutil.rmtree(old_data)
    legacy_rows = []
    for row in rows:
        if (
            row.get("role") == "dataset_payload"
            and row.get("component") == COMPONENT
            and row.get("split") == "development"
            and str(row["path"]).startswith(PAYLOAD_PREFIX + "/")
        ):
            moved = dict(row)
            moved["path"] = str(row["path"]).replace(
                PAYLOAD_PREFIX, LEGACY_PREFIX, 1
            )
            moved["split"] = "development_legacy"
            legacy_rows.append(moved)
    moved_paths = {row["path"] for row in legacy_rows}
    for table in sorted(legacy.rglob("*")):
        if not table.is_file():
            continue
        relative = table.relative_to(legacy).as_posix()
        if LEGACY_PREFIX + "/" + relative not in moved_paths:
            raise RuntimeError(f"Untracked legacy file after move: {relative}")
    for row in legacy_rows:
        path = root / str(row["path"])
        if not path.is_file() or _sha256(path) != row["sha256"]:
            moved_label = str(row["path"])
            raise RuntimeError(f"Legacy file changed during move: {moved_label}")

    new_data = development / "full"
    new_data.mkdir()
    for table in new_tables:
        shutil.copytree(table, new_data / table.name)
    new_rows = []
    new_files = 0
    new_bytes = 0
    for path in sorted(new_data.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(new_data).as_posix()
        new_rows.append(
            _manifest_row(
                path,
                root,
                role="dataset_payload",
                component=COMPONENT,
                split="development",
                source=PROVENANCE_SOURCE + "/data/" + relative,
            )
        )
        new_files += 1
        new_bytes += path.stat().st_size
    installed_tables = sorted(
        p for p in new_data.iterdir() if p.suffix == ".lance"
    )
    if len(installed_tables) != len(new_tables):
        raise RuntimeError("Installed table count disagrees with the payload source")

    package_contract_path = development / "registry_contract.json"
    _write_bytes(
        package_contract_path,
        _json_bytes(_sanitized_contract(contract)),
    )
    contract_row_path = package_contract_path.relative_to(root).as_posix()
    member_paths = [
        PAYLOAD_PREFIX + "/" + table.name for table in installed_tables
    ]

    payload["file_count"] = new_files
    payload["total_bytes"] = new_bytes
    payload["lance_table_count"] = len(installed_tables)
    payload["members"] = member_paths
    payload["provenance"] = {
        "source_logical_path": PROVENANCE_SOURCE,
        "excluded_paths": [],
    }
    evaluation = entry["development_evaluation"]
    evaluation["selection"] = dict(NEW_SELECTION)
    evaluation["payload"] = {
        "public_path": payload["public_path"],
        "payload_kind": payload["payload_kind"],
        "lance_table_count": payload["lance_table_count"],
        "members": list(payload["members"]),
    }

    rows = [
        row
        for row in rows
        if not (
            row.get("role") == "dataset_payload"
            and row.get("component") == COMPONENT
            and row.get("split") == "development"
            and str(row["path"]).startswith(PAYLOAD_PREFIX + "/")
        )
        and row.get("path") != contract_row_path
    ]
    rows.extend(legacy_rows)
    rows.extend(new_rows)
    rows.append(
        _manifest_row(package_contract_path, root, role="release_metadata")
    )

    version["dataset_version"] = DATASET_VERSION_AFTER
    version["amendments"] = version["amendments"] + [
        {
            "amendment_id": AMENDMENT_ID,
            "applied": "2026-09-10",
            "summary": (
                "action_delay Development payload replaced by the "
                "Public-Test-structured structural-parity payload (300 "
                "queries, six eval seeds 52-57 with 25/25 rooms and "
                "directions per seed, all eleven delays paired over every "
                "query); the legacy 66-table lexicographic 30-query payload "
                "moves to development_legacy_v1 byte-identical; the 300 "
                "(room, direction, x, y) start tuples are disjoint from the "
                "Public Test catalog (intersection 0)"
            ),
            "components": {
                COMPONENT: {
                    "legacy_tables_moved": len(old_tables),
                    "legacy_files_moved": old_files,
                    "legacy_bytes": old_bytes,
                    "new_tables": len(installed_tables),
                    "new_files": new_files,
                    "new_bytes": new_bytes,
                    "queries": {"before": 30, "after": 300},
                    "eval_seeds": {
                        "public_test": [42, 43, 44, 45, 46, 47],
                        "development": [52, 53, 54, 55, 56, 57],
                    },
                    "catalog_seed": {
                        "public_test": 20260728,
                        "development": 20260910,
                    },
                    "start_tuple_intersection_with_public_test": 0,
                    "disjointness_audited_by": (
                        "scripts/audit_tworoom_action_delay_h7_dev_"
                        "disjointness_v1.py"
                    ),
                    "legacy_payload_retained_at": LEGACY_PREFIX,
                },
            },
            "script": (
                "scripts/register_action_delay_dev_structural_parity_"
                "payload_v1.py"
            ),
        },
    ]
    _write_bytes(root / "VERSION.json", _json_bytes(version))
    _write_bytes(root / "task_registry.json", _json_bytes(registry))

    for name in ("VERSION.json", "task_registry.json"):
        path = root / name
        rows = [
            _manifest_row(path, root, role="release_metadata")
            if row.get("path") == name
            else row
            for row in rows
        ]
    rows.sort(key=lambda row: row["path"])
    _write_bytes(
        root / "manifest.jsonl",
        b"".join(_json_line(row) for row in rows),
    )
    manifest_after = _sha256(root / "manifest.jsonl")
    (root / "manifest.sha256").write_text(
        manifest_after + "  manifest.jsonl" + chr(10), encoding="ascii"
    )

    full_digest_after = _sha256(args.full_bundle_root / "manifest.jsonl")
    if full_digest_after != full_digest_before:
        raise RuntimeError("ContextWorld-v1-full changed during amendment")
    training_after = _tree_snapshot(component_dir / "training")
    if training_after != training_before:
        raise RuntimeError("action_delay training payloads changed")
    coarse_after = _tree_snapshot(development / "coarse")
    if coarse_after != coarse_before:
        raise RuntimeError("action_delay coarse development payload changed")
    legacy_after = _tree_snapshot(legacy)
    expected_legacy_tree = {
        str(row["path"]).replace(LEGACY_PREFIX + "/", "", 1): row["sha256"]
        for row in legacy_rows
    }
    if legacy_after != expected_legacy_tree:
        raise RuntimeError("development_legacy_v1 does not match its manifest rows")
    others_after = {
        row["path"]: row["sha256"]
        for row in rows
        if row.get("role") == "dataset_payload"
        and row.get("component") != COMPONENT
    }
    if others_after != others_before:
        raise RuntimeError("Another component payload changed during amendment")
    legacy_rows_after = [
        row
        for row in rows
        if row.get("role") == "dataset_payload"
        and row.get("split") == "development_legacy"
        and row.get("component") in ("speed", "door")
    ]
    if len(legacy_rows_after) != len(legacy_rows_before):
        raise RuntimeError("speed/door legacy rows changed during amendment")
    _verify_receipt(root)

    from contextworld.training.stablewm_bundle import (
        resolve_contextworld_development_payload,
    )

    resolution = {}
    for other in registry["components"]:
        other_id = str(other["component_id"])
        resolved = resolve_contextworld_development_payload(
            root, component=other_id
        )
        resolution[other_id] = {
            "payload_id": resolved["payload_id"],
            "members": len(resolved["relative_members"]),
            "manifest_sha256": resolved["manifest_sha256"][:12],
        }
    expected_components = {
        str(other["component_id"]) for other in registry["components"]
    }
    if set(resolution) != expected_components:
        raise RuntimeError("Not every component resolved its development payload")

    report = {
        "status": "amended",
        "amendment_id": AMENDMENT_ID,
        "bundle_root": str(root),
        "dataset_version": {
            "before": DATASET_VERSION_BEFORE,
            "after": DATASET_VERSION_AFTER,
        },
        "manifest_sha256": {"before": manifest_before, "after": manifest_after},
        "task_registry_sha256": _sha256(root / "task_registry.json"),
        "manifest_rows": len(rows),
        "components": {
            COMPONENT: {
                "legacy_tables_moved": len(old_tables),
                "legacy_files_moved": old_files,
                "legacy_bytes": old_bytes,
                "new_tables": len(installed_tables),
                "new_files": new_files,
                "new_bytes": new_bytes,
            }
        },
        "resolution": resolution,
        "full_bundle_untouched": True,
        "training_untouched": True,
        "coarse_untouched": True,
        "other_components_untouched": True,
        "runtime_seconds": round(time.monotonic() - started, 1),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
