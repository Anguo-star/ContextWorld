#!/usr/bin/env python3
"""Register the structural-parity Development payloads into ContextWorld-v1.

Stage-4 amendments of the ``ContextWorld-v1`` staging package.  Two
amendments have been applied through this script; both keep the legacy
payloads under ``development_legacy_v1/`` byte-identical and never touch
Public Test content or ``ContextWorld-v1-full``:

* ``development_structural_parity_v1`` (1.0.0-rc1 -> 1.0.1-rc1, historical):
  moved the legacy speed/door Development payloads to
  ``development_legacy_v1/`` and installed the Public-Test-structured Lance
  payloads.  Door was installed from the reset-isolated variant
  (``hidden_passage_dev_reset_isolated_v1``, 318 tables).
* ``door_dev_payload_loader_val_revert_v1`` (1.0.1-rc1 -> 1.0.2-rc1, this
  run): reverts ONLY the door Development payload to the loader_val variant
  (``hidden_passage_dev_structural_parity_v1``, 48 tables).  Reset-tuple
  deduplication against the Public Test had forced a train-union-loader_val
  gate pool, so 257/300 queries sat on training-seen gate positions and the
  Development main score saturated at 1.0000, destroying the dev set's
  selection ability; gate-position-level isolation already keeps every
  (gate position, episode, query) data row disjoint from the Public Test.
  The 98 reset coordinate tuples that coincide with Public Test resets are
  registered in the amendment record and deliberately not eliminated.

What this run changes (and nothing else):

* ``components/tworoom-door/v1/development/data``: the 318 ``hpdev-*``
  reset-isolated tables are removed (only after asserting their bytes are
  preserved at the ``hidden_passage_dev_reset_isolated_v1`` artifact root)
  and the 48 loader_val ``hpdev-*`` tables are installed.
* ``components/tworoom-door/v1/development/registry_contract.json`` is
  rewritten from the loader_val payload contract (sanitized).
* ``task_registry.json``: the door split=development payload entry and its
  ``development_evaluation`` binding are rewritten (the
  ``reset_tuple_isolation`` selection claim is dropped and replaced by the
  registered reset-tuple overlap note); the legacy and speed entries are
  untouched.
* ``manifest.jsonl`` rows under the door Development payload are replaced;
  the manifest receipt is recomputed.
* ``VERSION.json``: ``dataset_version`` 1.0.1-rc1 -> 1.0.2-rc1 plus an
  appended amendment record; the existing amendment is kept verbatim.

No full re-export is performed (``export_hf_clean`` would recopy ~17 GB and
``refresh_hf_clean_metadata`` cannot add payload rows); file bytes of every
pre-existing payload other than the replaced door tables are untouched.  The
package contains no Public Test artifacts and this script must not create
any; ``ContextWorld-v1-full`` is never opened for writing.
"""

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

DATASET_VERSION_BEFORE = "1.0.1-rc1"
DATASET_VERSION_AFTER = "1.0.2-rc1"
PREVIOUS_AMENDMENT_ID = "development_structural_parity_v1"
AMENDMENT_ID = "door_dev_payload_loader_val_revert_v1"
# Measured (K2b isolation audit): distinct reset coordinate tuples shared by
# the loader_val dev queries and the Public Test selected queries.  Shared
# coordinate values are not shared data rows (the gate positions are
# disjoint), so the overlap is registered, not eliminated.
RESET_TUPLE_OVERLAP_WITH_TEST = 98

_DOOR_SPEC: dict[str, Any] = {
    "dataset_dir": "components/tworoom-door/v1",
    "provenance_source": "evaluation/history3/hidden_passage_dev_structural_parity_v1",
    "superseded_provenance_source": (
        "evaluation/history3/hidden_passage_dev_reset_isolated_v1"
    ),
    "expected_benchmark": "tworoom_hidden_passage_history3_dev_structural_parity_v1",
    "development_evaluation": {
        "reader_id": "contextworld_door_dev_structural_parity_lance_v1",
        "selection": {
            "unit": "static_query_history_condition_episode",
            "eval_seeds": [42, 43, 44, 45, 46, 47],
            "directions": ["left_to_right", "right_to_left"],
            "unique_queries_per_eval_seed": 50,
            "per_direction_per_eval_seed": 25,
            "unique_queries": 300,
            "history_conditions": [
                "observed_passable",
                "observed_blocked",
                "did_not_attempt_crossing",
            ],
            "rows_per_episode": 20,
            "method": "seeded_disjoint_without_replacement_query_sets",
            "reset_tuple_overlap": (
                "98 reset coordinate tuples coincide with Public Test "
                "resets; registered, not isolated - gate positions are "
                "disjoint, so no (gate position, episode, query) data row "
                "is shared"
            ),
        },
    },
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
        raise RuntimeError(
            f"Manifest receipt mismatch: {fields[0]} != {observed}"
        )
    return observed


def _sanitized_contract(
    contract: dict[str, Any], *, component: str, benchmark: str
) -> dict[str, Any]:
    """In-package copy of the Lance registry contract, without private paths."""

    episode_layout = {
        key: value
        for key, value in contract["episode_layout"].items()
        if "path" not in key
    }
    selection = {
        key: value
        for key, value in contract.get("selection", {}).items()
        if key != "tracks"
    }
    if "tracks" in contract.get("selection", {}):
        selection["tracks"] = {
            name: {"reference_speeds": row["reference_speeds"]}
            for name, row in contract["selection"]["tracks"].items()
        }
    sanitized = {
        "schema_version": 1,
        "benchmark": benchmark,
        "component": component,
        "payload_kind": contract["payload_kind"],
        "reader_id": contract["reader_id"],
        "episode_layout": episode_layout,
        "lance_table_count": contract["lance_table_count"],
        "selection": selection,
    }
    for key in (
        "history_conditions",
        "true_future_rules",
        "stable_worldmodel_commit",
    ):
        if key in contract:
            sanitized[key] = contract[key]
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


def _assert_bytes_preserved_elsewhere(
    tables: list[Path], superseded_root: Path
) -> None:
    """Refuse to delete tables whose only copy lives in this package."""

    for table in tables:
        mirror = superseded_root / table.name
        if not mirror.is_dir():
            raise RuntimeError(
                f"Superseded table has no preserved copy: {mirror}"
            )
        local = {
            path.relative_to(table).as_posix(): _sha256(path)
            for path in sorted(table.rglob("*"))
            if path.is_file()
        }
        remote = {
            path.relative_to(mirror).as_posix(): _sha256(path)
            for path in sorted(mirror.rglob("*"))
            if path.is_file()
        }
        if local != remote:
            raise RuntimeError(
                f"Superseded table differs from its preserved copy: {table}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Revert the door Development payload of ContextWorld-v1 to the "
            "loader_val structural-parity payload (amendment 1.0.2-rc1)"
        )
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--door-development-root",
        type=Path,
        help="Directory containing data/ and development_registry_contract.json "
        "of the loader_val door payload",
        required=True,
    )
    parser.add_argument(
        "--superseded-door-root",
        type=Path,
        required=True,
        help="Artifact-root copy of the reset-isolated payload being removed; "
        "asserted byte-identical before anything is deleted",
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
    spec = _DOOR_SPEC
    component = "door"

    source = args.door_development_root.resolve()
    data = source / "data"
    contract_path = source / "development_registry_contract.json"
    if not data.is_dir() or not contract_path.is_file():
        raise FileNotFoundError(f"Loader_val door payload incomplete: {source}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if str(contract["benchmark"]) != spec["expected_benchmark"]:
        raise RuntimeError(
            f"Door contract benchmark is {contract['benchmark']!r}, expected "
            f"{spec['expected_benchmark']!r}"
        )
    tables = sorted(p for p in data.iterdir() if p.suffix == ".lance")
    if len(tables) != int(contract["lance_table_count"]):
        raise RuntimeError(
            f"door: table count {len(tables)} disagrees with its registry "
            f"contract {contract['lance_table_count']}"
        )

    manifest_before = _verify_receipt(root)
    full_digest_before = _sha256(args.full_bundle_root / "manifest.jsonl")
    version = json.loads((root / "VERSION.json").read_text(encoding="utf-8"))
    if version["dataset_version"] != DATASET_VERSION_BEFORE:
        raise RuntimeError(
            f"Unexpected dataset_version: {version['dataset_version']}"
        )
    amendments = version.get("amendments", [])
    if [row.get("amendment_id") for row in amendments] != [PREVIOUS_AMENDMENT_ID]:
        raise RuntimeError(
            "Expected exactly one prior amendment "
            f"{PREVIOUS_AMENDMENT_ID!r}, found "
            f"{[row.get('amendment_id') for row in amendments]}"
        )
    registry = json.loads((root / "task_registry.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    component_dir = root / spec["dataset_dir"]
    development = component_dir / "development"
    legacy = component_dir / "development_legacy_v1"
    old_data = development / "data"
    if not legacy.is_dir() or not old_data.is_dir():
        raise RuntimeError(
            "door: refusing to amend a tree that is not in the expected "
            "1.0.1-rc1 layout"
        )
    # The legacy payload and the speed payload must survive verbatim.
    legacy_before = _tree_snapshot(legacy)
    speed_before = {
        row["path"]: row["sha256"]
        for row in rows
        if row.get("component") == "speed" and row.get("split") == "development"
    }

    # 1. Remove the reset-isolated tables, but only once their bytes are
    #    proven to live on at the superseded artifact root.
    superseded_root = (
        args.superseded_door_root / "development" / "data"
    ).resolve()
    current_tables = sorted(p for p in old_data.iterdir() if p.suffix == ".lance")
    if not current_tables:
        raise RuntimeError("door: no Development payload installed to replace")
    _assert_bytes_preserved_elsewhere(current_tables, superseded_root)
    old_prefix = old_data.relative_to(root).as_posix()
    removed_rows = [
        row
        for row in rows
        if row.get("role") == "dataset_payload"
        and row.get("component") == component
        and row.get("split") == "development"
        and row["path"].startswith(old_prefix + "/")
    ]
    removed_files = len(removed_rows)
    removed_bytes = sum(row["bytes"] for row in removed_rows)
    removed_tables = len(current_tables)
    installed_files = sum(
        1 for path in old_data.rglob("*") if path.is_file()
    )
    if len(removed_rows) != installed_files:
        raise RuntimeError(
            f"door: manifest lists {len(removed_rows)} payload files but the "
            f"tree holds {installed_files}"
        )
    for table in current_tables:
        shutil.rmtree(table)

    # 2. Install the loader_val payload and rewrite its registry contract.
    new_data = old_data
    new_tables = 0
    for table in tables:
        shutil.copytree(table, new_data / table.name)
        new_tables += 1
    new_files = 0
    new_bytes = 0
    new_rows = []
    for path in sorted(new_data.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(new_data).as_posix()
        new_rows.append(
            _manifest_row(
                path,
                root,
                role="dataset_payload",
                component=component,
                split="development",
                source=(
                    spec["provenance_source"]
                    + "/development_lance/data/"
                    + relative
                ),
            )
        )
        new_files += 1
        new_bytes += path.stat().st_size
    package_contract_path = development / "registry_contract.json"
    _write_bytes(
        package_contract_path,
        _json_bytes(
            _sanitized_contract(
                contract,
                component=component,
                benchmark=str(contract["benchmark"]),
            )
        ),
    )

    # 3. Rewrite the manifest rows for this payload.
    contract_row_path = package_contract_path.relative_to(root).as_posix()
    rows = [
        row
        for row in rows
        if not (
            row.get("role") == "dataset_payload"
            and row.get("component") == component
            and row.get("split") == "development"
            and row["path"].startswith(old_prefix + "/")
        )
        and row.get("path") != contract_row_path
    ]
    rows.extend(new_rows)
    rows.append(
        _manifest_row(package_contract_path, root, role="release_metadata")
    )

    # 4. Rewrite the task registry entry for this component.
    matches = [
        entry
        for entry in registry["components"]
        if entry.get("component_id") == component
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Registry component lookup failed: {component}")
    entry = matches[0]
    payload = None
    for candidate in entry["payloads"]:
        if (
            candidate.get("split") == "development"
            and candidate.get("payload_id") == "data"
        ):
            payload = candidate
            break
    if payload is None:
        raise RuntimeError(f"No development payload for {component}")
    if (
        payload.get("provenance", {}).get("source_logical_path")
        != spec["superseded_provenance_source"]
    ):
        raise RuntimeError(
            "Installed door payload provenance is not the reset-isolated "
            f"variant: {payload.get('provenance')}"
        )
    payload["file_count"] = new_files
    payload["total_bytes"] = new_bytes
    payload["lance_table_count"] = new_tables
    payload["members"] = [
        spec["dataset_dir"] + "/development/data/" + table.name
        for table in sorted(new_data.iterdir())
        if table.suffix == ".lance"
    ]
    payload["provenance"] = {
        "source_logical_path": spec["provenance_source"],
        "excluded_paths": [],
    }
    evaluation = dict(entry["development_evaluation"])
    evaluation.update(spec["development_evaluation"])
    evaluation["payload"] = {
        "public_path": payload["public_path"],
        "payload_kind": payload["payload_kind"],
        "lance_table_count": payload["lance_table_count"],
        "members": list(payload["members"]),
    }
    entry["development_evaluation"] = evaluation

    # 5. Append the amendment record and bump the version.
    version["dataset_version"] = DATASET_VERSION_AFTER
    version["amendments"] = amendments + [
        {
            "amendment_id": AMENDMENT_ID,
            "applied": "2026-09-09",
            "summary": (
                "door Development payload reverted from "
                "tworoom_hidden_passage_history3_dev_reset_isolated_v1 to the "
                "loader_val structural-parity payload: reset-tuple "
                "deduplication against the Public Test forced a "
                "train-union-loader_val gate pool (257/300 queries on "
                "training-seen gate positions), destroying the dev set's "
                "selection ability, while gate-position-level isolation "
                "already keeps every (gate position, episode, query) data "
                "row disjoint from the Public Test"
            ),
            "supersedes": {
                "amendment_id": PREVIOUS_AMENDMENT_ID,
                "scope": "components/tworoom-door/v1 split=development only",
            },
            "components": {
                component: {
                    "removed_tables": removed_tables,
                    "removed_files": removed_files,
                    "removed_bytes": removed_bytes,
                    "installed_tables": new_tables,
                    "installed_files": new_files,
                    "installed_bytes": new_bytes,
                    "superseded_payload_retained_at": (
                        "evaluation/history3/"
                        "hidden_passage_dev_reset_isolated_v1 (artifact "
                        "root outside this package)"
                    ),
                    "reset_tuple_overlap_with_public_test": (
                        RESET_TUPLE_OVERLAP_WITH_TEST
                    ),
                    "reset_tuple_overlap_policy": (
                        "registered, not eliminated; shared coordinate "
                        "values do not create shared data rows because the "
                        "gate positions are disjoint"
                    ),
                },
            },
            "script": "scripts/register_dev_structural_parity_payloads_v1.py",
        }
    ]
    _write_bytes(root / "VERSION.json", _json_bytes(version))
    _write_bytes(root / "task_registry.json", _json_bytes(registry))

    # 6. Rebind manifest rows for regenerated metadata and rewrite the
    #    manifest + receipt.
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
        f"{manifest_after}  manifest.jsonl\n", encoding="ascii"
    )

    # 7. Post-conditions: nothing outside the door payload moved.
    full_digest_after = _sha256(args.full_bundle_root / "manifest.jsonl")
    if full_digest_after != full_digest_before:
        raise RuntimeError("ContextWorld-v1-full changed during amendment")
    legacy_after = _tree_snapshot(legacy)
    if legacy_after != legacy_before:
        raise RuntimeError("development_legacy_v1 changed during amendment")
    speed_after = {
        row["path"]: row["sha256"]
        for row in rows
        if row.get("component") == "speed" and row.get("split") == "development"
    }
    if speed_after != speed_before:
        raise RuntimeError("speed Development payload changed during amendment")

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
            component: {
                "removed_tables": removed_tables,
                "removed_files": removed_files,
                "removed_bytes": removed_bytes,
                "installed_tables": new_tables,
                "installed_files": new_files,
                "installed_bytes": new_bytes,
            }
        },
        "legacy_untouched": True,
        "speed_untouched": True,
        "full_bundle_untouched": True,
        "runtime_seconds": round(time.monotonic() - started, 1),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
