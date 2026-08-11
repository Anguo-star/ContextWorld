#!/usr/bin/env python3
"""Fully audit the expanded strict PushT contact-friction data release."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import lance


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.contact_friction_icl_data import (  # noqa: E402
    STRICT_CAUSAL_PROTOCOL,
    _pusht_physics_state_max_abs_gap,
    _read_lance_pairs,
    file_sha256,
)
from contextworld.release_migration import (  # noqa: E402
    absolute_json_path_audit,
)
from scripts.build_pusht_contact_friction_h3_data import (  # noqa: E402
    directory_file_receipts,
    directory_sha256,
)


DEFAULT_DATA_ROOT = (
    ROOT
    / "artifacts/synthesis/pusht_contact_friction_h3_release_v3"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/evaluation/history3/"
    "pusht_contact_friction_h3_strict_v3/expanded_data_release_audit.json"
)
EXPECTED_COUNTS = {
    "train": 8192,
    "loader_validation": 256,
    "validation": 256,
}
EXPECTED_MANIFEST_SHA256 = (
    "cbb9b1a1c030a3c66ea8acbf25c5e1a302f1c43907beeadcdc9d8bd1e989f3d5"
)
EXPECTED_TREE_SHA256 = (
    "bec565b20a959fd73249ab11b00d5899914fce91193d898b00c6eb6b26c01913"
)
EXPECTED_PORTABILITY_RECEIPT_SHA256 = (
    "01a6fc2519807b3de9f3d47f5a53233c79520da57247bbf4f3fc341c58a346d1"
)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    manifest_path = data_root / "manifest.json"
    request_path = data_root / "request.json"
    build_report_path = data_root / "build_report.json"
    portability_receipt_path = data_root / "portability_receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    portability_receipt = json.loads(
        portability_receipt_path.read_text(encoding="utf-8")
    )

    checks: dict[str, bool] = {
        "manifest_sha256_pinned": (
            file_sha256(manifest_path) == EXPECTED_MANIFEST_SHA256
        ),
        "build_report_manifest_sha256_matches": (
            build_report.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256
        ),
        "request_sha256_matches": (
            canonical_json_sha256(request) == manifest.get("request_sha256")
        ),
        "strict_protocol_matches": (
            manifest.get("protocol") == STRICT_CAUSAL_PROTOCOL
        ),
        "manifest_passed": manifest.get("passed") is True,
        "build_report_passed": build_report.get("passed") is True,
        "pair_counts_match": manifest.get("pair_counts") == EXPECTED_COUNTS,
        "cross_split_manifest_passed": (
            manifest.get("cross_split_audit", {}).get("passed") is True
        ),
        "expanded_training_coverage_passed": (
            manifest.get("training_coverage_vs_reused_release", {}).get(
                "passed"
            )
            is True
        ),
        "no_state_installations_after_x0": (
            manifest.get("causal_chain", {}).get(
                "state_installations_after_x0"
            )
            == 0
        ),
        "query_simulator_not_recreated": (
            manifest.get("causal_chain", {}).get(
                "query_simulator_recreated"
            )
            is False
        ),
        "portability_receipt_pinned_and_passed": (
            file_sha256(portability_receipt_path)
            == EXPECTED_PORTABILITY_RECEIPT_SHA256
            and portability_receipt.get("passed") is True
        ),
        "published_json_contains_no_absolute_paths": (
            absolute_json_path_audit(data_root).get("passed") is True
        ),
    }

    split_reports: dict[str, Any] = {}
    query_hashes: dict[str, set[str]] = {}
    template_ids: dict[str, set[str]] = {}
    for split, expected_pairs in EXPECTED_COUNTS.items():
        specification = manifest["splits"][split]
        table = data_root / specification["table_path"]
        arrays = _read_lance_pairs(
            table,
            expected_pairs=expected_pairs,
            expected_split=split,
        )
        query_gap = _pusht_physics_state_max_abs_gap(
            arrays.low_physics_states[:, 2],
            arrays.high_physics_states[:, 2],
        )
        observed_rows = int(lance.dataset(table).count_rows())
        observed_receipts = directory_file_receipts(table)
        observed_table_sha256 = directory_sha256(table)
        split_passed = bool(
            specification.get("passed") is True
            and arrays.pair_count == expected_pairs
            and observed_rows == 40 * expected_pairs
            and observed_table_sha256 == specification["table_sha256"]
            and len(observed_receipts) == int(specification["table_files"])
            and sum(row["bytes"] for row in observed_receipts)
            == int(specification["table_bytes"])
            and query_gap <= 5.0e-5
            and specification.get("state_installations_after_x0") == 0
            and specification.get("query_simulator_recreated") is False
            and specification.get("max_pair_query_pixel_difference") == 0
            and specification.get("max_pair_query_action_difference") == 0.0
        )
        split_reports[split] = {
            "pair_count": arrays.pair_count,
            "row_count": observed_rows,
            "table_sha256": observed_table_sha256,
            "table_files": len(observed_receipts),
            "table_bytes": sum(row["bytes"] for row in observed_receipts),
            "maximum_serialized_query_physics_gap": query_gap,
            "passed": split_passed,
        }
        query_hashes[split] = set(specification["query_hashes"])
        template_ids[split] = set(specification["template_ids"])
        del arrays
        gc.collect()

    pairs = [
        ("train", "loader_validation"),
        ("train", "validation"),
        ("loader_validation", "validation"),
    ]
    recomputed_overlap = {
        f"{left}__{right}": {
            "query_pixel_hashes": len(
                query_hashes[left] & query_hashes[right]
            ),
            "template_ids": len(template_ids[left] & template_ids[right]),
        }
        for left, right in pairs
    }
    checks["cross_split_overlap_recomputed_zero"] = all(
        count == 0
        for row in recomputed_overlap.values()
        for count in row.values()
    )
    checks["all_lance_payloads_decoded"] = all(
        report["passed"] for report in split_reports.values()
    )

    frozen_evaluation: dict[str, Any] = {}
    for split in ("loader_validation", "validation"):
        destination_table = data_root / f"{split}.lance"
        registered = manifest["splits"][split]["frozen_split_reuse"]
        destination_sha256 = directory_sha256(destination_table)
        migration_identity = portability_receipt["lance_tables"][split]
        passed = bool(
            registered.get("source") == "frozen_predecessor"
            and registered["source_table_sha256"]
            == registered["destination_table_sha256"]
            == destination_sha256
            and migration_identity.get("identical") is True
            and migration_identity["migration_before"]["tree_sha256"]
            == migration_identity["migration_after"]["tree_sha256"]
            == destination_sha256
        )
        frozen_evaluation[split] = {
            "source": "frozen_predecessor",
            "source_manifest_sha256": registered[
                "source_manifest_sha256"
            ],
            "source_table_sha256": registered["source_table_sha256"],
            "destination_table_sha256": destination_sha256,
            "migration_before_after_identical": migration_identity[
                "identical"
            ],
            "passed": passed,
        }
    checks["frozen_evaluation_bytes_preserved"] = all(
        row["passed"] for row in frozen_evaluation.values()
    )

    coverage = manifest["training_coverage_vs_reused_release"]
    stratification = coverage["stratification"]
    training = manifest["splits"]["train"]
    checks["training_strata_exactly_balanced"] = bool(
        stratification.get("complete_strata") == 2048
        and stratification.get("observed_strata") == 2048
        and stratification.get("pairs_per_stratum_minimum") == 4
        and stratification.get("pairs_per_stratum_maximum") == 4
        and stratification.get("balanced") is True
        and training.get("training_stratum_count") == 2048
        and training.get("training_stratum_minimum_count") == 4
        and training.get("training_stratum_maximum_count") == 4
        and set(training.get("training_stratum_counts", {}).values()) == {4}
    )

    children = sorted(path for path in data_root.rglob("*") if path.is_file())
    artifact_tree = {
        "files": len(children),
        "bytes": sum(path.stat().st_size for path in children),
        "sha256": directory_sha256(data_root),
    }
    checks["artifact_tree_sha256_pinned"] = (
        artifact_tree["sha256"] == EXPECTED_TREE_SHA256
    )
    checks["artifact_tree_shape_pinned"] = (
        artifact_tree["files"] == 13
        and artifact_tree["bytes"] == 1_030_165_054
    )

    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "data_release": data_root.name,
        "manifest_sha256": file_sha256(manifest_path),
        "portability_receipt_sha256": file_sha256(
            portability_receipt_path
        ),
        "artifact_tree": artifact_tree,
        "checks": checks,
        "split_reports": split_reports,
        "recomputed_cross_split_overlap": recomputed_overlap,
        "frozen_evaluation": frozen_evaluation,
        "passed": passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
