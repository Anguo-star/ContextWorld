#!/usr/bin/env python3
"""Bind the strict action-strength release to its distribution evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRICT_TRAINING = Path(
    "/tmp/contextworld_action_strength_strict_release/training"
)
DEFAULT_LEGACY_TRAINING = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "context_world/synthesis/"
    "pusht_hidden_actuation_replay_matched_h3_v2"
)
DEFAULT_PROTOCOL = CONTEXTWORLD_ROOT / (
    "configs/benchmark/pusht_action_strength_strict_causal_data_v1.yaml"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strict_causal_receipt(manifest: dict[str, Any]) -> dict[str, Any]:
    strict = manifest.get("strict_causal_chain_audit", {})
    split_checks = {}
    for split in ("train", "validation"):
        split_strict = manifest["splits"][split].get(
            "strict_causal_chain_audit", {}
        )
        split_checks[split] = bool(
            split_strict.get("passed") is True
            and split_strict.get("state_installations_after_x0") == 0
            and split_strict.get("query_simulator_recreated") is False
            and split_strict.get("full_state_dimensions") == 12
            and split_strict.get("max_pair_full_state_gap", float("inf"))
            <= split_strict.get("full_state_tolerance", -1.0)
            and split_strict.get("max_pair_query_pixel_difference") == 0
            and split_strict.get("max_pair_query_action_difference") == 0.0
        )
    checks = {
        "manifest_passed": manifest.get("passed") is True,
        "top_level_strict_audit_passed": strict.get("passed") is True,
        "no_state_installation_after_x0": (
            strict.get("state_installations_after_x0") == 0
        ),
        "single_simulator_per_trajectory": (
            strict.get("query_simulator_recreated") is False
        ),
        "full_x2_state_is_compared": (
            strict.get("full_state_dimensions") == 12
        ),
        "x2_full_state_within_tolerance": (
            strict.get("max_pair_full_state_gap", float("inf"))
            <= strict.get("full_state_tolerance", -1.0)
        ),
        "x2_pixels_identical": (
            strict.get("max_pair_query_pixel_difference") == 0
        ),
        "query_actions_identical": (
            strict.get("max_pair_query_action_difference") == 0.0
        ),
        "train_split_strict_audit_passed": split_checks["train"],
        "development_split_strict_audit_passed": split_checks[
            "validation"
        ],
    }
    return {
        "checks": checks,
        "top_level_audit": strict,
        "passed": all(checks.values()),
    }


def source_identity_receipt(
    strict_manifest: dict[str, Any],
    legacy_manifest: dict[str, Any],
) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    checks: list[bool] = []
    for split in ("train", "validation"):
        strict_split = strict_manifest["splits"][split]
        legacy_split = legacy_manifest["splits"][split]
        strict_templates = [row["template"] for row in strict_split["pairs"]]
        legacy_templates = [row["template"] for row in legacy_split["pairs"]]
        template_identity = strict_templates == legacy_templates
        candidate_identity = (
            strict_split.get("source_candidate_rows_sha256")
            == legacy_split.get("source_candidate_rows_sha256")
        )
        strict_partition = strict_manifest["source"]["episode_partition"][
            split
        ]
        legacy_partition = legacy_manifest["source"]["episode_partition"][
            split
        ]
        partition_identity = strict_partition == legacy_partition
        row_indices = [
            int(row["source_row_index"]) for row in strict_templates
        ]
        legacy_row_indices = [
            int(row["source_row_index"]) for row in legacy_templates
        ]
        rows_identity = row_indices == legacy_row_indices
        split_passed = all(
            (
                template_identity,
                candidate_identity,
                partition_identity,
                rows_identity,
            )
        )
        checks.append(split_passed)
        splits[split] = {
            "pair_count": len(strict_templates),
            "strict_template_population_sha256": canonical_sha256(
                strict_templates
            ),
            "legacy_template_population_sha256": canonical_sha256(
                legacy_templates
            ),
            "templates_byte_for_byte_identical": template_identity,
            "selected_source_rows_sha256": canonical_sha256(row_indices),
            "selected_source_rows_identical": rows_identity,
            "candidate_pool_receipt_identical": candidate_identity,
            "episode_partition_receipt_identical": partition_identity,
            "passed": split_passed,
        }
    return {"splits": splits, "passed": all(checks)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-training", type=Path, default=DEFAULT_STRICT_TRAINING
    )
    parser.add_argument(
        "--legacy-training", type=Path, default=DEFAULT_LEGACY_TRAINING
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strict_root = args.strict_training.expanduser().resolve()
    legacy_root = args.legacy_training.expanduser().resolve()
    protocol = args.protocol.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else strict_root / "distribution_audit.json"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    strict_path = strict_root / "manifest.json"
    legacy_path = legacy_root / "manifest.json"
    legacy_audit_path = legacy_root / "distribution_audit.json"
    for path in (strict_path, legacy_path, legacy_audit_path, protocol):
        if not path.is_file():
            raise FileNotFoundError(path)

    strict_manifest = json.loads(strict_path.read_text(encoding="utf-8"))
    legacy_manifest = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy_audit = json.loads(
        legacy_audit_path.read_text(encoding="utf-8")
    )
    causal = strict_causal_receipt(strict_manifest)
    identity = source_identity_receipt(strict_manifest, legacy_manifest)
    checks = {
        "prior_distribution_audit_passed": legacy_audit.get("passed") is True,
        "strict_causal_chain_passed": causal["passed"],
        "source_population_identity_passed": identity["passed"],
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "benchmark": "pusht_action_strength_history3_strict_causal_v1",
        "status": (
            "passed_strict_causal_and_distribution_identity_gate"
            if passed
            else "failed_strict_causal_or_distribution_identity_gate"
        ),
        "protocol": {
            "path": str(protocol),
            "sha256": file_sha256(protocol),
        },
        "inputs": {
            "strict_manifest": str(strict_path),
            "strict_manifest_sha256": file_sha256(strict_path),
            "legacy_manifest": str(legacy_path),
            "legacy_manifest_sha256": file_sha256(legacy_path),
            "prior_distribution_audit": str(legacy_audit_path),
            "prior_distribution_audit_sha256": file_sha256(
                legacy_audit_path
            ),
        },
        "reuse_basis": (
            "The strict release selects exactly the same replay templates, "
            "source rows, candidate pools, and episode partitions as the "
            "audited replay-matched release. Distribution metrics therefore "
            "describe the same population; only the trajectory construction "
            "and its stricter causal audit changed."
        ),
        "source_population_identity": identity,
        "strict_causal_chain": causal,
        "inherited_distribution_evidence": {
            "reference_population": legacy_audit["reference_population"],
            "feature_population_sizes": legacy_audit[
                "feature_population_sizes"
            ],
            "normalized_wasserstein_by_feature": legacy_audit[
                "normalized_wasserstein_by_feature"
            ],
            "summaries": legacy_audit["summaries"],
            "median_reduction_vs_v1": legacy_audit[
                "median_reduction_vs_v1"
            ],
            "distribution_checks": legacy_audit["distribution_checks"],
        },
        "checks": checks,
        "passed": passed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": passed,
                "checks": checks,
                "strict_manifest_sha256": file_sha256(strict_path),
                "prior_distribution_audit_sha256": file_sha256(
                    legacy_audit_path
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
