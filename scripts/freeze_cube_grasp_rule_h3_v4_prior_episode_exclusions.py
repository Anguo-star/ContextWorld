#!/usr/bin/env python3
"""Freeze source episodes that Cube v4 must never reuse.

The receipt binds every source episode used by the formal Cube v3 data and
the separate coupling-design pilot.  The two v3 smoke builds are verified as
subsets of the formal v3 population.  The old v3 RGB diagnostic is bound as a
design input but contributes no new episodes because it only reopened the
formal v3 Training/Development tables.

This command reads JSON receipts only.  It never opens Lance or Public Test.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


V3_PROTOCOL = "cube_gripper_carry_rule_history3_development_v3"
V4_PROTOCOL = "cube_gripper_carry_rule_history3_development_v4"
ACTIVE_SPLITS = ("train", "loader_validation")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_ids_sha256(values: Sequence[int]) -> str:
    canonical = sorted({int(value) for value in values})
    payload = ("\n".join(str(value) for value in canonical) + "\n").encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _formal_or_smoke_episodes(
    path: Path, *, expected_pair_count: int | None
) -> tuple[set[int], dict[str, Any]]:
    document = _load_object(path)
    if document.get("protocol") != V3_PROTOCOL:
        raise RuntimeError(f"Unexpected v3 protocol in {path}")
    if document.get("passed") is not True:
        raise RuntimeError(f"v3 build report is not passed: {path}")
    if document.get("active_splits") != list(ACTIVE_SPLITS):
        raise RuntimeError(f"Unexpected v3 active splits in {path}")
    splits = document.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(ACTIVE_SPLITS):
        raise RuntimeError(f"Malformed v3 split reports in {path}")
    episodes: set[int] = set()
    pair_count = 0
    split_counts: dict[str, int] = {}
    for split in ACTIVE_SPLITS:
        report = splits[split]
        if not isinstance(report, Mapping) or report.get("passed") is not True:
            raise RuntimeError(f"v3 split did not pass: {path}:{split}")
        split_pair_count = int(report.get("pair_count", -1))
        values = [int(value) for value in report.get("source_episodes", [])]
        if split_pair_count <= 0 or len(values) != split_pair_count:
            raise RuntimeError(f"v3 source episode count mismatch: {path}:{split}")
        if len(set(values)) != len(values):
            raise RuntimeError(f"duplicate v3 source episode: {path}:{split}")
        if episodes.intersection(values):
            raise RuntimeError(f"v3 source episode crosses splits: {path}")
        episodes.update(values)
        pair_count += split_pair_count
        split_counts[split] = split_pair_count
    if expected_pair_count is not None and pair_count != expected_pair_count:
        raise RuntimeError(
            f"Unexpected v3 pair count in {path}: {pair_count} != "
            f"{expected_pair_count}"
        )
    return episodes, {
        "path_recorded": False,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "pair_count": pair_count,
        "split_pair_counts": split_counts,
        "source_episode_count": len(episodes),
        "source_episode_ids_sha256": episode_ids_sha256(sorted(episodes)),
        "public_test_opened_or_generated": False,
    }


def _pilot_episodes(path: Path) -> tuple[set[int], dict[str, Any]]:
    document = _load_object(path)
    if document.get("schema_version") != 1 or document.get("role") != (
        "nonformal_v4_design_feasibility_not_a_frozen_gate"
    ):
        raise RuntimeError("Unexpected coupling-pilot identity")
    scope = document.get("scope")
    if not isinstance(scope, Mapping):
        raise RuntimeError("Coupling pilot lacks scope")
    if scope.get("public_test_opened_read_hashed_or_scored") is not False:
        raise RuntimeError("Coupling pilot did not keep Public Test closed")
    if scope.get("reference_model_training_or_scoring") is not False:
        raise RuntimeError("Coupling pilot unexpectedly used a reference model")
    variants = document.get("variants")
    if not isinstance(variants, Mapping) or not variants:
        raise RuntimeError("Coupling pilot lacks variants")
    episode_sets: list[set[int]] = []
    for name, variant in variants.items():
        if not isinstance(variant, Mapping):
            raise RuntimeError(f"Malformed pilot variant: {name}")
        rows = variant.get("rows_without_feature_vectors")
        if not isinstance(rows, list):
            raise RuntimeError(f"Pilot variant lacks rows: {name}")
        episodes = {int(row["source_episode"]) for row in rows}
        if len(episodes) != len(rows):
            raise RuntimeError(f"Pilot variant repeats source episodes: {name}")
        episode_sets.append(episodes)
    first = episode_sets[0]
    if any(values != first for values in episode_sets[1:]):
        raise RuntimeError("Coupling variants did not use the same pilot scenes")
    if len(first) != int(scope.get("sample_count", -1)):
        raise RuntimeError("Coupling pilot sample count mismatch")
    return first, {
        "path_recorded": False,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "role": document["role"],
        "selection_seed": int(scope["new_scene_and_selection_seed"]),
        "source_episode_count": len(first),
        "source_episode_ids_sha256": episode_ids_sha256(sorted(first)),
        "couplings_n": list(document["design"]["couplings_n"]),
        "reference_model_training_or_scoring": False,
        "public_test_opened_read_hashed_or_scored": False,
    }


def _diagnostic_receipt(path: Path) -> dict[str, Any]:
    document = _load_object(path)
    if document.get("status") != "completed_exploratory_diagnostic":
        raise RuntimeError("Unexpected v3 exploratory diagnostic status")
    scope = document.get("scope")
    if not isinstance(scope, Mapping):
        raise RuntimeError("v3 exploratory diagnostic lacks scope")
    public = scope.get("public_test")
    if not isinstance(public, Mapping) or any(
        public.get(name) is not False for name in ("opened", "read", "hashed", "scored")
    ):
        raise RuntimeError("v3 exploratory diagnostic did not keep Public closed")
    if scope.get("old_development_reused_for_exploratory_design") is not True:
        raise RuntimeError("v3 diagnostic reuse disclosure is missing")
    if scope.get("reference_model_training_or_scoring") is not False:
        raise RuntimeError("v3 diagnostic unexpectedly trained or scored models")
    return {
        "path_recorded": False,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "diagnostic_id": document.get("diagnostic_id"),
        "old_v3_development_used_for_exploratory_design": True,
        "contributes_new_source_episodes": False,
        "reference_model_training_or_scoring": False,
        "public_test_opened_read_hashed_or_scored": False,
    }


def freeze(
    *,
    formal_v3_report: Path,
    v3_smoke_reports: Sequence[Path],
    coupling_pilot: Path,
    exploratory_diagnostic: Path,
    v3_freeze_receipt: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite exclusion receipt {output}")
    formal_episodes, formal_receipt = _formal_or_smoke_episodes(
        formal_v3_report, expected_pair_count=2304
    )
    smoke_receipts: list[dict[str, Any]] = []
    for path in v3_smoke_reports:
        episodes, receipt = _formal_or_smoke_episodes(
            path, expected_pair_count=40
        )
        receipt["all_source_episodes_are_in_formal_v3"] = episodes <= formal_episodes
        if not receipt["all_source_episodes_are_in_formal_v3"]:
            raise RuntimeError("A v3 smoke used episodes outside formal v3")
        smoke_receipts.append(receipt)

    pilot_episodes, pilot_receipt = _pilot_episodes(coupling_pilot)
    if formal_episodes.intersection(pilot_episodes):
        raise RuntimeError("Coupling pilot overlaps formal v3 source episodes")
    diagnostic_receipt = _diagnostic_receipt(exploratory_diagnostic)

    v3_freeze = _load_object(v3_freeze_receipt)
    if v3_freeze.get("protocol_id") != V3_PROTOCOL or v3_freeze.get(
        "checks_passed"
    ) is not True:
        raise RuntimeError("Invalid v3 freeze receipt")
    source_h5 = v3_freeze.get("source_h5")
    if not isinstance(source_h5, Mapping):
        raise RuntimeError("v3 freeze receipt lacks source identity")

    excluded = sorted(formal_episodes | pilot_episodes)
    receipt = {
        "schema_version": 1,
        "protocol_id": V4_PROTOCOL,
        "receipt_id": "cube_gripper_carry_h3_v4_prior_episode_exclusions_v1",
        "status": "frozen_before_first_v4_data_build",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "source_episode_exclusions_for_v4_train_and_loader_validation",
        "source_h5": {
            "symbol": source_h5.get("symbol"),
            "sha256": source_h5.get("sha256"),
            "size_bytes": source_h5.get("size_bytes"),
            "row_count": source_h5.get("row_count"),
            "episode_count": source_h5.get("episode_count"),
            "path_recorded": False,
        },
        "inputs": {
            "v3_freeze_receipt": {
                "path_recorded": False,
                "sha256": file_sha256(v3_freeze_receipt),
                "size_bytes": v3_freeze_receipt.stat().st_size,
            },
            "formal_v3_build_report": formal_receipt,
            "v3_smoke_build_reports": smoke_receipts,
            "v4_coupling_design_pilot": pilot_receipt,
            "v3_failure_exploratory_diagnostic": diagnostic_receipt,
        },
        "excluded_source_episodes": excluded,
        "excluded_source_episode_count": len(excluded),
        "excluded_source_episode_ids_sha256": episode_ids_sha256(excluded),
        "components": {
            "formal_v3_source_episode_count": len(formal_episodes),
            "formal_v3_source_episode_ids_sha256": episode_ids_sha256(
                sorted(formal_episodes)
            ),
            "smoke_source_episodes_are_subsets_of_formal_v3": True,
            "coupling_pilot_source_episode_count": len(pilot_episodes),
            "coupling_pilot_source_episode_ids_sha256": episode_ids_sha256(
                sorted(pilot_episodes)
            ),
            "formal_v3_and_coupling_pilot_overlap_count": 0,
        },
        "authorized_active_splits": list(ACTIVE_SPLITS),
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "reference_model_training_or_scoring": False,
        "checks_passed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-v3-build-report", type=Path, required=True)
    parser.add_argument(
        "--v3-smoke-build-report", type=Path, action="append", default=[]
    )
    parser.add_argument("--coupling-pilot", type=Path, required=True)
    parser.add_argument("--exploratory-diagnostic", type=Path, required=True)
    parser.add_argument("--v3-freeze-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = freeze(
        formal_v3_report=args.formal_v3_build_report.expanduser().resolve(),
        v3_smoke_reports=[value.expanduser().resolve() for value in args.v3_smoke_build_report],
        coupling_pilot=args.coupling_pilot.expanduser().resolve(),
        exploratory_diagnostic=args.exploratory_diagnostic.expanduser().resolve(),
        v3_freeze_receipt=args.v3_freeze_receipt.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "checks_passed": receipt["checks_passed"],
                "excluded_source_episode_count": receipt[
                    "excluded_source_episode_count"
                ],
                "excluded_source_episode_ids_sha256": receipt[
                    "excluded_source_episode_ids_sha256"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
