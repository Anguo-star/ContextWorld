#!/usr/bin/env python3
"""Freeze the distinct one-use Cube v4r1 Public recovery authorization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.cube_grasp_rule_public_contract import (  # noqa: E402
    FREEZE_STATUS,
    PROTOCOL_ID,
    file_identity,
    read_yaml_nofollow,
)
from contextworld.benchmarks.cube_grasp_rule_public_recovery_contract import (  # noqa: E402
    DEFAULT_FREEZE_RECEIPT,
    DEFAULT_PREREGISTRATION,
    EXPECTED_RECOVERY_IMPLEMENTATION_KEYS,
    FREEZE_RECEIPT_ID,
    PREREGISTRATION_ID,
    RECOVERY_AUTHORIZATION_ID,
    validate_recovery_freeze_receipt_contract,
    validate_recovery_lineage,
    validate_recovery_preregistration_contract,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402
import scripts.freeze_cube_grasp_rule_h3_v4r1_public_release as base_freezer  # noqa: E402


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(name) == right.get(name)
        for name in ("path", "sha256", "size_bytes")
    )


def _validate_recovery_implementation_identities(
    prereg: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    entries = _mapping(
        prereg.get("identity", {}).get("recovery_implementation"),
        label="identity.recovery_implementation",
    )
    if set(entries) != set(EXPECTED_RECOVERY_IMPLEMENTATION_KEYS):
        raise RuntimeError("Cube Public recovery implementation set drifted")
    result: dict[str, dict[str, Any]] = {}
    for name, raw in entries.items():
        entry = _mapping(raw, label=f"recovery implementation {name}")
        path = resolve_contextworld_path(str(entry.get("path", "")))
        observed = file_identity(path, logical_path=str(entry.get("path", "")))
        if not _identity_equal(observed, entry):
            raise RuntimeError(f"recovery implementation identity drifted: {name}")
        result[str(name)] = observed
    return result


def freeze_public_recovery(
    *, preregistration: Path, output: Path
) -> dict[str, Any]:
    preregistration = preregistration.expanduser()
    if not preregistration.is_absolute():
        preregistration = ROOT / preregistration
    preregistration = preregistration.absolute()
    output = output.expanduser().resolve()
    prereg_raw, prereg = read_yaml_nofollow(
        preregistration, label="Cube Public recovery preregistration"
    )
    validate_recovery_preregistration_contract(prereg)

    planned = _mapping(prereg.get("planned_artifacts"), label="planned_artifacts")
    expected_output = resolve_contextworld_path(str(planned["freeze_receipt"]))
    if output != expected_output:
        raise ValueError("Cube Public recovery freeze output differs from preregistration")
    base_freezer._assert_absent(output, label="recovery freeze receipt")
    public_root = resolve_contextworld_path(str(planned["public_data_root"]))
    score_root = resolve_contextworld_path(str(planned["public_score_root"]))
    decision_path = resolve_contextworld_path(str(planned["public_release_decision"]))
    base_freezer._assert_absent(public_root, label="recovery Public data root")
    base_freezer._assert_absent(score_root, label="recovery Public score root")
    base_freezer._assert_absent(decision_path, label="recovery Public decision")

    implementations = base_freezer._validate_implementation_identities(prereg)
    recovery_implementations = _validate_recovery_implementation_identities(prereg)
    basis_identities, payloads = base_freezer._validate_basis(prereg)
    checkpoint_inputs = base_freezer._validate_checkpoint_chain(
        prereg, payloads=payloads
    )
    exclusions = base_freezer._union_receipt(
        prereg,
        prior=payloads["prior_exclusion_receipt"],
        build=payloads["development_build_report"],
    )
    runtime, runtime_inputs = base_freezer._validate_runtime(prereg)
    source_h5 = base_freezer._source_h5_receipt(prereg)
    lineage = validate_recovery_lineage(prereg, root=ROOT)

    prereg_identity = {
        "path": str(prereg["identity"]["preregistration_path"]),
        "sha256": hashlib.sha256(prereg_raw).hexdigest(),
        "size_bytes": len(prereg_raw),
    }
    frozen_inputs: dict[str, dict[str, Any]] = {
        name: {**identity, "rehash_on_entrypoint": True}
        for name, identity in basis_identities.items()
    }
    frozen_inputs.update(checkpoint_inputs)
    frozen_inputs.update(runtime_inputs)
    frozen_inputs["source_h5"] = source_h5

    receipt = {
        "schema_version": 1,
        "receipt_id": FREEZE_RECEIPT_ID,
        "receipt_path": str(planned["freeze_receipt"]),
        "preregistration_id": PREREGISTRATION_ID,
        "recovery_authorization_id": RECOVERY_AUTHORIZATION_ID,
        "protocol_id": PROTOCOL_ID,
        "status": FREEZE_STATUS,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": True,
        "preregistration": prereg_identity,
        "implementation_identities": implementations,
        "recovery_implementation_identities": recovery_implementations,
        "frozen_inputs": frozen_inputs,
        "runtime": {"stable_worldmodel": runtime},
        "public_exclusions": exclusions,
        "recovery_lineage": lineage,
        "authorization": {
            "public_generation_once": True,
            "public_scoring_once_after_successful_generation": True,
            "authorized_model_families": ["lewm"],
            "training_seeds": list(base_freezer.EXPECTED_SEEDS),
            "training_or_checkpoint_selection": False,
            "threshold_or_recipe_changes": False,
            "public_test_rerun_after_access": False,
            "suite_registration": False,
        },
        "public_test": {
            "access_status": "authorized_not_generated_not_opened_not_read_not_scored",
            "generated": False,
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "planned_artifacts": dict(planned),
    }
    validate_recovery_freeze_receipt_contract(
        prereg=prereg, freeze=receipt, root=ROOT
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_FREEZE_RECEIPT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = freeze_public_recovery(
        preregistration=args.prereg,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(args.output),
                "recovery_authorization_id": receipt[
                    "recovery_authorization_id"
                ],
                "public_test": receipt["public_test"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
