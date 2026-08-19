#!/usr/bin/env python3
"""Freeze the ActionStrength PLDM CEM stop after its failed three-seed ICL gate.

This is deliberately a terminal, additive receipt: it reads the already
frozen binding, raw Public ICL receipts, float32 recovery receipts, and their
three-seed aggregate.  It neither loads a model nor invokes either CEM runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPLETION_ID = "pusht_action_strength_pldm_reference_completion_v1"
RECOVERY_ID = "pusht_action_strength_pldm_float32_rescore_recovery_v1"
FORMAL_ROOT = Path(
    "artifacts/evaluation/history3/"
    "pusht_action_strength_pldm_reference_completion_v1/formal_icl_v1"
)
STOP_OUTPUT = FORMAL_ROOT / "cem_not_authorized_stop_v1.json"

RECOVERY_PREREG = Path(
    "configs/benchmark/pusht_action_strength_pldm_float32_rescore_recovery_v1.yaml"
)
RECOVERY_PREREG_SHA256 = "6526ec2223ac9366dddc2a3e215b003a21849e992da231b0d093c6d1901f0ebf"
BINDING_CONFIG = Path(
    "configs/benchmark/pusht_action_strength_pldm_evaluation_binding_v1.yaml"
)
BINDING_CONFIG_SHA256 = "da511217026febc81cac66850f4db10ec63a9bed6772e65bc132b96493e9ef12"
BINDING_RECEIPT = Path(
    "artifacts/evaluation/history3/"
    "pusht_action_strength_pldm_reference_completion_v1/"
    "evaluation_binding_v1/evaluation_binding_revalidation_v1.json"
)
BINDING_RECEIPT_SHA256 = "65d6f1c0565fb530bb69ccd3b631491a92189d6426196625bcccb358b59636ff"
RELEASE_CONFIG = Path("configs/benchmark/pusht_action_strength_icl_release_v1.yaml")
RELEASE_CONFIG_SHA256 = "e8e9fe068ef323b9dc92fab3a55a3154a305f6c920cd55718141e69854c778bc"
AGGREGATE = FORMAL_ROOT / "float32_rescore_recovery_v1/three_seed_negative_aggregate.json"
AGGREGATE_SHA256 = "b2bbf8c94e00246a10ec987c1c8f15476a65814b46c854e3371ca78507c33a17"

SEEDS = {
    13313: {
        "checkpoint_sha256": "beed2dfe852ab01f21386ea13cbe8f77ddf16264f71d2ca25c66003266e85a2f",
        "raw": FORMAL_ROOT / "seed_13313.json",
        "raw_sha256": "0949c239be93037d4c9c285fe041d57aeda0b9ded5842f5ba488f23f3dbbdfd7",
        "recovery": FORMAL_ROOT / "float32_rescore_recovery_v1/seed_13313.json",
        "recovery_sha256": "5ca4bb1d4687624fe40791f87331d588bae695431f78dfb6ae34026cfd01f0dc",
    },
    13314: {
        "checkpoint_sha256": "1f014cb345116c02cfbdbe6952a86a377138da1e8e8dd1dc87156c9ca9f47864",
        "raw": FORMAL_ROOT / "seed_13314.json",
        "raw_sha256": "ad8579ff0d20d6a00f3229882fe527be205ec1be62d04b0c2d308b440eca1bc9",
        "recovery": FORMAL_ROOT / "float32_rescore_recovery_v1/seed_13314.json",
        "recovery_sha256": "51067c41abebb6f5c3e9fcca389165187d2599a4adc51955bb12a9ce86d7d6c4",
    },
    13315: {
        "checkpoint_sha256": "3242de2bfa52dcf41c6a0d40b9126e8549cf87763d66cb75db86ca3a55240148",
        "raw": FORMAL_ROOT / "seed_13315.json",
        "raw_sha256": "a6719430527b8ee1883a05e2bb09d02f096ef6e98ddf25d37b49a4702b431665",
        "recovery": FORMAL_ROOT / "float32_rescore_recovery_v1/seed_13315.json",
        "recovery_sha256": "d1805511c84dbf3456371acf72d46b0741bcb778cdcb379f8dc1cb924eb23398",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(logical: Path | str, *, label: str) -> Path:
    path = (ROOT / logical).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    return path


def _logical(path: Path, *, label: str) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error


def _identity(logical: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    path = _resolve(logical, label=label)
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"{label} changed: expected={expected_sha256}, observed={observed}"
        )
    return {
        "path": _logical(path, label=label),
        "sha256": observed,
        "size_bytes": int(path.stat().st_size),
    }


def _self_identity() -> dict[str, Any]:
    path = Path(__file__).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _logical(path, label="stop freezer"),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _json(identity: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(identity["path"], label="frozen JSON")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "recovery_preregistration": _identity(
            RECOVERY_PREREG,
            RECOVERY_PREREG_SHA256,
            label="recovery preregistration",
        ),
        "evaluation_binding_config": _identity(
            BINDING_CONFIG,
            BINDING_CONFIG_SHA256,
            label="evaluation binding config",
        ),
        "evaluation_binding_receipt": _identity(
            BINDING_RECEIPT,
            BINDING_RECEIPT_SHA256,
            label="evaluation binding receipt",
        ),
        "release_config": _identity(
            RELEASE_CONFIG,
            RELEASE_CONFIG_SHA256,
            label="release config",
        ),
        "public_icl_aggregate": _identity(
            AGGREGATE,
            AGGREGATE_SHA256,
            label="three-seed recovery aggregate",
        ),
        "stop_freezer": _self_identity(),
        "raw_public_results": {},
        "recovery_receipts": {},
    }
    for seed, specification in SEEDS.items():
        snapshot["raw_public_results"][str(seed)] = _identity(
            specification["raw"],
            specification["raw_sha256"],
            label=f"raw Public ICL seed {seed}",
        )
        snapshot["recovery_receipts"][str(seed)] = _identity(
            specification["recovery"],
            specification["recovery_sha256"],
            label=f"float32 recovery seed {seed}",
        )
    return snapshot


def _validate_binding(snapshot: dict[str, Any]) -> None:
    binding = _json(snapshot["evaluation_binding_receipt"])
    if not (
        binding.get("status") == "passed_evaluation_binding_freeze"
        and binding.get("passed") is True
        and binding.get("binding", {}).get("sha256") == BINDING_CONFIG_SHA256
    ):
        raise RuntimeError("Passed evaluation binding is not intact")


def _validate_seed(seed: int, snapshot: dict[str, Any]) -> None:
    specification = SEEDS[seed]
    raw = _json(snapshot["raw_public_results"][str(seed)])
    recovery = _json(snapshot["recovery_receipts"][str(seed)])
    raw_adapter = raw.get("model", {}).get("adapter", {})
    verification = recovery.get("verification", {})
    bindings = recovery.get("bindings", {})
    if not (
        raw.get("status") == "completed"
        and raw.get("model", {}).get("training_seed") == seed
        and raw_adapter.get("checkpoint_sha256") == specification["checkpoint_sha256"]
        and raw.get("gate", {}).get("passed") is False
        and recovery.get("schema_version") == 1
        and recovery.get("recovery_id") == RECOVERY_ID
        and recovery.get("completion_id") == COMPLETION_ID
        and recovery.get("status") == "completed"
        and recovery.get("reconstruction", {}).get("gate", {}).get("passed") is False
        and bindings.get("checkpoint_sha256") == specification["checkpoint_sha256"]
        and bindings.get("raw_public_result", {}).get("path")
        == snapshot["raw_public_results"][str(seed)]["path"]
        and bindings.get("raw_public_result", {}).get("observed_sha256")
        == specification["raw_sha256"]
        and verification.get("passed") is True
        and verification.get("float32_scalar_aggregates_bitwise_equal") is True
        and verification.get("latent_summary_close") is True
        and verification.get("gate_exact_equal") is True
        and recovery.get("input_integrity", {}).get(
            "all_frozen_inputs_unchanged_during_recovery"
        )
        is True
    ):
        raise RuntimeError(f"Recovery evidence is not an intact failed gate: seed {seed}")


def _validate_aggregate(snapshot: dict[str, Any]) -> None:
    aggregate = _json(snapshot["public_icl_aggregate"])
    checkpoints = aggregate.get("checkpoints", [])
    expected_seed_rows = {int(row.get("training_seed", -1)): row for row in checkpoints}
    if not (
        aggregate.get("schema_version") == 1
        and aggregate.get("recovery_id") == RECOVERY_ID
        and aggregate.get("completion_id") == COMPLETION_ID
        and aggregate.get("status") == "completed"
        and aggregate.get("decision", {}).get("passed") is False
        and aggregate.get("decision", {}).get("formal_evaluation_completed") is True
        and aggregate.get("cem", {}).get("authorized") is False
        and aggregate.get("cem", {}).get("executed") is False
        and tuple(sorted(expected_seed_rows)) == tuple(sorted(SEEDS))
        and len(checkpoints) == 3
    ):
        raise RuntimeError("Three-seed recovery aggregate does not prove CEM is stopped")
    for seed, specification in SEEDS.items():
        row = expected_seed_rows[seed]
        expected_recovery = snapshot["recovery_receipts"][str(seed)]
        if not (
            row.get("passed") is False
            and row.get("checkpoint_sha256") == specification["checkpoint_sha256"]
            and row.get("recovery_receipt") == expected_recovery
        ):
            raise RuntimeError(f"Aggregate checkpoint row is not intact: seed {seed}")


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def freeze(output: Path) -> dict[str, Any]:
    expected_output = _resolve(STOP_OUTPUT, label="CEM stop output")
    actual_output = output.resolve()
    if actual_output != expected_output:
        raise ValueError(
            "CEM stop output must equal "
            f"{_logical(expected_output, label='CEM stop output')}"
        )
    before = _snapshot()
    _validate_binding(before)
    for seed in SEEDS:
        _validate_seed(seed, before)
    _validate_aggregate(before)
    after = _snapshot()
    if before != after:
        raise RuntimeError("A frozen CEM-stop input changed while it was read")
    raw_gates = [False for _ in SEEDS]
    return {
        "schema_version": 1,
        "completion_id": COMPLETION_ID,
        "status": "frozen_cem_not_authorized_after_failed_three_seed_public_icl",
        "output": {
            "path": _logical(expected_output, label="CEM stop output"),
            "content_sha256_not_embedded_to_avoid_self_reference": True,
        },
        "scope": {
            "model_evaluation_rerun_performed": False,
            "public_test_reopened": False,
            "action_planning_cem_executed": False,
            "original_pusht_retention_cem_executed": False,
            "checkpoint_selection_performed": False,
        },
        "evaluation_binding": {
            "config": before["evaluation_binding_config"],
            "receipt": before["evaluation_binding_receipt"],
        },
        "recovery_preregistration": before["recovery_preregistration"],
        "release_config": before["release_config"],
        "public_icl_aggregate": before["public_icl_aggregate"],
        "public_icl": {
            "passed": False,
            "passed_checkpoints": sum(raw_gates),
            "evaluated_checkpoints": len(raw_gates),
            "raw_public_gate_passed": {str(seed): False for seed in SEEDS},
            "float32_recovered_gate_passed": {str(seed): False for seed in SEEDS},
            "reason": "all_three_raw_and_float32_recovered_gates_are_false",
        },
        "cem": {
            "authorized": False,
            "executed": False,
            "reason": "three_seed_public_icl_gate_failed",
            "action_planning_cem": {"authorized": False, "executed": False},
            "original_pusht_retention_cem": {"authorized": False, "executed": False},
        },
        "input_integrity": {
            "all_frozen_inputs_unchanged_during_stop_freeze": True,
            "identities_before_stop_freeze": before,
            "identities_after_stop_freeze": after,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STOP_OUTPUT)
    args = parser.parse_args()
    output = _resolve(args.output, label="CEM stop output")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite CEM stop receipt: {output}")
    receipt = freeze(output)
    _write_exclusive(output, receipt)
    print(json.dumps({"status": receipt["status"], "output": receipt["output"]["path"]}))


if __name__ == "__main__":
    main()
