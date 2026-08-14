#!/usr/bin/env python3
"""Run the single authorized Cube v4r1 Public scoring campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.cube_grasp_rule_public_contract import (  # noqa: E402
    DEFAULT_FREEZE_RECEIPT,
    DEFAULT_PREREGISTRATION,
    file_identity,
    load_public_authorization,
)
from contextworld.benchmarks.cube_grasp_rule_public_score import (  # noqa: E402
    aggregate_public_results,
    build_adapter,
    evaluate_public_checkpoint,
    load_public_arrays,
    release_adapter,
    validate_public_publication,
)
from contextworld.paths import portable_contextworld_path  # noqa: E402


def _write_json_x(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_preflight(checkpoints: Any) -> list[dict[str, Any]]:
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        raise RuntimeError("Cube Public requires exactly three checkpoints")
    result: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for entry in checkpoints:
        if not isinstance(entry, Mapping):
            raise ValueError("Cube Public checkpoint entry must be a mapping")
        seed = int(entry["training_seed"])
        if seed in seen_seeds or entry.get("model_family") != "lewm":
            raise RuntimeError("Cube Public checkpoint family/seed mismatch")
        observed = file_identity(
            Path(str(entry["path"])), logical_path=str(entry["path"])
        )
        if any(
            observed.get(name) != entry.get(name)
            for name in ("path", "sha256", "size_bytes")
        ):
            raise RuntimeError(f"Cube Public checkpoint seed {seed} drifted")
        seen_seeds.add(seed)
        result.append(dict(entry))
    return result


def _adapter_runtime_preflight(
    *,
    authorization: Any,
    checkpoints: Sequence[Mapping[str, Any]],
    devices: Sequence[str],
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for checkpoint, device in zip(checkpoints, devices):
        adapter = build_adapter(
            authorization=authorization,
            checkpoint=checkpoint,
            device=device,
        )
        try:
            receipts.append(
                {
                    "training_seed": int(checkpoint["training_seed"]),
                    "device": str(device),
                    "model_state_sha256": adapter.frozen_state_hash(),
                    "adapter_id": adapter.metadata["adapter_id"],
                    "runtime_preflight_passed": True,
                }
            )
        finally:
            release_adapter(adapter)
            del adapter
    return receipts


def run_public_matrix(
    *,
    preregistration: Path,
    freeze_receipt: Path,
    output: Path,
    devices: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    authorization = load_public_authorization(
        preregistration_path=preregistration,
        freeze_receipt_path=freeze_receipt,
    )
    output = output.expanduser().resolve()
    if output != authorization.score_root:
        raise ValueError("Cube Public score root differs from the frozen path")
    try:
        os.lstat(output)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(
            "Cube Public score root already exists; the one-use campaign cannot rerun"
        )
    expected_devices = tuple(
        str(value)
        for value in authorization.preregistration["public_evaluation"]["devices"]
    )
    devices = tuple(str(value) for value in devices)
    if devices != expected_devices or len(devices) != 3:
        raise ValueError("Cube Public device assignment differs from preregistration")
    expected_batch = int(
        authorization.preregistration["public_evaluation"]["batch_size"]
    )
    if int(batch_size) != expected_batch:
        raise ValueError("Cube Public batch size differs from preregistration")

    checkpoints = _checkpoint_preflight(
        authorization.preregistration["public_evaluation"]["checkpoints"]
    )
    runtime_preflight = _adapter_runtime_preflight(
        authorization=authorization,
        checkpoints=checkpoints,
        devices=devices,
    )
    request_path = output / "matrix_request.json"
    access_path = output / "public_access_started.json"
    namespace_reserved = False
    access_started = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        namespace_reserved = True
        publication = validate_public_publication(
            authorization,
            verify_published_tree=False,
        )
        request = {
            "schema_version": 1,
            "preregistration": file_identity(
                authorization.preregistration_path,
                logical_path=authorization.preregistration["identity"][
                    "preregistration_path"
                ],
            ),
            "freeze_receipt": file_identity(
                authorization.freeze_receipt_path,
                logical_path=authorization.freeze_receipt_identity["path"],
            ),
            "public_success_marker": file_identity(
                authorization.public_root / "_SUCCESS.json",
                logical_path=portable_contextworld_path(
                    authorization.public_root / "_SUCCESS.json"
                ),
            ),
            "checkpoints": checkpoints,
            "devices": list(devices),
            "batch_size": expected_batch,
            "adapter_runtime_preflight": runtime_preflight,
            "model_visible_fields": ["history_pixels", "query_action_blocks"],
            "privileged_columns_passed_to_model": False,
            "training_or_checkpoint_selection": False,
            "threshold_or_recipe_changes": False,
            "rerun_authorized": False,
            "created_at_utc": _utc_now(),
        }
        _write_json_x(request_path, request)
        _write_json_x(
            access_path,
            {
                "schema_version": 1,
                "status": "public_access_started_irreversible_one_use_campaign",
                "started_at_utc": _utc_now(),
                "request": file_identity(
                    request_path,
                    logical_path=portable_contextworld_path(request_path),
                ),
                "public_data_metadata_preflight_passed": True,
                "checkpoint_identity_preflight_passed": True,
                "adapter_runtime_preflight_passed": True,
            },
        )
        access_started = True
        publication = validate_public_publication(authorization)
        arrays = load_public_arrays(authorization, publication)
        results = []
        for checkpoint, device in zip(checkpoints, devices):
            adapter = build_adapter(
                authorization=authorization,
                checkpoint=checkpoint,
                device=device,
            )
            try:
                result = evaluate_public_checkpoint(
                    adapter=adapter,
                    arrays=arrays,
                    authorization=authorization,
                    checkpoint_specification=checkpoint,
                    batch_size=expected_batch,
                    include_records=True,
                )
            finally:
                release_adapter(adapter)
                del adapter
            result_path = output / f"lewm_seed{checkpoint['training_seed']}.json"
            _write_json_x(result_path, result)
            results.append(result)
        matrix = aggregate_public_results(results, authorization=authorization)
        matrix["authorization"] = {
            "matrix_request": file_identity(
                request_path,
                logical_path=portable_contextworld_path(request_path),
            ),
            "public_access_started": file_identity(
                access_path,
                logical_path=portable_contextworld_path(access_path),
            ),
        }
        matrix_path = output / "matrix_score.json"
        _write_json_x(matrix_path, matrix)
        success_path = output / "_SUCCESS.json"
        _write_json_x(
            success_path,
            {
                "schema_version": 1,
                "status": "completed_one_use_public_scoring",
                "completed_at_utc": _utc_now(),
                "matrix_score": file_identity(
                    matrix_path,
                    logical_path=portable_contextworld_path(matrix_path),
                ),
                "checkpoint_results": [
                    file_identity(
                        output / f"lewm_seed{checkpoint['training_seed']}.json",
                        logical_path=portable_contextworld_path(
                            output / f"lewm_seed{checkpoint['training_seed']}.json"
                        ),
                    )
                    for checkpoint in checkpoints
                ],
                "public_test": {
                    "generated": True,
                    "hashed": True,
                    "opened": True,
                    "read": True,
                    "scored": True,
                    "used_for_training_or_selection": False,
                },
                "rerun_authorized": False,
            },
        )
        return {
            "status": "completed_one_use_public_scoring",
            "matrix_score": str(matrix_path),
            "passed": matrix["passed"],
            "checkpoints_passed": matrix["checkpoints_passed"],
            "success_marker": str(success_path),
        }
    except BaseException as error:
        failure_path = output / "infrastructure_failure_receipt.json"
        if namespace_reserved and not failure_path.exists():
            _write_json_x(
                failure_path,
                {
                    "schema_version": 1,
                    "status": (
                        "public_campaign_failed_after_access_no_rerun_authorized"
                        if access_started
                        else "public_campaign_preaccess_failed_attempt_consumed_no_rerun_authorized"
                    ),
                    "failed_at_utc": _utc_now(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "public_access_started": access_started,
                    "public_test_may_have_been_read": access_started,
                    "scoring_namespace_reserved": namespace_reserved,
                    "rerun_authorized": False,
                    "next_step": "archive and freeze a distinct recovery authorization",
                },
            )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument(
        "--freeze-receipt", type=Path, default=DEFAULT_FREEZE_RECEIPT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    result = run_public_matrix(
        preregistration=args.prereg,
        freeze_receipt=args.freeze_receipt,
        output=args.output,
        devices=devices,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
