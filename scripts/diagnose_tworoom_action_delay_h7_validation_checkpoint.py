#!/usr/bin/env python3
"""Score an arbitrary History-7 checkpoint on frozen Action Delay Validation.

Unlike the formal matrix evaluator, this post-hoc diagnostic accepts a
checkpoint outside the preregistered nine-model LeWM matrix.  It keeps the
frozen 300-query catalog, offline physical futures, latent targets, count
audits, and summary implementation unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMHistory7Adapter,
    StableWorldModelPLDMHistory7Adapter,
)
from contextworld.evaluation.action_delay_h7_score import (
    HORIZON_LOSS_RECORDS_PER_CHECKPOINT,
    MODEL_PREDICTIONS_PER_CHECKPOINT,
    TARGET_ENCODINGS_PER_CHECKPOINT,
    TRAJECTORY_COMPARISONS_PER_CHECKPOINT,
    load_h7_validation_assets,
    score_h7_validation_assets,
    summarize_h7_validation_records,
)
from contextworld.evaluation.action_delay_h7_validation import (
    QUERY_COUNT,
    file_sha256,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_action_delay_h7_scoring_v1.yaml"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _checkpoint_protocol(path: Path) -> dict[str, int]:
    value = _load_json(path)
    return {
        "history_size": int(value["wm"]["history_size"]),
        "num_preds": int(value["wm"]["num_preds"]),
        "frameskip": int(value["data"]["dataset"]["frameskip"]),
        "num_steps": int(value["data"]["dataset"]["num_steps"]),
        "action_encoder_input_dim": int(
            value["model"]["action_encoder"]["input_dim"]
        ),
    }


def _training_receipt(
    path: Path | None,
    *,
    checkpoint: Path,
    model_family: str,
) -> dict[str, Any]:
    if path is None:
        return {"required": False, "passed": True}
    path = path.expanduser().resolve()
    _require(path.is_file(), f"Training report does not exist: {path}")
    report = _load_json(path)
    checks = {
        "report_passed": report.get("passed") is True,
        "training_complete": report.get("training", {}).get(
            "training_complete"
        )
        is True,
        "global_step_exact": int(
            report.get("training", {}).get("global_step", -1)
        )
        == 1024,
        "world_size_exact": int(
            report.get("training", {}).get("world_size", -1)
        )
        == 8,
        "training_method_exact": report.get("model", {}).get(
            "training_method"
        )
        == model_family,
        "checkpoint_path_exact": Path(
            report.get("artifacts", {}).get("pretrained", "")
        ).resolve()
        == checkpoint,
        "checkpoint_hash_exact": report.get("artifacts", {}).get(
            "pretrained_sha256"
        )
        == file_sha256(checkpoint),
        "save_load_exact": report.get("save_load_exact") is True,
    }
    _require(
        all(checks.values()),
        "Training receipt failed: "
        + ", ".join(name for name, passed in checks.items() if not passed),
    )
    return {
        "required": True,
        "path": str(path),
        "sha256": file_sha256(path),
        "checks": checks,
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-family",
        choices=("lewm", "pldm"),
        required=True,
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stablewm-repo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(
        config.get("benchmark")
        == "tworoom_action_delay_history7_scoring_v1",
        "Expected the frozen History-7 scoring configuration",
    )
    checkpoint = args.checkpoint.expanduser().resolve()
    _require(checkpoint.is_file(), f"Checkpoint does not exist: {checkpoint}")
    checkpoint_config = checkpoint.parent / "config.json"
    _require(
        checkpoint_config.is_file(),
        f"Checkpoint config does not exist: {checkpoint_config}",
    )
    protocol = _checkpoint_protocol(checkpoint_config)
    _require(
        protocol
        == {
            "history_size": 7,
            "num_preds": 1,
            "frameskip": 5,
            "num_steps": 8,
            "action_encoder_input_dim": 10,
        },
        f"Checkpoint protocol is not History-7 Action Delay: {protocol}",
    )
    receipt = _training_receipt(
        args.training_report,
        checkpoint=checkpoint,
        model_family=str(args.model_family),
    )

    catalog_path = resolve_contextworld_path(
        config["source_identity"]["validation_catalog"]["path"],
        repo_root=ROOT,
    )
    catalog, assets = load_h7_validation_assets(
        catalog_path,
        repo_root=ROOT,
    )
    _require(len(assets) == QUERY_COUNT, "Validation must contain 300 queries")
    normalizer = resolve_contextworld_path(
        config["source_identity"]["normalizer"]["path"],
        repo_root=ROOT,
    )
    adapter_class = (
        StableWorldModelLeWMHistory7Adapter
        if args.model_family == "lewm"
        else StableWorldModelPLDMHistory7Adapter
    )
    adapter = adapter_class.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=str(
            args.stablewm_repo or config["stable_worldmodel"]["repo"]
        ),
        stablewm_ref=str(config["stable_worldmodel"]["commit"]),
        device=args.device,
    )
    state_before = adapter.frozen_state_hash()
    scored = score_h7_validation_assets(
        adapter,
        assets,
        batch_size=int(
            args.batch_size or config["evaluation"]["batch_size"]
        ),
    )
    state_after = adapter.frozen_state_hash()
    _require(state_before == state_after, "Model state changed during scoring")

    audit = scored["score_audit"]
    count_checks = {
        "queries_exact": audit["queries"] == QUERY_COUNT,
        "model_predictions_exact": (
            audit["model_predictions"]
            == MODEL_PREDICTIONS_PER_CHECKPOINT
        ),
        "target_encodings_exact": (
            audit["target_encodings"]
            == TARGET_ENCODINGS_PER_CHECKPOINT
        ),
        "trajectory_comparisons_exact": (
            audit["trajectory_comparisons"]
            == TRAJECTORY_COMPARISONS_PER_CHECKPOINT
        ),
        "horizon_loss_records_exact": (
            audit["horizon_loss_records"]
            == HORIZON_LOSS_RECORDS_PER_CHECKPOINT
        ),
        "offline_only": audit["online_environment_calls"] == 0,
        "no_privileged_fields": not audit[
            "privileged_fields_passed_to_adapter"
        ],
    }
    _require(
        all(count_checks.values()),
        f"Validation count audit failed: {count_checks}",
    )
    summary = summarize_h7_validation_records(scored["records"])

    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": 1,
        "benchmark": (
            "tworoom_action_delay_history7_validation_checkpoint_"
            "diagnostic_v1"
        ),
        "status": "completed_post_hoc_diagnostic",
        "claim_boundary": {
            "changes_formal_benchmark_result": False,
            "uses_frozen_validation": True,
            "hidden_test_used": False,
            "online_environment_used": False,
        },
        "label": str(args.label),
        "model_family": str(args.model_family),
        "identity": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_config": str(checkpoint_config),
            "checkpoint_config_sha256": file_sha256(checkpoint_config),
            "checkpoint_protocol": protocol,
            "frozen_scoring_config": str(config_path),
            "frozen_scoring_config_sha256": file_sha256(config_path),
            "frozen_catalog": str(catalog_path),
            "frozen_catalog_sha256": file_sha256(catalog_path),
            "catalog_content_manifest_sha256": catalog[
                "content_manifest_sha256"
            ],
            "normalizer": str(normalizer),
            "normalizer_sha256": file_sha256(normalizer),
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": file_sha256(Path(__file__).resolve()),
        },
        "training_receipt": receipt,
        "model": adapter.metadata,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "score_audit": {**audit, "checks": count_checks, "passed": True},
        "summary": summary,
        "records": scored["records"],
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "label": args.label,
                "model_family": args.model_family,
                "output": str(output),
                "trajectory": summary["trajectory"]["overall"],
                "by_horizon": {
                    horizon: values["overall"]
                    for horizon, values in summary["by_horizon"].items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
