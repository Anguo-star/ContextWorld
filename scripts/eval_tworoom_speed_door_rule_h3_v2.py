#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMAdapter,
    StableWorldModelPLDMAdapter,
)
from contextworld.evaluation.speed_door_rule_v2_score import (
    evaluate_v2_checkpoint_gate,
    score_v2_assets,
    summarize_v2_scores,
)
from contextworld.evaluation.speed_door_rule_v2_validation import (
    file_sha256,
    load_v2_validation_assets,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.config import load_config
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_speed_door_rule_h3_v2.yaml"
)
DEFAULT_NORMALIZER = (
    "artifacts/splits/tworoom_original_train_s3072_normalizer.json"
)
DEFAULT_NORMALIZER_SHA256 = (
    "7a5be7ea867bced446c1671b0b2c0ff6450ffc61e1a7bdbbfc5eaa0942f635db"
)


def _training_report_audit(
    path: Path | None,
    *,
    checkpoint: Path,
    training_model_id: str,
    training_seed: int,
) -> dict:
    if path is None:
        return {
            "required": False,
            "reason": "read_only_original_baseline",
            "passed": True,
        }
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifacts = dict(payload.get("artifacts", {}))
    training = dict(payload.get("training", {}))
    checks = {
        "report_passed": payload.get("passed") is True,
        "model_id_exact": (
            str(payload.get("model_id")) == str(training_model_id)
        ),
        "training_seed_exact": int(
            training.get("seed_before_model_initialization", -1)
        )
        == int(training_seed),
        "training_complete": training.get("training_complete") is True,
        "optimizer_steps_exact": int(training.get("global_step", -1))
        == 1024,
        "checkpoint_path_exact": Path(
            str(artifacts.get("pretrained", ""))
        ).expanduser().resolve()
        == checkpoint,
        "checkpoint_sha256_exact": str(
            artifacts.get("pretrained_sha256")
        )
        == file_sha256(checkpoint),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Training report does not bind the checkpoint: {checks}"
        )
    return {
        "required": True,
        "path": str(path),
        "sha256": file_sha256(path),
        "checks": checks,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score one checkpoint on frozen Speed × Door Rule v2 h1/h2 "
            "offline Validation"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--training-model-id")
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument(
        "--role",
        choices=("speed_only", "door_only", "joint", "descriptive"),
        required=True,
    )
    parser.add_argument(
        "--adapter", choices=("lewm", "pldm"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--normalizer", default=DEFAULT_NORMALIZER
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    normalizer = resolve_contextworld_path(
        args.normalizer, repo_root=ROOT
    )
    if file_sha256(normalizer) != DEFAULT_NORMALIZER_SHA256:
        raise RuntimeError("Frozen normalizer hash mismatch")
    catalog_path = resolve_contextworld_path(
        config["artifacts"]["validation_catalog"], repo_root=ROOT
    )
    assets, asset_audit = load_v2_validation_assets(
        catalog_path, repo_root=ROOT
    )
    report_audit = _training_report_audit(
        args.training_report,
        checkpoint=checkpoint,
        training_model_id=str(
            args.training_model_id or args.model_id
        ),
        training_seed=int(args.training_seed),
    )

    adapter_class = {
        "lewm": StableWorldModelLeWMAdapter,
        "pldm": StableWorldModelPLDMAdapter,
    }[args.adapter]
    adapter = adapter_class.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=str(config["stable_worldmodel"]["repo"]),
        stablewm_ref=str(config["stable_worldmodel"]["commit"]),
        device=str(args.device),
    )
    batch_size = int(args.batch_size or 24)
    scored = score_v2_assets(
        adapter,
        assets,
        batch_size=batch_size,
        epsilon=1.0e-12,
    )
    summary = summarize_v2_scores(
        scored["condition_records"], scored["suppression_records"]
    )
    gate = evaluate_v2_checkpoint_gate(
        summary=summary, config=config, role=str(args.role)
    )
    result = {
        "schema_version": 2,
        "benchmark": str(config["benchmark"]),
        "status": "completed",
        "model_id": str(args.model_id),
        "training_model_id": str(
            args.training_model_id or args.model_id
        ),
        "training_seed": int(args.training_seed),
        "role": str(args.role),
        "adapter_kind": str(args.adapter),
        "adapter": adapter.metadata,
        "checkpoint_sha256": file_sha256(checkpoint),
        "training_report_audit": report_audit,
        "asset_audit": asset_audit,
        "score_audit": scored["score_audit"],
        "summary": summary,
        "checkpoint_gate": gate,
        "condition_records": scored["condition_records"],
        "suppression_records": scored["suppression_records"],
        "identity": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "catalog": str(catalog_path),
            "catalog_sha256": file_sha256(catalog_path),
            "normalizer": str(normalizer),
            "normalizer_sha256": file_sha256(normalizer),
        },
        "claim_limit": (
            "Native latent losses are used only for within-checkpoint "
            "choices. h1 and h2 remain separate."
        ),
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else resolve_contextworld_path(
            config["artifacts"]["validation_results_root"],
            repo_root=ROOT,
        )
        / f"{args.model_id}_s{args.training_seed}.json"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "model_id": result["model_id"],
                "training_seed": result["training_seed"],
                "role": result["role"],
                "by_horizon": {
                    key: value["overall"]
                    for key, value in summary["by_horizon"].items()
                },
                "checkpoint_gate": gate,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
