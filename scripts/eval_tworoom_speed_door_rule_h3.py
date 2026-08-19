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
from contextworld.evaluation.speed_door_rule_score import (
    evaluate_checkpoint_gate,
    score_validation_assets,
    summarize_records,
)
from contextworld.evaluation.speed_door_rule_validation import (
    file_sha256,
    load_validation_assets,
    validate_frozen_config,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.config import load_config
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_speed_door_rule_h3_validation_v1.yaml"
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
    reported_checkpoint = Path(
        str(artifacts.get("pretrained", ""))
    ).expanduser().resolve()
    checks = {
        "report_passed": payload.get("passed") is True,
        "model_id_exact": (
            str(payload.get("model_id")) == training_model_id
        ),
        "training_seed_exact": (
            int(training.get("seed_before_model_initialization", -1))
            == int(training_seed)
        ),
        "training_complete": training.get("training_complete") is True,
        "optimizer_steps_exact": int(training.get("global_step", -1))
        == 1024,
        "checkpoint_path_exact": reported_checkpoint == checkpoint,
        "checkpoint_sha256_exact": (
            str(artifacts.get("pretrained_sha256"))
            == file_sha256(checkpoint)
        ),
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
            "Score one checkpoint on the frozen History=3 Speed × Door "
            "Rule composition Validation"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--training-model-id",
        help=(
            "Model ID recorded by the training report when the public "
            "evaluation label is a clearer alias."
        ),
    )
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument(
        "--adapter",
        choices=("lewm", "pldm"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    validate_frozen_config(config)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    normalizer = resolve_contextworld_path(
        config["adapter"]["normalizer"],
        repo_root=ROOT,
    )
    observed_normalizer_sha = file_sha256(normalizer)
    if observed_normalizer_sha != config["adapter"]["normalizer_sha256"]:
        raise RuntimeError("Frozen normalizer hash mismatch")
    catalog_path = resolve_contextworld_path(
        config["artifacts"]["catalog"],
        repo_root=ROOT,
    )
    assets, asset_audit = load_validation_assets(
        catalog_path,
        repo_root=ROOT,
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
    batch_size = int(
        args.batch_size or config["evaluation"]["batch_size"]
    )
    epsilon = float(config["decision_gates"]["epsilon"])
    scored = score_validation_assets(
        adapter,
        assets,
        batch_size=batch_size,
        epsilon=epsilon,
    )
    summary = summarize_records(scored["records"])
    checkpoint_gate = evaluate_checkpoint_gate(
        summary=summary,
        score_audit=scored["score_audit"],
        gates=config["decision_gates"],
    )
    result = {
        "schema_version": 1,
        "benchmark": str(config["benchmark"]),
        "status": "completed",
        "model_id": str(args.model_id),
        "training_model_id": str(
            args.training_model_id or args.model_id
        ),
        "training_seed": int(args.training_seed),
        "adapter_kind": str(args.adapter),
        "adapter": adapter.metadata,
        "checkpoint_sha256": file_sha256(checkpoint),
        "training_report_audit": report_audit,
        "asset_audit": asset_audit,
        "score_audit": scored["score_audit"],
        "summary": summary,
        "checkpoint_gate": checkpoint_gate,
        "records": scored["records"],
        "identity": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "catalog": str(catalog_path),
            "catalog_sha256": file_sha256(catalog_path),
            "normalizer": str(normalizer),
            "normalizer_sha256": observed_normalizer_sha,
        },
        "claim_limit": (
            "Raw native latent MSE is interpreted only within this "
            "checkpoint. The six accuracies and preregistered gates are "
            "comparable across checkpoints."
        ),
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (
            resolve_contextworld_path(
                config["artifacts"]["results_root"],
                repo_root=ROOT,
            )
            / f"{args.model_id}_s{args.training_seed}.json"
        )
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
                "overall": result["summary"]["overall"],
                "checkpoint_gate": result["checkpoint_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
