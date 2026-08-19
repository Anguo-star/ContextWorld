#!/usr/bin/env python3
"""Score one LeWM checkpoint on frozen History-3 action-delay assets."""

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

from contextworld.benchmarks.adapters import StableWorldModelLeWMAdapter
from contextworld.evaluation.action_delay_validation import (
    LOSS_RECORDS_PER_CHECKPOINT,
    MODEL_PREDICTIONS_PER_CHECKPOINT,
    QUERY_COUNT,
    file_sha256,
    load_validation_assets,
    score_validation_assets,
    summarize_validation_records,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_action_delay_h3_validation_v1.yaml"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON 顶层必须是对象：{path}")
    return payload


def _checkpoint_protocol(path: Path) -> dict[str, int]:
    payload = _load_json(path)
    return {
        "history_size": int(payload["wm"]["history_size"]),
        "num_preds": int(payload["wm"]["num_preds"]),
        "frameskip": int(payload["data"]["dataset"]["frameskip"]),
        "num_steps": int(payload["data"]["dataset"]["num_steps"]),
        "action_encoder_input_dim": int(
            payload["model"]["action_encoder"]["input_dim"]
        ),
    }


def _training_receipt(
    *,
    report_path: Path | None,
    checkpoint: Path,
    expected_steps: int | None,
) -> dict[str, Any] | None:
    if report_path is None:
        return None
    _require(report_path.is_file(), f"训练报告不存在：{report_path}")
    report = _load_json(report_path)
    _require(report.get("passed") is True, "训练报告未通过")
    global_step = int(report.get("training", {}).get("global_step", -1))
    if expected_steps is not None:
        _require(
            global_step == expected_steps,
            f"训练步数不是冻结值 {expected_steps}",
        )
    artifact = report.get("artifacts", {}).get("pretrained")
    _require(artifact is not None, "训练报告没有最终权重路径")
    _require(
        Path(artifact).resolve() == checkpoint,
        "训练报告与待评测 checkpoint 不一致",
    )
    expected_hash = report.get("artifacts", {}).get("pretrained_sha256")
    observed_hash = file_sha256(checkpoint)
    _require(
        expected_hash == observed_hash,
        "训练报告中的最终权重哈希不匹配",
    )
    return {
        "path": str(report_path),
        "sha256": file_sha256(report_path),
        "global_step": global_step,
        "checkpoint_sha256": observed_hash,
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stablewm-repo")
    parser.add_argument("--stablewm-ref")
    parser.add_argument(
        "--expected-training-steps",
        type=int,
        default=1024,
        help="Formal fine-tuning step count; use 0 for an unchanged baseline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(
        config["benchmark"]
        == "tworoom_action_delay_history3_validation_v1",
        "不是冻结的动作响应延迟 Validation 配置",
    )
    catalog_path = resolve_contextworld_path(
        config["artifacts"]["catalog"], repo_root=ROOT
    )
    build_report_path = catalog_path.parent / "build_report.json"
    build_report = _load_json(build_report_path)
    _require(
        build_report.get("status") == "passed",
        "动作响应延迟 Eval 构建报告未通过",
    )
    _require(
        build_report.get("catalog_sha256") == file_sha256(catalog_path),
        "冻结 Eval catalog 哈希不匹配",
    )

    checkpoint = resolve_contextworld_path(args.checkpoint, repo_root=ROOT)
    _require(checkpoint.is_file(), f"checkpoint 不存在：{checkpoint}")
    checkpoint_config = checkpoint.parent / "config.json"
    _require(
        checkpoint_config.is_file(),
        f"checkpoint 配置不存在：{checkpoint_config}",
    )
    protocol = _checkpoint_protocol(checkpoint_config)
    _require(protocol["history_size"] == 3, "checkpoint 不是 History=3")
    _require(protocol["frameskip"] == 5, "checkpoint frameskip 不是 5")
    _require(
        protocol["action_encoder_input_dim"] == 10,
        "checkpoint 动作输入维度不是 5×2",
    )

    report_path = (
        resolve_contextworld_path(args.training_report, repo_root=ROOT)
        if args.training_report is not None
        else None
    )
    receipt = _training_receipt(
        report_path=report_path,
        checkpoint=checkpoint,
        expected_steps=(
            None
            if int(args.expected_training_steps) == 0
            else int(args.expected_training_steps)
        ),
    )
    normalizer = resolve_contextworld_path(
        "artifacts/splits/tworoom_original_train_s3072_normalizer.json",
        repo_root=ROOT,
    )
    _require(
        file_sha256(normalizer)
        == "7a5be7ea867bced446c1671b0b2c0ff6450ffc61e1a7bdbbfc5eaa0942f635db",
        "冻结 action normalizer 哈希不匹配",
    )
    stablewm_repo = str(
        args.stablewm_repo or config["stable_worldmodel"]["repo"]
    )
    stablewm_ref = str(
        args.stablewm_ref or config["stable_worldmodel"]["commit"]
    )
    _require(
        stablewm_repo == str(config["stable_worldmodel"]["repo"]),
        "正式评测不允许替换 Stable-WorldModel 仓库",
    )
    _require(
        stablewm_ref == str(config["stable_worldmodel"]["commit"]),
        "正式评测不允许替换 Stable-WorldModel commit",
    )

    catalog, assets = load_validation_assets(
        catalog_path,
        repo_root=ROOT,
    )
    _require(len(assets) == QUERY_COUNT, "Eval 不是 50×6=300 个 query")
    adapter = StableWorldModelLeWMAdapter.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=stablewm_repo,
        stablewm_ref=stablewm_ref,
        device=args.device,
    )
    state_before = adapter.frozen_state_hash()
    scored = score_validation_assets(
        adapter,
        assets,
        batch_size=int(args.batch_size),
    )
    state_after = adapter.frozen_state_hash()
    _require(state_before == state_after, "评测期间模型状态发生变化")
    audit = scored["score_audit"]
    _require(
        audit["queries"] == QUERY_COUNT
        and audit["model_predictions"]
        == MODEL_PREDICTIONS_PER_CHECKPOINT
        and audit["loss_records"] == LOSS_RECORDS_PER_CHECKPOINT
        and audit["online_environment_calls"] == 0,
        "评分计数或离线执行约束不满足",
    )
    summary = summarize_validation_records(scored["records"])

    output = resolve_contextworld_path(args.output, repo_root=ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "model_id": str(args.model_id),
        "training_seed": int(args.training_seed),
        "identity": {
            "validation_config": str(config_path),
            "validation_config_sha256": file_sha256(config_path),
            "catalog": str(catalog_path),
            "catalog_sha256": file_sha256(catalog_path),
            "catalog_content_manifest_sha256": catalog[
                "content_manifest_sha256"
            ],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_config": str(checkpoint_config),
            "checkpoint_config_sha256": file_sha256(checkpoint_config),
            "checkpoint_protocol": protocol,
            "normalizer": str(normalizer),
            "normalizer_sha256": file_sha256(normalizer),
        },
        "training_receipt": receipt,
        "model": adapter.metadata,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "score_audit": audit,
        "summary": summary,
        "records": scored["records"],
    }
    write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "model_id": args.model_id,
                "training_seed": int(args.training_seed),
                "queries": audit["queries"],
                "model_predictions": audit["model_predictions"],
                "loss_records": audit["loss_records"],
                "overall": summary["overall"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
