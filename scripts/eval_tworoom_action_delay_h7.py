#!/usr/bin/env python3
"""Score one formal History-7 Action Delay checkpoint."""

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
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON 顶层必须是对象：{path}")
    return payload


def _model_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for role, rows in config["models"].items():
        for row in rows:
            slug = str(row["slug"])
            _require(slug not in models, f"模型 slug 重复：{slug}")
            models[slug] = {**row, "role": str(role)}
    return models


def _format_model_path(
    config: dict[str, Any],
    *,
    field: str,
    slug: str,
) -> Path:
    logical = str(config["model_artifact_pattern"][field]).format(
        slug=slug
    )
    return resolve_contextworld_path(logical, repo_root=ROOT)


def _validate_source_identity(config: dict[str, Any]) -> dict[str, Any]:
    audited = {}
    for name, identity in config["source_identity"].items():
        path = resolve_contextworld_path(identity["path"], repo_root=ROOT)
        observed = file_sha256(path)
        _require(
            observed == str(identity["sha256"]),
            f"冻结来源哈希变化：{name}",
        )
        audited[name] = {"path": str(path), "sha256": observed}
    return audited


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
    config: dict[str, Any],
    model: dict[str, Any],
    *,
    checkpoint: Path,
    report_path: Path,
) -> dict[str, Any]:
    _require(report_path.is_file(), f"训练报告不存在：{report_path}")
    report = _load_json(report_path)
    expected_steps = int(
        config["model_artifact_pattern"]["expected_training_steps"]
    )
    checks = {
        "report_passed": report.get("passed") is True,
        "model_id_exact": report.get("model_id") == model["model_id"],
        "training_complete": report.get("training", {}).get(
            "training_complete"
        )
        is True,
        "global_step_exact": int(
            report.get("training", {}).get("global_step", -1)
        )
        == expected_steps,
        "world_size_exact": int(
            report.get("training", {}).get("world_size", -1)
        )
        == 8,
        "initialization_applied": report.get(
            "initialization_checkpoint", {}
        ).get("applied")
        is True,
        "temporal_adaptation_passed": report.get(
            "initialization_checkpoint", {}
        )
        .get("temporal_adaptation_audit", {})
        .get("passed")
        is True,
        "initialized_state_exact": report.get(
            "initialization_checkpoint", {}
        ).get("initialized_state_sha256")
        == config["model_artifact_pattern"][
            "expected_initialization_state_sha256"
        ],
        "checkpoint_path_exact": Path(
            report.get("artifacts", {}).get("pretrained", "")
        ).resolve()
        == checkpoint,
        "checkpoint_hash_exact": report.get("artifacts", {}).get(
            "pretrained_sha256"
        )
        == file_sha256(checkpoint),
    }
    _require(
        all(checks.values()),
        "正式训练报告未通过评测前审计："
        + ", ".join(name for name, passed in checks.items() if not passed),
    )
    return {
        "path": str(report_path),
        "sha256": file_sha256(report_path),
        "checks": checks,
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--stablewm-repo")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(
        config.get("benchmark")
        == "tworoom_action_delay_history7_scoring_v1",
        "不是冻结的 History=7 Action Delay 评分配置",
    )
    models = _model_lookup(config)
    _require(args.model in models, f"未知模型：{args.model}")
    model = models[args.model]
    source_identity = _validate_source_identity(config)

    checkpoint = _format_model_path(
        config, field="checkpoint", slug=args.model
    )
    report_path = _format_model_path(
        config, field="training_report", slug=args.model
    )
    _require(checkpoint.is_file(), f"checkpoint 不存在：{checkpoint}")
    checkpoint_config = checkpoint.parent / "config.json"
    _require(
        checkpoint_config.is_file(),
        f"checkpoint config 不存在：{checkpoint_config}",
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
        f"checkpoint 不是冻结的 History=7 协议：{protocol}",
    )
    training_receipt = _training_receipt(
        config,
        model,
        checkpoint=checkpoint,
        report_path=report_path,
    )

    catalog_path = resolve_contextworld_path(
        config["source_identity"]["validation_catalog"]["path"],
        repo_root=ROOT,
    )
    catalog, assets = load_h7_validation_assets(
        catalog_path, repo_root=ROOT
    )
    _require(len(assets) == QUERY_COUNT, "Validation 不是 300 个 query")
    normalizer = resolve_contextworld_path(
        config["source_identity"]["normalizer"]["path"],
        repo_root=ROOT,
    )
    stablewm_repo = str(
        args.stablewm_repo or config["stable_worldmodel"]["repo"]
    )
    stablewm_commit = str(config["stable_worldmodel"]["commit"])
    batch_size = int(
        args.batch_size or config["evaluation"]["batch_size"]
    )
    adapter = StableWorldModelLeWMHistory7Adapter.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=stablewm_repo,
        stablewm_ref=stablewm_commit,
        device=args.device,
    )
    state_before = adapter.frozen_state_hash()
    scored = score_h7_validation_assets(
        adapter,
        assets,
        batch_size=batch_size,
    )
    state_after = adapter.frozen_state_hash()
    _require(state_before == state_after, "评测期间模型状态发生变化")
    audit = scored["score_audit"]
    expected = config["evaluation"]["expected_counts_per_checkpoint"]
    count_checks = {
        "queries_exact": audit["queries"] == QUERY_COUNT,
        "model_predictions_exact": audit["model_predictions"]
        == MODEL_PREDICTIONS_PER_CHECKPOINT
        == int(expected["model_predictions"]),
        "target_encodings_exact": audit["target_encodings"]
        == TARGET_ENCODINGS_PER_CHECKPOINT
        == int(expected["target_encodings"]),
        "trajectory_comparisons_exact": audit[
            "trajectory_comparisons"
        ]
        == TRAJECTORY_COMPARISONS_PER_CHECKPOINT
        == int(expected["trajectory_comparisons"]),
        "horizon_loss_records_exact": audit["horizon_loss_records"]
        == HORIZON_LOSS_RECORDS_PER_CHECKPOINT
        == int(expected["horizon_loss_records"]),
        "offline_only": audit["online_environment_calls"] == 0,
        "no_privileged_fields": not audit[
            "privileged_fields_passed_to_adapter"
        ],
    }
    _require(
        all(count_checks.values()),
        f"History=7 评分计数不一致：{count_checks}",
    )
    summary = summarize_h7_validation_records(scored["records"])

    output = resolve_contextworld_path(
        (
            args.output
            if args.output is not None
            else Path(config["artifacts"]["results_root"])
            / f"{args.model}.json"
        ),
        repo_root=ROOT,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "model_slug": str(args.model),
        "model_id": str(model["model_id"]),
        "training_role": str(model["role"]),
        "training_seed": int(model["training_seed"]),
        "identity": {
            "scoring_config": str(config_path),
            "scoring_config_sha256": file_sha256(config_path),
            "scoring_sources": {
                "entrypoint": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": file_sha256(Path(__file__).resolve()),
                },
                "implementation": {
                    "path": str(
                        ROOT
                        / "contextworld/evaluation/"
                        "action_delay_h7_score.py"
                    ),
                    "sha256": file_sha256(
                        ROOT
                        / "contextworld/evaluation/"
                        "action_delay_h7_score.py"
                    ),
                },
                "adapter": {
                    "path": str(
                        ROOT / "contextworld/benchmarks/adapters.py"
                    ),
                    "sha256": file_sha256(
                        ROOT / "contextworld/benchmarks/adapters.py"
                    ),
                },
            },
            "source_identity": source_identity,
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
        "training_receipt": training_receipt,
        "model": adapter.metadata,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "score_audit": {**audit, "checks": count_checks, "passed": True},
        "summary": summary,
        "records": scored["records"],
    }
    write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "model": args.model,
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
