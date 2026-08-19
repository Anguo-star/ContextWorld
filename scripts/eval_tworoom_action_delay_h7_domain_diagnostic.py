#!/usr/bin/env python3
"""Score one History-7 checkpoint on the frozen domain diagnostic."""

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
from contextworld.evaluation.action_delay_h7_domain_score import (
    load_domain_catalog,
    load_domain_track_assets,
    score_domain_track,
)
from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_domain_diagnostic_scoring_v1.yaml"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON 顶层必须是对象：{path}")
    return value


def _models(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for role, rows in config["models"].items():
        for row in rows:
            slug = str(row["slug"])
            _require(slug not in result, f"模型 slug 重复：{slug}")
            result[slug] = {**row, "role": role}
    return result


def _source_identity(config: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, row in config["source_identity"].items():
        path = resolve_contextworld_path(row["path"], repo_root=ROOT)
        observed = file_sha256(path)
        _require(
            observed == str(row["sha256"]),
            f"冻结输入哈希变化：{name}",
        )
        result[name] = {"path": str(path), "sha256": observed}
    return result


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
    model: dict[str, Any],
    *,
    checkpoint: Path,
    report_path: Path,
) -> dict[str, Any]:
    report = _load_json(report_path)
    checks = {
        "report_hash_exact": file_sha256(report_path)
        == model["training_report_sha256"],
        "report_passed": report.get("passed") is True,
        "model_id_exact": report.get("model_id") == model["model_id"],
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
        "checkpoint_path_exact": Path(
            report.get("artifacts", {}).get("pretrained", "")
        ).resolve()
        == checkpoint,
        "checkpoint_hash_exact": (
            file_sha256(checkpoint)
            == model["checkpoint_sha256"]
            == report.get("artifacts", {}).get("pretrained_sha256")
        ),
    }
    _require(
        all(checks.values()),
        "训练产物审计失败："
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
        == "tworoom_action_delay_history7_domain_diagnostic_scoring_v1",
        "不是冻结的 History=7 训练域诊断评分配置",
    )
    models = _models(config)
    _require(args.model in models, f"未知模型：{args.model}")
    model = models[args.model]
    source_identity = _source_identity(config)

    checkpoint = resolve_contextworld_path(
        model["checkpoint"],
        repo_root=ROOT,
    )
    report_path = resolve_contextworld_path(
        model["training_report"],
        repo_root=ROOT,
    )
    _require(checkpoint.is_file(), f"checkpoint 不存在：{checkpoint}")
    _require(
        file_sha256(checkpoint) == model["checkpoint_sha256"],
        f"checkpoint 哈希变化：{checkpoint}",
    )
    checkpoint_config = checkpoint.parent / "config.json"
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
        f"checkpoint 协议不一致：{protocol}",
    )
    training_receipt = _training_receipt(
        model,
        checkpoint=checkpoint,
        report_path=report_path,
    )
    catalog_path = resolve_contextworld_path(
        config["source_identity"]["diagnostic_catalog"]["path"],
        repo_root=ROOT,
    )
    catalog = load_domain_catalog(catalog_path)
    normalizer = resolve_contextworld_path(
        config["source_identity"]["normalizer"]["path"],
        repo_root=ROOT,
    )
    adapter = StableWorldModelLeWMHistory7Adapter.from_checkpoint(
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
    batch_size = int(
        args.batch_size or config["evaluation"]["batch_size"]
    )
    track_results = {}
    totals = {
        "queries": 0,
        "model_predictions": 0,
        "target_encodings": 0,
        "trajectory_comparisons": 0,
        "horizon_loss_records": 0,
        "online_environment_calls": 0,
    }
    for track in config["evaluation"]["tracks"]:
        assets = load_domain_track_assets(
            catalog,
            track=str(track),
            repo_root=ROOT,
        )
        result = score_domain_track(
            adapter,
            assets,
            batch_size=batch_size,
        )
        track_results[str(track)] = result
        for key in totals:
            totals[key] += int(result["audit"][key])
        print(
            f"[h7-domain-score] {args.model} completed {track}",
            flush=True,
        )
    state_after = adapter.frozen_state_hash()
    _require(state_before == state_after, "评测期间模型参数变化")
    expected = config["evaluation"]["expected_counts_per_checkpoint"]
    count_checks = {
        key: totals[key] == int(expected[key])
        for key in totals
    }
    count_checks["every_track_present"] = set(track_results) == set(
        config["evaluation"]["tracks"]
    )
    count_checks["no_privileged_fields"] = all(
        not row["audit"]["privileged_fields_passed_to_adapter"]
        for row in track_results.values()
    )
    _require(
        all(count_checks.values()),
        f"训练域诊断计数不一致：{count_checks}",
    )
    output = resolve_contextworld_path(
        args.output
        or Path(config["artifacts"]["results_root"])
        / f"{args.model}.json",
        repo_root=ROOT,
    )
    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "model_slug": args.model,
        "model_id": model["model_id"],
        "training_role": model["role"],
        "training_seed": int(model["training_seed"]),
        "identity": {
            "scoring_config": str(config_path),
            "scoring_config_sha256": file_sha256(config_path),
            "scoring_sources": {
                name: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for name, path in {
                    "adapter": (
                        ROOT / "contextworld/benchmarks/adapters.py"
                    ),
                    "implementation": (
                        ROOT
                        / "contextworld/evaluation/"
                        "action_delay_h7_domain_score.py"
                    ),
                    "entrypoint": Path(__file__).resolve(),
                }.items()
            },
            "source_identity": source_identity,
            "catalog_content_manifest_sha256": catalog[
                "content_manifest_sha256"
            ],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_config": str(checkpoint_config),
            "checkpoint_config_sha256": file_sha256(
                checkpoint_config
            ),
            "checkpoint_protocol": protocol,
            "normalizer": str(normalizer),
            "normalizer_sha256": file_sha256(normalizer),
        },
        "training_receipt": training_receipt,
        "model": adapter.metadata,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "score_audit": {
            **totals,
            "checks": count_checks,
            "passed": True,
        },
        "tracks": track_results,
    }
    write_json(output, result)
    print(
        json.dumps(
            {
                "model": args.model,
                "output": str(output),
                "audit": result["score_audit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
