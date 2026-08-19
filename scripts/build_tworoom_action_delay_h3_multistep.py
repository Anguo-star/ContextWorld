#!/usr/bin/env python3
"""构建冻结的 History=3 动作延迟多步与高端延迟 Eval。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_multistep import build_release
from contextworld.evaluation.action_delay_validation import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.stablewm import load_stable_worldmodel


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_action_delay_h3_multistep_extrap_v1.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="构建 50×6 动作延迟多步真实未来 Eval"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("benchmark") not in {
        "tworoom_action_delay_history3_multistep_extrap_v1",
        "tworoom_action_delay_history3_multistep_extrap_v2",
    }:
        raise ValueError("不是冻结的动作延迟多步扩展配置")
    if config.get("status") not in {
        "preregistered_before_catalog_generation_and_model_scoring",
        "preregistered_after_v1_failure_before_v2_catalog_generation_and_model_scoring",
    }:
        raise ValueError("配置没有在数据生成和模型评分前预注册")
    prior = config.get("source_identity", {}).get(
        "completed_multistep_v1"
    )
    if prior is not None:
        prior_path = resolve_contextworld_path(
            prior["final_summary"], repo_root=ROOT
        )
        if file_sha256(prior_path) != prior["final_summary_sha256"]:
            raise ValueError("v1 多步结果哈希发生变化")

    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    if stable_commit != str(config["stable_worldmodel"]["commit"]):
        raise ValueError("Stable-WorldModel commit 与冻结配置不一致")

    output_root = resolve_contextworld_path(
        (
            args.output_root
            if args.output_root is not None
            else config["artifacts"]["root"]
        ),
        repo_root=ROOT,
    )
    report = build_release(
        config=config,
        config_path=config_path,
        repo_root=ROOT,
        output_root=output_root,
    )
    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "status": report["status"],
                "checks": report["checks"],
                "counts": report["counts"],
                "catalog": report["catalog"],
                "catalog_sha256": report["catalog_sha256"],
                "stable_worldmodel_repo": str(stable_repo),
                "stable_worldmodel_commit": stable_commit,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
