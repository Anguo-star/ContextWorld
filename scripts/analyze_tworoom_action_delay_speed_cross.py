#!/usr/bin/env python3
"""汇总动作延迟模型在冻结速度历史 Eval 上的交叉因素诊断。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_validation import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h3_speed_cross_diagnostic_v1.yaml"
)
TRACKS = ("seen_for_speed_model", "unseen_speed_interpolation")
FORMAL_SEEDS = (3072, 4096, 5120)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    benchmark = str(config["benchmark"])
    _require(
        benchmark
        == "tworoom_action_delay_history3_speed_cross_diagnostic_v1",
        "不是动作延迟速度交叉诊断配置",
    )
    for name, identity in config["source_identity"].items():
        path = resolve_contextworld_path(identity["path"], repo_root=ROOT)
        _require(
            file_sha256(path) == identity["sha256"],
            f"冻结来源哈希发生变化：{name}",
        )
    results_root = resolve_contextworld_path(
        (
            args.results_root
            if args.results_root is not None
            else config["artifacts"]["results"]
        ),
        repo_root=ROOT,
    )
    output = resolve_contextworld_path(
        (
            args.output
            if args.output is not None
            else config["artifacts"]["final_summary"]
        ),
        repo_root=ROOT,
    )

    model_rows = {
        str(row["slug"]): {**row, "group": group}
        for group, rows in config["models"].items()
        for row in rows
    }
    models = {}
    result_files = {}
    for slug, model_row in model_rows.items():
        path = results_root / f"{slug}.json"
        _require(path.is_file(), f"缺少正式结果：{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        _require(
            payload.get("status") == "passed"
            and payload.get("benchmark") == benchmark
            and payload["model"]["slug"] == slug
            and int(payload["model"]["training_seed"])
            == int(model_row["training_seed"])
            and payload.get("online_environment_calls") == 0,
            f"结果身份或离线约束不满足：{path}",
        )
        track_rows = {}
        for track in TRACKS:
            summary = payload["tracks"][track]["summary"]
            _require(
                summary["count_audit"]["passed"],
                f"50×6 计数未通过：{slug}/{track}",
            )
            track_rows[track] = {
                "matching_speed_history_passed": bool(
                    summary["decision"]["passed"]
                ),
                "matching_below_both_other_histories_all_speeds": bool(
                    summary["decision"][
                        "matching_below_each_other_history_all_speeds"
                    ]
                ),
                "all_six_eval_seed_directions_positive": bool(
                    summary["decision"][
                        "all_eval_seed_directions_positive"
                    ]
                ),
                "relative_loss_reduction": float(
                    summary["overall"]["relative_loss_reduction"]
                ),
                "relative_loss_reduction_ci": summary["overall"][
                    "relative_loss_reduction_ci"
                ],
                "by_reference_speed": {
                    speed: {
                        "relative_loss_reduction": float(
                            row["relative_loss_reduction"]
                        ),
                        "matching_history_advantage": float(
                            row["matching_history_advantage"]
                        ),
                    }
                    for speed, row in summary[
                        "by_reference_speed"
                    ].items()
                },
            }
        models[slug] = {
            "group": model_row["group"],
            "training_seed": int(model_row["training_seed"]),
            "tracks": track_rows,
        }
        result_files[slug] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }

    groups = {}
    for group, rows in config["models"].items():
        groups[group] = {}
        for track in TRACKS:
            slugs = [str(row["slug"]) for row in rows]
            groups[group][track] = {
                "model_count": len(slugs),
                "models_passing_matching_speed_history_gate": sum(
                    models[slug]["tracks"][track][
                        "matching_speed_history_passed"
                    ]
                    for slug in slugs
                ),
                "relative_loss_reduction": _mean_std(
                    [
                        models[slug]["tracks"][track][
                            "relative_loss_reduction"
                        ]
                        for slug in slugs
                    ]
                ),
            }

    single_by_seed = {
        int(models[slug]["training_seed"]): slug
        for slug in models
        if models[slug]["group"] == "single_delay_control"
    }
    multi_by_seed = {
        int(models[slug]["training_seed"]): slug
        for slug in models
        if models[slug]["group"] == "multi_delay_target"
    }
    _require(
        tuple(sorted(single_by_seed)) == FORMAL_SEEDS
        and tuple(sorted(multi_by_seed)) == FORMAL_SEEDS,
        "缺少成对训练种子",
    )
    paired = {
        str(seed): {
            track: {
                "multi_slug": multi_by_seed[seed],
                "single_slug": single_by_seed[seed],
                "multi_minus_single_relative_loss_reduction": float(
                    models[multi_by_seed[seed]]["tracks"][track][
                        "relative_loss_reduction"
                    ]
                    - models[single_by_seed[seed]]["tracks"][track][
                        "relative_loss_reduction"
                    ]
                ),
            }
            for track in TRACKS
        }
        for seed in FORMAL_SEEDS
    }
    multi_all_pass = all(
        models[slug]["tracks"][track][
            "matching_speed_history_passed"
        ]
        for slug in multi_by_seed.values()
        for track in TRACKS
    )
    single_all_pass = all(
        models[slug]["tracks"][track][
            "matching_speed_history_passed"
        ]
        for slug in single_by_seed.values()
        for track in TRACKS
    )
    paired_all_positive = all(
        paired[str(seed)][track][
            "multi_minus_single_relative_loss_reduction"
        ]
        > 0.0
        for seed in FORMAL_SEEDS
        for track in TRACKS
    )
    original_slug = str(
        config["models"]["original_reference"][0]["slug"]
    )
    conclusions = {
        "all_multi_delay_models_show_matching_speed_history_response": (
            multi_all_pass
        ),
        "all_single_delay_controls_show_matching_speed_history_response": (
            single_all_pass
        ),
        "original_reference_shows_matching_speed_history_response": all(
            models[original_slug]["tracks"][track][
                "matching_speed_history_passed"
            ]
            for track in TRACKS
        ),
        "multi_minus_single_relative_response_positive_for_all_pairs": (
            paired_all_positive
        ),
        "multi_delay_specific_cross_factor_response_detected": bool(
            multi_all_pass
            and not single_all_pass
            and paired_all_positive
        ),
        "partial_multi_delay_cross_factor_response_detected": bool(
            paired_all_positive
            and any(
                all(
                    models[slug]["tracks"][track][
                        "matching_speed_history_passed"
                    ]
                    for slug in multi_by_seed.values()
                )
                for track in TRACKS
            )
        ),
        "this_is_speed_icl_training_evidence": False,
    }
    result = {
        "schema_version": 1,
        "benchmark": benchmark,
        "status": "completed",
        "identity": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
        },
        "interpretation": (
            "该实验只检查动作延迟训练是否对速度历史产生迁移或混淆；"
            "动作延迟模型没有接受多速度训练，因此不能据此声称速度 ICL。"
        ),
        "result_files": result_files,
        "models": models,
        "model_groups": groups,
        "paired_multi_minus_single": paired,
        "conclusions": conclusions,
    }
    write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "model_groups": groups,
                "conclusions": conclusions,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
