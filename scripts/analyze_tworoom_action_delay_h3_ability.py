#!/usr/bin/env python3
"""汇总动作延迟训练后的原始 TwoRoom 能力保留结果。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
    / "configs/benchmark/tworoom_action_delay_h3_ability_retention_v1.yaml"
)
DOMAINS = ("original_heldout", "speed5_matched")
EVAL_SEEDS = (42, 43, 44, 45, 46, 47)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int,
    confidence: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": float(values.mean()),
        "ci_lower": float(np.quantile(means, alpha)),
        "ci_upper": float(np.quantile(means, 1.0 - alpha)),
    }


def _load_model_domain(
    root: Path,
    *,
    slug: str,
    domain: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    records = {}
    files = []
    catalog_hash = None
    checkpoint_hash = None
    normalizer_hash = None
    for seed in EVAL_SEEDS:
        path = root / slug / domain / f"s{seed}.json"
        _require(path.is_file(), f"缺少正式结果：{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        _require(payload.get("status") == "passed", f"结果未通过：{path}")
        protocol = payload["protocol"]
        _require(
            int(protocol["eval_seed"]) == seed
            and int(protocol["evaluations"]) == 50
            and int(protocol["eval_budget"]) == 50
            and int(protocol["horizon"]) == 5
            and int(protocol["receding_horizon"]) == 5
            and int(protocol["cem_samples"]) == 300
            and int(protocol["cem_steps"]) == 30
            and int(protocol["cem_topk"]) == 30,
            f"规划配置或 50×6 计数不一致：{path}",
        )
        observed = (
            payload["catalog"]["sha256"],
            payload["checkpoint"]["sha256"],
            payload["normalizer"]["sha256"],
        )
        if catalog_hash is None:
            catalog_hash, checkpoint_hash, normalizer_hash = observed
        _require(
            observed == (catalog_hash, checkpoint_hash, normalizer_hash),
            f"同一模型/数据域的冻结身份发生变化：{path}",
        )
        for row in payload["raw_records"]:
            key = str(row["evaluation_id"])
            _require(key not in records, f"evaluation_id 重复：{key}")
            records[key] = row
        files.append({"path": str(path), "sha256": file_sha256(path)})
    _require(len(records) == 300, f"{slug}/{domain} 不是 50×6=300")
    return records, files


def _aggregate(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(records.values())

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(bool(row["success"]) for row in selected)
        return {
            "evaluations": len(selected),
            "successes": int(successes),
            "success_rate": float(successes / len(selected)),
            "mean_final_distance_px": float(
                np.mean([float(row["final_distance"]) for row in selected])
            ),
        }

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["eval_seed"])].append(row)
        by_stratum[
            f"{row['stratum']}|{row['room_relation']}"
        ].append(row)
    return {
        **summarize(rows),
        "by_seed": {
            str(seed): summarize(selected)
            for seed, selected in sorted(by_seed.items())
        },
        "by_stratum": {
            name: summarize(selected)
            for name, selected in sorted(by_stratum.items())
        },
    }


def _paired_comparison(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    seed: int,
    resamples: int,
    confidence: float,
    success_minimum: float,
    distance_maximum: float,
) -> dict[str, Any]:
    _require(set(reference) == set(candidate), "成对 evaluation_id 不一致")
    keys = sorted(reference)
    paired_fields = (
        "eval_seed",
        "evaluation_index",
        "source_kind",
        "source_path",
        "episode",
        "start_step",
        "goal_offset",
        "cem_group_seed",
        "stratum",
        "room_relation",
        "initial_state",
        "goal_state",
    )
    for key in keys:
        mismatches = [
            field
            for field in paired_fields
            if reference[key][field] != candidate[key][field]
        ]
        _require(
            not mismatches,
            f"成对样本元数据不一致：{key}/{mismatches}",
        )
    success_delta = np.asarray(
        [
            float(bool(candidate[key]["success"]))
            - float(bool(reference[key]["success"]))
            for key in keys
        ],
        dtype=np.float64,
    )
    distance_delta = np.asarray(
        [
            float(candidate[key]["final_distance"])
            - float(reference[key]["final_distance"])
            for key in keys
        ],
        dtype=np.float64,
    )
    success_ci = _mean_ci(
        success_delta,
        seed=seed,
        resamples=resamples,
        confidence=confidence,
    )
    distance_ci = _mean_ci(
        distance_delta,
        seed=seed ^ 0xD157A,
        resamples=resamples,
        confidence=confidence,
    )
    strata: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        row = reference[key]
        strata[f"{row['stratum']}|{row['room_relation']}"].append(key)
    collapsed = []
    for name, selected in strata.items():
        reference_success = sum(
            bool(reference[key]["success"]) for key in selected
        )
        candidate_success = sum(
            bool(candidate[key]["success"]) for key in selected
        )
        if reference_success > 0 and candidate_success == 0:
            collapsed.append(name)
    checks = {
        "success_rate_non_inferior": (
            success_ci["ci_lower"] >= success_minimum
        ),
        "final_distance_non_inferior": (
            distance_ci["ci_upper"] <= distance_maximum
        ),
        "no_solvable_stratum_collapse": not collapsed,
    }
    return {
        "evaluations": len(keys),
        "candidate_minus_reference_success_rate": success_ci,
        "candidate_minus_reference_final_distance_px": distance_ci,
        "margins": {
            "success_rate_minimum": success_minimum,
            "final_distance_px_maximum": distance_maximum,
        },
        "collapsed_solvable_strata": sorted(collapsed),
        "checks": checks,
        "passed": all(checks.values()),
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
    _require(
        config["benchmark"]
        == "tworoom_action_delay_history3_ability_retention_v1",
        "不是动作延迟能力保留配置",
    )
    for name, identity in config["source_identity"].items():
        path = resolve_contextworld_path(identity["path"], repo_root=ROOT)
        _require(
            file_sha256(path) == identity["sha256"],
            f"冻结来源哈希发生变化：{name}",
        )
    root = resolve_contextworld_path(
        (
            args.results_root
            if args.results_root is not None
            else config["artifacts"]["original_ability_retention_root"]
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
    records = {}
    result_files = {}
    aggregates = {}
    for slug in model_rows:
        aggregates[slug] = {}
        for domain in DOMAINS:
            selected, files = _load_model_domain(
                root, slug=slug, domain=domain
            )
            records[(slug, domain)] = selected
            result_files[f"{slug}:{domain}"] = files
            aggregates[slug][domain] = _aggregate(selected)

    protocol = config["evaluation"]["paired_non_inferiority"]
    reference_slug = str(protocol["reference"])
    comparisons = {}
    for slug, row in model_rows.items():
        if slug == reference_slug:
            continue
        comparisons[slug] = {}
        for domain_index, domain in enumerate(DOMAINS):
            comparisons[slug][domain] = _paired_comparison(
                records[(reference_slug, domain)],
                records[(slug, domain)],
                seed=(
                    int(protocol["bootstrap_seed"])
                    ^ (int(row["training_seed"]) << 4)
                    ^ domain_index
                ),
                resamples=int(protocol["bootstrap_resamples"]),
                confidence=float(protocol["confidence_level"]),
                success_minimum=float(
                    protocol[
                        "candidate_minus_reference_success_rate_minimum"
                    ]
                ),
                distance_maximum=float(
                    protocol[
                        "candidate_minus_reference_final_distance_px_maximum"
                    ]
                ),
            )

    group_summary = {}
    for group, rows in config["models"].items():
        group_summary[group] = {}
        for domain in DOMAINS:
            selected = [
                aggregates[str(row["slug"])][domain] for row in rows
            ]
            group_summary[group][domain] = {
                "model_count": len(selected),
                "success_rate": _mean_std(
                    [float(item["success_rate"]) for item in selected]
                ),
                "mean_final_distance_px": _mean_std(
                    [
                        float(item["mean_final_distance_px"])
                        for item in selected
                    ]
                ),
            }
    multi_slugs = [
        str(row["slug"])
        for row in config["models"]["multi_delay_target"]
    ]
    single_slugs = [
        str(row["slug"])
        for row in config["models"]["single_delay_control"]
    ]
    conclusions = {
        "multi_delay_preserves_original_heldout_ability": all(
            comparisons[slug]["original_heldout"]["passed"]
            for slug in multi_slugs
        ),
        "multi_delay_preserves_speed5_base_domain_ability": all(
            comparisons[slug]["speed5_matched"]["passed"]
            for slug in multi_slugs
        ),
        "single_delay_controls_preserve_original_heldout_ability": all(
            comparisons[slug]["original_heldout"]["passed"]
            for slug in single_slugs
        ),
        "speed5_result_is_speed_icl_evidence": False,
    }
    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "identity": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
        },
        "interpretation": {
            "original_heldout": "原始 TwoRoom 规划能力保留的主检验",
            "speed5_matched": (
                "固定速度 5 基础动力学域的次级检验，不是速度 ICL"
            ),
        },
        "result_files": result_files,
        "models": aggregates,
        "model_groups": group_summary,
        "paired_vs_original_reference": comparisons,
        "conclusions": conclusions,
    }
    write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "model_groups": group_summary,
                "conclusions": conclusions,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
