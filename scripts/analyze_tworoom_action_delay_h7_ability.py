#!/usr/bin/env python3
"""Aggregate paired History-7 original-ability retention results."""

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

from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_action_delay_h7_ability_retention_v1.yaml"
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


def _models(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for role, rows in config["models"].items():
        for row in rows:
            slug = str(row["slug"])
            _require(slug not in result, f"模型 slug 重复：{slug}")
            result[slug] = {**row, "role": role}
    return result


def _load_domain(
    root: Path,
    *,
    slug: str,
    domain: str,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    records = {}
    files = []
    identity = None
    planner = config["evaluation"]["planner"]
    for seed in EVAL_SEEDS:
        path = root / slug / domain / f"s{seed}.json"
        _require(path.is_file(), f"缺少能力保持结果：{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload.get("protocol", {})
        checks = {
            "status": payload.get("status") == "passed",
            "history_size": int(protocol.get("history_size", -1)) == 7,
            "action_block": int(protocol.get("action_block", -1)) == 5,
            "eval_seed": int(protocol.get("eval_seed", -1)) == seed,
            "evaluations": int(protocol.get("evaluations", -1)) == 50,
            "budget": int(protocol.get("eval_budget", -1))
            == int(planner["eval_budget_raw_steps"]),
            "horizon": int(protocol.get("horizon", -1))
            == int(planner["horizon_action_blocks"]),
            "receding_horizon": int(
                protocol.get("receding_horizon", -1)
            )
            == int(planner["receding_horizon_action_blocks"]),
            "cem_samples": int(protocol.get("cem_samples", -1))
            == int(planner["cem_samples"]),
            "cem_steps": int(protocol.get("cem_steps", -1))
            == int(planner["cem_steps"]),
            "cem_topk": int(protocol.get("cem_topk", -1))
            == int(planner["cem_topk"]),
            "weights_frozen": payload.get("frozen_weight_audit", {}).get(
                "passed"
            )
            is True,
        }
        _require(all(checks.values()), f"能力保持协议不一致：{path}")
        observed_identity = (
            payload["catalog"]["sha256"],
            payload["checkpoint"]["sha256"],
            payload["normalizer"]["sha256"],
        )
        if identity is None:
            identity = observed_identity
        _require(
            identity == observed_identity,
            f"同一模型/域的身份变化：{path}",
        )
        for row in payload["raw_records"]:
            key = str(row["evaluation_id"])
            _require(key not in records, f"evaluation_id 重复：{key}")
            records[key] = row
        files.append({"path": str(path), "sha256": file_sha256(path)})
    _require(len(records) == 300, f"{slug}/{domain} 不是 300 次")
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
            str(seed): summarize(values)
            for seed, values in sorted(by_seed.items())
        },
        "by_stratum": {
            name: summarize(values)
            for name, values in sorted(by_stratum.items())
        },
    }


def _paired_noninferiority(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    seed: int,
    config: dict[str, Any],
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
        _require(
            all(
                reference[key][field] == candidate[key][field]
                for field in paired_fields
            ),
            f"成对样本元数据不一致：{key}",
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
    protocol = config["evaluation"]["paired_non_inferiority"]
    success = _mean_ci(
        success_delta,
        seed=seed,
        resamples=int(protocol["bootstrap_resamples"]),
        confidence=float(protocol["confidence_level"]),
    )
    distance = _mean_ci(
        distance_delta,
        seed=seed ^ 0xD157A,
        resamples=int(protocol["bootstrap_resamples"]),
        confidence=float(protocol["confidence_level"]),
    )
    strata: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        row = reference[key]
        strata[f"{row['stratum']}|{row['room_relation']}"].append(key)
    collapsed = [
        name
        for name, selected in strata.items()
        if any(reference[key]["success"] for key in selected)
        and not any(candidate[key]["success"] for key in selected)
    ]
    checks = {
        "success_rate_non_inferior": success["ci_lower"]
        >= float(protocol["success_rate_delta_minimum"]),
        "final_distance_non_inferior": distance["ci_upper"]
        <= float(protocol["final_distance_delta_px_maximum"]),
        "no_solvable_stratum_collapse": not collapsed,
    }
    return {
        "evaluations": len(keys),
        "candidate_minus_reference_success_rate": success,
        "candidate_minus_reference_final_distance_px": distance,
        "collapsed_solvable_strata": sorted(collapsed),
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(
        config.get("benchmark")
        == "tworoom_action_delay_history7_ability_retention_v1",
        "不是 History=7 能力保持配置",
    )
    for name, identity in config["source_identity"].items():
        path = resolve_contextworld_path(identity["path"], repo_root=ROOT)
        _require(
            file_sha256(path) == identity["sha256"],
            f"冻结来源哈希变化：{name}",
        )
    root = resolve_contextworld_path(
        config["artifacts"]["results_root"], repo_root=ROOT
    )
    models = _models(config)
    records = {}
    aggregates = {}
    files = {}
    for slug in models:
        aggregates[slug] = {}
        for domain in DOMAINS:
            selected, selected_files = _load_domain(
                root, slug=slug, domain=domain, config=config
            )
            records[(slug, domain)] = selected
            aggregates[slug][domain] = _aggregate(selected)
            files[f"{slug}:{domain}"] = selected_files

    by_role_seed = {
        (str(row["role"]), int(row["training_seed"])): slug
        for slug, row in models.items()
    }
    comparisons = {}
    for seed in (3072, 4096, 5120):
        reference_slug = by_role_seed[("original_only", seed)]
        comparisons[str(seed)] = {}
        for role in ("single_delay_control", "multi_delay_target"):
            candidate_slug = by_role_seed[(role, seed)]
            comparisons[str(seed)][role] = {}
            for domain_index, domain in enumerate(DOMAINS):
                comparisons[str(seed)][role][domain] = (
                    _paired_noninferiority(
                        records[(reference_slug, domain)],
                        records[(candidate_slug, domain)],
                        seed=(
                            int(
                                config["evaluation"][
                                    "paired_non_inferiority"
                                ]["bootstrap_seed"]
                            )
                            ^ (seed << 4)
                            ^ domain_index
                            ^ (0 if role == "single_delay_control" else 1)
                        ),
                        config=config,
                    )
                )
            comparisons[str(seed)][role]["passed"] = all(
                comparisons[str(seed)][role][domain]["passed"]
                for domain in DOMAINS
            )

    group_summary = {}
    for role in config["models"]:
        slugs = [
            slug for slug, row in models.items() if row["role"] == role
        ]
        group_summary[role] = {
            domain: {
                "models": len(slugs),
                "success_rate": _mean_std(
                    [aggregates[slug][domain]["success_rate"] for slug in slugs]
                ),
                "mean_final_distance_px": _mean_std(
                    [
                        aggregates[slug][domain]["mean_final_distance_px"]
                        for slug in slugs
                    ]
                ),
            }
            for domain in DOMAINS
        }
    multi_passed = all(
        comparisons[str(seed)]["multi_delay_target"][
            "original_heldout"
        ]["passed"]
        for seed in (3072, 4096, 5120)
    )
    output = resolve_contextworld_path(
        (
            args.output
            if args.output is not None
            else config["artifacts"]["summary"]
        ),
        repo_root=ROOT,
    )
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "identity": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "result_files": files,
        },
        "aggregates": aggregates,
        "group_summary": group_summary,
        "paired_vs_same_seed_original_only": comparisons,
        "decision": {
            "multi_delay_original_ability_retained": multi_passed,
            "required_multi_seeds": 3,
            "speed5_reported_separately": True,
        },
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "group_summary": group_summary,
                "decision": payload["decision"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
