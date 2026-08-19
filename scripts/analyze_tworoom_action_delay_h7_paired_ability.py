#!/usr/bin/env python3
"""Audit and aggregate an Action Delay CEM ability-retention matrix."""

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
    / "configs/benchmark/"
    "tworoom_action_delay_h7_paired_ability_retention_v1.yaml"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.size > 0, "不能汇总空指标")
    return {
        "mean": float(array.mean()),
        "sample_std": (
            float(array.std(ddof=1)) if array.size > 1 else 0.0
        ),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
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
    result: dict[str, dict[str, Any]] = {}
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
    model: dict[str, Any],
    domain: str,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    records: dict[str, dict[str, Any]] = {}
    files: list[dict[str, str]] = []
    planner = config["evaluation"]["planner"]
    domain_config = config["evaluation"]["domains"][domain]
    expected_normalizer = config["source_identity"]["normalizer"]["sha256"]
    expected_commit = config["stable_worldmodel"]["commit"]
    for seed in config["evaluation"]["eval_seeds"]:
        path = root / slug / domain / f"s{int(seed)}.json"
        _require(path.is_file(), f"缺少 CEM 结果：{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        protocol = payload.get("protocol", {})
        checks = {
            "status": payload.get("status") == "passed",
            "history_size": int(protocol.get("history_size", -1))
            == int(model["history_size"]),
            "action_block": int(protocol.get("action_block", -1)) == 5,
            "eval_seed": int(protocol.get("eval_seed", -1)) == int(seed),
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
            "checkpoint": payload.get("checkpoint", {}).get("sha256")
            == model["checkpoint_sha256"],
            "catalog": payload.get("catalog", {}).get("sha256")
            == domain_config["catalog_sha256"],
            "normalizer": payload.get("normalizer", {}).get("sha256")
            == expected_normalizer,
            "stable_worldmodel": payload.get("stable_worldmodel", {}).get(
                "commit"
            )
            == expected_commit,
        }
        _require(
            all(checks.values()),
            f"CEM 协议或身份不一致：{path}; "
            f"{[name for name, passed in checks.items() if not passed]}",
        )
        for row in payload["raw_records"]:
            key = str(row["evaluation_id"])
            _require(key not in records, f"evaluation_id 重复：{key}")
            records[key] = row
        files.append({"path": str(path), "sha256": file_sha256(path)})
    _require(
        len(records)
        == int(config["evaluation"]["evaluations_per_model_per_domain"])
        == 300,
        f"{slug}/{domain} 不是 300 次独立规划",
    )
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
                np.mean(
                    [float(row["final_distance"]) for row in selected]
                )
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
        str(config.get("benchmark", "")).startswith(
            "tworoom_action_delay_history7_"
        ),
        "不是 History=7 Action Delay 能力保持配置",
    )
    for name, identity in config["source_identity"].items():
        path = resolve_contextworld_path(identity["path"], repo_root=ROOT)
        _require(
            path.is_file() and file_sha256(path) == identity["sha256"],
            f"冻结来源身份变化：{name}",
        )
    runner_report = resolve_contextworld_path(
        config["artifacts"]["runner_report"], repo_root=ROOT
    )
    runner = json.loads(runner_report.read_text(encoding="utf-8"))
    _require(
        runner.get("status") == "passed"
        and int(runner.get("jobs", -1))
        == int(config["evaluation"]["expected_jobs"])
        and int(runner.get("independent_planning_evaluations", -1))
        == int(
            config["evaluation"][
                "expected_independent_planning_evaluations"
            ]
        ),
        "runner report 未通过完整性审计",
    )

    root = resolve_contextworld_path(
        config["artifacts"]["results_root"], repo_root=ROOT
    )
    models = _models(config)
    records: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    aggregates: dict[str, dict[str, Any]] = {}
    result_files: dict[str, list[dict[str, str]]] = {}
    domains = tuple(config["evaluation"]["domains"])
    for slug, model in models.items():
        aggregates[slug] = {}
        for domain in domains:
            selected, selected_files = _load_domain(
                root,
                slug=slug,
                model=model,
                domain=domain,
                config=config,
            )
            records[(slug, domain)] = selected
            aggregates[slug][domain] = _aggregate(selected)
            result_files[f"{slug}:{domain}"] = selected_files

    reference_slug = str(
        config["evaluation"]["paired_non_inferiority"][
            "primary_reference"
        ]
    )
    base_seed = int(
        config["evaluation"]["paired_non_inferiority"]["bootstrap_seed"]
    )
    vs_h3_reference: dict[str, dict[str, Any]] = {}
    candidate_slugs = [
        slug for slug in models if slug != reference_slug
    ]
    for model_index, slug in enumerate(candidate_slugs):
        vs_h3_reference[slug] = {}
        for domain_index, domain in enumerate(domains):
            vs_h3_reference[slug][domain] = _paired_noninferiority(
                records[(reference_slug, domain)],
                records[(slug, domain)],
                seed=base_seed ^ (model_index << 5) ^ domain_index,
                config=config,
            )

    by_role_seed = {
        (str(model["role"]), int(model["training_seed"])): slug
        for slug, model in models.items()
    }
    vs_same_seed_h7_original: dict[str, dict[str, Any]] = {}
    if "original_h7_control" in config["models"]:
        candidate_roles = [
            role
            for role in config["models"]
            if role not in {"original_h3_reference", "original_h7_control"}
        ]
        for family_index, role in enumerate(candidate_roles):
            vs_same_seed_h7_original[role] = {}
            for seed in (3072, 4096, 5120):
                reference = by_role_seed[("original_h7_control", seed)]
                candidate = by_role_seed[(role, seed)]
                vs_same_seed_h7_original[role][str(seed)] = {}
                for domain_index, domain in enumerate(domains):
                    vs_same_seed_h7_original[role][str(seed)][domain] = (
                        _paired_noninferiority(
                            records[(reference, domain)],
                            records[(candidate, domain)],
                            seed=(
                                base_seed
                                ^ (family_index << 12)
                                ^ (seed << 2)
                                ^ domain_index
                            ),
                            config=config,
                        )
                    )

    group_summary: dict[str, dict[str, Any]] = {}
    for role in config["models"]:
        slugs = [
            slug
            for slug, model in models.items()
            if model["role"] == role
        ]
        group_summary[role] = {
            domain: {
                "models": len(slugs),
                "success_rate": _stats(
                    [
                        aggregates[slug][domain]["success_rate"]
                        for slug in slugs
                    ]
                ),
                "mean_final_distance_px": _stats(
                    [
                        aggregates[slug][domain][
                            "mean_final_distance_px"
                        ]
                        for slug in slugs
                    ]
                ),
            }
            for domain in domains
        }

    target_role = str(
        config["evaluation"]
        .get("decision", {})
        .get("target_role", "paired_pldm")
    )
    target_slugs = [
        slug
        for slug, model in models.items()
        if model["role"] == target_role
    ]
    target_primary_seed_pass = {
        slug: bool(vs_h3_reference[slug]["original_heldout"]["passed"])
        for slug in target_slugs
    }
    target_retained = bool(target_slugs) and all(
        target_primary_seed_pass.values()
    )
    output = resolve_contextworld_path(
        args.output if args.output else config["artifacts"]["summary"],
        repo_root=ROOT,
    )
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "question": config["question"],
        "identity": {
            "config": {
                "path": str(config_path),
                "sha256": file_sha256(config_path),
            },
            "runner_report": {
                "path": str(runner_report),
                "sha256": file_sha256(runner_report),
            },
            "result_files": result_files,
            "stable_worldmodel_commit": config["stable_worldmodel"][
                "commit"
            ],
        },
        "protocol": {
            "models": len(models),
            "domains": list(domains),
            "eval_seeds": config["evaluation"]["eval_seeds"],
            "evaluations_per_model_per_domain": 300,
            "independent_planning_evaluations": int(
                config["evaluation"][
                    "expected_independent_planning_evaluations"
                ]
            ),
            "planner": config["evaluation"]["planner"],
            "paired_non_inferiority": config["evaluation"][
                "paired_non_inferiority"
            ],
        },
        "models": models,
        "aggregates": aggregates,
        "group_summary": group_summary,
        "paired_vs_original_h3_reference": vs_h3_reference,
        "paired_vs_same_seed_original_h7": vs_same_seed_h7_original,
        "decision": {
            "primary_reference": reference_slug,
            "primary_domain": "original_heldout",
            "target_role": target_role,
            "target_seed_pass": target_primary_seed_pass,
            "target_seeds_passed": int(
                sum(target_primary_seed_pass.values())
            ),
            "required_target_seeds": len(target_slugs),
            "target_original_cem_ability_retained": target_retained,
            "speed5_matched_is_secondary_not_a_gate": True,
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
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
