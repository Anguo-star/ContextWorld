#!/usr/bin/env python3
"""Aggregate completed hidden-passage History-3 checkpoint evaluations."""

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

from contextworld.evaluation.hidden_passage_validation import (
    canonical_sha256,
    file_sha256,
    summarize_validation_records,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from scripts.eval_tworoom_hidden_passage_h3_latent import (
    validate_training_provenance,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_validation_v2.yaml"
)


def _public_comparison(
    result: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    by_rule = {}
    for rule, row in summary["by_true_rule"].items():
        overall = row["overall"]
        by_rule[rule] = {
            "same_vs_other_symmetric_contrast": overall[
                "symmetric_contrast"
            ]["same_vs_other_rule_history"],
            "same_vs_no_attempt_symmetric_contrast": overall[
                "symmetric_contrast"
            ]["same_vs_no_crossing_attempt"],
            "strict_win_rate": overall["strict_win_rate"],
            "matching_vs_opposite_history_win_rate": overall[
                "matching_vs_opposite_history_win_rate"
            ],
            "same_history_two_target_accuracy": overall[
                "same_history_two_target_accuracy"
            ],
            "overall_both_paired_effects_positive": row[
                "overall_both_paired_effects_positive"
            ],
        }
    return {
        "model_id": result["model_id"],
        "training_seed": int(result["training_seed"]),
        "model_slug": result["model_slug"],
        "checkpoint_sha256": result["model"]["checkpoint_sha256"],
        "checkpoint_validation_passed": summary["decision"]["passed"],
        "failed_checkpoint_gates": summary["decision"]["failed_checks"],
        "paired_static_query_bootstrap": summary[
            "paired_static_query_bootstrap"
        ],
        "target_latent_separation": summary["target_latent_separation"],
        "two_target_ties": summary["two_target_ties"],
        "by_true_rule": by_rule,
    }


def _attribution_checks(
    passed_by_result: dict[tuple[str, int], bool],
    config: dict[str, Any],
) -> dict[str, bool]:
    gate = config["comparison"].get("attribution_gate", {})
    required_all_pass_model_ids = gate.get("required_all_pass_model_ids")
    required_seed_count = gate.get(
        "pldm_joint_and_fixed_required_training_seeds"
    )
    if required_all_pass_model_ids is None and required_seed_count is not None:
        required_all_pass_model_ids = list(
            config["comparison"]["required_results"]
        )
    required_all_pass_model_ids = required_all_pass_model_ids or []
    if required_all_pass_model_ids:
        expected_count = int(
            gate.get(
                "required_passed_training_seeds",
                required_seed_count,
            )
        )
        checks: dict[str, bool] = {}
        for raw_model_id in required_all_pass_model_ids:
            model_id = str(raw_model_id)
            registered_seeds = tuple(
                map(
                    int,
                    config["comparison"]["required_results"][model_id],
                )
            )
            checks[
                f"{model_id}/has_exact_registered_seed_count"
            ] = len(registered_seeds) == expected_count
            checks[
                f"{model_id}/passes_every_registered_training_seed"
            ] = all(
                passed_by_result[(model_id, seed)]
                for seed in registered_seeds
            )
        return checks

    target_model_id = gate.get("target_model_id")
    if target_model_id is not None:
        target_model_id = str(target_model_id)
        target_seeds = tuple(
            map(
                int,
                config["comparison"]["required_results"][
                    target_model_id
                ],
            )
        )
        expected_count = int(
            gate.get(
                "required_passed_training_seeds",
                len(target_seeds),
            )
        )
        return {
            "target_recipe_has_exact_registered_seed_count": (
                len(target_seeds) == expected_count
            ),
            "target_recipe_passes_every_registered_training_seed": all(
                passed_by_result[(target_model_id, seed)]
                for seed in target_seeds
            ),
        }

    def seeds(model_id: str) -> tuple[int, ...]:
        return tuple(
            map(
                int,
                config["comparison"]["required_results"][model_id],
            )
        )

    return {
        "mixed_rules_passes_all_three_training_seeds": all(
            passed_by_result[("H3_Passage_MixedRules", seed)]
            for seed in seeds("H3_Passage_MixedRules")
        ),
        "original_baseline_fails": not passed_by_result[
            ("H3_Original_LEWM", 3072)
        ],
        "passable_only_family_does_not_pass_all_three_seeds": not all(
            passed_by_result[("H3_Passage_PassableOnly", seed)]
            for seed in seeds("H3_Passage_PassableOnly")
        ),
        "blocked_only_family_does_not_pass_all_three_seeds": not all(
            passed_by_result[("H3_Passage_BlockedOnly", seed)]
            for seed in seeds("H3_Passage_BlockedOnly")
        ),
    }


def aggregate_validation_results(
    *,
    results: list[dict[str, Any]],
    paths: list[Path],
    config: dict[str, Any],
    config_path: Path,
    expected_catalog_sha256: str,
) -> dict[str, Any]:
    """Fail closed on the exact result matrix registered by the config."""

    if len(results) != len(paths):
        raise ValueError("Result/path counts differ")
    required = {
        (str(model_id), int(seed))
        for model_id, seeds in config["comparison"][
            "required_results"
        ].items()
        for seed in seeds
    }
    observed = [
        (str(result.get("model_id")), int(result.get("training_seed", -1)))
        for result in results
    ]
    if len(set(observed)) != len(observed):
        raise ValueError("Duplicate model-id/training-seed result")
    if set(observed) != required:
        raise ValueError(
            "Result matrix is incomplete or contains extras: "
            f"missing={sorted(required - set(observed))}, "
            f"extra={sorted(set(observed) - required)}"
        )

    config_hash = file_sha256(config_path)
    expected_normalizer_hash = str(
        config["adapter"]["normalizer_sha256"]
    )
    expected_stable_commit = str(
        config["stable_worldmodel"]["commit"]
    )
    adapter_ids = {
        "StableWorldModelLeWMAdapter": "stable_worldmodel_lewm_v1",
        "StableWorldModelPLDMAdapter": "stable_worldmodel_pldm_v1",
    }
    expected_adapter_id = config["adapter"].get("id")
    if expected_adapter_id is None:
        implementation = str(
            config["adapter"].get(
                "implementation",
                "StableWorldModelLeWMAdapter",
            )
        )
        expected_adapter_id = adapter_ids.get(implementation)
        if expected_adapter_id is None:
            raise ValueError(
                f"Unknown adapter implementation: {implementation}"
            )
    expected_adapter_id = str(expected_adapter_id)
    expected_adapter_protocol = {
        "history_tokens": 3,
        "action_block_raw_steps": 5,
        "action_dim": 2,
        "future_action_blocks": 5,
        "native_target_encoder": True,
    }
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    checkpoint_hashes = []
    slugs = []
    for path, result, identity in zip(paths, results, observed):
        if result.get("status") != "completed":
            raise ValueError(f"Incomplete result: {path}")
        if result.get("benchmark") != config["benchmark"]:
            raise ValueError(f"Benchmark mismatch: {path}")
        if result["identity"]["config_sha256"] != config_hash:
            raise ValueError(f"Config hash mismatch: {path}")
        if (
            result["identity"]["catalog_sha256"]
            != expected_catalog_sha256
        ):
            raise ValueError(f"Catalog hash mismatch: {path}")
        if (
            result["identity"]["normalizer_sha256"]
            != expected_normalizer_hash
        ):
            raise ValueError(f"Normalizer hash mismatch: {path}")
        if (
            result["model"]["stable_worldmodel_commit"]
            != expected_stable_commit
        ):
            raise ValueError(f"Stable-WorldModel commit mismatch: {path}")
        if result["model"]["adapter_id"] != expected_adapter_id:
            raise ValueError(f"Adapter implementation mismatch: {path}")
        if result["model"]["protocol"] != expected_adapter_protocol:
            raise ValueError(f"Adapter protocol mismatch: {path}")
        provenance = validate_training_provenance(
            config=config,
            model_id=identity[0],
            training_seed=identity[1],
            checkpoint=resolve_contextworld_path(
                result["identity"]["checkpoint"],
                repo_root=ROOT,
            ),
            training_report=resolve_contextworld_path(
                result["identity"]["training_report"],
                repo_root=ROOT,
            ),
        )
        if canonical_sha256(provenance) != canonical_sha256(
            result.get("training_provenance")
        ):
            raise ValueError(
                f"Stored training provenance differs from source files: {path}"
            )
        score_audit = result["score_audit"]
        if (
            not score_audit.get("passed")
            or int(score_audit["model_predictions"]) != 900
            or int(score_audit["target_encodings"]) != 600
            or int(score_audit["records"]) != 1800
            or score_audit["frozen_state_hash_before"]
            != score_audit["frozen_state_hash_after"]
        ):
            raise ValueError(f"Score audit failed: {path}")
        data_audit = result["data_audit"]
        if (
            not data_audit.get("passed")
            or data_audit.get("content_manifest_sha256")
            != data_audit.get("content_manifest_recomputed_sha256")
            or int(data_audit.get("online_environment_calls", -1)) != 0
        ):
            raise ValueError(f"Frozen data audit failed: {path}")

        recomputed = summarize_validation_records(
            result["records"],
            eval_seeds=config["evaluation"]["eval_seeds"],
            unique_queries_per_seed=int(
                config["evaluation"]["unique_queries_per_seed"]
            ),
            gates=config["gates"],
        )
        if canonical_sha256(recomputed) != canonical_sha256(
            result["summary"]
        ):
            raise ValueError(f"Stored summary differs from records: {path}")
        summaries[identity] = recomputed
        checkpoint_hashes.append(
            str(result["model"]["checkpoint_sha256"])
        )
        slugs.append(str(result["model_slug"]))

    if len(set(checkpoint_hashes)) != len(required):
        raise ValueError("Every formal result must use a unique checkpoint")
    if len(set(slugs)) != len(required):
        raise ValueError("Every formal result must use a unique model_slug")

    passed_by_result = {
        identity: bool(summary["decision"]["passed"])
        for identity, summary in summaries.items()
    }
    attribution_checks = _attribution_checks(
        passed_by_result,
        config,
    )
    target_model_id = config["comparison"].get(
        "attribution_gate",
        {},
    ).get("target_model_id")
    attribution_gate = config["comparison"].get(
        "attribution_gate",
        {},
    )
    required_all_pass_model_ids = attribution_gate.get(
        "required_all_pass_model_ids",
        [],
    )
    if (
        not required_all_pass_model_ids
        and attribution_gate.get(
            "pldm_joint_and_fixed_required_training_seeds"
        )
        is not None
    ):
        required_all_pass_model_ids = list(
            config["comparison"]["required_results"]
        )
    if required_all_pass_model_ids:
        attribution_interpretation = (
            "本对照只在配置列出的每个训练配方都以预注册训练种子全数"
            "通过时成立。该门槛验证结果稳定性；训练目标的因果解释还"
            "必须结合只改变训练目标的配对实验。"
        )
    elif target_model_id is None:
        attribution_interpretation = (
            "训练归因只在双规则模型 3/3 通过、原始基线失败，且两个"
            "单规则模型族都不是 3/3 通过时成立。没有尝试穿门的历史"
            "只作辅助基线，不能单独支持归因。"
        )
    else:
        attribution_interpretation = (
            f"{target_model_id} 的三个预注册训练种子必须全部通过。"
            "这一门控判断该固定表示配方是否稳定有效；没有匹配的固定"
            "表示单规则对照时，不把收益进一步归因于双规则数据本身。"
        )
    ordered = sorted(
        zip(paths, results, observed),
        key=lambda row: row[2],
    )
    return {
        "schema_version": 2,
        "benchmark": config["benchmark"],
        "status": "completed",
        "catalog_sha256": expected_catalog_sha256,
        "config_sha256": config_hash,
        "normalizer_sha256": expected_normalizer_hash,
        "stable_worldmodel_commit": expected_stable_commit,
        "models": [
            _public_comparison(result, summaries[identity])
            for _, result, identity in ordered
        ],
        "result_files": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "model_id": result["model_id"],
                "training_seed": int(result["training_seed"]),
                "model_slug": result["model_slug"],
            }
            for path, result, _ in ordered
        ],
        "attribution": {
            "passed": bool(all(attribution_checks.values())),
            "checks": attribution_checks,
            "checkpoint_pass_by_model_and_seed": {
                f"{model_id}/s{seed}": passed_by_result[(model_id, seed)]
                for model_id, seed in sorted(required)
            },
            "interpretation": attribution_interpretation,
        },
        "comparison_contract": {
            "required_result_count": len(required),
            "native_latent_mse_cross_checkpoint_comparison_allowed": False,
            "decision_contract": config.get("gates", {}).get(
                "decision_contract",
                "all_histories_strict_v1",
            ),
            "primary_cross_checkpoint_metrics": (
                [
                    "paired bootstrap confidence bounds",
                    "matching-vs-opposite-history win rate",
                    "matching-history two-target accuracy",
                    "seed and direction consistency",
                ]
                if config.get("gates", {}).get("decision_contract")
                == "informative_history_rule_switch_v2"
                else [
                    "paired bootstrap confidence bounds",
                    "strict win rate",
                    "matching-history two-target accuracy",
                    "seed and direction consistency",
                ]
            ),
            "no_crossing_attempt_role": config.get("metrics", {}).get(
                "no_crossing_attempt_role",
                "auxiliary baseline; never sufficient for attribution",
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate hidden-passage checkpoint results without comparing "
            "native latent MSE across checkpoints"
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = [
        resolve_contextworld_path(path, repo_root=ROOT)
        for path in args.results
    ]
    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    catalog_path = resolve_contextworld_path(
        config["artifacts"]["catalog"],
        repo_root=ROOT,
    )
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)
    expected_catalog_sha256 = file_sha256(catalog_path)

    output = resolve_contextworld_path(args.output, repo_root=ROOT)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    payload = aggregate_validation_results(
        results=results,
        paths=paths,
        config=config,
        config_path=config_path,
        expected_catalog_sha256=expected_catalog_sha256,
    )
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
