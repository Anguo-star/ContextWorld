#!/usr/bin/env python3
"""Run the three-seed fixed-representation hidden-passage confirmation.

The unseen-door Validation is intentionally gated on all three checkpoints
passing the informative-history train-seen 50x6 rule-switch diagnostic.
Existing artifacts are reused only through the same fail-closed validators
as the original ten-model run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.hidden_passage_validation import file_sha256
from contextworld.paths import artifact_root, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from scripts.run_tworoom_hidden_passage_h3_pipeline import (
    Recipe,
    ScoreJob,
    TrainingJob,
    _run_aggregate_stage,
    _run_preflight_stage,
    _run_score_stage,
    _run_training_stage,
)


MODEL_ID = "H3_Passage_MixedRules_FrozenRepresentation"
SEEDS = (3072, 4096, 5120)
TRAINING_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_hidden_passage_h3_fixed_representation_training_v1.yaml"
)
TRAIN_SEEN_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_hidden_passage_h3_fixed_representation_train_seen_eval_v2.yaml"
)
UNSEEN_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_hidden_passage_h3_fixed_representation_validation_v2.yaml"
)
RECIPE = Recipe(
    model_id=MODEL_ID,
    shell_variant="fixed-mixed",
    run_prefix="h3_passage_mixed_rules_fixed_representation_v2",
    score_slug_prefix="h3_passage_fixed_representation",
    display_name="固定原始图像表示的双规则模型",
    training_group="passage_mixed",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置不是映射：{path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_static_contract(
    *,
    training: dict[str, Any],
    train_seen: dict[str, Any],
    unseen: dict[str, Any],
) -> dict[str, Any]:
    protocol = training["training_protocol"]
    _require(
        tuple(map(int, protocol["paired_training_seeds"])) == SEEDS,
        "训练种子不是预注册的 3072/4096/5120",
    )
    _require(
        protocol.get("frozen_model_modules") == ["encoder", "projector"]
        and protocol.get("force_frozen_modules_eval_mode") is True,
        "正式配方没有固定 encoder/projector 及其运行统计",
    )
    _require(
        protocol["group_sampling"]
        == {MODEL_ID: {"passage_mixed": 1.0}},
        "正式配方不是双规则 synthetic-only",
    )
    _require(
        protocol["distributed_execution"]["audit_scheduling"]
        == {
            "policy": "sibling_shared_flock",
            "maximum_concurrency": 8,
            "scope": (
                "per_rank_full_audit_and_fit_start_storage_revalidation"
            ),
            "lock_protocol": (
                "contextworld.hidden_passage_h3."
                "audit_scheduling_lock.v2"
            ),
            "lock_order": "release_shared_then_audit_shared",
            "collective_holds_lock": False,
            "topology_scope": "single_node_8gpu",
            "concurrent_training_runs_per_release": 1,
        },
        "固定表示确认实验没有使用 8 个独立只读校核并行协议",
    )
    _require(
        protocol["distributed_execution"]["rank_cpu_affinity"]
        == {
            "policy": "local_rank_disjoint_contiguous_from_zero",
            "cpus_per_rank": 8,
            "expected_world_size": 8,
            "scope": "full_rank_process",
            "apply_before_stableworldmodel_and_lance_import": True,
        },
        "固定表示确认实验没有在 Lance 导入前限制各 rank 线程池",
    )
    profile = protocol["profiles"]["passage_formal"]
    _require(
        int(profile["optimizer_steps"]) == 1024
        and int(profile["effective_global_batch"]) == 1024
        and int(profile["total_logical_draws"]) == 1_048_576,
        "正式训练预算发生变化",
    )
    expected_results = {MODEL_ID: list(SEEDS)}
    for label, config in (
        ("训练门位置", train_seen),
        ("未见门位置", unseen),
    ):
        _require(
            config["comparison"]["required_results"] == expected_results,
            f"{label}不是精确的三种子矩阵",
        )
        _require(
            len(config["evaluation"]["eval_seeds"]) == 6
            and int(config["evaluation"]["unique_queries_per_seed"]) == 50
            and int(config["evaluation"]["unique_queries"]) == 300,
            f"{label}没有为每个条件独立冻结 50×6",
        )
        _require(
            int(config["evaluation"]["model_predictions_per_checkpoint"])
            == 900
            and int(config["evaluation"]["loss_records_per_checkpoint"])
            == 1800,
            f"{label}不是每模型 900 次预测/1,800 条 loss",
        )
        referenced = resolve_contextworld_path(
            config["training_provenance"]["passage_formal"][
                "training_benchmark_config"
            ],
            repo_root=ROOT,
        )
        _require(
            referenced == TRAINING_CONFIG.resolve(),
            f"{label}引用了错误的训练配置",
        )
        catalog = resolve_contextworld_path(
            config["artifacts"]["catalog"],
            repo_root=ROOT,
        )
        _require(catalog.is_file(), f"{label} catalog 不存在：{catalog}")
        payload = json.loads(catalog.read_text(encoding="utf-8"))
        _require(
            payload.get("benchmark") == config["benchmark"],
            f"{label} catalog 与配置身份不一致",
        )
    _require(
        training["evaluation_gate"]["stage_2_unseen_door_validation"][
            "locked_until_stage_1_passes"
        ]
        is True,
        "未见门位置 Validation 没有被训练门位置 3/3 门控",
    )
    return {
        "passed": True,
        "model": MODEL_ID,
        "training_seeds": list(SEEDS),
        "training_budget": profile,
        "frozen_modules": ["encoder", "projector"],
        "each_eval_condition": "50x6=300",
        "decision_contract": (
            "informative_history_rule_switch_v2"
        ),
    }


def _training_jobs(training: dict[str, Any]) -> tuple[TrainingJob, ...]:
    root = resolve_contextworld_path(
        training["artifacts"]["training_root"],
        repo_root=ROOT,
    )
    reports = resolve_contextworld_path(
        training["artifacts"]["reports"],
        repo_root=ROOT,
    )
    steps = int(
        training["training_protocol"]["profiles"]["passage_formal"][
            "optimizer_steps"
        ]
    )
    jobs = []
    for seed in SEEDS:
        run_name = f"{RECIPE.run_prefix}_passage_formal_s{seed}"
        preflight_name = f"{RECIPE.run_prefix}_passage_pilot_s{seed}"
        run_dir = root / "checkpoints" / run_name
        jobs.append(
            TrainingJob(
                recipe=RECIPE,
                seed=seed,
                run_name=run_name,
                run_dir=run_dir,
                report=reports / f"{run_name}.json",
                checkpoint=run_dir / f"weights_final_step_{steps}.pt",
                preflight_run_name=preflight_name,
                preflight_report=(
                    reports / f"{preflight_name}_preflight.json"
                ),
            )
        )
    return tuple(jobs)


def _score_jobs(
    *,
    config: dict[str, Any],
    training_jobs: tuple[TrainingJob, ...],
) -> tuple[ScoreJob, ...]:
    root = resolve_contextworld_path(
        config["artifacts"]["output_root"],
        repo_root=ROOT,
    )
    jobs = []
    for training_job in training_jobs:
        slug = f"{RECIPE.score_slug_prefix}_s{training_job.seed}"
        jobs.append(
            ScoreJob(
                model_id=MODEL_ID,
                seed=training_job.seed,
                model_slug=slug,
                display_name=RECIPE.display_name,
                checkpoint=training_job.checkpoint,
                training_report=training_job.report,
                output=root / "results" / f"{slug}.json",
            )
        )
    return tuple(jobs)


def _gate_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = payload.get("attribution", {}).get("checks", {})
    passed = payload.get("attribution", {}).get("passed")
    _require(
        isinstance(checks, dict) and len(checks) == 2,
        "三种子 aggregate 没有完整门控检查",
    )
    return {
        "passed": passed is True and all(checks.values()),
        "checks": checks,
        "checkpoint_pass_by_model_and_seed": payload["attribution"][
            "checkpoint_pass_by_model_and_seed"
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.artifact_root is not None:
        os.environ["CONTEXTWORLD_ARTIFACT_ROOT"] = str(
            args.artifact_root.expanduser().resolve()
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    training = _load_yaml(TRAINING_CONFIG)
    train_seen = _load_yaml(TRAIN_SEEN_CONFIG)
    unseen = _load_yaml(UNSEEN_CONFIG)
    static = _validate_static_contract(
        training=training,
        train_seen=train_seen,
        unseen=unseen,
    )
    jobs = _training_jobs(training)
    plan = {
        "status": "dry_run" if args.dry_run else "planned",
        "static_contract": static,
        "training_jobs": [
            {
                "seed": job.seed,
                "run_name": job.run_name,
                "checkpoint": str(job.checkpoint),
            }
            for job in jobs
        ],
        "gate": (
            "三个模型必须全部通过训练门位置 50x6，才运行未见门位置"
        ),
    }
    if args.dry_run:
        return plan

    stages: dict[str, Any] = {}
    # The data split seed, catalog bytes, loader plan, model boundary, and
    # frozen-module declaration are identical across training seeds. One
    # recipe-level preflight is therefore sufficient; every formal DDP run
    # still performs an independent full logical audit on all eight ranks.
    stages["preflight"] = _run_preflight_stage(
        jobs[:1],
        validation_config=train_seen,
        training_config=training,
        python=args.python,
    )
    stages["preflight"]["scope"] = "recipe_level_seed_independent"
    stages["preflight"]["formal_runs_still_audit_all_ranks"] = True
    stages["training"] = _run_training_stage(
        jobs,
        training_config_path=TRAINING_CONFIG.resolve(),
        validation_config=train_seen,
        python=args.python,
    )

    train_seen_jobs = _score_jobs(
        config=train_seen,
        training_jobs=jobs,
    )
    stages["train_seen_score"] = _run_score_stage(
        train_seen_jobs,
        validation_config_path=TRAIN_SEEN_CONFIG.resolve(),
        validation_config=train_seen,
        python=args.python,
        device=args.device,
    )
    train_seen_aggregate = resolve_contextworld_path(
        train_seen["artifacts"]["output_root"],
        repo_root=ROOT,
    ) / "aggregate_rule_switch_v2.json"
    stages["train_seen_aggregate"] = _run_aggregate_stage(
        train_seen_jobs,
        output=train_seen_aggregate,
        validation_config_path=TRAIN_SEEN_CONFIG.resolve(),
        validation_config=train_seen,
        python=args.python,
    )
    train_seen_gate = _gate_payload(train_seen_aggregate)

    unseen_aggregate = None
    unseen_gate = None
    if train_seen_gate["passed"]:
        unseen_jobs = _score_jobs(
            config=unseen,
            training_jobs=jobs,
        )
        stages["unseen_score"] = _run_score_stage(
            unseen_jobs,
            validation_config_path=UNSEEN_CONFIG.resolve(),
            validation_config=unseen,
            python=args.python,
            device=args.device,
        )
        unseen_aggregate = resolve_contextworld_path(
            unseen["artifacts"]["output_root"],
            repo_root=ROOT,
        ) / "aggregate_rule_switch_v2.json"
        stages["unseen_aggregate"] = _run_aggregate_stage(
            unseen_jobs,
            output=unseen_aggregate,
            validation_config_path=UNSEEN_CONFIG.resolve(),
            validation_config=unseen,
            python=args.python,
        )
        unseen_gate = _gate_payload(unseen_aggregate)

    report = {
        "schema_version": 1,
        "status": (
            "completed"
            if train_seen_gate["passed"]
            else "stopped_at_train_seen_gate"
        ),
        "static_contract": static,
        "config_identity": {
            "training": {
                "path": str(TRAINING_CONFIG),
                "sha256": file_sha256(TRAINING_CONFIG),
            },
            "train_seen": {
                "path": str(TRAIN_SEEN_CONFIG),
                "sha256": file_sha256(TRAIN_SEEN_CONFIG),
            },
            "unseen": {
                "path": str(UNSEEN_CONFIG),
                "sha256": file_sha256(UNSEEN_CONFIG),
            },
        },
        "stages": stages,
        "train_seen_gate": train_seen_gate,
        "unseen_validation_executed": train_seen_gate["passed"],
        "unseen_gate": unseen_gate,
        "aggregates": {
            "train_seen": str(train_seen_aggregate),
            "unseen": (
                str(unseen_aggregate)
                if unseen_aggregate is not None
                else None
            ),
        },
    }
    report_path = (
        artifact_root(ROOT)
        / "evaluation/history3/"
        "hidden_passage_fixed_representation_confirmation_v2.json"
    )
    write_json(report_path, report)
    return {**report, "report": str(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "训练固定原始图像表示的三个双规则模型；训练门位置 3/3 "
            "通过后才运行未见门位置 Validation。"
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cuda-visible-devices",
        default="0,1,2,3,4,5,6,7",
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
