#!/usr/bin/env python3
"""Run the fail-closed History-3 hidden-passage comparison.

This file is only an orchestration layer.  Data construction, training,
Validation scoring, and aggregation remain owned by their dedicated entry
points.  Existing outputs are reused only after their complete identity has
been validated; an invalid or ambiguous output stops the pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.hidden_passage_validation import (
    canonical_sha256,
    file_sha256,
    summarize_validation_records,
)
from contextworld.paths import (
    artifact_root,
    resolve_contextworld_path,
)
from contextworld.training.tworoom_data import (
    hidden_passage_training_release_root,
)
from scripts.analyze_tworoom_hidden_passage_h3 import (
    aggregate_validation_results,
)
from scripts.eval_tworoom_hidden_passage_h3_latent import (
    TRAINING_RUN_EXCLUSIVITY_CONTRACT,
    _audit_training_run_exclusivity,
    validate_training_provenance,
)


DEFAULT_VALIDATION_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_validation_v2.yaml"
)
PASSAGE_INTERNAL_ENVIRONMENT = (
    "CONTEXTWORLD_H3_RANK0_ATTESTATION_V1",
    "CONTEXTWORLD_H3_RANK0_ATTESTATION_V2",
    "CONTEXTWORLD_H3_RANK0_SECRET",
    "CONTEXTWORLD_H3_RANK0_ISSUER",
)
DEFAULT_TRAINING_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_training_v1.yaml"
)
TRAIN_ENTRY = ROOT / "scripts/run_h3_hidden_passage_train.sh"
SCORE_ENTRY = ROOT / "scripts/eval_tworoom_hidden_passage_h3_latent.py"
AGGREGATE_ENTRY = ROOT / "scripts/analyze_tworoom_hidden_passage_h3.py"

TRAINING_SEEDS = (3072, 4096, 5120)
ORIGINAL_MODEL_ID = "H3_Original_LEWM"
ORIGINAL_TRAINING_SEED = 3072
AUDIT_SCHEDULING_CONTRACT = {
    "policy": "sibling_exclusive_flock",
    "maximum_concurrency": 1,
    "scope": "per_rank_full_audit_and_fit_start_storage_revalidation",
    "lock_protocol": "contextworld.hidden_passage_h3.audit_scheduling_lock.v1",
    "lock_order": "release_shared_then_audit_exclusive",
    "collective_holds_lock": False,
    "topology_scope": "single_node_8gpu",
    "concurrent_training_runs_per_release": 1,
}


@dataclass(frozen=True)
class Recipe:
    model_id: str
    shell_variant: str
    run_prefix: str
    score_slug_prefix: str
    display_name: str
    training_group: str


RECIPES = (
    Recipe(
        model_id="H3_Passage_PassableOnly",
        shell_variant="passable",
        run_prefix="h3_passage_passable_only",
        score_slug_prefix="h3_original_init_plus_synth_passable",
        display_name=(
            "原始 H3 初始化 + 仅“门可通过”合成数据继续训练"
        ),
        training_group="passage_passable",
    ),
    Recipe(
        model_id="H3_Passage_BlockedOnly",
        shell_variant="blocked",
        run_prefix="h3_passage_blocked_only",
        score_slug_prefix="h3_original_init_plus_synth_blocked",
        display_name=(
            "原始 H3 初始化 + 仅“门不可通过”合成数据继续训练"
        ),
        training_group="passage_blocked",
    ),
    Recipe(
        model_id="H3_Passage_MixedRules",
        shell_variant="mixed",
        run_prefix="h3_passage_mixed_rules",
        score_slug_prefix="h3_original_init_plus_synth_mixed_rules",
        display_name=(
            "原始 H3 初始化 + “门可通过/不可通过”合成数据继续训练"
        ),
        training_group="passage_mixed",
    ),
)


@dataclass(frozen=True)
class TrainingJob:
    recipe: Recipe
    seed: int
    run_name: str
    run_dir: Path
    report: Path
    checkpoint: Path
    preflight_run_name: str
    preflight_report: Path

    @property
    def label(self) -> str:
        return f"{self.recipe.display_name}（训练 seed={self.seed}）"


@dataclass(frozen=True)
class ScoreJob:
    model_id: str
    seed: int
    model_slug: str
    display_name: str
    checkpoint: Path
    training_report: Path
    output: Path

    @property
    def label(self) -> str:
        return f"{self.display_name}（训练 seed={self.seed}）"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置不是映射：{path}")
    return payload


def _required_result_identities(
    validation_config: dict[str, Any],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (str(model_id), int(seed))
        for model_id, seeds in validation_config["comparison"][
            "required_results"
        ].items()
        for seed in seeds
    )


def validate_static_contract(
    *,
    validation_config: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate the frozen 9-train/10-score matrix without touching artifacts."""

    expected_trained = {
        (recipe.model_id, seed)
        for recipe in RECIPES
        for seed in TRAINING_SEEDS
    }
    expected_results = {
        (ORIGINAL_MODEL_ID, ORIGINAL_TRAINING_SEED),
        *expected_trained,
    }
    observed_results = set(
        _required_result_identities(validation_config)
    )
    _require(
        observed_results == expected_results,
        "Validation v2 结果矩阵不是原始模型 + 3 配方 × 3 seed",
    )
    _require(
        len(observed_results) == 10,
        "Validation v2 必须精确包含 10 个结果",
    )

    protocol = training_config["training_protocol"]
    _require(
        tuple(map(int, protocol["paired_training_seeds"]))
        == TRAINING_SEEDS,
        "训练配置中的三个 seed 与 runner 不一致",
    )
    _require(
        int(protocol["history_tokens"]) == 3
        and int(protocol["raw_steps_per_action_block"]) == 5,
        "训练配置不是 History=3、每个动作块 5 个原始步",
    )
    _require(
        protocol.get("synthetic_only") is True,
        "三个继续训练配方必须是 synthetic-only",
    )
    group_sampling = protocol["group_sampling"]
    for recipe in RECIPES:
        _require(
            group_sampling.get(recipe.model_id)
            == {recipe.training_group: 1.0},
            f"{recipe.model_id} 的训练数据配方不唯一",
        )

    profile = protocol["profiles"]["passage_formal"]
    provenance = validation_config["training_provenance"][
        "passage_formal"
    ]
    _require(
        int(profile["optimizer_steps"])
        == int(provenance["optimizer_steps"])
        == 1024,
        "正式训练步数没有在训练/Validation 配置中一致冻结",
    )
    _require(
        int(profile["total_logical_draws"])
        == int(provenance["total_logical_draws"])
        == 1_048_576,
        "正式训练逻辑样本数没有一致冻结",
    )
    _require(
        int(provenance["topology"]["devices"]) == 8,
        "History=3 hidden-passage 正式训练必须使用冻结的 8 GPU 拓扑",
    )
    _require(
        validation_config["evaluation"]["model_predictions_per_checkpoint"]
        == 900
        and validation_config["evaluation"]["loss_records_per_checkpoint"]
        == 1800,
        "每个 checkpoint 的 Validation 计数不是 900 次预测/1800 条 loss",
    )
    return {
        "passed": True,
        "history_tokens": 3,
        "formal_training_jobs": len(expected_trained),
        "validation_results": len(expected_results),
        "training_seeds": list(TRAINING_SEEDS),
    }


def validate_validation_artifacts(
    *,
    validation_config_path: Path,
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    """Bind execution to the successfully built, frozen Validation v2 files."""

    output_root = resolve_contextworld_path(
        validation_config["artifacts"]["output_root"],
        repo_root=ROOT,
    )
    catalog = resolve_contextworld_path(
        validation_config["artifacts"]["catalog"],
        repo_root=ROOT,
    )
    exclusion = resolve_contextworld_path(
        validation_config["artifacts"]["training_exclusion_manifest"],
        repo_root=ROOT,
    )
    report_path = output_root / "build_report.json"
    for path in (catalog, exclusion, report_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Validation v2 冻结文件不存在：{path}"
            )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        report.get("status") == "passed"
        and report.get("benchmark") == validation_config["benchmark"]
        and all(report.get("checks", {}).values()),
        f"Validation v2 build report 未通过：{report_path}",
    )
    identity = report.get("identity", {})
    _require(
        identity.get("config_sha256")
        == file_sha256(validation_config_path)
        and identity.get("catalog_sha256") == file_sha256(catalog)
        and identity.get("training_exclusion_manifest_sha256")
        == file_sha256(exclusion)
        and identity.get("stable_worldmodel_commit")
        == validation_config["stable_worldmodel"]["commit"],
        "Validation v2 build report 与当前配置/catalog/exclusion 不一致",
    )
    catalog_payload = json.loads(catalog.read_text(encoding="utf-8"))
    exclusion_payload = json.loads(exclusion.read_text(encoding="utf-8"))
    content_sha256 = str(report["content_manifest_sha256"])
    _require(
        catalog_payload.get("benchmark") == validation_config["benchmark"]
        and catalog_payload.get("content_manifest_sha256")
        == content_sha256
        and exclusion_payload.get("benchmark")
        == validation_config["benchmark"]
        and exclusion_payload.get("content_manifest_sha256")
        == content_sha256,
        "Validation v2 的 catalog、排除清单与 build report 内容身份不一致",
    )
    return {
        "passed": True,
        "build_report": str(report_path),
        "catalog": str(catalog),
        "catalog_sha256": identity["catalog_sha256"],
        "training_exclusion_manifest": str(exclusion),
        "training_exclusion_manifest_sha256": identity[
            "training_exclusion_manifest_sha256"
        ],
        "content_manifest_sha256": content_sha256,
    }


def training_jobs(
    training_config: dict[str, Any],
) -> tuple[TrainingJob, ...]:
    training_root = resolve_contextworld_path(
        training_config["artifacts"]["training_root"],
        repo_root=ROOT,
    )
    report_root = resolve_contextworld_path(
        training_config["artifacts"]["reports"],
        repo_root=ROOT,
    )
    optimizer_steps = int(
        training_config["training_protocol"]["profiles"][
            "passage_formal"
        ]["optimizer_steps"]
    )
    jobs = []
    for recipe in RECIPES:
        for seed in TRAINING_SEEDS:
            run_name = f"{recipe.run_prefix}_passage_formal_s{seed}"
            preflight_run_name = (
                f"{recipe.run_prefix}_passage_pilot_s{seed}"
            )
            jobs.append(
                TrainingJob(
                    recipe=recipe,
                    seed=seed,
                    run_name=run_name,
                    run_dir=training_root / "checkpoints" / run_name,
                    report=report_root / f"{run_name}.json",
                    checkpoint=(
                        training_root
                        / "checkpoints"
                        / run_name
                        / f"weights_final_step_{optimizer_steps}.pt"
                    ),
                    preflight_run_name=preflight_run_name,
                    preflight_report=(
                        report_root
                        / f"{preflight_run_name}_preflight.json"
                    ),
                )
            )
    return tuple(jobs)


def score_jobs(
    *,
    validation_config: dict[str, Any],
    training_config: dict[str, Any],
    train_jobs: Iterable[TrainingJob],
) -> tuple[ScoreJob, ...]:
    validation_root = resolve_contextworld_path(
        validation_config["artifacts"]["output_root"],
        repo_root=ROOT,
    )
    results_root = validation_root / "results"
    original = training_config["comparison"]["unchanged_baseline"]
    original_contract = validation_config["training_provenance"][
        "original_baseline"
    ]
    jobs = [
        ScoreJob(
            model_id=ORIGINAL_MODEL_ID,
            seed=ORIGINAL_TRAINING_SEED,
            model_slug="h3_original_lewm_s3072",
            display_name=(
                "原始 LeWM（原始 TwoRoom 数据训练，不做合成数据继续训练）"
            ),
            checkpoint=resolve_contextworld_path(
                original["checkpoint"],
                repo_root=ROOT,
            ),
            training_report=resolve_contextworld_path(
                original_contract["training_report"],
                repo_root=ROOT,
            ),
            output=results_root / "h3_original_lewm_s3072.json",
        )
    ]
    by_model = {recipe.model_id: recipe for recipe in RECIPES}
    for job in train_jobs:
        recipe = by_model[job.recipe.model_id]
        slug = f"{recipe.score_slug_prefix}_s{job.seed}"
        jobs.append(
            ScoreJob(
                model_id=recipe.model_id,
                seed=job.seed,
                model_slug=slug,
                display_name=recipe.display_name,
                checkpoint=job.checkpoint,
                training_report=job.report,
                output=results_root / f"{slug}.json",
            )
        )
    return tuple(jobs)


def _training_environment(
    *,
    job: TrainingJob,
    python: str,
) -> dict[str, str]:
    root = artifact_root(ROOT)
    return {
        "PYTHON_BIN": python,
        "TRAINING_SEED": str(job.seed),
        "RUN_NAME": job.run_name,
        "CONTEXTWORLD_ARTIFACT_ROOT": str(root),
        "OUTPUT_ROOT": str(root / "training/runs"),
        "REPORT_DIR": str(root / "training/reports"),
        "LOG_DIR": str(root / "training/logs"),
    }


def preflight_command(
    job: TrainingJob,
    *,
    python: str,
) -> tuple[list[str], dict[str, str]]:
    environment = _training_environment(job=job, python=python)
    environment["RUN_NAME"] = job.preflight_run_name
    return (
        [
            "bash",
            str(TRAIN_ENTRY),
            job.recipe.shell_variant,
            "preflight",
        ],
        environment,
    )


def training_command(
    job: TrainingJob,
    *,
    python: str,
    resume: bool,
) -> tuple[list[str], dict[str, str]]:
    return (
        [
            "bash",
            str(TRAIN_ENTRY),
            job.recipe.shell_variant,
            "formal-resume" if resume else "formal",
        ],
        _training_environment(job=job, python=python),
    )


def score_command(
    job: ScoreJob,
    *,
    validation_config_path: Path,
    validation_config: dict[str, Any],
    python: str,
    device: str,
) -> list[str]:
    catalog = resolve_contextworld_path(
        validation_config["artifacts"]["catalog"],
        repo_root=ROOT,
    )
    normalizer = resolve_contextworld_path(
        validation_config["adapter"]["normalizer"],
        repo_root=ROOT,
    )
    return [
        python,
        str(SCORE_ENTRY),
        "--config",
        str(validation_config_path),
        "--catalog",
        str(catalog),
        "--checkpoint",
        str(job.checkpoint),
        "--training-report",
        str(job.training_report),
        "--model-id",
        job.model_id,
        "--training-seed",
        str(job.seed),
        "--model-slug",
        job.model_slug,
        "--normalizer",
        str(normalizer),
        "--output",
        str(job.output),
        "--device",
        device,
    ]


def aggregate_command(
    jobs: Iterable[ScoreJob],
    *,
    validation_config_path: Path,
    output: Path,
    python: str,
) -> list[str]:
    return [
        python,
        str(AGGREGATE_ENTRY),
        "--config",
        str(validation_config_path),
        "--results",
        *(str(job.output) for job in jobs),
        "--output",
        str(output),
    ]


def _command_record(
    *,
    stage: str,
    label: str,
    command: list[str],
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "label": label,
        "command": command,
        "environment": environment or {},
    }


def dry_run_plan(
    *,
    stage: str,
    validation_config_path: Path,
    validation_config: dict[str, Any],
    training_config: dict[str, Any],
    python: str,
    device: str,
) -> dict[str, Any]:
    trains = training_jobs(training_config)
    scores = score_jobs(
        validation_config=validation_config,
        training_config=training_config,
        train_jobs=trains,
    )
    validation_root = resolve_contextworld_path(
        validation_config["artifacts"]["output_root"],
        repo_root=ROOT,
    )
    aggregate_output = validation_root / "aggregate.json"
    records: list[dict[str, Any]] = []
    selected = (
        ("preflight", "train", "score", "aggregate")
        if stage == "all"
        else (stage,)
    )
    if "preflight" in selected:
        for job in trains:
            command, environment = preflight_command(job, python=python)
            records.append(
                _command_record(
                    stage="preflight",
                    label=job.label,
                    command=command,
                    environment=environment,
                )
            )
    if "train" in selected:
        for job in trains:
            command, environment = training_command(
                job,
                python=python,
                resume=False,
            )
            records.append(
                _command_record(
                    stage="train",
                    label=job.label,
                    command=command,
                    environment=environment,
                )
            )
    if "score" in selected:
        for job in scores:
            records.append(
                _command_record(
                    stage="score",
                    label=job.label,
                    command=score_command(
                        job,
                        validation_config_path=validation_config_path,
                        validation_config=validation_config,
                        python=python,
                        device=device,
                    ),
                )
            )
    if "aggregate" in selected:
        records.append(
            _command_record(
                stage="aggregate",
                label="精确汇总原始模型 + 9 个继续训练模型",
                command=aggregate_command(
                    scores,
                    validation_config_path=validation_config_path,
                    output=aggregate_output,
                    python=python,
                ),
            )
        )
    return {
        "status": "dry_run",
        "stage": stage,
        "formal_training_jobs": len(trains),
        "validation_score_jobs": len(scores),
        "commands": records,
        "note": (
            "正式执行时，只有身份校验通过的已有结果才会明确标记为"
            "“复用-已验证”；不完整训练仅在发现合法 last.ckpt 后改用"
            " formal-resume。"
        ),
    }


def _same_path(observed: str | Path, expected: Path) -> bool:
    if not isinstance(observed, (str, Path)) or not str(observed):
        return False
    return Path(observed).expanduser().resolve() == expected.resolve()


def validate_existing_preflight(
    job: TrainingJob,
    *,
    validation_config: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    path = job.preflight_report
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("passed") is True
        and payload.get("run_kind") == "training_data_plan_preflight",
        f"已有 preflight 不是通过状态：{path}",
    )
    _require(
        payload.get("model_id") == job.recipe.model_id
        and payload.get("run_name") == job.preflight_run_name,
        f"已有 preflight 的模型或 run_name 不匹配：{path}",
    )
    _require(
        payload.get("stable_worldmodel", {}).get("commit")
        == validation_config["stable_worldmodel"]["commit"],
        f"已有 preflight 的 Stable-WorldModel commit 不匹配：{path}",
    )
    plan = payload.get("training_plan", {})
    _require(
        plan.get("profile") == "passage_pilot"
        and int(plan.get("training_seed", -1)) == job.seed
        and int(plan.get("data_split_seed", -1)) == 3072,
        f"已有 preflight 的 profile/seed 不匹配：{path}",
    )
    distributed = payload.get("distributed_execution_contract", {})
    expected_audit_scheduling = training_config["training_protocol"][
        "distributed_execution"
    ]["audit_scheduling"]
    _require(
        distributed.get("rendezvous_timeout_seconds_declared") == 7200
        and distributed.get("rendezvous_timeout_scope")
        == "passage_multi_gpu_only"
        and distributed.get("rendezvous_timeout_seconds_applied") == 7200
        and distributed.get("rendezvous_timeout_override_applied") is True
        and distributed.get("transport_configuration")
        == "framework_defaults_with_frozen_rendezvous_timeout"
        and distributed.get("audit_scheduling")
        == expected_audit_scheduling,
        f"已有 preflight 没有冻结 7200 秒 passage DDP 等待：{path}",
    )
    _require(
        distributed.get("training_run_exclusivity")
        == TRAINING_RUN_EXCLUSIVITY_CONTRACT,
        f"已有 preflight 没有冻结 root-run 排他锁协议：{path}",
    )
    data = payload.get("data", {})
    _require(
        data.get("group_weights") == {job.recipe.training_group: 1.0}
        and set(data.get("groups", {})) == {job.recipe.training_group},
        f"已有 preflight 的合成数据配方不匹配：{path}",
    )
    scope = data.get("training_data_scope", {})
    _require(
        scope.get("synthetic_only") is True
        and scope.get("original_samples_included") is False,
        f"已有 preflight 不是 synthetic-only：{path}",
    )
    boundary = payload.get("model_contract", {})
    _require(
        boundary.get("model_boundary_keys") == ["pixels", "action"]
        and boundary.get("privileged_fields_at_model_boundary") == [],
        f"已有 preflight 的模型输入边界不合格：{path}",
    )
    expected_catalog = resolve_contextworld_path(
        training_config["data"]["catalogs"][
            job.recipe.training_group
        ],
        repo_root=ROOT,
    )
    group = data["groups"][job.recipe.training_group]
    split_audit = group.get("catalog_split_audit", {})
    quality = training_config["data_quality"]["groups"][
        job.recipe.training_group
    ]
    expected_hashes = {
        "catalog": quality.get("required_catalog_sha256"),
        "manifest": quality.get("required_manifest_sha256"),
        "synthesis_report": quality.get(
            "required_synthesis_report_sha256"
        ),
    }
    _require(
        _same_path(group.get("catalog", ""), expected_catalog)
        and split_audit.get("required_artifact_hashes")
        == expected_hashes
        and all(
            isinstance(value, str) and len(value) == 64
            for value in expected_hashes.values()
        ),
        f"已有 preflight 的正式训练 catalog/hashes 已过期：{path}",
    )
    exclusion_spec = training_config["data"][
        "training_exclusion_manifest"
    ]
    exclusion_audit = data.get("training_exclusion_audit", {})
    _require(
        exclusion_audit.get("passed") is True
        and exclusion_audit.get("sha256") == exclusion_spec["sha256"]
        and exclusion_audit.get("content_sha256")
        == exclusion_spec["content_sha256"],
        f"已有 preflight 没有绑定当前 Validation v2 排除清单：{path}",
    )
    expected_init = training_config["training_protocol"][
        "initialization_checkpoint"
    ]
    observed_init = payload.get("initialization_checkpoint", {})
    _require(
        observed_init.get("configured") is True
        and observed_init.get("applied") is False
        and observed_init.get("reason") == "preflight_hash_audit_only"
        and observed_init.get("sha256") == expected_init["sha256"]
        and observed_init.get("config_sha256")
        == expected_init["config_sha256"]
        and observed_init.get("role") == expected_init["role"],
        f"已有 preflight 没有绑定原始 H3 初始化 checkpoint：{path}",
    )
    expected_frozen = list(
        training_config.get("training_protocol", {}).get(
            "frozen_model_modules",
            [],
        )
    )
    observed_frozen = payload.get("frozen_model_modules", {})
    if expected_frozen:
        _require(
            expected_frozen == ["encoder", "projector"]
            and observed_frozen.get("configured") is True
            and observed_frozen.get("applied") is False
            and observed_frozen.get("reason")
            == "preflight_does_not_instantiate_model"
            and observed_frozen.get("modules") == expected_frozen
            and observed_frozen.get("force_eval_mode") is True,
            f"已有 preflight 没有绑定固定 encoder/projector 配方：{path}",
        )
    else:
        _require(
            observed_frozen.get("configured") in (None, False),
            f"已有 preflight 意外启用了固定表示：{path}",
        )
    return payload


def validate_existing_training(
    job: TrainingJob,
    *,
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    if not job.report.is_file():
        raise FileNotFoundError(job.report)
    payload = json.loads(job.report.read_text(encoding="utf-8"))
    _require(
        payload.get("run_name") == job.run_name,
        f"正式训练报告的 run_name 不匹配：{job.report}",
    )
    _require(
        _same_path(
            payload.get("artifacts", {}).get("pretrained", ""),
            job.checkpoint,
        ),
        f"正式训练报告指向了意外 checkpoint：{job.report}",
    )
    provenance = validate_training_provenance(
        config=validation_config,
        model_id=job.recipe.model_id,
        training_seed=job.seed,
        checkpoint=job.checkpoint,
        training_report=job.report,
    )
    return provenance


def validate_partial_training_checkpoint(
    job: TrainingJob,
    *,
    training_config_path: Path,
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate both the run identity and complete trainer state before resume."""

    resume_checkpoint = job.run_dir / "last.ckpt"
    checkpoint_config_path = job.run_dir / "config.json"
    if not resume_checkpoint.is_file():
        raise FileNotFoundError(resume_checkpoint)
    if not checkpoint_config_path.is_file():
        raise FileNotFoundError(
            "last.ckpt 没有同目录 config.json，无法证明断点属于当前配方："
            f"{checkpoint_config_path}"
        )
    checkpoint_config = json.loads(
        checkpoint_config_path.read_text(encoding="utf-8")
    )
    _require(
        checkpoint_config.get("output_model_name") == job.run_name
        and checkpoint_config.get("subdir") == job.run_name,
        f"断点 config 的 run_name 不匹配：{checkpoint_config_path}",
    )
    _require(
        int(checkpoint_config.get("seed", -1)) == job.seed,
        f"断点 config 的训练 seed 不匹配：{checkpoint_config_path}",
    )
    protocol = {
        "history_size": int(
            checkpoint_config.get("wm", {}).get("history_size", -1)
        ),
        "num_preds": int(
            checkpoint_config.get("wm", {}).get("num_preds", -1)
        ),
        "frameskip": int(
            checkpoint_config.get("data", {})
            .get("dataset", {})
            .get("frameskip", -1)
        ),
        "num_steps": int(
            checkpoint_config.get("data", {})
            .get("dataset", {})
            .get("num_steps", -1)
        ),
        "action_encoder_input_dim": int(
            checkpoint_config.get("model", {})
            .get("action_encoder", {})
            .get("input_dim", -1)
        ),
    }
    _require(
        protocol
        == {
            "history_size": 3,
            "num_preds": 1,
            "frameskip": 5,
            "num_steps": 4,
            "action_encoder_input_dim": 10,
        },
        f"断点 config 不是冻结的 History=3 LeWM：{protocol}",
    )
    context = checkpoint_config.get("contextworld_benchmark", {})
    _require(
        context.get("model_id") == job.recipe.model_id
        and context.get("profile") == "passage_formal"
        and _same_path(
            context.get("benchmark_config", ""),
            training_config_path,
        ),
        f"断点 config 的模型/正式 profile/benchmark 不匹配："
        f"{checkpoint_config_path}",
    )
    distributed = context.get("distributed_execution_contract", {})
    training_config = yaml.safe_load(
        training_config_path.read_text(encoding="utf-8")
    )
    expected_audit_scheduling = training_config["training_protocol"][
        "distributed_execution"
    ]["audit_scheduling"]
    _require(
        distributed.get("rendezvous_timeout_seconds_declared") == 7200
        and distributed.get("rendezvous_timeout_scope")
        == "passage_multi_gpu_only"
        and distributed.get("rendezvous_timeout_seconds_applied") == 7200
        and distributed.get("rendezvous_timeout_override_applied") is True
        and distributed.get("transport_configuration")
        == "framework_defaults_with_frozen_rendezvous_timeout"
        and distributed.get("audit_scheduling")
        == expected_audit_scheduling,
        "断点 config 没有绑定当前 7200 秒 passage DDP 等待配置："
        f"{checkpoint_config_path}",
    )
    _require(
        distributed.get("training_run_exclusivity")
        == TRAINING_RUN_EXCLUSIVITY_CONTRACT,
        "断点 config 没有绑定当前 root-run 排他锁协议，旧协议 "
        f"last.ckpt 禁止恢复：{checkpoint_config_path}",
    )
    plan = context.get("training_plan", {})
    expected_topology = validation_config["training_provenance"][
        "passage_formal"
    ]["topology"]
    _require(
        int(plan.get("training_seed", -1)) == job.seed
        and int(plan.get("data_split_seed", -1)) == 3072
        and int(plan.get("optimizer_steps_total", -1)) == 1024
        and int(plan.get("optimizer_steps_per_epoch", -1)) == 256
        and int(plan.get("logical_epochs", -1)) == 4
        and int(plan.get("total_global_sample_draws", -1)) == 1_048_576
        and int(plan.get("devices", -1))
        == int(expected_topology["devices"])
        and plan.get("execution_topology")
        == expected_topology["execution_topology"],
        f"断点 config 的训练预算/seed/8 GPU 拓扑不匹配："
        f"{checkpoint_config_path}",
    )
    _require(
        plan.get("data_quality_gates", {})
        .get(job.recipe.training_group, {})
        .get("passed")
        is True,
        f"断点 config 未记录通过的正式数据门禁：{checkpoint_config_path}",
    )
    data = context.get("data", {})
    scope = data.get("training_data_scope", {})
    _require(
        data.get("group_weights")
        == {job.recipe.training_group: 1.0}
        and set(data.get("groups", {})) == {job.recipe.training_group}
        and scope.get("synthetic_only") is True
        and scope.get("original_samples_included") is False,
        f"断点 config 的 synthetic-only 配方不匹配："
        f"{checkpoint_config_path}",
    )
    release_root = hidden_passage_training_release_root(
        training_config_path,
        repo_root=ROOT,
        model_id=job.recipe.model_id,
    )
    _require(
        release_root is not None,
        f"断点 config 无法解析 sealed release：{checkpoint_config_path}",
    )
    root_lock_audit = _audit_training_run_exclusivity(
        report=None,
        checkpoint_data=data,
        release_root=release_root,
        verify_lock_available=False,
    )
    expected_init = validation_config["training_provenance"][
        "passage_formal"
    ]["initialization_checkpoint"]
    observed_init = context.get("initialization_checkpoint", {})
    _require(
        observed_init.get("sha256") == expected_init["sha256"]
        and observed_init.get("config_sha256")
        == expected_init["config_sha256"]
        and observed_init.get("role") == expected_init["role"],
        f"断点 config 的原始 H3 初始化身份不匹配："
        f"{checkpoint_config_path}",
    )
    boundary = context.get("model_input_boundary", {})
    _require(
        boundary.get("model_boundary_keys") == ["pixels", "action"]
        and boundary.get("privileged_fields_at_model_boundary") == [],
        f"断点 config 的模型输入边界不合格：{checkpoint_config_path}",
    )

    # Reuse the training entry point's exact optimizer/scheduler/RNG audit.
    from scripts.train_tworoom_step1 import (
        _full_state_checkpoint_metadata,
    )

    full_state = _full_state_checkpoint_metadata(
        resume_checkpoint,
        expected_optimizer_steps=1024,
        require_incomplete=True,
        expected_world_size=int(expected_topology["world_size"]),
        optimizer_steps_per_epoch=256,
    )
    return {
        "passed": True,
        "checkpoint_config": str(checkpoint_config_path),
        "checkpoint_config_sha256": file_sha256(
            checkpoint_config_path
        ),
        "resume_checkpoint": str(resume_checkpoint),
        "resume_checkpoint_sha256": file_sha256(resume_checkpoint),
        "full_state": full_state,
        "training_run_exclusivity": root_lock_audit,
    }


def _expected_adapter_protocol() -> dict[str, Any]:
    return {
        "history_tokens": 3,
        "action_block_raw_steps": 5,
        "action_dim": 2,
        "future_action_blocks": 5,
        "native_target_encoder": True,
    }


def validate_existing_score(
    job: ScoreJob,
    *,
    validation_config_path: Path,
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    if not job.output.is_file():
        raise FileNotFoundError(job.output)
    result = json.loads(job.output.read_text(encoding="utf-8"))
    _require(
        result.get("status") == "completed"
        and result.get("benchmark") == validation_config["benchmark"],
        f"已有评分不是完成的 Validation v2 结果：{job.output}",
    )
    _require(
        result.get("model_id") == job.model_id
        and int(result.get("training_seed", -1)) == job.seed
        and result.get("model_slug") == job.model_slug,
        f"已有评分的模型/seed/slug 不匹配：{job.output}",
    )
    identity = result.get("identity", {})
    catalog = resolve_contextworld_path(
        validation_config["artifacts"]["catalog"],
        repo_root=ROOT,
    )
    normalizer = resolve_contextworld_path(
        validation_config["adapter"]["normalizer"],
        repo_root=ROOT,
    )
    expected_paths = {
        "catalog": catalog,
        "checkpoint": job.checkpoint,
        "training_report": job.training_report,
        "normalizer": normalizer,
    }
    for name, expected in expected_paths.items():
        _require(
            _same_path(identity.get(name, ""), expected),
            f"已有评分的 {name} 路径不匹配：{job.output}",
        )
    _require(
        identity.get("config_sha256")
        == file_sha256(validation_config_path)
        and identity.get("catalog_sha256") == file_sha256(catalog)
        and identity.get("normalizer_sha256") == file_sha256(normalizer),
        f"已有评分的配置/数据/normalizer 哈希不匹配：{job.output}",
    )
    provenance = validate_training_provenance(
        config=validation_config,
        model_id=job.model_id,
        training_seed=job.seed,
        checkpoint=job.checkpoint,
        training_report=job.training_report,
    )
    _require(
        canonical_sha256(provenance)
        == canonical_sha256(result.get("training_provenance")),
        f"已有评分保存的训练来源与当前文件不一致：{job.output}",
    )
    model = result.get("model", {})
    _require(
        model.get("checkpoint_sha256") == file_sha256(job.checkpoint)
        and model.get("stable_worldmodel_commit")
        == validation_config["stable_worldmodel"]["commit"]
        and model.get("adapter_id") == "stable_worldmodel_lewm_v1"
        and model.get("protocol") == _expected_adapter_protocol(),
        f"已有评分的 checkpoint/adapter 身份不匹配：{job.output}",
    )
    score_audit = result.get("score_audit", {})
    _require(
        score_audit.get("passed") is True
        and int(score_audit.get("model_predictions", -1)) == 900
        and int(score_audit.get("target_encodings", -1)) == 600
        and int(score_audit.get("records", -1)) == 1800
        and score_audit.get("frozen_state_hash_before")
        == score_audit.get("frozen_state_hash_after"),
        f"已有评分的 900 预测/1800 loss 审计失败：{job.output}",
    )
    data_audit = result.get("data_audit", {})
    _require(
        data_audit.get("passed") is True
        and data_audit.get("content_manifest_sha256")
        == data_audit.get("content_manifest_recomputed_sha256")
        and int(data_audit.get("online_environment_calls", -1)) == 0,
        f"已有评分的离线数据审计失败：{job.output}",
    )
    recomputed = summarize_validation_records(
        result.get("records", []),
        eval_seeds=validation_config["evaluation"]["eval_seeds"],
        unique_queries_per_seed=int(
            validation_config["evaluation"]["unique_queries_per_seed"]
        ),
        gates=validation_config["gates"],
    )
    _require(
        canonical_sha256(recomputed)
        == canonical_sha256(result.get("summary")),
        f"已有评分的摘要不能由 1800 条 loss 记录重算得到：{job.output}",
    )
    return result


def validate_existing_aggregate(
    *,
    output: Path,
    score_jobs_: tuple[ScoreJob, ...],
    validation_config_path: Path,
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    if not output.is_file():
        raise FileNotFoundError(output)
    paths = [job.output for job in score_jobs_]
    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    catalog = resolve_contextworld_path(
        validation_config["artifacts"]["catalog"],
        repo_root=ROOT,
    )
    expected = aggregate_validation_results(
        results=results,
        paths=paths,
        config=validation_config,
        config_path=validation_config_path,
        expected_catalog_sha256=file_sha256(catalog),
    )
    observed = json.loads(output.read_text(encoding="utf-8"))
    _require(
        canonical_sha256(observed) == canonical_sha256(expected),
        f"已有 aggregate 与当前 10 个结果不一致：{output}",
    )
    return observed


def _run_command(
    *,
    label: str,
    command: list[str],
    environment: dict[str, str] | None = None,
) -> None:
    inherited_internal = sorted(
        name
        for name in PASSAGE_INTERNAL_ENVIRONMENT
        if name in os.environ or name in (environment or {})
    )
    if inherited_internal:
        raise RuntimeError(
            "Pipeline refuses internal hidden-passage launch state: "
            f"{inherited_internal}"
        )
    print(f"[运行] {label}", flush=True)
    print(
        json.dumps(
            {
                "command": command,
                "environment": environment or {},
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    child_environment = os.environ.copy()
    child_environment.update(environment or {})
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=child_environment,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"命令失败（returncode={completed.returncode}）：{label}"
        )


def _run_preflight_stage(
    jobs: tuple[TrainingJob, ...],
    *,
    validation_config: dict[str, Any],
    training_config: dict[str, Any],
    python: str,
) -> dict[str, int]:
    reused = 0
    executed = 0
    for job in jobs:
        if job.preflight_report.exists():
            validate_existing_preflight(
                job,
                validation_config=validation_config,
                training_config=training_config,
            )
            reused += 1
            print(f"[复用-已验证] preflight：{job.label}", flush=True)
            continue
        command, environment = preflight_command(job, python=python)
        _run_command(
            label=f"preflight：{job.label}",
            command=command,
            environment=environment,
        )
        validate_existing_preflight(
            job,
            validation_config=validation_config,
            training_config=training_config,
        )
        executed += 1
    return {"jobs": len(jobs), "reused": reused, "executed": executed}


def _run_training_stage(
    jobs: tuple[TrainingJob, ...],
    *,
    training_config_path: Path,
    validation_config: dict[str, Any],
    python: str,
) -> dict[str, int]:
    reused = 0
    resumed = 0
    fresh = 0
    for job in jobs:
        if job.report.exists():
            validate_existing_training(
                job,
                validation_config=validation_config,
            )
            reused += 1
            print(f"[复用-已验证] 正式训练：{job.label}", flush=True)
            continue

        run_contents = (
            tuple(job.run_dir.iterdir()) if job.run_dir.is_dir() else ()
        )
        resume_checkpoint = job.run_dir / "last.ckpt"
        if run_contents and not resume_checkpoint.is_file():
            raise FileExistsError(
                "训练目录已有文件但没有可验证的 last.ckpt；拒绝覆盖或"
                f"猜测恢复状态：{job.run_dir}"
            )
        if job.checkpoint.exists():
            raise FileExistsError(
                "发现最终 checkpoint 但缺少正式训练报告；拒绝静默复用："
                f"{job.checkpoint}"
            )
        use_resume = resume_checkpoint.is_file()
        if use_resume:
            resume_audit = validate_partial_training_checkpoint(
                job,
                training_config_path=training_config_path,
                validation_config=validation_config,
            )
            print(
                "[断点-已验证] "
                f"{job.label}，global_step="
                f"{resume_audit['full_state']['global_step']}",
                flush=True,
            )
        command, environment = training_command(
            job,
            python=python,
            resume=use_resume,
        )
        _run_command(
            label=(
                f"{'断点恢复' if use_resume else '全新'}正式训练："
                f"{job.label}"
            ),
            command=command,
            environment=environment,
        )
        validate_existing_training(
            job,
            validation_config=validation_config,
        )
        if use_resume:
            resumed += 1
        else:
            fresh += 1
    return {
        "jobs": len(jobs),
        "reused": reused,
        "resumed": resumed,
        "fresh": fresh,
    }


def _run_score_stage(
    jobs: tuple[ScoreJob, ...],
    *,
    validation_config_path: Path,
    validation_config: dict[str, Any],
    python: str,
    device: str,
) -> dict[str, int]:
    reused = 0
    executed = 0
    for job in jobs:
        if job.output.exists():
            validate_existing_score(
                job,
                validation_config_path=validation_config_path,
                validation_config=validation_config,
            )
            reused += 1
            print(f"[复用-已验证] Validation v2：{job.label}", flush=True)
            continue
        # This rejects pilot/smoke/incomplete reports before a GPU is opened.
        validate_training_provenance(
            config=validation_config,
            model_id=job.model_id,
            training_seed=job.seed,
            checkpoint=job.checkpoint,
            training_report=job.training_report,
        )
        job.output.parent.mkdir(parents=True, exist_ok=True)
        _run_command(
            label=f"Validation v2：{job.label}",
            command=score_command(
                job,
                validation_config_path=validation_config_path,
                validation_config=validation_config,
                python=python,
                device=device,
            ),
        )
        validate_existing_score(
            job,
            validation_config_path=validation_config_path,
            validation_config=validation_config,
        )
        executed += 1
    return {"jobs": len(jobs), "reused": reused, "executed": executed}


def _run_aggregate_stage(
    jobs: tuple[ScoreJob, ...],
    *,
    output: Path,
    validation_config_path: Path,
    validation_config: dict[str, Any],
    python: str,
) -> dict[str, int]:
    for job in jobs:
        validate_existing_score(
            job,
            validation_config_path=validation_config_path,
            validation_config=validation_config,
        )
    if output.exists():
        validate_existing_aggregate(
            output=output,
            score_jobs_=jobs,
            validation_config_path=validation_config_path,
            validation_config=validation_config,
        )
        print(
            f"[复用-已验证] 精确 10 结果 aggregate：{output}",
            flush=True,
        )
        return {"jobs": 1, "reused": 1, "executed": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        label="精确汇总原始模型 + 9 个继续训练模型",
        command=aggregate_command(
            jobs,
            validation_config_path=validation_config_path,
            output=output,
            python=python,
        ),
    )
    validate_existing_aggregate(
        output=output,
        score_jobs_=jobs,
        validation_config_path=validation_config_path,
        validation_config=validation_config,
    )
    return {"jobs": 1, "reused": 0, "executed": 1}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.artifact_root is not None:
        os.environ["CONTEXTWORLD_ARTIFACT_ROOT"] = str(
            args.artifact_root.expanduser().resolve()
        )
    validation_config_path = args.validation_config.resolve()
    training_config_path = args.training_config.resolve()
    validation_config = _load_yaml(validation_config_path)
    training_config = _load_yaml(training_config_path)
    referenced_training_config = resolve_contextworld_path(
        validation_config["training_provenance"]["passage_formal"][
            "training_benchmark_config"
        ],
        repo_root=ROOT,
    )
    _require(
        training_config_path == referenced_training_config,
        "runner 的训练配置必须与 Validation v2 明确引用的训练配置相同；"
        f"runner={training_config_path}, "
        f"Validation={referenced_training_config}",
    )
    static = validate_static_contract(
        validation_config=validation_config,
        training_config=training_config,
    )
    if args.dry_run:
        return {
            "static_contract": static,
            **dry_run_plan(
                stage=args.stage,
                validation_config_path=validation_config_path,
                validation_config=validation_config,
                training_config=training_config,
                python=args.python,
                device=args.device,
            ),
        }

    validation_artifacts = validate_validation_artifacts(
        validation_config_path=validation_config_path,
        validation_config=validation_config,
    )
    trains = training_jobs(training_config)
    scores = score_jobs(
        validation_config=validation_config,
        training_config=training_config,
        train_jobs=trains,
    )
    validation_root = resolve_contextworld_path(
        validation_config["artifacts"]["output_root"],
        repo_root=ROOT,
    )
    aggregate_output = validation_root / "aggregate.json"
    stages = (
        ("preflight", "train", "score", "aggregate")
        if args.stage == "all"
        else (args.stage,)
    )
    results: dict[str, Any] = {}
    for stage in stages:
        if stage == "preflight":
            results[stage] = _run_preflight_stage(
                trains,
                validation_config=validation_config,
                training_config=training_config,
                python=args.python,
            )
        elif stage == "train":
            results[stage] = _run_training_stage(
                trains,
                training_config_path=training_config_path,
                validation_config=validation_config,
                python=args.python,
            )
        elif stage == "score":
            results[stage] = _run_score_stage(
                scores,
                validation_config_path=validation_config_path,
                validation_config=validation_config,
                python=args.python,
                device=args.device,
            )
        elif stage == "aggregate":
            results[stage] = _run_aggregate_stage(
                scores,
                output=aggregate_output,
                validation_config_path=validation_config_path,
                validation_config=validation_config,
                python=args.python,
            )
        else:  # pragma: no cover - argparse owns this boundary.
            raise AssertionError(stage)
    return {
        "status": "completed",
        "stage": args.stage,
        "static_contract": static,
        "validation_artifacts": validation_artifacts,
        "stages": results,
        "aggregate": str(aggregate_output),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "History=3 门能否通过：9 个正式训练 + 原始模型在内的 10 个"
            " Validation v2 结果 + 精确汇总。任何已有输出都先验明身份。"
        )
    )
    parser.add_argument(
        "--stage",
        choices=("preflight", "train", "score", "aggregate", "all"),
        required=True,
    )
    parser.add_argument(
        "--validation-config",
        type=Path,
        default=DEFAULT_VALIDATION_CONFIG,
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_TRAINING_CONFIG,
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "可选的 ContextWorld artifact 根目录；不传时使用项目统一路径。"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印冻结矩阵与完整命令，不读取或写入实验结果。",
    )
    return parser.parse_args(argv)


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
