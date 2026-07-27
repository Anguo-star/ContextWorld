#!/usr/bin/env python3
"""Score original/tiny-overfit H3 checkpoints on exact training examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.adapters import StableWorldModelLeWMAdapter
from contextworld.evaluation.hidden_passage_validation import (
    HISTORY_CONDITIONS,
    file_sha256,
    load_validation_assets,
    paired_effect_rows,
    score_validation_assets,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from scripts.eval_tworoom_hidden_passage_h3_latent import (
    _checkpoint_protocol,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_tiny_overfit_eval_v1.yaml"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _audit_checkpoint(
    *,
    config: dict[str, Any],
    role: str,
    checkpoint: Path,
    training_report: Path,
) -> dict[str, Any]:
    contract = config["checkpoint_contract"][role]
    checkpoint = checkpoint.resolve()
    training_report = training_report.resolve()
    checkpoint_config_path = checkpoint.parent / "config.json"
    for path in (checkpoint, checkpoint_config_path, training_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha256 = file_sha256(checkpoint)
    checkpoint_config_sha256 = file_sha256(checkpoint_config_path)
    training_report_sha256 = file_sha256(training_report)
    checkpoint_config = json.loads(
        checkpoint_config_path.read_text(encoding="utf-8")
    )
    report = json.loads(training_report.read_text(encoding="utf-8"))
    protocol = _checkpoint_protocol(checkpoint_config)
    _require(
        protocol
        == {
            "history_size": 3,
            "num_preds": 1,
            "frameskip": 5,
            "num_steps": 4,
            "action_encoder_input_dim": 10,
        },
        f"Checkpoint is not the frozen History=3 LeWM protocol: {protocol}",
    )
    context = checkpoint_config.get("contextworld_benchmark", {})
    _require(
        str(context.get("model_id")) == str(contract["model_id"]),
        "Checkpoint model identity differs from diagnostic contract",
    )
    _require(
        report.get("passed") is True
        and report.get("save_load_exact") is True
        and report.get("training", {}).get("training_complete") is True,
        "Training report is not a completed exact-save run",
    )
    artifacts = report.get("artifacts", {})
    _require(
        Path(str(artifacts.get("pretrained", ""))).resolve() == checkpoint
        and artifacts.get("pretrained_sha256") == checkpoint_sha256
        and Path(
            str(artifacts.get("pretrained_config", ""))
        ).resolve()
        == checkpoint_config_path
        and artifacts.get("pretrained_config_sha256")
        == checkpoint_config_sha256,
        "Training report/checkpoint artifact identity mismatch",
    )

    audit_kind = str(contract.get("audit_kind", role))
    if audit_kind in {"original", "frozen_original"}:
        _require(
            checkpoint
            == resolve_contextworld_path(
                contract["checkpoint"], repo_root=ROOT
            )
            and checkpoint_sha256 == contract["checkpoint_sha256"]
            and checkpoint_config_sha256
            == contract["checkpoint_config_sha256"]
            and training_report
            == resolve_contextworld_path(
                contract["training_report"], repo_root=ROOT
            )
            and training_report_sha256
            == contract["training_report_sha256"],
            "Original H3 identity differs from the frozen contract",
        )
    elif audit_kind in {"tiny_overfit", "tiny_training"}:
        training_config = resolve_contextworld_path(
            contract["training_config"], repo_root=ROOT
        )
        expected_training_config_sha256 = contract.get(
            "training_config_sha256"
        )
        if expected_training_config_sha256 is not None:
            _require(
                file_sha256(training_config)
                == str(expected_training_config_sha256),
                "Tiny diagnostic training config hash mismatch",
            )
        plan = context.get("training_plan", {})
        data = context.get("data", {})
        training = report["training"]
        _require(
            context.get("benchmark_config") == str(training_config)
            and report.get("run_name") == contract["run_name"]
            and int(plan.get("training_seed", -1))
            == int(contract["training_seed"])
            and int(plan.get("optimizer_steps_total", -1))
            == int(contract["expected_optimizer_steps"])
            and int(training.get("global_step", -1))
            == int(contract["expected_optimizer_steps"])
            and int(plan.get("total_global_sample_draws", -1))
            == int(contract["expected_total_logical_draws"])
            and data.get("group_weights")
            == contract["expected_group_weights"]
            and data.get("training_data_scope", {}).get(
                "original_samples_included"
            )
            is False,
            "Tiny-overfit training plan differs from the frozen diagnostic",
        )
        source_init = report.get("initialization_checkpoint", {})
        _require(
            source_init.get("configured") is True
            and source_init.get("applied") is True
            and source_init.get("state_exact") is True,
            "Tiny-overfit model was not initialized exactly from original H3",
        )
        for field, observed in (
            ("checkpoint_sha256", checkpoint_sha256),
            ("checkpoint_config_sha256", checkpoint_config_sha256),
            ("training_report_sha256", training_report_sha256),
        ):
            expected = contract.get(field)
            if expected is not None:
                _require(
                    observed == str(expected),
                    f"Frozen {role} {field} mismatch",
                )
        expected_frozen = list(
            contract.get("expected_frozen_modules", [])
        )
        configured_frozen = context.get("frozen_model_modules", {})
        report_frozen = report.get("frozen_model_modules", {})
        _require(
            list(configured_frozen.get("modules", []))
            == expected_frozen
            and list(report_frozen.get("modules", []))
            == expected_frozen,
            "Frozen-module declaration differs from diagnostic contract",
        )
        if expected_frozen:
            _require(
                configured_frozen.get("configured") is True
                and configured_frozen.get("force_eval_mode") is True
                and report_frozen.get("applied") is True
                and report_frozen.get("passed") is True
                and all(
                    report_frozen.get("state_unchanged", {}).get(name)
                    is True
                    for name in expected_frozen
                ),
                "Frozen representation was not held exact during training",
            )
    else:
        raise ValueError(
            f"Unknown diagnostic audit kind {audit_kind!r} for {role!r}"
        )

    return {
        "passed": True,
        "role": role,
        "model_id": contract["model_id"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_config": str(checkpoint_config_path),
        "checkpoint_config_sha256": checkpoint_config_sha256,
        "training_report": str(training_report),
        "training_report_sha256": training_report_sha256,
        "protocol": protocol,
    }


def _switch_diagnostic(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_query: dict[str, dict[str, str]] = {}
    for row in records:
        if str(row["true_rule"]) != "passable":
            continue
        condition = str(row["history_condition"])
        if condition not in {
            "observed_passable",
            "observed_blocked",
        }:
            continue
        by_query.setdefault(str(row["static_query_id"]), {})[
            condition
        ] = str(row["predicted_rule"])
    if not by_query or any(
        set(values)
        != {"observed_passable", "observed_blocked"}
        for values in by_query.values()
    ):
        raise ValueError("Incomplete paired history predictions")
    switches = 0
    correct = 0
    for values in by_query.values():
        passable = values["observed_passable"]
        blocked = values["observed_blocked"]
        switches += passable != blocked
        correct += passable == "passable" and blocked == "blocked"
    total = len(by_query)
    return {
        "queries": total,
        "history_changes_selected_target": switches,
        "history_target_switch_rate": switches / total,
        "correct_directional_switches": correct,
        "correct_directional_switch_rate": correct / total,
    }


def _tiny_diagnostic_summary(
    records: list[dict[str, Any]],
    *,
    expected_static_queries: int,
) -> dict[str, Any]:
    """Summarize exact seen examples without invoking the 50x6 formal gate."""

    expected_records = (
        int(expected_static_queries) * 2 * len(HISTORY_CONDITIONS)
    )
    _require(
        len(records) == expected_records,
        "Tiny diagnostic record count mismatch: "
        f"expected {expected_records}, got {len(records)}",
    )
    paired = paired_effect_rows(records)
    _require(
        len(paired) == int(expected_static_queries) * 2,
        "Tiny diagnostic paired-row count mismatch",
    )
    static_queries = {str(row["static_query_id"]) for row in paired}
    _require(
        len(static_queries) == int(expected_static_queries),
        "Tiny diagnostic static-query count mismatch",
    )

    by_true_rule: dict[str, dict[str, Any]] = {}
    for rule in ("passable", "blocked"):
        selected = [
            row for row in paired if str(row["true_rule"]) == rule
        ]
        _require(
            len(selected) == int(expected_static_queries),
            f"Tiny diagnostic {rule} pair count mismatch",
        )
        by_true_rule[rule] = {
            "overall": {
                "pairs": len(selected),
                "native_latent_mse": {
                    "matching_history": mean(
                        float(row["same_history_loss"])
                        for row in selected
                    ),
                    "other_history": mean(
                        float(row["other_history_loss"])
                        for row in selected
                    ),
                    "no_crossing_attempt_history": mean(
                        float(row["no_evidence_history_loss"])
                        for row in selected
                    ),
                },
                "paired_advantage": {
                    "matching_vs_other_history": mean(
                        float(row["same_vs_other_advantage"])
                        for row in selected
                    ),
                    "matching_vs_no_crossing_attempt": mean(
                        float(row["same_vs_no_evidence_advantage"])
                        for row in selected
                    ),
                },
                "strict_win_rate": mean(
                    bool(row["strict_win"]) for row in selected
                ),
                "same_history_two_target_accuracy": mean(
                    bool(row["same_history_true_target_closer"])
                    for row in selected
                ),
            }
        }
    return {
        "diagnostic_only": True,
        "formal_50x6_gate_applied": False,
        "records": len(records),
        "paired_rows": len(paired),
        "unique_static_queries": len(static_queries),
        "by_true_rule": by_true_rule,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score one original or tiny-overfit History=3 checkpoint on "
            "frozen exact training examples"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--role",
        choices=(
            "original",
            "tiny_overfit",
            "joint_update",
            "fixed_representation",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(
        config.get("status")
        == "diagnostic_frozen_before_catalog_generation_and_scoring",
        "Tiny-overfit Eval config is not frozen",
    )
    catalog = resolve_contextworld_path(
        config["artifacts"]["catalog"], repo_root=ROOT
    )
    catalog_payload = json.loads(catalog.read_text(encoding="utf-8"))
    _require(
        catalog_payload.get("benchmark") == config["benchmark"],
        "Tiny-overfit catalog/config benchmark mismatch",
    )
    normalizer = resolve_contextworld_path(
        config["adapter"]["normalizer"], repo_root=ROOT
    )
    _require(
        file_sha256(normalizer)
        == config["adapter"]["normalizer_sha256"],
        "Frozen normalizer hash mismatch",
    )
    checkpoint = resolve_contextworld_path(
        args.checkpoint, repo_root=ROOT
    )
    training_report = resolve_contextworld_path(
        args.training_report, repo_root=ROOT
    )
    checkpoint_audit = _audit_checkpoint(
        config=config,
        role=args.role,
        checkpoint=checkpoint,
        training_report=training_report,
    )
    output = resolve_contextworld_path(args.output, repo_root=ROOT)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    assets, data_audit = load_validation_assets(
        catalog,
        repo_root=ROOT,
    )
    adapter = StableWorldModelLeWMAdapter.from_checkpoint(
        checkpoint,
        normalizer=normalizer,
        repo_root=ROOT,
        stablewm_repo=str(config["stable_worldmodel"]["repo"]),
        stablewm_ref=str(config["stable_worldmodel"]["commit"]),
        device=args.device,
    )
    scored = score_validation_assets(
        adapter,
        assets,
        batch_size=int(config["evaluation"]["batch_size"]),
    )
    summary = _tiny_diagnostic_summary(
        scored["records"],
        expected_static_queries=int(
            config["evaluation"]["unique_queries_per_seed"]
        ),
    )
    switch = _switch_diagnostic(scored["records"])
    thresholds = config["gates"]["diagnostic_overfit"]
    accuracies = {
        rule: float(
            summary["by_true_rule"][rule]["overall"][
                "same_history_two_target_accuracy"
            ]
        )
        for rule in ("passable", "blocked")
    }
    threshold_checks = {
        "passable_matching_history_accuracy": (
            accuracies["passable"]
            > float(
                thresholds[
                    "minimum_matching_history_accuracy_exclusive"
                ]
            )
        ),
        "blocked_matching_history_accuracy": (
            accuracies["blocked"]
            > float(
                thresholds[
                    "minimum_matching_history_accuracy_exclusive"
                ]
            )
        ),
        "history_target_switch_rate": (
            switch["history_target_switch_rate"]
            > float(
                thresholds[
                    "minimum_history_target_switch_rate_exclusive"
                ]
            )
        ),
        "correct_directional_switch_rate": (
            switch["correct_directional_switch_rate"]
            > float(
                thresholds[
                    "minimum_correct_directional_switch_rate_exclusive"
                ]
            )
        ),
    }
    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "diagnostic_only": True,
        "role": args.role,
        "identity": {
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "catalog": str(catalog),
            "catalog_sha256": file_sha256(catalog),
            "normalizer": str(normalizer),
            "normalizer_sha256": file_sha256(normalizer),
        },
        "checkpoint_audit": checkpoint_audit,
        "data_audit": data_audit,
        "score_audit": scored["score_audit"],
        "summary": summary,
        "switch_diagnostic": switch,
        "overfit_gate": {
            "passed": all(threshold_checks.values()),
            "checks": threshold_checks,
            "matching_history_accuracy": accuracies,
            "thresholds": thresholds,
        },
        "records": scored["records"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "role": args.role,
                "matching_history_accuracy": accuracies,
                "switch_diagnostic": switch,
                "overfit_gate_passed": result["overfit_gate"]["passed"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
