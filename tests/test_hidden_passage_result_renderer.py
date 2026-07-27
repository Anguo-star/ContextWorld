from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from contextworld.evaluation.hidden_passage_validation import (
    PAIRED_BOOTSTRAP_METRICS,
)
from scripts import render_tworoom_hidden_passage_h3_results as renderer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_validation_v2.yaml"
)


def _config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _checkpoint_summary(*, passed: bool, offset: float) -> dict[str, Any]:
    lower = 0.05 if passed else -0.05
    intervals = {
        metric_id: {
            "mean": 0.10 + offset,
            "lower": lower + offset,
            "upper": 0.15 + offset,
        }
        for metric_id in PAIRED_BOOTSTRAP_METRICS
    }
    return {
        "by_true_rule": {
            "passable": {
                "overall": {
                    "native_latent_mse": {
                        "same_rule_history": 123.0 + offset,
                        "other_rule_history": 456.0 + offset,
                    },
                    "paired_advantage": {
                        "same_vs_other_rule_history": 0.21 + offset,
                        "same_vs_no_crossing_attempt": 0.22 + offset,
                    },
                    "same_history_two_target_accuracy": 0.80,
                    "strict_win_rate": 0.75,
                }
            },
            "blocked": {
                "overall": {
                    "native_latent_mse": {
                        "same_rule_history": 321.0 + offset,
                        "other_rule_history": 654.0 + offset,
                    },
                    "paired_advantage": {
                        "same_vs_other_rule_history": 0.31 + offset,
                        "same_vs_no_crossing_attempt": 0.32 + offset,
                    },
                    "same_history_two_target_accuracy": 0.85,
                    "strict_win_rate": 0.70,
                }
            },
        },
        "paired_static_query_bootstrap": {
            "confidence": 0.95,
            "metrics": intervals,
        },
        "decision": {
            "thresholds": {
                "minimum_bootstrap_lower_bound_exclusive": 0.0,
            },
            "checks": {
                "paired_static_query_bootstrap_lower_bounds_above_threshold": (
                    passed
                ),
            },
            "passed": passed,
            "failed_checks": [] if passed else ["synthetic_failed_gate"],
        },
        "metric_contract": {
            "raw_latent_mse_cross_checkpoint_comparison_allowed": False,
        },
    }


def _fixture(
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    Path,
    list[dict[str, Any]],
    list[Path],
    dict[str, Any],
]:
    config = _config()
    passed_by_identity: dict[tuple[str, int], bool] = {}
    results = []
    result_paths = []
    offset = 0.0
    for model_id, seeds in config["comparison"]["required_results"].items():
        for seed in seeds:
            passed = model_id == "H3_Passage_MixedRules"
            passed_by_identity[(str(model_id), int(seed))] = passed
            result = {
                "model_id": str(model_id),
                "training_seed": int(seed),
                "summary": _checkpoint_summary(
                    passed=passed,
                    offset=offset,
                ),
            }
            path = tmp_path / f"{model_id}-s{seed}.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            results.append(result)
            result_paths.append(path)
            offset += 0.001

    checks = {
        "mixed_rules_passes_all_three_training_seeds": True,
        "original_baseline_fails": True,
        "passable_only_family_does_not_pass_all_three_seeds": True,
        "blocked_only_family_does_not_pass_all_three_seeds": True,
    }
    aggregate = {
        "schema_version": 2,
        "benchmark": config["benchmark"],
        "status": "completed",
        "comparison_contract": {
            "native_latent_mse_cross_checkpoint_comparison_allowed": False,
        },
        "attribution": {
            "passed": True,
            "checks": checks,
            "checkpoint_pass_by_model_and_seed": {
                f"{model_id}/s{seed}": passed
                for (model_id, seed), passed in sorted(
                    passed_by_identity.items()
                )
            },
        },
    }
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    return aggregate, aggregate_path, results, result_paths, config


def _build(
    *,
    aggregate: dict[str, Any],
    aggregate_path: Path,
    results: list[dict[str, Any]],
    result_paths: list[Path],
    config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    return renderer.build_reader_outputs(
        stored_aggregate=aggregate,
        aggregate_path=aggregate_path,
        results=results,
        result_paths=result_paths,
        config=config,
        config_path=CONFIG,
        expected_catalog_sha256="a" * 64,
    )


def test_renderer_revalidates_then_reports_ten_reader_facing_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate, aggregate_path, results, paths, config = _fixture(tmp_path)
    calls = []

    def strict_revalidation(**kwargs):
        calls.append(kwargs)
        return copy.deepcopy(aggregate)

    monkeypatch.setattr(
        renderer,
        "aggregate_validation_results",
        strict_revalidation,
    )
    payload, markdown = _build(
        aggregate=aggregate,
        aggregate_path=aggregate_path,
        results=list(reversed(results)),
        result_paths=list(reversed(paths)),
        config=config,
    )

    assert len(calls) == 1
    assert calls[0]["expected_catalog_sha256"] == "a" * 64
    assert payload["source"]["strict_identity_revalidation_passed"] is True
    assert len(payload["checkpoint_results"]) == 10
    assert all(
        len(row["confidence_intervals_95"]) == 6
        for row in payload["checkpoint_results"]
    )
    assert payload["checkpoint_results"][0]["training_recipe_zh"] == (
        "原始 TwoRoom 数据训练（基线）"
    )
    first_passable = payload["checkpoint_results"][0]["by_true_rule"][
        "passable"
    ]
    assert first_passable["same_vs_other_history_advantage"] == 0.21
    assert first_passable["matching_history_target_accuracy"] == 0.80
    assert payload["attribution"]["passed"] is True
    assert all(
        family["attribution_condition_satisfied"]
        for family in payload["attribution"]["three_training_families"]
    )

    assert "原始 H3 续训：同时用两种规则合成数据" in markdown
    assert "六项 95% 置信区间" in markdown
    assert "三种训练数据配方的 3/3 归因检查" in markdown
    assert "H3_Passage_MixedRules" not in markdown
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert '"native_latent_mse":' not in serialized
    assert '"same_rule_history":' not in serialized
    assert payload["metric_contract"][
        "raw_latent_mse_cross_checkpoint_comparison_allowed"
    ] is False


def test_renderer_rejects_a_stored_aggregate_that_differs_after_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate, aggregate_path, results, paths, config = _fixture(tmp_path)
    trusted = copy.deepcopy(aggregate)
    aggregate["attribution"]["passed"] = False
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    monkeypatch.setattr(
        renderer,
        "aggregate_validation_results",
        lambda **_: trusted,
    )

    with pytest.raises(
        ValueError,
        match="differs from strict identity revalidation",
    ):
        _build(
            aggregate=aggregate,
            aggregate_path=aggregate_path,
            results=results,
            result_paths=paths,
            config=config,
        )


def test_renderer_rejects_cross_checkpoint_raw_loss_permission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate, aggregate_path, results, paths, config = _fixture(tmp_path)
    aggregate["comparison_contract"][
        "native_latent_mse_cross_checkpoint_comparison_allowed"
    ] = True
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    monkeypatch.setattr(
        renderer,
        "aggregate_validation_results",
        lambda **_: copy.deepcopy(aggregate),
    )

    with pytest.raises(ValueError, match="does not forbid"):
        _build(
            aggregate=aggregate,
            aggregate_path=aggregate_path,
            results=results,
            result_paths=paths,
            config=config,
        )


def test_renderer_recomputes_the_six_ci_final_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate, aggregate_path, results, paths, config = _fixture(tmp_path)
    mixed = next(
        row
        for row in results
        if row["model_id"] == "H3_Passage_MixedRules"
    )
    metric = PAIRED_BOOTSTRAP_METRICS[0]
    mixed["summary"]["paired_static_query_bootstrap"]["metrics"][metric][
        "lower"
    ] = -0.01
    monkeypatch.setattr(
        renderer,
        "aggregate_validation_results",
        lambda **_: copy.deepcopy(aggregate),
    )

    with pytest.raises(ValueError, match="CI decision differs"):
        _build(
            aggregate=aggregate,
            aggregate_path=aggregate_path,
            results=results,
            result_paths=paths,
            config=config,
        )
