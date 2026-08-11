from __future__ import annotations

import pytest

from contextworld.benchmarks.public_score import (
    compare_paired_query_decisions,
    make_component_result,
    make_public_scoreboard,
    make_public_scoreboard_from_spec,
    make_retention_result,
)
from contextworld.benchmarks.public_score_cli import main as public_score_main


_PUBLIC_TEST_HASH = "a" * 64


def _query_submission(
    model_name: str,
    decisions: dict[str, int],
    *,
    component_id: str = "action_strength",
    public_test_id: str = "action_strength_public_test_v1",
    decision_metric: str = "calibrated_icl_correct",
) -> dict:
    return {
        "component_id": component_id,
        "public_test_id": public_test_id,
        "public_test_sha256": _PUBLIC_TEST_HASH,
        "model_name": model_name,
        "decision_metric": decision_metric,
        "query_decisions": decisions,
    }


def _paired_component(**overrides):
    kwargs = {
        "component_id": "action_strength",
        "component_name": "PushT action strength",
        "method_name": "reference_lewm",
        "primary_metric_id": "correct_future_rate",
        "primary_metric_label": "Real next-state choice accuracy",
        "per_seed_primary_values": [0.97, 0.96, 0.965],
        "per_seed_gate_passes": [True, True, True],
        "ability_passed": True,
        "required_training_seeds": 3,
        "evidence_scope": "behavioral",
        "original_task_retention": make_retention_result(
            result="PASS",
            metric_id="standard_task_cem_success_rate",
            metric_label="Standard-task CEM success rate",
            per_seed_values=[224 / 300, 216 / 300, 232 / 300],
            baseline_value=225 / 300,
        ),
        "diagnostics": {
            "correct_history_rate": [0.98, 0.99, 0.98],
            "context_switch_rate": [0.99, 0.99, 0.99],
        },
    }
    kwargs.update(overrides)
    return make_component_result(**kwargs)


def test_component_public_view_has_only_two_outcomes() -> None:
    result = _paired_component()
    public = result["public"]

    assert set(public) == {
        "component_id",
        "component_name",
        "method_name",
        "icl_ability",
        "original_task_retention",
    }
    assert public["icl_ability"]["result"] == "PASS"
    assert public["icl_ability"]["primary_metric"]["mean"] == pytest.approx(
        0.965
    )
    assert public["icl_ability"]["training_seed_stability"] == {
        "passed_checkpoints": 3,
        "evaluated_checkpoints": 3,
        "required_checkpoints": 3,
        "all_required_seeds_passed": True,
    }
    assert public["original_task_retention"]["result"] == "PASS"
    assert "correct_history_rate" not in public
    assert result["diagnostics"]["correct_history_rate"] == [0.98, 0.99, 0.98]


def test_a_failed_seed_prevents_a_formal_pass() -> None:
    with pytest.raises(ValueError, match="one or more required training seeds"):
        _paired_component(per_seed_gate_passes=[True, False, True])


def test_training_attribution_requires_matched_controls() -> None:
    with pytest.raises(ValueError, match="matched control evidence"):
        _paired_component(evidence_scope="training_attributed")

    result = _paired_component(
        evidence_scope="training_attributed",
        training_attribution={
            "control_kind": "matched_no_factor_training_control",
            "paired_training_seeds": 3,
            "all_paired_effects_favor_target": True,
        },
    )
    assert (
        result["public"]["icl_ability"]["claim"]
        == "training_attributed_icl_demonstrated"
    )

    failed = _paired_component(
        ability_passed=False,
        evidence_scope="training_attributed",
        training_attribution={
            "control_kind": "matched_no_factor_training_control",
            "paired_training_seeds": 3,
            "all_paired_effects_favor_target": False,
        },
    )
    assert failed["public"]["icl_ability"]["claim"] == "icl_not_demonstrated"


def test_not_applicable_retention_is_explicit() -> None:
    retention = make_retention_result(
        result="N/A",
        reason="This component has no meaningful closed-loop original task.",
    )
    result = _paired_component(original_task_retention=retention)
    assert result["public"]["original_task_retention"] == retention

    with pytest.raises(ValueError, match="requires a reason"):
        make_retention_result(result="N/A")


def test_not_evaluated_retention_is_distinct_and_unscored() -> None:
    retention = make_retention_result(
        result="NOT_EVALUATED",
        reason="The applicable original-task CEM run is not complete.",
    )
    assert retention == {
        "result": "NOT_EVALUATED",
        "reason": "The applicable original-task CEM run is not complete.",
    }
    with pytest.raises(ValueError, match="requires a reason"):
        make_retention_result(result="NOT_EVALUATED")
    with pytest.raises(ValueError, match="cannot contain a score"):
        make_retention_result(
            result="NOT_EVALUATED",
            reason="Pending.",
            metric_id="cem_success_rate",
        )


def test_scoreboard_never_produces_a_cross_component_average() -> None:
    action_strength = _paired_component()
    door = _paired_component(
        component_id="door_rule",
        component_name="TwoRoom door rule",
        method_name="fixed_encoder_lewm",
        primary_metric_id="correct_target_choice_rate",
        primary_metric_label="Door outcome choice accuracy",
        per_seed_primary_values=[0.91, 0.93, 0.92],
    )
    scoreboard = make_public_scoreboard([door, action_strength])

    assert "overall_score" not in scoreboard
    assert "average" not in scoreboard
    assert [
        row["component_id"] for row in scoreboard["component_results"]
    ] == ["action_strength", "door_rule"]
    assert all("diagnostics" not in row for row in scoreboard["component_results"])


def test_optional_uncertainty_is_a_paired_query_interval() -> None:
    result = _paired_component(
        uncertainty={
            "kind": "paired_bootstrap_95_percent_interval",
            "lower": 0.94,
            "upper": 0.98,
        }
    )
    assert result["public"]["icl_ability"]["primary_metric"]["uncertainty"] == {
        "kind": "paired_bootstrap_95_percent_interval",
        "lower": 0.94,
        "upper": 0.98,
    }


def test_paired_model_comparison_does_not_call_a_tie_superior() -> None:
    decisions = {f"q{index:03d}": index % 2 for index in range(100)}
    result = compare_paired_query_decisions(
        model_a=_query_submission("model-a", decisions),
        model_b=_query_submission("model-b", decisions),
        bootstrap_resamples=1_000,
        bootstrap_seed=7,
    )

    assert result["models"]["model_a"]["accuracy"] == 0.5
    assert result["models"]["model_b"]["accuracy"] == 0.5
    assert result["paired_accuracy_difference"]["value"] == 0.0
    assert result["superiority"]["model_a_superior"] is False
    assert result["superiority"]["superior_model"] is None


def test_paired_model_comparison_requires_positive_lower_bound() -> None:
    model_a = {f"q{index:03d}": 1 for index in range(100)}
    model_b = {
        f"q{index:03d}": int(index >= 70) for index in range(100)
    }
    result = compare_paired_query_decisions(
        model_a=_query_submission("model-a", model_a),
        model_b=_query_submission("model-b", model_b),
        bootstrap_resamples=2_000,
        bootstrap_seed=11,
    )

    interval = result["paired_accuracy_difference"][
        "paired_bootstrap_95_percent_interval"
    ]
    assert result["models"]["model_a"]["accuracy"] == 1.0
    assert result["models"]["model_b"]["accuracy"] == 0.3
    assert result["paired_accuracy_difference"]["value"] == 0.7
    assert interval["lower"] > 0.0
    assert result["superiority"] == {
        "criterion": "only_if_lower_bound_gt_0",
        "rule": (
            "model_a_is_superior_only_if_the_paired_bootstrap_95_percent_"
            "lower_bound_of_a_minus_b_is_strictly_greater_than_zero"
        ),
        "model_a_superior": True,
        "superior_model": "model-a",
    }


def test_paired_model_comparison_rejects_unpaired_queries() -> None:
    with pytest.raises(ValueError, match="identical query ids"):
        compare_paired_query_decisions(
            model_a=_query_submission("model-a", {"q1": 1, "q2": 0}),
            model_b=_query_submission("model-b", {"q1": 1, "q3": 0}),
        )


def test_paired_model_comparison_rejects_cross_component_scores() -> None:
    with pytest.raises(ValueError, match="Cross-component"):
        compare_paired_query_decisions(
            model_a=_query_submission("model-a", {"q1": 1}),
            model_b=_query_submission(
                "model-b", {"q1": 1}, component_id="door_rule"
            ),
        )


def test_paired_model_comparison_rejects_raw_latent_mse() -> None:
    with pytest.raises(ValueError, match="raw latent MSE"):
        compare_paired_query_decisions(
            model_a=_query_submission(
                "model-a",
                {"q1": 1},
                decision_metric="raw_latent_mse",
            ),
            model_b=_query_submission("model-b", {"q1": 1}),
        )


def _scoreboard_spec() -> dict:
    return {
        "schema_version": 1,
        "result_kind": "contextworld_public_scoreboard_spec",
        "components": [
            {
                "component_id": "door_rule",
                "component_name": "TwoRoom door rule",
                "method_name": "fixed_encoder_lewm",
                "primary_metric": {
                    "id": "correct_target_choice_rate",
                    "label": "Door outcome choice accuracy",
                    "per_seed_values": [0.91, 0.93, 0.92],
                },
                "per_seed_gate_passes": [True, True, True],
                "ability_passed": True,
                "required_training_seeds": 3,
                "evidence_scope": "behavioral",
                "original_task_retention": {
                    "result": "NOT_EVALUATED",
                    "reason": "The original-task CEM run is pending.",
                },
            }
        ],
    }


def test_scoreboard_generator_exposes_no_diagnostics_or_suite_average() -> None:
    scoreboard = make_public_scoreboard_from_spec(_scoreboard_spec())
    rendered = str(scoreboard).lower()

    assert "overall_score" not in scoreboard
    assert "raw_latent" not in rendered
    assert "latent_mse" not in rendered
    assert "diagnostics" not in rendered
    assert scoreboard["component_results"][0]["original_task_retention"] == {
        "result": "NOT_EVALUATED",
        "reason": "The original-task CEM run is pending.",
    }


def test_scoreboard_cli_writes_generated_json(tmp_path) -> None:
    import json

    input_path = tmp_path / "formal-results.json"
    output_path = tmp_path / "scoreboard.json"
    input_path.write_text(json.dumps(_scoreboard_spec()), encoding="utf-8")

    public_score_main(
        ["--input", str(input_path), "--output", str(output_path)]
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["result_kind"] == "contextworld_public_scoreboard"
    assert len(payload["component_results"]) == 1


def test_scoreboard_generator_rejects_raw_latent_public_metric() -> None:
    spec = _scoreboard_spec()
    spec["components"][0]["primary_metric"] = {
        "id": "raw_latent_mse",
        "label": "Raw latent MSE",
        "per_seed_values": [0.1, 0.1, 0.1],
    }
    with pytest.raises(ValueError, match="Raw latent loss"):
        make_public_scoreboard_from_spec(spec)
