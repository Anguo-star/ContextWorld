from __future__ import annotations

import copy
import json
import math

import pytest

from contextworld.benchmarks.reference_decision import (
    aggregate_method_decision,
    main,
    reference_decision_for_result,
)


def _action_result(
    *,
    completion: bool = True,
    seed: int | None = None,
    checkpoint: str | None = None,
    recipe: str | None = None,
    adapter_id: str | None = None,
) -> dict:
    core = {
        "physical_group_macro_accuracy": 1.0,
        "minimum_physical_group_accuracy": 1.0,
        "paired_query_bootstrap_95_percent_interval": {"lower": 1.0},
    }
    gate = {
        "checks": {
            "physical_group_macro_accuracy": True,
            "minimum_physical_group_accuracy": True,
            "bootstrap_lower_bound": True,
        },
        "passed": True,
    }
    gate_metrics = {
        "correct_history_rate": 1.0 if completion else 0.0,
        "context_switch_rate": 1.0 if completion else 0.0,
        "latent_response": {
            "target_latent_separation": {
                "zero_separation_pair_count": 0 if completion else 1,
                "minimum_target_response_mse": 1.0,
            },
            "response_gain": 1.0 if completion else 0.0,
            "normalized_response_error": 0.0 if completion else 1.5,
        },
    }
    result = {
        "core_h1": core,
        "gate": gate,
        "gate_completion": {
            "metrics": gate_metrics,
            "uncertainty": {
                "lower_bounds": {
                    "correct_history_rate": 1.0,
                    "context_switch_rate": 1.0,
                }
            },
            # A stale stored result must not override recomputation.
            "passed": True,
        },
    }
    if any(value is not None for value in (seed, checkpoint, recipe, adapter_id)):
        result["model"] = {
            "training_seed": seed,
            "training_recipe": recipe,
            "adapter": {
                "adapter_id": adapter_id,
                "checkpoint_sha256": checkpoint,
            },
        }
    return result


def _door_result() -> dict:
    """Build complete numeric Door evidence for both split entry points."""

    cells = {
        f"s{seed}/{direction}": {
            "paired_advantage": {"same_vs_other_rule_history": 1.0},
            "same_history_two_target_accuracy": 0.9,
        }
        for seed in (42, 43, 44, 45, 46, 47)
        for direction in ("left_to_right", "right_to_left")
    }
    rules = {
        rule: {
            "overall": {
                "paired_advantage": {
                    "same_vs_other_rule_history": 1.0,
                },
                "same_history_two_target_accuracy": 0.9,
                "matching_vs_opposite_history_win_rate": 0.9,
            },
            "by_eval_seed_and_direction": copy.deepcopy(cells),
        }
        for rule in ("passable", "blocked")
    }
    bootstrap_metrics = {
        name: {"mean": 1.0, "lower": 0.5, "upper": 1.5}
        for name in (
            "passable/same_vs_other_rule_history",
            "blocked/same_vs_other_rule_history",
            "passable/matching_history_two_target_margin",
            "blocked/matching_history_two_target_margin",
        )
    }
    summary = {
        "by_true_rule": rules,
        "paired_static_query_bootstrap": {
            "metrics": bootstrap_metrics,
        },
        "target_latent_separation": {
            "queries": 300,
            "minimum_mse": 1.0,
        },
        "decision": {
            "passed": True,
            "checks": {
                "matching_history_beats_opposite_history_for_each_true_rule": True,
                "matching_history_beats_opposite_history_in_every_seed_direction_cell": True,
                "matching_history_target_accuracy_above_threshold_for_each_rule": True,
                (
                    "matching_history_target_accuracy_above_threshold_in_every_seed_direction_cell"
                ): True,
                "matching_history_beats_opposite_history_on_majority_queries_for_each_rule": True,
                "required_bootstrap_lower_bounds_above_threshold": True,
                "target_latents_are_separated_for_every_query": True,
            },
        },
    }
    gate_completion = {
        "metrics": {
            "correct_history_rate": 1.0,
            "context_switch_rate": 1.0,
            "worst_rule_correct_target_choice_rate": 1.0,
            "latent_response": {
                "target_latent_separation": {
                    "zero_separation_pair_count": 0,
                    "minimum_target_response_mse": 1.0,
                },
                "response_gain": 1.0,
                "normalized_response_error": 0.0,
            },
        },
        "uncertainty": {
            "lower_bounds": {
                "context_switch_rate": 1.0,
                "worst_rule_correct_target_choice_rate": 1.0,
            }
        },
        "passed": True,
    }
    return {"summary": summary, "gate_completion": gate_completion}


def _door_development_result() -> dict:
    result = _door_result()
    return {
        "metrics": {
            key: copy.deepcopy(value)
            for key, value in result["summary"].items()
            if key != "decision"
        },
        "gate_completion_inputs": copy.deepcopy(result["gate_completion"]),
    }


def test_primary_gate_pass_does_not_override_failed_additive_gate() -> None:
    receipt = reference_decision_for_result(
        "action_delay", _action_result(completion=False)
    )
    assert receipt["original_gate"]["passed"] is True
    assert receipt["gate_completion"]["passed"] is False
    assert receipt["passed"] is False


def test_three_seed_method_receipt_is_the_same_per_checkpoint_conjunction() -> None:
    rows = [
        _action_result(
            seed=seed,
            checkpoint=f"{seed:064x}",
            recipe="contextworld_action_delay",
            adapter_id="stable_worldmodel_lewm_v1",
        )
        for seed in (3072, 3073, 3074)
    ]
    aggregate = aggregate_method_decision("action_delay", rows)
    individual = [
        reference_decision_for_result("action_delay", row)["passed"]
        for row in rows
    ]
    assert individual == [True, True, True]
    assert aggregate["passed"] is True
    assert [row["passed"] for row in aggregate["checkpoints"]] == individual


def test_three_seed_method_rejects_duplicate_training_seed() -> None:
    rows = [
        _action_result(
            seed=seed,
            checkpoint=f"{index + 1:064x}",
            recipe="contextworld_action_delay",
            adapter_id="stable_worldmodel_lewm_v1",
        )
        for index, seed in enumerate((3072, 3072, 3074))
    ]
    aggregate = aggregate_method_decision("action_delay", rows)
    assert aggregate["passed"] is False
    assert aggregate["identity"]["training_seeds_present"] is True
    assert aggregate["identity"]["training_seeds_distinct"] is False
    assert "training_seeds_distinct_failed" in aggregate["reason_codes"]


def test_three_seed_method_rejects_duplicate_checkpoint_hash() -> None:
    rows = [
        _action_result(
            seed=seed,
            checkpoint="a" * 64 if index < 2 else "b" * 64,
            recipe="contextworld_action_delay",
            adapter_id="stable_worldmodel_lewm_v1",
        )
        for index, seed in enumerate((3072, 3073, 3074))
    ]
    aggregate = aggregate_method_decision("action_delay", rows)
    assert aggregate["passed"] is False
    assert aggregate["identity"]["checkpoint_hashes_present"] is True
    assert aggregate["identity"]["checkpoint_hashes_distinct"] is False
    assert "checkpoint_hashes_distinct_failed" in aggregate["reason_codes"]


def test_three_seed_method_without_identity_is_descriptive_only() -> None:
    aggregate = aggregate_method_decision(
        "action_delay", [_action_result() for _ in range(3)]
    )
    assert aggregate["passed"] is False
    assert aggregate["status"] == "rejected"
    assert "training_seeds_present_failed" in aggregate["reason_codes"]
    assert "checkpoint_hashes_present_failed" in aggregate["reason_codes"]


def test_three_seed_method_rejects_non_three_checkpoint_contract() -> None:
    with pytest.raises(ValueError, match="exactly three checkpoints"):
        aggregate_method_decision(
            "action_delay", [], required_checkpoints=2
        )


def test_development_decision_free_inputs_are_rescored_and_missing_inputs_fail() -> None:
    result = _action_result()
    dev = {
        "metrics": {
            **result["core_h1"],
            "gate_completion_inputs": result["gate_completion"],
        }
    }
    # Development payloads intentionally have no old ``gate``/``passed``.
    receipt = reference_decision_for_result(
        "action_delay", dev, split="development"
    )
    assert receipt["passed"] is True
    missing = reference_decision_for_result(
        "action_delay", {"core_h1": result["core_h1"]}, split="development"
    )
    assert missing["passed"] is False


def test_door_numeric_gate_rejects_stale_true_flags_with_bad_accuracy_and_win() -> None:
    result = _door_result()
    for row in result["summary"]["by_true_rule"].values():
        row["overall"]["same_history_two_target_accuracy"] = 0.0
        row["overall"]["matching_vs_opposite_history_win_rate"] = 0.0

    receipt = reference_decision_for_result("door", result, split="test")

    assert receipt["original_gate"]["passed"] is False
    assert receipt["passed"] is False
    assert receipt["original_gate"]["legacy_diagnostics"]["stored_passed"] is True
    assert receipt["original_gate"]["legacy_diagnostics"]["agrees_with_numeric"] is False
    assert (
        "original_rule_passable_target_accuracy"
        in receipt["original_gate"]["reason_codes"]
    )


def test_door_development_gate_rejects_negative_per_cell_advantage() -> None:
    result = _door_development_result()
    result["metrics"]["by_true_rule"]["blocked"][
        "by_eval_seed_and_direction"
    ]["s42/left_to_right"]["paired_advantage"][
        "same_vs_other_rule_history"
    ] = -1.0

    receipt = reference_decision_for_result(
        "door", result, split="development"
    )

    assert receipt["original_gate"]["passed"] is False
    assert receipt["passed"] is False
    assert (
        "original_rule_blocked_s42/left_to_right_paired_advantage"
        in receipt["original_gate"]["reason_codes"]
    )


@pytest.mark.parametrize("missing", ["rule", "cell"])
def test_door_numeric_gate_rejects_missing_required_rule_or_cell(
    missing: str,
) -> None:
    result = _door_result()
    if missing == "rule":
        del result["summary"]["by_true_rule"]["blocked"]
    else:
        del result["summary"]["by_true_rule"]["passable"][
            "by_eval_seed_and_direction"
        ]["s46/right_to_left"]

    receipt = reference_decision_for_result("door", result, split="test")

    assert receipt["original_gate"]["passed"] is False
    assert receipt["passed"] is False
    reason_codes = receipt["original_gate"]["reason_codes"]
    if missing == "rule":
        assert "original_rule_missing" in reason_codes
    else:
        assert "original_rule_passable_cell_missing" in reason_codes


def test_door_numeric_gate_rejects_nonfinite_numeric_evidence() -> None:
    result = _door_result()
    result["summary"]["by_true_rule"]["passable"][
        "overall"
    ]["same_history_two_target_accuracy"] = math.nan

    receipt = reference_decision_for_result("door", result, split="test")

    assert receipt["original_gate"]["passed"] is False
    assert receipt["passed"] is False


def test_door_numeric_gate_accepts_complete_test_and_development_inputs() -> None:
    test_receipt = reference_decision_for_result(
        "door", _door_result(), split="test"
    )
    development_receipt = reference_decision_for_result(
        "door", _door_development_result(), split="development"
    )

    assert test_receipt["original_gate"]["passed"] is True
    assert test_receipt["passed"] is True
    assert development_receipt["original_gate"]["passed"] is True
    assert development_receipt["passed"] is True


def test_reference_decision_cli_writes_single_and_method_receipts(tmp_path) -> None:
    rows = [
        _action_result(
            seed=seed,
            checkpoint=f"{seed:064x}",
            recipe="contextworld_action_delay",
            adapter_id="stable_worldmodel_lewm_v1",
        )
        for seed in (3072, 3073, 3074)
    ]
    inputs = []
    for index, row in enumerate(rows):
        path = tmp_path / f"result-{index}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        inputs.append(path)
    single_output = tmp_path / "single.json"
    assert (
        main(
            [
                "single",
                "--component",
                "action_delay",
                "--input",
                str(inputs[0]),
                "--output",
                str(single_output),
            ]
        )
        == 0
    )
    assert json.loads(single_output.read_text())["passed"] is True
    method_output = tmp_path / "method.json"
    argv = ["method", "--component", "action_delay"]
    for path in inputs:
        argv.extend(("--input", str(path)))
    argv.extend(("--output", str(method_output)))
    assert main(argv) == 0
    method = json.loads(method_output.read_text())
    assert method["passed"] is True
    assert method["identity"]["training_seeds_distinct"] is True
