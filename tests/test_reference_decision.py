from __future__ import annotations

import json

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
