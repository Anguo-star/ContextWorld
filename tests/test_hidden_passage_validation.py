from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from contextworld.benchmarks.adapters import (
    AdapterProtocol,
    SpeedICLModelAdapter,
)
from contextworld.evaluation import hidden_passage_validation as validation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_validation_v2.yaml"
)


def _config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


class _DummyHistoryRuleAdapter(SpeedICLModelAdapter):
    def __init__(self) -> None:
        self.rollout_calls = 0
        self.encode_calls = 0
        self.last_pixels_shape: tuple[int, ...] | None = None
        self.last_actions_shape: tuple[int, ...] | None = None

    @property
    def protocol(self) -> AdapterProtocol:
        return AdapterProtocol(
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=2,
            future_action_blocks=1,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {"adapter_id": "dummy_history_rule"}

    def encode_pixels(
        self,
        pixels: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        del batch_size
        self.encode_calls += 1
        return np.asarray(pixels, dtype=np.float32).reshape(len(pixels), -1) / 255.0

    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        del batch_size
        self.rollout_calls += 1
        pixels = np.asarray(input_pixels, dtype=np.uint8)
        actions = np.asarray(raw_action_blocks, dtype=np.float32)
        self.last_pixels_shape = pixels.shape
        self.last_actions_shape = actions.shape
        assert pixels.shape[1] == 3
        assert actions.shape[1:] == (3, 5, 2)
        assert np.count_nonzero(actions) == 0
        query = pixels[:, -1].astype(np.float32).reshape(len(pixels), -1)
        # The first history frame contains only the history-condition marker.
        query[:, 2] = pixels[:, 0, 0, 0, 2]
        return (query / 255.0)[:, None]

    def frozen_state_hash(self) -> str:
        return "dummy-frozen-state"


class _GeometryOnlyDoorAdapter(_DummyHistoryRuleAdapter):
    @property
    def protocol(self) -> AdapterProtocol:
        return AdapterProtocol(
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=2,
            future_action_blocks=1,
            native_target_encoder=False,
            decoder_required=True,
        )


def test_door_adapter_audit_only_requires_public_geometry_fields() -> None:
    result = validation._adapter_protocol_audit(_GeometryOnlyDoorAdapter())

    assert result["passed"] is True
    assert set(result["checks"]) == {
        "history_tokens",
        "action_block_raw_steps",
        "action_dim",
        "at_least_one_future",
    }


def _query_pixels(index: int) -> np.ndarray:
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    pixels[0, 0, 0] = np.uint8(index % 256)
    pixels[0, 0, 1] = np.uint8(index // 256)
    return pixels


def _synthetic_assets() -> list[dict[str, Any]]:
    assets = []
    conditions = {
        "observed_passable": 200,
        "observed_blocked": 0,
        "did_not_attempt_crossing": 100,
    }
    for seed_index, eval_seed in enumerate((42, 43, 44, 45, 46, 47)):
        for evaluation_index in range(50):
            index = seed_index * 50 + evaluation_index
            query = _query_pixels(index)
            histories = {}
            actions = {}
            for condition, marker in conditions.items():
                history = np.stack([query.copy(), query.copy(), query.copy()])
                history[0, 0, 0, 2] = np.uint8(marker)
                histories[condition] = history
                actions[condition] = np.zeros((3, 5, 2), dtype=np.float32)
            passable = query.copy()
            passable[0, 0, 2] = np.uint8(200)
            assets.append(
                {
                    "query_id": f"s{eval_seed}-e{evaluation_index:03d}",
                    "static_query_id": f"q{index:03d}",
                    "eval_seed": eval_seed,
                    "evaluation_index": evaluation_index,
                    "direction": (
                        "left_to_right"
                        if evaluation_index < 25
                        else "right_to_left"
                    ),
                    "template_id": f"dummy-{index:03d}",
                    "query_pixels": query,
                    "histories": histories,
                    "actions": actions,
                    "targets": {
                        "passable": passable,
                        "blocked": query.copy(),
                    },
                    # A privileged field is deliberately present in the asset.
                    # The adapter API must still receive only pixels/actions.
                    "privileged_rule_for_test": index % 2,
                }
            )
    return assets


def test_validation_assignment_is_300_unique_queries_in_six_balanced_groups() -> None:
    config = _config()
    assignments = validation.select_validation_assignments(config)

    assert len(assignments) == 300
    assert len({row.template.template_id for row in assignments}) == 300
    assert set(config["data"]["generation"]["eval_only_door_positions"]) == set(
        range(30, 195, 4)
    )
    for eval_seed in (42, 43, 44, 45, 46, 47):
        rows = [row for row in assignments if row.eval_seed == eval_seed]
        assert len(rows) == 50
        assert sum(
            row.template.direction == "left_to_right" for row in rows
        ) == 25
        assert sum(
            row.template.direction == "right_to_left" for row in rows
        ) == 25
        assert {row.evaluation_index for row in rows} == set(range(50))


def test_each_eval_seed_controls_a_disjoint_without_replacement_selection() -> None:
    config = _config()
    assignments = validation.select_validation_assignments(config)
    selected = {
        eval_seed: {
            row.template.template_id
            for row in assignments
            if row.eval_seed == eval_seed
        }
        for eval_seed in config["evaluation"]["eval_seeds"]
    }
    assert all(len(values) == 50 for values in selected.values())
    assert sum(map(len, selected.values())) == len(
        set().union(*selected.values())
    )

    changed = copy.deepcopy(config)
    changed["evaluation"]["eval_seeds"][0] = 142
    changed_assignments = validation.select_validation_assignments(changed)
    changed_first = {
        row.template.template_id
        for row in changed_assignments
        if row.eval_seed == 142
    }
    assert changed_first != selected[42]


def test_no_attempt_history_is_continuous_and_shares_query_and_actions() -> None:
    assignments = validation.select_validation_assignments(_config())
    for direction in ("left_to_right", "right_to_left"):
        template = next(
            row.template
            for row in assignments
            if row.template.direction == direction
        )
        arrays, audit = validation.build_rollout_matrix(template)

        assert audit["passed"]
        assert audit["checks"]["no_attempt_rule_invariant"]
        assert audit["checks"]["no_attempt_stays_on_approach_side"]
        assert audit["checks"]["all_histories_end_at_same_query_pixels"]
        assert audit["checks"]["all_histories_end_at_same_query_state"]
        assert audit["checks"]["all_action_blocks_identical"]
        query = arrays["query_pixels"]
        actions = []
        for condition in validation.HISTORY_CONDITIONS:
            assert np.array_equal(
                arrays[f"{condition}_history_pixels"][-1],
                query,
            )
            actions.append(arrays[f"{condition}_action_blocks"])
        assert all(np.array_equal(actions[0], value) for value in actions[1:])
        assert np.array_equal(
            arrays["target_blocked_pixels"],
            arrays["query_pixels"],
        )
        assert not np.array_equal(
            arrays["target_passable_pixels"],
            arrays["query_pixels"],
        )


def test_dummy_adapter_scores_exact_2x3_matrix_without_environment(
    monkeypatch,
) -> None:
    assets = _synthetic_assets()
    asset_audit = validation.audit_validation_assets(
        assets,
        eval_seeds=(42, 43, 44, 45, 46, 47),
        unique_queries_per_seed=50,
    )
    assert asset_audit["passed"]

    def environment_call_is_forbidden(*args, **kwargs):
        raise AssertionError(f"Scoring called the environment: {args} {kwargs}")

    monkeypatch.setattr(
        validation,
        "simulate_template",
        environment_call_is_forbidden,
    )
    adapter = _DummyHistoryRuleAdapter()
    scored = validation.score_validation_assets(
        adapter,
        assets,
        batch_size=64,
    )
    summary = validation.summarize_validation_records(
        scored["records"],
        eval_seeds=(42, 43, 44, 45, 46, 47),
        unique_queries_per_seed=50,
        gates=_config()["gates"],
    )

    assert len(scored["records"]) == 1800
    assert scored["score_audit"]["model_predictions"] == 900
    assert scored["score_audit"]["target_encodings"] == 600
    assert scored["score_audit"]["online_environment_calls"] == 0
    assert scored["score_audit"]["privileged_fields_passed_to_adapter"] == []
    assert adapter.rollout_calls == 1
    assert adapter.encode_calls == 1
    assert adapter.last_pixels_shape == (900, 3, 2, 2, 3)
    assert adapter.last_actions_shape == (900, 3, 5, 2)
    assert summary["count_audit"]["passed"]
    assert summary["count_audit"]["records"] == 1800
    assert summary["decision"]["passed"]
    assert (
        summary["paired_static_query_bootstrap"]["unit"]
        == "static_query_within_eval_seed_direction"
    )
    assert set(
        summary["paired_static_query_bootstrap"]["strata"].values()
    ) == {25}
    assert summary["target_latent_separation"]["minimum_mse"] > 0.0
    assert summary["two_target_ties"]["ties"] == 300
    assert summary["two_target_ties"]["tie_rate"] == 1.0 / 3.0
    assert summary["metric_contract"]["unchanged_query_relative_loss_used"] is False
    for rule in validation.TRUE_RULES:
        assert (
            summary["by_true_rule"][rule]["overall"]["strict_win_rate"]
            == 1.0
        )
        assert (
            summary["by_true_rule"][rule]["overall"][
                "same_history_two_target_accuracy"
            ]
            == 1.0
        )


def test_formal_summary_rejects_a_result_that_only_works_for_one_rule() -> None:
    adapter = _DummyHistoryRuleAdapter()
    scored = validation.score_validation_assets(
        adapter,
        _synthetic_assets(),
        batch_size=128,
    )
    for row in scored["records"]:
        if (
            row["true_rule"] == "blocked"
            and row["history_condition"] == "observed_blocked"
        ):
            row["true_next_frame_latent_mse"] = 10.0
    summary = validation.summarize_validation_records(
        scored["records"],
        eval_seeds=(42, 43, 44, 45, 46, 47),
        unique_queries_per_seed=50,
        gates=_config()["gates"],
    )

    assert summary["by_true_rule"]["passable"][
        "overall_both_paired_effects_positive"
    ]
    assert not summary["by_true_rule"]["blocked"][
        "overall_both_paired_effects_positive"
    ]
    assert not summary["decision"]["passed"]


def test_asset_audit_rejects_reusing_one_query_as_multiple_eval_seeds() -> None:
    assets = _synthetic_assets()
    assets[50]["query_pixels"] = assets[0]["query_pixels"].copy()
    audit = validation.audit_validation_assets(
        assets,
        eval_seeds=(42, 43, 44, 45, 46, 47),
        unique_queries_per_seed=50,
    )

    assert not audit["passed"]
    assert not audit["checks"]["query_pixels_unique"]


def _replace_two_target_losses(
    records: list[dict[str, Any]],
    losses_by_input: dict[tuple[str, str], tuple[float, float]],
) -> None:
    for row in records:
        key = (str(row["query_id"]), str(row["history_condition"]))
        passable_loss, blocked_loss = losses_by_input[key]
        losses = {
            "passable": float(passable_loss),
            "blocked": float(blocked_loss),
        }
        true_rule = str(row["true_rule"])
        other_rule = "blocked" if true_rule == "passable" else "passable"
        if passable_loss < blocked_loss:
            predicted_rule = "passable"
        elif blocked_loss < passable_loss:
            predicted_rule = "blocked"
        else:
            predicted_rule = "tie"
        row.update(
            {
                "true_next_frame_latent_mse": losses[true_rule],
                "other_target_latent_mse": losses[other_rule],
                "two_target_margin": (
                    losses[other_rule] - losses[true_rule]
                ),
                "predicted_rule": predicted_rule,
                "true_target_closer": predicted_rule == true_rule,
            }
        )


def test_decision_rejects_positive_advantages_when_one_rule_is_never_selected() -> None:
    scored = validation.score_validation_assets(
        _DummyHistoryRuleAdapter(),
        _synthetic_assets(),
        batch_size=128,
    )
    losses_by_input = {}
    for asset in _synthetic_assets():
        query_id = str(asset["query_id"])
        losses_by_input[(query_id, "observed_passable")] = (1.0, 0.9)
        losses_by_input[(query_id, "observed_blocked")] = (2.0, 0.1)
        losses_by_input[(query_id, "did_not_attempt_crossing")] = (3.0, 0.5)
    _replace_two_target_losses(scored["records"], losses_by_input)

    summary = validation.summarize_validation_records(
        scored["records"],
        eval_seeds=(42, 43, 44, 45, 46, 47),
        unique_queries_per_seed=50,
        gates=_config()["gates"],
    )

    assert summary["by_true_rule"]["passable"][
        "overall_both_paired_effects_positive"
    ]
    assert summary["by_true_rule"]["blocked"][
        "overall_both_paired_effects_positive"
    ]
    assert (
        summary["by_true_rule"]["passable"]["overall"][
            "same_history_two_target_accuracy"
        ]
        == 0.0
    )
    assert not summary["decision"]["passed"]
    assert (
        "evidence_history_target_accuracy_above_threshold_for_each_rule"
        in summary["decision"]["failed_checks"]
    )


def test_stratified_bootstrap_rejects_a_tiny_unstable_positive_mean() -> None:
    assets = _synthetic_assets()
    scored = validation.score_validation_assets(
        _DummyHistoryRuleAdapter(),
        assets,
        batch_size=128,
    )
    losses_by_input = {}
    for asset in assets:
        query_id = str(asset["query_id"])
        local_index = int(asset["evaluation_index"]) % 25
        majority = local_index < 13
        if majority:
            losses_by_input[(query_id, "observed_passable")] = (0.0, 1.0)
            losses_by_input[(query_id, "observed_blocked")] = (1.0, 0.0)
            losses_by_input[(query_id, "did_not_attempt_crossing")] = (1.0, 1.0)
        else:
            losses_by_input[(query_id, "observed_passable")] = (1.0, 0.1)
            losses_by_input[(query_id, "observed_blocked")] = (0.1, 1.0)
            losses_by_input[(query_id, "did_not_attempt_crossing")] = (0.1, 0.1)
    _replace_two_target_losses(scored["records"], losses_by_input)

    summary = validation.summarize_validation_records(
        scored["records"],
        eval_seeds=(42, 43, 44, 45, 46, 47),
        unique_queries_per_seed=50,
        gates=_config()["gates"],
    )

    assert summary["decision"][
        "both_effects_positive_in_every_seed_direction_cell"
    ]
    assert all(
        row["overall"]["same_history_two_target_accuracy"] > 0.5
        for row in summary["by_true_rule"].values()
    )
    assert all(
        row["overall"]["strict_win_rate"] > 0.5
        for row in summary["by_true_rule"].values()
    )
    assert not summary["decision"]["checks"][
        "paired_static_query_bootstrap_lower_bounds_above_threshold"
    ]
    assert not summary["decision"]["passed"]


def test_rule_switch_v2_treats_no_attempt_as_an_auxiliary_default() -> None:
    assets = _synthetic_assets()
    scored = validation.score_validation_assets(
        _DummyHistoryRuleAdapter(),
        assets,
        batch_size=128,
    )
    losses_by_input = {}
    for asset in assets:
        query_id = str(asset["query_id"])
        losses_by_input[(query_id, "observed_passable")] = (0.0, 2.0)
        losses_by_input[(query_id, "observed_blocked")] = (2.0, 0.0)
        # No interaction evidence has no unique correct rule. A model may
        # reasonably retain its blocked prior in this auxiliary condition.
        losses_by_input[(query_id, "did_not_attempt_crossing")] = (2.0, 0.0)
    _replace_two_target_losses(scored["records"], losses_by_input)

    legacy_gates = _config()["gates"]
    legacy = validation.summarize_validation_records(
        copy.deepcopy(scored["records"]),
        eval_seeds=(42, 43, 44, 45, 46, 47),
        unique_queries_per_seed=50,
        gates=legacy_gates,
    )
    assert not legacy["decision"]["passed"]

    rule_switch_gates = copy.deepcopy(legacy_gates)
    rule_switch_gates.update(
        {
            "decision_contract": (
                validation.INFORMATIVE_HISTORY_RULE_SWITCH_V2
            ),
            "minimum_matching_vs_opposite_history_win_rate_exclusive": 0.5,
        }
    )
    rule_switch_gates.pop("minimum_strict_win_rate_exclusive")
    rule_switch_gates["paired_bootstrap"]["required_metrics"] = list(
        validation.INFORMATIVE_HISTORY_BOOTSTRAP_METRICS
    )
    revised = validation.summarize_validation_records(
        copy.deepcopy(scored["records"]),
        eval_seeds=(42, 43, 44, 45, 46, 47),
        unique_queries_per_seed=50,
        gates=rule_switch_gates,
    )

    assert revised["decision"]["passed"]
    assert (
        revised["decision"]["decision_contract"]
        == validation.INFORMATIVE_HISTORY_RULE_SWITCH_V2
    )
    assert (
        revised["metric_contract"]["no_crossing_attempt_role"]
        == "auxiliary_default_tendency_only"
    )
    assert set(revised["paired_static_query_bootstrap"]["metrics"]) == set(
        validation.INFORMATIVE_HISTORY_BOOTSTRAP_METRICS
    )


def test_target_latent_collapse_is_a_hard_failure() -> None:
    scored = validation.score_validation_assets(
        _DummyHistoryRuleAdapter(),
        _synthetic_assets(),
        batch_size=128,
    )
    for row in scored["records"]:
        row["target_pair_latent_mse"] = 0.0
    summary = validation.summarize_validation_records(
        scored["records"],
        eval_seeds=(42, 43, 44, 45, 46, 47),
        unique_queries_per_seed=50,
        gates=_config()["gates"],
    )

    assert not summary["decision"]["checks"][
        "target_latents_are_separated_for_every_query"
    ]
    assert not summary["decision"]["passed"]


def test_loader_recomputes_and_rejects_catalog_content_manifest(
    tmp_path: Path,
) -> None:
    config = _config()
    catalog = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "frozen_before_model_scoring",
        "protocol": {
            "true_future_rules": list(validation.TRUE_RULES),
            "history_conditions": list(validation.HISTORY_CONDITIONS),
            "history_tokens": 3,
            "raw_steps_per_action_block": 5,
            "unchanged_query_relative_loss_is_forbidden": True,
        },
        "bundles": [],
        "content_manifest_sha256": "0" * 64,
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")

    try:
        validation.load_validation_assets(path, repo_root=ROOT)
    except RuntimeError as exc:
        assert "content manifest mismatch" in str(exc)
    else:
        raise AssertionError("Tampered catalog content manifest was accepted")
