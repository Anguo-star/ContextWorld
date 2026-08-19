from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from contextworld.evaluation.hidden_passage_validation import (
    _is_recorded_h3_training_data_path_successor,
    candidate_templates,
    file_sha256,
    select_validation_assignments,
)
from contextworld.synthesis.config import load_config
from contextworld.training.tworoom_data import CATALOG_BY_GROUP
from scripts.build_tworoom_hidden_passage_h3_tiny_overfit_data import (
    _select,
)
from scripts.eval_tworoom_hidden_passage_h3_overfit_diagnostic import (
    _switch_diagnostic,
    _tiny_diagnostic_summary,
)
from scripts.train_tworoom_step1 import (
    _apply_frozen_modules,
    _finalize_frozen_modules,
    _frozen_module_spec,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SEEN_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_train_seen_eval_v1.yaml"
)
TINY_EVAL_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_tiny_overfit_eval_v1.yaml"
)
FROZEN_REPRESENTATION_TRAINING_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_hidden_passage_h3_tiny_frozen_representation_training_v1.yaml"
)
TRAINING_DATA_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_training_data_v1.yaml"
)
LEGACY_TRAINING_DATA_CONFIG_SHA256 = (
    "c9ab2054c3421582d3464634e574711f3035686b41a11c4fc610bc9442bc5f82"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_recorded_training_data_portability_successor_is_fail_closed() -> None:
    config = load_config(TRAINING_DATA_CONFIG)
    observed = file_sha256(TRAINING_DATA_CONFIG)

    assert _is_recorded_h3_training_data_path_successor(
        expected_sha256=LEGACY_TRAINING_DATA_CONFIG_SHA256,
        observed_sha256=observed,
        config=config,
        repo_root=ROOT,
    )

    geometry_drift = deepcopy(config)
    geometry_drift["protocol"]["agent_speed"] = 6.0
    assert not _is_recorded_h3_training_data_path_successor(
        expected_sha256=LEGACY_TRAINING_DATA_CONFIG_SHA256,
        observed_sha256=observed,
        config=geometry_drift,
        repo_root=ROOT,
    )


def test_train_seen_diagnostic_uses_exact_frozen_training_geometry() -> None:
    config = _load(TRAIN_SEEN_CONFIG)
    templates = candidate_templates(config, repo_root=ROOT)
    assignments = select_validation_assignments(
        config,
        repo_root=ROOT,
    )

    assert len(templates) == 1_920
    assert len(assignments) == 300
    assert len({row.template.template_id for row in assignments}) == 300
    assert len({template.door_position for template in templates}) == 96
    assert not (
        {template.door_position for template in templates}
        & set(config["data"]["generation"]["eval_only_door_positions"])
    )
    assert all("-w" in template.template_id for template in templates)


def test_tiny_overfit_eval_is_eight_exact_training_examples() -> None:
    config = _load(TINY_EVAL_CONFIG)
    templates = candidate_templates(config, repo_root=ROOT)
    assignments = select_validation_assignments(
        config,
        repo_root=ROOT,
    )

    assert len(templates) == 80
    assert {template.door_position for template in templates} == {36}
    assert len(assignments) == 8
    assert sum(
        row.template.direction == "left_to_right"
        for row in assignments
    ) == 4
    assert sum(
        row.template.direction == "right_to_left"
        for row in assignments
    ) == 4


def test_tiny_subset_selection_requires_both_rules() -> None:
    rows = [
        {
            "split": "train",
            "rule": rule,
            "scenario_id": f"door36-{rule}",
            "clip_count": 80,
            "factors": {"door.position": [36, 36, 36]},
        }
        for rule in ("passable", "blocked")
    ]
    selected = _select(
        rows,
        split="train",
        doors=[36],
        rules=["passable", "blocked"],
        expected_shards=2,
        expected_clips=160,
    )
    assert [row["rule"] for row in selected] == [
        "blocked",
        "passable",
    ]
    assert CATALOG_BY_GROUP["passage_tiny_overfit"] == (
        "passage_tiny_overfit"
    )


def test_switch_diagnostic_counts_only_correct_directional_changes() -> None:
    records = []
    predictions = {
        "q0": ("passable", "blocked"),
        "q1": ("blocked", "blocked"),
    }
    for query_id, (passable, blocked) in predictions.items():
        for condition, predicted in (
            ("observed_passable", passable),
            ("observed_blocked", blocked),
        ):
            records.append(
                {
                    "static_query_id": query_id,
                    "true_rule": "passable",
                    "history_condition": condition,
                    "predicted_rule": predicted,
                }
            )
    result = _switch_diagnostic(records)
    assert result["queries"] == 2
    assert result["history_target_switch_rate"] == 0.5
    assert result["correct_directional_switch_rate"] == 0.5


def test_tiny_summary_does_not_apply_formal_50x6_bootstrap() -> None:
    records = []
    for query_index in range(8):
        for true_rule in ("passable", "blocked"):
            for condition in (
                "observed_passable",
                "observed_blocked",
                "did_not_attempt_crossing",
            ):
                matching = condition == f"observed_{true_rule}"
                records.append(
                    {
                        "query_id": f"q{query_index}/{true_rule}",
                        "static_query_id": f"q{query_index}",
                        "eval_seed": 242,
                        "evaluation_index": query_index,
                        "direction": (
                            "left_to_right"
                            if query_index < 4
                            else "right_to_left"
                        ),
                        "true_rule": true_rule,
                        "history_condition": condition,
                        "true_next_frame_latent_mse": (
                            0.1 if matching else 0.2
                        ),
                        "true_target_closer": matching,
                        "two_target_margin": 0.1 if matching else -0.1,
                    }
                )
    summary = _tiny_diagnostic_summary(
        records,
        expected_static_queries=8,
    )
    assert summary["formal_50x6_gate_applied"] is False
    assert summary["unique_static_queries"] == 8
    assert (
        summary["by_true_rule"]["passable"]["overall"][
            "same_history_two_target_accuracy"
        ]
        == 1.0
    )


def test_frozen_representation_contract_keeps_state_exact() -> None:
    import torch

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(2, 2),
                torch.nn.BatchNorm1d(2),
            )
            self.predictor = torch.nn.Linear(2, 2)

    specification = _frozen_module_spec(
        FROZEN_REPRESENTATION_TRAINING_CONFIG
    )
    model = Model()
    audit = _apply_frozen_modules(model, specification)
    with torch.no_grad():
        model.predictor.weight.add_(1.0)
    finalized = _finalize_frozen_modules(model, audit)

    assert finalized["passed"] is True
    assert finalized["state_unchanged"] == {
        "encoder": True,
        "projector": True,
    }
    assert all(
        not parameter.requires_grad
        for module in (model.encoder, model.projector)
        for parameter in module.parameters()
    )
    assert any(parameter.requires_grad for parameter in model.predictor.parameters())
