from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from contextworld.evaluation.icl_model import file_sha256
from scripts.analyze_tworoom_door_ability import (
    ExpectedModel,
    FrozenProtocol,
    _assert_same_planning_queries,
    _assert_unique_checkpoint_hashes,
    _audit_training_report,
    _is_formal_analysis,
    _load_planning_model,
    _load_rollout_model,
    _paired_bootstrap,
    _planning_comparison,
    _rollout_comparison,
)


def _model(
    tmp_path: Path,
    *,
    slug: str = "h3_door_multi_v2_s3072",
    training_seed: int = 3072,
) -> ExpectedModel:
    checkpoint = tmp_path / slug / "weights_final_step_12840.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes((slug + "-checkpoint").encode())
    return ExpectedModel(
        group="multi_door_target",
        training_seed=training_seed,
        slug=slug,
        model_id="M_door_multi_v2",
        checkpoint=checkpoint,
        report_name=f"{slug}.json",
        expected_steps=12840,
        training_groups={"original": 0.5, "door_multi_v2": 0.5},
    )


def _planning_entry(index: int) -> dict:
    return {
        "evaluation_id": f"q{index}",
        "evaluation_index": index,
        "eval_seed": 42,
        "source_kind": "original_h5",
        "source_path": "artifacts/source.h5",
        "episode": 100 + index,
        "start_step": index,
        "goal_offset": 25,
        "cem_group_seed": 42,
        "stratum": "original_future25",
    }


def _rollout_entry(index: int, domain: str = "original_heldout") -> dict:
    return {
        "evaluation_id": f"r-{domain}-{index}",
        "evaluation_index": index,
        "eval_seed": 42,
        "source_kind": "original_h5",
        "source_path": "artifacts/source.h5",
        "episode": 200 + index,
        "start_step": index,
        "domain": domain,
    }


def _frozen(tmp_path: Path, *, resamples: int = 200) -> FrozenProtocol:
    protocol = tmp_path / "ability.yaml"
    normalizer = tmp_path / "normalizer.json"
    planning = tmp_path / "planning_catalog.json"
    rollout = tmp_path / "rollout_catalog.json"
    protocol.write_text("protocol: test\n", encoding="utf-8")
    normalizer.write_text("{}", encoding="utf-8")
    planning.write_text("{}", encoding="utf-8")
    rollout.write_text("{}", encoding="utf-8")
    planning_entries = {_planning_entry(index)["evaluation_id"]: _planning_entry(index) for index in range(2)}
    rollout_entries = {_rollout_entry(index)["evaluation_id"]: _rollout_entry(index) for index in range(2)}
    return FrozenProtocol(
        protocol_path=protocol,
        protocol_sha256=file_sha256(protocol),
        stable_worldmodel_commit="a" * 40,
        eval_seeds=(42,),
        evaluations_per_seed=2,
        planning_parameters={
            "eval_budget": 50,
            "horizon": 5,
            "receding_horizon": 5,
            "cem_samples": 300,
            "cem_steps": 30,
            "cem_topk": 30,
        },
        horizons=(1, 2, 3, 5),
        bootstrap_seed=3072,
        bootstrap_resamples=resamples,
        confidence_level=0.95,
        success_margin_pp=-5.0,
        distance_margin_px=5.0,
        require_no_stratum_collapse=True,
        normalizer=normalizer,
        normalizer_sha256=file_sha256(normalizer),
        planning_catalog=planning,
        planning_catalog_sha256=file_sha256(planning),
        planning_entries_by_seed={42: planning_entries},
        rollout_catalog=rollout,
        rollout_catalog_sha256=file_sha256(rollout),
        rollout_entries_by_domain={"original_heldout": rollout_entries},
    )


def _binding(model: ExpectedModel) -> dict:
    return {"checkpoint_sha256": file_sha256(model.checkpoint)}


def _common_result(model: ExpectedModel, frozen: FrozenProtocol, *, rollout: bool) -> dict:
    catalog = frozen.rollout_catalog if rollout else frozen.planning_catalog
    catalog_hash = frozen.rollout_catalog_sha256 if rollout else frozen.planning_catalog_sha256
    return {
        "status": "passed",
        "checkpoint": {"path": str(model.checkpoint), "sha256": file_sha256(model.checkpoint)},
        "normalizer": {"path": str(frozen.normalizer), "sha256": frozen.normalizer_sha256},
        "catalog": {"path": str(catalog), "sha256": catalog_hash},
        "stable_worldmodel": {"commit": frozen.stable_worldmodel_commit},
        "frozen_weight_audit": {
            "passed": True,
            "state_dict_sha256_before": "weights",
            "state_dict_sha256_after": "weights",
        },
    }


def _planning_record(index: int, *, success: bool = True, distance: float = 2.0) -> dict:
    return {
        **_planning_entry(index),
        "room_relation": "cross_room",
        "initial_state": [10.0, 20.0],
        "goal_state": [180.0, 20.0],
        "final_state": [178.0, 20.0],
        "success": success,
        "final_distance": distance,
    }


def _rollout_record(index: int, *, offset: float = 0.0) -> dict:
    return {
        **_rollout_entry(index),
        "horizons": {
            str(horizon): {
                "latent_mse": offset + 0.01 * horizon,
                "latent_rmse": offset + 0.02 * horizon,
                "latent_cosine_distance": offset + 0.03 * horizon,
            }
            for horizon in (1, 2, 3, 5)
        },
    }


def test_training_report_is_bound_to_actual_unique_checkpoint(tmp_path: Path) -> None:
    model = _model(tmp_path)
    report = {
        "passed": True,
        "save_load_exact": True,
        "model_id": model.model_id,
        "run_name": model.slug,
        "data": {"seed": 3072, "group_weights": model.training_groups},
        "training": {
            "training_complete": True,
            "global_step": 12840,
            "expected_optimizer_steps": 12840,
            "plan": {
                "data_split_seed": 3072,
                "training_seed": model.training_seed,
                "optimizer_steps_total": 12840,
            },
        },
        "stable_worldmodel": {"commit": "a" * 40},
        "artifacts": {
            "pretrained": str(model.checkpoint),
            "pretrained_sha256": file_sha256(model.checkpoint),
        },
    }
    path = tmp_path / model.report_name
    path.write_text(json.dumps(report), encoding="utf-8")
    binding = _audit_training_report(
        model,
        path,
        stable_worldmodel_commit="a" * 40,
        expected_data_split_seed=3072,
    )
    assert binding["checkpoint_sha256"] == file_sha256(model.checkpoint)
    with pytest.raises(RuntimeError, match="same checkpoint hash"):
        _assert_unique_checkpoint_hashes({"a": binding, "b": dict(binding)})
    report["artifacts"]["pretrained_sha256"] = "stale"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Training-report binding failed"):
        _audit_training_report(
            model,
            path,
            stable_worldmodel_commit="a" * 40,
            expected_data_split_seed=3072,
        )


@pytest.mark.parametrize("training_seed", [3072, 4096, 5120])
def test_training_report_keeps_data_and_training_seeds_separate(
    tmp_path: Path, training_seed: int
) -> None:
    model = _model(
        tmp_path,
        slug=f"h3_door_multi_v2_s{training_seed}",
        training_seed=training_seed,
    )
    report = {
        "passed": True,
        "save_load_exact": True,
        "model_id": model.model_id,
        "run_name": model.slug,
        "data": {"seed": 3072, "group_weights": model.training_groups},
        "training": {
            "training_complete": True,
            "global_step": 12840,
            "expected_optimizer_steps": 12840,
            "plan": {
                "data_split_seed": 3072,
                "training_seed": training_seed,
                "optimizer_steps_total": 12840,
            },
        },
        "stable_worldmodel": {"commit": "a" * 40},
        "artifacts": {
            "pretrained": str(model.checkpoint),
            "pretrained_sha256": file_sha256(model.checkpoint),
        },
    }
    path = tmp_path / model.report_name
    path.write_text(json.dumps(report), encoding="utf-8")

    binding = _audit_training_report(
        model,
        path,
        stable_worldmodel_commit="a" * 40,
        expected_data_split_seed=3072,
    )

    assert binding["checks"]["data_seed"]
    assert binding["checks"]["plan_data_split_seed"]
    assert binding["checks"]["training_seed"]


def test_planning_loader_requires_exact_query_and_cem_schedule(tmp_path: Path) -> None:
    model = _model(tmp_path)
    frozen = _frozen(tmp_path)
    result = _common_result(model, frozen, rollout=False)
    result.update(
        {
            "protocol": {
                "action_block": 5,
                "history_size": 3,
                "eval_seed": 42,
                "evaluations": 2,
                **frozen.planning_parameters,
            },
            "aggregate": {"evaluations": 2},
            "raw_records": [_planning_record(index) for index in range(2)],
        }
    )
    path = tmp_path / model.slug / "planning_original_heldout" / "seed42.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    records, _ = _load_planning_model(tmp_path, model, _binding(model), frozen)
    assert len(records) == 2
    result["raw_records"][0]["cem_group_seed"] = 999
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="query/CEM binding changed"):
        _load_planning_model(tmp_path, model, _binding(model), frozen)


def test_rollout_loader_requires_complete_original_heldout_horizons(tmp_path: Path) -> None:
    model = _model(tmp_path)
    frozen = _frozen(tmp_path)
    result = _common_result(model, frozen, rollout=True)
    result.update(
        {
            "protocol": {
                "action_block": 5,
                "history_size": 3,
                "horizons_action_blocks": [1, 2, 3, 5],
            },
            "raw_records": [_rollout_record(index) for index in range(2)],
        }
    )
    path = tmp_path / model.slug / "rollout_error.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    domains, _ = _load_rollout_model(tmp_path, model, _binding(model), frozen)
    assert len(domains["original_heldout"]) == 2
    result["raw_records"].pop()
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="partially present"):
        _load_rollout_model(tmp_path, model, _binding(model), frozen)


def test_planning_noninferiority_uses_paired_ci_and_stratum_collapse(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path, resamples=300)
    reference = {f"q{index}": _planning_record(index) for index in range(2)}
    identical = {key: dict(row) for key, row in reference.items()}
    passed = _planning_comparison(reference, identical, frozen=frozen, bootstrap_seed=3)
    assert passed["passed"]
    assert passed["candidate_minus_original_success_rate_percentage_points"]["point"] == 0.0

    failed = {
        key: {**row, "success": False, "final_distance": 20.0}
        for key, row in reference.items()
    }
    result = _planning_comparison(reference, failed, frozen=frozen, bootstrap_seed=3)
    assert not result["passed"]
    assert not result["gates"]["success_rate_non_inferior"]
    assert not result["gates"]["final_distance_non_inferior"]
    assert not result["gates"]["no_solvable_stratum_collapse"]


def test_rollout_reports_all_true_future_horizons_but_is_not_a_gate(tmp_path: Path) -> None:
    frozen = _frozen(tmp_path, resamples=100)
    reference = {f"r-original_heldout-{index}": _rollout_record(index) for index in range(2)}
    candidate = {
        key: _rollout_record(index, offset=0.01)
        for index, key in enumerate(sorted(reference))
    }
    result = _rollout_comparison(reference, candidate, frozen=frozen, bootstrap_seed=5)
    assert set(result["horizons"]) == {"h1", "h2", "h3", "h5"}
    assert result["formal_noninferiority_gate"] is False
    for row in result["horizons"].values():
        assert row["candidate_minus_original_latent_mse"]["point"] == pytest.approx(0.01)


def test_bootstrap_is_deterministic_and_partial_can_never_be_formal() -> None:
    values = np.asarray([-1.0, 0.0, 1.0])
    first = _paired_bootstrap(values, seed=7, resamples=100, confidence=0.95)
    second = _paired_bootstrap(values, seed=7, resamples=100, confidence=0.95)
    assert first == second
    assert _is_formal_analysis(complete_matrix=True, allow_partial=False, comparison_count=6)
    assert not _is_formal_analysis(complete_matrix=True, allow_partial=True, comparison_count=6)
    assert not _is_formal_analysis(complete_matrix=False, allow_partial=False, comparison_count=6)


def test_pairing_rejects_changed_query_even_when_ids_match() -> None:
    reference = {"q0": _planning_record(0)}
    candidate = {"q0": {**_planning_record(0), "cem_group_seed": 99}}
    with pytest.raises(RuntimeError, match="Paired planning query changed"):
        _assert_same_planning_queries(reference, candidate)
