from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from contextworld.evaluation.icl_model import file_sha256
from scripts import analyze_tworoom_door_planning as analysis


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"


def test_planning_training_identity_uses_training_seed_not_data_seed(
    tmp_path: Path,
) -> None:
    report_paths = []
    for training_seed in (3072, 4096, 5120):
        expected = analysis.TRAINING_BINDINGS[
            ("multi_door_target", training_seed)
        ]
        report = {
            "passed": True,
            "save_load_exact": True,
            "model_id": expected["model_id"],
            "run_name": expected["run_name"],
            "data": {"seed": 3072},
            "training": {
                "training_complete": True,
                "plan": {
                    "data_split_seed": 3072,
                    "training_seed": training_seed,
                },
            },
            "stable_worldmodel": {"commit": "stable"},
            "artifacts": {
                "pretrained_sha256": f"checkpoint-{training_seed}"
            },
        }
        path = tmp_path / expected["report_name"]
        path.write_text(json.dumps(report), encoding="utf-8")
        report_paths.append(path)

    identities, paths, commit = analysis._load_training_identities(
        report_paths,
        expected_data_split_seed=3072,
        require_complete=False,
    )

    assert set(paths) == {
        ("multi_door_target", 3072),
        ("multi_door_target", 4096),
        ("multi_door_target", 5120),
    }
    assert set(identities) == {
        "checkpoint-3072",
        "checkpoint-4096",
        "checkpoint-5120",
    }
    assert commit == "stable"


def test_paired_bootstrap_respects_metric_direction_and_ties() -> None:
    rng = np.random.default_rng(3)
    indices = rng.integers(0, 4, size=(100, 4))
    lower = analysis._bootstrap_delta_summary(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([2.0, 2.0, 4.0, 3.0]),
        indices=indices,
        higher_is_better=False,
    )
    assert lower["target_better_pairs"] == 2
    assert lower["fixed_door_control_better_pairs"] == 1
    assert lower["ties"] == 1

    higher = analysis._bootstrap_delta_summary(
        np.asarray([0.5, 0.2, 0.1, 0.4]),
        np.asarray([0.4, 0.2, 0.3, 0.1]),
        indices=indices,
        higher_is_better=True,
    )
    assert higher["target_better_pairs"] == 2
    assert higher["fixed_door_control_better_pairs"] == 1
    assert higher["ties"] == 1


def test_model_summary_reports_requested_fixed_and_cem_metrics() -> None:
    common = {
        "eval_seed": 42,
        "success": True,
        "doorway_crossing": True,
        "final_distance_px": 4.0,
        "steps_to_success": 17,
    }
    fixed = analysis._aggregate_model_records(
        [
            {
                **common,
                "exact_environment_endpoint_regret_px": 2.0,
                "predicted_cost_vs_true_endpoint_distance_spearman": 0.7,
            },
            {
                **common,
                "success": False,
                "steps_to_success": None,
                "final_distance_px": 14.0,
                "exact_environment_endpoint_regret_px": 4.0,
                "predicted_cost_vs_true_endpoint_distance_spearman": 0.3,
            },
        ],
        mode="fixed",
    )
    assert fixed["success_rate_percent"] == 50.0
    assert fixed["endpoint_regret_px_lower_is_better"]["mean"] == 3.0
    assert fixed["tie_aware_spearman_higher_is_better"]["mean"] == 0.5

    planning = analysis._aggregate_model_records([common], mode="planning")
    assert planning["doorway_crossing_rate_percent"] == 100.0
    assert planning["steps_to_success_success_only"]["mean"] == 17.0
    assert "tie_aware_spearman_higher_is_better" not in planning


def test_complete_matrix_is_exactly_seven_models_by_track_door_seed() -> None:
    tracks = {
        "validation_seen": (49, 89, 129, 169),
        "validation_interpolation": (53, 85, 117, 149),
    }
    seeds = (42, 43, 44, 45, 46, 47)
    rows = []
    for binding in analysis.TRAINING_BINDINGS.values():
        for track, doors in tracks.items():
            for door in doors:
                for seed in seeds:
                    rows.append(
                        {
                            "model": {"slug": binding["run_name"]},
                            "track": track,
                            "door_position": door,
                            "eval_seed": seed,
                            "records": [{}] * 50,
                        }
                    )
    audit = analysis._audit_matrix(
        rows, tracks=tracks, eval_seeds=seeds, require_complete=True
    )
    assert audit["complete_formal_matrix"]
    assert audit["expected_result_files"] == 7 * 8 * 6
    assert audit["observed_records"] == 7 * 8 * 6 * 50

    with pytest.raises(RuntimeError, match="incomplete"):
        analysis._audit_matrix(
            rows[:-1], tracks=tracks, eval_seeds=seeds, require_complete=True
        )


def _fixed_signature_row() -> dict:
    return {
        "query_id": "q0",
        "evaluation_id": "s42-e000-q0",
        "evaluation_index": 0,
        "direction": "left_to_right",
        "door_relative_vertical_offset_px": 0,
        "history_pixels_sha256": "pixels",
        "history_actions_sha256": "actions",
        "candidate_bank_raw_sha256": "raw-bank",
        "candidate_bank_normalized_sha256": "normalized-bank",
    }


def test_pairing_audit_rejects_candidate_or_cem_seed_changes() -> None:
    first = {
        "model": {"slug": "first"},
        "track": "validation_seen",
        "door_position": 49,
        "eval_seed": 42,
        "records": [_fixed_signature_row()],
    }
    second = {
        **first,
        "model": {"slug": "second"},
        "records": [{**_fixed_signature_row(), "candidate_bank_raw_sha256": "changed"}],
    }
    with pytest.raises(RuntimeError, match="pairing failed"):
        analysis._audit_cross_model_pairing(
            [first, second], mode="fixed", require_complete=False
        )

    planning = {
        **_fixed_signature_row(),
        "cem_seed": 7,
        "cem_rng_state_sha256_before": "rng",
        "fixed_context": {"normalized_actions_sha256": "normalized-actions"},
    }
    first["records"] = [planning]
    second["records"] = [{**planning, "cem_seed": 8}]
    with pytest.raises(RuntimeError, match="pairing failed"):
        analysis._audit_cross_model_pairing(
            [first, second], mode="planning", require_complete=False
        )


def _write_partial_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    normalizer = tmp_path / "normalizer.json"
    normalizer.write_text("{}", encoding="utf-8")
    stable_commit = "stable-commit"
    checkpoint_hash = "checkpoint-fixed-3072"
    binding = analysis.TRAINING_BINDINGS[("fixed_door_control", 3072)]
    report = {
        "passed": True,
        "save_load_exact": True,
        "model_id": binding["model_id"],
        "run_name": binding["run_name"],
        "data": {"seed": 3072},
        "training": {
            "training_complete": True,
            "plan": {
                "data_split_seed": 3072,
                "training_seed": 3072,
            },
        },
        "stable_worldmodel": {"commit": stable_commit},
        "artifacts": {"pretrained_sha256": checkpoint_hash},
    }
    report_path = tmp_path / binding["report_name"]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    bundles = []
    for index in range(50):
        direction = "left_to_right" if index < 25 else "right_to_left"
        query_id = f"q{index}"
        bundles.append(
            {
                "query_id": query_id,
                "track": "validation_seen",
                "task": "cross_room_navigation",
                "eval_seed": 42,
                "evaluation_index": index,
                "query_factors": {"agent.speed": 5.0, "door.position": 49},
                "door_position": 49,
                "direction": direction,
                "door_relative_vertical_offset_px": (0, 20, 40)[index % 3],
                "cem_seed": 1000 + index,
                "history_pixels_sha256": f"pixels-{index}",
                "history_actions_sha256": f"actions-{index}",
                "fixed_candidate_raw_actions_sha256": f"bank-{index}",
            }
        )
    catalog = {
        "status": "smoke_only",
        "split_role": "validation",
        "config": {"sha256": file_sha256(CONFIG)},
        "stable_worldmodel": {"commit": stable_commit},
        "protocol": {
            "agent_speed": 5.0,
            "task": "cross_room_navigation",
            "history_tokens": 3,
            "action_block_raw_steps": 5,
            "candidates_per_query": 300,
            "candidate_horizon_action_blocks": 10,
            "eval_seeds": [42, 43, 44, 45, 46, 47],
            "queries_per_door_per_eval_seed": 50,
        },
        "bundles": bundles,
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    rows = []
    for bundle in bundles:
        index = bundle["evaluation_index"]
        query_id = bundle["query_id"]
        rows.append(
            {
                "evaluation_id": f"s42-e{index:03d}-{query_id}",
                "evaluation_index": index,
                "eval_seed": 42,
                "query_id": query_id,
                "track": "validation_seen",
                "task": "cross_room_navigation",
                "door_position": 49,
                "agent_speed": 5.0,
                "direction": bundle["direction"],
                "door_relative_vertical_offset_px": bundle[
                    "door_relative_vertical_offset_px"
                ],
                "history_pixels_sha256": bundle["history_pixels_sha256"],
                "history_actions_sha256": bundle["history_actions_sha256"],
                "candidate_bank_raw_sha256": bundle[
                    "fixed_candidate_raw_actions_sha256"
                ],
                "candidate_bank_normalized_sha256": f"normalized-{index}",
                "exact_environment_endpoint_regret_px": 1.0,
                "predicted_cost_vs_true_endpoint_distance_spearman": 0.5,
                "success": True,
                "final_distance_px": 2.0,
                "steps_to_success": 10,
                "doorway_crossing": True,
            }
        )
    result = {
        "benchmark": "tworoom_door_fixed_candidates_v1",
        "status": "passed",
        "evidence_role": "planning_action_ranking_not_latent_accuracy",
        "run_kind": "confirmation",
        "track": "validation_seen",
        "door_position": 49,
        "eval_seed": 42,
        "config": {"sha256": file_sha256(CONFIG)},
        "catalog": {
            "path": str(catalog_path.resolve()),
            "sha256": file_sha256(catalog_path),
            "cell_audit": {"passed": True},
        },
        "model": {"checkpoint": "checkpoint.pt", "sha256": checkpoint_hash},
        "normalizer": {
            "path": str(normalizer),
            "sha256": file_sha256(normalizer),
        },
        "stable_worldmodel": {"commit": stable_commit},
        "protocol": {
            "history_size": 3,
            "action_block": 5,
            "queries": 50,
            "candidates_per_query": 300,
            "horizon_action_blocks": 10,
            "horizon_raw_steps": 50,
            "same_frozen_candidate_bank_across_models": True,
            "agent_speed": 5.0,
        },
        "frozen_weight_audit": {
            "state_dict_sha256_before": "state",
            "state_dict_sha256_after": "state",
            "passed": True,
        },
        "count_audit": {"records": 50, "expected_records": 50, "passed": True},
        "records": rows,
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return normalizer, report_path, catalog_path, result_path


def test_partial_end_to_end_is_never_formal(tmp_path: Path, monkeypatch) -> None:
    artifact_base = tmp_path / "artifacts"
    monkeypatch.setenv("CONTEXTWORLD_ARTIFACT_ROOT", str(artifact_base))
    normalizer, report, catalog, result = _write_partial_inputs(tmp_path)
    canonical_root = (
        artifact_base
        / "evaluation/history3/door_visual_generalization_v1"
    )
    canonical_catalog = canonical_root / "planning/validation/catalog.json"
    canonical_catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.replace(canonical_catalog)
    canonical_result = canonical_root / "fixed_results/partial.json"
    canonical_result.parent.mkdir(parents=True, exist_ok=True)
    result.replace(canonical_result)
    catalog = canonical_catalog
    result = canonical_result
    moved_payload = json.loads(result.read_text(encoding="utf-8"))
    moved_payload["catalog"]["path"] = str(catalog)
    result.write_text(json.dumps(moved_payload), encoding="utf-8")
    args = analysis.parse_args(
        [
            "--mode",
            "fixed",
            "--allow-partial",
            "--catalog",
            str(catalog),
            "--results",
            str(result),
            "--training-reports",
            str(report),
            "--expected-normalizer",
            str(normalizer),
            "--output",
            str(tmp_path / "summary.json"),
            "--bootstrap-samples",
            "20",
        ]
    )
    summary = analysis.run(args)
    assert summary["status"] == "partial_analysis_only"
    assert not summary["formal_analysis"]
    assert summary["interpretation"]["partial_results_are_never_formal"]
    assert summary["by_track_door_model"]["validation_seen"]["doors"]["49"]

    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["records"][0]["candidate_bank_raw_sha256"] = "wrong"
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Candidate bank/catalog mismatch"):
        analysis.run(args)
