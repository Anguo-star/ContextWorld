from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from contextworld.benchmarks.action_strength_icl_data import (
    ACTION_STRENGTH_RELEASE_ID,
    _action_strength_pair_causal_coverage,
    _strict_causal_manifest_checks,
    action_strength_icl_evaluation_plans,
    audit_action_strength_icl_release,
    load_action_strength_icl_release,
)
from contextworld.benchmarks.action_strength_icl_score import (
    _prediction_contract_sha256,
    _prediction_gate,
    _prediction_metrics,
    score_action_strength_retention_report,
)


def test_prediction_contract_is_stable_when_reference_results_are_added() -> None:
    release = load_action_strength_icl_release()
    original = _prediction_contract_sha256(release)

    with_results = copy.deepcopy(release)
    with_results["reference_results"] = {
        "new_three_seed_result": {
            "path": "artifacts/new-result.json",
            "sha256": "1" * 64,
        }
    }
    with_results["reference_method"] = {
        "status": "passed",
        "formal_three_seed_method_claim": True,
    }
    assert _prediction_contract_sha256(with_results) == original

    changed_gate = copy.deepcopy(release)
    changed_gate["scoring"]["hidden_future_prediction"]["gates"][
        "correct_future_rate_minimum"
    ] = 0.99
    assert _prediction_contract_sha256(changed_gate) != original


def test_release_uses_self_explanatory_action_strength_name() -> None:
    release = load_action_strength_icl_release()
    assert release["release_id"] == ACTION_STRENGTH_RELEASE_ID
    assert release["scope"]["capability"] == (
        "infer_action_strength_from_recent_interaction"
    )
    assert release["scope"]["display_name_zh"] == "PushT 推手移动幅度 ICL"
    assert release["scope"]["strength_values"] == [60, 140]
    assert release["scope"]["public_test_included"] is True
    assert release["scope"]["sealed_test_included"] is False
    assert release["training"]["artifact_tree"]["root"] == (
        "artifacts/synthesis/pusht_action_strength_h3_release_v1"
    )
    assert release["evaluation"]["artifact_tree"]["root"] == (
        "artifacts/evaluation/history3/"
        "pusht_action_strength_h3_public_test_v1"
    )
    assert release["evaluation"]["planning_oracle"][
        "causal_execution"
    ] == "replay_x0_to_x1_to_x2_before_each_candidate"
    assert set(release["reference_results"]) == {
        "latent_response_summary",
        "reference_method_summary",
        "strict_causal_data_compatibility_audit",
        "strict_result_compatibility_audit",
    }
    assert release["reference_method"]["training_seeds"] == [
        13313,
        13314,
        13315,
    ]
    assert release["reference_method"]["artifact_tree"] == {
        "root": (
            "artifacts/evaluation/history3/"
            "pusht_action_strength_h3_release_v1"
        ),
        "files": 26,
        "bytes": 217250121,
        "sha256": (
            "c9cc76d0e0a5a67c53ead868581bb56a"
            "42ef6fc32a5ae0c09d2728bb65f8eaef"
        ),
    }
    for specification in (
        *release["training"]["upstream"].values(),
        release["training"]["initialization"],
    ):
        assert "source_symbol" in specification
        assert "local_source" not in specification
    for name, specification in release["reference_results"].items():
        assert set(specification) >= {"path", "sha256"}, name
        assert specification["path"] == specification["path"].strip(), name
        assert len(specification["sha256"]) == 64, name


def test_reference_method_package_is_exact_and_self_contained() -> None:
    release = load_action_strength_icl_release()
    repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / release["reference_method"]["artifact_tree"][
        "root"
    ]
    root_files = {
            "action_planning_oracle.json",
            "action_planning_oracle_report.json",
            "latent_response_summary.json",
            "reference_method_summary.json",
        "reference_training_scales.json",
        "standard_pusht_query_catalog.json",
        "strict_causal_data_compatibility_audit.json",
        "strict_result_compatibility_audit.json",
    }
    seed_files = {
        "checkpoint.pt",
        "model_config.json",
        "training_report.json",
        "prediction_result.json",
        "action_planning_result.json",
        "standard_pusht_cem_result.json",
    }
    expected = set(root_files)
    for seed in (13313, 13314, 13315):
        expected.update(f"seed_{seed}/{name}" for name in seed_files)
    observed = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
    }
    assert observed == expected

    forbidden = (
        "/opt/",
        "data/world_model/context_world/evaluation/history3/"
        "pusht_hidden_actuation",
        "research",
    )
    checkpoints = {
        f"seed_{seed}/checkpoint.pt" for seed in (13313, 13314, 13315)
    }
    for relative in sorted(observed - checkpoints):
        text = (package_root / relative).read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), relative

    summary = json.loads(
        (package_root / "reference_method_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(summary["per_seed"]) == {"13313", "13314", "13315"}
    assert summary["aggregate"] == {
        "correct_future_rate_mean": 0.9661458333333334,
        "correct_history_rate_mean": 0.984375,
        "rule_switch_rate_mean": 0.99609375,
        "worst_strength_correct_future_rate_mean": 0.9440104166666666,
        "correct_action_region_rate_mean": 0.9733072916666666,
        "standard_pusht_cem_successes": 672,
        "standard_pusht_cem_evaluations": 900,
        "all_three_seeds_passed": True,
    }


def test_reference_package_binds_portable_training_and_public_test() -> None:
    release = load_action_strength_icl_release()
    repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / release["reference_method"]["artifact_tree"][
        "root"
    ]
    training_manifest = release["training"]["manifest_sha256"]
    public_manifest = release["evaluation"]["manifest_sha256"]
    training_receipt = release["training"]["artifacts"][
        "portability_receipt"
    ]["sha256"]
    public_receipt = release["evaluation"]["artifacts"][
        "portability_receipt"
    ]["sha256"]

    summary = json.loads(
        (package_root / "reference_method_summary.json").read_text()
    )
    assert summary["data_binding"]["training_manifest_sha256"] == (
        training_manifest
    )
    assert summary["data_binding"][
        "training_portability_receipt_sha256"
    ] == training_receipt
    assert summary["data_binding"]["public_test_manifest_sha256"] == (
        public_manifest
    )
    assert summary["data_binding"][
        "public_test_portability_receipt_sha256"
    ] == public_receipt

    scales = json.loads(
        (package_root / "reference_training_scales.json").read_text()
    )
    assert scales["formal_training_manifest_sha256"] == training_manifest
    assert scales["formal_training_portability_receipt_sha256"] == (
        training_receipt
    )

    causal = json.loads(
        (package_root / "strict_causal_data_compatibility_audit.json").read_text()
    )
    assert causal["formal_manifests"] == {
        "training_and_development": training_manifest,
        "public_test": public_manifest,
    }
    assert causal["formal_portability_receipts"] == {
        "training_and_development": training_receipt,
        "public_test": public_receipt,
    }

    oracle = json.loads(
        (package_root / "action_planning_oracle_report.json").read_text()
    )
    assert oracle["public_test_manifest_sha256"] == public_manifest
    assert oracle["public_test_portability_receipt_sha256"] == public_receipt

    for seed in (13313, 13314, 13315):
        seed_root = package_root / f"seed_{seed}"
        training = json.loads((seed_root / "training_report.json").read_text())
        prediction = json.loads(
            (seed_root / "prediction_result.json").read_text()
        )
        planning = json.loads(
            (seed_root / "action_planning_result.json").read_text()
        )
        assert training["data"]["formal_training_manifest_sha256"] == (
            training_manifest
        )
        assert training["data"][
            "formal_training_portability_receipt_sha256"
        ] == training_receipt
        assert prediction["training_manifest_sha256"] == training_manifest
        assert prediction["training_portability_receipt_sha256"] == (
            training_receipt
        )
        assert prediction["public_test_manifest_sha256"] == public_manifest
        assert prediction["public_test_portability_receipt_sha256"] == (
            public_receipt
        )
        assert planning["public_test_manifest_sha256"] == public_manifest
        assert planning["public_test_portability_receipt_sha256"] == (
            public_receipt
        )


def test_release_audit_checks_all_three_current_reference_seeds() -> None:
    audit = audit_action_strength_icl_release(full=False)
    assert audit["passed"] is True
    assert audit["reference_method"]["passed"] is True
    assert audit["reference_method"]["checks"][
        "summary_gates_match_config"
    ] is True
    assert audit["reference_method"]["checks"][
        "latent_response_summary_matches_config"
    ] is True
    assert set(audit["reference_method"]["per_seed"]) == {
        "13313",
        "13314",
        "13315",
    }
    assert all(
        row["passed"]
        for row in audit["reference_method"]["per_seed"].values()
    )
    assert audit["reference_method"]["contaminated_files"] == {}
    assert audit["reference_method"]["config_contamination"] == []
    assert audit["artifact_trees"]["reference_method"]["passed"] is True
    assert all(
        row["required_for_release_audit"] is False and row["passed"] is True
        for row in audit["upstream_inputs"].values()
    )


def test_legacy_positive_result_does_not_pass_current_gate() -> None:
    release = load_action_strength_icl_release()
    legacy = release["reference_method"]["per_seed"][13313]
    metrics = {
        name: legacy[name]
        for name in (
            "correct_future_rate",
            "correct_history_rate",
            "rule_switch_rate",
            "worst_strength_correct_future_rate",
        )
    }
    gate = _prediction_gate(metrics, release=release)
    assert gate["passed"] is False
    assert gate["checks"]["target_latent_separation"] is False
    assert gate["checks"]["response_gain"] is False
    assert gate["checks"]["normalized_response_error"] is False


def test_strict_causal_manifest_checks_are_hard_gates() -> None:
    strict = {
        "passed": True,
        "state_installations_after_x0": 0,
        "query_simulator_recreated": False,
        "full_state_dimensions": 12,
        "max_pair_full_state_gap": 5.0e-6,
        "full_state_tolerance": 1.0e-5,
        "max_pair_query_pixel_difference": 0,
        "max_pair_query_action_difference": 0.0,
    }
    manifest = {
        "strict_causal_chain_audit": strict,
        "splits": {
            "train": {"strict_causal_chain_audit": copy.deepcopy(strict)},
            "validation": {
                "strict_causal_chain_audit": copy.deepcopy(strict)
            },
        },
    }
    checks = _strict_causal_manifest_checks(
        manifest,
        splits=("train", "validation"),
        prefix="training",
    )
    assert all(checks.values())

    manifest["splits"]["validation"]["strict_causal_chain_audit"][
        "state_installations_after_x0"
    ] = 1
    failed = _strict_causal_manifest_checks(
        manifest,
        splits=("train", "validation"),
        prefix="training",
    )
    assert failed[
        "training_validation_strict_no_state_installation_after_x0"
    ] is False


def test_pair_causal_coverage_checks_every_formal_pair() -> None:
    def pair(split: str, index: int) -> dict:
        return {
            "template": {"template_id": f"{split}-{index:05d}"},
            "audit": {
                "template_id": f"{split}-{index:05d}",
                "passed": True,
                "full_state_dimensions": 12,
                "full_state_components": [f"state-{axis}" for axis in range(12)],
                "query_physics_max_abs_gap": 5.0e-6,
                "query_physics_tolerance": 1.0e-5,
                "pair_query_pixel_difference": 0,
                "pair_query_action_difference": 0.0,
                "history_effect": 1.0,
                "true_future_effect": 1.0,
                "state_installations_after_x0": 0,
                "query_simulator_recreated": False,
                "checks": {
                    "initial_pixels_identical": True,
                    "initial_state_identical": True,
                    "low_recovery_natural": True,
                    "high_recovery_natural": True,
                    "query_physics_within_numerical_tolerance": True,
                    "query_pixels_identical": True,
                    "query_matches_initial_low": True,
                    "query_matches_initial_high": True,
                    "actions_identical": True,
                    "middle_pixels_different": True,
                    "future_pixels_different": True,
                    "no_state_installations_after_x0": True,
                },
            },
        }

    expected = {"training": 2048, "development": 256, "public_test": 256}
    groups = {
        split: [pair(split, index) for index in range(count)]
        for split, count in expected.items()
    }
    split_audits = {
        split: {"passed": True, "pair_count": count}
        for split, count in expected.items()
    }
    coverage = _action_strength_pair_causal_coverage(
        pair_groups=groups,
        split_audits=split_audits,
        expected_counts=expected,
    )
    assert coverage["passed"] is True
    assert coverage["audited_pair_count"] == 2560
    assert set(coverage["passed_pair_counts"].values()) == {2560}

    groups["public_test"][0]["audit"]["pair_query_pixel_difference"] = 1
    failed = _action_strength_pair_causal_coverage(
        pair_groups=groups,
        split_audits=split_audits,
        expected_counts=expected,
    )
    assert failed["passed"] is False
    assert failed["passed_pair_counts"]["query_rgb_exact"] == 2559


def test_prediction_metrics_require_history_conditioned_real_future() -> None:
    target_low = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    target_high = np.asarray([[2.0, 2.0], [3.0, 3.0]])
    metrics, records = _prediction_metrics(
        pair_ids=("pair-0", "pair-1"),
        predicted_low=target_low,
        predicted_high=target_high,
        target_low=target_low,
        target_high=target_high,
    )
    assert metrics["correct_future_rate"] == 1.0
    assert metrics["correct_history_rate"] == 1.0
    assert metrics["rule_switch_rate"] == 1.0
    assert metrics["worst_strength_correct_future_rate"] == 1.0
    assert metrics["joint_icl_pair_success_rate"] == 1.0
    assert len(records) == 2
    release = load_action_strength_icl_release()
    gate = _prediction_gate(metrics, release=release)
    assert gate["passed"] is True
    assert gate["checks"]["target_latent_separation"] is True
    assert gate["checks"]["response_gain"] is True
    assert gate["checks"]["normalized_response_error"] is True


def test_retention_scorer_binds_exact_300_episode_protocol(
    tmp_path: Path,
) -> None:
    release = load_action_strength_icl_release()
    expected = release["scoring"]["original_task_retention"]
    query_catalog = tmp_path / "query_catalog.json"
    query_catalog.write_text("{}\n", encoding="utf-8")
    report = {
        "protocol": {
            "eval_seeds": expected["eval_seeds"],
            "num_eval_per_seed": 50,
            "history_len": 3,
            "horizon": 5,
            "receding_horizon": 5,
            "action_block": 5,
            "cem_samples": 300,
            "cem_iterations": 30,
            "cem_topk": 30,
        },
        "query_catalog": {
            "path": str(query_catalog),
            "sha256": expected["query_catalog_sha256"],
        },
        "models": [
            {
                "model": "candidate",
                "checkpoint_sha256": "1" * 64,
                "aggregate": {
                    "success_count": 224,
                    "evaluation_count": 300,
                    "success_rate": 224 / 300,
                },
            }
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(report, sort_keys=True),
        encoding="utf-8",
    )
    score = score_action_strength_retention_report(
        report_path=report_path,
        model_name="candidate",
    )
    assert score["gate"]["passed"] is True
    assert score["score"]["difference_from_standard_only"] == -1


def test_evaluation_plan_keeps_prediction_planning_and_retention_separate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    upstream_h5 = tmp_path / "pusht.h5"
    upstream_h5.write_bytes(b"upstream")
    monkeypatch.setenv("CONTEXTWORLD_PUSHT_H5", str(upstream_h5))
    plans = action_strength_icl_evaluation_plans(
        checkpoint=checkpoint,
        model_name="candidate",
        output_root=tmp_path / "results",
    )
    assert set(plans["commands"]) == {
        "action_strength_planning",
        "standard_pusht_retention",
    }
    assert "score-planning" in plans["commands"][
        "action_strength_planning"
    ]["score_command"]
    assert "score-retention" in plans["commands"][
        "standard_pusht_retention"
    ]["score_command"]
