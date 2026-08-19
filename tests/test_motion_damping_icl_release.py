from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from contextworld.benchmarks import motion_damping_icl_cli
from contextworld.benchmarks.adapters import AdapterProtocol
from contextworld.benchmarks.motion_damping_icl_data import (
    MOTION_DAMPING_RELEASE_ID,
    MotionDampingICLDevelopmentDataset,
    audit_motion_damping_icl_release,
    load_motion_damping_icl_release,
    motion_damping_development_data_contract,
)
from contextworld.benchmarks.motion_damping_icl_score import (
    evaluate_motion_damping_icl_development_model,
    motion_damping_prediction_gate,
    motion_damping_prediction_metrics,
    rescore_motion_damping_icl_development_result,
)


def test_release_name_and_public_splits_are_explicit() -> None:
    release = load_motion_damping_icl_release()
    assert release["release_id"] == MOTION_DAMPING_RELEASE_ID
    assert release["scope"]["display_name_zh"] == "PushT 运动阻尼 ICL"
    assert release["scope"]["history_tokens"] == 3
    assert release["scope"]["damping_values"] == [0.2, 1.0]
    assert release["scope"]["public_test_included"] is True
    assert release["scope"]["sealed_test_included"] is False
    assert release["data"]["pair_counts"] == {
        "train": 8192,
        "loader_validation": 256,
        "validation": 256,
    }
    matrix = release["training"]["reference_matrix"]
    assert matrix["status"] == "failed_development"
    assert matrix["completed_development_seeds"] == [14321]
    assert matrix["remaining_seeds_run"] is False
    assert matrix["public_model_scoring_opened"] is False
    assert matrix["reported_endpoint"] == {
        "model_family": "LeWM",
        "recipe": "mixed_frozen_image_paired_future_ranking_twin_1p00",
        "training_seed": 14321,
        "optimizer_step": 8192,
    }
    assert release["scoring"]["hidden_future_prediction"]["gates"] == {
        "correct_future_rate_minimum": 0.95,
        "correct_history_rate_minimum": 0.95,
        "context_switch_rate_minimum": 0.95,
        "worst_damping_correct_future_rate_minimum": 0.90,
        "target_latent_separation_required": True,
        "response_gain_minimum": 0.50,
        "normalized_response_error_strict_maximum": 1.00,
    }


def test_release_data_and_public_test_are_auditable() -> None:
    audit = audit_motion_damping_icl_release(full=False)
    assert audit["passed"]
    assert audit["causal_data_contract"]["passed"]
    assert audit["causal_data_contract"]["x0_policy"] == (
        "balanced_visible_start"
    )
    total_pairs = sum(audit["counts"].values())
    assert audit["causal_data_contract"]["evidence_scope"] == (
        f"all_{total_pairs}_pairs_and_{2 * total_pairs}_clean_replays"
    )
    assert audit["row_counts"] == {
        "train": 327680,
        "loader_validation": 10240,
        "validation": 10240,
    }
    assert audit["data_checks"][
        "frozen_evaluation_split_receipts_passed"
    ]
    assert audit["data_checks"][
        "frozen_evaluation_table_hashes_preserved"
    ]
    assert audit["reference_result"]["passed"]
    assert audit["reference_result"]["status"] == "failed_development"
    assert audit["reference_result"]["failed_metrics"] == [
        "correct_history_rate"
    ]
    assert audit["reference_result"]["public_model_scoring_opened"] is False
    assert audit["reference_result"]["positive_reference_claim"] is False
    loss_identity = audit["files"]["identity.stablewm_loss"]
    assert loss_identity["required_for_release_audit"] is False
    assert loss_identity["passed"] is True
    assert loss_identity["role"] == "frozen_reference_training_provenance"


def test_compact_negative_reference_decision_is_public_safe() -> None:
    release = load_motion_damping_icl_release()
    specification = release["reference_results"]["current_decision"]
    path = Path(__file__).resolve().parents[1] / specification["path"]
    decision = json.loads(path.read_text(encoding="utf-8"))
    serialized = json.dumps(decision, sort_keys=True)
    assert "/opt/" not in serialized
    assert "candidates" not in serialized
    assert "attempt" not in serialized
    assert decision["status"] == "failed_development"
    assert decision["data_release"]["published_manifest_sha256"] == release[
        "data"
    ]["manifest_sha256"]
    receipt_path = (
        Path(__file__).resolve().parents[1]
        / release["data"]["artifacts"]["portability_receipt"]["path"]
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert decision["data_release"]["training_manifest_sha256"] == receipt[
        "metadata_sha256"
    ]["manifest.json"]["before"]
    assert decision["data_release"]["portability_receipt_sha256"] == release[
        "data"
    ]["artifacts"]["portability_receipt"]["sha256"]
    endpoint = decision["reported_endpoint"]
    assert endpoint["metrics"] == {
        "correct_future_rate": 0.974609375,
        "correct_history_rate": 0.529296875,
        "context_switch_rate": 0.9765625,
        "worst_damping_correct_future_rate": 0.97265625,
    }
    assert endpoint["failed_metrics"] == ["correct_history_rate"]
    assert endpoint["passed"] is False
    assert decision["public_model_scoring_opened"] is False
    assert decision["additional_training_seeds_run"] is False
    assert decision["original_task_cem_run"] is False
    assert decision["positive_reference_claim"] is False
    legacy_gate = motion_damping_prediction_gate(
        endpoint["metrics"], release=load_motion_damping_icl_release()
    )
    assert legacy_gate["passed"] is False
    assert legacy_gate["checks"]["target_latent_separation"] is False


def test_prediction_metrics_compare_matching_real_futures() -> None:
    faster = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    no_extra = np.asarray([[2.0, 2.0], [3.0, 3.0]])
    metrics, records = motion_damping_prediction_metrics(
        pair_ids=("pair-0", "pair-1"),
        predicted_faster_decay=faster,
        predicted_no_extra_decay=no_extra,
        target_faster_decay=faster,
        target_no_extra_decay=no_extra,
    )
    assert metrics["correct_future_rate"] == 1.0
    assert metrics["correct_history_rate"] == 1.0
    assert metrics["context_switch_rate"] == 1.0
    assert metrics["worst_damping_correct_future_rate"] == 1.0
    assert metrics["joint_icl_pair_success_rate"] == 1.0
    assert len(records) == 2
    gate = motion_damping_prediction_gate(
        metrics, release=load_motion_damping_icl_release()
    )
    assert gate["passed"]
    assert gate["checks"]["target_latent_separation"]
    assert gate["checks"]["response_gain"]
    assert gate["checks"]["normalized_response_error"]


def test_development_contract_is_pinned_to_loader_validation_only() -> None:
    release = load_motion_damping_icl_release()
    contract = motion_damping_development_data_contract(release)
    assert contract == {
        "split": "loader_validation",
        "lance_table": "loader_validation.lance",
        "pair_count": 256,
        "lance_table_sha256": release["data"]["table_sha256"][
            "loader_validation"
        ],
        "data_manifest_sha256": release["data"]["manifest_sha256"],
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
    }
    dataset = MotionDampingICLDevelopmentDataset(release=release)
    assert dataset.identity["passed"] is True
    assert dataset.describe()["public_test_opened"] is False


def test_development_contract_fails_closed_if_it_names_public_test(
    tmp_path: Path,
) -> None:
    release = load_motion_damping_icl_release()
    path = Path(release["_config_path"])
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["evaluation"]["development"]["split"] = "validation"
    candidate = tmp_path / "motion-damping-public-misuse.yaml"
    candidate.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="loader_validation"):
        load_motion_damping_icl_release(candidate)


def test_development_score_keeps_public_closed_and_is_rescorable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextworld.benchmarks.motion_damping_icl_score as score_api

    release = load_motion_damping_icl_release()

    class Arrays:
        pair_ids = tuple(f"development-pair-{index}" for index in range(256))
        faster_decay_pixels = np.zeros((256, 4, 2, 2, 3), dtype=np.uint8)
        no_extra_decay_pixels = np.zeros((256, 4, 2, 2, 3), dtype=np.uint8)
        raw_action_blocks = np.zeros((256, 4, 5, 2), dtype=np.float32)
        pair_count = 256

    Arrays.faster_decay_pixels[:, 1] = 10
    Arrays.faster_decay_pixels[:, 3] = 10
    Arrays.no_extra_decay_pixels[:, 1] = 20
    Arrays.no_extra_decay_pixels[:, 3] = 20

    class DevelopmentDataset:
        def __init__(self, *, release, repo_root):
            del repo_root
            self.development = motion_damping_development_data_contract(release)

        @property
        def identity(self):
            return {"passed": True}

        @property
        def arrays(self):
            return Arrays

        @property
        def is_full_protocol(self):
            return True

        def describe(self):
            return {"split": "Development", "public_test_opened": False}

    class Adapter:
        protocol = AdapterProtocol(
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=2,
            future_action_blocks=1,
        )

        @property
        def metadata(self):
            return {
                "checkpoint": "/tmp/motion-damping-baseline.ckpt",
                "checkpoint_sha256": "b" * 64,
            }

        def rollout_latents(self, pixels, actions, *, batch_size):
            del actions, batch_size
            values = np.asarray(pixels, dtype=np.float32)[:, 1].mean(
                axis=(1, 2, 3)
            )
            return values[:, None, None]

        def encode_pixels(self, pixels, *, batch_size):
            del batch_size
            values = np.asarray(pixels, dtype=np.float32).mean(
                axis=(1, 2, 3)
            )
            return values[:, None]

        def frozen_state_hash(self):
            return "frozen"

    monkeypatch.setattr(
        score_api,
        "MotionDampingICLDevelopmentDataset",
        DevelopmentDataset,
    )
    monkeypatch.setattr(
        score_api,
        "MotionDampingICLEvalDataset",
        lambda *args, **kwargs: pytest.fail("Public dataset was constructed"),
    )
    result = evaluate_motion_damping_icl_development_model(
        adapter=Adapter(),
        model_name="original-pldm",
        training_recipe="original_task_only",
        training_seed=None,
    )
    assert result["metrics"]["correct_future_rate"] == 1.0
    assert result["model"]["checkpoint"]["sha256"] == "b" * 64
    assert result["contract"]["development_split"] == "loader_validation"
    assert result["public_test"]["read"] is False
    assert len(result["records"]) == 256
    output = tmp_path / "development-result.json"
    output.write_text(json.dumps(result), encoding="utf-8")
    assert rescore_motion_damping_icl_development_result(output) == result


def test_cli_keeps_public_eval_and_development_eval_separate() -> None:
    common = [
        "--checkpoint",
        "checkpoint.ckpt",
        "--adapter",
        "lewm",
        "--model-name",
        "baseline",
        "--output",
        "result.json",
    ]
    public = motion_damping_icl_cli.parse_args(
        ["eval", *common, "--without-records"]
    )
    assert public.command == "eval"
    assert public.without_records is True
    development = motion_damping_icl_cli.parse_args(
        ["eval-development", *common]
    )
    assert development.command == "eval-development"
    assert not hasattr(development, "without_records")
    score = motion_damping_icl_cli.parse_args(
        ["score-development", "--input", "input.json", "--output", "result.json"]
    )
    assert score.command == "score-development"
    with pytest.raises(SystemExit):
        motion_damping_icl_cli.parse_args(
            ["eval-development", *common, "--without-records"]
        )
