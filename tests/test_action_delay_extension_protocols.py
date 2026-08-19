from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from contextworld.paths import resolve_contextworld_path


ROOT = Path(__file__).resolve().parents[1]
ABILITY_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_action_delay_h3_ability_retention_v1.yaml"
)
CROSS_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h3_speed_cross_diagnostic_v1.yaml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _models(config: dict) -> list[dict]:
    return [
        row
        for rows in config["models"].values()
        for row in rows
    ]


def test_ability_retention_is_two_independent_50x6_domains() -> None:
    config = yaml.safe_load(ABILITY_CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "preregistered_before_model_scoring"
    assert config["evaluation"]["eval_seeds"] == [42, 43, 44, 45, 46, 47]
    assert config["evaluation"][
        "evaluations_per_seed_per_model_per_domain"
    ] == 50
    assert config["evaluation"]["evaluations_per_model_per_domain"] == 300
    assert set(config["evaluation"]["domains"]) == {
        "original_heldout",
        "speed5_matched",
    }
    assert len(_models(config)) == 7
    assert len(config["models"]["multi_delay_target"]) == 3
    assert config["evaluation"]["domains"]["speed5_matched"][
        "interpretation"
    ] == "not_speed_icl"
    for identity in config["source_identity"].values():
        path = resolve_contextworld_path(identity["path"], repo_root=ROOT)
        assert path.is_file()
        assert _sha256(path) == identity["sha256"]


def test_speed_cross_diagnostic_uses_only_offline_true_latent() -> None:
    config = yaml.safe_load(CROSS_CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == (
        "preregistered_before_action_delay_model_scoring_on_frozen_catalogs"
    )
    assert config["data"]["online_environment_during_scoring"] is False
    assert config["evaluation"]["prediction_horizon_action_blocks"] == 1
    assert config["evaluation"]["target"] == (
        "frozen_offline_true_next_frame_pixels"
    )
    assert config["evaluation"][
        "evaluations_per_reference_speed_per_history_per_seed"
    ] == 50
    assert config["evaluation"]["eval_seeds"] == [42, 43, 44, 45, 46, 47]
    assert len(_models(config)) == 7
    excluded = set(config["metrics"]["excluded"])
    assert {
        "inferred_speed",
        "projected_pixel_position",
        "position_error_px",
        "displacement_error_px",
        "raw_latent_loss_comparison_between_checkpoints",
    } <= excluded
    for identity in config["source_identity"].values():
        path = resolve_contextworld_path(identity["path"], repo_root=ROOT)
        assert path.is_file()
        assert _sha256(path) == identity["sha256"]


def test_completed_extension_summaries_have_unambiguous_claims() -> None:
    paths = {
        "multistep": resolve_contextworld_path(
            "artifacts/evaluation/history3/"
            "action_delay_multistep_extrap_v2/final_summary.json",
            repo_root=ROOT,
        ),
        "ability": resolve_contextworld_path(
            "artifacts/evaluation/history3/"
            "action_delay_ability_retention_v1/final_summary.json",
            repo_root=ROOT,
        ),
        "cross": resolve_contextworld_path(
            "artifacts/evaluation/history3/"
            "action_delay_speed_cross_diagnostic_v1/final_summary.json",
            repo_root=ROOT,
        ),
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    multistep = payloads["multistep"]["conclusions"]
    assert multistep["new_queries_confirm_one_step_action_delay_icl"] is True
    assert multistep[
        "ready_for_one_step_speed_delay_combination_design"
    ] is True
    assert multistep[
        "ready_for_multistep_or_planning_speed_delay_combination"
    ] is False
    assert multistep["delay5_high_endpoint_one_step_passed"] is True

    ability = payloads["ability"]["conclusions"]
    assert ability["multi_delay_preserves_original_heldout_ability"] is True
    assert ability["speed5_result_is_speed_icl_evidence"] is False

    cross = payloads["cross"]["conclusions"]
    assert cross["partial_multi_delay_cross_factor_response_detected"] is True
    assert cross["multi_delay_specific_cross_factor_response_detected"] is False
    assert cross["this_is_speed_icl_training_evidence"] is False
