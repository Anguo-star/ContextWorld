from pathlib import Path

import yaml

from contextworld.benchmarks.adapters import StableWorldModelPLDMAdapter
from scripts.diagnose_tworoom_action_delay_h7_checkpoint import (
    StableWorldModelPLDMHistory7DiagnosticAdapter,
    TRACKS_BY_SCOPE,
)
from scripts.diagnose_tworoom_action_delay_h7_capacity import (
    _selection_metrics,
)
from scripts.train_tworoom_step1 import _training_method
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEWM = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_training_v1.yaml"
)
PLDM = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_pldm_root_cause_v1.yaml"
)
CAPACITY = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_capacity_diagnostic_v1.yaml"
)
LEWM_CAPACITY_EXTENSION = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_lewm_capacity_extension_v1.yaml"
)
LEWM_ID = "H7_ActionDelay_Multi"
PLDM_ID = "H7_ActionDelay_Multi_PLDM_Diagnostic"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_pldm_root_cause_control_changes_only_registered_factors() -> None:
    lewm = _load(LEWM)
    pldm = _load(PLDM)

    assert pldm["status"] == "preregistered_before_pldm_training"
    assert pldm["claim_boundary"]["required_training_seeds"] == [3072]
    assert pldm["stable_worldmodel"] == lewm["stable_worldmodel"]
    assert pldm["data"]["original_read_only"] == lewm["data"][
        "original_read_only"
    ]
    assert pldm["data"]["original_split"] == lewm["data"]["original_split"]
    assert pldm["data"]["frozen_normalizer"] == lewm["data"][
        "frozen_normalizer"
    ]
    assert pldm["data"]["catalogs"]["action_delay_multi"] == lewm["data"][
        "catalogs"
    ]["action_delay_multi"]
    assert pldm["data_quality"]["groups"]["action_delay_multi"] == lewm[
        "data_quality"
    ]["groups"]["action_delay_multi"]

    lewm_protocol = lewm["training_protocol"]
    pldm_protocol = pldm["training_protocol"]
    for key in (
        "history_tokens",
        "num_preds",
        "raw_steps_per_action_block",
        "model_visible_fields",
        "initialization_checkpoint",
        "early_stopping",
        "checkpoint_selection",
        "distributed_execution",
    ):
        assert pldm_protocol[key] == lewm_protocol[key]
    assert pldm_protocol["paired_training_seeds"] == [3072]
    assert pldm_protocol["training_method"] == "pldm"
    assert lewm_protocol["training_method"] == "lewm"
    assert pldm_protocol["group_sampling"][PLDM_ID] == lewm_protocol[
        "group_sampling"
    ][LEWM_ID]
    assert pldm["models"] == [
        {
            "model_id": PLDM_ID,
            "display_name": (
                "相同多延迟配方、仅改用 PLDM 目标的单种子根因对照"
            ),
            "training_groups": ["original", "action_delay_multi"],
        }
    ]


def test_pldm_root_cause_runner_and_history7_adapter_are_explicit() -> None:
    assert _training_method(LEWM) == "lewm"
    assert _training_method(PLDM) == "pldm"
    assert issubclass(
        StableWorldModelPLDMHistory7DiagnosticAdapter,
        StableWorldModelPLDMAdapter,
    )
    assert (
        StableWorldModelPLDMHistory7DiagnosticAdapter.required_history_tokens
        == 7
    )
    assert (
        StableWorldModelPLDMHistory7DiagnosticAdapter
        .maximum_future_action_blocks
        == 3
    )
    assert TRACKS_BY_SCOPE == {
        "training_replay": (
            "training_replay_delay_0",
            "training_replay_delay_4",
            "training_replay_delay_8",
        ),
        "loader_validation": (
            "loader_validation_delay_0",
            "loader_validation_delay_4",
            "loader_validation_delay_8",
        ),
    }


def test_capacity_diagnostic_is_bounded_and_preregistered() -> None:
    config = _load(CAPACITY)
    assert config["status"] == "preregistered_before_capacity_diagnostic"
    assert config["claim_boundary"]["trains_new_formal_model"] is False
    assert config["claim_boundary"]["hidden_test_remains_sealed"] is True
    assert config["training"]["frozen_modules"] == ["encoder", "projector"]
    assert config["training"]["trainable_modules"] == [
        "predictor",
        "pred_proj",
        "action_encoder",
    ]
    assert set(config["training"]["variants"]) == {
        "unpaired_final",
        "paired_full",
        "paired_final",
    }


def test_capacity_selection_metrics_recognize_exact_delay_binding() -> None:
    targets = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0], [3.0, 1.0]],
        ],
        dtype=np.float32,
    )
    metrics = _selection_metrics(targets.copy(), targets)
    assert metrics["exact_target_selection_rate"] == 1.0
    assert metrics["exact_history_selection_rate"] == 1.0
    assert metrics["matching_history_strict_win_rate"] == 1.0
    assert metrics["latent_alignment"][
        "prediction_to_target_pair_magnitude_ratio"
    ] == 1.0
    assert np.isclose(
        metrics["latent_alignment"]["pair_direction_cosine_mean"],
        1.0,
    )


def test_lewm_capacity_extension_changes_only_budget_and_variant_scope() -> None:
    base = _load(CAPACITY)
    extension = _load(LEWM_CAPACITY_EXTENSION)
    assert extension["benchmark"] == base["benchmark"]
    assert extension["status"] == base["status"]
    assert extension["inputs"]["normalizer"] == base["inputs"][
        "normalizer"
    ]
    assert extension["inputs"]["paired_domain_catalog"] == base["inputs"][
        "paired_domain_catalog"
    ]
    assert extension["inputs"]["failed_checkpoints"]["lewm"] == base[
        "inputs"
    ]["failed_checkpoints"]["lewm"]
    assert extension["data"]["train_tracks"] == base["data"]["train_tracks"]
    assert extension["data"]["heldout_tracks"] == base["data"][
        "heldout_tracks"
    ]
    assert extension["data"]["queries_per_track"] == 32
    assert extension["training"]["optimizer"] == base["training"][
        "optimizer"
    ]
    assert extension["training"]["examples_per_step"] == 96
    assert extension["training"]["optimizer_steps"] == 1024
    assert list(extension["training"]["variants"]) == ["paired_final"]
    assert extension["decision"]["training_capacity_pass"] == base[
        "decision"
    ]["training_capacity_pass"]
    assert extension["decision"]["heldout_transfer_signal"] == base[
        "decision"
    ]["heldout_transfer_signal"]
