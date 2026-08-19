"""Tests for the causal population used by transition-conditional SIGReg."""

import pytest
import torch

from scripts import run_pusht_hidden_actuation_mixed as mixed
from scripts.run_pusht_hidden_actuation_mixed import (
    build_transition_conditional_population,
)


def test_transition_population_uses_causal_target_and_prediction_masks():
    embeddings = torch.arange(4 * 4 * 2).reshape(4, 4, 2).float()
    prediction = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float() + 1000
    pairs = torch.tensor([[2, 3]], dtype=torch.long)

    population, transition_pairs, active = (
        build_transition_conditional_population(
            embeddings,
            prediction,
            pairs,
        )
    )

    torch.testing.assert_close(population[:4], embeddings[:, 1:])
    torch.testing.assert_close(population[4:], prediction)
    torch.testing.assert_close(
        transition_pairs,
        torch.tensor([[2, 3], [6, 7]], dtype=torch.long),
    )
    torch.testing.assert_close(
        active,
        torch.tensor(
            [
                [True, False],
                [False, False],
                [True, True],
            ]
        ),
    )


def test_transition_population_rejects_non_history3_shapes():
    embeddings = torch.zeros(4, 3, 2)
    prediction = torch.zeros(4, 2, 2)
    pairs = torch.tensor([[2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="expects 3 outputs"):
        build_transition_conditional_population(
            embeddings,
            prediction,
            pairs,
        )


def test_transition_population_rejects_overlapping_pairs():
    embeddings = torch.zeros(4, 4, 2)
    prediction = torch.zeros(4, 3, 2)
    pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    with pytest.raises(ValueError, match="disjoint"):
        build_transition_conditional_population(
            embeddings,
            prediction,
            pairs,
        )


def test_pldm_control_uses_the_official_pldm_model_config():
    assert mixed.VARIANT_WEIGHTS["mixed_pldm_joint"] == (
        "pldm",
        1.0,
        "official_vcreg_and_temporal_alignment",
    )
    assert (
        mixed.model_config_name_for_variant("mixed_pldm_joint") == "pldm"
    )
    assert (
        mixed.model_config("pldm")["_target_"]
        == "stable_worldmodel.wm.pldm.pldm.PLDM"
    )
    assert (
        mixed.model_config_name_for_variant("mixed_native_sigreg_0p09")
        == "lewm"
    )


def test_batch_partition_defaults_to_balanced_mixture():
    assert mixed.resolve_batch_partition(128, None) == (64, 64)


def test_batch_partition_supports_standard_only_control():
    assert mixed.resolve_batch_partition(128, 128) == (128, 0)


@pytest.mark.parametrize(
    ("batch_size", "original_batch_size"),
    [(128, 0), (128, 129), (128, 65)],
)
def test_batch_partition_rejects_invalid_or_broken_pairs(
    batch_size,
    original_batch_size,
):
    with pytest.raises(ValueError):
        mixed.resolve_batch_partition(batch_size, original_batch_size)
