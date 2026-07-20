from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from contextworld.evaluation.icl_catalog import (
    diagnostic_action_blocks,
    validate_context_query_catalog,
)
from contextworld.evaluation.icl_model import state_dict_sha256
from contextworld.paths import artifact_path, resolve_contextworld_path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = artifact_path(
    "evaluation/icl/tworoom_icl_v1_validation_context_query_catalog.json",
    repo_root=ROOT,
)


def test_diagnostic_actions_return_exactly_and_k2_has_intermediate_motion() -> None:
    direction = np.asarray([1.0, 0.5], dtype=np.float32)
    k1 = diagnostic_action_blocks(direction, 1)
    k2 = diagnostic_action_blocks(direction, 2)

    assert k1.shape == (1, 5, 2)
    assert k2.shape == (2, 5, 2)
    assert np.array_equal(k1.sum(axis=(0, 1)), np.zeros(2, dtype=np.float32))
    assert np.array_equal(k2.sum(axis=(0, 1)), np.zeros(2, dtype=np.float32))
    assert np.any(k2[0].sum(axis=0) != 0)
    assert np.array_equal(k2[0], -k2[1])


def test_validation_catalog_is_strictly_paired_and_query_disjoint() -> None:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))

    assert catalog["summary"]["bundles"] == 68
    assert catalog["summary"]["by_family"] == {
        "speed": 32,
        "door": 16,
        "speed_door_composition": 20,
    }
    assert catalog["protocol"]["supported_context_budgets"] == [0, 1, 2]
    assert catalog["protocol"]["maximum_prior_context_transitions"] == 2

    query_ids = [bundle["query_id"] for bundle in catalog["bundles"]]
    payloads = [bundle["payload"] for bundle in catalog["bundles"]]
    assert len(set(query_ids)) == len(query_ids)
    assert len(set(payloads)) == len(payloads)

    for bundle in catalog["bundles"]:
        payload_path = resolve_contextworld_path(
            bundle["payload"], repo_root=ROOT
        )
        with np.load(payload_path, allow_pickle=False) as payload:
            reference_actions = payload["context_b2_correct_actions"]
            reference_state = payload["context_b2_correct_states"][0]
            for condition in bundle["conditions"]:
                assert np.array_equal(
                    payload[f"context_b2_{condition}_actions"], reference_actions
                )
                assert np.array_equal(
                    payload[f"context_b2_{condition}_states"][0], reference_state
                )
                assert np.array_equal(
                    payload[f"context_b2_{condition}_next_states"][-1],
                    payload["query_state"],
                )
            assert np.array_equal(
                payload["candidate_pixels"][bundle["correct_candidate_index"]],
                payload["target_pixels"],
            )


def test_validation_catalog_can_be_scoped_to_speed_only() -> None:
    report = validate_context_query_catalog(
        CATALOG,
        repo_root=ROOT,
        replay_simulator=False,
        family="speed",
    )

    assert report["passed"]
    assert report["families"] == ["speed"]
    assert report["bundles"] == 32


def test_state_dict_hash_detects_mutation_and_is_stable_without_mutation() -> None:
    model = torch.nn.Linear(3, 2)
    first = state_dict_sha256(model)
    second = state_dict_sha256(model)
    assert first == second

    with torch.no_grad():
        model.weight[0, 0].add_(1.0)
    assert state_dict_sha256(model) != first
