from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from contextworld.benchmarks.cube_grasp_rule_icl_data import (
    CubeGraspRuleEvalArrays,
)
from contextworld.benchmarks.cube_grasp_rule_public_contract import (
    PublicAuthorization,
)
from contextworld.benchmarks.cube_grasp_rule_public_score import (
    aggregate_public_results,
    evaluate_public_checkpoint,
)
import scripts.run_cube_grasp_rule_h3_v4r1_public_matrix as public_runner


SEEDS = (17321, 17322, 17323)
RECIPE = "mixed_frozen_image_paired_future_fit_1p00"


def _authorization(tmp_path: Path) -> PublicAuthorization:
    prereg_path = tmp_path / "prereg.yaml"
    freeze_path = tmp_path / "freeze.json"
    prereg_path.write_text("frozen: true\n", encoding="utf-8")
    freeze_path.write_text('{"frozen": true}\n', encoding="utf-8")
    public_root = tmp_path / "public"
    public_root.mkdir()
    (public_root / "_SUCCESS.json").write_text(
        '{"status": "fixture"}\n', encoding="utf-8"
    )
    checkpoints = [
        {
            "model_name": f"cube_gripper_carry_lewm_seed{seed}",
            "model_family": "lewm",
            "training_recipe": RECIPE,
            "training_seed": seed,
            "path": str(tmp_path / f"seed{seed}.pt"),
            "sha256": f"{index + 1}" * 64,
            "size_bytes": 1,
            "model_state_sha256": "a" * 64,
        }
        for index, seed in enumerate(SEEDS)
    ]
    prereg = {
        "preregistration_id": "contextworld_cube_gripper_carry_h3_v4r1_public_release_v1",
        "identity": {"preregistration_path": str(prereg_path)},
        "planned_artifacts": {
            "public_data_root": str(public_root),
            "public_score_root": str(tmp_path / "score"),
            "public_release_decision": str(tmp_path / "decision.json"),
        },
        "public_evaluation": {
            "checkpoints": checkpoints,
            "devices": ["cuda:0", "cuda:1", "cuda:2"],
            "batch_size": 64,
        },
        "scoring": {
            "hidden_future_prediction": {
                "gates": {
                    "correct_future_rate_minimum": 0.75,
                    "correct_history_rate_minimum": 0.75,
                    "context_switch_rate_minimum": 0.90,
                    "worst_rule_correct_future_rate_minimum": 0.70,
                    "target_latent_separation_required": True,
                    "response_gain_minimum": 0.50,
                    "normalized_response_error_strict_maximum": 1.00,
                },
                "uncertainty": {
                    "lower_bound_minimum": {
                        "correct_future_rate": 0.70,
                        "correct_history_rate": 0.70,
                        "context_switch_rate": 0.85,
                    }
                },
            }
        },
    }
    return PublicAuthorization(
        preregistration_path=prereg_path,
        freeze_receipt_path=freeze_path,
        preregistration=prereg,
        freeze_receipt={},
        freeze_receipt_identity={
            "path": str(tmp_path / "freeze.json"),
            "sha256": "f" * 64,
            "size_bytes": 1,
        },
    )


def _arrays() -> CubeGraspRuleEvalArrays:
    pair_count = 256
    cannot = np.zeros((pair_count, 4, 1, 1, 1), dtype=np.float32)
    can = np.full((pair_count, 4, 1, 1, 1), 255, dtype=np.float32)
    return CubeGraspRuleEvalArrays(
        pair_ids=tuple(f"pair-{index:04d}" for index in range(pair_count)),
        cannot_hold_pixels=cannot,
        can_hold_pixels=can,
        raw_action_blocks=np.zeros((pair_count, 4, 5, 5), dtype=np.float32),
        cannot_hold_states=np.zeros((pair_count, 4, 1), dtype=np.float64),
        can_hold_states=np.ones((pair_count, 4, 1), dtype=np.float64),
    )


class PerfectAdapter:
    protocol = SimpleNamespace(
        history_tokens=3,
        action_block_raw_steps=5,
        action_dim=5,
        future_action_blocks=5,
        native_target_encoder=True,
        decoder_required=False,
    )
    metadata = {"adapter_id": "test-perfect-cube", "device": "cpu"}

    def frozen_state_hash(self) -> str:
        return "a" * 64

    def rollout_latents(
        self, histories: np.ndarray, actions: np.ndarray, *, batch_size: int
    ) -> np.ndarray:
        del actions, batch_size
        values = np.where(histories.mean(axis=tuple(range(1, histories.ndim))) > 0, 1.0, -1.0)
        return values[:, None, None]

    def encode_pixels(self, pixels: np.ndarray, *, batch_size: int) -> np.ndarray:
        del batch_size
        values = np.where(pixels.mean(axis=tuple(range(1, pixels.ndim))) > 0, 1.0, -1.0)
        return values[:, None]


def test_perfect_public_checkpoint_passes_frozen_gates(tmp_path: Path) -> None:
    authorization = _authorization(tmp_path)
    result = evaluate_public_checkpoint(
        adapter=PerfectAdapter(),
        arrays=_arrays(),
        authorization=authorization,
        checkpoint_specification=authorization.preregistration[
            "public_evaluation"
        ]["checkpoints"][0],
        batch_size=64,
        include_records=False,
    )
    assert result["gate"]["passed"] is True
    assert result["metrics"]["correct_future_rate"] == 1.0
    assert result["metrics"]["correct_history_rate"] == 1.0
    assert result["metrics"]["context_switch_rate"] == 1.0
    assert result["freeze_receipt_sha256"] == "f" * 64
    assert result["data"]["model_visible_fields"] == [
        "history_pixels",
        "query_action_blocks",
    ]
    assert result["data"]["privileged_columns_passed_to_model"] is False


def test_public_aggregate_requires_exact_frozen_three_seed_matrix(
    tmp_path: Path,
) -> None:
    authorization = _authorization(tmp_path)
    results = []
    for checkpoint in authorization.preregistration["public_evaluation"][
        "checkpoints"
    ]:
        results.append(
            evaluate_public_checkpoint(
                adapter=PerfectAdapter(),
                arrays=_arrays(),
                authorization=authorization,
                checkpoint_specification=checkpoint,
                batch_size=64,
                include_records=False,
            )
        )
    matrix = aggregate_public_results(results, authorization=authorization)
    assert matrix["passed"] is True
    assert matrix["checkpoints_passed"] == 3
    assert matrix["training_seeds"] == list(SEEDS)

    results[-1]["model"]["training_seed"] = SEEDS[1]
    with pytest.raises(RuntimeError, match="exactly three frozen seeds"):
        aggregate_public_results(results, authorization=authorization)


def test_public_aggregate_recomputes_gate_and_rejects_provenance_drift(
    tmp_path: Path,
) -> None:
    authorization = _authorization(tmp_path)
    results = [
        evaluate_public_checkpoint(
            adapter=PerfectAdapter(),
            arrays=_arrays(),
            authorization=authorization,
            checkpoint_specification=checkpoint,
            batch_size=64,
            include_records=False,
        )
        for checkpoint in authorization.preregistration["public_evaluation"][
            "checkpoints"
        ]
    ]
    forged_gate = copy.deepcopy(results)
    forged_gate[0]["gate"]["passed"] = False
    with pytest.raises(RuntimeError, match="gate was not recomputed exactly"):
        aggregate_public_results(forged_gate, authorization=authorization)

    forged_recipe = copy.deepcopy(results)
    forged_recipe[0]["model"]["training_recipe"] = "selected_from_public"
    with pytest.raises(RuntimeError, match="model provenance drifted"):
        aggregate_public_results(forged_recipe, authorization=authorization)

    nonfinite = copy.deepcopy(results)
    nonfinite[0]["metrics"]["correct_future_rate"] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        aggregate_public_results(nonfinite, authorization=authorization)


def test_runner_preloads_all_adapters_before_public_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _authorization(tmp_path)
    built: list[tuple[int, str]] = []
    released: list[int] = []

    class Adapter:
        metadata = {"adapter_id": "preflight-adapter"}

        def __init__(self, seed: int) -> None:
            self.seed = seed

        def frozen_state_hash(self) -> str:
            return f"{self.seed:064x}"[-64:]

    def build_adapter(*, authorization, checkpoint, device):
        del authorization
        seed = int(checkpoint["training_seed"])
        built.append((seed, device))
        return Adapter(seed)

    monkeypatch.setattr(public_runner, "build_adapter", build_adapter)
    monkeypatch.setattr(
        public_runner,
        "release_adapter",
        lambda adapter: released.append(adapter.seed),
    )
    checkpoints = authorization.preregistration["public_evaluation"][
        "checkpoints"
    ]
    receipts = public_runner._adapter_runtime_preflight(
        authorization=authorization,
        checkpoints=checkpoints,
        devices=("cuda:0", "cuda:1", "cuda:2"),
    )
    assert built == list(zip(SEEDS, ("cuda:0", "cuda:1", "cuda:2")))
    assert released == list(SEEDS)
    assert [row["training_seed"] for row in receipts] == list(SEEDS)
    assert all(row["runtime_preflight_passed"] for row in receipts)


def test_runner_persists_preaccess_failure_after_score_namespace_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _authorization(tmp_path)
    monkeypatch.setattr(
        public_runner, "load_public_authorization", lambda **_: authorization
    )
    monkeypatch.setattr(
        public_runner, "_checkpoint_preflight", lambda checkpoints: checkpoints
    )
    monkeypatch.setattr(
        public_runner,
        "_adapter_runtime_preflight",
        lambda **_: [{"runtime_preflight_passed": True}] * 3,
    )
    monkeypatch.setattr(
        public_runner,
        "validate_public_publication",
        lambda _, **__: {"passed": True},
    )
    original_write = public_runner._write_json_x

    def fail_at_access(path: Path, value: dict) -> None:
        if path.name == "public_access_started.json":
            raise RuntimeError("forced marker failure")
        original_write(path, value)

    monkeypatch.setattr(public_runner, "_write_json_x", fail_at_access)
    output = authorization.score_root
    with pytest.raises(RuntimeError, match="forced marker failure"):
        public_runner.run_public_matrix(
            preregistration=authorization.preregistration_path,
            freeze_receipt=authorization.freeze_receipt_path,
            output=output,
            devices=("cuda:0", "cuda:1", "cuda:2"),
            batch_size=64,
        )
    failure = json.loads(
        (output / "infrastructure_failure_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["public_access_started"] is False
    assert failure["public_test_may_have_been_read"] is False
    assert failure["rerun_authorized"] is False


def test_runner_marks_access_before_full_public_tree_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _authorization(tmp_path)
    monkeypatch.setattr(
        public_runner, "load_public_authorization", lambda **_: authorization
    )
    monkeypatch.setattr(
        public_runner, "_checkpoint_preflight", lambda checkpoints: checkpoints
    )
    monkeypatch.setattr(
        public_runner,
        "_adapter_runtime_preflight",
        lambda **_: [{"runtime_preflight_passed": True}] * 3,
    )
    output = authorization.score_root
    validation_calls: list[bool] = []

    def validate_publication(_, *, verify_published_tree: bool = True):
        validation_calls.append(verify_published_tree)
        if verify_published_tree:
            assert (output / "public_access_started.json").is_file()
            raise RuntimeError("forced Public tree validation failure")
        return {"metadata_preflight_passed": True}

    monkeypatch.setattr(
        public_runner, "validate_public_publication", validate_publication
    )
    with pytest.raises(RuntimeError, match="forced Public tree validation failure"):
        public_runner.run_public_matrix(
            preregistration=authorization.preregistration_path,
            freeze_receipt=authorization.freeze_receipt_path,
            output=output,
            devices=("cuda:0", "cuda:1", "cuda:2"),
            batch_size=64,
        )

    failure = json.loads(
        (output / "infrastructure_failure_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation_calls == [False, True]
    assert failure["status"] == (
        "public_campaign_failed_after_access_no_rerun_authorized"
    )
    assert failure["public_access_started"] is True
    assert failure["public_test_may_have_been_read"] is True
    assert failure["rerun_authorized"] is False
