from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from contextworld.benchmarks import door_icl_score
from contextworld.benchmarks.adapters import (
    AdapterProtocol,
    SpeedICLModelAdapter,
)
from contextworld.benchmarks.door_icl_data import (
    DoorICLEvalDataset,
    RELEASE_ID,
    _tree_fingerprint,
    load_door_icl_release,
)
from contextworld.benchmarks.door_icl_score import (
    evaluate_door_icl_model,
    score_door_icl_results,
)
from contextworld.evaluation.hidden_passage_validation import (
    INFORMATIVE_HISTORY_BOOTSTRAP_METRICS,
    file_sha256,
)


class FakeDoorAdapter(SpeedICLModelAdapter):
    def __init__(self, checkpoint_sha256: str = "a" * 64) -> None:
        self.checkpoint_sha256 = checkpoint_sha256

    @property
    def protocol(self) -> AdapterProtocol:
        return AdapterProtocol(
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=2,
            future_action_blocks=1,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "adapter_id": "fake_door",
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    def encode_pixels(
        self,
        pixels: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        del batch_size
        return (
            np.asarray(pixels, dtype=np.float32).reshape(len(pixels), -1)
            / 255.0
        )

    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        del batch_size
        pixels = np.asarray(input_pixels, dtype=np.uint8)
        actions = np.asarray(raw_action_blocks, dtype=np.float32)
        assert actions.shape[1:] == (3, 5, 2)
        query = pixels[:, -1].astype(np.float32).reshape(len(pixels), -1)
        query[:, 2] = pixels[:, 0, 0, 0, 2]
        return (query / 255.0)[:, None]

    def frozen_state_hash(self) -> str:
        return f"frozen-{self.checkpoint_sha256}"


def _query_pixels(index: int) -> np.ndarray:
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    pixels[0, 0, 0] = np.uint8(index % 256)
    pixels[0, 0, 1] = np.uint8(index // 256)
    return pixels


def _synthetic_assets() -> list[dict[str, Any]]:
    assets = []
    markers = {
        "observed_passable": 200,
        "observed_blocked": 0,
        "did_not_attempt_crossing": 100,
    }
    for seed_index, eval_seed in enumerate((42, 43, 44, 45, 46, 47)):
        for evaluation_index in range(50):
            index = seed_index * 50 + evaluation_index
            query = _query_pixels(index)
            histories = {}
            actions = {}
            for condition, marker in markers.items():
                history = np.stack([query.copy(), query.copy(), query.copy()])
                history[0, 0, 0, 2] = np.uint8(marker)
                histories[condition] = history
                actions[condition] = np.zeros(
                    (3, 5, 2),
                    dtype=np.float32,
                )
            passable = query.copy()
            passable[0, 0, 2] = np.uint8(200)
            assets.append(
                {
                    "query_id": f"s{eval_seed}-e{evaluation_index:03d}",
                    "static_query_id": f"q{index:03d}",
                    "eval_seed": eval_seed,
                    "evaluation_index": evaluation_index,
                    "direction": (
                        "left_to_right"
                        if evaluation_index < 25
                        else "right_to_left"
                    ),
                    "template_id": f"dummy-{index:03d}",
                    "query_pixels": query,
                    "histories": histories,
                    "actions": actions,
                    "targets": {
                        "passable": passable,
                        "blocked": query.copy(),
                    },
                }
            )
    return assets


def _release_config(tmp_path: Path) -> Path:
    path = tmp_path / "door-release.yaml"
    payload = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "release_status": "validation_release_candidate",
        "scope": {"sealed_test_included": False},
        "training": {"paired_training_seeds": [3072, 4096, 5120]},
        "evaluation": {
            "catalog_sha256": "catalog",
            "content_manifest_sha256": "content",
            "normalizer_sha256": "normalizer",
            "eval_seeds": [42, 43, 44, 45, 46, 47],
            "queries_per_eval_seed": 50,
        },
        "scoring": {
            "gates": {
                "decision_contract": "informative_history_rule_switch_v2",
                "minimum_same_history_two_target_accuracy_exclusive": 0.5,
                "minimum_matching_vs_opposite_history_win_rate_exclusive": 0.5,
                "minimum_target_pair_latent_mse_exclusive": 1.0e-12,
                "paired_bootstrap": {
                    "unit": "static_query_within_eval_seed_direction",
                    "strata": "eval_seed_x_direction",
                    "method": "percentile",
                    "resamples": 100,
                    "confidence": 0.95,
                    "seed": 20260725,
                    "minimum_lower_bound_exclusive": 0.0,
                    "required_metrics": list(
                        INFORMATIVE_HISTORY_BOOTSTRAP_METRICS
                    ),
                },
            }
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_default_release_is_validation_only() -> None:
    release = load_door_icl_release()
    assert release["release_status"] == "validation_release_candidate"
    assert release["scope"]["sealed_test_included"] is False
    assert release["evaluation"]["queries"] == 300
    assert release["evaluation"]["model_predictions_per_checkpoint"] == 900
    assert release["evaluation"]["loss_records_per_checkpoint"] == 1800


def test_tree_fingerprint_detects_same_size_content_change(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a").write_bytes(b"abc")
    first = _tree_fingerprint(root, hash_contents=True)
    (root / "a").write_bytes(b"ABC")
    second = _tree_fingerprint(root, hash_contents=True)
    assert first["files"] == second["files"] == 1
    assert first["bytes"] == second["bytes"] == 3
    assert first["sha256"] != second["sha256"]


def test_public_dataset_selects_seed_and_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    assets = _synthetic_assets()
    release = {
        "evaluation": {
            "catalog": str(catalog),
            "catalog_sha256": file_sha256(catalog),
            "eval_seeds": [42, 43, 44, 45, 46, 47],
            "queries_per_eval_seed": 50,
        }
    }
    monkeypatch.setattr(
        "contextworld.benchmarks.door_icl_data.load_validation_assets",
        lambda *args, **kwargs: (assets, {"passed": True}),
    )
    dataset = DoorICLEvalDataset(
        release=release,
        repo_root=tmp_path,
        eval_seeds=[43],
        limit_per_seed=2,
    )
    assert len(dataset) == 2
    assert not dataset.is_full_protocol
    assert dataset[0].eval_seed == 43
    assert dataset[0].query_pixels.shape == (2, 2, 3)
    assert set(dataset[0].histories) == {
        "observed_passable",
        "observed_blocked",
        "did_not_attempt_crossing",
    }


def test_full_eval_and_three_seed_public_rescore(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release_path = _release_config(tmp_path)

    class FakeDataset:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.raw_assets = _synthetic_assets()
            self.is_full_protocol = True

        def describe(self) -> dict[str, Any]:
            return {
                "track": "unseen_door_positions",
                "queries": 300,
                "full_protocol": True,
            }

    monkeypatch.setattr(
        door_icl_score,
        "DoorICLEvalDataset",
        FakeDataset,
    )
    result = evaluate_door_icl_model(
        adapter=FakeDoorAdapter(),
        model_name="fake-door-model",
        training_recipe="fake",
        training_seed=3072,
        release_config=release_path,
        repo_root=tmp_path,
        batch_size=64,
    )
    assert result["full_protocol"]
    assert result["formal_checkpoint_passed"]
    assert len(result["records"]) == 1800
    assert result["score_audit"]["model_predictions"] == 900
    assert result["summary"]["decision"]["passed"]

    paths = []
    for index, seed in enumerate((3072, 4096, 5120), start=1):
        copy = json.loads(json.dumps(result))
        copy["model"]["training_seed"] = seed
        copy["model"]["adapter"]["checkpoint_sha256"] = f"{index:064x}"
        path = tmp_path / f"result-{seed}.json"
        path.write_text(json.dumps(copy), encoding="utf-8")
        paths.append(path)
    summary = score_door_icl_results(
        result_paths=paths,
        method_name="fake-three-seed-method",
        release_config=release_path,
    )
    assert summary["formal_claim_level"] == "three_seed_method_result"
    assert summary["passed_checkpoints"] == 3
    assert summary["method_passed"] is True
    assert summary["mean_correct_target_choice_rate"] == 1.0
    assert summary["mean_correct_history_win_rate"] == 1.0


def test_public_rescore_rejects_a_changed_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release_path = _release_config(tmp_path)

    class FakeDataset:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.raw_assets = _synthetic_assets()
            self.is_full_protocol = True

        def describe(self) -> dict[str, Any]:
            return {"queries": 300, "full_protocol": True}

    monkeypatch.setattr(
        door_icl_score,
        "DoorICLEvalDataset",
        FakeDataset,
    )
    result = evaluate_door_icl_model(
        adapter=FakeDoorAdapter(),
        model_name="fake",
        training_recipe="fake",
        training_seed=3072,
        release_config=release_path,
        repo_root=tmp_path,
    )
    result["records"][0]["true_next_frame_latent_mse"] += 1.0
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    try:
        score_door_icl_results(
            result_paths=[path],
            method_name="changed",
            release_config=release_path,
        )
    except RuntimeError as error:
        assert "summary changed" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Changed records must fail independent rescoring")
