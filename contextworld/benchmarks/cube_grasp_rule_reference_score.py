from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

import numpy as np

from contextworld.benchmarks.adapters import CubeGraspRuleICLModelAdapter
from contextworld.benchmarks.cube_grasp_rule_icl_data import _read_lance_pairs
from contextworld.benchmarks.cube_grasp_rule_icl_score import (
    _validate_cube_adapter_protocol,
    cube_grasp_rule_prediction_gate,
    cube_grasp_rule_prediction_metrics,
)
from contextworld.benchmarks.cube_grasp_rule_reference_training import (
    DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
    file_sha256,
    load_cube_reference_training_prereg,
    validate_cube_reference_training_report,
)
from contextworld.paths import repository_root, resolve_contextworld_path


CUBE_REFERENCE_DEVELOPMENT_BENCHMARK = (
    "cube_history3_gripper_carry_v4r1_development_v1"
)


class CubeGraspRuleDevelopmentDataset:
    """The frozen v4r1 Loader Validation split; never a Public/Test reader."""

    def __init__(
        self,
        *,
        prereg: dict[str, Any],
        repo_root: Path | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.prereg = prereg
        self.root = resolve_contextworld_path(
            prereg["data"]["artifact_tree"]["root"],
            repo_root=self.repo_root,
        )
        self._arrays = None

    @property
    def arrays(self):
        if self._arrays is None:
            evaluation = self.prereg["evaluation"]
            if evaluation["split"] != "loader_validation":
                raise RuntimeError("Cube Development reader refuses non-Development split")
            self._arrays = _read_lance_pairs(
                self.root / evaluation["lance_table"],
                expected_pairs=int(evaluation["pair_count"]),
                expected_split="loader_validation",
            )
        return self._arrays

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "split": "Development",
            "lance_table": "loader_validation.lance",
            "pair_count": self.arrays.pair_count,
            "condition_count": 2 * self.arrays.pair_count,
            "history_tokens": 3,
            "online_environment_calls": 0,
            "public_test_opened": False,
        }


def _expected_contract_identity(
    prereg: dict[str, Any], *, prereg_path: Path
) -> dict[str, Any]:
    freeze_path = Path(prereg["_freeze_receipt_path"])
    return {
        "preregistration_id": prereg["preregistration_id"],
        "preregistration_sha256": file_sha256(prereg_path),
        "freeze_receipt_sha256": file_sha256(freeze_path),
        "data_manifest_sha256": prereg["data"]["manifest_sha256"],
    }


def evaluate_cube_reference_development_checkpoint(
    *,
    adapter: CubeGraspRuleICLModelAdapter,
    model_family: str,
    model_name: str,
    training_recipe: str,
    training_seed: int,
    prereg_config: Path | str = DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
    repo_root: Path | None = None,
    batch_size: int = 64,
    include_records: bool = True,
    loaded_prereg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    prereg = loaded_prereg or load_cube_reference_training_prereg(
        prereg_config, require_freeze=True, repo_root=root
    )
    if Path(prereg["_config_path"]).resolve() != Path(prereg_config).expanduser().resolve():
        raise RuntimeError("Cube Development loaded prereg path drifted")
    if model_family not in {"lewm", "pldm"}:
        raise ValueError("Cube model_family must be lewm or pldm")
    matrix = prereg["training"]["reference_matrix"]
    expected_recipe = matrix["models"][model_family]["variant"]
    if training_recipe != expected_recipe:
        raise ValueError("Cube Development recipe does not match preregistration")
    if training_seed not in matrix["training_seeds"]:
        raise ValueError("Cube Development seed does not match preregistration")
    if int(batch_size) != int(prereg["evaluation"]["inference_batch_size"]):
        raise ValueError("Cube Development inference batch size drifted")
    training_cell = validate_cube_reference_training_report(
        prereg,
        model_family=model_family,
        training_seed=training_seed,
        prereg_path=Path(prereg["_config_path"]),
        repo_root=root,
    )
    _validate_cube_adapter_protocol(adapter)
    metadata = adapter.metadata
    runtime = prereg["runtime"]["stable_worldmodel"]
    if (
        Path(str(metadata.get("checkpoint", ""))).resolve()
        != Path(training_cell["checkpoint"])
        or metadata.get("checkpoint_sha256")
        != training_cell["checkpoint_sha256"]
        or metadata.get("stable_worldmodel_commit") != runtime["expected_ref"]
        or Path(str(metadata.get("stable_worldmodel_repo", ""))).resolve()
        != Path(str(runtime["repo"])).expanduser().resolve()
    ):
        raise RuntimeError("Cube Development adapter is not the frozen training cell")
    dataset = CubeGraspRuleDevelopmentDataset(prereg=prereg, repo_root=root)
    arrays = dataset.arrays
    histories = np.concatenate(
        [arrays.cannot_hold_pixels[:, :3], arrays.can_hold_pixels[:, :3]]
    )
    actions = np.concatenate(
        [arrays.raw_action_blocks[:, :3], arrays.raw_action_blocks[:, :3]]
    )
    before = adapter.frozen_state_hash()
    if before != training_cell["model_state_sha256"]:
        raise RuntimeError("Cube Development checkpoint state differs from training report")
    predicted = adapter.rollout_latents(histories, actions, batch_size=batch_size)
    count = arrays.pair_count
    if (
        predicted.ndim != 3
        or predicted.shape[0] != 2 * count
        or predicted.shape[1] != 1
        or not np.isfinite(predicted).all()
    ):
        raise RuntimeError(
            "Cube Development adapter must return finite "
            "(2 * pairs, 1, latent_dim) predictions"
        )
    true_futures = np.concatenate(
        [arrays.cannot_hold_pixels[:, 3], arrays.can_hold_pixels[:, 3]]
    )
    encoded = adapter.encode_pixels(true_futures, batch_size=batch_size)
    if (
        encoded.shape != (2 * count, predicted.shape[2])
        or not np.isfinite(encoded).all()
    ):
        raise RuntimeError("Cube Development target latents do not match predictions")
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Cube checkpoint changed during Development scoring")
    metrics, records = cube_grasp_rule_prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_cannot_hold=predicted[:count, 0],
        predicted_can_hold=predicted[count:, 0],
        target_cannot_hold=encoded[:count],
        target_can_hold=encoded[count:],
    )
    prereg_path = Path(prereg["_config_path"])
    result = {
        "schema_version": 1,
        "benchmark": CUBE_REFERENCE_DEVELOPMENT_BENCHMARK,
        "submission_kind": "single_checkpoint",
        "status": "completed",
        "contract": _expected_contract_identity(prereg, prereg_path=prereg_path),
        "model": {
            "family": model_family,
            "name": str(model_name),
            "training_recipe": training_recipe,
            "training_seed": int(training_seed),
            "adapter": adapter.metadata,
            "state_sha256_before": before,
            "state_sha256_after": after,
            "training_checkpoint": {
                "path": str(training_cell["checkpoint"]),
                "sha256": training_cell["checkpoint_sha256"],
                "size_bytes": training_cell["checkpoint_size_bytes"],
                "model_state_sha256": training_cell["model_state_sha256"],
                "training_report": str(training_cell["report"]),
            },
        },
        "data": dataset.describe(),
        "metrics": metrics,
        "gate": cube_grasp_rule_prediction_gate(metrics, release=prereg),
        "claim_scope": "Development_only_not_Public_or_release",
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
    }
    if include_records:
        result["records"] = records
    return result


def score_cube_reference_development_results(
    *,
    result_paths: Iterable[Path | str],
    model_family: str,
    prereg_config: Path | str = DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
    loaded_prereg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prereg = loaded_prereg or load_cube_reference_training_prereg(
        prereg_config, require_freeze=True
    )
    if Path(prereg["_config_path"]).resolve() != Path(prereg_config).expanduser().resolve():
        raise RuntimeError("Cube Development loaded prereg path drifted")
    if model_family not in {"lewm", "pldm"}:
        raise ValueError("Cube model_family must be lewm or pldm")
    prereg_path = Path(prereg["_config_path"])
    expected_contract = _expected_contract_identity(prereg, prereg_path=prereg_path)
    matrix = prereg["training"]["reference_matrix"]
    expected_seeds = sorted(int(value) for value in matrix["training_seeds"])
    expected_recipe = matrix["models"][model_family]["variant"]
    results = []
    validated_cells: dict[int, dict[str, Any]] = {}
    for value in result_paths:
        path = Path(value)
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("schema_version") != 1
            or row.get("benchmark") != CUBE_REFERENCE_DEVELOPMENT_BENCHMARK
            or row.get("submission_kind") != "single_checkpoint"
            or row.get("status") != "completed"
            or row.get("claim_scope") != "Development_only_not_Public_or_release"
        ):
            raise ValueError(f"Unsupported Cube Development result: {path}")
        if row.get("contract") != expected_contract:
            raise RuntimeError(f"Cube Development contract mismatch: {path}")
        model = row.get("model", {})
        seed = model.get("training_seed")
        if (
            model.get("family") != model_family
            or model.get("training_recipe") != expected_recipe
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise RuntimeError(f"Cube Development model identity mismatch: {path}")
        training_cell = validate_cube_reference_training_report(
            prereg,
            model_family=model_family,
            training_seed=seed,
            prereg_path=prereg_path,
        )
        validated_cells[seed] = training_cell
        expected_checkpoint = {
            "path": str(training_cell["checkpoint"]),
            "sha256": training_cell["checkpoint_sha256"],
            "size_bytes": training_cell["checkpoint_size_bytes"],
            "model_state_sha256": training_cell["model_state_sha256"],
            "training_report": str(training_cell["report"]),
        }
        if model.get("training_checkpoint") != expected_checkpoint:
            raise RuntimeError(f"Cube Development checkpoint identity mismatch: {path}")
        if (
            model.get("state_sha256_before")
            != training_cell["model_state_sha256"]
            or model.get("state_sha256_after")
            != training_cell["model_state_sha256"]
        ):
            raise RuntimeError(f"Cube Development model-state identity mismatch: {path}")
        public = row.get("public_test", {})
        if public != {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        }:
            raise RuntimeError(f"Cube Development result opened Public Test: {path}")
        if "latent_response" not in row.get("metrics", {}):
            raise ValueError(f"Cube Development result lacks anti-spoof metrics: {path}")
        expected_gate = cube_grasp_rule_prediction_gate(row["metrics"], release=prereg)
        if row.get("gate") != expected_gate:
            raise RuntimeError(f"Cube Development gate was not recomputed: {path}")
        results.append(row)
    seeds = sorted(row["model"].get("training_seed") for row in results)
    if seeds != expected_seeds:
        raise ValueError(
            f"Cube {model_family} Development scoring requires seeds {expected_seeds}"
        )
    names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_rule_correct_future_rate",
        "other_minus_correct_mse_margin_mean",
        "joint_icl_pair_success_rate",
    )
    result = {
        "schema_version": 1,
        "benchmark": CUBE_REFERENCE_DEVELOPMENT_BENCHMARK,
        "submission_kind": "three_seed_method",
        "status": "completed",
        "model_family": model_family,
        "training_recipe": expected_recipe,
        "training_seeds": seeds,
        "checkpoint_results": results,
        "aggregate": {
            name: {
                "mean": float(statistics.mean(row["metrics"][name] for row in results)),
                "minimum": float(min(row["metrics"][name] for row in results)),
                "maximum": float(max(row["metrics"][name] for row in results)),
            }
            for name in names
        },
        "passed": all(row["gate"]["passed"] for row in results),
        "public_test_opened": False,
    }
    validate_cube_reference_development_method(
        result,
        prereg=prereg,
        model_family=model_family,
        validated_cells=validated_cells,
    )
    return result


def validate_cube_reference_development_method(
    method: dict[str, Any],
    *,
    prereg: dict[str, Any],
    model_family: str,
    validated_cells: Mapping[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Recompute an embedded three-seed method before finalization."""

    if model_family not in {"lewm", "pldm"}:
        raise ValueError("Cube model_family must be lewm or pldm")
    matrix = prereg["training"]["reference_matrix"]
    expected_seeds = sorted(int(value) for value in matrix["training_seeds"])
    expected_recipe = matrix["models"][model_family]["variant"]
    rows = method.get("checkpoint_results")
    if (
        method.get("schema_version") != 1
        or method.get("benchmark") != CUBE_REFERENCE_DEVELOPMENT_BENCHMARK
        or method.get("submission_kind") != "three_seed_method"
        or method.get("status") != "completed"
        or method.get("model_family") != model_family
        or method.get("training_recipe") != expected_recipe
        or method.get("training_seeds") != expected_seeds
        or method.get("public_test_opened") is not False
        or not isinstance(rows, list)
        or len(rows) != 3
    ):
        raise RuntimeError(f"Cube {model_family} Development method is incomplete")
    contract = _expected_contract_identity(
        prereg, prereg_path=Path(prereg["_config_path"])
    )
    observed_seeds: list[int] = []
    names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_rule_correct_future_rate",
        "other_minus_correct_mse_margin_mean",
        "joint_icl_pair_success_rate",
    )
    for row in rows:
        model = row.get("model", {})
        seed = model.get("training_seed")
        if (
            row.get("schema_version") != 1
            or row.get("benchmark") != CUBE_REFERENCE_DEVELOPMENT_BENCHMARK
            or row.get("submission_kind") != "single_checkpoint"
            or row.get("status") != "completed"
            or row.get("contract") != contract
            or row.get("claim_scope") != "Development_only_not_Public_or_release"
            or model.get("family") != model_family
            or model.get("training_recipe") != expected_recipe
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise RuntimeError(f"Cube {model_family} embedded result identity drifted")
        training_cell = (
            validated_cells[seed]
            if validated_cells is not None and seed in validated_cells
            else validate_cube_reference_training_report(
                prereg,
                model_family=model_family,
                training_seed=seed,
                prereg_path=Path(prereg["_config_path"]),
            )
        )
        expected_checkpoint = {
            "path": str(training_cell["checkpoint"]),
            "sha256": training_cell["checkpoint_sha256"],
            "size_bytes": training_cell["checkpoint_size_bytes"],
            "model_state_sha256": training_cell["model_state_sha256"],
            "training_report": str(training_cell["report"]),
        }
        if (
            model.get("training_checkpoint") != expected_checkpoint
            or model.get("state_sha256_before") != training_cell["model_state_sha256"]
            or model.get("state_sha256_after") != training_cell["model_state_sha256"]
            or row.get("gate")
            != cube_grasp_rule_prediction_gate(row.get("metrics", {}), release=prereg)
            or row.get("public_test")
            != {
                "access_status": "closed_not_read_not_scored",
                "opened": False,
                "read": False,
                "hashed": False,
                "scored": False,
            }
        ):
            raise RuntimeError(f"Cube {model_family} embedded result provenance drifted")
        observed_seeds.append(seed)
    if sorted(observed_seeds) != expected_seeds:
        raise RuntimeError(f"Cube {model_family} embedded seed set drifted")
    expected_aggregate = {
        name: {
            "mean": float(statistics.mean(row["metrics"][name] for row in rows)),
            "minimum": float(min(row["metrics"][name] for row in rows)),
            "maximum": float(max(row["metrics"][name] for row in rows)),
        }
        for name in names
    }
    expected_pass = all(row["gate"]["passed"] for row in rows)
    if method.get("aggregate") != expected_aggregate or method.get("passed") is not expected_pass:
        raise RuntimeError(f"Cube {model_family} Development aggregate drifted")
    return method


__all__ = [
    "CUBE_REFERENCE_DEVELOPMENT_BENCHMARK",
    "CubeGraspRuleDevelopmentDataset",
    "evaluate_cube_reference_development_checkpoint",
    "score_cube_reference_development_results",
    "validate_cube_reference_development_method",
]
