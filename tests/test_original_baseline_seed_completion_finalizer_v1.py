from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]

CARRIED = {
    ("pusht", "lewm"): 3073,
    ("pusht", "pldm"): 3072,
    ("reacher", "lewm"): 3072,
    ("reacher", "pldm"): 3072,
    ("tworoom", "pldm"): 3072,
    ("cube", "lewm"): 3073,
    ("cube", "pldm"): 3073,
}
TWOROOM_SEEDS = (42, 43, 44, 45, 46, 47)
STANDARD_SEEDS = {"pusht": (42, 43, 44, 45, 46, 47), "reacher": (42, 43, 44), "cube": (42, 43, 44)}
QUERIES = {"tworoom": 50, "pusht": 50, "reacher": 100, "cube": 100}


def _load_finalizer():
    path = ROOT / "scripts/finalize_contextworld_original_baseline_seed_completion_v1.py"
    spec = importlib.util.spec_from_file_location("seed_completion_finalizer_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _identity(module: Any, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": module.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _outcomes(seed: int, count: int) -> list[bool]:
    return [((seed + index) % 4) != 0 for index in range(count)]


def _digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _state_pair(digest: str) -> dict[str, Any]:
    state = {"state_dict_sha256": digest, "parameter_count": 7}
    return {"before": state, "after": dict(state), "passed": True}


class Fixture:
    def __init__(self, tmp_path: Path):
        self.module = _load_finalizer()
        self.root = tmp_path
        self.commits = {"tworoom": "a" * 40, "standard": "b" * 40}
        self._build_inputs()
        self._build_cells()
        self._build_prereg_and_policy_receipts()
        self._build_receipts()
        self._build_parent_freeze()
        self._build_recovery()

    # ---- fixture construction -------------------------------------------
    def _build_inputs(self) -> None:
        self.inputs: dict[str, dict[str, Any]] = {}
        for environment in ("tworoom", "pusht", "reacher", "cube"):
            directory = self.root / "inputs" / environment
            directory.mkdir(parents=True)
            catalog = directory / "catalog.json"
            dataset = directory / "dataset.h5"
            catalog.write_text(f"{environment}-catalog\n", encoding="utf-8")
            dataset.write_bytes(f"{environment}-dataset".encode("utf-8"))
            row = {
                "catalog": _identity(self.module, catalog),
                "dataset": _identity(self.module, dataset),
            }
            if environment == "tworoom":
                normalizer = directory / "normalizer.json"
                normalizer.write_text("{}\n", encoding="utf-8")
                row["normalizer"] = _identity(self.module, normalizer)
            self.inputs[environment] = row
        plan_dir = self.root / "plan"
        plan_dir.mkdir()
        self.plan_configs = {}
        for environment in ("pusht", "reacher", "cube"):
            plan = plan_dir / f"{environment}.yaml"
            plan.write_text(f"plan: {environment}\n", encoding="utf-8")
            identity = _identity(self.module, plan)
            self.plan_configs[environment] = {
                "path": f"plan/{environment}.yaml",
                "sha256": identity["sha256"],
                "size_bytes": identity["size_bytes"],
            }

    def _build_cells(self) -> None:
        runner_by_env_family = {
            ("pusht", "lewm"): "standard_runner",
            ("pusht", "pldm"): "standard_runner",
            ("reacher", "lewm"): "standard_runner",
            ("reacher", "pldm"): "standard_runner",
            ("cube", "lewm"): "cube_lewm_evaluator_v2",
            ("cube", "pldm"): "cube_pldm_wrapper",
            ("tworoom", "lewm"): "tworoom_runner",
            ("tworoom", "pldm"): "tworoom_runner",
        }
        new_seed_plan: list[tuple[str, str, int]] = []
        for (environment, family), carried_seed in CARRIED.items():
            for seed in (3072, 3073, 3074):
                if seed != carried_seed:
                    new_seed_plan.append((environment, family, seed))
        new_seed_plan.extend(("tworoom", "lewm", seed) for seed in (3072, 3073, 3074))
        assert len(new_seed_plan) == 17
        self.cells: list[dict[str, Any]] = []
        for environment, family, seed in new_seed_plan:
            cell_id = f"{environment}_{family}_seed{seed}"
            directory = self.root / "checkpoints" / cell_id
            directory.mkdir(parents=True)
            weight = directory / "weights.ckpt"
            config = directory / "config.yaml"
            weight.write_bytes(cell_id.encode("utf-8"))
            config.write_text(f"model: {cell_id}\n", encoding="utf-8")
            eval_seeds = TWOROOM_SEEDS if environment == "tworoom" else STANDARD_SEEDS[environment]
            cell = {
                "cell_id": cell_id,
                "environment": environment,
                "family": family,
                "training_seed": seed,
                "runner": runner_by_env_family[(environment, family)],
                "mujoco_gl": "egl",
                "checkpoint": _identity(self.module, weight),
                "effective_loader_config": _identity(self.module, config),
                "eval_seeds": list(eval_seeds),
                "queries_per_seed": QUERIES[environment],
                "evaluations": 300,
                "output_directory": str(
                    self.root
                    / "artifacts/evaluation/original_baseline_seed_completion_v1"
                    / environment
                    / family
                    / f"seed{seed}"
                ),
            }
            if environment == "tworoom":
                cell["output_files"] = [f"seed{value}.json" for value in eval_seeds]
            self.cells.append(cell)

    def _build_prereg_and_policy_receipts(self) -> None:
        implementation_dir = self.root / "scripts"
        implementation_dir.mkdir()
        self.implementation = {}
        for name in ("standard_runner", "tworoom_runner", "cube_pldm_wrapper", "cube_lewm_evaluator_v2"):
            path = implementation_dir / f"{name}.py"
            path.write_text(f"# frozen {name}\n", encoding="utf-8")
            self.implementation[name] = _identity(self.module, path)
        self.implementation["note"] = "non-identity entry is skipped"
        namespace = self.root / "artifacts/evaluation/original_baseline_seed_completion_v1"
        namespace.mkdir(parents=True)
        preflight = self.root / "configs/benchmark/preflight.json"
        freeze = self.root / "configs/benchmark/freeze.json"
        _write_json(preflight, {"passed": True})
        _write_json(freeze, {"status": "frozen"})
        carried_members: list[dict[str, Any]] = [
            {
                "environment": environment,
                "family": family,
                "training_seed": seed,
                "successes": 150,
                "evaluations": 300,
                "source": "original_baseline_cem_v1",
            }
            for (environment, family), seed in CARRIED.items()
        ]
        carried_members.append(
            {
                "environment": "tworoom",
                "family": "lewm",
                "training_seed": 3072,
                "successes": 273,
                "evaluations": 300,
                "source": "original_baseline_cem_v1",
                "family_statistics_membership": "excluded_lineage_note_only",
                "reason": "repo-trained lineage, lightning triplet evaluated instead",
            }
        )
        self.prereg_payload = {
            "schema_version": 1,
            "preregistration_id": "contextworld_original_baseline_seed_completion_v1",
            "status": "frozen_before_cem_execution",
            "scientific_scope": {
                "environments": ["tworoom", "pusht", "reacher", "cube"],
                "families": ["lewm", "pldm"],
                "training_seed_set_per_family": [3072, 3073, 3074],
                "newly_executed_member_cells": 17,
                "newly_executed_episodes": 5100,
                "formal_suite_scoreboard_eligible": False,
                "cross_environment_average_authorized": False,
                "pass_fail_threshold": None,
            },
            "authority": {
                field: False
                for field in (
                    "training_authorized",
                    "finetuning_authorized",
                    "checkpoint_selection_authorized",
                    "model_or_recipe_change_authorized",
                    "result_based_retry_authorized",
                    "checkpoint_swap_authorized",
                    "public_test_access_authorized",
                    "formal_scoreboard_mutation_authorized",
                )
            },
            "runtimes": {
                "tworoom": {"root": str(self.root / "rt-tworoom"), "expected_commit": self.commits["tworoom"]},
                "pusht_reacher_cube": {
                    "root": str(self.root / "rt-standard"),
                    "expected_commit": self.commits["standard"],
                    "plan_configs": self.plan_configs,
                },
            },
            "frozen_environment_inputs": self.inputs,
            "implementation": self.implementation,
            "already_evaluated_family_members": carried_members,
            "new_member_cells": self.cells,
            "execution_policy": {
                "output_root": "artifacts/evaluation/original_baseline_seed_completion_v1",
                "preflight_receipt": str(preflight),
                "freeze_receipt": str(freeze),
                "result_summary": "artifacts/evaluation/original_baseline_seed_completion_v1/family_summary.json",
            },
        }
        self.prereg_path = self.root / (
            "configs/benchmark/contextworld_original_baseline_seed_completion_prereg_v1.yaml"
        )
        self.prereg_path.parent.mkdir(parents=True, exist_ok=True)
        self.prereg_path.write_text(yaml.safe_dump(self.prereg_payload), encoding="utf-8")

    def _tworoom_receipt(self, cell: dict[str, Any], seed: int) -> dict[str, Any]:
        outcomes = _outcomes(seed + cell["training_seed"], 50)
        digest = _digest(f"state-{cell['cell_id']}")
        return {
            "status": "passed",
            "protocol": {
                "eval_seed": seed,
                "evaluations": 50,
                "history_size": 3,
                "action_block": 5,
                "eval_budget": 50,
                "horizon": 5,
                "receding_horizon": 5,
                "cem_samples": 300,
                "cem_steps": 30,
                "cem_topk": 30,
            },
            "frozen_input_preflight": {
                "passed": True,
                "runtime": {
                    "root": str(self.root / "rt-tworoom"),
                    "commit": self.commits["tworoom"],
                    "clean": True,
                },
                "checkpoint": dict(cell["checkpoint"]),
                "config": dict(cell["effective_loader_config"]),
                "catalog": dict(self.inputs["tworoom"]["catalog"]),
                "normalizer": dict(self.inputs["tworoom"]["normalizer"]),
                "source_dataset": dict(self.inputs["tworoom"]["dataset"]),
            },
            "stable_worldmodel": {"commit": self.commits["tworoom"]},
            "frozen_weight_audit": {
                "passed": True,
                "state_dict_sha256_before": digest,
                "state_dict_sha256_after": digest,
            },
            "raw_records": [
                {"eval_seed": seed, "evaluation_index": index, "success": success}
                for index, success in enumerate(outcomes)
            ],
            "aggregate": {
                "evaluations": 50,
                "successes": sum(outcomes),
                "success_rate": sum(outcomes) / 50,
            },
        }

    def _standard_receipt(self, cell: dict[str, Any]) -> dict[str, Any]:
        digest = _digest(f"state-{cell['cell_id']}")
        environment = cell["environment"]
        seeds = STANDARD_SEEDS[environment]
        queries = QUERIES[environment]
        seed_rows = []
        audit_rows = []
        for seed in seeds:
            outcomes = _outcomes(seed + cell["training_seed"], queries)
            seed_rows.append(
                {
                    "eval_seed": seed,
                    "episode_successes": outcomes,
                    "success_count": sum(outcomes),
                    "evaluation_count": queries,
                    "success_rate": sum(outcomes) / queries,
                    "frozen_state_audit": _state_pair(digest),
                }
            )
            audit_rows.append({"eval_seed": seed, **_state_pair(digest)})
        total = sum(row["success_count"] for row in seed_rows)
        return {
            "status": "standard_original_task_real_environment_cem",
            "task": environment,
            "runtime": {
                "root": str(self.root / "rt-standard"),
                "commit": self.commits["standard"],
                "clean": True,
            },
            "protocol": {
                "history_len": 3,
                "goal_offset_steps": 25,
                "eval_budget": 50,
                "horizon": 5,
                "receding_horizon": 5,
                "action_block": 5,
                "cem_samples": 300,
                "cem_iterations": 30,
                "cem_topk": 30,
                "eval_seeds": list(seeds),
                "num_eval_per_seed": queries,
                "dataset": dict(self.inputs[environment]["dataset"]),
            },
            "query_catalog": {"frozen_source": dict(self.inputs[environment]["catalog"])},
            "public_test": {
                "contextworld_public_test_read": False,
                "contextworld_public_test_scored": False,
            },
            "model": {
                "model": cell["cell_id"],
                "checkpoint": cell["checkpoint"]["path"],
                "checkpoint_sha256": cell["checkpoint"]["sha256"],
                "checkpoint_size_bytes": cell["checkpoint"]["size_bytes"],
                "config": cell["effective_loader_config"]["path"],
                "config_sha256": cell["effective_loader_config"]["sha256"],
                "config_size_bytes": cell["effective_loader_config"]["size_bytes"],
                "frozen_state_audit": {
                    "passed": True,
                    "scope": "actual_policy_model_per_seed",
                    "seeds": audit_rows,
                },
                "seeds": seed_rows,
                "aggregate": {
                    "success_count": total,
                    "evaluation_count": 300,
                    "success_rate": total / 300,
                },
            },
        }

    def _cube_pldm_receipt(self, cell: dict[str, Any]) -> dict[str, Any]:
        digest = _digest(f"state-{cell['cell_id']}")
        seed_rows = []
        for seed in STANDARD_SEEDS["cube"]:
            outcomes = _outcomes(seed + cell["training_seed"], 100)
            seed_rows.append(
                {
                    "eval_seed": seed,
                    "episode_successes": outcomes,
                    "success_count": sum(outcomes),
                    "evaluation_count": 100,
                    "success_rate": sum(outcomes) / 100,
                    "actual_evaluated_model_state": _state_pair(digest),
                }
            )
        total = sum(row["success_count"] for row in seed_rows)
        return {
            "status": "standard_original_task_real_environment_cem",
            "task": "cube",
            "runtime": {
                "root": str(self.root / "rt-standard"),
                "commit": self.commits["standard"],
                "clean": True,
            },
            "public_test": {"opened": False, "read": False, "hashed": False, "scored": False},
            "query_catalog": {
                "frozen_source": self.inputs["cube"]["catalog"]["path"],
                "sha256": self.inputs["cube"]["catalog"]["sha256"],
            },
            "model": {
                "family": "pldm",
                "strict_load": True,
                "checkpoint": dict(cell["checkpoint"]),
                "config": dict(cell["effective_loader_config"]),
                "loaded_state_consistent_across_seeds": True,
                "loaded_state_dict_sha256": digest,
                "seeds": seed_rows,
                "aggregate": {
                    "success_count": total,
                    "evaluation_count": 300,
                    "success_rate": total / 300,
                },
            },
        }

    def _cube_lewm_receipt(self, cell: dict[str, Any]) -> dict[str, Any]:
        seed_rows = []
        for seed in STANDARD_SEEDS["cube"]:
            outcomes = _outcomes(seed + cell["training_seed"], 100)
            seed_rows.append(
                {
                    "eval_seed": seed,
                    "episode_successes": outcomes,
                    "success_count": sum(outcomes),
                    "query_count": 100,
                    "success_rate": sum(outcomes) / 100,
                }
            )
        total = sum(row["success_count"] for row in seed_rows)
        dataset = self.inputs["cube"]["dataset"]
        return {
            "status": "standard_original_task_real_environment_cem",
            "task": "cube",
            "runtime": {
                "root": str(self.root / "rt-standard"),
                "commit": self.commits["standard"],
                "clean": True,
            },
            "protocol": {
                "history_len": 3,
                "goal_offset_steps": 25,
                "eval_budget": 50,
                "horizon": 5,
                "receding_horizon": 5,
                "action_block": 5,
                "cem_samples": 300,
                "cem_iterations": 30,
                "cem_topk": 30,
                "eval_seeds": list(STANDARD_SEEDS["cube"]),
                "num_eval_per_seed": 100,
                "dataset": dataset["path"],
                "dataset_sha256": dataset["sha256"],
                "dataset_size_bytes": dataset["size_bytes"],
                "source": str(self.root / "plan/cube.yaml"),
                "source_sha256": self.plan_configs["cube"]["sha256"],
                "videos_written": False,
            },
            "public_test": {"opened": False, "read": False, "hashed": False, "scored": False},
            "query_catalog": {
                "frozen_source": self.inputs["cube"]["catalog"]["path"],
                "sha256": self.inputs["cube"]["catalog"]["sha256"],
            },
            "models": [
                {
                    "model": "baseline_lewm",
                    "checkpoint": cell["checkpoint"]["path"],
                    "checkpoint_sha256": cell["checkpoint"]["sha256"],
                    "config": cell["effective_loader_config"]["path"],
                    "config_sha256": cell["effective_loader_config"]["sha256"],
                    "seeds": seed_rows,
                    "aggregate": {
                        "success_count": total,
                        "evaluation_count": 300,
                        "success_rate": total / 300,
                    },
                }
            ],
        }

    def _build_receipts(self) -> None:
        for cell in self.cells:
            directory = Path(cell["output_directory"])
            if cell["environment"] == "tworoom":
                for seed in cell["eval_seeds"]:
                    _write_json(directory / f"seed{seed}.json", self._tworoom_receipt(cell, seed))
            elif cell["runner"] == "standard_runner":
                _write_json(directory / "aggregate.json", self._standard_receipt(cell))
            elif cell["runner"] == "cube_pldm_wrapper":
                _write_json(directory / "aggregate.json", self._cube_pldm_receipt(cell))
            else:
                _write_json(directory / "aggregate.json", self._cube_lewm_receipt(cell))

    def _build_parent_freeze(self) -> None:
        summary_path = self.root / "parent/matrix_summary.json"
        _write_json(summary_path, {"summary_id": "contextworld_original_baseline_cem_matrix_v1"})
        cells = [
            {
                "environment": environment,
                "family": family,
                "success_count": 150,
                "evaluation_count": 300,
                "success_rate": 0.5,
            }
            for (environment, family) in CARRIED
        ]
        cells.append(
            {
                "environment": "tworoom",
                "family": "lewm",
                "success_count": 273,
                "evaluation_count": 300,
                "success_rate": 0.91,
            }
        )
        _write_json(
            self.root / "configs/benchmark/contextworld_original_baseline_cem_results_freeze_v1.json",
            {
                "freeze_id": "contextworld_original_baseline_cem_results_freeze_v1",
                "matrix_summary": _identity(self.module, summary_path),
                "cells": cells,
            },
        )

    def _build_recovery(self) -> None:
        recovery_path = self.root / (
            "configs/benchmark/original_baseline_seed_completion_tworoom_pldm_seed3074_"
            "eval43_infra_relaunch_recovery_v1.yaml"
        )
        recovery_path.write_text(
            yaml.safe_dump(
                {
                    "recovery_id": self.module.EVAL43_RECOVERY_ID,
                    "scope": {
                        "result_observed_before_failure": False,
                        "result_based_retry": False,
                        "relaunch_count_authorized": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        seed43 = (
            self.root
            / "artifacts/evaluation/original_baseline_seed_completion_v1/tworoom/pldm/seed3074/seed43.json"
        )
        receipt = {
            "recovery_id": self.module.EVAL43_RECOVERY_ID,
            "relaunch": {
                "job_id": "tworoom_pldm_seed3074_eval43",
                "exit_code": 0,
                "output": _identity(self.module, seed43),
            },
        }
        _write_json(
            self.root
            / "artifacts/evaluation/original_baseline_seed_completion_v1/tworoom/pldm/seed3074/"
            "eval43_infra_relaunch_recovery_v1.json",
            receipt,
        )

    # ---- helpers ---------------------------------------------------------
    def rewrite_prereg(self) -> None:
        self.prereg_path.write_text(yaml.safe_dump(self.prereg_payload), encoding="utf-8")

    def finalize(self):
        return self.module.finalize(prereg_path=self.prereg_path, repo_root=self.root)

    @property
    def summary_path(self) -> Path:
        return (
            self.root
            / "artifacts/evaluation/original_baseline_seed_completion_v1/family_summary.json"
        )


@pytest.fixture()
def fixture(tmp_path: Path) -> Fixture:
    return Fixture(tmp_path)


def test_finalize_writes_family_summary(fixture: Fixture) -> None:
    summary = fixture.finalize()
    assert fixture.summary_path.is_file()
    assert summary["counts"]["newly_executed_member_cells"] == 17
    assert len(summary["families"]) == 8
    by_key = {(row["environment"], row["family"]): row for row in summary["families"]}
    for family in summary["families"]:
        assert tuple(member["training_seed"] for member in family["members"]) == (3072, 3073, 3074)
        statistics = family["statistics"]
        rates = statistics["success_rates"]
        mean = sum(rates) / 3
        assert statistics["mean"] == pytest.approx(mean)
        variance = sum((rate - mean) ** 2 for rate in rates) / 2
        assert statistics["sample_variance"] == pytest.approx(variance)
        assert statistics["sample_std"] == pytest.approx(math.sqrt(variance))
    tworoom_lewm = by_key[("tworoom", "lewm")]
    assert len(tworoom_lewm["lineage_notes"]) == 1
    note = tworoom_lewm["lineage_notes"][0]
    assert note["success_count"] == 273 and note["excluded_from_family_statistics"] is True
    assert all(
        member["provenance"] == "new_cell_this_preregistration"
        for member in tworoom_lewm["members"]
    )
    disclosures = summary["execution_disclosures"]
    assert disclosures[0]["job_id"] == "tworoom_pldm_seed3074_eval43"
    assert disclosures[0]["recovery_id"] == fixture.module.EVAL43_RECOVERY_ID


def test_missing_relaunch_receipt_fails(fixture: Fixture) -> None:
    receipt = (
        fixture.root
        / "artifacts/evaluation/original_baseline_seed_completion_v1/tworoom/pldm/seed3074/"
        "eval43_infra_relaunch_recovery_v1.json"
    )
    receipt.unlink()
    with pytest.raises(FileNotFoundError):
        fixture.finalize()
    assert not fixture.summary_path.exists()


def test_checkpoint_drift_fails(fixture: Fixture) -> None:
    cell = next(row for row in fixture.cells if row["cell_id"] == "pusht_lewm_seed3072")
    cell["checkpoint"]["sha256"] = "0" * 64
    fixture.rewrite_prereg()
    with pytest.raises(fixture.module.FinalizationError, match="checkpoint"):
        fixture.finalize()
    assert not fixture.summary_path.exists()


def test_recount_mismatch_fails(fixture: Fixture) -> None:
    receipt_path = (
        fixture.root
        / "artifacts/evaluation/original_baseline_seed_completion_v1/reacher/pldm/seed3073/aggregate.json"
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["model"]["aggregate"]["success_count"] += 1
    payload["model"]["aggregate"]["success_rate"] = (
        payload["model"]["aggregate"]["success_count"] / 300
    )
    _write_json(receipt_path, payload)
    with pytest.raises(fixture.module.FinalizationError, match="aggregate"):
        fixture.finalize()


def test_carried_member_binding_fails_on_drift(fixture: Fixture) -> None:
    freeze_path = (
        fixture.root
        / "configs/benchmark/contextworld_original_baseline_cem_results_freeze_v1.json"
    )
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    payload["cells"][0]["success_count"] += 1
    _write_json(freeze_path, payload)
    with pytest.raises(fixture.module.FinalizationError, match="frozen parent matrix"):
        fixture.finalize()


def test_refuses_to_overwrite_summary(fixture: Fixture) -> None:
    fixture.summary_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        fixture.finalize()


def test_partial_tworoom_cell_fails(fixture: Fixture) -> None:
    receipt = (
        fixture.root
        / "artifacts/evaluation/original_baseline_seed_completion_v1/tworoom/lewm/seed3073/seed44.json"
    )
    receipt.unlink()
    with pytest.raises(FileNotFoundError):
        fixture.finalize()
