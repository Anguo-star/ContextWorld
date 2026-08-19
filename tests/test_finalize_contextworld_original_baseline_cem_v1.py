from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_finalizer():
    path = ROOT / "scripts/finalize_contextworld_original_baseline_cem_v1.py"
    spec = importlib.util.spec_from_file_location("cem_finalizer_under_test", path)
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


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _aggregate(successes: list[bool]) -> dict[str, Any]:
    count = sum(successes)
    return {
        "success_count": count,
        "evaluation_count": len(successes),
        "success_rate": count / len(successes),
    }


def _outcomes(seed: int, count: int) -> list[bool]:
    return [((seed + index) % 5) != 0 for index in range(count)]


def _state_pair(digest: str, *, parameters: int = 11) -> dict[str, Any]:
    state = {"state_dict_sha256": digest, "parameter_count": parameters}
    return {"before": state, "after": dict(state), "passed": True}


def _protocol(
    *, dataset: dict[str, Any], seeds: tuple[int, ...], queries_per_seed: int
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "eval_seeds": list(seeds),
        "num_eval_per_seed": queries_per_seed,
        "evaluation_count": len(seeds) * queries_per_seed,
        "goal_offset_steps": 25,
        "eval_budget": 50,
        "history_len": 3,
        "horizon": 5,
        "receding_horizon": 5,
        "action_block": 5,
        "cem_samples": 300,
        "cem_iterations": 30,
        "cem_topk": 30,
        "videos_written": False,
    }


def _make_fixture(tmp_path: Path) -> tuple[Any, dict[str, Any]]:
    module = _load_finalizer()
    commits = {"tworoom": "a" * 40, "standard": "b" * 40}
    input_identities: dict[str, dict[str, Any]] = {}
    for environment in ("tworoom", "pusht", "reacher", "cube"):
        directory = tmp_path / "inputs" / environment
        directory.mkdir(parents=True)
        catalog = directory / "catalog.json"
        dataset = directory / "dataset.h5"
        catalog.write_text(f"{environment}-catalog\n", encoding="utf-8")
        dataset.write_bytes(f"{environment}-dataset".encode("utf-8"))
        row: dict[str, Any] = {
            "catalog": _identity(module, catalog),
            "dataset": _identity(module, dataset),
        }
        if environment == "tworoom":
            normalizer = directory / "normalizer.json"
            normalizer.write_text("{}\n", encoding="utf-8")
            row["normalizer"] = _identity(module, normalizer)
        input_identities[environment] = row

    weights: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    for environment in ("tworoom", "pusht", "reacher", "cube"):
        for family in ("lewm", "pldm"):
            checkpoint_id = f"{environment}_{family}_original"
            directory = tmp_path / "checkpoints" / checkpoint_id
            directory.mkdir(parents=True)
            weight = directory / "weights.ckpt"
            config = directory / "config.yaml"
            weight.write_bytes(checkpoint_id.encode("utf-8"))
            config.write_text(f"model: {checkpoint_id}\n", encoding="utf-8")
            weights[checkpoint_id] = _identity(module, weight)
            configs[checkpoint_id] = _identity(module, config)
            checkpoint_rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "environment": environment,
                    "family": family,
                    "weights": weights[checkpoint_id],
                }
            )

    registry = tmp_path / "base_checkpoint_registry.yaml"
    registry.write_text(
        yaml.safe_dump({"checkpoints": checkpoint_rows}, sort_keys=False),
        encoding="utf-8",
    )
    base_icl = tmp_path / "base_icl_freeze.json"
    _write_json(base_icl, {"status": "frozen"})
    implementation = tmp_path / "implementation.py"
    implementation.write_text("# synthetic frozen implementation\n", encoding="utf-8")
    input_identity_audit = tmp_path / "input_identity_audit.json"
    _write_json(
        input_identity_audit,
        {
            "schema_version": 1,
            "audit_id": "contextworld_original_baseline_cem_input_identity_audit_v1",
            "content_hash_authority": "full_file_sha256_streamed_before_cem_freeze",
            "datasets": {
                environment: {
                    **input_identities[environment]["dataset"],
                    "content_hash_checked": True,
                }
                for environment in ("tworoom", "pusht", "reacher", "cube")
            },
        },
    )

    preflight = tmp_path / "preflight.json"
    new_ids = [
        row["checkpoint_id"]
        for row in checkpoint_rows
        if row["checkpoint_id"] != "cube_lewm_original"
    ]
    _write_json(
        preflight,
        {
            "schema_version": 1,
            "preflight_id": "contextworld_original_baseline_cem_preflight_v1",
            "status": "passed_without_cem_execution",
            "passed": True,
            "cem_episodes_consumed": 0,
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "runtime_checkouts": {
                "tworoom": {
                    "root": str(tmp_path / "runtime" / "tworoom"),
                    "commit": commits["tworoom"],
                    "clean": True,
                },
                "pusht_reacher_cube": {
                    "root": str(tmp_path / "runtime" / "standard"),
                    "commit": commits["standard"],
                    "clean": True,
                },
            },
            "new_execution_models": [
                {
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_sha256": weights[checkpoint_id]["sha256"],
                    "config_sha256": configs[checkpoint_id]["sha256"],
                    "strict_load": True,
                }
                for checkpoint_id in new_ids
            ],
        },
    )

    results = tmp_path / "results"
    for family in ("lewm", "pldm"):
        checkpoint_id = f"tworoom_{family}_original"
        state_digest = _digest(f"state-{checkpoint_id}")
        for seed in (42, 43, 44, 45, 46, 47):
            outcomes = _outcomes(seed, 50)
            raw_records = [
                {
                    "eval_seed": seed,
                    "evaluation_index": index,
                    "success": success,
                }
                for index, success in enumerate(outcomes)
            ]
            report = {
                "status": "passed",
                "protocol": {
                    "history_size": 3,
                    "eval_seed": seed,
                    "evaluations": 50,
                    "eval_budget": 50,
                    "horizon": 5,
                    "receding_horizon": 5,
                    "action_block": 5,
                    "cem_samples": 300,
                    "cem_steps": 30,
                    "cem_topk": 30,
                },
                "frozen_input_preflight": {
                    "runtime": {
                        "root": str(tmp_path / "runtime" / "tworoom"),
                        "commit": commits["tworoom"],
                        "clean": True,
                    },
                    "checkpoint": weights[checkpoint_id],
                    "config": configs[checkpoint_id],
                    "catalog": input_identities["tworoom"]["catalog"],
                    "normalizer": input_identities["tworoom"]["normalizer"],
                    "source_dataset": input_identities["tworoom"]["dataset"],
                    "passed": True,
                },
                "checkpoint": {
                    "path": weights[checkpoint_id]["path"],
                    "sha256": weights[checkpoint_id]["sha256"],
                },
                "catalog": {
                    "path": input_identities["tworoom"]["catalog"]["path"],
                    "sha256": input_identities["tworoom"]["catalog"]["sha256"],
                },
                "normalizer": {
                    "path": input_identities["tworoom"]["normalizer"]["path"],
                    "sha256": input_identities["tworoom"]["normalizer"]["sha256"],
                },
                "stable_worldmodel": {
                    "repo": str(tmp_path / "runtime" / "tworoom"),
                    "commit": commits["tworoom"],
                },
                "frozen_weight_audit": {
                    "state_dict_sha256_before": state_digest,
                    "state_dict_sha256_after": state_digest,
                    "passed": True,
                },
                "aggregate": {
                    "successes": sum(outcomes),
                    "evaluations": 50,
                    "success_rate": sum(outcomes) / 50,
                },
                "raw_records": raw_records,
            }
            _write_json(
                results / "tworoom" / family / f"seed{seed}.json", report
            )

    def standard_report(environment: str, family: str) -> dict[str, Any]:
        checkpoint_id = f"{environment}_{family}_original"
        seeds = (42, 43, 44, 45, 46, 47) if environment == "pusht" else (42, 43, 44)
        queries = 50 if environment == "pusht" else 100
        digest = _digest(f"state-{checkpoint_id}")
        seed_rows = []
        audit_rows = []
        for seed in seeds:
            outcomes = _outcomes(seed, queries)
            audit = _state_pair(digest)
            seed_rows.append(
                {
                    "eval_seed": seed,
                    "episode_successes": outcomes,
                    **_aggregate(outcomes),
                    "frozen_state_audit": audit,
                }
            )
            audit_rows.append({"eval_seed": seed, **audit})
        all_outcomes = [value for row in seed_rows for value in row["episode_successes"]]
        return {
            "status": "standard_original_task_real_environment_cem",
            "task": environment,
            "runtime": {
                "root": str(tmp_path / "runtime" / "standard"),
                "commit": commits["standard"],
                "clean": True,
            },
            "protocol": _protocol(
                dataset=input_identities[environment]["dataset"],
                seeds=seeds,
                queries_per_seed=queries,
            ),
            "query_catalog": {
                "frozen_source": input_identities[environment]["catalog"],
            },
            "public_test": {
                "contextworld_public_test_read": False,
                "contextworld_public_test_scored": False,
            },
            "model": {
                "checkpoint": weights[checkpoint_id]["path"],
                "checkpoint_sha256": weights[checkpoint_id]["sha256"],
                "checkpoint_size_bytes": weights[checkpoint_id]["size_bytes"],
                "config": configs[checkpoint_id]["path"],
                "config_sha256": configs[checkpoint_id]["sha256"],
                "config_size_bytes": configs[checkpoint_id]["size_bytes"],
                "frozen_state_audit": {
                    "scope": "actual_policy_model_per_seed",
                    "passed": True,
                    "seeds": audit_rows,
                },
                "seeds": seed_rows,
                "aggregate": _aggregate(all_outcomes),
            },
        }

    for environment in ("pusht", "reacher"):
        for family in ("lewm", "pldm"):
            _write_json(
                results / environment / family / "aggregate.json",
                standard_report(environment, family),
            )

    cube_checkpoint = "cube_pldm_original"
    cube_digest = _digest("state-cube-pldm")
    cube_seed_rows = []
    for seed in (42, 43, 44):
        outcomes = _outcomes(seed, 100)
        cube_seed_rows.append(
            {
                "eval_seed": seed,
                "episode_successes": outcomes,
                **_aggregate(outcomes),
                "actual_evaluated_model_state": _state_pair(cube_digest),
            }
        )
    cube_outcomes = [value for row in cube_seed_rows for value in row["episode_successes"]]
    _write_json(
        results / "cube" / "pldm" / "aggregate.json",
        {
            "status": "standard_original_task_real_environment_cem",
            "task": "cube",
            "runtime": {
                "root": str(tmp_path / "runtime" / "standard"),
                "commit": commits["standard"],
                "clean": True,
            },
            "protocol": _protocol(
                dataset=input_identities["cube"]["dataset"],
                seeds=(42, 43, 44),
                queries_per_seed=100,
            ),
            "query_catalog": {
                "frozen_source": input_identities["cube"]["catalog"],
            },
            "public_test": {
                "opened": False,
                "read": False,
                "hashed": False,
                "scored": False,
            },
            "model": {
                "family": "pldm",
                "strict_load": True,
                "checkpoint": weights[cube_checkpoint],
                "config": configs[cube_checkpoint],
                "loaded_state_consistent_across_seeds": True,
                "loaded_state_dict_sha256": cube_digest,
                "seeds": cube_seed_rows,
                "aggregate": _aggregate(cube_outcomes),
            },
        },
    )

    cube_lewm = "cube_lewm_original"
    cube_lewm_rows = []
    for seed in (42, 43, 44):
        outcomes = _outcomes(seed, 100)
        cube_lewm_rows.append(
            {
                "eval_seed": seed,
                "episode_successes": outcomes,
                "query_count": 100,
                **_aggregate(outcomes),
            }
        )
    cube_lewm_outcomes = [
        value for row in cube_lewm_rows for value in row["episode_successes"]
    ]
    cube_aggregate = tmp_path / "cube_lewm" / "aggregate.json"
    _write_json(
        cube_aggregate,
        {
            "status": "standard_original_task_real_environment_cem",
            "task": "cube",
            "runtime": {
                "root": str(tmp_path / "runtime" / "standard"),
                "commit": commits["standard"],
                "clean": True,
            },
            "protocol": {
                **{
                    key: value
                    for key, value in _protocol(
                        dataset=input_identities["cube"]["dataset"],
                        seeds=(42, 43, 44),
                        queries_per_seed=100,
                    ).items()
                    if key != "dataset"
                },
                "dataset": input_identities["cube"]["dataset"]["path"],
                "dataset_sha256": input_identities["cube"]["dataset"]["sha256"],
                "dataset_size_bytes": input_identities["cube"]["dataset"]["size_bytes"],
            },
            "query_catalog": {
                "frozen_source": input_identities["cube"]["catalog"]["path"],
                "sha256": input_identities["cube"]["catalog"]["sha256"],
            },
            "models": [
                {
                    "model": "baseline_lewm",
                    "checkpoint": weights[cube_lewm]["path"],
                    "checkpoint_sha256": weights[cube_lewm]["sha256"],
                    "config_sha256": configs[cube_lewm]["sha256"],
                    "seeds": cube_lewm_rows,
                    "aggregate": _aggregate(cube_lewm_outcomes),
                }
            ],
        },
    )
    cube_receipt = tmp_path / "cube_lewm" / "freeze_receipt.json"
    _write_json(
        cube_receipt,
        {
            "status": "frozen_authorized",
            "model_preflight": {
                "runtime": {
                    "root": str(tmp_path / "runtime" / "standard"),
                    "commit": commits["standard"],
                    "clean": True,
                },
                "models": [
                    {
                        "model": "baseline_lewm",
                        "checkpoint": weights[cube_lewm]["path"],
                        "checkpoint_sha256": weights[cube_lewm]["sha256"],
                        "config_sha256": configs[cube_lewm]["sha256"],
                        "parameter_count": 11,
                        "strict_load": True,
                    }
                ],
            },
            "query_catalog": input_identities["cube"]["catalog"],
            "static_identities": {"dataset": input_identities["cube"]["dataset"]},
        },
    )

    execution_cells = []
    for environment in ("tworoom", "pusht", "reacher", "cube"):
        for family in ("lewm", "pldm"):
            if (environment, family) == ("cube", "lewm"):
                continue
            checkpoint_id = f"{environment}_{family}_original"
            seeds = (42, 43, 44, 45, 46, 47) if environment in {"tworoom", "pusht"} else (42, 43, 44)
            queries = 50 if environment in {"tworoom", "pusht"} else 100
            output_kind = "six_seed_receipts" if environment == "tworoom" else "aggregate"
            execution_cells.append(
                {
                    "environment": environment,
                    "family": family,
                    "checkpoint_id": checkpoint_id,
                    "runtime": "tworoom" if environment == "tworoom" else "pusht_reacher_cube",
                    "mujoco_gl": "egl" if environment in {"tworoom", "pusht"} else "osmesa",
                    "checkpoint": weights[checkpoint_id],
                    "effective_loader_config": configs[checkpoint_id],
                    "environment_inputs": environment,
                    "eval_seeds": list(seeds),
                    "queries_per_seed": queries,
                    "evaluations": 300,
                    "output_directory": str((results / environment / family).resolve()),
                    "output_kind": output_kind,
                    "output_files": (
                        [f"seed{seed}.json" for seed in seeds]
                        if output_kind == "six_seed_receipts"
                        else ["aggregate.json"]
                    ),
                }
            )

    output = results / "matrix_summary.json"
    prereg = tmp_path / "prereg.yaml"
    prereg_payload = {
        "schema_version": 1,
        "preregistration_id": "contextworld_original_baseline_cem_v1",
        "status": "frozen_before_cem_execution",
        "scientific_scope": {
            "environments": ["tworoom", "pusht", "reacher", "cube"],
            "families": ["lewm", "pldm"],
            "matrix_cells": 8,
            "exact_legacy_cells_reused": 1,
            "newly_executed_cells": 7,
            "total_matrix_episodes": 2400,
            "newly_executed_episodes": 2100,
            "formal_suite_scoreboard_eligible": False,
            "cross_environment_average_authorized": False,
            "pass_fail_threshold": None,
        },
        "authority": {
            "cem_execution_authorized": True,
            "authorized_new_cells": 7,
            "authorized_new_episodes": 2100,
            "training_authorized": False,
            "finetuning_authorized": False,
            "checkpoint_selection_authorized": False,
            "model_or_recipe_change_authorized": False,
            "result_based_retry_authorized": False,
            "checkpoint_swap_authorized": False,
            "public_test_access_authorized": False,
            "formal_scoreboard_mutation_authorized": False,
            "component_release_claim_mutation_authorized": False,
        },
        "base_checkpoint_registry": _identity(module, registry),
        "base_icl_result_freeze": _identity(module, base_icl),
        "input_identity_audit": _identity(module, input_identity_audit),
        "preflight": _identity(module, preflight),
        "protocol": {
            "history_tokens": 3,
            "action_block_raw_steps": 5,
            "goal_offset_raw_steps": 25,
            "execution_budget_raw_steps": 50,
            "horizon_action_blocks": 5,
            "receding_horizon_action_blocks": 5,
            "cem_candidates": 300,
            "cem_iterations": 30,
            "cem_topk": 30,
            "videos_written": False,
        },
        "runtimes": {
            "tworoom": {
                "expected_commit": commits["tworoom"],
                "clean_checkout_required": True,
            },
            "pusht_reacher_cube": {
                "expected_commit": commits["standard"],
                "clean_checkout_required": True,
            },
        },
        "frozen_environment_inputs": input_identities,
        "implementation": {"synthetic_finalizer_dependency": _identity(module, implementation)},
        "reuse_cells": [
            {
                "environment": "cube",
                "family": "lewm",
                "checkpoint_id": "cube_lewm_original",
                "source": _identity(module, cube_receipt),
                "result": _identity(module, cube_aggregate),
                "exact_reuse_contract_passed": True,
            }
        ],
        "execution_cells": execution_cells,
        "execution_policy": {
            "output_root": str(results.resolve()),
            "result_summary": str(output.resolve()),
        },
    }
    prereg.write_text(
        yaml.safe_dump(prereg_payload, sort_keys=False), encoding="utf-8"
    )
    return module, {
        "prereg": prereg,
        "results": results,
        "cube_aggregate": cube_aggregate,
        "cube_receipt": cube_receipt,
        "output": output,
        "standard_aggregate": results / "pusht" / "lewm" / "aggregate.json",
    }


def _finalize(module: Any, fixture: dict[str, Any]) -> dict[str, Any]:
    return module.finalize(
        prereg_path=fixture["prereg"],
        results_root=fixture["results"],
        cube_lewm_aggregate=fixture["cube_aggregate"],
        cube_lewm_freeze_receipt=fixture["cube_receipt"],
        output=fixture["output"],
        repo_root=fixture["prereg"].parent,
    )


def test_finalizer_closes_eight_cells_without_cross_environment_average(
    tmp_path: Path,
) -> None:
    module, fixture = _make_fixture(tmp_path)

    summary = _finalize(module, fixture)

    assert summary["counts"] == {
        "matrix_cells": 8,
        "episodes_per_cell": 300,
        "total_matrix_episodes": 2400,
    }
    assert len(summary["cells"]) == 8
    assert {row["evaluation_count"] for row in summary["cells"]} == {300}
    assert sum(row["success_count"] for row in summary["cells"]) > 0
    assert "cross_environment_average" not in summary
    assert summary["scope"]["cross_environment_average_reported"] is False
    assert fixture["output"].is_file()
    persisted = json.loads(fixture["output"].read_text(encoding="utf-8"))
    assert persisted["cells"] == summary["cells"]
    assert {row["provenance"] for row in summary["cells"]} == {
        "six_seed_receipts",
        "aggregate",
        "frozen_reuse_receipt_and_aggregate",
    }


def test_finalizer_rejects_missing_receipt_before_summary_write(tmp_path: Path) -> None:
    module, fixture = _make_fixture(tmp_path)
    (fixture["results"] / "tworoom" / "lewm" / "seed47.json").unlink()

    with pytest.raises(FileNotFoundError, match="TwoRoom lewm seed 47 receipt"):
        _finalize(module, fixture)

    assert not fixture["output"].exists()


def test_finalizer_rejects_actual_policy_model_state_drift_and_overwrite(
    tmp_path: Path,
) -> None:
    module, fixture = _make_fixture(tmp_path)
    aggregate = json.loads(fixture["standard_aggregate"].read_text(encoding="utf-8"))
    aggregate["model"]["seeds"][0]["frozen_state_audit"]["after"][
        "state_dict_sha256"
    ] = _digest("drift")
    _write_json(fixture["standard_aggregate"], aggregate)

    with pytest.raises(module.FinalizationError, match="model state changed"):
        _finalize(module, fixture)
    assert not fixture["output"].exists()

    # Recreate an entirely closed fixture to test exclusive final output write.
    module, fixture = _make_fixture(tmp_path / "exclusive")
    _finalize(module, fixture)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _finalize(module, fixture)
