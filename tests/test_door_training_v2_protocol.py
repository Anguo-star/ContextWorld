from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from contextworld.synthesis.config import (
    build_compiler,
    load_config,
    scenario_requests,
)
from contextworld.synthesis.validator import (
    validate_independent_seed_assignment,
    validate_minimum_episode_start_oracle,
)
from contextworld.synthesis.reset_constraints import (
    apply_tworoom_reset_constraints,
)
from contextworld.training.tworoom_data import (
    CATALOG_BY_GROUP,
    _catalog_split_audit,
    _validate_complete_synthesis_report,
    _validate_paired_door_catalogs,
)
from scripts.train_tworoom_step1 import _load_distributed_execution_contract


ROOT = Path(__file__).resolve().parents[1]
FIXED_CONFIG = (
    ROOT / "configs/synthesis/tworoom_door_fixed49_matched_v2.yaml"
)
MULTI_CONFIG = ROOT / "configs/synthesis/tworoom_door_multi_v2.yaml"
TRAINING_CONFIG = ROOT / "configs/benchmark/tworoom_door_training_v2.yaml"
TRAINING_RUNNER = ROOT / "scripts/run_h3_door_train.sh"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _complete_synthesis_report(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    catalog = (tmp_path / "catalog.json").resolve()
    manifest = (tmp_path / "manifest.jsonl").resolve()
    report = (tmp_path / "report.json").resolve()
    catalog.write_text("{}", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    payload = {
        "passed": True,
        "compile_only": False,
        "preflight_passed": True,
        "catalog": str(catalog),
        "manifest": str(manifest),
        "loader_compatibility": {"passed": True},
        "scenarios": [
            {"scenario_id": "scenario-a", "passed": True},
            {"scenario_id": "scenario-b", "passed": True},
        ],
        "collection_status": {
            "scenario-a": "collected",
            "scenario-b": "reused",
        },
    }
    return payload, report, catalog, manifest


def _run_complete_report_gate(
    payload: dict, report: Path, catalog: Path, manifest: Path
) -> dict:
    return _validate_complete_synthesis_report(
        payload,
        report_path=report,
        catalog_path=catalog,
        manifest_path=manifest,
        expected_scenario_ids={"scenario-a", "scenario-b"},
    )


def _requests(path: Path):
    return scenario_requests(_yaml(path))


def test_door_recipes_have_exact_paired_collection_budget() -> None:
    fixed = _yaml(FIXED_CONFIG)
    multi = _yaml(MULTI_CONFIG)
    fixed_requests = _requests(FIXED_CONFIG)
    multi_requests = _requests(MULTI_CONFIG)

    assert fixed["seed"] == multi["seed"]
    assert fixed["scenario_generation_seed"] == multi[
        "scenario_generation_seed"
    ]
    assert fixed["collection"] == multi["collection"]
    assert fixed["controlled_constants"] == multi["controlled_constants"] == {
        "agent_speed": 5.0
    }
    assert len(fixed_requests) == len(multi_requests) == 608
    assert Counter(row.split for row in fixed_requests) == {
        "train": 512,
        "val": 96,
    }
    assert Counter(row.split for row in multi_requests) == {
        "train": 512,
        "val": 96,
    }
    assert sum(row.episodes for row in fixed_requests if row.split == "train") == (
        512 * 32
    )
    assert sum(row.episodes for row in fixed_requests if row.split == "val") == (
        96 * 16
    )
    assert sum(row.episodes for row in multi_requests if row.split == "train") == (
        512 * 32
    )
    assert sum(row.episodes for row in multi_requests if row.split == "val") == (
        96 * 16
    )

    fixed_by_group = {row.seed_group: row for row in fixed_requests}
    multi_by_group = {row.seed_group: row for row in multi_requests}
    assert len(fixed_by_group) == len(multi_by_group) == 608
    assert fixed_by_group.keys() == multi_by_group.keys()
    for seed_group, left in fixed_by_group.items():
        right = multi_by_group[seed_group]
        assert left.split == right.split
        assert left.regime == right.regime
        assert left.episodes == right.episodes
        assert left.reset_constraints == right.reset_constraints
        assert left.atoms[0].kind == right.atoms[0].kind == "door_position"
        assert float(left.atoms[0].value) == 49.0


def test_door_recipes_assign_one_independent_seed_block_per_scenario() -> None:
    for config_path in (FIXED_CONFIG, MULTI_CONFIG):
        config = load_config(config_path)
        requests = scenario_requests(config)
        scenarios = build_compiler(config, ROOT).compile_all(requests)
        audit = validate_independent_seed_assignment(
            scenarios, config["validation"]["independent_seed_assignment"]
        )
        assert audit["passed"]
        assert audit["splits"]["train"]["observed"]["seed_groups"] == 512
        assert audit["splits"]["val"]["observed"]["seed_groups"] == 96
        assert audit["splits"]["train"]["observed"][
            "one_scenario_per_seed_group"
        ]
        assert audit["splits"]["val"]["observed"][
            "one_scenario_per_seed_group"
        ]

    fixed = load_config(FIXED_CONFIG)
    multi = load_config(MULTI_CONFIG)
    fixed_compiled = build_compiler(fixed, ROOT).compile_all(
        scenario_requests(fixed)
    )
    multi_compiled = build_compiler(multi, ROOT).compile_all(
        scenario_requests(multi)
    )
    fixed_seeds = {
        row.seed_group: (row.env_seed, row.policy_seed)
        for row in fixed_compiled
    }
    multi_seeds = {
        row.seed_group: (row.env_seed, row.policy_seed)
        for row in multi_compiled
    }
    assert fixed_seeds == multi_seeds


def test_paired_seed_groups_replay_the_same_reset_and_goal() -> None:
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    fixed = load_config(FIXED_CONFIG)
    multi = load_config(MULTI_CONFIG)
    fixed_rows = {
        row.seed_group: row
        for row in build_compiler(fixed, ROOT).compile_all(
            scenario_requests(fixed)
        )
    }
    multi_rows = {
        row.seed_group: row
        for row in build_compiler(multi, ROOT).compile_all(
            scenario_requests(multi)
        )
    }
    env = TwoRoomEnv(render_mode="rgb_array")
    try:
        for seed_group, left in fixed_rows.items():
            right = multi_rows[seed_group]
            apply_tworoom_reset_constraints(env, left.reset_constraints)
            left_observation, _ = env.reset(
                seed=left.env_seed,
                options={
                    "variation": left.variation,
                    "variation_values": left.variation_values,
                },
            )
            apply_tworoom_reset_constraints(env, right.reset_constraints)
            right_observation, _ = env.reset(
                seed=right.env_seed,
                options={
                    "variation": right.variation,
                    "variation_values": right.variation_values,
                },
            )
            np.testing.assert_array_equal(
                left_observation[:4], right_observation[:4]
            )
    finally:
        env.close()


def test_door_recipes_read_back_the_pinned_default_speed() -> None:
    for config_path in (FIXED_CONFIG, MULTI_CONFIG):
        config = load_config(config_path)
        scenario = build_compiler(config, ROOT).compile_all(
            scenario_requests(config)[:1]
        )[0]
        oracle = validate_minimum_episode_start_oracle(
            [scenario], config["validation"]["minimum_episode_start_oracle"]
        )
        assert oracle["passed"]
        assert oracle["expected_agent_speed"] == 5.0
        assert oracle["observed_agent_speeds"] == [5.0]
        assert oracle["agent_speed_readback_passed"]
        assert oracle["agent_speed_mismatches"] == []


def test_loader_validation_uses_training_doors_and_not_eval_holdouts() -> None:
    fixed = _requests(FIXED_CONFIG)
    multi = _requests(MULTI_CONFIG)
    benchmark = _yaml(TRAINING_CONFIG)
    heldout = set(benchmark["door_support"]["eval_heldout_values"])

    fixed_train = {
        int(row.atoms[0].value) for row in fixed if row.split == "train"
    }
    fixed_val = {int(row.atoms[0].value) for row in fixed if row.split == "val"}
    multi_train = {
        int(row.atoms[0].value) for row in multi if row.split == "train"
    }
    multi_val = {int(row.atoms[0].value) for row in multi if row.split == "val"}

    assert fixed_train == fixed_val == {49}
    assert multi_train == multi_val == set(
        benchmark["door_support"]["multi_synthetic_train"]
    )
    assert not heldout & (fixed_train | fixed_val | multi_train | multi_val)

    multi_val_counts = Counter(
        int(row.atoms[0].value) for row in multi if row.split == "val"
    )
    assert set(multi_val_counts.values()) == {6}


def test_door_groups_are_wired_to_the_additive_training_profile() -> None:
    benchmark = _yaml(TRAINING_CONFIG)
    protocol = benchmark["training_protocol"]

    assert protocol["profile"] == "additive"
    distributed = protocol["distributed_execution"]
    assert distributed["strategy"] == "ddp"
    assert distributed["transport_configuration"] == "framework_defaults"
    assert distributed["primary_formal_launch"] == "fresh"
    assert distributed["resume_role"] == "disaster_recovery_only"
    assert distributed["resume_scope"] == "complete_epoch_boundary_only"
    assert distributed["recovery_acceptance"] == {
        "single_gpu_epoch_boundary": {
            "parameter_equivalence": "bitwise",
        },
        "four_gpu_epoch_boundary": {
            "data_order": "exact",
            "rng_state": "exact",
            "global_step": "exact",
            "scheduler_state": "exact",
            "parameter_equivalence": "numerical",
            "maximum_absolute_parameter_difference": 2.0e-9,
            "bytewise_identity_required": False,
        },
    }
    verification = distributed["recovery_verification"]
    assert verification["single_gpu"] == {
        "passed": True,
        "compared_parameter_tensors": 303,
        "non_bitwise_parameter_tensors": 0,
        "observed_maximum_absolute_parameter_difference": 0.0,
        "serialized_pretrained_sha256_equal": True,
    }
    assert verification["four_gpu"]["passed"] is True
    assert verification["four_gpu"]["data_order_exact"] is True
    assert verification["four_gpu"]["rng_state_exact"] is True
    assert verification["four_gpu"]["global_step_exact"] is True
    assert verification["four_gpu"]["scheduler_state_exact"] is True
    assert verification["four_gpu"]["compared_parameter_tensors"] == 303
    assert verification["four_gpu"]["non_bitwise_parameter_tensors"] == 22
    assert (
        verification["four_gpu"][
            "observed_maximum_absolute_parameter_difference"
        ]
        <= distributed["recovery_acceptance"]["four_gpu_epoch_boundary"][
            "maximum_absolute_parameter_difference"
        ]
    )
    assert (
        verification["four_gpu"]["serialized_pretrained_sha256_equal"]
        is False
    )
    runner = TRAINING_RUNNER.read_text(encoding="utf-8")
    assert "NCCL_" not in runner
    assert protocol["paired_training_seeds"] == [3072, 4096, 5120]
    assert protocol["group_sampling"]["M_door_fixed49_v2"] == {
        "original": 0.5,
        "door_fixed49_v2": 0.5,
    }
    assert protocol["group_sampling"]["M_door_multi_v2"] == {
        "original": 0.5,
        "door_multi_v2": 0.5,
    }
    assert benchmark["models"] == [
        {
            "model_id": "M_door_fixed49_v2",
            "display_name": "History-3 固定门位置匹配控制",
            "training_groups": ["original", "door_fixed49_v2"],
        },
        {
            "model_id": "M_door_multi_v2",
            "display_name": "History-3 多门位置目标",
            "training_groups": ["original", "door_multi_v2"],
        },
    ]
    assert CATALOG_BY_GROUP["door_fixed49_v2"] == "door_fixed49_v2"
    assert CATALOG_BY_GROUP["door_multi_v2"] == "door_multi_v2"
    assert benchmark["data_quality"]["groups"]["door_fixed49_v2"][
        "require_complete_synthesis_report"
    ] is True
    assert benchmark["data_quality"]["groups"]["door_multi_v2"][
        "require_complete_synthesis_report"
    ] is True
    fixed_quality = benchmark["data_quality"]["groups"]["door_fixed49_v2"]
    multi_quality = benchmark["data_quality"]["groups"]["door_multi_v2"]
    for quality in (fixed_quality, multi_quality):
        assert quality["exact_train_scenarios"] == 512
        assert quality["exact_validation_scenarios"] == 96
        for key in (
            "required_catalog_sha256",
            "required_manifest_sha256",
            "required_synthesis_report_sha256",
        ):
            assert len(quality[key]) == 64
            int(quality[key], 16)
    assert fixed_quality["factor_support_contract"] == {
        "factor": "door.position",
        "train_values_from_door_support": "fixed_synthetic_train",
        "validation_values_from_door_support": "fixed_loader_validation",
    }
    assert multi_quality["factor_support_contract"] == {
        "factor": "door.position",
        "train_values_from_door_support": "multi_synthetic_train",
        "validation_values_from_door_support": "multi_loader_validation",
    }
    pairing = benchmark["paired_collection_contract"]
    assert pairing["reset_and_goal_pairing"] == "exact_by_seed_group"
    assert "may diverge" in pairing["expert_actions"]
    assert "identical action sequences are not required" in pairing[
        "expert_actions"
    ]


def test_complete_synthesis_report_gate_accepts_collected_or_reused(
    tmp_path: Path,
) -> None:
    payload, report, catalog, manifest = _complete_synthesis_report(tmp_path)

    audit = _run_complete_report_gate(payload, report, catalog, manifest)

    assert audit == {
        "required": True,
        "compile_only": False,
        "preflight_passed": True,
        "loader_compatibility_passed": True,
        "scenario_results": 2,
        "collection_status_entries": 2,
        "collection_status_counts": {"collected": 1, "reused": 1},
        "passed": True,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("passed", False, "did not pass"),
        ("compile_only", True, "not a completed collection"),
        ("compile_only", None, "not a completed collection"),
        ("preflight_passed", False, "preflight did not pass"),
        ("preflight_passed", None, "preflight did not pass"),
        ("loader_compatibility", None, "loader compatibility"),
        ("loader_compatibility", {"passed": False}, "loader compatibility"),
    ],
)
def test_complete_synthesis_report_gate_rejects_incomplete_top_level_state(
    tmp_path: Path, field: str, value, message: str
) -> None:
    payload, report, catalog, manifest = _complete_synthesis_report(tmp_path)
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value

    with pytest.raises(ValueError, match=message):
        _run_complete_report_gate(payload, report, catalog, manifest)


@pytest.mark.parametrize("field", ["catalog", "manifest"])
def test_complete_synthesis_report_gate_rejects_missing_artifact_binding(
    tmp_path: Path, field: str
) -> None:
    payload, report, catalog, manifest = _complete_synthesis_report(tmp_path)
    payload.pop(field)

    with pytest.raises(ValueError, match=f"missing {field}"):
        _run_complete_report_gate(payload, report, catalog, manifest)


def test_complete_synthesis_report_gate_requires_every_catalog_scenario(
    tmp_path: Path,
) -> None:
    payload, report, catalog, manifest = _complete_synthesis_report(tmp_path)
    payload["scenarios"].pop()
    with pytest.raises(ValueError, match="report/catalog scenario sets differ"):
        _run_complete_report_gate(payload, report, catalog, manifest)

    payload, report, catalog, manifest = _complete_synthesis_report(tmp_path)
    payload["scenarios"][1]["passed"] = False
    with pytest.raises(ValueError, match="scenario did not pass"):
        _run_complete_report_gate(payload, report, catalog, manifest)


def test_complete_synthesis_report_gate_requires_every_collection(
    tmp_path: Path,
) -> None:
    payload, report, catalog, manifest = _complete_synthesis_report(tmp_path)
    payload["collection_status"].pop("scenario-b")
    with pytest.raises(ValueError, match="collection/catalog scenario sets differ"):
        _run_complete_report_gate(payload, report, catalog, manifest)

    payload, report, catalog, manifest = _complete_synthesis_report(tmp_path)
    payload["collection_status"]["scenario-b"] = "pending"
    with pytest.raises(ValueError, match="not collected"):
        _run_complete_report_gate(payload, report, catalog, manifest)


def _minimal_catalog_artifact(
    root: Path, name: str, rows: list[dict]
) -> Path:
    catalog_dir = root / "synthesis" / "catalogs"
    manifest_dir = root / "synthesis" / "manifests"
    report_dir = root / "synthesis" / "reports"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    sections = {"train": [], "val": [], "ood_test": []}
    encoded_rows = []
    for index, source in enumerate(rows):
        row = dict(source)
        fingerprint = f"{index + 1:064x}"
        scenario_id = f"scenario-{row['split']}-{fingerprint[:10]}"
        output = root / "data" / scenario_id
        output.mkdir(parents=True)
        row.update(
            {
                "schema_version": 1,
                "scenario_id": scenario_id,
                "fingerprint": fingerprint,
                "output_path": str(output),
                "collection_status": "collected",
                "stable_worldmodel_commit": "stable-ref",
                "pixel_codec": {
                    "format": "png",
                    "compress_level": 1,
                    "lossless": True,
                },
            }
        )
        sections[row["split"]].append(str(output))
        encoded_rows.append(row)
    catalog_path = catalog_dir / f"{name}.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "train": {"synthetic": sections["train"]},
                "val": {"synthetic": sections["val"]},
                "ood_test": {"synthetic": sections["ood_test"]},
                "pixel_codec": {
                    "format": "png",
                    "compress_level": 1,
                    "lossless": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (manifest_dir / f"{name}.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in encoded_rows),
        encoding="utf-8",
    )
    (report_dir / f"{name}.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    return catalog_path


def _paired_row(seed_group: str, split: str, door: int) -> dict:
    return {
        "seed_group": seed_group,
        "split": split,
        "regime": "train_broad_cross" if split == "train" else "validation_id",
        "env_id": "swm/TwoRoom-v1",
        "env_seed": 101,
        "policy_seed": 202,
        "episodes": 32 if split == "train" else 16,
        "task": "tworoom",
        "max_episode_steps": 100,
        "image_shape": [224, 224],
        "reset_constraints": {"target_room": "opposite"},
        "factors": {"door.position": door},
    }


def test_catalog_gate_enforces_exact_counts_support_and_hashes(
    tmp_path: Path,
) -> None:
    catalog = _minimal_catalog_artifact(
        tmp_path,
        "door_fixed",
        [
            _paired_row("train-0", "train", 49),
            _paired_row("val-0", "val", 49),
        ],
    )
    audit = _catalog_split_audit(
        catalog,
        repo_root=tmp_path,
        expected_stablewm_commit="stable-ref",
        expected_split_scenario_counts={"train": 1, "validation": 1},
        factor_support_contract={
            "factor": "door.position",
            "expected_by_split": {"train": [49], "validation": [49]},
        },
    )
    assert audit["exact_split_scenario_counts"]["passed"]
    assert audit["factor_support"]["passed"]

    with pytest.raises(ValueError, match="exact split counts"):
        _catalog_split_audit(
            catalog,
            repo_root=tmp_path,
            expected_stablewm_commit="stable-ref",
            expected_split_scenario_counts={"train": 2, "validation": 1},
        )
    with pytest.raises(ValueError, match="factor support"):
        _catalog_split_audit(
            catalog,
            repo_root=tmp_path,
            expected_stablewm_commit="stable-ref",
            factor_support_contract={
                "factor": "door.position",
                "expected_by_split": {"train": [57], "validation": [49]},
            },
        )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        _catalog_split_audit(
            catalog,
            repo_root=tmp_path,
            expected_stablewm_commit="stable-ref",
            required_artifact_hashes={"catalog": "0" * 64},
        )


def test_paired_catalog_gate_allows_only_door_position_to_differ(
    tmp_path: Path,
) -> None:
    fixed = _minimal_catalog_artifact(
        tmp_path / "fixed", "fixed", [_paired_row("pair-0", "train", 49)]
    )
    multi = _minimal_catalog_artifact(
        tmp_path / "multi", "multi", [_paired_row("pair-0", "train", 89)]
    )
    config = {
        "data": {
            "catalogs": {
                "door_fixed49_v2": str(fixed),
                "door_multi_v2": str(multi),
            }
        }
    }
    audit = _validate_paired_door_catalogs(config, repo_root=tmp_path)
    assert audit["paired_seed_groups"] == 1
    assert audit["passed"]

    multi_manifest = multi.parent.parent / "manifests" / "multi.jsonl"
    record = json.loads(multi_manifest.read_text(encoding="utf-8"))
    record["policy_seed"] += 1
    multi_manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only door.position may differ"):
        _validate_paired_door_catalogs(config, repo_root=tmp_path)


def test_distributed_runtime_keeps_framework_transport_and_loads_recovery_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acceptance = {
        "single_gpu_epoch_boundary": {
            "parameter_equivalence": "bitwise",
        },
        "four_gpu_epoch_boundary": {
            "data_order": "exact",
            "rng_state": "exact",
            "global_step": "exact",
            "scheduler_state": "exact",
            "parameter_equivalence": "numerical",
            "maximum_absolute_parameter_difference": 2.0e-9,
            "bytewise_identity_required": False,
        },
    }
    verification = _yaml(TRAINING_CONFIG)["training_protocol"][
        "distributed_execution"
    ]["recovery_verification"]
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "training_protocol": {
                    "distributed_execution": {
                        "transport_configuration": "framework_defaults",
                        "primary_formal_launch": "fresh",
                        "resume_role": "disaster_recovery_only",
                        "resume_scope": "complete_epoch_boundary_only",
                        "recovery_acceptance": acceptance,
                        "recovery_verification": verification,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NCCL_ALGO", "Tree")

    audit = _load_distributed_execution_contract(config, devices=4)

    assert audit["runtime_mode"] == "multi_gpu"
    assert audit["transport_configuration"] == "framework_defaults"
    assert audit["transport_overrides_applied"] is False
    assert audit["recovery_acceptance"] == acceptance
    assert audit["recovery_verification"] == verification
    assert audit["resume_role"] == "disaster_recovery_only"
    assert os.environ["NCCL_ALGO"] == "Tree"

    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["training_protocol"]["distributed_execution"][
        "nccl_environment"
    ] = {"NCCL_ALGO": "Ring"}
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must not freeze NCCL"):
        _load_distributed_execution_contract(config, devices=4)


def test_distributed_recovery_gate_rejects_excess_four_gpu_drift(
    tmp_path: Path,
) -> None:
    benchmark = _yaml(TRAINING_CONFIG)
    benchmark["training_protocol"]["distributed_execution"][
        "recovery_verification"
    ]["four_gpu"][
        "observed_maximum_absolute_parameter_difference"
    ] = 2.0000001e-9
    config = tmp_path / "benchmark.yaml"
    config.write_text(yaml.safe_dump(benchmark), encoding="utf-8")

    with pytest.raises(ValueError, match="does not satisfy"):
        _load_distributed_execution_contract(config, devices=4)
