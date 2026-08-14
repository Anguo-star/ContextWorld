from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import yaml

from contextworld.benchmarks import cube_grasp_rule_reference_training as reference_training
from contextworld.benchmarks import cube_grasp_rule_reference_score as reference_score
from contextworld.benchmarks.cube_grasp_rule_reference_training import (
    CUBE_REFERENCE_TRAINING_ID,
    CUBE_REFERENCE_TRAINING_PROTOCOL,
    DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
    load_cube_reference_training_prereg,
)
import scripts.finalize_cube_grasp_rule_h3_v4r1_reference_development as finalizer
import scripts.freeze_cube_grasp_rule_h3_v4r1_reference_training as freezer
import scripts.run_cube_grasp_rule_h3_train as cube_train


ROOT = Path(__file__).resolve().parents[1]
PINNED_STABLE = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world/"
    "upstream/stable-worldmodel-875e607fc08aa72e"
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_reference_preregistration_loads_without_training_authorization() -> None:
    prereg = load_cube_reference_training_prereg(
        DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
        require_freeze=False,
    )
    assert prereg["preregistration_id"] == CUBE_REFERENCE_TRAINING_ID
    assert prereg["scope"]["flattened_action_input_dim"] == 25
    assert prereg["scope"]["authorized_splits"] == [
        "train",
        "loader_validation",
    ]
    assert prereg["public_test"]["opened"] is False
    assert (
        prereg["training"]["upstream"]["original_h5"]["path"]
        == prereg["training"]["upstream"]["original_h5"]["local_source"]
    )
    assert (
        prereg["training"]["reference_matrix"]["initial_checkpoints"]["lewm"]
        ["checkpoint"]
        == prereg["training"]["reference_matrix"]["initial_checkpoints"]["lewm"]
        ["local_source"]
    )
    assert prereg["infrastructure_recovery"]["prior_training_state"][
        "optimizer_steps"
    ] == 0
    assert prereg["training"]["reference_matrix"]["training_seeds"] == [
        17321,
        17322,
        17323,
    ]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("two_axis", "scope.raw_action_dim"),
        ("public_read", "public_test.read=false"),
        ("duplicate_seed", "three integer seeds"),
        ("variant_drift", "model variants drifted"),
        ("adaptive_stop", "execution policy drifted"),
        ("learning_rate", "fixed recipe"),
        ("worker_count", "fixed recipe"),
        ("inference_batch", "score only frozen Development"),
        ("missing_h5_alias", "original_h5.path must equal"),
        ("checkpoint_alias_drift", "initial_checkpoints.lewm.checkpoint must equal"),
        ("recovery_recipe_drift", "change only runtime compatibility"),
    ],
)
def test_reference_preregistration_rejects_contract_drift(
    tmp_path: Path, mutation: str, match: str
) -> None:
    payload = yaml.safe_load(
        DEFAULT_CUBE_REFERENCE_TRAINING_PREREG.read_text(encoding="utf-8")
    )
    if mutation == "two_axis":
        payload["scope"]["raw_action_dim"] = 2
    elif mutation == "public_read":
        payload["public_test"]["read"] = True
    elif mutation == "duplicate_seed":
        payload["training"]["reference_matrix"]["training_seeds"][-1] = 17322
    elif mutation == "variant_drift":
        payload["training"]["reference_matrix"]["models"]["pldm"][
            "variant"
        ] = "other"
    elif mutation == "adaptive_stop":
        payload["training"]["reference_matrix"]["execution_policy"][
            "adaptive_stopping"
        ] = True
    elif mutation == "learning_rate":
        payload["training"]["reference_matrix"]["common"]["learning_rate"] = 1e-4
    elif mutation == "worker_count":
        payload["training"]["reference_matrix"]["common"]["data_loader_workers"] = 8
    elif mutation == "missing_h5_alias":
        del payload["training"]["upstream"]["original_h5"]["path"]
    elif mutation == "checkpoint_alias_drift":
        payload["training"]["reference_matrix"]["initial_checkpoints"]["lewm"][
            "checkpoint"
        ] = "/different.ckpt"
    elif mutation == "recovery_recipe_drift":
        payload["infrastructure_recovery"]["recovery_change"][
            "model_recipe_changed"
        ] = True
    else:
        payload["evaluation"]["inference_batch_size"] = 128
    path = tmp_path / "prereg.yaml"
    _write_yaml(path, payload)
    with pytest.raises(ValueError, match=match):
        load_cube_reference_training_prereg(path, require_freeze=False)


def test_formal_cube_args_reject_scientific_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        release_config=DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
        data_root=None,
        variant=None,
        optimizer_steps=8,
        original_h5=None,
        original_lance=None,
        checkpoint=None,
        contrast_scales=None,
        num_workers=4,
        eval_batch_size=64,
        model="lewm",
        seed=17321,
        output=Path("unused"),
    )
    monkeypatch.setattr(cube_train.trainer, "parse_args", lambda: args)
    monkeypatch.setattr(
        cube_train,
        "load_cube_reference_training_prereg",
        lambda *args, **kwargs: {
            "training": {
                "reference_matrix": {
                    "common": {
                        "data_loader_workers": 4,
                        "loader_validation_batch_size": 64,
                    }
                }
            }
        },
    )
    cube_train._install_fail_closed_formal_args()
    with pytest.raises(ValueError, match="--optimizer-steps"):
        cube_train.trainer.parse_args()


def test_cube_reference_trainer_rejects_legacy_release(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "release_id": "legacy"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only the frozen v4r1"):
        cube_train._load_training_contract_for_requested_model(path)


def test_shared_trainer_resolves_v2_data_and_checkpoint_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prereg = load_cube_reference_training_prereg(
        DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
        require_freeze=False,
    )
    for name in (
        "CONTEXTWORLD_CUBE_H5",
        "CONTEXTWORLD_CUBE_LANCE",
        "CONTEXTWORLD_CUBE_LEWM_INIT_CHECKPOINT",
    ):
        monkeypatch.delenv(name, raising=False)
    original_h5, original_lance, checkpoint = cube_train.trainer._training_inputs(
        prereg,
        model="lewm",
        original_h5=None,
        original_lance=None,
        checkpoint=None,
    )
    assert original_h5 == Path(
        prereg["training"]["upstream"]["original_h5"]["path"]
    )
    assert original_lance == Path(
        prereg["training"]["upstream"]["original_lance"]["path"]
    )
    assert checkpoint == Path(
        prereg["training"]["reference_matrix"]["initial_checkpoints"]["lewm"]
        ["checkpoint"]
    )


def test_generated_data_tree_rejects_same_size_mutation(tmp_path: Path) -> None:
    train = tmp_path / "train.lance"
    development = tmp_path / "loader_validation.lance"
    train.mkdir()
    development.mkdir()
    (train / "data").write_bytes(b"train")
    (development / "data").write_bytes(b"develop")
    (tmp_path / "manifest.json").write_bytes(b"manifest")
    (tmp_path / "_SUCCESS.json").write_bytes(b"success")
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    prereg = {
        "data": {
            "artifact_tree": {
                "root": str(tmp_path),
                "files": len(files),
                "bytes": sum(path.stat().st_size for path in files),
                "tree_sha256_without_success_marker": reference_training._directory_sha256(
                    tmp_path, excluded=frozenset({"_SUCCESS.json"})
                ),
            },
            "lance_tables": {
                "train": "train.lance",
                "loader_validation": "loader_validation.lance",
            },
            "table_tree_sha256": {
                "train": reference_training._directory_sha256(train),
                "loader_validation": reference_training._directory_sha256(development),
            },
        }
    }
    reference_training.cube_reference_data_tree_identity(prereg, repo_root=tmp_path)
    (train / "data").write_bytes(b"TRAIN")
    with pytest.raises(RuntimeError, match="tree SHA256 drifted"):
        reference_training.cube_reference_data_tree_identity(prereg, repo_root=tmp_path)


def _fake_prereg(tmp_path: Path) -> dict:
    prereg_path = tmp_path / "prereg.yaml"
    freeze_path = tmp_path / "freeze.json"
    prereg_path.write_text("prereg\n", encoding="utf-8")
    freeze_path.write_text("{}\n", encoding="utf-8")
    return {
        "preregistration_id": CUBE_REFERENCE_TRAINING_ID,
        "_config_path": str(prereg_path),
        "_freeze_receipt_path": str(freeze_path),
        "data": {"manifest_sha256": "manifest"},
        "scoring": {
            "hidden_future_prediction": {
                "gates": {
                    "correct_future_rate_minimum": 0.75,
                    "correct_history_rate_minimum": 0.75,
                    "context_switch_rate_minimum": 0.9,
                    "worst_rule_correct_future_rate_minimum": 0.7,
                    "target_latent_separation_required": True,
                    "response_gain_minimum": 0.5,
                    "normalized_response_error_strict_maximum": 1.0,
                },
                "uncertainty": {
                    "lower_bound_minimum": {
                        "correct_future_rate": 0.7,
                        "correct_history_rate": 0.7,
                        "context_switch_rate": 0.85,
                    }
                },
            }
        },
        "training": {
            "reference_matrix": {
                "training_seeds": [17321, 17322, 17323],
                "models": {
                    "lewm": {"variant": "lewm_recipe"},
                    "pldm": {"variant": "pldm_recipe"},
                },
            }
        },
        "planned_artifacts": {"training_root": str(tmp_path / "training")},
    }


def _fake_training_cell(*, prereg: dict, family: str, seed: int) -> dict:
    root = Path(prereg["planned_artifacts"]["training_root"]) / f"{family}_seed{seed}"
    checkpoint = root / f"{family}_recipe_step4096.pt"
    return {
        "checkpoint": checkpoint,
        "report": root / "training_report.json",
        "checkpoint_sha256": "c" * 64,
        "checkpoint_size_bytes": 1,
        "model_state_sha256": "d" * 64,
    }


def _development_result(
    *, prereg: dict, family: str, seed: int, passed: bool = True
) -> dict:
    contract = reference_score._expected_contract_identity(
        prereg, prereg_path=Path(prereg["_config_path"])
    )
    metrics = {
        "correct_future_rate": 1.0,
        "correct_history_rate": 1.0,
        "context_switch_rate": 1.0,
        "worst_rule_correct_future_rate": 1.0,
        "other_minus_correct_mse_margin_mean": 1.0,
        "joint_icl_pair_success_rate": 1.0,
        "paired_bootstrap_95_lower_bound": {
            "correct_future_rate": 1.0,
            "correct_history_rate": 1.0,
            "context_switch_rate": 1.0,
        },
        "latent_response": {
            "target_latent_separation": {"passed": True},
            "response_gain": 1.0,
            "normalized_response_error": 0.0,
        },
    }
    cell = _fake_training_cell(prereg=prereg, family=family, seed=seed)
    return {
        "schema_version": 1,
        "benchmark": reference_score.CUBE_REFERENCE_DEVELOPMENT_BENCHMARK,
        "submission_kind": "single_checkpoint",
        "status": "completed",
        "contract": contract,
        "model": {
            "family": family,
            "training_recipe": f"{family}_recipe",
            "training_seed": seed,
            "state_sha256_before": cell["model_state_sha256"],
            "state_sha256_after": cell["model_state_sha256"],
            "training_checkpoint": {
                "path": str(cell["checkpoint"]),
                "sha256": cell["checkpoint_sha256"],
                "size_bytes": cell["checkpoint_size_bytes"],
                "model_state_sha256": cell["model_state_sha256"],
                "training_report": str(cell["report"]),
            },
        },
        "metrics": metrics,
        "gate": reference_score.cube_grasp_rule_prediction_gate(
            metrics, release=prereg
        ),
        "claim_scope": "Development_only_not_Public_or_release",
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
    }


def test_development_method_requires_all_three_fixed_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg = _fake_prereg(tmp_path)
    monkeypatch.setattr(
        reference_score,
        "load_cube_reference_training_prereg",
        lambda *args, **kwargs: prereg,
    )
    monkeypatch.setattr(
        reference_score,
        "validate_cube_reference_training_report",
        lambda _prereg, *, model_family, training_seed, **kwargs: _fake_training_cell(
            prereg=prereg, family=model_family, seed=training_seed
        ),
    )
    paths = []
    for seed in (17321, 17322, 17323):
        path = tmp_path / f"seed{seed}.json"
        path.write_text(
            json.dumps(_development_result(prereg=prereg, family="lewm", seed=seed)),
            encoding="utf-8",
        )
        paths.append(path)
    score = reference_score.score_cube_reference_development_results(
        result_paths=paths,
        model_family="lewm",
        prereg_config=Path(prereg["_config_path"]),
    )
    assert score["passed"] is True
    assert score["training_seeds"] == [17321, 17322, 17323]
    assert score["public_test_opened"] is False

    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    payload["model"]["training_seed"] = 17322
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        reference_score.score_cube_reference_development_results(
            result_paths=paths,
            model_family="lewm",
            prereg_config=Path(prereg["_config_path"]),
        )


def test_development_method_rejects_gate_only_fabrication(tmp_path: Path) -> None:
    prereg = _fake_prereg(tmp_path)
    method = {
        "schema_version": 1,
        "benchmark": reference_score.CUBE_REFERENCE_DEVELOPMENT_BENCHMARK,
        "submission_kind": "three_seed_method",
        "status": "completed",
        "model_family": "lewm",
        "training_recipe": "lewm_recipe",
        "training_seeds": [17321, 17322, 17323],
        "checkpoint_results": [
            {"gate": {"passed": True}} for _ in range(3)
        ],
        "aggregate": {},
        "passed": True,
        "public_test_opened": False,
    }
    with pytest.raises(RuntimeError, match="embedded result identity drifted"):
        reference_score.validate_cube_reference_development_method(
            method, prereg=prereg, model_family="lewm"
        )


def test_freezer_authorizes_exact_six_jobs_and_keeps_public_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg_path = tmp_path / "prereg.yaml"
    prereg_path.write_text("fixture\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "active_splits": ["train", "loader_validation"],
                "build_passed": True,
                "public_test_generated": False,
                "public_test_opened": False,
            }
        ),
        encoding="utf-8",
    )
    decision_path = tmp_path / "data_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "status": "passed_development",
                "scope": "data_readiness_only_not_reference_model_or_public",
                "protocol_id": "cube_gripper_carry_rule_history3_development_v4",
                "claims": {
                    "data_readiness_passed": True,
                    "positive_reference_model_claim_allowed": False,
                    "public_test_claim_allowed": False,
                    "release_claim_allowed": False,
                },
                "reference_model_phase": {
                    "trainer_invoked": False,
                    "optimizer_steps_run": 0,
                    "checkpoints_created": False,
                    "lewm_or_pldm_development_scoring_run": False,
                    "public_test_model_scoring_opened": False,
                },
                "public_test": {
                    "access_status": "closed_not_read_not_scored",
                    "generated": False,
                    "hashed": False,
                    "opened": False,
                    "read": False,
                    "scored": False,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "freeze.json"
    training_root = tmp_path / "train_out"
    score_root = tmp_path / "score_out"
    data_root = tmp_path / "data"
    data_root.mkdir()
    prereg = {
        "preregistration_id": CUBE_REFERENCE_TRAINING_ID,
        "protocol_id": CUBE_REFERENCE_TRAINING_PROTOCOL,
        "identity": {"fixture": {"path": str(prereg_path)}},
        "runtime": {"stable_worldmodel": {}},
        "data": {
            "protocol": "cube_gripper_carry_rule_history3_development_v4",
            "artifact_tree": {"root": str(data_root)},
            "artifacts": {
                "manifest": {"path": str(manifest_path)},
                "data_readiness_decision": {"path": str(decision_path)},
            },
        },
        "training": {
            "upstream": {"original_h5": {}, "original_lance": {}},
            "reference_matrix": {
                "training_seeds": [17321, 17322, 17323],
                "common": {"optimizer_steps": 4096},
                "initial_checkpoints": {"lewm": {}, "pldm": {}},
            },
        },
        "planned_artifacts": {
            "freeze_receipt": str(output),
            "training_root": str(training_root),
            "development_score_root": str(score_root),
        },
    }
    monkeypatch.setattr(
        freezer,
        "load_cube_reference_training_prereg",
        lambda *args, **kwargs: prereg,
    )
    monkeypatch.setattr(
        freezer,
        "_verify_declared_file",
        lambda entry, **kwargs: {
            "path": entry["path"],
            "sha256": "fixture",
            "size_bytes": Path(entry["path"]).stat().st_size,
        },
    )
    monkeypatch.setattr(freezer, "_verify_source_h5", lambda *args, **kwargs: {"sha256": "h5"})
    monkeypatch.setattr(freezer, "_verify_original_lance", lambda *args, **kwargs: {"files": {}})
    monkeypatch.setattr(
        freezer,
        "_verify_checkpoint",
        lambda *args, family, **kwargs: {"sha256": family},
    )
    monkeypatch.setattr(
        freezer,
        "_verify_stable_worldmodel_runtime",
        lambda *args, **kwargs: {"path": str(tmp_path), "commit": "fixture"},
    )
    monkeypatch.setattr(
        freezer,
        "cube_reference_data_tree_identity",
        lambda *args, **kwargs: {"tree": "fixture"},
    )
    monkeypatch.setattr(
        freezer,
        "cube_reference_infrastructure_recovery_identity",
        lambda *args, **kwargs: {"v1": "zero_step"},
    )
    monkeypatch.setattr(
        freezer,
        "resolve_cube_reference_training_input",
        lambda _prereg, name, **kwargs: {
            "original_h5": tmp_path / "source.h5",
            "original_lance": tmp_path / "source.lance",
        }[name],
    )
    monkeypatch.setattr(
        freezer,
        "resolve_cube_reference_initial_checkpoint",
        lambda _prereg, family, **kwargs: tmp_path / f"{family}.ckpt",
    )
    monkeypatch.setattr(
        freezer,
        "_verify_checkpoint_runtime_compatibility",
        lambda **kwargs: {"lewm": {}, "pldm": {}},
    )
    receipt = freezer.freeze(
        prereg_path=prereg_path,
        source_h5=tmp_path / "source.h5",
        original_lance=tmp_path / "source.lance",
        lewm_checkpoint=tmp_path / "lewm.ckpt",
        pldm_checkpoint=tmp_path / "pldm.ckpt",
        output=output,
    )
    assert len(receipt["authorization"]["jobs"]) == 6
    assert receipt["authorization"]["total_optimizer_steps_authorized"] == 24576
    assert receipt["authorization"]["public_model_scoring_authorized"] is False
    assert receipt["infrastructure_recovery"] == {"v1": "zero_step"}
    assert receipt["public_test"]["opened"] is False


def test_finalizer_requires_complete_family_and_keeps_cem_public_unauthorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg_path = tmp_path / "prereg.yaml"
    freeze_path = tmp_path / "freeze.json"
    prereg_path.write_text("fixture\n", encoding="utf-8")
    freeze_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "decision.json"
    training_root = tmp_path / "training"
    score_root = tmp_path / "score"
    training_root.mkdir()
    score_root.mkdir()
    training_matrix_path = training_root / "matrix_report.json"
    training_matrix_path.write_text("{}\n", encoding="utf-8")
    prereg = {
        "preregistration_id": CUBE_REFERENCE_TRAINING_ID,
        "_config_path": str(prereg_path),
        "_freeze_receipt_path": str(freeze_path),
        "training": {"reference_matrix": {"training_seeds": [17321, 17322, 17323]}},
        "planned_artifacts": {
            "development_decision": str(output),
            "development_score_root": str(score_root),
            "training_root": str(training_root),
        },
    }
    monkeypatch.setattr(
        finalizer,
        "load_cube_reference_training_prereg",
        lambda *args, **kwargs: prereg,
    )
    monkeypatch.setattr(
        finalizer,
        "validate_cube_reference_development_method",
        lambda method, **kwargs: method,
    )
    methods = {}
    for family, passed in (("lewm", True), ("pldm", False)):
        methods[family] = {
            "benchmark": reference_score.CUBE_REFERENCE_DEVELOPMENT_BENCHMARK,
            "submission_kind": "three_seed_method",
            "status": "completed",
            "model_family": family,
            "training_recipe": f"{family}_recipe",
            "training_seeds": [17321, 17322, 17323],
            "public_test_opened": False,
            "checkpoint_results": [
                {"gate": {"passed": passed}} for _ in range(3)
            ],
            "aggregate": {},
            "passed": passed,
        }
    matrix_path = score_root / "matrix_score.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "preregistration_id": CUBE_REFERENCE_TRAINING_ID,
                "training_root": str(training_root),
                "authorization_chain": {
                    "preregistration": {
                        "path": str(prereg_path),
                        "sha256": finalizer.file_sha256(prereg_path),
                    },
                    "freeze_receipt": {
                        "path": str(freeze_path),
                        "sha256": finalizer.file_sha256(freeze_path),
                    },
                    "training_matrix": {
                        "path": str(training_matrix_path),
                        "sha256": finalizer.file_sha256(training_matrix_path),
                    },
                },
                "public_test_opened": False,
                "methods": methods,
            }
        ),
        encoding="utf-8",
    )
    decision = finalizer.finalize(
        prereg_path=prereg_path,
        matrix_score_path=matrix_path,
        output=output,
    )
    assert decision["status"] == "passed_development"
    assert decision["passing_families"] == ["lewm"]
    assert decision["original_task_retention"]["authorized"] is False
    assert decision["public_test"]["opened"] is False


@pytest.mark.skipif(not PINNED_STABLE.is_dir(), reason="Pinned runtime is unavailable")
def test_pinned_stable_runtime_preflights_eager_optional_loss_constructor() -> None:
    environment = os.environ.copy()
    environment["CONTEXTWORLD_STABLE_WORLDMODEL_REPO"] = str(PINNED_STABLE)
    environment["MPLCONFIGDIR"] = "/tmp/contextworld-mpl-test"
    command = [
        sys.executable,
        "-c",
        (
            "import scripts.run_cube_grasp_rule_h3_train as c; "
            "c._install_cube_action_dimensions(); "
            "assert c.trainer.mixed.model_config('lewm')['action_encoder']['input_dim'] == 25; "
            "assert c.trainer.mixed.model_config('pldm')['action_encoder']['input_dim'] == 25; "
            "loss=c.trainer.mixed.ConditionalSIGReg(include_unpaired=False, complete_haar_population=False); "
            "assert type(loss).__name__ == 'PinnedConditionalSIGReg'; "
            "assert c._PINNED_LOSS_COMPATIBILITY['conditional_sigreg_false_only'] is True"
        ),
    ]
    subprocess.run(command, cwd=ROOT, env=environment, check=True, capture_output=True)
