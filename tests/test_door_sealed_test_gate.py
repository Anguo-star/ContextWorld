from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from contextworld.evaluation.icl_model import file_sha256
from contextworld.evaluation.sealed_test_gate import (
    SealedTestGateError,
    require_sealed_test_gate,
    validate_sealed_test_gate,
    validation_result_manifest_sha256,
)
from scripts import analyze_tworoom_door_planning as planning_analyzer
from scripts import analyze_tworoom_door_visual_generalization as visual_analyzer
from scripts import build_tworoom_door_planning_catalogs as planning_builder
from scripts import build_tworoom_door_visual_catalogs as visual_builder
from scripts import eval_tworoom_door_fixed_candidates as fixed_evaluator
from scripts import eval_tworoom_door_planning as planning_evaluator
from scripts import eval_tworoom_door_true_future_latent as latent_evaluator
from scripts import run_tworoom_door_eval as eval_runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"
SCHEMA = ROOT / "configs/benchmark/tworoom_door_sealed_test_gate_v1.schema.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _valid_gate_fixture(
    tmp_path: Path, monkeypatch
) -> tuple[Path, dict, Path, dict, Path, dict]:
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("CONTEXTWORLD_ARTIFACT_ROOT", str(artifact_root))
    schema = tmp_path / "configs/gate.schema.json"
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_bytes(SCHEMA.read_bytes())
    manifest_path = tmp_path / "configs/gate.json"
    record_path = tmp_path / "configs/validation-freeze.json"
    implementation_file = tmp_path / "contextworld/impl.py"
    implementation_file.parent.mkdir(parents=True, exist_ok=True)
    implementation_file.write_text("VALUE = 1\n", encoding="utf-8")

    original_checkpoint = (
        artifact_root
        / "training/runs/checkpoints/h3_origheldout_s3072/weights_final_step_6420.pt"
    )
    original_checkpoint.parent.mkdir(parents=True)
    original_checkpoint.write_bytes(b"original")
    ability_protocol = tmp_path / "configs/ability.yaml"
    ability_protocol.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {
                        "model_id": "M_origheldout",
                        "training_groups": ["original"],
                        "checkpoint": str(original_checkpoint),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = {
        "benchmark": "tworoom_door_visual_generalization_v1",
        "training_protocol": {"optimizer_steps": 12840},
        "ability_retention": {"protocol": str(ability_protocol)},
        "models": {
            "original_reference": {"required_training_seeds": [3072]},
            "fixed_door_control": {
                "required_training_seeds": [3072, 4096, 5120],
                "training_groups": {"original": 0.5, "door_fixed49_v2": 0.5},
            },
            "multi_door_target": {
                "required_training_seeds": [3072, 4096, 5120],
                "training_groups": {"original": 0.5, "door_multi_v2": 0.5},
            },
        },
        "sealed_test_gate": {
            "manifest": str(manifest_path),
            "schema": str(schema),
            "validation_freeze_record": str(record_path),
            "required_implementation_files": ["contextworld/impl.py"],
            "validation_evidence": {
                role: {"expected_result_files": 1}
                for role in (
                    "prediction",
                    "fixed_candidate",
                    "closed_loop_planning",
                    "original_ability",
                )
            },
        },
    }
    config_path = tmp_path / "configs/door.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    config_hash = file_sha256(config_path)

    checkpoint_rows = [
        {
            "group": "original_reference",
            "slug": "h3_origheldout_s3072",
            "training_seed": 3072,
            "path": str(original_checkpoint),
            "sha256": file_sha256(original_checkpoint),
        }
    ]
    for group, synthetic in (
        ("fixed_door_control", "door_fixed49_v2"),
        ("multi_door_target", "door_multi_v2"),
    ):
        for seed in (3072, 4096, 5120):
            slug = f"h3_{synthetic}_s{seed}"
            checkpoint = (
                artifact_root
                / "training/runs/checkpoints"
                / slug
                / "weights_final_step_12840.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(slug.encode("utf-8"))
            checkpoint_rows.append(
                {
                    "group": group,
                    "slug": slug,
                    "training_seed": seed,
                    "path": str(checkpoint),
                    "sha256": file_sha256(checkpoint),
                }
            )
    checkpoint_hashes = {row["sha256"] for row in checkpoint_rows}

    evidence = {}
    for role in (
        "prediction",
        "fixed_candidate",
        "closed_loop_planning",
        "original_ability",
    ):
        result_path = tmp_path / "results" / f"{role}.json"
        _write_json(result_path, {"role": role})
        input_files = [{"path": str(result_path), "sha256": file_sha256(result_path)}]
        if role == "prediction":
            identity = {
                "models": {
                    row["slug"]: {
                        "model": {"checkpoint_sha256": row["sha256"]}
                    }
                    for row in checkpoint_rows
                },
                "model_matrix_audit": {"complete_formal_matrix": True},
                "decision": {
                    "visible_geometry_generalization_validation_gate_passed": True
                },
            }
        elif role in ("fixed_candidate", "closed_loop_planning"):
            identity = {
                "formal_analysis": True,
                "model_matrix_audit": {"complete_formal_matrix": True},
                "training_report_binding_audit": {
                    "bindings": {
                        row["slug"]: {"checkpoint_sha256": row["sha256"]}
                        for row in checkpoint_rows
                    }
                },
            }
        else:
            identity = {
                "formal_analysis": True,
                "matrix_audit": {"complete_formal_matrix": True},
                "training_report_bindings": {
                    row["slug"]: {"checkpoint_sha256": row["sha256"]}
                    for row in checkpoint_rows
                },
            }
        report = {
            "status": "passed",
            "config": {"sha256": config_hash},
            "input_files": input_files,
            **identity,
        }
        if role != "original_ability":
            report["evaluation_split"] = "validation"
        report_path = tmp_path / "reports" / f"{role}.json"
        _write_json(report_path, report)
        result_digest, _ = validation_result_manifest_sha256(
            input_files, repo_root=tmp_path
        )
        evidence[role] = {
            "analysis_report": {
                "path": str(report_path),
                "sha256": file_sha256(report_path),
            },
            "result_file_count": 1,
            "result_files_manifest_sha256": result_digest,
        }

    record = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "validation_frozen",
        "validation_config": {
            "path": str(config_path),
            "sha256": config_hash,
        },
        "implementation": {
            "commit": "a" * 40,
            "tree": "c" * 40,
            "files": [
                {
                    "path": "contextworld/impl.py",
                    "sha256": file_sha256(implementation_file),
                }
            ],
        },
        "checkpoints": checkpoint_rows,
        "validation_evidence": evidence,
        "preregistered_gate": {
            "name": "visible_geometry_generalization_validation_gate",
            "passed": True,
            "source_analysis_report_sha256": evidence["prediction"][
                "analysis_report"
            ]["sha256"],
        },
    }
    _write_json(record_path, record)
    manifest = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "unlocked_after_validation_freeze",
        "schema": str(schema),
        "validation_freeze_record": {
            "path": str(record_path),
            "sha256": file_sha256(record_path),
        },
        "freeze": {
            "commit": "b" * 40,
            "recorded_at_utc": "2026-07-22T00:00:00Z",
            "immutable": True,
        },
    }
    _write_json(manifest_path, manifest)
    return config_path, config, manifest_path, manifest, record_path, record


def test_validation_never_requires_or_reads_a_gate_manifest(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.json"
    result = require_sealed_test_gate(
        split="validation",
        config_path=tmp_path / "also-missing.yaml",
        config={"sealed_test_gate": {"manifest": str(missing)}},
        repo_root=tmp_path,
    )
    assert result == {"required": False, "passed": True, "split": "validation"}


def test_complete_frozen_gate_passes_and_binds_all_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, config, manifest_path, _, _, _ = _valid_gate_fixture(
        tmp_path, monkeypatch
    )
    result = validate_sealed_test_gate(
        config_path=config_path,
        config=config,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        verify_git=False,
    )
    assert result["passed"]
    assert result["checkpoint_count"] == 7
    assert set(result["evidence"]) == {
        "prediction",
        "fixed_candidate",
        "closed_loop_planning",
        "original_ability",
    }


def test_gate_hard_fails_when_a_result_or_checkpoint_hash_changes(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, config, manifest_path, _, _, record = _valid_gate_fixture(
        tmp_path, monkeypatch
    )
    result_path = Path(
        json.loads(
            Path(
                record["validation_evidence"]["prediction"]["analysis_report"][
                    "path"
                ]
            ).read_text(encoding="utf-8")
        )["input_files"][0]["path"]
    )
    result_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(SealedTestGateError, match="Validation result hash mismatch"):
        validate_sealed_test_gate(
            config_path=config_path,
            config=config,
            manifest_path=manifest_path,
            repo_root=tmp_path,
            verify_git=False,
        )

    # Restore the result, then alter one of the seven actual checkpoint files.
    _write_json(result_path, {"role": "prediction"})
    checkpoint = Path(record["checkpoints"][0]["path"])
    checkpoint.write_bytes(b"tampered-checkpoint")
    with pytest.raises(SealedTestGateError, match="Checkpoint hash mismatch"):
        validate_sealed_test_gate(
            config_path=config_path,
            config=config,
            manifest_path=manifest_path,
            repo_root=tmp_path,
            verify_git=False,
        )


def test_partial_or_unrecorded_gate_cannot_be_formal(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, config, manifest_path, manifest, record_path, record = _valid_gate_fixture(
        tmp_path, monkeypatch
    )
    record["preregistered_gate"]["passed"] = False
    _write_json(record_path, record)
    manifest["validation_freeze_record"]["sha256"] = file_sha256(record_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(SealedTestGateError, match="Validation freeze record failed"):
        validate_sealed_test_gate(
            config_path=config_path,
            config=config,
            manifest_path=manifest_path,
            repo_root=tmp_path,
            verify_git=False,
        )
    record["preregistered_gate"]["passed"] = True
    _write_json(record_path, record)
    manifest["validation_freeze_record"]["sha256"] = file_sha256(record_path)
    manifest["freeze"]["immutable"] = False
    _write_json(manifest_path, manifest)
    with pytest.raises(SealedTestGateError, match="gate manifest failed"):
        validate_sealed_test_gate(
            config_path=config_path,
            config=config,
            manifest_path=manifest_path,
            repo_root=tmp_path,
            verify_git=False,
        )


def test_draft_2020_12_schema_is_actually_enforced(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, config, manifest_path, manifest, _, _ = _valid_gate_fixture(
        tmp_path, monkeypatch
    )
    schema_path = Path(manifest["schema"])
    _write_json(
        schema_path,
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "required": ["schema_only_required_field"],
        },
    )
    with pytest.raises(SealedTestGateError, match="Draft 2020-12"):
        validate_sealed_test_gate(
            config_path=config_path,
            config=config,
            manifest_path=manifest_path,
            repo_root=tmp_path,
            verify_git=False,
        )


def test_two_commit_freeze_and_unlock_is_constructible_in_a_real_git_repo(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, config, manifest_path, unlocked_manifest, record_path, frozen_record = (
        _valid_gate_fixture(tmp_path, monkeypatch)
    )

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Door Gate Test")
    git("config", "user.email", "door-gate@example.invalid")

    _write_json(
        record_path,
        {
            "schema_version": 1,
            "benchmark": config["benchmark"],
            "status": "pending_validation",
            "reason": "Validation pending",
        },
    )
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "benchmark": config["benchmark"],
            "status": "locked_pending_validation",
            "reason": "Validation pending",
        },
    )
    git("add", ".")
    git("commit", "-q", "-m", "freeze implementation and pending gate")
    implementation_commit = git("rev-parse", "HEAD")
    implementation_tree = git("rev-parse", f"{implementation_commit}^{{tree}}")

    frozen_record["implementation"]["commit"] = implementation_commit
    frozen_record["implementation"]["tree"] = implementation_tree
    _write_json(record_path, frozen_record)
    git("add", str(record_path.relative_to(tmp_path)))
    git("commit", "-q", "-m", "freeze validation evidence")
    freeze_commit = git("rev-parse", "HEAD")

    unlocked_manifest["validation_freeze_record"]["sha256"] = file_sha256(record_path)
    unlocked_manifest["freeze"]["commit"] = freeze_commit
    _write_json(manifest_path, unlocked_manifest)
    git("add", str(manifest_path.relative_to(tmp_path)))
    git("commit", "-q", "-m", "unlock sealed test")

    result = validate_sealed_test_gate(
        config_path=config_path,
        config=config,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        verify_git=True,
    )
    assert result["passed"]
    assert result["freeze_commit"] == freeze_commit
    assert result["validation_freeze_record_sha256"] == file_sha256(record_path)

    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SealedTestGateError, match="not committed at HEAD"):
        validate_sealed_test_gate(
            config_path=config_path,
            config=config,
            manifest_path=manifest_path,
            repo_root=tmp_path,
            verify_git=True,
        )


def test_validation_catalog_copy_is_rejected_before_catalog_loader(
    tmp_path: Path, monkeypatch
) -> None:
    called = False

    def forbidden_loader(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("catalog loader must not be called")

    monkeypatch.setattr(fixed_evaluator, "load_door_planning_cell", forbidden_loader)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            fixed_evaluator.__name__,
            "--split",
            "validation",
            "--catalog",
            str(tmp_path / "copied-sealed-test-catalog.json"),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--normalizer",
            str(tmp_path / "missing-normalizer.json"),
            "--output",
            str(tmp_path / "unused.json"),
            "--track",
            "validation_seen",
            "--door-position",
            "49",
            "--seed",
            "42",
        ],
    )
    args = fixed_evaluator.parse_args()
    with pytest.raises(SealedTestGateError, match="Non-canonical"):
        fixed_evaluator.run(args)
    assert not called


def test_every_sealed_builder_evaluator_and_runner_rejects_locked_gate(
    tmp_path: Path, monkeypatch,
) -> None:
    def parsed(module, values):
        monkeypatch.setattr(sys, "argv", [module.__name__, *values])
        return module.parse_args()

    visual_args = parsed(
        visual_builder,
        ["--split", "sealed_test", "--output-root", str(tmp_path / "visual")],
    )
    planning_builder_args = parsed(
        planning_builder,
        ["--split", "sealed_test", "--output", str(tmp_path / "planning.json")],
    )
    latent_args = parsed(
        latent_evaluator,
        [
            "--split", "sealed_test", "--model-slug", "missing",
            "--group", "original_reference", "--training-seed", "3072",
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--output", str(tmp_path / "latent.json"),
        ],
    )
    fixed_args = parsed(
        fixed_evaluator,
        [
            "--split", "sealed_test", "--catalog", str(tmp_path / "missing.json"),
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--normalizer", str(tmp_path / "missing-normalizer.json"),
            "--output", str(tmp_path / "fixed.json"), "--track", "test_interpolation",
            "--door-position", "61", "--seed", "42",
        ],
    )
    planning_args = parsed(
        planning_evaluator,
        [
            "--split", "sealed_test", "--catalog", str(tmp_path / "missing.json"),
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--normalizer", str(tmp_path / "missing-normalizer.json"),
            "--output", str(tmp_path / "cem.json"), "--track", "test_interpolation",
            "--door-position", "61", "--seed", "42",
        ],
    )
    visual_analysis_args = parsed(
        visual_analyzer,
        ["--split", "sealed_test", "--artifact-root", str(tmp_path / "visual")],
    )
    planning_analysis_args = planning_analyzer.parse_args(
        [
            "--mode", "fixed", "--split", "sealed_test",
            "--artifact-root", str(tmp_path / "planning-analysis"),
        ]
    )
    calls = [
        lambda: visual_builder.run(visual_args),
        lambda: planning_builder.run(planning_builder_args),
        lambda: latent_evaluator.run(latent_args),
        lambda: fixed_evaluator.run(fixed_args),
        lambda: planning_evaluator.run(planning_args),
        lambda: visual_analyzer.run(visual_analysis_args),
        lambda: planning_analyzer.run(planning_analysis_args),
        lambda: eval_runner.run(
            eval_runner.parse_args(
                ["--split", "sealed_test", "--mode", "latent", "--dry-run"]
            )
        ),
    ]
    for call in calls:
        with pytest.raises(SealedTestGateError, match="remains locked"):
            call()


def test_checked_in_gate_is_locked_and_schema_declares_full_unlock_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / config["sealed_test_gate"]["manifest"]).read_text(encoding="utf-8")
    )
    schema = json.loads(
        (ROOT / config["sealed_test_gate"]["schema"]).read_text(encoding="utf-8")
    )
    record = json.loads(
        (ROOT / config["sealed_test_gate"]["validation_freeze_record"]).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "locked_pending_validation"
    assert record["status"] == "pending_validation"
    unlocked_required = schema["allOf"][0]["then"]["required"]
    assert {
        "validation_freeze_record",
        "freeze",
    } <= set(unlocked_required)
    record_required = schema["$defs"]["validation_freeze_record"]["required"]
    assert {
        "validation_config",
        "implementation",
        "checkpoints",
        "validation_evidence",
        "preregistered_gate",
    } <= set(record_required)
    assert "tree" in schema["$defs"]["implementation"]["required"]
