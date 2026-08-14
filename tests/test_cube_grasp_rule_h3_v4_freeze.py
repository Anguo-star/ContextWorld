from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

import scripts.freeze_cube_grasp_rule_h3_v4_development as freezer


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "sha256": freezer.file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    repo.mkdir()
    artifacts.mkdir()
    monkeypatch.setattr(freezer, "ROOT", repo)

    identity: dict[str, dict[str, object]] = {}
    for name in freezer.REQUIRED_IDENTITY_KEYS:
        path = repo / "identity" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
        identity[name] = _identity(path)

    pilot = repo / "evidence" / "coupling.json"
    _write_json(
        pilot,
        {
            "scope": {
                "reference_model_training_or_scoring": False,
                "public_test_opened_read_hashed_or_scored": False,
            },
            "design": {"couplings_n": [0.30, 0.40]},
        },
    )
    diagnostic = repo / "evidence" / "diagnostic.json"
    _write_json(
        diagnostic,
        {
            "scope": {
                "reference_model_training_or_scoring": False,
                "public_test": {
                    "opened": False,
                    "read": False,
                    "hashed": False,
                    "scored": False,
                },
            }
        },
    )
    decision = repo / "evidence" / "decision.json"
    _write_json(
        decision,
        {
            "status": "failed_development",
            "public_test": {
                "opened": False,
                "read": False,
                "hashed": False,
                "scored": False,
            },
        },
    )
    feasibility = repo / "evidence" / "feasibility.json"
    narrative = repo / "evidence" / "feasibility.md"
    _write_json(feasibility, {"passed": True})
    narrative.write_text("frozen feasibility\n", encoding="utf-8")
    action_support = repo / "evidence" / "v4_action_support.json"
    _write_json(
        action_support,
        {
            "status": "passed",
            "passed": True,
            "scope": {
                "public_test_opened": False,
                "total_concrete_profiles": 2304,
                "formal_catalog_namespace": {
                    "catalog_index_offset": 1_000_000,
                    "offset_modulo_anchor_count": 0,
                    "preformal_catalog_indices_excluded": [0, 1],
                },
            },
        },
    )
    preformal = repo / "evidence" / "v4_preformal_content.json"
    _write_json(
        preformal,
        {
            "protocol_id": freezer.PROTOCOL,
            "status": freezer.RECEIPT_STATUS,
            "checks_passed": True,
            "reconstruction_contract": {
                "lance_opened_or_generated": False,
                "formal_build_attempted": False,
            },
            "excluded_source_episode_count": 17,
            "prior_content_exclusions": {
                field: {"count": 18}
                for field in (
                    "action_profile_ids",
                    "scene_template_content_hashes",
                    "pair_content_hashes",
                    "query_pixel_hashes",
                )
            },
            "public_test": {
                "opened": False,
                "read": False,
                "hashed": False,
                "scored": False,
            },
            "reference_model_training_or_scoring": False,
        },
    )
    evidence_paths = {
        "coupling_pilot": pilot,
        "exploratory_diagnostic": diagnostic,
        "v3_failed_development_decision": decision,
        "action_feasibility_report": feasibility,
        "action_feasibility_narrative": narrative,
        "v4_action_support_audit": action_support,
        "v4_preformal_content_receipt": preformal,
    }
    evidence = {name: _identity(path) for name, path in evidence_paths.items()}

    basis = artifacts / Path(freezer.BASIS_CANONICAL_PATH).relative_to("artifacts")
    _write_json(
        basis,
        {
            "protocol_id": freezer.PROTOCOL,
            "status": "frozen_before_first_v4_data_build",
            "checks_passed": True,
            "excluded_source_episode_count": freezer.BASIS_EPISODE_COUNT,
            "excluded_source_episode_ids_sha256": (
                freezer.BASIS_EPISODE_IDS_SHA256
            ),
        },
    )
    monkeypatch.setattr(freezer, "BASIS_RECEIPT_SHA256", freezer.file_sha256(basis))

    source = tmp_path / "source.h5"
    actions = np.asarray(
        [[0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 5.0]],
        dtype=np.float32,
    )
    with h5py.File(source, "w") as handle:
        handle.create_dataset("action", data=actions)
        handle.create_dataset("ep_len", data=np.asarray([2], dtype=np.int32))
    canonical = np.ascontiguousarray(actions, dtype="<f4")
    std = np.std(canonical.astype(np.float64), axis=0, ddof=0).astype("<f4")
    monkeypatch.setattr(freezer, "SOURCE_FILE_SHA256", freezer.file_sha256(source))
    monkeypatch.setattr(freezer, "SOURCE_SIZE_BYTES", source.stat().st_size)
    monkeypatch.setattr(freezer, "SOURCE_ROW_COUNT", 2)
    monkeypatch.setattr(freezer, "SOURCE_EPISODE_COUNT", 1)
    monkeypatch.setattr(freezer, "ACTION_SHAPE", (2, 5))
    monkeypatch.setattr(freezer, "ACTION_FINITE_ROW_COUNT", 2)
    monkeypatch.setattr(freezer, "ACTION_EXCLUDED_NONFINITE_ROW_COUNT", 0)
    monkeypatch.setattr(
        freezer,
        "ACTION_DATA_SHA256",
        hashlib.sha256(canonical.tobytes()).hexdigest(),
    )
    monkeypatch.setattr(
        freezer,
        "ACTION_FINITE_CONTENT_SHA256",
        hashlib.sha256(canonical.tobytes()).hexdigest(),
    )
    monkeypatch.setattr(
        freezer, "ACTION_STD_FLOAT32", tuple(float(value) for value in std)
    )
    monkeypatch.setattr(
        freezer,
        "ACTION_STD_FLOAT32_SHA256",
        hashlib.sha256(std.tobytes()).hexdigest(),
    )

    action_contract = {
        "name": "action",
        "shape": [2, 5],
        "dtype": "float32",
        "finite_row_count": 2,
        "excluded_nonfinite_row_count": 0,
        "row_major_little_endian_float32_sha256": freezer.ACTION_DATA_SHA256,
        "finite_rows_content_sha256": freezer.ACTION_FINITE_CONTENT_SHA256,
        "population_std_float32": list(freezer.ACTION_STD_FLOAT32),
        "population_std_float32_sha256": freezer.ACTION_STD_FLOAT32_SHA256,
    }
    prereg = repo / "configs/benchmark/cube_gripper_carry_h3_development_prereg_v4.yaml"
    prereg.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "status": freezer.PREREG_STATUS,
        "phase": "development_only",
        "protocol_id": freezer.PROTOCOL,
        "scientific_change": {
            "sole_change": "can_hold_vertical_force_coupling_n",
            "v3_baseline_vertical_force_coupling_n": 0.30,
            "v4_vertical_force_coupling_n": 0.40,
            "capability_semantics_unchanged": True,
            "history3_causal_sequence_unchanged": True,
            "action_profiles_and_constraints_unchanged_except_new_seeds": True,
        },
        "learnability_gates": {
            "rgb_history_probe": {
                "recipe_unchanged_from_v3": True,
                "thresholds_unchanged_from_v3": True,
                "recipe": json.loads(json.dumps(freezer.PROBE_RECIPE)),
                "thresholds": json.loads(json.dumps(freezer.PROBE_THRESHOLDS)),
            }
        },
        "prior_episode_exclusion": {
            "basis_receipt": {
                "path": freezer.BASIS_CANONICAL_PATH,
                "sha256": freezer.BASIS_RECEIPT_SHA256,
                "size_bytes": basis.stat().st_size,
            },
            "basis_episode_count": freezer.BASIS_EPISODE_COUNT,
            "basis_episode_ids_sha256": freezer.BASIS_EPISODE_IDS_SHA256,
        },
        "identity": identity,
        "frozen_evidence": evidence,
        "source_and_catalog": {
            "source_symbol": freezer.SOURCE_SYMBOL,
            "formal_source_must_be_supplied_explicitly": True,
            "frozen_source_identity": {
                "row_count": 2,
                "episode_count": 1,
                "size_bytes": source.stat().st_size,
                "sha256": freezer.SOURCE_FILE_SHA256,
                "action_dataset": action_contract,
            },
        },
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "validation_lance_access_allowed": False,
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "reference_model_phase": {"training_and_scoring_authorized": False},
    }
    prereg.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return {
        "repo": repo,
        "artifacts": artifacts,
        "basis": basis,
        "source": source,
        "prereg": prereg,
        "document": document,
        "output": tmp_path / "receipt.json",
    }


def _run(fixture: dict[str, object]) -> dict[str, object]:
    return freezer.freeze(
        prereg_path=fixture["prereg"],
        artifact_root=fixture["artifacts"],
        source_h5=fixture["source"],
        output=fixture["output"],
    )


def test_recursive_placeholder_rejection() -> None:
    assert freezer._contains_placeholder({"nested": [0, {"value": "TBD"}]})
    assert not freezer._contains_placeholder({"nested": [0, {"value": "fixed"}]})


def test_float32_vector_normalizes_equivalent_decimal_spellings() -> None:
    short = [0.28941983, 0.39371696, 0.6431365, 0.39280161, 0.25030735]
    expanded = [float(value) for value in np.asarray(short, dtype="<f4")]
    assert short != expanded
    assert freezer._canonical_float32_vector(
        short, length=5, label="test vector"
    ) == expanded
    assert freezer._canonical_float32_vector(
        expanded, length=5, label="test vector"
    ) == expanded


@pytest.mark.parametrize("invalid", ([0.0] * 4, [0.0, 0.0, 0.0, 0.0, True]))
def test_float32_vector_rejects_wrong_shape_or_non_numeric(invalid: list[object]) -> None:
    with pytest.raises(RuntimeError):
        freezer._canonical_float32_vector(invalid, length=5, label="test vector")


def test_cli_requires_source_artifact_root_and_output_and_rejects_public() -> None:
    with pytest.raises(SystemExit):
        freezer.parse_args([])
    with pytest.raises(ValueError, match="Public"):
        freezer.parse_args(
            [
                "--artifact-root",
                "artifacts",
                "--source-h5",
                "source.h5",
                "--output",
                "public/receipt.json",
            ]
        )


def test_freeze_emits_builder_compatible_receipt_and_is_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = _run(fixture)
    assert receipt["protocol_id"] == freezer.PROTOCOL
    assert receipt["status"] == "frozen_before_first_v4_build"
    assert receipt["authorized_splits"] == ["train", "loader_validation"]
    assert receipt["identity"]["v4_builder"]["sha256"] == (
        fixture["document"]["identity"]["v4_builder"]["sha256"]
    )
    assert receipt["identity"]["v4_physics"]["sha256"] == (
        fixture["document"]["identity"]["v4_physics"]["sha256"]
    )
    assert receipt["identity"]["v3_physics_dependency"]
    assert receipt["scientific_change"]["v4_vertical_force_coupling_n"] == 0.40
    assert receipt["rgb_history_probe"]["recipe_unchanged_from_v3"]
    assert receipt["rgb_history_probe"]["thresholds_unchanged_from_v3"]
    assert receipt["prior_episode_exclusion_basis"][
        "excluded_source_episode_count"
    ] == 2320
    assert receipt["source_h5"]["path_recorded"] is False
    assert receipt["public_test"]["read"] is False
    assert receipt["reference_model_training_or_scoring_authorized"] is False
    with pytest.raises(FileExistsError):
        _run(fixture)


@pytest.mark.parametrize(
    "mutation",
    (
        "status",
        "phase",
        "protocol",
        "public",
        "training",
        "coupling",
        "probe_recipe",
        "probe_threshold",
        "identity_hash",
        "missing_v3_dependency",
        "basis_hash",
        "basis_count",
        "basis_digest",
        "pilot_hash",
        "source_hash",
        "action_hash",
    ),
)
def test_freeze_rejects_contract_or_identity_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    document = fixture["document"]
    if mutation == "status":
        document["status"] = "draft"
    elif mutation == "phase":
        document["phase"] = "release"
    elif mutation == "protocol":
        document["protocol_id"] = "wrong"
    elif mutation == "public":
        document["public_test"]["read"] = True
    elif mutation == "training":
        document["reference_model_phase"]["training_and_scoring_authorized"] = True
    elif mutation == "coupling":
        document["scientific_change"]["v4_vertical_force_coupling_n"] = 0.41
    elif mutation == "probe_recipe":
        document["learnability_gates"]["rgb_history_probe"]["recipe"][
            "resize_shape"
        ] = [32, 32]
    elif mutation == "probe_threshold":
        document["learnability_gates"]["rgb_history_probe"]["thresholds"][
            "overall_accuracy_minimum"
        ] = 0.74
    elif mutation == "identity_hash":
        document["identity"]["v4_builder"]["sha256"] = "0" * 64
    elif mutation == "missing_v3_dependency":
        del document["identity"]["v3_physics_dependency"]
    elif mutation == "basis_hash":
        document["prior_episode_exclusion"]["basis_receipt"]["sha256"] = "0" * 64
    elif mutation == "basis_count":
        basis = json.loads(fixture["basis"].read_text())
        basis["excluded_source_episode_count"] = 2319
        _write_json(fixture["basis"], basis)
        monkeypatch.setattr(freezer, "BASIS_RECEIPT_SHA256", freezer.file_sha256(fixture["basis"]))
        document["prior_episode_exclusion"]["basis_receipt"]["sha256"] = freezer.BASIS_RECEIPT_SHA256
    elif mutation == "basis_digest":
        basis = json.loads(fixture["basis"].read_text())
        basis["excluded_source_episode_ids_sha256"] = "0" * 64
        _write_json(fixture["basis"], basis)
        monkeypatch.setattr(freezer, "BASIS_RECEIPT_SHA256", freezer.file_sha256(fixture["basis"]))
        document["prior_episode_exclusion"]["basis_receipt"]["sha256"] = freezer.BASIS_RECEIPT_SHA256
    elif mutation == "pilot_hash":
        document["frozen_evidence"]["coupling_pilot"]["sha256"] = "0" * 64
    elif mutation == "source_hash":
        document["source_and_catalog"]["frozen_source_identity"]["sha256"] = "0" * 64
    else:
        document["source_and_catalog"]["frozen_source_identity"]["action_dataset"][
            "row_major_little_endian_float32_sha256"
        ] = "0" * 64
    fixture["prereg"].write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises((RuntimeError, FileNotFoundError)):
        _run(fixture)
