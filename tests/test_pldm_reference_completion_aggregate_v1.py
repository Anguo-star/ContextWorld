from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import contextworld.benchmarks.pldm_reference_completion_aggregate as aggregate
from contextworld.benchmarks.public_score import make_public_scoreboard_from_spec
from contextworld.paths import resolve_contextworld_path


ROOT = Path(__file__).resolve().parents[1]


def _identity(path: Path, logical_path: str) -> dict[str, Any]:
    return {
        "path": logical_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _write_json(root: Path, logical_path: str, payload: dict[str, Any]) -> Path:
    path = root / logical_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _formal_spec() -> dict[str, Any]:
    return {
        "completion_config": "configs/speed.yml",
        "completion_id": "speed-completion",
        "component_id": "speed",
        "component_name": "TwoRoom 速度",
        "formal_result": {
            "public_icl_aggregate": "artifacts/speed/aggregate.json",
            "raw_public_results_root": "artifacts/speed/raw",
            "recovery_root": "artifacts/speed/recovery",
            "cem_binding": "artifacts/speed/cem_binding.json",
            "cem_not_authorized_stop": "artifacts/speed/stop.json",
            "action_planning_aggregate": "artifacts/speed/planning.json",
            "original_task_retention_aggregate": "artifacts/speed/retention.json",
            "primary_metric": {
                "id": "metric",
                "label": "Metric",
                "value_path": "metrics.value",
                "gate_path": "gate.passed",
                "seed_path": "model.training_seed",
            },
            "aggregate_metric_key": "metric",
            "method_name": "PLDM test method",
            "behavioral_claim_boundary": "configs/speed-boundary.yml",
        },
        "retention_metric": {"id": "retention", "label": "Retention"},
    }


def _public_icl(cem: dict[str, bool]) -> dict[str, Any]:
    return {
        "metric": {
            "id": "metric",
            "label": "Metric",
            "value_path": "metrics.value",
            "gate_path": "gate.passed",
            "seed_path": "model.training_seed",
        },
        "records": [
            {"training_seed": seed, "value": 1.0, "passed": True}
            for seed in (1, 2, 3)
        ],
        "aggregate": {"path": "artifacts/speed/aggregate.json", "sha256": "a" * 64, "size_bytes": 1},
        "cem": cem,
        "ability_passed": True,
        "development": {"synthetic": True},
    }


def _cem_material(kind: str) -> dict[str, Any]:
    return {
        "metric": {"id": kind, "label": kind},
        "records": [
            {"training_seed": seed, "value": 1.0, "passed": True}
            for seed in (1, 2, 3)
        ],
        "aggregate": {"path": f"artifacts/speed/{kind}.json", "sha256": "b" * 64, "size_bytes": 1},
        "decision": {"passed": True},
    }


def _patch_formal_state(monkeypatch: pytest.MonkeyPatch, cem: dict[str, bool]) -> None:
    monkeypatch.setattr(
        aggregate,
        "_load_completion",
        lambda specification, **_: (
            {},
            {"path": "configs/speed.yml", "sha256": "c" * 64, "size_bytes": 1},
            "configs/release.yml",
            [1, 2, 3],
        ),
    )
    monkeypatch.setattr(
        aggregate,
        "_read_yaml",
        lambda path, **_: {"release_id": "speed-release"},
    )
    monkeypatch.setattr(
        aggregate,
        "_identity",
        lambda path, **_: {"path": str(path), "sha256": "d" * 64, "size_bytes": 1},
    )
    monkeypatch.setattr(
        aggregate,
        "_public_icl_from_recovery",
        lambda *args, **kwargs: _public_icl(cem),
    )
    monkeypatch.setattr(
        aggregate,
        "_speed_behavioral_claim_boundary",
        lambda **kwargs: {"path": "configs/speed-boundary.yml", "sha256": "e" * 64, "size_bytes": 1},
    )
    monkeypatch.setattr(
        aggregate,
        "_cem_aggregate",
        lambda **kwargs: _cem_material(str(kwargs["expected_kind"])),
    )
    monkeypatch.setattr(
        aggregate,
        "_speed_positive_cem_binding",
        lambda **kwargs: (
            {"path": "artifacts/speed/cem_binding.json", "sha256": "f" * 64, "size_bytes": 1},
            {"cem_binding_id": "tworoom_speed_pldm_cem_binding_v1"},
        ),
    )


def test_positive_icl_gate_is_frozen_before_cem_and_later_cem_completes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pass authorizes CEM at ICL freeze time, not after it has run."""

    _patch_formal_state(monkeypatch, {"authorized": True, "executed": False})

    material = aggregate._formal_completion_material(
        "speed", _formal_spec(), repo_root=tmp_path
    )

    assert material["outcome"] == "passed_public_icl_cem_completed"
    assert material["formal_public_icl"]["cem"] == {
        "authorized": True,
        "executed": False,
    }
    assert material["cem_finalization"]["executed"] is True
    assert material["training_attribution"] == {
        "claim": False,
        "paired_training_controls_available": False,
        "reason": (
            "该参考完成没有预注册的配对训练对照；Public ICL 行仅报告行为结果，"
            "不把表现归因于合成训练因素。"
        ),
    }


def test_positive_icl_gate_rejects_claim_that_cem_already_executed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_formal_state(monkeypatch, {"authorized": True, "executed": True})

    with pytest.raises(ValueError, match="pre-CEM authorization"):
        aggregate._formal_completion_material("speed", _formal_spec(), repo_root=tmp_path)


@pytest.mark.parametrize(
    ("gates", "expected"),
    [((True, True, True), True), ((True, False, True), False)],
    ids=["cem-authorized", "cem-stop-required"],
)
def test_speed_public_icl_probe_uses_speed_raw_receipt_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gates: tuple[bool, bool, bool],
    expected: bool,
) -> None:
    """The branch probe accepts the Speed envelope used by final validation."""

    root = tmp_path / "ContextWorld"
    root.mkdir()
    for seed, gate in zip((1, 2, 3), gates, strict=True):
        _write_json(
            root,
            f"artifacts/speed/raw/seed_{seed}.json",
            {
                "schema_version": 1,
                "benchmark": "speed-release",
                "submission_kind": "single_model",
                "status": "passed",
                "full_protocol": True,
                "release_config": {
                    "path": "configs/benchmark/tworoom_speed_release_v1.yaml",
                    "sha256": "a" * 64,
                },
                "model": {
                    "training_seed": seed,
                    "checkpoint_sha256": "b" * 64,
                },
                "metrics": {"value": 0.9},
                "gate": {"passed": gate},
            },
        )
    monkeypatch.setattr(
        aggregate,
        "_load_completion",
        lambda specification, **_: ({}, {}, "configs/speed-release.yaml", [1, 2, 3]),
    )
    monkeypatch.setattr(
        aggregate,
        "_read_yaml",
        lambda path, **_: {"release_id": "speed-release"},
    )

    assert aggregate._probe_public_icl_branch(
        "speed", _formal_spec(), repo_root=root
    ) == (expected, None)


def test_cem_stop_accepts_one_of_three_failure_but_binds_exact_gate_count(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ContextWorld"
    root.mkdir()
    aggregate_path = _write_json(root, "artifacts/formal/aggregate.json", {"receipt": "icl"})
    aggregate_identity = _identity(aggregate_path, "artifacts/formal/aggregate.json")
    development = {
        "config": {"path": "configs/development.yml", "sha256": "c" * 64, "size_bytes": 1},
        "manifest": {"path": "artifacts/development/manifest.json", "sha256": "d" * 64, "size_bytes": 1},
        "receipts": [
            {
                "seed": seed,
                "receipt": {
                    "path": f"artifacts/development/seed_{seed}.json",
                    "sha256": "e" * 64,
                    "size_bytes": 1,
                },
            }
            for seed in (1, 2, 3)
        ],
    }
    stop_path = _write_json(
        root,
        "artifacts/formal/cem_stop.json",
        {
            "schema_version": 1,
            "completion_id": "speed-completion",
            "cem": {"authorized": False, "executed": False},
            "public_icl": {
                "passed": False,
                "passed_checkpoints": 1,
                "evaluated_checkpoints": 3,
            },
            "public_icl_aggregate": aggregate_identity,
            "development": development,
        },
    )

    missing_development = json.loads(stop_path.read_text(encoding="utf-8"))
    missing_development.pop("development")
    stop_path.write_text(json.dumps(missing_development) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Development chain"):
        aggregate._cem_not_authorized_stop(
            component="speed",
            path="artifacts/formal/cem_stop.json",
            completion_id="speed-completion",
            public_icl_identity=aggregate_identity,
            ability_passed=False,
            passed_checkpoints=1,
            gate_by_seed={1: True, 2: False, 3: False},
            expected_development=development,
            repo_root=root,
        )
    missing_development["development"] = development
    stop_path.write_text(json.dumps(missing_development) + "\n", encoding="utf-8")

    assert aggregate._cem_not_authorized_stop(
        component="speed",
        path="artifacts/formal/cem_stop.json",
        completion_id="speed-completion",
        public_icl_identity=aggregate_identity,
        ability_passed=False,
        passed_checkpoints=1,
        gate_by_seed={1: True, 2: False, 3: False},
        expected_development=development,
        repo_root=root,
    ) == _identity(stop_path, "artifacts/formal/cem_stop.json")

    payload = json.loads(stop_path.read_text(encoding="utf-8"))
    payload["public_icl"]["passed_checkpoints"] = 0
    stop_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact 1/3"):
        aggregate._cem_not_authorized_stop(
            component="speed",
            path="artifacts/formal/cem_stop.json",
            completion_id="speed-completion",
            public_icl_identity=aggregate_identity,
            ability_passed=False,
            passed_checkpoints=1,
            gate_by_seed={1: True, 2: False, 3: False},
            expected_development=development,
            repo_root=root,
        )


def test_new_outputs_are_kept_in_checkout_even_when_inputs_resolve_externally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "ContextWorld"
    root.mkdir()
    external_artifacts = tmp_path / "canonical-artifacts"
    monkeypatch.setenv("CONTEXTWORLD_ARTIFACT_ROOT", str(external_artifacts))
    logical_path = "artifacts/evaluation/addendum/public_scoreboard.json"

    assert resolve_contextworld_path(logical_path, repo_root=root) == (
        external_artifacts / "evaluation/addendum/public_scoreboard.json"
    )
    assert aggregate._new_output_path(logical_path, repo_root=root) == root / logical_path


def test_local_finalization_validation_never_falls_back_to_external_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "ContextWorld"
    root.mkdir()
    external_artifacts = tmp_path / "canonical-artifacts"
    logical_path = "artifacts/evaluation/addendum/public_scoreboard.json"
    _write_json(
        external_artifacts.parent,
        "canonical-artifacts/evaluation/addendum/public_scoreboard.json",
        {"external": True},
    )
    monkeypatch.setenv("CONTEXTWORLD_ARTIFACT_ROOT", str(external_artifacts))

    assert resolve_contextworld_path(logical_path, repo_root=root).is_file()
    with pytest.raises(FileNotFoundError, match="local finalization output"):
        aggregate._read_local_output_json(logical_path, repo_root=root)


def test_new_output_path_rejects_a_symlinked_checkout_component(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ContextWorld"
    root.mkdir()
    external = tmp_path / "external-artifacts"
    external.mkdir()
    (root / "artifacts").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="traverses a symlink"):
        aggregate._new_output_path(
            "artifacts/evaluation/addendum/public_scoreboard.json",
            repo_root=root,
        )


def _public_addendum_bundle() -> dict[str, Any]:
    return {
        "rows_added": [
            {
                "component_id": "speed",
                "evidence_scope": "behavioral",
                "ability_passed": True,
                "original_task_retention": {"result": "PASS"},
            },
            {
                "component_id": "action_strength",
                "evidence_scope": "behavioral",
                "ability_passed": False,
                "original_task_retention": {"result": "NOT_EVALUATED"},
            },
        ],
        "aggregate": {
            "completion_results": {
                "speed": {
                    "reader_result": {"evidence_scope": "behavioral"},
                    "training_attribution": {
                        "claim": False,
                        "paired_training_controls_available": False,
                    },
                },
                "action_strength": {},
            },
            "public_scoreboard_addendum": {
                "components_added": ["speed", "action_strength"],
                "development_only_components_not_added": [
                    "contact_friction",
                    "motion_damping",
                ],
                "formal_reference_rows_added": 2,
                "formal_reference_rows_after": 13,
            },
        },
    }


def test_public_addendum_summary_accepts_the_canonical_formal_boundary() -> None:
    assert aggregate._validated_public_addendum_summary(
        _public_addendum_bundle()
    ) == {
        "formal_reference_rows": 13,
        "formal_reference_rows_added": 2,
        "components_added": ["speed", "action_strength"],
        "development_only_components_not_added": [
            "contact_friction",
            "motion_damping",
        ],
        "speed_evidence_scope": "behavioral",
        "speed_training_attribution_claim": False,
        "action_strength_formal_row_included": True,
        "action_strength_ability_passed": False,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda bundle: bundle["rows_added"][0].update(
            {"evidence_scope": "training_attributed"}
        ),
        lambda bundle: bundle["aggregate"]["completion_results"]["speed"][
            "reader_result"
        ].update({"evidence_scope": "training_attributed"}),
        lambda bundle: bundle["aggregate"]["completion_results"]["speed"][
            "training_attribution"
        ].update({"claim": True}),
        lambda bundle: bundle["rows_added"][1].update({"ability_passed": True}),
        lambda bundle: bundle["rows_added"][1].update(
            {"component_id": "contact_friction"}
        ),
        lambda bundle: bundle["aggregate"]["public_scoreboard_addendum"].update(
            {"development_only_components_not_added": ["contact_friction"]}
        ),
    ],
)
def test_public_addendum_summary_rejects_claim_and_row_boundary_drift(
    mutate: Any,
) -> None:
    bundle = _public_addendum_bundle()
    mutate(bundle)

    with pytest.raises(ValueError):
        aggregate._validated_public_addendum_summary(bundle)


def test_incremental_scoreboard_keeps_spec_prefix_but_uses_canonical_full_sort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The addendum never reuses an obsolete positional-scoreboard contract."""

    base_rows = [
        {
            "component_id": f"base_{index:02d}",
            "component_name": f"Base {index}",
            "method_name": "Frozen base",
            "primary_metric": {"id": "base_metric", "label": "Base metric", "per_seed_values": [1.0]},
            "per_seed_gate_passes": [True],
            "ability_passed": True,
            "required_training_seeds": 1,
            "evidence_scope": "behavioral",
            "original_task_retention": {"result": "N/A", "reason": "not applicable"},
        }
        for index in range(11)
    ]
    base_specification = {
        "schema_version": 1,
        "result_kind": "contextworld_public_scoreboard_spec",
        "components": base_rows,
    }
    base_scoreboard = make_public_scoreboard_from_spec(base_specification)
    root = tmp_path / "ContextWorld"
    root.mkdir()
    preregistration = {
        "_config_path": str(root / "configs/benchmark/aggregate.yaml"),
        "outputs": {
            "aggregate_freeze": "configs/benchmark/contextworld_pldm_reference_completion_aggregate_results_freeze_v1.json"
        },
        "scoreboard_addendum": {
            "preregistration": "configs/benchmark/addendum.yaml",
            "decision": "configs/benchmark/addendum_decision.json",
            "decision_id": "addendum-decision",
            "output_namespace": "artifacts/evaluation/addendum",
            "specification": "artifacts/evaluation/addendum/public_scoreboard_spec.json",
            "scoreboard": "artifacts/evaluation/addendum/public_scoreboard.json",
        },
        "completion_inputs": {"speed": {}, "action_strength": {}, "contact_friction": {}, "motion_damping": {}},
    }
    identities = {
        "specification": {"path": "artifacts/base/spec.json", "sha256": "1" * 64, "size_bytes": 1},
        "scoreboard": {"path": "artifacts/base/scoreboard.json", "sha256": "2" * 64, "size_bytes": 1},
    }

    def formal_material(component: str) -> dict[str, Any]:
        metric = {"id": f"{component}_metric", "label": f"{component} metric"}
        retention = (
            None
            if component == "action_strength"
            else {
                "metric": {"id": "retention", "label": "Retention"},
                "records": [
                    {"training_seed": seed, "value": 1.0, "passed": True}
                    for seed in (1, 2, 3)
                ],
            }
        )
        return {
            "completion_id": f"{component}-completion",
            "reader_result": {
                "component_id": component,
                "component_name": component,
                "method_name": f"{component} PLDM",
                "evidence_scope": "behavioral",
                "primary_metric": metric,
            },
            "formal_public_icl": {
                "records": [
                    {"training_seed": seed, "value": 1.0, "passed": True}
                    for seed in (1, 2, 3)
                ]
            },
            "original_task_retention_cem": retention,
        }

    monkeypatch.setattr(
        aggregate,
        "audit_completion_aggregate_readiness",
        lambda **kwargs: {"ready": True},
    )
    monkeypatch.setattr(
        aggregate,
        "load_completion_aggregate_preregistration",
        lambda *args, **kwargs: preregistration,
    )
    monkeypatch.setattr(
        aggregate,
        "_validate_historical_base",
        lambda *args, **kwargs: (
            base_specification,
            base_scoreboard,
            identities["specification"],
            identities["scoreboard"],
        ),
    )
    monkeypatch.setattr(
        aggregate,
        "_validate_baseline_cem",
        lambda *args, **kwargs: ({"path": "configs/base.json", "sha256": "3" * 64, "size_bytes": 1}, {"speed": 1.0, "action_strength": 1.0}),
    )
    monkeypatch.setattr(
        aggregate,
        "_validate_addendum_preregistration",
        lambda *args, **kwargs: {"path": "configs/benchmark/addendum.yaml", "sha256": "4" * 64, "size_bytes": 1},
    )
    monkeypatch.setattr(
        aggregate,
        "_formal_completion_material",
        lambda component, *args, **kwargs: formal_material(component),
    )
    monkeypatch.setattr(
        aggregate,
        "_development_only_material",
        lambda component, *args, **kwargs: {"completion_id": f"{component}-completion", "finalized": True},
    )
    monkeypatch.setattr(
        aggregate,
        "_identity",
        lambda path, **kwargs: {"path": str(path), "sha256": "5" * 64, "size_bytes": 1},
    )

    bundle = aggregate.build_completion_aggregate_and_scoreboard(repo_root=root)

    assert bundle["scoreboard_specification"]["components"][:11] == base_rows
    assert bundle["scoreboard"] == make_public_scoreboard_from_spec(
        bundle["scoreboard_specification"]
    )
    base_by_key = {
        (row["component_id"], row["method_name"]): row
        for row in base_scoreboard["component_results"]
    }
    observed_by_key = {
        (row["component_id"], row["method_name"]): row
        for row in bundle["scoreboard"]["component_results"]
    }
    assert {key: observed_by_key[key] for key in base_by_key} == base_by_key
    assert len(bundle["scoreboard"]["component_results"]) == 13
    assert bundle["aggregate"]["completion_results"].keys() == {
        "speed",
        "action_strength",
        "contact_friction",
        "motion_damping",
    }


def test_action_strength_real_negative_chain_is_formal_and_cem_is_not_evaluated() -> None:
    """The frozen ActionStrength failure remains a valid formal reference row."""

    preregistration = aggregate.load_completion_aggregate_preregistration(repo_root=ROOT)
    material = aggregate._formal_completion_material(
        "action_strength",
        preregistration["completion_inputs"]["action_strength"],
        repo_root=ROOT,
    )

    assert material["outcome"] == "failed_public_icl_cem_not_authorized"
    assert material["formal_public_icl"]["ability_passed"] is False
    assert material["formal_public_icl"]["cem"] == {
        "authorized": False,
        "executed": False,
    }
    assert material["cem_finalization"]["executed"] is False
    assert material["formal_public_icl"]["post_freeze_maintenance"]["score_consistency_amendment"]["path"] == (
        aggregate.ACTION_STRENGTH_SCORE_CONSISTENCY_AMENDMENT
    )


def test_speed_claim_boundary_keeps_future_public_row_behavioral_only() -> None:
    preregistration = aggregate.load_completion_aggregate_preregistration(repo_root=ROOT)
    specification = preregistration["completion_inputs"]["speed"]
    _completion, completion_identity, release_path, _seeds = aggregate._load_completion(
        specification, repo_root=ROOT
    )
    assert release_path is not None
    release_identity = aggregate._identity(release_path, repo_root=ROOT)
    release = aggregate._read_yaml(release_path, repo_root=ROOT)

    boundary = aggregate._speed_behavioral_claim_boundary(
        path=specification["formal_result"]["behavioral_claim_boundary"],
        completion_id=specification["completion_id"],
        release_id=release["release_id"],
        completion_identity=completion_identity,
        release_identity=release_identity,
        repo_root=ROOT,
    )

    assert boundary["path"] == "configs/benchmark/tworoom_speed_pldm_behavioral_claim_boundary_v1.yaml"
