from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_pusht_pldm_reference_completion_v1.py"
BINDING_RUNNER_PATH = (
    ROOT / "scripts/freeze_pusht_pldm_evaluation_binding_v1.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location(
        "pldm_reference_completion_runner", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _binding_runner():
    spec = importlib.util.spec_from_file_location(
        "pldm_reference_completion_binding", BINDING_RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(name: str) -> Path:
    return ROOT / "configs/benchmark" / name


def test_completion_configs_are_pinned_to_current_data_and_original_pldm():
    expected = {
        "pusht_contact_friction_pldm_reference_completion_v1.yaml": (
            "contact_friction",
            [13313, 13314, 13315],
            "mixed_pldm_joint",
            "pusht_contact_friction_h3_release_v3",
        ),
        "pusht_motion_damping_pldm_reference_completion_v1.yaml": (
            "motion_damping",
            [14321, 14322, 14323],
            "mixed_pldm_identifiable_future_joint",
            "pusht_motion_damping_h3_release_v4",
        ),
    }
    for name, (component, seeds, recipe, data_version) in expected.items():
        payload = yaml.safe_load(_config(name).read_text(encoding="utf-8"))
        assert payload["status"] == "preregistered_development_only"
        assert payload["scope"]["component"] == component
        assert payload["scope"]["current_release_data_version"] == data_version
        assert payload["training"]["model_family"] == "PLDM"
        assert payload["training"]["recipe"] == recipe
        assert payload["training"]["seeds"] == seeds
        assert payload["training"]["pilot_seed"] == seeds[0]
        assert payload["initialization"]["checkpoint_id"] == "pusht_pldm_original"
        assert payload["initialization"]["strict_state_dict_load_required"] is True
        assert payload["stable_worldmodel"]["commit"] == (
            "875e607fc08aa72eacb94d5d178127804134cc06"
        )


def test_completion_runner_cannot_authorize_public_or_cem_in_overlay():
    runner = _runner()
    config_path = _config("pusht_contact_friction_pldm_reference_completion_v1.yaml")
    _, completion = runner.load_completion(config_path)
    _, source = runner.load_source_release(completion)
    original_status = source["training"]["reference_matrix"]["status"]
    overlay = runner.build_training_overlay(completion, source)

    assert original_status == "failed_development"
    assert source["training"]["reference_matrix"]["status"] == original_status
    assert overlay["training"]["reference_matrix"]["status"] == "planned_not_executed"
    assert overlay["training"]["reference_matrix"]["public_model_scoring_opened"] is False
    assert completion["source_release"]["data"]["public_test"] == runner.PUBLIC_CLOSED | {
        "lance_table_name_only": "validation.lance"
    }


def test_runner_has_development_only_commands_and_pinned_runtime_guard():
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"eval-development"' in source
    assert '"score-development"' in source
    assert "_pinned_stable_worldmodel" in source
    assert 'commands.add_parser("cem")' not in source
    assert 'commands.add_parser("public")' not in source


def test_evaluation_binding_is_additive_and_development_only():
    source = BINDING_RUNNER_PATH.read_text(encoding="utf-8")
    assert 'commands.add_parser("bind")' in source
    assert 'commands.add_parser("evaluate-development")' in source
    assert 'commands.add_parser("cem")' not in source
    assert 'commands.add_parser("public")' not in source
    assert '"eval-development"' in source
    assert '"score-development"' in source
    assert '"contextworld/benchmarks/adapters.py"' in source
    assert '"contextworld/benchmarks/contact_friction_icl_score.py"' in source
    assert '"contextworld/benchmarks/motion_damping_icl_score.py"' in source


def test_evaluation_binding_public_receipt_never_resolves_public_path():
    runner = _binding_runner()
    release = {
        "evaluation": {
            "lance_table": "validation.lance",
            "development": {"public_test": dict(runner.PUBLIC_CLOSED)},
        }
    }
    receipt = runner._public_receipt(release)
    assert receipt["table_name_only"] == "validation.lance"
    assert receipt["path_resolved"] is False
    assert receipt["path_statted"] is False
    assert receipt["path_walked"] is False
    assert receipt["path_hashed"] is False
    assert receipt["path_decoded"] is False
    assert receipt["accessed_by_binding"] is False
    assert receipt["accessed_by_evaluator"] is False
