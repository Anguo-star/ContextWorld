from __future__ import annotations

import copy
from pathlib import Path

from contextworld.benchmarks import cube_original_task_retention as v1
from contextworld.benchmarks import cube_original_task_retention_v2 as v2


ROOT = Path(__file__).resolve().parents[1]


def test_cube_cem_v2_changes_only_render_backend_and_namespace() -> None:
    original = v1.load_cube_cem_retention_prereg(require_freeze=False)
    recovery = v2.load_cube_cem_retention_v2_prereg(require_freeze=False)

    assert recovery["scope"] == original["scope"]
    assert recovery["runtime"] == original["runtime"]
    assert recovery["data"] == original["data"]
    assert recovery["authorization"] == original["authorization"]
    assert recovery["prior_development"] == original["prior_development"]
    assert recovery["public_test"] == original["public_test"]

    evaluation = copy.deepcopy(recovery["evaluation"])
    assert evaluation.pop("mujoco_gl") == "osmesa"
    assert evaluation.pop("environment_preflight_required") is True
    assert evaluation.pop("environment_preflight_num_envs") == 1
    assert evaluation.pop("environment_preflight_world_evaluate_called") is False
    assert evaluation.pop("environment_preflight_cem_episodes_consumed") == 0
    evaluation["mujoco_gl"] = "egl"
    assert evaluation == original["evaluation"]
    assert all(
        "v2" in str(value) for value in recovery["planned_artifacts"].values()
    )
    assert recovery["recovery"]["query_catalog_changed"] is False
    assert recovery["recovery"]["cem_parameter_changed"] is False


def test_cube_cem_v1_failure_is_zero_episode_and_not_a_model_result() -> None:
    recovery = v2.load_cube_cem_retention_v2_prereg(require_freeze=False)
    failure = recovery["predecessor_observed"]["failure_receipt"]
    import json

    payload = json.loads(Path(failure["path"]).read_text(encoding="utf-8"))
    assert payload["execution"]["world_evaluate_calls"] == 0
    assert payload["execution"]["cem_episodes_completed"] == 0
    assert payload["execution"]["aggregate_reports_created"] == 0
    assert payload["scientific_interpretation"][
        "retention_pass_or_fail_observed"
    ] is False
    assert payload["public_test"] == v2.closed_public_contract()


def test_cube_cem_v2_source_identities_match() -> None:
    recovery = v2.load_cube_cem_retention_v2_prereg(require_freeze=False)
    for name, entry in recovery["identity"].items():
        path = v2.resolve_declared_path(entry["path"], repo_root=ROOT)
        assert path.is_file(), name
        assert not path.is_symlink(), name
        assert v2.file_sha256(path) == entry["sha256"], name
        assert path.stat().st_size == entry["size_bytes"], name


def test_cube_cem_v2_wrapper_forces_osmesa_before_importing_v1() -> None:
    source = (
        ROOT / "scripts/eval_cube_original_task_cem_frozen_v2.py"
    ).read_text(encoding="utf-8")
    assignment = source.index('os.environ["MUJOCO_GL"] = "osmesa"')
    import_v1 = source.index("SPEC.loader.exec_module(v1)")
    assert assignment < import_v1
    assert "world_evaluate_called" in source
    assert "cem_episodes_consumed" in source
    assert "num_envs=1" in source
    assert "[contextworld] MUJOCO_GL=osmesa" in source
