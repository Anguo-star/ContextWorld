from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import contextworld.benchmarks as public_benchmarks
from contextworld.benchmarks import (
    portal_exit_icl_score,
    reacher_arm_mass_icl_score,
)
from contextworld.benchmarks.adapters import (
    ActionDelayICLModelAdapter,
    ActionStrengthICLModelAdapter,
    AdapterProtocol,
    ContactFrictionICLModelAdapter,
    CubeGraspRuleICLModelAdapter,
    DoorICLModelAdapter,
    LatentWorldModelAdapter,
    MotionDampingICLModelAdapter,
    PortalExitICLModelAdapter,
    ReacherArmMassICLModelAdapter,
    SpeedICLModelAdapter,
    validate_adapter_protocol,
)


class _ExternalAdapter(LatentWorldModelAdapter):
    """A framework-free adapter implemented by a downstream user."""

    def __init__(self, protocol: AdapterProtocol | None = None) -> None:
        self._protocol = protocol or AdapterProtocol(
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=2,
            future_action_blocks=1,
        )
        self.rollout_calls = 0

    @property
    def protocol(self) -> AdapterProtocol:
        return self._protocol

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "adapter_id": "external_generic",
            "checkpoint_sha256": "a" * 64,
        }

    def encode_pixels(
        self, pixels: np.ndarray, *, batch_size: int
    ) -> np.ndarray:
        del batch_size
        values = np.asarray(pixels, dtype=np.float32)
        return values.reshape(len(values), -1).mean(axis=1, keepdims=True)

    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        del raw_action_blocks, batch_size
        self.rollout_calls += 1
        values = np.asarray(input_pixels, dtype=np.float32)
        return values[:, -1].reshape(len(values), -1).mean(
            axis=1, keepdims=True
        )[:, None]

    def frozen_state_hash(self) -> str:
        return "external-generic-frozen"


def test_generic_adapter_is_public_and_legacy_names_remain_aliases() -> None:
    aliases = (
        SpeedICLModelAdapter,
        DoorICLModelAdapter,
        ActionDelayICLModelAdapter,
        ActionStrengthICLModelAdapter,
        ContactFrictionICLModelAdapter,
        MotionDampingICLModelAdapter,
        PortalExitICLModelAdapter,
        ReacherArmMassICLModelAdapter,
        CubeGraspRuleICLModelAdapter,
    )

    assert public_benchmarks.LatentWorldModelAdapter is LatentWorldModelAdapter
    assert "LatentWorldModelAdapter" in public_benchmarks.__all__
    assert all(alias is LatentWorldModelAdapter for alias in aliases)
    assert isinstance(_ExternalAdapter(), SpeedICLModelAdapter)


def test_generic_adapter_base_remains_abstract() -> None:
    with pytest.raises(TypeError):
        LatentWorldModelAdapter()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("history_tokens", 7),
        ("action_block_raw_steps", 4),
        ("action_dim", 5),
        ("future_action_blocks", 0),
    ),
)
def test_shared_validator_keeps_task_geometry_checks(
    field: str,
    value: int,
) -> None:
    adapter = _ExternalAdapter(
        replace(AdapterProtocol(3, 5, 2, 1), **{field: value})
    )

    with pytest.raises(ValueError, match=field):
        validate_adapter_protocol(
            adapter,
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=2,
            minimum_future_action_blocks=1,
            task_name="test task",
        )


def _task_arrays(*, first: str, second: str) -> SimpleNamespace:
    pixels = np.zeros((1, 4, 1, 1, 3), dtype=np.uint8)
    actions = np.zeros((1, 3, 5, 2), dtype=np.float32)
    return SimpleNamespace(
        pair_ids=("pair-0",),
        pair_count=1,
        raw_action_blocks=actions,
        **{first: pixels.copy(), second: pixels.copy()},
    )


def _release(tmp_path: Path) -> dict[str, object]:
    config = tmp_path / "release.yaml"
    config.write_text("fixture: true\n", encoding="utf-8")
    return {
        "_config_path": str(config),
        "release_id": "fixture-release",
        "data": {"manifest_sha256": "fixture-manifest"},
    }


def test_one_external_generic_adapter_runs_multiple_task_scorers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The same generic implementation can satisfy compatible H3 tasks."""

    adapter = _ExternalAdapter()
    release = _release(tmp_path)

    portal_arrays = _task_arrays(
        first="near_border_pixels",
        second="farther_from_border_pixels",
    )
    monkeypatch.setattr(
        portal_exit_icl_score,
        "load_portal_exit_icl_release",
        lambda *_args, **_kwargs: release,
    )
    monkeypatch.setattr(
        portal_exit_icl_score,
        "PortalExitICLEvalDataset",
        lambda **_kwargs: SimpleNamespace(
            arrays=portal_arrays,
            describe=lambda: {"fixture": "portal"},
        ),
    )
    monkeypatch.setattr(
        portal_exit_icl_score,
        "portal_exit_prediction_metrics",
        lambda **_kwargs: ({"fixture": 1.0}, []),
    )
    monkeypatch.setattr(
        portal_exit_icl_score,
        "portal_exit_prediction_gate",
        lambda *_args, **_kwargs: {"passed": True},
    )

    portal = portal_exit_icl_score.evaluate_portal_exit_icl_model(
        adapter=adapter,
        model_name="external",
        training_recipe="fixture",
        training_seed=1,
        repo_root=tmp_path,
    )

    reacher_arrays = _task_arrays(
        first="lighter_pixels",
        second="heavier_pixels",
    )
    monkeypatch.setattr(
        reacher_arm_mass_icl_score,
        "load_reacher_arm_mass_icl_release",
        lambda *_args, **_kwargs: release,
    )
    monkeypatch.setattr(
        reacher_arm_mass_icl_score,
        "ReacherArmMassICLEvalDataset",
        lambda **_kwargs: SimpleNamespace(
            arrays=reacher_arrays,
            describe=lambda: {"fixture": "reacher"},
        ),
    )
    monkeypatch.setattr(
        reacher_arm_mass_icl_score,
        "reacher_arm_mass_prediction_metrics",
        lambda **_kwargs: ({"fixture": 1.0}, []),
    )
    monkeypatch.setattr(
        reacher_arm_mass_icl_score,
        "reacher_arm_mass_prediction_gate",
        lambda *_args, **_kwargs: {"passed": True},
    )

    reacher = reacher_arm_mass_icl_score.evaluate_reacher_arm_mass_icl_model(
        adapter=adapter,
        model_name="external",
        training_recipe="fixture",
        training_seed=1,
        repo_root=tmp_path,
    )

    assert portal["benchmark"] == "tworoom_history3_portal_exit_icl_v1"
    assert reacher["benchmark"] == "reacher_history3_arm_mass_icl_v1"
    assert adapter.rollout_calls == 2


def test_dependency_extras_keep_core_small_and_do_not_pin_remote_sources() -> None:
    pyproject = (
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    ).read_text(encoding="utf-8")

    assert "[project.optional-dependencies]" in pyproject
    for group in ("eval", "stablewm", "dev"):
        assert f"{group} = [" in pyproject
    assert '"pylance>=4.0.0"' in pyproject
    assert '"torch>=' in pyproject
    assert '"pytest>=' in pyproject
    stablewm = pyproject.split("stablewm = [", 1)[1].split("]\n", 1)[0]
    dev = pyproject.split("dev = [", 1)[1].split("]\n", 1)[0]
    assert '"pylance>=4.0.0"' in stablewm
    assert '"stable-worldmodel[env,format]==0.1.1"' in stablewm
    for dependency in (
        '"torch>=',
        '"torchvision>=',
        '"lancedb>=',
        '"hydra-core>=',
    ):
        assert dependency in stablewm
    assert '"stable-worldmodel[env,format]==0.1.1"' in dev
    assert '"torch>=' in dev
    assert 'formal_checkout_policy = "component-release-config-pinned"' in pyproject
    assert '"lance>=' not in pyproject
    assert "git+" not in pyproject
    assert "http://" not in pyproject
    assert "https://" not in pyproject


def test_public_adapter_import_does_not_require_optional_eval_packages() -> None:
    """The documented adapter path must work with only the core dependencies."""

    source = """
import builtins

real_import = builtins.__import__
blocked = {
    'PIL', 'gymnasium', 'h5py', 'hydra', 'lance', 'lancedb', 'omegaconf',
    'pyarrow', 'pymunk', 'scipy', 'stable_worldmodel', 'torch', 'torchvision',
}

def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in blocked:
        raise ModuleNotFoundError(f'blocked optional dependency: {name}')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from contextworld.benchmarks.adapters import LatentWorldModelAdapter
from contextworld.benchmarks import AdapterProtocol
assert LatentWorldModelAdapter.__name__ == 'LatentWorldModelAdapter'
assert AdapterProtocol.__name__ == 'AdapterProtocol'
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_lazy_dataset_export_explains_the_eval_extra() -> None:
    source = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] == 'lance':
        raise ModuleNotFoundError("No module named 'lance'", name='lance')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import contextworld.benchmarks as benchmarks
try:
    benchmarks.MotionDampingICLEvalDataset
except ModuleNotFoundError as exc:
    assert 'contextworld[eval]' in str(exc)
else:
    raise AssertionError('optional dataset export unexpectedly imported')
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_engineering_identity_amendment_is_additive_and_nine_component_scoped() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (
            root
            / "configs/benchmark/"
            "contextworld_icl_suite_v2_engineering_identity_amendment_prereg_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    amendment = payload["engineering_identity_amendment"]
    authority = payload["membership_authority"]

    assert amendment["status"] == "preregistered_pending_integrity_reseal_v2"
    assert set(amendment["approved_component_identity_updates"]) == {
        "speed",
        "door",
        "action_delay",
        "action_strength",
        "contact_friction",
        "motion_damping",
        "robot_arm_mass",
        "portal_exit",
        "cube_gripper_carry",
    }
    assert "data_paths_data_hashes_or_artifact_trees" in amendment[
        "field_constraints"
    ]["prohibited"]
    assert authority["config_alone_grants_membership"] is False
    assert authority["public_test_rerun_authorized"] is False
