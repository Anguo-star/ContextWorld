from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Dataset:
    column_names = ("episode_idx", "step_idx")

    def __init__(self) -> None:
        self.episodes = np.asarray([0, 0, 1, 1], dtype=np.int64)
        self.steps = np.asarray([0, 25, 0, 25], dtype=np.int64)

    def get_col_data(self, name: str):
        return self.episodes if name == "episode_idx" else self.steps

    def get_row_data(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        return {
            "episode_idx": self.episodes[rows],
            "step_idx": self.steps[rows],
        }


def test_standard_runner_accepts_only_the_frozen_query_identity(tmp_path: Path) -> None:
    runner = _load(
        "test_standard_original_cem",
        "scripts/eval_standard_original_task_cem_frozen_v1.py",
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "42": {
                    "row_indices": [0, 2],
                    "episode_indices": [0, 1],
                    "start_steps": [0, 0],
                }
            }
        ),
        encoding="utf-8",
    )
    _, queries = runner._load_frozen_queries(
        catalog, dataset=_Dataset(), seeds=(42,), count=2
    )
    assert queries[42]["row_indices"].tolist() == [0, 2]

    payload = json.loads(catalog.read_text(encoding="utf-8"))
    payload["42"]["episode_indices"] = [1, 0]
    catalog.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity drifted"):
        runner._load_frozen_queries(
            catalog, dataset=_Dataset(), seeds=(42,), count=2
        )

    payload = {
        "42": {
            "row_indices": [1, 3],
            "episode_indices": [0, 1],
            "start_steps": [25, 25],
        }
    }
    catalog.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-eligible starts"):
        runner._load_frozen_queries(
            catalog, dataset=_Dataset(), seeds=(42,), count=2
        )


def test_standard_runner_has_no_query_sampling_entrypoint() -> None:
    runner = _load(
        "test_standard_original_cem_no_sampling",
        "scripts/eval_standard_original_task_cem_frozen_v1.py",
    )
    assert not hasattr(runner, "select_queries")
    assert set(runner.TASKS) == {"pusht", "reacher"}


class _PreflightDataset:
    column_names = ("episode_idx", "step_idx", "action", "proprio", "state")

    def __init__(self) -> None:
        self._columns = {
            "episode_idx": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "step_idx": np.asarray([0, 25, 0, 25], dtype=np.int64),
            "action": np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
            "proprio": np.asarray([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]]),
            "state": np.asarray([[8.0, 8.0], [9.0, 9.0], [10.0, 10.0], [11.0, 11.0]]),
        }

    def get_col_data(self, name: str):
        return self._columns[name]

    def get_row_data(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        return {
            name: values[rows]
            for name, values in self._columns.items()
            if name in {"episode_idx", "step_idx"}
        }


class _PreflightPushTEnv:
    def _set_state(self, state):
        del state

    def _set_goal_state(self, goal_state):
        del goal_state


class _PreflightWorld:
    instances: list["_PreflightWorld"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.evaluate_called = False
        self.envs = SimpleNamespace(
            envs=[SimpleNamespace(unwrapped=_PreflightPushTEnv())]
        )
        self.instances.append(self)

    def evaluate(self, *args, **kwargs):  # pragma: no cover - must never run.
        del args, kwargs
        self.evaluate_called = True
        raise AssertionError("CEM evaluation must not run in preflight")

    def close(self) -> None:
        self.closed = True


def test_standard_preflight_closes_inputs_processors_and_world_without_cem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(
        "test_standard_original_cem_preflight",
        "scripts/eval_standard_original_task_cem_frozen_v1.py",
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("output_model_name: lewm\n", encoding="utf-8")
    dataset_path = tmp_path / "original.h5"
    dataset_path.write_bytes(b"dataset")
    input_audit = tmp_path / "input_audit.json"
    input_audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "audit_id": "contextworld_original_baseline_cem_input_identity_audit_v1",
                "datasets": {
                    "pusht": {
                        "path": str(dataset_path.resolve()),
                        "sha256": "c" * 64,
                        "size_bytes": dataset_path.stat().st_size,
                        "content_hash_checked": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "42": {
                    "row_indices": [0, 2],
                    "episode_indices": [0, 1],
                    "start_steps": [0, 0],
                }
            }
        ),
        encoding="utf-8",
    )
    dataset = _PreflightDataset()
    runtime_root = tmp_path / "runtime"
    plan_config = runtime_root / "scripts/plan/config/pusht.yaml"
    plan_config.parent.mkdir(parents=True)
    plan_config.write_text("plan: frozen\n", encoding="utf-8")
    _PreflightWorld.instances.clear()
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setattr(
        runner.base,
        "install_runtime",
        lambda root, ref: {"root": str(root), "commit": ref, "clean": True},
    )
    monkeypatch.setattr(
        runner.base,
        "parse_models",
        lambda values: {"lewm": checkpoint},
    )
    monkeypatch.setattr(runner.base, "checkpoint_config_path", lambda path: config)
    monkeypatch.setattr(
        runner,
        "_model_identity",
        lambda path: {"state_dict_sha256": "a" * 64, "parameter_count": 1},
    )
    monkeypatch.setattr(runner, "_load_dataset", lambda path, cache_keys: dataset)
    monkeypatch.setattr(runner.base, "swm", SimpleNamespace(World=_PreflightWorld))

    args = SimpleNamespace(
        task="pusht",
        stable_worldmodel_root=runtime_root,
        expected_ref="b" * 40,
        expected_plan_config_sha256=runner.base.file_sha256(plan_config),
        expected_plan_config_size=plan_config.stat().st_size,
        model=[f"lewm={checkpoint}"],
        expected_checkpoint_sha256=runner.base.file_sha256(checkpoint),
        expected_checkpoint_size=checkpoint.stat().st_size,
        expected_config_sha256=runner.base.file_sha256(config),
        expected_config_size=config.stat().st_size,
        dataset=dataset_path,
        expected_dataset_sha256="c" * 64,
        expected_dataset_size=dataset_path.stat().st_size,
        input_identity_audit=input_audit,
        expected_input_identity_audit_sha256=runner.base.file_sha256(input_audit),
        expected_input_identity_audit_size=input_audit.stat().st_size,
        query_catalog=catalog,
        expected_catalog_sha256=runner.base.file_sha256(catalog),
        expected_catalog_size=catalog.stat().st_size,
        eval_seeds="42",
        num_eval=2,
    )
    result = runner.preflight(args)

    assert result["cem_episodes_consumed"] == 0
    assert result["inputs"]["query_count"] == 2
    assert result["inputs"]["processor_columns"] == [
        "action",
        "goal_proprio",
        "goal_state",
        "proprio",
        "state",
    ]
    assert result["environment_preflight"]["world_evaluate_called"] is False
    assert [row["method"] for row in result["environment_preflight"]["task_callables"]] == [
        "_set_state",
        "_set_goal_state",
    ]
    assert len(_PreflightWorld.instances) == 1
    assert _PreflightWorld.instances[0].kwargs["num_envs"] == 1
    assert _PreflightWorld.instances[0].closed is True


def test_standard_policy_state_audit_targets_the_policy_model() -> None:
    runner = _load(
        "test_standard_original_cem_policy_state",
        "scripts/eval_standard_original_task_cem_frozen_v1.py",
    )
    model = torch.nn.Linear(2, 1)
    policy = SimpleNamespace(
        solver=SimpleNamespace(model=SimpleNamespace(model=model))
    )
    before = runner._policy_model_identity(policy)
    with torch.no_grad():
        model.weight.add_(1.0)
    after = runner._policy_model_identity(policy)
    assert before["state_dict_sha256"] != after["state_dict_sha256"]


def test_standard_checkpoint_format_matches_the_file_type() -> None:
    runner = _load(
        "test_standard_original_cem_checkpoint_format",
        "scripts/eval_standard_original_task_cem_frozen_v1.py",
    )
    assert runner._checkpoint_format(Path("baseline.ckpt")) == (
        "legacy_lightning_ckpt"
    )
    assert runner._checkpoint_format(Path("trained_reference.pt")) == (
        "save_pretrained_pt"
    )


class _EvaluationWorld:
    instances: list["_EvaluationWorld"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.policy = None
        self.instances.append(self)

    def set_policy(self, policy) -> None:
        self.policy = policy

    def evaluate(self, **kwargs):
        del kwargs
        return {"episode_successes": np.asarray([True, False])}

    def close(self) -> None:
        self.closed = True


def test_standard_evaluation_reports_the_actual_policy_state_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load(
        "test_standard_original_cem_evaluation_audit",
        "scripts/eval_standard_original_task_cem_frozen_v1.py",
    )
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "config.yaml"
    config.write_text("output_model_name: lewm\n", encoding="utf-8")
    dataset_path = tmp_path / "original.h5"
    dataset_path.write_bytes(b"dataset")
    input_audit = tmp_path / "input_audit.json"
    input_audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "audit_id": "contextworld_original_baseline_cem_input_identity_audit_v1",
                "datasets": {
                    "pusht": {
                        "path": str(dataset_path.resolve()),
                        "sha256": "c" * 64,
                        "size_bytes": dataset_path.stat().st_size,
                        "content_hash_checked": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "42": {
                    "row_indices": [0, 2],
                    "episode_indices": [0, 1],
                    "start_steps": [0, 0],
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime"
    plan_config = runtime / "scripts/plan/config/pusht.yaml"
    plan_config.parent.mkdir(parents=True)
    plan_config.write_text("plan: frozen\n", encoding="utf-8")
    dataset = _PreflightDataset()
    models: list[torch.nn.Module] = []
    _EvaluationWorld.instances.clear()
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setattr(
        runner.base,
        "install_runtime",
        lambda root, ref: {"root": str(root), "commit": ref, "clean": True},
    )
    monkeypatch.setattr(
        runner.base,
        "parse_models",
        lambda values: {"lewm": checkpoint},
    )
    monkeypatch.setattr(runner.base, "checkpoint_config_path", lambda path: config)
    monkeypatch.setattr(runner, "_load_dataset", lambda path, cache_keys: dataset)
    monkeypatch.setattr(runner.base, "swm", SimpleNamespace(World=_EvaluationWorld))

    def build_policy(checkpoint_path, *, device, seed, processors):
        del checkpoint_path, device, seed, processors
        model = torch.nn.Linear(2, 1)
        models.append(model)
        return SimpleNamespace(
            solver=SimpleNamespace(model=SimpleNamespace(model=model))
        )

    monkeypatch.setattr(runner.base, "build_policy", build_policy)
    monkeypatch.setattr(
        runner,
        "_model_identity",
        lambda path: pytest.fail("evaluation must audit its actual policy model"),
    )
    args = SimpleNamespace(
        task="pusht",
        stable_worldmodel_root=runtime,
        expected_ref="b" * 40,
        expected_plan_config_sha256=runner.base.file_sha256(plan_config),
        expected_plan_config_size=plan_config.stat().st_size,
        model=[f"lewm={checkpoint}"],
        expected_checkpoint_sha256=runner.base.file_sha256(checkpoint),
        expected_checkpoint_size=checkpoint.stat().st_size,
        expected_config_sha256=runner.base.file_sha256(config),
        expected_config_size=config.stat().st_size,
        dataset=dataset_path,
        expected_dataset_sha256="c" * 64,
        expected_dataset_size=dataset_path.stat().st_size,
        input_identity_audit=input_audit,
        expected_input_identity_audit_sha256=runner.base.file_sha256(input_audit),
        expected_input_identity_audit_size=input_audit.stat().st_size,
        query_catalog=catalog,
        expected_catalog_sha256=runner.base.file_sha256(catalog),
        expected_catalog_size=catalog.stat().st_size,
        eval_seeds="42",
        num_eval=2,
        output=tmp_path / "result",
        device="cpu",
    )
    result = runner.evaluate(args)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))

    audit = report["model"]["frozen_state_audit"]
    assert audit["scope"] == "actual_policy_model_per_seed"
    assert audit["passed"] is True
    assert audit["seeds"][0]["before"] == audit["seeds"][0]["after"]
    assert report["model"]["seeds"][0]["frozen_state_audit"]["passed"] is True
    assert len(models) == 1
    assert _EvaluationWorld.instances[0].closed is True


def test_tworoom_wrapper_requires_all_execution_identities() -> None:
    runner = _load(
        "test_tworoom_original_cem",
        "scripts/eval_tworoom_original_baseline_cem_frozen_v1.py",
    )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "eval",
                "--stable-worldmodel-root",
                "/tmp/runtime",
                "--expected-ref",
                "a" * 40,
                "--checkpoint",
                "/tmp/model.ckpt",
                "--expected-checkpoint-sha256",
                "b" * 64,
                "--expected-checkpoint-size",
                "1",
            ]
        )


def test_cube_frozen_runner_accepts_legacy_checkpoints_and_catalogs() -> None:
    source = (ROOT / "scripts/eval_cube_original_task_cem_frozen.py").read_text(
        encoding="utf-8"
    )
    assert "legacy Lightning .ckpt" in source
    assert "--query-catalog" in source
    assert "verify_runtime" in source
    assert "--expected-ref" in source
