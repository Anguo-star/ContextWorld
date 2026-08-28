"""The public StableWM entry maps one contract to four method families."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_stablewm_eval as evaluator  # noqa: E402
import run_stablewm_family_entry as family_entry  # noqa: E402
import run_stablewm_plan as planner  # noqa: E402
import run_stablewm_train as launcher  # noqa: E402


@pytest.fixture
def stablewm_repo(tmp_path: Path) -> Path:
    config = tmp_path / "scripts/train/config"
    config.mkdir(parents=True)
    train = config.parent
    (train / "lewm.py").write_text("from x import build_training_logger\n",
                                   encoding="utf-8")
    (train / "viswm.py").write_text("from x import build_training_logger\n",
                                    encoding="utf-8")
    (train / "pldm.py").write_text("from x import build_training_logger\n",
                                   encoding="utf-8")
    (train / "prejepa.py").write_text("enabled = cfg.wandb.enabled\n", encoding="utf-8")
    common_logger = {
        "logger_backend": "none",
        "swanlab": {
            "enabled": False,
            "config": {}
        },
        "wandb": {
            "enabled": False,
            "config": {}
        },
    }
    (config / "lewm.yaml").write_text(
        yaml.safe_dump({
            **common_logger,
            "trainer": {
                "max_epochs": 100
            },
            "loss": {
                "regularizer": "sigreg",
                "sigreg": {
                    "weight": 0.09
                },
                "visreg": {
                    "weight": 0.09,
                    "kwargs": {}
                },
            },
        }),
        encoding="utf-8",
    )
    (config / "viswm.yaml").write_text(
        yaml.safe_dump({
            "defaults": ["lewm", "_self_"],
            "output_model_name": "viswm",
            "optimizer": {"lr": 1e-4},
            "loss": {
                "regularizer": "visreg",
                "visreg": {"weight": 4.5},
            },
        }),
        encoding="utf-8",
    )
    (config / "pldm.yaml").write_text(
        yaml.safe_dump({
            **common_logger, "trainer": {
                "max_epochs": 100
            }
        }),
        encoding="utf-8",
    )
    (config / "prejepa.yaml").write_text(
        yaml.safe_dump({"trainer": {
            "max_epochs": 10
        }}),
        encoding="utf-8",
    )
    data_config = config / "data"
    data_config.mkdir()
    for group in ("tworoom", "pusht"):
        (data_config / f"{group}.yaml").write_text(
            yaml.safe_dump({
                "dataset": {
                    "keys_to_load": ["pixels", "action", "proprio"]
                }
            }),
            encoding="utf-8",
        )
    plan = tmp_path / "scripts/plan"
    plan.mkdir(parents=True)
    (plan / "eval_wm.py").write_text("", encoding="utf-8")
    return tmp_path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pixels", shape=(2, 8, 8, 3), dtype="uint8")
        handle.create_dataset("action", shape=(2, 2), dtype="float32")
        handle.create_dataset("proprio", shape=(2, 2), dtype="float32")
    return path


def _args(
    stablewm_repo: Path,
    dataset: Path,
    *extra: str,
) -> launcher.argparse.Namespace:
    return launcher.parse_args([
        "--component",
        "speed",
        "--dataset",
        str(dataset),
        "--stablewm-repo",
        str(stablewm_repo),
        "--checkpoint-root",
        str(dataset.parent / "checkpoints-root"),
        *extra,
    ])


def _pairs(entries: list[str]) -> dict[str, str]:
    result = {}
    for item in entries:
        key, separator, value = item.partition("=")
        if separator:
            result[key] = value
    return result


def _enable_prejepa_common_logger(stablewm_repo: Path) -> None:
    train = stablewm_repo / "scripts/train"
    (train / "prejepa.py").write_text(
        "from stable_worldmodel.loggers import build_training_logger\n",
        encoding="utf-8",
    )
    (train / "config/prejepa.yaml").write_text(
        yaml.safe_dump(
            {
                "trainer": {"max_epochs": 10},
                "logger_backend": "none",
                "swanlab": {
                    "enabled": False,
                    "config": {
                        "project": None,
                        "workspace": None,
                        "experiment_name": None,
                        "id": None,
                        "logdir": None,
                        "mode": None,
                    },
                },
                "wandb": {"enabled": False, "config": {}},
            }
        ),
        encoding="utf-8",
    )


def _build(
    args: launcher.argparse.Namespace,
    stablewm_repo: Path,
) -> tuple[launcher.Target, dict[str, str], list[str]]:
    contract = launcher.load_profile_contract()
    target = launcher.resolve_target(args, contract)
    entries = launcher.build_overrides(
        args,
        contract,
        target,
        run_name="run",
        seed=3072,
        stablewm_repo=stablewm_repo,
    )
    return target, _pairs(entries), entries


class TestFamilyDialects:

    def test_planner_masks_broken_optional_flash_attention_before_swm_import(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entry = tmp_path / "eval_wm.py"
        entry.write_text("", encoding="utf-8")
        events: list[str] = []

        class FakePolicy:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        fake_swm = type(
            "FakeStableWorldModel",
            (),
            {"policy": type("FakePolicyModule", (), {"WorldModelPolicy": FakePolicy})},
        )()
        monkeypatch.setattr(sys, "argv", ["run_stablewm_plan.py"])
        monkeypatch.setitem(sys.modules, "stable_worldmodel", fake_swm)
        monkeypatch.setattr(
            planner,
            "_prepare_optional_flash_attention",
            lambda: events.append("flash_checked") or True,
        )
        monkeypatch.setattr(
            planner.runpy,
            "run_path",
            lambda *args, **kwargs: events.append("upstream_ran"),
        )

        assert planner.main([
            "--upstream-entry", str(entry),
            "--history-keys", "pixels,proprio",
        ]) == 0
        assert events == ["flash_checked", "upstream_ran"]

    def test_planner_can_disable_unrequested_upstream_videos(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entry = tmp_path / "eval_wm.py"
        entry.write_text("", encoding="utf-8")

        class FakePolicy:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        class FakeWorld:
            def evaluate(self, *args: object, **kwargs: object) -> object:
                return kwargs.get("video")

        fake_swm = type(
            "FakeStableWorldModel",
            (),
            {
                "World": FakeWorld,
                "policy": type(
                    "FakePolicyModule", (), {"WorldModelPolicy": FakePolicy}
                ),
            },
        )()
        observed: list[object] = []
        monkeypatch.setitem(sys.modules, "stable_worldmodel", fake_swm)
        monkeypatch.setattr(
            planner,
            "_prepare_optional_flash_attention",
            lambda: False,
        )
        monkeypatch.setattr(
            planner.runpy,
            "run_path",
            lambda *args, **kwargs: observed.append(
                fake_swm.World().evaluate(video=tmp_path / "video")
            ),
        )

        assert planner.main([
            "--upstream-entry", str(entry),
            "--history-keys", "pixels",
            "--disable-videos",
        ]) == 0
        assert observed == [None]

    def test_seed_list_accepts_one_or_multiple_runs(
            self, stablewm_repo: Path, dataset: Path) -> None:
        one = _args(stablewm_repo, dataset, "--seeds", "3072")
        three = _args(
            stablewm_repo,
            dataset,
            "--seeds",
            "3072,3073,3074",
        )

        assert one.seeds == (3072,)
        assert three.seeds == (3072, 3073, 3074)

    def test_frozen_release_reproduction_uses_the_same_public_entry(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        status = launcher.main([
            "--component",
            "door",
            "--family",
            "pldm",
            "--training-track",
            "historical_release",
            "--seeds",
            "3072",
            "--print-command",
        ])

        output = capsys.readouterr().out
        assert status == 0
        assert "mode=release-reproduction" in output
        assert "run_h3_hidden_passage_train.sh" in output
        assert "pldm-mixed" in output

    def test_component_selects_its_family_data_yaml_without_manual_mapping(
            self, stablewm_repo: Path, dataset: Path) -> None:
        args = _args(stablewm_repo, dataset, "--family", "lewm")

        target, pairs, entries = _build(args, stablewm_repo)

        assert target.data_group == "tworoom"
        assert "data=tworoom" in entries
        assert pairs["data.dataset.name"] == str(dataset)

    @pytest.mark.parametrize(
        "family,dataset_key,batch_key,embed_key",
        [
            ("lewm", "data.dataset.name", "loader.batch_size", "embed_dim"),
            ("viswm", "data.dataset.name", "loader.batch_size", "embed_dim"),
            ("pldm", "data.dataset.name", "loader.batch_size", "wm.embed_dim"),
            ("prejepa", "dataset_name", "batch_size", None),
        ],
    )
    def test_same_public_options_reach_the_real_yaml_keys(
        self,
        stablewm_repo: Path,
        dataset: Path,
        family: str,
        dataset_key: str,
        batch_key: str,
        embed_key: str | None,
    ) -> None:
        extra = [
            "--family",
            family,
            "--batch-size",
            "64",
            "--max-epochs",
            "20",
            "--accumulate",
            "2",
        ]
        if family != "prejepa":
            extra += ["--data-group", "pusht", "--embed-dim", "256"]
        args = _args(stablewm_repo, dataset, *extra)

        _, pairs, entries = _build(args, stablewm_repo)

        assert pairs[dataset_key] == str(dataset)
        assert pairs[batch_key] == "64"
        assert pairs["trainer.max_epochs"] == "20"
        assert pairs["++trainer.accumulate_grad_batches"] == "2"
        if embed_key:
            assert pairs[embed_key] == "256"
        assert not any("action_encoder.input_dim" in item for item in entries)

    def test_prejepa_logger_free_run_repairs_the_missing_wandb_flag(
            self, stablewm_repo: Path, dataset: Path) -> None:
        args = _args(stablewm_repo, dataset, "--family", "prejepa")

        _, pairs, _ = _build(args, stablewm_repo)

        assert pairs["++wandb.enabled"] == "false"

    def test_prejepa_action_delay_defaults_to_bf16_mixed(
        self, stablewm_repo: Path, dataset: Path
    ) -> None:
        args = launcher.parse_args(
            [
                "--component",
                "action_delay",
                "--dataset",
                str(dataset),
                "--family",
                "prejepa",
                "--stablewm-repo",
                str(stablewm_repo),
                "--checkpoint-root",
                str(dataset.parent / "checkpoints-root"),
            ]
        )

        _, pairs, _ = _build(args, stablewm_repo)

        assert pairs["trainer.precision"] == "bf16-mixed"

    def test_explicit_precision_overrides_action_delay_safety_default(
        self, stablewm_repo: Path, dataset: Path
    ) -> None:
        args = launcher.parse_args(
            [
                "--component",
                "action_delay",
                "--dataset",
                str(dataset),
                "--family",
                "prejepa",
                "--precision",
                "32-true",
                "--stablewm-repo",
                str(stablewm_repo),
                "--checkpoint-root",
                str(dataset.parent / "checkpoints-root"),
            ]
        )

        _, pairs, _ = _build(args, stablewm_repo)

        assert pairs["trainer.precision"] == "32-true"

    def test_prejepa_rejects_loader_knobs_it_does_not_read(self, stablewm_repo: Path,
                                                           dataset: Path) -> None:
        args = _args(
            stablewm_repo,
            dataset,
            "--family",
            "prejepa",
            "--prefetch-factor",
            "4",
        )

        with pytest.raises(SystemExit, match="does not expose"):
            _build(args, stablewm_repo)

    @pytest.mark.parametrize("family", ["lewm", "pldm", "prejepa"])
    def test_zero_workers_is_rejected_before_pytorch(
        self,
        stablewm_repo: Path,
        dataset: Path,
        family: str,
    ) -> None:
        extra = ["--family", family, "--num-workers", "0"]
        if family != "prejepa":
            extra += ["--data-group", "tworoom"]
        args = _args(stablewm_repo, dataset, *extra)

        with pytest.raises(SystemExit, match="must be positive"):
            _build(args, stablewm_repo)


class TestTargetAndStorageSafety:
    def test_rejects_stablepretraining_without_full_state_manager_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            launcher.importlib.metadata,
            "version",
            lambda _: "0.1.6",
        )

        with pytest.raises(SystemExit, match=r"requires stable-pretraining>=0\.1\.8"):
            launcher.validate_stablepretraining_version()

    def test_accepts_supported_stablepretraining_version(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            launcher.importlib.metadata,
            "version",
            lambda _: "0.1.8",
        )

        assert launcher.validate_stablepretraining_version() == "0.1.8"

    def test_resume_defaults_to_auto(
        self,
        stablewm_repo: Path,
        dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("CW_RESUME", raising=False)

        assert _args(stablewm_repo, dataset).resume == "auto"

    def test_resume_reset_is_available_from_the_environment(
        self,
        stablewm_repo: Path,
        dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CW_RESUME", "reset")

        assert _args(stablewm_repo, dataset).resume == "reset"

    def test_component_training_defaults_to_joint_scratch_not_historical(
        self,
        stablewm_repo: Path,
        dataset: Path,
    ) -> None:
        args = _args(stablewm_repo, dataset, "--family", "pldm")

        assert args.training_track == "joint_scratch_v1"
        assert launcher._uses_release_recipe(args) is False

    def test_post_eval_defaults_to_fifty_by_six(
        self,
        stablewm_repo: Path,
        dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("CW_EVAL_NUM", raising=False)
        monkeypatch.delenv("CW_EVAL_SEEDS", raising=False)

        training = _args(stablewm_repo, dataset)
        evaluation = evaluator.parse_args([
            "--family", "prejepa",
            "--original-env", "tworoom",
            "--dataset", str(dataset),
            "--checkpoint", str(dataset),
            "--stablewm-repo", str(stablewm_repo),
        ])

        assert training.eval_num == 50
        assert training.eval_seeds == "42,43,44,45,46,47"
        assert evaluation.num_eval == 50
        assert evaluation.eval_seeds == (42, 43, 44, 45, 46, 47)

    def test_h5_action_width_is_checked_before_training(
            self, stablewm_repo: Path, tmp_path: Path) -> None:
        path = tmp_path / "cube.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("pixels", shape=(2, 8, 8, 3), dtype="uint8")
            handle.create_dataset("action", shape=(2, 2), dtype="float32")
            handle.create_dataset("observation", shape=(2, 28), dtype="float32")
        args = launcher.parse_args([
            "--family", "prejepa",
            "--original-env", "cube",
            "--dataset", str(path),
            "--stablewm-repo", str(stablewm_repo),
            "--checkpoint-root", str(tmp_path / "ckpt"),
        ])
        target = launcher.resolve_target(args, launcher.load_profile_contract())

        with pytest.raises(SystemExit, match="raw action width 2"):
            launcher.validate_training_dataset_schema(
                target=target,
                family="prejepa",
                stablewm_repo=stablewm_repo,
            )

    def test_h5_auxiliary_width_is_checked_before_training(
            self, stablewm_repo: Path, tmp_path: Path) -> None:
        path = tmp_path / "cube.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("pixels", shape=(2, 8, 8, 3), dtype="uint8")
            handle.create_dataset("action", shape=(2, 5), dtype="float32")
            handle.create_dataset("observation", shape=(2, 2), dtype="float32")
        args = launcher.parse_args([
            "--family", "prejepa",
            "--original-env", "cube",
            "--dataset", str(path),
            "--stablewm-repo", str(stablewm_repo),
            "--checkpoint-root", str(tmp_path / "ckpt"),
        ])
        target = launcher.resolve_target(args, launcher.load_profile_contract())

        with pytest.raises(SystemExit, match="observation width 2"):
            launcher.validate_training_dataset_schema(
                target=target,
                family="prejepa",
                stablewm_repo=stablewm_repo,
            )

    @pytest.mark.parametrize(
        "component",
        [
            "speed",
            "door",
            "action_delay",
            "portal_exit",
            "action_strength",
            "contact_friction",
            "motion_damping",
            "robot_arm_mass",
            "cube_gripper_carry",
        ],
    )
    def test_prejepa_component_profile_is_pixels_and_action_only(
        self,
        stablewm_repo: Path,
        dataset: Path,
        component: str,
    ) -> None:
        args = launcher.parse_args([
            "--family", "prejepa",
            "--component", component,
            "--dataset", str(dataset),
            "--stablewm-repo", str(stablewm_repo),
            "--checkpoint-root", str(dataset.parent / "ckpt"),
        ])

        target = launcher.resolve_target(args, launcher.load_profile_contract())

        assert target.encoding_key is None
        assert target.encoding_dim is None
        assert launcher._family_model_columns(
            family="prejepa",
            target=target,
            stablewm_repo=stablewm_repo,
        ) == ("pixels", "action")

    @pytest.mark.parametrize("component", ["robot_arm_mass", "cube_gripper_carry"])
    def test_prejepa_component_never_remaps_observation(
        self,
        stablewm_repo: Path,
        dataset: Path,
        component: str,
    ) -> None:
        args = launcher.parse_args([
            "--family", "prejepa",
            "--component", component,
            "--dataset", str(dataset),
            "--stablewm-repo", str(stablewm_repo),
            "--checkpoint-root", str(dataset.parent / "ckpt"),
        ])
        target = launcher.resolve_target(args, launcher.load_profile_contract())
        entries = launcher.build_overrides(
            args,
            launcher.load_profile_contract(),
            target,
            run_name="run",
            seed=3072,
            stablewm_repo=stablewm_repo,
        )

        assert "~wm.encoding.proprio" in entries
        assert not any("wm.encoding.observation" in item for item in entries)

    def test_prejepa_component_schema_does_not_require_privileged_state(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "cube-component.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("pixels", shape=(2, 8, 8, 3), dtype="uint8")
            handle.create_dataset("action", shape=(2, 5), dtype="float32")
        args = launcher.parse_args([
            "--family", "prejepa",
            "--component", "cube_gripper_carry",
            "--dataset", str(path),
            "--stablewm-repo", str(stablewm_repo),
            "--checkpoint-root", str(tmp_path / "ckpt"),
        ])
        target = launcher.resolve_target(args, launcher.load_profile_contract())

        launcher.validate_training_dataset_schema(
            target=target,
            family="prejepa",
            stablewm_repo=stablewm_repo,
        )

    def test_prejepa_component_cannot_reintroduce_a_state_encoder(
        self,
        stablewm_repo: Path,
        dataset: Path,
    ) -> None:
        args = _args(
            stablewm_repo,
            dataset,
            "--family", "prejepa",
            "--override", "+wm.encoding.observation=10",
        )

        with pytest.raises(SystemExit, match="fixes model inputs"):
            _build(args, stablewm_repo)

    def test_component_post_eval_requires_a_completed_icl_result(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
    ) -> None:
        target = launcher.Target(
            label="speed",
            dataset=dataset,
            data_group="tworoom",
            history_size=3,
            action_dim=2,
            environment="tworoom",
        )
        checkpoint_root = tmp_path / "checkpoints-root"
        run_name = "speed_prejepa_s3072"
        checkpoint = (
            checkpoint_root / "checkpoints" / run_name / "weights_epoch_10.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        output = checkpoint.parent / "eval_results/benchmark_icl/speed/result.json"
        output.parent.mkdir(parents=True)
        output.write_text("{}\n", encoding="utf-8")
        manifest = checkpoint.parent / "eval_results/manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "steps": [
                        {
                            "id": "benchmark_icl/speed",
                            "status": "completed",
                            "output": str(output),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        args = argparse.Namespace(eval_result_subdir="")

        assert launcher._development_component_icl_failure(
            args=args,
            target=target,
            checkpoint_root=checkpoint_root,
            run_name=run_name,
            epoch=10,
        ) is None

        manifest.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "steps": [
                        {
                            "id": "benchmark_icl/speed",
                            "status": "not_compatible",
                            "reason": "state input",
                            "output": str(output),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert "did not complete" in launcher._development_component_icl_failure(
            args=args,
            target=target,
            checkpoint_root=checkpoint_root,
            run_name=run_name,
            epoch=10,
        )

    def test_component_post_eval_keeps_cem_when_dataset_is_available(
        self,
        stablewm_repo: Path,
        dataset: Path,
    ) -> None:
        args = _args(
            stablewm_repo,
            dataset,
            "--family", "prejepa",
            "--post-eval", "--eval-epoch", "10",
            "--benchmark-root", str(dataset.parent / "ContextWorld-v1"),
        )
        contract = launcher.load_profile_contract()
        target = launcher.resolve_target(args, contract)

        command = launcher._post_eval_command(
            args,
            target=target,
            run_name="speed_prejepa_s3072",
            checkpoint_root=dataset.parent / "checkpoints-root",
            stablewm_repo=stablewm_repo,
            profile=contract["families"]["prejepa"],
            frameskip=5,
            training_seed=3072,
            original_dataset=dataset,
        )

        assert "--component" in command
        assert "--icl-only" not in command
        assert command[command.index("--dataset") + 1] == str(dataset)
        assert command[command.index("--benchmark-root") + 1] == str(
            dataset.parent / "ContextWorld-v1"
        )
        assert command[command.index("--num-eval") + 1] == "50"
        assert command[command.index("--eval-seeds") + 1] == (
            "42,43,44,45,46,47"
        )

    def test_component_post_eval_resolves_matching_original_dataset(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
    ) -> None:
        original_root = tmp_path / "original-data"
        original_dataset = original_root / "quentinll/tworoom.h5"
        original_dataset.parent.mkdir(parents=True)
        original_dataset.write_bytes(dataset.read_bytes())
        args = _args(
            stablewm_repo,
            dataset,
            "--family", "prejepa",
            "--post-eval", "--eval-epoch", "10",
            "--dataset-root", str(original_root),
        )
        contract = launcher.load_profile_contract()
        target = launcher.resolve_target(args, contract)

        resolved = launcher._original_dataset_for_post_eval(
            args,
            contract,
            target,
        )

        assert resolved == original_dataset

    def test_component_post_eval_fails_closed_without_original_dataset(
        self,
        stablewm_repo: Path,
        dataset: Path,
    ) -> None:
        args = _args(
            stablewm_repo,
            dataset,
            "--family", "prejepa",
            "--post-eval", "--eval-epoch", "10",
        )
        contract = launcher.load_profile_contract()
        target = launcher.resolve_target(args, contract)

        with pytest.raises(SystemExit, match="Complete component post-eval"):
            launcher._original_dataset_for_post_eval(args, contract, target)

    def test_native_lance_contract_accepts_the_declared_action_width(
            self, tmp_path: Path) -> None:
        launcher._validate_lance_column_contract(
            path=tmp_path / "native.lance",
            columns={"episode_idx", "step_idx", "pixels", "action", "proprio"},
            required=("episode_idx", "step_idx", "pixels", "action", "proprio"),
            action_width=5,
            expected_action_dim=5,
            target_label="custom",
        )

    def test_cube_block_projection_is_rejected_with_the_required_adapter(
            self, tmp_path: Path) -> None:
        with pytest.raises(
            SystemExit, match="cube_block_projection_to_sequence_v1"
        ):
            launcher._validate_lance_column_contract(
                path=tmp_path / "cube.lance",
                columns={
                    "episode_idx",
                    "model_step_idx",
                    "pixels",
                    "action_block",
                    "physical_state",
                },
                required=(
                    "episode_idx",
                    "step_idx",
                    "pixels",
                    "action",
                    "proprio",
                ),
                action_width=None,
                expected_action_dim=5,
                target_label="cube_gripper_carry",
            )

    def test_action_width_is_derived_and_checked_not_rewritten(
            self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="does not pad, truncate, or hard-code"):
            launcher._validate_lance_column_contract(
                path=tmp_path / "wrong-width.lance",
                columns={"episode_idx", "step_idx", "pixels", "action", "proprio"},
                required=("episode_idx", "step_idx", "pixels", "action", "proprio"),
                action_width=2,
                expected_action_dim=5,
                target_label="custom-cube",
            )

    def test_public_loader_step_string_metadata_is_rejected_before_training(
            self, tmp_path: Path) -> None:
        with pytest.raises(
            SystemExit, match="stablewm_step_metadata_to_episode_table_v1"
        ):
            launcher._validate_lance_column_contract(
                path=tmp_path / "legacy-metadata.lance",
                columns={
                    "episode_idx",
                    "step_idx",
                    "pixels",
                    "action",
                    "proprio",
                    "pair_id",
                },
                required=(
                    "episode_idx",
                    "step_idx",
                    "pixels",
                    "action",
                    "proprio",
                ),
                action_width=2,
                expected_action_dim=2,
                target_label="action_strength",
                forbidden_string_columns=("pair_id",),
            )

    def test_original_mapping_resolves_one_root_to_the_exact_file(
            self, stablewm_repo: Path, tmp_path: Path) -> None:
        dataset = tmp_path / "quentinll/ogbench/cube_single_expert.h5"
        dataset.parent.mkdir(parents=True)
        dataset.write_bytes(b"")
        args = launcher.parse_args([
            "--family",
            "prejepa",
            "--original-env",
            "cube",
            "--dataset-root",
            str(tmp_path),
            "--stablewm-repo",
            str(stablewm_repo),
            "--checkpoint-root",
            str(tmp_path / "ckpt"),
        ])

        target, pairs, entries = _build(args, stablewm_repo)

        assert target.dataset == dataset
        assert pairs["dataset_name"] == str(dataset)
        assert "~wm.encoding.proprio" in entries
        assert "+wm.encoding.observation=10" in entries
        assert not any("action_dim" in item for item in entries)

    def test_collection_root_is_rejected_in_favour_of_an_exact_table(
            self, stablewm_repo: Path, tmp_path: Path) -> None:
        collection = tmp_path / "collection"
        (collection / "a.lance").mkdir(parents=True)
        args = _args(stablewm_repo, collection, "--family", "prejepa")

        with pytest.raises(SystemExit, match="exact H5 file or .lance table"):
            launcher.resolve_target(args, launcher.load_profile_contract())

    def test_resume_never_refuses_a_nonempty_run(self, tmp_path: Path) -> None:
        run = tmp_path / "checkpoints/run"
        run.mkdir(parents=True)
        (run / "config.yaml").write_text("x", encoding="utf-8")

        with pytest.raises(SystemExit, match="refuses existing state"):
            launcher.validate_resume(
                tmp_path,
                "run",
                "never",
                family="prejepa",
                identity_sha256="recipe",
            )

    def test_resume_reset_archives_exact_run_state_and_starts_fresh(
        self,
        tmp_path: Path,
    ) -> None:
        checkpoint_root = tmp_path / "checkpoint-root"
        output_root = tmp_path / "output-root"
        run_name = "speed_prejepa_joint_scratch_v1_s3072"
        checkpoint_dir = checkpoint_root / "checkpoints" / run_name
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "weights_epoch_4.pt").write_bytes(b"old weights")
        (checkpoint_dir / "eval_results").mkdir()
        hydra_dir = output_root / run_name
        hydra_dir.mkdir(parents=True)
        (hydra_dir / "train.log").write_text("old log", encoding="utf-8")

        matching_spt = []
        for suffix in ("uuid-rank-zero", "uuid-worker"):
            spt_run = checkpoint_root / "runs/20260828/120000" / suffix
            spt_run.mkdir(parents=True)
            (spt_run / launcher.SPT_RUN_MARKER_FILENAME).write_text(
                json.dumps({
                    "schema_version": launcher.SPT_RUN_MARKER_SCHEMA,
                    "run_name": run_name,
                    "training_identity_sha256": "old-recipe",
                }),
                encoding="utf-8",
            )
            matching_spt.append(spt_run)
        (matching_spt[0] / "checkpoints").mkdir()
        (matching_spt[0] / "checkpoints/last.ckpt").write_bytes(b"full state")

        other_spt = checkpoint_root / "runs/20260828/130000/other-uuid"
        other_spt.mkdir(parents=True)
        (other_spt / launcher.SPT_RUN_MARKER_FILENAME).write_text(
            json.dumps({
                "schema_version": launcher.SPT_RUN_MARKER_SCHEMA,
                "run_name": "another-run",
                "training_identity_sha256": "other-recipe",
            }),
            encoding="utf-8",
        )

        plan = launcher._plan_run_reset(
            checkpoint_root,
            run_name,
            output_root=output_root,
        )
        receipts = launcher._execute_run_reset(
            plan,
            identity_sha256="new-recipe",
        )

        assert len(plan.moves) == 4
        assert not checkpoint_dir.exists()
        assert not hydra_dir.exists()
        assert all(not path.exists() for path in matching_spt)
        assert other_spt.is_dir()
        assert len(receipts) == 2
        assert all(path.is_file() for path in receipts)
        receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
        assert receipt["schema_version"] == launcher.RESET_RECEIPT_SCHEMA
        assert receipt["run_name"] == run_name
        assert receipt["replacement_training_identity_sha256"] == "new-recipe"
        assert {entry["kind"] for entry in receipt["moves"]} == {
            "stablewm_checkpoint",
            "stablepretraining_run",
            "hydra_output",
        }
        archived = [Path(entry["archive"]) for entry in receipt["moves"]]
        assert all(path.exists() for path in archived)
        assert launcher.validate_resume(
            checkpoint_root,
            run_name,
            "never",
            family="prejepa",
            identity_sha256="new-recipe",
        ) is None

    def test_resume_reset_rejects_a_symlink_before_moving_other_state(
        self,
        tmp_path: Path,
    ) -> None:
        checkpoint_root = tmp_path / "checkpoint-root"
        run_name = "run"
        real = tmp_path / "real-checkpoint"
        real.mkdir()
        checkpoint_dir = checkpoint_root / "checkpoints" / run_name
        checkpoint_dir.parent.mkdir(parents=True)
        checkpoint_dir.symlink_to(real, target_is_directory=True)
        hydra_dir = tmp_path / "output" / run_name
        hydra_dir.mkdir(parents=True)
        (hydra_dir / "keep.txt").write_text("keep", encoding="utf-8")

        with pytest.raises(SystemExit, match="symlinked StableWM checkpoint"):
            launcher._plan_run_reset(
                checkpoint_root,
                run_name,
                output_root=hydra_dir.parent,
            )

        assert checkpoint_dir.is_symlink()
        assert (hydra_dir / "keep.txt").is_file()

    @pytest.mark.parametrize("family", ["lewm", "pldm", "prejepa"])
    def test_new_job_required_resume_needs_a_matching_full_state_checkpoint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        family: str,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        with pytest.raises(SystemExit, match="no full-state StablePretraining"):
            launcher.validate_resume(
                tmp_path,
                "run",
                "required",
                family=family,
                identity_sha256="recipe",
            )

    def test_resume_auto_never_restarts_an_incomplete_manual_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        run = tmp_path / "checkpoints/run"
        run.mkdir(parents=True)
        (run / "weights_epoch_5.pt").write_bytes(b"inference-only")

        with pytest.raises(SystemExit, match="refusing to restart"):
            launcher.validate_resume(
                tmp_path,
                "run",
                "auto",
                family="prejepa",
                identity_sha256="recipe",
            )

    def test_restart_count_without_job_identity_is_not_a_native_requeue(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")

        assert launcher._stablepretraining_native_requeue() is False

    def test_native_scheduler_requeue_is_left_to_stablepretraining(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = tmp_path / "checkpoints/run"
        run.mkdir(parents=True)
        (run / "weights_epoch_5.pt").write_bytes(b"inference-only")
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")

        launcher.validate_resume(
            tmp_path,
            "run",
            "auto",
            family="prejepa",
            identity_sha256="recipe",
        )
        launcher.validate_resume(
            tmp_path,
            "run",
            "required",
            family="prejepa",
            identity_sha256="recipe",
        )

    @pytest.mark.parametrize("policy", ["auto", "required"])
    def test_new_job_uses_matching_portable_full_state_checkpoint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        policy: str,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        spt_run = tmp_path / "runs/20260822/001122/uuid-one"
        checkpoint = spt_run / "checkpoints/last.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"full trainer state")
        (spt_run / launcher.SPT_RUN_MARKER_FILENAME).write_text(
            json.dumps({
                "schema_version": launcher.SPT_RUN_MARKER_SCHEMA,
                "run_name": "run",
                "training_identity_sha256": "recipe",
            }),
            encoding="utf-8",
        )
        stablewm_run = tmp_path / "checkpoints/run"
        stablewm_run.mkdir(parents=True)
        (stablewm_run / "config.yaml").write_text("incomplete", encoding="utf-8")

        resolved = launcher.validate_resume(
            tmp_path,
            "run",
            policy,
            family="prejepa",
            identity_sha256="recipe",
        )

        assert resolved == checkpoint.resolve()

    def test_portable_resume_ignores_another_recipe_checkpoint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        spt_run = tmp_path / "runs/20260822/001122/uuid-other"
        checkpoint = spt_run / "checkpoints/last.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"full trainer state")
        (spt_run / launcher.SPT_RUN_MARKER_FILENAME).write_text(
            json.dumps({
                "schema_version": launcher.SPT_RUN_MARKER_SCHEMA,
                "run_name": "run",
                "training_identity_sha256": "different-recipe",
            }),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit, match="no full-state StablePretraining"):
            launcher.validate_resume(
                tmp_path,
                "run",
                "required",
                family="pldm",
                identity_sha256="recipe",
            )

    def test_family_entry_delegates_full_state_resume_to_spt_manager(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pickle

        family_entry._prepare_optional_flash_attention()
        import stable_pretraining as spt

        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        spt_run = tmp_path / "runs/uuid"
        resume = tmp_path / "prior/last.ckpt"
        resume.parent.mkdir(parents=True)
        resume.write_bytes(b"full trainer state")

        class FakeManager:
            def __init__(
                self,
                *args: object,
                ckpt_path: object = None,
                weights_only: bool = True,
                **kwargs: object,
            ) -> None:
                self.args = args
                self.kwargs = {
                    **kwargs,
                    "ckpt_path": ckpt_path,
                    "weights_only": weights_only,
                }

            def _resolve_run_dir(self) -> Path:
                return spt_run

        monkeypatch.setattr(spt, "Manager", FakeManager)
        family_entry._install_manager_bridge(
            run_name="tworoom_prejepa_original_s3073",
            identity_sha256="recipe",
            resume_checkpoint=resume,
        )

        manager = spt.Manager(ckpt_path="upstream-weights.ckpt")
        resolved = manager._resolve_run_dir()

        assert resolved == spt_run
        assert manager.kwargs["ckpt_path"] == str(resume)
        assert manager.kwargs["weights_only"] is False
        assert pickle.loads(pickle.dumps(spt.Manager)) is spt.Manager
        marker = json.loads(
            (spt_run / family_entry.RUN_MARKER_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert marker == {
            "schema_version": family_entry.RUN_MARKER_SCHEMA,
            "run_name": "tworoom_prejepa_original_s3073",
            "training_identity_sha256": "recipe",
        }

    def test_native_requeue_refuses_an_unmarked_legacy_spt_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        family_entry._prepare_optional_flash_attention()
        import stable_pretraining as spt

        legacy_run = tmp_path / "runs/20260822/001122/legacy"
        legacy_run.mkdir(parents=True)

        class FakeManager:
            _early_preempt_fallback = False

            def __init__(
                self,
                *args: object,
                ckpt_path: object = None,
                weights_only: bool = True,
                **kwargs: object,
            ) -> None:
                pass

            def _resolve_run_dir(self) -> Path:
                return legacy_run

        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        monkeypatch.setattr(spt, "Manager", FakeManager)
        family_entry._install_manager_bridge(
            run_name="tworoom_prejepa_original_s3073",
            identity_sha256="recipe",
            resume_checkpoint=None,
        )

        with pytest.raises(RuntimeError, match="without its immutable"):
            spt.Manager()._resolve_run_dir()
        assert not (legacy_run / family_entry.RUN_MARKER_FILENAME).exists()

    def test_run_name_cannot_escape_the_checkpoint_root(self, stablewm_repo: Path,
                                                        dataset: Path) -> None:
        args = _args(
            stablewm_repo,
            dataset,
            "--family",
            "prejepa",
            "--run-name",
            "../shared",
        )
        target = launcher.resolve_target(args, launcher.load_profile_contract())

        with pytest.raises(SystemExit, match="Run names"):
            launcher._run_name(args, target, 3072, (3072, ))

    @pytest.mark.parametrize(
        "override",
        [
            "subdir=elsewhere",
            "data.dataset.name=/wrong.h5",
            "service.api_key=visible-secret",
            "optimizer.lr=${oc.env:LEARNING_RATE}",
            "hydra.searchpath=[file:///tmp/external-config]",
            "model@external_package=alternate",
        ],
    )
    def test_raw_overrides_cannot_bypass_path_or_secret_safety(
        self,
        stablewm_repo: Path,
        dataset: Path,
        override: str,
    ) -> None:
        args = _args(
            stablewm_repo,
            dataset,
            "--family",
            "lewm",
            "--data-group",
            "pusht",
            "--override",
            override,
        )

        with pytest.raises(SystemExit):
            _build(args, stablewm_repo)


class TestRecoveryPaths:

    def test_resume_reset_print_only_does_not_move_state(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        checkpoint_root = tmp_path / "persistent-checkpoints"
        run_name = "tworoom_prejepa_original_s3072"
        checkpoint_dir = checkpoint_root / "checkpoints" / run_name
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "weights_epoch_10.pt").write_bytes(b"old")
        spt_run = checkpoint_root / "runs/20260828/120000/uuid-old"
        spt_run.mkdir(parents=True)
        (spt_run / launcher.SPT_RUN_MARKER_FILENAME).write_text(
            json.dumps({
                "schema_version": launcher.SPT_RUN_MARKER_SCHEMA,
                "run_name": run_name,
                "training_identity_sha256": "old-recipe",
            }),
            encoding="utf-8",
        )

        status = launcher.main([
            "--family",
            "prejepa",
            "--original-env",
            "tworoom",
            "--dataset",
            str(dataset),
            "--stablewm-repo",
            str(stablewm_repo),
            "--checkpoint-root",
            str(checkpoint_root),
            "--output",
            str(checkpoint_root),
            "--seeds",
            "3072",
            "--resume",
            "reset",
            "--print-command",
        ])

        assert status == 0
        assert checkpoint_dir.is_dir()
        assert spt_run.is_dir()
        assert not (checkpoint_root / launcher.RESET_ARCHIVE_DIRNAME).exists()
        output = capsys.readouterr().out
        assert "reset planned: 2 existing state directories" in output
        assert "full_state_resume=fresh-after-reset" in output

    def test_resume_reset_archives_completed_epoch_and_runs_training(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        checkpoint_root = tmp_path / "persistent-checkpoints"
        run_name = "tworoom_prejepa_original_s3072"
        checkpoint_dir = checkpoint_root / "checkpoints" / run_name
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "weights_epoch_10.pt").write_bytes(b"completed old run")
        hydra_dir = checkpoint_root / run_name
        hydra_dir.mkdir(parents=True)
        (hydra_dir / "old.log").write_text("old", encoding="utf-8")
        calls: list[list[str]] = []

        class Completed:
            returncode = 0

        def run(command: list[str], **_: object) -> Completed:
            calls.append(command)
            return Completed()

        monkeypatch.setattr(launcher.subprocess, "run", run)

        status = launcher.main([
            "--family",
            "prejepa",
            "--original-env",
            "tworoom",
            "--dataset",
            str(dataset),
            "--stablewm-repo",
            str(stablewm_repo),
            "--checkpoint-root",
            str(checkpoint_root),
            "--output",
            str(checkpoint_root),
            "--benchmark-root",
            str(tmp_path / "ContextWorld-v1"),
            "--seeds",
            "3072",
            "--resume",
            "reset",
            "--post-eval",
            "--eval-epoch",
            "10",
        ])

        assert status == 0
        assert len(calls) == 2
        assert calls[0][1].endswith("scripts/train/prejepa.py")
        assert calls[1][1].endswith("scripts/run_stablewm_eval.py")
        assert (checkpoint_dir / launcher.TRAINING_IDENTITY_FILENAME).is_file()
        archived_weights = list(
            checkpoint_root.glob(
                f"{launcher.RESET_ARCHIVE_DIRNAME}/{run_name}/*/"
                "stablewm_checkpoint/weights_epoch_10.pt"
            )
        )
        archived_hydra = list(
            checkpoint_root.glob(
                f"{launcher.RESET_ARCHIVE_DIRNAME}/{run_name}/*/"
                "hydra_output/old.log"
            )
        )
        assert len(archived_weights) == 1
        assert len(archived_hydra) == 1

    def test_resume_reset_is_incompatible_with_eval_only(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
    ) -> None:
        checkpoint_root = tmp_path / "persistent-checkpoints"
        checkpoint = (
            checkpoint_root
            / "checkpoints/tworoom_prejepa_original_s3072/weights_epoch_10.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"keep")

        with pytest.raises(SystemExit, match="cannot be combined with CW_EVAL_ONLY"):
            launcher.main([
                "--family",
                "prejepa",
                "--original-env",
                "tworoom",
                "--dataset",
                str(dataset),
                "--stablewm-repo",
                str(stablewm_repo),
                "--checkpoint-root",
                str(checkpoint_root),
                "--seeds",
                "3072",
                "--resume",
                "reset",
                "--eval-only",
                "--eval-epoch",
                "10",
            ])

        assert checkpoint.is_file()
        assert not (checkpoint_root / launcher.RESET_ARCHIVE_DIRNAME).exists()

    def test_slurm_requeue_index_cannot_span_multiple_training_seeds(
        self,
        stablewm_repo: Path,
        dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)

        with pytest.raises(SystemExit, match="only one CW_SEEDS value"):
            launcher.main(
                [
                    "--family",
                    "prejepa",
                    "--original-env",
                    "tworoom",
                    "--dataset",
                    str(dataset),
                    "--stablewm-repo",
                    str(stablewm_repo),
                    "--checkpoint-root",
                    str(dataset.parent / "persistent-checkpoints"),
                    "--seeds",
                    "3072,3073",
                    "--print-command",
                ]
            )

    def test_non_slurm_launcher_keeps_serial_seed_sweeps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        launcher._validate_scheduler_seed_isolation(["seed-a", "seed-b"])

    def test_training_persists_stablepretraining_state_at_checkpoint_root(
        self,
        stablewm_repo: Path,
        dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        checkpoint_root = dataset.parent / "persistent-checkpoints"
        calls: list[tuple[list[str], dict[str, str]]] = []

        class Completed:
            returncode = 0

        def run(command: list[str], **kwargs: object) -> Completed:
            calls.append((command, kwargs["env"]))  # type: ignore[index]
            return Completed()

        monkeypatch.setattr(launcher.subprocess, "run", run)

        status = launcher.main([
            "--family",
            "prejepa",
            "--original-env",
            "tworoom",
            "--dataset",
            str(dataset),
            "--stablewm-repo",
            str(stablewm_repo),
            "--checkpoint-root",
            str(checkpoint_root),
            "--seeds",
            "3072,3073",
            "--resume",
            "auto",
        ])

        assert status == 0
        assert len(calls) == 2
        assert all(
            environment["STABLEWM_HOME"] == str(checkpoint_root)
            and environment["SPT_CACHE_DIR"] == str(checkpoint_root)
            and environment["CONTEXTWORLD_SPT_BRIDGE"] == "1"
            and "CONTEXTWORLD_STABLEWM_BUNDLE" not in environment
            and environment["PYTHONPATH"].split(":")[0]
            == str(launcher.STABLEWM_BOOTSTRAP_DIR)
            for _, environment in calls
        )
        assert calls[0][1]["CONTEXTWORLD_SPT_RUN_NAME"] == (
            "tworoom_prejepa_original_s3072"
        )
        assert calls[1][1]["CONTEXTWORLD_SPT_RUN_NAME"] == (
            "tworoom_prejepa_original_s3073"
        )
        assert "subdir=tworoom_prejepa_original_s3072" in calls[0][0]
        assert "subdir=tworoom_prejepa_original_s3073" in calls[1][0]
        for seed in (3072, 3073):
            identity_path = (
                checkpoint_root
                / "checkpoints"
                / f"tworoom_prejepa_original_s{seed}"
                / launcher.TRAINING_IDENTITY_FILENAME
            )
            payload = json.loads(identity_path.read_text(encoding="utf-8"))
            assert payload["schema_version"] == launcher.TRAINING_IDENTITY_SCHEMA
            assert payload["identity"]["seed"] == seed
            assert payload["identity"]["hydra_overrides"]
            dependencies = payload["identity"]["training_dependencies"]
            assert "stable-pretraining" in dependencies
            assert "version" in dependencies["stable-pretraining"]

    def test_multi_seed_post_eval_finishes_all_training_before_evaluation(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        calls: list[tuple[str, str]] = []

        class Completed:
            returncode = 0

        def run(command: list[str], **_: object) -> Completed:
            if command[1].endswith("run_stablewm_eval.py"):
                run_name = command[command.index("--run-name") + 1]
                calls.append(("eval", run_name))
            else:
                subdir = next(
                    value for value in command if value.startswith("subdir=")
                )
                calls.append(("train", subdir.removeprefix("subdir=")))
            return Completed()

        monkeypatch.setattr(launcher.subprocess, "run", run)

        status = launcher.main([
            "--family",
            "prejepa",
            "--original-env",
            "tworoom",
            "--dataset",
            str(dataset),
            "--stablewm-repo",
            str(stablewm_repo),
            "--benchmark-root",
            str(tmp_path / "ContextWorld-v1"),
            "--checkpoint-root",
            str(tmp_path / "persistent-checkpoints"),
            "--seeds",
            "3073,3074",
            "--resume",
            "auto",
            "--post-eval",
            "--eval-epoch",
            "10",
        ])

        assert status == 0
        assert calls == [
            ("train", "tworoom_prejepa_original_s3073"),
            ("train", "tworoom_prejepa_original_s3074"),
            ("eval", "tworoom_prejepa_original_s3073"),
            ("eval", "tworoom_prejepa_original_s3074"),
        ]

    def test_failed_training_seed_does_not_block_later_training_or_eval(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        calls: list[tuple[str, str]] = []

        class Completed:
            def __init__(self, returncode: int) -> None:
                self.returncode = returncode

        def run(command: list[str], **_: object) -> Completed:
            if command[1].endswith("run_stablewm_eval.py"):
                run_name = command[command.index("--run-name") + 1]
                calls.append(("eval", run_name))
                return Completed(0)
            run_name = next(
                value.removeprefix("subdir=")
                for value in command
                if value.startswith("subdir=")
            )
            calls.append(("train", run_name))
            return Completed(7 if run_name.endswith("s3072") else 0)

        monkeypatch.setattr(launcher.subprocess, "run", run)

        status = launcher.main([
            "--family", "prejepa",
            "--original-env", "tworoom",
            "--dataset", str(dataset),
            "--stablewm-repo", str(stablewm_repo),
            "--benchmark-root", str(tmp_path / "ContextWorld-v1"),
            "--checkpoint-root", str(tmp_path / "persistent-checkpoints"),
            "--seeds", "3072,3073",
            "--post-eval", "--eval-epoch", "10",
        ])

        assert status == 7
        assert calls == [
            ("train", "tworoom_prejepa_original_s3072"),
            ("train", "tworoom_prejepa_original_s3073"),
            ("eval", "tworoom_prejepa_original_s3073"),
        ]

    def test_failed_eval_seed_does_not_block_later_eval_seed(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        calls: list[tuple[str, str]] = []

        class Completed:
            def __init__(self, returncode: int) -> None:
                self.returncode = returncode

        def run(command: list[str], **_: object) -> Completed:
            if command[1].endswith("run_stablewm_eval.py"):
                run_name = command[command.index("--run-name") + 1]
                calls.append(("eval", run_name))
                return Completed(9 if run_name.endswith("s3072") else 8)
            run_name = next(
                value.removeprefix("subdir=")
                for value in command
                if value.startswith("subdir=")
            )
            calls.append(("train", run_name))
            return Completed(0)

        monkeypatch.setattr(launcher.subprocess, "run", run)

        status = launcher.main([
            "--family", "prejepa",
            "--original-env", "tworoom",
            "--dataset", str(dataset),
            "--stablewm-repo", str(stablewm_repo),
            "--benchmark-root", str(tmp_path / "ContextWorld-v1"),
            "--checkpoint-root", str(tmp_path / "persistent-checkpoints"),
            "--seeds", "3072,3073",
            "--post-eval", "--eval-epoch", "10",
        ])

        assert status == 9
        assert calls == [
            ("train", "tworoom_prejepa_original_s3072"),
            ("train", "tworoom_prejepa_original_s3073"),
            ("eval", "tworoom_prejepa_original_s3072"),
            ("eval", "tworoom_prejepa_original_s3073"),
        ]

    def test_new_job_forwards_identity_matched_last_checkpoint_to_every_rank(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
        checkpoint_root = tmp_path / "persistent-checkpoints"
        identity = {
            "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
            "identity_sha256": "recipe",
            "identity": {"seed": 3073},
        }
        spt_run = checkpoint_root / "runs/20260822/001122/uuid-prior"
        checkpoint = spt_run / "checkpoints/last.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"full trainer state")
        (spt_run / launcher.SPT_RUN_MARKER_FILENAME).write_text(
            json.dumps({
                "schema_version": launcher.SPT_RUN_MARKER_SCHEMA,
                "run_name": "tworoom_prejepa_original_s3073",
                "training_identity_sha256": "recipe",
            }),
            encoding="utf-8",
        )
        calls: list[tuple[list[str], dict[str, str]]] = []

        class Completed:
            returncode = 0

        def run(command: list[str], **kwargs: object) -> Completed:
            calls.append((command, kwargs["env"]))  # type: ignore[index]
            return Completed()

        monkeypatch.setattr(
            launcher,
            "_training_identity_document",
            lambda **_: identity,
        )
        monkeypatch.setattr(launcher.subprocess, "run", run)

        status = launcher.main([
            "--family",
            "prejepa",
            "--original-env",
            "tworoom",
            "--dataset",
            str(dataset),
            "--stablewm-repo",
            str(stablewm_repo),
            "--benchmark-root",
            str(tmp_path / "ContextWorld-v1"),
            "--checkpoint-root",
            str(checkpoint_root),
            "--seeds",
            "3073",
            "--resume",
            "auto",
        ])

        assert status == 0
        assert len(calls) == 1
        command, environment = calls[0]
        assert command[1].endswith("scripts/train/prejepa.py")
        assert environment["CONTEXTWORLD_SPT_RESUME_CHECKPOINT"] == str(
            checkpoint.resolve()
        )
        assert environment["CONTEXTWORLD_SPT_RUN_NAME"] == (
            "tworoom_prejepa_original_s3073"
        )

    def test_training_identity_never_overwrites_an_existing_recipe(
        self,
        tmp_path: Path,
    ) -> None:
        run_name = "one-run"
        run_dir = tmp_path / "checkpoints" / run_name
        run_dir.mkdir(parents=True)
        path = run_dir / launcher.TRAINING_IDENTITY_FILENAME
        original = {
            "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
            "identity_sha256": "first",
            "identity": {"recipe": 1},
        }
        path.write_text(json.dumps(original), encoding="utf-8")

        with pytest.raises(SystemExit, match="training identity differs"):
            launcher._install_training_identity(
                tmp_path,
                run_name,
                {
                    "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
                    "identity_sha256": "second",
                    "identity": {"recipe": 2},
                },
            )

        assert json.loads(path.read_text(encoding="utf-8")) == original

    def test_failed_preflight_empty_run_directory_is_safe_to_retry(
        self,
        tmp_path: Path,
    ) -> None:
        run_name = "tworoom_prejepa_original_s3073"
        run_dir = tmp_path / "checkpoints" / run_name
        run_dir.mkdir(parents=True)
        identity = {
            "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
            "identity_sha256": "recipe",
            "identity": {"seed": 3073},
        }

        launcher._install_training_identity(tmp_path, run_name, identity)

        assert json.loads(
            (run_dir / launcher.TRAINING_IDENTITY_FILENAME).read_text(
                encoding="utf-8"
            )
        ) == identity

    def test_auto_retry_accepts_an_exact_identity_only_preflight(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        run_name = "tworoom_prejepa_original_s3073"
        run_dir = tmp_path / "checkpoints" / run_name
        run_dir.mkdir(parents=True)
        hydra_run = tmp_path / "hydra" / run_name
        (run_dir / launcher.TRAINING_IDENTITY_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
                    "identity_sha256": "recipe",
                    "identity": {
                        "seed": 3073,
                        "hydra_overrides": [f"hydra.run.dir={hydra_run}"],
                    },
                }
            ),
            encoding="utf-8",
        )

        assert launcher.validate_resume(
            tmp_path,
            run_name,
            "auto",
            family="prejepa",
            identity_sha256="recipe",
        ) is None

    def test_auto_retry_accepts_a_stale_identity_when_training_never_started(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        run_name = "tworoom_prejepa_original_s3073"
        run_dir = tmp_path / "checkpoints" / run_name
        run_dir.mkdir(parents=True)
        hydra_run = tmp_path / "hydra" / run_name
        (run_dir / launcher.TRAINING_IDENTITY_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
                    "identity_sha256": "another-recipe",
                    "identity": {
                        "seed": 3073,
                        "hydra_overrides": [f"hydra.run.dir={hydra_run}"],
                    },
                }
            ),
            encoding="utf-8",
        )

        assert launcher.validate_resume(
            tmp_path,
            run_name,
            "auto",
            family="prejepa",
            identity_sha256="recipe",
        ) is None

    def test_auto_retry_rejects_a_stale_identity_after_spt_started(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        run_name = "tworoom_prejepa_original_s3073"
        run_dir = tmp_path / "checkpoints" / run_name
        run_dir.mkdir(parents=True)
        hydra_run = tmp_path / "hydra" / run_name
        (run_dir / launcher.TRAINING_IDENTITY_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
                    "identity_sha256": "another-recipe",
                    "identity": {
                        "seed": 3073,
                        "hydra_overrides": [f"hydra.run.dir={hydra_run}"],
                    },
                }
            ),
            encoding="utf-8",
        )
        spt_run = tmp_path / "runs/20260822/001122/uuid-started"
        spt_run.mkdir(parents=True)
        (spt_run / launcher.SPT_RUN_MARKER_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": launcher.SPT_RUN_MARKER_SCHEMA,
                    "run_name": run_name,
                    "training_identity_sha256": "another-recipe",
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit, match="refusing to restart"):
            launcher.validate_resume(
                tmp_path,
                run_name,
                "auto",
                family="prejepa",
                identity_sha256="recipe",
            )

    def test_auto_retry_rebinds_a_proven_zero_step_sanity_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        run_name = "speed_prejepa_s3072"
        old_identity = {
            "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
            "identity_sha256": "old-recipe",
            "identity": {"seed": 3072},
        }
        run_dir = tmp_path / "checkpoints" / run_name
        run_dir.mkdir(parents=True)
        (run_dir / launcher.TRAINING_IDENTITY_FILENAME).write_text(
            json.dumps(old_identity),
            encoding="utf-8",
        )
        (run_dir / "config.yaml").write_text("seed: 3072\n", encoding="utf-8")

        rank_zero = tmp_path / "runs/20260824/151050/rank-zero"
        (rank_zero / "checkpoints").mkdir(parents=True)
        marker = {
            "schema_version": launcher.SPT_RUN_MARKER_SCHEMA,
            "run_name": run_name,
            "training_identity_sha256": "old-recipe",
        }
        (rank_zero / launcher.SPT_RUN_MARKER_FILENAME).write_text(
            json.dumps(marker), encoding="utf-8"
        )
        (rank_zero / "run_meta.json").write_text("{}", encoding="utf-8")
        (rank_zero / "hparams.yaml").write_text("{}\n", encoding="utf-8")
        (rank_zero / "sidecar.json").write_text("{}", encoding="utf-8")
        (rank_zero / "summary.json").write_text(
            json.dumps({"step": 0, "epoch": 0, "metrics": {}}),
            encoding="utf-8",
        )

        worker = tmp_path / "runs/20260824/151208/worker-rank"
        worker.mkdir(parents=True)
        (worker / launcher.SPT_RUN_MARKER_FILENAME).write_text(
            json.dumps(marker), encoding="utf-8"
        )
        (worker / "run_meta.json").write_text("{}", encoding="utf-8")

        assert launcher.validate_resume(
            tmp_path,
            run_name,
            "auto",
            family="prejepa",
            identity_sha256="new-recipe",
        ) is None

        new_identity = {
            "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
            "identity_sha256": "new-recipe",
            "identity": {"seed": 3072},
        }
        launcher._install_training_identity(
            tmp_path,
            run_name,
            new_identity,
            replace_preflight_reservation=True,
        )
        assert json.loads(
            (run_dir / launcher.TRAINING_IDENTITY_FILENAME).read_text(
                encoding="utf-8"
            )
        ) == new_identity

    def test_auto_retry_rejects_a_run_that_reached_an_optimizer_step(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        run_name = "speed_prejepa_s3072"
        run_dir = tmp_path / "checkpoints" / run_name
        run_dir.mkdir(parents=True)
        (run_dir / launcher.TRAINING_IDENTITY_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
                    "identity_sha256": "old-recipe",
                    "identity": {"seed": 3072},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "config.yaml").write_text("seed: 3072\n", encoding="utf-8")
        spt_run = tmp_path / "runs/20260824/151050/rank-zero"
        spt_run.mkdir(parents=True)
        (spt_run / launcher.SPT_RUN_MARKER_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": launcher.SPT_RUN_MARKER_SCHEMA,
                    "run_name": run_name,
                    "training_identity_sha256": "old-recipe",
                }
            ),
            encoding="utf-8",
        )
        (spt_run / "run_meta.json").write_text("{}", encoding="utf-8")
        (spt_run / "summary.json").write_text(
            json.dumps({"step": 1, "epoch": 0, "metrics": {"loss": 1.0}}),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit, match="refusing to restart"):
            launcher.validate_resume(
                tmp_path,
                run_name,
                "auto",
                family="prejepa",
                identity_sha256="new-recipe",
            )

    def test_training_identity_rebind_is_opt_in_and_preflight_only(
        self,
        tmp_path: Path,
    ) -> None:
        run_name = "tworoom_prejepa_original_s3073"
        run_dir = tmp_path / "checkpoints" / run_name
        run_dir.mkdir(parents=True)
        hydra_run = tmp_path / "hydra" / run_name
        path = run_dir / launcher.TRAINING_IDENTITY_FILENAME
        path.write_text(
            json.dumps(
                {
                    "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
                    "identity_sha256": "old-recipe",
                    "identity": {
                        "hydra_overrides": [f"hydra.run.dir={hydra_run}"],
                    },
                }
            ),
            encoding="utf-8",
        )
        expected = {
            "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
            "identity_sha256": "new-recipe",
            "identity": {
                "hydra_overrides": [f"hydra.run.dir={hydra_run}"],
            },
        }

        launcher._install_training_identity(
            tmp_path,
            run_name,
            expected,
            replace_preflight_reservation=True,
        )

        assert json.loads(path.read_text(encoding="utf-8")) == expected

    def test_completed_epoch_automatically_recovers_at_eval(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("CW_EVAL_ONLY", raising=False)
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        checkpoint_root = tmp_path / "root"
        checkpoint = (
            checkpoint_root
            / "checkpoints/tworoom_prejepa_original_s3072/weights_epoch_10.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"complete inference checkpoint")
        (checkpoint.parent / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "output_model_name": "tworoom_prejepa_original_s3072",
                    "subdir": "tworoom_prejepa_original_s3072",
                    "seed": 3072,
                    "dataset_name": str(dataset),
                    "frameskip": 5,
                    "wm": {"history_size": 3, "num_preds": 1},
                    "trainer": {"max_epochs": 10},
                }
            ),
            encoding="utf-8",
        )
        identity = {
            "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
            "identity_sha256": "verified-test-identity",
            "identity": {"recipe": "test"},
        }
        (checkpoint.parent / launcher.TRAINING_IDENTITY_FILENAME).write_text(
            json.dumps(identity),
            encoding="utf-8",
        )
        calls: list[tuple[list[str], dict[str, str]]] = []

        def forbidden(*_: object, **__: object) -> None:
            raise AssertionError("completed training must bypass trainer resume")

        class Completed:
            returncode = 0

        def run(command: list[str], **kwargs: object) -> Completed:
            calls.append((command, kwargs["env"]))  # type: ignore[index]
            return Completed()

        monkeypatch.setattr(
            launcher,
            "_training_identity_document",
            lambda **_: identity,
        )
        monkeypatch.setattr(launcher, "validate_resume", forbidden)
        monkeypatch.setattr(launcher.subprocess, "run", run)

        status = launcher.main([
            "--family",
            "prejepa",
            "--original-env",
            "tworoom",
            "--dataset",
            str(dataset),
            "--stablewm-repo",
            str(stablewm_repo),
            "--benchmark-root",
            str(tmp_path / "ContextWorld-v1"),
            "--checkpoint-root",
            str(checkpoint_root),
            "--seeds",
            "3072",
            "--resume",
            "auto",
            "--post-eval",
            "--eval-epoch",
            "10",
            "--eval-result-subdir",
            "recovery-v2",
        ])

        assert status == 0
        assert len(calls) == 1
        command, environment = calls[0]
        assert command[1].endswith("run_stablewm_eval.py")
        assert "--suite" in command
        assert command[command.index("--run-name") + 1] == (
            "tworoom_prejepa_original_s3072"
        )
        assert command[command.index("--epoch") + 1] == "10"
        assert command[command.index("--result-subdir") + 1] == "recovery-v2"
        assert environment["SPT_CACHE_DIR"] == str(checkpoint_root)

    def test_resume_never_cannot_turn_a_stale_epoch_into_eval_recovery(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        checkpoint_root = tmp_path / "root"
        checkpoint = (
            checkpoint_root
            / "checkpoints/tworoom_prejepa_original_s3072/weights_epoch_10.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"stale")

        with pytest.raises(SystemExit, match="refuses existing state"):
            launcher.main([
                "--family",
                "prejepa",
                "--original-env",
                "tworoom",
                "--dataset",
                str(dataset),
                "--stablewm-repo",
                str(stablewm_repo),
                "--benchmark-root",
                str(tmp_path / "ContextWorld-v1"),
                "--checkpoint-root",
                str(checkpoint_root),
                "--seeds",
                "3072",
                "--resume",
                "never",
                "--post-eval",
                "--eval-epoch",
                "10",
                "--print-command",
            ])

    def test_auto_eval_recovery_rejects_mismatched_training_identity(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
    ) -> None:
        checkpoint_root = tmp_path / "root"
        checkpoint = (
            checkpoint_root
            / "checkpoints/tworoom_prejepa_original_s3072/weights_epoch_10.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"stale")
        (checkpoint.parent / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "output_model_name": "tworoom_prejepa_original_s3072",
                    "subdir": "tworoom_prejepa_original_s3072",
                    "seed": 9999,
                    "dataset_name": str(dataset),
                    "frameskip": 5,
                    "wm": {"history_size": 3, "num_preds": 1},
                    "trainer": {"max_epochs": 10},
                }
            ),
            encoding="utf-8",
        )
        (checkpoint.parent / launcher.TRAINING_IDENTITY_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": launcher.TRAINING_IDENTITY_SCHEMA,
                    "identity_sha256": "stale-recipe",
                    "identity": {"seed": 9999},
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit, match="training identity differs"):
            launcher.main([
                "--family",
                "prejepa",
                "--original-env",
                "tworoom",
                "--dataset",
                str(dataset),
                "--stablewm-repo",
                str(stablewm_repo),
                "--benchmark-root",
                str(tmp_path / "ContextWorld-v1"),
                "--checkpoint-root",
                str(checkpoint_root),
                "--seeds",
                "3072",
                "--resume",
                "auto",
                "--post-eval",
                "--eval-epoch",
                "10",
                "--print-command",
            ])

    def test_eval_only_never_logs_into_swanlab_or_validates_training_resume(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checkpoint_root = tmp_path / "root"
        checkpoint = (
            checkpoint_root
            / "checkpoints/tworoom_prejepa_original_s3072/weights_epoch_10.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"complete inference checkpoint")
        monkeypatch.setenv("SWANLAB_API_KEY", "sentinel-secret")
        monkeypatch.delenv("SPT_CACHE_DIR", raising=False)
        commands: list[list[str]] = []

        def forbidden(*_: object, **__: object) -> None:
            raise AssertionError("eval-only must not enter training setup")

        class Completed:
            returncode = 0

        def run(command: list[str], **_: object) -> Completed:
            commands.append(command)
            return Completed()

        monkeypatch.setattr(launcher, "validate_resume", forbidden)
        monkeypatch.setattr(
            launcher,
            "_login_swanlab_without_exposing_key",
            forbidden,
        )
        monkeypatch.setattr(launcher.subprocess, "run", run)

        status = launcher.main([
            "--family",
            "prejepa",
            "--original-env",
            "tworoom",
            "--dataset",
            str(dataset),
            "--stablewm-repo",
            str(stablewm_repo),
            "--benchmark-root",
            str(tmp_path / "ContextWorld-v1"),
            "--checkpoint-root",
            str(checkpoint_root),
            "--seeds",
            "3072",
            "--resume",
            "required",
            "--logger",
            "swanlab",
            "--eval-only",
            "--eval-epoch",
            "10",
        ])

        assert status == 0
        assert len(commands) == 1
        assert commands[0][1].endswith("run_stablewm_eval.py")
        assert "sentinel-secret" not in " ".join(commands[0])


class TestLoggingAndLossBoundaries:

    def test_swanlab_is_profiled_without_putting_the_key_in_argv(
        self,
        stablewm_repo: Path,
        dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        secret = "secret-that-must-not-appear"
        monkeypatch.setenv("SWANLAB_API_KEY", secret)
        status = launcher.main([
            "--family",
            "lewm",
            "--component",
            "speed",
            "--data-group",
            "pusht",
            "--dataset",
            str(dataset),
            "--stablewm-repo",
            str(stablewm_repo),
            "--checkpoint-root",
            str(dataset.parent / "ckpt"),
            "--logger",
            "swanlab",
            "--swanlab-project",
            "contextworld",
            "--print-command",
        ])

        assert status == 0
        output = capsys.readouterr().out
        assert "logger_backend=swanlab" in output
        assert "swanlab.config.project=contextworld" in output
        assert secret not in output

    def test_prejepa_rejects_swanlab_when_the_trainer_does_not_consume_it(
            self, stablewm_repo: Path, dataset: Path) -> None:
        args = _args(
            stablewm_repo,
            dataset,
            "--family",
            "prejepa",
            "--logger",
            "swanlab",
        )

        with pytest.raises(SystemExit, match="does not call build_training_logger"):
            _build(args, stablewm_repo)

    def test_compatible_prejepa_uses_the_common_swanlab_contract(
        self, stablewm_repo: Path, dataset: Path
    ) -> None:
        _enable_prejepa_common_logger(stablewm_repo)
        args = _args(
            stablewm_repo,
            dataset,
            "--family",
            "prejepa",
            "--logger",
            "swanlab",
            "--swanlab-project",
            "contextworld",
        )

        _, pairs, _ = _build(args, stablewm_repo)

        assert pairs["logger_backend"] == "swanlab"
        assert pairs["swanlab.enabled"] == "true"
        assert pairs["swanlab.config.project"] == "contextworld"
        assert pairs["swanlab.config.experiment_name"] == "run"

    def test_prejepa_swanlab_login_precedes_the_training_process(
        self,
        stablewm_repo: Path,
        dataset: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _enable_prejepa_common_logger(stablewm_repo)
        monkeypatch.setenv("SWANLAB_API_KEY", "injected-secret")
        events: list[str] = []

        def login(environment: dict[str, str]) -> None:
            assert environment["SWANLAB_API_KEY"] == "injected-secret"
            events.append("login")

        class Completed:
            returncode = 0

        def run(*_args: object, **_kwargs: object) -> Completed:
            events.append("train")
            return Completed()

        monkeypatch.setattr(launcher, "_login_swanlab_without_exposing_key", login)
        monkeypatch.setattr(launcher.subprocess, "run", run)

        status = launcher.main(
            [
                "--family",
                "prejepa",
                "--component",
                "speed",
                "--dataset",
                str(dataset),
                "--stablewm-repo",
                str(stablewm_repo),
                "--checkpoint-root",
                str(dataset.parent / "swanlab-checkpoints"),
                "--logger",
                "swanlab",
                "--seeds",
                "3072",
            ]
        )

        assert status == 0
        assert events == ["login", "train"]

    def test_viswm_is_an_independent_family_with_method_owned_options(
            self, stablewm_repo: Path, dataset: Path) -> None:
        args = _args(
            stablewm_repo,
            dataset,
            "--family",
            "viswm",
            "--data-group",
            "pusht",
            "--viswm-weight",
            "4.5",
        )

        _, pairs, _ = _build(args, stablewm_repo)

        assert "loss.regularizer" not in pairs
        assert pairs["loss.visreg.weight"] == "4.5"

    def test_lewm_cli_has_no_visreg_method_switch(
            self, stablewm_repo: Path, dataset: Path) -> None:
        with pytest.raises(SystemExit):
            _args(
                stablewm_repo,
                dataset,
                "--family",
                "lewm",
                "--data-group",
                "pusht",
                "--lewm-regularizer",
                "visreg",
            )


class TestExplicitEvaluation:

    @pytest.fixture(autouse=True)
    def _development_bundle_root(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Suites must name the public Development bundle explicitly."""

        root = tmp_path / "ContextWorld-v1"
        root.mkdir()
        monkeypatch.setenv("CONTEXTWORLD_BENCHMARK_ROOT", str(root))

    def test_original_metric_receipt_parser_returns_typed_values(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "metrics.txt"
        path.write_text(
            "==== RESULTS ====\n"
            "metrics: {'success_rate': 58.0, "
            "'episode_successes': array([True, False])}\n"
            "evaluation_time: 2071.4979 seconds\n",
            encoding="utf-8",
        )

        assert evaluator._parse_original_metrics(path, num_eval=50) == {
            "success_rate_percent": 58.0,
            "successful_episodes": 29,
            "evaluation_time_seconds": 2071.4979,
        }

    def test_viswm_icl_uses_the_shared_lewm_checkpoint_adapter(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
    ) -> None:
        checkpoint_path = tmp_path / "weights_epoch_10.pt"
        args = evaluator.parse_args(
            [
                "--family",
                "viswm",
                "--component",
                "speed",
                "--checkpoint",
                str(checkpoint_path),
                "--stablewm-repo",
                str(stablewm_repo),
                "--print-command",
            ]
        )
        checkpoint = evaluator.ResolvedCheckpoint(
            path=checkpoint_path,
            run_name="speed_viswm_joint_scratch_v1_s3072",
            epoch=10,
            checkpoint_root=tmp_path,
            policy="weights_epoch_10.pt",
        )

        steps = evaluator._build_icl_steps(
            args,
            checkpoint=checkpoint,
            stablewm_repo=stablewm_repo,
            stablewm_ref="a" * 40,
            benchmark_root=tmp_path / "ContextWorld-v1",
            components=("speed",),
            eval_root=tmp_path / "eval",
            contract=evaluator.load_contract(),
        )

        command = steps[0].command
        assert command[command.index("--adapter") + 1] == "lewm"

    def test_eval_revision_reads_git_metadata_without_running_git(
        self,
        stablewm_repo: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        revision = "f" * 40
        git_dir = stablewm_repo / ".git"
        ref = git_dir / "refs/heads/main"
        ref.parent.mkdir(parents=True)
        ref.write_text(revision + "\n", encoding="utf-8")
        (git_dir / "HEAD").write_text(
            "ref: refs/heads/main\n", encoding="utf-8"
        )

        def reject_git(*_: object, **__: object) -> None:
            raise AssertionError("evaluation must not invoke git rev-parse")

        monkeypatch.setattr(evaluator.subprocess, "run", reject_git)

        assert evaluator._stablewm_ref(stablewm_repo, None) == revision

    def test_explicit_eval_revision_must_be_a_full_sha(
        self, stablewm_repo: Path
    ) -> None:
        with pytest.raises(SystemExit, match="40-digit SHA"):
            evaluator._stablewm_ref(stablewm_repo, "875e607")

    def test_suite_reuses_original_cem_and_all_environment_icl_scorers(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        status = evaluator.main([
            "--suite",
            "--family",
            "prejepa",
            "--original-env",
            "tworoom",
            "--dataset",
            str(dataset),
            "--run-name",
            "tworoom_prejepa_original_s3072",
            "--epoch",
            "10",
            "--checkpoint-root",
            str(tmp_path / "checkpoint-root"),
            "--stablewm-repo",
            str(stablewm_repo),
            "--stablewm-ref",
            "a" * 40,
            "--training-seed",
            "3072",
            "--print-command",
        ])

        output = capsys.readouterr().out
        assert status == 0
        assert "original-cem:" in output
        for component in ("speed", "door", "action_delay", "portal_exit"):
            assert f"component={component}" in output
            assert f"benchmark_icl/{component}/result.json" in output
        assert "contact_friction" not in output

    def test_component_suite_skips_only_cem_when_original_data_is_absent(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        status = evaluator.main([
            "--suite",
            "--family",
            "lewm",
            "--component",
            "contact_friction",
            "--checkpoint",
            str(tmp_path / "run" / "weights_epoch_10.pt"),
            "--stablewm-repo",
            str(stablewm_repo),
            "--stablewm-ref",
            "b" * 40,
            "--print-command",
        ])

        output = capsys.readouterr().out
        assert status == 0
        assert "original-cem skipped" in output
        assert "component=contact_friction" in output
        assert "benchmark_icl/contact_friction/result.json" in output
        assert "--evaluation-split development" in output

    @pytest.mark.parametrize(
        "environment,components",
        [
            ("tworoom", ("speed", "door", "action_delay", "portal_exit")),
            ("pusht", ("action_strength", "contact_friction", "motion_damping")),
            ("reacher", ("robot_arm_mass",)),
            ("cube", ("cube_gripper_carry",)),
        ],
    )
    def test_every_public_suite_icl_command_uses_development_bundle(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        environment: str,
        components: tuple[str, ...],
    ) -> None:
        """All nine ICL commands are Development-only and bundle-rooted."""

        status = evaluator.main([
            "--suite",
            "--icl-only",
            "--family",
            "lewm",
            "--original-env",
            environment,
            "--checkpoint",
            str(tmp_path / "run" / "weights_epoch_10.pt"),
            "--stablewm-repo",
            str(stablewm_repo),
            "--stablewm-ref",
            "c" * 40,
            "--print-command",
        ])

        output = capsys.readouterr().out
        root = tmp_path / "ContextWorld-v1"
        assert status == 0
        assert output.count("--evaluation-split development") == len(components)
        assert output.count(f"--benchmark-root {root}") == len(components)
        for component in components:
            assert f"component={component}" in output

    def test_suite_rejects_an_unconfigured_development_bundle(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("CONTEXTWORLD_BENCHMARK_ROOT")

        with pytest.raises(SystemExit, match="CONTEXTWORLD_BENCHMARK_ROOT"):
            evaluator.main([
                "--suite",
                "--icl-only",
                "--family",
                "lewm",
                "--component",
                "speed",
                "--checkpoint",
                str(tmp_path / "run" / "weights_epoch_10.pt"),
                "--stablewm-repo",
                str(stablewm_repo),
                "--stablewm-ref",
                "c" * 40,
            ])

    def test_original_suite_never_silently_skips_its_primary_cem(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(SystemExit, match="primary target"):
            evaluator.main([
                "--suite",
                "--family",
                "lewm",
                "--original-env",
                "tworoom",
                "--checkpoint",
                str(tmp_path / "run/weights_epoch_10.pt"),
                "--stablewm-repo",
                str(stablewm_repo),
                "--stablewm-ref",
                "b" * 40,
                "--print-command",
            ])

    def test_standalone_suite_writes_results_beside_the_checkpoint(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checkpoint = tmp_path / "checkpoints/run/weights_epoch_10.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")

        calls: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> argparse.Namespace:
            calls.append(command)
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"status":"completed"}\n', encoding="utf-8")
            return argparse.Namespace(returncode=0)

        monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

        arguments = [
            "--suite",
            "--family",
            "lewm",
            "--component",
            "speed",
            "--checkpoint",
            str(checkpoint),
            "--stablewm-repo",
            str(stablewm_repo),
            "--stablewm-ref",
            "c" * 40,
            "--result-subdir",
            "attempt-2",
        ]
        status = evaluator.main(arguments)

        eval_root = checkpoint.parent / "eval_results/attempt-2"
        manifest = json.loads(
            (eval_root / "manifest.json").read_text(encoding="utf-8")
        )
        assert status == 0
        assert manifest["status"] == "completed"
        assert manifest["steps"][0]["status"] == "skipped"
        assert manifest["steps"][1]["status"] == "completed"
        assert manifest["steps"][1]["evaluation_split"] == "development"
        assert manifest["steps"][1]["benchmark_root"] == str(
            tmp_path / "ContextWorld-v1"
        )
        assert manifest["request"]["evaluation"]["icl_evaluation_split"] == (
            "development"
        )
        assert (eval_root / "benchmark_icl/speed/result.json").is_file()
        assert manifest["outputs"][0]["path"] == (
            "benchmark_icl/speed/result.json"
        )
        assert manifest["schema_version"] == evaluator.SUITE_MANIFEST_SCHEMA
        assert manifest["request_sha256"] == evaluator._json_sha256(
            manifest["request"]
        )

        original_manifest = (eval_root / "manifest.json").read_bytes()
        assert evaluator.main(arguments) == 0
        assert len(calls) == 1
        assert (eval_root / "manifest.json").read_bytes() == original_manifest

        with pytest.raises(SystemExit, match="different"):
            evaluator.main([*arguments, "--eval-batch-size", "65"])
        assert len(calls) == 1
        assert (eval_root / "manifest.json").read_bytes() == original_manifest

        result = eval_root / "benchmark_icl/speed/result.json"
        result.write_text('{"status":"tampered"}\n', encoding="utf-8")
        with pytest.raises(SystemExit, match="size/SHA256"):
            evaluator.main(arguments)
        assert len(calls) == 1

    def test_suite_records_original_exception_and_runs_icl(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checkpoint = tmp_path / "checkpoints/run/weights_epoch_10.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")

        def fail_original(*_: object, **__: object) -> int:
            raise SystemExit("upstream evaluator wrote no metrics")

        monkeypatch.setattr(evaluator, "_run_original", fail_original)

        def fake_run(command: list[str], **_: object) -> argparse.Namespace:
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"status":"completed"}\n', encoding="utf-8")
            return argparse.Namespace(returncode=0)

        monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

        status = evaluator.main([
            "--suite",
            "--family",
            "lewm",
            "--component",
            "speed",
            "--dataset",
            str(dataset),
            "--checkpoint",
            str(checkpoint),
            "--stablewm-repo",
            str(stablewm_repo),
            "--stablewm-ref",
            "d" * 40,
        ])

        manifest = json.loads(
            (checkpoint.parent / "eval_results/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert status == 1
        assert manifest["status"] == "failed"
        assert all(
            row["status"] == "failed"
            for row in manifest["steps"]
            if row["kind"] == "original_environment_cem"
        )
        assert all(
            row["status"] == "completed"
            for row in manifest["steps"]
            if row["kind"] == "benchmark_icl"
        )

    def test_suite_runs_every_cem_seed_and_remaining_icl_steps_after_failure(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checkpoint = tmp_path / "checkpoints/run/weights_epoch_10.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        executed: list[str] = []
        cem_seeds: list[int] = []

        def fake_run(command: list[str], **_: object) -> argparse.Namespace:
            seed_arg = next(
                (argument for argument in command if argument.startswith("seed=")),
                None,
            )
            if seed_arg is not None:
                seed = int(seed_arg.split("=", 1)[1])
                cem_seeds.append(seed)
                if seed == 42:
                    return argparse.Namespace(returncode=7)
                if seed == 43:
                    return argparse.Namespace(returncode=0)
                metrics_relative = next(
                    argument.split("=", 1)[1]
                    for argument in command
                    if argument.startswith("output.filename=")
                )
                metrics = checkpoint.parent / metrics_relative
                metrics.parent.mkdir(parents=True, exist_ok=True)
                metrics.write_text(
                    "metrics: {'success_rate': 50.0}\n"
                    "evaluation_time: 1 seconds\n",
                    encoding="utf-8",
                )
                return argparse.Namespace(returncode=0)
            component = command[command.index("--task") + 1]
            executed.append(component)
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"status":"completed"}\n', encoding="utf-8")
            return argparse.Namespace(returncode=0)

        monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

        status = evaluator.main([
            "--suite",
            "--family",
            "lewm",
            "--original-env",
            "tworoom",
            "--dataset",
            str(dataset),
            "--checkpoint",
            str(checkpoint),
            "--stablewm-repo",
            str(stablewm_repo),
            "--stablewm-ref",
            "e" * 40,
            "--eval-seeds",
            "42,43,44",
        ])

        manifest = json.loads(
            (checkpoint.parent / "eval_results/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert status == 7
        assert cem_seeds == [42, 43, 44]
        assert executed == ["speed", "door", "action_delay", "portal_exit"]
        assert manifest["status"] == "failed"
        cem = [
            row
            for row in manifest["steps"]
            if row["kind"] == "original_environment_cem"
        ]
        assert [row["status"] for row in cem] == ["failed", "failed", "completed"]
        assert [row["eval_seed"] for row in cem] == [42, 43, 44]
        assert all(
            row["status"] == "completed"
            for row in manifest["steps"]
            if row["kind"] == "benchmark_icl"
        )

    def test_eval_uses_the_exact_checkpoint_and_never_overwrites_by_default(
            self, stablewm_repo: Path, dataset: Path, tmp_path: Path) -> None:
        checkpoint_root = tmp_path / "root"
        checkpoint = checkpoint_root / "checkpoints/run/weights_epoch_5.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"")
        args = evaluator.parse_args([
            "--family",
            "pldm",
            "--original-env",
            "tworoom",
            "--dataset",
            str(dataset),
            "--run-name",
            "run",
            "--epoch",
            "5",
            "--checkpoint-root",
            str(checkpoint_root),
            "--stablewm-repo",
            str(stablewm_repo),
            "--eval-seeds",
            "42,43",
        ])

        _, commands = evaluator.build_commands(args)

        assert len(commands) == 2
        assert "policy=run/weights_epoch_5.pt" in commands[0][-1]
        assert "++plan_config.action_block=5" in commands[0][-1]
        assert commands[0][1] != commands[1][1]
        assert "eval_results" in str(commands[0][2])

    def test_eval_run_name_cannot_escape_checkpoint_root(self, stablewm_repo: Path,
                                                         dataset: Path,
                                                         tmp_path: Path) -> None:
        args = evaluator.parse_args([
            "--family",
            "lewm",
            "--original-env",
            "tworoom",
            "--dataset",
            str(dataset),
            "--run-name",
            "../outside",
            "--epoch",
            "5",
            "--checkpoint-root",
            str(tmp_path),
            "--stablewm-repo",
            str(stablewm_repo),
            "--print-command",
        ])

        with pytest.raises(SystemExit, match="Run names"):
            evaluator.build_commands(args)

    @pytest.mark.parametrize(
        "environment,state_key,extra",
        [
            ("tworoom", "proprio", []),
            ("pusht", "proprio", []),
            (
                "reacher",
                "observation",
                [
                    "objective.terms.0.1.pred_key=predicted_observation_emb",
                    "objective.terms.0.1.goal_key=observation_goal_emb",
                    "dataset.keys_to_cache=[action,observation]",
                ],
            ),
            (
                "cube",
                "observation",
                [
                    "objective.terms.0.1.pred_key=predicted_observation_emb",
                    "objective.terms.0.1.goal_key=observation_goal_emb",
                    "dataset.keys_to_cache=[action,observation]",
                ],
            ),
        ],
    )
    def test_prejepa_cem_uses_split_objective_and_state_history_bridge(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        environment: str,
        state_key: str,
        extra: list[str],
    ) -> None:
        args = evaluator.parse_args(
            [
                "--family",
                "prejepa",
                "--original-env",
                environment,
                "--dataset",
                str(dataset),
                "--run-name",
                "run",
                "--epoch",
                "10",
                "--checkpoint-root",
                str(tmp_path / "root"),
                "--stablewm-repo",
                str(stablewm_repo),
                "--print-command",
            ]
        )

        _, commands = evaluator.build_commands(args)
        command = commands[0][-1]

        assert command[1].endswith("scripts/run_stablewm_plan.py")
        assert command[command.index("--history-keys") + 1] == (
            f"pixels,{state_key}"
        )
        assert "objective=goal_mse_pixels_proprio" in command
        for value in extra:
            assert value in command

    @pytest.mark.parametrize(
        "history_size,action_width,error",
        [
            (7, 10, "trained with history_size=7"),
            (3, 14, "trained with action_block=7"),
        ],
    )
    def test_prejepa_cem_rejects_checkpoint_geometry_mismatch(
        self,
        stablewm_repo: Path,
        dataset: Path,
        tmp_path: Path,
        history_size: int,
        action_width: int,
        error: str,
    ) -> None:
        checkpoint = tmp_path / "run/weights_epoch_10.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (checkpoint.parent / "config.json").write_text(
            json.dumps(
                {
                    "history_size": history_size,
                    "extra_encoders": {
                        "modules": {
                            "proprio": {"in_chans": 2},
                            "action": {"in_chans": action_width},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        args = evaluator.parse_args(
            [
                "--family",
                "prejepa",
                "--original-env",
                "tworoom",
                "--dataset",
                str(dataset),
                "--checkpoint",
                str(checkpoint),
                "--stablewm-repo",
                str(stablewm_repo),
                "--print-command",
            ]
        )

        with pytest.raises(SystemExit, match=error):
            evaluator.build_commands(args)

    def test_state_conditioned_prejepa_keeps_strict_row_and_runs_diagnostic(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checkpoint = tmp_path / "run/weights_epoch_10.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (checkpoint.parent / "config.json").write_text(
            json.dumps(
                {
                    "history_size": 3,
                    "extra_encoders": {
                        "modules": {"proprio": {}, "action": {}}
                    },
                }
            ),
            encoding="utf-8",
        )
        calls: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> argparse.Namespace:
            calls.append(command)
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"status":"completed"}\n', encoding="utf-8")
            return argparse.Namespace(returncode=0)

        monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

        status = evaluator.main(
            [
                "--suite",
                "--icl-only",
                "--family",
                "prejepa",
                "--original-env",
                "tworoom",
                "--checkpoint",
                str(checkpoint),
                "--stablewm-repo",
                str(stablewm_repo),
                "--stablewm-ref",
                "e" * 40,
                "--result-subdir",
                "state-input-contract",
            ]
        )

        manifest = json.loads(
            (
                checkpoint.parent
                / "eval_results/state-input-contract/manifest.json"
            ).read_text(encoding="utf-8")
        )
        assert status == 0
        assert manifest["status"] == "completed"
        strict = [
            row for row in manifest["steps"] if row["kind"] == "benchmark_icl"
        ]
        diagnostic = [
            row
            for row in manifest["steps"]
            if row["kind"] == "benchmark_icl_diagnostic"
        ]
        assert len(strict) == len(diagnostic) == 4
        assert all(row["status"] == "not_compatible" for row in strict)
        assert all("only pixels and actions" in row["reason"] for row in strict)
        assert all(row["status"] == "completed" for row in diagnostic)
        assert all(
            row["protocol_track"] == evaluator.DEVELOPMENT_ICL_PROTOCOL_TRACK
            for row in strict
        )
        assert all(
            row["protocol_track"]
            == evaluator.DEVELOPMENT_DIAGNOSTIC_ICL_PROTOCOL_TRACK
            for row in diagnostic
        )
        assert len({row["id"] for row in manifest["steps"]}) == len(
            manifest["steps"]
        )
        assert all("protocol_track" in row for row in manifest["steps"])

        diagnostic_by_component = {
            command[command.index("--task") + 1]: command for command in calls
        }
        assert set(diagnostic_by_component) == {
            "speed", "door", "action_delay", "portal_exit"
        }
        for command in diagnostic_by_component.values():
            assert command[
                command.index("--prejepa-missing-context-policy") + 1
            ] == "normalized_zero"
        action_delay = diagnostic_by_component["action_delay"]
        assert action_delay[action_delay.index("--history-adapter") + 1] == (
            "h3_tail_projection"
        )

    def test_state_free_prejepa_runs_the_strict_icl_track(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checkpoint = tmp_path / "run/weights_epoch_10.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (checkpoint.parent / "config.json").write_text(
            json.dumps(
                {
                    "history_size": 3,
                    "extra_encoders": {"modules": {"action": {}}},
                }
            ),
            encoding="utf-8",
        )
        calls: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> argparse.Namespace:
            calls.append(command)
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"status":"completed"}\n', encoding="utf-8")
            return argparse.Namespace(returncode=0)

        monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
        status = evaluator.main(
            [
                "--suite",
                "--icl-only",
                "--family",
                "prejepa",
                "--component",
                "speed",
                "--checkpoint",
                str(checkpoint),
                "--stablewm-repo",
                str(stablewm_repo),
                "--stablewm-ref",
                "f" * 40,
            ]
        )

        manifest = json.loads(
            (checkpoint.parent / "eval_results/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        strict = [
            row for row in manifest["steps"] if row["kind"] == "benchmark_icl"
        ]
        assert status == 0
        assert len(calls) == len(strict) == 1
        assert strict[0]["status"] == "completed"
        assert not any(
            row["kind"] == "benchmark_icl_diagnostic"
            for row in manifest["steps"]
        )
        assert "--prejepa-missing-context-policy" not in calls[0]

    def test_suite_continues_after_icl_failure_and_missing_output(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checkpoint = tmp_path / "run/weights_epoch_10.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        executed: list[str] = []

        def fake_run(command: list[str], **_: object) -> argparse.Namespace:
            component = command[command.index("--task") + 1]
            executed.append(component)
            if component == "speed":
                return argparse.Namespace(returncode=7)
            if component == "door":
                return argparse.Namespace(returncode=0)
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text('{"status":"completed"}\n', encoding="utf-8")
            return argparse.Namespace(returncode=0)

        monkeypatch.setattr(evaluator.subprocess, "run", fake_run)
        status = evaluator.main(
            [
                "--suite",
                "--icl-only",
                "--family",
                "lewm",
                "--original-env",
                "tworoom",
                "--checkpoint",
                str(checkpoint),
                "--stablewm-repo",
                str(stablewm_repo),
                "--stablewm-ref",
                "a" * 40,
            ]
        )

        manifest = json.loads(
            (checkpoint.parent / "eval_results/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        rows = {
            row["component"]: row
            for row in manifest["steps"]
            if row["kind"] == "benchmark_icl"
        }
        assert status == 7
        assert executed == ["speed", "door", "action_delay", "portal_exit"]
        assert manifest["status"] == "failed"
        assert rows["speed"]["status"] == "failed"
        assert rows["door"]["status"] == "failed"
        assert rows["door"]["error"]["type"] == "RuntimeError"
        assert rows["action_delay"]["status"] == "completed"
        assert rows["portal_exit"]["status"] == "completed"


class TestTrainingMethodOverlay:
    """CW_METHOD is orthogonal to CW_FAMILY and driven only by the profile."""

    FAMILIES = ("lewm", "viswm", "pldm", "prejepa")

    @staticmethod
    def _expose_conditional_joint(stablewm_repo: Path, family: str) -> None:
        """Mirror the checkout's own disabled loss.conditional_joint block."""

        contract = launcher.load_profile_contract()
        config = (
            stablewm_repo
            / "scripts/train/config"
            / f"{contract['families'][family]['config_name']}.yaml"
        )
        payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        loss = payload.setdefault("loss", {})
        loss["conditional_joint"] = {
            "enabled": False,
            "weight": 0.0,
            "group_width": 2,
        }
        config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    @staticmethod
    def _component_args(
        stablewm_repo: Path,
        tmp_path: Path,
        family: str,
        component: str = "contact_friction",
        *extra: str,
    ) -> argparse.Namespace:
        return launcher.parse_args([
            "--component",
            component,
            "--family",
            family,
            "--method",
            "coja_v1",
            "--stablewm-repo",
            str(stablewm_repo),
            "--checkpoint-root",
            str(tmp_path / "checkpoints-root"),
            *extra,
        ])

    @staticmethod
    def _contact_friction_target() -> launcher.Target:
        prefix = launcher.CONTEXTWORLD_DATASET_URI_PREFIX
        return launcher.Target(
            label="contact_friction",
            dataset=f"{prefix}contact_friction",
            data_group="pusht",
            history_size=3,
            action_dim=2,
            environment="pusht",
        )

    @pytest.fixture
    def registered_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            launcher,
            "describe_contextworld_dataset",
            lambda uri: {
                "component": "contact_friction",
                "history_length": 3,
                "frameskip": 5,
                "action_dimension": 2,
                "conditional_joint": {
                    "method": "coja_v1",
                    "group_width": 2,
                    "relation_kind": "public_pair_identity_v1",
                },
            },
        )

    def test_coja_is_registered_for_every_base_family_without_new_names(
        self,
    ) -> None:
        contract = launcher.load_profile_contract()
        profile = launcher.method_profile(contract, "coja_v1")

        assert sorted(profile["families"]) == sorted(self.FAMILIES)
        assert set(profile["families"]) <= set(contract["families"])
        assert sorted(profile["components"]) == ["contact_friction"]

    @pytest.mark.parametrize("family", FAMILIES)
    def test_every_base_family_renders_the_same_registered_coja_keys(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        registered_identity: None,
        family: str,
    ) -> None:
        self._expose_conditional_joint(stablewm_repo, family)
        contract = launcher.load_profile_contract()
        args = self._component_args(stablewm_repo, tmp_path, family)

        launcher._validate_method(args, contract)
        entries = launcher.build_overrides(
            args,
            contract,
            self._contact_friction_target(),
            run_name="run",
            seed=3072,
            stablewm_repo=stablewm_repo,
        )
        pairs = _pairs(entries)

        assert pairs["loss.conditional_joint.enabled"] == "true"
        assert pairs["loss.conditional_joint.weight"] == "0.09"
        assert pairs["loss.conditional_joint.group_width"] == "2"
        assert pairs["trainer.use_distributed_sampler"] == "false"

    @pytest.mark.parametrize("family", FAMILIES)
    def test_coja_run_name_never_shares_the_native_run_directory(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        family: str,
    ) -> None:
        args = self._component_args(stablewm_repo, tmp_path, family)

        name = launcher._run_name(
            args, self._contact_friction_target(), 3072, (3072,)
        )

        assert name == (
            f"contact_friction_{family}_joint_scratch_v1_coja_v1_s3072"
        )

    @pytest.mark.parametrize("family", FAMILIES)
    def test_unsupported_component_fails_closed_for_every_family(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        family: str,
    ) -> None:
        contract = launcher.load_profile_contract()
        args = self._component_args(
            stablewm_repo, tmp_path, family, "motion_damping"
        )

        with pytest.raises(SystemExit) as failure:
            launcher._validate_method(args, contract)

        assert "motion_damping" in str(failure.value)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_checkout_without_the_loss_interface_fails_closed(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        registered_identity: None,
        family: str,
    ) -> None:
        contract = launcher.load_profile_contract()
        args = self._component_args(stablewm_repo, tmp_path, family)

        with pytest.raises(SystemExit) as failure:
            launcher.build_overrides(
                args,
                contract,
                self._contact_friction_target(),
                run_name="run",
                seed=3072,
                stablewm_repo=stablewm_repo,
            )

        assert "loss.conditional_joint" in str(failure.value)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_original_environment_training_never_accepts_coja(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        family: str,
    ) -> None:
        contract = launcher.load_profile_contract()
        args = launcher.parse_args([
            "--original-env",
            "pusht",
            "--family",
            family,
            "--method",
            "coja_v1",
            "--stablewm-repo",
            str(stablewm_repo),
            "--checkpoint-root",
            str(tmp_path / "checkpoints-root"),
        ])

        with pytest.raises(SystemExit) as failure:
            launcher._validate_method(args, contract)

        assert "original-environment" in str(failure.value)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_historical_release_track_never_accepts_coja(
        self,
        stablewm_repo: Path,
        tmp_path: Path,
        family: str,
    ) -> None:
        contract = launcher.load_profile_contract()
        args = self._component_args(
            stablewm_repo,
            tmp_path,
            family,
            "contact_friction",
            "--training-track",
            "historical_release",
        )

        with pytest.raises(SystemExit) as failure:
            launcher._validate_method(args, contract)

        assert "training tracks" in str(failure.value)
