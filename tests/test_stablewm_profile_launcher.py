"""The public StableWM entry maps one contract to three real YAML dialects."""

from __future__ import annotations

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
import run_stablewm_train as launcher  # noqa: E402


@pytest.fixture
def stablewm_repo(tmp_path: Path) -> Path:
    config = tmp_path / "scripts/train/config"
    config.mkdir(parents=True)
    train = config.parent
    (train / "lewm.py").write_text("from x import build_training_logger\n",
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
        "component,expected_key,expected_dim",
        [
            ("robot_arm_mass", "observation", 6),
            ("cube_gripper_carry", "observation", 28),
            ("action_strength", "proprio", 4),
        ],
    )
    def test_component_profile_inherits_the_environment_encoding(
        self,
        stablewm_repo: Path,
        dataset: Path,
        component: str,
        expected_key: str,
        expected_dim: int,
    ) -> None:
        args = launcher.parse_args([
            "--family", "prejepa",
            "--component", component,
            "--dataset", str(dataset),
            "--stablewm-repo", str(stablewm_repo),
            "--checkpoint-root", str(dataset.parent / "ckpt"),
        ])

        target = launcher.resolve_target(args, launcher.load_profile_contract())

        assert target.encoding_key == expected_key
        assert target.encoding_dim == expected_dim

    @pytest.mark.parametrize("component", ["robot_arm_mass", "cube_gripper_carry"])
    def test_prejepa_component_observation_is_remapped(
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
        assert "+wm.encoding.observation=10" in entries

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

        with pytest.raises(SystemExit, match="non-empty run directory"):
            launcher.validate_resume(tmp_path, "run", "never")

    def test_resume_required_means_full_training_state(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="full-state checkpoint"):
            launcher.validate_resume(tmp_path, "run", "required")
        checkpoint = tmp_path / "checkpoints/run/run_weights.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"")

        launcher.validate_resume(tmp_path, "run", "required")

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

    def test_visreg_is_only_emitted_when_the_checkout_declares_it(
            self, stablewm_repo: Path, dataset: Path) -> None:
        args = _args(
            stablewm_repo,
            dataset,
            "--family",
            "lewm",
            "--data-group",
            "pusht",
            "--lewm-regularizer",
            "visreg",
            "--lewm-visreg-weight",
            "0.09",
        )

        _, pairs, _ = _build(args, stablewm_repo)

        assert pairs["loss.regularizer"] == "visreg"
        assert pairs["loss.visreg.weight"] == "0.09"


class TestExplicitEvaluation:

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
