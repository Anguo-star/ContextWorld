"""Focused contracts for ContextWorld's true multi-step StableWM overlay.

These tests deliberately do not import a live Stable-WorldModel checkout.  They
check the two things that must remain true across upstream YAML variants:

* ``wm.num_preds`` remains the upstream one-step target-offset setting; and
* ``CW_ROLLOUT_STEPS`` selects a separate, differentiable autoregressive path.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_stablewm_train as launcher  # noqa: E402
import run_stablewm_rollout_entry as rollout_entry  # noqa: E402
from contextworld.training import stablewm_rollout  # noqa: E402


def _fake_stablewm_repo(root: Path) -> Path:
    """Write only the files the public launcher statically inspects."""

    train = root / "scripts/train"
    config = train / "config"
    config.mkdir(parents=True)
    logger = {
        "logger_backend": "none",
        "swanlab": {"enabled": False, "config": {}},
        "wandb": {"enabled": False, "config": {}},
    }
    for family in ("lewm", "pldm"):
        (train / f"{family}.py").write_text(
            "from x import build_training_logger\n", encoding="utf-8"
        )
        (config / f"{family}.yaml").write_text(
            yaml.safe_dump({**logger, "trainer": {"max_epochs": 10}}),
            encoding="utf-8",
        )
    (train / "viswm.py").write_text(
        "from x import build_training_logger\n", encoding="utf-8"
    )
    (config / "viswm.yaml").write_text(
        yaml.safe_dump({
            "defaults": ["lewm", "_self_"],
            "optimizer": {"lr": 0.0001},
            "loss": {"regularizer": "visreg", "visreg": {"weight": 1.0}},
        }),
        encoding="utf-8",
    )
    (train / "prejepa.py").write_text(
        "enabled = cfg.wandb.enabled\n", encoding="utf-8"
    )
    (config / "prejepa.yaml").write_text(
        yaml.safe_dump({"trainer": {"max_epochs": 10}}), encoding="utf-8"
    )
    data = config / "data"
    data.mkdir()
    (data / "pusht.yaml").write_text(
        yaml.safe_dump({"dataset": {"keys_to_load": ["pixels", "action"]}}),
        encoding="utf-8",
    )
    return root


def _component_args(
    stablewm_repo: Path,
    tmp_path: Path,
    *,
    family: str = "lewm",
    method: str = "native",
    component: str = "motion_damping",
    rollout_steps: int = 3,
    training_track: str = "joint_scratch_v1",
    extra: tuple[str, ...] = (),
) -> launcher.argparse.Namespace:
    return launcher.parse_args(
        [
            "--component",
            component,
            "--family",
            family,
            "--method",
            method,
            "--training-track",
            training_track,
            "--rollout-steps",
            str(rollout_steps),
            "--benchmark-root",
            str(tmp_path / "ContextWorld-v1"),
            "--dataset-root",
            str(tmp_path / "original-data"),
            "--stablewm-repo",
            str(stablewm_repo),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            *extra,
        ]
    )


def _runtime_identity(*, method: str) -> dict[str, object]:
    return {
        "component": "motion_damping",
        "history_length": 3,
        "frameskip": 5,
        "action_dimension": 2,
        "payload_id": "data",
        "member_count": 1,
        "weights": {"original": 0.5, "synthetic": 0.5},
        "epoch_size": None,
        "conditional_joint": (
            {
                "method": "coja_v1",
                "group_width": 2,
                "relation_kind": "public_pair_identity_v1",
            }
            if method == "coja_v1"
            else None
        ),
    }


def _target(dataset: str) -> launcher.Target:
    return launcher.Target(
        label="motion_damping",
        dataset=dataset,
        data_group="pusht",
        history_size=3,
        action_dim=2,
        environment="pusht",
        encoding_key=None,
        encoding_dim=None,
    )


def _enable_coja_config(
    stablewm_repo: Path, *, family: str, contract: dict[str, object]
) -> None:
    config_name = str(contract["families"][family]["config_name"])  # type: ignore[index]
    config = stablewm_repo / "scripts/train/config" / f"{config_name}.yaml"
    payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    payload.setdefault("loss", {})["conditional_joint"] = {
        "enabled": False,
        "weight": 0.0,
        "group_width": 2,
    }
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")


class TestRolloutLauncherContract:
    def test_ddp_child_reenters_rollout_entry_from_inherited_contract(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trainer = tmp_path / "prejepa.py"
        trainer.touch()
        monkeypatch.setenv(rollout_entry._TRAINER_ENV, str(trainer))
        monkeypatch.setenv(rollout_entry._FAMILY_ENV, "prejepa")
        monkeypatch.setenv(rollout_entry._STEPS_ENV, "3")

        args, hydra_args = rollout_entry._parse_args(
            ["--config-name=prejepa", "wm.num_preds=1", "n_steps=6"]
        )

        assert args.trainer_script == trainer
        assert args.family == "prejepa"
        assert args.rollout_steps == 3
        assert args.rollout_context == "sliding"
        assert hydra_args == [
            "--config-name=prejepa",
            "wm.num_preds=1",
            "n_steps=6",
        ]

    def test_rank_zero_keeps_wrapper_as_argv_zero_for_lightning_ddp(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trainer_path = tmp_path / "prejepa.py"
        trainer_path.touch()
        observed: dict[str, object] = {}

        class Trainer:
            @staticmethod
            def run() -> None:
                observed["argv"] = list(sys.argv)
                observed["trainer"] = os.environ[rollout_entry._TRAINER_ENV]
                observed["family"] = os.environ[rollout_entry._FAMILY_ENV]
                observed["steps"] = os.environ[rollout_entry._STEPS_ENV]
                observed["context"] = os.environ[rollout_entry._CONTEXT_ENV]
                observed["hydra_main_module"] = os.environ[
                    rollout_entry._HYDRA_MAIN_MODULE_ENV
                ]

        monkeypatch.setattr(
            rollout_entry, "_load_trainer", lambda _path: Trainer
        )
        monkeypatch.setattr(
            rollout_entry, "install_rollout_forward", lambda *_a, **_k: None
        )
        monkeypatch.setattr(sys, "argv", ["pytest"])

        assert rollout_entry.main(
            [
                "--trainer-script",
                str(trainer_path),
                "--family",
                "prejepa",
                "--rollout-steps",
                "3",
                "--",
                "--config-name=prejepa",
                "n_steps=6",
            ]
        ) == 0

        argv = observed["argv"]
        assert isinstance(argv, list)
        assert argv[0] == str(Path(rollout_entry.__file__).resolve())
        assert argv[1:] == ["--config-name=prejepa", "n_steps=6"]
        assert observed == {
            "argv": argv,
            "trainer": str(trainer_path.resolve()),
            "family": "prejepa",
            "steps": "3",
            "context": "sliding",
            "hydra_main_module": "__main__",
        }

    def test_default_one_step_command_keeps_the_direct_upstream_entry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Steps=1 must not import or route through the rollout overlay."""

        stablewm_repo = _fake_stablewm_repo(tmp_path / "stablewm")
        dataset_uri = "contextworld://v1/default-one-step"
        target = _target(dataset_uri)
        monkeypatch.setattr(launcher, "validate_stablepretraining_version", lambda: "0.1.8")
        monkeypatch.setattr(launcher, "resolve_target", lambda *_: target)
        monkeypatch.setattr(launcher, "validate_training_dataset_schema", lambda **_: None)
        monkeypatch.setattr(
            launcher,
            "describe_contextworld_dataset",
            lambda _: _runtime_identity(method="native"),
        )
        monkeypatch.setattr(
            launcher,
            "_training_identity_document",
            lambda **_: {"identity_sha256": "a" * 64, "identity": {}},
        )
        monkeypatch.setattr(launcher, "validate_resume", lambda *_, **__: None)

        assert launcher.main(
            [
                "--component", "motion_damping",
                "--family", "lewm",
                "--stablewm-repo", str(stablewm_repo),
                "--checkpoint-root", str(tmp_path / "checkpoints"),
                "--seeds", "3072",
                "--print-command",
            ]
        ) == 0

        output = capsys.readouterr().out
        assert "rollout_steps=1" in output
        assert str(launcher.ROLLOUT_ENTRY_SCRIPT) not in output
        assert str(stablewm_repo / "scripts/train/lewm.py") in output
        # It remains the untouched H+1 upstream sequence path.
        assert "data.dataset.num_steps=" not in output

    @pytest.mark.parametrize("method", ("native", "coja_v1"))
    def test_multistep_command_uses_the_overlay_entry_for_native_and_coja(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        method: str,
    ) -> None:
        stablewm_repo = _fake_stablewm_repo(tmp_path / "stablewm")
        contract = launcher.load_profile_contract()
        if method == "coja_v1":
            _enable_coja_config(stablewm_repo, family="lewm", contract=contract)
        dataset_uri = "contextworld://v1/multistep-command"
        target = _target(dataset_uri)
        monkeypatch.setattr(launcher, "validate_stablepretraining_version", lambda: "0.1.8")
        monkeypatch.setattr(launcher, "resolve_target", lambda *_: target)
        monkeypatch.setattr(launcher, "validate_training_dataset_schema", lambda **_: None)
        monkeypatch.setattr(
            launcher,
            "describe_contextworld_dataset",
            lambda _: _runtime_identity(method=method),
        )
        monkeypatch.setattr(
            launcher,
            "_training_identity_document",
            lambda **_: {"identity_sha256": "b" * 64, "identity": {}},
        )
        monkeypatch.setattr(launcher, "validate_resume", lambda *_, **__: None)

        assert launcher.main(
            [
                "--component", "motion_damping",
                "--family", "lewm",
                "--method", method,
                "--rollout-steps", "3",
                "--stablewm-repo", str(stablewm_repo),
                "--checkpoint-root", str(tmp_path / "checkpoints"),
                "--seeds", "3072",
                "--print-command",
            ]
        ) == 0

        output = capsys.readouterr().out
        assert str(launcher.ROLLOUT_ENTRY_SCRIPT) in output
        assert "--rollout-steps 3" in output
        assert "wm.num_preds=1" in output
        assert "data.dataset.num_steps=6" in output
        suffix = "_coja_v1" if method == "coja_v1" else ""
        assert f"motion_damping_lewm_joint_scratch_v1{suffix}_rollout3_s3072" in output

    @pytest.mark.parametrize("method", ("native", "coja_v1"))
    def test_motion_rollout_routes_registered_data_for_native_and_coja(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
    ) -> None:
        stablewm_repo = _fake_stablewm_repo(tmp_path / "stablewm")
        args = _component_args(stablewm_repo, tmp_path, method=method)
        contract = launcher.load_profile_contract()
        captured: dict[str, object] = {}

        def build_uri(*_args: object, **kwargs: object) -> str:
            captured.update(kwargs)
            return "contextworld://v1/motion-rollout"

        monkeypatch.setattr(launcher, "build_contextworld_dataset_uri", build_uri)
        monkeypatch.setattr(
            launcher,
            "describe_contextworld_dataset",
            lambda _: _runtime_identity(method=method),
        )

        target = launcher.resolve_target(args, contract)

        assert target.history_size == 3
        assert captured["component"] == "motion_damping"
        assert captured["rollout_steps"] == 3
        assert str(captured["rollout_artifact_root"]).endswith(
            "artifacts/synthesis/pusht_motion_damping_h3_rollout_k10_v1"
        )
        assert captured["conditional_joint_method"] == (
            "coja_v1" if method == "coja_v1" else None
        )

    @pytest.mark.parametrize("family", ("lewm", "viswm", "pldm", "prejepa"))
    @pytest.mark.parametrize("method", ("native", "coja_v1"))
    def test_rollout_keeps_target_offset_one_and_sets_history_plus_steps(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        family: str,
        method: str,
    ) -> None:
        stablewm_repo = _fake_stablewm_repo(tmp_path / "stablewm")
        args = _component_args(stablewm_repo, tmp_path, family=family, method=method)
        contract = launcher.load_profile_contract()

        # The public profile itself is the fail-closed registration check.
        assert launcher._rollout_recipe(args, contract) is not None
        if method == "coja_v1":
            # Make the local fake config expose the upstream optional loss block.
            _enable_coja_config(stablewm_repo, family=family, contract=contract)

        # COJA checks the URI's relation contract while rendering its optional
        # loss keys.  This focused test exercises launcher dialects, not the
        # multi-gigabyte bundle reader, so provide the already-validated view.
        monkeypatch.setattr(
            launcher,
            "describe_contextworld_dataset",
            lambda _: _runtime_identity(method=method),
        )

        entries = launcher.build_overrides(
            args,
            contract,
            _target("contextworld://v1/motion-rollout"),
            run_name="motion_damping_rollout3",
            seed=3072,
            stablewm_repo=stablewm_repo,
        )
        pairs = {
            key: value
            for item in entries
            for key, separator, value in [item.partition("=")]
            if separator
        }
        sequence_key = (
            "n_steps" if family == "prejepa" else "data.dataset.num_steps"
        )
        num_pred_key = "wm.num_preds"

        assert pairs[num_pred_key] == "1"
        assert pairs[sequence_key] == "6"  # H=3 plus K=3, not num_preds=3.
        if method == "coja_v1":
            assert pairs["loss.conditional_joint.enabled"] == "true"

    @pytest.mark.parametrize(
        "component,track,expected",
        [
            ("action_strength", "joint_scratch_v1", "no query-anchored"),
            ("motion_damping", "historical_release", "unavailable on training track"),
        ],
    )
    def test_rollout_is_registered_only_for_motion_joint_scratch(
        self,
        tmp_path: Path,
        component: str,
        track: str,
        expected: str,
    ) -> None:
        stablewm_repo = _fake_stablewm_repo(tmp_path / "stablewm")
        args = _component_args(
            stablewm_repo, tmp_path, component=component, training_track=track
        )

        with pytest.raises(SystemExit, match=expected):
            launcher._rollout_recipe(args, launcher.load_profile_contract())

    def test_rollout_rejects_num_preds_as_an_alias(
        self, tmp_path: Path
    ) -> None:
        stablewm_repo = _fake_stablewm_repo(tmp_path / "stablewm")
        args = _component_args(
            stablewm_repo, tmp_path, extra=("--num-preds", "3")
        )

        with pytest.raises(SystemExit, match="target-offset"):
            launcher._rollout_recipe(args, launcher.load_profile_contract())

    def test_expanding_context_is_typed_and_allocates_predictor_capacity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stablewm_repo = _fake_stablewm_repo(tmp_path / "stablewm")
        args = _component_args(
            stablewm_repo,
            tmp_path,
            family="lewm",
            extra=("--rollout-context", "expanding"),
        )
        contract = launcher.load_profile_contract()
        monkeypatch.setattr(
            launcher,
            "describe_contextworld_dataset",
            lambda _: _runtime_identity(method="native"),
        )

        entries = launcher.build_overrides(
            args,
            contract,
            _target("contextworld://v1/motion-rollout"),
            run_name="motion_damping_rollout3_expanding",
            seed=3072,
            stablewm_repo=stablewm_repo,
        )

        assert "model.predictor.num_frames=5" in entries
        name = launcher._run_name(
            args,
            _target("contextworld://v1/motion-rollout"),
            3072,
            (3072,),
        )
        assert name == (
            "motion_damping_lewm_joint_scratch_v1_"
            "rollout3_expanding_s3072"
        )


class _IdentityPredictor:
    """A no-parameter predictor makes the rollout gradient path observable."""

    def __init__(self) -> None:
        self.contexts: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []

    def predict(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.contexts.append(context)
        self.actions.append(action)
        # The last prediction is exactly the latest context state.  Thus the
        # final step can only differentiate through an earlier prediction.
        return context


class TestAutoregressiveLoss:
    @pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
    def test_every_step_is_scored_and_later_loss_backpropagates_through_earlier_prediction(
        self,
    ) -> None:
        history = 2
        steps = 3
        # The final historical state is 1.  The identity predictor therefore
        # emits 1 at every rollout step, while targets give losses 1, 1, and 9.
        embeddings = torch.tensor(
            [[[0.0], [1.0], [0.0], [2.0], [4.0]]], requires_grad=True
        )
        actions = torch.arange(5.0).reshape(1, 5, 1)
        model = _IdentityPredictor()

        loss, predictions, step_losses = stablewm_rollout._latent_rollout(
            model,
            embeddings,
            actions,
            history_size=history,
            rollout_steps=steps,
            detach_targets=True,
        )

        assert len(predictions) == steps
        assert len(step_losses) == steps
        assert torch.allclose(
            torch.stack(step_losses), torch.tensor([1.0, 1.0, 9.0])
        )
        assert torch.allclose(loss, torch.tensor(11.0 / 3.0))
        # The context/action window shifts after each predicted state; this is
        # not the upstream target-offset ``num_preds`` behavior.
        assert [context.shape[1] for context in model.contexts] == [2, 2, 2]
        assert [action[:, 0, 0].item() for action in model.actions] == [0.0, 1.0, 2.0]
        assert [action[:, -1, 0].item() for action in model.actions] == [1.0, 2.0, 3.0]

        # Backprop only the final loss.  It has no parameter of its own; the
        # nonzero gradient on the last historical state proves step 3 reaches
        # it through step 2 -> step 1 predictions without a detach.
        step_losses[-1].backward()
        assert embeddings.grad is not None
        assert embeddings.grad[0, history - 1, 0].abs().item() > 0.0

    def test_expanding_context_keeps_all_observed_history(self) -> None:
        history = 2
        steps = 3
        embeddings = torch.tensor(
            [[[0.0], [1.0], [0.0], [2.0], [4.0]]], requires_grad=True
        )
        actions = torch.arange(5.0).reshape(1, 5, 1)
        model = _IdentityPredictor()

        _loss, _predictions, step_losses = stablewm_rollout._latent_rollout(
            model,
            embeddings,
            actions,
            history_size=history,
            rollout_steps=steps,
            detach_targets=True,
            context_mode="expanding",
        )

        assert [context.shape[1] for context in model.contexts] == [2, 3, 4]
        assert [action.shape[1] for action in model.actions] == [2, 3, 4]
        # The first observed state remains the first token at every horizon.
        assert [context[0, 0, 0].item() for context in model.contexts] == [
            0.0,
            0.0,
            0.0,
        ]
        step_losses[-1].backward()
        assert embeddings.grad is not None
        assert embeddings.grad[0, history - 1, 0].abs().item() > 0.0

    def test_forward_builder_propagates_expanding_context(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.context_lengths: list[int] = []

            def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                return {"emb": batch["emb"], "act_emb": batch["act_emb"]}

            def predict(
                self, context: torch.Tensor, _actions: torch.Tensor
            ) -> torch.Tensor:
                self.context_lengths.append(int(context.shape[1]))
                return context

        class Module:
            def __init__(self) -> None:
                self.model = Model()

            def log_dict(self, *_args: object, **_kwargs: object) -> None:
                return None

        def native_forward(
            owner: Module,
            values: dict[str, torch.Tensor],
            _stage: str,
            _cfg: object,
            **_: object,
        ) -> dict[str, torch.Tensor]:
            encoded = owner.model.encode(values)
            pred = owner.model.predict(
                encoded["emb"][:, :2], encoded["act_emb"][:, :2]
            )
            native = torch.nn.functional.mse_loss(
                pred, encoded["emb"][:, 1:]
            )
            return {"loss": native, "pred_loss": native}

        module = Module()
        cfg = SimpleNamespace(wm=SimpleNamespace(history_size=2))
        batch = {
            "pixels": torch.zeros((1, 5, 1)),
            "action": torch.zeros((1, 5, 1)),
            "emb": torch.tensor([[[0.0], [1.0], [0.0], [2.0], [4.0]]]),
            "act_emb": torch.zeros((1, 5, 1)),
        }
        wrapped = stablewm_rollout._build_lewm_like_forward(
            native_forward,
            rollout_steps=3,
            context_mode="expanding",
        )

        wrapped(module, batch, "fit", cfg)

        # First length 2 is the unchanged native prefix; the next three are
        # the autoregressive diagnostic contexts.
        assert module.model.context_lengths == [2, 2, 3, 4]

    def test_overlay_replaces_only_the_native_prediction_term(self) -> None:
        """Representation losses remain native while prediction becomes K-step."""

        class Model:
            def __init__(self) -> None:
                self.prefix_lengths: list[int] = []

            def encode(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                self.prefix_lengths.append(int(batch["pixels"].shape[1]))
                return {
                    "emb": batch["emb"],
                    "act_emb": batch["act_emb"],
                }

            def predict(
                self, context: torch.Tensor, _actions: torch.Tensor
            ) -> torch.Tensor:
                return context

        class Module:
            def __init__(self) -> None:
                self.model = Model()
                self.logged: list[dict[str, torch.Tensor]] = []

            def log_dict(self, values: dict[str, torch.Tensor], **_: object) -> None:
                self.logged.append(values)

        module = Module()
        cfg = SimpleNamespace(wm=SimpleNamespace(history_size=2))
        native_prefix_lengths: list[int] = []
        batch = {
            "pixels": torch.zeros((1, 5, 1)),
            "action": torch.zeros((1, 5, 1)),
            "emb": torch.tensor([[[0.0], [1.0], [0.0], [2.0], [4.0]]]),
            "act_emb": torch.zeros((1, 5, 1)),
        }

        def native_forward(
            owner: Module,
            values: dict[str, torch.Tensor],
            _stage: str,
            _cfg: object,
            **_: object,
        ) -> dict[str, torch.Tensor]:
            native_prefix_lengths.append(int(values["pixels"].shape[1]))
            encoded = owner.model.encode(values)
            prediction = owner.model.predict(
                encoded["emb"][:, :2], encoded["act_emb"][:, :2]
            )
            native = torch.nn.functional.mse_loss(prediction, encoded["emb"][:, 1:])
            representation = torch.tensor(7.0)
            return {
                "loss": native + representation,
                "pred_loss": native,
                "representation_loss": representation,
            }

        wrapped = stablewm_rollout._build_lewm_like_forward(
            native_forward, rollout_steps=3
        )
        output = wrapped(module, batch, "fit", cfg)

        # The full encoding is computed once; the native objective sees only
        # its ordinary H+1 prefix through the cached encoder.
        assert module.model.prefix_lengths == [5]
        assert native_prefix_lengths == [3]
        assert torch.allclose(output["rollout_loss"], torch.tensor(11.0 / 3.0))
        assert torch.allclose(output["pred_loss"], output["rollout_loss"])
        assert torch.allclose(output["representation_loss"], torch.tensor(7.0))
        assert torch.allclose(output["loss"], torch.tensor(7.0 + 11.0 / 3.0))
        assert set(output) >= {
            "rollout_step_1_loss",
            "rollout_step_2_loss",
            "rollout_step_3_loss",
        }
        assert len(module.logged) == 1
