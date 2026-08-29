"""The cloud router reaches every task without inventing an interface.

The nine tasks were built over months and do not agree on how to say
"family", "seed" or "mode". A router is only worth having if it gets each of
those right, so the divergences get one test apiece -- each of them is a
command that would otherwise fail late, on a GPU, after the data loaded.

Nothing here runs training. These tests assert the command that *would* run.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import h5py
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cloud_train as router  # noqa: E402


def _args(**keywords: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "task": "speed",
        "env": None,
        "family": "lewm",
        "seed": 3072,
        "seeds": (3072,),
        "mode": "preflight",
        "stage": "paired",
        "variant": None,
        "dataset": None,
        "run_name": None,
        "output": None,
        "batch_size": None,
        "print_command": True,
        "extra": [],
    }
    defaults.update(keywords)
    return argparse.Namespace(**defaults)


def _command(**keywords: object) -> str:
    return " ".join(router.build_plan(_args(**keywords)).command)


class TestEveryTaskIsReachable:
    @pytest.mark.parametrize("task", router.TASKS)
    @pytest.mark.parametrize("family", ["lewm", "pldm"])
    def test_the_baseline_families_route(self, task: str, family: str) -> None:
        plan = router.build_plan(
            _args(task=task, family=family, dataset="d")
        )

        assert plan.command

    @pytest.mark.parametrize("task", router.TASKS)
    def test_prejepa_routes_uniformly(self, task: str) -> None:
        """The new family needs no per-task entry; that is the point of it."""

        plan = router.build_plan(
            _args(task=task, family="prejepa", dataset="d")
        )

        assert "run_stablewm_train.py" in " ".join(plan.command)
        assert f"--component {task}" in " ".join(plan.command)
        assert "--seeds 3072" in " ".join(plan.command)

    @pytest.mark.parametrize("task", router.TASKS)
    def test_every_launcher_it_names_exists(self, task: str) -> None:
        """A routing table that points at a moved file is worse than none."""

        for family in ("lewm", "pldm"):
            command = router.build_plan(
                _args(task=task, family=family, dataset="d")
            ).command
            script = Path(command[1] if command[0] == "bash" else command[1])

            assert script.is_file(), f"{task}/{family} -> missing {script}"


class TestEnvironmentParsing:
    def test_comma_separated_seeds_expand_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CW_SEEDS", "3072, 3073,3074")
        monkeypatch.setenv("CW_TASK", "original")
        monkeypatch.setenv("CW_ENV", "tworoom")

        args = router.parse_args([])

        assert args.seeds == (3072, 3073, 3074)

    @pytest.mark.parametrize("legacy", ["CW_SEED", "CW_ALL_SEEDS"])
    def test_legacy_seed_variables_fail_with_a_migration_message(
        self,
        legacy: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv(legacy, "3072")

        with pytest.raises(SystemExit):
            router.parse_args(["--task", "speed"])
        assert "CW_SEEDS" in capsys.readouterr().err

    def test_invalid_boolean_fails_before_launch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CW_PRINT_ONLY", "sometimes")

        with pytest.raises(SystemExit, match="CW_PRINT_ONLY"):
            router.parse_args(["--task", "speed"])

class TestTheDivergencesThatWouldFailLate:
    def test_door_uses_mixture_names_not_family_names(self) -> None:
        """``run_h3_hidden_passage_train.sh`` rejects "lewm" outright.

        Its variants are named after the data mixture. The release config's
        ``shell_variant`` fields are the source of this mapping.
        """

        assert "fixed-mixed" in _command(task="door", family="lewm")
        assert "pldm-mixed" in _command(task="door", family="pldm")
        assert " lewm " not in _command(task="door", family="lewm")

    def test_the_door_variants_match_the_frozen_release(self) -> None:
        """Pinned against the config rather than restated, so a change there
        fails here instead of silently retraining a different mixture."""

        release = yaml.safe_load(
            (ROOT / "configs/benchmark/tworoom_door_icl_release_v1.yaml")
            .read_text(encoding="utf-8")
        )
        declared = set()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "shell_variant":
                        declared.add(value)
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(release)

        assert declared, "release declares no shell_variant; guard is vacuous"
        assert set(router.DOOR_VARIANT.values()) == declared

    def test_speed_pldm_is_a_different_program(self) -> None:
        """Speed's two families are not two modes of one launcher."""

        lewm = _command(task="speed", family="lewm")
        pldm = _command(task="speed", family="pldm")

        assert "run_h3_speed_isolated_train.sh" in lewm
        assert "run_pldm_reference_completion.py" in pldm
        assert "--component speed" in pldm

    def test_speed_lewm_passes_seed_through_the_environment(self) -> None:
        """That launcher reads TRAINING_SEED; there is no seed flag to pass."""

        plan = router.build_plan(_args(task="speed", family="lewm", seed=5120))

        assert plan.env["TRAINING_SEED"] == "5120"
        assert "5120" not in " ".join(plan.command)

    def test_action_delay_passes_family_and_seed_positionally(self) -> None:
        """``FAMILY="${1:?}"`` and ``TRAINING_SEED="${2:-3072}"``."""

        command = router.build_plan(
            _args(task="action_delay", family="pldm", seed=4096)
        ).command

        assert command[-2:] == ["pldm", "4096"]

    def test_action_delay_has_two_ordered_stages(self) -> None:
        """The curriculum stage verifies stage one's checkpoint hash."""

        paired = _command(task="action_delay", stage="paired")
        curriculum = _command(task="action_delay", stage="curriculum")

        assert "paired" in paired and "curriculum" not in paired
        assert "curriculum" in curriculum

    def test_action_strength_lewm_needs_its_recipe_string(self) -> None:
        """``--variants`` is validated against a table; a family name fails."""

        command = _command(task="action_strength", family="lewm")

        assert router.ACTION_STRENGTH_VARIANT in command

    def test_the_action_strength_recipe_matches_the_frozen_release(
        self,
    ) -> None:
        release = yaml.safe_load(
            (ROOT / "configs/benchmark/pusht_action_strength_icl_release_v1.yaml")
            .read_text(encoding="utf-8")
        )
        recipe = release["training"]["recipes"]["reference_method"]

        assert router.ACTION_STRENGTH_VARIANT == recipe["runner_variant"]

    def test_the_five_shared_engine_tasks_use_flags(self) -> None:
        """One interface for five tasks -- the only group that agrees."""

        for task in router.SHARED_ENGINE:
            command = _command(task=task, family="pldm", seed=9)

            assert "--model pldm" in command
            assert "--seed 9" in command
            assert "--output" in command


class TestItRoutesRatherThanDecides:
    def test_it_edits_no_pinned_launcher(self) -> None:
        """Two targets are byte-pinned by frozen release configs.

        Routing above them is the reason this file exists rather than a new
        mode inside them.
        """

        historical = {
            "contextworld_icl_suite_v1.yaml",
            "contextworld_icl_suite_v2.yaml",
            "contextworld_icl_suite_v2_recovery_v2.yaml",
        }
        pinned: dict[str, set[str]] = {}
        for config in sorted((ROOT / "configs/benchmark").glob("*.yaml")):
            if config.name in historical:
                continue
            payload = yaml.safe_load(config.read_text(encoding="utf-8"))

            def walk(node: object) -> None:
                if isinstance(node, dict):
                    for key, value in node.items():
                        if key == "source_sha256" and isinstance(value, dict):
                            for path, digest in value.items():
                                if str(path).startswith("scripts/"):
                                    pinned.setdefault(
                                        str(path), set()
                                    ).add(digest)
                        walk(value)
                elif isinstance(node, list):
                    for value in node:
                        walk(value)

            walk(payload)

        assert pinned, "no script is pinned; this guard would be vacuous"
        for path, digests in pinned.items():
            live = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()

            assert digests == {live}, f"{path} no longer matches its pins"

    def test_it_does_not_validate_the_launcher_s_own_vocabulary(self) -> None:
        """Modes differ per launcher and change over time. Guessing here
        would make valid runs unreachable for no gain."""

        assert "formal-resume" in _command(task="door", mode="formal-resume")

    def test_trailing_arguments_are_forwarded_untouched(self) -> None:
        """The router must never be the reason something is unreachable."""

        args = _args(task="contact_friction", extra=["--", "--dry-run"])
        plan = router.build_plan(args)
        command = [*plan.command, *[x for x in args.extra if x != "--"]]

        assert command[-1] == "--dry-run"

    def test_prejepa_defaults_to_the_baseline_batch_size(self) -> None:
        """prejepa.yaml ships 32; lewm and pldm both ship 128."""

        plan = router.build_plan(
            _args(task="speed", family="prejepa", dataset="d")
        )

        assert "--batch-size" in plan.command
        index = plan.command.index("--batch-size")
        assert plan.command[index + 1] == str(router.BASELINE_BATCH_SIZE)
        assert "128" in plan.note

    def test_an_explicit_batch_size_wins(self) -> None:
        plan = router.build_plan(
            _args(task="speed", family="prejepa", dataset="d", batch_size=64)
        )
        index = plan.command.index("--batch-size")

        assert plan.command[index + 1] == "64"


class TestTheOriginalBaselineRegime:
    """`CW_TASK=original` reaches the four unmodified task datasets.

    This is a different regime from the nine ICL capabilities: same families,
    different data, and the seeds are what make a baseline column reportable
    as mean +/- std.
    """

    @pytest.mark.parametrize("env", router.ORIGINAL_ENVIRONMENTS)
    @pytest.mark.parametrize("family", ["lewm", "pldm", "prejepa"])
    def test_every_environment_and_family_routes(
        self, env: str, family: str
    ) -> None:
        plan = router.build_plan(
            _args(task="original", env=env, family=family)
        )
        command = " ".join(plan.command)

        assert "run_stablewm_train.py" in command
        assert f"--original-env {env}" in command
        assert f"--family {family}" in command

    def test_it_does_not_need_a_dataset_for_prejepa(self) -> None:
        """The benchmark route requires CW_DATASET; here the environment
        determines the data, so requiring it too would be noise."""

        plan = router.build_plan(
            _args(task="original", env="cube", family="prejepa")
        )

        assert plan.command

    def test_prejepa_original_defaults_to_the_comparable_batch_size(self) -> None:
        plan = router.build_plan(
            _args(task="original", env="cube", family="prejepa")
        )
        index = plan.command.index("--batch-size")

        assert plan.command[index + 1] == str(router.BASELINE_BATCH_SIZE)
        assert "comparability" in plan.note

    def test_explicit_original_batch_size_wins(self) -> None:
        plan = router.build_plan(
            _args(
                task="original",
                env="cube",
                family="prejepa",
                batch_size=64,
            )
        )
        index = plan.command.index("--batch-size")

        assert plan.command[index + 1] == "64"

    def test_an_explicit_original_dataset_reaches_the_launcher(self) -> None:
        plan = router.build_plan(
            _args(
                task="original",
                env="tworoom",
                family="prejepa",
                dataset="/datasets/tworoom.h5",
            )
        )

        index = plan.command.index("--dataset")

        assert plan.command[index + 1] == "/datasets/tworoom.h5"

    def test_original_forwards_one_resolved_seed(self) -> None:
        plan = router.build_plan(
            _args(task="original", env="pusht", family="lewm", seed=3074)
        )
        command = " ".join(plan.command)

        assert "--seeds 3074" in command

    def test_multiple_seeds_create_sequential_isolated_runs(self) -> None:
        args = _args(
            task="contact_friction",
            family="pldm",
            seeds=(3072, 3073),
            run_name="formal",
            output="/runs",
        )

        runs = router._seed_runs(args)

        assert [run.seed for run in runs] == [3072, 3073]
        assert [run.run_name for run in runs] == ["formal_s3072", "formal_s3073"]
        assert [run.output for run in runs] == [
            "/runs/contact_friction_pldm_s3072",
            "/runs/contact_friction_pldm_s3073",
        ]

    def test_a_missing_environment_fails_early(self) -> None:
        with pytest.raises(SystemExit, match="CW_ENV"):
            router.build_plan(_args(task="original", env=None))

    def test_an_ICL_task_name_is_not_an_environment(self) -> None:
        """'speed' is a capability, not one of the four base environments."""

        with pytest.raises(SystemExit, match="CW_ENV"):
            router.build_plan(_args(task="original", env="speed"))

    def test_the_launcher_it_names_exists(self) -> None:
        plan = router.build_plan(
            _args(task="original", env="tworoom", family="lewm")
        )

        assert Path(plan.command[1]).is_file()


class TestFailuresArriveEarly:
    def test_an_unknown_task_is_rejected(self) -> None:
        with pytest.raises(SystemExit, match="Unknown task"):
            router.build_plan(_args(task="nope"))

    def test_an_unknown_family_is_rejected(self) -> None:
        with pytest.raises(SystemExit, match="Unknown family"):
            router.build_plan(_args(family="dino"))

    def test_prejepa_without_data_fails_before_the_gpu(self) -> None:
        with pytest.raises(SystemExit, match="dataset"):
            router.build_plan(_args(family="prejepa", dataset=None))


class TestTheCloudContract:
    def test_the_shell_entry_uses_one_public_python_entry(self) -> None:
        """The job template can only call `bash <path>`."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert 'exec "$PYTHON_BIN" "$ROOT/scripts/run_stablewm_train.py"' in text
        assert text.startswith("#!/usr/bin/env bash")

    def test_the_shell_entry_locates_the_repo_itself(self) -> None:
        """work_dir varies; the script must not assume a fixed checkout."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert "BASH_SOURCE" in text

    def test_it_does_not_hardcode_a_single_data_root(self) -> None:
        """The cloud mounts /opt/huawei/dataset/ag_data; the dev box adds an
        `explorer-env` segment. Hardcoding either breaks the other."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert "/opt/huawei/dataset/ag_data" in text
        assert "/opt/huawei/explorer-env/dataset/ag_data" in text
        assert "CW_DATA_ROOT" in text

    def test_it_exports_the_paths_the_launcher_reads(self) -> None:
        """The GUI sets only CW_* variables; the rest must be derived and
        exported here, because there is nowhere else to set them."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        for exported in (
            "CONTEXTWORLD_STABLE_WORLDMODEL_REPO",
            "CONTEXTWORLD_DATASET_ROOT",
            "CONTEXTWORLD_BENCHMARK_ROOT",
            "HF_HUB_CACHE",
        ):
            assert f"export {exported}" in text or (
                f": \"${{{exported}" in text
            ), exported

    def test_it_separates_original_bundle_and_internal_data_roots(
        self,
    ) -> None:
        """Original LeWM data and ContextWorld's own outputs are different
        trees. One variable covering both would silently train a baseline on
        synthesized data, or a capability on original data."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert "CONTEXTWORLD_DATASET_ROOT" in text
        assert "CONTEXTWORLD_BENCHMARK_ROOT" in text
        assert "CONTEXTWORLD_ARTIFACT_ROOT" in text
        assert "export CONTEXTWORLD_BENCHMARK_ROOT" in text
        assert "export CONTEXTWORLD_ARTIFACT_ROOT" in text

    def test_the_artifact_root_is_not_left_to_inference(self) -> None:
        """contextworld.paths falls back to repo.parents[1]/data/... , which
        is wrong whenever work_dir is not two levels below the data root --
        exactly the cloud's layout."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert "benchmark artifact root does not exist" in text

    def test_the_three_roots_are_reported_distinctly(self) -> None:
        """An operator reading the log should see which is which."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert "original data" in text
        assert "benchmark bundle" in text
        assert "historical artifact archive" in text

    def test_an_explicit_data_root_is_honoured(self) -> None:
        """A user who sets CW_DATA_ROOT is trusted; detection is skipped."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert 'if [ -z "${CW_DATA_ROOT:-}" ]' in text

    def test_explicit_leaf_paths_do_not_require_the_umbrella_root(self) -> None:
        """CW_DATA_ROOT is a fallback, not a prerequisite."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert "This umbrella root is only a" in text
        assert "fallback" in text

    def test_checkpoint_root_controls_stablewm_checkpoints(self) -> None:
        """Hydra's run directory is not where upstream saves weights."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert "CW_CHECKPOINT_ROOT" in text
        assert 'STABLEWM_HOME="$CW_CHECKPOINT_ROOT"' in text
        assert 'SPT_CACHE_DIR="$STABLEWM_HOME"' in text
        assert "export SPT_CACHE_DIR" in text

    def test_an_actual_run_freezes_the_stablewm_revision(
        self, tmp_path: Path
    ) -> None:
        """A live checkout update cannot split training and post-evaluation."""

        stablewm = tmp_path / "stable-worldmodel"
        (stablewm / "stable_worldmodel").mkdir(parents=True)
        (stablewm / "scripts/train").mkdir(parents=True)
        (stablewm / "scripts/plan").mkdir(parents=True)
        source = stablewm / "stable_worldmodel/__init__.py"
        source.write_text("revision = 1\n", encoding="utf-8")
        (stablewm / "scripts/train/prejepa.py").write_text(
            "# trainer\n", encoding="utf-8"
        )
        (stablewm / "scripts/plan/eval_wm.py").write_text(
            "# evaluator\n", encoding="utf-8"
        )

        def git(*arguments: str) -> str:
            return subprocess.run(
                ["git", "-C", str(stablewm), *arguments],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        git("init", "-q")
        git("config", "user.email", "contextworld@example.invalid")
        git("config", "user.name", "ContextWorld test")
        git("add", ".")
        git("commit", "-q", "-m", "initial")
        frozen_ref = git("rev-parse", "HEAD")

        dataset = tmp_path / "training.lance"
        dataset.touch()
        checkpoint_root = tmp_path / "checkpoints"
        probe = tmp_path / "python-probe"
        probe.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'EFFECTIVE=%s\\n' \"$CONTEXTWORLD_STABLE_WORLDMODEL_REPO\"\n"
            "printf 'REF=%s\\n' \"$CW_STABLEWM_REF\"\n"
            "for argument in \"$@\"; do printf 'ARG=%s\\n' \"$argument\"; done\n",
            encoding="utf-8",
        )
        probe.chmod(0o755)
        no_git_bin = tmp_path / "no-git-bin"
        no_git_bin.mkdir()
        (no_git_bin / "git").write_text(
            "#!/usr/bin/env bash\necho 'git must not be called' >&2\nexit 99\n",
            encoding="utf-8",
        )
        (no_git_bin / "git").chmod(0o755)
        environment = {
            "PATH": f"{no_git_bin}:{os.environ['PATH']}",
            "HOME": str(tmp_path),
            "CW_TASK": "speed",
            "CW_FAMILY": "prejepa",
            "CW_DATASET": str(dataset),
            "CW_CHECKPOINT_ROOT": str(checkpoint_root),
            "CW_PRINT_ONLY": "0",
            "CONTEXTWORLD_STABLE_WORLDMODEL_REPO": str(stablewm),
            "PYTHON_BIN": str(probe),
        }
        bypass = tmp_path / "must-not-bypass-snapshot"

        first = subprocess.run(
            [
                "bash",
                str(SCRIPTS / "cloud_train.sh"),
                "--stablewm-repo",
                str(bypass),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        snapshot = (
            checkpoint_root
            / ".contextworld/stable-worldmodel"
            / frozen_ref
        )
        assert first.returncode == 0, first.stdout + first.stderr
        assert (snapshot / ".git/HEAD").read_text(
            encoding="utf-8"
        ).strip() == frozen_ref
        assert f"EFFECTIVE={snapshot}" in first.stdout
        assert f"REF={frozen_ref}" in first.stdout
        assert first.stdout.splitlines()[-2:] == [
            "ARG=--stablewm-repo",
            f"ARG={snapshot}",
        ]

        source.write_text("revision = 2\n", encoding="utf-8")
        git("add", ".")
        git("commit", "-q", "-m", "advance live checkout")
        environment["CW_STABLEWM_REF"] = frozen_ref
        second = subprocess.run(
            ["bash", str(SCRIPTS / "cloud_train.sh")],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert second.returncode == 0, second.stdout + second.stderr
        assert f"EFFECTIVE={snapshot}" in second.stdout
        assert (snapshot / "stable_worldmodel/__init__.py").read_text(
            encoding="utf-8"
        ) == "revision = 1\n"

        # A source-only cloud mount has no usable Git metadata at all. Its
        # runtime content fingerprint becomes the stable 40-digit ref.
        (stablewm / ".git").rename(stablewm / ".git-hidden")
        environment.pop("CW_STABLEWM_REF")
        source_only_root = tmp_path / "source-only-checkpoints"
        environment["CW_CHECKPOINT_ROOT"] = str(source_only_root)
        third = subprocess.run(
            ["bash", str(SCRIPTS / "cloud_train.sh")],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert third.returncode == 0, third.stdout + third.stderr
        source_only_ref = next(
            line.removeprefix("REF=")
            for line in third.stdout.splitlines()
            if line.startswith("REF=")
        )
        assert len(source_only_ref) == 40
        assert all(
            character in "0123456789abcdef" for character in source_only_ref
        )
        assert (
            source_only_root
            / ".contextworld/stable-worldmodel"
            / source_only_ref
            / ".git/HEAD"
        ).read_text(encoding="utf-8").strip() == source_only_ref

    def test_original_data_does_not_require_contextworld_artifacts(self) -> None:
        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert '[ "${CW_TASK:-}" != "original" ]' in text

    def test_explicit_original_paths_work_without_an_artifact_root(
        self, tmp_path: Path
    ) -> None:
        dataset = tmp_path / "tworoom.h5"
        with h5py.File(dataset, "w") as handle:
            handle.create_dataset("pixels", shape=(2, 8, 8, 3), dtype="uint8")
            handle.create_dataset("action", shape=(2, 2), dtype="float32")
            handle.create_dataset("proprio", shape=(2, 2), dtype="float32")
        stablewm = tmp_path / "stablewm"
        (stablewm / "scripts/train/config").mkdir(parents=True)
        (stablewm / "scripts/train/prejepa.py").write_text(
            "enabled = cfg.wandb.enabled\n", encoding="utf-8"
        )
        (stablewm / "scripts/train/config/prejepa.yaml").write_text(
            "trainer:\n  max_epochs: 10\n", encoding="utf-8"
        )
        benchmark_root = tmp_path / "ContextWorld-v1"
        benchmark_root.mkdir()
        for filename in ("task_registry.json", "manifest.jsonl", "manifest.sha256"):
            (benchmark_root / filename).write_text("{}\n", encoding="utf-8")
        checkpoint_root = tmp_path / "checkpoints"
        environment = dict(os.environ)
        for name in (
            "CONTEXTWORLD_DATASET_ROOT",
            "CONTEXTWORLD_ARTIFACT_ROOT",
            "STABLEWM_HOME",
            "SPT_CACHE_DIR",
        ):
            environment.pop(name, None)
        environment.update(
            {
                "CW_DATA_ROOT": str(tmp_path / "unused-umbrella"),
                "CW_TASK": "original",
                "CW_ENV": "tworoom",
                "CW_FAMILY": "prejepa",
                "CW_PRINT_ONLY": "1",
                "CW_POST_TRAIN_EVAL": "1",
                "CW_DATASET": str(dataset),
                "CONTEXTWORLD_BENCHMARK_ROOT": str(benchmark_root),
                "CW_CHECKPOINT_ROOT": f"{checkpoint_root}/",
                "STABLEWM_HOME": f"{checkpoint_root}/.",
                "SPT_CACHE_DIR": f"{checkpoint_root}//",
                "CONTEXTWORLD_STABLE_WORLDMODEL_REPO": str(stablewm),
                "PYTHON_BIN": sys.executable,
            }
        )

        completed = subprocess.run(
            ["bash", str(SCRIPTS / "cloud_train.sh")],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        assert f"original dataset={dataset}" in completed.stdout
        assert f"checkpoint root={checkpoint_root}" in completed.stdout
        assert f"spt cache={checkpoint_root}" in completed.stdout
        assert f"benchmark bundle={benchmark_root}" in completed.stdout
        assert "historical artifact archive=<not needed>" in completed.stdout
        assert f"dataset_name={dataset}" in completed.stdout
        assert f"--benchmark-root {benchmark_root}" in completed.stdout

    def test_one_dataset_root_selects_the_original_file_by_environment(
        self, tmp_path: Path
    ) -> None:
        dataset_root = tmp_path / "data" / "world_model"
        dataset = dataset_root / "quentinll" / "tworoom.h5"
        dataset.parent.mkdir(parents=True)
        with h5py.File(dataset, "w") as handle:
            handle.create_dataset("pixels", shape=(2, 8, 8, 3), dtype="uint8")
            handle.create_dataset("action", shape=(2, 2), dtype="float32")
            handle.create_dataset("proprio", shape=(2, 2), dtype="float32")
        stablewm = tmp_path / "stablewm"
        (stablewm / "scripts/train/config").mkdir(parents=True)
        (stablewm / "scripts/train/prejepa.py").write_text(
            "enabled = cfg.wandb.enabled\n", encoding="utf-8"
        )
        (stablewm / "scripts/train/config/prejepa.yaml").write_text(
            "trainer:\n  max_epochs: 10\n", encoding="utf-8"
        )
        checkpoint_root = tmp_path / "checkpoints"
        environment = dict(os.environ)
        for name in (
            "CW_DATASET",
            "CONTEXTWORLD_ARTIFACT_ROOT",
            "STABLEWM_HOME",
        ):
            environment.pop(name, None)
        environment.update(
            {
                "CW_DATA_ROOT": str(tmp_path / "unused-umbrella"),
                "CW_TASK": "original",
                "CW_ENV": "tworoom",
                "CW_FAMILY": "prejepa",
                "CW_PRINT_ONLY": "1",
                "CW_CHECKPOINT_ROOT": str(checkpoint_root),
                "CONTEXTWORLD_DATASET_ROOT": str(dataset_root),
                "CONTEXTWORLD_STABLE_WORLDMODEL_REPO": str(stablewm),
                "PYTHON_BIN": sys.executable,
            }
        )

        completed = subprocess.run(
            ["bash", str(SCRIPTS / "cloud_train.sh")],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        assert f"dataset_name={dataset}" in completed.stdout
        assert "subdir=tworoom_prejepa_original_s3072" in completed.stdout
        assert "route=stablewm-train" in completed.stdout

    def test_cloud_duplicate_parameter_argv_is_filtered_without_secret_leak(
        self, tmp_path: Path
    ) -> None:
        dataset_root = tmp_path / "data" / "world_model"
        dataset = dataset_root / "quentinll" / "tworoom.h5"
        dataset.parent.mkdir(parents=True)
        with h5py.File(dataset, "w") as handle:
            handle.create_dataset("pixels", shape=(2, 8, 8, 3), dtype="uint8")
            handle.create_dataset("action", shape=(2, 2), dtype="float32")
            handle.create_dataset("proprio", shape=(2, 2), dtype="float32")
        stablewm = tmp_path / "stablewm"
        (stablewm / "scripts/train/config").mkdir(parents=True)
        (stablewm / "scripts/train/prejepa.py").write_text(
            "enabled = cfg.wandb.enabled\n", encoding="utf-8"
        )
        (stablewm / "scripts/train/config/prejepa.yaml").write_text(
            "trainer:\n  max_epochs: 10\n", encoding="utf-8"
        )
        environment = dict(os.environ)
        for name in ("CW_DATASET", "CONTEXTWORLD_ARTIFACT_ROOT", "STABLEWM_HOME"):
            environment.pop(name, None)
        environment.update(
            {
                "CW_TASK": "original",
                "CW_ENV": "tworoom",
                "CW_FAMILY": "prejepa",
                "CW_PRINT_ONLY": "1",
                "CW_CHECKPOINT_ROOT": str(tmp_path / "checkpoints"),
                "CONTEXTWORLD_DATASET_ROOT": str(dataset_root),
                "CONTEXTWORLD_STABLE_WORLDMODEL_REPO": str(stablewm),
                "PYTHON_BIN": sys.executable,
            }
        )
        secret = "must-not-appear-in-cloud-log"

        completed = subprocess.run(
            [
                "bash",
                str(SCRIPTS / "cloud_train.sh"),
                "--CW_FAMILY",
                "prejepa",
                "--work_dir",
                str(ROOT),
                "--CW_ENV",
                "tworoom",
                "--SWANLAB_API_KEY",
                secret,
                "--run_shell_script",
                "scripts/cloud_train.sh",
                "--CW_PRINT_ONLY",
                "0",
                "--np",
                "8",
                "--max-epochs",
                "2",
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        output = completed.stdout + completed.stderr
        assert completed.returncode == 0, output
        assert "ignored duplicate platform arguments=7" in output
        assert "trainer.max_epochs=2" in output
        assert secret not in output
        assert "unrecognized arguments" not in output

    def test_historical_release_uses_the_same_public_entry(
        self, tmp_path: Path
    ) -> None:
        artifact_root = tmp_path / "context_world"
        artifact_root.mkdir()
        environment = dict(os.environ)
        for name in (
            "CW_COMPONENT",
            "CW_DATASET",
            "CW_CHECKPOINT_ROOT",
            "STABLEWM_HOME",
            "CONTEXTWORLD_STABLE_WORLDMODEL_REPO",
        ):
            environment.pop(name, None)
        environment.update(
            {
                "CW_TASK": "door",
                "CW_FAMILY": "pldm",
                "CW_TRAINING_TRACK": "historical_release",
                "CW_SEEDS": "3072",
                "CW_PRINT_ONLY": "1",
                "CONTEXTWORLD_ARTIFACT_ROOT": str(artifact_root),
                "PYTHON_BIN": sys.executable,
            }
        )

        completed = subprocess.run(
            ["bash", str(SCRIPTS / "cloud_train.sh")],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr
        assert "route=stablewm-train" in completed.stdout
        assert "mode=release-reproduction" in completed.stdout
        assert "run_h3_hidden_passage_train.sh" in completed.stdout
        assert "pldm-mixed" in completed.stdout

    def test_a_relative_dataset_root_fails_before_python(
        self, tmp_path: Path
    ) -> None:
        environment = dict(os.environ)
        environment.pop("CW_DATASET", None)
        environment.update(
            {
                "CW_TASK": "original",
                "CW_ENV": "tworoom",
                "CW_FAMILY": "prejepa",
                "CONTEXTWORLD_DATASET_ROOT": "relative/data/world_model",
                "CW_CHECKPOINT_ROOT": str(tmp_path / "checkpoints"),
            }
        )

        completed = subprocess.run(
            ["bash", str(SCRIPTS / "cloud_train.sh")],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 2
        assert "CONTEXTWORLD_DATASET_ROOT must be absolute" in completed.stderr

    def test_a_relative_cloud_dataset_fails_before_python(
        self, tmp_path: Path
    ) -> None:
        environment = dict(os.environ)
        environment.update(
            {
                "CW_TASK": "original",
                "CW_ENV": "tworoom",
                "CW_FAMILY": "prejepa",
                "CW_DATASET": "relative/tworoom.h5",
                "CW_CHECKPOINT_ROOT": str(tmp_path / "checkpoints"),
            }
        )

        completed = subprocess.run(
            ["bash", str(SCRIPTS / "cloud_train.sh")],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 2
        assert "CW_DATASET must be an absolute path" in completed.stderr

    @pytest.mark.parametrize(
        "variable,option",
        [
            ("CW_TASK", "task"),
            ("CW_FAMILY", "family"),
            ("CW_MODE", "mode"),
            ("CW_DATASET", "dataset"),
        ],
    )
    def test_each_option_is_settable_from_the_environment(
        self, variable: str, option: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The platform passes environment variables, not flags."""

        expected = "door"
        monkeypatch.setenv("CW_TASK", "door")
        monkeypatch.setenv(variable, expected)
        parsed = router.parse_args([])

        assert str(getattr(parsed, option)) == expected

    def test_seed_list_is_settable_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CW_TASK", "door")
        monkeypatch.setenv("CW_SEEDS", "7,11")

        assert router.parse_args([]).seeds == (7, 11)

    def test_a_missing_task_fails_with_a_usable_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CW_TASK", raising=False)

        with pytest.raises(SystemExit):
            router.parse_args([])
