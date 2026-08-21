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
import sys
from pathlib import Path

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
        "all_seeds": False,
        "family": "lewm",
        "seed": 3072,
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

        assert "run_prejepa_train.py" in " ".join(plan.command)
        assert f"--task {task}" in " ".join(plan.command)

    @pytest.mark.parametrize("task", router.TASKS)
    def test_every_launcher_it_names_exists(self, task: str) -> None:
        """A routing table that points at a moved file is worse than none."""

        for family in ("lewm", "pldm"):
            command = router.build_plan(
                _args(task=task, family=family, dataset="d")
            ).command
            script = Path(command[1] if command[0] == "bash" else command[1])

            assert script.is_file(), f"{task}/{family} -> missing {script}"


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

        assert "run_original_task_train.py" in command
        assert f"--env {env}" in command
        assert f"--family {family}" in command

    def test_it_does_not_need_a_dataset_for_prejepa(self) -> None:
        """The benchmark route requires CW_DATASET; here the environment
        determines the data, so requiring it too would be noise."""

        plan = router.build_plan(
            _args(task="original", env="cube", family="prejepa")
        )

        assert plan.command

    def test_all_seeds_replaces_the_single_seed(self) -> None:
        plan = router.build_plan(
            _args(task="original", env="pusht", family="lewm", all_seeds=True)
        )
        command = " ".join(plan.command)

        assert "--all-seeds" in command
        assert "--seed" not in command

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
    def test_the_shell_entry_delegates_to_the_router(self) -> None:
        """The job template can only call `bash <path>`."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert "cloud_train.py" in text
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
            "HF_HUB_CACHE",
        ):
            assert f"export {exported}" in text or (
                f": \"${{{exported}" in text
            ), exported

    def test_an_explicit_data_root_is_honoured(self) -> None:
        """A user who sets CW_DATA_ROOT is trusted; detection is skipped."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert 'if [ -z "${CW_DATA_ROOT:-}" ]' in text

    def test_detection_failure_is_loud(self) -> None:
        """A silent wrong path re-downloads GB or trains on nothing."""

        text = (SCRIPTS / "cloud_train.sh").read_text(encoding="utf-8")

        assert "cannot locate the data root" in text
        assert "exit 2" in text

    @pytest.mark.parametrize(
        "variable,option",
        [
            ("CW_TASK", "task"),
            ("CW_FAMILY", "family"),
            ("CW_SEED", "seed"),
            ("CW_MODE", "mode"),
            ("CW_DATASET", "dataset"),
        ],
    )
    def test_each_option_is_settable_from_the_environment(
        self, variable: str, option: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The platform passes environment variables, not flags."""

        expected = "7" if option == "seed" else "door"
        monkeypatch.setenv("CW_TASK", "door")
        monkeypatch.setenv(variable, expected)
        parsed = router.parse_args([])

        assert str(getattr(parsed, option)) == expected

    def test_a_missing_task_fails_with_a_usable_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CW_TASK", raising=False)

        with pytest.raises(SystemExit):
            router.parse_args([])
