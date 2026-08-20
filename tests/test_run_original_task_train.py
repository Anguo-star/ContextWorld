"""The original-task launcher speaks each family's dialect correctly.

This is the baseline regime: the four unmodified environments the nine ICL
capabilities are built on. Every fact asserted here was verified against the
live Stable-WorldModel checkout -- the dataset columns by loading each
dataset, the override keys by composing each config. Those checks need GPUs
and tens of GB of data, so what they found is pinned here instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_original_task_train as launcher  # noqa: E402


def _args(**keywords: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "run_name": None,
        "output": None,
        "batch_size": None,
        "num_workers": None,
        "devices": None,
        "accumulate": None,
        "max_epochs": None,
        "precision": None,
        "dataset_override": False,
        "override": [],
    }
    defaults.update(keywords)
    return argparse.Namespace(**defaults)


def _pairs(env: str, family: str, seed: int = 3072, **keywords: object):
    entries = launcher.build_overrides(
        launcher.ENVIRONMENTS[env], family, seed, _args(**keywords)
    )
    pairs = {}
    for entry in entries:
        key, _, value = entry.partition("=")
        pairs[key] = value
    return pairs, entries


class TestTheFourEnvironments:
    def test_exactly_four(self) -> None:
        assert sorted(launcher.ENVIRONMENTS) == [
            "cube", "pusht", "reacher", "tworoom"
        ]

    def test_the_three_baseline_seeds(self) -> None:
        """The frozen baseline families were completed to these three, so a
        new family reports mean +/- std against the same set."""

        assert launcher.BASELINE_SEEDS == (3072, 3073, 3074)

    @pytest.mark.parametrize(
        "env,action_dim,key,encoding_dim",
        [
            ("tworoom", 2, "proprio", 2),
            ("pusht", 2, "proprio", 4),
            ("reacher", 2, "observation", 6),
            ("cube", 5, "observation", 28),
        ],
    )
    def test_geometry_matches_the_loaded_datasets(
        self, env: str, action_dim: int, key: str, encoding_dim: int
    ) -> None:
        """Verified by loading each dataset through swm.data.load_dataset."""

        environment = launcher.ENVIRONMENTS[env]

        assert environment.action_dim == action_dim
        assert environment.encoding_key == key
        assert environment.encoding_dim == encoding_dim


class TestTheTwoConfigDialects:
    """lewm/pldm and prejepa disagree about how data is selected."""

    @pytest.mark.parametrize(
        "env,group",
        [
            ("tworoom", "tworoom"),
            ("pusht", "pusht"),
            ("reacher", "dmc"),
            ("cube", "ogb"),
        ],
    )
    def test_lewm_selects_data_through_the_defaults_group(
        self, env: str, group: str
    ) -> None:
        """Two environments' group names do not match their own names --
        reacher is 'dmc' and cube is 'ogb'. Verified by hydra composition."""

        pairs, _ = _pairs(env, "lewm")

        assert pairs["data"] == group
        assert "dataset_name" not in pairs

    @pytest.mark.parametrize("env", sorted(launcher.ENVIRONMENTS))
    def test_prejepa_selects_data_through_a_flat_name(self, env: str) -> None:
        """prejepa.yaml has no data defaults group."""

        pairs, _ = _pairs(env, "prejepa")

        assert "dataset_name" in pairs
        assert "data" not in pairs

    @pytest.mark.parametrize("family", ["lewm", "pldm", "prejepa"])
    def test_every_family_names_the_run_the_same_way(
        self, family: str
    ) -> None:
        """``exp_name`` is the wm_exp wrapper's spelling; these configs
        reject it outright. Caught by hydra composition, not by inspection."""

        pairs, _ = _pairs("tworoom", family)

        assert "output_model_name" in pairs
        assert "exp_name" not in pairs

    def test_batch_size_goes_under_loader_for_lewm_only(self) -> None:
        lewm, _ = _pairs("tworoom", "lewm", batch_size=64)
        prejepa, _ = _pairs("tworoom", "prejepa", batch_size=64)

        assert lewm["loader.batch_size"] == "64"
        assert prejepa["batch_size"] == "64"


class TestPreJEPAEncoding:
    """prejepa.py raises if an encoding key is missing from the dataset."""

    @pytest.mark.parametrize("env", ["tworoom", "pusht"])
    def test_proprio_environments_keep_the_default(self, env: str) -> None:
        _, entries = _pairs(env, "prejepa")

        assert not any("wm.encoding" in entry for entry in entries)

    @pytest.mark.parametrize("env", ["reacher", "cube"])
    def test_observation_environments_are_remapped(self, env: str) -> None:
        """These two carry 'observation', not 'proprio'. Leaving the default
        would fail at dataset load with a confusing message."""

        _, entries = _pairs(env, "prejepa")

        assert "~wm.encoding.proprio" in entries
        assert "+wm.encoding.observation=10" in entries

    def test_the_remap_only_applies_to_prejepa(self) -> None:
        """lewm/pldm take their keys from the data config group."""

        _, entries = _pairs("reacher", "lewm")

        assert not any("wm.encoding" in entry for entry in entries)


class TestItRemainsALauncher:
    @pytest.mark.parametrize("env", sorted(launcher.ENVIRONMENTS))
    @pytest.mark.parametrize("family", ["lewm", "pldm", "prejepa"])
    def test_no_loss_or_objective_override(
        self, env: str, family: str
    ) -> None:
        pairs, _ = _pairs(env, family)

        for key in pairs:
            assert not key.startswith("loss")
            assert "regularizer" not in key

    @pytest.mark.parametrize("env", sorted(launcher.ENVIRONMENTS))
    @pytest.mark.parametrize("family", ["lewm", "pldm", "prejepa"])
    def test_action_width_is_left_to_upstream(
        self, env: str, family: str
    ) -> None:
        """lewm.py:285 and prejepa.py:209 both derive it from the dataset.
        Setting it here would be a second, staler source of truth."""

        pairs, _ = _pairs(env, family)

        assert not any("input_dim" in key for key in pairs)
        assert not any("action_dim" in key for key in pairs)

    def test_hardware_flags_are_omitted_unless_asked_for(self) -> None:
        pairs, _ = _pairs("tworoom", "lewm")

        assert "trainer.devices" not in pairs
        assert "trainer.max_epochs" not in pairs
        assert "loader.batch_size" not in pairs

    def test_prejepa_is_raised_to_the_baseline_batch_size(self) -> None:
        """prejepa.yaml ships 32; lewm and pldm both ship 128."""

        pairs, _ = _pairs("tworoom", "prejepa")

        assert pairs["batch_size"] == str(launcher.BASELINE_BATCH_SIZE)

    def test_lewm_keeps_its_own_batch_size_default(self) -> None:
        """It is already 128 upstream, so overriding it adds a second copy."""

        pairs, _ = _pairs("tworoom", "lewm")

        assert "loader.batch_size" not in pairs

    def test_explicit_overrides_are_forwarded_last(self) -> None:
        _, entries = _pairs(
            "tworoom", "lewm", override=["trainer.max_epochs=1"]
        )

        assert entries[-1] == "trainer.max_epochs=1"


class TestDatasetResolution:
    def test_an_absolute_path_is_used_when_it_exists(
        self, tmp_path: Path
    ) -> None:
        """A relative name resolves under STABLEWM_HOME/datasets, where an
        empty directory from an interrupted download shadows the real file
        and silently triggers a multi-GB re-download."""

        environment = launcher.ENVIRONMENTS["tworoom"]
        target = tmp_path / environment.dataset_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")

        assert launcher.dataset_argument(environment, tmp_path) == str(target)

    def test_it_falls_back_to_the_plain_name(self, tmp_path: Path) -> None:
        """If we cannot resolve it, upstream's own lookup should still run."""

        environment = launcher.ENVIRONMENTS["tworoom"]

        assert (
            launcher.dataset_argument(environment, tmp_path)
            == environment.dataset_name
        )

    def test_no_root_means_no_opinion(self) -> None:
        environment = launcher.ENVIRONMENTS["cube"]

        assert (
            launcher.dataset_argument(environment, None)
            == environment.dataset_name
        )

    def test_a_missing_explicit_root_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="Dataset root not found"):
            launcher.resolve_dataset_root(str(tmp_path / "nope"))

    def test_lewm_keeps_its_data_group_unless_asked(self) -> None:
        """The group is the source of truth; overriding the path silently
        would make the config and the run disagree."""

        pairs, _ = _pairs("tworoom", "lewm")

        assert "data.dataset.name" not in pairs


class TestRunNaming:
    def test_runs_are_named_by_env_family_and_seed(self) -> None:
        assert (
            launcher.run_name("cube", "prejepa", 3074)
            == "cube_prejepa_original_s3074"
        )

    def test_an_explicit_name_wins(self) -> None:
        pairs, _ = _pairs("cube", "prejepa", run_name="mine")

        assert pairs["output_model_name"] == "mine"

    @pytest.mark.parametrize("seed", launcher.BASELINE_SEEDS)
    def test_each_seed_gets_its_own_run(self, seed: int) -> None:
        pairs, _ = _pairs("tworoom", "lewm", seed=seed)

        assert pairs["seed"] == str(seed)
        assert pairs["output_model_name"].endswith(f"_s{seed}")
