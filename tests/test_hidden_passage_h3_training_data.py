from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import contextworld.evaluation.hidden_passage_h3_data as h3_data
from contextworld.evaluation.hidden_passage_h3_data import (
    GROUP_RULES,
    LOGICAL_CONTENT_COLUMNS,
    SHARD_COMPLETION_PROTOCOL,
    HiddenPassageShardPlan,
    _publish_hidden_passage_shard_completion,
    _remove_incomplete_planned_shard,
    audit_hidden_passage_release_assets,
    collect_hidden_passage_shard,
    directory_sha256,
    door_splits_for_scale,
    episode_plans_for_door,
    hidden_passage_release_lock,
    shard_completion_marker_path,
    templates_for_door,
    verify_hidden_passage_shard_completion,
)
from contextworld.synthesis.config import load_config
from contextworld.synthesis.stablewm import load_stable_worldmodel
from contextworld.training.tworoom_data import _catalog_split_audit


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_hidden_passage_h3_training_data_v1.yaml"
)


def _config():
    return load_config(CONFIG)


def test_formal_door_split_excludes_frozen_validation_positions() -> None:
    config = _config()
    splits = door_splits_for_scale(config, "formal")
    eval_only = set(range(30, 195, 4))
    safe = set(range(36, 189))

    assert len(eval_only) == 42
    assert len(safe & eval_only) == 38
    assert len(safe - eval_only) == 115
    assert len(splits.train) == 96
    assert len(splits.val) == 16
    assert len(splits.guard) == 3
    assert set(splits.test) == eval_only
    assert not set(splits.train) & eval_only
    assert not set(splits.val) & eval_only
    assert (
        set(splits.train) | set(splits.val) | set(splits.guard)
        == safe - eval_only
    )


def test_formal_and_small_template_counts_are_exact() -> None:
    config = _config()
    formal = templates_for_door(
        config,
        scale="formal",
        door_position=37,
    )
    small = templates_for_door(
        config,
        scale="small",
        door_position=37,
    )

    assert len(formal) == 2 * 4 * 10 == 80
    assert len(small) == 2 * 2 * 2 == 8
    assert len({row.template_id for row in formal}) == 80
    assert all(
        "passable" not in row.template_id
        and "blocked" not in row.template_id
        for row in formal
    )


def test_small_episode_plans_are_paired_and_rule_hidden_from_actions() -> None:
    plans = episode_plans_for_door(
        _config(),
        scale="small",
        door_position=37,
    )

    assert set(plans) == {"passable", "blocked"}
    assert len(plans["passable"]) == len(plans["blocked"]) == 8
    passable = {row.template_id: row for row in plans["passable"]}
    blocked = {row.template_id: row for row in plans["blocked"]}
    assert set(passable) == set(blocked)
    for template_id in passable:
        left = passable[template_id]
        right = blocked[template_id]
        assert left.collection_actions.shape == (20, 2)
        assert (
            left.expected_hashes["model_actions"]
            == right.expected_hashes["model_actions"]
        )
        assert (
            left.expected_hashes["initial_pixels"]
            == right.expected_hashes["initial_pixels"]
        )
        assert (
            left.expected_hashes["query_pixels"]
            == right.expected_hashes["query_pixels"]
        )
        assert (
            left.expected_hashes["future_pixels"]
            != right.expected_hashes["future_pixels"]
        )


def test_public_group_names_are_stable() -> None:
    assert GROUP_RULES == {
        "passage_passable": ("passable",),
        "passage_blocked": ("blocked",),
        "passage_mixed": ("passable", "blocked"),
    }


class _FakeLanceWriter:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.mkdir(parents=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def write_episodes(self, episodes) -> None:
        (self.path / "data.bin").write_bytes(b"non-empty-lance")


def _fake_shard(tmp_path: Path) -> HiddenPassageShardPlan:
    return HiddenPassageShardPlan(
        split="train",
        door_position=49,
        rule="passable",
        pair_id="pair-49",
        fingerprint="f" * 64,
        scenario_id="scenario-49",
        table_path=tmp_path / "scenario-49.lance",
        episode_manifest_path=tmp_path / "scenario-49.jsonl",
    )


def _fake_episode_rows() -> list[dict]:
    return [
        {
            "episode_index": 0,
            "template_id": "template-0",
            "rule": "passable",
            **{
                f"raw_{column}_sha256": f"{index:064x}"
                for index, column in enumerate(
                    LOGICAL_CONTENT_COLUMNS,
                    start=1,
                )
            },
        }
    ]


def test_shard_copy_does_not_rename_a_nonempty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = _fake_shard(tmp_path)
    real_replace = os.replace

    def reject_nonempty_directory_rename(source, destination):
        if Path(source).is_dir() and any(Path(source).iterdir()):
            raise PermissionError("simulated shared-filesystem EPERM")
        return real_replace(source, destination)

    monkeypatch.setattr(
        h3_data,
        "build_lance_writer",
        lambda swm, path, pixel_codec: _FakeLanceWriter(path),
    )
    monkeypatch.setattr(os, "replace", reject_nonempty_directory_rename)
    collect_hidden_passage_shard(
        None,
        plans=[object()],
        table_path=shard.table_path,
        config={
            "collection": {"staging_root": str(tmp_path / "staging")},
            "storage": {"pixel_codec": {"format": "png"}},
        },
    )

    assert (shard.table_path / "data.bin").is_file()
    assert not shard_completion_marker_path(shard.table_path).exists()


def test_interrupted_copy_is_incomplete_and_safe_to_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = _fake_shard(tmp_path)

    def interrupted_copy(source, destination):
        destination = Path(destination)
        destination.mkdir(parents=True)
        (destination / "only-one-file").write_bytes(b"partial")
        raise RuntimeError("simulated interrupted shared-filesystem copy")

    monkeypatch.setattr(
        h3_data,
        "build_lance_writer",
        lambda swm, path, pixel_codec: _FakeLanceWriter(path),
    )
    monkeypatch.setattr(h3_data.shutil, "copytree", interrupted_copy)
    with pytest.raises(RuntimeError, match="simulated interrupted"):
        collect_hidden_passage_shard(
            None,
            plans=[object()],
            table_path=shard.table_path,
            config={
                "collection": {
                    "staging_root": str(tmp_path / "staging")
                },
                "storage": {"pixel_codec": {"format": "png"}},
            },
        )

    assert shard.table_path.is_dir()
    assert not shard_completion_marker_path(shard.table_path).exists()
    with pytest.raises(ValueError, match="no regular episode manifest"):
        verify_hidden_passage_shard_completion(
            table_path=shard.table_path,
            episode_manifest_path=shard.episode_manifest_path,
            expected_scenario_id=shard.scenario_id,
            expected_fingerprint=shard.fingerprint,
        )

    _remove_incomplete_planned_shard(shard)
    assert not any(
        path.exists() or path.is_symlink()
        for path in (
            shard.table_path,
            shard.episode_manifest_path,
            shard_completion_marker_path(shard.table_path),
        )
    )


def test_tampered_completion_marker_is_rejected(
    tmp_path: Path,
) -> None:
    shard = _fake_shard(tmp_path)
    shard.table_path.mkdir()
    (shard.table_path / "data.bin").write_bytes(b"complete")
    rows = _fake_episode_rows()
    shard.episode_manifest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    completion = _publish_hidden_passage_shard_completion(
        shard=shard,
        episode_rows=rows,
    )
    assert completion["passed"]

    marker_path = shard_completion_marker_path(shard.table_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["fingerprint"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="marker identity failed"):
        verify_hidden_passage_shard_completion(
            table_path=shard.table_path,
            episode_manifest_path=shard.episode_manifest_path,
            expected_scenario_id=shard.scenario_id,
            expected_fingerprint=shard.fingerprint,
        )


def test_directory_hash_keeps_regular_semantics_and_rejects_aliases_and_fifo(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree.lance"
    tree.mkdir()
    (tree / "a").write_bytes(b"one")
    nested = tree / "nested"
    nested.mkdir()
    (nested / "b").write_bytes(b"two")

    legacy = h3_data.hashlib.sha256()
    for value in sorted(path for path in tree.rglob("*") if path.is_file()):
        legacy.update(value.relative_to(tree).as_posix().encode("utf-8"))
        legacy.update(b"\0")
        legacy.update(value.read_bytes())
        legacy.update(b"\0")
    assert directory_sha256(tree) == legacy.hexdigest()

    alias = tree / "alias"
    alias.symlink_to(tree / "a")
    with pytest.raises(ValueError, match="unsafe node"):
        directory_sha256(tree)
    alias.unlink()

    fifo = tree / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="kind=fifo"):
        directory_sha256(tree)


def test_fresh_marker_requires_sidecar_rows_to_be_exact(
    tmp_path: Path,
) -> None:
    shard = _fake_shard(tmp_path)
    shard.table_path.mkdir()
    (shard.table_path / "data.bin").write_bytes(b"complete")
    rows = _fake_episode_rows()
    sidecar_row = {**rows[0], "unsealed_extra_field": True}
    shard.episode_manifest_path.write_text(
        json.dumps(sidecar_row) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly equals"):
        _publish_hidden_passage_shard_completion(
            shard=shard,
            episode_rows=rows,
        )
    assert not shard_completion_marker_path(shard.table_path).exists()


def test_resume_removal_rejects_symlink_leaf_without_touching_target(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    table = release / "tables" / "train" / "scenario.lance"
    sidecar = (
        release / "episode_manifests" / "train" / "scenario.jsonl"
    )
    table.parent.mkdir(parents=True)
    sidecar.parent.mkdir(parents=True)
    target = tmp_path / "outside"
    target.mkdir()
    (target / "keep").write_text("do not delete", encoding="utf-8")
    table.symlink_to(target, target_is_directory=True)
    shard = HiddenPassageShardPlan(
        split="train",
        door_position=49,
        rule="passable",
        pair_id="pair",
        fingerprint="f" * 64,
        scenario_id="scenario",
        table_path=table,
        episode_manifest_path=sidecar,
    )

    with pytest.raises(ValueError, match="observed=symlink"):
        _remove_incomplete_planned_shard(shard)
    assert (target / "keep").read_text(encoding="utf-8") == "do not delete"
    assert table.is_symlink()


def test_release_asset_scan_rejects_extra_sidecar_and_symlink_alias(
    tmp_path: Path,
) -> None:
    release = tmp_path / "formal"
    table = release / "tables" / "train" / "scenario.lance"
    table.mkdir(parents=True)
    (table / "data.bin").write_bytes(b"data")
    marker = shard_completion_marker_path(table)
    marker.write_text("{}", encoding="utf-8")
    sidecar = (
        release / "episode_manifests" / "train" / "scenario.jsonl"
    )
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{}\n", encoding="utf-8")

    assert audit_hidden_passage_release_assets(
        release_root=release,
        expected_tables=[table],
        expected_markers=[marker],
        expected_sidecars=[sidecar],
    )["passed"]

    extra = sidecar.with_name("extra.jsonl")
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="release assets differ"):
        audit_hidden_passage_release_assets(
            release_root=release,
            expected_tables=[table],
            expected_markers=[marker],
            expected_sidecars=[sidecar],
        )
    extra.unlink()

    alias = table.parent / "alias.lance"
    alias.symlink_to(table, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe node"):
        audit_hidden_passage_release_assets(
            release_root=release,
            expected_tables=[table],
            expected_markers=[marker],
            expected_sidecars=[sidecar],
        )


def test_release_lock_does_not_follow_a_preexisting_symlink(
    tmp_path: Path,
) -> None:
    release = tmp_path / "formal"
    release.mkdir()
    target = tmp_path / "target"
    target.write_text("keep", encoding="utf-8")
    lock = tmp_path / ".formal.hidden-passage.lock"
    lock.symlink_to(target)

    with pytest.raises(OSError):
        with hidden_passage_release_lock(release, exclusive=True):
            pass
    assert target.read_text(encoding="utf-8") == "keep"


def _small_build_hashes(output_root: Path) -> dict:
    manifest = (
        output_root
        / "manifests"
        / "tworoom_hidden_passage_h3_mixed_v1.jsonl"
    )
    scenario_records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    output = {}
    for record in scenario_records:
        key = (
            record["split"],
            int(record["factors"]["door.position"][0]),
            record["rule"],
        )
        episode_manifest = Path(record["episode_manifest"])
        if not episode_manifest.is_absolute():
            episode_manifest = output_root.parents[0] / episode_manifest
        if not episode_manifest.is_file():
            episode_manifest = (
                output_root
                / "episode_manifests"
                / record["split"]
                / f"{record['scenario_id']}.jsonl"
            )
        episodes = [
            json.loads(line)
            for line in episode_manifest.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        output[key] = {
            "content_sha256": record["content_sha256"],
            "episode_manifest_sha256": record[
                "episode_manifest_sha256"
            ],
            "episode_hashes": {
                (episode["template_id"], episode["rule"]): {
                    name: value
                    for name, value in episode.items()
                    if name.endswith("_sha256")
                }
                for episode in episodes
            },
        }
    return output


@pytest.mark.skipif(
    os.environ.get("CONTEXTWORLD_RUN_H3_DATA_INTEGRATION") != "1",
    reason=(
        "Set CONTEXTWORLD_RUN_H3_DATA_INTEGRATION=1 to rebuild the small "
        "Lance dataset in serial and parallel modes"
    ),
)
def test_small_serial_and_parallel_build_hashes_are_identical(
    tmp_path: Path,
) -> None:
    script = ROOT / "scripts/build_tworoom_hidden_passage_h3_training_data.py"
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    common = [
        sys.executable,
        str(script),
        "--config",
        str(CONFIG),
        "--scale",
        "small",
    ]
    serial_run = subprocess.run(
        [*common, "--output-root", str(serial), "--workers", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert serial_run.returncode == 0, (
        f"{serial_run.stdout}\n{serial_run.stderr}"
    )
    parallel_run = subprocess.run(
        [*common, "--output-root", str(parallel), "--workers", "4"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert parallel_run.returncode == 0, (
        f"{parallel_run.stdout}\n{parallel_run.stderr}"
    )

    serial_hashes = _small_build_hashes(serial)
    parallel_hashes = _small_build_hashes(parallel)
    assert len(serial_hashes) == len(parallel_hashes) == 6
    assert sum(
        len(value["episode_hashes"])
        for value in serial_hashes.values()
    ) == 48
    assert serial_hashes == parallel_hashes

    removed_table = sorted((parallel / "tables").rglob("*.lance"))[0]
    shutil.rmtree(removed_table)
    resume_run = subprocess.run(
        [
            *common,
            "--output-root",
            str(parallel),
            "--workers",
            "4",
            "--resume-partial",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert resume_run.returncode == 0, (
        f"{resume_run.stdout}\n{resume_run.stderr}"
    )
    assert _small_build_hashes(parallel) == serial_hashes

    unmarked_table = sorted((parallel / "tables").rglob("*.lance"))[0]
    shard_completion_marker_path(unmarked_table).unlink()
    unmarked_resume = subprocess.run(
        [
            *common,
            "--output-root",
            str(parallel),
            "--workers",
            "4",
            "--resume-partial",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert unmarked_resume.returncode == 0, (
        f"{unmarked_resume.stdout}\n{unmarked_resume.stderr}"
    )
    assert _small_build_hashes(parallel) == serial_hashes

    config = _config()
    swm, _, stable_commit = load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    group_expectations = {
        "passable": (3, 24),
        "blocked": (3, 24),
        "mixed": (6, 48),
    }
    for suffix, (shards, episodes) in group_expectations.items():
        catalog = (
            parallel
            / "catalogs"
            / f"tworoom_hidden_passage_h3_{suffix}_v1.json"
        )
        audit = _catalog_split_audit(
            catalog,
            swm=swm,
            repo_root=ROOT,
            expected_stablewm_commit=stable_commit,
            require_complete_synthesis_report=True,
            expected_split_scenario_counts={
                "train": 2 if suffix != "mixed" else 4,
                "validation": 1 if suffix != "mixed" else 2,
                "test": 0,
            },
        )
        assert audit["logical_content_audit"] == {
            "required": True,
            "shards_verified": shards,
            "episodes_verified": episodes,
            "completion_markers_verified": shards,
            "completion_protocol": SHARD_COMPLETION_PROTOCOL,
            "columns": [
                "pixels",
                "action",
                "proprio",
                "state",
                "goal_state",
                "terminated",
                "truncated",
                "variation_agent_speed",
                "variation_door_number",
                "variation_door_position",
                "variation_passage_open",
            ],
            "passed": True,
        }

    import lance
    import pyarrow as pa

    passable_manifest = (
        parallel
        / "manifests"
        / "tworoom_hidden_passage_h3_passable_v1.jsonl"
    )
    first_record = json.loads(
        passable_manifest.read_text(encoding="utf-8").splitlines()[0]
    )
    table_path = Path(first_record["output_path"])
    lance_table = lance.dataset(table_path).to_table()
    action_index = lance_table.schema.get_field_index("action")
    actions = lance_table.column(action_index).combine_chunks()
    flat_actions = actions.values.to_numpy(
        zero_copy_only=False
    ).copy()
    flat_actions[0] = 0.25
    changed_actions = pa.FixedSizeListArray.from_arrays(
        pa.array(flat_actions, type=pa.float32()),
        2,
    )
    changed_table = lance_table.set_column(
        action_index,
        "action",
        changed_actions,
    )
    lance.write_dataset(changed_table, table_path, mode="overwrite")

    with pytest.raises(
        ValueError,
        match=(
            "completion marker content binding failed|"
            "logical content differs"
        ),
    ):
        _catalog_split_audit(
            (
                parallel
                / "catalogs"
                / "tworoom_hidden_passage_h3_passable_v1.json"
            ),
            swm=swm,
            repo_root=ROOT,
            expected_stablewm_commit=stable_commit,
            require_complete_synthesis_report=True,
        )
