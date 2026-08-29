"""The HF exporter must publish only registered Train/Dev/Test artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from contextworld.benchmarks.hf_clean_export import (
    CleanExportError,
    EXPECTED_COMPONENTS,
    TWOROOM_NORMALIZER_RELATIVE_PATH,
    build_export_plan,
    export_hf_clean,
    refresh_hf_clean_metadata,
)
from contextworld.benchmarks.external_model_cli import (
    _public_test_bundle_binding,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/benchmark/contextworld_hf_clean_export_v1.yaml"


def _source_fixture(tmp_path: Path) -> Path:
    source_root = tmp_path / "benchmark"
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    for component_id, component in contract["components"].items():
        for row in component["sources"]:
            source = source_root / row["source"]
            source.mkdir(parents=True, exist_ok=True)
            if row["source"].endswith(".lance"):
                tables = [source]
            elif component_id == "speed":
                tables = [source / "regime" / "part.lance"]
            elif component_id in {"door", "action_delay"}:
                tables = [source / "part-a.lance", source / "part-b.lance"]
            elif row["split"] == "test":
                tables = [source / "validation.lance"]
            else:
                tables = []
            for table in tables:
                (table / "_versions").mkdir(parents=True, exist_ok=True)
                (table / "data").mkdir(parents=True, exist_ok=True)
                (table / "_versions" / "1.manifest").write_bytes(b"v1")
                (table / "data" / "payload.bin").write_bytes(
                    f"{row['source']}\n".encode("utf-8")
                )
            if row.get("exclude"):
                excluded = source / row["exclude"][0]
                excluded.mkdir(parents=True, exist_ok=True)
                (excluded / "must-not-copy.json").write_text(
                    '{"internal":"score"}\n', encoding="utf-8"
                )

    # These are deliberately present in the source tree but are not mapped.
    (source_root / "evaluation/public_test").mkdir(parents=True)
    (source_root / "evaluation/public_test/secret.bin").write_bytes(b"test")
    (source_root / "training/swanlab").mkdir(parents=True)
    (source_root / "training/swanlab/metadata.json").write_text(
        '{"token": "must-not-copy"}\n', encoding="utf-8"
    )
    return source_root


def _manifest(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_plan_has_exactly_nine_components_and_creates_nothing(
    tmp_path: Path,
) -> None:
    source = _source_fixture(tmp_path)
    output = tmp_path / "output"

    plan = build_export_plan(
        contract_path=CONTRACT, suite_export_root=source, repo_root=ROOT
    )

    assert tuple(row["component_id"] for row in plan["components"]) == (
        EXPECTED_COMPONENTS
    )
    assert plan["public_test_policy"] == "public_offline_final_reporting"
    assert plan["inventory"]["file_count"] > 0
    assert plan["inventory"]["total_bytes"] > 0
    assert not output.exists()

    by_id = {row["component_id"]: row for row in plan["components"]}
    assert {
        payload["payload_kind"] for payload in by_id["speed"]["payloads"]
    } == {"nested_lance_collection"}
    assert {
        payload["payload_kind"] for payload in by_id["door"]["payloads"]
    } == {"lance_collection"}
    assert {
        payload["payload_kind"]
        for payload in by_id["action_delay"]["payloads"]
    } == {"lance_collection"}
    for component_id in EXPECTED_COMPONENTS:
        assert not any(
            payload["direct_stable_worldmodel_load"]
            for payload in by_id[component_id]["payloads"]
        )
    for component_id in EXPECTED_COMPONENTS[3:-1]:
        assert all(
            payload["single_dataset_entrypoint"]
            for payload in by_id[component_id]["payloads"]
        )
        assert {
            payload["stable_worldmodel_adapter_required"]
            for payload in by_id[component_id]["payloads"]
        } == {"stablewm_step_metadata_to_episode_table_v1"}
    cube_payloads = by_id["cube_gripper_carry"]["payloads"]
    assert all(payload["single_dataset_entrypoint"] for payload in cube_payloads)
    assert not any(
        payload["direct_stable_worldmodel_load"] for payload in cube_payloads
    )
    assert {
        payload["stable_worldmodel_adapter_required"]
        for payload in cube_payloads
    } == {"cube_block_projection_to_sequence_v1"}


def test_clean_export_contains_registered_train_dev_and_public_test(
    tmp_path: Path,
) -> None:
    source = _source_fixture(tmp_path)
    output = tmp_path / "output"

    summary = export_hf_clean(
        contract_path=CONTRACT,
        suite_export_root=source,
        output=output,
        repo_root=ROOT,
    )

    assert summary["component_count"] == 9
    assert summary["public_test_included"] is True
    assert not (output / "evaluation").exists()
    assert not (output / "training").exists()
    assert not any(path.is_symlink() for path in output.rglob("*"))

    registry = json.loads((output / "task_registry.json").read_text())
    assert registry["public_test"] == {
        "included": True,
        "policy": "public_offline_final_reporting",
        "evaluation_interface": "offline_final_reporting",
        "selection_policy": (
            "development_only_model_selection_test_final_reporting"
        ),
    }
    assert len(registry["components"]) == 9
    for component in registry["components"]:
        for payload in component["payloads"]:
            assert "public_path" in payload
            assert "payload_kind" in payload
            assert "single_dataset_entrypoint" in payload
            assert "stable_worldmodel_sequence_schema" in payload
            assert "source" not in payload
            assert "source_logical_path" in payload["provenance"]
        development = component["development_evaluation"]
        assert development["status"] == "public_development_only"
        assert development["split"] == "development"
        assert development["payload"]["public_path"].startswith(
            f"components/{component['dataset_id']}/v1/development/"
        )
        assert development["payload"]["members"]
        normalization = development["action_normalization"]
        assert normalization["transform"] == "zscore"
        assert len(normalization["mean"]) == component["action_dimension"]
        assert len(normalization["std"]) == component["action_dimension"]
        assert all(value > 0.0 for value in normalization["std"])
        public_test = component["public_test_evaluation"]
        assert public_test["status"] == "public_final_reporting_only"
        assert public_test["split"] == "test"
        assert public_test["artifact_root"].startswith("artifacts/")
        assert public_test["payloads"]
        assert public_test["official_scoreboard_row"] is False

    assert not any(
        "score_receipts/" in str(row["path"])
        for row in _manifest(output / "manifest.jsonl")
    )
    for component_id in EXPECTED_COMPONENTS:
        binding = _public_test_bundle_binding(output, task=component_id)
        assert binding["task"] == component_id
        assert binding["manifest_payload_files"] > 0
        assert binding["selection_policy"] == (
            "development_only_model_selection_test_final_reporting"
        )

    by_id = {component["component_id"]: component for component in registry["components"]}
    action_delay = by_id["action_delay"]["development_evaluation"]
    assert action_delay["payload_id"] == "full"
    assert action_delay["selection"] == {
        "reference_condition": 0,
        "contrasts": list(range(1, 11)),
        "profiles": 6,
        "pairs_per_contrast_per_profile": 5,
        "selected_pair_count": 300,
        "method": "lexicographic_first_episode_ids_v1",
    }
    for component_id in ("speed", "door", "action_delay"):
        assert (
            by_id[component_id]["development_evaluation"]["normalizer_path"]
            == TWOROOM_NORMALIZER_RELATIVE_PATH
        )
    normalizer = json.loads(
        (output / TWOROOM_NORMALIZER_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    assert normalizer["protocol"] == "tworoom_original_train_s3072_unbiased_zscore_v1"
    assert normalizer["statistics_scope"] == "original_9000_train_episodes_only"
    assert normalizer["columns"]["action"]["std_unbiased"] == pytest.approx(
        [0.867571689163936, 0.8688840167517821]
    )
    assert normalizer["columns"]["proprio"]["std_unbiased"] == pytest.approx(
        [36.85458874773545, 38.17356572449523]
    )

    speed_card = (
        output / "components/tworoom-speed/v1/component_card.md"
    ).read_text(encoding="utf-8")
    assert "Do not pass their split" in speed_card
    cube_card = (
        output / "components/cube-gripper-carry/v1/component_card.md"
    ).read_text(encoding="utf-8")
    assert "must not be passed directly" in cube_card
    assert "cube_block_projection_to_sequence_v1" in cube_card
    assert "CW_DATASET=<clean-root>" not in cube_card
    action_card = (
        output / "components/pusht-action-strength/v1/component_card.md"
    ).read_text(encoding="utf-8")
    assert "string-valued metadata" in action_card
    assert "stablewm_step_metadata_to_episode_table_v1" in action_card

    rows = _manifest(output / "manifest.jsonl")
    assert rows
    for row in rows:
        path = output / str(row["path"])
        assert path.is_file()
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert not str(row["path"]).startswith("/")


def test_symlink_inside_a_registered_source_is_rejected(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    first = next(iter(contract["components"].values()))["sources"][0]
    mapped_source = source / first["source"]
    payload = next(path for path in mapped_source.rglob("*") if path.is_file())
    (mapped_source / "link").symlink_to(payload)

    with pytest.raises(CleanExportError, match="Symlink"):
        export_hf_clean(
            contract_path=CONTRACT,
            suite_export_root=source,
            output=tmp_path / "output",
            repo_root=ROOT,
        )


def test_credential_like_text_is_rejected(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    first = next(iter(contract["components"].values()))["sources"][0]
    fake_credential = "ghp_" + "abcdefghijklmnopqrstuvwxyz123456"
    (source / first["source"] / "metadata.json").write_text(
        f'{{"token":"{fake_credential}"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(CleanExportError, match="Credential-like"):
        export_hf_clean(
            contract_path=CONTRACT,
            suite_export_root=source,
            output=tmp_path / "output",
            repo_root=ROOT,
        )


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(CleanExportError, match="already exists"):
        export_hf_clean(
            contract_path=CONTRACT,
            suite_export_root=source,
            output=output,
            repo_root=ROOT,
        )


def test_direct_write_supports_managed_mount_semantics(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    output = tmp_path / "direct-output"

    summary = export_hf_clean(
        contract_path=CONTRACT,
        suite_export_root=source,
        output=output,
        repo_root=ROOT,
        atomic_publish=False,
    )

    assert summary["status"] == "clean_staging_created"
    assert (output / "manifest.jsonl").is_file()
    assert (output / "task_registry.json").is_file()


def test_metadata_refresh_preserves_payloads_and_rebuilds_manifest(
    tmp_path: Path,
) -> None:
    source = _source_fixture(tmp_path)
    output = tmp_path / "refresh-output"
    created = export_hf_clean(
        contract_path=CONTRACT,
        suite_export_root=source,
        output=output,
        repo_root=ROOT,
    )
    payload = next(
        row for row in _manifest(output / "manifest.jsonl")
        if row["role"] == "dataset_payload"
    )
    payload_path = output / str(payload["path"])
    before = payload_path.read_bytes()

    refreshed = refresh_hf_clean_metadata(
        contract_path=CONTRACT,
        output=output,
        repo_root=ROOT,
    )

    assert refreshed["status"] == "clean_staging_metadata_refreshed"
    assert refreshed["manifest_sha256"] == created["manifest_sha256"]
    assert payload_path.read_bytes() == before
