from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import tomli
import yaml

from contextworld.benchmarks import suite_data


ROOT = Path(__file__).resolve().parents[1]
METADATA_ROOT = (
    ROOT
    / "contextworld/_release_metadata/contextworld_icl_suite_v2_release"
)
RECEIPTS = {
    "specification": (
        "artifacts/evaluation/contextworld_icl_suite_v2_release/"
        "public_scoreboard_spec.json",
        "public_scoreboard_spec.json",
    ),
    "scoreboard": (
        "artifacts/evaluation/contextworld_icl_suite_v2_release/"
        "public_scoreboard.json",
        "public_scoreboard.json",
    ),
}


def _suite_with_frozen_receipts() -> dict:
    release = yaml.safe_load(
        (
            ROOT
            / "configs/benchmark/contextworld_icl_suite_v2_recovery_v2.yaml"
        ).read_text(encoding="utf-8")
    )
    return {
        "_config_path": str(ROOT / "configs/benchmark/suite.yaml"),
        "public_results": release["public_results"],
    }


def test_packaged_receipts_match_the_frozen_suite_identities() -> None:
    suite = _suite_with_frozen_receipts()
    for key, (_, filename) in RECEIPTS.items():
        path = METADATA_ROOT / filename
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == suite[
            "public_results"
        ][key]["sha256"]


def test_public_result_resolution_prefers_canonical_then_verified_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _suite_with_frozen_receipts()
    canonical = tmp_path / "canonical.json"
    canonical.write_bytes(
        (METADATA_ROOT / RECEIPTS["specification"][1]).read_bytes()
    )
    monkeypatch.setattr(
        suite_data, "resolve_contextworld_path", lambda *_args, **_kwargs: canonical
    )
    assert suite_data._bundled_public_result(
        suite, "specification", repo_root=tmp_path
    ) == canonical

    missing = tmp_path / "missing.json"
    monkeypatch.setattr(
        suite_data, "resolve_contextworld_path", lambda *_args, **_kwargs: missing
    )
    for key, (_, filename) in RECEIPTS.items():
        assert suite_data._bundled_public_result(
            suite, key, repo_root=tmp_path
        ) == METADATA_ROOT / filename


def test_packaged_public_result_fallback_rejects_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _suite_with_frozen_receipts()
    bad_root = tmp_path / "release_metadata"
    bad_root.mkdir()
    (bad_root / RECEIPTS["scoreboard"][1]).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        suite_data,
        "resolve_contextworld_path",
        lambda *_args, **_kwargs: tmp_path / "missing.json",
    )
    monkeypatch.setattr(
        suite_data, "_PACKAGED_SUITE_V2_RELEASE_METADATA_ROOT", bad_root
    )
    with pytest.raises(RuntimeError, match="frozen receipt identity"):
        suite_data._bundled_public_result(suite, "scoreboard", repo_root=tmp_path)


def test_build_metadata_lists_only_default_info_contracts() -> None:
    project = tomli.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    data_files = project["tool"]["setuptools"]["data-files"]
    assert set(data_files) == {"configs/benchmark"}
    assert set(data_files["configs/benchmark"]) == {
        "configs/benchmark/contextworld_icl_suite_v1.yaml",
        "configs/benchmark/contextworld_icl_suite_v2_recovery_v2.yaml",
        "configs/benchmark/contextworld_icl_suite_v2_integrity_reseal_v1.yaml",
        "configs/benchmark/contextworld_icl_suite_v2_public_document_amendment_v1.yaml",
        "configs/benchmark/contextworld_historical_package_pin_correction_v1.yaml",
        "configs/benchmark/tworoom_speed_icl_release_v1.yaml",
        "configs/benchmark/tworoom_door_icl_release_v1.yaml",
        "configs/benchmark/tworoom_action_delay_icl_release_v1.yaml",
        "configs/benchmark/pusht_action_strength_icl_release_v1.yaml",
        "configs/benchmark/pusht_contact_friction_icl_release_v1.yaml",
        "configs/benchmark/pusht_motion_damping_icl_release_v1.yaml",
        "configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml",
        "configs/benchmark/reacher_arm_mass_icl_release_v1.yaml",
        "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml",
    }
    assert project["tool"]["setuptools"]["package-data"]["contextworld"] == [
        "_release_metadata/contextworld_icl_suite_v2_release/*.json",
        "_release_metadata/contextworld_icl_suite_v2_release/README.md",
    ]
