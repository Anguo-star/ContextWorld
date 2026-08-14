from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import stat

import pytest
import yaml

from scripts import finalize_cube_grasp_rule_h3_v4r1_development as finalizer


ARTIFACT_BASE = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world"
)
PREREG = finalizer.ROOT / (
    "configs/benchmark/"
    "cube_gripper_carry_h3_development_recovery_prereg_v4r1.yaml"
)
FREEZE = ARTIFACT_BASE / (
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "development_recovery_freeze_receipt_v1.json"
)
PRIOR = ARTIFACT_BASE / (
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "prior_episode_exclusions_final_v1.json"
)
BUILD = ARTIFACT_BASE / (
    "synthesis/cube_gripper_carry_rule_h3_development_v4r1"
)
PROBE = ARTIFACT_BASE / (
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "rgb_history_probe_v1.json"
)


def _load() -> dict[str, dict]:
    return {
        "prereg": yaml.safe_load(PREREG.read_text()),
        "freeze": json.loads(FREEZE.read_text()),
        "prior": json.loads(PRIOR.read_text()),
        "request": json.loads((BUILD / "request.json").read_text()),
        "report": json.loads((BUILD / "build_report.json").read_text()),
        "manifest": json.loads((BUILD / "manifest.json").read_text()),
        "success": json.loads((BUILD / "_SUCCESS.json").read_text()),
        "probe": json.loads(PROBE.read_text()),
    }


@pytest.fixture(scope="module")
def documents() -> dict[str, dict]:
    return _load()


def _identity(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return finalizer._identity(raw)


def _release_receipts() -> dict[str, dict]:
    return finalizer._release_receipts(BUILD)


@pytest.fixture(scope="module")
def release_receipts() -> dict[str, dict]:
    return _release_receipts()


def _fake_release(root: Path, *, extra_directory: bool = False) -> None:
    root.mkdir()
    for name in ("request.json", "build_report.json", "manifest.json", "_SUCCESS.json"):
        (root / name).write_text("{}\n")
    for split in ("train", "loader_validation"):
        table = root / f"{split}.lance"
        for directory in ("_transactions", "_versions", "data"):
            (table / directory).mkdir(parents=True)
        (table / "_transactions/0.txn").write_bytes(b"transaction")
        (table / "_versions/1.manifest").write_bytes(b"manifest")
        (table / "data/0.lance").write_bytes(b"data")
    if extra_directory:
        (root / "unexpected-empty-directory").mkdir()


def test_canonical_chain_finalizes_passed_data_readiness(tmp_path: Path) -> None:
    output = tmp_path / "development_decision.json"
    payload = finalizer.finalize(
        prereg_path=PREREG,
        freeze_path=FREEZE,
        prior_path=PRIOR,
        artifact_root=BUILD,
        probe_path=PROBE,
        output=output,
        enforce_canonical_paths=False,
    )
    assert payload["status"] == "passed_development"
    assert payload["scope"] == (
        "data_readiness_only_not_reference_model_or_public"
    )
    assert payload["required_gate_summary"]["all_required"] is True
    assert payload["rgb_history_probe"]["overall_accuracy"] == 0.791015625
    assert payload["public_test"]["read"] is False
    assert payload["reference_model_phase"]["optimizer_steps_run"] == 0
    assert payload["claims"]["release_claim_allowed"] is False
    assert json.loads(output.read_text()) == payload


def test_canonical_paths_are_required_by_cli(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="output must be the canonical path"):
        finalizer.finalize(
            prereg_path=PREREG,
            freeze_path=FREEZE,
            prior_path=PRIOR,
            artifact_root=BUILD,
            probe_path=PROBE,
            output=tmp_path / "decision.json",
        )


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "decision.json"
    output.write_text("sentinel")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalizer.finalize(
            prereg_path=PREREG,
            freeze_path=FREEZE,
            prior_path=PRIOR,
            artifact_root=BUILD,
            probe_path=PROBE,
            output=output,
            enforce_canonical_paths=False,
        )
    assert output.read_text() == "sentinel"


def test_public_shaped_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="forbidden Public component"):
        finalizer.finalize(
            prereg_path=PREREG,
            freeze_path=FREEZE,
            prior_path=PRIOR,
            artifact_root=BUILD,
            probe_path=PROBE,
            output=tmp_path / "validation" / "decision.json",
            enforce_canonical_paths=False,
        )


def test_symlink_input_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "probe.json"
    link.symlink_to(PROBE)
    with pytest.raises(RuntimeError, match="non-symlink"):
        finalizer.finalize(
            prereg_path=PREREG,
            freeze_path=FREEZE,
            prior_path=PRIOR,
            artifact_root=BUILD,
            probe_path=link,
            output=tmp_path / "decision.json",
            enforce_canonical_paths=False,
        )


def test_write_fsync_failure_removes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "decision.json"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(finalizer.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        finalizer._write_exclusive(output, {"status": "test"})
    assert not output.exists()


def test_output_parent_symlink_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked-parent"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="non-symlink directory"):
        finalizer._write_exclusive(link / "decision.json", {"status": "test"})
    assert not (real / "decision.json").exists()


def test_output_parent_replacement_is_detected_and_old_file_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "decision-parent"
    parent.mkdir()
    moved = tmp_path / "held-original-parent"
    original_fsync = finalizer.os.fsync
    replaced = False

    def replace_parent(descriptor: int) -> None:
        nonlocal replaced
        if not replaced and stat.S_ISREG(os.fstat(descriptor).st_mode):
            replaced = True
            parent.rename(moved)
            parent.mkdir()
        original_fsync(descriptor)

    monkeypatch.setattr(finalizer.os, "fsync", replace_parent)
    with pytest.raises(RuntimeError, match="parent changed"):
        finalizer._write_exclusive(parent / "decision.json", {"status": "test"})
    assert not (parent / "decision.json").exists()
    assert not (moved / "decision.json").exists()


def test_release_rejects_extra_empty_directory(tmp_path: Path) -> None:
    root = tmp_path / "release"
    _fake_release(root, extra_directory=True)
    with pytest.raises(RuntimeError, match="directory inventory mismatch"):
        finalizer._HeldRelease.open(root)


def test_held_release_detects_root_replacement(tmp_path: Path) -> None:
    root = tmp_path / "release"
    _fake_release(root)
    held = finalizer._HeldRelease.open(root)
    moved = tmp_path / "original-release"
    try:
        root.rename(moved)
        root.mkdir()
        with pytest.raises(RuntimeError, match="artifact root identity changed"):
            held.reverify()
    finally:
        held.close()


def test_held_release_detects_file_byte_change(tmp_path: Path) -> None:
    root = tmp_path / "release"
    _fake_release(root)
    held = finalizer._HeldRelease.open(root)
    try:
        (root / "request.json").write_text('{"changed":true}\n')
        with pytest.raises(RuntimeError, match="release file bytes changed"):
            held.reverify()
    finally:
        held.close()


@pytest.mark.parametrize("entry_kind", ("file", "directory"))
def test_held_release_detects_entry_added_after_scan(
    tmp_path: Path, entry_kind: str
) -> None:
    root = tmp_path / "release"
    _fake_release(root)
    held = finalizer._HeldRelease.open(root)
    try:
        added = root / "train.lance" / f"late-{entry_kind}"
        if entry_kind == "file":
            added.write_bytes(b"late")
        else:
            added.mkdir()
        with pytest.raises(RuntimeError, match="directory entries changed"):
            held.reverify()
    finally:
        held.close()


def test_post_write_release_check_failure_removes_decision(tmp_path: Path) -> None:
    output = tmp_path / "decision.json"

    def fail_post_write() -> None:
        raise RuntimeError("release changed during decision write")

    with pytest.raises(RuntimeError, match="release changed"):
        finalizer._write_exclusive(
            output,
            {"status": "test"},
            post_write_check=fail_post_write,
        )
    assert not output.exists()


def test_prereg_model_or_public_drift_is_rejected(documents: dict[str, dict]) -> None:
    prereg = deepcopy(documents["prereg"])
    prereg["reference_model_phase"]["optimizer_steps_authorized"] = 1
    with pytest.raises(RuntimeError, match="model phase"):
        finalizer._validate_prereg(
            prereg,
            {
                "sha256": finalizer.EXPECTED_INPUTS["preregistration"][0],
                "size_bytes": finalizer.EXPECTED_INPUTS["preregistration"][1],
            },
        )
    prereg = deepcopy(documents["prereg"])
    prereg["public_test"]["read"] = True
    with pytest.raises(RuntimeError, match="read must be false"):
        finalizer._validate_prereg(
            prereg,
            {
                "sha256": finalizer.EXPECTED_INPUTS["preregistration"][0],
                "size_bytes": finalizer.EXPECTED_INPUTS["preregistration"][1],
            },
        )


def test_freeze_and_prior_chain_drift_is_rejected(documents: dict[str, dict]) -> None:
    prereg_identity = _identity(PREREG)
    freeze_identity = _identity(FREEZE)
    freeze = deepcopy(documents["freeze"])
    freeze["authorization_inputs"]["infrastructure_failure_decision"][
        "sha256"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="failure evidence"):
        finalizer._validate_freeze(freeze, prereg_identity)
    prior = deepcopy(documents["prior"])
    prior["prior_content_exclusions"]["query_pixel_hashes"]["count"] -= 1
    with pytest.raises(RuntimeError, match="query_pixel_hashes"):
        finalizer._validate_prior(
            prior, prereg_identity, freeze_identity
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("causal", "causal data contract failed"),
        ("replay", "fresh simulator replay gate failed"),
        ("overlap", "cross-split gate failed"),
        ("anchor", "formal split contract failed"),
        ("public", "top-level contract mismatch"),
    ),
)
def test_build_gate_mutations_fail_closed(
    documents: dict[str, dict], mutation: str, message: str
) -> None:
    report = deepcopy(documents["report"])
    if mutation == "causal":
        report["causal_data_contract"]["passed"] = False
    elif mutation == "replay":
        report["fresh_simulator_replay"]["passed"] = False
    elif mutation == "overlap":
        report["cross_split_audit"]["passed"] = False
    elif mutation == "anchor":
        report["splits"]["train"]["action_anchor_counts"]["endpoint4"] -= 1
    elif mutation == "public":
        report["public_test_opened"] = True
    with pytest.raises(RuntimeError, match=message):
        finalizer._validate_build(report, documents["request"])


def test_success_marker_and_manifest_drift_are_rejected(
    documents: dict[str, dict], release_receipts: dict[str, dict]
) -> None:
    success = deepcopy(documents["success"])
    success["publication"]["success_marker_written_last"] = False
    with pytest.raises(RuntimeError, match="publication receipt"):
        finalizer._validate_success(success, release_receipts)
    manifest = deepcopy(documents["manifest"])
    first = next(iter(manifest["files"]))
    manifest["files"][first] = "0" * 64
    with pytest.raises(RuntimeError, match="manifest file digest"):
        finalizer._validate_manifest(
            manifest, _identity(PRIOR), release_receipts
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("passed", "top-level contract mismatch"),
        ("gate", "not every frozen"),
        ("overall", "primary RGB-history metrics"),
        ("bootstrap", "bootstrap result"),
        ("shortcut", "negative control mismatch"),
        ("public", "read must be false"),
    ),
)
def test_probe_mutations_fail_closed(
    documents: dict[str, dict],
    release_receipts: dict[str, dict],
    mutation: str,
    message: str,
) -> None:
    probe = deepcopy(documents["probe"])
    if mutation == "passed":
        probe["passed"] = False
    elif mutation == "gate":
        probe["gates"]["overall_accuracy_at_least_0_75"] = False
    elif mutation == "overall":
        probe["primary_probe"]["metrics"]["overall_accuracy"] = 0.99
    elif mutation == "bootstrap":
        probe["pair_cluster_anchor_stratified_bootstrap"][
            "lower_bound_2_5_percent"
        ] = 0.70
    elif mutation == "shortcut":
        probe["negative_controls"]["action_only"]["accuracy"] = 0.51
    elif mutation == "public":
        probe["public_test"]["read"] = True
    with pytest.raises(RuntimeError, match=message):
        finalizer._validate_probe(
            probe,
            _identity(PREREG),
            _identity(FREEZE),
            _identity(PRIOR),
            release_receipts,
        )
