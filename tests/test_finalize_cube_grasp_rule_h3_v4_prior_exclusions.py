from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.finalize_cube_grasp_rule_h3_v4_prior_exclusions as finalizer


def test_episode_digest_matches_builder_namespace() -> None:
    values = [2, 5, 9]
    payload = b"".join(value.to_bytes(8, "little", signed=True) for value in values)
    expected = hashlib.sha256(
        b"contextworld-cube-prior-source-episodes-v1\0" + payload
    ).hexdigest()
    assert finalizer.excluded_source_episodes_sha256(values) == expected


def test_basis_episode_digest_keeps_legacy_newline_ascii_namespace() -> None:
    values = [2, 5, 9]
    expected = hashlib.sha256(b"2\n5\n9\n").hexdigest()
    assert finalizer.basis_episode_ids_sha256(values) == expected
    assert expected != finalizer.excluded_source_episodes_sha256(values)


def test_basis_and_final_authorization_statuses_are_intentionally_distinct() -> None:
    assert finalizer.BASIS_STATUS == "frozen_before_first_v4_data_build"
    assert finalizer.STATUS == "frozen_before_first_v4_build"


def test_content_digest_matches_builder_namespace() -> None:
    values = sorted(["01" * 32, "ab" * 32])
    expected = hashlib.sha256(
        b"contextworld-cube-prior-content-exclusions-v1\0"
        b"action_profile_ids\0"
        + b"".join(bytes.fromhex(value) for value in values)
    ).hexdigest()
    assert finalizer.canonical_content_digest(
        values, field_name="action_profile_ids"
    ) == expected


@pytest.mark.parametrize("values", [[1, 1], [2, 1], [-1]])
def test_episode_digest_rejects_noncanonical_values(values: list[int]) -> None:
    with pytest.raises(ValueError):
        finalizer.excluded_source_episodes_sha256(values)


def test_cli_requires_explicit_inputs() -> None:
    with pytest.raises(SystemExit):
        finalizer.parse_args([])


@pytest.mark.parametrize(
    "value", ("public/receipt.json", "x/validation.lance/report.json")
)
def test_public_shaped_paths_are_rejected(value: str) -> None:
    with pytest.raises(RuntimeError, match="Public"):
        finalizer._reject_public_path(value, label="test")


def test_prior_content_union_does_not_require_scene_subset() -> None:
    # The same source episode may receive a different smoke scene seed.  The
    # final receipt must union those hashes instead of pretending source
    # episode inclusion proves content inclusion.
    formal = {"01" * 32}
    smoke = {"02" * 32}
    formal.update(smoke)
    assert formal == {"01" * 32, "02" * 32}


def _write_preformal_receipt(path: Path) -> None:
    episodes = [0, 17]
    content = {}
    for index, field in enumerate(finalizer.CONTENT_FIELDS, start=1):
        values = [f"{index:02x}" * 32]
        content[field] = {
            "values": values,
            "count": 1,
            "sha256": finalizer.canonical_content_digest(
                values, field_name=field
            ),
        }
    value = {
        "protocol_id": finalizer.V4_PROTOCOL,
        "status": finalizer.STATUS,
        "checks_passed": True,
        "reconstruction_contract": {
            "existing_pilot_replayed_not_reselected": True,
            "existing_real_mujoco_tests_replayed": True,
            "lance_opened_or_generated": False,
            "formal_build_attempted": False,
            "coupling_or_probe_recipe_changed": False,
        },
        "input_identities": {"source_h5": {"sha256": "aa" * 32}},
        "excluded_source_episodes": episodes,
        "excluded_source_episode_count": len(episodes),
        "excluded_source_episodes_sha256": (
            finalizer.excluded_source_episodes_sha256(episodes)
        ),
        "prior_content_exclusions": content,
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "reference_model_training_or_scoring": False,
    }
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_preformal_content_receipt_is_validated_and_extracted(tmp_path: Path) -> None:
    path = tmp_path / "preformal.json"
    _write_preformal_receipt(path)
    episodes, content, artifact, inputs = (
        finalizer._extract_preformal_content_evidence(
            path, logical_path="artifacts/evaluation/history3/preformal.json"
        )
    )
    assert episodes == {0, 17}
    assert {field: len(values) for field, values in content.items()} == {
        field: 1 for field in finalizer.CONTENT_FIELDS
    }
    assert artifact["role"] == "v4_preformal_smokes_and_pilots"
    assert artifact["formal_lance_build_attempted"] is False
    assert inputs["source_h5"]["sha256"] == "aa" * 32


def test_preformal_content_receipt_rejects_public_or_digest_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "preformal.json"
    _write_preformal_receipt(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["public_test"]["read"] = True
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Public"):
        finalizer._extract_preformal_content_evidence(path, logical_path="safe.json")
