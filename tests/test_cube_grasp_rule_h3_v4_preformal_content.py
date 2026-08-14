from __future__ import annotations

import hashlib

import pytest

import scripts.freeze_cube_grasp_rule_h3_v4_preformal_content as audit


def test_content_digest_matches_formal_builder_namespace() -> None:
    values = sorted(["01" * 32, "ab" * 32])
    expected = hashlib.sha256(
        b"contextworld-cube-prior-content-exclusions-v1\0"
        b"action_profile_ids\0"
        + b"".join(bytes.fromhex(value) for value in values)
    ).hexdigest()
    assert audit.canonical_content_digest(
        values, field_name="action_profile_ids"
    ) == expected


def test_episode_digest_matches_formal_builder_namespace() -> None:
    values = [0, 2, 5]
    expected = hashlib.sha256(
        b"contextworld-cube-prior-source-episodes-v1\0"
        + b"".join(value.to_bytes(8, "little", signed=True) for value in values)
    ).hexdigest()
    assert audit.excluded_source_episodes_sha256(values) == expected


@pytest.mark.parametrize("values", ([1, 1], [2, 1], [-1]))
def test_episode_digest_rejects_noncanonical_values(values: list[int]) -> None:
    with pytest.raises(ValueError):
        audit.excluded_source_episodes_sha256(values)


def test_preformal_and_formal_catalog_namespaces_are_disjoint() -> None:
    assert audit.REAL_PAIR_CATALOG_INDICES == (0, 1)
    assert max(audit.REAL_PAIR_CATALOG_INDICES) < audit.V4_FORMAL_CATALOG_INDEX_OFFSET
    assert audit.PILOT_CATALOG_START + audit.PILOT_COUNT < (
        audit.V4_FORMAL_CATALOG_INDEX_OFFSET
    )


def test_public_paths_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="Public"):
        audit.parse_args(
            [
                "--source-h5",
                "source.h5",
                "--source-h5-sha256",
                "0" * 64,
                "--pilot-json",
                "pilot.json",
                "--pilot-json-sha256",
                "0" * 64,
                "--formal-v3-report",
                "formal.json",
                "--formal-v3-report-sha256",
                "0" * 64,
                "--pilot-runner",
                "pilot.py",
                "--pilot-runner-sha256",
                "0" * 64,
                "--historical-physics-snapshot",
                "physics.py",
                "--historical-physics-snapshot-sha256",
                "0" * 64,
                "--historical-test-snapshot",
                "test.py",
                "--historical-test-snapshot-sha256",
                "0" * 64,
                "--output",
                "public/receipt.json",
            ]
        )


def test_cli_requires_all_explicit_identities() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args([])
