"""Branch coverage for the three-level pin grading shared by the pin audits.

Every judgment site extended to the three-level scale is built on
``grade_source_pin`` plus one of the two replacement factories in
``tests/pin_grading.py``; this module pins down each tier of that shared
 machinery — in particular that "semantic fingerprint unchanged" passes with
a warning and that genuine semantic drift still fails — so the individual
site tests only have to exercise their wiring.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from contextworld.benchmarks.source_fingerprint import (
    ACCEPTED_TRANSITION,
    BYTE_MATCH,
    DRIFTED,
    SEMANTIC_UNCHANGED,
    fingerprint_file,
    grade_source_pin,
)

import pin_grading
from pin_grading import (
    accepted_pin_transitions,
    graded_require_identity,
    graded_require_static_identity,
)


ROOT = pin_grading.ROOT
ADAPTERS = "contextworld/benchmarks/adapters.py"

ORIGINAL = '''"""A pinned runtime source."""

LIMIT = 3


def clamp(value):
    """Keep values inside the limit."""
    return max(-LIMIT, min(LIMIT, value))
'''

COMMENT_ONLY_DRIFT = '''"""A pinned runtime source, with a clarified docstring."""

# The limit is part of the frozen contract.
LIMIT = 3


def clamp(value):
    """Keep values inside the limit (inclusive)."""
    return max(-LIMIT, min(LIMIT, value))
'''

SEMANTIC_DRIFT = '''"""A pinned runtime source."""

LIMIT = 4


def clamp(value):
    """Keep values inside the limit."""
    return max(-LIMIT, min(LIMIT, value))
'''


@pytest.fixture()
def history_repo(tmp_path: Path) -> Path:
    """A git repository whose HEAD holds ``ORIGINAL`` for one source file."""

    source = tmp_path / "contextworld/benchmarks/frozen.py"
    source.parent.mkdir(parents=True)
    source.write_text(ORIGINAL, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "add",
            "-A",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "freeze",
        ],
        check=True,
    )
    return tmp_path


def _grade(repo: Path, pinned: str, *, config: str = "configs/x.yaml"):
    return grade_source_pin(
        repo_root=repo,
        config_relative=config,
        path_relative="contextworld/benchmarks/frozen.py",
        pinned_sha256=pinned,
        accepted={},
    )


def test_byte_match_passes_silently(history_repo: Path) -> None:
    pinned = fingerprint_file(
        history_repo / "contextworld/benchmarks/frozen.py"
    ).byte_sha256

    grading = _grade(history_repo, pinned)

    assert grading.status == BYTE_MATCH
    assert grading.message is None


def test_comment_only_drift_passes_with_a_warning(history_repo: Path) -> None:
    pinned = fingerprint_file(
        history_repo / "contextworld/benchmarks/frozen.py"
    ).byte_sha256
    (history_repo / "contextworld/benchmarks/frozen.py").write_text(
        COMMENT_ONLY_DRIFT, encoding="utf-8"
    )

    grading = _grade(history_repo, pinned)

    assert grading.status == SEMANTIC_UNCHANGED
    assert "semantic fingerprint is unchanged" in grading.message


def test_semantic_drift_fails_even_when_bytes_almost_match(
    history_repo: Path,
) -> None:
    pinned = fingerprint_file(
        history_repo / "contextworld/benchmarks/frozen.py"
    ).byte_sha256
    drifted = history_repo / "contextworld/benchmarks/frozen.py"
    drifted.write_text(SEMANTIC_DRIFT, encoding="utf-8")

    grading = _grade(history_repo, pinned)

    assert grading.status == DRIFTED
    assert not grading.passed
    assert "may have changed" in grading.message


def test_accepted_transition_passes_and_binds_the_registered_state(
    history_repo: Path,
) -> None:
    pinned = fingerprint_file(
        history_repo / "contextworld/benchmarks/frozen.py"
    ).byte_sha256
    drifted = history_repo / "contextworld/benchmarks/frozen.py"
    drifted.write_text(SEMANTIC_DRIFT, encoding="utf-8")
    observed = fingerprint_file(drifted)
    accepted = {
        (
            "configs/x.yaml",
            "contextworld/benchmarks/frozen.py",
            pinned,
        ): {
            "accepted_current_sha256": observed.byte_sha256,
            "accepted_current_semantic_sha256": observed.semantic_sha256,
        }
    }

    grading = grade_source_pin(
        repo_root=history_repo,
        config_relative="configs/x.yaml",
        path_relative="contextworld/benchmarks/frozen.py",
        pinned_sha256=pinned,
        accepted=accepted,
    )

    assert grading.status == ACCEPTED_TRANSITION


def test_stale_acceptance_still_fails(history_repo: Path) -> None:
    """A correction whose recorded state no longer matches must not excuse."""

    pinned = fingerprint_file(
        history_repo / "contextworld/benchmarks/frozen.py"
    ).byte_sha256
    drifted = history_repo / "contextworld/benchmarks/frozen.py"
    drifted.write_text(SEMANTIC_DRIFT, encoding="utf-8")
    accepted = {
        (
            "configs/x.yaml",
            "contextworld/benchmarks/frozen.py",
            pinned,
        ): {
            "accepted_current_sha256": "0" * 64,
            "accepted_current_semantic_sha256": "1" * 64,
        }
    }

    grading = grade_source_pin(
        repo_root=history_repo,
        config_relative="configs/x.yaml",
        path_relative="contextworld/benchmarks/frozen.py",
        pinned_sha256=pinned,
        accepted=accepted,
    )

    assert grading.status == DRIFTED
    assert "moved after the correction was accepted" in grading.message


def test_non_python_drift_has_no_semantic_pardon(tmp_path: Path) -> None:
    data = tmp_path / "contextworld/benchmarks/table.json"
    data.parent.mkdir(parents=True)
    data.write_text('{"v": 1}', encoding="utf-8")
    pinned = fingerprint_file(data).byte_sha256
    data.write_text('{"v": 2}', encoding="utf-8")

    grading = grade_source_pin(
        repo_root=tmp_path,
        config_relative="configs/x.yaml",
        path_relative="contextworld/benchmarks/table.json",
        pinned_sha256=pinned,
        accepted={},
    )

    assert grading.status == DRIFTED
    assert "non-Python" in grading.message


def test_missing_source_and_escaping_path_fail_fast(tmp_path: Path) -> None:
    missing = grade_source_pin(
        repo_root=tmp_path,
        config_relative="configs/x.yaml",
        path_relative="contextworld/benchmarks/gone.py",
        pinned_sha256="a" * 64,
        accepted={},
    )
    assert missing.status == DRIFTED
    assert "missing" in missing.message

    outside = tmp_path / "contextworld/benchmarks/escape.py"
    outside.parent.mkdir(parents=True)
    (tmp_path / "elsewhere.py").write_text("x = 1\n", encoding="utf-8")
    escaping = grade_source_pin(
        repo_root=tmp_path,
        config_relative="configs/x.yaml",
        path_relative="../elsewhere.py",
        pinned_sha256="a" * 64,
        accepted={},
    )
    assert escaping.status == DRIFTED
    assert "escapes the repository" in escaping.message


def _contract_observer(repo: Path):
    def observe(specification, repo_root):
        path = Path(specification["path"])
        resolved = path if path.is_absolute() else repo / path
        return {
            "path": str(specification["path"]),
            "sha256": fingerprint_file(resolved).byte_sha256
            if resolved.is_file()
            else "missing",
            "size_bytes": resolved.stat().st_size if resolved.is_file() else -1,
        }

    return observe


def _logical_observer(repo: Path):
    def observe(path: str):
        resolved = repo / path
        return {
            "path": path,
            "sha256": fingerprint_file(resolved).byte_sha256,
            "size_bytes": resolved.stat().st_size,
        }

    return observe


def test_graded_require_identity_warns_on_semantic_unchanged(
    history_repo: Path,
) -> None:
    source = history_repo / "contextworld/benchmarks/frozen.py"
    pinned = fingerprint_file(source).byte_sha256
    source.write_text(COMMENT_ONLY_DRIFT, encoding="utf-8")

    def original(specification, *, label, repo_root=None):
        raise AssertionError("byte drift must not reach the original checker")

    graded = graded_require_identity(
        original,
        config_relative="configs/x.yaml",
        repo_root=history_repo,
        observe=_contract_observer(history_repo),
    )

    with pytest.warns(UserWarning, match="semantic fingerprint is unchanged"):
        identity = graded(
            {"path": "contextworld/benchmarks/frozen.py", "sha256": pinned},
            label="implementation.adapter_boundary",
        )
    assert identity["sha256"] == pinned


def test_graded_require_identity_still_fails_semantic_drift(
    history_repo: Path,
) -> None:
    source = history_repo / "contextworld/benchmarks/frozen.py"
    pinned = fingerprint_file(source).byte_sha256
    source.write_text(SEMANTIC_DRIFT, encoding="utf-8")

    graded = graded_require_identity(
        lambda *args, **kwargs: pytest.fail("unreachable"),
        config_relative="configs/x.yaml",
        repo_root=history_repo,
        observe=_contract_observer(history_repo),
    )

    with pytest.raises(RuntimeError, match="identity drifted"):
        graded(
            {"path": "contextworld/benchmarks/frozen.py", "sha256": pinned},
            label="implementation.adapter_boundary",
        )


def test_graded_require_static_identity_returns_declared_values_on_acceptance(
    history_repo: Path,
) -> None:
    source = history_repo / "contextworld/benchmarks/frozen.py"
    pinned = fingerprint_file(source).byte_sha256
    source.write_text(SEMANTIC_DRIFT, encoding="utf-8")
    observed = fingerprint_file(source)
    accepted = {
        ("configs/x.yaml", "contextworld/benchmarks/frozen.py", pinned): {
            "accepted_current_sha256": observed.byte_sha256,
            "accepted_current_semantic_sha256": observed.semantic_sha256,
        }
    }

    graded = graded_require_static_identity(
        config_relative="configs/x.yaml",
        repo_root=history_repo,
        observe=_logical_observer(history_repo),
        accepted=accepted,
    )

    with pytest.warns(UserWarning, match="registered accepted correction"):
        identity = graded(
            {
                "path": "contextworld/benchmarks/frozen.py",
                "sha256": pinned,
                "size_bytes": len(ORIGINAL.encode()),
            },
            label="implementation adapter_boundary",
        )

    assert identity == {
        "path": "contextworld/benchmarks/frozen.py",
        "sha256": pinned,
        "size_bytes": len(ORIGINAL.encode()),
    }


def test_graded_require_static_identity_fails_unregistered_drift(
    history_repo: Path,
) -> None:
    source = history_repo / "contextworld/benchmarks/frozen.py"
    pinned = fingerprint_file(source).byte_sha256
    source.write_text(SEMANTIC_DRIFT, encoding="utf-8")

    graded = graded_require_static_identity(
        config_relative="configs/x.yaml",
        repo_root=history_repo,
        observe=_logical_observer(history_repo),
    )

    with pytest.raises(RuntimeError, match="identity drifted"):
        graded(
            {
                "path": "contextworld/benchmarks/frozen.py",
                "sha256": pinned,
                "size_bytes": 1,
            },
            label="implementation adapter_boundary",
        )


def test_graded_bound_input_identity_warns_on_semantic_unchanged(
    history_repo: Path,
) -> None:
    source = history_repo / "contextworld/benchmarks/frozen.py"
    pinned = fingerprint_file(source).byte_sha256
    source.write_text(COMMENT_ONLY_DRIFT, encoding="utf-8")

    def same(left, right):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.get("path") == right.get("path")
            and left.get("sha256") == right.get("sha256")
            and left.get("size_bytes") == right.get("size_bytes")
        )

    graded = pin_grading.graded_bound_input_identity(
        config_relative="configs/x.yaml",
        repo_root=history_repo,
        observe=_logical_observer(history_repo),
        same_identity=same,
    )

    with pytest.warns(UserWarning, match="semantic fingerprint is unchanged"):
        identity = graded(
            {
                "path": "contextworld/benchmarks/frozen.py",
                "sha256": pinned,
                "size_bytes": len(ORIGINAL.encode()),
            },
            label="bound input contextworld/benchmarks/frozen.py",
        )
    assert identity["sha256"] == pinned


def test_graded_bound_input_identity_fails_semantic_drift(
    history_repo: Path,
) -> None:
    source = history_repo / "contextworld/benchmarks/frozen.py"
    pinned = fingerprint_file(source).byte_sha256
    source.write_text(SEMANTIC_DRIFT, encoding="utf-8")

    graded = pin_grading.graded_bound_input_identity(
        config_relative="configs/x.yaml",
        repo_root=history_repo,
        observe=_logical_observer(history_repo),
        same_identity=lambda left, right: left == right,
    )

    with pytest.raises(RuntimeError, match="identity drifted"):
        graded(
            {
                "path": "contextworld/benchmarks/frozen.py",
                "sha256": pinned,
                "size_bytes": 1,
            },
            label="bound input contextworld/benchmarks/frozen.py",
        )


def test_graded_failed_audit_files_splits_local_and_external(
    history_repo: Path,
) -> None:
    """The audit-set grader excuses registered local pins only."""

    source = history_repo / "contextworld/benchmarks/frozen.py"
    pinned = fingerprint_file(source).byte_sha256
    source.write_text(SEMANTIC_DRIFT, encoding="utf-8")
    observed = fingerprint_file(source)
    accepted = {
        ("configs/x.yaml", "contextworld/benchmarks/frozen.py", pinned): {
            "accepted_current_sha256": observed.byte_sha256,
            "accepted_current_semantic_sha256": observed.semantic_sha256,
        }
    }
    audit = {
        "files": {
            "identity.ok": {"passed": True, "path": "contextworld/benchmarks/ok.py"},
            "identity.excused": {
                "passed": False,
                "logical_path": "contextworld/benchmarks/frozen.py",
                "expected_sha256": pinned,
            },
            "identity.package": {
                "passed": False,
                "logical_path": "pyproject.toml",
                "expected_sha256": "0" * 64,
            },
            "identity.external": {
                "passed": False,
                "path": str(history_repo.parent / "stable-worldmodel/pldm.yaml"),
                "expected_sha256": "1" * 64,
            },
        }
    }
    original_loader = pin_grading.accepted_pin_transitions
    pin_grading.accepted_pin_transitions = lambda: accepted
    try:
        failed, excused = pin_grading.graded_failed_audit_files(
            audit,
            config_relative="configs/x.yaml",
            repo_root=history_repo,
        )
    finally:
        pin_grading.accepted_pin_transitions = original_loader

    assert failed == {"identity.package", "identity.external"}
    assert excused == {"identity.excused"}


def test_release_copy_with_graded_pins_rewrites_only_excused_pins(
    history_repo: Path,
) -> None:
    source = history_repo / "contextworld/benchmarks/frozen.py"
    pinned = fingerprint_file(source).byte_sha256
    source.write_text(SEMANTIC_DRIFT, encoding="utf-8")
    observed = fingerprint_file(source)
    accepted = {
        ("configs/x.yaml", "contextworld/benchmarks/frozen.py", pinned): {
            "accepted_current_sha256": observed.byte_sha256,
            "accepted_current_semantic_sha256": observed.semantic_sha256,
        }
    }
    release = history_repo / "configs/x.yaml"
    release.parent.mkdir()
    ok = history_repo / "contextworld/benchmarks/ok.py"
    ok.write_text("x = 1\n", encoding="utf-8")
    ok_sha = fingerprint_file(ok).byte_sha256
    release.write_text(
        yaml.safe_dump(
            {
                "identity": {
                    "adapters": {
                        "path": "contextworld/benchmarks/frozen.py",
                        "sha256": pinned,
                    },
                    "unchanged": {
                        "path": "contextworld/benchmarks/ok.py",
                        "sha256": ok_sha,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    original_loader = pin_grading.accepted_pin_transitions
    pin_grading.accepted_pin_transitions = lambda: accepted
    try:
        destination = pin_grading.release_copy_with_graded_pins(
            release,
            config_relative="configs/x.yaml",
            repo_root=history_repo,
            destination=history_repo / "audit-copy.yaml",
        )
        # The same drifted pin under a config with no registered acceptance
        # refuses to produce an audit copy at all.
        with pytest.raises(AssertionError, match="may have changed"):
            pin_grading.release_copy_with_graded_pins(
                release,
                config_relative="configs/unregistered.yaml",
                repo_root=history_repo,
                destination=history_repo / "audit-copy-2.yaml",
            )
    finally:
        pin_grading.accepted_pin_transitions = original_loader

    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert payload["identity"]["adapters"]["sha256"] == observed.byte_sha256
    assert payload["identity"]["unchanged"]["sha256"] == ok_sha


def test_correction_registry_covers_the_newly_registered_transitions() -> None:
    """The rows this audit relies on must exist and bind the live sources."""

    accepted = accepted_pin_transitions()
    live = fingerprint_file(ROOT / ADAPTERS)
    pinned = "cc9e758b7081a57251e8cd026e9ac9ff8a17e3f300d52f464bdad871edcf26b2"
    for config in (
        "configs/benchmark/tworoom_speed_pldm_infrastructure_development_v1.yaml",
        "configs/benchmark/tworoom_speed_pldm_cem_prereg_v1.yaml",
        "configs/benchmark/tworoom_speed_pldm_evaluation_binding_v1.yaml",
        "configs/benchmark/contextworld_complete_reference_comparison_prereg_v1.yaml",
        "configs/benchmark/pusht_action_strength_pldm_evaluation_binding_v1.yaml",
    ):
        row = accepted[(config, ADAPTERS, pinned)]
        assert row["accepted_current_sha256"] == live.byte_sha256
        assert row["accepted_current_semantic_sha256"] == (
            live.semantic_sha256
        )
        # The registered transition must not be a no-op.
        assert pinned != row["accepted_current_sha256"]


def test_registry_rejects_a_record_that_changed_model_results(
    tmp_path: Path,
) -> None:
    from contextworld.benchmarks.source_fingerprint import (
        load_accepted_pin_transitions,
    )

    bad = tmp_path / "correction.yaml"
    bad.write_text(
        "status: accepted_metadata_correction\n"
        "scope:\n"
        "  model_results_changed: true\n"
        "finding:\n"
        "  classification: runtime_source_fingerprint_updated_behavior_unchanged\n"
        "accepted_transitions: []\n"
        "superseded_release_lineage:\n"
        "  release_config: c\n"
        "  affected_records: []\n"
        "historical_execution_receipts: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not change model results"):
        load_accepted_pin_transitions(bad)
