"""Which suite manifest is the live one, and which are historical markers.

A previous debugging session lost hours to this distinction.  The export API's
default ``release_config`` is the v1 manifest, and several v2-era manifests
froze on 2026-08-14 as deliberate historical commit markers.  Running the
exporter against any of them fails the frozen-input gate with a long list of
hash mismatches that *looks* like the working tree has drifted.

It has not.  The live view is the pointer-only successor overlay resolved by
``resolve_suite_v2_cli_default_config``.  The mismatches are the governance
working as designed: a historical marker must not appear to have signed bytes
that were written after it.

The failure mode these tests exist to prevent is the tempting "fix" --
refreshing the stale hashes in a historical manifest so the exporter stops
complaining.  That is explicitly forbidden by the overlay's own
``prohibited_changes`` list, and it would destroy the audit trail it looks
like it is repairing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from contextworld.benchmarks.suite_data import (
    SUITE_V2_RECOVERY_CONFIG,
    _assert_frozen_export_inputs,
    load_icl_suite_release,
    resolve_suite_v2_cli_default_config,
)
from contextworld.paths import repository_root
from scripts import freeze_tworoom_speed_pldm_cem_binding_v1 as cem_binding
from scripts import run_tworoom_speed_pldm_cem_v1 as cem_runner

import pin_grading


OVERLAY_PATH = (
    "configs/benchmark/contextworld_icl_suite_v2_current_results_overlay_v1.yaml"
)

CEM_PREREG_CONFIG = "configs/benchmark/tworoom_speed_pldm_cem_prereg_v1.yaml"


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return repository_root().resolve()


def _install_graded_cem_static_identity(monkeypatch, repo_root: Path) -> None:
    """Grade the Speed CEM pins the v2 decision validation walks through."""

    monkeypatch.setattr(
        cem_binding,
        "_require_static_identity",
        pin_grading.graded_require_static_identity(
            config_relative=CEM_PREREG_CONFIG,
            repo_root=repo_root,
            observe=lambda path: cem_binding._source(path),
        ),
    )
    monkeypatch.setattr(
        cem_runner,
        "_source_identity",
        pin_grading.graded_bound_input_identity(
            config_relative=CEM_PREREG_CONFIG,
            repo_root=repo_root,
            observe=lambda path: cem_runner.identity(
                cem_runner.resolve_source(path, repo_root=cem_runner.ROOT),
                repo_root=cem_runner.ROOT,
            ),
            same_identity=cem_runner._same_identity,
        ),
    )


def _skip_without_pinned_speed_runtime(repo_root: Path) -> None:
    completion = yaml.safe_load(
        (
            repo_root
            / "configs/benchmark/tworoom_speed_pldm_reference_completion_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    ready, reason = pin_grading.stablewm_runtime_readiness(
        completion["runtime"]["stable_worldmodel"]
    )
    if not ready:
        pytest.skip(reason)


def test_the_resolved_current_view_passes_the_frozen_export_gate(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live manifest must accept the live tree.

    This is the assertion whose absence sent a debugging session chasing a
    non-existent drift.  If it fails, either a source really did change
    without its manifest, or the resolver is pointing at a historical file.
    """

    _install_graded_cem_static_identity(monkeypatch, repo_root)
    _skip_without_pinned_speed_runtime(repo_root)
    config = resolve_suite_v2_cli_default_config(repo_root=repo_root)
    suite = load_icl_suite_release(config)

    _assert_frozen_export_inputs(suite, repo_root=repo_root)


def test_the_resolver_points_at_the_successor_overlay(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the identity of the live view so a silent repoint is visible."""

    _install_graded_cem_static_identity(monkeypatch, repo_root)
    _skip_without_pinned_speed_runtime(repo_root)
    config = resolve_suite_v2_cli_default_config(repo_root=repo_root)

    assert config == repo_root / OVERLAY_PATH


def test_the_overlay_forbids_rewriting_the_historical_manifests(
    repo_root: Path,
) -> None:
    """The prohibition is what makes the stale hashes correct rather than a bug."""

    payload = yaml.safe_load((repo_root / OVERLAY_PATH).read_text(encoding="utf-8"))
    overlay = payload["current_results_overlay"]

    assert (
        "rewrite_historical_v1_config_or_scoreboard"
        in overlay["prohibited_changes"]
    )


def test_the_recovery_manifest_remains_a_historical_marker(
    repo_root: Path,
) -> None:
    """Recovery-v2 is expected to be stale, and must stay that way.

    Its hashes describe the tree as of 2026-08-14.  A future contributor who
    "fixes" the exporter by refreshing them will fail here, which is the
    point: the fix is to pass the resolved current config instead.
    """

    suite = load_icl_suite_release(SUITE_V2_RECOVERY_CONFIG)

    with pytest.raises(RuntimeError, match="Suite export inputs are not frozen"):
        _assert_frozen_export_inputs(suite, repo_root=repo_root)
