from __future__ import annotations

import json
from pathlib import Path

import pytest

import contextworld.benchmarks.suite_data as suite_data
import contextworld.benchmarks.suite_v2_integrity_reseal_v2 as reseal_v2
from contextworld.benchmarks.suite_v2_cli import main as suite_v2_cli_main
from contextworld.benchmarks.suite_data import (
    SUITE_V2_CURRENT_RESULTS_OVERLAY_CONFIG_LOGICAL_PATH,
    SUITE_V2_INTEGRITY_RESEAL_CONFIG_LOGICAL_PATH,
    SUITE_V2_INTEGRITY_RESEAL_V2_DECISION,
    load_icl_suite_release,
    load_public_scoreboard,
    require_suite_membership_activation,
    resolve_suite_v2_cli_default_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _redirect_final_decision(
    monkeypatch: pytest.MonkeyPatch, decision_path: Path
) -> None:
    """Route only the final-v2 decision lookup to a controlled test file."""

    original = suite_data.resolve_no_symlink_contextworld_path

    def resolve(
        value: str | Path,
        *,
        repo_root: Path | None = None,
        label: str,
        allow_missing: bool = False,
    ) -> Path:
        if str(value) == SUITE_V2_INTEGRITY_RESEAL_V2_DECISION:
            return decision_path
        return original(
            value,
            repo_root=repo_root,
            label=label,
            allow_missing=allow_missing,
        )

    monkeypatch.setattr(suite_data, "resolve_no_symlink_contextworld_path", resolve)


def test_default_results_show_the_frozen_archive_when_final_decision_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _redirect_final_decision(monkeypatch, tmp_path / "missing-decision.json")

    default = resolve_suite_v2_cli_default_config(repo_root=ROOT)

    assert default == ROOT / SUITE_V2_INTEGRITY_RESEAL_CONFIG_LOGICAL_PATH
    assert len(load_public_scoreboard(default, repo_root=ROOT)["component_results"]) == 11


def test_default_info_describes_the_archive_without_reactivating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _redirect_final_decision(monkeypatch, tmp_path / "missing-decision.json")

    suite_v2_cli_main(["info"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["release_view"] == "historical_archive"
    assert payload["active_release"] is False
    assert payload["read_only"] is True
    assert payload["archive"]["formal_reference_rows"] == 11
    assert payload["membership_activation"] == {
        "required": True,
        "active": False,
        "status": "historical_archive_read_only",
        "current_release_claimed": False,
    }


def test_archive_default_keeps_active_audit_and_export_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _redirect_final_decision(monkeypatch, tmp_path / "missing-decision.json")

    with pytest.raises(RuntimeError, match="historical Suite v2 archive is read-only"):
        suite_v2_cli_main(["audit", "--component", "speed"])
    with pytest.raises(RuntimeError, match="historical Suite v2 archive cannot be exported"):
        suite_v2_cli_main(
            ["export", "--destination", str(tmp_path / "bundle")]
        )


def test_present_invalid_decision_never_silently_falls_back_to_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision_path = tmp_path / "invalid-decision.json"
    decision_path.write_text(json.dumps({"passed": False}) + "\n", encoding="utf-8")
    _redirect_final_decision(monkeypatch, decision_path)

    def reject(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("fixture decision is invalid")

    monkeypatch.setattr(reseal_v2, "validate_integrity_reseal_v2_decision", reject)

    with pytest.raises(RuntimeError, match="current-results default is unavailable"):
        resolve_suite_v2_cli_default_config(repo_root=ROOT)
    with pytest.raises(RuntimeError, match="current-results default is unavailable"):
        suite_v2_cli_main(["info"])


def test_overlay_config_cannot_be_loaded_before_a_final_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _redirect_final_decision(monkeypatch, tmp_path / "missing-decision.json")

    with pytest.raises(RuntimeError, match="final v2 decision is missing"):
        load_icl_suite_release(ROOT / SUITE_V2_CURRENT_RESULTS_OVERLAY_CONFIG_LOGICAL_PATH)


def test_accepted_final_decision_selects_and_activates_current_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decision_path = tmp_path / "accepted-decision.json"
    decision_path.write_text("{}\n", encoding="utf-8")
    _redirect_final_decision(monkeypatch, decision_path)
    monkeypatch.setattr(
        reseal_v2,
        "validate_integrity_reseal_v2_decision",
        lambda *_args, **_kwargs: {"passed": True, "reseal_id": reseal_v2.RESEAL_ID},
    )
    monkeypatch.setattr(
        suite_data,
        "_apply_current_results_overlay_materials",
        lambda _suite, **_kwargs: {
            "formal_reference_rows": 11,
            "components_with_formal_results": ["speed"],
            "scoreboard_extension_authorized": False,
            "formal_reference_rows_added": 0,
        },
    )

    default = resolve_suite_v2_cli_default_config(repo_root=ROOT)
    suite = load_icl_suite_release(default)
    activation = require_suite_membership_activation(suite, repo_root=ROOT)

    assert default == ROOT / SUITE_V2_CURRENT_RESULTS_OVERLAY_CONFIG_LOGICAL_PATH
    assert activation["status"] == (
        "suite_registration_passed_with_current_results_overlay_v1"
    )
    assert activation["current_results"]["formal_reference_rows"] == 11
