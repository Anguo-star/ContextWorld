"""Test-side three-level grading for byte-pinned runtime sources.

The production validators stay byte-exact: a release contract pin must keep
matching the blob it was published with.  What this helper adds is the audit
judgment the pin tests run *around* those validators, so that a registered,
hash-bound metadata transition (or a provably documentation-only edit) is
distinguished from a change that could alter evaluation behaviour:

* byte match                                            -> pass silently
* accepted transition registered in the correction yaml -> pass silently
* semantic fingerprint unchanged (recovered from history) -> pass with a warning
* anything else                                          -> the validator's own
  ``RuntimeError``/failure stands.

``monkeypatch.setattr``-ing a validator with :func:`graded_require_identity`
or :func:`graded_require_static_identity` keeps every other validation the
script performs; only the drift verdict is taught the lower tiers, and the
values returned for an accepted pin are the *declared* ones, so sealed chains
that recorded the historical identities stay internally consistent.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from contextworld.benchmarks.source_fingerprint import (
    ACCEPTED_TRANSITION,
    BYTE_MATCH,
    DRIFTED,
    SEMANTIC_UNCHANGED,
    grade_source_pin,
    load_accepted_pin_transitions,
    load_additive_scoring_extension_transitions,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE_PIN_CORRECTION = (
    ROOT / "configs/benchmark/contextworld_runtime_source_pin_correction_v1.yaml"
)
# Additive Public Test gate completion for speed / action delay / door.  Kept in
# its own record with its own loader because its claim is different from a
# behaviour-neutral metadata correction: new fields are emitted, every
# pre-existing field is bit-identical, and every sealed result was re-executed.
ADDITIVE_GATE_PIN_TRANSITION = (
    ROOT
    / "configs/benchmark/contextworld_additive_test_gate_completion_pin_transition_v1.yaml"
)


def accepted_pin_transitions() -> dict[tuple[str, str, str], dict[str, Any]]:
    """Load the correction registry shared by every pin-audit test."""

    accepted = dict(load_accepted_pin_transitions(RUNTIME_SOURCE_PIN_CORRECTION))
    overlap = accepted.keys() & load_additive_scoring_extension_transitions(
        ADDITIVE_GATE_PIN_TRANSITION
    ).keys()
    if overlap:
        raise ValueError(
            f"A pin is registered in both correction records: {sorted(overlap)}"
        )
    accepted.update(
        load_additive_scoring_extension_transitions(ADDITIVE_GATE_PIN_TRANSITION)
    )
    return accepted


def graded_failed_audit_files(
    audit: dict[str, Any],
    *,
    config_relative: str,
    repo_root: Path,
) -> tuple[set[str], set[str]]:
    """Split a per-file release audit into (graded failures, excused pins).

    ``audit["files"]`` rows carry ``path``/``expected_sha256`` from the
    release contract.  A repo-local source whose drift is excused by the
    three-level scale (registered acceptance, or an unchanged semantic
    fingerprint — the latter with a warning) leaves the failed set; every
    other row, including external and missing files, stays failed.  Returns
    the remaining failures plus the names of the excused entries.
    """

    failed: set[str] = set()
    excused: set[str] = set()
    accepted = accepted_pin_transitions()
    root = repo_root.resolve()
    for name, result in audit.get("files", {}).items():
        if result.get("passed"):
            continue
        # Audits report either an absolute "path" or a repo-relative
        # "logical_path" for the same pin; accept both shapes.
        raw = str(result.get("logical_path") or result.get("path") or "")
        path = Path(raw)
        try:
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            failed.add(name)
            continue
        grading = grade_source_pin(
            repo_root=root,
            config_relative=config_relative,
            path_relative=relative,
            pinned_sha256=str(result["expected_sha256"]),
            accepted=accepted,
        )
        if grading.status == SEMANTIC_UNCHANGED:
            warnings.warn(f"{name}: {grading.message}", stacklevel=2)
        if grading.passed:
            excused.add(name)
        else:
            failed.add(name)
    return failed, excused


def release_copy_with_graded_pins(
    release_path: Path,
    *,
    config_relative: str,
    repo_root: Path,
    destination: Path,
) -> Path:
    """Write a runtime-evidence audit copy of a release yaml with graded pins.

    The immutable release keeps its historical byte pins; this mirrors the
    established package-pin pattern of auditing a temporary copy in which the
    only rewritten fields are identity pins whose byte drift the three-level
    scale excuses — a registered accepted transition, or an unchanged semantic
    fingerprint (rewritten with a warning).  Any other drift raises here, so
    the audit that consumes the copy can never inherit a silent exemption.
    """

    payload = yaml.safe_load(release_path.read_text(encoding="utf-8"))
    accepted = accepted_pin_transitions()
    for specification in payload.get("identity", {}).values():
        if not (
            isinstance(specification, dict)
            and isinstance(specification.get("path"), str)
            and isinstance(specification.get("sha256"), str)
        ):
            continue
        grading = grade_source_pin(
            repo_root=repo_root,
            config_relative=config_relative,
            path_relative=specification["path"],
            pinned_sha256=specification["sha256"],
            accepted=accepted,
        )
        if grading.status == BYTE_MATCH:
            continue
        assert grading.status in (ACCEPTED_TRANSITION, SEMANTIC_UNCHANGED), (
            grading.message
        )
        if grading.status == SEMANTIC_UNCHANGED:
            warnings.warn(grading.message, stacklevel=2)
        specification["sha256"] = hashlib.sha256(
            (repo_root / specification["path"]).read_bytes()
        ).hexdigest()
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return destination


def stablewm_runtime_readiness(
    runtime_section: Mapping[str, Any],
) -> tuple[bool, str]:
    """Probe whether the pinned Stable-WorldModel runtime is available.

    Returns ``(ready, skip_reason)``.  A release or preregistration that pins
    the third-party runtime by file hash records that file at its pinned
    runtime ref; when the environment does not hold that state (no pinned
    worktree, or a sibling checkout at a different commit) validating against
    it is an environment precondition, not a code defect — the caller skips
    with the returned actionable reason instead of failing.  Both key
    spellings seen in the configs (``worktree``/``expected_ref`` and
    ``repo``/``commit``) are accepted, and ``pldm_config`` may be relative or
    absolute.
    """

    worktree = Path(
        str(runtime_section.get("worktree") or runtime_section.get("repo") or "")
    )
    expected_ref = str(
        runtime_section.get("expected_ref") or runtime_section.get("commit") or ""
    )
    raw_config = Path(str(runtime_section.get("pldm_config", "")))
    pldm_config = raw_config if raw_config.is_absolute() else worktree / raw_config
    if pldm_config.is_file():
        observed = hashlib.sha256(pldm_config.read_bytes()).hexdigest()
        if observed == str(runtime_section["pldm_config_sha256"]):
            return True, ""
    return False, (
        f"pinned Stable-WorldModel runtime is unavailable in this environment: "
        f"{pldm_config} does not match sha256 "
        f"{str(runtime_section['pldm_config_sha256'])[:12]}… (the release pins "
        f"runtime ref {expected_ref}); check out that ref to run this "
        f"validation, e.g. git -C <stable-worldmodel-checkout> checkout "
        f"{expected_ref}"
    )


def skip_when_stablewm_checkout_mismatches(error: RuntimeError) -> None:
    """Turn a smoke-pin checkout mismatch into an explicit, actionable skip."""

    message = str(error)
    if "differs from the smoke pin" not in message:
        raise error
    import re

    import pytest

    expected = re.search(r"expected ([0-9a-f]{40})", message)
    checkout = (
        f"git -C ../stable-worldmodel checkout {expected.group(1)}"
        if expected
        else "git -C ../stable-worldmodel checkout <expected_ref>"
    )
    pytest.skip(
        f"{message} — this test executes the pinned third-party runtime; "
        f"check out the pinned ref to run it ({checkout}), or provide the "
        "pinned worktree"
    )


def graded_bound_input_identity(
    *,
    config_relative: str,
    repo_root: Path,
    observe: Callable[[str], dict[str, Any]],
    same_identity: Callable[[Any, Any], bool],
) -> Callable[..., dict[str, Any]]:
    """Build a three-level replacement for a runner's bound-input checker.

    ``observe(path)`` returns the live identity mapping for a logical source
    path; ``same_identity(observed, declared)`` is the runner's own equality.
    On an excused tier the *declared* identity is returned so snapshots
    recorded with the historical bytes stay internally consistent.
    """

    accepted = accepted_pin_transitions()

    def replacement(value: Any, *, label: str) -> dict[str, Any]:
        raw = value.get("path") if isinstance(value, Mapping) else None
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"{label} lacks a source path")
        observed = observe(raw)
        if same_identity(observed, value):
            return observed
        pinned = value.get("sha256")
        if not isinstance(pinned, str):
            raise RuntimeError(f"{label} identity drifted")
        grading = grade_source_pin(
            repo_root=repo_root,
            config_relative=config_relative,
            path_relative=raw,
            pinned_sha256=pinned,
            accepted=accepted,
        )
        if _excuse(grading):
            warnings.warn(f"{label}: {grading.message}", stacklevel=3)
            return {
                "path": raw,
                "sha256": pinned,
                "size_bytes": int(value.get("size_bytes")),
            }
        raise RuntimeError(f"{label} identity drifted")

    return replacement


def install_graded_contract_require_identity(
    monkeypatch, target_module, *, config_relative: str
) -> None:
    """Give one script module a three-level ``require_identity``.

    The Speed PLDM freezer/evaluator scripts import ``require_identity`` from
    the shared development contract; this replaces it on the script module
    only, for the duration of one test, with the graded judgment.  Byte-exact
    pins behave identically, excused drift returns the declared (historical)
    identity so sealed chains stay consistent, and unregistered drift still
    raises the contract's own ``RuntimeError``.
    """

    from contextworld.benchmarks import (
        speed_pldm_infrastructure_development as contract,
    )

    def observe(specification, repo_root):
        return contract.identity(
            contract.resolve_source(
                specification["path"], repo_root=repo_root
            ),
            repo_root=repo_root,
        )

    monkeypatch.setattr(
        target_module,
        "require_identity",
        graded_require_identity(
            contract.require_identity,
            config_relative=config_relative,
            repo_root=contract.root(),
            observe=observe,
        ),
    )


def _excuse(grading) -> bool:
    return grading.status in (ACCEPTED_TRANSITION, SEMANTIC_UNCHANGED)


def graded_require_identity(
    original: Callable[..., dict[str, Any]],
    *,
    config_relative: str,
    repo_root: Path,
    observe: Callable[[dict[str, Any], Path | None], dict[str, Any]],
    accepted: Mapping[tuple[str, str, str], dict[str, Any]] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build a three-level replacement for the contract's ``require_identity``.

    ``observe`` must return the live identity mapping (path/sha256/size_bytes)
    for a specification without raising, mirroring what ``original`` compares
    against.  ``accepted`` overrides the correction registry, for tests.
    """

    accepted = accepted if accepted is not None else accepted_pin_transitions()
    repo_root_default = repo_root

    def replacement(
        specification: dict[str, Any], *, label: str, repo_root: Path | None = None
    ) -> dict[str, Any]:
        if not (
            isinstance(specification, dict)
            and isinstance(specification.get("path"), str)
            and specification["path"]
            and isinstance(specification.get("sha256"), str)
        ):
            raise ValueError(f"{label} needs non-empty path and sha256")
        effective_root = repo_root if repo_root is not None else repo_root_default
        observed = observe(specification, repo_root)
        if observed["sha256"] == specification["sha256"]:
            return original(specification, label=label, repo_root=repo_root)
        grading = grade_source_pin(
            repo_root=effective_root,
            config_relative=config_relative,
            path_relative=str(specification["path"]),
            pinned_sha256=str(specification["sha256"]),
            accepted=accepted,
        )
        if _excuse(grading):
            warnings.warn(f"{label}: {grading.message}", stacklevel=3)
            declared_size = specification.get("size_bytes")
            return {
                "path": observed["path"],
                "sha256": str(specification["sha256"]),
                "size_bytes": (
                    int(declared_size)
                    if declared_size is not None
                    else observed["size_bytes"]
                ),
            }
        raise RuntimeError(
            f"{label} identity drifted: expected={specification['sha256']}, "
            f"observed={observed['sha256']}"
        )

    return replacement


def graded_require_static_identity(
    *,
    config_relative: str,
    repo_root: Path,
    observe: Callable[[str], dict[str, Any]],
    accepted: Mapping[tuple[str, str, str], dict[str, Any]] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build a three-level replacement for ``_require_static_identity``.

    ``observe(path)`` must return the live identity mapping for a logical
    source path without raising.  ``accepted`` overrides the correction
    registry, for tests.
    """

    accepted = accepted if accepted is not None else accepted_pin_transitions()

    def replacement(value: Any, *, label: str) -> dict[str, Any]:
        if not (
            isinstance(value, dict)
            and isinstance(value.get("path"), str)
            and isinstance(value.get("sha256"), str)
            and isinstance(value.get("size_bytes"), int)
        ):
            raise ValueError(f"{label} needs path, SHA-256, and byte size")
        expected = str(value["sha256"])
        expected_size = int(value["size_bytes"])
        observed = observe(str(value["path"]))
        if (
            observed["sha256"] == expected
            and int(observed["size_bytes"]) == expected_size
        ):
            return observed
        grading = grade_source_pin(
            repo_root=repo_root,
            config_relative=config_relative,
            path_relative=str(value["path"]),
            pinned_sha256=expected,
            accepted=accepted,
        )
        if _excuse(grading):
            warnings.warn(f"{label}: {grading.message}", stacklevel=3)
            return {
                "path": observed["path"],
                "sha256": expected,
                "size_bytes": expected_size,
            }
        raise RuntimeError(f"{label} identity drifted")

    return replacement
