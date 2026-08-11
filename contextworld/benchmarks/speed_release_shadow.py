"""Build and audit the portable metadata shadow for Speed ICL v1.

The large training and evaluation payloads remain in the canonical artifact
store.  Only small JSON metadata files are copied into the repository.  This
module rewrites known machine-local paths to stable logical identities and
rejects every unknown absolute path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import tempfile
from typing import Any, Iterable


class SpeedMetadataPathError(ValueError):
    """Raised when Speed release metadata contains an unknown machine path."""


SPEED_INDEPENDENT_METADATA_FILES = (
    "artifacts/synthesis/catalogs/tworoom_speed_single_matched_v2.json",
    "artifacts/synthesis/reports/tworoom_speed_single_matched_v2.json",
    "artifacts/synthesis/catalogs/tworoom_speed_full_v1.json",
    "artifacts/synthesis/reports/tworoom_speed_full_v1.json",
    "artifacts/evaluation/history3/speed_continuous_causal_audit.json",
    "artifacts/evaluation/history3/speed_next_latent_v4/final_summary.json",
    "artifacts/evaluation/history3/speed_multistep_extrap_v5/final_summary.json",
    "artifacts/evaluation/history3/speed_isolated_v2/final_summary.json",
)

SPEED_INDEPENDENT_MANIFEST_FILES = (
    "artifacts/synthesis/manifests/tworoom_speed_single_matched_v2.jsonl",
    "artifacts/synthesis/manifests/tworoom_speed_full_v1.jsonl",
)

SPEED_SHARED_METADATA_FILES = (
    "artifacts/splits/tworoom_original_train_s3072_normalizer.json",
)

SPEED_CATALOG_DIRECTORIES = (
    "artifacts/evaluation/history3/speed_multistep_extrap_v5/catalogs",
    "artifacts/evaluation/history3/speed_isolated_v2/catalogs",
)

SPEED_FORMAL_RESULT_FILES = (
    "artifacts/evaluation/history3/speed_next_latent_v4/final_summary.json",
    "artifacts/evaluation/history3/speed_multistep_extrap_v5/final_summary.json",
    "artifacts/evaluation/history3/speed_isolated_v2/final_summary.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_fingerprint(path: Path) -> dict[str, Any]:
    files = sorted(value for value in path.rglob("*") if value.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for child in files:
        relative = child.relative_to(path).as_posix()
        size = child.stat().st_size
        child_hash = _sha256(child)
        total_bytes += size
        digest.update(
            f"{relative}\0{size}\0{child_hash}\n".encode("utf-8")
        )
    return {
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _lexical_absolute(value: str | Path) -> Path:
    return Path(os.path.abspath(Path(value).expanduser()))


def _is_absolute(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _relative_to(value: Path, root: Path) -> Path | None:
    try:
        return value.relative_to(root)
    except ValueError:
        return None


def sanitize_speed_release_metadata(
    value: Any,
    *,
    repo_root: Path,
    canonical_artifact_root: Path,
    original_h5: Path,
    stable_worldmodel_root: Path,
) -> Any:
    """Replace only the four frozen, known path roots in Speed metadata."""

    repo = _lexical_absolute(repo_root)
    canonical = _lexical_absolute(canonical_artifact_root)
    original = _lexical_absolute(original_h5)
    stablewm = _lexical_absolute(stable_worldmodel_root)

    def rewrite(item: Any, location: str) -> Any:
        if isinstance(item, dict):
            return {
                key: rewrite(child, f"{location}.{key}")
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [
                rewrite(child, f"{location}[{index}]")
                for index, child in enumerate(item)
            ]
        if not isinstance(item, str) or not _is_absolute(item):
            return item
        if PureWindowsPath(item).is_absolute() and not Path(item).is_absolute():
            raise SpeedMetadataPathError(
                f"{location}: unknown absolute metadata path: {item}"
            )
        path = _lexical_absolute(item)
        if path == original:
            return "upstream/lewm-tworooms/tworoom.h5"
        relative = _relative_to(path, canonical)
        if relative is not None:
            return (Path("artifacts") / relative).as_posix()
        relative = _relative_to(path, repo)
        if relative is not None:
            return relative.as_posix() if relative.parts else "."
        relative = _relative_to(path, stablewm)
        if relative is not None:
            identity = Path("upstream/stable-worldmodel")
            return (
                (identity / relative).as_posix()
                if relative.parts
                else identity.as_posix()
            )
        raise SpeedMetadataPathError(
            f"{location}: unknown absolute metadata path: {path}"
        )

    return rewrite(value, "$")


def absolute_json_paths(value: Any) -> list[dict[str, str]]:
    """Return every absolute path string and its JSON-tree location."""

    found: list[dict[str, str]] = []

    def walk(item: Any, location: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                walk(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{location}[{index}]")
        elif isinstance(item, str) and _is_absolute(item):
            found.append({"location": location, "value": item})

    walk(value, "$")
    return found


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _sanitize_json_file(
    source: Path,
    target: Path,
    *,
    repo_root: Path,
    canonical_artifact_root: Path,
    original_h5: Path,
    stable_worldmodel_root: Path,
) -> dict[str, Any]:
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    portable = sanitize_speed_release_metadata(
        source_payload,
        repo_root=repo_root,
        canonical_artifact_root=canonical_artifact_root,
        original_h5=original_h5,
        stable_worldmodel_root=stable_worldmodel_root,
    )
    remaining = absolute_json_paths(portable)
    if remaining:
        raise SpeedMetadataPathError(
            f"sanitized metadata still contains absolute paths: {remaining}"
        )
    source_hash = _sha256(source)
    _write_json_atomic(target, portable)
    return {
        "source_sha256": source_hash,
        "portable_sha256": _sha256(target),
        "absolute_path_count": 0,
    }


def _catalog_relative_files(path: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            child.relative_to(path).as_posix()
            for child in path.rglob("*")
            if child.is_file()
        )
    )


def copy_speed_catalog_shadow(
    source: Path,
    target: Path,
    *,
    repo_root: Path,
    canonical_artifact_root: Path,
    original_h5: Path,
    stable_worldmodel_root: Path,
) -> dict[str, Any]:
    """Copy a complete catalog directory, then sanitize its build report."""

    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_files = _catalog_relative_files(source)
    if "build_report.json" not in source_files:
        raise ValueError(f"Catalog directory lacks build_report.json: {source}")
    try:
        # This workspace filesystem rejects directory rename/link operations.
        # Copy directly, validate the complete directory, and remove the
        # incomplete target on every failure.  No canonical-root symlink is
        # ever created.
        shutil.copytree(source, target, copy_function=shutil.copy2)
        if _catalog_relative_files(target) != source_files:
            raise RuntimeError("Catalog copy changed the relative file set")
        payload_hashes = {}
        for relative in source_files:
            if relative == "build_report.json":
                continue
            source_hash = _sha256(source / relative)
            copied_hash = _sha256(target / relative)
            if copied_hash != source_hash:
                raise RuntimeError(f"Catalog payload changed: {relative}")
            payload_hashes[relative] = source_hash
        report = _sanitize_json_file(
            source / "build_report.json",
            target / "build_report.json",
            repo_root=repo_root,
            canonical_artifact_root=canonical_artifact_root,
            original_h5=original_h5,
            stable_worldmodel_root=stable_worldmodel_root,
        )
        for relative in source_files:
            if absolute_json_paths(
                json.loads((target / relative).read_text(encoding="utf-8"))
            ):
                raise SpeedMetadataPathError(
                    f"Portable catalog contains a machine path: {relative}"
                )
    except BaseException:
        if target.exists():
            shutil.rmtree(target)
        raise
    return {
        "path": target.relative_to(repo_root).as_posix(),
        "source_files": len(source_files),
        "payload_sha256": payload_hashes,
        "build_report": report,
        "tree": _tree_fingerprint(target),
    }


def expected_speed_shadow_files() -> tuple[str, ...]:
    catalog_files = []
    catalog_members = {
        SPEED_CATALOG_DIRECTORIES[0]: (
            "build_report.json",
            "extrapolation_high.json",
            "extrapolation_low.json",
            "seen_for_multi.json",
            "unseen_interpolation.json",
        ),
        SPEED_CATALOG_DIRECTORIES[1]: (
            "build_report.json",
            "calibration.json",
            "seen_for_multi.json",
            "unseen_interpolation.json",
        ),
    }
    for directory, names in catalog_members.items():
        catalog_files.extend(f"{directory}/{name}" for name in names)
    return tuple(
        sorted(
            set(SPEED_INDEPENDENT_METADATA_FILES)
            | set(SPEED_INDEPENDENT_MANIFEST_FILES)
            | set(SPEED_SHARED_METADATA_FILES)
            | set(catalog_files)
        )
    )


def owned_speed_shadow_files() -> tuple[str, ...]:
    """Return the Speed-owned subset, excluding the shared normalizer."""

    shared = set(SPEED_SHARED_METADATA_FILES)
    return tuple(
        logical
        for logical in expected_speed_shadow_files()
        if logical not in shared
    )


def selected_file_fingerprint(
    repo_root: Path,
    logical_paths: Iterable[str],
) -> dict[str, Any]:
    paths = tuple(sorted(set(str(value) for value in logical_paths)))
    digest = hashlib.sha256()
    total_bytes = 0
    missing = []
    for logical in paths:
        path = repo_root / logical
        if not path.is_file():
            missing.append(logical)
            continue
        size = path.stat().st_size
        child_hash = _sha256(path)
        total_bytes += size
        digest.update(f"{logical}\0{size}\0{child_hash}\n".encode("utf-8"))
    return {
        "files": len(paths) - len(missing),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "missing": missing,
    }


def build_speed_portable_shadow(
    *,
    repo_root: Path,
    canonical_artifact_root: Path,
    original_h5: Path,
    stable_worldmodel_root: Path,
) -> dict[str, Any]:
    """Build Speed-owned shadow files; the shared normalizer is not written."""

    repo = repo_root.resolve()
    canonical = canonical_artifact_root.resolve()
    catalogs = []
    for logical in SPEED_CATALOG_DIRECTORIES:
        relative = Path(*Path(logical).parts[1:])
        catalogs.append(
            copy_speed_catalog_shadow(
                canonical / relative,
                repo / logical,
                repo_root=repo,
                canonical_artifact_root=canonical,
                original_h5=original_h5,
                stable_worldmodel_root=stable_worldmodel_root,
            )
        )

    files = []
    for logical in SPEED_INDEPENDENT_METADATA_FILES:
        target = repo / logical
        relative = Path(*Path(logical).parts[1:])
        source = (
            target
            if logical.endswith("speed_continuous_causal_audit.json")
            else canonical / relative
        )
        if target.exists() and source != target:
            raise FileExistsError(target)
        if not source.is_file():
            raise FileNotFoundError(source)
        audit = _sanitize_json_file(
            source,
            target,
            repo_root=repo,
            canonical_artifact_root=canonical,
            original_h5=original_h5,
            stable_worldmodel_root=stable_worldmodel_root,
        )
        files.append({"path": logical, **audit})

    for logical in SPEED_INDEPENDENT_MANIFEST_FILES:
        target = repo / logical
        relative = Path(*Path(logical).parts[1:])
        source = canonical / relative
        if target.exists():
            raise FileExistsError(target)
        if not source.is_file():
            raise FileNotFoundError(source)
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            remaining = absolute_json_paths(json.loads(line))
            if remaining:
                raise SpeedMetadataPathError(
                    f"{logical}:{line_number} contains a machine path: "
                    f"{remaining}"
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if _sha256(target) != _sha256(source):
            raise RuntimeError(f"Manifest copy changed content: {logical}")
        files.append(
            {
                "path": logical,
                "source_sha256": _sha256(source),
                "portable_sha256": _sha256(target),
                "absolute_path_count": 0,
            }
        )

    built_paths = tuple(
        sorted(
            set(SPEED_INDEPENDENT_METADATA_FILES)
            | set(SPEED_INDEPENDENT_MANIFEST_FILES)
            | {
                f"{directory}/{member}"
                for directory in SPEED_CATALOG_DIRECTORIES
                for member in _catalog_relative_files(repo / directory)
            }
        )
    )
    fingerprint = selected_file_fingerprint(repo, built_paths)
    if fingerprint["missing"]:
        raise RuntimeError(f"Speed shadow is incomplete: {fingerprint['missing']}")
    return {
        "schema_version": 1,
        "status": "passed",
        "written_shared_normalizer": False,
        "files": files,
        "catalog_directories": catalogs,
        "built_file_set": fingerprint,
    }


def audit_speed_portable_shadow(
    release: dict[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    """Verify the exact portable shadow contract frozen in the release YAML."""

    specification = release.get("portable_shadow", {})
    expected_paths = expected_speed_shadow_files()
    owned_paths = owned_speed_shadow_files()
    fingerprint = selected_file_fingerprint(repo_root, owned_paths)
    absolute_paths = []
    for logical in expected_paths:
        path = repo_root / logical
        if path.is_file() and path.suffix == ".json":
            for row in absolute_json_paths(
                json.loads(path.read_text(encoding="utf-8"))
            ):
                absolute_paths.append({"path": logical, **row})
        elif path.is_file() and path.suffix == ".jsonl":
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                for row in absolute_json_paths(json.loads(line)):
                    absolute_paths.append(
                        {"path": logical, "line": line_number, **row}
                    )

    catalog_audits = {}
    expected_catalogs = specification.get("catalog_directories", {})
    for logical in SPEED_CATALOG_DIRECTORIES:
        path = repo_root / logical
        observed = _tree_fingerprint(path) if path.is_dir() else None
        expected = expected_catalogs.get(logical)
        catalog_audits[logical] = {
            "path": logical,
            "observed": observed,
            "expected": expected,
            "passed": observed == expected,
        }

    result_directories = {
        str(Path(logical).parent) for logical in SPEED_FORMAL_RESULT_FILES
    }
    formal_results_only = True
    result_directory_files = {}
    for logical in sorted(result_directories):
        directory = repo_root / logical
        observed = (
            sorted(
                child.name
                for child in directory.iterdir()
                if child.is_file()
            )
            if directory.is_dir()
            else []
        )
        result_directory_files[logical] = observed
        formal_results_only = formal_results_only and observed == [
            "final_summary.json"
        ]

    shared_audits = {}
    for logical in SPEED_SHARED_METADATA_FILES:
        path = repo_root / logical
        observed = _sha256(path) if path.is_file() else None
        expected = release["evaluation"]["normalizer_sha256"]
        shared_audits[logical] = {
            "path": logical,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "passed": observed == expected,
        }

    expected_fingerprint = specification.get("owned_file_tree", {})
    fingerprint_passed = bool(
        not fingerprint["missing"]
        and {
            key: fingerprint[key] for key in ("files", "bytes", "sha256")
        }
        == expected_fingerprint
    )
    passed = bool(
        fingerprint_passed
        and not absolute_paths
        and all(row["passed"] for row in catalog_audits.values())
        and all(row["passed"] for row in shared_audits.values())
        and formal_results_only
    )
    return {
        "path": "artifacts",
        "owned_file_tree": fingerprint,
        "expected_owned_file_tree": expected_fingerprint,
        "shared_files": shared_audits,
        "absolute_json_paths": absolute_paths,
        "catalog_directories": catalog_audits,
        "formal_result_directory_files": result_directory_files,
        "formal_results_only": formal_results_only,
        "passed": passed,
    }


__all__ = [
    "SPEED_CATALOG_DIRECTORIES",
    "SPEED_FORMAL_RESULT_FILES",
    "SPEED_INDEPENDENT_METADATA_FILES",
    "SPEED_SHARED_METADATA_FILES",
    "SpeedMetadataPathError",
    "absolute_json_paths",
    "audit_speed_portable_shadow",
    "build_speed_portable_shadow",
    "copy_speed_catalog_shadow",
    "expected_speed_shadow_files",
    "owned_speed_shadow_files",
    "sanitize_speed_release_metadata",
    "selected_file_fingerprint",
]
