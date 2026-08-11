"""Metadata-only, atomic portability migration for benchmark releases."""

from __future__ import annotations

import copy
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
from contextlib import contextmanager
from typing import Any, Iterable, Mapping

import lance
import numpy as np

from contextworld.release_metadata import (
    FROZEN_PREDECESSOR,
    frozen_predecessor_reference,
    portable_release_metadata,
    write_portable_release_json,
)


METADATA_FILES = ("request.json", "manifest.json", "build_report.json")
LANCE_SPLITS = ("train", "loader_validation", "validation")
DEFAULT_LANCE_TABLES = {
    split: f"{split}.lance" for split in LANCE_SPLITS
}
PORTABILITY_RECEIPT = "portability_receipt.json"


def _normalized_lance_tables(
    lance_tables: Mapping[str, str | Path] | None,
) -> dict[str, str]:
    """Validate logical Lance names and release-relative table paths."""

    raw = DEFAULT_LANCE_TABLES if lance_tables is None else lance_tables
    if not raw:
        raise ValueError("lance_tables must contain at least one table")
    normalized: dict[str, str] = {}
    for logical_name, raw_path in raw.items():
        name = str(logical_name)
        path = Path(raw_path)
        if not name or name in normalized:
            raise ValueError(f"Invalid or duplicate Lance name: {name!r}")
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Lance table path must be release-relative: {raw_path}"
            )
        relative = path.as_posix()
        if relative in {"", "."}:
            raise ValueError(f"Invalid Lance table path: {raw_path}")
        normalized[name] = relative
    return normalized


def _validated_sha256(value: str, *, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return normalized


def _normalized_semantic_sources(
    semantic_sources: Mapping[str | Path, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]]:
    """Validate exact external-path replacements used by old metadata."""

    normalized: dict[str, dict[str, str]] = {}
    symbols: dict[str, dict[str, str]] = {}
    for raw_path, raw_specification in (semantic_sources or {}).items():
        path = os.path.abspath(Path(raw_path).expanduser())
        if not Path(path).is_absolute():
            raise ValueError(f"Semantic source path is not absolute: {raw_path}")
        symbol = str(raw_specification.get("symbol", ""))
        digest_role = str(
            raw_specification.get("digest_role", "content_sha256")
        )
        if not symbol or Path(symbol).is_absolute():
            raise ValueError(f"Invalid semantic source symbol: {symbol!r}")
        if digest_role not in {
            "content_sha256",
            "file_sha256",
            "manifest_sha256",
            "tree_sha256",
        }:
            raise ValueError(
                f"Unsupported semantic source digest role: {digest_role}"
            )
        specification = {
            "symbol": symbol,
            "digest_role": digest_role,
            "sha256": _validated_sha256(
                raw_specification.get("sha256", ""),
                label=f"semantic_sources[{path!r}].sha256",
            ),
        }
        prior = symbols.get(symbol)
        if prior is not None and prior != specification:
            raise ValueError(
                f"Semantic source symbol has conflicting identities: {symbol}"
            )
        symbols[symbol] = specification
        normalized[path] = specification
    return normalized


def _rewrite_semantic_sources(
    value: Any,
    *,
    semantic_sources: Mapping[str, Mapping[str, str]],
    used_symbols: set[str],
) -> Any:
    """Replace exact machine paths by a nearby semantic name and digest."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        digest_overrides: dict[str, str] = {}
        for key, child in value.items():
            specification = None
            if isinstance(child, str) and (
                Path(child).is_absolute()
                or PureWindowsPath(child).is_absolute()
            ):
                specification = semantic_sources.get(
                    os.path.abspath(Path(child).expanduser())
                )
            if specification is None:
                result[key] = _rewrite_semantic_sources(
                    child,
                    semantic_sources=semantic_sources,
                    used_symbols=used_symbols,
                )
                continue

            symbol = specification["symbol"]
            used_symbols.add(symbol)
            if key in {"path", "root"}:
                existing_source = result.get("source", value.get("source"))
                if existing_source is not None and existing_source != symbol:
                    raise ValueError(
                        "Cannot replace path/root beside a conflicting "
                        f"source field: {existing_source!r}"
                    )
                result["source"] = symbol
                digest_key = specification["digest_role"]
            else:
                result[key] = symbol
                digest_key = f"{key}_sha256"
            digest_overrides[digest_key] = specification["sha256"]
        result.update(digest_overrides)
        return result
    if isinstance(value, list):
        return [
            _rewrite_semantic_sources(
                child,
                semantic_sources=semantic_sources,
                used_symbols=used_symbols,
            )
            for child in value
        ]
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def model_visible_payload_sha256(table_path: Path) -> dict[str, Any]:
    """Hash exactly the stored pixels/actions consumed by reference loaders."""

    dataset = lance.dataset(str(table_path))
    scanner = dataset.scanner(
        columns=["episode_idx", "step_idx", "pixels", "action"]
    )
    digest = hashlib.sha256()
    row_count = 0
    for batch in scanner.to_batches():
        episodes = np.asarray(batch.column("episode_idx"), dtype=np.int32)
        steps = np.asarray(batch.column("step_idx"), dtype=np.int32)
        pixels = batch.column("pixels").to_pylist()
        actions = np.asarray(
            batch.column("action").to_pylist(), dtype=np.float32
        ).reshape(-1, 2)
        for episode, step, pixel, action in zip(
            episodes,
            steps,
            pixels,
            actions,
            strict=True,
        ):
            digest.update(
                np.asarray([episode, step], dtype=np.int32).tobytes()
            )
            digest.update(np.asarray([len(pixel)], dtype=np.int64).tobytes())
            digest.update(pixel)
            digest.update(np.ascontiguousarray(action).tobytes())
            row_count += 1
    return {"row_count": row_count, "sha256": digest.hexdigest()}


def lance_table_identity(table_path: Path) -> dict[str, Any]:
    files = sorted(value for value in table_path.rglob("*") if value.is_file())
    visible = model_visible_payload_sha256(table_path)
    row_count = int(lance.dataset(str(table_path)).count_rows())
    if visible["row_count"] != row_count:
        raise RuntimeError(
            f"Model-visible scan row count changed for {table_path}: "
            f"scanner={visible['row_count']} dataset={row_count}"
        )
    return {
        "table": table_path.name,
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "tree_sha256": directory_sha256(table_path),
        "row_count": row_count,
        "model_visible_payload_sha256": visible["sha256"],
    }


def _lance_file_paths(
    root: Path,
    lance_tables: Mapping[str, str | Path] | None = None,
) -> Iterable[Path]:
    for relative in _normalized_lance_tables(lance_tables).values():
        table = root / relative
        if not table.is_dir():
            raise FileNotFoundError(f"Missing Lance table: {table}")
        yield from sorted(
            value for value in table.rglob("*") if value.is_file()
        )


def _hardlink_lance_audit(
    source: Path,
    candidate: Path,
    lance_tables: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    source_files = {
        path.relative_to(source).as_posix(): path
        for path in _lance_file_paths(source, lance_tables)
    }
    candidate_files = {
        path.relative_to(candidate).as_posix(): path
        for path in _lance_file_paths(candidate, lance_tables)
    }
    same_names = source_files.keys() == candidate_files.keys()
    hardlinked = 0
    if same_names:
        for name, source_path in source_files.items():
            candidate_path = candidate_files[name]
            source_stat = source_path.stat()
            candidate_stat = candidate_path.stat()
            if (
                source_stat.st_dev == candidate_stat.st_dev
                and source_stat.st_ino == candidate_stat.st_ino
            ):
                hardlinked += 1
    passed = bool(same_names and hardlinked == len(source_files))
    return {
        "source_file_count": len(source_files),
        "candidate_file_count": len(candidate_files),
        "same_relative_paths": same_names,
        "hardlinked_file_count": hardlinked,
        "passed": passed,
    }


def _hardlink_copy(source: str, destination: str) -> str:
    os.link(source, destination, follow_symlinks=False)
    return destination


def hardlink_clone_release(source: Path, destination: Path) -> None:
    """Clone a release on the same filesystem without copying payload bytes."""

    source = source.resolve()
    destination = Path(os.path.abspath(destination))
    if destination.exists():
        raise FileExistsError(f"Staging path already exists: {destination}")
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise RuntimeError("Release and staging parent are not on one filesystem")
    try:
        shutil.copytree(
            source,
            destination,
            copy_function=_hardlink_copy,
            symlinks=True,
        )
    except BaseException:
        # ``copytree`` creates directories before the first link attempt.  A
        # failed preparation must never leave a candidate that could later be
        # mistaken for a validated staging tree.
        if destination.exists():
            shutil.rmtree(destination)
        raise


def _collapse_predecessor_path_keys(value: Any) -> Any:
    """Remove path-shaped predecessor keys after lexical sanitization."""

    if isinstance(value, dict):
        result = {
            key: _collapse_predecessor_path_keys(child)
            for key, child in value.items()
        }
        predecessor_seen = False
        for key in ("source_root", "legacy_table"):
            if result.get(key) == FROZEN_PREDECESSOR:
                result.pop(key)
                predecessor_seen = True
        if result.get("root") == FROZEN_PREDECESSOR:
            result.pop("root")
            predecessor_seen = True
        if predecessor_seen:
            result["source"] = FROZEN_PREDECESSOR
        return result
    if isinstance(value, list):
        return [_collapse_predecessor_path_keys(child) for child in value]
    return value


def _frozen_table_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for split in ("loader_validation", "validation"):
        reuse = manifest.get("splits", {}).get(split, {}).get(
            "frozen_split_reuse", {}
        )
        digest = reuse.get("source_table_sha256")
        if not isinstance(digest, str):
            raise ValueError(
                f"Missing frozen predecessor table SHA for {split}"
            )
        hashes[split] = digest
    return hashes


def _normalize_top_predecessor_reference(
    request: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    table_hashes: dict[str, str] | None = None
    for field in (
        "evaluation_reuse_source",
        "evaluation_tables_reused_byte_for_byte",
    ):
        specification = request.get(field)
        if not isinstance(specification, dict):
            continue
        if table_hashes is None:
            table_hashes = _frozen_table_hashes(manifest)
        manifest_digest = specification.get(
            "manifest_sha256",
            specification.get("source_manifest_sha256"),
        )
        if not isinstance(manifest_digest, str):
            raise ValueError(
                f"Missing frozen predecessor manifest SHA in {field}"
            )
        normalized = frozen_predecessor_reference(
            manifest_sha256=manifest_digest,
            table_sha256=table_hashes,
        )
        if isinstance(specification.get("splits"), list):
            normalized["splits"] = list(specification["splits"])
        request[field] = normalized
        manifest[field] = copy.deepcopy(normalized)


def portable_metadata_payloads(
    *,
    request: dict[str, Any],
    manifest: dict[str, Any],
    build_report: dict[str, Any],
    release_root: Path,
    frozen_predecessor_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return coupled portable metadata with request/manifest hashes updated."""

    portable_request = _collapse_predecessor_path_keys(
        portable_release_metadata(
            request,
            release_root=release_root,
            frozen_predecessor_root=frozen_predecessor_root,
        )
    )
    portable_manifest = _collapse_predecessor_path_keys(
        portable_release_metadata(
            manifest,
            release_root=release_root,
            frozen_predecessor_root=frozen_predecessor_root,
        )
    )
    _normalize_top_predecessor_reference(
        portable_request,
        portable_manifest,
    )
    for key, child in portable_request.items():
        portable_manifest[key] = copy.deepcopy(child)
    portable_manifest["request_sha256"] = canonical_json_sha256(
        portable_request
    )

    report_source = copy.deepcopy(build_report)
    # A historical temporary build directory is not part of the release
    # identity.  ``root`` always describes the directory that contains this
    # report after publication.
    report_source["root"] = "."
    portable_report = _collapse_predecessor_path_keys(
        portable_release_metadata(
            report_source,
            release_root=release_root,
            frozen_predecessor_root=frozen_predecessor_root,
        )
    )
    portable_report["root"] = "."
    # The manifest file hash is injected by ``prepare_release_migration`` after
    # the canonical portable manifest has been written.
    return portable_request, portable_manifest, portable_report


def _absolute_paths(value: Any, *, location: str = "$") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(
                _absolute_paths(child, location=f"{location}.{key}")
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(
                _absolute_paths(child, location=f"{location}[{index}]")
            )
    elif isinstance(value, str) and (
        Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        rows.append({"location": location, "value": value})
    return rows


def absolute_json_path_audit(root: Path) -> dict[str, Any]:
    files = sorted(root.rglob("*.json"))
    rows: list[dict[str, str]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in _absolute_paths(payload):
            rows.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    **row,
                }
            )
    return {
        "json_files": [path.relative_to(root).as_posix() for path in files],
        "absolute_path_count": len(rows),
        "absolute_paths": rows,
        "passed": not rows,
    }


def _release_metadata_files(root: Path) -> tuple[str, ...]:
    names = []
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root)
        if PORTABILITY_RECEIPT == relative.as_posix():
            continue
        if any(part.endswith(".lance") for part in relative.parts):
            continue
        names.append(relative.as_posix())
    missing = [name for name in METADATA_FILES if name not in names]
    if missing:
        raise FileNotFoundError(
            f"Release is missing required metadata files: {missing}"
        )
    extras = sorted(name for name in names if name not in METADATA_FILES)
    return (*METADATA_FILES, *extras)


def _metadata_sha256(
    root: Path,
    names: Iterable[str],
) -> dict[str, str]:
    return {name: file_sha256(root / name) for name in names}


def _lance_identities(
    root: Path,
    lance_tables: Mapping[str, str | Path] | None = None,
) -> dict[str, dict[str, Any]]:
    tables = _normalized_lance_tables(lance_tables)
    return {
        name: lance_table_identity(root / relative)
        for name, relative in tables.items()
    }


def _refresh_local_manifest_references(
    value: Any,
    *,
    manifest_sha256: str,
) -> Any:
    """Refresh SHA companions for portable references to ``manifest.json``."""

    if isinstance(value, dict):
        result = {
            key: _refresh_local_manifest_references(
                child,
                manifest_sha256=manifest_sha256,
            )
            for key, child in value.items()
        }
        for key, child in result.items():
            if child == "manifest.json":
                digest_key = f"{key}_sha256"
                if digest_key in result:
                    result[digest_key] = manifest_sha256
        return result
    if isinstance(value, list):
        return [
            _refresh_local_manifest_references(
                child,
                manifest_sha256=manifest_sha256,
            )
            for child in value
        ]
    return value


def _identity_equal(
    before: dict[str, Any],
    after: dict[str, Any],
) -> bool:
    return before == after


def prepare_release_migration(
    *,
    release_root: Path,
    frozen_predecessor_root: Path | None = None,
    staging_root: Path | None = None,
    lance_tables: Mapping[str, str | Path] | None = None,
    semantic_sources: Mapping[
        str | Path,
        Mapping[str, str],
    ] | None = None,
) -> dict[str, Any]:
    """Create a metadata-only staging and validate in-place Lance identity."""

    release_root = release_root.expanduser().resolve()
    predecessor = (
        None
        if frozen_predecessor_root is None
        else Path(os.path.abspath(frozen_predecessor_root.expanduser()))
    )
    tables = _normalized_lance_tables(lance_tables)
    sources = _normalized_semantic_sources(semantic_sources)
    staging = (
        release_root.with_name(f".{release_root.name}.portability-staging")
        if staging_root is None
        else Path(os.path.abspath(staging_root.expanduser()))
    )
    metadata_names = _release_metadata_files(release_root)
    if (release_root / PORTABILITY_RECEIPT).exists():
        raise FileExistsError(
            f"Release is already portable: {release_root / PORTABILITY_RECEIPT}"
        )
    if staging.exists():
        raise FileExistsError(f"Staging path already exists: {staging}")

    old_metadata = _metadata_sha256(release_root, metadata_names)
    before_lance = _lance_identities(release_root, tables)
    staging.mkdir()
    try:
        raw_payloads = {
            name: json.loads(
                (release_root / name).read_text(encoding="utf-8")
            )
            for name in metadata_names
        }
        used_symbols: set[str] = set()
        source_payloads = {
            name: _rewrite_semantic_sources(
                payload,
                semantic_sources=sources,
                used_symbols=used_symbols,
            )
            for name, payload in raw_payloads.items()
        }
        declared_symbols = {
            specification["symbol"] for specification in sources.values()
        }
        if used_symbols != declared_symbols:
            raise ValueError(
                "Semantic source mappings were not used exactly: "
                f"declared={sorted(declared_symbols)} "
                f"used={sorted(used_symbols)}"
            )
        request, manifest, report = portable_metadata_payloads(
            request=source_payloads["request.json"],
            manifest=source_payloads["manifest.json"],
            build_report=source_payloads["build_report.json"],
            release_root=release_root,
            frozen_predecessor_root=predecessor,
        )
        request = write_portable_release_json(
            staging / "request.json",
            request,
            release_root=release_root,
            frozen_predecessor_root=predecessor,
        )
        manifest = write_portable_release_json(
            staging / "manifest.json",
            manifest,
            release_root=release_root,
            frozen_predecessor_root=predecessor,
        )
        report["manifest_sha256"] = file_sha256(staging / "manifest.json")
        report = write_portable_release_json(
            staging / "build_report.json",
            report,
            release_root=release_root,
            frozen_predecessor_root=predecessor,
        )
        manifest_digest = file_sha256(staging / "manifest.json")
        for name in metadata_names:
            if name in METADATA_FILES:
                continue
            portable_extra = _collapse_predecessor_path_keys(
                portable_release_metadata(
                    source_payloads[name],
                    release_root=release_root,
                    frozen_predecessor_root=predecessor,
                )
            )
            portable_extra = _refresh_local_manifest_references(
                portable_extra,
                manifest_sha256=manifest_digest,
            )
            write_portable_release_json(
                staging / name,
                portable_extra,
                release_root=release_root,
                frozen_predecessor_root=predecessor,
            )

        new_metadata = _metadata_sha256(staging, metadata_names)
        # The tables remain at their original paths.  This second complete read
        # is the migration-after identity, not a shortcut through cached values.
        after_lance = _lance_identities(release_root, tables)
        lance_equal = {
            name: _identity_equal(before_lance[name], after_lance[name])
            for name in tables
        }
        if not all(lance_equal.values()):
            raise RuntimeError(
                f"In-place Lance identity changed while staging: {lance_equal}"
            )

        receipt = {
            "schema_version": 1,
            "migration": "metadata_only_portability_v1",
            "status": "passed",
            "atomic_switch_ready": True,
            "release_root": ".",
            "metadata_exchange_order": list(metadata_names),
            "atomicity": {
                "mode": (
                    "per_file_atomic_with_release_lock_and_"
                    "transactional_rollback"
                ),
                "directory_level_atomic": False,
                "fixed_exchange_order": list(metadata_names),
                "fsync_after_each_replace": True,
                "reverse_order_rollback": True,
            },
            "metadata_sha256": {
                name: {
                    "before": old_metadata[name],
                    "after": new_metadata[name],
                    "changed": old_metadata[name] != new_metadata[name],
                }
                for name in metadata_names
            },
            "request_contract": {
                "canonical_sha256": canonical_json_sha256(request),
                "manifest_request_sha256": manifest["request_sha256"],
                "passed": (
                    canonical_json_sha256(request)
                    == manifest["request_sha256"]
                ),
            },
            "manifest_contract": {
                "file_sha256": new_metadata["manifest.json"],
                "build_report_manifest_sha256": report["manifest_sha256"],
                "passed": (
                    new_metadata["manifest.json"]
                    == report["manifest_sha256"]
                ),
            },
            "lance_storage_contract": {
                "mode": "in_place_read_only",
                "directories_moved": False,
                "files_rewritten": False,
                "passed": True,
            },
            "lance_tables": {
                name: {
                    "relative_path": tables[name],
                    "migration_before": before_lance[name],
                    "migration_after": after_lance[name],
                    "identical": lance_equal[name],
                }
                for name in tables
            },
            "semantic_sources": {
                specification["symbol"]: {
                    specification["digest_role"]: specification["sha256"],
                }
                for specification in sources.values()
            },
            "absolute_path_audit": {
                "json_files": [*metadata_names, PORTABILITY_RECEIPT],
                "absolute_path_count": 0,
                "passed": True,
            },
            "passed": bool(
                all(lance_equal.values())
                and canonical_json_sha256(request)
                == manifest["request_sha256"]
                and new_metadata["manifest.json"]
                == report["manifest_sha256"]
            ),
        }
        if not receipt["passed"] or _absolute_paths(receipt):
            raise RuntimeError(
                "Portability receipt did not pass before writing"
            )
        write_portable_release_json(
            staging / PORTABILITY_RECEIPT,
            receipt,
            release_root=release_root,
            frozen_predecessor_root=predecessor,
        )
        absolute_audit = absolute_json_path_audit(staging)
        if not absolute_audit["passed"]:
            raise RuntimeError(
                f"Portable staging contains absolute paths: {absolute_audit}"
            )
        return {
            "schema_version": 1,
            "status": "prepared",
            "release_root": str(release_root),
            "staging_root": str(staging),
            "receipt": receipt,
            "absolute_path_audit": absolute_audit,
            "passed": True,
        }
    except BaseException:
        shutil.rmtree(staging)
        raise


def _rename_exchange(left: Path, right: Path) -> None:
    """Atomically exchange two same-filesystem paths with renameat2."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is unavailable; refusing non-atomic switch")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(left),
        -100,
        os.fsencode(right),
        2,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"{left} <-> {right}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_file_fsynced(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    _fsync_file(destination)
    _fsync_directory(destination.parent)


def _atomic_replace(staged: Path, active: Path) -> None:
    """Atomically replace one active metadata file on the same filesystem."""

    os.replace(staged, active)


@contextmanager
def release_advisory_lock(release_root: Path):
    """Hold a non-blocking release-level migration lock."""

    lock_path = release_root.with_name(f".{release_root.name}.portability.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another portability migration holds {lock_path}"
            ) from error
        yield lock_path
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _remove_empty_tree(path: Path) -> None:
    directories = sorted(
        (value for value in path.rglob("*") if value.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for directory in directories:
        directory.rmdir()
    path.rmdir()


def _validate_prepared_candidate(
    release_root: Path,
    staging_root: Path,
) -> dict[str, Any]:
    receipt = json.loads(
        (staging_root / PORTABILITY_RECEIPT).read_text(encoding="utf-8")
    )
    if receipt.get("passed") is not True:
        raise RuntimeError("Staged portability receipt is not passed")
    metadata_names = tuple(receipt["metadata_exchange_order"])
    if metadata_names[: len(METADATA_FILES)] != METADATA_FILES:
        raise RuntimeError("Staged metadata exchange order is not canonical")
    if set(metadata_names) != set(receipt["metadata_sha256"]):
        raise RuntimeError("Staged metadata receipt file set is inconsistent")
    if tuple(_release_metadata_files(release_root)) != metadata_names:
        raise RuntimeError("Source metadata file set changed after staging")
    lance_tables = {
        name: specification.get("relative_path", f"{name}.lance")
        for name, specification in receipt["lance_tables"].items()
    }
    lance_tables = _normalized_lance_tables(lance_tables)
    current_metadata = _metadata_sha256(release_root, metadata_names)
    staged_metadata = _metadata_sha256(staging_root, metadata_names)
    for name in metadata_names:
        specification = receipt["metadata_sha256"][name]
        if current_metadata[name] != specification["before"]:
            raise RuntimeError(f"Source metadata changed after staging: {name}")
        if staged_metadata[name] != specification["after"]:
            raise RuntimeError(f"Staged metadata changed after validation: {name}")
    current_lance = _lance_identities(release_root, lance_tables)
    for name in lance_tables:
        specification = receipt["lance_tables"][name]
        if current_lance[name] != specification["migration_before"]:
            raise RuntimeError(f"Source Lance changed after staging: {name}")
    absolute_audit = absolute_json_path_audit(staging_root)
    if not absolute_audit["passed"]:
        raise RuntimeError("Staged JSON contains an absolute path")
    return {
        "receipt": receipt,
        "source_metadata": current_metadata,
        "staged_metadata": staged_metadata,
        "lance_tables": current_lance,
        "lance_table_paths": lance_tables,
        "metadata_names": metadata_names,
        "absolute_path_audit": absolute_audit,
        "passed": True,
    }


def commit_prepared_migration(
    *,
    release_root: Path,
    staging_root: Path | None = None,
    backup_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically activate a validated candidate and retain the old release."""

    release_root = release_root.expanduser().resolve()
    staging = (
        release_root.with_name(f".{release_root.name}.portability-staging")
        if staging_root is None
        else Path(os.path.abspath(staging_root.expanduser()))
    )
    with release_advisory_lock(release_root):
        if (release_root / PORTABILITY_RECEIPT).exists():
            raise FileExistsError(
                "Active release already has a portability receipt; "
                "refusing a second commit"
            )
        validation = _validate_prepared_candidate(release_root, staging)
        metadata_names = validation["metadata_names"]
        old_manifest = validation["source_metadata"]["manifest.json"]
        backup = (
            release_root.with_name(
                f"{release_root.name}.pre-portable-backup-"
                f"{old_manifest[:12]}"
            )
            if backup_root is None
            else Path(os.path.abspath(backup_root.expanduser()))
        )
        if backup.exists():
            raise FileExistsError(f"Backup path already exists: {backup}")

        backup.mkdir()
        for name in metadata_names:
            _copy_file_fsynced(release_root / name, backup / name)
        if _metadata_sha256(backup, metadata_names) != validation[
            "source_metadata"
        ]:
            raise RuntimeError("Rollback backup metadata verification failed")
        _fsync_directory(backup)
        _fsync_directory(backup.parent)

        replaced: list[str] = []
        receipt_installed = False
        try:
            # The filesystem does not support directory exchange or hardlinks.
            # Each replace is still atomic; the advisory lock, fsynced backup,
            # and reverse rollback make the fixed-order sequence transactional.
            for name in metadata_names:
                _atomic_replace(staging / name, release_root / name)
                replaced.append(name)
                _fsync_directory((release_root / name).parent)
                _fsync_directory((staging / name).parent)

            _atomic_replace(
                staging / PORTABILITY_RECEIPT,
                release_root / PORTABILITY_RECEIPT,
            )
            receipt_installed = True
            _fsync_directory(release_root)
            _fsync_directory(staging)

            active_metadata = _metadata_sha256(
                release_root,
                metadata_names,
            )
            if active_metadata != validation["staged_metadata"]:
                raise RuntimeError(
                    "Activated metadata differs from staged metadata"
                )
            active_lance = _lance_identities(
                release_root,
                validation["lance_table_paths"],
            )
            receipt = validation["receipt"]
            for split in validation["lance_table_paths"]:
                if (
                    active_lance[split]
                    != receipt["lance_tables"][split]["migration_after"]
                ):
                    raise RuntimeError(
                        f"Lance identity changed across switch: {split}"
                    )
            if not absolute_json_path_audit(release_root)["passed"]:
                raise RuntimeError("Activated JSON is not portable")
            _remove_empty_tree(staging)
            _fsync_directory(release_root.parent)
        except BaseException:
            if receipt_installed:
                _atomic_replace(
                    release_root / PORTABILITY_RECEIPT,
                    staging / PORTABILITY_RECEIPT,
                )
            for name in reversed(replaced):
                # Preserve the portable candidate for inspection/retry, then
                # atomically restore the active file from the fsynced backup.
                _copy_file_fsynced(release_root / name, staging / name)
                restore = release_root / (
                    f".{name.replace('/', '__')}.restore.tmp"
                )
                _copy_file_fsynced(backup / name, restore)
                _atomic_replace(restore, release_root / name)
                _fsync_directory((release_root / name).parent)
                _fsync_directory((staging / name).parent)
            _fsync_directory(release_root)
            _fsync_directory(staging)
            raise

        return {
            "schema_version": 1,
            "status": "committed",
            "release_root": str(release_root),
            "backup_root": str(backup),
            "old_manifest_sha256": old_manifest,
            "new_manifest_sha256": active_metadata["manifest.json"],
            "portability_receipt_sha256": file_sha256(
                release_root / PORTABILITY_RECEIPT
            ),
            "lance_tables": active_lance,
            "passed": True,
        }


__all__ = [
    "LANCE_SPLITS",
    "METADATA_FILES",
    "PORTABILITY_RECEIPT",
    "absolute_json_path_audit",
    "canonical_json_sha256",
    "commit_prepared_migration",
    "directory_sha256",
    "file_sha256",
    "hardlink_clone_release",
    "lance_table_identity",
    "model_visible_payload_sha256",
    "portable_metadata_payloads",
    "prepare_release_migration",
]
