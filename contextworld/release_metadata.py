"""Portable, fail-closed metadata helpers for benchmark releases.

Release metadata must remain meaningful after its directory is copied to a
different machine.  These helpers deliberately accept only two absolute path
roots:

* the release being written, represented by ``.``; and
* an optional frozen predecessor, represented by ``frozen_predecessor``.

Any other absolute path is rejected instead of being silently serialized.
The functions are purely lexical: neither allowed root needs to exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath
import tempfile
from typing import Any, Mapping


FROZEN_PREDECESSOR = "frozen_predecessor"


class NonPortableMetadataPathError(ValueError):
    """Raised when release metadata contains an unknown absolute path."""


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    """Return an absolute, normalized path without requiring it to exist."""

    return Path(os.path.abspath(Path(path).expanduser()))


def _is_absolute_path(value: str) -> bool:
    """Recognize native and Windows absolute paths in JSON string values."""

    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def portable_release_path(
    value: str | os.PathLike[str],
    *,
    release_root: str | os.PathLike[str],
    frozen_predecessor_root: str | os.PathLike[str] | None = None,
) -> str:
    """Convert one known absolute path to its release-local representation.

    Relative strings are returned unchanged.  Absolute paths below
    ``release_root`` become paths relative to ``.``.  Paths below the optional
    predecessor root collapse to the symbolic ``frozen_predecessor`` identity;
    their manifest/table SHA receipts carry the exact identity.  No filesystem
    lookup is performed.
    """

    raw = os.fspath(value)
    if not _is_absolute_path(raw):
        return Path(raw).as_posix() if isinstance(value, Path) else raw

    # PureWindowsPath cannot be compared safely with native POSIX roots.  A
    # Windows absolute path on a POSIX build is therefore unknown and rejected.
    if PureWindowsPath(raw).is_absolute() and not Path(raw).is_absolute():
        raise NonPortableMetadataPathError(
            f"unknown absolute metadata path: {raw}"
        )

    absolute = _lexical_absolute(raw)
    current = _lexical_absolute(release_root)
    try:
        relative = absolute.relative_to(current)
    except ValueError:
        pass
    else:
        return "." if not relative.parts else relative.as_posix()

    if frozen_predecessor_root is not None:
        predecessor = _lexical_absolute(frozen_predecessor_root)
        try:
            absolute.relative_to(predecessor)
        except ValueError:
            pass
        else:
            return FROZEN_PREDECESSOR

    raise NonPortableMetadataPathError(
        f"unknown absolute metadata path: {absolute}"
    )


def portable_release_metadata(
    value: Any,
    *,
    release_root: str | os.PathLike[str],
    frozen_predecessor_root: str | os.PathLike[str] | None = None,
) -> Any:
    """Recursively make JSON-compatible release metadata path-portable.

    Dictionaries and lists are copied recursively, so the input object is not
    mutated.  ``Path`` values are converted to strings.  If an unknown absolute
    path is encountered, the error reports its location in the JSON tree.
    """

    def rewrite(item: Any, location: str) -> Any:
        if isinstance(item, Mapping):
            return {
                key: rewrite(child, f"{location}.{key}")
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [
                rewrite(child, f"{location}[{index}]")
                for index, child in enumerate(item)
            ]
        if isinstance(item, tuple):
            return [
                rewrite(child, f"{location}[{index}]")
                for index, child in enumerate(item)
            ]
        if isinstance(item, os.PathLike):
            raw = os.fspath(item)
        elif isinstance(item, str) and _is_absolute_path(item):
            raw = item
        else:
            return item
        try:
            return portable_release_path(
                raw,
                release_root=release_root,
                frozen_predecessor_root=frozen_predecessor_root,
            )
        except NonPortableMetadataPathError as error:
            raise NonPortableMetadataPathError(
                f"{location}: {error}"
            ) from error

    return rewrite(value, "$")


def _validated_sha256(value: str, *, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return normalized


def frozen_predecessor_reference(
    *,
    manifest_sha256: str,
    table_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build the only predecessor identity stored in portable metadata.

    The returned receipt intentionally contains no filesystem path.  It can be
    audited after the predecessor directory has been archived or removed.
    """

    if not table_sha256:
        raise ValueError("table_sha256 must contain at least one table")
    return {
        "source": FROZEN_PREDECESSOR,
        "manifest_sha256": _validated_sha256(
            manifest_sha256,
            label="manifest_sha256",
        ),
        "table_sha256": {
            str(name): _validated_sha256(
                digest,
                label=f"table_sha256[{name!r}]",
            )
            for name, digest in sorted(table_sha256.items())
        },
    }


def write_portable_release_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    release_root: str | os.PathLike[str],
    frozen_predecessor_root: str | os.PathLike[str] | None = None,
) -> Any:
    """Atomically write one sanitized JSON metadata file.

    The temporary file is created beside the destination and replaced with
    ``os.replace``.  This function writes metadata only; it never reads or
    modifies Lance data.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    portable = portable_release_metadata(
        value,
        release_root=release_root,
        frozen_predecessor_root=frozen_predecessor_root,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        mode = (
            destination.stat().st_mode & 0o777
            if destination.exists()
            else 0o644
        )
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(portable, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return portable


__all__ = [
    "FROZEN_PREDECESSOR",
    "NonPortableMetadataPathError",
    "frozen_predecessor_reference",
    "portable_release_metadata",
    "portable_release_path",
    "write_portable_release_json",
]
