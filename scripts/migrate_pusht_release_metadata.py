#!/usr/bin/env python3
"""Prepare or commit one metadata-only PushT release portability migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.release_migration import (  # noqa: E402
    commit_prepared_migration,
    prepare_release_migration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("prepare", "commit"),
        help=(
            "prepare builds and verifies metadata-only staging; commit uses "
            "per-file atomic replacement with a lock and rollback backup"
        ),
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--frozen-predecessor-root", type=Path)
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument(
        "--lance-table",
        action="append",
        default=[],
        metavar="NAME=RELATIVE_PATH",
        help=(
            "logical table name and release-relative Lance directory; repeat "
            "for releases that do not use the default three tables"
        ),
    )
    parser.add_argument(
        "--semantic-source",
        action="append",
        default=[],
        metavar="SYMBOL|ABSOLUTE_PATH|DIGEST_ROLE|SHA256",
        help=(
            "replace one historical external path by a semantic identity and "
            "verified digest; repeat for each external source"
        ),
    )
    args = parser.parse_args()
    if args.mode == "commit" and args.frozen_predecessor_root is not None:
        parser.error("commit does not read the frozen predecessor")
    if args.mode == "commit" and (args.lance_table or args.semantic_source):
        parser.error(
            "commit reads table/source identities from the staged receipt"
        )
    return args


def _lance_tables(values: list[str]) -> dict[str, str] | None:
    if not values:
        return None
    result: dict[str, str] = {}
    for value in values:
        try:
            name, relative = value.split("=", 1)
        except ValueError as error:
            raise SystemExit(
                f"Invalid --lance-table {value!r}; expected NAME=PATH"
            ) from error
        if name in result:
            raise SystemExit(f"Duplicate --lance-table name: {name}")
        result[name] = relative
    return result


def _semantic_sources(values: list[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for value in values:
        parts = value.split("|")
        if len(parts) != 4:
            raise SystemExit(
                "Invalid --semantic-source; expected "
                "SYMBOL|ABSOLUTE_PATH|DIGEST_ROLE|SHA256"
            )
        symbol, path, digest_role, digest = parts
        if path in result:
            raise SystemExit(f"Duplicate semantic source path: {path}")
        result[path] = {
            "symbol": symbol,
            "digest_role": digest_role,
            "sha256": digest,
        }
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        result = prepare_release_migration(
            release_root=args.release_root,
            frozen_predecessor_root=args.frozen_predecessor_root,
            staging_root=args.staging_root,
            lance_tables=_lance_tables(args.lance_table),
            semantic_sources=_semantic_sources(args.semantic_source),
        )
    else:
        result = commit_prepared_migration(
            release_root=args.release_root,
            staging_root=args.staging_root,
            backup_root=args.backup_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
