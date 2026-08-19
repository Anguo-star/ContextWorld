#!/usr/bin/env python3
"""Verify and write the derived 18-cell original-baseline matrix summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextworld.benchmarks.original_baseline_results import (
    DEFAULT_MATRIX_SUMMARY,
    build_original_baseline_summary,
)
from contextworld.paths import repository_root
from contextworld.synthesis.manifest import write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the frozen eight-checkpoint/eighteen-cell descriptive "
            "baseline matrix and derive its machine-readable summary."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--output", type=Path, default=DEFAULT_MATRIX_SUMMARY)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify all inputs and print compact status without writing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    payload = build_original_baseline_summary(repo_root=root)
    output = args.output.expanduser()
    if not output.is_absolute():
        output = root / output
    if not args.check_only:
        write_json(output.resolve(), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "counts": payload["counts"],
                "output": None if args.check_only else str(output.resolve()),
                "check_only": bool(args.check_only),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
