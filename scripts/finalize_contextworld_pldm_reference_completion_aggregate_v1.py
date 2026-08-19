#!/usr/bin/env python3
"""Check or exclusively finalize the four PLDM reference-completion record.

With no argument this command is read-only: it reports every currently known
missing or inconsistent receipt and creates nothing.  ``--finalize`` is the
one explicit write path; it refuses to overwrite the preregistered aggregate
freeze or either additive-scoreboard file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextworld.benchmarks.pldm_reference_completion_aggregate import (
    AGGREGATE_CONFIG,
    audit_completion_aggregate_readiness,
    validate_written_completion_aggregate_and_scoreboard,
    write_completion_aggregate_and_scoreboard,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-config", type=Path, default=AGGREGATE_CONFIG)
    parser.add_argument("--repo-root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="read-only readiness audit (default)")
    mode.add_argument("--finalize", action="store_true", help="exclusively create all preregistered outputs")
    mode.add_argument(
        "--validate-written",
        action="store_true",
        help="rebuild and verify existing exclusive outputs without writing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    kwargs = {
        "aggregate_config": args.aggregate_config,
        "repo_root": args.repo_root,
    }
    if args.finalize:
        result = write_completion_aggregate_and_scoreboard(**kwargs)
    elif args.validate_written:
        result = validate_written_completion_aggregate_and_scoreboard(**kwargs)
    else:
        result = audit_completion_aggregate_readiness(**kwargs)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.finalize and not args.validate_written and not result["ready"]:
        raise SystemExit("PLDM reference-completion aggregate is blocked")


if __name__ == "__main__":
    main()
