#!/usr/bin/env python3
"""Audit the non-executable original-baseline CEM completion draft."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.benchmarks.original_baseline_cem_completion import (  # noqa: E402
    DEFAULT_ORIGINAL_BASELINE_CEM_COMPLETION_BLOCKED_AUDIT,
    DEFAULT_ORIGINAL_BASELINE_CEM_COMPLETION_DRAFT,
    audit_original_baseline_cem_completion_draft,
    write_blocked_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prereg",
        type=Path,
        default=DEFAULT_ORIGINAL_BASELINE_CEM_COMPLETION_DRAFT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ORIGINAL_BASELINE_CEM_COMPLETION_BLOCKED_AUDIT,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate and print the blocked audit without writing it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        payload = audit_original_baseline_cem_completion_draft(
            args.prereg, repo_root=REPO_ROOT
        )
    else:
        payload = write_blocked_audit(
            args.output,
            prereg_path=args.prereg,
            repo_root=REPO_ROOT,
        )
    summary = {
        "status": payload["status"],
        "freeze_generated": payload["freeze_generated"],
        "cem_execution_started": payload["cem_execution_started"],
        "counts": payload["counts"],
        "output": None if args.check_only else str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

