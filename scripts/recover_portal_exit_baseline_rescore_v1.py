#!/usr/bin/env python3
"""Write the additive float32 Portal Exit original-baseline recoveries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextworld.benchmarks.portal_exit_rescore_recovery import (
    DEFAULT_RECOVERY_RECEIPTS,
    recover_portal_exit_rescore_recovery,
)
from contextworld.paths import repository_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover only the frozen original Portal Exit LeWM/PLDM record "
            "rescores with the evaluator's float32 MSE aggregation."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument(
        "--family",
        choices=("all", "lewm", "pldm"),
        default="all",
        help="which frozen Portal Exit receipt to recover (default: both)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional receipt path; valid only when recovering one family",
    )
    args = parser.parse_args(argv)
    if args.output is not None and args.family == "all":
        parser.error("--output requires --family lewm or --family pldm")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    families = ("lewm", "pldm") if args.family == "all" else (args.family,)
    receipts = []
    for family in families:
        receipt = recover_portal_exit_rescore_recovery(
            family=family,
            repo_root=args.repo_root,
            output_path=args.output or DEFAULT_RECOVERY_RECEIPTS[family],
        )
        receipts.append(
            {
                "family": family,
                "status": receipt["status"],
                "verification_passed": receipt["verification"]["passed"],
                "model_gate_passed": receipt["verification"][
                    "recomputed_model_gate_passed"
                ],
                "output": receipt["output"]["path"],
            }
        )
    print(json.dumps({"status": "completed", "recoveries": receipts}, sort_keys=True))


if __name__ == "__main__":
    main()
