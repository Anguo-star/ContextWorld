#!/usr/bin/env python3
"""Create the additive float32 recovery receipt for Action Strength LeWM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextworld.benchmarks.action_strength_rescore_recovery import (
    DEFAULT_FREEZE_RECEIPT,
    DEFAULT_PREREGISTRATION,
    DEFAULT_RAW_RECEIPT,
    DEFAULT_RECOVERY_RECEIPT,
    recover_action_strength_lewm_baseline,
)
from contextworld.benchmarks.action_strength_icl_data import (
    DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
)
from contextworld.paths import repository_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover only the frozen original PushT LeWM Action Strength "
            "rescore with the evaluator's float32 MSE aggregation."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--raw-receipt", type=Path, default=DEFAULT_RAW_RECEIPT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--freeze-receipt", type=Path, default=DEFAULT_FREEZE_RECEIPT)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RECOVERY_RECEIPT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = recover_action_strength_lewm_baseline(
        repo_root=args.repo_root,
        raw_receipt=args.raw_receipt,
        preregistration=args.preregistration,
        freeze_receipt=args.freeze_receipt,
        release_config=args.release_config,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "recovery_id": receipt["recovery_id"],
                "verification_passed": receipt["verification"]["passed"],
                "model_gate_passed": receipt["verification"][
                    "recomputed_model_gate_passed"
                ],
                "output": receipt["output"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
