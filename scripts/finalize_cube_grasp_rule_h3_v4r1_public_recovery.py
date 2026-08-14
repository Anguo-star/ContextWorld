#!/usr/bin/env python3
"""Finalize the completed Cube v4r1 Public recovery campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.cube_grasp_rule_public_recovery_contract import (
    DECISION_ID,
    DEFAULT_FREEZE_RECEIPT,
    DEFAULT_PREREGISTRATION,
    load_public_recovery_authorization,
)
import scripts.finalize_cube_grasp_rule_h3_v4r1_public_release as base_finalizer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument(
        "--freeze-receipt", type=Path, default=DEFAULT_FREEZE_RECEIPT
    )
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = base_finalizer.finalize_public_release(
        preregistration=args.prereg,
        freeze_receipt=args.freeze_receipt,
        score_root=args.score_root,
        output=args.output,
        authorization_loader=load_public_recovery_authorization,
        decision_id=DECISION_ID,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "checkpoints_passed": result["public_evaluation"][
                    "checkpoints_passed"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
