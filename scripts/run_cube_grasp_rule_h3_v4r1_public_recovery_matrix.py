#!/usr/bin/env python3
"""Run the fixed three-seed Cube v4r1 Public recovery score exactly once."""

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
    DEFAULT_FREEZE_RECEIPT,
    DEFAULT_PREREGISTRATION,
    load_public_recovery_authorization,
)
import scripts.run_cube_grasp_rule_h3_v4r1_public_matrix as base_runner


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument(
        "--freeze-receipt", type=Path, default=DEFAULT_FREEZE_RECEIPT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    result = base_runner.run_public_matrix(
        preregistration=args.prereg,
        freeze_receipt=args.freeze_receipt,
        output=args.output,
        devices=devices,
        batch_size=args.batch_size,
        authorization_loader=load_public_recovery_authorization,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
