#!/usr/bin/env python3
"""Build the authorized Cube v4r1 Public recovery dataset exactly once."""

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
import scripts.build_cube_grasp_rule_h3_v4r1_public_data as base_builder


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument(
        "--freeze-receipt", type=Path, default=DEFAULT_FREEZE_RECEIPT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--staging-root", type=Path, default=base_builder.DEFAULT_STAGING_ROOT
    )
    parser.add_argument("--workers", type=int, default=base_builder.DEFAULT_WORKERS)
    parser.add_argument(
        "--jpeg-quality", type=int, default=base_builder.DEFAULT_JPEG_QUALITY
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = base_builder.build_public_data(
        source=args.source,
        preregistration=args.prereg,
        freeze_receipt=args.freeze_receipt,
        output=args.output,
        staging_root=args.staging_root,
        workers=args.workers,
        jpeg_quality=args.jpeg_quality,
        authorization_loader=load_public_recovery_authorization,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
