"""Check or explicitly create the inactive Suite v2 integrity-reseal-v2 marker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextworld.benchmarks.suite_v2_integrity_reseal_v2 import (
    RESEAL_CONFIG,
    audit_integrity_reseal_v2_readiness,
    validate_integrity_reseal_v2_decision,
    write_integrity_reseal_v2_decision,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reseal-config", type=Path, default=RESEAL_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--write-decision", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.check_only == args.write_decision:
        raise SystemExit("choose exactly one of --check-only or --write-decision")
    if args.check_only:
        audit = audit_integrity_reseal_v2_readiness(
            reseal_config=args.reseal_config
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        if not audit["ready"]:
            raise SystemExit("integrity reseal v2 is blocked")
        return
    if args.output is None:
        raise SystemExit("--output is required with --write-decision")
    decision = write_integrity_reseal_v2_decision(
        args.output, reseal_config=args.reseal_config
    )
    validation = validate_integrity_reseal_v2_decision(
        decision, reseal_config=args.reseal_config
    )
    print(
        json.dumps(
            {
                "status": decision["status"],
                "output": str(args.output.expanduser().resolve()),
                "validation": validation,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
