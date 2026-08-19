"""Prepare the one-use Suite v2 integrity-reseal decision after inputs settle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextworld.benchmarks.suite_v2_integrity_reseal import (
    RESEAL_CONFIG,
    audit_current_identity_drift,
    validate_integrity_reseal_decision,
    write_integrity_reseal_decision,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reseal-config", type=Path, default=RESEAL_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.check_only:
        payload = audit_current_identity_drift()
        missing = payload["missing_required_descriptive_result_freezes"]
        if missing:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            raise SystemExit(
                "integrity reseal is blocked: " + "; ".join(missing)
            )
        if args.output is not None and args.output.is_file():
            decision = json.loads(args.output.read_text(encoding="utf-8"))
            payload["decision_validation"] = validate_integrity_reseal_decision(
                decision, reseal_config=args.reseal_config
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.output is None:
        raise SystemExit("--output is required unless --check-only is used")
    try:
        decision = write_integrity_reseal_decision(
            args.output, reseal_config=args.reseal_config
        )
    except ValueError as error:
        raise SystemExit(f"integrity reseal is blocked: {error}") from error
    print(
        json.dumps(
            {
                "status": decision["status"],
                "output": str(args.output.expanduser().resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
