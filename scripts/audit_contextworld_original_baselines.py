from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextworld.benchmarks.original_baseline_matrix import (
    DEFAULT_ORIGINAL_BASELINE_FREEZE,
    DEFAULT_ORIGINAL_BASELINE_PREREG,
    audit_original_baseline_prereg,
)
from contextworld.synthesis.manifest import write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the frozen eight-checkpoint original baseline matrix"
    )
    parser.add_argument("--prereg", type=Path, default=DEFAULT_ORIGINAL_BASELINE_PREREG)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_ORIGINAL_BASELINE_FREEZE)
    parser.add_argument("--verify-local-checkpoints", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = audit_original_baseline_prereg(
        args.prereg,
        freeze_path=args.freeze,
        verify_local_checkpoints=args.verify_local_checkpoints,
    )
    if args.output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        target = args.output.expanduser().resolve()
        write_json(target, payload)
        print(json.dumps({"status": payload["status"], "output": str(target)}, sort_keys=True))


if __name__ == "__main__":
    main()
