from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contextworld.benchmarks.suite_data import (
    COMPONENT_IDS,
    DEFAULT_SUITE_RELEASE_CONFIG,
    audit_icl_suite_release,
    export_icl_suite_artifacts,
    load_icl_suite_release,
    load_public_scoreboard,
)
from contextworld.synthesis.manifest import write_json


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    if output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    target = output.expanduser().resolve()
    write_json(target, payload)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "output": str(target),
            },
            sort_keys=True,
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-benchmark",
        description=(
            "Unified release, audit and export commands for the "
            "ContextWorld ICL Benchmark suite"
        ),
    )
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_SUITE_RELEASE_CONFIG,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser(
        "info",
        help="Print the unified suite contract and component entry points",
    )
    info.add_argument("--output", type=Path, default=None)

    audit = subparsers.add_parser(
        "audit",
        help="Audit the common code version and selected benchmark data",
    )
    audit.add_argument(
        "--component",
        action="append",
        choices=COMPONENT_IDS,
        default=None,
        help="Audit one component; repeat to select multiple components",
    )
    audit.add_argument(
        "--full",
        action="store_true",
        help="Hash every training tree and every offline Eval payload",
    )
    audit.add_argument("--original-h5", type=Path, default=None)
    audit.add_argument("--output", type=Path, default=None)

    results = subparsers.add_parser(
        "results",
        help="Print the frozen compact public reference-result table",
    )
    results.add_argument("--output", type=Path, default=None)

    export = subparsers.add_parser(
        "export",
        help="Export one README plus the integrated benchmark data tree",
    )
    export.add_argument("--destination", type=Path, required=True)
    export.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    export.add_argument(
        "--without-upstream-original",
        action="store_true",
        help=(
            "Create a smaller package without upstream TwoRoom, PushT or "
            "Reacher training data and initialization checkpoints"
        ),
    )
    export.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "info":
        suite = load_icl_suite_release(args.release_config)
        payload = {
            key: value
            for key, value in suite.items()
            if not key.startswith("_")
        }
        payload["commands"] = {
            "suite": "contextworld-benchmark",
            **{
                component_id: suite["components"][component_id]["cli"]
                for component_id in COMPONENT_IDS
            },
        }
    elif args.command == "audit":
        payload = audit_icl_suite_release(
            release_config=args.release_config,
            components=args.component,
            full=args.full,
            original_h5=args.original_h5,
        )
    elif args.command == "results":
        payload = load_public_scoreboard(args.release_config)
    elif args.command == "export":
        payload = export_icl_suite_artifacts(
            args.destination,
            release_config=args.release_config,
            mode=args.mode,
            include_upstream_original=(
                not args.without_upstream_original
            ),
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
