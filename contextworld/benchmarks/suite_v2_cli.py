from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contextworld.benchmarks.suite_data import (
    DEFAULT_SUITE_V2_RELEASE_CONFIG,
    SUITE_V2_COMPONENT_IDS,
    audit_icl_suite_release,
    export_icl_suite_artifacts,
    load_icl_suite_release,
    load_public_scoreboard,
    require_suite_membership_activation,
)
from contextworld.synthesis.manifest import write_json


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    if output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    target = output.expanduser().resolve()
    write_json(target, payload)
    print(json.dumps({"status": payload.get("status"), "output": str(target)}))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-benchmark",
        description="ContextWorld ICL Benchmark Suite v2 (nine components)",
    )
    parser.add_argument(
        "--release-config", type=Path, default=DEFAULT_SUITE_V2_RELEASE_CONFIG
    )
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info")
    info.add_argument("--output", type=Path)
    audit = commands.add_parser("audit")
    audit.add_argument(
        "--component", action="append", choices=SUITE_V2_COMPONENT_IDS
    )
    audit.add_argument("--full", action="store_true")
    audit.add_argument("--original-h5", type=Path)
    audit.add_argument("--output", type=Path)
    results = commands.add_parser("results")
    results.add_argument("--output", type=Path)
    export = commands.add_parser("export")
    export.add_argument("--destination", type=Path, required=True)
    export.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    export.add_argument("--without-upstream-original", action="store_true")
    export.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "info":
        suite = load_icl_suite_release(args.release_config)
        membership_activation = require_suite_membership_activation(suite)
        payload = {
            key: value for key, value in suite.items() if not key.startswith("_")
        }
        payload["membership_activation"] = membership_activation
        payload["commands"] = {
            "suite": "contextworld-benchmark",
            **{
                component_id: suite["components"][component_id]["cli"]
                for component_id in SUITE_V2_COMPONENT_IDS
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
            include_upstream_original=not args.without_upstream_original,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
