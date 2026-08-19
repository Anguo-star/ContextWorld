from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from contextworld.benchmarks.suite_data import (
    SUITE_V2_COMPONENT_IDS,
    audit_icl_suite_release,
    export_icl_suite_artifacts,
    load_historical_v2_archive_view,
    load_icl_suite_release,
    load_public_scoreboard,
    require_suite_membership_activation,
    resolve_suite_v2_cli_default_config,
)
from contextworld.synthesis.manifest import write_json


def _enable_source_checkout_auditors() -> None:
    """Make repository-only audit modules visible to an editable CLI.

    A generated console script starts with ``/usr/local/bin`` as
    ``sys.path[0]`` rather than the caller's working directory.  Editable
    installs therefore find the ``contextworld`` package but not the
    repository's top-level ``scripts`` namespace used by deep result audits.
    Add the checkout only when this module is actually running from one; a
    wheel installation has no sibling ``scripts`` directory and is unchanged.
    """

    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "scripts").is_dir():
        value = str(checkout)
        if value not in sys.path:
            sys.path.insert(0, value)


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
        "--release-config",
        type=Path,
        default=None,
        help=(
            "Explicit release config. Without this option, the command shows "
            "the verified current table after the final v2 decision, or the "
            "v1 archived table while that decision is absent."
        ),
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
    _enable_source_checkout_auditors()
    args = parse_args(argv)
    release_config = args.release_config or resolve_suite_v2_cli_default_config()
    archive_view = load_historical_v2_archive_view(release_config)
    if args.command == "info":
        suite = (
            archive_view["suite"]
            if archive_view is not None
            else load_icl_suite_release(release_config)
        )
        payload = {
            key: value for key, value in suite.items() if not key.startswith("_")
        }
        if archive_view is not None:
            # The old activation marker intentionally is not revalidated here:
            # this is evidence inspection, not a claim that the old release is
            # currently active.
            payload.pop("membership_authority", None)
            payload["release_view"] = archive_view["release_view"]
            payload["active_release"] = False
            payload["read_only"] = True
            payload["archive"] = {
                key: archive_view[key]
                for key in (
                    "release_config",
                    "config_identity",
                    "formal_reference_rows",
                    "components_with_formal_results",
                )
            }
            payload["membership_activation"] = {
                "required": True,
                "active": False,
                "status": "historical_archive_read_only",
                "current_release_claimed": False,
            }
        else:
            payload["membership_activation"] = require_suite_membership_activation(
                suite
            )
        payload["commands"] = {
            "suite": "contextworld-benchmark",
            **{
                component_id: suite["components"][component_id]["cli"]
                for component_id in SUITE_V2_COMPONENT_IDS
            },
        }
    elif args.command == "audit":
        if archive_view is not None:
            raise RuntimeError(
                "The historical Suite v2 archive is read-only; an active "
                "release audit requires a valid final v2 decision."
            )
        payload = audit_icl_suite_release(
            release_config=release_config,
            components=args.component,
            full=args.full,
            original_h5=args.original_h5,
        )
    elif args.command == "results":
        payload = load_public_scoreboard(release_config)
    elif args.command == "export":
        if archive_view is not None:
            raise RuntimeError(
                "The historical Suite v2 archive cannot be exported as an "
                "active release; a valid final v2 decision is required."
            )
        payload = export_icl_suite_artifacts(
            args.destination,
            release_config=release_config,
            mode=args.mode,
            include_upstream_original=not args.without_upstream_original,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _emit(payload, args.output)


if __name__ == "__main__":
    main()
