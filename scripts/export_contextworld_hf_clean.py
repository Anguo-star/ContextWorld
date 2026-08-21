#!/usr/bin/env python3
"""Plan or build the Public-Test-withheld ContextWorld HF staging tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.hf_clean_export import (  # noqa: E402
    build_export_plan,
    export_hf_clean,
    refresh_hf_clean_metadata,
)


DEFAULT_CONTRACT = (
    ROOT / "configs/benchmark/contextworld_hf_clean_export_v1.yaml"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a clean Hugging Face staging tree containing only the "
            "nine components' Training and Development data."
        )
    )
    parser.add_argument(
        "--suite-export-root",
        type=Path,
        help=(
            "ContextWorld artifact root containing the registered synthesis/ "
            "payloads; required for plan and execute modes"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Copy payloads. Without this flag, only validate and print the plan.",
    )
    mode.add_argument(
        "--refresh-metadata",
        action="store_true",
        help=(
            "Refresh generated metadata in an existing staging export after "
            "verifying its manifest and payload mapping; never recopy data."
        ),
    )
    parser.add_argument(
        "--full-plan",
        action="store_true",
        help="Include every registered Lance member in plan-only JSON.",
    )
    parser.add_argument(
        "--direct-write",
        action="store_true",
        help=(
            "Write directly to a new output directory instead of publishing "
            "by atomic rename. Use only on managed mounts that forbid rename."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.direct_write and not args.execute:
        raise SystemExit("--direct-write is only valid with --execute")
    if args.refresh_metadata:
        result = refresh_hf_clean_metadata(
            contract_path=args.contract,
            output=args.output,
            repo_root=ROOT,
        )
    elif args.execute:
        if args.suite_export_root is None:
            raise SystemExit("--execute requires --suite-export-root")
        result = export_hf_clean(
            contract_path=args.contract,
            suite_export_root=args.suite_export_root,
            output=args.output,
            repo_root=ROOT,
            atomic_publish=not args.direct_write,
        )
    else:
        if args.suite_export_root is None:
            raise SystemExit("plan-only mode requires --suite-export-root")
        plan = build_export_plan(
            contract_path=args.contract,
            suite_export_root=args.suite_export_root,
            repo_root=ROOT,
        )
        if args.full_plan:
            result = plan
        else:
            result = {
                "export_id": plan["export_id"],
                "status": plan["status"],
                "public_test_policy": plan["public_test_policy"],
                "inventory": plan["inventory"],
                "components": [
                    {
                        "component_id": component["component_id"],
                        "dataset_id": component["dataset_id"],
                        "payloads": [
                            {
                                key: payload[key]
                                for key in (
                                    "split",
                                    "payload_id",
                                    "public_path",
                                    "payload_kind",
                                    "file_count",
                                    "total_bytes",
                                    "lance_table_count",
                                    "single_dataset_entrypoint",
                                    "single_dataset_path",
                                    "stable_worldmodel_sequence_schema",
                                    "stable_worldmodel_adapter_required",
                                    "direct_stable_worldmodel_load",
                                    "cw_dataset_entrypoint",
                                )
                            }
                            for payload in component["payloads"]
                        ],
                    }
                    for component in plan["components"]
                ],
            }
        result = {
            **result,
            "mode": "plan_only",
            "output_will_be": str(args.output.expanduser().resolve()),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
