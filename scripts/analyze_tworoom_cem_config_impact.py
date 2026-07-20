#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.cem_config_impact_analysis import (
    load_config,
    run_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and analyze the preregistered TwoRoom CEM planning "
            "configuration impact experiment."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "configs/benchmark/tworoom_cem_config_impact_v1.yaml"
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run the complete audit without writing the final summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    result = run_analysis(
        config=load_config(config_path),
        config_path=config_path,
        repo_root=REPO_ROOT,
        write_output=not args.no_write,
    )
    compact = {
        "status": result["status"],
        "output": result["output"],
        "protocol_and_count_audit": result[
            "protocol_and_count_audit"
        ],
        "primary_fast_minus_slow_effect": result[
            "primary_fast_minus_slow_effect"
        ],
        "stage_conclusion": result["stage_conclusion"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
