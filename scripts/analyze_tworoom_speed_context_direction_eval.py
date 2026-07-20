#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.context_direction_analysis import (
    run_directional_analysis,
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload["status"] != "preregistered_before_execution":
        raise ValueError("Expected the frozen preregistered configuration")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the preregistered heldout wrong-slower/wrong-faster "
            "SpeedFull planning evaluation"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "configs/benchmark/"
            "tworoom_speed_context_direction_eval_v2.yaml"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    result = run_directional_analysis(
        config=_load_config(config_path),
        config_path=config_path,
        repo_root=REPO_ROOT,
    )
    comparisons = result["paired_comparisons"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": result["output"],
                "count_audit": result["protocol_and_count_audit"],
                "success_rates_percent": {
                    name: summary["success_rate_percent"]
                    for name, summary in result["conditions"].items()
                },
                "effects_points": {
                    "correct_minus_wrong_slow": comparisons[
                        "correct_vs_wrong_slow"
                    ][
                        "correct_minus_wrong_slow_success_rate_points"
                    ],
                    "correct_minus_wrong_fast": comparisons[
                        "correct_vs_wrong_fast"
                    ][
                        "correct_minus_wrong_fast_success_rate_points"
                    ],
                    "wrong_fast_minus_wrong_slow": comparisons[
                        "wrong_fast_vs_wrong_slow"
                    ][
                        "wrong_fast_minus_wrong_slow_success_rate_points"
                    ],
                },
                "decisions": result["decisions"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
