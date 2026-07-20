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

from contextworld.evaluation.icl_sensitive_analysis import (
    run_calibration_diagnostics,
    run_calibration_selection,
    run_formal_analysis,
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload["status"] != "preregistered_before_execution":
        raise ValueError("Expected the frozen preregistered configuration")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select ICL-sensitive distances or analyze formal runs"
    )
    parser.add_argument(
        "stage",
        choices=("select", "diagnose", "formal"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/benchmark/tworoom_speed_icl_sensitive_eval_v1.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    if args.stage == "select":
        result = run_calibration_selection(
            config=config,
            config_path=config_path,
            repo_root=REPO_ROOT,
        )
        display = {
            "status": result["status"],
            "selected_distance_bins": result[
                "selected_distance_bins"
            ],
            "formal_eval_authorized": result[
                "formal_eval_authorized"
            ],
        }
    elif args.stage == "diagnose":
        result = run_calibration_diagnostics(
            config=config,
            config_path=config_path,
            repo_root=REPO_ROOT,
        )
        display = {
            "status": result["status"],
            "evidence_level": result["evidence_level"],
            "difficulty_bins": result[
                "difficulty_eligible_ignoring_context_effect"
            ]["distance_bins"],
            "prompt_speed_bias": result[
                "prompt_speed_bias_relabeling"
            ],
            "output": result["output"],
        }
    else:
        result = run_formal_analysis(
            config=config,
            config_path=config_path,
            repo_root=REPO_ROOT,
        )
        display = {
            "status": result["status"],
            "primary_decision": result["primary_decision"],
            "specificity_diagnostic": result[
                "specificity_diagnostic"
            ],
        }
    print(
        json.dumps(
            display,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
