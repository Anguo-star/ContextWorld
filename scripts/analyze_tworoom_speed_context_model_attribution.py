#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.context_model_attribution_analysis import (
    audit_static_inputs,
    load_attribution_config,
    run_model_attribution_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or analyze the preregistered four-model directional-v2 "
            "speed-context attribution comparison."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "configs/benchmark/"
            "tworoom_speed_context_model_attribution_v1.yaml"
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Verify frozen inputs and count arithmetic without reading scores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_attribution_config(config_path)
    if args.audit_only:
        result = audit_static_inputs(
            config=config,
            config_path=config_path,
            repo_root=REPO_ROOT,
        )
    else:
        result = run_model_attribution_analysis(
            config=config,
            config_path=config_path,
            repo_root=REPO_ROOT,
        )
    compact = {
        "status": result["status"],
        "output": result.get("output"),
        "count_audit": result.get("protocol_and_count_audit"),
        "model_fast_minus_slow": (
            {}
            if args.audit_only
            else {
                model_id: {
                    "effect_points": row[
                        "stable_fast_over_slow_gate"
                    ]["effect_points"],
                    "p_value": row["stable_fast_over_slow_gate"][
                        "paired_p_value"
                    ],
                    "passed": row["stable_fast_over_slow_gate"][
                        "passed"
                    ],
                }
                for model_id, row in result["models"].items()
            }
        ),
        "decisions": result.get("decisions"),
        "static_audit": (
            result if args.audit_only else result["static_input_audit"]
        ),
    }
    print(
        json.dumps(
            compact,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
