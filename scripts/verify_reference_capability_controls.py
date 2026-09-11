#!/usr/bin/env python3
"""Run the cheap reference capability controls and emit JSON evidence.

The controls use only tiny synthetic arrays and the checked-in score kernels;
they do not open model checkpoints or frozen evaluation artifacts.  The JSON
is diagnostic evidence and explicitly makes no population FPR claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_controls():
    tests_path = ROOT / "tests/test_reference_capability_controls.py"
    spec = importlib.util.spec_from_file_location(
        "reference_capability_controls", tests_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load controls: {tests_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(module)
    return module


def _assert_expected_controls(evidence: dict) -> None:
    failures: list[str] = []
    for component in ("speed", "action_delay"):
        observed = [
            bool(evidence[component][mode]["passed"])
            for mode in ("oracle", "no_history", "reversed", "tiny")
        ]
        if observed != [True, False, False, False]:
            failures.append(f"{component} pass pattern changed: {observed}")
    paired = evidence["shared_paired_latent"]
    for task, checks in paired["oracle"]["checks_by_task"].items():
        if not all(checks.values()):
            failures.append(f"{task} oracle gate failed: {checks}")
    for mode in ("no_history", "reversed", "tiny"):
        for task, checks in paired[mode]["checks_by_task"].items():
            if checks["response_gain"]:
                failures.append(f"{task} {mode} response-gain control passed")
    if evidence["scope"]["population_false_positive_rate_estimated"] is not False:
        failures.append("synthetic controls must not claim population FPR")
    if failures:
        raise RuntimeError("; ".join(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts/evaluation/baseline_completion_v3/validation/capability_controls.json",
        help="JSON evidence path (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    controls = _load_controls()
    evidence = controls.collect_control_evidence()
    _assert_expected_controls(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
