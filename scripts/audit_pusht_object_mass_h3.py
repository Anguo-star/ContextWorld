#!/usr/bin/env python3
"""Run the frozen PushT object-mass History=3 feasibility audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.pusht_object_mass_h3 import (
    audit_object_mass_history3,
)


DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/evaluation/history3/"
    "pusht_object_mass_h3_feasibility_v1/audit.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit_object_mass_history3()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
