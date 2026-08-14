#!/usr/bin/env python3
"""Run the pinned Stable-WorldModel standard Cube CEM retention test."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
STABLE_SCRIPT_ROOT = (
    ROOT.parent
    / "stable-worldmodel/research/conditional_dynamics_representation/scripts"
)
if str(STABLE_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(STABLE_SCRIPT_ROOT))

from eval_original_task_cem import main  # noqa: E402


if __name__ == "__main__":
    main()
