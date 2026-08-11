"""CLI for rendering the minimal ContextWorld public result table as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contextworld.benchmarks.public_score import (
    make_public_scoreboard_from_spec,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m contextworld.benchmarks.public_score_cli",
        description=(
            "Render formal per-seed ICL correctness and original-task "
            "retention results into the minimal public scoreboard JSON"
        ),
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Public scoreboard spec must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    scoreboard = make_public_scoreboard_from_spec(_load(args.input))
    rendered = json.dumps(scoreboard, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
