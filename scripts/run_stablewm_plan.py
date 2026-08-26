#!/usr/bin/env python3
"""Run Stable-WorldModel's planner with an explicit history-key bridge.

Stable-WorldModel owns the environment rollout, CEM solver and world-model
cost calculation.  Its public ``WorldModelPolicy`` already accepts
``history_keys``, but the historical ``scripts/plan/eval_wm.py`` entry point
does not expose that constructor argument through Hydra.  Multimodal PreJEPA
checkpoints therefore stack pixel history while leaving their state stream at
one frame.

This launcher adds no planning or model logic.  It injects the caller-selected
``history_keys`` into the existing policy constructor and then executes the
upstream entry point unchanged.  It can be removed once upstream exposes the
same setting directly.
"""

from __future__ import annotations

import argparse
import functools
import runpy
import sys
from pathlib import Path

from run_stablewm_family_entry import _prepare_optional_flash_attention


def _parse_bridge_args(argv: list[str]) -> tuple[Path, tuple[str, ...], list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream-entry", type=Path, required=True)
    parser.add_argument("--history-keys", required=True)
    args, remaining = parser.parse_known_args(argv)

    entry = args.upstream_entry.expanduser().resolve()
    if not entry.is_file():
        raise SystemExit(f"Stable-WorldModel planner entry point not found: {entry}")
    keys = tuple(part.strip() for part in args.history_keys.split(",") if part.strip())
    if not keys or keys[0] != "pixels" or len(keys) != len(set(keys)):
        raise SystemExit(
            "--history-keys must be a unique comma-separated list beginning "
            "with pixels"
        )
    return entry, keys, remaining


def main(argv: list[str] | None = None) -> int:
    entry, history_keys, upstream_argv = _parse_bridge_args(
        list(sys.argv[1:] if argv is None else argv)
    )

    # Match the training entry: a stale optional flash-attn wheel must not
    # prevent StableWM from using PyTorch's supported attention fallback.
    _prepare_optional_flash_attention()
    import stable_worldmodel as swm

    policy_class = swm.policy.WorldModelPolicy
    original_init = policy_class.__init__

    @functools.wraps(original_init)
    def init_with_history_keys(self, *args, **kwargs):
        kwargs.setdefault("history_keys", history_keys)
        return original_init(self, *args, **kwargs)

    policy_class.__init__ = init_with_history_keys
    sys.argv = [str(entry), *upstream_argv]
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        # Preserve the normal interpreter behavior for ``sys.exit("message")``
        # so an upstream configuration error is not reduced to a silent 1.
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
