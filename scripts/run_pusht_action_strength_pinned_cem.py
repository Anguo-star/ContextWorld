#!/usr/bin/env python3
"""Run frozen ActionStrength CEM sources against an explicit Stable-WM pin.

The two published CEM scripts historically resolve ``../stable-worldmodel``
at import time.  This additive launcher leaves those sources untouched: it
verifies their hashes and executes an in-memory copy whose *only* replacement
is that one runtime-root assignment.  That makes the actual runtime explicit
and prevents a local sibling checkout from silently changing a formal PLDM
evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RUNTIME_ASSIGNMENT = (
    'STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / "stable-worldmodel"'
)
TRACKS = {
    "planning": {
        "source": SCRIPTS / "eval_pusht_replay_matched_hidden_cem.py",
        "module": "_pinned_action_strength_planning",
        "dependency": SCRIPTS / "eval_pusht_hidden_actuation_cem.py",
    },
    "retention": {
        "source": SCRIPTS / "eval_pusht_standard_cem_retention.py",
        "module": "_pinned_action_strength_retention",
        "dependency": None,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worktree_head(worktree: Path) -> str:
    pointer = worktree / ".git"
    if pointer.is_file():
        raw = pointer.read_text(encoding="utf-8").strip()
        if not raw.startswith("gitdir: "):
            raise RuntimeError(f"Unsupported .git pointer: {pointer}")
        gitdir = Path(raw[len("gitdir: ") :]).expanduser()
    elif pointer.is_dir():
        gitdir = pointer
    else:
        raise FileNotFoundError(f"Stable-WorldModel worktree lacks .git: {worktree}")
    head = (gitdir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head[len("ref: ") :]
        ref_path = gitdir / ref
        if not ref_path.is_file():
            # Linked worktrees keep ordinary refs in their common gitdir.
            common = (gitdir / "commondir").read_text(encoding="utf-8").strip()
            ref_path = (gitdir / common / ref).resolve()
        head = ref_path.read_text(encoding="utf-8").strip()
    if len(head) != 40:
        raise RuntimeError(f"Could not resolve a commit from {gitdir / 'HEAD'}")
    return head


def _verify_source(path: Path, expected_sha256: str, *, role: str) -> None:
    if len(expected_sha256) != 64:
        raise ValueError(f"{role} SHA-256 must be explicit")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{role} changed after binding: expected {expected_sha256}, got {actual}"
        )
    source = path.read_text(encoding="utf-8")
    if source.count(RUNTIME_ASSIGNMENT) != 1:
        raise RuntimeError(
            f"{role} no longer has exactly one injectable runtime assignment"
        )


def _clear_runtime_modules() -> None:
    for name in list(sys.modules):
        if name == "eval_wm" or name.startswith(("stable_worldmodel", "stable_pretraining")):
            sys.modules.pop(name, None)


def _load_with_pinned_root(
    *,
    module_name: str,
    source_path: Path,
    stablewm_root: Path,
) -> types.ModuleType:
    source = source_path.read_text(encoding="utf-8")
    replacement = f"STABLE_WORLD_MODEL_ROOT = Path({str(stablewm_root)!r}).resolve()"
    if source.count(RUNTIME_ASSIGNMENT) != 1:
        raise RuntimeError(f"Cannot inject Stable-WorldModel root into {source_path}")
    source = source.replace(RUNTIME_ASSIGNMENT, replacement, 1)
    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


def _prepare_runtime(
    *,
    track: str,
    stablewm_root: Path,
    expected_ref: str,
    runner_sha256: str,
    dependency_sha256: str | None,
) -> tuple[types.ModuleType, dict[str, Any]]:
    stablewm_root = stablewm_root.expanduser().resolve()
    if not stablewm_root.is_dir():
        raise FileNotFoundError(stablewm_root)
    observed_ref = _worktree_head(stablewm_root)
    if observed_ref != expected_ref:
        raise RuntimeError(
            "Stable-WorldModel ref differs from the formal binding: "
            f"expected {expected_ref}, got {observed_ref}"
        )
    specification = TRACKS[track]
    source_path = Path(specification["source"])
    _verify_source(source_path, runner_sha256, role=f"{track} runner")
    dependency = specification["dependency"]
    if dependency is not None:
        if dependency_sha256 is None:
            raise ValueError("Planning requires --dependency-sha256")
        _verify_source(
            Path(dependency),
            dependency_sha256,
            role="planning shared model loader",
        )

    _clear_runtime_modules()
    for path in (str(stablewm_root), str(ROOT), str(SCRIPTS)):
        while path in sys.path:
            sys.path.remove(path)
    sys.path[:0] = [str(stablewm_root), str(ROOT), str(SCRIPTS)]
    if dependency is not None:
        _load_with_pinned_root(
            module_name="eval_pusht_hidden_actuation_cem",
            source_path=Path(dependency),
            stablewm_root=stablewm_root,
        )
    runner = _load_with_pinned_root(
        module_name=str(specification["module"]),
        source_path=source_path,
        stablewm_root=stablewm_root,
    )
    receipt: dict[str, Any] = {
        "track": track,
        "stable_worldmodel_root": str(stablewm_root),
        "stable_worldmodel_expected_ref": expected_ref,
        "stable_worldmodel_observed_ref": observed_ref,
        "runner": {"path": str(source_path), "sha256": runner_sha256},
        "runtime_assignment_injected_in_memory_only": True,
        "original_runner_modified": False,
    }
    if dependency is not None:
        receipt["shared_model_loader"] = {
            "path": str(dependency),
            "sha256": dependency_sha256,
        }
    return runner, receipt


def _strict_load_check(
    *,
    track: str,
    runner: types.ModuleType,
    checkpoint: Path,
) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    import torch

    if track == "planning":
        loader = sys.modules["eval_pusht_hidden_actuation_cem"]
        model = loader.load_model(checkpoint, torch.device("cpu"))
    else:
        # This uses the unmodified retention source's normal model loader and
        # its sibling config.json contract, with the pinned runtime injected.
        dataset = None
        try:
            import stable_worldmodel as swm

            model = swm.wm.utils.load_pretrained(str(checkpoint))
        finally:
            del dataset
    model.eval()
    state = model.state_dict()
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "model_state_sha256": digest.hexdigest(),
    }


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", choices=tuple(TRACKS))
    parser.add_argument("--stable-worldmodel-root", type=Path, required=True)
    parser.add_argument("--expected-ref", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--dependency-sha256", default=None)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--strict-load-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=None,
        help="Optional additive JSON receipt; refuses to overwrite.",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the frozen runner; place them after '--'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner, receipt = _prepare_runtime(
        track=args.track,
        stablewm_root=args.stable_worldmodel_root,
        expected_ref=str(args.expected_ref),
        runner_sha256=str(args.runner_sha256),
        dependency_sha256=(
            None if args.dependency_sha256 is None else str(args.dependency_sha256)
        ),
    )
    if args.strict_load_checkpoint is not None:
        receipt["strict_load"] = _strict_load_check(
            track=args.track,
            runner=runner,
            checkpoint=args.strict_load_checkpoint,
        )
    if args.receipt_output is not None:
        _write_receipt(args.receipt_output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if args.verify_only:
        if args.runner_args:
            raise ValueError("--verify-only cannot forward runner arguments")
        return
    if not args.runner_args:
        raise ValueError("Runner arguments are required after '--'")
    old_argv = list(sys.argv)
    try:
        sys.argv = [str(TRACKS[args.track]["source"]), *args.runner_args]
        runner.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
