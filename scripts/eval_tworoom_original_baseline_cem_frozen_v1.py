#!/usr/bin/env python3
"""Run one frozen TwoRoom original-task CEM cell from a legacy checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


base = _module(
    ROOT / "scripts/eval_cube_original_task_cem_frozen.py",
    "contextworld_tworoom_legacy_checkpoint_runtime_v1",
)
tworoom = _module(
    ROOT / "scripts/eval_tworoom_ability_catalog.py",
    "contextworld_tworoom_ability_catalog_frozen_v1",
)

from contextworld.paths import resolve_contextworld_path  # noqa: E402


def _identity(
    path: Path, *, expected_sha256: str, expected_size: int | None, label: str
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    if expected_size is not None and resolved.stat().st_size != expected_size:
        raise RuntimeError(f"{label} size drifted")
    observed = base.file_sha256(resolved)
    if observed != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 drifted: expected {expected_sha256}, got {observed}"
        )
    return {
        "path": str(resolved),
        "sha256": observed,
        "size_bytes": resolved.stat().st_size,
    }


def _catalog_contract(catalog: Path) -> tuple[dict[str, Any], list[Path]]:
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if (
        payload.get("schema_version") != 1
        or payload.get("catalog") != "tworoom_original_heldout_eval_catalog_v1"
        or not isinstance(entries, list)
        or len(entries) != 300
    ):
        raise RuntimeError("Frozen TwoRoom catalog must contain 300 entries")
    expected_seeds = tuple(range(42, 48))
    counts = Counter(int(row.get("eval_seed", -1)) for row in entries)
    if counts != Counter({seed: 50 for seed in expected_seeds}):
        raise RuntimeError("Frozen TwoRoom catalog seed/count contract drifted")
    for seed in expected_seeds:
        rows = [row for row in entries if int(row["eval_seed"]) == seed]
        if (
            {int(row.get("cem_group_seed", -1)) for row in rows} != {seed}
            or sorted(int(row.get("evaluation_index", -1)) for row in rows)
            != list(range(50))
            or len({str(row.get("evaluation_id", "")) for row in rows}) != 50
            or {str(row.get("source_kind", "")) for row in rows}
            != {"original_h5"}
            or {int(row.get("goal_offset", -1)) for row in rows} != {25}
        ):
            raise RuntimeError("Frozen TwoRoom catalog query contract drifted")
    paths = sorted({str(row.get("source_path", "")) for row in entries})
    if len(paths) != 1 or not paths[0]:
        raise RuntimeError("Frozen TwoRoom catalog source set drifted")
    return payload, [resolve_contextworld_path(paths[0], repo_root=ROOT)]


def _install_legacy_loader(runtime_root: Path, expected_ref: str) -> dict[str, Any]:
    runtime = base.install_runtime(runtime_root, expected_ref)

    def load_legacy(checkpoint: Path, _swm: Any, *, cache_dir: Path) -> Any:
        del _swm, cache_dir
        return base.load_checkpoint_model(Path(checkpoint).expanduser().resolve())

    tworoom.load_pretrained_cost_model = load_legacy
    return runtime


def _verified_inputs(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _identity(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        expected_size=args.expected_checkpoint_size,
        label="canonical checkpoint",
    )
    config_path = base.checkpoint_config_path(Path(checkpoint["path"]))
    config = _identity(
        config_path,
        expected_sha256=args.expected_config_sha256,
        expected_size=args.expected_config_size,
        label="checkpoint loader config",
    )
    catalog = _identity(
        args.catalog,
        expected_sha256=args.expected_catalog_sha256,
        expected_size=args.expected_catalog_size,
        label="frozen query catalog",
    )
    normalizer = _identity(
        args.normalizer,
        expected_sha256=args.expected_normalizer_sha256,
        expected_size=args.expected_normalizer_size,
        label="frozen normalizer",
    )
    catalog_payload, sources = _catalog_contract(Path(catalog["path"]))
    if len(sources) != 1:
        raise RuntimeError("Expected one heldout dataset source")
    source = _identity(
        sources[0],
        expected_sha256=args.expected_source_sha256,
        expected_size=args.expected_source_size,
        label="frozen heldout dataset",
    )
    return {
        "checkpoint": checkpoint,
        "config": config,
        "catalog": catalog,
        "catalog_payload": catalog_payload,
        "normalizer": normalizer,
        "source_dataset": source,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    runtime = _install_legacy_loader(
        args.stable_worldmodel_root, args.expected_ref
    )
    inputs = _verified_inputs(args)
    checkpoint = inputs["checkpoint"]
    model = base.load_checkpoint_model(Path(checkpoint["path"]))
    try:
        protocol = tworoom.infer_model_protocol(model, action_dim=2)
        if protocol != {"action_block": 5, "history_size": 3}:
            raise RuntimeError(f"Unexpected model protocol: {protocol}")
        model_row = {
            **checkpoint,
            "config": inputs["config"],
            "parameter_count": sum(
                int(parameter.numel()) for parameter in model.parameters()
            ),
            "state_dict_sha256": tworoom.state_dict_sha256(model),
            "protocol": protocol,
            "strict_load": True,
        }
    finally:
        del model

    # Exercise every runtime/data boundary without calling ``World.evaluate``.
    swm, stable_root, stable_commit = tworoom.load_stable_worldmodel(
        ROOT, str(args.stable_worldmodel_root.expanduser().resolve()), args.expected_ref
    )
    if stable_commit != args.expected_ref or stable_root != args.stable_worldmodel_root.expanduser().resolve():
        raise RuntimeError("TwoRoom runtime import identity drifted")
    tworoom.register_tworoom_eval_env()
    process = tworoom.frozen_normalizer_process(Path(inputs["normalizer"]["path"]))
    if not {"action", "proprio"}.issubset(process):
        raise RuntimeError("Frozen TwoRoom normalizer contract drifted")
    dataset = tworoom._h5_dataset(swm, Path(inputs["source_dataset"]["path"]))
    entries = inputs["catalog_payload"]["entries"]
    chunks = dataset.load_chunk(
        [int(row["episode"]) for row in entries],
        [int(row["start_step"]) for row in entries],
        [int(row["start_step"]) + 26 for row in entries],
    )
    if len(chunks) != 300:
        raise RuntimeError("Frozen TwoRoom catalog rows are not loadable")
    world = swm.World(
        "swm/TwoRoom-v1",
        num_envs=1,
        max_episode_steps=100,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )
    try:
        target = world.envs.envs[0].unwrapped
        for callable_row in tworoom._original_callables():
            if not callable(getattr(target, callable_row["method"], None)):
                raise RuntimeError(
                    f"TwoRoom task callable missing: {callable_row['method']}"
                )
    finally:
        world.close()
    return {
        "runtime": runtime,
        "model": model_row,
        "inputs": {
            key: value
            for key, value in inputs.items()
            if key != "catalog_payload"
        },
        "runtime_preflight": {
            "catalog_rows_loaded": 300,
            "world_constructed": True,
            "task_callables_verified": True,
        },
        "cem_episodes_consumed": 0,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _install_legacy_loader(
        args.stable_worldmodel_root, args.expected_ref
    )
    inputs = _verified_inputs(args)
    checkpoint = inputs["checkpoint"]
    catalog = inputs["catalog"]
    normalizer = inputs["normalizer"]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the scientific output before any CEM work.  A failed job leaves
    # the reservation in place and therefore cannot be silently retried.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(
            {"status": "reserved_before_cem", "seed": args.seed},
            stream,
            sort_keys=True,
        )
        stream.write("\n")
    runner_args = argparse.Namespace(
        catalog=Path(catalog["path"]),
        checkpoint=Path(checkpoint["path"]),
        normalizer=Path(normalizer["path"]),
        output=output,
        seed=args.seed,
        stablewm_repo=str(args.stable_worldmodel_root.expanduser().resolve()),
        stablewm_ref=args.expected_ref,
        device=args.device,
        eval_budget=50,
        horizon=5,
        receding_horizon=5,
        cem_samples=300,
        cem_steps=30,
        cem_topk=30,
        expected_history_size=3,
    )
    report = tworoom.run(runner_args)
    if report["stable_worldmodel"]["commit"] != args.expected_ref:
        raise RuntimeError("TwoRoom runtime identity drifted")
    report["frozen_input_preflight"] = {
        "runtime": runtime,
        **{
            key: value
            for key, value in inputs.items()
            if key != "catalog_payload"
        },
        "passed": True,
    }
    # ``run`` already created the file. Rewrite once to include the complete
    # preflight closure; the output namespace was exclusive before execution.
    tworoom.write_json(output, report)
    return {
        "report": str(output),
        "sha256": base.file_sha256(output),
        "aggregate": report["aggregate"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight-model", "eval"))
    parser.add_argument("--stable-worldmodel-root", type=Path, required=True)
    parser.add_argument("--expected-ref", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-size", type=int, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-config-size", type=int, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--expected-catalog-sha256", required=True)
    parser.add_argument("--expected-catalog-size", type=int, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--expected-normalizer-sha256", required=True)
    parser.add_argument("--expected-normalizer-size", type=int, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-source-size", type=int, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "eval":
        required = (
            "seed",
            "output",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error("eval requires: " + ", ".join(missing))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = preflight(args) if args.command == "preflight-model" else evaluate(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
