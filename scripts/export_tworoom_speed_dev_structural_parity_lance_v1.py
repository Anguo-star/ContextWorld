#!/usr/bin/env python3
"""Export the Development speed catalogs to Lance + isolation audit.

Reads the four Development track catalogs built by
``build_tworoom_speed_dev_structural_parity_v1.py`` (the
Public-Test-structured Development replica of
``tworoom_history3_speed_multistep_extrap_v5``) and writes one Lance table
per (track, history condition).  Every episode is one (static query,
history condition, reference speed) evaluation sample of 40 raw steps
(8 action blocks):

* rows 0-9:   the two context blocks simulated at the condition speed
              (history tokens 1 and 2 come from rows 0 and 5);
* rows 10-34: the five query/future blocks simulated at the reference
              speed (query frame = row 10, true futures 1..5 = rows
              15, 20, 25, 30, 35);
* rows 35-39: a zero-action tail so the last future frame owns a row.

Block-boundary frames are verified against the frozen npz payload of the
catalog bundle during generation and re-verified from the written table,
so the Lance payload is exactly the catalog content.  Stratification keys
(eval_seed, evaluation_index, static/query ids, condition, speeds) are
carried as per-episode constant columns because the pinned Lance schema
has no episode side table.  The aggregation step re-runs the
Test-isolation audit on the finished catalogs' identities.

Tables are independent: ``--only track:condition`` exports a single table
(writing ``audits/<table>.audit.json``), completed tables are skipped on
re-run, and a plain invocation aggregates every audit into the global
reports.  This supports running several tables as parallel processes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_catalog import _factor_options
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.lance import build_lance_writer
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel

PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
PIXEL_CODEC = {"format": "png", "compress_level": 1, "lossless": True}
EPISODE_ROWS = 40
TOKEN_ROWS = (0, 5, 10, 15, 20, 25, 30, 35)

TEST_CATALOG_DIR_DEFAULT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "ContextWorld-v1-full/artifacts/evaluation/history3/"
    "speed_multistep_extrap_v5/catalogs"
)
TEST_TRACK_CATALOGS = (
    "seen_for_multi.json",
    "unseen_interpolation.json",
    "extrapolation_low.json",
    "extrapolation_high.json",
)


def _simulate_rows(
    factors: dict[str, Any],
    reset_state: np.ndarray,
    goal_state: np.ndarray,
    action_blocks: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-raw-step variant of ``_simulate_blocks``.

    Returns ``(frames, states)`` with ``frames[i]``/``states[i]`` the render
    and agent state after ``i`` raw steps, so a K-block rollout yields K * 5
    + 1 rows.  Boundary rows are bit-identical to ``_simulate_blocks``
    because both drive the same env with the same reset options and renders
    do not mutate state (verified by the caller against the frozen npz).
    """

    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    env = TwoRoomEnv(render_mode="rgb_array")
    frames: list[np.ndarray] = []
    states: list[np.ndarray] = []
    try:
        env.reset(
            seed=int(seed),
            options={
                "variation": (),
                "variation_values": _factor_options(factors),
                "state": np.asarray(reset_state, dtype=np.float32).copy(),
                "target_state": np.asarray(goal_state, dtype=np.float32).copy(),
            },
        )
        frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
        states.append(env.agent_position.detach().cpu().numpy().copy())
        for block in np.asarray(action_blocks, dtype=np.float32):
            for action in block:
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    raise RuntimeError(
                        "Per-step rollout terminated: "
                        f"factors={factors}, action={action.tolist()}"
                    )
                frames.append(
                    np.asarray(env.render(), dtype=np.uint8).copy()
                )
                states.append(
                    env.agent_position.detach().cpu().numpy().copy()
                )
    finally:
        env.close()
    return np.stack(frames), np.stack(states).astype(np.float32)


def _load_npz(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    payload_path = resolve_contextworld_path(bundle["payload"], repo_root=ROOT)
    with np.load(payload_path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]).copy() for name in payload.files}


def _episode_for(
    bundle: dict[str, Any],
    condition: str,
    npz: dict[str, np.ndarray],
    context_rows: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Assemble one 40-row episode dict from cached context rows."""

    reference_speed = float(bundle["query_factors"]["agent.speed"])
    condition_speed = float(
        bundle["conditions"][condition]["factors"]["agent.speed"]
    )
    door = int(bundle["query_factors"]["door.position"])
    template = bundle["template"]
    reset = np.asarray(template["reset_state"], dtype=np.float32)
    goal = np.asarray(template["goal_state"], dtype=np.float32)
    future_actions = np.concatenate(
        [
            np.asarray(npz["future_actions"], dtype=np.float32),
            np.zeros((1, 5, 2), dtype=np.float32),
        ]
    )
    future_frames, future_states = _simulate_rows(
        {"agent.speed": reference_speed, "door.position": door},
        reset,
        goal,
        future_actions,
        seed=int(bundle["simulator_seed"]),
    )
    context_frames, context_states = context_rows
    frames = np.concatenate([context_frames[:10], future_frames[:30]], axis=0)
    states = np.concatenate([context_states[:10], future_states[:30]], axis=0)
    if frames.shape[0] != EPISODE_ROWS or states.shape[0] != EPISODE_ROWS:
        raise RuntimeError("Episode row assembly produced the wrong length")

    # Verify against the frozen catalog payload.
    expected_context = npz[f"context_b2_{condition}_pixels"]
    expected_context_next = npz[f"context_b2_{condition}_next_pixels"]
    checks = {
        "context_first": np.array_equal(frames[0], expected_context[0]),
        "context_middle": np.array_equal(
            frames[5], expected_context_next[0]
        ),
        "query": np.array_equal(frames[10], npz["query_pixels"]),
        "future_1": np.array_equal(
            frames[15], npz["future_next_pixels"][0]
        ),
        "future_2": np.array_equal(
            frames[20], npz["future_next_pixels"][1]
        ),
        "future_3": np.array_equal(
            frames[25], npz["future_next_pixels"][2]
        ),
        "future_4": np.array_equal(
            frames[30], npz["future_next_pixels"][3]
        ),
        "future_5": np.array_equal(
            frames[35], npz["future_next_pixels"][4]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"{bundle['query_id']}/{condition}: per-step rollout differs "
            f"from the frozen payload at {failed}"
        )

    actions = np.concatenate(
        [
            np.asarray(
                npz[f"context_b2_{condition}_actions"], dtype=np.float32
            ),
            future_actions,
        ]
    ).reshape(EPISODE_ROWS, 2)
    goal_rows = np.repeat(
        np.asarray(goal, dtype=np.float32)[None], EPISODE_ROWS, axis=0
    )
    truncated = np.zeros((EPISODE_ROWS, 1), dtype=np.float32)
    truncated[-1] = 1.0
    return {
        "pixels": list(frames),
        "action": list(actions),
        "proprio": list(states),
        "state": list(states.copy()),
        "goal_state": list(goal_rows),
        "terminated": list(np.zeros((EPISODE_ROWS, 1), dtype=np.float32)),
        "truncated": list(truncated),
        "variation_agent_speed": list(
            np.full((EPISODE_ROWS, 1), reference_speed, dtype=np.float32)
        ),
        "dev_query_id": [str(bundle["query_id"])] * EPISODE_ROWS,
        "dev_static_query_id": [str(bundle["static_query_id"])]
        * EPISODE_ROWS,
        "dev_track": [str(bundle["track"])] * EPISODE_ROWS,
        "dev_condition": [str(condition)] * EPISODE_ROWS,
        "dev_template_id": [str(template["template_id"])] * EPISODE_ROWS,
        "dev_eval_seed": list(
            np.full((EPISODE_ROWS, 1), int(bundle["eval_seed"]), np.float32)
        ),
        "dev_evaluation_index": list(
            np.full(
                (EPISODE_ROWS, 1),
                int(bundle["evaluation_index"]),
                np.float32,
            )
        ),
        "dev_reference_speed": list(
            np.full((EPISODE_ROWS, 1), reference_speed, np.float32)
        ),
        "dev_condition_speed": list(
            np.full((EPISODE_ROWS, 1), condition_speed, np.float32)
        ),
    }


def _context_rows_for(
    bundle: dict[str, Any], condition: str
) -> tuple[np.ndarray, np.ndarray]:
    npz = _load_npz(bundle)
    condition_speed = float(
        bundle["conditions"][condition]["factors"]["agent.speed"]
    )
    door = int(bundle["query_factors"]["door.position"])
    template = bundle["template"]
    context_actions = np.asarray(
        npz[f"context_b2_{condition}_actions"], dtype=np.float32
    )
    frames, states = _simulate_rows(
        {"agent.speed": condition_speed, "door.position": door},
        np.asarray(template["reset_state"], dtype=np.float32),
        np.asarray(template["goal_state"], dtype=np.float32),
        context_actions,
        seed=int(bundle["simulator_seed"]),
    )
    if frames.shape[0] != 11:
        raise RuntimeError("Context rollout does not cover 10 raw steps")
    return frames, states


def _audit_table(
    table_path: Path,
    expected_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    import lance
    from PIL import Image

    def decode(blob: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(bytes(blob))) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    dataset = lance.dataset(table_path)
    columns = [
        "episode_idx",
        "step_idx",
        "pixels",
        "action",
        "dev_query_id",
        "dev_condition",
        "dev_reference_speed",
        "dev_eval_seed",
    ]
    table = dataset.to_table(columns=columns)
    episode_ids = np.asarray(table["episode_idx"].to_numpy(), dtype=np.int64)
    step_ids = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
    blobs = table["pixels"].to_pylist()
    conditions = table["dev_condition"].to_pylist()
    query_ids = table["dev_query_id"].to_pylist()
    reference_speeds = np.asarray(
        [value[0] for value in table["dev_reference_speed"].to_pylist()]
    )
    seeds = np.asarray([value[0] for value in table["dev_eval_seed"].to_pylist()])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)

    failures: list[str] = []
    for index, expected in enumerate(expected_episodes):
        rows = np.flatnonzero(episode_ids == index)
        if len(rows) != EPISODE_ROWS or not np.array_equal(
            step_ids[rows], np.arange(EPISODE_ROWS)
        ):
            failures.append(f"{expected['query_id']}: row layout broken")
            continue
        observed = np.stack(
            [decode(blobs[int(row)]) for row in rows[list(TOKEN_ROWS)]]
        )
        if not np.array_equal(observed, expected["token_frames"]):
            failures.append(f"{expected['query_id']}: token frames differ")
        if not np.array_equal(
            actions[rows].reshape(EPISODE_ROWS, 2), expected["actions"]
        ):
            failures.append(f"{expected['query_id']}: actions differ")
        row_conditions = np.asarray(conditions)[rows]
        if any(value != expected["condition"] for value in row_conditions):
            failures.append(f"{expected['query_id']}: condition column differs")
        if query_ids[rows[0]] != expected["query_id"]:
            failures.append(f"{expected['query_id']}: query id column differs")
        if (
            np.float32(reference_speeds[rows[0]])
            != np.float32(expected["reference_speed"])
            or int(seeds[rows[0]]) != int(expected["eval_seed"])
        ):
            failures.append(f"{expected['query_id']}: speed/seed column differs")
    return {
        "episodes": len(expected_episodes),
        "rows": int(len(episode_ids)),
        "failures": failures,
        "passed": not failures,
    }


def _table_spec(
    config: dict[str, Any], track_name: str, condition: str
) -> dict[str, Any]:
    track = config["data"]["tracks"][track_name]
    catalog_path = resolve_contextworld_path(track["catalog"], repo_root=ROOT)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    by_static: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bundle in catalog["bundles"]:
        by_static[str(bundle["static_query_id"])].append(bundle)
    ordered_statics = sorted(by_static)
    episode_keys = [
        [
            str(bundle["query_id"]),
            float(bundle["query_factors"]["agent.speed"]),
        ]
        for static_id in ordered_statics
        for bundle in sorted(
            by_static[static_id],
            key=lambda b: float(b["query_factors"]["agent.speed"]),
        )
    ]
    fingerprint = hashlib.sha256(
        json.dumps(episode_keys).encode("utf-8")
    ).hexdigest()[:10]
    table_name = f"twmsdev-{track_name}-{condition}-{fingerprint}.lance"
    return {
        "track": str(track_name),
        "condition": str(condition),
        "table_name": table_name,
        "catalog_path": str(catalog_path),
        "catalog": catalog,
        "by_static": by_static,
        "ordered_statics": ordered_statics,
        "reference_speeds": [float(s) for s in track["speeds"]],
        "episode_keys": episode_keys,
    }


def _export_one_table(
    spec: dict[str, Any],
    output_root: Path,
    staging_root: Path,
) -> dict[str, Any]:
    swm, _, commit = load_stable_worldmodel(
        ROOT, _export_one_table.stablewm_repo, PINNED_STABLEWM
    )
    if commit != PINNED_STABLEWM:
        raise RuntimeError(f"StableWM commit mismatch: {commit}")
    condition = spec["condition"]
    ordered_statics = spec["ordered_statics"]
    by_static = spec["by_static"]
    expected_meta: list[dict[str, Any]] = []
    started = time.monotonic()

    def audit_generator() -> Iterator[dict[str, Any]]:
        for static_id in ordered_statics:
            group = sorted(
                by_static[static_id],
                key=lambda b: float(b["query_factors"]["agent.speed"]),
            )
            context_cache = None
            for bundle in group:
                npz = _load_npz(bundle)
                if context_cache is None:
                    context_cache = (static_id, _context_rows_for(bundle, condition))
                episode = _episode_for(bundle, condition, npz, context_cache[1])
                expected_meta.append(
                    {
                        "query_id": str(bundle["query_id"]),
                        "condition": condition,
                        "reference_speed": float(
                            bundle["query_factors"]["agent.speed"]
                        ),
                        "eval_seed": int(bundle["eval_seed"]),
                        "token_frames": np.stack(
                            [
                                np.asarray(episode["pixels"][row])
                                for row in TOKEN_ROWS
                            ]
                        ),
                        "actions": np.asarray(
                            episode["action"], dtype=np.float32
                        ).reshape(EPISODE_ROWS, 2),
                    }
                )
                yield episode

    staged = staging_root / spec["table_name"]
    if staged.exists():
        shutil.rmtree(staged)
    writer = build_lance_writer(swm, staged, pixel_codec=dict(PIXEL_CODEC))
    with writer as opened:
        opened.write_episodes(audit_generator())
    table_path = output_root / "data" / spec["table_name"]
    if table_path.exists():
        shutil.rmtree(table_path)
    shutil.copytree(staged, table_path)
    shutil.rmtree(staged, ignore_errors=True)
    audit = _audit_table(table_path, expected_meta)
    audit.update(
        {
            "table": str(table_path.relative_to(output_root)),
            "track": spec["track"],
            "condition": condition,
            "runtime_seconds": round(time.monotonic() - started, 1),
        }
    )
    if not audit["passed"]:
        raise RuntimeError(f"Table audit failed: {audit}")
    write_json(output_root / "audits" / f"{spec['table_name']}.audit.json", audit)
    return audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    config_path = args.config.resolve()
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_root).resolve()
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "audits").mkdir(parents=True, exist_ok=True)
    staging_root = Path(args.staging_root).resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    _export_one_table.stablewm_repo = args.stablewm_repo

    only = None
    if args.only:
        only = set(str(value) for value in args.only)

    specs = []
    for track_name in config["data"]["tracks"]:
        catalog_path = resolve_contextworld_path(
            config["data"]["tracks"][track_name]["catalog"], repo_root=ROOT
        )
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        for condition in catalog["summary"]["history_conditions"]:
            spec = _table_spec(config, str(track_name), str(condition))
            specs.append(spec)

    if only is not None:
        selected = [s for s in specs if f"{s['track']}:{s['condition']}" in only]
        unknown = only - {
            f"{s['track']}:{s['condition']}" for s in specs
        }
        if unknown:
            raise ValueError(f"Unknown --only targets: {sorted(unknown)}")
        results = []
        for spec in selected:
            results.append(_export_one_table(spec, output_root, staging_root))
            print(
                f"[speed-lance] {spec['track']}/{spec['condition']}: "
                f"{results[-1]['episodes']} episodes, {results[-1]['rows']} rows",
                flush=True,
            )
        return {"status": "passed", "tables": results}

    # Aggregation pass: reuse tables whose audit already passed.
    audits: list[dict[str, Any]] = []
    reused = 0
    for spec in specs:
        audit_path = output_root / "audits" / f"{spec['table_name']}.audit.json"
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("passed") and (
                output_root / "data" / spec["table_name"]
            ).is_dir():
                audits.append(audit)
                reused += 1
                continue
        audits.append(_export_one_table(spec, output_root, staging_root))
        print(
            f"[speed-lance] {spec['track']}/{spec['condition']}: "
            f"{audits[-1]['episodes']} episodes, {audits[-1]['rows']} rows",
            flush=True,
        )
    if len(audits) != len(specs) or not all(a["passed"] for a in audits):
        raise RuntimeError("Not all speed tables passed their audits")

    members = [
        {
            "track": audit["track"],
            "condition": audit["condition"],
            "path": str(Path("data") / Path(audit["table"]).name),
        }
        for audit in sorted(audits, key=lambda a: (a["track"], a["condition"]))
    ]

    # ---- coverage + isolation from the four Development catalogs
    coverage: dict[str, Any] = {"tracks": {}}
    dev_static_ids: set[str] = set()
    dev_query_hashes: set[str] = set()
    dev_resets: set[tuple[float, float]] = set()
    for track_name, track in config["data"]["tracks"].items():
        catalog_path = resolve_contextworld_path(track["catalog"], repo_root=ROOT)
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        by_static = defaultdict(list)
        for bundle in catalog["bundles"]:
            by_static[str(bundle["static_query_id"])].append(bundle)
        dev_static_ids.update(
            str(b["static_query_id"]) for b in catalog["bundles"]
        )
        dev_query_hashes.update(
            str(b["query_pixels_sha256"]) for b in catalog["bundles"]
        )
        dev_resets.update(
            tuple(map(float, b["template"]["reset_state"]))
            for b in catalog["bundles"]
        )
        seed_counts = Counter(int(b["eval_seed"]) for b in catalog["bundles"])
        coverage["tracks"][str(track_name)] = {
            "static_queries": len(by_static),
            "conditions": list(catalog["summary"]["history_conditions"]),
            "reference_speeds": [float(s) for s in track["speeds"]],
            "episodes_by_condition": {
                audit["condition"]: audit["episodes"]
                for audit in audits
                if audit["track"] == str(track_name)
            },
            "episodes_by_eval_seed": dict(sorted(seed_counts.items())),
        }

    test_resets: set[tuple[float, float]] = set()
    test_static_ids: set[str] = set()
    test_query_hashes: set[str] = set()
    test_catalog_summaries = {}
    for name in TEST_TRACK_CATALOGS:
        path = Path(args.test_catalog_dir) / name
        test_catalog = json.loads(path.read_text(encoding="utf-8"))
        test_resets.update(
            tuple(map(float, g["reset_state"]))
            for g in test_catalog["geometry_bank"]
        )
        test_static_ids.update(
            str(b["static_query_id"]) for b in test_catalog["bundles"]
        )
        test_query_hashes.update(
            str(b["query_pixels_sha256"]) for b in test_catalog["bundles"]
        )
        test_catalog_summaries[name] = {
            "bundles": len(test_catalog["bundles"]),
            "geometries": len(test_catalog["geometry_bank"]),
        }
    isolation = {
        "test_catalog_dir": str(args.test_catalog_dir),
        "test_catalogs": test_catalog_summaries,
        "checks": {
            "reset_states_disjoint": not (dev_resets & test_resets),
            "static_query_ids_disjoint": not (
                dev_static_ids & test_static_ids
            ),
            "query_pixels_sha256_disjoint": not (
                dev_query_hashes & test_query_hashes
            ),
        },
        "intersections": {
            "reset_states": sorted(map(list, dev_resets & test_resets)),
            "static_query_ids": sorted(dev_static_ids & test_static_ids),
            "query_pixels_sha256": sorted(
                dev_query_hashes & test_query_hashes
            ),
        },
        "counts": {
            "dev_reset_states": len(dev_resets),
            "dev_static_query_ids": len(dev_static_ids),
            "dev_query_pixel_hashes": len(dev_query_hashes),
            "test_reset_states": len(test_resets),
            "test_static_query_ids": len(test_static_ids),
            "test_query_pixel_hashes": len(test_query_hashes),
        },
    }
    isolation["passed"] = all(isolation["checks"].values())

    contract = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "payload_kind": "development_structural_parity_v1",
        "reader_id": "contextworld_speed_dev_structural_parity_lance_v1",
        "episode_layout": {
            "rows_per_episode": EPISODE_ROWS,
            "model_tokens": 8,
            "frames_at_steps": list(TOKEN_ROWS),
            "token_semantics": (
                "rows 0/5/10 are the three history tokens (context at the "
                "condition speed, ending at the shared query); rows "
                "15/20/25/30/35 are the true futures after 1..5 query "
                "blocks at the reference speed"
            ),
            "history_condition_column": "dev_condition",
            "reference_speed_column": "dev_reference_speed",
            "condition_speed_column": "dev_condition_speed",
            "variation_agent_speed_semantics": (
                "the reference (query) speed; the condition speed of rows "
                "0-9 is dev_condition_speed"
            ),
            "pixel_codec": PIXEL_CODEC,
        },
        "lance_table_count": len(members),
        "members": members,
        "selection": {
            "kind": "frozen_dev_track_catalogs",
            "tracks": {
                name: {
                    "reference_speeds": [float(s) for s in track["speeds"]],
                    "catalog": str(
                        resolve_contextworld_path(
                            track["catalog"], repo_root=ROOT
                        )
                    ),
                }
                for name, track in config["data"]["tracks"].items()
            },
            "eval_seeds": config["evaluation"]["eval_seeds"],
            "unique_queries_per_reference_speed_per_seed": int(
                config["evaluation"][
                    "unique_queries_per_reference_speed_per_seed"
                ]
            ),
        },
        "stable_worldmodel_commit": PINNED_STABLEWM,
    }

    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed"
        if isolation["passed"] and all(a["passed"] for a in audits)
        else "failed",
        "isolation_audit": isolation,
        "coverage": coverage,
        "table_audits": audits,
        "tables": len(members),
        "tables_reused_from_previous_run": reused,
        "episodes": sum(a["episodes"] for a in audits),
        "rows": sum(a["rows"] for a in audits),
        "runtime_seconds": round(time.monotonic() - started, 1),
    }
    write_json(output_root / "isolation_audit.json", isolation)
    write_json(output_root / "coverage.json", coverage)
    write_json(output_root / "development_registry_contract.json", contract)
    write_json(output_root / "lance_export_report.json", report)
    shutil.rmtree(staging_root, ignore_errors=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/benchmark/tworoom_speed_dev_structural_parity_v1.yaml"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument(
        "--test-catalog-dir", type=Path, default=TEST_CATALOG_DIR_DEFAULT
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/tmp/contextworld-speed-dev-lance-staging"),
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="track:condition table to export (repeatable); skips the "
        "global aggregation reports",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    if "tables" in result and isinstance(result.get("tables"), list) and result[
        "status"
    ] == "passed" and "rows" not in result:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "exported": [
                        f"{row['track']}/{row['condition']}"
                        for row in result["tables"]
                    ],
                },
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "tables": result["tables"],
                    "episodes": result["episodes"],
                    "rows": result["rows"],
                    "isolation_passed": result["isolation_audit"]["passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
