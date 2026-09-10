#!/usr/bin/env python3
"""Export the door Development validation catalog to Lance + isolation audit.

Reads the Development catalog built by
``build_tworoom_hidden_passage_h3_validation.py`` under the
``tworoom_hidden_passage_history3_dev_structural_parity_v1`` config (the
Public-Test-structured Development replica on the frozen loader_val doors)
and writes one Lance table per (door position, history condition) using the
pinned StableWM collection path (``World.collect`` + the ContextWorld Lance
writer), so the payload is registerable by
``stablewm_bundle.resolve_contextworld_development_payload`` (which only
accepts ``.lance`` directories).

Every episode is one (static query, history condition) row-group of 20 raw
steps whose block-boundary frames are exactly the frozen npz arrays:

* ``observed_passable``    episode: rule=passable, frames = passable history
  + ``target_passable``;
* ``observed_blocked``     episode: rule=blocked, frames = blocked history
  + ``target_blocked``;
* ``did_not_attempt_crossing`` episode: no-attempt reset under rule=passable
  (the no-attempt history is rule-invariant), frames = no-attempt history +
  ``target_passable``.

Stratification keys (``static_query_id``, ``query_id``, ``eval_seed``,
``evaluation_index``, ``direction``, condition, door position) are carried as
per-episode constant columns (``dev_*``) because the pinned Lance schema has
no episode side table.

The script also runs the Test-isolation audit: the Development queries must
not share any door position, template id, static-query id, or rendered query
pixel hash with the frozen Public Test validation catalog.
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
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.hidden_passage import (
    DIRECTIONS,
    HiddenPassageTemplate,
    simulate_template,
)
from contextworld.evaluation.hidden_passage_validation import (
    HISTORY_CONDITIONS,
    TRUE_RULES,
    file_sha256,
    make_no_attempt_template,
)
from contextworld.evaluation.hidden_passage_env import register_hidden_passage_env
from contextworld.evaluation.hidden_passage_lance import (
    _ScriptedPolicy,
    _collection_actions,
)
from contextworld.evaluation.hidden_passage_h3_data import _OneEpisodeCapture
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.lance import build_lance_writer
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel

PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
CONDITION_RULE = {
    "observed_passable": "passable",
    "observed_blocked": "blocked",
    # The no-crossing history never touches the wall, so it is rule-invariant;
    # its episode runs under the passable rule and therefore ends at the
    # passable true future (a duplicate of that target, kept for a uniform
    # four-frame episode contract).
    "did_not_attempt_crossing": "passable",
}
PIXEL_CODEC = {"format": "png", "compress_level": 1, "lossless": True}
EPISODE_ROWS = 20


class _EnrichingWriter:
    """World.collect writer that injects constant per-episode dev columns."""

    def __init__(self, base: Any, metadata: list[dict[str, Any]]) -> None:
        self._base = base
        self._metadata = metadata

    def __enter__(self) -> "_EnrichingWriter":
        self._base.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._base.__exit__(*exc)

    def write_episodes(self, episodes: Any) -> None:
        def enriched():
            for index, episode in enumerate(episodes):
                if index >= len(self._metadata):
                    raise RuntimeError("World produced more episodes than planned")
                meta = self._metadata[index]
                rows = len(episode["action"])
                for key, value in meta["strings"].items():
                    episode[key] = [str(value)] * rows
                for key, value in meta["floats"].items():
                    episode[key] = [(float(value),)] * rows
                yield episode

            if len(self._metadata) != index + 1:
                raise RuntimeError(
                    f"World produced {index + 1} episodes, planned "
                    f"{len(self._metadata)}"
                )

        self._base.write_episodes(enriched())


def _condition_template(
    bundle: dict[str, Any],
    condition: str,
    passable_query_state: np.ndarray,
) -> HiddenPassageTemplate:
    template = HiddenPassageTemplate(**bundle["template"])
    if condition == "did_not_attempt_crossing":
        return make_no_attempt_template(template, passable_query_state)
    return template


def _episodes_for_bundle(
    world: Any,
    swm: Any,
    bundle: dict[str, Any],
    npz: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Collect the three condition episodes of one static query."""

    query_state = np.asarray(npz["query_state"], dtype=np.float32)
    episodes = []
    for condition in HISTORY_CONDITIONS:
        rule = CONDITION_RULE[condition]
        template = _condition_template(bundle, condition, query_state)
        reference = simulate_template(template, rule=rule)
        # The frozen npz is the authority: the collected episode must
        # reproduce its frames and actions exactly.
        expected_frames = np.concatenate(
            [
                np.asarray(npz[f"{condition}_history_pixels"], dtype=np.uint8),
                np.asarray(npz[f"target_{rule}_pixels"], dtype=np.uint8)[None],
            ],
            axis=0,
        )
        expected_reference_frames = np.concatenate(
            [
                np.asarray(reference["history_pixels"], dtype=np.uint8),
                np.asarray(reference["target_pixels"], dtype=np.uint8)[None],
            ],
            axis=0,
        )
        if not np.array_equal(expected_frames, expected_reference_frames):
            raise RuntimeError(
                f"{bundle['query_id']}/{condition}: fresh simulation differs "
                "from the frozen catalog payload"
            )
        expected_actions = np.concatenate(
            [
                np.asarray(reference["history_actions"], dtype=np.float32).reshape(-1, 2),
                np.asarray(reference["query_action"], dtype=np.float32),
                np.zeros((5, 2), dtype=np.float32),
            ]
        )
        metadata = {
            "strings": {
                "dev_query_id": bundle["query_id"],
                "dev_static_query_id": bundle["static_query_id"],
                "dev_condition": condition,
                "dev_env_rule": rule,
                "dev_direction": bundle["direction"],
                "dev_template_id": template.template_id,
            },
            "floats": {
                "dev_eval_seed": int(bundle["eval_seed"]),
                "dev_evaluation_index": int(bundle["evaluation_index"]),
                "dev_door_position": int(template.door_position),
            },
        }
        capture_plan = _OneEpisodeCapture()
        policy_actions = _collection_actions(reference)
        policy = _ScriptedPolicy(policy_actions)
        world.set_policy(policy)
        world.collect(
            episodes=1,
            seed=int(template.simulator_seed),
            options={
                "variation": (
                    "agent.speed",
                    "door.number",
                    "door.position",
                    "passage.open",
                ),
                "variation_values": {
                    "agent.speed": np.asarray([5.0], dtype=np.float32),
                    "door.number": 1,
                    "door.position": np.asarray(
                        [int(template.door_position)] * 3, dtype=np.int64
                    ),
                    "passage.open": 1 if rule == "passable" else 0,
                },
                "state": np.asarray(template.reset_state, dtype=np.float32),
                "target_state": np.asarray(template.goal_state, dtype=np.float32),
            },
            writer=capture_plan,
            progress=False,
        )
        episode = capture_plan.one()
        pixels = np.stack([np.asarray(frame) for frame in episode["pixels"]])
        if pixels.shape != (EPISODE_ROWS, *expected_frames.shape[1:]):
            raise RuntimeError(
                f"{bundle['query_id']}/{condition}: episode has {pixels.shape} "
                "frames"
            )
        if not np.array_equal(
            pixels[[0, 5, 10, 15]], expected_frames
        ):
            raise RuntimeError(
                f"{bundle['query_id']}/{condition}: collected boundary frames "
                "differ from the frozen catalog payload"
            )
        if not np.array_equal(
            np.asarray(episode["action"], dtype=np.float32).reshape(-1, 2),
            expected_actions,
        ):
            raise RuntimeError(
                f"{bundle['query_id']}/{condition}: collected actions differ "
                "from the frozen catalog payload"
            )
        episodes.append({"condition": condition, "episode": episode, "meta": metadata})
    return episodes


def _audit_table(
    table_path: Path,
    rows: list[dict[str, Any]],
    npz_by_query: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    import lance

    dataset = lance.dataset(table_path)
    table = dataset.to_table(
        columns=[
            "episode_idx",
            "step_idx",
            "pixels",
            "action",
            "dev_query_id",
            "dev_condition",
            "dev_eval_seed",
        ]
    )
    episode_ids = np.asarray(table["episode_idx"].to_numpy(), dtype=np.int64)
    step_ids = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
    condition_column = table["dev_condition"].to_pylist()
    query_column = table["dev_query_id"].to_pylist()
    seed_column = np.asarray(
        [value[0] for value in table["dev_eval_seed"].to_pylist()]
    )
    from PIL import Image

    def decode(blob: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(bytes(blob))) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    failures = []
    for index, row in enumerate(rows):
        selection = np.flatnonzero(episode_ids == index)
        if not np.array_equal(step_ids[selection], np.arange(EPISODE_ROWS)):
            failures.append(f"{row['query_id']}: step indices are not a 20-row clip")
            continue
        npz = npz_by_query[row["query_id"]]
        expected = np.concatenate(
            [
                np.asarray(
                    npz[f"{row['condition']}_history_pixels"], dtype=np.uint8
                ),
                np.asarray(
                    npz[f"target_{CONDITION_RULE[row['condition']]}_pixels"],
                    dtype=np.uint8,
                )[None],
            ],
            axis=0,
        )
        blobs = table["pixels"].to_pylist()
        observed = np.stack(
            [decode(blobs[int(position)]) for position in selection[[0, 5, 10, 15]]]
        )
        if not np.array_equal(observed, expected):
            failures.append(f"{row['query_id']}/{row['condition']}: frame mismatch")
        row_conditions = np.asarray(condition_column)[selection]
        if any(value != row["condition"] for value in row_conditions):
            failures.append(f"{row['query_id']}: condition column is not constant")
        if query_column[selection[0]] != row["query_id"]:
            failures.append(f"{row['query_id']}: query id column mismatch")
        if int(seed_column[selection[0]]) != int(row["eval_seed"]):
            failures.append(f"{row['query_id']}: eval seed column mismatch")
    return {
        "table": str(table_path),
        "episodes": len(rows),
        "rows": int(len(episode_ids)),
        "failures": failures,
        "passed": not failures,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    catalog_path = args.catalog.resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("status") != "frozen_before_model_scoring":
        raise ValueError("Development catalog is not frozen")
    protocol = catalog["protocol"]
    if list(protocol["history_conditions"]) != list(HISTORY_CONDITIONS):
        raise ValueError("History conditions differ from the Test protocol")
    if list(protocol["true_future_rules"]) != list(TRUE_RULES):
        raise ValueError("True-future rules differ from the Test protocol")
    bundles = sorted(
        catalog["bundles"],
        key=lambda row: (int(row["eval_seed"]), int(row["evaluation_index"])),
    )
    if len(bundles) != int(catalog["summary"]["unique_queries"]):
        raise ValueError("Bundle count disagrees with the catalog summary")

    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    (output_root / "data").mkdir(parents=True)
    # Lance commits rename directories, which the shared output filesystem
    # rejects; stage tables on local disk and copy them in (the same pattern
    # as collect_hidden_passage_shard).
    staging_root = Path(args.staging_root).resolve()
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    npz_by_query: dict[str, dict[str, np.ndarray]] = {}
    for bundle in bundles:
        payload_path = resolve_contextworld_path(bundle["payload"], repo_root=ROOT)
        with np.load(payload_path, allow_pickle=False) as payload:
            npz_by_query[bundle["query_id"]] = {
                name: np.asarray(payload[name]).copy() for name in payload.files
            }

    swm, _, commit = load_stable_worldmodel(ROOT, args.stablewm_repo, PINNED_STABLEWM)
    if commit != PINNED_STABLEWM:
        raise RuntimeError(f"StableWM commit mismatch: {commit}")
    register_hidden_passage_env()
    world = swm.World(
        "contextworld/TwoRoomHiddenPassage-v1",
        num_envs=1,
        max_episode_steps=EPISODE_ROWS,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )

    by_door_condition: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    table_audits: list[dict[str, Any]] = []
    members: list[str] = []
    try:
        for position, bundle in enumerate(bundles):
            episodes = _episodes_for_bundle(world, swm, bundle, npz_by_query[bundle["query_id"]])
            for record in episodes:
                by_door_condition[
                    (
                        int(bundle["template"]["door_position"]),
                        record["condition"],
                    )
                ].append(
                    {
                        "query_id": bundle["query_id"],
                        "static_query_id": bundle["static_query_id"],
                        "eval_seed": int(bundle["eval_seed"]),
                        "direction": bundle["direction"],
                        "condition": record["condition"],
                        "meta": record["meta"],
                        "episode": record["episode"],
                    }
                )
            if position % 50 == 0:
                print(
                    f"[door-lance] collected {position + 1}/{len(bundles)} queries",
                    flush=True,
                )
    finally:
        world.close()

    for (door, condition), rows in sorted(by_door_condition.items()):
        fingerprint = hashlib.sha256(
            "\n".join(row["query_id"] for row in sorted(rows, key=lambda r: r["query_id"])).encode("utf-8")
        ).hexdigest()[:10]
        table_name = f"hpdev-d{door:03d}-{condition}-{fingerprint}.lance"
        staged_path = staging_root / table_name
        base_writer = build_lance_writer(
            swm, staged_path, pixel_codec=dict(PIXEL_CODEC)
        )
        metadata = [row["meta"] for row in rows]
        writer = _EnrichingWriter(base_writer, metadata)
        with writer:
            writer.write_episodes(iter(row["episode"] for row in rows))
        table_path = output_root / "data" / table_name
        shutil.copytree(staged_path, table_path)
        members.append(str(table_path.relative_to(output_root)))
        table_audits.append(
            _audit_table(
                table_path,
                [
                    {
                        "query_id": row["query_id"],
                        "condition": row["condition"],
                        "eval_seed": row["eval_seed"],
                    }
                    for row in rows
                ],
                npz_by_query,
            )
        )
        if not table_audits[-1]["passed"]:
            raise RuntimeError(f"Table audit failed: {table_audits[-1]}")

    # ---- Test isolation audit (the Public Test catalog is read-only input)
    test_catalog = json.loads(Path(args.test_catalog).read_text(encoding="utf-8"))
    test_doors = {int(b["template"]["door_position"]) for b in test_catalog["bundles"]}
    test_template_ids = {b["template"]["template_id"] for b in test_catalog["bundles"]}
    test_static_ids = {b["static_query_id"] for b in test_catalog["bundles"]}
    test_query_hashes = {
        b["query_pixels_sha256"] for b in test_catalog["bundles"]
    }
    dev_doors = {int(b["template"]["door_position"]) for b in bundles}
    dev_template_ids = {b["template"]["template_id"] for b in bundles}
    dev_static_ids = {b["static_query_id"] for b in bundles}
    dev_query_hashes = {b["query_pixels_sha256"] for b in bundles}
    isolation = {
        "test_catalog": str(args.test_catalog),
        "test_catalog_sha256": file_sha256(Path(args.test_catalog)),
        "dev_catalog": str(catalog_path),
        "dev_catalog_sha256": file_sha256(catalog_path),
        "checks": {
            "door_positions_disjoint": not (dev_doors & test_doors),
            "template_ids_disjoint": not (dev_template_ids & test_template_ids),
            "static_query_ids_disjoint": not (dev_static_ids & test_static_ids),
            "query_pixels_sha256_disjoint": not (
                dev_query_hashes & test_query_hashes
            ),
            "dev_doors": sorted(dev_doors),
            "test_doors": sorted(test_doors),
            "dev_query_count": len(bundles),
            "test_query_count": len(test_catalog["bundles"]),
            "intersections": {
                "doors": sorted(dev_doors & test_doors),
                "template_ids": sorted(dev_template_ids & test_template_ids),
                "static_query_ids": sorted(dev_static_ids & test_static_ids),
                "query_pixels_sha256": sorted(dev_query_hashes & test_query_hashes),
            },
        },
    }
    isolation["passed"] = all(
        value
        for key, value in isolation["checks"].items()
        if isinstance(value, bool)
    )

    by_seed = Counter(int(b["eval_seed"]) for b in bundles)
    by_seed_direction = Counter(
        (int(b["eval_seed"]), str(b["direction"])) for b in bundles
    )
    by_condition = Counter(row["condition"] for rows in by_door_condition.values() for row in rows)
    coverage = {
        "queries_by_eval_seed": dict(sorted(by_seed.items())),
        "queries_by_eval_seed_and_direction": {
            f"s{seed}/{direction}": by_seed_direction[(seed, direction)]
            for seed in sorted(by_seed)
            for direction in DIRECTIONS
        },
        "strata_count": len(by_seed_direction),
        "strata_size": sorted(set(by_seed_direction.values())),
        "episodes_by_condition": dict(sorted(by_condition.items())),
        "doors": sorted(dev_doors),
        "both_directions_covered": set(by_seed_direction) == {
            (seed, direction)
            for seed in sorted(by_seed)
            for direction in DIRECTIONS
        },
    }

    contract = {
        "schema_version": 1,
        "benchmark": catalog["benchmark"],
        "payload_kind": "development_structural_parity_v1",
        "reader_id": "contextworld_door_dev_structural_parity_lance_v1",
        "history_conditions": list(HISTORY_CONDITIONS),
        "true_future_rules": list(TRUE_RULES),
        "episode_layout": {
            "rows_per_episode": EPISODE_ROWS,
            "model_tokens": 4,
            "frames_at_steps": [0, 5, 10, 15],
            "token_semantics": "three history frames then the true future "
            "under dev_env_rule",
            "condition_to_target_rule": dict(CONDITION_RULE),
            "target_extraction": (
                "target_passable = frame 15 of the observed_passable episode; "
                "target_blocked = frame 15 of the observed_blocked episode; "
                "the did_not_attempt_crossing episode repeats target_passable "
                "at frame 15 and its rule-invariant history at frames 0-10"
            ),
            "pixel_codec": PIXEL_CODEC,
            "stratification_columns": [
                "dev_eval_seed",
                "dev_direction",
                "dev_static_query_id",
                "dev_query_id",
                "dev_evaluation_index",
                "dev_condition",
                "dev_door_position",
            ],
        },
        "lance_table_count": len(members),
        "members": members,
        "selection": {
            "kind": "frozen_validation_catalog_static_queries",
            "unique_queries": len(bundles),
            "eval_seeds": sorted(by_seed),
            "unique_queries_per_eval_seed": 50,
            "per_direction_per_eval_seed": 25,
        },
        "source_catalog": {
            "path": str(catalog_path),
            "sha256": file_sha256(catalog_path),
        },
        "stable_worldmodel_commit": PINNED_STABLEWM,
    }

    report = {
        "schema_version": 1,
        "benchmark": catalog["benchmark"],
        "status": "passed"
        if isolation["passed"] and all(a["passed"] for a in table_audits)
        else "failed",
        "isolation_audit": isolation,
        "coverage": coverage,
        "table_audits": table_audits,
        "tables": len(members),
        "episodes": sum(a["episodes"] for a in table_audits),
        "rows": sum(a["rows"] for a in table_audits),
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
        "--catalog",
        type=Path,
        required=True,
        help="Development validation catalog.json (Test structure)",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--test-catalog",
        type=Path,
        default=Path(
            "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
            "ContextWorld-v1-full/artifacts/evaluation/history3/"
            "hidden_passage_validation_v2/catalog.json"
        ),
    )
    parser.add_argument("--stablewm-repo", default=None)
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/tmp/contextworld-dev-parity-lance-staging"),
        help="Local-disk staging for Lance writes (the output FS rejects "
        "Lance's directory renames); tables are copied to the output root "
        "after the write commits.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.stablewm_repo is None:
        arguments.stablewm_repo = (
            Path(__file__).resolve().parents[1] / "../stable-worldmodel"
        )
    result = run(arguments)
    print(
        json.dumps(
            {
                "status": result["status"],
                "tables": result["tables"],
                "episodes": result["episodes"],
                "isolation_passed": result["isolation_audit"]["passed"],
                "coverage": result["coverage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
