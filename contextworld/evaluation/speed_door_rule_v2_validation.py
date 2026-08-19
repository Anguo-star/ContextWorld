from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)

from .speed_door_rule_v2_design import (
    require_valid_speed_door_rule_v2_design,
)
from .speed_door_rule_v2_feasibility import (
    audit_v2_query_bundles,
    future_action_blocks,
    simulate_v2_future,
)


RULES = ("passable", "blocked")
HORIZONS = (1, 2)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def factor_key(speed: float, rule: str) -> str:
    return f"s{float(speed):04.1f}_{rule}".replace(".", "p")


def passable_target_key(speed: float) -> str:
    return f"passable_s{float(speed):04.1f}".replace(".", "p")


def physical_target_keys(speeds: tuple[float, ...]) -> tuple[str, ...]:
    return ("blocked",) + tuple(
        passable_target_key(speed) for speed in speeds
    )


def _payload_content_sha256(arrays: dict[str, np.ndarray]) -> str:
    return canonical_sha256(
        {
            name: array_sha256(value)
            for name, value in sorted(arrays.items())
        }
    )


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_v1_source(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], Path]:
    specification = config["evaluation"]["paired_query_source"]
    path = resolve_contextworld_path(
        specification["catalog"], repo_root=repo_root
    )
    observed = file_sha256(path)
    expected = str(specification["catalog_sha256"])
    if observed != expected:
        raise RuntimeError(
            f"Paired query catalog hash mismatch: {observed} != {expected}"
        )
    with path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    expected_queries = int(config["evaluation"]["unique_base_queries"])
    if (
        catalog.get("status") != "frozen_before_model_scoring"
        or len(catalog.get("bundles", ())) != expected_queries
    ):
        raise RuntimeError("Paired v1 query catalog identity/count failed")
    return catalog, path


def _base_payload(
    bundle: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, np.ndarray]:
    path = resolve_contextworld_path(
        bundle["payload"], repo_root=repo_root
    )
    if file_sha256(path) != str(bundle["payload_sha256"]):
        raise RuntimeError(f"Paired query payload hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            name: np.asarray(payload[name]).copy()
            for name in payload.files
        }
    if _payload_content_sha256(arrays) != str(
        bundle["payload_content_sha256"]
    ):
        raise RuntimeError(f"Paired query payload content mismatch: {path}")
    return arrays


def _build_payload(
    *,
    config: dict[str, Any],
    bundle: dict[str, Any],
    base: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    speeds = tuple(map(float, config["protocol"]["eval_speeds"]))
    factors = tuple(
        (speed, rule) for speed in speeds for rule in RULES
    )
    # Reuse one environment across the six factor resets.
    from .hidden_passage_env import make_hidden_passage_env

    env = make_hidden_passage_env(render_mode="rgb_array")
    try:
        grid = {
            factor: simulate_v2_future(
                env,
                bundle=bundle,
                speed=factor[0],
                rule=factor[1],
                config=config,
            )
            for factor in factors
        }
    finally:
        env.close()

    arrays: dict[str, np.ndarray] = {
        "query_pixels": np.asarray(
            next(iter(grid.values()))["query_pixels"], dtype=np.uint8
        ),
        "query_state": np.asarray(
            next(iter(grid.values()))["query_state"], dtype=np.float32
        ),
        "goal_state": np.asarray(
            bundle["template"]["goal_state"], dtype=np.float32
        ),
    }
    future_blocks = future_action_blocks(
        str(bundle["direction"]), config
    )
    for speed, rule in factors:
        key = factor_key(speed, rule)
        base_key = key
        history = np.asarray(
            base[f"{base_key}_history_pixels"], dtype=np.uint8
        )
        old_actions = np.asarray(
            base[f"{base_key}_action_blocks"], dtype=np.float32
        )
        if old_actions.shape != (3, 5, 2):
            raise RuntimeError(
                f"Unexpected paired action shape: {old_actions.shape}"
            )
        arrays[f"{key}_history_pixels"] = history
        arrays[f"{key}_action_blocks"] = np.concatenate(
            [old_actions[:2], future_blocks],
            axis=0,
        ).astype(np.float32)

    for horizon in HORIZONS:
        blocked = [
            grid[(speed, "blocked")]["targets"][horizon]
            for speed in speeds
        ]
        if not all(
            np.array_equal(blocked[0]["pixels"], value["pixels"])
            and np.array_equal(blocked[0]["state"], value["state"])
            for value in blocked[1:]
        ):
            raise RuntimeError("Blocked v2 future is not shared across speeds")
        arrays[f"h{horizon}_blocked_target_pixels"] = np.asarray(
            blocked[0]["pixels"], dtype=np.uint8
        )
        arrays[f"h{horizon}_blocked_target_state"] = np.asarray(
            blocked[0]["state"], dtype=np.float32
        )
        for speed in speeds:
            target = grid[(speed, "passable")]["targets"][horizon]
            target_key = passable_target_key(speed)
            arrays[f"h{horizon}_{target_key}_target_pixels"] = np.asarray(
                target["pixels"], dtype=np.uint8
            )
            arrays[f"h{horizon}_{target_key}_target_state"] = np.asarray(
                target["state"], dtype=np.float32
            )

    query_hash = array_sha256(arrays["query_pixels"])
    if query_hash != str(bundle["query_pixels_sha256"]):
        raise RuntimeError("v2 query pixels differ from paired v1 query")
    target_hashes = {
        f"h{horizon}/{target_key}": array_sha256(
            arrays[f"h{horizon}_{target_key}_target_pixels"]
        )
        for horizon in HORIZONS
        for target_key in physical_target_keys(speeds)
    }
    for horizon in HORIZONS:
        values = {
            target_hashes[f"h{horizon}/{target_key}"]
            for target_key in physical_target_keys(speeds)
        }
        if len(values) != len(speeds) + 1:
            raise RuntimeError(
                f"h{horizon} does not contain four physical futures"
            )
    audit = {
        "query_pixels_sha256": query_hash,
        "history_pixels_sha256": {
            factor_key(*factor): array_sha256(
                arrays[f"{factor_key(*factor)}_history_pixels"]
            )
            for factor in factors
        },
        "action_blocks_sha256": {
            factor_key(*factor): array_sha256(
                arrays[f"{factor_key(*factor)}_action_blocks"]
            )
            for factor in factors
        },
        "target_pixels_sha256": target_hashes,
    }
    return arrays, audit


def _manifest_projection(catalog: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": catalog["schema_version"],
        "benchmark": catalog["benchmark"],
        "status": catalog["status"],
        "protocol": catalog["protocol"],
        "summary": catalog["summary"],
        "bundles": [
            {
                key: bundle[key]
                for key in (
                    "query_id",
                    "eval_seed",
                    "evaluation_index",
                    "direction",
                    "door_position",
                    "template",
                    "payload_sha256",
                    "payload_content_sha256",
                    "query_pixels_sha256",
                    "history_pixels_sha256",
                    "action_blocks_sha256",
                    "target_pixels_sha256",
                )
            }
            for bundle in catalog["bundles"]
        ],
    }


def build_v2_validation_catalog(
    *,
    config: dict[str, Any],
    repo_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    design = require_valid_speed_door_rule_v2_design(config)
    source, source_path = _load_v1_source(config, repo_root=repo_root)
    prototype = audit_v2_query_bundles(
        config=config,
        bundles=source["bundles"],
        require_full_catalog=True,
    )
    if not prototype["passed"]:
        raise RuntimeError(f"v2 physical prototype failed: {prototype}")

    speeds = tuple(map(float, config["protocol"]["eval_speeds"]))
    factors = tuple(
        (speed, rule) for speed in speeds for rule in RULES
    )
    payload_root = Path(output_root) / "payloads"
    payload_root.mkdir(parents=True, exist_ok=True)
    bundles = []
    for source_bundle in source["bundles"]:
        base = _base_payload(source_bundle, repo_root=repo_root)
        arrays, audit = _build_payload(
            config=config,
            bundle=source_bundle,
            base=base,
        )
        payload_path = payload_root / f"{source_bundle['query_id']}.npz"
        if payload_path.exists():
            raise FileExistsError(payload_path)
        _atomic_savez(payload_path, arrays)
        with np.load(payload_path, allow_pickle=False) as payload:
            roundtrip = {
                name: np.asarray(payload[name])
                for name in payload.files
            }
        if (
            set(roundtrip) != set(arrays)
            or any(
                not np.array_equal(roundtrip[name], value)
                for name, value in arrays.items()
            )
        ):
            raise RuntimeError(f"v2 payload roundtrip failed: {payload_path}")
        bundle = {
            "query_id": str(source_bundle["query_id"]),
            "eval_seed": int(source_bundle["eval_seed"]),
            "evaluation_index": int(
                source_bundle["evaluation_index"]
            ),
            "direction": str(source_bundle["direction"]),
            "door_position": int(source_bundle["door_position"]),
            "template": dict(source_bundle["template"]),
            "factor_conditions": [
                {
                    "key": factor_key(speed, rule),
                    "speed": speed,
                    "rule": rule,
                }
                for speed, rule in factors
            ],
            "physical_target_keys": list(
                physical_target_keys(speeds)
            ),
            "payload": portable_contextworld_path(
                payload_path, repo_root=repo_root
            ),
            "payload_sha256": file_sha256(payload_path),
            "payload_content_sha256": _payload_content_sha256(arrays),
            **audit,
        }
        bundles.append(bundle)

    eval_seeds = tuple(map(int, config["evaluation"]["eval_seeds"]))
    by_seed = Counter(bundle["eval_seed"] for bundle in bundles)
    by_direction = Counter(bundle["direction"] for bundle in bundles)
    summary = {
        "unique_base_queries": len(bundles),
        "eval_seeds": list(eval_seeds),
        "unique_base_queries_per_seed": int(
            config["evaluation"]["unique_base_queries_per_seed"]
        ),
        "by_eval_seed": {
            str(seed): by_seed[seed] for seed in eval_seeds
        },
        "by_direction": dict(sorted(by_direction.items())),
        "eval_speeds": list(speeds),
        "rules": list(RULES),
        "histories_per_query": len(factors),
        "horizons": list(HORIZONS),
        "physical_target_classes_per_horizon": len(speeds) + 1,
        "prediction_sequences_per_checkpoint": len(bundles)
        * len(factors),
        "prediction_endpoints_per_checkpoint": len(bundles)
        * len(factors)
        * len(HORIZONS),
        "distinct_target_encodings_per_checkpoint": len(bundles)
        * (len(speeds) + 1)
        * len(HORIZONS),
    }
    catalog = {
        "schema_version": 2,
        "benchmark": str(config["benchmark"]),
        "status": "frozen_before_training_or_model_scoring",
        "protocol": {
            "history_tokens": 3,
            "raw_steps_per_action_block": 5,
            "future_action_blocks": 2,
            "horizons": list(HORIZONS),
            "factor_conditions": [
                {
                    "key": factor_key(speed, rule),
                    "speed": speed,
                    "rule": rule,
                }
                for speed, rule in factors
            ],
            "physical_target_keys": list(
                physical_target_keys(speeds)
            ),
            "model_visible_fields": ["pixels", "action"],
            "online_environment_calls_during_scoring": 0,
        },
        "summary": summary,
        "bundles": bundles,
    }
    catalog["content_manifest_sha256"] = canonical_sha256(
        _manifest_projection(catalog)
    )
    exclusion = {
        "schema_version": 2,
        "benchmark": str(config["benchmark"]),
        "purpose": (
            "Exclude every Eval-only door and selected query image from "
            "Speed × Door Rule v2 training data"
        ),
        "eval_only_door_positions": sorted(
            {int(bundle["door_position"]) for bundle in bundles}
            | {110}
        ),
        "eval_speeds": list(speeds),
        "query_records": [
            {
                "query_id": bundle["query_id"],
                "template_id": bundle["template"]["template_id"],
                "eval_seed": bundle["eval_seed"],
                "door_position": bundle["door_position"],
                "direction": bundle["direction"],
                "query_pixels_sha256": bundle[
                    "query_pixels_sha256"
                ],
            }
            for bundle in bundles
        ],
        "query_count": len(bundles),
        "content_manifest_sha256": catalog[
            "content_manifest_sha256"
        ],
    }
    checks = {
        "design_contract_passed": bool(design["passed"]),
        "physical_prototype_300_of_300_passed": bool(
            prototype["passed"] and prototype["query_count"] == 300
        ),
        "query_count_exact": len(bundles)
        == int(config["evaluation"]["unique_base_queries"]),
        "per_seed_counts_exact": all(
            by_seed[seed]
            == int(config["evaluation"]["unique_base_queries_per_seed"])
            for seed in eval_seeds
        ),
        "directions_balanced": (
            by_direction["left_to_right"]
            == by_direction["right_to_left"]
            == len(bundles) // 2
        ),
        "query_ids_unique": len(
            {bundle["query_id"] for bundle in bundles}
        )
        == len(bundles),
        "query_pixels_unique": len(
            {bundle["query_pixels_sha256"] for bundle in bundles}
        )
        == len(bundles),
        "source_catalog_hash_exact": file_sha256(source_path)
        == str(
            config["evaluation"]["paired_query_source"][
                "catalog_sha256"
            ]
        ),
    }
    report = {
        "schema_version": 2,
        "benchmark": str(config["benchmark"]),
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "physical_prototype": prototype,
        "source_catalog": portable_contextworld_path(
            source_path, repo_root=repo_root
        ),
        "source_catalog_sha256": file_sha256(source_path),
        "content_manifest_sha256": catalog[
            "content_manifest_sha256"
        ],
        "claim_limit": (
            "Frozen offline data and physical identifiability only; "
            "not a model composition result."
        ),
    }
    return catalog, exclusion, report


def load_v2_validation_assets(
    catalog_path: Path,
    *,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog_path = Path(catalog_path).resolve()
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    if catalog.get("status") != "frozen_before_training_or_model_scoring":
        raise ValueError("v2 Validation catalog is not frozen")
    observed = canonical_sha256(_manifest_projection(catalog))
    if observed != catalog.get("content_manifest_sha256"):
        raise RuntimeError("v2 Validation content manifest mismatch")
    factors = tuple(
        (float(row["speed"]), str(row["rule"]))
        for row in catalog["protocol"]["factor_conditions"]
    )
    target_keys = tuple(catalog["protocol"]["physical_target_keys"])
    horizons = tuple(map(int, catalog["protocol"]["horizons"]))
    assets = []
    for bundle in catalog["bundles"]:
        payload_path = resolve_contextworld_path(
            bundle["payload"], repo_root=repo_root
        )
        if file_sha256(payload_path) != bundle["payload_sha256"]:
            raise RuntimeError(f"v2 payload hash mismatch: {payload_path}")
        with np.load(payload_path, allow_pickle=False) as payload:
            arrays = {
                name: np.asarray(payload[name]).copy()
                for name in payload.files
            }
        if _payload_content_sha256(arrays) != bundle[
            "payload_content_sha256"
        ]:
            raise RuntimeError("v2 payload content mismatch")
        histories = {
            factor: np.asarray(
                arrays[f"{factor_key(*factor)}_history_pixels"],
                dtype=np.uint8,
            )
            for factor in factors
        }
        actions = {
            factor: np.asarray(
                arrays[f"{factor_key(*factor)}_action_blocks"],
                dtype=np.float32,
            )
            for factor in factors
        }
        targets = {
            horizon: {
                target_key: np.asarray(
                    arrays[
                        f"h{horizon}_{target_key}_target_pixels"
                    ],
                    dtype=np.uint8,
                )
                for target_key in target_keys
            }
            for horizon in horizons
        }
        if any(value.shape != (3, 224, 224, 3) for value in histories.values()):
            raise RuntimeError("v2 history shape mismatch")
        if any(value.shape != (4, 5, 2) for value in actions.values()):
            raise RuntimeError("v2 action shape mismatch")
        if len({array_sha256(value) for value in actions.values()}) != 1:
            raise RuntimeError("v2 future/context actions differ by factor")
        assets.append(
            {
                "query_id": str(bundle["query_id"]),
                "eval_seed": int(bundle["eval_seed"]),
                "evaluation_index": int(bundle["evaluation_index"]),
                "direction": str(bundle["direction"]),
                "door_position": int(bundle["door_position"]),
                "template_id": str(bundle["template"]["template_id"]),
                "histories": histories,
                "actions": actions,
                "targets": targets,
            }
        )
    audit = {
        "passed": len(assets)
        == int(catalog["summary"]["unique_base_queries"]),
        "catalog": str(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "content_manifest_sha256": observed,
        "unique_base_queries": len(assets),
        "factor_conditions": len(factors),
        "target_classes": len(target_keys),
        "horizons": list(horizons),
    }
    return assets, audit


__all__ = [
    "build_v2_validation_catalog",
    "factor_key",
    "file_sha256",
    "load_v2_validation_assets",
    "passable_target_key",
    "physical_target_keys",
]
