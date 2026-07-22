#!/usr/bin/env python3
"""Analyze original-ability retention for the seven door-study models.

Formal analysis requires the complete original reference plus the three paired
training seeds of both door recipes.  Every checkpoint is bound to its final
training report and to the hashes recorded by the frozen planning and rollout
evaluators.  ``--allow-partial`` is intentionally exploratory: it never emits
a formal non-inferiority decision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_model import file_sha256
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"
)
GROUPS = ("original_reference", "fixed_door_control", "multi_door_target")
PAIRED_TRAINING_SEEDS = (3072, 4096, 5120)
HORIZONS = (1, 2, 3, 5)
ROLLOUT_METRICS = ("latent_mse", "latent_rmse", "latent_cosine_distance")


@dataclass(frozen=True)
class ExpectedModel:
    group: str
    training_seed: int
    slug: str
    model_id: str
    checkpoint: Path
    report_name: str
    expected_steps: int
    training_groups: dict[str, float]


@dataclass(frozen=True)
class FrozenProtocol:
    protocol_path: Path
    protocol_sha256: str
    stable_worldmodel_commit: str
    eval_seeds: tuple[int, ...]
    evaluations_per_seed: int
    planning_parameters: dict[str, int]
    horizons: tuple[int, ...]
    bootstrap_seed: int
    bootstrap_resamples: int
    confidence_level: float
    success_margin_pp: float
    distance_margin_px: float
    require_no_stratum_collapse: bool
    normalizer: Path
    normalizer_sha256: str
    planning_catalog: Path
    planning_catalog_sha256: str
    planning_entries_by_seed: dict[int, dict[str, dict[str, Any]]]
    rollout_catalog: Path
    rollout_catalog_sha256: str
    rollout_entries_by_domain: dict[str, dict[str, dict[str, Any]]]


def _load_mapping(path: Path, *, yaml_input: bool = False) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) if yaml_input else json.loads(text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a mapping in {path}")
    return payload


def _same_path(left: Any, right: Path) -> bool:
    try:
        return Path(str(left)).expanduser().resolve() == right.resolve()
    except (OSError, TypeError, ValueError):
        return False


def _stablewm_commit(protocol: dict[str, Any]) -> str:
    prefix = "stable_worldmodel_commit_"
    values = [
        str(value).removeprefix(prefix)
        for value in protocol["training_protocol"].get("fixed_components", [])
        if str(value).startswith(prefix)
    ]
    if len(values) != 1 or len(values[0]) != 40:
        raise RuntimeError("Ability protocol must pin one StableWorldModel commit")
    return values[0]


def _expected_models(
    config: dict[str, Any], protocol: dict[str, Any]
) -> list[ExpectedModel]:
    if tuple(config.get("models", {})) != GROUPS:
        raise RuntimeError(
            "Door config must declare original, fixed-door, then multi-door groups"
        )
    original_rows = [
        row
        for row in protocol["models"]
        if list(row.get("training_groups", [])) == ["original"]
    ]
    if len(original_rows) != 1:
        raise RuntimeError("Ability protocol must declare one original-only model")
    original = original_rows[0]
    original_checkpoint = resolve_contextworld_path(
        original["checkpoint"], repo_root=ROOT
    )
    original_id = str(original["model_id"])
    sampling = protocol["training_protocol"]["group_sampling"]
    original_weights = {
        str(key): float(value) for key, value in sampling[original_id].items()
    }
    original_steps = int(
        protocol["training_protocol"]["exposure_contract"]["single_domain"][
            "optimizer_steps"
        ]
    )
    expected: list[ExpectedModel] = []
    for group, row in config["models"].items():
        seeds = tuple(int(value) for value in row["required_training_seeds"])
        if group == "original_reference":
            if seeds != (3072,):
                raise RuntimeError("Original door reference must be training seed 3072")
            expected.append(
                ExpectedModel(
                    group=group,
                    training_seed=3072,
                    slug=original_checkpoint.parent.name,
                    model_id=original_id,
                    checkpoint=original_checkpoint,
                    report_name=f"{original_checkpoint.parent.name}.json",
                    expected_steps=original_steps,
                    training_groups=original_weights,
                )
            )
            continue
        if seeds != PAIRED_TRAINING_SEEDS:
            raise RuntimeError(f"Unexpected paired training seeds for {group}: {seeds}")
        weights = {str(key): float(value) for key, value in row["training_groups"].items()}
        synthetic = sorted(set(weights) - {"original"})
        if len(synthetic) != 1:
            raise RuntimeError(f"Expected one synthetic group for {group}")
        model_id = f"M_{synthetic[0]}"
        for seed in seeds:
            slug = f"h3_{synthetic[0]}_s{seed}"
            expected.append(
                ExpectedModel(
                    group=group,
                    training_seed=seed,
                    slug=slug,
                    model_id=model_id,
                    checkpoint=artifact_path(
                        "training",
                        "runs",
                        "checkpoints",
                        slug,
                        f"weights_final_step_{int(config['training_protocol']['optimizer_steps'])}.pt",
                        repo_root=ROOT,
                    ),
                    report_name=f"{slug}.json",
                    expected_steps=int(config["training_protocol"]["optimizer_steps"]),
                    training_groups=weights,
                )
            )
    if len(expected) != 7 or len({row.slug for row in expected}) != 7:
        raise RuntimeError("Door ability matrix must contain seven unique models")
    return expected


def _unique_entries(
    rows: Iterable[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("evaluation_id"))
        if not key or key == "None" or key in output:
            raise RuntimeError(f"{label} has missing or duplicate evaluation_id")
        output[key] = row
    return output


def _load_frozen_protocol(
    config: dict[str, Any], config_path: Path
) -> tuple[dict[str, Any], FrozenProtocol]:
    if not str(config.get("status", "")).startswith("preregistered_"):
        raise RuntimeError("Door config is not preregistered")
    protocol_path = resolve_contextworld_path(
        config["ability_retention"]["protocol"], repo_root=ROOT
    )
    protocol = _load_mapping(protocol_path, yaml_input=True)
    evaluation = protocol["evaluation_protocol"]
    noninferiority = evaluation["non_inferiority"]
    seeds = tuple(int(value) for value in evaluation["eval_seeds"])
    count = int(evaluation["num_eval_per_seed"])
    if seeds != (42, 43, 44, 45, 46, 47) or count != 50:
        raise RuntimeError("Formal ability retention requires six seeds x 50 queries")
    if int(config["ability_retention"]["per_eval_per_model"]) != len(seeds) * count:
        raise RuntimeError("Door config and ability protocol disagree on Eval count")
    planning = {
        key: int(evaluation["planning"][key])
        for key in (
            "eval_budget",
            "horizon",
            "receding_horizon",
            "cem_samples",
            "cem_steps",
            "cem_topk",
        )
    }
    horizons = tuple(int(value) for value in evaluation["rollout_horizons_action_blocks"])
    if horizons != HORIZONS:
        raise RuntimeError(f"Formal rollout horizons must be {HORIZONS}")
    if int(noninferiority["paired_bootstrap_resamples"]) != 10_000:
        raise RuntimeError("Formal ability analysis requires 10,000 bootstrap draws")

    artifacts = protocol["artifacts"]
    normalizer = resolve_contextworld_path(artifacts["frozen_normalizer"], repo_root=ROOT)
    planning_catalog = resolve_contextworld_path(
        artifacts["original_eval_catalog"], repo_root=ROOT
    )
    rollout_catalog = resolve_contextworld_path(artifacts["rollout_catalog"], repo_root=ROOT)
    for path in (config_path, protocol_path, normalizer, planning_catalog, rollout_catalog):
        if not path.is_file():
            raise FileNotFoundError(path)

    planning_payload = _load_mapping(planning_catalog)
    if planning_payload.get("status") != "frozen":
        raise RuntimeError("Original-heldout planning catalog is not frozen")
    planning_protocol = planning_payload["protocol"]
    expected_planning_protocol = {**planning, "num_eval_per_seed": count}
    for key, expected_value in expected_planning_protocol.items():
        if int(planning_protocol.get(key, -1)) != expected_value:
            raise RuntimeError(f"Planning catalog protocol mismatch: {key}")
    if tuple(map(int, planning_protocol.get("eval_seeds", []))) != seeds:
        raise RuntimeError("Planning catalog Eval seeds changed")
    planning_by_seed = {}
    for seed in seeds:
        selected = [
            row
            for row in planning_payload.get("entries", [])
            if int(row.get("eval_seed", -1)) == seed
        ]
        if len(selected) != count:
            raise RuntimeError(f"Planning catalog seed {seed} is not 50-query complete")
        planning_by_seed[seed] = _unique_entries(selected, label=f"planning catalog seed {seed}")

    rollout_payload = _load_mapping(rollout_catalog)
    if rollout_payload.get("status") != "frozen":
        raise RuntimeError("Rollout catalog is not frozen")
    rollout_protocol = rollout_payload["protocol"]
    if tuple(map(int, rollout_protocol.get("eval_seeds", []))) != seeds:
        raise RuntimeError("Rollout catalog Eval seeds changed")
    if tuple(map(int, rollout_protocol.get("rollout_horizons", []))) != horizons:
        raise RuntimeError("Rollout catalog horizons changed")
    if int(rollout_protocol.get("num_eval_per_seed_per_domain", -1)) != count:
        raise RuntimeError("Rollout catalog count changed")
    rollout_by_domain: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rollout_payload.get("entries", []):
        rollout_by_domain.setdefault(str(row.get("domain")), {})
        key = str(row.get("evaluation_id"))
        if key in rollout_by_domain[str(row.get("domain"))]:
            raise RuntimeError("Rollout catalog has duplicate evaluation_id")
        rollout_by_domain[str(row.get("domain"))][key] = row
    original = rollout_by_domain.get("original_heldout", {})
    counts = Counter(int(row.get("eval_seed", -1)) for row in original.values())
    if counts != Counter({seed: count for seed in seeds}):
        raise RuntimeError("Original-heldout rollout catalog is not 50 x 6 complete")

    return protocol, FrozenProtocol(
        protocol_path=protocol_path,
        protocol_sha256=file_sha256(protocol_path),
        stable_worldmodel_commit=_stablewm_commit(protocol),
        eval_seeds=seeds,
        evaluations_per_seed=count,
        planning_parameters=planning,
        horizons=horizons,
        bootstrap_seed=int(noninferiority["paired_bootstrap_seed"]),
        bootstrap_resamples=int(noninferiority["paired_bootstrap_resamples"]),
        confidence_level=float(noninferiority["confidence_level"]),
        success_margin_pp=float(noninferiority["success_margin_percentage_points"]),
        distance_margin_px=float(noninferiority["final_distance_margin_px"]),
        require_no_stratum_collapse=bool(
            noninferiority["require_no_solvable_stratum_collapse"]
        ),
        normalizer=normalizer,
        normalizer_sha256=file_sha256(normalizer),
        planning_catalog=planning_catalog,
        planning_catalog_sha256=file_sha256(planning_catalog),
        planning_entries_by_seed=planning_by_seed,
        rollout_catalog=rollout_catalog,
        rollout_catalog_sha256=file_sha256(rollout_catalog),
        rollout_entries_by_domain=rollout_by_domain,
    )


def _audit_training_report(
    model: ExpectedModel,
    report_path: Path,
    *,
    stable_worldmodel_commit: str,
    expected_data_split_seed: int,
) -> dict[str, Any]:
    report = _load_mapping(report_path)
    if not model.checkpoint.is_file():
        raise FileNotFoundError(model.checkpoint)
    checkpoint_hash = file_sha256(model.checkpoint)
    expected_groups = model.training_groups
    observed_groups = {
        str(key): float(value)
        for key, value in report["data"]["group_weights"].items()
    }
    training_plan = report["training"]["plan"]
    checks = {
        "report_passed": report.get("passed") is True,
        "save_load_exact": report.get("save_load_exact") is True,
        "training_complete": report["training"].get("training_complete") is True,
        "model_id": str(report.get("model_id")) == model.model_id,
        "run_name": str(report.get("run_name")) == model.slug,
        "data_seed": int(report["data"]["seed"])
        == expected_data_split_seed,
        "plan_data_split_seed": int(training_plan["data_split_seed"])
        == expected_data_split_seed,
        "training_seed": int(training_plan["training_seed"])
        == model.training_seed,
        "training_groups": observed_groups == expected_groups,
        "stable_worldmodel_commit": str(report["stable_worldmodel"]["commit"])
        == stable_worldmodel_commit,
        "optimizer_steps": int(report["training"]["global_step"])
        == model.expected_steps
        == int(report["training"]["expected_optimizer_steps"])
        == int(report["training"]["plan"]["optimizer_steps_total"]),
        "checkpoint_path": _same_path(report["artifacts"]["pretrained"], model.checkpoint),
        "checkpoint_hash": str(report["artifacts"]["pretrained_sha256"])
        == checkpoint_hash,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Training-report binding failed for {model.slug}: {checks}")
    return {
        "path": str(report_path),
        "sha256": file_sha256(report_path),
        "checkpoint": str(model.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checks": checks,
    }


def _common_result_audit(
    payload: dict[str, Any],
    model: ExpectedModel,
    binding: dict[str, Any],
    frozen: FrozenProtocol,
    *,
    catalog: Path,
    catalog_hash: str,
) -> None:
    checks = {
        "status": payload.get("status") == "passed",
        "checkpoint_path": _same_path(payload.get("checkpoint", {}).get("path"), model.checkpoint),
        "checkpoint_hash": payload.get("checkpoint", {}).get("sha256")
        == binding["checkpoint_sha256"],
        "normalizer_path": _same_path(
            payload.get("normalizer", {}).get("path"), frozen.normalizer
        ),
        "normalizer_hash": payload.get("normalizer", {}).get("sha256")
        == frozen.normalizer_sha256,
        "catalog_path": _same_path(payload.get("catalog", {}).get("path"), catalog),
        "catalog_hash": payload.get("catalog", {}).get("sha256") == catalog_hash,
        "stable_worldmodel_commit": payload.get("stable_worldmodel", {}).get("commit")
        == frozen.stable_worldmodel_commit,
        "frozen_weights": bool(payload.get("frozen_weight_audit", {}).get("passed")),
        "frozen_weight_hash": payload.get("frozen_weight_audit", {}).get(
            "state_dict_sha256_before"
        )
        == payload.get("frozen_weight_audit", {}).get("state_dict_sha256_after"),
        "history_size": int(payload.get("protocol", {}).get("history_size", -1)) == 3,
        "action_block": int(payload.get("protocol", {}).get("action_block", -1)) == 5,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen result binding failed for {model.slug}: {checks}")


def _catalog_fields_match(record: dict[str, Any], catalog: dict[str, Any]) -> bool:
    fields = (
        "evaluation_id",
        "evaluation_index",
        "eval_seed",
        "source_kind",
        "source_path",
        "episode",
        "start_step",
    )
    optional = ("goal_offset", "cem_group_seed", "stratum", "domain")
    return all(record.get(field) == catalog.get(field) for field in fields) and all(
        field not in catalog or record.get(field) == catalog.get(field)
        for field in optional
    )


def _load_planning_model(
    root: Path,
    model: ExpectedModel,
    binding: dict[str, Any],
    frozen: FrozenProtocol,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    combined: dict[str, dict[str, Any]] = {}
    provenance = []
    for seed in frozen.eval_seeds:
        path = root / model.slug / "planning_original_heldout" / f"seed{seed}.json"
        payload = _load_mapping(path)
        _common_result_audit(
            payload,
            model,
            binding,
            frozen,
            catalog=frozen.planning_catalog,
            catalog_hash=frozen.planning_catalog_sha256,
        )
        protocol = payload["protocol"]
        expected_protocol = {
            "eval_seed": seed,
            "evaluations": frozen.evaluations_per_seed,
            **frozen.planning_parameters,
        }
        if any(int(protocol.get(key, -1)) != value for key, value in expected_protocol.items()):
            raise RuntimeError(f"Planning protocol changed in {path}")
        records = _unique_entries(payload.get("raw_records", []), label=str(path))
        catalog = frozen.planning_entries_by_seed[seed]
        if set(records) != set(catalog) or len(records) != frozen.evaluations_per_seed:
            raise RuntimeError(f"Planning seed {seed} is not exactly 50-query complete")
        for key, record in records.items():
            if not _catalog_fields_match(record, catalog[key]):
                raise RuntimeError(f"Planning query/CEM binding changed for {key}")
            if int(record.get("eval_seed", -1)) != seed:
                raise RuntimeError(f"Planning record is in the wrong seed file: {key}")
            if not math.isfinite(float(record["final_distance"])):
                raise RuntimeError(f"Non-finite planning distance: {key}")
        if int(payload.get("aggregate", {}).get("evaluations", -1)) != len(records):
            raise RuntimeError(f"Planning aggregate count changed in {path}")
        overlap = set(combined) & set(records)
        if overlap:
            raise RuntimeError(f"Planning IDs repeat across Eval seeds: {sorted(overlap)[:1]}")
        combined.update(records)
        provenance.append({"path": str(path), "sha256": file_sha256(path)})
    if len(combined) != len(frozen.eval_seeds) * frozen.evaluations_per_seed:
        raise RuntimeError(f"Planning result is not 50 x 6 complete for {model.slug}")
    return combined, provenance


def _load_rollout_model(
    root: Path,
    model: ExpectedModel,
    binding: dict[str, Any],
    frozen: FrozenProtocol,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str]]:
    path = root / model.slug / "rollout_error.json"
    payload = _load_mapping(path)
    _common_result_audit(
        payload,
        model,
        binding,
        frozen,
        catalog=frozen.rollout_catalog,
        catalog_hash=frozen.rollout_catalog_sha256,
    )
    if tuple(map(int, payload["protocol"].get("horizons_action_blocks", []))) != frozen.horizons:
        raise RuntimeError(f"Rollout horizons changed for {model.slug}")
    records = _unique_entries(payload.get("raw_records", []), label=str(path))
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for domain, expected in frozen.rollout_entries_by_domain.items():
        selected = {key: records[key] for key in expected if key in records}
        if not selected:
            continue
        if set(selected) != set(expected):
            raise RuntimeError(f"Rollout domain {domain} is only partially present")
        for key, record in selected.items():
            if not _catalog_fields_match(record, expected[key]):
                raise RuntimeError(f"Rollout query binding changed for {key}")
            if set(record.get("horizons", {})) != {str(value) for value in frozen.horizons}:
                raise RuntimeError(f"Rollout horizon cells are incomplete for {key}")
            for horizon in frozen.horizons:
                metrics = record["horizons"][str(horizon)]
                for metric in ROLLOUT_METRICS:
                    value = float(metrics[metric])
                    if not math.isfinite(value):
                        raise RuntimeError(f"Non-finite {metric} for {key}/h{horizon}")
        output[domain] = selected
    known_ids = set().union(*(set(rows) for rows in frozen.rollout_entries_by_domain.values()))
    if set(records) - known_ids:
        raise RuntimeError("Rollout result contains queries outside the frozen catalog")
    if "original_heldout" not in output:
        raise RuntimeError(f"Original-heldout rollout is missing for {model.slug}")
    original = output["original_heldout"]
    counts = Counter(int(row["eval_seed"]) for row in original.values())
    if counts != Counter({seed: frozen.evaluations_per_seed for seed in frozen.eval_seeds}):
        raise RuntimeError(f"Original-heldout rollout is not 50 x 6 for {model.slug}")
    return output, {"path": str(path), "sha256": file_sha256(path)}


def _assert_same_planning_queries(
    reference: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]
) -> list[str]:
    if set(reference) != set(candidate):
        raise RuntimeError("Planning evaluation IDs differ between paired models")
    fields = (
        "eval_seed",
        "evaluation_index",
        "source_kind",
        "source_path",
        "episode",
        "start_step",
        "goal_offset",
        "cem_group_seed",
        "stratum",
        "room_relation",
        "initial_state",
        "goal_state",
    )
    for key in sorted(reference):
        changed = [field for field in fields if reference[key].get(field) != candidate[key].get(field)]
        if changed:
            raise RuntimeError(f"Paired planning query changed for {key}: {changed}")
    return sorted(reference)


def _paired_bootstrap(
    differences: np.ndarray,
    *,
    seed: int,
    resamples: int,
    confidence: float,
    scale: float = 1.0,
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Paired bootstrap requires a finite non-empty vector")
    rng = np.random.default_rng(int(seed))
    alpha = (1.0 - float(confidence)) / 2.0
    chunk = max(1, min(2_000, int(resamples)))
    means = np.empty(int(resamples), dtype=np.float64)
    for start in range(0, int(resamples), chunk):
        stop = min(start + chunk, int(resamples))
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "point": float(values.mean() * scale),
        "ci_lower": float(np.quantile(means, alpha) * scale),
        "ci_upper": float(np.quantile(means, 1.0 - alpha) * scale),
        "paired_queries": len(values),
        "bootstrap_resamples": int(resamples),
        "confidence_level": float(confidence),
    }


def _assert_unique_checkpoint_hashes(
    bindings: dict[str, dict[str, Any]], *, expected_count: int | None = None
) -> list[str]:
    hashes = [str(row["checkpoint_sha256"]) for row in bindings.values()]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Two model labels are bound to the same checkpoint hash")
    if expected_count is not None and len(hashes) != int(expected_count):
        raise RuntimeError(
            f"Expected {expected_count} checkpoint bindings, observed {len(hashes)}"
        )
    return hashes


def _is_formal_analysis(
    *, complete_matrix: bool, allow_partial: bool, comparison_count: int
) -> bool:
    return bool(complete_matrix and not allow_partial and comparison_count == 6)


def _planning_summary(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(records.values())
    by_seed = defaultdict(list)
    by_stratum = defaultdict(list)
    for row in rows:
        by_seed[int(row["eval_seed"])].append(row)
        by_stratum[f"{row['stratum']}|{row['room_relation']}"].append(row)

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(bool(row["success"]) for row in selected)
        return {
            "evaluations": len(selected),
            "successes": int(successes),
            "success_rate_percentage": float(100.0 * successes / len(selected)),
            "mean_final_distance_px": float(np.mean([float(row["final_distance"]) for row in selected])),
        }

    return {
        **summarize(rows),
        "by_eval_seed": {str(seed): summarize(values) for seed, values in sorted(by_seed.items())},
        "by_stratum": {key: summarize(values) for key, values in sorted(by_stratum.items())},
    }


def _planning_comparison(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    frozen: FrozenProtocol,
    bootstrap_seed: int,
) -> dict[str, Any]:
    keys = _assert_same_planning_queries(reference, candidate)
    success = _paired_bootstrap(
        np.asarray([
            float(bool(candidate[key]["success"])) - float(bool(reference[key]["success"]))
            for key in keys
        ]),
        seed=bootstrap_seed,
        resamples=frozen.bootstrap_resamples,
        confidence=frozen.confidence_level,
        scale=100.0,
    )
    distance = _paired_bootstrap(
        np.asarray([
            float(candidate[key]["final_distance"]) - float(reference[key]["final_distance"])
            for key in keys
        ]),
        seed=bootstrap_seed ^ 0xD157A,
        resamples=frozen.bootstrap_resamples,
        confidence=frozen.confidence_level,
    )
    strata: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        row = reference[key]
        strata[f"{row['stratum']}|{row['room_relation']}"].append(key)
    collapsed = []
    strata_report = {}
    for name, selected in sorted(strata.items()):
        reference_successes = sum(bool(reference[key]["success"]) for key in selected)
        candidate_successes = sum(bool(candidate[key]["success"]) for key in selected)
        collapse = reference_successes > 0 and candidate_successes == 0
        if collapse:
            collapsed.append(name)
        strata_report[name] = {
            "evaluations": len(selected),
            "reference_successes": int(reference_successes),
            "candidate_successes": int(candidate_successes),
            "solvable_stratum_collapsed": bool(collapse),
        }
    gates = {
        "success_rate_non_inferior": success["ci_lower"] >= frozen.success_margin_pp,
        "final_distance_non_inferior": distance["ci_upper"] <= frozen.distance_margin_px,
        "no_solvable_stratum_collapse": (
            not collapsed if frozen.require_no_stratum_collapse else True
        ),
    }
    return {
        "candidate_minus_original_success_rate_percentage_points": success,
        "candidate_minus_original_final_distance_px": distance,
        "margins": {
            "success_rate_percentage_points": frozen.success_margin_pp,
            "final_distance_px": frozen.distance_margin_px,
            "gate_uses_paired_bootstrap_confidence_bounds": True,
        },
        "strata": strata_report,
        "collapsed_solvable_strata": collapsed,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _rollout_summary(
    domains: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    output = {}
    for domain, records in sorted(domains.items()):
        output[domain] = {}
        for horizon in HORIZONS:
            rows = [row["horizons"][str(horizon)] for row in records.values()]
            output[domain][f"h{horizon}"] = {
                "evaluations": len(rows),
                **{
                    f"mean_{metric}": float(np.mean([float(row[metric]) for row in rows]))
                    for metric in ROLLOUT_METRICS
                },
            }
    return output


def _rollout_comparison(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    frozen: FrozenProtocol,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise RuntimeError("Original-heldout rollout IDs differ between models")
    keys = sorted(reference)
    fields = ("eval_seed", "evaluation_index", "source_kind", "source_path", "episode", "start_step", "domain")
    for key in keys:
        if any(reference[key].get(field) != candidate[key].get(field) for field in fields):
            raise RuntimeError(f"Paired rollout query changed for {key}")
    horizons = {}
    for horizon_index, horizon in enumerate(frozen.horizons):
        metrics = {}
        for metric_index, metric in enumerate(ROLLOUT_METRICS):
            differences = np.asarray([
                float(candidate[key]["horizons"][str(horizon)][metric])
                - float(reference[key]["horizons"][str(horizon)][metric])
                for key in keys
            ])
            metrics[f"candidate_minus_original_{metric}"] = _paired_bootstrap(
                differences,
                seed=bootstrap_seed + 101 * horizon_index + 10_007 * metric_index,
                resamples=frozen.bootstrap_resamples,
                confidence=frozen.confidence_level,
            )
        horizons[f"h{horizon}"] = {"evaluations": len(keys), **metrics}
    return {
        "domain": "original_heldout",
        "horizons": horizons,
        "formal_noninferiority_gate": False,
        "interpretation": (
            "descriptive true-future latent error only; native latent scales "
            "are checkpoint-specific and are not used for cross-model ranking"
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    config = _load_mapping(config_path, yaml_input=True)
    protocol, frozen = _load_frozen_protocol(config, config_path)
    models = _expected_models(config, protocol)
    ability_root = (
        args.artifact_root.expanduser().resolve()
        if args.artifact_root is not None
        else artifact_path(
            "evaluation",
            "history3",
            str(config["benchmark"]).removeprefix("tworoom_"),
            "ability_retention",
            repo_root=ROOT,
        )
    )
    report_root = resolve_contextworld_path(args.training_report_root, repo_root=ROOT)

    bindings: dict[str, dict[str, Any]] = {}
    missing = []
    for model in models:
        report_path = report_root / model.report_name
        required = [report_path, model.checkpoint]
        required.extend(
            ability_root / model.slug / "planning_original_heldout" / f"seed{seed}.json"
            for seed in frozen.eval_seeds
        )
        required.append(ability_root / model.slug / "rollout_error.json")
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            missing.extend(absent)
            continue
        bindings[model.slug] = _audit_training_report(
            model,
            report_path,
            stable_worldmodel_commit=frozen.stable_worldmodel_commit,
            expected_data_split_seed=int(
                protocol["data"]["original_split"]["seed"]
            ),
        )
    complete = len(bindings) == len(models) and not missing
    if not args.allow_partial and not complete:
        raise RuntimeError("Formal analysis requires the complete seven-model matrix; missing:\n" + "\n".join(sorted(set(missing))))
    checkpoint_hashes = _assert_unique_checkpoint_hashes(bindings)

    planning = {}
    planning_files = {}
    rollout = {}
    rollout_files = {}
    model_by_slug = {row.slug: row for row in models}
    for slug, binding in bindings.items():
        model = model_by_slug[slug]
        planning[slug], planning_files[slug] = _load_planning_model(
            ability_root, model, binding, frozen
        )
        rollout[slug], rollout_files[slug] = _load_rollout_model(
            ability_root, model, binding, frozen
        )

    original = next(row for row in models if row.group == "original_reference")
    comparisons = {}
    if original.slug in planning and original.slug in rollout:
        for index, candidate in enumerate(models):
            if candidate.group == "original_reference" or candidate.slug not in planning:
                continue
            comparison = {
                "group": candidate.group,
                "training_seed": candidate.training_seed,
                "candidate": candidate.slug,
                "original_reference": original.slug,
                "planning": _planning_comparison(
                    planning[original.slug],
                    planning[candidate.slug],
                    frozen=frozen,
                    bootstrap_seed=frozen.bootstrap_seed + 100_000 * index,
                ),
                "rollout_true_future_latent_error": _rollout_comparison(
                    rollout[original.slug]["original_heldout"],
                    rollout[candidate.slug]["original_heldout"],
                    frozen=frozen,
                    bootstrap_seed=frozen.bootstrap_seed + 100_000 * index + 50_000,
                ),
            }
            comparisons[candidate.slug] = comparison

    formal_analysis = _is_formal_analysis(
        complete_matrix=complete,
        allow_partial=args.allow_partial,
        comparison_count=len(comparisons),
    )
    group_decisions = {}
    for group in ("fixed_door_control", "multi_door_target"):
        candidates = [row for row in models if row.group == group]
        available = [comparisons[row.slug] for row in candidates if row.slug in comparisons]
        group_decisions[group] = {
            "training_seeds": [row.training_seed for row in candidates],
            "all_three_seeds_non_inferior": (
                all(row["planning"]["passed"] for row in available)
                if formal_analysis and len(available) == 3
                else None
            ),
            "formal": formal_analysis,
        }

    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "analysis": "original_ability_retention",
        "status": "passed" if formal_analysis else "partial_exploratory_only",
        "formal_analysis": formal_analysis,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "ability_retention_protocol": {
            "path": str(frozen.protocol_path),
            "sha256": frozen.protocol_sha256,
        },
        "protocol": {
            "formal_domain": "original_heldout",
            "eval_seeds": list(frozen.eval_seeds),
            "evaluations_per_seed_per_model": frozen.evaluations_per_seed,
            "planning_evaluations_per_model": len(frozen.eval_seeds) * frozen.evaluations_per_seed,
            "rollout_horizons_action_blocks": list(frozen.horizons),
            "paired_bootstrap": {
                "unit": "query",
                "resamples": frozen.bootstrap_resamples,
                "seed": frozen.bootstrap_seed,
                "confidence_level": frozen.confidence_level,
            },
            "noninferiority": {
                "success_margin_percentage_points": frozen.success_margin_pp,
                "final_distance_margin_px": frozen.distance_margin_px,
                "uses_confidence_bound": True,
                "require_no_solvable_stratum_collapse": frozen.require_no_stratum_collapse,
            },
            "speed5_matched_rollout_if_present": "descriptive_only",
        },
        "matrix_audit": {
            "expected_models": [row.slug for row in models],
            "complete_models": sorted(bindings),
            "missing_files": sorted(set(missing)),
            "seven_unique_checkpoint_hashes": complete and len(set(checkpoint_hashes)) == 7,
            "complete_formal_matrix": complete,
        },
        "training_report_bindings": bindings,
        "models": {
            slug: {
                "group": model_by_slug[slug].group,
                "training_seed": model_by_slug[slug].training_seed,
                "checkpoint_sha256": bindings[slug]["checkpoint_sha256"],
                "planning": _planning_summary(planning[slug]),
                "rollout_true_future_latent_error": _rollout_summary(rollout[slug]),
                "planning_files": planning_files[slug],
                "rollout_file": rollout_files[slug],
            }
            for slug in sorted(bindings)
        },
        "paired_candidate_vs_original": comparisons,
        "formal_group_decisions": group_decisions,
        "interpretation_limits": {
            "partial_results_are_formal": False,
            "planning_noninferiority_domain": "original_heldout_only",
            "rollout_error_is_a_noninferiority_gate": False,
            "native_latent_error_cross_checkpoint_ranking": False,
        },
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else ability_root / "original_ability_retention_summary.json"
    )
    write_json(output, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--training-report-root",
        type=Path,
        default=Path("artifacts/training/reports"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["formal_group_decisions"], indent=2, sort_keys=True))
