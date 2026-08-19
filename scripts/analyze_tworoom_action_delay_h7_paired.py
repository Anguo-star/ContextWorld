#!/usr/bin/env python3
"""Aggregate the three-seed paired History-7 Action Delay experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3072, 4096, 5120)
FAMILIES = ("pldm", "lewm")
HORIZONS = ("1", "2", "3")
DELAYS = tuple(range(11))
TRAINING_DELAYS = (0, 4, 8)


def _artifact_root() -> Path:
    configured = os.environ.get("CONTEXTWORLD_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (ROOT.parents[1] / "data/world_model/context_world").resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: Iterable[float]) -> dict[str, float]:
    rows = [float(value) for value in values]
    _require(rows, "Cannot summarize an empty metric")
    return {
        "mean": float(statistics.fmean(rows)),
        "sample_std": float(statistics.stdev(rows))
        if len(rows) > 1
        else 0.0,
        "minimum": float(min(rows)),
        "maximum": float(max(rows)),
    }


def _slug(family: str, seed: int) -> str:
    return f"h7_action_delay_paired_{family}_formal_s{seed}"


def _domain_summary(payload: dict[str, Any]) -> dict[str, Any]:
    aggregate = payload["aggregate_source_h1"]
    tracks = payload["tracks"]
    _require(
        {
            int(row["source_delay"]) for row in tracks.values()
        }
        == set(TRAINING_DELAYS),
        "Domain result must contain delays 0, 4, and 8",
    )
    ordered_tracks = sorted(
        tracks.values(),
        key=lambda value: int(value["source_delay"]),
    )
    alignments = [row["latent_alignment_h1"] for row in ordered_tracks]
    return {
        "source_h1_units": int(aggregate["source_h1_units"]),
        "exact_target_selection_rate": float(
            aggregate["exact_target_selection_rate"]
        ),
        "exact_history_selection_rate": float(
            aggregate["exact_history_selection_rate"]
        ),
        "matching_history_strict_win_rate": float(
            aggregate["matching_history_strict_win_rate"]
        ),
        "selected_target_counts": aggregate["selected_target_counts"],
        "prediction_to_target_pair_magnitude_ratio": float(
            statistics.fmean(
                row["prediction_to_target_pair_magnitude_ratio"]
                for row in alignments
            )
        ),
        "minimum_track_magnitude_ratio": float(
            min(
                row["prediction_to_target_pair_magnitude_ratio"]
                for row in alignments
            )
        ),
        "pair_direction_cosine_mean": float(
            statistics.fmean(
                row["pair_direction_cosine_mean"] for row in alignments
            )
        ),
        "minimum_track_direction_cosine": float(
            min(row["pair_direction_cosine_mean"] for row in alignments)
        ),
        "pair_direction_positive_fraction": float(
            statistics.fmean(
                row["pair_direction_positive_fraction"]
                for row in alignments
            )
        ),
        "by_delay": {
            str(int(row["source_delay"])): {
                "exact_target_selection_rate": float(
                    row["exact_target_selection_rate"]
                ),
                "exact_history_selection_rate": float(
                    row["exact_history_selection_rate"]
                ),
                "matching_history_strict_win_rate": float(
                    row["matching_history_strict_win_rate"]
                ),
                "prediction_to_target_pair_magnitude_ratio": float(
                    row["latent_alignment_h1"][
                        "prediction_to_target_pair_magnitude_ratio"
                    ]
                ),
                "pair_direction_cosine_mean": float(
                    row["latent_alignment_h1"][
                        "pair_direction_cosine_mean"
                    ]
                ),
                "pair_direction_positive_fraction": float(
                    row["latent_alignment_h1"][
                        "pair_direction_positive_fraction"
                    ]
                ),
            }
            for row in ordered_tracks
        },
    }


def _validation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    horizons = {
        horizon: dict(summary["by_horizon"][horizon]["overall"])
        for horizon in HORIZONS
    }
    return {
        "query_count": int(payload["score_audit"]["queries"]),
        "horizons": horizons,
        "trajectory": dict(summary["trajectory"]["overall"]),
        "by_delay_h1": {
            str(delay): dict(
                summary["by_horizon"]["1"]["by_target_delay"][str(delay)]
            )
            for delay in DELAYS
        },
    }


def _domain_passed(row: dict[str, Any]) -> bool:
    return (
        row["exact_target_selection_rate"] >= 0.75
        and row["exact_history_selection_rate"] >= 0.75
        and row["minimum_track_magnitude_ratio"] >= 0.50
        and row["minimum_track_direction_cosine"] >= 0.50
    )


def _every_training_delay_target_passed(row: dict[str, Any]) -> bool:
    return all(
        row["by_delay"][str(delay)][
            "exact_target_selection_rate"
        ]
        >= 0.75
        for delay in TRAINING_DELAYS
    )


def _validation_passed(row: dict[str, Any]) -> bool:
    return all(
        row["horizons"][horizon]["exact_target_selection_rate"] >= 0.60
        and row["horizons"][horizon]["exact_history_selection_rate"] >= 0.60
        and row["horizons"][horizon][
            "matching_history_strict_win_rate"
        ]
        >= 0.50
        for horizon in HORIZONS
    )


def _aggregate_family(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    domain_metrics = (
        "exact_target_selection_rate",
        "exact_history_selection_rate",
        "matching_history_strict_win_rate",
        "prediction_to_target_pair_magnitude_ratio",
        "pair_direction_cosine_mean",
        "pair_direction_positive_fraction",
    )
    validation_metrics = (
        "exact_target_selection_rate",
        "exact_history_selection_rate",
        "matching_history_strict_win_rate",
        "physical_target_group_selection_rate",
        "physical_history_group_selection_rate",
        "mean_history_loss_ratio",
        "mean_history_margin",
    )
    by_horizon = {}
    for horizon in HORIZONS:
        by_horizon[horizon] = {
            metric: _stats(
                row["validation"]["horizons"][horizon][metric]
                for row in rows
            )
            for metric in validation_metrics
        }
    by_delay_h1 = {}
    for delay in DELAYS:
        by_delay_h1[str(delay)] = {
            metric: _stats(
                row["validation"]["by_delay_h1"][str(delay)][metric]
                for row in rows
            )
            for metric in (
                "exact_target_selection_rate",
                "exact_history_selection_rate",
                "physical_target_group_selection_rate",
                "physical_history_group_selection_rate",
            )
        }
    training_domain_by_delay = {
        str(delay): {
            metric: _stats(
                row["training_domain"]["by_delay"][str(delay)][metric]
                for row in rows
            )
            for metric in domain_metrics
        }
        for delay in TRAINING_DELAYS
    }
    heldout_domain_by_delay = {
        str(delay): {
            metric: _stats(
                row["heldout_same_distribution"]["by_delay"][
                    str(delay)
                ][metric]
                for row in rows
            )
            for metric in domain_metrics
        }
        for delay in TRAINING_DELAYS
    }
    return {
        "models": len(rows),
        "training_seeds": [int(row["training_seed"]) for row in rows],
        "training_domain_h1": {
            metric: _stats(row["training_domain"][metric] for row in rows)
            for metric in domain_metrics
        },
        "training_domain_h1_by_delay": training_domain_by_delay,
        "heldout_same_distribution_h1": {
            metric: _stats(
                row["heldout_same_distribution"][metric] for row in rows
            )
            for metric in domain_metrics
        },
        "heldout_same_distribution_h1_by_delay": (
            heldout_domain_by_delay
        ),
        "validation_by_horizon": by_horizon,
        "validation_by_delay_h1": by_delay_h1,
        "training_domain_gate_passed_seeds": sum(
            bool(row["gates"]["training_domain_h1_passed"])
            for row in rows
        ),
        "heldout_same_distribution_gate_passed_seeds": sum(
            bool(
                row["gates"][
                    "heldout_same_distribution_h1_passed"
                ]
            )
            for row in rows
        ),
        "training_domain_every_delay_target_75_passed_seeds": sum(
            _every_training_delay_target_passed(
                row["training_domain"]
            )
            for row in rows
        ),
        "heldout_every_delay_target_75_passed_seeds": sum(
            _every_training_delay_target_passed(
                row["heldout_same_distribution"]
            )
            for row in rows
        ),
        "formal_validation_gate_passed_seeds": sum(
            bool(row["gates"]["formal_validation_passed"]) for row in rows
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = _artifact_root()
    result_root = (
        artifact_root
        / "evaluation/history7/action_delay_paired_repair_v1/model_results"
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else artifact_root
        / "evaluation/history7/action_delay_paired_repair_v1/"
        "comparison_summary.json"
    )

    model_rows = []
    artifacts = {}
    for seed in SEEDS:
        for family in FAMILIES:
            slug = _slug(family, seed)
            checkpoint_path = (
                artifact_root
                / "training/runs/checkpoints"
                / slug
                / "weights_final_step_1024.pt"
            )
            training_report_path = (
                artifact_root / "training/reports" / f"{slug}.json"
            )
            domain_path = result_root / f"{slug}_training_domain.json"
            heldout_path = (
                result_root
                / f"{slug}_heldout_same_distribution.json"
            )
            validation_path = result_root / f"{slug}_validation.json"
            _require(
                checkpoint_path.is_file(),
                f"Missing checkpoint: {checkpoint_path}",
            )
            _require(
                training_report_path.is_file(),
                f"Missing training report: {training_report_path}",
            )
            _require(domain_path.is_file(), f"Missing result: {domain_path}")
            _require(
                heldout_path.is_file(),
                f"Missing result: {heldout_path}",
            )
            _require(
                validation_path.is_file(),
                f"Missing result: {validation_path}",
            )
            domain_payload = _load_json(domain_path)
            heldout_payload = _load_json(heldout_path)
            validation_payload = _load_json(validation_path)
            _require(
                domain_payload.get("label") == slug
                and heldout_payload.get("label") == slug
                and validation_payload.get("label") == slug,
                f"Result label mismatch for {slug}",
            )
            _require(
                domain_payload.get("model_family") == family
                and heldout_payload.get("model_family") == family
                and validation_payload.get("model_family") == family,
                f"Result family mismatch for {slug}",
            )
            domain = _domain_summary(domain_payload)
            heldout = _domain_summary(heldout_payload)
            validation = _validation_summary(validation_payload)
            model_rows.append(
                {
                    "label": slug,
                    "model_family": family,
                    "training_seed": seed,
                    "training_domain": domain,
                    "heldout_same_distribution": heldout,
                    "validation": validation,
                    "gates": {
                        "training_domain_h1_passed": _domain_passed(
                            domain
                        ),
                        "heldout_same_distribution_h1_passed": (
                            _domain_passed(heldout)
                        ),
                        "training_domain_every_delay_target_75": (
                            _every_training_delay_target_passed(
                                domain
                            )
                        ),
                        "heldout_every_delay_target_75": (
                            _every_training_delay_target_passed(
                                heldout
                            )
                        ),
                        "formal_validation_passed": _validation_passed(
                            validation
                        ),
                    },
                }
            )
            artifacts[slug] = {
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": _sha256(checkpoint_path),
                },
                "training_report": {
                    "path": str(training_report_path),
                    "sha256": _sha256(training_report_path),
                },
                "training_domain": {
                    "path": str(domain_path),
                    "sha256": _sha256(domain_path),
                },
                "heldout_same_distribution": {
                    "path": str(heldout_path),
                    "sha256": _sha256(heldout_path),
                },
                "validation": {
                    "path": str(validation_path),
                    "sha256": _sha256(validation_path),
                },
            }

    by_family = {
        family: _aggregate_family(
            [
                row
                for row in model_rows
                if row["model_family"] == family
            ]
        )
        for family in FAMILIES
    }
    pldm_domain_passes = int(
        by_family["pldm"]["training_domain_gate_passed_seeds"]
    )
    pldm_validation_passes = int(
        by_family["pldm"]["formal_validation_gate_passed_seeds"]
    )
    pldm_heldout_passes = int(
        by_family["pldm"][
            "heldout_same_distribution_gate_passed_seeds"
        ]
    )
    pldm_training_every_delay_passes = int(
        by_family["pldm"][
            "training_domain_every_delay_target_75_passed_seeds"
        ]
    )
    pldm_heldout_every_delay_passes = int(
        by_family["pldm"][
            "heldout_every_delay_target_75_passed_seeds"
        ]
    )
    pldm_domain_stable = pldm_domain_passes == len(SEEDS)
    pldm_heldout_stable = pldm_heldout_passes == len(SEEDS)
    pldm_validation_passed = pldm_validation_passes == len(SEEDS)
    if pldm_domain_stable and pldm_heldout_stable:
        learning_conclusion = (
            "PLDM 在三档训练来源和同分布未见 query 上均稳定通过了"
            "预注册的总体能力门；但最慢档延迟 8 仍未单独达到 "
            "75% 真实下一状态选择率。"
        )
    elif pldm_domain_stable:
        learning_conclusion = (
            "PLDM 在三档训练来源上学会了历史—下一状态绑定，"
            "但没有在同分布未见 query 上稳定通过。"
        )
    else:
        learning_conclusion = (
            "PLDM 没有在三个种子上稳定通过三档训练域能力门。"
        )
    if pldm_validation_passed:
        validation_conclusion = (
            "PLDM 同时通过了冻结 11 档、多步 Validation。"
        )
    data_build_report = (
        artifact_root
        / "synthesis/action_delay_h7_paired_v1/build_report.json"
    )
    data_catalog = (
        artifact_root
        / "synthesis/action_delay_h7_paired_v1/catalogs/"
        "tworoom_action_delay_h7_paired_v1.json"
    )
    data_manifest = (
        artifact_root
        / "synthesis/action_delay_h7_paired_v1/manifests/"
        "tworoom_action_delay_h7_paired_v1.jsonl"
    )
    validation_catalog = (
        artifact_root
        / "evaluation/history7/action_delay_validation_v1/catalog.json"
    )
    identity_paths = {
        "data_build_report": data_build_report,
        "data_catalog": data_catalog,
        "data_manifest": data_manifest,
        "frozen_validation_catalog": validation_catalog,
        "data_protocol": (
            ROOT
            / "configs/benchmark/"
            "tworoom_action_delay_h7_paired_training_data_v1.yaml"
        ),
        "pldm_training_protocol": (
            ROOT
            / "configs/benchmark/"
            "tworoom_action_delay_h7_paired_pldm_v1.yaml"
        ),
        "lewm_training_protocol": (
            ROOT
            / "configs/benchmark/"
            "tworoom_action_delay_h7_paired_lewm_v1.yaml"
        ),
    }
    for name, path in identity_paths.items():
        _require(path.is_file(), f"Missing identity artifact {name}: {path}")
    else:
        validation_conclusion = (
            "PLDM 在冻结 11 档 Validation 上仍未通过完整能力门，"
            "失败位置由逐 horizon 指标给出。"
        )
    payload = {
        "schema_version": 1,
        "benchmark": (
            "tworoom_action_delay_history7_paired_repair_comparison_v1"
        ),
        "status": "completed",
        "question": (
            "同一 query 成套提供延迟 0、4、8，并提高最后一个转移的 loss "
            "权重后，LeWM 或 PLDM 能否根据七帧历史预测动作延迟？"
        ),
        "chance_references": {
            "training_domain_three_way": 1.0 / 3.0,
            "validation_eleven_way": 1.0 / 11.0,
        },
        "identity": {
            "stable_worldmodel_commit": (
                "ad2bc44579f2b5b65c004fd2c9d8edc8ebaa43ce"
            ),
            "files": {
                name: {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for name, path in identity_paths.items()
            },
        },
        "protocol": {
            "training_seeds": list(SEEDS),
            "training_delays": list(TRAINING_DELAYS),
            "validation_delays": list(DELAYS),
            "independent_queries_per_delay": 300,
            "validation_horizons": [1, 2, 3],
            "last_transition_normalized_loss_share": 7.0 / 13.0,
            "training_domain_gate": {
                "exact_target_selection_rate_minimum": 0.75,
                "exact_history_selection_rate_minimum": 0.75,
                "prediction_to_target_pair_magnitude_ratio_minimum": 0.50,
                "pair_direction_cosine_mean_minimum": 0.50,
            },
            "formal_validation_gate": {
                "every_seed_and_horizon_required": True,
                "exact_target_selection_rate_minimum": 0.60,
                "exact_history_selection_rate_minimum": 0.60,
                "matching_history_strict_win_rate_minimum": 0.50,
            },
        },
        "models": model_rows,
        "by_family": by_family,
        "decision": {
            "pldm_training_domain_stable": pldm_domain_stable,
            "pldm_heldout_same_distribution_stable": (
                pldm_heldout_stable
            ),
            "pldm_training_domain_every_delay_target_75": (
                pldm_training_every_delay_passes == len(SEEDS)
            ),
            "pldm_heldout_every_delay_target_75": (
                pldm_heldout_every_delay_passes == len(SEEDS)
            ),
            "pldm_formal_validation_passed": pldm_validation_passed,
            "lewm_training_domain_stable": int(
                by_family["lewm"]["training_domain_gate_passed_seeds"]
            )
            == len(SEEDS),
            "lewm_heldout_same_distribution_stable": int(
                by_family["lewm"][
                    "heldout_same_distribution_gate_passed_seeds"
                ]
            )
            == len(SEEDS),
            "lewm_formal_validation_passed": int(
                by_family["lewm"][
                    "formal_validation_gate_passed_seeds"
                ]
            )
            == len(SEEDS),
            "stage_conclusion": (
                f"{learning_conclusion} "
                f"{validation_conclusion} "
                "LeWM 与 PLDM 使用完全相同的数据、初始化、预算和 "
                "loss 权重，因此两者差异可归因到训练目标。"
            ),
        },
        "artifacts": artifacts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[action-delay-h7-paired-analysis] wrote {output}", flush=True)
    print(
        json.dumps(payload["decision"], ensure_ascii=False, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
