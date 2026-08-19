#!/usr/bin/env python3
"""Diagnose three-way versus eleven-way Action Delay scoring resolution."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import artifact_path
from contextworld.synthesis.manifest import write_json


SEEDS = (3072, 4096, 5120)
FAMILIES = ("pldm", "lewm")
TRAINING_DELAYS = (0, 4, 8)
ALL_DELAYS = tuple(range(11))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stats(values: Iterable[float]) -> dict[str, float]:
    rows = [float(value) for value in values]
    _require(rows, "不能汇总空指标")
    return {
        "mean": float(statistics.fmean(rows)),
        "sample_std": (
            float(statistics.stdev(rows)) if len(rows) > 1 else 0.0
        ),
        "minimum": float(min(rows)),
        "maximum": float(max(rows)),
    }


def _slug(family: str, seed: int) -> str:
    return f"h7_action_delay_paired_{family}_formal_s{seed}"


def score_h1_candidate_resolution(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    h1 = [row for row in records if int(row["horizon"]) == 1]
    queries = sorted({str(row["query_id"]) for row in h1})
    _require(len(queries) == 300, f"冻结 Validation 应有 300 query，实际 {len(queries)}")
    lookup: dict[tuple[str, int, int], float] = {}
    for row in h1:
        key = (
            str(row["query_id"]),
            int(row["history_delay"]),
            int(row["target_delay"]),
        )
        _require(key not in lookup, f"h1 loss 重复：{key}")
        lookup[key] = float(row["latent_mse"])
    _require(
        len(lookup) == 300 * 11 * 11,
        f"h1 loss 矩阵不完整：{len(lookup)}",
    )

    restricted_target_correct = 0
    restricted_history_correct = 0
    restricted_by_history = {
        delay: {"correct": 0, "units": 0} for delay in TRAINING_DELAYS
    }
    full_confusion = {
        delay: Counter({candidate: 0 for candidate in ALL_DELAYS})
        for delay in TRAINING_DELAYS
    }
    full_abs_errors: list[float] = []
    full_selected: list[float] = []
    for query in queries:
        for history_delay in TRAINING_DELAYS:
            selected = min(
                TRAINING_DELAYS,
                key=lambda target: (
                    lookup[(query, history_delay, target)],
                    target,
                ),
            )
            correct = selected == history_delay
            restricted_target_correct += int(correct)
            restricted_by_history[history_delay]["correct"] += int(correct)
            restricted_by_history[history_delay]["units"] += 1

            full_selected_delay = min(
                ALL_DELAYS,
                key=lambda target: (
                    lookup[(query, history_delay, target)],
                    target,
                ),
            )
            full_confusion[history_delay][full_selected_delay] += 1
            full_selected.append(float(full_selected_delay))
            full_abs_errors.append(
                float(abs(full_selected_delay - history_delay))
            )

        for target_delay in TRAINING_DELAYS:
            selected_history = min(
                TRAINING_DELAYS,
                key=lambda history: (
                    lookup[(query, history, target_delay)],
                    history,
                ),
            )
            restricted_history_correct += int(
                selected_history == target_delay
            )

    units = len(queries) * len(TRAINING_DELAYS)
    return {
        "queries": len(queries),
        "restricted_three_way": {
            "candidate_delays": list(TRAINING_DELAYS),
            "target_selection_units": units,
            "exact_target_selection_rate": (
                restricted_target_correct / units
            ),
            "history_selection_units": units,
            "exact_history_selection_rate": (
                restricted_history_correct / units
            ),
            "by_history_delay": {
                str(delay): {
                    "units": row["units"],
                    "exact_target_selection_rate": (
                        row["correct"] / row["units"]
                    ),
                }
                for delay, row in restricted_by_history.items()
            },
        },
        "full_eleven_way": {
            "candidate_delays": list(ALL_DELAYS),
            "target_selection_units": units,
            "mean_selected_delay": statistics.fmean(full_selected),
            "mean_absolute_delay_error": statistics.fmean(full_abs_errors),
            "selected_target_confusion_counts": {
                str(history): {
                    str(target): int(count)
                    for target, count in sorted(counts.items())
                }
                for history, counts in full_confusion.items()
            },
            "selected_target_confusion_rates": {
                str(history): {
                    str(target): float(count / len(queries))
                    for target, count in sorted(counts.items())
                }
                for history, counts in full_confusion.items()
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=artifact_path(
            "evaluation/history7/action_delay_paired_repair_v1/"
            "model_results",
            repo_root=ROOT,
        ),
    )
    parser.add_argument(
        "--paired-summary",
        type=Path,
        default=artifact_path(
            "evaluation/history7/action_delay_paired_repair_v1/"
            "comparison_summary.json",
            repo_root=ROOT,
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_path(
            "evaluation/history7/action_delay_paired_repair_v1/"
            "candidate_resolution_diagnostic.json",
            repo_root=ROOT,
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    paired_summary_path = args.paired_summary.expanduser().resolve()
    paired_summary = json.loads(
        paired_summary_path.read_text(encoding="utf-8")
    )
    model_results: dict[str, Any] = {}
    input_files: dict[str, dict[str, str]] = {}
    for family in FAMILIES:
        for seed in SEEDS:
            slug = _slug(family, seed)
            path = result_root / f"{slug}_validation.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            _require(
                payload.get("status") == "completed_post_hoc_diagnostic"
                and payload.get("score_audit", {}).get("passed") is True,
                f"冻结 Validation 结果未通过审计：{path}",
            )
            _require(
                payload.get("label") == slug
                and payload.get("model_family") == family,
                f"模型身份不一致：{path}",
            )
            model_results[slug] = {
                "model_family": family,
                "training_seed": seed,
                **score_h1_candidate_resolution(payload["records"]),
            }
            input_files[slug] = {
                "path": str(path),
                "sha256": file_sha256(path),
            }

    family_results: dict[str, Any] = {}
    for family in FAMILIES:
        selected = [
            model_results[_slug(family, seed)] for seed in SEEDS
        ]
        heldout_three_way = float(
            paired_summary["by_family"][family][
                "heldout_same_distribution_h1"
            ]["exact_target_selection_rate"]["mean"]
        )
        frozen_three_way_values = [
            row["restricted_three_way"]["exact_target_selection_rate"]
            for row in selected
        ]
        frozen_history_values = [
            row["restricted_three_way"]["exact_history_selection_rate"]
            for row in selected
        ]
        aggregate_confusion = {
            str(history): {
                str(target): int(
                    sum(
                        row["full_eleven_way"][
                            "selected_target_confusion_counts"
                        ][str(history)][str(target)]
                        for row in selected
                    )
                )
                for target in ALL_DELAYS
            }
            for history in TRAINING_DELAYS
        }
        confusion_units = 300 * len(SEEDS)
        family_results[family] = {
            "models": len(selected),
            "heldout_same_distribution_three_way_target_rate": (
                heldout_three_way
            ),
            "frozen_validation_same_geometry_three_way_target_rate": (
                _stats(frozen_three_way_values)
            ),
            "frozen_validation_same_geometry_three_way_history_rate": (
                _stats(frozen_history_values)
            ),
            "absolute_three_way_target_rate_difference": abs(
                statistics.fmean(frozen_three_way_values)
                - heldout_three_way
            ),
            "full_eleven_way_selected_target_confusion_counts": (
                aggregate_confusion
            ),
            "full_eleven_way_selected_target_confusion_rates": {
                str(history): {
                    str(target): float(count / confusion_units)
                    for target, count in row.items()
                }
                for history, row in aggregate_confusion.items()
            },
        }

    pldm_gap = family_results["pldm"][
        "absolute_three_way_target_rate_difference"
    ]
    geometry_shift_explains_gap = pldm_gap > 0.03
    output = args.output.expanduser().resolve()
    payload = {
        "schema_version": 1,
        "benchmark": (
            "tworoom_action_delay_history7_candidate_resolution_"
            "diagnostic_v1"
        ),
        "status": "completed",
        "question": (
            "PLDM 从三选一约 84% 到十一选一约 50% 的差距，是否由 "
            "Validation query 几何分布变化造成？"
        ),
        "claim_boundary": {
            "model_rerun": False,
            "uses_frozen_raw_latent_losses": True,
            "changes_formal_validation_gate": False,
            "diagnostic_only": True,
        },
        "identity": {
            "paired_summary": {
                "path": str(paired_summary_path),
                "sha256": file_sha256(paired_summary_path),
            },
            "validation_result_files": input_files,
        },
        "models": model_results,
        "by_family": family_results,
        "decision": {
            "query_geometry_shift_explains_pldm_gap": (
                geometry_shift_explains_gap
            ),
            "pldm_three_way_rate_reproduced_on_frozen_geometry": (
                not geometry_shift_explains_gap
            ),
            "pldm_remaining_error_pattern": (
                "ordered_but_compressed_response_with_adjacent_delay_"
                "selection"
            ),
            "lewm_error_pattern": (
                "history_independent_middle_delay_selection"
            ),
            "root_cause_scope": (
                "This isolates candidate resolution and response amplitude; "
                "it does not by itself prove the optimizer-level cause."
            ),
        },
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "by_family": family_results,
                "decision": payload["decision"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
