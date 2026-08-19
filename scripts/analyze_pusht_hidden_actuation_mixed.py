#!/usr/bin/env python3
"""Seal the mixed Push-T hidden-dynamics and ability-retention result.

The script only combines registered metrics.  It does not select checkpoints,
rank action candidates, or introduce a post-hoc planning metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world/"
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "mixed_retention_seed3073_step2048"
)

PREDICTION_REPORTS = {
    "mixed_native_sigreg_0p09": "native/mixed_report.json",
    "mixed_conditional_sigreg_0p01": "conditional_0p01/mixed_report.json",
    "mixed_conditional_sigreg_0p05": "conditional_0p05/mixed_report.json",
    "mixed_conditional_sigreg_0p01_include_unpaired": (
        "conditional_0p01_include_unpaired/mixed_report.json"
    ),
    "mixed_conditional_sigreg_0p01_complete_haar": (
        "conditional_0p01_complete_haar/mixed_report.json"
    ),
    "mixed_conditional_sigreg_0p05_complete_haar": (
        "conditional_0p05_complete_haar/mixed_report.json"
    ),
}
STANDARD_REPORTS = {
    "source": "standard_pusht_cem/source/aggregate.json",
    "mixed_native_sigreg_0p09": (
        "standard_pusht_cem/mixed_native_0p09/aggregate.json"
    ),
    "mixed_conditional_sigreg_0p01": (
        "standard_pusht_cem/mixed_conditional_0p01/aggregate.json"
    ),
    "mixed_conditional_sigreg_0p01_include_unpaired": (
        "standard_pusht_cem/mixed_conditional_0p01_include_unpaired/"
        "aggregate.json"
    ),
}
HIDDEN_CEM_REPORT = "hidden_cem_seed4096/aggregate.json"
NATIVE_CONTROL = "mixed_native_sigreg_0p09"
PREDICTION_GATED_ADAPTIVE_CANDIDATES = (
    "mixed_conditional_sigreg_0p01_include_unpaired",
    "mixed_conditional_sigreg_0p01_complete_haar",
    "mixed_conditional_sigreg_0p05_complete_haar",
)

PREDICTION_GATES = {
    "target_selection": 0.95,
    "correct_history": 0.95,
    "rule_switch": 0.95,
    "worst_mode": 0.90,
}
HIDDEN_CEM_GATES = {
    "mode_action_classification": 0.90,
    "real_environment_success": 0.80,
}
STANDARD_NONINFERIORITY_MARGIN = 0.05
METHOD_DISPLAY_NAMES_ZH = {
    "source": "续训前原模型",
    "mixed_native_sigreg_0p09": "原始正则混训版",
    "mixed_conditional_sigreg_0p01": "成对差异保护版 0.01",
    "mixed_conditional_sigreg_0p05": "成对差异保护版 0.05",
    "mixed_conditional_sigreg_0p01_include_unpaired": (
        "差异保护兼顾原样本版 0.01"
    ),
    "mixed_conditional_sigreg_0p01_complete_haar": (
        "成对均值与差异完整版 0.01"
    ),
    "mixed_conditional_sigreg_0p05_complete_haar": (
        "成对均值与差异完整版 0.05"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to ROOT/mixed_retention_summary.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing summary after all source checks pass",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_single(items: list[dict[str, Any]], *, context: str) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError(f"{context}: expected one item, found {len(items)}")
    return items[0]


def prediction_summary(report: dict[str, Any]) -> dict[str, Any]:
    result = require_single(report["results"], context="mixed prediction report")
    final = result["snapshots"][-1]
    if final["optimizer_step"] != report["training_contract"]["max_steps"]:
        raise ValueError("Prediction report does not end at the registered step")
    metrics = final["hidden_evaluation"]
    values = {
        "target_selection": metrics["two_real_future_target_selection_rate"],
        "correct_history": metrics["correct_history_preference_rate"],
        "rule_switch": metrics["correct_rule_switch_rate"],
        "worst_mode": metrics["worst_mode_target_selection_rate"],
        "prediction_mse_margin": metrics["prediction_mse"][
            "incorrect_minus_correct_margin"
        ],
        "paired_to_unrelated_ratio": metrics["representation_geometry"][
            "prediction_space"
        ]["paired_to_unrelated_ratio"],
        "future_effective_rank": metrics["representation_geometry"][
            "future_effective_rank"
        ],
    }
    values["gate_passed"] = all(
        values[name] >= threshold
        for name, threshold in PREDICTION_GATES.items()
    )
    values["checkpoint_sha256"] = result["final_checkpoint"]["sha256"]
    return values


def standard_summary(
    report: dict[str, Any],
    *,
    expected_name: str,
) -> dict[str, Any]:
    model = require_single(report["models"], context=expected_name)
    if model["model"] != expected_name:
        aliases = {
            "mixed_native_sigreg_0p09": "mixed_native_0p09",
            "mixed_conditional_sigreg_0p01": "mixed_conditional_0p01",
            "mixed_conditional_sigreg_0p01_include_unpaired": (
                "mixed_conditional_0p01_include_unpaired"
            ),
        }
        if model["model"] != aliases.get(expected_name):
            raise ValueError(
                f"{expected_name}: report contains model {model['model']!r}"
            )
    aggregate = model["aggregate"]
    if aggregate["evaluation_count"] != 300:
        raise ValueError(
            f"{expected_name}: expected 300 evaluations, "
            f"found {aggregate['evaluation_count']}"
        )
    return {
        "success_count": aggregate["success_count"],
        "evaluation_count": aggregate["evaluation_count"],
        "success_rate": aggregate["success_rate"],
        "success_count_by_eval_seed": {
            str(seed["eval_seed"]): seed["success_count"]
            for seed in model["seeds"]
        },
        "query_catalog_sha256": report["query_catalog"]["sha256"],
    }


def hidden_cem_summaries(report: dict[str, Any]) -> dict[str, Any]:
    summaries = {}
    aliases = {
        "mixed_native_0p09": "mixed_native_sigreg_0p09",
        "mixed_conditional_0p01": "mixed_conditional_sigreg_0p01",
    }
    for model in report["models"]:
        name = aliases.get(model["model"], model["model"])
        summary = model["summary"]
        values = {
            "real_environment_success": summary[
                "real_environment_success_rate"
            ],
            "mode_action_classification": summary[
                "oracle_mode_action_classification_rate"
            ],
            "correct_low_greater_than_high": summary[
                "correct_low_greater_than_high_rate"
            ],
            "mean_absolute_amplitude_regret": summary[
                "mean_absolute_amplitude_regret"
            ],
            "oracle_grid_success": summary["oracle_grid_success_rate"],
        }
        values["gate_passed"] = (
            values["mode_action_classification"]
            >= HIDDEN_CEM_GATES["mode_action_classification"]
            and values["real_environment_success"]
            >= HIDDEN_CEM_GATES["real_environment_success"]
        )
        summaries[name] = values
    return summaries


def source_record(root: Path, relative: str) -> dict[str, str]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    root = Path(os.path.abspath(args.root.expanduser()))
    output = (
        Path(os.path.abspath(args.output.expanduser()))
        if args.output
        else root / "mixed_retention_summary.json"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")

    prediction = {}
    sources = {}
    for name, relative in PREDICTION_REPORTS.items():
        path = root / relative
        prediction[name] = prediction_summary(load_json(path))
        sources[f"prediction/{name}"] = source_record(root, relative)

    standard = {}
    query_hashes = set()
    for name, relative in STANDARD_REPORTS.items():
        path = root / relative
        standard[name] = standard_summary(
            load_json(path),
            expected_name=name,
        )
        query_hashes.add(standard[name]["query_catalog_sha256"])
        sources[f"standard_pusht/{name}"] = source_record(root, relative)
    if len(query_hashes) != 1:
        raise ValueError(
            "Standard Push-T reports do not use the same query catalog: "
            f"{sorted(query_hashes)}"
        )

    native_rate = standard[NATIVE_CONTROL]["success_rate"]
    for name, values in standard.items():
        delta = values["success_rate"] - native_rate
        values["absolute_difference_from_replay_matched_native"] = delta
        values["noninferior_to_replay_matched_native"] = (
            delta >= -STANDARD_NONINFERIORITY_MARGIN
        )

    hidden_cem_path = root / HIDDEN_CEM_REPORT
    hidden_cem = hidden_cem_summaries(load_json(hidden_cem_path))
    sources["hidden_cem"] = source_record(root, HIDDEN_CEM_REPORT)

    selected = "mixed_conditional_sigreg_0p01"
    adaptive = "mixed_conditional_sigreg_0p01_include_unpaired"
    complete_haar = (
        "mixed_conditional_sigreg_0p01_complete_haar",
        "mixed_conditional_sigreg_0p05_complete_haar",
    )
    selected_joint_pass = (
        prediction[selected]["gate_passed"]
        and hidden_cem[selected]["gate_passed"]
        and standard[selected]["noninferior_to_replay_matched_native"]
    )
    for name in PREDICTION_GATED_ADAPTIVE_CANDIDATES:
        if prediction[name]["gate_passed"]:
            raise ValueError(
                f"{name} passed prediction but has no registered CEM report"
            )
        skipped = {"status": "skipped_by_registered_prediction_gate"}
        hidden_cem[name] = skipped
        standard.setdefault(name, skipped.copy())

    payload = {
        "schema_version": 1,
        "owner": "ContextWorld",
        "benchmark": "pusht_hidden_actuation_history3_action_coverage_v2",
        "experiment": "mixed_original_hidden_ability_retention_v1",
        "status": "complete_no_jointly_passing_candidate",
        "method_display_names_zh": METHOD_DISPLAY_NAMES_ZH,
        "scope": {
            "training_seed": 3073,
            "standard_pusht_eval_seeds": [42, 43, 44, 45, 46, 47],
            "standard_pusht_query_count": 300,
            "hidden_cem_eval_seed": 4096,
            "claim_level": "single_training_seed_validation",
        },
        "registered_gates": {
            "hidden_prediction": PREDICTION_GATES,
            "hidden_real_cem": HIDDEN_CEM_GATES,
            "standard_pusht_noninferiority_margin_absolute": (
                STANDARD_NONINFERIORITY_MARGIN
            ),
        },
        "source_reports": sources,
        "standard_query_catalog": {
            "all_reports_identical": True,
            "sha256": next(iter(query_hashes)),
        },
        "hidden_prediction": prediction,
        "hidden_real_cem": hidden_cem,
        "standard_pusht_real_cem": standard,
        "decisions": {
            "pre_registered_selected_candidate": selected,
            "selected_candidate_joint_gate_passed": selected_joint_pass,
            "unpaired_population_adaptive_candidate": adaptive,
            "adaptive_hidden_prediction_gate_passed": prediction[adaptive][
                "gate_passed"
            ],
            "complete_haar_adaptive_candidates": list(complete_haar),
            "complete_haar_hidden_prediction_gate_passed": {
                name: prediction[name]["gate_passed"]
                for name in complete_haar
            },
            "adaptive_cem_status": "skipped_by_registered_prediction_gate",
            "jointly_passing_candidate": None,
        },
        "conclusion": [
            (
                "Replay-matched native SIGReg retains standard Push-T ability "
                "but does not learn the hidden-dynamics prediction."
            ),
            (
                "Conditional SIGReg at 0.01 learns the hidden dynamics, but "
                "fails both strict hidden CEM and standard Push-T retention."
            ),
            (
                "Adding unpaired samples to the same active-time population "
                "does not recover the hidden prediction gate, so missing "
                "unpaired coverage is not a sufficient explanation."
            ),
            (
                "A full invertible Haar population also fails the registered "
                "hidden-prediction gate at both 0.01 and 0.05; population "
                "coverage and invertibility do not prevent a pooled marginal "
                "statistic from diluting conditional high-pass structure."
            ),
            (
                "No current candidate simultaneously passes hidden prediction, "
                "hidden real-environment CEM, and standard Push-T retention."
            ),
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "status": payload["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
