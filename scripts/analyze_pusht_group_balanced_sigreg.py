#!/usr/bin/env python3
"""Seal the group-balanced SIGReg Push-T validation result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world"
)
DEFAULT_ROOT = ARTIFACT_ROOT / (
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "group_balanced_sigreg_seed5121_step2048"
)
DEFAULT_CALIBRATION = ARTIFACT_ROOT / (
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "group_balanced_sigreg_seed4097_step2048/"
    "first_batch_gradient_report.json"
)
DEFAULT_NATIVE_CONTROL = ARTIFACT_ROOT / (
    "evaluation/history3/pusht_hidden_actuation_h3_v1/"
    "mixed_retention_seed3073_step2048/standard_pusht_cem/"
    "mixed_native_0p09/aggregate.json"
)
PREDICTION_GATES = {
    "target_selection": 0.95,
    "correct_history": 0.95,
    "rule_switch": 0.95,
    "worst_mode": 0.90,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--calibration-gradient",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--native-control",
        type=Path,
        default=DEFAULT_NATIVE_CONTROL,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def only(items: list[dict[str, Any]], context: str) -> dict[str, Any]:
    if len(items) != 1:
        raise ValueError(f"{context}: expected one item, found {len(items)}")
    return items[0]


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    root = Path(os.path.abspath(args.root.expanduser()))
    calibration_path = args.calibration_gradient.expanduser().resolve()
    native_control_path = args.native_control.expanduser().resolve()
    output = (
        Path(os.path.abspath(args.output.expanduser()))
        if args.output
        else root / "summary.json"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")

    paths = {
        "calibration_gradient_seed4097": calibration_path,
        "held_out_gradient_seed5121": root
        / "first_batch_gradient_report.json",
        "training_and_hidden_prediction": root
        / "training/mixed_report.json",
        "standard_pusht_cem": root
        / "standard_pusht_cem/aggregate.json",
        "hidden_actuation_cem": root
        / "hidden_actuation_cem_seed5120/aggregate.json",
        "replay_matched_native_standard_cem": native_control_path,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    calibration = load(paths["calibration_gradient_seed4097"])
    held_out = load(paths["held_out_gradient_seed5121"])
    training_report = load(paths["training_and_hidden_prediction"])
    training = only(training_report["results"], "training report")
    final = training["snapshots"][-1]
    if final["optimizer_step"] != 2048:
        raise ValueError("Training report does not end at step 2048")
    hidden_metrics = final["hidden_evaluation"]
    prediction = {
        "target_selection": hidden_metrics[
            "two_real_future_target_selection_rate"
        ],
        "correct_history": hidden_metrics[
            "correct_history_preference_rate"
        ],
        "rule_switch": hidden_metrics["correct_rule_switch_rate"],
        "worst_mode": hidden_metrics["worst_mode_target_selection_rate"],
        "prediction_mse_margin": hidden_metrics["prediction_mse"][
            "incorrect_minus_correct_margin"
        ],
        "paired_to_unrelated_ratio": hidden_metrics[
            "representation_geometry"
        ]["prediction_space"]["paired_to_unrelated_ratio"],
    }
    prediction["gate_passed"] = all(
        prediction[name] >= threshold
        for name, threshold in PREDICTION_GATES.items()
    )

    standard_report = load(paths["standard_pusht_cem"])
    standard_model = only(standard_report["models"], "standard CEM")
    standard = dict(standard_model["aggregate"])
    native_report = load(paths["replay_matched_native_standard_cem"])
    native_model = only(native_report["models"], "native standard CEM")
    native = dict(native_model["aggregate"])
    if (
        standard_report["query_catalog"]["sha256"]
        != native_report["query_catalog"]["sha256"]
    ):
        raise ValueError("Standard CEM query catalogs do not match")
    standard["replay_matched_native_success_rate"] = native["success_rate"]
    standard["difference_from_replay_matched_native"] = (
        standard["success_rate"] - native["success_rate"]
    )
    standard["noninferiority_margin_absolute"] = 0.05
    standard["noninferior"] = (
        standard["difference_from_replay_matched_native"] >= -0.05
    )
    standard["query_catalog_sha256"] = standard_report["query_catalog"][
        "sha256"
    ]

    hidden_report = load(paths["hidden_actuation_cem"])
    hidden_model = only(hidden_report["models"], "hidden CEM")
    hidden = dict(hidden_model["summary"])
    hidden["mode_action_gate_passed"] = (
        hidden["oracle_mode_action_classification_rate"] >= 0.90
    )
    hidden["strict_success_gate_passed"] = (
        hidden["real_environment_success_rate"] >= 0.80
    )
    hidden["gate_passed"] = (
        hidden["oracle_grid_success_rate"] >= 0.95
        and hidden["mode_action_gate_passed"]
        and hidden["strict_success_gate_passed"]
    )

    payload = {
        "schema_version": 1,
        "benchmark": "pusht_hidden_actuation_history3_action_coverage_v2",
        "experiment": "group_balanced_sigreg_joint_retention_v2",
        "owner": "ContextWorld",
        "status": "complete_no_joint_pass",
        "method": {
            "display_name_zh": "分组平衡 SIGReg",
            "formula": (
                "active: 0.5 * (SIGReg(full batch) + "
                "SIGReg(matched difference / sqrt(2))); "
                "inactive: SIGReg(full batch)"
            ),
            "external_weight": 0.05,
            "effective_active_weight_per_group": 0.025,
            "training_seed": 5121,
        },
        "gradient_screen": {
            "calibration_seed4097": {
                "weight": 0.02,
                "status": calibration["status"],
                "total_probe_t1_direction": calibration[
                    "gradient_effects"
                ]["probe_revealed_t1"]["group_balanced_total"]["all"][
                    "predicted_distance_change_per_unit_lr"
                ],
                "total_future_t3_direction": calibration[
                    "gradient_effects"
                ]["history_conditioned_future_t3"][
                    "group_balanced_total"
                ]["all"]["predicted_distance_change_per_unit_lr"],
            },
            "held_out_seed5121": {
                "weight": 0.05,
                "status": held_out["status"],
                "all_gates_passed": held_out["all_gates_passed"],
                "total_probe_t1_direction": held_out[
                    "gradient_effects"
                ]["probe_revealed_t1"]["group_balanced_total"]["all"][
                    "predicted_distance_change_per_unit_lr"
                ],
                "total_future_t3_direction": held_out[
                    "gradient_effects"
                ]["history_conditioned_future_t3"][
                    "group_balanced_total"
                ]["all"]["predicted_distance_change_per_unit_lr"],
            },
        },
        "hidden_prediction": prediction,
        "standard_pusht_cem": standard,
        "hidden_actuation_cem": hidden,
        "joint_decision": {
            "hidden_prediction_passed": prediction["gate_passed"],
            "standard_pusht_noninferior": standard["noninferior"],
            "hidden_actuation_cem_passed": hidden["gate_passed"],
            "jointly_passed": (
                prediction["gate_passed"]
                and standard["noninferior"]
                and hidden["gate_passed"]
            ),
            "interpretation": (
                "Separately scoring the marginal and paired-difference "
                "populations restores hidden-future discrimination, but it "
                "does not restore standard latent planning calibration or "
                "strict hidden-actuation execution success."
            ),
        },
        "sources": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    atomic_write(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "status": payload["status"],
                "joint_decision": payload["joint_decision"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
