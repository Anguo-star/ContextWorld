#!/usr/bin/env python3
"""Aggregate the completed History-7 Action Delay root-cause evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


ARTIFACTS = {
    "lewm_trajectory_root": (
        "artifacts/evaluation/history7/action_delay_root_cause_v1/"
        "lewm_checkpoint_trajectory"
    ),
    "pldm_trajectory_root": (
        "artifacts/evaluation/history7/action_delay_pldm_root_cause_v1/"
        "checkpoint_trajectory"
    ),
    "pldm_training_report": (
        "artifacts/training/reports/"
        "h7_action_delay_multi_pldm_root_cause_s3072.json"
    ),
    "pldm_training_replay": (
        "artifacts/evaluation/history7/action_delay_pldm_root_cause_v1/"
        "training_replay_final.json"
    ),
    "lewm_validation": (
        "artifacts/evaluation/history7/action_delay_validation_v1/"
        "model_results/h7_action_delay_multi_formal_s3072.json"
    ),
    "pldm_validation": (
        "artifacts/evaluation/history7/action_delay_pldm_root_cause_v1/"
        "frozen_validation_final.json"
    ),
    "lewm_capacity": (
        "artifacts/evaluation/history7/"
        "action_delay_capacity_diagnostic_v1/lewm.json"
    ),
    "pldm_capacity": (
        "artifacts/evaluation/history7/"
        "action_delay_capacity_diagnostic_v1/pldm.json"
    ),
    "lewm_capacity_extension": (
        "artifacts/evaluation/history7/"
        "action_delay_capacity_diagnostic_v1/lewm_1024_extension.json"
    ),
}
DEFAULT_OUTPUT = (
    "artifacts/evaluation/history7/action_delay_root_cause_v1/"
    "root_cause_summary.json"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _resolve(logical: str) -> Path:
    path = resolve_contextworld_path(logical, repo_root=ROOT)
    _require(path.is_file() or path.is_dir(), f"Missing artifact: {path}")
    return path


def _alignment_average(payload: dict[str, Any]) -> dict[str, float]:
    rows = [
        row["latent_alignment_h1"]
        for row in payload["tracks"].values()
    ]
    return {
        key: float(mean(float(row[key]) for row in rows))
        for key in (
            "target_pair_mse",
            "prediction_pair_mse",
            "prediction_to_target_pair_magnitude_ratio",
            "pair_direction_cosine_mean",
            "pair_direction_positive_fraction",
            "centered_delay_pattern_cosine_mean",
        )
    }


def _trajectory(root: Path, family: str) -> list[dict[str, Any]]:
    output = []
    for epoch in range(1, 5):
        path = root / f"{family}_epoch_{epoch}.json"
        payload = _read(path)
        _require(
            payload.get("status") == "completed_post_hoc_diagnostic"
            and payload.get("model_family") == family,
            f"Invalid checkpoint diagnostic: {path}",
        )
        source = payload["aggregate_source_h1"]
        output.append(
            {
                "epoch": epoch,
                "optimizer_step": epoch * 256,
                "exact_target_selection_rate": float(
                    source["exact_target_selection_rate"]
                ),
                "exact_history_selection_rate": float(
                    source["exact_history_selection_rate"]
                ),
                "delay4_selected_rate": float(
                    source["selected_target_rates"]["4"]
                ),
                "latent_alignment": _alignment_average(payload),
                "artifact": str(path),
                "artifact_sha256": file_sha256(path),
            }
        )
    return output


def _validation(payload: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, section in (
        ("h1", payload["summary"]["by_horizon"]["1"]),
        ("h2", payload["summary"]["by_horizon"]["2"]),
        ("h3", payload["summary"]["by_horizon"]["3"]),
        ("trajectory", payload["summary"]["trajectory"]),
    ):
        overall = section["overall"]
        result[name] = {
            "exact_target_selection_rate": float(
                overall["exact_target_selection_rate"]
            ),
            "exact_history_selection_rate": float(
                overall["exact_history_selection_rate"]
            ),
            "matching_history_strict_win_rate": float(
                overall["matching_history_strict_win_rate"]
            ),
            "physical_target_group_selection_rate": float(
                overall["physical_target_group_selection_rate"]
            ),
            "physical_history_group_selection_rate": float(
                overall["physical_history_group_selection_rate"]
            ),
            "mean_history_margin": float(
                overall["mean_history_margin"]
            ),
            "training_seen": {
                "exact_target_selection_rate": float(
                    section["by_track"]["training_seen"][
                        "exact_target_selection_rate"
                    ]
                ),
                "exact_history_selection_rate": float(
                    section["by_track"]["training_seen"][
                        "exact_history_selection_rate"
                    ]
                ),
            },
        }
    return result


def _capacity(payload: dict[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("status")
        == "completed_post_hoc_capacity_diagnostic",
        "Capacity result is incomplete",
    )
    return {
        variant: {
            split: {
                "exact_target_selection_rate": float(
                    row["final"][split][
                        "exact_target_selection_rate"
                    ]
                ),
                "exact_history_selection_rate": float(
                    row["final"][split][
                        "exact_history_selection_rate"
                    ]
                ),
                "prediction_to_target_pair_magnitude_ratio": float(
                    row["final"][split]["latent_alignment"][
                        "prediction_to_target_pair_magnitude_ratio"
                    ]
                ),
            }
            for split in ("train", "heldout")
        }
        for variant, row in payload["variants"].items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        name: _resolve(logical)
        for name, logical in ARTIFACTS.items()
    }
    lewm_trajectory = _trajectory(paths["lewm_trajectory_root"], "lewm")
    pldm_trajectory = _trajectory(paths["pldm_trajectory_root"], "pldm")
    pldm_report = _read(paths["pldm_training_report"])
    pldm_replay = _read(paths["pldm_training_replay"])
    lewm_validation_payload = _read(paths["lewm_validation"])
    pldm_validation_payload = _read(paths["pldm_validation"])
    lewm_capacity_payload = _read(paths["lewm_capacity"])
    pldm_capacity_payload = _read(paths["pldm_capacity"])
    lewm_extension_payload = _read(paths["lewm_capacity_extension"])

    _require(
        pldm_report.get("passed") is True
        and pldm_report.get("model", {}).get("training_method") == "pldm"
        and pldm_report.get("training", {}).get("global_step") == 1024
        and pldm_report.get("save_load_exact") is True,
        "PLDM training report failed its receipt audit",
    )
    _require(
        pldm_replay.get("status") == "completed_post_hoc_diagnostic",
        "PLDM training-replay result is incomplete",
    )
    _require(
        lewm_validation_payload.get("status") == "completed"
        and pldm_validation_payload.get("status")
        == "completed_post_hoc_diagnostic",
        "Frozen Validation results are incomplete",
    )

    lewm_capacity = _capacity(lewm_capacity_payload)
    pldm_capacity = _capacity(pldm_capacity_payload)
    lewm_extension = _capacity(lewm_extension_payload)
    capacity_threshold = 0.90
    heldout_signal_threshold = 0.60
    lewm_capacity_rate = lewm_extension["paired_final"]["train"][
        "exact_target_selection_rate"
    ]
    pldm_capacity_rate = pldm_capacity["paired_final"]["train"][
        "exact_target_selection_rate"
    ]
    lewm_heldout_rate = lewm_extension["paired_final"]["heldout"][
        "exact_target_selection_rate"
    ]
    pldm_heldout_rate = pldm_capacity["paired_final"]["heldout"][
        "exact_target_selection_rate"
    ]
    checks = {
        "lewm_native_never_learned_correct_targets": all(
            row["exact_target_selection_rate"] < 0.40
            for row in lewm_trajectory
        ),
        "lewm_native_converged_to_delay4": (
            lewm_trajectory[-1]["delay4_selected_rate"] > 0.99
        ),
        "pldm_native_correct_target_selection_failed": (
            pldm_trajectory[-1]["exact_target_selection_rate"] < 0.40
        ),
        "pldm_native_converged_to_delay4": (
            pldm_trajectory[-1]["delay4_selected_rate"] > 0.99
        ),
        "pldm_native_learned_aligned_history_direction": (
            pldm_trajectory[-1]["latent_alignment"][
                "pair_direction_cosine_mean"
            ]
            > 0.90
        ),
        "pldm_native_history_amplitude_remained_too_small": (
            pldm_trajectory[-1]["latent_alignment"][
                "prediction_to_target_pair_magnitude_ratio"
            ]
            < 0.25
        ),
        "lewm_shared_predictor_capacity_passed": (
            lewm_capacity_rate >= capacity_threshold
        ),
        "pldm_shared_predictor_capacity_passed": (
            pldm_capacity_rate >= capacity_threshold
        ),
        "lewm_heldout_transfer_signal_passed": (
            lewm_heldout_rate >= heldout_signal_threshold
        ),
        "pldm_heldout_transfer_signal_passed": (
            pldm_heldout_rate >= heldout_signal_threshold
        ),
        "unpaired_final_supervision_insufficient_for_both": (
            lewm_capacity["unpaired_final"]["train"][
                "exact_target_selection_rate"
            ]
            < capacity_threshold
            and pldm_capacity["unpaired_final"]["train"][
                "exact_target_selection_rate"
            ]
            < capacity_threshold
        ),
        "same_query_pairing_alone_sufficient_for_pldm": (
            pldm_capacity["paired_full"]["train"][
                "exact_target_selection_rate"
            ]
            >= capacity_threshold
        ),
        "same_query_pairing_alone_insufficient_for_lewm_at_512": (
            lewm_capacity["paired_full"]["train"][
                "exact_target_selection_rate"
            ]
            < capacity_threshold
        ),
    }
    _require(all(checks.values()), f"Root-cause checks failed: {checks}")

    output = resolve_contextworld_path(args.output, repo_root=ROOT)
    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_action_delay_history7_root_cause_v1",
        "status": "completed",
        "question": (
            "History=7 Action Delay 没学到，是 LeWM 特有问题、PLDM "
            "也学不到，还是当前训练监督没有迫使模型使用历史？"
        ),
        "chance_references": {
            "training_replay_three_way": 1.0 / 3.0,
            "frozen_validation_eleven_way": 1.0 / 11.0,
        },
        "native_training": {
            "lewm_checkpoint_trajectory": lewm_trajectory,
            "pldm_checkpoint_trajectory": pldm_trajectory,
            "pldm_training_receipt": {
                "path": str(paths["pldm_training_report"]),
                "sha256": file_sha256(paths["pldm_training_report"]),
                "checkpoint": pldm_report["artifacts"]["pretrained"],
                "checkpoint_sha256": pldm_report["artifacts"][
                    "pretrained_sha256"
                ],
                "optimizer_steps": 1024,
                "world_size": 8,
                "passed": True,
            },
        },
        "frozen_validation": {
            "lewm": _validation(lewm_validation_payload),
            "pldm": _validation(pldm_validation_payload),
            "artifacts": {
                "lewm": {
                    "path": str(paths["lewm_validation"]),
                    "sha256": file_sha256(paths["lewm_validation"]),
                },
                "pldm": {
                    "path": str(paths["pldm_validation"]),
                    "sha256": file_sha256(paths["pldm_validation"]),
                },
            },
        },
        "capacity_diagnostic": {
            "thresholds": {
                "training_capacity": capacity_threshold,
                "heldout_transfer_signal": heldout_signal_threshold,
            },
            "lewm_512": lewm_capacity,
            "lewm_1024_paired_final": lewm_extension,
            "pldm_512": pldm_capacity,
            "artifacts": {
                name: {
                    "path": str(paths[name]),
                    "sha256": file_sha256(paths[name]),
                }
                for name in (
                    "lewm_capacity",
                    "pldm_capacity",
                    "lewm_capacity_extension",
                )
            },
        },
        "root_cause_checks": checks,
        "conclusion": {
            "primary": (
                "共享 History=7 预测器具有所需容量；正式配方失败的主因是"
                "同一个 query 没有同时提供三种延迟监督，并且真正依赖历史的"
                "最后一个转移在总体损失中权重过低。"
            ),
            "lewm": (
                "原生 LeWM 从第一个 epoch 起就未学会真实下一帧，随后越来越"
                "集中到 delay 4。明确配套三延迟并强化末步后，1,024 步训练"
                "准确率达到 95.49%，说明不是 LeWM 结构学不会。"
            ),
            "pldm": (
                "原生 PLDM 学到了正确的历史调节方向，但调节幅度约为真实差异"
                "的 20%，因此仍集中到 delay 4。加入同 query 三延迟配套后，"
                "即使继续平均监督全部位置也达到 91.67%，说明 PLDM 目标更有"
                "利于历史绑定，但不能单独修复当前未配套训练配方。"
            ),
            "not_supported": [
                "PLDM 也没有能力学习 Action Delay",
                "LeWM 或 PLDM 看不到七帧历史",
                "Encoder 无法区分三种真实下一帧",
                "简单增加原配方训练步数已经被证明足够",
            ],
        },
        "next_formal_recipe": {
            "data": (
                "每个训练 query 同时提供 delay 0、4、8 三条历史，几何、"
                "query 和动作保持一致。"
            ),
            "loss": (
                "显式提高最后一个 history-dependent 转移的预测权重；"
                "保留全序列损失作为辅助项。"
            ),
            "models": [
                "PLDM 联合训练，作为首选正式候选",
                "LeWM 联合训练，作为目标函数归因对照",
            ],
            "evaluation_order": [
                "先过训练域真实下一帧选择门槛",
                "再过未见几何 Validation",
                "最后才进入多步与 CEM",
            ],
            "required_ablation": (
                "正式配方中分别比较：未配套、仅配套、配套加末步加权；"
                "不能把容量诊断的冻结表示结果直接写成正式 Benchmark 成绩。"
            ),
        },
        "artifact_identity": {
            name: {
                "path": str(path),
                "sha256": (
                    file_sha256(path) if path.is_file() else None
                ),
            }
            for name, path in paths.items()
            if path.is_file()
        },
    }
    write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "checks": checks,
                "native_final": {
                    "lewm_target": lewm_trajectory[-1][
                        "exact_target_selection_rate"
                    ],
                    "pldm_target": pldm_trajectory[-1][
                        "exact_target_selection_rate"
                    ],
                    "pldm_direction_cosine": pldm_trajectory[-1][
                        "latent_alignment"
                    ]["pair_direction_cosine_mean"],
                },
                "capacity": {
                    "lewm_train": lewm_capacity_rate,
                    "lewm_heldout": lewm_heldout_rate,
                    "pldm_train": pldm_capacity_rate,
                    "pldm_heldout": pldm_heldout_rate,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
