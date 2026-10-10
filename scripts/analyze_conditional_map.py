#!/usr/bin/env python3
"""Summarize two-task latent bridge and original/frozen checkpoint controls."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from diagnose_conditional_map import TASKS, metric, pair_diff
from diagnose_task_history_readout import sha, write


def _load_state(path: Path) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def _projector_changes(first: dict, second: dict) -> tuple[int, int]:
    import torch

    def projector(state):
        return {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.removeprefix("model.").startswith("projector.")
        }

    original = projector(first)
    frozen = projector(second)
    assert original and original.keys() == frozen.keys()
    return len(original), sum(not torch.equal(original[key], frozen[key]) for key in original)


def summarize(task: str, source: Path, panels: Path, models: dict, validation: dict) -> dict:
    folder = source / task
    bridge = json.loads((folder / "result.json").read_text())
    assert bridge["task"] == task and not bridge["gate_pass"]
    assert bridge["no_model_update"] and bridge["no_test_read"]
    manifest_path = panels / task / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert sha(manifest_path) == bridge["panel_manifest_sha256"]
    dev_path = panels / task / manifest["splits"]["development"]["path"]
    assert sha(dev_path) == bridge["split_sha256"]["development"]
    with np.load(dev_path, allow_pickle=False) as data_file:
        data = {key: data_file[key] for key in data_file.files}

    encoded = {}
    model_meta = {}
    for regime in ("original", "joint", "frozen"):
        model_meta[regime] = json.loads(
            (folder / f"{regime}_meta.json").read_text())
        array_path = folder / f"{regime}_development.npz"
        assert sha(
            array_path) == model_meta[regime]["development_latent"]["sha256"]
        with np.load(array_path, allow_pickle=False) as array_file:
            encoded[regime] = {key: array_file[key]
                               for key in array_file.files}

    original = models[model_meta["original"]["model_id"]]
    frozen = models[model_meta["frozen"]["model_id"]]
    validate = validation["tasks"][task]
    assert original["checkpoint_sha256"] == validate["checkpoint_sha256"]["T0"]
    assert frozen["checkpoint_sha256"] == validate["checkpoint_sha256"]["T3"]
    assert validate["changed_encoder_keys_vs_T0"]["T3"] == 0
    projector_count, projector_changed = _projector_changes(
        _load_state(Path(original["checkpoint"])),
        _load_state(Path(frozen["checkpoint"])),
    )
    assert projector_changed > 0

    response = {}
    for regime in ("original", "joint", "frozen"):
        target, queries, groups = pair_diff(encoded[regime]["target"], data)
        predicted, check_queries, check_groups = pair_diff(
            encoded[regime]["prediction"], data)
        assert np.array_equal(queries, check_queries) and np.array_equal(
            groups, check_groups)
        response[regime] = metric(predicted, target)
    assert abs(response["joint"]["nre"] - bridge["native"]
               ["joint"]["response"]["nre"]) < 1e-10
    assert abs(response["frozen"]["nre"] - bridge["native"]
               ["frozen"]["response"]["nre"]) < 1e-10

    target_difference = np.abs(
        encoded["original"]["target"] - encoded["frozen"]["target"])
    return {
        "task": task,
        "paired_development_queries": len(np.unique(data["query_ids"])),
        "training_pairs_for_bridge": 512,
        "checkpoint_sha256": {regime: model_meta[regime]["checkpoint_sha256"] for regime in model_meta},
        "panel_development_sha256": bridge["split_sha256"]["development"],
        "backbone_tensor_count": validate["encoder_key_count"],
        "backbone_changed_T0_to_T3": validate["changed_encoder_keys_vs_T0"]["T3"],
        "projector_tensor_count": projector_count,
        "projector_changed_T0_to_T3": projector_changed,
        "target_T0_T3_max_abs_difference": float(target_difference.max()),
        "native_response": response,
        "T2_to_T3_true_target_linear_bridge": {
            "ridge_alpha": bridge["ridge_alpha"],
            "training_group_cv_nre": bridge["train_group_cv_nre"][str(bridge["ridge_alpha"])],
            "development": bridge["true_target_map_fidelity"],
            "common_space_prediction_interpretable": bridge["gate_pass"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("/tmp/cw-conditional-map-20261010"))
    parser.add_argument("--panels", type=Path,
                        default=Path("/tmp/cw-cross-task-mechanism-20261010/native"))
    parser.add_argument("--models", type=Path,
                        default=Path("/tmp/cw-icl-validity-20261007/models.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/research/data/conditional_map_v1"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    models = {entry["id"]: entry for entry in json.loads(
        args.models.read_text())}
    validation = json.loads(Path(
        "docs/research/data/matched_encoder_contrast_v1/encoder_freeze_validation.json").read_text())
    tasks = [summarize(task, args.source, args.panels,
                       models, validation) for task in TASKS]
    payload = {
        "schema": "contextworld.conditional_map_summary.v1",
        "scope": "LeWM, two tasks, one training seed per policy, matched Development queries; no world-model training or Test access",
        "interpretation": "The Train-fitted linear bridge does not support common-latent predictor comparison. Frozen ViT backbone does not freeze the learned target projector.",
        "tasks": tasks,
    }
    rows = [{
        "task": item["task"],
        "T0_native_response_nre": item["native_response"]["original"]["nre"],
        "T2_native_response_nre": item["native_response"]["joint"]["nre"],
        "T3_native_response_nre": item["native_response"]["frozen"]["nre"],
        "T2_to_T3_target_bridge_dev_nre": item["T2_to_T3_true_target_linear_bridge"]["development"]["nre"],
        "projector_tensors_changed_T0_to_T3": item["projector_changed_T0_to_T3"],
        "projector_tensor_count": item["projector_tensor_count"],
    } for item in tasks]
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_text = stream.getvalue()
    if args.check:
        assert json.loads(
            (args.output / "summary.json").read_text()) == payload
        assert (args.output / "summary.csv").read_text() == csv_text
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        write(args.output / "summary.json", payload)
        (args.output / "summary.csv").write_text(csv_text)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
