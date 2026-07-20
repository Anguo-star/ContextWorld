#!/usr/bin/env python3
"""Evaluate frozen 1/2/3/5-block native-latent rollout error."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.icl_model import state_dict_sha256
from contextworld.evaluation.protocol import (
    frozen_normalizer_process,
    infer_model_protocol,
    load_pretrained_cost_model,
)
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from scripts.eval_tworoom_step1 import image_transform


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
HORIZONS = (1, 2, 3, 5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _h5_dataset(swm: Any, path: Path):
    return swm.data.HDF5Dataset(
        path=path,
        frameskip=1,
        num_steps=1,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )


def _lance_dataset(swm: Any, path: Path):
    return swm.data.LanceDataset(
        path=path,
        frameskip=1,
        num_steps=1,
        keys_to_load=["pixels", "action"],
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _assets(
    dataset: Any,
    entries: list[dict[str, Any]],
    action_standardizer: Any,
) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.asarray([entry["episode"] for entry in entries], dtype=np.int64)
    starts = np.asarray([entry["start_step"] for entry in entries], dtype=np.int64)
    chunks = dataset.load_chunk(episodes, starts, starts + 36)
    pixels = []
    actions = []
    for chunk in chunks:
        raw_pixels = _to_numpy(chunk["pixels"])
        if raw_pixels.shape[1] == 3:
            raw_pixels = raw_pixels.transpose(0, 2, 3, 1)
        pixels.append(raw_pixels[np.arange(0, 36, 5)])
        raw_actions = _to_numpy(chunk["action"][:35]).reshape(7, 5, 2)
        normalized = action_standardizer.transform(
            raw_actions.reshape(-1, 2)
        ).astype(np.float32)
        actions.append(normalized.reshape(7, 10))
    return np.stack(pixels), np.stack(actions)


def _evaluate_group(
    *,
    model: Any,
    transform: Any,
    dataset: Any,
    entries: list[dict[str, Any]],
    action_standardizer: Any,
    device: str,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    pixel_values, action_values = _assets(
        dataset, entries, action_standardizer
    )
    transformed = torch.stack(
        [
            torch.stack([transform(frame) for frame in sequence])
            for sequence in pixel_values
        ]
    ).to(device)
    actions = torch.from_numpy(action_values).to(device)
    with torch.inference_mode():
        target_embeddings = model.encode({"pixels": transformed})["emb"]
        rollout = model.rollout(
            {"pixels": transformed[:, None, :3]},
            actions[:, None],
            history_size=3,
        )["predicted_emb"][:, 0]
    predictions = rollout[:, 3:8]
    targets = target_embeddings[:, 3:8]
    records = []
    for row, entry in enumerate(entries):
        horizon_metrics = {}
        for horizon in HORIZONS:
            prediction = predictions[row, horizon - 1]
            target = targets[row, horizon - 1]
            mse = float(F.mse_loss(prediction, target).item())
            horizon_metrics[str(horizon)] = {
                "latent_mse": mse,
                "latent_rmse": float(np.sqrt(mse)),
                "latent_cosine_distance": float(
                    (1.0 - F.cosine_similarity(
                        prediction[None], target[None]
                    )).item()
                ),
            }
        records.append({**entry, "horizons": horizon_metrics})
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    catalog_path = resolve_contextworld_path(args.catalog, repo_root=REPO_ROOT)
    checkpoint = resolve_contextworld_path(args.checkpoint, repo_root=REPO_ROOT)
    normalizer = resolve_contextworld_path(args.normalizer, repo_root=REPO_ROOT)
    output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = list(catalog["entries"])
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    process = frozen_normalizer_process(normalizer)
    model = load_pretrained_cost_model(
        checkpoint,
        swm,
        cache_dir=artifact_path("evaluation/model_cache", repo_root=REPO_ROOT),
    )
    protocol = infer_model_protocol(model, action_dim=2)
    if protocol != {"action_block": 5, "history_size": 3}:
        raise RuntimeError(f"Unexpected model protocol: {protocol}")
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    setattr(model, "history_size", 3)
    setattr(model, "interpolate_pos_encoding", True)
    before = state_dict_sha256(model)
    transform = image_transform(224)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[(entry["source_kind"], entry["source_path"])].append(entry)
    records = []
    for index, ((source_kind, logical_path), values) in enumerate(
        grouped.items(), start=1
    ):
        print(f"[{index}/{len(grouped)}] {source_kind} n={len(values)}", flush=True)
        path = resolve_contextworld_path(logical_path, repo_root=REPO_ROOT)
        dataset = (
            _h5_dataset(swm, path)
            if source_kind == "original_h5"
            else _lance_dataset(swm, path)
        )
        for start in range(0, len(values), args.batch_size):
            records.extend(
                _evaluate_group(
                    model=model,
                    transform=transform,
                    dataset=dataset,
                    entries=values[start : start + args.batch_size],
                    action_standardizer=process["action"],
                    device=args.device,
                )
            )
    after = state_dict_sha256(model)
    if before != after:
        raise RuntimeError("Model weights changed during rollout evaluation")
    aggregates = []
    for domain in sorted({record["domain"] for record in records}):
        selected = [record for record in records if record["domain"] == domain]
        for horizon in HORIZONS:
            metrics = [
                record["horizons"][str(horizon)] for record in selected
            ]
            aggregates.append(
                {
                    "domain": domain,
                    "horizon_action_blocks": horizon,
                    "evaluations": len(metrics),
                    "mean_latent_mse": float(
                        np.mean([metric["latent_mse"] for metric in metrics])
                    ),
                    "mean_latent_rmse": float(
                        np.mean([metric["latent_rmse"] for metric in metrics])
                    ),
                    "mean_latent_cosine_distance": float(
                        np.mean(
                            [
                                metric["latent_cosine_distance"]
                                for metric in metrics
                            ]
                        )
                    ),
                }
            )
    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_original_ability_rollout_error_v1",
        "status": "passed",
        "catalog": {"path": str(catalog_path), "sha256": _sha256(catalog_path)},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "normalizer": {"path": str(normalizer), "sha256": _sha256(normalizer)},
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "protocol": {
            **protocol,
            "history_observation_blocks": 3,
            "frameskip": 5,
            "horizons_action_blocks": list(HORIZONS),
            "endpoint_error_definition": (
                "native latent RMSE between the autoregressive predicted "
                "endpoint and the encoded realized endpoint"
            ),
        },
        "frozen_weight_audit": {
            "state_dict_sha256_before": before,
            "state_dict_sha256_after": after,
            "passed": before == after,
        },
        "aggregates": aggregates,
        "raw_records": records,
    }
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=Path(
            "artifacts/splits/tworoom_original_train_s3072_normalizer.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"]}, sort_keys=True))
