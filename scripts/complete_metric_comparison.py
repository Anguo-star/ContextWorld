#!/usr/bin/env python3
"""Complete v1 same-output metrics from cached receipts and LeWM features only."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

TASKS = {"action_strength", "contact_friction", "motion_damping"}
MODELS = 8
SCENES = 256
HORIZONS = 5
REPLICATES = 1000
RTOL = 2e-6
ATOL = 1e-6


class DataError(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_rows(rows: list[tuple[str, ...]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update("\0".join(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataError(f"Expected JSON object: {path}")
    return value


def number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DataError(f"Invalid {label}: {value!r}") from error
    if not math.isfinite(result):
        raise DataError(f"Non-finite {label}: {value!r}")
    return result


def check_close(residuals: dict[str, float], name: str, actual: Any, expected: Any) -> None:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise DataError(f"Invalid values while checking {name}")
    delta = np.abs(a - b)
    residuals[name] = max(residuals.get(name, 0.0), float(np.max(delta, initial=0.0)))
    if not np.allclose(a, b, rtol=RTOL, atol=ATOL):
        raise DataError(f"{name} mismatch; max abs residual={residuals[name]:.12g}")


def native_sums(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Recompute full binary latent errors from [condition,candidate,time,feature]."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 4 or p.shape[0] != 2 or p.shape[2] != HORIZONS:
        raise DataError(f"Unexpected native feature shape: {p.shape}, {y.shape}")
    pm, ym = np.mean(p, axis=0), np.mean(y, axis=0)
    gain_num = float(np.sum((p[1] - p[0]) * (y[1] - y[0]), dtype=np.float64))
    gain_den = float(np.sum((y[1] - y[0]) ** 2, dtype=np.float64))
    if gain_den <= 0:
        raise DataError("Zero target-response energy in LeWM scene")
    return {
        "E": np.sum((p - y) ** 2, axis=(0, 1, 3), dtype=np.float64),
        "W": np.sum((p - y[::-1]) ** 2, axis=(0, 1, 3), dtype=np.float64),
        "R": np.sum(((p - pm[None]) - (y - ym[None])) ** 2, axis=(0, 1, 3), dtype=np.float64),
        "C": 2 * np.sum((pm - ym) ** 2, axis=(0, 2), dtype=np.float64),
        "B": np.sum((y - ym[None]) ** 2, axis=(0, 1, 3), dtype=np.float64),
        "gain_num": gain_num,
        "gain_den": gain_den,
    }


def bootstrap_ci(groups: dict[str, np.ndarray], seed: int) -> dict[str, list[float]]:
    names = sorted(groups)
    if len(names) < 2:
        raise DataError("At least two source groups are required")
    matrix = np.stack([groups[name] for name in names])  # response, bias, E, B
    rng = np.random.default_rng(seed)
    draws = np.empty((REPLICATES, 3), dtype=np.float64)
    for i in range(REPLICATES):
        totals = matrix[rng.integers(0, len(names), len(names))].sum(axis=0)
        r, c, e, b = totals
        if b <= 0:
            raise DataError("Bootstrap draw has zero B")
        draws[i] = r / b, c / b, e / b
    ci = np.quantile(draws, [0.025, 0.975], axis=0, method="linear")
    return {
        "nre": [float(ci[0, 0]), float(ci[1, 0])],
        "common_bias_ratio": [float(ci[0, 1]), float(ci[1, 1])],
        "complete_error_ratio": [float(ci[0, 2]), float(ci[1, 2])],
    }


def build(root: Path, comparison_path: Path, queries_path: Path, out_dir: Path) -> dict[str, Any]:
    root, comparison_path, queries_path, out_dir = (
        p.expanduser().resolve() for p in (root, comparison_path, queries_path, out_dir)
    )
    outputs = {
        "json": out_dir / "same_output_metric_comparison_v2.json",
        "csv": out_dir / "same_output_metric_comparison_v2.csv",
        "queries": out_dir / "same_output_metric_comparison_v2_queries.csv.gz",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(path.exists() or path.is_symlink() for path in outputs.values()):
        raise FileExistsError("Refusing to overwrite a v2 output")

    v1 = read_json(comparison_path)
    with gzip.open(queries_path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        query_fields, queries = list(reader.fieldnames or ()), list(reader)
    models_path = root / "models.json"
    model_defs = json.loads(models_path.read_text(encoding="utf-8"))
    model_rows = {row["model_id"]: row for row in v1.get("models", [])}
    if v1.get("schema") != "cw_metric_comparison_v1" or len(model_rows) != MODELS:
        raise DataError("Expected eight cw_metric_comparison_v1 models")
    defs = {row["id"]: row for row in model_defs}
    qby_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    qkeys = set()
    for row in queries:
        key = (row["id"], row["scene_id"])
        if key in qkeys:
            raise DataError(f"Duplicate query: {key}")
        qkeys.add(key)
        qby_model[row["id"]].append(row)
    if set(model_rows) != set(defs) or set(model_rows) != set(qby_model):
        raise DataError("V1 model, models.json, and query ids differ")
    if len(queries) != MODELS * SCENES or any(len(rows) != SCENES for rows in qby_model.values()):
        raise DataError("Unexpected v1 query coverage")
    actual_models_sha = sha(models_path)
    expected_models_sha = v1.get("inputs", {}).get("models_json_sha256")
    if expected_models_sha and actual_models_sha != expected_models_sha:
        raise DataError("models.json hash differs from v1 provenance")
    seed = int(v1.get("inputs", {}).get("bootstrap_seed", 20261008))

    manifests: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]], str, Path]] = {}
    for task in TASKS:
        path = root / "panels" / task / "manifest.json"
        manifest = read_json(path)
        scenes = {str(s["scene_id"]): s for s in manifest.get("scenes", [])}
        if len(scenes) != SCENES:
            raise DataError(f"Unexpected {task} manifest coverage")
        manifests[task] = manifest, scenes, sha(path), path

    residuals: dict[str, float] = {}
    receipt_hashes: dict[str, dict[str, Any]] = {}
    scene_receipt_hash_rows: list[tuple[str, ...]] = []
    lewm_feature_hash_rows: list[tuple[str, ...]] = []
    panel_hashes: dict[str, dict[str, Any]] = {}
    for task, (_, _, digest, path) in manifests.items():
        panel_hashes[task] = {"path": str(path), "sha256": digest, "scene_count": SCENES}

    new_queries: dict[tuple[str, str], dict[str, Any]] = {}
    decomp_by_model: dict[str, dict[str, Any]] = {}
    for model_id, summary in model_rows.items():
        task, family = str(summary["task"]), str(summary["family"])
        regime = str(summary["regime"])
        manifest, manifest_scenes, panel_sha, _ = manifests[task]
        result_dir = root / "results" / model_id
        receipt_path = result_dir / "receipt.json"
        receipt = read_json(receipt_path)
        if (receipt.get("model_id") != model_id or receipt.get("task") != task
                or receipt.get("family") != family or receipt.get("scenes") != SCENES
                or receipt.get("checkpoint_sha256") != defs[model_id].get("checkpoint_sha256")
                or receipt.get("panel_sha256") != panel_sha):
            raise DataError(f"Receipt identity/manifest mismatch: {model_id}")
        entries = receipt.get("entry_hashes", {})
        if len(entries) != SCENES:
            raise DataError(f"Unexpected receipt entry count: {model_id}")
        receipt_hashes[model_id] = {
            "path": str(receipt_path), "sha256": sha(receipt_path), "scene_count": len(entries)
        }
        if {r["scene_id"] for r in qby_model[model_id]} != set(manifest_scenes):
            raise DataError(f"V1 queries do not match {task} panel manifest")

        group_totals: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.float64))
        e_total = r_total = c_total = b_total = g_numerator = g_denominator = g_energy = 0.0
        independent_count = 0
        for query in qby_model[model_id]:
            scene_id = query["scene_id"]
            source = manifest_scenes[scene_id]
            query_group = str(query["source_group"])
            manifest_group = source.get("source_group") or source.get("bootstrap_cluster")
            if manifest_group is not None and str(manifest_group) != query_group:
                raise DataError(f"Source group mismatch: {model_id}/{scene_id}")
            source_sha = str(source.get("sha256", ""))
            if not source_sha:
                raise DataError(f"Missing source SHA: {task}/{scene_id}")
            entry_name = Path(str(source["path"])).stem + ".json"
            if entry_name not in entries:
                entry_name = scene_id + ".json"
            entry_path = result_dir / entry_name
            if not entry_path.is_file() or entry_path.is_symlink():
                raise DataError(f"Missing scene receipt: {entry_path}")
            entry_sha = sha(entry_path)
            if entries.get(entry_name) != entry_sha:
                raise DataError(f"Scene receipt hash mismatch: {entry_path}")
            scene_receipt_hash_rows.append((model_id, scene_id, entry_name, entry_sha))
            scene = read_json(entry_path)
            if (scene.get("model_id") != model_id or scene.get("scene_id") != scene_id
                    or scene.get("source_sha256") != source_sha):
                raise DataError(f"Scene/source identity mismatch: {model_id}/{scene_id}")

            k, candidates = int(scene["conditions"]), int(scene["candidates"])
            if k != 2 or candidates != int(query["K"]) or len(scene["horizons"]) != HORIZONS:
                raise DataError(f"Unexpected scene dimensions: {model_id}/{scene_id}")
            normalizer = float(k * candidates * HORIZONS)
            free = scene["modes"]["free"]
            receipt_e = np.asarray(free["prediction_error_sum"], dtype=np.float64)
            receipt_r = np.asarray(free["response_error_sum"], dtype=np.float64)
            receipt_b = np.asarray(scene["target_separation_energy"], dtype=np.float64)
            receipt_w = np.asarray(scene["wrong_allother"]["energy_sum"], dtype=np.float64)
            if any(x.shape != (HORIZONS,) for x in (receipt_e, receipt_r, receipt_b, receipt_w)):
                raise DataError(f"Bad energy vector: {model_id}/{scene_id}")
            check_close(residuals, f"{family}_receipt_E_vs_matched", receipt_e, scene["matched"]["energy_sum"])

            if family == "lewm":
                feature_name = str(scene["features_file"])
                feature_path = result_dir / feature_name
                if not feature_path.is_file() or feature_path.is_symlink() or sha(feature_path) != scene["features_file_sha256"]:
                    raise DataError(f"LeWM native feature cache hash mismatch: {feature_path}")
                lewm_feature_hash_rows.append((model_id, scene_id, feature_name, sha(feature_path)))
                with np.load(feature_path, allow_pickle=False) as cached:
                    stats = native_sums(cached["pred"], cached["target"])
                e, w, r, c, b = (stats[key] for key in ("E", "W", "R", "C", "B"))
                check_close(residuals, "lewm_E_direct_vs_receipt", e, receipt_e)
                check_close(residuals, "lewm_R_direct_vs_receipt", r, receipt_r)
                check_close(residuals, "lewm_B_direct_vs_receipt", b, receipt_b)
                check_close(residuals, "lewm_W_direct_vs_receipt", w, receipt_w)
                check_close(residuals, "lewm_E_equals_R_plus_bias", e, r + c)
                scene_gain = stats["gain_num"] / stats["gain_den"]
                g_numerator += stats["gain_num"] / normalizer
                g_denominator += stats["gain_den"] / normalizer
                independent_count += 1
            elif family == "dinowm":
                # Use validated native receipt values; never score pooled features.
                e, w, r, b = receipt_e, receipt_w, receipt_r, receipt_b
                c = e - r
                scene_gain = number(query["canonical_G_all5"], "G") / 4.0
            else:
                raise DataError(f"Unexpected family: {family}")

            es, ws, rs, cs, bs = (float(x.sum(dtype=np.float64)) for x in (e, w, r, c, b))
            en, wn, rn, cn, bn = (x / normalizer for x in (es, ws, rs, cs, bs))
            check_close(residuals, f"{family}_query_E", en, number(query["canonical_Eall_norm"], "query E"))
            check_close(residuals, f"{family}_query_W", wn, number(query["canonical_Wall_norm"], "query W"))
            check_close(residuals, f"{family}_query_B", bn, number(query["canonical_Ball_norm"], "query B"))
            if bs > 0:
                scene_g = (ws - es) / bs
                scene_s = 100 * (1 - es / bs)
                check_close(residuals, f"{family}_query_G", scene_g, number(query["canonical_G_all5"], "query G"))
                check_close(residuals, f"{family}_query_S", scene_s, number(query["canonical_S_all5"], "query S"))
                check_close(residuals, f"{family}_G_equals_4Gain", scene_g, 4 * scene_gain)
            else:
                scene_g = ""
            group_totals[query_group] += [rn, cn, en, bn]
            e_total += en
            r_total += rn
            c_total += cn
            b_total += bn
            g_energy += wn - en
            new_queries[(model_id, scene_id)] = {
                "response_error_sum_all5": rs,
                "response_error_energy_per_condition_candidate_horizon": rn,
                "common_prediction_bias_sum_all5": cs,
                "common_bias_energy_per_condition_candidate_horizon": cn,
                "complete_error_sum_all5": es,
                "complete_error_energy_per_condition_candidate_horizon": en,
                "target_separation_energy_sum_all5": bs,
                "target_separation_energy_per_condition_candidate_horizon": bn,
                "G_all5_recomputed": scene_g,
                "Gain_full_native_or_receipt_identity": scene_gain,
                "G_minus_4Gain": scene_g - 4 * scene_gain if scene_g != "" else "",
                "verification_basis": "independent full-native feature recompute" if family == "lewm" else "validated native receipt values; Gain=G/4 identity",
            }

        groups = len(group_totals)
        if groups != int(summary["n_source_groups"]) or b_total <= 0:
            raise DataError(f"Source group count or model-level B invalid: {model_id}")
        nre, bias_ratio, e_ratio = r_total / b_total, c_total / b_total, e_total / b_total
        check_close(residuals, f"{family}_E_over_B_decomposition", e_ratio, nre + bias_ratio)
        g_value = g_energy / b_total
        if family == "lewm":
            gain = g_numerator / g_denominator
            check_close(residuals, "lewm_model_G_equals_4Gain", g_value, 4 * gain)
            check_close(residuals, "lewm_model_G_matches_v1", g_value, summary["point_estimates"]["canonical_all5"]["G"])
        else:
            gain = g_value / 4
            check_close(residuals, "dino_receipt_model_G_matches_v1", g_value, summary["point_estimates"]["canonical_all5"]["G"])
        if groups not in {128, 243, 256}:
            raise DataError(f"Unexpected number of source groups: {model_id}")
        ci = bootstrap_ci(group_totals, seed)
        decomp_by_model[model_id] = {
            "nre": nre,
            "common_bias_ratio": bias_ratio,
            "complete_error_ratio": e_ratio,
            "ci95": ci,
            "energy_sums_normalized_by_scene": {
                "response": r_total, "common_bias": c_total, "complete_error": e_total, "target_separation_B": b_total
            },
            "gain_identity": {"G": g_value, "Gain": gain, "G_minus_4Gain": g_value - 4 * gain},
            "n_scenes": SCENES,
            "n_source_groups": groups,
            "bootstrap": {"replicates": REPLICATES, "seed": seed, "sampling_unit": "source group; retain all member scenes"},
            "verification": {
                "basis": "independent full-native feature recompute" if family == "lewm" else "validated native receipt E/R/B; pooled features not scored",
                "independently_recomputed_scene_count": independent_count,
                "receipt_scene_count": SCENES,
                "full_DINO_independent_recompute_claimed": False,
            },
        }

    result = deepcopy(v1)
    for row in result["models"]:
        row["response_decomposition"] = decomp_by_model[row["model_id"]]
    result["response_decomposition"] = {
        "schema": "cw_metric_response_decomposition_v2",
        "scope": {"tasks": sorted(TASKS), "model_count": MODELS, "scene_receipts": MODELS * SCENES, "scenes_per_model": SCENES, "LeWM_full_native_recomputations": len(lewm_feature_hash_rows), "source_groups_by_task": {task: next(m["response_decomposition"]["n_source_groups"] for m in result["models"] if m["task"] == task) for task in sorted(TASKS)}, "all_candidates_including_zero_retained": True},
        "definitions": {"nre": "sum centered response error / sum target separation B", "common_bias_ratio": "sum(E-response error) / sum(B); LeWM independently checked from condition-mean prediction bias", "complete_error_ratio": "sum complete matched error E / sum(B) = NRE + common bias/B", "gain_identity": "two-condition G=(E_wrong-E_matched)/B = 4*Gain"},
        "aggregation": "Within each scene sum the two conditions, all candidates, and five horizons, then divide by 2*candidate_count*5. Sum these normalized energies across scenes before ratios. Model-level B is positive; no zero-B scene ratio is fabricated.",
        "bootstrap": {"replicates": REPLICATES, "seed": seed, "unit": "source group; all member scenes retained", "interval": "percentile 95% CI"},
        "verification_scope": {"LeWM": "512/512 full native pred/target feature caches independently recomputed; E, response, common bias, B, G and 4*Gain checked.", "DINO-WM": "1536/1536 native receipt values checked against v1 query summaries and current panel source identities; pooled features not loaded or scored; no full-DINO independent recomputation claimed.", "source_identity": "Per-scene source SHA matched by scene_id to the task panel manifest; top-level receipt panel hash matched that manifest.", "claim_boundary": "Supplementary cached-model diagnosis for root-cause analysis; not unified physical accuracy."},
        "maximum_absolute_residuals": dict(sorted(residuals.items())),
        "input_hashes": {
            "comparison_json": {"path": str(comparison_path), "sha256": sha(comparison_path)},
            "queries_csv_gz": {"path": str(queries_path), "sha256": sha(queries_path)},
            "models_json": {"path": str(models_path), "sha256": actual_models_sha},
            "panel_manifests": panel_hashes,
            "model_receipts": receipt_hashes,
            "scene_receipt_records": {"count": len(scene_receipt_hash_rows), "sha256_sorted_identity_digest": digest_rows(scene_receipt_hash_rows)},
            "LeWM_native_feature_caches": {"count": len(lewm_feature_hash_rows), "sha256_sorted_identity_digest": digest_rows(lewm_feature_hash_rows)},
        },
        "v1_values_preserved": True,
    }
    result["outputs_v2"] = {key: str(path) for key, path in outputs.items()}

    extra_fields = list(next(iter(new_queries.values())))
    qheader = query_fields + extra_fields
    merged_queries = []
    for row in queries:
        merged = dict(row)
        merged.update(new_queries[(row["id"], row["scene_id"])])
        merged_queries.append(merged)
    csv_fields = ["model_id", "task", "family", "regime", "n_scenes", "n_source_groups", "canonical_R", "canonical_S", "canonical_G", "nre", "nre_ci95_low", "nre_ci95_high", "common_bias_ratio", "common_bias_ci95_low", "common_bias_ci95_high", "complete_error_ratio", "complete_error_ci95_low", "complete_error_ci95_high", "Gain", "G_minus_4Gain", "verification_basis"]
    summary_csv = []
    for row in result["models"]:
        d, old = row["response_decomposition"], row["point_estimates"]["canonical_all5"]
        ci = d["ci95"]
        summary_csv.append({"model_id": row["model_id"], "task": row["task"], "family": row["family"], "regime": row["regime"], "n_scenes": d["n_scenes"], "n_source_groups": d["n_source_groups"], "canonical_R": old["R"], "canonical_S": old["S"], "canonical_G": old["G"], "nre": d["nre"], "nre_ci95_low": ci["nre"][0], "nre_ci95_high": ci["nre"][1], "common_bias_ratio": d["common_bias_ratio"], "common_bias_ci95_low": ci["common_bias_ratio"][0], "common_bias_ci95_high": ci["common_bias_ratio"][1], "complete_error_ratio": d["complete_error_ratio"], "complete_error_ci95_low": ci["complete_error_ratio"][0], "complete_error_ci95_high": ci["complete_error_ratio"][1], "Gain": d["gain_identity"]["Gain"], "G_minus_4Gain": d["gain_identity"]["G_minus_4Gain"], "verification_basis": d["verification"]["basis"]})

    # V2 is additive; preserve every original v1 field and all frozen values.
    for old, new in zip(v1["models"], result["models"]):
        if any(new.get(key) != value for key, value in old.items()):
            raise DataError("An original v1 model field changed")
    for key, value in v1.items():
        if key != "models" and result.get(key) != value:
            raise DataError(f"An original v1 top-level field changed: {key}")

    with outputs["json"].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with outputs["csv"].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(summary_csv)
    with outputs["queries"].open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
            writer = csv.DictWriter(text, fieldnames=qheader)
            writer.writeheader()
            writer.writerows(merged_queries)
            text.flush()
            text.detach()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root, args.comparison, args.queries, args.output_dir)
    print("outputs=" + json.dumps(result["outputs_v2"], sort_keys=True))
    print("task | model | NRE [95% CI] | common bias/B [95% CI] | E/B [95% CI]")
    for row in result["models"]:
        d, ci = row["response_decomposition"], row["response_decomposition"]["ci95"]
        print(f"{row['task']} | {row['family']} {row['regime']} | {d['nre']:.6f} [{ci['nre'][0]:.6f}, {ci['nre'][1]:.6f}] | {d['common_bias_ratio']:.6f} [{ci['common_bias_ratio'][0]:.6f}, {ci['common_bias_ratio'][1]:.6f}] | {d['complete_error_ratio']:.6f} [{ci['complete_error_ratio'][0]:.6f}, {ci['complete_error_ratio'][1]:.6f}]")
    print("max_abs_residuals=" + json.dumps(result["response_decomposition"]["maximum_absolute_residuals"], sort_keys=True))


if __name__ == "__main__":
    main()
