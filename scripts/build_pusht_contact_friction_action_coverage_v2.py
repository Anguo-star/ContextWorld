#!/usr/bin/env python3
"""Build contact-friction Training v2 with balanced query-action coverage.

The 2,048-pair count and both evaluation splits stay unchanged.  Only the
Training query action is scaled across nine preregistered magnitudes so the
model can learn how friction and action magnitude jointly affect the future.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.pusht_contact_friction_h3 import (
    ENDPOINT_MODES,
    ContactFrictionTemplate,
    array_sha256,
    simulate_contact_friction_clip,
    validate_contact_friction_pair,
)
from build_pusht_contact_friction_h3_data import (
    _cross_split_audit,
    _episode_rows,
    _orientation_bin,
    _position_bin,
    _write_lance_episodes,
    canonical_json_sha256,
    directory_sha256,
    file_sha256,
    safe_output_path,
)


PROTOCOL = "pusht_contact_friction_history3_action_coverage_v2"
DEFAULT_SOURCE = (
    ROOT / "artifacts/synthesis/pusht_contact_friction_h3_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/synthesis/"
    "pusht_contact_friction_h3_action_coverage_v2"
)
QUERY_SCALES = (
    0.500,
    0.625,
    0.750,
    0.875,
    1.000,
    1.125,
    1.250,
    1.375,
    1.500,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def _build_training(
    *,
    root: Path,
    source_manifest: dict,
    resolution: int,
    jpeg_quality: int,
) -> dict:
    source = source_manifest["splits"]["train"]
    source_pairs = source["pairs"]
    pair_reports = []
    query_hashes = set()
    action_hashes = set()
    template_ids = set()
    scale_counts = {str(value): 0 for value in QUERY_SCALES}
    started = time.monotonic()

    def episodes():
        for pair_index, source_pair in enumerate(source_pairs):
            source_template = ContactFrictionTemplate(
                **source_pair["template"]
            )
            scale = QUERY_SCALES[pair_index % len(QUERY_SCALES)]
            scaled_query = (
                np.asarray(source_template.query_actions, dtype=np.float64)
                * scale
            )
            template = replace(
                source_template,
                query_actions=tuple(map(tuple, scaled_query.tolist())),
            )
            low = simulate_contact_friction_clip(
                template,
                mode=ENDPOINT_MODES[0],
                resolution=resolution,
            )
            high = simulate_contact_friction_clip(
                template,
                mode=ENDPOINT_MODES[1],
                resolution=resolution,
            )
            audit = validate_contact_friction_pair(low, high)
            if not audit["passed"]:
                failed = [
                    key
                    for key, passed in audit["checks"].items()
                    if not passed
                ]
                raise RuntimeError(
                    f"Scaled training pair {pair_index} failed: {failed}"
                )
            query_hash = audit["hashes"]["query_pixels"]
            if query_hash in query_hashes:
                raise RuntimeError(
                    f"Duplicate query pixels at pair {pair_index}"
                )
            query_hashes.add(query_hash)
            action_hashes.add(audit["hashes"]["raw_actions"])
            template_ids.add(template.template_id)
            scale_counts[str(scale)] += 1
            template_payload = asdict(template)
            pair_reports.append(
                {
                    "pair_index": pair_index,
                    "catalog_index": source_pair["catalog_index"],
                    "template": template_payload,
                    "query_action_scale": scale,
                    "orientation_bin": _orientation_bin(template_payload),
                    "position_bin": _position_bin(template_payload),
                    "audit": audit,
                }
            )
            yield _episode_rows(
                low,
                split="train",
                catalog_index=source_pair["catalog_index"],
            )
            yield _episode_rows(
                high,
                split="train",
                catalog_index=source_pair["catalog_index"],
            )
            if (
                len(pair_reports) <= 4
                or len(pair_reports) % 64 == 0
                or len(pair_reports) == len(source_pairs)
            ):
                print(
                    f"training {len(pair_reports)}/{len(source_pairs)} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

    table_path = root / "train.lance"
    _write_lance_episodes(
        table_path,
        episodes(),
        jpeg_quality=jpeg_quality,
    )
    corrections = [
        value
        for pair in pair_reports
        for value in pair["audit"][
            "query_precanonical_correction"
        ].values()
    ]
    history_gaps = [
        pair["audit"]["history_visible_response_gap"]["px_equivalent"]
        for pair in pair_reports
    ]
    future_gaps = [
        pair["audit"]["future_gap"]["block_position_px"]
        for pair in pair_reports
    ]
    orientation_counts = {
        str(index): sum(
            row["orientation_bin"] == index for row in pair_reports
        )
        for index in range(8)
    }
    position_counts = {}
    for row in pair_reports:
        key = row["position_bin"]
        position_counts[key] = position_counts.get(key, 0) + 1
    table_files = [
        path for path in table_path.rglob("*") if path.is_file()
    ]
    return {
        "split": "train",
        "pair_count": len(pair_reports),
        "episode_count": 2 * len(pair_reports),
        "raw_rows": 2 * len(pair_reports) * 20,
        "table_path": "train.lance",
        "table_files": len(table_files),
        "table_bytes": sum(path.stat().st_size for path in table_files),
        "table_sha256": directory_sha256(table_path),
        "catalog_seed": source["catalog_seed"],
        "catalog_attempts": len(pair_reports),
        "accepted_catalog_indices_sha256": (
            source["accepted_catalog_indices_sha256"]
        ),
        "failure_counts": {},
        "query_hash_count": len(query_hashes),
        "query_hashes": sorted(query_hashes),
        "action_hash_count": len(action_hashes),
        "action_hashes": sorted(action_hashes),
        "template_ids": sorted(template_ids),
        "orientation_bin_counts": orientation_counts,
        "position_bin_counts": dict(sorted(position_counts.items())),
        "query_action_scale_grid": list(QUERY_SCALES),
        "query_action_scale_counts": scale_counts,
        "minimum_history_gap_px_equivalent": float(min(history_gaps)),
        "maximum_query_correction": float(max(corrections)),
        "minimum_future_block_position_gap_px": float(min(future_gaps)),
        "pairs": pair_reports,
        "passed": bool(
            len(pair_reports) == len(source_pairs)
            and all(row["audit"]["passed"] for row in pair_reports)
            and all(value > 0 for value in orientation_counts.values())
            and max(scale_counts.values()) - min(scale_counts.values()) <= 1
        ),
    }


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = safe_output_path(args.output)
    source_manifest_path = source / "manifest.json"
    required = (
        source_manifest_path,
        source / "loader_validation.lance",
        source / "validation.lance",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing source artifact(s):\n" + "\n".join(map(str, missing))
        )
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    request = {
        "protocol": PROTOCOL,
        "source_protocol": source_manifest["protocol"],
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "pair_counts": source_manifest["pair_counts"],
        "history_tokens": 3,
        "raw_steps_per_action_block": 5,
        "training_query_action_scale_grid": list(QUERY_SCALES),
        "loader_validation_query_action_scale": 1.0,
        "validation_query_action_scale": 1.0,
        "resolution": int(args.resolution),
        "jpeg_quality": int(args.jpeg_quality),
        "sealed_test_included": False,
    }
    with tempfile.TemporaryDirectory(
        prefix="pusht-contact-friction-action-coverage-",
        dir="/tmp",
    ) as temporary:
        root = Path(temporary) / output.name
        root.mkdir()
        (root / "request.json").write_text(
            json.dumps(request, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reports = {
            "train": _build_training(
                root=root,
                source_manifest=source_manifest,
                resolution=int(args.resolution),
                jpeg_quality=int(args.jpeg_quality),
            ),
            "loader_validation": source_manifest["splits"][
                "loader_validation"
            ],
            "validation": source_manifest["splits"]["validation"],
        }
        shutil.copytree(
            source / "loader_validation.lance",
            root / "loader_validation.lance",
        )
        shutil.copytree(
            source / "validation.lance",
            root / "validation.lance",
        )
        cross_split = _cross_split_audit(reports)
        if not cross_split["passed"]:
            raise RuntimeError(
                f"Cross-split isolation failed: {cross_split}"
            )
        manifest = {
            **request,
            "friction_values": source_manifest["friction_values"],
            "catalog_seed": source_manifest["catalog_seed"],
            "split_catalog_seeds": source_manifest[
                "split_catalog_seeds"
            ],
            "request_sha256": canonical_json_sha256(request),
            "splits": reports,
            "cross_split_audit": cross_split,
            "passed": bool(
                cross_split["passed"]
                and all(report["passed"] for report in reports.values())
            ),
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = {
            "protocol": PROTOCOL,
            "status": "passed" if manifest["passed"] else "failed",
            "root": str(output),
            "manifest": "manifest.json",
            "manifest_sha256": file_sha256(manifest_path),
            "pair_counts": request["pair_counts"],
            "training_query_action_scale_counts": reports["train"][
                "query_action_scale_counts"
            ],
            "cross_split_audit": cross_split,
            "split_metrics": {
                split: {
                    key: reports[split][key]
                    for key in (
                        "minimum_history_gap_px_equivalent",
                        "maximum_query_correction",
                        "minimum_future_block_position_gap_px",
                    )
                }
                for split in ("train", "loader_validation", "validation")
            },
            "passed": manifest["passed"],
        }
        (root / "build_report.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copytree(root, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
