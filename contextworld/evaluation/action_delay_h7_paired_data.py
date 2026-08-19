from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel
from contextworld.synthesis.validator import validate_loader_mix

from . import action_delay_h7_data as base
from .action_delay import canonical_sha256
from .action_delay_h7_validation import file_sha256


GROUP = "action_delay_paired"
DELAYS = base.MULTI_DELAYS


def _configured_delays(config: dict[str, Any]) -> tuple[int, ...]:
    delays = tuple(
        int(value)
        for value in config["protocol"]["training_delay_values"]
    )
    if (
        not delays
        or delays != tuple(sorted(set(delays)))
        or delays[0] < 0
        or delays[-1] > base.MAXIMUM_DELAY_STEPS
    ):
        raise ValueError(
            "Paired H7 delays must be unique, sorted, and within "
            f"0..{base.MAXIMUM_DELAY_STEPS}: {delays}"
        )
    return delays


def _validate_paired_config(config: dict[str, Any]) -> dict[str, bool]:
    protocol = config["protocol"]
    delays = _configured_delays(config)
    checks = {
        "history_tokens_are_seven": int(protocol["history_tokens"])
        == base.HISTORY_TOKENS,
        "num_preds_is_one": int(protocol["num_preds"])
        == base.NUM_PREDS,
        "action_block_is_five": int(
            protocol["raw_steps_per_action_block"]
        )
        == base.ACTION_BLOCK,
        "episode_has_ten_model_frames": int(
            protocol["episode_model_frames"]
        )
        == base.EPISODE_MODEL_FRAMES,
        "episode_has_fifty_raw_rows": int(protocol["rows_per_episode"])
        == base.RAW_STEPS,
        "formal_clip_starts_at_zero": int(
            protocol["strict_training_clip_start_raw_step"]
        )
        == base.FORMAL_CLIP_STARTS[0],
        "formal_reader_keeps_one_clip": int(
            protocol["formal_training_clips_per_episode"]
        )
        == 1,
        "declared_delay_support_is_valid": bool(delays),
        "model_fields_are_pixels_and_action": tuple(
            protocol["model_visible_fields"]
        )
        == base.MODEL_KEYS,
    }
    for split in base.SPLITS:
        counts = config["counts"][split]
        shards = int(counts["shards"])
        pair_shards = int(counts["paired_shards"])
        episodes = int(counts["episodes_per_shard"])
        checks[f"{split}_shards_are_complete_bundles"] = (
            shards == pair_shards * len(delays)
        )
        checks[f"{split}_episodes_per_shard_are_160"] = episodes == 160
        checks[f"{split}_clip_count_is_exact"] = int(
            counts["clips"]
        ) == shards * episodes
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Invalid paired H7 data protocol: {failed}")
    return checks


def build_paired_shard_plans(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> list[base.ActionDelayH7ShardPlan]:
    """Build exact same-query delay bundles."""

    _validate_paired_config(config)
    delays = _configured_delays(config)
    output_root = resolve_contextworld_path(
        config["output_root"],
        repo_root=repo_root,
    )
    catalog_seed = int(config["catalog_seed"])
    shards: list[base.ActionDelayH7ShardPlan] = []
    for split in base.SPLITS:
        pair_shards = int(config["counts"][split]["paired_shards"])
        episodes_per_shard = int(
            config["counts"][split]["episodes_per_shard"]
        )
        for pair_shard_index in range(pair_shards):
            templates = tuple(
                base.training_template(
                    catalog_seed=catalog_seed,
                    split=split,
                    shard_index=pair_shard_index,
                    episode_index=episode_index,
                )
                for episode_index in range(episodes_per_shard)
            )
            for delay_index, delay_steps in enumerate(delays):
                shard_index = (
                    pair_shard_index * len(delays) + delay_index
                )
                episodes = tuple(
                    base.ActionDelayH7EpisodePlan(
                        template=template,
                        delay_steps=int(delay_steps),
                        split=split,
                        shard_index=shard_index,
                        episode_index=episode_index,
                    )
                    for episode_index, template in enumerate(templates)
                )
                fingerprint = canonical_sha256(
                    {
                        "benchmark": config["benchmark"],
                        "group": GROUP,
                        "split": split,
                        "pair_shard_index": pair_shard_index,
                        "delay_steps": int(delay_steps),
                        "episode_templates": [
                            asdict(value) for value in templates
                        ],
                    }
                )
                scenario_id = (
                    f"ad-h7-paired-{split}-p{pair_shard_index:03d}-"
                    f"d{delay_steps}-{fingerprint[:10]}"
                )
                shards.append(
                    base.ActionDelayH7ShardPlan(
                        group=GROUP,
                        split=split,
                        shard_index=shard_index,
                        delay_steps=int(delay_steps),
                        scenario_id=scenario_id,
                        fingerprint=fingerprint,
                        table_path=(
                            output_root
                            / "tables"
                            / split
                            / f"{scenario_id}.lance"
                        ),
                        episodes=episodes,
                    )
                )
    return shards


def _paired_key(
    shard: base.ActionDelayH7ShardPlan,
    *,
    delay_count: int,
) -> tuple[str, int]:
    return shard.split, int(shard.shard_index) // delay_count


def _delay_balance(
    shards: list[base.ActionDelayH7ShardPlan],
    *,
    delays: tuple[int, ...],
) -> dict[str, Any]:
    counts = {
        split: {
            delay: sum(
                len(shard.episodes)
                for shard in shards
                if shard.split == split
                and int(shard.delay_steps) == delay
            )
            for delay in delays
        }
        for split in base.SPLITS
    }
    expected = {
        split: {
            delay: sum(
                len(shard.episodes)
                for shard in shards
                if shard.split == split
            )
            // len(delays)
            for delay in delays
        }
        for split in base.SPLITS
    }
    return {
        "counts": {
            split: {
                str(delay): counts[split][delay] for delay in delays
            }
            for split in base.SPLITS
        },
        "expected": {
            split: {
                str(delay): expected[split][delay] for delay in delays
            }
            for split in base.SPLITS
        },
        "passed": counts == expected,
    }


def audit_paired_triplets(
    shards: list[base.ActionDelayH7ShardPlan],
    audits: list[dict[str, Any]],
    *,
    delays: tuple[int, ...] = DELAYS,
) -> dict[str, Any]:
    """Prove that only delay and its physical future differ per bundle."""

    grouped: dict[
        tuple[str, int],
        list[tuple[base.ActionDelayH7ShardPlan, dict[str, Any]]],
    ] = defaultdict(list)
    for shard, audit in zip(shards, audits, strict=True):
        grouped[
            _paired_key(shard, delay_count=len(delays))
        ].append((shard, audit))

    expected_delays = set(delays)
    complete = all(
        {shard.delay_steps for shard, _ in rows} == expected_delays
        and len(rows) == len(delays)
        for rows in grouped.values()
    )
    templates_exact = complete and all(
        all(
            [
                asdict(left.template) == asdict(right.template)
                for left, right in zip(
                    ordered[0][0].episodes,
                    candidate[0].episodes,
                    strict=True,
                )
            ]
        )
        for rows in grouped.values()
        for ordered in [
            sorted(rows, key=lambda row: row[0].delay_steps)
        ]
        for candidate in ordered[1:]
    )
    exact_fields = {}
    for field in (
        "initial_pixels_sha256",
        "query_pixels_sha256",
        "model_action_sha256",
    ):
        exact_fields[field] = complete and all(
            all(candidate[1][field] == ordered[0][1][field]
                for candidate in ordered[1:])
            for rows in grouped.values()
            for ordered in [
                sorted(rows, key=lambda row: row[0].delay_steps)
            ]
        )
    target_physical_groups_exact = complete and all(
        all(
            (
                left[1]["target_pixels_sha256"][episode_index]
                == right[1]["target_pixels_sha256"][episode_index]
            )
            == (
                min(
                    int(left[0].delay_steps),
                    base.ACTION_BLOCK,
                )
                == min(
                    int(right[0].delay_steps),
                    base.ACTION_BLOCK,
                )
            )
            for left_index, left in enumerate(ordered)
            for right in ordered[left_index + 1 :]
            for episode_index in range(len(ordered[0][0].episodes))
        )
        for rows in grouped.values()
        for ordered in [
            sorted(rows, key=lambda row: row[0].delay_steps)
        ]
    )
    checks = {
        "every_bundle_contains_declared_delays": complete,
        "template_geometry_and_simulator_seed_exact": templates_exact,
        "initial_pixels_exact": exact_fields[
            "initial_pixels_sha256"
        ],
        "query_pixels_exact": exact_fields["query_pixels_sha256"],
        "commanded_actions_exact": exact_fields[
            "model_action_sha256"
        ],
        "next_frame_physical_equivalence_groups_exact": (
            target_physical_groups_exact
        ),
    }
    return {
        "pair_shards": len(grouped),
        "query_triplets": sum(
            len(rows[0][0].episodes) for rows in grouped.values()
        ),
        "query_bundles": sum(
            len(rows[0][0].episodes) for rows in grouped.values()
        ),
        "physical_clips": sum(
            len(shard.episodes) for shard in shards
        ),
        "delays_per_query": list(delays),
        "physical_next_state_groups": sorted(
            {min(delay, base.ACTION_BLOCK) for delay in delays}
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _manifest_record(
    shard: base.ActionDelayH7ShardPlan,
    *,
    config: dict[str, Any],
    repo_root: Path,
    audit: dict[str, Any],
    stable_commit: str,
    collection_status: str,
    delays: tuple[int, ...],
) -> dict[str, Any]:
    pair_shard_index = int(shard.shard_index) // len(delays)
    record = base._manifest_record(
        shard,
        config=config,
        repo_root=repo_root,
        audit=audit,
        stable_commit=stable_commit,
        collection_status=collection_status,
    )
    record.update(
        {
            "pair_shard_index": pair_shard_index,
            "pair_id": (
                f"action-delay-h7-paired-{shard.split}-"
                f"p{pair_shard_index:03d}"
            ),
            "seed_group": (
                f"action-delay-h7-paired-{shard.split}-"
                f"p{pair_shard_index:03d}"
            ),
            "regime": (
                "train_action_delay_history7_paired"
                if shard.split == "train"
                else "validation_action_delay_history7_paired"
            ),
        }
    )
    return record


def build_paired_training_release(
    *,
    config: dict[str, Any],
    repo_root: Path,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    protocol_checks = _validate_paired_config(config)
    delays = _configured_delays(config)
    if workers < 1:
        raise ValueError("workers must be positive")
    output_root = resolve_contextworld_path(
        config["output_root"],
        repo_root=repo_root,
    )
    output_root.mkdir(parents=True, exist_ok=resume)
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        repo_root,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    shards = build_paired_shard_plans(config, repo_root=repo_root)
    missing = [shard for shard in shards if not shard.table_path.exists()]
    if missing and not resume and len(missing) != len(shards):
        raise FileExistsError(
            "Paired H7 release is partially populated; use --resume"
        )
    base._collect_missing(
        swm,
        missing=missing,
        config=config,
        repo_root=repo_root,
        workers=workers,
    )
    audits = base._audit_group(
        swm,
        shards=shards,
        config=config,
        repo_root=repo_root,
        workers=workers,
    )
    failed = [
        audit["scenario_id"] for audit in audits if not audit["passed"]
    ]
    if failed:
        raise RuntimeError(f"Paired H7 shard audits failed: {failed}")

    records = [
        _manifest_record(
            shard,
            config=config,
            repo_root=repo_root,
            audit=audit,
            stable_commit=stable_commit,
            # Describe how the release artifact originated, not whether the
            # current audit invocation happened to reuse its bytes.
            collection_status="collected",
            delays=delays,
        )
        for shard, audit in zip(shards, audits, strict=True)
    ]
    stem = str(config["catalog_stem"])
    catalog_path = output_root / "catalogs" / f"{stem}.json"
    manifest_path = output_root / "manifests" / f"{stem}.jsonl"
    report_path = output_root / "reports" / f"{stem}.json"
    base._write_jsonl(manifest_path, records)
    catalog = base._catalog_payload(
        config=config,
        group=GROUP,
        records=records,
    )
    catalog["delay_support"] = list(delays)
    catalog["pairing_contract"] = {
        "same_query_bundle_delays": list(delays),
        # Retain the v1 field only for the original three-delay release.
        **(
            {"same_query_triplet_delays": list(delays)}
            if len(delays) == 3
            else {}
        ),
        "pair_id_manifest_field": "pair_id",
        "model_visible_fields": list(base.MODEL_KEYS),
        "physical_next_state_group": (
            f"min(delay_steps, {base.ACTION_BLOCK})"
        ),
    }
    write_json(catalog_path, catalog)

    loader = validate_loader_mix(
        swm,
        original_dataset=resolve_contextworld_path(
            config["original_dataset"]["path"],
            repo_root=repo_root,
        ),
        synthetic_dataset=shards[0].table_path,
        cache_dir=Path(
            f"/tmp/contextworld-action-delay-h7-{stem}-loader"
        ),
    )
    pair_audit = audit_paired_triplets(
        shards,
        audits,
        delays=delays,
    )
    representative = [
        (shard, audit)
        for shard, audit in zip(shards, audits, strict=True)
        if shard.delay_steps == delays[0]
    ]
    isolation = base._isolation_audit(
        config=config,
        repo_root=repo_root,
        plans=[row[0] for row in representative],
        audits=[row[1] for row in representative],
    )
    balance = _delay_balance(shards, delays=delays)
    observed_counts = {
        split: {
            "shards": sum(shard.split == split for shard in shards),
            "paired_shards": sum(
                shard.split == split and shard.delay_steps == delays[0]
                for shard in shards
            ),
            "query_bundles": sum(
                len(shard.episodes)
                for shard in shards
                if shard.split == split
                and shard.delay_steps == delays[0]
            ),
            "formal_model_clips": sum(
                audit["formal_model_clips"]
                for shard, audit in zip(shards, audits, strict=True)
                if shard.split == split
            ),
        }
        for split in base.SPLITS
    }
    expected_counts = {
        split: {
            "shards": int(config["counts"][split]["shards"]),
            "paired_shards": int(
                config["counts"][split]["paired_shards"]
            ),
            "query_bundles": (
                int(config["counts"][split]["paired_shards"])
                * int(config["counts"][split]["episodes_per_shard"])
            ),
            "formal_model_clips": int(
                config["counts"][split]["clips"]
            ),
        }
        for split in base.SPLITS
    }
    group_checks = {
        "loader_compatible": bool(loader["passed"]),
        "all_shards_pass_raw_physical_replay": all(
            audit["passed"] for audit in audits
        ),
        "same_query_bundles_exact": pair_audit["passed"],
        "delay_counts_exactly_balanced": balance["passed"],
        "validation_and_split_isolation_pass": isolation["passed"],
        "counts_exact": observed_counts == expected_counts,
    }
    group_report = {
        "schema_version": 1,
        "experiment": stem,
        "benchmark": config["benchmark"],
        "group": GROUP,
        "compile_only": False,
        "preflight_passed": bool(loader["passed"]),
        "passed": all(group_checks.values()),
        "checks": group_checks,
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "catalog": str(catalog_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "collection_status": {
            record["scenario_id"]: record["collection_status"]
            for record in records
        },
        "loader_compatibility": loader,
        "scenarios": [
            {
                "scenario_id": audit["scenario_id"],
                "passed": audit["passed"],
                "checks": audit["checks"],
            }
            for audit in audits
        ],
        "pairing": pair_audit,
        "delay_balance": balance,
        "isolation": isolation,
        "counts": observed_counts,
    }
    write_json(report_path, group_report)
    artifacts = {
        "catalog": portable_contextworld_path(
            catalog_path,
            repo_root=repo_root,
        ),
        "catalog_sha256": file_sha256(catalog_path),
        "manifest": portable_contextworld_path(
            manifest_path,
            repo_root=repo_root,
        ),
        "manifest_sha256": file_sha256(manifest_path),
        "synthesis_report": portable_contextworld_path(
            report_path,
            repo_root=repo_root,
        ),
        "synthesis_report_sha256": file_sha256(report_path),
    }
    checks = {
        **protocol_checks,
        **group_checks,
        "catalog_manifest_and_report_published": all(
            Path(path).is_file()
            for path in (catalog_path, manifest_path, report_path)
        ),
        "model_projection_is_pixels_and_actions_only": (
            base.MODEL_KEYS == ("pixels", "action")
        ),
    }
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "scope": (
            "same_query_delay_bundles_with_full_physical_replay; "
            "no_model_training"
        ),
        "checks": checks,
        "identity": {
            "stable_worldmodel_repo": str(stable_repo),
            "stable_worldmodel_commit": stable_commit,
        },
        "temporal_contract": catalog["temporal_contract"],
        "pairing": pair_audit,
        "delay_balance": balance,
        "isolation": isolation,
        "counts": observed_counts,
        "artifacts": artifacts,
        "physical_counts": {
            "shards": len(shards),
            "query_bundles": pair_audit["query_bundles"],
            "episodes": sum(len(shard.episodes) for shard in shards),
            "raw_rows_replayed": sum(
                len(shard.episodes) * base.RAW_STEPS
                for shard in shards
            ),
            "formal_model_clips": sum(
                audit["formal_model_clips"] for audit in audits
            ),
            "misaligned_default_clips_rejected": sum(
                audit["clip_filter"]["removed_misaligned_clips"]
                for audit in audits
            ),
        },
    }
    write_json(output_root / "build_report.json", report)
    return report


__all__ = [
    "DELAYS",
    "GROUP",
    "audit_paired_triplets",
    "build_paired_shard_plans",
    "build_paired_training_release",
]
