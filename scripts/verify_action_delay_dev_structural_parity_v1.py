#!/usr/bin/env python3
"""Independent verification of the Action Delay H7 Development structural-parity release.

This checks three frozen/in-flight artifacts against each other and against
the benchmark's own rule (Development and Public Test must share identical
evaluation structure and have zero data-row overlap):

  * Development release  : artifacts/evaluation/history7/action_delay_dev_structural_parity_v1/
  * Public Test release  : artifacts/evaluation/history7/action_delay_validation_v1/
  * Newly exported Lance : --lance-root (66 tables; optional, may not exist yet)

Deliberate independence: this script does NOT read dev_disjointness_audit.json,
training_exclusion_manifest.json, build_report.json, or any *_build_report.json
self-report, and it does NOT import or invoke
scripts/export_tworoom_action_delay_dev_structural_parity_lance_v1.py,
scripts/build_tworoom_action_delay_h7_dev_structural_parity_v1.py, or
scripts/audit_tworoom_action_delay_h7_dev_disjointness_v1.py. Every number
below is recomputed from the catalogs / assets / Lance tables directly. The
only reused code is generic, pre-existing hashing/path utilities
(contextworld.evaluation.action_delay.array_sha256 and
contextworld.paths.resolve_contextworld_path) that both the frozen Test and
the Development builder already depend on -- reusing them is what lets a
recomputed hash mean the same thing the catalog says it means.

Section D (Lance export) is independently derived from:
  * the 16-column schema and "ad-h7-paired-val-p<PPP>-d<D>-*.lance" naming
    already frozen in the legacy development/full payload,
  * the dev_query_id / dev_* stratification-column convention already
    established (and frozen) for the door and speed structural-parity
    payloads in contextworld/benchmarks/bundle_development.py, and
  * the npz asset shapes themselves: history_pixels is (11 delays, 7 history
    tokens, H, W, 3) and true_future_pixels is (11 delays, 3 future horizons,
    H, W, 3) -- i.e. 10 block-boundary frames per delay condition (steps
    0,5,...,30 from history_pixels, steps 35,40,45 from true_future_pixels),
    matching the exporter's own docstring claim of "10 block-boundary frames"
    without relying on that docstring being true.

Run with the frozen eval venv (system python's torch build will not import):
    <cwenv>/bin/python scripts/verify_action_delay_dev_structural_parity_v1.py \
        [--lance-root DIR] [--sample-episodes N] [--seed N]

Exits non-zero if any check fails. Section D is SKIPPED (not a failure) when
--lance-root is omitted or does not yet exist.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay import array_sha256  # noqa: E402
from contextworld.paths import resolve_contextworld_path  # noqa: E402

DEV_RELEASE = ROOT / "artifacts/evaluation/history7/action_delay_dev_structural_parity_v1"
TEST_RELEASE = ROOT / "artifacts/evaluation/history7/action_delay_validation_v1"

# The legacy 66-table payload this release supersedes.  Per
# scripts/register_action_delay_dev_structural_parity_payload_v1.py this
# directory is slated to be moved to development_legacy_v1 once the new
# payload is registered; check both locations so this script keeps working
# whether or not that registration has happened yet.
LEGACY_REFERENCE_TABLE_CANDIDATES = [
    Path(
        "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/ContextWorld-v1"
        "/components/tworoom-action-delay/v1/development/full"
        "/ad-h7-paired-val-p000-d0-12fb796bc9.lance"
    ),
    Path(
        "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/ContextWorld-v1"
        "/components/tworoom-action-delay/v1/development_legacy_v1"
        "/ad-h7-paired-val-p000-d0-12fb796bc9.lance"
    ),
]

EXPECTED_DELAYS = list(range(11))
EXPECTED_TEST_SEEDS = [42, 43, 44, 45, 46, 47]
EXPECTED_DEV_SEEDS = [52, 53, 54, 55, 56, 57]
EXPECTED_STATUS = "frozen_before_model_scoring"
EXPECTED_COUNTS_KEYS = (
    "delay_conditions_per_query",
    "distinct_queries",
    "history_conditioned_model_predictions_per_checkpoint",
    "aggregate_three_step_target_comparisons_per_checkpoint",
)

TABLE_NAME_RE = re.compile(r"^ad-h7-paired-val-p(\d{3})-d(\d{1,2})-[0-9a-fA-F]+\.lance$")
EPISODE_ROWS = 50
BLOCK_BOUNDARY_ROWS = tuple(range(0, EPISODE_ROWS, 5))  # 0,5,...,45 (10 boundaries)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []
        self.any_fail = False

    def add(self, section: str, check: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        if not ok:
            self.any_fail = True
        self.rows.append((section, check, status, detail))

    def skip(self, section: str, check: str, detail: str = "") -> None:
        self.rows.append((section, check, "SKIP", detail))

    def error(self, section: str, check: str, detail: str = "") -> None:
        self.any_fail = True
        self.rows.append((section, check, "ERROR", detail))

    def render(self) -> str:
        check_w = max([len("CHECK")] + [len(r[1]) for r in self.rows])
        lines = []
        header = f"{'SEC':<4} {'CHECK':<{check_w}} {'STATUS':<6} DETAIL"
        lines.append(header)
        lines.append("-" * len(header))
        for section, check, status, detail in self.rows:
            lines.append(f"{section:<4} {check:<{check_w}} {status:<6} {detail}")
        n_pass = sum(1 for r in self.rows if r[2] == "PASS")
        n_fail = sum(1 for r in self.rows if r[2] == "FAIL")
        n_error = sum(1 for r in self.rows if r[2] == "ERROR")
        n_skip = sum(1 for r in self.rows if r[2] == "SKIP")
        lines.append("-" * len(header))
        lines.append(
            f"TOTAL: {n_pass} passed, {n_fail} failed, {n_error} errored, "
            f"{n_skip} skipped"
        )
        lines.append("OVERALL: " + ("FAIL" if self.any_fail else "PASS"))
        return "\n".join(lines)


def _fmt_examples(values: Iterable[Any], limit: int = 5) -> str:
    values = sorted(values, key=lambda v: str(v))
    shown = values[:limit]
    suffix = f" (+{len(values) - limit} more)" if len(values) > limit else ""
    return f"{shown}{suffix}"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Section A: data-row disjointness against Public Test
# --------------------------------------------------------------------------


def _start_tuple(query: dict[str, Any]) -> tuple[tuple[float, ...], str, str]:
    template = query["template"]
    return (tuple(template["reset_state"]), query["room"], query["direction"])


A_FIELDS: list[tuple[str, Callable[[dict[str, Any]], Any]]] = [
    ("template_sha256", lambda q: q["template_sha256"]),
    ("initial_pixels_sha256", lambda q: q["initial_pixels_sha256"]),
    ("query_pixels_sha256", lambda q: q["query_pixels_sha256"]),
    ("payload_sha256", lambda q: q["payload_sha256"]),
    ("asset_sha256", lambda q: q["asset_sha256"]),
    ("query_id", lambda q: q["query_id"]),
    ("(reset_state, room, direction)", _start_tuple),
]


def section_a_disjointness(
    report: Report, dev_queries: list[dict[str, Any]], test_queries: list[dict[str, Any]]
) -> None:
    for name, extractor in A_FIELDS:
        dev_values = {extractor(q) for q in dev_queries}
        test_values = {extractor(q) for q in test_queries}
        intersection = dev_values & test_values
        detail = f"dev_distinct={len(dev_values)} test_distinct={len(test_values)} intersection={len(intersection)}"
        if intersection:
            detail += f" examples={_fmt_examples(intersection)}"
        report.add("A", f"{name} disjoint", len(intersection) == 0, detail)

    # simulator_seed, "if present"
    dev_has = all("simulator_seed" in q["template"] for q in dev_queries)
    test_has = all("simulator_seed" in q["template"] for q in test_queries)
    if dev_has and test_has:
        dev_seeds = {q["template"]["simulator_seed"] for q in dev_queries}
        test_seeds = {q["template"]["simulator_seed"] for q in test_queries}
        intersection = dev_seeds & test_seeds
        detail = (
            f"dev_distinct={len(dev_seeds)} test_distinct={len(test_seeds)} "
            f"intersection={len(intersection)}"
        )
        if intersection:
            detail += f" examples={_fmt_examples(intersection)}"
        report.add("A", "simulator_seed disjoint", len(intersection) == 0, detail)
    else:
        report.skip(
            "A",
            "simulator_seed disjoint",
            f"simulator_seed not present on every query (dev_all_present={dev_has}, "
            f"test_all_present={test_has})",
        )


# --------------------------------------------------------------------------
# Section B: structural parity with Public Test
# --------------------------------------------------------------------------


def _counter(values: Iterable[Any]) -> Counter:
    return Counter(values)


def section_b_structural_parity(
    report: Report, dev_catalog: dict[str, Any], test_catalog: dict[str, Any]
) -> None:
    dev_queries = dev_catalog["queries"]
    test_queries = test_catalog["queries"]

    # 300 distinct queries on both sides.
    dev_ids = [q["query_id"] for q in dev_queries]
    test_ids = [q["query_id"] for q in test_queries]
    dev_distinct = len(set(dev_ids))
    test_distinct = len(set(test_ids))
    report.add(
        "B",
        "300 distinct queries",
        dev_distinct == 300 and test_distinct == 300 and len(dev_ids) == 300 and len(test_ids) == 300,
        f"dev_rows={len(dev_ids)} dev_distinct={dev_distinct} "
        f"test_rows={len(test_ids)} test_distinct={test_distinct}",
    )

    # protocol.delay_values identical (both [0..10])
    dev_delays = dev_catalog["protocol"]["delay_values"]
    test_delays = test_catalog["protocol"]["delay_values"]
    report.add(
        "B",
        "protocol.delay_values == [0..10] on both sides",
        dev_delays == EXPECTED_DELAYS and test_delays == EXPECTED_DELAYS and dev_delays == test_delays,
        f"dev={dev_delays} test={test_delays}",
    )

    # 6 eval seeds x 50 queries per seed (Test 42-47, Dev 52-57).
    dev_seed_counts = _counter(q["eval_seed"] for q in dev_queries)
    test_seed_counts = _counter(q["eval_seed"] for q in test_queries)
    dev_seeds_ok = sorted(dev_seed_counts) == EXPECTED_DEV_SEEDS and all(
        c == 50 for c in dev_seed_counts.values()
    )
    test_seeds_ok = sorted(test_seed_counts) == EXPECTED_TEST_SEEDS and all(
        c == 50 for c in test_seed_counts.values()
    )
    report.add(
        "B",
        "6 eval seeds x 50 queries/seed",
        dev_seeds_ok and test_seeds_ok,
        f"dev={dict(sorted(dev_seed_counts.items()))} test={dict(sorted(test_seed_counts.items()))}",
    )

    # room split 150/150
    dev_room_counts = _counter(q["room"] for q in dev_queries)
    test_room_counts = _counter(q["room"] for q in test_queries)
    dev_room_ok = len(dev_room_counts) == 2 and set(dev_room_counts.values()) == {150}
    test_room_ok = len(test_room_counts) == 2 and set(test_room_counts.values()) == {150}
    report.add(
        "B",
        "room split 150/150",
        dev_room_ok and test_room_ok,
        f"dev={dict(dev_room_counts)} test={dict(test_room_counts)}",
    )

    # direction split 150/150
    dev_dir_counts = _counter(q["direction"] for q in dev_queries)
    test_dir_counts = _counter(q["direction"] for q in test_queries)
    dev_dir_ok = len(dev_dir_counts) == 2 and set(dev_dir_counts.values()) == {150}
    test_dir_ok = len(test_dir_counts) == 2 and set(test_dir_counts.values()) == {150}
    report.add(
        "B",
        "direction split 150/150",
        dev_dir_ok and test_dir_ok,
        f"dev={dict(dev_dir_counts)} test={dict(test_dir_counts)}",
    )

    # per-(eval_seed, room) = 25/25
    def _per_seed_pair_ok(queries: list[dict[str, Any]], key: str) -> tuple[bool, dict[str, dict[Any, int]]]:
        by_seed: dict[Any, Counter] = defaultdict(Counter)
        for q in queries:
            by_seed[q["eval_seed"]][q[key]] += 1
        ok = all(
            len(counts) == 2 and set(counts.values()) == {25} for counts in by_seed.values()
        )
        return ok, {str(seed): dict(counts) for seed, counts in sorted(by_seed.items())}

    dev_room_seed_ok, dev_room_seed_detail = _per_seed_pair_ok(dev_queries, "room")
    test_room_seed_ok, test_room_seed_detail = _per_seed_pair_ok(test_queries, "room")
    report.add(
        "B",
        "per-(eval_seed, room) = 25/25",
        dev_room_seed_ok and test_room_seed_ok,
        f"dev={dev_room_seed_detail} test={test_room_seed_detail}",
    )

    dev_dir_seed_ok, dev_dir_seed_detail = _per_seed_pair_ok(dev_queries, "direction")
    test_dir_seed_ok, test_dir_seed_detail = _per_seed_pair_ok(test_queries, "direction")
    report.add(
        "B",
        "per-(eval_seed, direction) = 25/25",
        dev_dir_seed_ok and test_dir_seed_ok,
        f"dev={dev_dir_seed_detail} test={test_dir_seed_detail}",
    )

    # counts block: four required keys must be identical.
    dev_counts = dev_catalog["counts"]
    test_counts = test_catalog["counts"]
    for key in EXPECTED_COUNTS_KEYS:
        dev_value = dev_counts.get(key, "<missing>")
        test_value = test_counts.get(key, "<missing>")
        report.add(
            "B",
            f"counts.{key} identical",
            key in dev_counts and key in test_counts and dev_value == test_value,
            f"dev={dev_value} test={test_value}",
        )

    # status
    dev_status = dev_catalog.get("status")
    test_status = test_catalog.get("status")
    report.add(
        "B",
        f"status == {EXPECTED_STATUS!r}",
        dev_status == EXPECTED_STATUS and test_status == EXPECTED_STATUS,
        f"dev={dev_status!r} test={test_status!r}",
    )


# --------------------------------------------------------------------------
# Section C: asset integrity of the Development release
# --------------------------------------------------------------------------


def section_c_asset_integrity(report: Report, dev_queries: list[dict[str, Any]]) -> None:
    missing_files: list[str] = []
    bad_asset_hash: list[str] = []
    bad_array_hash: list[tuple[str, str]] = []
    missing_array_key: list[tuple[str, str]] = []
    checked = 0

    for query in dev_queries:
        query_id = query["query_id"]
        asset_path = resolve_contextworld_path(query["asset"], repo_root=ROOT)
        if not asset_path.exists():
            missing_files.append(query_id)
            continue
        checked += 1

        actual_asset_hash = file_sha256(asset_path)
        if actual_asset_hash != query["asset_sha256"]:
            bad_asset_hash.append(query_id)

        try:
            with np.load(asset_path, allow_pickle=False) as bundle:
                arrays = {name: bundle[name] for name in bundle.files}
        except Exception as exc:  # pragma: no cover - defensive
            bad_asset_hash.append(f"{query_id} (unreadable npz: {exc})")
            continue

        for array_name, expected_hash in query["array_sha256"].items():
            if array_name not in arrays:
                missing_array_key.append((query_id, array_name))
                continue
            actual_hash = array_sha256(arrays[array_name])
            if actual_hash != expected_hash:
                bad_array_hash.append((query_id, array_name))
        del arrays  # release the (~tens of MB) npz payload before the next query

    report.add(
        "C",
        "asset npz exists (300 expected)",
        len(missing_files) == 0,
        f"checked={checked}/300 missing={len(missing_files)}"
        + (f" examples={_fmt_examples(missing_files)}" if missing_files else ""),
    )
    report.add(
        "C",
        "asset_sha256 matches file",
        len(bad_asset_hash) == 0,
        f"checked={checked} mismatches={len(bad_asset_hash)}"
        + (f" examples={_fmt_examples(bad_asset_hash)}" if bad_asset_hash else ""),
    )
    report.add(
        "C",
        "array_sha256 entries present in npz",
        len(missing_array_key) == 0,
        f"checked={checked} missing_entries={len(missing_array_key)}"
        + (f" examples={_fmt_examples(missing_array_key)}" if missing_array_key else ""),
    )
    report.add(
        "C",
        "array_sha256 matches recomputed hash",
        len(bad_array_hash) == 0,
        f"checked={checked} mismatches={len(bad_array_hash)}"
        + (f" examples={_fmt_examples(bad_array_hash)}" if bad_array_hash else ""),
    )


# --------------------------------------------------------------------------
# Section D: exported Lance tables (independent of the exporter's own claims)
# --------------------------------------------------------------------------


def _open_legacy_reference_schema(report: Report) -> Any | None:
    import lance

    for candidate in LEGACY_REFERENCE_TABLE_CANDIDATES:
        if candidate.exists():
            schema = lance.dataset(candidate).schema
            report.add(
                "D",
                "legacy reference table located",
                True,
                f"found at {candidate}",
            )
            return schema
    report.error(
        "D",
        "legacy reference table located",
        "not found at any candidate path: "
        + ", ".join(str(p) for p in LEGACY_REFERENCE_TABLE_CANDIDATES),
    )
    return None


def _episode_rows_sorted(episode_idx: np.ndarray, step_idx: np.ndarray, episode: int) -> np.ndarray:
    """Row indices for one episode, sorted by step_idx (never assume row order == step order)."""

    rows = np.flatnonzero(episode_idx == episode)
    return rows[np.argsort(step_idx[rows])]


def _decode_png(raw: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        return np.array(image.convert("RGB"))


def section_d_lance_export(
    report: Report,
    lance_root: Path | None,
    dev_queries_by_id: dict[str, dict[str, Any]],
    *,
    sample_episodes: int,
    seed: int,
) -> None:
    if lance_root is None:
        report.skip("D", "lance export checks", "--lance-root not provided")
        return
    if not lance_root.exists():
        report.skip(
            "D",
            "lance export checks",
            f"{lance_root} does not exist yet (export not done)",
        )
        return

    import lance

    entries = sorted(p for p in lance_root.iterdir() if p.is_dir())
    matched: list[tuple[Path, str, int]] = []  # (path, profile, delay)
    unmatched: list[str] = []
    for entry in entries:
        match = TABLE_NAME_RE.match(entry.name)
        if match is None:
            unmatched.append(entry.name)
        else:
            matched.append((entry, match.group(1), int(match.group(2))))

    report.add(
        "D",
        "exactly 66 tables named ad-h7-paired-val-p<PPP>-d<D>-*.lance",
        len(matched) == 66 and len(unmatched) == 0,
        f"matched={len(matched)} unmatched={len(unmatched)}"
        + (f" unmatched_examples={_fmt_examples(unmatched)}" if unmatched else ""),
    )

    profiles = sorted({profile for _, profile, _ in matched})
    delays = sorted({delay for _, _, delay in matched})
    combo_counts = Counter((profile, delay) for _, profile, delay in matched)
    grid_ok = (
        len(profiles) == 6
        and delays == EXPECTED_DELAYS
        and len(combo_counts) == len(matched)
        and all(c == 1 for c in combo_counts.values())
    )
    report.add(
        "D",
        "profile x delay forms a 6x11 grid, no duplicates",
        grid_ok,
        f"profiles={profiles} delays={delays} n_combos={len(combo_counts)} n_tables={len(matched)}",
    )

    legacy_schema = _open_legacy_reference_schema(report)

    # Per-table scan: schema/columns, row/episode/step structure, delay
    # column consistency, and the dev_query_id crosswalk -- all in one pass
    # over the (cheap, non-pixel) columns so we only touch the lance dataset
    # once per table for these checks.
    schema_missing: dict[str, list[str]] = {}
    schema_type_mismatch: dict[str, list[str]] = {}
    row_count_bad: list[str] = []
    episode_count_bad: list[str] = []
    step_shape_bad: list[str] = []
    delay_column_bad: list[str] = []
    query_id_to_tables: dict[str, set[str]] = defaultdict(set)
    delay_to_tables: dict[int, set[str]] = defaultdict(set)
    all_dev_query_ids: set[str] = set()
    per_table_dev_query_ids: dict[str, set[str]] = {}
    table_open_error: list[str] = []

    for path, profile, delay in matched:
        name = path.name
        try:
            ds = lance.dataset(path)
            schema = ds.schema
            row_count = ds.count_rows()

            if legacy_schema is not None:
                missing = []
                mismatched = []
                for field in legacy_schema:
                    if field.name not in schema.names:
                        missing.append(field.name)
                        continue
                    actual_type = schema.field(field.name).type
                    if not actual_type.equals(field.type):
                        mismatched.append(f"{field.name}({actual_type} != {field.type})")
                if missing:
                    schema_missing[name] = missing
                if mismatched:
                    schema_type_mismatch[name] = mismatched

            if row_count != 2500:
                row_count_bad.append(f"{name}(rows={row_count})")

            needed_columns = ["episode_idx", "step_idx", "variation_action_delay_steps"]
            has_query_id_column = "dev_query_id" in schema.names
            if has_query_id_column:
                needed_columns.append("dev_query_id")
            table = ds.to_table(columns=needed_columns)
            episode_idx = np.asarray(table.column("episode_idx").to_numpy(), dtype=np.int64)
            step_idx = np.asarray(table.column("step_idx").to_numpy(), dtype=np.int64)
            delay_values = np.asarray(
                [v[0] for v in table.column("variation_action_delay_steps").to_pylist()],
                dtype=np.float64,
            )

            episodes = sorted(int(v) for v in np.unique(episode_idx))
            if len(episodes) != 50:
                episode_count_bad.append(f"{name}(n_episodes={len(episodes)})")

            bad_clip = False
            for episode in episodes:
                rows = _episode_rows_sorted(episode_idx, step_idx, episode)
                if not np.array_equal(step_idx[rows], np.arange(EPISODE_ROWS)):
                    bad_clip = True
                    break
            if bad_clip:
                step_shape_bad.append(name)

            if not np.allclose(delay_values, float(delay)):
                observed = sorted(set(delay_values.tolist()))
                delay_column_bad.append(f"{name}(expected={delay}, observed={observed})")

            if has_query_id_column:
                query_ids = table.column("dev_query_id").to_pylist()
                distinct_ids = set(query_ids)
                per_table_dev_query_ids[name] = distinct_ids
                all_dev_query_ids |= distinct_ids
                for qid in distinct_ids:
                    query_id_to_tables[qid].add(name)
                delay_to_tables[delay].add(name)
        except Exception as exc:  # pragma: no cover - defensive
            table_open_error.append(f"{name}: {exc}")

    if table_open_error:
        report.error("D", "all 66 tables open cleanly", "; ".join(table_open_error[:5]))
    else:
        report.add("D", "all 66 tables open cleanly", True, f"{len(matched)} tables opened")

    if legacy_schema is not None:
        n_legacy_fields = len(legacy_schema)
        ok = not schema_missing and not schema_type_mismatch
        detail = f"legacy_field_count={n_legacy_fields}"
        if schema_missing:
            example_table = next(iter(schema_missing))
            detail += f" missing_in={len(schema_missing)}_tables e.g. {example_table}:{schema_missing[example_table]}"
        if schema_type_mismatch:
            example_table = next(iter(schema_type_mismatch))
            detail += (
                f" type_mismatch_in={len(schema_type_mismatch)}_tables "
                f"e.g. {example_table}:{schema_type_mismatch[example_table]}"
            )
        report.add("D", "16 legacy columns present with matching arrow types", ok, detail)
    else:
        report.skip("D", "16 legacy columns present with matching arrow types", "no legacy schema to compare against")

    report.add(
        "D",
        "each table has exactly 2500 rows",
        len(row_count_bad) == 0,
        f"bad_tables={len(row_count_bad)}" + (f" examples={_fmt_examples(row_count_bad)}" if row_count_bad else ""),
    )
    report.add(
        "D",
        "each table has exactly 50 distinct episode_idx",
        len(episode_count_bad) == 0,
        f"bad_tables={len(episode_count_bad)}"
        + (f" examples={_fmt_examples(episode_count_bad)}" if episode_count_bad else ""),
    )
    report.add(
        "D",
        "each episode's step_idx is exactly 0..49",
        len(step_shape_bad) == 0,
        f"bad_tables={len(step_shape_bad)}" + (f" examples={_fmt_examples(step_shape_bad)}" if step_shape_bad else ""),
    )
    report.add(
        "D",
        "variation_action_delay_steps == table's delay on every row",
        len(delay_column_bad) == 0,
        f"bad_tables={len(delay_column_bad)}"
        + (f" examples={_fmt_examples(delay_column_bad)}" if delay_column_bad else ""),
    )

    has_dev_query_id_everywhere = len(per_table_dev_query_ids) == len(matched) and len(matched) > 0
    if not has_dev_query_id_everywhere:
        report.error(
            "D",
            "dev_query_id column present on every table",
            f"present_on={len(per_table_dev_query_ids)}/{len(matched)} tables",
        )
    else:
        report.add("D", "dev_query_id column present on every table", True, f"present_on={len(matched)}/{len(matched)} tables")

    catalog_query_ids = set(dev_queries_by_id.keys())
    extra_in_lance = all_dev_query_ids - catalog_query_ids
    missing_from_lance = catalog_query_ids - all_dev_query_ids
    report.add(
        "D",
        "300 distinct dev_query_id, exactly matching Development catalog",
        len(all_dev_query_ids) == 300 and not extra_in_lance and not missing_from_lance,
        f"lance_distinct={len(all_dev_query_ids)} catalog_distinct={len(catalog_query_ids)} "
        f"extra_in_lance={len(extra_in_lance)} missing_from_lance={len(missing_from_lance)}"
        + (f" extra_examples={_fmt_examples(extra_in_lance)}" if extra_in_lance else "")
        + (f" missing_examples={_fmt_examples(missing_from_lance)}" if missing_from_lance else ""),
    )

    wrong_table_count_per_query = {
        qid: len(tables) for qid, tables in query_id_to_tables.items() if len(tables) != 11
    }
    report.add(
        "D",
        "each dev_query_id appears in exactly 11 tables",
        len(wrong_table_count_per_query) == 0,
        f"violations={len(wrong_table_count_per_query)}"
        + (f" examples={_fmt_examples(wrong_table_count_per_query.items())}" if wrong_table_count_per_query else ""),
    )

    wrong_table_count_per_delay = {
        delay: len(tables) for delay, tables in delay_to_tables.items() if len(tables) != 6
    }
    all_delays_present = sorted(delay_to_tables.keys()) == EXPECTED_DELAYS
    report.add(
        "D",
        "each delay (0..10) appears in exactly 6 tables",
        all_delays_present and len(wrong_table_count_per_delay) == 0,
        f"delays_present={sorted(delay_to_tables.keys())} violations={len(wrong_table_count_per_delay)}"
        + (
            f" examples={_fmt_examples((d, c) for d, c in wrong_table_count_per_delay.items())}"
            if wrong_table_count_per_delay
            else ""
        ),
    )

    _section_d_pixel_spotcheck(
        report,
        matched,
        per_table_dev_query_ids,
        dev_queries_by_id,
        sample_episodes=sample_episodes,
        seed=seed,
    )


def _section_d_pixel_spotcheck(
    report: Report,
    matched: list[tuple[Path, str, int]],
    per_table_dev_query_ids: dict[str, set[str]],
    dev_queries_by_id: dict[str, dict[str, Any]],
    *,
    sample_episodes: int,
    seed: int,
) -> None:
    import lance

    table_by_name = {path.name: (path, delay) for path, _profile, delay in matched}
    usable_tables = [name for name in table_by_name if name in per_table_dev_query_ids]
    if not usable_tables:
        report.skip(
            "D",
            f"pixel spot-check ({sample_episodes} episodes vs frozen npz)",
            "no tables with a readable dev_query_id column",
        )
        return

    rng = random.Random(seed)
    sample_names = rng.sample(usable_tables, k=min(sample_episodes, len(usable_tables)))

    mismatches: list[str] = []
    errors: list[str] = []
    checked = 0

    for table_name in sample_names:
        table_path, delay = table_by_name[table_name]
        try:
            ds = lance.dataset(table_path)
            table = ds.to_table(columns=["episode_idx", "step_idx", "pixels", "dev_query_id"])
            episode_idx = np.asarray(table.column("episode_idx").to_numpy(), dtype=np.int64)
            step_idx = np.asarray(table.column("step_idx").to_numpy(), dtype=np.int64)
            pixels = table.column("pixels").to_pylist()
            dev_query_id = table.column("dev_query_id").to_pylist()

            episodes = sorted(int(v) for v in np.unique(episode_idx))
            episode = rng.choice(episodes)
            rows = _episode_rows_sorted(episode_idx, step_idx, episode)
            query_ids_in_episode = {dev_query_id[int(r)] for r in rows}
            if len(query_ids_in_episode) != 1:
                errors.append(f"{table_name}/ep{episode}: dev_query_id not constant ({query_ids_in_episode})")
                continue
            query_id = query_ids_in_episode.pop()

            query = dev_queries_by_id.get(query_id)
            if query is None:
                errors.append(f"{table_name}/ep{episode}: dev_query_id {query_id!r} not in Development catalog")
                continue

            asset_path = resolve_contextworld_path(query["asset"], repo_root=ROOT)
            with np.load(asset_path, allow_pickle=False) as bundle:
                target_delays = bundle["target_delays"]
                delay_positions = np.flatnonzero(target_delays == delay)
                if len(delay_positions) != 1:
                    errors.append(
                        f"{table_name}/ep{episode}: delay {delay} not uniquely present in "
                        f"target_delays={target_delays.tolist()}"
                    )
                    continue
                delay_idx = int(delay_positions[0])
                history_pixels = bundle["history_pixels"][delay_idx]  # (7, H, W, 3)
                true_future_pixels = bundle["true_future_pixels"][delay_idx]  # (3, H, W, 3)

            checked += 1
            for boundary_row in BLOCK_BOUNDARY_ROWS:
                token_index = boundary_row // 5
                if token_index < 7:
                    expected = history_pixels[token_index]
                else:
                    expected = true_future_pixels[token_index - 7]
                decoded = _decode_png(pixels[int(rows[boundary_row])])
                if not np.array_equal(decoded, expected):
                    mismatches.append(f"{table_name}/ep{episode}/row{boundary_row}(query={query_id})")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{table_name}: {exc}")

    ok = checked > 0 and not mismatches and not errors
    detail = f"episodes_checked={checked}/{len(sample_names)} mismatched_frames={len(mismatches)} errors={len(errors)}"
    if mismatches:
        detail += f" mismatch_examples={_fmt_examples(mismatches)}"
    if errors:
        detail += f" error_examples={_fmt_examples(errors)}"
    report.add("D", f"pixel spot-check ({len(sample_names)} episodes vs frozen npz)", ok, detail)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--lance-root",
        type=Path,
        default=None,
        help="Directory holding the 66 newly exported Lance tables. If omitted or "
        "not-yet-existing, section D is skipped with a clear message.",
    )
    parser.add_argument(
        "--sample-episodes",
        type=int,
        default=10,
        help="Number of (table, episode) samples to pixel-check in section D (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260910,
        help="Random seed for the section-D episode sample (default: fixed, for reproducibility).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = Report()

    dev_catalog = load_json(DEV_RELEASE / "catalog.json")
    test_catalog = load_json(TEST_RELEASE / "catalog.json")
    dev_queries = dev_catalog["queries"]
    test_queries = test_catalog["queries"]
    dev_queries_by_id = {q["query_id"]: q for q in dev_queries}

    print(f"Development release : {DEV_RELEASE}")
    print(f"Public Test release : {TEST_RELEASE}")
    print(f"Lance root (--lance-root): {args.lance_root}")
    print()

    section_a_disjointness(report, dev_queries, test_queries)
    section_b_structural_parity(report, dev_catalog, test_catalog)
    section_c_asset_integrity(report, dev_queries)
    lance_root = args.lance_root.resolve() if args.lance_root is not None else None
    section_d_lance_export(
        report,
        lance_root,
        dev_queries_by_id,
        sample_episodes=args.sample_episodes,
        seed=args.seed,
    )

    print(report.render())
    return 1 if report.any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
