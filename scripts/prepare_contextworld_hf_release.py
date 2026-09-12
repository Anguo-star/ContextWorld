#!/usr/bin/env python3
"""Stage the ContextWorld v3 HF dataset release (native Lance snapshot).

Modes (mutually exclusive): plan (default) | --execute | --verify.
Reads Training+Development from ContextWorld-v1 and Test ONLY from the sibling
ContextWorld-v1-full bundle; never touches source bytes. Standalone stdlib CLI.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_REL = "configs/benchmark/contextworld_joint_scratch_v1_reference_results_freeze_v3.json"
TEMPLATE_REL = "docs/templates/ContextWorld_HF_Dataset_Card.md"
BASELINE_ID = "contextworld_joint_scratch_v1_reference_results_freeze_v3"
DATASET_VERSION = "contextworld-v3-hf-rc1"
REPRESENTATION = "native_lance_snapshot"
RELEASE_STATUS = "staging_not_public_release"
LEGAL_FILES = ("LICENSE", "DATA_LICENSE", "NOTICE")
DEV_BUNDLE, TEST_BUNDLE = "ContextWorld-v1", "ContextWorld-v1-full"
REQUIRED_COMPONENTS = 9
REQUIRED_SPLITS = ("training", "development", "test")
SOURCE_ROLES = {"dataset_payload", "release_metadata", "legal_metadata", "component_documentation"}
OUTPUT_ROLES = {"dataset_payload", "release_metadata", "legal_metadata", "dataset_card",
                "component_documentation", "provenance", "reference", "inventory"}
NORMALIZER_RE = re.compile(r"normalizers/[A-Za-z0-9_./-]+")
PINS = {
    "baseline": "01298ca407c4a72bdd9a1238879ae7109b1141a7cfdd2b25065cb0bdec87fef0",
    "dev_manifest": "4c5b9cdc84006caeac2a9770501f943affe2b81e32d70c27ab504770b28a93c4",
    "dev_registry": "7952ff61e59e1959d1b31f5c7aba9506b6410944f7063dad1c16a650b7aa6c46",
    "test_manifest": "81eb29d8eeaefb91750187831b9120b9241c78b4903047cd4b720a186c7f6d97",
    "test_registry": "26b78ae7d3bbaa2c170080bc1eb8791582a9183e7d606322aaf6bae1b01dcbdd",
}


class ReleaseError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            h.update(chunk)
    return total, h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def is_safe_rel(path):
    if not isinstance(path, str):
        return False
    p = Path(path)
    if p.is_absolute() or "\\" in path or not path or path.startswith("/"):
        return False
    return ".." not in p.parts and p.as_posix() == path and path != "."


def regular_file(root, relative):
    require(is_safe_rel(relative), f"Unsafe relative path: {relative!r}")
    require(not root.is_symlink(), f"Symlink root is forbidden: {root}")
    path = root
    for part in Path(relative).parts:
        path = path / part
        require(not path.is_symlink(), f"Symlink is forbidden: {path}")
    require(path.is_file(), f"Missing regular file: {path}")
    return path


def check_pin(path, expected, label):
    regular_file(path.parent, path.name)
    _, digest = sha256_file(path)
    if digest != expected:
        raise ReleaseError(f"{label}: sha256 pin mismatch for {path} (got {digest}, want {expected})")


def load_manifest_rows(root, label):
    mpath = root / "manifest.jsonl"
    seen, rows = set(), []
    for lineno, line in enumerate(mpath.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReleaseError(f"{label} manifest.jsonl:{lineno}: invalid JSON ({exc})")
        role = row.get("role")
        if role not in SOURCE_ROLES:
            raise ReleaseError(f"{label} manifest.jsonl:{lineno}: unexpected role {role!r}")
        for key in ("path", "bytes", "sha256"):
            if key not in row:
                raise ReleaseError(f"{label} manifest.jsonl:{lineno}: missing {key}")
        if not is_safe_rel(row["path"]):
            raise ReleaseError(f"{label} manifest.jsonl:{lineno}: unsafe path {row['path']!r}")
        if row["path"] in seen:
            raise ReleaseError(f"{label} manifest.jsonl:{lineno}: duplicate path {row['path']!r}")
        seen.add(row["path"])
        rows.append(row)
    if not rows:
        raise ReleaseError(f"{label}: empty manifest")
    sidecar = regular_file(root, "manifest.sha256")
    require(sidecar.read_text().split() == [sha256_file(mpath)[1], "manifest.jsonl"],
            f"{label}: manifest.sha256 sidecar mismatch")
    return rows


def load_bundle(root, name, role, man_pin, reg_pin):
    root = Path(root)
    if not root.is_dir():
        raise ReleaseError(f"{name} root is not a directory: {root}")
    check_pin(root / "manifest.jsonl", man_pin, f"{name} manifest")
    check_pin(root / "task_registry.json", reg_pin, f"{name} registry")
    rows = load_manifest_rows(root, name)
    registry = json.loads((root / "task_registry.json").read_text())
    comps = {}
    for comp in registry.get("components", []):
        cid = comp.get("component_id")
        if not cid or cid in comps:
            raise ReleaseError(f"{name}: missing/duplicate component_id {cid!r}")
        for payload in comp.get("payloads", []):
            require(is_safe_rel(payload.get("public_path")), f"{name}: unsafe payload path")
            for member in payload.get("members", []):
                if not is_safe_rel(member):
                    raise ReleaseError(f"{name}: unsafe member {member!r}")
        comps[cid] = comp
    if not comps:
        raise ReleaseError(f"{name}: registry has no components")
    return {"name": name, "role": role, "root": root, "rows": rows,
            "registry": registry, "components": comps}


def require(cond, msg):
    if not cond:
        raise ReleaseError(msg)


def check_alignment(dev, tst):
    require(dev["registry"].get("schema_version") == tst["registry"].get("schema_version"),
            "registry schema_version mismatch between bundles")
    dmap, tmap = dev["components"], tst["components"]
    require(len(dmap) == REQUIRED_COMPONENTS, f"dev bundle must have {REQUIRED_COMPONENTS} components")
    require(sorted(dmap) == sorted(tmap), "component sets differ between dev and test registries")
    for cid, dcomp in dmap.items():
        tcomp = tmap[cid]
        require(sorted(dcomp.keys()), f"{cid}: empty component")
        require("development_evaluation" in dcomp, f"{cid}: dev component lacks development_evaluation")
        require("public_test_evaluation" in tcomp, f"{cid}: test component lacks public_test_evaluation")
        require(tcomp["public_test_evaluation"].get("split") == "test",
                f"{cid}: public_test_evaluation split must be 'test'")
        for field in ("component_id", "dataset_id", "environment", "history_length",
                      "release_config", "release_config_sha256"):
            require(dcomp.get(field) == tcomp.get(field),
                    f"{cid}: identity field {field} misaligned between dev/test registries")


def select_payload_rows(bundle, splits):
    """Select role=dataset_payload rows whose split tag is in `splits`; verify each
    row is covered by exactly one same-component/same-split payload prefix
    (public_path union members) and that per-payload coverage equals file_count."""
    root = bundle["root"]
    selected, per_split, per_payload = [], defaultdict(lambda: [0, 0]), Counter()
    for row in bundle["rows"]:
        if row["role"] != "dataset_payload":
            continue
        if row.get("split") not in splits:
            continue
        cid, split = row.get("component"), row.get("split")
        require(cid in bundle["components"], f"{bundle['name']}: payload row {row['path']!r} has unknown component {cid!r}")
        comp = bundle["components"][cid]
        hits = []
        for payload in comp["payloads"]:
            if payload.get("split") != split:
                continue
            prefixes = [payload.get("public_path")] + list(payload.get("members", []))
            if any(p and (row["path"] == p or row["path"].startswith(p + "/")) for p in prefixes):
                hits.append(payload)
        require(len(hits) == 1,
                f"{bundle['name']}: row {row['path']!r} covered by {len(hits)} payloads (need exactly 1)")
        payload = hits[0]
        per_payload[(cid, split, payload.get("payload_id"))] += 1
        src = regular_file(root, row["path"])
        require(src.stat().st_size == row["bytes"], f"Source size mismatch: {src}")
        selected.append(row)
        stats = per_split[(comp.get("dataset_id", cid), split)]
        stats[0] += 1
        stats[1] += row["bytes"]
    for comp in bundle["components"].values():
        for payload in comp["payloads"]:
            if payload.get("split") not in splits:
                continue
            pid = payload.get("payload_id")
            got = per_payload.get((comp['component_id'], payload['split'], pid), 0)
            want = payload.get("file_count")
            require(got >= 1, f"{bundle['name']}: payload {pid!r} has no manifest coverage")
            if want is not None:
                require(got == want, f"{bundle['name']}: payload {pid!r} covered by {got} rows, registry declares {want}")
    return selected, per_split


def merge_registry(dev, tst, export_id):
    merged = json.loads(json.dumps(dev["registry"]))  # deep copy
    tmap = tst["components"]
    for comp in merged["components"]:
        cid = comp["component_id"]
        tcomp = tmap[cid]
        keep = [json.loads(json.dumps(p)) for p in comp["payloads"] if p.get("split") in ("training", "development")]
        append = [json.loads(json.dumps(p)) for p in tcomp["payloads"] if p.get("split") == "test"]
        require(keep and append, f"{cid}: merged component must keep training/development and gain test payloads")
        comp["payloads"] = keep + append
        comp["public_test_evaluation"] = json.loads(json.dumps(tcomp["public_test_evaluation"]))
        for p in comp["payloads"]:
            require(p.get("split") != "development_legacy", f"{cid}: legacy split must not survive merge")
    merged["public_test"] = json.loads(json.dumps(tst["registry"]["public_test"]))
    merged["export_id"] = export_id
    merged["release_status"] = RELEASE_STATUS
    return merged


def collect_metadata_copies(dev):
    """Byte-exact copies from dev root: registry_contract.json files and every
    normalizer explicitly referenced by the dev registry. Roles stay release_metadata."""
    refs = sorted(set(NORMALIZER_RE.findall(json.dumps(dev["registry"]))))
    rows_by_path = {r["path"]: r for r in dev["rows"]}
    contracts = sorted(r["path"] for r in dev["rows"]
                       if r["role"] == "release_metadata"
                       and fnmatch.fnmatch(r["path"], "components/*/v1/development/registry_contract.json"))
    copies = []
    for rel in contracts + [p for p in refs if p not in contracts]:
        row = rows_by_path.get(rel)
        require(row is not None, f"dev bundle: referenced metadata {rel!r} missing from source manifest")
        require(row["role"] == "release_metadata", f"dev bundle: {rel!r} must keep role release_metadata")
        src = regular_file(dev["root"], rel)
        require(sha256_file(src) == (row["bytes"], row["sha256"]), f"Metadata hash mismatch: {rel}")
        copies.append({"rel": rel, "src": src, "bundle": DEV_BUNDLE,
                       "bytes": row["bytes"], "sha256": row["sha256"]})
    return copies


def build_plan(args):
    dev = load_bundle(Path(args.development_root), DEV_BUNDLE, "training_and_development", PINS["dev_manifest"], PINS["dev_registry"])
    tst = load_bundle(Path(args.test_root), TEST_BUNDLE, "test_only", PINS["test_manifest"], PINS["test_registry"])
    check_alignment(dev, tst)
    check_pin(resolve_repo(BASELINE_REL), PINS["baseline"], "baseline")
    dev_sel, dev_stats = select_payload_rows(dev, ("training", "development"))
    tst_sel, tst_stats = select_payload_rows(tst, ("test",))
    legacy = [r for r in dev["rows"] if r["role"] == "dataset_payload" and r.get("split") not in ("training", "development")]
    stale = [r for r in tst["rows"] if r["role"] == "dataset_payload" and r.get("split") != "test"]
    template = resolve_repo(TEMPLATE_REL)
    require(template.is_file(), f"missing dataset card template: {template}")
    selected = [dict(r, source_bundle=DEV_BUNDLE) for r in dev_sel]
    selected += [dict(r, source_bundle=TEST_BUNDLE) for r in tst_sel]
    require(len({r['path'] for r in selected}) == len(selected), "Duplicate destination payload path")
    return {"dev": dev, "tst": tst, "selected": selected,
            "by_component_split": {f"{k[0]}|{k[1]}": {"files": v[0], "bytes": v[1]} for k, v in {**dev_stats, **tst_stats}.items()},
            "legacy_omitted": len(legacy), "stale_test_omitted": len(stale),
            "metadata_copies": collect_metadata_copies(dev), "template": template}


def resolve_repo(rel):
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def split_inventory_table(by_component_split):
    lines = ["| Component | Split | Files | Bytes |", "|---|---|---:|---:|"]
    tf = tb = 0
    for key in sorted(by_component_split):
        cid, split = key.split("|", 1)
        stats = by_component_split[key]
        tf += stats["files"]
        tb += stats["bytes"]
        lines.append(f"| `{cid}` | {split} | {stats['files']} | {stats['bytes']} |")
    lines.append(f"| **TOTAL** | all | **{tf}** | **{tb}** |")
    return "\n".join(lines)


def render_card(comp, by_component_split, dataset_id):
    did = comp["dataset_id"]
    rows = "\n".join(f"- {s}: {by_component_split[f'{did}|{s}']['files']} files, "
                     f"{by_component_split[f'{did}|{s}']['bytes']} bytes"
                     for s in REQUIRED_SPLITS if f"{did}|{s}" in by_component_split)
    de = comp.get("development_evaluation", {})
    pte = comp.get("public_test_evaluation", {})
    return (f"# {did}\n\n"
            f"Generated ContextWorld v3 HF release card for component `{did}` "
            f"(component_id `{comp['component_id']}`).\n\n"
            f"- Description: {comp.get('description', '')}\n"
            f"- Capability category: {comp.get('capability_category', '')}\n"
            f"- Environment: `{json.dumps(comp.get('environment'))}`\n"
            f"- History length: {comp.get('history_length')} | frameskip: {comp.get('frameskip')} | "
            f"action dimension: {comp.get('action_dimension')}\n"
            f"- Prediction horizons (action blocks): `{json.dumps(comp.get('prediction_horizons_action_blocks'))}`\n\n"
            f"## Payload inventory\n{rows}\n\n"
            f"## Development evaluation (authoritative selection)\n"
            f"- Input contract: `{json.dumps(de.get('input_contract'))}`\n"
            f"- Normalizer: `{de.get('normalizer_path')}`\n\n"
            f"## Public test evaluation\n"
            f"- Selection policy: `{pte.get('selection_policy')}` | status: `{pte.get('status')}`\n\n"
            f"Representation: native Lance snapshot; no parquet conversion.\n")


def build_generated(plan, export_id):
    dev, tst, sel = plan["dev"], plan["tst"], plan["selected"]
    payload_files = len(sel)
    payload_bytes = sum(r["bytes"] for r in sel)
    files = {}
    merged = merge_registry(dev, tst, export_id)
    version = {
        "schema_version": 1, "dataset_version": DATASET_VERSION, "baseline_id": BASELINE_ID,
        "baseline_sha256": PINS["baseline"], "representation": REPRESENTATION,
        "release_status": RELEASE_STATUS, "public_test_included": True, "export_id": export_id,
        "source_bundles": {
            DEV_BUNDLE: {"role": "training_and_development",
                         "manifest_sha256": PINS["dev_manifest"], "task_registry_sha256": PINS["dev_registry"]},
            TEST_BUNDLE: {"role": "test_only",
                          "manifest_sha256": PINS["test_manifest"], "task_registry_sha256": PINS["test_registry"]}},
        "reference": {"path": f"reference/{BASELINE_ID}.json", "sha256": PINS["baseline"]},
    }
    files["VERSION.json"] = json.dumps(version, indent=2, sort_keys=True) + "\n"
    files["task_registry.json"] = json.dumps(merged, indent=2, sort_keys=True) + "\n"
    inventory = {
        "schema_version": "contextworld.inventory.v1", "dataset_version": DATASET_VERSION,
        "baseline_id": BASELINE_ID,
        "totals": {"payload_files": payload_files, "payload_bytes": payload_bytes},
        "by_component": {}, "by_split": {},
        "omitted_legacy": {"files": plan["legacy_omitted"], "source_bundle": DEV_BUNDLE},
        "unused_source_payload_files": {DEV_BUNDLE: plan["legacy_omitted"],
                                        TEST_BUNDLE: plan["stale_test_omitted"]},
    }
    for key, stats in plan["by_component_split"].items():
        cid, split = key.split("|", 1)
        inventory["by_component"].setdefault(cid, {})[split] = stats
        inventory["by_split"].setdefault(split, {})[cid] = stats
    files["inventory.json"] = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    files[".gitattributes"] = (
        "*.lance filter=lfs diff=lfs merge=lfs -text\n"
        "*.txn filter=lfs diff=lfs merge=lfs -text\n"
        "*.manifest filter=lfs diff=lfs merge=lfs -text\n")
    template = plan["template"].read_text()
    readme = (template.replace("{{SPLIT_INVENTORY}}", split_inventory_table(plan["by_component_split"]))
              .replace("{{BASELINE_ID}}", BASELINE_ID).replace("{{BASELINE_SHA256}}", PINS["baseline"]))
    files["README.md"] = readme
    for comp in merged["components"]:
        card = render_card(comp, plan["by_component_split"], comp["dataset_id"])
        files[f"components/{comp['dataset_id']}/v1/component_card.md"] = card
    return files, merged, version, inventory


def copy_job(job, progress):
    src, dst = job["src"], job["dst"]
    h = hashlib.sha256()
    copied = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fin, open(dst, "xb") as fout:
        while True:
            chunk = fin.read(1 << 20)
            if not chunk:
                break
            fout.write(chunk)
            h.update(chunk)
            copied += len(chunk)
    digest = h.hexdigest()
    if job.get("declared_bytes") is not None and copied != job["declared_bytes"]:
        raise ReleaseError(f"size mismatch copying {job['rel']}: {copied} != {job['declared_bytes']}")
    if job.get("declared_sha256") and digest != job["declared_sha256"]:
        raise ReleaseError(f"sha256 mismatch copying {job['rel']}")
    _, redigest = sha256_file(dst)
    if redigest != digest:
        raise ReleaseError(f"destination re-hash mismatch for {job['rel']}")
    with progress["lock"]:
        progress["done"] += 1
        if progress["done"] % 250 == 0:
            print(f"[hf-release] copied {progress['done']}/{progress['total']} files", file=sys.stderr, flush=True)
    return {"rel": job["rel"], "bytes": copied, "sha256": digest}


def cmd_execute(args):
    plan = build_plan(args)
    output = Path(args.output).absolute()
    require(not output.exists() and not output.is_symlink(), f"Output already exists: {output}")
    for source in (plan["dev"]["root"], plan["tst"]["root"]):
        source = source.resolve()
        target = output.resolve()
        require(target != source and source not in target.parents and target not in source.parents,
                f"Output overlaps source: {source}")
    export_id = "contextworld-v3-hf-rc1"
    files, merged, version, inventory = build_generated(plan, export_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    direct = getattr(args, "direct_write", False)
    staging = output if direct else output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    roots = {DEV_BUNDLE: plan["dev"]["root"], TEST_BUNDLE: plan["tst"]["root"]}
    jobs, identities = [], {}

    def add(rel, src, role, size=None, digest=None, **identity):
        require(is_safe_rel(rel), f"Unsafe output path: {rel}")
        require(rel not in identities and rel not in files, f"Duplicate output: {rel}")
        regular_file(src.parent, src.name)
        if size is None or digest is None:
            size, digest = sha256_file(src)
        identities[rel] = {"role": role, **identity}
        jobs.append({"rel": rel, "src": src, "dst": staging / rel,
                     "declared_bytes": size, "declared_sha256": digest})

    for row in plan["selected"]:
        bundle = row["source_bundle"]
        src = regular_file(roots[bundle], row["path"])
        add(row["path"], src, "dataset_payload", row["bytes"], row["sha256"],
            component=row["component"], split=row["split"], source_bundle=bundle,
            source_path=row["path"], source_sha256=row["sha256"])
    for meta in plan["metadata_copies"]:
        add(meta["rel"], meta["src"], "release_metadata", meta["bytes"], meta["sha256"],
            source_bundle=DEV_BUNDLE, source_path=meta["rel"], source_sha256=meta["sha256"])
    for name, source in roots.items():
        for filename in ("manifest.jsonl", "manifest.sha256", "task_registry.json"):
            add(f"provenance/{name}/{filename}", regular_file(source, filename), "provenance")
    add(f"reference/{BASELINE_ID}.json", resolve_repo(BASELINE_REL), "reference",
        digest=PINS["baseline"], size=resolve_repo(BASELINE_REL).stat().st_size)
    for name in LEGAL_FILES:
        add(name, regular_file(REPO_ROOT, name), "legal_metadata")

    staging.mkdir()
    try:
        if direct:
            (staging / ".build_in_progress").write_text("not a completed release candidate\n")
        progress = {"lock": threading.Lock(), "done": 0, "total": len(jobs)}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda job: copy_job(job, progress), jobs))
        manifest = [{"path": r["rel"], "bytes": r["bytes"], "sha256": r["sha256"],
                     **identities[r["rel"]]} for r in results]
        for rel, content in files.items():
            require(is_safe_rel(rel), f"Unsafe generated path: {rel}")
            path = staging / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode("utf-8")
            with path.open("xb") as stream:
                stream.write(data)
            role = "component_documentation" if rel.startswith("components/") else "release_metadata"
            manifest.append({"path": rel, "bytes": len(data), "sha256": sha256_bytes(data), "role": role})
        manifest.sort(key=lambda row: row["path"])
        data = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in manifest)
        (staging / "manifest.jsonl").write_bytes(data)
        digest = sha256_bytes(data)
        (staging / "manifest.sha256").write_text(f"{digest}  manifest.jsonl\n")
        if direct:
            (staging / ".build_in_progress").unlink()
        else:
            require(not output.exists(), f"Output appeared during build: {output}")
            staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return {"status": "built", "output": str(output), "dataset_version": DATASET_VERSION,
            "manifest_sha256": digest, "manifest_files": len(manifest), **inventory["totals"]}


def expected_source_rows(dev, tst):
    result = {}
    for bundle, splits in ((dev, {"training", "development"}), (tst, {"test"})):
        for row in bundle["rows"]:
            if row["role"] == "dataset_payload" and row.get("split") in splits:
                require(row["path"] not in result, f"Duplicate expected payload: {row['path']}")
                result[row["path"]] = (bundle["name"], row)
    return result


def verify_package(output, workers=8):
    root = Path(output).absolute()
    manifest_path = regular_file(root, "manifest.jsonl")
    digest = sha256_file(manifest_path)[1]
    sidecar = regular_file(root, "manifest.sha256").read_text().split()
    require(sidecar == [digest, "manifest.jsonl"], "Manifest checksum mismatch")
    rows = {}
    for line in manifest_path.read_text().splitlines():
        row = json.loads(line)
        rel = row.get("path")
        require(is_safe_rel(rel), f"Unsafe manifest path: {rel!r}")
        require(rel not in rows and rel not in {"manifest.jsonl", "manifest.sha256"},
                f"Duplicate/self-referencing manifest entry: {rel}")
        require(row.get("role") in OUTPUT_ROLES, f"Unknown manifest role: {rel}")
        rows[rel] = row
    require(rows, "Empty candidate manifest")
    listed = set(rows) | {"manifest.jsonl", "manifest.sha256"}
    actual = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(dirs):
            path = base / name
            relative = path.relative_to(root).as_posix()
            require(not path.is_symlink(), f"Symlink directory: {relative}")
            if relative in {".git", ".cache/huggingface"}:
                dirs.remove(name)
        for name in names:
            path = base / name
            require(not path.is_symlink(), f"Symlink file: {path}")
            actual.add(path.relative_to(root).as_posix())
    require(actual == listed,
            f"Candidate inventory differs; missing={sorted(listed-actual)[:5]}, extra={sorted(actual-listed)[:5]}")

    def verify_row(row):
        path = regular_file(root, row["path"])
        require(sha256_file(path) == (row["bytes"], row["sha256"]),
                f"File integrity mismatch: {row['path']}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(verify_row, rows.values()))
    for name in ("VERSION.json", "task_registry.json", "inventory.json", "README.md", *LEGAL_FILES):
        require(name in rows, f"Required metadata missing: {name}")
    version = json.loads((root / "VERSION.json").read_text())
    require(version.get("baseline_id") == BASELINE_ID and version.get("baseline_sha256") == PINS["baseline"],
            "Wrong baseline identity")
    require(version.get("dataset_version") == DATASET_VERSION and version.get("representation") == REPRESENTATION,
            "Wrong release representation/version")
    check_pin(regular_file(root, f"reference/{BASELINE_ID}.json"), PINS["baseline"], "baseline")
    dev = load_bundle(root / "provenance" / DEV_BUNDLE, DEV_BUNDLE, "training_and_development",
                      PINS["dev_manifest"], PINS["dev_registry"])
    tst = load_bundle(root / "provenance" / TEST_BUNDLE, TEST_BUNDLE, "test_only",
                      PINS["test_manifest"], PINS["test_registry"])
    check_alignment(dev, tst)
    registry = json.loads((root / "task_registry.json").read_text())
    require(registry == merge_registry(dev, tst, version["export_id"]), "Registry differs from frozen source contracts")
    expected = expected_source_rows(dev, tst)
    observed = {p: r for p, r in rows.items() if r["role"] == "dataset_payload"}
    require(set(expected) == set(observed), "Payload coverage differs from frozen source manifests")
    for path, (bundle, source) in expected.items():
        row = observed[path]
        for key in ("component", "split", "bytes", "sha256"):
            require(row.get(key) == source.get(key), f"Frozen payload {key} changed: {path}")
        require(row.get("source_bundle") == bundle and row.get("source_path") == path
                and row.get("source_sha256") == source["sha256"], f"Wrong source attribution: {path}")
    # Small metadata sidecars and normalizers keep their frozen bytes too.
    normalizers = set(NORMALIZER_RE.findall(json.dumps(dev["registry"])))
    for source in dev["rows"]:
        path = source["path"]
        if path in normalizers or fnmatch.fnmatch(path, "components/*/v1/development/registry_contract.json"):
            require(path in rows and rows[path]["role"] == "release_metadata"
                    and (rows[path]["bytes"], rows[path]["sha256"]) == (source["bytes"], source["sha256"]),
                    f"Frozen metadata missing or changed: {path}")
    coverage = {(r["component"], r["split"]) for r in observed.values()}
    require(coverage == {(c, s) for c in dev["components"] for s in REQUIRED_SPLITS}, "Incomplete task/split coverage")
    total_bytes = sum(row["bytes"] for row in observed.values())
    inventory = json.loads((root / "inventory.json").read_text())
    require(inventory["totals"] == {"payload_files": len(observed), "payload_bytes": total_bytes},
            "Inventory totals disagree with payloads")
    return {"status": "passed", "manifest_sha256": digest, "manifest_files": len(rows),
            "payload_files": len(observed), "payload_bytes": total_bytes,
            "component_count": len(dev["components"]), "component_split_count": len(coverage),
            "baseline_sha256": PINS["baseline"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--development-root", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--direct-write", action="store_true",
                        help="Create a new output directly when the mount cannot rename populated directories.")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 64:
        parser.error("--workers must be between 1 and 64")
    if not args.verify and (args.development_root is None or args.test_root is None):
        parser.error("plan/execute require --development-root and --test-root")
    if args.direct_write and not args.execute:
        parser.error("--direct-write requires --execute")
    try:
        if args.verify:
            result = verify_package(args.output, args.workers)
        elif args.execute:
            result = cmd_execute(args)
        else:
            plan = build_plan(args)
            result = {"status": "plan_only", "output": str(args.output),
                      "payload_files": len(plan["selected"]),
                      "payload_bytes": sum(r["bytes"] for r in plan["selected"]),
                      "omitted_legacy_files": plan["legacy_omitted"],
                      "by_component_split": plan["by_component_split"]}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ReleaseError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"HF release preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
