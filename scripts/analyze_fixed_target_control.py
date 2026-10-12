#!/usr/bin/env python3
"""Archive the completed, fixed-target LeWM control without loading weights."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


DEFAULT_QUEUE = Path("/opt/huawei/explorer-env/dataset/ag_data/ckpt/lewm-fixed-target-control-v1/queue_status.json")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs/research/data/fixed_target_control_v1"
DEFAULT_DOC = Path(__file__).resolve().parents[1] / "docs/ICL_Metric_Study.md"
BEGIN = "<!-- BEGIN FIXED_TARGET_CONTROL -->"
END = "<!-- END FIXED_TARGET_CONTROL -->"
TASKS = ("contact_friction", "robot_arm_mass")
EPOCHS = (1, 5, 10)
SPLITS = {"training": 512, "development": 256}
METRICS = ("response_nre", "full_error", "full_error_over_B", "history_benefit", "history_benefit_over_B")
DELTAS = ("delta_response_nre", "delta_full_error", "delta_full_error_over_B", "delta_history_benefit_over_B")
FIELDS = ("task", "split", "epoch", "pairs", "response_nre", "full_error_over_B", "history_benefit_over_B", "B_conditional_mean_error")


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build(queue_path):
    queue = read_json(queue_path)
    require(queue["status"] == "completed" and set(queue["tasks"]) == set(TASKS), "incomplete or unexpected queue")
    archive = {"schema": "contextworld.fixed_target_control_archive.v1", "scope": "native one-step query, physical step 5; training 512 pairs, development 256 pairs", "source_queue": str(queue_path), "source_queue_sha256": digest(queue_path), "tasks": {}}
    rows = []
    for task in TASKS:
        item = queue["tasks"][task]
        require(item["status"] == "completed" and item["training_returncode"] == 0, f"{task}: training incomplete")
        require(set(map(int, item["evaluations"])) == set(EPOCHS), f"{task}: evaluation epochs")
        run = Path(item["run_dir"])
        recipe_path, loader_path, end_path = (run / n for n in ("recipe.json", "loader_actual.json", "train_end_audit.json"))
        recipe, loader, end = map(read_json, (recipe_path, loader_path, end_path))
        require(loader["world_size"] == end["world_size"] == item["train_end"]["world_size"] == 8, f"{task}: world size")
        require(loader["max_epochs"] == 10 and loader["max_steps"] == -1, f"{task}: ten epoch recipe")
        require(end["global_step"] == loader["estimated_optimizer_steps"] == item["train_end"]["global_step"], f"{task}: final step")
        require(end == item["train_end"] and end["frozen_state_unchanged"], f"{task}: final audit")
        frozen = end["frozen_hashes"]
        require(frozen["encoder"]["tensors"] == 198 and frozen["projector"]["tensors"] == 9, f"{task}: frozen tensor count")
        require(read_json(run / "initial_frozen_hashes.json") == frozen, f"{task}: initial frozen identity")
        for epoch in range(1, 11):
            checkpoint = run / f"weights_epoch_{epoch}.pt"
            audit_path = run / f"epoch_{epoch:02d}_audit.json"
            require(checkpoint.is_file() and checkpoint.stat().st_size > 0, f"{task}: checkpoint {epoch}")
            audit = read_json(audit_path)
            require(audit["epoch"] == epoch and audit["global_step"] == epoch * loader["actual_num_training_batches_per_rank"], f"{task}: epoch {epoch} step")
            require(audit["world_size"] == 8 and audit["frozen_state_unchanged"] and audit["frozen_hashes"] == frozen, f"{task}: epoch {epoch} frozen identity")
        require(read_json(run / "epoch_10_audit.json")["global_step"] == end["global_step"], f"{task}: epoch 10 end")
        entry = {"run_dir": str(run), "recipe": {"path": str(recipe_path), "sha256": digest(recipe_path), "full_t0_sha256": recipe["full_t0_sha256"], "full_t0_checkpoint": recipe["full_t0_checkpoint"], "stablewm_repo": recipe["stablewm_repo"], "reference_run": recipe["reference_run"]}, "loader": {"path": str(loader_path), "sha256": digest(loader_path), **loader}, "train_end_audit": {"path": str(end_path), "sha256": digest(end_path), **end}, "epochs": {}}
        baseline = {}
        panel = {}
        manifest = None
        stable_ref = None
        for epoch in EPOCHS:
            result_path = Path(item["evaluations"][str(epoch)]["result"])
            require(result_path == run / "fixed_target_eval" / f"epoch_{epoch:02d}" / "result.json", f"{task}: result path")
            result = read_json(result_path)
            require(result["schema"] == "contextworld.fixed_target_control.v1" and result["task"] == task, f"{task}: result identity")
            require(result["t0_checkpoint_sha256"] == recipe["full_t0_sha256"], f"{task}: T0 identity")
            require(result["new_checkpoint_sha256"] == digest(run / f"weights_epoch_{epoch}.pt"), f"{task}: epoch {epoch} checkpoint hash")
            require(result["visual_identity"] == {"tensor_and_buffer_count": 207, "bitwise_equal": True}, f"{task}: visual identity")
            require(result["no_fitting"] and result["no_test_read"], f"{task}: evaluation scope")
            require(all(x["before"] == x["after"] for x in result["model_state_unchanged"].values()), f"{task}: evaluation mutated model")
            manifest = manifest or result["panel_manifest_sha256"]
            stable_ref = stable_ref or result["stable_ref"]
            require(result["panel_manifest_sha256"] == manifest and result["stable_ref"] == stable_ref, f"{task}: manifest or code changed")
            compact_splits = {}
            for split, pairs in SPLITS.items():
                raw = result["splits"][split]
                require(raw == item["evaluations"][str(epoch)]["splits"][split], f"{task}: queue/result mismatch")
                require(raw["pairs"] == pairs and raw["rows"] == 2 * pairs and raw["action_blocks_equal_per_pair"], f"{task}: {split} panel size")
                require(raw["t0_cache_sha_and_meta_valid"] and raw["t0_cache_comparison"]["target_max_abs_difference"] == 0, f"{task}: {split} T0 cache")
                if split == "development":
                    require(raw["t0_cache_comparison"]["prediction_max_abs_difference"] == 0, f"{task}: development T0 prediction")
                panel.setdefault(split, raw["panel_sha256"])
                require(raw["panel_sha256"] == panel[split], f"{task}: {split} panel changed")
                baseline.setdefault(split, {"old": raw["old"], "B": raw["B_conditional_mean_error"]})
                require(raw["old"] == baseline[split]["old"] and raw["B_conditional_mean_error"] == baseline[split]["B"], f"{task}: {split} T0 changed")
                compact_splits[split] = {"panel_sha256": raw["panel_sha256"], "pairs": pairs, "rows": raw["rows"], "B_conditional_mean_error": raw["B_conditional_mean_error"], "old": {k: raw["old"][k] for k in METRICS}, "new": {k: raw["new"][k] for k in METRICS}, "deltas": {k: raw[k] for k in DELTAS}}
                if epoch == 1:
                    rows.append({"task": task, "split": split, "epoch": 0, "pairs": pairs, "response_nre": raw["old"]["response_nre"], "full_error_over_B": raw["old"]["full_error_over_B"], "history_benefit_over_B": raw["old"]["history_benefit_over_B"], "B_conditional_mean_error": raw["B_conditional_mean_error"]})
                rows.append({"task": task, "split": split, "epoch": epoch, "pairs": pairs, "response_nre": raw["new"]["response_nre"], "full_error_over_B": raw["new"]["full_error_over_B"], "history_benefit_over_B": raw["new"]["history_benefit_over_B"], "B_conditional_mean_error": raw["B_conditional_mean_error"]})
            entry["epochs"][str(epoch)] = {"result_path": str(result_path), "result_sha256": digest(result_path), "checkpoint_path": str(run / f"weights_epoch_{epoch}.pt"), "checkpoint_sha256": result["new_checkpoint_sha256"], "splits": compact_splits}
        entry["panel_manifest_sha256"] = manifest
        entry["panel_sha256"] = panel
        entry["stable_ref"] = stable_ref
        archive["tasks"][task] = entry
    return archive, rows


def render_csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def table(archive):
    print(render_table(archive), end="")


def render_table(archive):
    lines = ["| 任务 | 阶段 | 响应 NRE ↓ | 完整 E/B ↓ | 历史收益 G ↑ |", "|---|---|---:|---:|---:|"]
    names = {"contact_friction": "接触摩擦", "robot_arm_mass": "机械臂质量"}
    for task in TASKS:
        epochs = archive["tasks"][task]["epochs"]
        first = epochs["1"]["splits"]["development"]
        for stage, values in [("T0", first["old"])] + [(f"混合训练 {e} 轮", epochs[str(e)]["splits"]["development"]["new"]) for e in EPOCHS]:
            lines.append(f"| {names[task]} | {stage} | {values['response_nre']:.3f} | {values['full_error_over_B']:.3f} | {values['history_benefit_over_B']:.3f} |")
    return "\n".join(lines) + "\n"


def update_doc(path, block, check):
    source = path.read_text()
    require(source.count(BEGIN) == source.count(END) == 1, f"{path}: expected one generated block")
    before, rest = source.split(BEGIN, 1)
    current, after = rest.split(END, 1)
    replacement = "\n" + block.rstrip() + "\n"
    if check:
        require(current == replacement, f"{path}: generated block differs")
    else:
        path.write_text(before + BEGIN + replacement + END + after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="validate saved JSON/CSV against source results")
    parser.add_argument("--render-doc", action="store_true", help="update or check the generated Markdown block")
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()
    archive, rows = build(args.queue)
    json_text = json.dumps(archive, indent=2, ensure_ascii=False) + "\n"
    csv_text = render_csv(rows)
    json_path, csv_path = args.output_dir / "summary.json", args.output_dir / "summary.csv"
    if args.check:
        require(json_path.read_text() == json_text and csv_path.read_text() == csv_text, "saved summary differs from source")
        print("Saved JSON and CSV match all six raw results and ten audited epochs per task.")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json_text)
        csv_path.write_text(csv_text)
        print(f"Archived {len(EPOCHS) * len(TASKS)} results, {len(rows)} split rows.")
    if args.render_doc:
        update_doc(args.doc, render_table(archive), args.check)
    table(archive)


if __name__ == "__main__":
    main()
