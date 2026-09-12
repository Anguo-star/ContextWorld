# ContextWorld v3 HF dataset guide

This document has two audiences. Dataset users should read **Release layout**
and **Downloading**. Repository maintainers who prepare the Hugging Face
release candidate should also read **Building the release candidate**.

The v3 release candidate lives at
`artifacts/releases/ContextWorld-v3-hf` (the root of the future HF dataset
repo). It merges the frozen **Training and Development** splits of all nine
components with the frozen **Test** artifacts, byte-for-byte, so the sealed v3
baseline
(`contextworld_joint_scratch_v1_reference_results_freeze_v3`) and the repo's
loader semantics are preserved exactly. Development is for model/recipe
selection; Test is for final reporting only. This guide describes packaging
and loading, not data generation (see
[Data generation methodology](Data_Generation.md)).

## Release layout

```text
ContextWorld-v3-hf/
├── README.md               # rendered HF dataset card (see template below)
├── LICENSE                 # code license (MIT)
├── DATA_LICENSE            # synthetic data license (CC BY 4.0)
├── NOTICE                  # attribution
├── VERSION.json
├── task_registry.json      # entry point for software
├── manifest.jsonl
├── manifest.sha256
├── inventory.json          # generated split/file inventory (no hardcoded counts)
├── normalizers/
├── components/<component>/v1/{training,development}/   # frozen raw Training + Development
│     (tworoom-speed, tworoom-door, tworoom-action-delay, tworoom-portal-exit,
│      pusht-action-strength, pusht-contact-friction, pusht-motion-damping,
│      reacher-arm-mass, cube-gripper-carry; legacy Development leftovers excluded)
├── artifacts/evaluation/... and artifacts/synthesis/... # frozen Test, raw paths unchanged
├── provenance/
│   ├── ContextWorld-v1/{manifest.jsonl,manifest.sha256,task_registry.json}
│   └── ContextWorld-v1-full/{manifest.jsonl,manifest.sha256,task_registry.json}
└── reference/contextworld_joint_scratch_v1_reference_results_freeze_v3.json
```

The `reference/` file is an exact copy of the sealed v3 freeze; verify it
against SHA-256
`01298ca407c4a72bdd9a1238879ae7109b1141a7cfdd2b25065cb0bdec87fef0`.

## Downloading

Files are distributed in their **native Lance/JSON form, byte-for-byte**. No
conversion is applied, so a released revision reproduces the exact bytes the
sealed v3 baseline and the repo loaders expect. This bundle does not claim
`datasets.load_dataset()` compatibility, Parquet conversion, or dataset-viewer
support. Its card sets `viewer: false` using the
[official viewer configuration](https://huggingface.co/docs/hub/datasets-viewer-configure).
A derived Parquet view, if ever needed, would be a separately
versioned artifact and is unnecessary for this release.

Download an immutable revision with `huggingface_hub` (official guides:
[download files](https://huggingface.co/docs/huggingface_hub/guides/download),
[dataset cards](https://huggingface.co/docs/hub/datasets-cards),
[adding datasets](https://huggingface.co/docs/hub/datasets-adding), which
explicitly permits non-`datasets`-library workflows):

```bash
export HF_DATASET_REPO='ORG/DATASET'        # replace with the published repository
export HF_DATASET_REVISION='COMMIT_SHA'     # replace with its immutable revision
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["HF_DATASET_REPO"],
    repo_type="dataset",
    revision=os.environ["HF_DATASET_REVISION"],
    local_dir="./ContextWorld-v3-hf",
)
PY
```

Then verify byte integrity offline with the repo script (see
**Building the release candidate**): `--verify --output ./ContextWorld-v3-hf`
must pass without network access.

## Loading the data

The prepared local candidate and the downloaded snapshot use the same native
files and relative paths. From this checkout, select the current candidate:

```bash
export CONTEXTWORLD_BENCHMARK_ROOT="$(pwd)/artifacts/releases/ContextWorld-v3-hf"
```

After downloading, point that same variable at the downloaded directory:

```bash
export CONTEXTWORLD_BENCHMARK_ROOT=/absolute/path/ContextWorld-v3-hf
contextworld-benchmark info
```

Training and post-training ICL evaluation share one root resolver. An explicit
`--benchmark-root`/`CONTEXTWORLD_BENCHMARK_ROOT` takes priority. Otherwise they
use `<CONTEXTWORLD_DATASET_ROOT>/ContextWorld-v3-hf` if present, then the local
`artifacts/releases/ContextWorld-v3-hf` candidate. They never automatically
select the legacy `ContextWorld-v1` directory. An explicit directory can have
any name; no conversion or changes to the files inside it are required.

For built-in Stable-WorldModel training, select a task and family through the
ContextWorld launcher (do not set `CW_DATASET`; the launcher reads
`task_registry.json` and builds a manifest-bound dataset request):

```text
CW_TASK=action_strength
CW_FAMILY=lewm
CW_TRAINING_TRACK=joint_scratch_v1
CONTEXTWORLD_BENCHMARK_ROOT=/absolute/path/ContextWorld-v3-hf
CONTEXTWORLD_DATASET_ROOT=/absolute/path/data/world_model
CW_CHECKPOINT_ROOT=/absolute/path/checkpoints/lewm-contextworld
```

`CONTEXTWORLD_DATASET_ROOT` supplies the separately downloaded original
environment H5 files for naive training, original/synthetic mixtures, and
optional CEM evaluation. These files are not part of the ICL dataset snapshot.

External models implement `LatentWorldModelAdapter`; see the
[external model adapter contract](External_Model_Adapter_Contract.md). Score
on **Development first** for all method/recipe/checkpoint selection:

```bash
python -m contextworld.benchmarks.external_model_cli \
  --benchmark-root "$CONTEXTWORLD_BENCHMARK_ROOT" \
  --evaluation-split development \
  --task contact_friction \
  --adapter your_package.module:YourAdapter \
  --checkpoint /path/to/checkpoint \
  --model-name your-model \
  --output /path/to/dev-result.json
```

Only after all choices are frozen, run the public offline **Test** split once
as the final report (same command with `--evaluation-split test`). Test
output is an offline final report, not a hosted-scoreboard row.

## What the bundle does not contain

The distribution intentionally excludes: model checkpoints and training logs;
internal `score_receipts`; third-party
source checkouts; original LeWM training datasets; and the original-environment
datasets used for CEM evaluation (they stay outside the v3 release). Native
raw storage only; no derived views are shipped.

Some source-bound Test collections retain historical summary metadata, including
Speed's `final_summary.json`. It is copied as source material and must not be
read as the current v3 reference score table.

## Building the release candidate

For repository maintainers only; end users should download a published
revision instead. Source data is never modified — the script only reads the
frozen roots and writes a new candidate directory.

The candidate is built by `scripts/prepare_contextworld_hf_release.py`
(8 workers by default). First review the plan without copying payloads:

```bash
python scripts/prepare_contextworld_hf_release.py \
  --development-root /absolute/path/to/ContextWorld-v1 \
  --test-root /absolute/path/to/ContextWorld-v1-full \
  --output artifacts/releases/ContextWorld-v3-hf
```

Then build **once** with `--execute` (the script rejects an existing output
directory). On a managed mount that rejects renaming populated directories,
use `--execute --direct-write`; this still requires a new directory and rejects
incomplete builds during verification. After building, verify offline byte hashes:

```bash
python scripts/prepare_contextworld_hf_release.py \
  --verify --output artifacts/releases/ContextWorld-v3-hf

# With the repository's [eval] dependencies installed, read each task/split:
python scripts/check_contextworld_hf_loading.py \
  --benchmark-root artifacts/releases/ContextWorld-v3-hf
```

`--verify` re-checks every file against the recorded SHA-256 and works both
locally and on a `snapshot_download` result.

**Do not** use the legacy `scripts/export_contextworld_hf_clean.py` (or its
`--refresh-metadata` mode) to regenerate or refresh this v3 candidate: its
clean-export config still contains obsolete Speed/Door Development readers
and does not describe the merged v3 layout.

### Dataset card

`README.md` is rendered from
[`templates/ContextWorld_HF_Dataset_Card.md`](templates/ContextWorld_HF_Dataset_Card.md)
by substituting exactly three tokens: `{{BASELINE_ID}}`,
`{{BASELINE_SHA256}}`, and `{{SPLIT_INVENTORY}}` (generated from
`inventory.json`; no file counts or sizes are hardcoded). The card is
metadata/documentation only and must stay factual.

## Publication status

The sealed v3 baseline passed final acceptance on 2026-09-10; building this
candidate is preparation only. The local candidate contains 4,757 payload files
(20,489,896,395 bytes); all 4,785 distributed manifest entries passed hash
verification, and the nine tasks passed 27 native split-loading checks.
See the [local acceptance record](reference/contextworld_v3_hf_candidate_2026-09-11.json).
Remaining release blockers are tracked in
[ContextWorld_Public_v1_Release_Readiness.md](ContextWorld_Public_v1_Release_Readiness.md):
the HF namespace/repo and immutable revision/URL are **not yet supplied**, and
the post-upload download smoke test and the publication record remain pending.
The publication record must bind the data revision, code revision, manifest
checksum and frozen baseline. Building the directory does not upload anything.
