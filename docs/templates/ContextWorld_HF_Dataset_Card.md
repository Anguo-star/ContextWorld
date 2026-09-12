---
license: cc-by-4.0
pretty_name: ContextWorld v3
viewer: false
tags:
  - image
  - robotics
  - world-model
  - image-observations
  - latent-dynamics-model
  - offline-benchmark
  - in-context-learning
  - planning
---

# ContextWorld v3

ContextWorld is a research benchmark of nine synthetic world-model tasks with
image observations and low-dimensional actions. Each task pairs history
conditioning with a controllable physical property, so that a latent world
model can be scored on whether its predictions actually use history. The code,
loaders, and reference training recipes live at
[github.com/Anguo-star/ContextWorld](https://github.com/Anguo-star/ContextWorld).

- **Frozen baseline identity:** `{{BASELINE_ID}}`
- **Freeze SHA-256:** `{{BASELINE_SHA256}}`

## Data inventory

The counts below refer to storage files, not benchmark queries or trajectories.

{{SPLIT_INVENTORY}}

This dataset ships native Lance/JSON files, byte-for-byte as used by the
benchmark loaders. Download a fixed revision with
`huggingface_hub.snapshot_download(repo_type="dataset", revision=...)`; the
files are not `datasets.load_dataset()`-compatible Parquet views and no
dataset-viewer rendering is claimed.

## Tasks

| # | Component | Capability | History length |
|---|---|---|---|
| 1 | tworoom-speed | movement-speed inference and extrapolation | 3 |
| 2 | tworoom-door | hidden-passage (door) rule | 3 |
| 3 | tworoom-action-delay | delayed-action response (h1, groups 0, 1, 2, 3, 4, 5–10) | 7 |
| 4 | tworoom-portal-exit | portal-exit outcome | 3 |
| 5 | pusht-action-strength | action-strength response | 3 |
| 6 | pusht-contact-friction | contact-friction response | 3 |
| 7 | pusht-motion-damping | motion-damping response | 3 |
| 8 | reacher-arm-mass | robot-arm-mass response | 3 |
| 9 | cube-gripper-carry | binary gripper carry rule (History=3) | 3 |

`task_registry.json` is the authoritative per-task contract (environment,
history length, action dimension, payload layout, members, selection rule,
normalization, runtime views).

## Splits and protocol

- **Training / Development:** used for training, model, recipe, and checkpoint
  selection. Development scoring is model-independent via the
  `LatentWorldModelAdapter` contract.
- **Test:** frozen, offline, final reporting only. Do not select on Test.
- The current 6-row in-context-learning matrix reports 7 tasks on Test and
  friction/damping on Development; historical Test disclosures for those two
  tasks are recorded in the repository, not reported as current Test passes.

## Reference results

The reference results come from the joint-scratch training track with three
independent training seeds per method; primary scores are reported as
**mean ± sample standard deviation across seeds**. The count of seeds passing
the full gate is recorded as supplemental information, not as a score. A
model not passing a gate is a valid reference point for measuring whether new
methods improve that metric; that result alone does not establish a data defect.
All frozen bounds,
thresholds, per-task history provenance, and known claim limits are fixed in
`{{BASELINE_ID}}` in the repository (`configs/benchmark/` and
`reference/` in this bundle); no public archive DOI exists for this freeze,
and none is claimed.

Historical runtime provenance is incomplete and external independent reproduction
is not claimed. The reference JSON retains historical source paths as identity
records; those paths are not required locations for loading this dataset.
Some byte-bound Test collections retain historical summary metadata, including
Speed's `final_summary.json`; these are not the current v3 reference scores.

## Download and load

After publication, set `HF_DATASET_REPO` and `HF_DATASET_REVISION` to the dataset
repository and its immutable commit. Download the complete native snapshot:

```python
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_DATASET_REPO"],
    repo_type="dataset",
    revision=os.environ["HF_DATASET_REVISION"],
    local_dir="./ContextWorld-v3-hf",
)
```

From the ContextWorld code checkout, verify and inspect the downloaded directory:

```bash
python scripts/prepare_contextworld_hf_release.py --verify --output /path/to/ContextWorld-v3-hf
export CONTEXTWORLD_BENCHMARK_ROOT=/absolute/path/to/ContextWorld-v3-hf
contextworld-benchmark info
```

Install the code package with `pip install -e ".[eval]"` for data reading and
scoring. The dataset repository and public revision are pending publication;
this local candidate does not claim an available download yet.

## Out of scope

The bundle intentionally contains no model checkpoints, no training logs, and
no original-environment datasets used for CEM planning evaluation.

## License and citation

Code is MIT-licensed; the synthetic data is released under CC BY 4.0 (see
`DATA_LICENSE` and `NOTICE`). To cite this dataset, reference the repository
by its title and URL together with the freeze identity:

```bibtex
@misc{contextworld_v3,
  title  = {ContextWorld (github.com/Anguo-star/ContextWorld)},
  howpublished = {\url{https://github.com/Anguo-star/ContextWorld}},
  note   = {Frozen baseline {{BASELINE_ID}}, SHA-256 {{BASELINE_SHA256}}}
}
```
