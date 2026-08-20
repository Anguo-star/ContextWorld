# ContextWorld adapter contract — what an external model must satisfy

Everything below was read out of the code, not from documentation. It is what
`contextworld-external-eval` will actually check at runtime.

## The five members

Subclass `contextworld.benchmarks.adapters.LatentWorldModelAdapter` and
implement:

| member | kind | contract |
|---|---|---|
| `protocol` | property → `AdapterProtocol` | declares your geometry |
| `metadata` | property → `dict` | JSON-serialisable provenance |
| `encode_pixels(pixels, *, batch_size)` | method | `[B,H,W,3]` uint8 → `[B,...]` latents |
| `rollout_latents(input_pixels, raw_action_blocks, *, batch_size)` | method | see below |
| `frozen_state_hash()` | method | hash params+buffers, must not mutate state |

Plus a constructor. The registry prefers `from_contextworld_request(request)`
and falls back to `from_checkpoint(...)`.

## Geometry per task

`validate_adapter_protocol` enforces exact equality on the first three and
`>=` on the last.

| task | history | block steps | action_dim | min future blocks |
|---|---|---|---|---|
| speed | 3 | 5 | 2 | 5 |
| door | 3 | 5 | 2 | 1 |
| action_delay | **7** | 5 | 2 | 1 |
| portal_exit | 3 | 5 | 2 | 1 |
| action_strength | 3 | 5 | 2 | 1 |
| contact_friction | 3 | 5 | 2 | 1 |
| motion_damping | 3 | 5 | 2 | 1 |
| robot_arm_mass | 3 | 5 | 2 | 1 |
| cube_gripper_carry | 3 | 5 | **5** | 1 |

Two things to note. `action_delay` needs **history 7** — that is why the
built-ins ship `History7` variants. `cube_gripper_carry` has **action_dim 5**,
every other task is 2.

A single adapter can serve eight of the nine if it declares
`history_tokens=3, action_block_raw_steps=5, action_dim=2,
future_action_blocks=5`. Cube needs a separate instance (action_dim 5),
action_delay needs another (history 7).

The built-ins are split exactly this way, which is the pattern to copy:

| built-in | history | future blocks | action_dim |
|---|---|---|---|
| `StableWorldModelLeWMAdapter` | 3 | 5 | 2 |
| `StableWorldModelPLDMAdapter` | 3 | 5 | 2 |
| `StableWorldModelLeWMHistory7Adapter` | 7 | 3 | 2 |
| `StableWorldModelPLDMHistory7Adapter` | 7 | 3 | 2 |

So the practical shape is one class parameterised by
`required_history_tokens` / `raw_action_dim`, instantiated three ways.

## What the benchmark never asks for

* **No decoder.** `decoder_required=False`. Predictions are compared against
  targets encoded by *your own* frozen encoder, in your own latent space.
  Latent width and model family are deliberately unconstrained.
* **No pixel reconstruction, no reward, no value head.**

This is what makes DINO-WM a clean fit: the frozen-DINO-encoder-plus-latent-
predictor shape is exactly the shape the benchmark expects.

## Array shapes

```
encode_pixels(pixels=[B,H,W,3] uint8)              -> [B, D...]   (your latent)
rollout_latents(
    input_pixels     = [B, history, H, W, 3] uint8,
    raw_action_blocks= [B, T, 5, action_dim] float32,
) -> [B, future, D...]
```

`raw_action_blocks` arrives as **raw environment actions**. Normalisation is
the adapter's job — this is deliberate, so the frozen dataset stays
model-agnostic.

The future-length rule, from the reference implementation:

```
expected_future = T - (history_tokens - 1)
```

History H consumes H−1 context actions, so T action tokens request
T−(H−1) future predictions. Return exactly that many; returning a different
count raises.

## Two construction shapes

The task set is split, and `AdapterRequest` rejects receiving both shapes at
once.

| shape | tasks |
|---|---|
| **normalizer artifact** (`action_normalizer`) | speed, door, action_delay |
| **explicit `action_mean`/`action_std`** | action_strength, contact_friction, motion_damping, portal_exit, robot_arm_mass, cube_gripper_carry |

`portal_exit` alone uses `std_unbiased`; the other five statistics tasks use
`std_population`.

You never choose which shape you get. `build_request` reads the action
geometry out of the task's **frozen release config**, not from the command
line, "so an external model is normalized exactly as the baselines were and
cannot quietly evaluate under a different contract". Your adapter must
therefore accept both shapes and normalise accordingly.

## Pixel preprocessing

`_preprocess_pixels` normalises with **ImageNet** mean/std. This is not an
arbitrary ContextWorld choice — it is what Stable-WorldModel trains with.
`scripts/train/lewm.py`, `pldm.py` and `prejepa.py` all build their transform
from `spt.data.dataset_stats.ImageNet`, so the benchmark's preprocessing is
aligned with the training pipeline by construction.

An adapter for a Stable-WorldModel family should therefore reuse
`_preprocess_pixels` as-is. Only a model trained *outside* Stable-WorldModel
under different statistics needs its own chain — and in that case the
mismatch is the integrator's to resolve, not the benchmark's.

## Two routes for a new model family

**Route A — Stable-WorldModel family (preferred).** If the model exists as a
Stable-WorldModel world model, the work is small. `_load_model` already
instantiates any family from `scripts/train/config/{name}.yaml`, so a new
adapter is mostly `model_config_name = "<family>"` plus the geometry class
attributes. Normalisation, checkpoint loading and the frozen-state hash all
come for free, and the evaluation is aligned with the baselines by
construction.

**Route B — fully external model.** Implement the five members directly. You
own preprocessing, checkpoint loading and latent representation. Use this
only when the model cannot be expressed as a Stable-WorldModel family.

### DINO-WM specifically

Stable-WorldModel already ships a DINOv2-based world model: the `prejepa`
family. Its `scripts/train/config/prejepa.yaml` matches DINO-WM's published
architecture closely — `dinov2_small`, `patch_size: 14`, `image_size: 224`,
`history_size: 3`, `frameskip: 5`, predictor `depth 6 / heads 16 /
mlp_dim 2048 / dim_head 64`, action and proprio encoding width 10.

**This is already implemented.** `contextworld/benchmarks/prejepa_adapters.py`
provides all eight variants, and `--adapter prejepa` works for every task:

```bash
contextworld-external-eval --task speed --adapter prejepa \
    --checkpoint /path/to/prejepa.ckpt --model-name dino-wm
```

Train the checkpoint with Stable-WorldModel's own
`scripts/train/prejepa.py`; ContextWorld does not train.

Three `prejepa` API differences are absorbed by a rollout shim, each with a
silent failure mode if left unhandled:

* `rollout` takes no `history_size` argument.
* The visual stream is `predicted_visual`, not `predicted_emb`. Targets from
  `encode_pixels` are visual-only, so scoring the concatenated action slots
  would compare a model's action encoding against itself.
* `rollout` caches its initial embedding on `info['id']`/`info['step_idx']`,
  which benchmark bundles do not carry. The cache is dropped per call; a
  stale hit would score one bundle from another's initial state.

### Adding another family

Adapters for a new Stable-WorldModel family go in a **new module**, not in
`adapters.py` — that file's bytes are pinned by the speed and door release
configs, so editing it invalidates published provenance. Then add one entry
to `_BUILTIN_FAMILIES` in `external_model_cli.py`:

```python
_BUILTIN_FAMILIES = {
    "lewm":    (_ADAPTERS, "LeWM"),
    "pldm":    (_ADAPTERS, "PLDM"),
    "prejepa": (f"{_SCORE}.prejepa_adapters", "PreJEPA"),
}
```

Class names follow `StableWorldModel{infix}{task}Adapter`, so one entry
reaches all nine tasks. The CLI help text is derived from this table.

## Invocation

```bash
contextworld-external-eval \
    --task speed \
    --adapter your_package.module:YourAdapter \
    --checkpoint /path/to/weights.pt \
    --model-name dino-wm
```

`--adapter` accepts a built-in name (`lewm`, `pldm`), an import path
`package.module:ClassName`, or an installed `contextworld.adapters` entry
point. Built-in names cannot be overridden by an installed package.

Results are stamped `external_unofficial` and nested under a `result` key so
they cannot be replayed as a frozen submission. Running this touches no
hash-pinned file.
