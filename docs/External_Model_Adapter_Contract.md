# ContextWorld external model adapter contract

This document defines the runtime interface that an external latent world
model implements to use ContextWorld Development scorers. The checks described
below are enforced by `contextworld-external-eval`.

The `ContextWorld-v1` bundle does not restrict evaluation to LeWM, PLDM or
PreJEPA. Those names identify built-in reference integrations. An external
model may train with its own code and data loader; to use the public
Development scorers it supplies the adapter below. The adapter translates the
model interface, not the benchmark definition, and it never receives Public
Test examples.

## Required interface

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

A single adapter can serve seven of the nine if it declares
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

- **No decoder.** `decoder_required=False`. Predictions are compared against
  targets encoded by *your own* frozen encoder, in your own latent space.
  Latent width and model family are deliberately unconstrained.
- **No pixel reconstruction, no reward, no value head.**

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
the adapter's job, so the distributed dataset remains
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

The task determines which shape is supplied. `build_request` reads the action
geometry from the versioned task configuration rather than the command line.
This ensures that external models and reference models use the same
normalisation. An adapter intended to support all tasks must accept both
shapes.

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

**Route A — Stable-WorldModel family.** Reuse the family's native
checkpoint loader and its public `encode` / `rollout` interfaces where
available. A thin ContextWorld adapter must still verify the versioned input
contract—preprocessing, history length, action-block geometry, required
context streams, and prediction key—because these details differ across
families. `model_config_name` alone is therefore not a compatibility
guarantee.

**Route B — fully external model.** Implement the five members directly. You
own preprocessing, checkpoint loading and latent representation. Use this
only when the model cannot be expressed as a Stable-WorldModel family.

### DINO-WM / PreJEPA

Stable-WorldModel provides the DINOv2-based `prejepa` family. ContextWorld
includes a native PreJEPA adapter for checkpoints that satisfy the official v1
input interface.

That contract is intentionally narrow: the ICL scorer supplies RGB history
and raw actions only. A checkpoint is eligible only when its predictor
requires no additional context stream and its trained history length, action
block, and action dimension match the selected task. In particular,
original-data PreJEPA checkpoints that require `proprio` or `observation`
cannot be scored through the official v1 interface. The result records the
machine-readable status `not_compatible`, meaning that the checkpoint's input
requirements do not match the benchmark interface. It is not a model failure
or a benchmark score. A direct evaluation request rejects such a checkpoint by
default.

For diagnostic comparison only, callers may explicitly set
`--prejepa-missing-context-policy normalized_zero`. The adapter then supplies
zeros in each missing stream's normalized model-input space. The output is
labelled `external_diagnostic_non_frozen_v1` so downstream tools cannot confuse
it with an official result. This route does not expose simulator state or
change model weights. It must be reported as an auxiliary analysis, not as an
official v1 ICL score. A History=3 PreJEPA
checkpoint can be evaluated by the History=7 Action Delay scorer only with the
additional explicit `--history-adapter h3_tail_projection` option.

For an eligible native `.pt` checkpoint with its accompanying `config.json`,
the built-in adapter loads Stable-WorldModel's public `encode` and `rollout`
interfaces. It encodes targets from the pixel stream and reads predictions
from `predicted_pixels_emb`. It does not score action embeddings as visual
state.

The adapter also preserves the temporal meaning of the request: for history
length `H`, it passes the first `H-1` normalized action blocks as
`action_history` and the remaining blocks as future actions. This is why the
adapter returns `T-(H-1)` predictions for `T` supplied action blocks. The
upstream rollout cache is cleared for every independent benchmark bundle.

For example, a compatible state-free checkpoint can be evaluated with:

```bash
python -m contextworld.benchmarks.external_model_cli --task speed --adapter prejepa \
    --checkpoint /path/to/weights_epoch_10.pt --model-name dino-wm
```

An original state-conditioned checkpoint can be inspected diagnostically with:

```bash
python -m contextworld.benchmarks.external_model_cli --task speed --adapter prejepa \
    --checkpoint /path/to/weights_epoch_10.pt --model-name dino-wm \
    --prejepa-missing-context-policy normalized_zero
```

ContextWorld's training launcher delegates to the selected model family's
native trainer. The adapter and evaluator do not replace the model's training
implementation.

### Contributing a built-in integration

External users do not need to modify ContextWorld: an import path or installed
entry point is sufficient. A model family intended for inclusion as a built-in
integration should be implemented in its own module, cover the three required
geometries, and include protocol-validation and frozen-state tests. Repository
maintainers then register the family with the CLI without changing the public
adapter base class.

## Invocation

```bash
python -m contextworld.benchmarks.external_model_cli \
    --task speed \
    --adapter your_package.module:YourAdapter \
    --checkpoint /path/to/weights.pt \
    --model-name dino-wm
```

`--adapter` accepts a built-in name (`lewm`, `pldm`, `prejepa`), an import path
`package.module:ClassName`, or an installed `contextworld.adapters` entry
point. Built-in names cannot be overridden by an installed package.

Development results from an external adapter carry the machine-readable status
`external_unofficial`. This distinguishes user-run Development evaluations
from sealed Public Test submissions; it does not imply that the scorer or task
definition is unofficial.
