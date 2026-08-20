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
