# Design — Combine Graph-Param Buffers

## Context

Each scene MaterialX nodegraph currently gets a dedicated typed param buffer:

```slang
// generated_materials.slang (emit_megakernel_aggregator)
[[vk::binding(25, 0)]] StructuredBuffer<GraphParams_Marble_3D> graphParams_Marble_3D;
[[vk::binding(26, 0)]] StructuredBuffer<GraphParams_Tiled_Wood> graphParams_Tiled_Wood;
[[vk::binding(27, 0)]] StructuredBuffer<GraphParams_Tiled_Brass> graphParams_Tiled_Brass;

GraphParams_Marble_3D _graphParams_Marble_3D(uint matId) { return graphParams_Marble_3D[matId]; }
```

`GRAPH_BINDING_BASE = 25`; binding `25 + idx` per fragment
(`megakernel_sources.py`). The wavefront per-material shade modules
(`emit_wavefront_shade_module`) and the wavefront path pass bind the same
per-graph buffers by their Slang global name. Host fills one `StorageBuffer` per
target into `_graph_param_buffers[target]` and binds it at `25 + idx`
(`renderer._upload_graph_param_buffers`, `vk_compute`, `metal_compute`,
`metal_wavefront`).

On Metal each of those buffers consumes one argument-table slot (cap = 31,
indices 0–30). N graphs ⇒ N slots; the neural build needs 3 more for
`neuralWeights`/`Biases`/`Layers`. With N ≥ 2 the neural buffers spill past 30 →
`SLANG_FAIL`.

## Goals / Non-Goals

- **Goal**: graph params occupy exactly **one** buffer slot regardless of graph
  count, on both backends, with byte-identical shaded output.
- **Goal**: the neural + online-training wavefront Metal build fits the 31-slot
  cap for the reported command and for any future graph count.
- **Non-goal**: combining the neural weight/bias/layer trio (independent slot
  win, separate change). Changing graph fragment sources, the per-material shade
  partitioning, or the `evalSceneGraph` dispatch shape.

## Decision: one matId-major byte buffer + a single shared stride

A probe of Slang's `ByteAddressBuffer.Load<T>` settled the design: `Load<T>`
reads a struct in **scalar layout on both the SPIR-V and Metal targets**
(verified — `sizeof` and the emitted field offsets are tight/scalar on each).
That has two consequences that simplify everything:

1. The per-slot stride is **target-independent** (= the host's existing
   `scalar_stride`), so a single literal works in the shader for both backends.
2. The former Metal **MSL repack** of graph params (the `float3`→16 B relocation
   in `_upload_graph_param_buffers`) is no longer needed — the host packs one
   scalar blob that both backends read identically.

A material maps to **exactly one** graph, so the combined buffer is laid out
**matId-major** with a single fixed stride (the max scalar `GraphParams` size
over the scene's graphs, 16-aligned): slot `matId` holds that material's params.

```slang
[[vk::binding(25, 0)]] ByteAddressBuffer graphParamsCombined;
static const uint GRAPH_PARAM_STRIDE = 32u;   // max scalar GraphParams size, 16-aligned

GraphParams_Marble_3D _graphParams_Marble_3D(uint matId)
{
    return graphParamsCombined.Load<GraphParams_Marble_3D>(matId * GRAPH_PARAM_STRIDE);
}
```

**Why a single stride, not per-graph base offsets** — `material_capacity` can
grow at runtime *without* re-emitting the shaders (the resize path only
reallocates buffers). Per-graph base offsets `cap * Σ stride` would bake
`material_capacity` into shader constants and go stale on a resize. The
matId-major layout needs **no** per-graph base (base 0 for every graph) and its
stride depends only on the scene's graph SET — which only changes on a scene
reload, and that already re-emits. So the baked constant stays valid across a
capacity resize; only the host buffer is reallocated (larger).

`graph_param_combined_stride(fragments)` is the single source of truth, called
by both the emitter (bakes `GRAPH_PARAM_STRIDE`) and the host (sizes + packs).
The Vulkan-correct path already proves `scalar_stride` == the Slang struct's
layout, and `Load<T>` reads that same scalar layout — the per-material A/B
render is the guardrail (a stride/offset error collapses a material to black).

**Alternatives rejected**

- *Per-graph (base, stride) offset table*: needs `material_capacity` baked in →
  stale on resize. The matId-major single stride avoids it entirely.
- *Fold more fixed globals out* (extend `SKINNY_METAL_*` compile-outs): buys a
  constant, not an N-independent bound; breaks again at higher graph counts.
- *Combine the neural trio instead*: frees 2 slots but graph count still grows
  the table. Orthogonal; not pursued (the env-CDF combine, below, is the
  cleaner read-only −1 for the residual baseline slot).

## The residual slot — env-CDF combine

Graph-param combine makes the graph contribution a constant 1 slot, but the
neural + online-training wavefront flavour is **1 over at baseline** (even with a
single graph). Metal assigns buffer indices by kernel-parameter order, not
vk::binding, so re-slotting bindings is unreliable — only **reducing the bound
count by one** deterministically lowers the max index. The two environment
importance-sampling CDFs are a clean read-only pair, built once together, so
they fold into one buffer with no gating:

```slang
[[vk::binding(31)]] StructuredBuffer<float> envDistCdf;   // [marginal | conditional]
static const int ENV_COND_CDF_BASE = ENV_DIST_H + 1;      // conditional element offset
```

`envSearchCdf` already takes a base offset, so the conditional reads just add
`ENV_COND_CDF_BASE`. Host concatenates `marginal + conditional` into one
`env_dist_buffer`; binding 32 is retired. Unconditional (both backends, both
modes) — the env A/B is covered by the same per-material render check.

## Layout & alignment

- Combined graph buffer size = `material_capacity * GRAPH_PARAM_STRIDE + pad`,
  grown with `material_capacity`. Each slot 16-aligned ⇒ `Load<T>` aligned.
- Zero-graph scene: no `graphParamsCombined` decl emitted and no buffer bound
  (the accessors that would reference it are absent too).
- Env buffer: `(ENV_DIST_H+1) + ENV_DIST_H*(ENV_DIST_W+1)` floats; conditional
  starts exactly at element `ENV_COND_CDF_BASE = ENV_DIST_H+1`, so a plain
  `marginal + conditional` byte concat aligns.

## Backend bind sites (all bind-by-name, single slot now)

- `vk_compute.py`: graph-param descriptor spec collapses from `for idx … binding
  25+idx` to one `binding 25` storage-buffer entry (`graphParamsCombined`).
- `metal_compute.py` / `metal_wavefront.py`: `graph_param_layouts` reflection
  keyed per target → a single combined-buffer reflection + the emitted offset
  table; the per-dispatch bind map gains one `graphParamsCombined` entry instead
  of N `graphParams_<name>` entries.
- `renderer.py`: `_upload_graph_param_buffers` packs the concatenated blob into
  one `StorageBuffer` (`self._graph_params_combined`) at the codegen offsets;
  `_scene_graph_bindings` reduces to the single slot.

## Guardrail (kept from metal-record-drain)

The post-link reflected slot-count check stays: it SHALL raise
`RuntimeError(kernel, count)` when any entry point exceeds 31 buffers, so a future
regression surfaces as a named error, not a raw `SLANG_FAIL`. After this change
the heaviest neural wavefront kernel for an N-graph scene drops from
`fixed + N + 3` to `fixed + 1 + 3` buffers.

## Test strategy

- **Codegen unit** (no GPU): `emit_megakernel_aggregator` / shade-module emit
  one `ByteAddressBuffer` at slot 25 + correct `GRAPH_BASE`/`GRAPH_STRIDE`
  constants for a 3-fragment fixture; offsets are 16-B aligned and monotonic.
- **Guarded GPU build** (thermal rule — one Metal compile process,
  `scripts/guarded_metal.sh`, `-m 'not gpu'` elsewhere): the reported command's
  wavefront path pass builds (`wfPathShadeFlat` slot count ≤ 31, logged).
- **Vulkan/Metal A/B**: `three_materials_demo.usda` converged image equivalence
  before vs after (pure data-relocation refactor — same bytes, same shading).
- **Bit-identity**: single-graph and zero-graph scenes render unchanged; default
  (non-neural) Metal + Vulkan builds compile and match.
