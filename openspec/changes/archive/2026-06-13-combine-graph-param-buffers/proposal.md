# Combine Graph-Param Buffers

## Why

The command

```
skinny --backend metal --execution-mode wavefront \
  --neural-trainer mlx --neural-handoff interop --online-training \
  --integrator path assets/three_materials_demo.usda
```

crashes at Metal pipeline creation:

```
wavefront_path.slang(217): error : 'buffer' attribute parameter is out of
bounds: must be between 0 and 30
  ...neuralWeights_1 [[buffer(31)]], neuralBiases_1 [[buffer(32)]],
     neuralLayers_1 [[buffer(33)]]...
→ create_compute_pipeline → SLANG_FAIL (metal_wavefront.py:130)
```

Metal caps a kernel's buffer argument table at 31 slots (indices 0–30). The
neural directional proposal's three frozen-weight buffers (`neuralWeights`,
`neuralBiases`, `neuralLayers`) land at indices 31/32/33 and overflow.

The existing slot-cap fold tricks (`SKINNY_METAL_NEURAL` compiles out
`toolBuffer`/`recordBuf`/`recordCounter`; `SKINNY_METAL_RECORDS` compiles out
`lightSplatBuffer`/`gizmoSegments`) were budgeted against the metal-neural-interop
/ metal-record-drain test scenes, which carried ~zero procedural MaterialX graph
materials. But every distinct scene MaterialX nodegraph emits **its own** typed
param buffer at `GRAPH_BINDING_BASE + idx` (binding 25, 26, 27, …) — a
scene-dependent, unbounded contributor to the argument table the fold budget
never reserved room for. `three_materials_demo.usda` carries three graph
materials (`Marble_3D`, `Tiled_Wood`, `Tiled_Brass`); two reach the flat shade
kernel and push the neural buffers past slot 30.

This is not specific to this asset: any neural + online-training wavefront render
on a Metal scene with ≥2 graph materials hits the cap. Folding more fixed globals
out only buys a constant; the graph-material count grows without bound.

## What Changes

- Replace the N per-graph typed `StructuredBuffer<GraphParams_X>` (bindings 25,
  26, 27, …) with **one** byte-addressed param buffer bound at a single slot
  (`GRAPH_BINDING_BASE`). The generated `_graphParams_X(matId)` accessors read
  their struct out of the combined buffer at a compile-time per-graph base offset
  plus `matId * sizeof(GraphParams_X)`. Graph-material count no longer grows the
  argument table — it stays at exactly one slot for any number of graphs.
- This drops the neural buffers' effective Metal index by (N − 1) for an
  N-graph scene. It is necessary but not sufficient on its own: the neural +
  online-training wavefront flavour is **one slot over the cap at its baseline**
  (even with a single graph material — which is why the metal-record-drain test
  scenes, carrying zero graph materials, fit at exactly 31). Reclaim that last
  slot by also folding the two environment importance-sampling CDF buffers
  (`envMarginalCdf` binding 31 + `envCondCdf` binding 32) into **one**
  concatenated buffer `envDistCdf` at binding 31 — the marginal CDF followed by
  the conditional CDF at element offset `ENV_DIST_H+1`. Both are read-only,
  built once together host-side, so this is an unconditional (both backends,
  both execution modes) clean −1 with no gating. Binding 32 is retired.
- Together the two combines clear the 31-slot cap for the reported command and
  for every future neural + multi-graph Metal scene, and free argument-table
  headroom on the megakernel and on Vulkan (fewer descriptors), at no functional
  cost (Metal `Load<T>` and the env CDFs read the same bytes as before).
- Host side: one combined `StorageBuffer` packed from the per-target param
  blobs at the codegen-emitted offsets, bound by the single Slang global name on
  both backends (megakernel aggregator, wavefront per-material shade passes, and
  the wavefront path pass). The per-target `_graph_param_buffers` upload becomes
  a single concatenated upload.
- Keep the existing `metal-record-drain` post-link reflected slot check as the
  guardrail: it SHALL still raise a clear `RuntimeError` (kernel name + count) if
  any future combination exceeds 31, rather than surfacing a raw `SLANG_FAIL`.
- Tests: a guarded GPU test that the failing command's pipeline builds on a
  Metal host; a Vulkan/Metal A/B that `three_materials_demo.usda` renders
  equivalently before/after the layout change (combined buffer is a pure
  refactor of where the same bytes live); a codegen unit test for the offset
  table; default-render bit-identity for single-graph and zero-graph scenes.
- Docs: `docs/Architecture.md` (descriptor binding map — single combined graph
  param slot), `docs/Wavefront.md` / `docs/Megakernel.md` (graph param layout),
  `CHANGELOG.md`, and the CLAUDE.md compatibility note if the supported-combo
  wording changes.

## Capabilities

### New Capabilities

(none — this hardens existing capabilities)

### Modified Capabilities

- `metal-backend`: the 31-buffer argument-table budget SHALL be independent of
  the scene's MaterialX graph-material count — graph params occupy exactly one
  buffer slot regardless of how many distinct graphs the scene carries, so the
  neural + online-training wavefront build fits the cap for any graph count.
- `per-material-pipeline`: the material code generator SHALL pack all per-graph
  param blocks into a single combined param buffer addressed by a compile-time
  per-graph offset, rather than one buffer binding per graph, while preserving
  the same per-graph fragment sources and per-material shade pipelines.

## Impact

- Codegen: `src/skinny/megakernel_sources.py`
  (`emit_generated_materials`, `emit_megakernel_aggregator`,
  `emit_wavefront_shade_module`, `GRAPH_BINDING_BASE` → single binding + offset
  table emission).
- Shaders: the generated `generated_materials.slang` / `wavefront/shade_*.slang`
  param declarations + `_graphParams_X` accessors (now `Load<T>` from the
  combined `ByteAddressBuffer`). `environment.slang` — the two env CDF
  StructuredBuffers fold into one `envDistCdf` at binding 31 (conditional at
  `ENV_COND_CDF_BASE = ENV_DIST_H+1`); binding 32 retired.
- Host: `src/skinny/renderer.py` (`_upload_graph_param_buffers` → one combined
  `_graph_params_combined`; the wavefront shade-pass / path-pass / material-pass
  bind maps; `env_dist_buffer` replacing `env_marginal_buffer` +
  `env_cond_buffer` in alloc/upload/bind/destroy + the descriptor-pool count),
  `src/skinny/vk_compute.py` + `src/skinny/metal_compute.py` (graph-param +
  env-CDF descriptor/bind spec → single slots each),
  `src/skinny/metal_wavefront.py` (`graph_param_layouts` reflection retired —
  the byte buffer needs no per-field MSL layout).
- Tests: guarded Metal pipeline-build test for the failing command; Vulkan/Metal
  graph-material A/B; codegen offset-table unit test.
- Docs: `docs/Architecture.md`, `docs/Wavefront.md`, `docs/Megakernel.md`,
  `CHANGELOG.md`.
- Out of scope: combining the neural weight/bias/layer buffers (a separate,
  independent slot win); changing the megakernel `evalSceneGraph` dispatch
  structure; any change to the graph fragment sources or per-material shade
  partitioning (`per-material-pipeline` semantics preserved).
