## Why

Adding or removing a model at runtime currently rebuilds far more than the model
itself. `add_model()` / `remove_node()` both call
`_resync_geometry_from_stage()`, which re-reads the whole stage and runs three
costs that scale with the *entire* scene rather than the one prim that changed:

1. **Shader recompile.** The renderer is a single megakernel
   (`main_pass.slang`). Every MaterialX nodegraph is stitched into one
   auto-generated `switch (graphId)` in `generated_materials.slang` and compiled
   to one SPIR-V. Adding a model that introduces a *new* material graph changes
   the graph-set signature and forces a full `slangc` recompile of the whole
   kernel plus a pipeline recreate — seconds of stall.
2. **Geometry re-upload.** All meshes are concatenated back-to-back into a single
   shared vertex/index/BVH buffer (`_upload_meshes_concatenated()`). Any add or
   remove re-concatenates and re-uploads the *whole* scene, even for an already
   baked mesh.
3. **Mesh bake** — already content-hash cached, so only genuinely new geometry
   bakes. Not a target here.

The fix has two independent axes. Costs (1) and (3) live on the *material* axis;
cost (2) lives on the *mesh* axis. We address both: a switchable **wavefront
execution backend** that compiles each material as its own small compute kernel
(so adding a material compiles one kernel, never the megakernel), and a
**geometry suballocator** so add/remove touch only the changed mesh's slab.

The megakernel stays. Megakernel and wavefront have opposite sweet spots —
megakernel wins on simple scenes (no VRAM round-trips), wavefront wins on
heavy/divergent material sets (coherent execution + per-material compile) — so a
runtime mode switch is a long-term feature, not a migration crutch.

## What Changes

- Add an **execution-mode axis** — `megakernel` (current) and `wavefront` —
  orthogonal to the existing integrator axis (`path` / `bdpt`). Selected like
  `backend`, included in `_current_state_hash()` (so a switch resets
  accumulation), persisted in `settings.json`, and exposed in `ALL_PARAMS` plus
  every front-end (GLFW, Qt, debug viewport) for parity.
- Add a **wavefront backend** that splits the in-kernel path loop into staged
  compute dispatches communicating through GPU path-state buffers and ray queues:
  generate → intersect → logic → shade. The **shade stage is one compute pipeline
  per material**, so adding a material compiles a single small kernel and
  registers it — the megakernel `switch`/recompile path is never taken in
  wavefront mode.
- Split the material codegen behind **two emitters** over the same
  `GraphFragment` source: the existing megakernel switch-stitch emitter, and a
  new per-material-entry emitter that produces a `shadeMaterial_X` compute entry
  **and** an `evalBSDF_X` entry (the latter needed for bdpt connection events).
- Bound wavefront memory with **tiled / streaming** execution (fixed-size path
  streams with terminated-lane refill) rather than one path-state slot per pixel.
- Extend the wavefront backend to **bdpt** so the mode switch is unconditional:
  camera + light subpath wavefronts, vertex storage, and a connection+shadow
  stage with MIS. Matches the current bdpt scope (flat first-hit only), not
  general bidirectional.
- Add a **geometry suballocator**: per-mesh slabs in the shared vertex/index/BVH
  buffers with stable offsets and a free-list. Add writes one slab + instance
  record(s); remove frees a slab + drops instance record(s); neither
  re-concatenates the scene. Mode-independent (both backends read these buffers).
- Add **multi-pipeline + indirect-dispatch** support to the Vulkan compute layer,
  modeled on the existing `vk_skinning.py` two-pipeline precedent.
- **Vulkan-first.** Wavefront is a Vulkan backend feature, like GPU skinning/BVH
  refit. On Metal the execution mode is forced to / pinned at `megakernel`.

## Capabilities

### New Capabilities
- `wavefront-execution`: the execution-mode axis, the staged wavefront pipeline
  (path and bdpt), tiled/streaming memory bound, mode switching + persistence +
  front-end parity + UI gating, and mega-vs-wavefront A/B parity.
- `per-material-pipeline`: per-material shade/eval compute kernels under the
  wavefront backend; adding a material compiles one kernel instead of the whole
  megakernel; the two-emitter codegen contract.
- `geometry-suballocation`: incremental per-mesh slab management of the shared
  geometry buffers so runtime add/remove touch only the changed mesh.

### Modified Capabilities
<!-- none — these are new capabilities; existing specs are untouched -->

## Impact

- **Code:** `src/skinny/renderer.py` (execution-mode field, state hash,
  persistence, dispatch dispatch-on-mode, suballocator replacing
  `_upload_meshes_concatenated` / `_ensure_mesh_buffer_capacity`, incremental
  add/remove paths in `add_model` / `remove_node` / `_resync_geometry_from_stage`);
  `src/skinny/vk_compute.py` (two-emitter codegen split in
  `_emit_generated_materials`; multi-pipeline + indirect-dispatch support);
  new `src/skinny/vk_wavefront.py` (wavefront pipeline set + queues + dispatch
  loop, modeled on `vk_skinning.py`); `src/skinny/app.py`,
  `src/skinny/ui/qt/viewport.py`, debug viewport (mode toggle parity);
  `src/skinny/ui/` ALL_PARAMS entry.
- **Shaders:** new `src/skinny/shaders/wavefront/*` (generate, intersect, logic,
  build-indirect-args, and the per-material `shadeMaterial_X` / `evalBSDF_X`
  templates); reuse the BVH traversal, light, sampler, and §1–§6 estimator
  includes so material/integrator logic is shared, not duplicated. `main_pass.slang`
  (megakernel) untouched. Any shader change requires recompiling SPIR-V.
- **APIs:** renderer gains an execution-mode property + cycle/set method; geometry
  upload API changes from concatenate-all to slab add/free.
- **Settings:** new persisted `execution_mode` field in `settings.json`.
- **Backends:** Vulkan-first; Metal pinned to megakernel (documented like the
  Vulkan-only skinning/refit pipelines).
- **Tests:** headless A/B parity (megakernel vs wavefront, per integrator) as a CI
  gate using the existing `tests/test_headless.py` harness; suballocator unit
  tests (stable offsets across add/remove, free-list reuse, grow/compact);
  front-end mode-toggle parity test; indirect-dispatch correctness vs conservative
  direct-dispatch fallback.
