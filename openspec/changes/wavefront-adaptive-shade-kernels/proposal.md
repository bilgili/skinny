## Why

The wavefront path tracer shades hits through exactly **two** size-fixed
kernels: `wfPathShadeFlat` (slot 0 — flat + MaterialX graphs, ~1.4 MB) and
`wfPathShade` (slot 1 — the *catch-all*: skin + every python material + debug,
~2.8 MB). The catch-all imports the whole material/integrator tree, so it sits
right at MoltenVK's ~2.83 MB Metal-compile danger line. Under sustained GPU load
the Metal compile of that kernel **flakes** — observed in the back-to-back GPU
suite (`test_wavefront_bdpt_matches_megakernel`,
`test_shade_pipelines_cache_per_material` flaked only inside the full run, pass
alone). This is the recorded "5.4 remainder": *per-graph (vs per-type) shade
granularity*.

The catch-all is big precisely because it is a **union** — skin + N python
materials + debug in one kernel. Split it and each kernel imports only its own
material, so every kernel is inherently small. The fix is to stop bundling by
material *type* and instead partition shade work into **size-bounded** kernels,
none of which approaches the compile limit.

## What Changes

- **Adaptive, size-gated shade-kernel grouping (path tracer only).** Replace the
  fixed `flat | catch-all` 2-slot split with a per-scene set of shade kernels
  whose SPIR-V never exceeds a safety threshold below the MoltenVK limit. The
  flat/graph kernel stays slot 0 unchanged.
- **Measured two-pass bin-pack.** On a material-set change (scene load / add
  material / live python-material edit): (1) compile each non-flat shade member
  isolated and read its `.spv` byte size (cache-keyed, so repeats are free);
  (2) first-fit-decreasing bin-pack members into groups with
  `Σ(isolated sizes) ≤ THRESHOLD` (≈2.4 MB), skin pinned to its own group;
  (3) compile the grouped kernels + build a `materialId → slot` map. A group's
  real size ≤ the sum of its members (shared imports counted once), so packing
  by the sum **never under-splits** past the limit.
- **Dynamic slot count via fixed `MAX_SLOTS` padding.** `WF_NUM_SLOTS` (today a
  static `2u`) becomes a fixed maximum (`MAX_SLOTS = 32`). The classify /
  build-args / clear-counts shaders loop `0..MAX_SLOTS` (trivial); counting-sort
  buffers are sized to `MAX_SLOTS` (tiny). The CPU records exactly `used`
  indirect shade dispatches — raising the cap costs only shader loop iterations
  and a few bytes, **not** dispatch count. 32 makes overflow effectively
  impossible for real scenes.
- **Classifier change.** `wfSlotForType(matType)` → `wfSlotForMaterial(matId)`,
  reading a small CPU-built `materialSlot[]` buffer (new set-1 binding); flat /
  graph → slot 0.
- **`WavefrontPathPass` takes the grouping** (groups + member→slot map) instead
  of the `build_catchall: bool`, and builds one shade pipeline per group.

## Out of Scope (explicit non-goals)

- **bdpt is untouched.** Its 2-slot split is connection *strategy* (NEE / FULL)
  with BSDF eval inline — not a material-kernel split. Different axis.
- **megakernel is byte-unchanged**, never recompiled.
- **Shade-coherence perf tuning (goal A)** — coarsening the grouping to cut
  dispatch count on material-heavy scenes — is a **follow-up change**, gated by
  profiling and exposed as an opt-in knob. This change is compile-safety (goal
  B) only.
- **Shrinking the skin kernel itself** is out of scope. If skin *alone* measures
  over the limit, no grouping fixes it; pass 1 surfaces that immediately as a
  separate effort.

## Capabilities

### Modified Capabilities
- `per-material-pipeline`: the wavefront shade stage partitions materials into
  **size-bounded** compute pipelines (adaptive measured grouping under the Metal
  compile limit) rather than a fixed flat/catch-all pair; no shade pipeline
  approaches the MoltenVK ~2.83 MB danger line.

## Impact

- **Shaders:** `wavefront/wf_shade_common.slang` (`WF_NUM_SLOTS` → `MAX_SLOTS`;
  `wfSlotForType` → `wfSlotForMaterial` + `materialSlot[]` binding);
  `wavefront/wavefront_path.slang` (classify reads the map; grouped shade entries
  switch among their members; single-member fast path). Runtime-generated `.spv`
  — no checked-in shader bytes change; `main_pass.slang` untouched.
- **Python:** `vk_wavefront.py` (`WavefrontPathPass` takes a grouping; one
  pipeline per group; emit per-group shade entries; measure isolated `.spv`
  sizes); a new `wavefront_shade_packer` module (the pure bin-pack);
  `wavefront_layout.queue_buffer_sizes` (size to `MAX_SLOTS`); `renderer.py`
  (`_ensure_wavefront_path_pass` builds the grouping from `_material_types` +
  python ids, uploads `materialSlot[]`, records `used` dispatches).
- **Tests:** A/B parity on a forced-split multi-material scene; a per-`.spv`
  size-regression assertion (the bug guard); a CPU bin-pack unit test (FFD,
  skin isolated, deterministic); a slot-map / classify counting-sort test; a
  `MAX_SLOTS` headroom test.
- **Behaviour:** identical image (every grouping is the same estimator); the
  flaky catch-all compile is eliminated because no shade kernel nears the limit.

## Notes

Builds on the archived `wavefront-execution-backend` and `per-material-pipeline`
work. The follow-up perf-tune change (goal A) will reuse this grouping
infrastructure, biasing the pack threshold to trade dispatch count vs shade
coherence.
