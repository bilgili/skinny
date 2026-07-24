# renderer-module-carveout

## Why

`src/skinny/renderer.py` is a 12,114-line god module: one class, one mutable
`self`, owning unrelated concerns — uniform/material byte packing, the
accumulation state hash, MLT bootstrap/chain seeding, lens/framing math,
wavefront pass construction for both backends, Metal bind maps, USD live-edit,
detail maps, spectral tables, and the gizmo overlay. It carries 117
`is_metal`/`_metal` sites, including a 13-method `_render_*_metal` /
`_ensure_*_metal` sibling family that duplicates orchestration per backend.
Locality is near zero, and pure logic is untestable without a GPU: 67 test
files import `skinny.renderer`, and anything touching a frame stands up a
device.

The codebase already proves the fix works: `wavefront_driver.py` (463 lines,
zero backend leaks) holds the staged loop once behind a duck-typed recorder
protocol, and `mlt_bootstrap.py` holds the pure MLT resample hostlessly. This
change is the umbrella carve-out that extends that pattern to the remaining
renderer-resident clusters — as a pure refactor with bit-identical rendering.

## What Changes

This is the umbrella proposal. Two sibling proposals already own adjacent
scope and are referenced, not duplicated:

- **`reflection-owned-byte-layouts`** (sibling) — owns the *byte packing* side
  (`pack_flat_material`, `pack_std_surface_params*`, `_pack_uniforms` /
  `_pack_uniforms_msl` serialization and offset/reflection ownership).
- **`param-registry-accumulation-reset`** (sibling) — owns the accumulation
  state hash (`_current_state_hash`) and its parameter registry.

This proposal's own scope, in independently landable stages:

1. **MLT chain-state module** — extract the MLT host orchestration that still
   lives on `Renderer` (`_next_mlt_seed`, `_mlt_uniform_tail_active`,
   `_mlt_iterations_per_frame`, `_mlt_pass_key`, the
   `_run_wavefront_mlt_bootstrap` / `_run_wavefront_mlt_bootstrap_metal`
   round-trip) into a dedicated module with a hostless-testable pure core,
   completing the family started by `mlt_bootstrap.py` +
   `wavefront_driver.record_mlt_*`.
2. **Frame-constant derivation module** — extract the derived-value
   computation currently inside `_pack_uniforms` (camera view/proj inverses,
   the lens FOV-framing ratio and `film_half_h_world`, the detail-flag
   bitfield, the exposure/imaging-ratio fold, the emissive total power, and
   the proposal-mask/reuse capability folding) into pure state→values
   functions. The packing method keeps its existing side-effect call sites
   and append order (it is not itself made pure here — that is the sibling's
   packer scope); what this stage delivers is that every *derived value* is
   computed by device-free functions, testable without a GPU.
3. **Wavefront pass-object seam** — move the `_ensure_wavefront_*` /
   `_ensure_wavefront_*_metal` and `_render_wavefront_*` /
   `_render_wavefront_*_metal` method-pair family behind the duck-typed
   pass-object seam that `wavefront_driver.py` already demonstrates: one
   ensure/dispatch path per integrator on the renderer, backend divergence in
   per-backend pass factories/adapters. This reduces renderer-resident
   `is_metal`/`_metal` sites without changing the seam itself.
4. **Extraction pattern + follow-on ordering** — define the reusable carve-out
   pattern (pure core + orchestration seam + bit-identity gate) and the order
   for the remaining clusters (USD live-edit, gizmo overlay, detail maps) as
   follow-on changes; those clusters are *not* implemented here.

Constraints (all stages): pure refactor, bit-identical rendering, RGB `.spv`
byte-unchanged (no shader edits), parity matrix gates stay green, one
extraction stage per PR-able task group.

Recorded decisions respected, not re-litigated: `openspec/specs/metal-backend`
mandates every renderer path either runs a Metal-equivalent or short-circuits
on `is_metal` — the problem is volume-in-one-file, not the seam's existence;
CLAUDE.md documents the `resource_module`/`is_metal` split as intentional.

## Capabilities

### New Capabilities

- `renderer-module-structure` — structural requirements on the renderer host
  layer: MLT chain state, frame-constant derivation, and wavefront pass
  construction/dispatch live in dedicated modules with hostless-testable pure
  cores, and every carve-out stage lands with bit-identity gates.

### Modified Capabilities

None. No existing capability's behavior changes: `metal-backend`,
`wavefront-execution`, and `metropolis-light-transport` requirements are
preserved verbatim by the bit-identity constraint.

## Impact

- **Code**: `src/skinny/renderer.py` shrinks (MLT orchestration, derivation
  math, and the `_metal` sibling family move out); new modules
  `src/skinny/mlt_chain.py` and `src/skinny/frame_derive.py` (names final in
  design); `src/skinny/vk_wavefront.py` / `src/skinny/metal_wavefront.py`
  gain the pass-factory surface. No shader, no `.spv`, no binding change.
- **Tests**: new hostless unit tests for the extracted pure cores; existing
  parity matrix (`tests/pbrt/`) is the regression gate per stage; a
  byte-equality golden test pins `_pack_uniforms` output across the stage-2
  extraction.
- **Sequencing with siblings**: the ordering between stage 2 and
  `reflection-owned-byte-layouts` is **soft** — their scopes are disjoint
  (byte-layouts v1 touches module-level layout tables and
  `_pack_uniforms_msl`; stage 2 touches only the value expressions inside
  `_pack_uniforms`), so the real coupling is a textual merge conflict on the
  same method, avoided by landing stage 2 first when practical. This change
  does not touch `_current_state_hash`, which stays whole for
  `param-registry-accumulation-reset`. All three renderer-cluster changes
  touch `tests/test_mlt_host.py` (it asserts against renderer-resident
  source); each change's tasks re-point that file's assertions to the new
  authority as they move code.
- **Docs**: `docs/Architecture.md` module map, `docs/Wavefront.md` (pass-seam
  wording), `docs/MetropolisLightTransport.md` (host-orchestration module),
  CLAUDE.md architecture notes.
