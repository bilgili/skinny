# Design: reflection-owned-byte-layouts

## Context

Host code and Slang shaders share byte layouts by convention, and the
convention is hand-copied in three places:

- **FrameConstants** (`common.slang`): `_FC_SCALAR_FIELDS` (renderer.py:180),
  `_FC_MLT_FIELDS` / `_FC_SCALAR_FIELDS_MLT` (:223/:232),
  `_TILE_ORIGIN_Y_OFFSET` (:240), the `_VK_UNIFORM_BUFFER_BYTES` sizing assert
  (:284), and the append order inside `_pack_uniforms` (:10452).
- **Material params**: `pack_std_surface_params` (:853) + `_STD_SURFACE_FIELDS`
  (:973) + `pack_std_surface_params_msl` (:992) mirror `StdSurfaceParams`
  (`mtlx_std_surface.slang`); `pack_flat_material` (:635) +
  `FLAT_MATERIAL_STRIDE` (:321, comment-only offset map) mirror
  `FlatMaterialParams` (`common.slang:57`).
- **Wavefront sizing** (`wavefront_layout.py`, 473 lines): field lists for
  `WavefrontPathState`, `RecVertex`, `VisiblePoint`, `SppmAccum`,
  `BDPTVertex`, `WfBdptAux`, and the MLT chain structs, each taking
  `spectral: bool` to re-express `#if defined(SKINNY_SPECTRAL)` blocks.

What already exists and works (verified in the worktree):

- **Metal runtime reflection**: `metal_compute.ComputePipeline` reflects
  `uniform_layout`/`uniform_size` (:719), `mtlx_skin_layout` (:745), and
  `std_surface_layout` (:645) from the SlangPy compiled program.
  `_pack_uniforms_msl` consumes it with a coverage drift-guard assert
  (renderer.py:10153). This is the precedent to extend. It requires a live
  Metal device + slang session — it is **not** hostless.
- **Vulkan has no field-level reflection.** `vk_compute.ComputePipeline`
  hand-authors the descriptor-set layout (binding *types* only, no struct
  offsets) and compiles via `slangc -fvk-use-scalar-layout`. Scalar layout
  means every Vulkan offset is a pure running sum of scalar field sizes — a
  deterministic function of the declared field order alone.
- **A hostless Slang struct parser already exists — in two copies**:
  `tests/test_wavefront_state.py::_parse_struct_fields` and a near-duplicate
  at `tests/test_sppm_state.py:56` regex-parse a struct body out of the
  `.slang` source, strip comments, resolve `#if defined(SKINNY_SPECTRAL)` per
  variant, normalize the `Spectrum` typealias, and handle nested
  `SampledWavelengths`. Consolidation retires both copies into one `src/`
  module. gpu-marked companion tests (`_reflect_msl_layout`) lock the Python
  MSL math to real Metal reflection.
- **FrameConstants is gate-laden and its blob order is NOT declaration
  order**: the struct carries `#if defined(SKINNY_METAL) uint tileOriginY;`
  (common.slang:414) immediately before the `#if defined(SKINNY_MLT)` tail
  (:421). The host scalar blob always carries `tileOriginY` **last** — under
  an MLT pack the 32 B tail sits where the Vulkan filler word would be and
  `tileOriginY` follows it (`_FC_SCALAR_FIELDS_MLT`, renderer.py:232), so on
  Vulkan-MLT (whose SPIR-V struct has no `tileOriginY` at all) `mltSigma`
  lands at 564 and the trailing word is benign filler inside the 768 B UBO.
  `tests/test_mlt_host.py:189–211` pins exactly this today.

So all the ingredients exist; they are just distributed across tests and
per-site tables. The change is consolidation, not invention.

## Goals / Non-Goals

**Goals**

- One module (`src/skinny/slang_layout.py`) owns every host-mirrored byte
  layout: field order parsed from the authoritative `.slang` declaration,
  scalar and MSL offsets/strides computed by the existing deterministic math.
- Packers (`_pack_uniforms`, `_pack_uniforms_msl`, `pack_std_surface_params*`,
  `pack_flat_material`) and allocators (`wavefront_layout` sizers) consume it.
- Layout drift (field reorder/retype/insert in a mirrored Slang struct) fails
  a **hostless** test instead of garbling GPU output.
- Vulkan SPIR-V byte-unchanged; rendered output bit-identical; every existing
  hostless and gpu-marked layout test kept at full strength.

**Non-Goals**

- No shader edits, no new bindings, no `.spv` recompile.
- No change to *what* is packed (field values, defaults, alias maps like
  OpenPBR→standard_surface stay in the packers).
- `SkinParameters.pack()` (std140, 80 B, single site with its own documented
  offsets and `test_skin_parameters_pack_size` guard) is out of scope for v1 —
  std140 is a third layout dialect and the struct is stable; see Open
  Questions.
- No general Slang front-end. The parser handles exactly the declaration
  subset the mirrored structs use, and **raises** on anything it cannot
  classify — it never guesses.

## Decisions

### D1 — Layout source: parse the `.slang` declarations, not a snapshot, not runtime reflection

`slang_layout.py` derives each struct's ordered `(name, type)` field list by
parsing the authoritative `.slang` source (promotion of
`test_wavefront_state._parse_struct_fields` into `src/`), then computes:

- scalar offsets/stride with `SLANG_SCALAR_SIZES` running sums (exactly what
  `-fvk-use-scalar-layout` guarantees on Vulkan), and
- MSL offsets/stride with the `SLANG_MSL_SIZES`/`SLANG_MSL_ALIGNS` walk
  already in `wavefront_layout._struct_stride`.

Variant defines are resolved per query, as the test parser does today, with
exactly three resolvable gates: `SKINNY_SPECTRAL`, `SKINNY_MLT`, and
`SKINNY_METAL` (required — `FrameConstants` gates `tileOriginY` behind it, so
a two-define whitelist cannot parse the registered struct at all). Semantics
per variant: a compiled-struct layout query resolves the gates to that
target's defines (Vulkan RGB: all off; Vulkan MLT: `SKINNY_MLT` only —
no `tileOriginY` in the SPIR-V struct; Metal: `SKINNY_METAL` on, plus
`SKINNY_MLT`/`SKINNY_SPECTRAL` as built). Any other `#if` in a mirrored
struct raises. Parsed results are cached per (file mtime, struct, variant);
parsing is lazy so app startup never depends on structs it does not use.

**The FC scalar-blob rule (declaration order is not blob order).** The
host-side scalar blob for `FrameConstants` is defined as: the variant's
declared fields in order, with `tileOriginY` **always present and always
relocated to the tail** (after the MLT tail when present). This reproduces
today's `_FC_SCALAR_FIELDS` / `_FC_SCALAR_FIELDS_MLT` exactly: base blob ends
`…sppmGroupPmfEnv, tileOriginY` (568 B, `_TILE_ORIGIN_Y_OFFSET` = 564); MLT
blob ends `…sppmGroupPmfEnv, mltSigma…mltSeed, tileOriginY` (600 B), so
`mltSigma` sits at 564 where the Vulkan-MLT SPIR-V expects it, and on Vulkan
variants whose compiled struct lacks `tileOriginY` the trailing word is benign
filler inside the 768 B UBO (renderer.py:277–287). The blob rule lives in the
layout module as part of the `FrameConstants` registration, and
`tests/test_mlt_host.py:189–211` is re-pointed at the derived tables as the
**permanent** blob-order lock (mltSigma@564, `tileOriginY` last, +32 B MLT
delta) — not deleted with the hand tables.

**Why not the alternatives:**

- *Runtime SlangPy reflection as the source* — requires a GPU device and a
  slang session; checks would be gpu-marked, violating the hostless
  preference, and the Vulkan front-end does not even use SlangPy. Rejected as
  the *source*; retained as the *ground truth check* (D3).
- *`slangc -reflection-json` at build/test time* — needs the slangc toolchain
  in every test environment and one compile per variant; slower and no more
  authoritative than the source itself given scalar layout is deterministic.
  Rejected.
- *Checked-in reflected-layout snapshot + regen script + drift gate* — works
  and is fully hostless, but adds a regen step humans forget and a
  stale-snapshot failure mode; the parser already exists, is exercised in CI
  today, and reads the same file the shader compiler reads (cannot go stale).
  Rejected for v1; documented as the fallback if the parser ever proves
  fragile against new Slang syntax (the module's API — `fields()`,
  `scalar_layout()`, `msl_layout()` — would be unchanged by that swap).

**Answer to "where does reflection data come from on the Vulkan path":**
verified — there is no field-level SPIR-V reflection in
`vk_compute.ComputePipeline` today (descriptor reflection covers binding types
only). None is needed: under `-fvk-use-scalar-layout` the offsets are a pure
function of declared field order, so the parsed field list plus the
registered blob rule above (declaration order alone is NOT sufficient for
`FrameConstants` — see the tileOriginY relocation) is the reflection
equivalent, hostlessly. The compiler-side ground truth is D3.

**Type coverage is an extension, not pure consolidation.** `FrameConstants`
needs types and features the existing tables lack: `float4x4` (camera
matrices), `uint2` (`pickPixel`), `uint3` (`sppmGridRes`), and generalized
nested-struct flattening (`camera.<field>`, today special-cased only for
`SampledWavelengths`). `SLANG_SCALAR_SIZES` / `SLANG_MSL_SIZES` /
`SLANG_MSL_ALIGNS` grow accordingly (scalar: float4x4 64, uint2 8, uint3 12;
MSL: float4x4 64/16, uint2 8/8, uint3 16/16). Every **new** MSL size/align
rule is trusted only after the gpu-marked `fc` reflection lock (Stage 1,
task 2.4) confirms it against Slang's Metal target — the same discipline the
existing wavefront MSL rules earned.

### D2 — Consumers keep their shape; only the tables change owner

- `_FC_SCALAR_FIELDS` / `_FC_MLT_FIELDS` / `_FC_SCALAR_FIELDS_MLT` become
  `slang_layout` queries over `FrameConstants` (base and `SKINNY_MLT`
  variants) **under the blob rule of D1** (tileOriginY relocated to the
  tail); `_TILE_ORIGIN_Y_OFFSET` and the `_VK_UNIFORM_BUFFER_BYTES` assert
  derive from the same query. `_pack_uniforms`'s `struct.pack` call sites are
  untouched in v1. Coverage (derived table == `len(_pack_uniforms())`) is a
  **runtime pack-site guard** — `_pack_uniforms` needs a constructed
  `Renderer` (and `skinny.renderer` imports `vulkan` at module load), so this
  check cannot be hostless; it generalizes the :10153 guard to the Vulkan
  upload path. The hostless side pins the derived blobs directly (D3).
  Direct table consumers migrate with the tables: `tests/test_mlt_host.py`
  reads `_FC_SCALAR_FIELDS*` / `_TILE_ORIGIN_Y_OFFSET` by name (:189–211)
  and is re-pointed at the derived tables in the same stage, keeping its
  assertions as the permanent blob-order lock.
- `_STD_SURFACE_FIELDS` is deleted; `pack_std_surface_params_msl` iterates the
  derived scalar layout of `StdSurfaceParams`. `FLAT_MATERIAL_STRIDE` and
  `STD_SURFACE_STRIDE` become derived constants; the comment-only offset map
  at renderer.py:302–320 is replaced by a pointer to the module.
- `wavefront_layout.py` keeps its entire public API (constants,
  `*_size(msl=, spectral=)` helpers, `queue_buffer_sizes`,
  `sppm_buffer_sizes`, `mlt_buffer_sizes`, flag bits) but its private
  `_*_fields` lists are replaced by parsed field lists. Callers
  (vk_wavefront, metal_wavefront, drivers, tests) need zero edits.
- `_pack_uniforms_msl` continues to consume **live** `pipeline.uniform_layout`
  reflection at runtime — the `metal-backend` spec requires reflected offsets
  and that stays true. New: it cross-asserts the live reflection against the
  derived MSL layout (both directions of drift now fail loudly).

### D3 — Drift gates: hostless primary, gpu-marked ground truth, both mandatory

Three layers, no layer weakened:

1. **Hostless equality/coverage gates (new, primary).** Per mirrored struct ×
   variant: derived field list has no gap/overlap; derived strides == pinned
   golden values (68 scalar / 96 MSL path state, 76/112 RecVertex, 180/240
   VisiblePoint, 16 RGB / 20 spectral SppmAccum, 120 RGB / 128 spectral
   BDPTVertex scalar, 92/136 WfBdptAux scalar, 16/32/16 MLT, 256
   flat/std-surface). Packer-output-length equality is hostless where the
   packer is hostlessly constructible (`pack_flat_material`,
   `pack_std_surface_params` take a plain namespace). For `FrameConstants`
   the packer is renderer-bound, so the hostless gates pin the **derived
   blobs** directly: golden blob lengths (568 base / 600 MLT) and a golden
   field-order lock (the derived (name, size) sequence, plus the re-pointed
   `test_mlt_host` blob-order assertions) — a shader-side field reorder then
   fails hostlessly and forces the human to update the golden *and* the
   `struct.pack` body together; the blob↔packer length equality itself is
   the runtime pack-site guard of D2. Pinning the numbers keeps the gate
   non-tautological: if a parser change *and* a shader change move together,
   the pinned constant still trips and forces a human re-measure — the same
   discipline the parity manifest uses.
2. **gpu-marked MSL reflection locks (existing, extended).** The
   `_reflect_msl_layout` lock tests in `test_wavefront_state.py` and the
   round-trip in `test_metal_std_surface_layout.py` /
   `test_metal_flat_material_layout.py` remain the ground truth that the
   Python MSL math equals what Slang's Metal target actually emits; extended
   to cover `fc` (uniform) and any struct newly routed through the module.
3. **Runtime asserts (existing, strengthened).** The `_pack_uniforms_msl`
   coverage assert stays; plus the D2 cross-check of live reflection vs
   derived layout.

During migration each stage also carries a temporary **table-equality test**
(derived output == old hand table, byte-for-byte) that lands *before* the hand
table is deleted, then is removed with it.

### D4 — Bit-identical adoption, enforced per stage

No `.slang` file changes ⇒ the Vulkan SPIR-V cache key inputs are unchanged ⇒
SPIR-V byte-identical by construction (asserted once by hashing `main_pass.spv`
pre/post). Packer outputs are gated byte-identical by the D3 stage-equality
tests, and a matrix-harness spot check (path/bdpt, mega/wave, Metal) confirms
pixel-identical accumulation on an unchanged seed.

## Risks / Trade-offs

- **[Risk] Regex parser meets Slang syntax it mis-reads** (new nesting, new
  vector type, multi-declarator lines). → Mitigation: the parser whitelists
  known types and the three known defines (`SKINNY_SPECTRAL`, `SKINNY_MLT`,
  `SKINNY_METAL`) and **raises** on anything else — an
  unparseable struct is a hostless test failure, never a silent wrong offset;
  the gpu-marked reflection locks are a second, independent net; the snapshot
  design (D1 alternative) is the documented escape hatch with an unchanged
  module API.
- **[Risk] Tautology — source-derived layout checked against source-derived
  layout.** → Mitigation: pinned golden strides (layer 1) and live-reflection
  locks (layer 2) are both independent of the parser.
- **[Risk] A migration stage silently changes packed bytes.** → Mitigation:
  the temporary table-equality test lands green before its hand table is
  deleted; each stage is a separate commit gated on the hostless sweep plus
  the SPIR-V hash check.
- **[Risk] Import-time parsing slows or breaks startup.** → Mitigation: lazy +
  cached per (mtime, struct, variant); the renderer touches only the structs
  the session uses.
- **[Trade-off] `struct.pack` call sites still encode field order implicitly**
  (v1 keeps them for the shortest diff). The coverage gate makes order drift
  fail hostlessly, but a same-size field *swap* inside the packer body is
  caught only by value-level tests (e.g. `test_struct_layout`'s
  unpack-at-offset checks, which now read offsets from the module). Full
  offset-driven packing is a possible v2; not needed to kill the drift class
  named here.

## Migration Plan

Staged; each stage independently green, bit-identical-verified, and mergeable.

1. **Stage 0 — module.** Add `slang_layout.py` (parser moved from
   `test_wavefront_state.py` + layout math shared with `wavefront_layout`),
   with its own hostless tests (golden strides, gap/overlap, raise-on-unknown).
   No consumer changes yet.
2. **Stage 1 — uniforms.** Derive the FrameConstants tables (base + MLT
   variant, under the D1 blob rule), `_TILE_ORIGIN_Y_OFFSET`,
   `_VK_UNIFORM_BUFFER_BYTES` bound; equality test vs hand tables → land the
   gpu-marked `fc` MSL reflection lock (validates the new
   float4x4/uint2/uint3/nested MSL rules against live Metal reflection)
   **before** arming the runtime raise-on-mismatch cross-assert — armed the
   other way round, wrong derived MSL math would crash every Metal run at
   startup; re-point `tests/test_mlt_host.py` at the derived tables → delete
   hand tables. Verify: hostless sweep, `main_pass.spv` hash unchanged, one
   Metal megakernel + one MLT wavefront frame pixel-identical.
3. **Stage 2 — StdSurface / flat material.** Derive `_STD_SURFACE_FIELDS`
   replacement, `STD_SURFACE_STRIDE`, `FLAT_MATERIAL_STRIDE`; point
   `test_struct_layout` / `test_metal_std_surface_layout` /
   `test_metal_flat_material_layout` offset lookups at the module. Verify:
   packer outputs byte-identical, gpu-marked std-surface round-trip green.
4. **Stage 3 — wavefront sizers.** Swap `wavefront_layout`'s private field
   lists to parsed ones behind the unchanged public API; keep every existing
   hostless test and gpu lock green as-is (they are the baseline, not
   replaceable). Verify: all `*_STRIDE` constants numerically unchanged
   (pinned), RGB and spectral, scalar and MSL.
5. **Stage 4 — docs + closure.** `docs/Architecture.md` layout-ownership
   section; run the full hostless matrix tests plus the guarded gpu layout
   locks; matrix-harness spot check for bit-identical output.

Rollback per stage = revert that stage's commit; the facade APIs make each
swap independently reversible.

**Cross-change coordination (soft, conflict-avoidance only):**
`tests/test_mlt_host.py` and the `_pack_uniforms` region are also touched
textually by the other renderer-cluster changes (notably
`renderer-module-carveout` Stage B); sequence Stage 1 relative to that
change's landing to avoid merge conflicts — there is **no semantic
prerequisite** in either direction, and each change must simply update the
`test_mlt_host` assertions to whichever layout authority is current when it
lands.

## Open Questions

- Should `SkinParameters.pack()` (std140) and `INSTANCE_STRIDE` /
  `Light`-buffer records join the module in a follow-up? They are
  single-authored today (lower drift risk), but bringing them in would make
  the module's claim ("all host-mirrored layouts") complete. Proposed: yes,
  as a separate small change once v1 has soaked.
- Should v2 move `_pack_uniforms` to offset-driven writes (killing the
  implicit order in the `struct.pack` body)? Deferred; measure whether the
  residual same-size-swap risk ever bites first.
- Parser home: `slang_layout.py` proposed standalone; folding it into
  `wavefront_layout.py` (renamed) is acceptable if reviewers prefer one file —
  API is what matters, not the module count.
