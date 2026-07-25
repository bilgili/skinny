# Tasks: reflection-owned-byte-layouts

## 1. Layout module (Stage 0)

- [x] 1.1 Create `src/skinny/slang_layout.py`: consolidate BOTH parser copies
      (`tests/test_wavefront_state.py::_parse_struct_fields` and the
      near-duplicate at `tests/test_sppm_state.py:56`) into one module.
      Resolvable preprocessor gates are exactly `SKINNY_SPECTRAL`,
      `SKINNY_MLT`, and `SKINNY_METAL` (FrameConstants gates `tileOriginY`
      behind `SKINNY_METAL` at common.slang:414 — a two-define whitelist
      cannot parse it); parser raises on any other gate, declaration form,
      or field type.
- [x] 1.2 Add scalar + MSL layout math (reuse/share
      `wavefront_layout._struct_stride`, `SLANG_SCALAR_SIZES`,
      `SLANG_MSL_SIZES`/`SLANG_MSL_ALIGNS`) AND extend the type tables for
      what FrameConstants needs but the existing tables lack: `float4x4`
      (scalar 64, MSL 64/align 16), `uint2` (8, 8/8), `uint3` (12, 16/16),
      plus generalized nested-struct flattening (`camera.<field>`;
      `SampledWavelengths` stops being a special case). New MSL size/align
      rules are provisional until the task-2.4 gpu lock confirms them.
      Expose per (struct, variant): ordered fields, scalar offsets/stride,
      MSL offsets/stride; lazy, cached per (file mtime, struct, variant).
- [x] 1.3 Register the owned structs and their source files:
      `FrameConstants` + `FlatMaterialParams` (`common.slang`),
      `StdSurfaceParams` (`mtlx_std_surface.slang`), `WavefrontPathState`
      (`wavefront/wavefront_state.slang`), `RecVertex`
      (`wavefront/wf_records.slang`), `VisiblePoint`/`SppmAccum`
      (`integrators/sppm_state.slang`), `BDPTVertex`
      (`integrators/bdpt.slang`), `WfBdptAux`
      (`wavefront/wavefront_bdpt.slang`), MLT chain structs
      (`wavefront/wavefront_mlt.slang` + `common.slang` MLT tail).
      `FrameConstants` registers with the design-D1 scalar-blob rule:
      declared fields in variant order, `tileOriginY` always present and
      relocated to the tail (after the MLT tail when present).
- [x] 1.4 Hostless module tests (`tests/test_slang_layout.py`), each golden
      labelled by axis: path state 68 scalar / 96 MSL (RGB), RecVertex
      76 scalar / 112 MSL, VisiblePoint 180 scalar / 240 MSL (RGB),
      SppmAccum 16 RGB / 20 spectral (scalar==MSL), BDPTVertex scalar
      120 RGB / 128 spectral, WfBdptAux scalar 92 RGB / 136 spectral, MLT
      structs 16/32/16 (scalar==MSL), FlatMaterialParams 256 scalar,
      StdSurfaceParams 256 scalar; FrameConstants derived blobs pinned
      hostlessly — 568 B base / 600 B MLT and a golden field-order lock on
      the derived (name, size) sequence; gap/overlap coverage; a
      raise-on-unknown case (unsupported type AND unsupported `#if` gate).

## 2. FrameConstants adoption (Stage 1)

- [x] 2.1 Temporary equality test, green before any table edit:
      module-derived `(name, size)` tuples for `FrameConstants` base and MLT
      variants (blob rule applied) == `_FC_SCALAR_FIELDS` /
      `_FC_SCALAR_FIELDS_MLT`; derived `tileOriginY` scalar offset ==
      `_TILE_ORIGIN_Y_OFFSET` (564); MLT blob == base blob + 32 B with
      `mltSigma` at 564.
- [x] 2.2 Replace `_FC_SCALAR_FIELDS` / `_FC_MLT_FIELDS` /
      `_FC_SCALAR_FIELDS_MLT` / `_TILE_ORIGIN_Y_OFFSET` with module queries;
      derive the `_VK_UNIFORM_BUFFER_BYTES` import-time assert bound from the
      derived blob length (buffer constant itself stays 768).
- [x] 2.3 Runtime coverage guard at the pack sites (NOT hostless —
      `_pack_uniforms` needs a constructed `Renderer` and `skinny.renderer`
      imports `vulkan` at module load): derived field table covers
      `len(_pack_uniforms())` / `len(_pack_uniforms(mlt_tail=True))`
      exactly, asserted on the Vulkan upload path (generalizing the
      renderer.py:10153 Metal guard); hostless side is covered by the 1.4
      blob-length + field-order goldens.
- [x] 2.4 gpu-marked `fc` MSL reflection lock (guarded runner, `-m gpu`):
      module-derived MSL layout of `FrameConstants` == live
      `pipeline.uniform_layout` offsets and `uniform_size`, RGB and MLT
      variants. This is the ground-truth confirmation of the new
      float4x4/uint2/uint3/nested MSL rules and MUST land green before 2.5.
- [x] 2.5 In `_pack_uniforms_msl`, cross-assert live `pipeline.uniform_layout`
      offsets/size against the derived MSL layout (raise on mismatch, before
      upload) — armed only after 2.4 is green.
- [x] 2.6 Re-point `tests/test_mlt_host.py:189–211` at the derived tables:
      keep its assertions verbatim (MLT tail sits before `tileOriginY`,
      `mltSigma` at 564, +32 B delta) as the PERMANENT blob-order lock; do
      not delete with the hand tables.
- [x] 2.7 Remove the temporary 2.1 equality test with the hand tables; verify
      `build/spv_cache` `main_pass.spv` hash unchanged and one Metal
      megakernel frame + one MLT wavefront frame render bit-identical
      (`--backend metal`, guarded runner, one Metal process).

## 3. StdSurface / flat-material adoption (Stage 2)

- [x] 3.1 Temporary equality test: derived `StdSurfaceParams` scalar layout ==
      `_STD_SURFACE_FIELDS` running offsets; derived scalar stride == 256 ==
      `STD_SURFACE_STRIDE`; derived `FlatMaterialParams` scalar stride ==
      `FLAT_MATERIAL_STRIDE` == 256 with offsets compared at float4-row
      granularity (the Slang struct declares float4-wrapped rows,
      common.slang:57–105 — the renderer.py:302–320 comment map's scalar
      sub-offsets are packer-internal, not struct fields).
- [x] 3.2 Delete `_STD_SURFACE_FIELDS`; `pack_std_surface_params_msl` iterates
      the derived scalar layout; `STD_SURFACE_STRIDE` /
      `FLAT_MATERIAL_STRIDE` become derived constants; replace the comment
      offset map with a pointer to the module.
- [x] 3.3 Point offset lookups in `tests/test_struct_layout.py`,
      `tests/test_metal_std_surface_layout.py`, and
      `tests/test_metal_flat_material_layout.py` at the module (assert
      values unchanged — tests keep their unpack-at-offset value checks).
- [x] 3.4 Verify: `pack_std_surface_params` / `pack_flat_material` outputs
      byte-identical on a corpus of real materials (suite scenes); gpu-marked
      std-surface and flat-material round-trips green; remove the 3.1
      temporary test.

## 4. Wavefront sizer adoption (Stage 3)

- [x] 4.1 Swap `wavefront_layout.py`'s private `_*_fields` lists (path state,
      RecVertex, VisiblePoint, SppmAccum, BDPTVertex, WfBdptAux, MLT) to
      module-parsed field lists; public constants and sizer signatures
      unchanged; keep `REC_MAX_BOUNCES`, `MLT_MAX_DIMS`, `MLT_RECORD_SLOTS`,
      flag bits as-is.
- [x] 4.2 Keep every existing hostless lock in `tests/test_wavefront_state.py`
      and `tests/test_sppm_state.py` green — assertions unmodified, parser
      imports now coming from `src/` (both local parser copies deleted); do
      not weaken or delete any assertion; all `*_STRIDE` constants
      numerically unchanged (RGB + spectral, scalar + MSL).
- [x] 4.3 Extend the gpu-marked `_reflect_msl_layout` lock set to cover
      `StdSurfaceParams` via the module's MSL layout (the `fc` lock landed in
      task 2.4); guarded runner, `-m gpu`.

## 5. Verification and docs (Stage 4)

- [ ] 5.1 Full hostless sweep: `.venv/bin/pytest` (layout tests, matrix
      construction, metrics, import) — zero regressions.
- [x] 5.2 gpu-marked layout locks + kill-harness rules respected:
      `PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest
      tests/test_wavefront_state.py tests/test_metal_std_surface_layout.py
      tests/test_metal_flat_material_layout.py -m gpu -q` (one Metal process,
      progress logged).
- [x] 5.3 Parity spot check bit-identical pre/post: path + bdpt ×
      megakernel + wavefront on Metal at fixed seed via the parity harness;
      confirm every produced `.spv` byte-identical.
- [x] 5.4 Update `docs/Architecture.md` with a "Byte-layout ownership"
      subsection (module, owned structs, blob rule, drift gates); sweep other
      docs per the documentation-upkeep rule; `ruff check src/`.
- [ ] 5.5 `openspec validate reflection-owned-byte-layouts` clean; archive
      after merge.

## Change notes (implementation)

* **Stage 0** — `src/skinny/slang_layout.py` + `tests/test_slang_layout.py`
  (66 hostless tests). Every golden stride matched the hand-authored value on
  the first derivation, including the 568 B / 600 B `fc` blobs and
  `mltSigma@564`.
* **Type coverage** — `float4x4` (64 / 64@16), `uint2` (8 / 8@8), `uint3`
  (12 / 16@16) and recursive nested-struct flattening (`camera.<field>`,
  `SampledWavelengths`) were added for `FrameConstants`. The task-2.4 gpu lock
  confirmed all of them against live Metal reflection: 656 B RGB, 688 B MLT,
  65/73 fields, every offset equal.
* **Guard placement (task 2.3)** — the coverage guard sits at `_pack_uniforms`'s
  return rather than at each upload call site: one guard covers all four Vulkan
  upload sites *and* the Metal packer's scalar source.
* **`_VK_UNIFORM_BUFFER_BYTES`** — the import-time bound now derives from the
  MLT blob (600 B), the longest a Vulkan upload carries; it previously checked
  only the 568 B base blob.
* **Byte-invariance evidence** — derived tables equalled the hand tables
  byte-for-byte before each deletion (temporary migration test, since removed);
  the material packers hash identically to `main` over an 8-material corpus
  (flat + std-surface + MSL relocation, RGB and spectral); `int_caustic`
  renders on Metal are bit-identical pre/post for path/bdpt × megakernel/wavefront
  and mlt/wavefront (maxdiff 0 on all five).
* **SPIR-V** — no `.slang` source is modified (`git diff main -- shaders` is
  empty). A fresh `slangc` compile of `main_pass.slang` from either tree with
  the same include dirs is byte-identical; the raw pre/post artifacts differ
  only because the primary checkout carries three stale *generated*
  `wavefront/shade_*.slang` files (untracked codegen), which is per-worktree
  drift, not a layout change.
* **Test consumers re-pointed** — `test_mlt_host` (permanent blob-order lock,
  now also anchored to `slang_layout.fc_scalar_blob`), `test_sppm_selection`
  (was grepping renderer source text for `("sppmGroupPmfE", 4)`),
  `test_struct_layout`, `test_metal_std_surface_layout`, and both duplicate
  parsers in `test_wavefront_state` / `test_sppm_state`.
* **Out of scope, unchanged** — `SkinParameters.pack()` (std140),
  `INSTANCE_STRIDE`, the light-buffer records; see the design's Open Questions.
