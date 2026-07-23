# Change: reflection-owned-byte-layouts

## Why

The host↔GPU interface is a byte-offset agreement with the Slang structs, but
the layout is authored **twice, by hand, in three independent places**:

1. **FrameConstants** — `_FC_SCALAR_FIELDS` (renderer.py:180) plus the MLT tail
   tables (`_FC_MLT_FIELDS` / `_FC_SCALAR_FIELDS_MLT` / `_TILE_ORIGIN_Y_OFFSET`,
   :223–240) hand-mirror the Slang `FrameConstants` struct in `common.slang`;
   `_pack_uniforms` (renderer.py:10452, ~83 `struct.pack` appends) consumes the
   same order a third time implicitly. The Metal path already does this RIGHT:
   `_pack_uniforms_msl` (:10118) reuses the packed scalar blob and relocates
   every field via **reflected** offsets (`pipeline.uniform_layout`) with a
   drift-guard assert (:10153). The hand table survives only because the Vulkan
   direct-upload path still reads it.
2. **StdSurfaceParams** — `pack_std_surface_params` (renderer.py:853, 256 B
   scalar) and the hand-listed `_STD_SURFACE_FIELDS` relocation table (:973)
   consumed by `pack_std_surface_params_msl` (:992) re-author the
   `mtlx_std_surface.slang` struct field-by-field; `pack_flat_material` (:635)
   and `FLAT_MATERIAL_STRIDE = 256` (:321) do the same for `FlatMaterialParams`
   (`common.slang:57`) via a comment-only offset map.
3. **Wavefront record sizing** — `wavefront_layout.py` (473 lines) re-lists
   every wavefront/SPPM/BDPT/MLT shader struct's fields
   (`_path_state_fields`, `_visible_point_fields`, `_sppm_accum_fields`,
   `_bdpt_vertex_fields`, `_wf_bdpt_aux_fields`, the MLT sizers) to mirror
   `#if defined(SKINNY_SPECTRAL)` blocks in `wavefront_state.slang` /
   `wavefront_bdpt.slang`. Only comments tie the two authorings together.

A field reorder or retype in any of these Slang structs today produces **no
exception** — it surfaces as garbled GPU output (or a silently undersized
allocation) that has to be diagnosed visually. The pattern that already fixed
this once (`_pack_uniforms_msl`: derive offsets, assert coverage) exists but is
not extended to the other two authorings or to the Vulkan side.

## What Changes

- **New module `src/skinny/slang_layout.py`** becomes the single owner of every
  host-mirrored byte layout. It derives each struct's ordered field list by
  parsing the authoritative `.slang` declaration (promoting the proven
  `_parse_struct_fields` machinery from `tests/test_wavefront_state.py` into
  `src/`), and computes scalar (Vulkan, `-fvk-use-scalar-layout`) and MSL
  (Metal) offsets/strides with the deterministic layout math already living in
  `wavefront_layout.py`. Compile-variant gates (`SKINNY_SPECTRAL`,
  `SKINNY_MLT`) are resolved per variant, exactly as the test parser does today.
- **Packers and allocators become consumers.** `_FC_SCALAR_FIELDS` /
  `_FC_MLT_FIELDS` / `_TILE_ORIGIN_Y_OFFSET` / `_VK_UNIFORM_BUFFER_BYTES`,
  `_STD_SURFACE_FIELDS`, the `FLAT_MATERIAL_STRIDE` offset map, and the
  `wavefront_layout.py` field lists are replaced by (or verified against and
  then derived from) `slang_layout` queries. `struct.pack` call sites keep
  their shape; only the tables feeding them change owner.
- **Silent drift becomes a failing hostless test.** A stride/offset-equality
  check per mirrored struct (derived layout ↔ packer output length ↔ pinned
  golden strides) runs under plain `pytest` with no GPU. The existing
  gpu-marked Metal reflection lock tests are kept unweakened and extended to
  `fc`, `StdSurfaceParams`, and `FlatMaterialParams` as the ground truth for
  the MSL layout rules.
- **No shader changes.** No `.slang` file is touched, so the Vulkan SPIR-V is
  byte-unchanged by construction. Each migration stage is gated on
  byte-identical packer output (and bit-identical rendered output via the
  existing parity harness).

Non-breaking: no public API, CLI, or file-format change. `wavefront_layout.py`
keeps its public constants/sizers as a thin facade over `slang_layout`.

## Capabilities

### New Capabilities

- `shader-byte-layouts` — single derived authority for host-mirrored GPU byte
  layouts: parse-derived field lists, deterministic scalar/MSL layout math,
  hostless drift gates, and the bit-identical adoption constraints.

### Modified Capabilities

None. The `metal-backend` requirement "MSL-correct uniform layout on Metal"
(runtime offsets from compiled-module reflection, `set_data`-only upload) and
the `wavefront-execution` requirement "Wavefront path-state carries a spectrum
under the spectral define" (Python stream-layout mirrors match the shader
structs in both variants) remain satisfied verbatim — this change strengthens
how the mirrors are produced without altering what either spec requires.

## Impact

- Affected code: `src/skinny/renderer.py` (FrameConstants tables, StdSurface /
  flat-material packers), `src/skinny/wavefront_layout.py` (field lists become
  derived; public API unchanged), new `src/skinny/slang_layout.py`.
- Affected tests: `tests/test_wavefront_state.py` and
  `tests/test_sppm_state.py` (their duplicate parsers move to `src/`; tests
  keep independent pinned-stride + gpu reflection locks),
  `tests/test_mlt_host.py` (reads `_FC_SCALAR_FIELDS*` /
  `_TILE_ORIGIN_Y_OFFSET` by name at :189–211 — re-pointed at the derived
  tables and kept as the permanent MLT blob-order lock),
  `tests/test_struct_layout.py`, `tests/test_metal_std_surface_layout.py`,
  `tests/test_metal_flat_material_layout.py` — none weakened; new hostless
  drift-gate tests added.
- Cross-change coordination (soft, conflict-avoidance only):
  `tests/test_mlt_host.py` and the `_pack_uniforms` region are also touched
  textually by the other renderer-cluster changes (e.g.
  `renderer-module-carveout` Stage B). Sequence to avoid merge conflicts;
  there is no semantic prerequisite in either direction — each change
  updates the `test_mlt_host` assertions to whichever layout authority is
  current when it lands.
- Affected docs: `docs/Architecture.md` (layout-ownership section),
  `CLAUDE.md`/`AGENTS.md` pointers if they reference the hand tables.
- Not affected: all `.slang` sources and `.spv` artifacts (byte-unchanged),
  rendered output (bit-identical, gated), descriptor binding map, public
  Python API.
