# Change: flat-material-field-table

## Why

`shader-byte-layouts` made the flat-material **stride** derived from the Slang
declaration. The **field meanings inside that stride** stayed hand-written, and
that is where the contract actually lives.

`pack_flat_material` (`renderer.py:792`) is a 60-argument positional
`struct.pack`. The struct declares 14 opaque `float4` rows
(`shaders/common.slang:57-102`), so `slang_layout` derives the row offsets but
not which lane inside a row means `roughness` and which means `metallic` —
that map is a **docstring** at `renderer.py:627-672`. The only drift guard is a
size assert. Transposing two same-typed arguments changes every rendered pixel
and passes every test.

The names that reach the packer are raw strings, and **three tables claim to
own the vocabulary**, with no shared constant:

- `usd_loader._STD_SURFACE_TO_FLAT` (`usd_loader.py:763`) — **5 entries**
- `mtlx_synthesis._STD_SURFACE_TO_FLAT_PACK` (`mtlx_synthesis.py:1147`) —
  **12 entries**, carrying the comment "Keep in sync with the override keys
  `pack_flat_material` consumes in renderer.py"
- `mtlx_synthesis._PREVIEW_SURFACE_FLAT_KEYS` — a third frozenset, with its own
  "Keep in sync with pack_flat_material" comment

The first two disagree by 7 entries. Today that is harmless only because those
7 names happen to be spelled identically in both dialects and
`_store_shader_override` writes the raw name unconditionally — a coincidence,
not a design.

`pack_flat_material` reads **31 distinct override key strings**, authored
across `pbrt/materials.py` and `pbrt/media.py`, merged and derived in
`usd_loader.py`, advertised as editable by `mtlx_synthesis.py`, and consumed in
`renderer.py`. A typo anywhere in that chain is a silently ignored override.

The same chain is where the `customData["skinnyOverrides"]` merge ordering bug
lives: `usd_loader.py:1246-1253` has to **re-run**
`_derive_opacity_from_subsurface` because the first derivation ran before
customData was merged.

`pack_std_surface_params` (`renderer.py:913`) has the same shape — 64 floats,
8 hand-written offset comments, size-only assert.

## What Changes

- Add a field table for the flat-material and std-surface records: name →
  (row, lane, kind, default), derived from the Slang declaration where the
  declaration is specific enough and pinned by a permanent golden where it is
  not (the opaque `float4` rows).
- Packing takes names, not positions. A misspelled or unknown override key is
  an error at the packing seam, not a silently dropped value.
- The three alias tables read the field table instead of restating it; the
  "keep in sync" comments become tests.
- Add a transposition gate: a permanent golden of (name → byte offset) for
  every field, so swapping two same-typed fields fails.
- Fix the `skinnyOverrides` merge ordering so no derivation runs twice — the
  merge and the derivations are ordered once at the intake seam.
- Pure refactor of the packed bytes: for the same inputs, the emitted bytes are
  identical before and after.

## Capabilities

### Modified Capabilities

- `shader-byte-layouts`: ownership extends from strides and row offsets down to
  **named fields** for the flat-material and std-surface records, with
  name-keyed packing and a transposition gate — a size-equal assert is no
  longer sufficient for these two structs.
  The same requirement takes ownership of the override key vocabulary that
  `pbrt-material-resolution`'s resolver emits and that three hand-synced alias
  tables currently restate — the resolver's own behaviour is unchanged; only
  the vocabulary gains an owner, and unknown keys stop being silently dropped.

## Impact

- Modified: `src/skinny/slang_layout.py` (field table), `renderer.py`
  (`pack_flat_material`, `pack_std_surface_params`, plus their MSL variants),
  `usd_loader.py` (alias table, merge/derivation ordering),
  `mtlx_synthesis.py` (two tables), and the pbrt authors that emit key strings.
- Unchanged: shader structs, stride values, GPU behaviour, rendered images.
- Depends on `renderer-pure-core-extraction` for the packers to be hostlessly
  testable; without it these tests remain gated on the Vulkan SDK.
- Docs: `docs/Architecture.md` byte-layout section; `docs/PbrtImport.md` for
  the override key vocabulary.
