# Design review — flat-material-field-table

Adversarial review, 2026-07-27, against the tree at `8247148`. Recorded rather
than folded. **Fold before implementing.**

**Verdict: premise wrong in three places; rewrite before it is scheduled.** The
goal is right; the mechanism, the gap analysis and the enforcement point are all
mis-stated.

## MAJOR

**M1 — D1's core premise is false: lane assignment *is* derivable.**
`shaders/common.slang:110-141` declares **24 `property` accessors** in one
uniform machine-parseable form — `property float roughness { get { return
_diffuseColorRoughness.w; } }`, `property uint channelMask { get { return
asuint(_normalScaleChannelMask.w); } }`, and so on. Every meaningful lane is
named there; the **only** unnamed lane in the record is
`_cloudDensityWispinessFrequency.w` (pad). And `slang_layout.py:105`
(`_SKIP_PREFIXES = ("property", …)`) **deliberately discards** exactly this
information today.

This matters beyond tidiness. A golden "captured from the current packer" (task
1.1) records what the packer *does*, including any existing lane bug; a lane map
derived from the property block records what the shader *reads*. Those are the
two sides the change exists to reconcile — pinning one from the other makes the
gate vacuous.

Fix: extend `parse_struct_fields` with a property-accessor pass (unwrap
`asuint`/`asfloat`), derive lanes, pin only the one pad lane. Rewrite D1 as
"derive lanes from the property block; the golden is the independent leg, not
the source." This also settles the first open question: the table must live in
`slang_layout.py`, because its derivation source is the parser.

**M2 — `StdSurfaceParams` already has a derived name→offset field table.**
`slang_layout.scalar_layout("StdSurfaceParams").offsets` is name→(offset,size)
for all 44 fields — all individually named in `mtlx_std_surface.slang`, no
opaque rows — and `renderer.py:948` already consumes it as
`_STD_SURFACE_SCALAR_ENTRIES`. The remaining work for this record is one line of
scope: make `pack_std_surface_params` (`renderer.py:827`) walk that entry list
instead of a 64-value positional tuple. No table, no golden, no lane
declaration.

Also "8 hand-written offset comments, size-only assert" is wrong twice: the size
assert (`:975`) is inside the *MSL relocator*, and the scalar packer has **no**
assert — its field order is already pinned externally by
`tests/test_slang_layout.py:64`.

**M3 — "The only drift guard is a size assert" is false, which invalidates D3
and its spec scenario.** Two guards exist, both hostless:
`tests/test_slang_layout.py:63-64` pins the **full ordered declared field list**
(name + type) of both records — a shader-side transposition of two same-typed
fields already fails today under plain `pytest`. And
`tests/test_struct_layout.py:225,227,229,251-253,259` already assert packer
**output values at byte offsets** (`opacityTextureIdx`@76,
`opacityThreshold`@92, `channelMask`@108, `transmissionColor`@128,
`diffuseRoughness`@140, `specularColor`@144).

The genuinely uncovered mode is narrower and different in kind: transposing two
same-typed **positional arguments in the packer's `struct.pack` call**
(`:797-826`). A name→shader-offset golden cannot catch that — the shader offsets
do not move. What catches it is name-keyed packing (which the change does
propose) plus a golden over **packed bytes** for a material whose every field
has a distinct value.

Fix: restate the gap as "partial coverage: ~6 of 30 packer fields pinned"; make
the gate a packed-byte fingerprint; drop the name→offset golden, already owned
by `shader-byte-layouts`.

**M4 — D2 ("unknown key is an error") is unimplementable at the packing seam.**
`parameter_overrides` is a shared multi-consumer dict, not this packer's
argument list. `pack_flat_material` reads 30 keys; `pack_std_surface_params`
(`:827-947`) reads **44** from the same dict. `usd_loader.py:504`
(`_store_shader_override`) writes the raw MaterialX input name for **every**
authored shader input, plus the OpenPBR alias (`:507`) and the flat alias
(`:510`) — so a standard_surface/OpenPBR material carries `base_weight`,
`fuzz_color`, `geometry_opacity`, `emission_luminance`, … which no packer reads.
`pbrt/media.py:76,128,129,175` adds `pbrt_medium`, `volume_grid_field`,
`subsurface_eta`; `pbrt/api.py:401` adds `volume_interface`.

Under the spec as written, essentially every real material errors. Report-only
staging would surface hundreds of legitimate keys, with no criterion separating
them from stale ones.

Fix: invert the check. Own the *authoring* vocabulary as a union set and
validate at the author sites — `_store_shader_override`, the `skinnyOverrides`
merge (`usd_loader.py:732`, `:1244`), `apply_material_override`
(`renderer.py:7029`), `apply_material_overrides` (`:7064`) — not at the packer.
Or scope packer rejection to "keys in this packer's declared dialect", which
needs a dialect column the design lacks.

(No dynamically-constructed override key names exist — only `str(k)` passthrough
at `usd_loader.py:732` and registry-driven names at `params.py:408-415`. A
static vocabulary is feasible; the problem is scope, not dynamism.)

**M5 — D5: the double derivation is not a merge-ordering bug, and the
prescribed fix is a behaviour change.** The two `_derive_opacity_from_subsurface`
calls are in different functions on different objects. `usd_loader.py:1176` runs
inside `_load_mtlx_materials`, building the **shared per-`.mtlx`-leaf table
entry** from the MaterialX document alone — there is no stage prim in scope, so
`skinnyOverrides` is *structurally* unavailable, not merely out of order.
`:1253` runs inside `_merge_prim_overrides`, per prim binding, on an independent
copy (`:1204`), after merging that prim's customData. The second derivation has
an input the first cannot have.

Task 5.1 is unachievable without either deleting the first derivation — changing
the shared table entry for every other consumer — or hoisting all of
`_load_mtlx_materials`'s folds (including the inline `transmission` fold at
`:1157-1171`) to binding time.

Second defect: the spec prescribes "merge, then opacity-from-transmission,
opacity-from-subsurface, coat canonicalisation". But `_canonicalize_coat` has
exactly one call site (`usd_loader.py:748`, `_extract_material`), and
`_load_mtlx_materials` never calls `_derive_opacity_from_transmission` at all.
Imposing that order on the mtlx path **adds a coat fold that does not run
today**, contradicting "Unchanged: rendered images".

Fix: cut D5, or rescope to what it is — "the `.mtlx` fallback derives twice by
construction because the shared table entry predates the binding; make
idempotence explicit and assert it" — and drop the third spec requirement.

## MINOR

- Citations: `pack_flat_material` is `renderer.py:609` not 792;
  `pack_std_surface_params` is `:827` not 913; `_STD_SURFACE_TO_FLAT_PACK` is
  `mtlx_synthesis.py:1144` not 1147. Correct as cited:
  `_STD_SURFACE_TO_FLAT` `usd_loader.py:763`, the re-run `:1246-1253`, the
  docstring map `:629-673`.
- **64 arguments, not 60** (`:797-798`; `struct.calcsize` = 256).
- **30 override keys, not 31** (29 in the body plus `volume_interface`
  transitively via `_material_is_volume`, `:498`).
- "14 opaque float4 rows" understates what is derived: the struct declares **25**
  fields (`common.slang:59-108`); 11 are individually named scalars/uints with
  fully derived names and offsets. Only ~9 rows are genuinely lane-packed.
- **The 7-entry disagreement is real but the conclusion is not.**
  `_STD_SURFACE_TO_FLAT` is a **rename** table read with an identity fallback at
  three of four call sites (`usd_loader.py:688,1108,1115,1131` use `.get(k, k)`);
  its 7 identity entries are *correctly absent*. `_STD_SURFACE_TO_FLAT_PACK` is a
  **whitelist** whose `.values()` feed `FLAT_PACK_WRITABLE_KEYS`
  (`mtlx_synthesis.py:1174-1177`), where identity entries are load-bearing.
  Budget one task for the audit, not seven — and D4 must not collapse a rename
  table and a whitelist into one projection.
- Ownership: the derivation half overlaps `shader-byte-layouts`' existing
  "Layout drift fails hostless tests" requirement — make it `MODIFIED` and keep
  only the new obligation (name-keyed packing + byte fingerprint) as `ADDED`.
  On the vocabulary side, note that `resolve_material`'s internal lobe names
  differ from the packer's (`coat_ior` at `pbrt/materials.py:540,575` vs the
  packer's `coat_IOR`, translated by `_MTLX_LOBES` at `:659`, dropped by
  `_USD_LOBES` at `:636-645`). The vocabulary owner must own that translation or
  it re-diverges immediately.
- The `renderer-pure-core-extraction` dependency holds for task 3.2 only; task
  2.2 needs no SDK (`tests/test_slang_layout.py` is already hostless).
- Stale comment to fix in passing: `common.slang:50` still says "128 B / record";
  it has been 256 B since the volume/cloud rows landed.
