# Design: flat-material-field-table

## Context

Two byte-layout authorities already exist and work: `slang_layout.py` parses
the Slang declaration for strides and offsets, and `shader_variants.py` owns
the build matrix. This change extends the first one down a level, to the
fields inside the opaque rows.

Why the rows are opaque: `FlatMaterialParams` declares 14 `float4` rows rather
than named scalars. That is deliberate (it keeps the shader-side layout stable
and lets the packer pack densely) but it means the derivation stops at the row
and the meaning of each lane is carried by a docstring plus the argument order
of a 60-argument function.

## Goals / Non-Goals

**Goals**
- One table naming every field of the two records and its byte position.
- Name-keyed packing; unknown key is an error.
- A gate that catches transposition of two same-typed fields.
- One ordering for override merge and derivation at the intake seam.

**Non-Goals**
- Renaming the shader struct or splitting the `float4` rows into named
  scalars. That would change shader source and risk the compiled artifacts.
- Bringing `SkinParameters.pack`, `INSTANCE_STRIDE`, or the light records into
  the table. They are documented single-author exceptions; this change touches
  only the two material records where three external tables already try to
  mirror the vocabulary.
- Changing which overrides exist or what they mean.

## Decisions

### D1 — Derive where possible, pin where not

Row offsets and stride come from `slang_layout` as today. Lane assignment
inside a row cannot be derived from a `float4` declaration, so it is declared
in the table and **pinned by a permanent golden captured from the current
packer** — record reality, then let the table be the authority. Same discipline
as `shader-variant-key-module`'s flag-tuple fixture.

### D2 — Name-keyed packing, unknown key is an error

Today an override with a misspelled key is silently ignored, and the only
signal is a wrong-looking render. The packer takes a mapping and rejects keys
that are not in the table. This turns the 31-string chain from a convention
into a checked contract.

The risk is a scene that *currently* carries a stale key and renders fine.
Mitigation: run the whole pbrt corpus and the confirming suite through the new
packer in a report-only mode first, list every unknown key found, and decide
each (fix the author, or add the key) before switching to hard rejection.

### D3 — Transposition gate is a name→offset golden

The existing `assert off == len(scalar)` catches size drift only. The gate here
is: for every field name, its byte offset within the record, pinned. Swapping
two same-typed fields moves both offsets and fails. This is the single most
valuable test in the change — it covers the failure mode that is currently
invisible.

### D4 — Alias tables read the table; the comments become tests

`_STD_SURFACE_TO_FLAT` (5), `_STD_SURFACE_TO_FLAT_PACK` (12) and
`_PREVIEW_SURFACE_FLAT_KEYS` all restate part of the vocabulary. They become
projections of the field table plus their genuine dialect mapping. The two
"Keep in sync with pack_flat_material" comments become assertions.

Note the 7-entry disagreement is currently benign only because the names happen
to match across dialects. Before collapsing, verify each of the 7 explicitly —
a disagreement that is benign by coincidence may be hiding an intent.

### D5 — Merge ordering fixed here, not in scene intake

`usd_loader.py:1246-1253` re-runs `_derive_opacity_from_subsurface` because the
first derivation ran before `customData["skinnyOverrides"]` was merged. Three
separate readers of that key exist, each with its own merge/derive order. The
fix belongs with the vocabulary owner: one merge step, then one derivation
step, stated once. `scene-intake-interface` explicitly defers this here.

### D6 — MSL variants follow, they do not diverge

`pack_std_surface_params_msl` and `pack_mtlx_skin_array_msl` derive their
entries from `slang_layout` already. They consume the same field table with the
MSL offsets; there is no second table.

## Risks / Trade-offs

- **Risk: a corpus scene depends on a key the table does not have.** Handled by
  D2's report-only stage. Do not skip it — the corpus and the confirming suite
  are the only places these keys are exercised end to end.
- **Risk: report-only reveals that two dialect names must map to one field with
  different coercion.** Then the table needs a per-dialect coercion column;
  decide when measured, not now.
- **Trade-off: name-keyed packing is slower than positional.** Material packing
  happens on upload, not per frame, and `_upload_flat_materials` is already
  263 lines of Python. Measure, but this is not a hot path.
- **Dependency:** these packers live above `import vulkan` in `renderer.py`, so
  their tests skip without the SDK. `renderer-pure-core-extraction` should land
  first, or the transposition gate is only enforced on Vulkan-capable hosts —
  which is precisely the wrong place for it, since the Metal path packs the same
  bytes.

## Open Questions

- Should the field table live in `slang_layout.py` or in a new module beside
  it? Leaning: inside `slang_layout`, since it is the same authority one level
  down, and a second module would need to import it anyway.
- Do the pbrt authors emit table-derived constants, or keep string literals
  validated at the seam? Leaning: constants for the ~31 known keys, since a
  typo then fails at import rather than at pack time.
