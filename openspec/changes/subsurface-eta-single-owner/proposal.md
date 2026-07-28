# Change: subsurface-eta-single-owner

## Why

A pbrt `subsurface` material with a named-spectrum `eta` crashes the importer.

```
MakeNamedMaterial "s" "string type" "subsurface" "spectrum eta" "glass-LASF9"
```

```
ValueError: could not convert string to float: 'glass-LASF9'
  src/skinny/pbrt/api.py:380      _author_material
  src/skinny/pbrt/media.py:150    subsurface_overrides
  src/skinny/pbrt/parser.py:61    Param.floats
```

The crash is not one missing guard. The pbrt `subsurface` branch has **two full
implementations** of "pbrt parameters to medium coefficients":

| Site | Named-spectrum `eta` | Unit pre-division | Emits `ior` | Reaches the stage |
|------|----------------------|-------------------|-------------|-------------------|
| `materials._subsurface_overrides` (line 384) | guarded by hand | no | no | no — filtered by `api._OVERRIDE_ONLY_INPUTS` |
| `media.subsurface_overrides` (line 133) | **unguarded** | yes | yes | yes |

The two copies already diverge in three ways. This is the defect. A third reader,
`resolve_material`'s `lobes["ior"] = scalar("eta", 1.33)`, is correct today: it
routes through `get_float_texture`, which resolves a named glass to its d-line
index.

`KnownBugs.md` item 4 records the crash and states the conclusion:
*"The two `subsurface_overrides` implementations should be one."*

The spec `pbrt-material-resolution` already requires exactly one reader:
*"pbrt-param interpretation SHALL live in exactly one resolver ... and subsurface
coefficient precedence."* `media.subsurface_overrides` violates that requirement.

**Downstream consequence.** `material_pack.pack_flat_material` reads the boundary
IOR from the `ior` lane and never reads `subsurface_eta`
(`slang_layout.INTAKE_ONLY_KEYS`). The two values must agree, or the rendered
boundary IOR does not match the authored glass. Today they agree by coincidence,
because every corpus scene writes a numeric `eta`.

## What Changes

**Ownership.** The resolver in `materials.py` owns pbrt-parameter interpretation,
including the `eta` rule. `media.py` owns the USD emission convention: the mm
per scene unit pre-division and the `ior` carry. Neither owns both.

**One `eta` resolution per material.** `resolve_material` reads `eta` once,
through `get_float_texture`, and passes the resolved float to the coefficient
builder. `lobes["ior"]` and `subsurface_eta` then carry one value by
construction, not by coincidence.

**`media.subsurface_overrides` delegates.** It calls the resolver's coefficient
builder and applies only the unit convention and the `ior` carry. Its own
`ParamSet` reads are removed.

## Impact

- Affected specs: `pbrt-material-resolution`
- Affected code: `src/skinny/pbrt/materials.py`, `src/skinny/pbrt/media.py`
- Affected docs: `KnownBugs.md` (item 4 is resolved), `docs/PbrtImport.md`
- Byte-identical importer output for every scene with a numeric `eta`. The
  change adds behaviour for named-spectrum and texture-bound `eta` only.

## Non-Goals

- `KnownBugs.md` item 3, the texture-bound `reflectance` crash in
  `materials._subsurface_overrides` (`ParamSet.rgb` raises). Same module, same
  bug class, different parameter. It needs its own change.
- The four frozen flavour divergences in `KnownBugs.md` item 3's parent section.
