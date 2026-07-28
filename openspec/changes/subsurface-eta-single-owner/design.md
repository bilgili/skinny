# Design: subsurface-eta-single-owner

## Context

Three sites read the pbrt `eta` parameter on the `subsurface` branch. The table
below records what each one does today.

| # | Site | Reads `eta` with | Named spectrum | Texture | Output |
|---|------|------------------|----------------|---------|--------|
| 1 | `materials.resolve_material`, `scalar("eta", 1.33)` | `get_float_texture` | d-line index | default + note | `lobes["ior"]` |
| 2 | `materials._subsurface_overrides` | `p.get` + hand-rolled branch, else `ParamSet.floats` | d-line index | **raises** | `lobes["subsurface_*"]` |
| 3 | `media.subsurface_overrides` | `ParamSet.floats` | **raises** | **raises** | `skinnyOverrides` on the stage |

Site 2 and site 3 both build the full coefficient set. Site 2 does not reach the
stage: `api._OVERRIDE_ONLY_INPUTS` filters `subsurface_sigma_a`,
`subsurface_sigma_s`, `subsurface_g` and `subsurface_eta` out of the authored
shader inputs on both the UsdPreviewSurface path and the MaterialX path. Site 3
writes the values the renderer reads.

The two builders are not interchangeable. Site 3 pre-divides both sigma by
`PBRT_STAGE_METERS_PER_UNIT * 1000`, and adds an `ior` key. Site 2 does neither.
The same key names therefore hold different numbers in the two places.

## Goals

- One owner for pbrt-parameter interpretation on the `subsurface` branch.
- `lobes["ior"]` and `subsurface_eta` hold one value by construction.
- Byte-identical importer output for a numeric `eta`.

## Decision

### D1 — Split the two responsibilities that both copies conflate

The two copies exist because each site needs a coefficient dict, and each site
built one. They conflate two different jobs:

- **Interpretation**: which pbrt parameters a `subsurface` material reads, with
  what defaults, precedence, and named-spectrum or texture substitution.
- **Emission**: the USD stage unit convention (mm per scene unit) and which key
  the renderer reads the boundary IOR from.

Interpretation belongs to the resolver in `materials.py`, which the
`pbrt-material-resolution` spec already names as the sole owner. Emission
belongs to `media.py`, which owns every other `skinnyOverrides` medium payload.

Data flow after the change:

```
api._author_material       -> materials.map_material(pbrt_material)
api._author_material_mtlx       -> resolve_material
                                     -> get_float_texture(p, "eta")   <- THE read
                                     -> subsurface_medium_overrides(p, eta)
                                          -> subsurface_coefficients(...)
                                -> inputs {subsurface_sigma_a/_s/_g/_eta, ior, ...}
                           -> media.subsurface_overrides(inputs)
                                + mm-per-unit division
                                + "ior" carry
```

`media.subsurface_overrides` never sees a `ParamSet`. Both `_USD_LOBES` and
`_MTLX_LOBES` identity-map the four `subsurface_*` lobes, so `inputs` carries
them on either flavour.

### D2 — `eta` is read once, in the resolver, and both lanes descend from it

`resolve_material` already resolves `eta` for `lobes["ior"]` through
`get_float_texture`, which handles all three parameter types. That resolved
float is passed into the coefficient builder as an argument.

`eta` is a **required** argument, not an optional one. An `eta=None` fallback
would be a second contract — "interpret these params" versus "interpret these
params, I already resolved eta" — and the two differ in the accessor arguments
used and in note routing. The only caller that wanted the fallback was
`media.subsurface_overrides`, and D1 removes its need for one.

`subsurface.subsurface_coefficients` returns `eta` unchanged on every precedence
branch, so `subsurface_eta == ior` holds for all four branches.

**Discarded alternative:** copy the named-spectrum guard from site 2 into site 3.
It fixes the reported crash in two lines. It also leaves two precedence chains
that already differ in three ways, and leaves the texture-bound `eta` crash in
both. The next divergence costs nothing to introduce.

**Discarded alternative:** keep `media.subsurface_overrides(params)` and have it
call the resolver's builder with `eta=None`. This was the first implementation
and a pre-merge review measured it wrong: `api._author_material` calls
`map_material` **and** `media.subsurface_overrides` separately, so `eta` was
resolved **twice per material** — and the emitter's copy is the one that reaches
the stage, since `_OVERRIDE_ONLY_INPUTS` filters the resolver's `subsurface_eta`
out of the shader inputs and the loader applies `skinnyOverrides` last. That
leaves one implementation but two invocations, which is not the same guarantee:
the two calls are not even argument-equivalent (the resolver passes
`textures=`/`base_dir=`, the emitter passed neither), so the moment the `ior`
lane gains real texture support the textures-blind lane silently wins. D1 as
written — emission consumes the resolved intermediate — is what makes the
agreement structural.

### D3 — One read means one note, with nothing to suppress

`resolve_material` appends a note when a texture-bound or unrecognised-spectrum
`eta` degrades to the pbrt default. Because there is now exactly one read, there
is exactly one note — no site has to pass `notes=None` to suppress a duplicate.
Under the discarded `eta=None` variant the second reader *did* need that
suppression, which is another way of seeing that it was a second read.

## Risks

- **Report drift.** A texture-bound or unrecognised-spectrum `eta` produces one
  note, from site 1. Verified by the byte-identical corpus check plus a direct
  note assertion.
- **`ior` default.** Site 1 spells the default `1.33` as a literal; the
  coefficient chain spells it `subsurface.ETA_DEFAULT`, which is `1.33`. The
  change makes site 1 use the named constant so one edit moves both.
- **Emission contract goes implicit.** `media.subsurface_overrides` spells its
  five emitted keys out rather than spreading the resolved dict, so a key added
  to the resolver cannot silently land on `skinnyOverrides` and from there in
  `Material.parameter_overrides`.
