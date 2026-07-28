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
api._author_material          -> media.subsurface_overrides(params)
api._author_material_mtlx     ->   materials.subsurface_medium_overrides(params)
                                     -> subsurface.subsurface_coefficients(...)
                                   + mm-per-unit division
                                   + "ior" carry
```

`media.subsurface_overrides` performs no `ParamSet` read of its own.

### D2 — The resolver reads `eta` once and passes the value down

`resolve_material` already resolves `eta` for `lobes["ior"]` through
`get_float_texture`, which handles all three parameter types. That resolved
float is passed into the coefficient builder as an argument.

`subsurface_medium_overrides(params, eta=None)` resolves `eta` itself when the
caller supplies none. `media.subsurface_overrides` is such a caller: `api` hands
it a raw `ParamSet` and there is no resolver result in scope.

`subsurface.subsurface_coefficients` returns `eta` unchanged on every precedence
branch, so `subsurface_eta == ior` holds for all four branches.

**Discarded alternative:** copy the named-spectrum guard from site 2 into site 3.
It fixes the reported crash in two lines. It also leaves two precedence chains
that already differ in three ways, and leaves the texture-bound `eta` crash in
both. The next divergence costs nothing to introduce.

**Discarded alternative:** delete `materials._subsurface_overrides` and have
`api` read the coefficients from the resolver's lobes. `api` would then apply
the unit division to lobe values, which puts a stage convention inside the
authoring loop, and the resolved lobes are asserted by two hostless tests
(`test_subsurface_coeffs.py`, `test_material_resolve.py`) that exist to prove the
two mappers agree.

### D3 — The resolved `eta` carries no note at the coefficient site

`resolve_material` appends a note when a texture-bound `eta` degrades to the
scalar default. The coefficient builder must not append a second note for the
same read, or the import report gains a duplicate line and the report baselines
shift. When the builder resolves `eta` itself it passes `notes=None`, which is
what site 2 does today.

## Risks

- **Report drift.** A texture-bound or unrecognised-spectrum `eta` produces one
  note, from site 1. Verified by the byte-identical corpus check plus a direct
  note assertion.
- **`ior` default.** Site 1 spells the default `1.33` as a literal; the
  coefficient chain spells it `subsurface.ETA_DEFAULT`, which is `1.33`. The
  change makes site 1 use the named constant so one edit moves both.
