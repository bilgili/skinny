# pbrt Procedural `cloud` Medium

## Why

`nanovdb-volume-rendering` shipped grid-file heterogeneous media (`type "nanovdb"` →
disney-cloud, bunny-cloud). But pbrt's canonical `clouds.pbrt` scene uses a **different** medium
source — `MakeNamedMedium "c" "string type" "cloud"` — pbrt's built-in *procedural* cloud: an
analytic fBm/Perlin density inside the unit cube, with **no grid file**. Converting it today emits
no volume (importer reports `medium:cloud … heterogeneous media unsupported`) and the bounding
sphere falls back to a grey diffuse ball. This change makes `clouds.pbrt` render as a cloud by
implementing the procedural density as a new medium kind through the seam the volume walk already
exposes — no new GPU resource, no new transport.

A second, smaller gap the same scene exposes: pbrt `Material ""` (empty-string material) on a
`MediumInterface` shape is a null/interface boundary exactly like `Material "interface"`, but the
importer maps it to the default grey `UsdPreviewSurface` instead of routing it to the volume path.

## What Changes

- **Procedural `cloud` medium import**: the pbrt importer stops skipping `MakeNamedMedium` of type
  `cloud`; it emits the medium as a procedural volume (a `UsdVol.Volume` with an fBm-source field
  or, more simply, the medium params on the bound interface material's `skinnyOverrides` — decided
  in design.md) carrying `sigma_a`/`sigma_s`/`g`/`density`/`wispiness`/`frequency`. Covers
  `clouds.pbrt` (`density 2`, default `wispiness 1`, `frequency 5`).
- **`MEDIUM_CLOUD` transport**: a new `densityAt` case that evaluates pbrt's exact
  `CloudMedium::Density` — classic Perlin `Noise`/`DNoise` (the 256-entry `NoisePerm` table +
  `Grad` + quintic `NoiseWeight`), the 5-octave fBm sum, the 2-iteration wispiness domain warp, and
  the altitude falloff — analytic, **no texture, no new binding**. `mediumMajorant` = the packed
  σ_t (density clamps to [0,1], same global-majorant argument as the grid). Reuses the whole volume
  walk (majorant/null-collision, HG phase, NEE, escape continuation) unchanged.
- **`Material ""` → null boundary**: the importer treats an empty-string material bound to a
  `MediumInterface` shape as an interface/null boundary (same encoding as `Material "interface"` —
  lobe-less + `volume_interface` marker), not the grey-diffuse fallback.
- **Parity gate**: add `clouds` to the pbrt corpus with a pbrt-truth reference and the standing
  mega≡wave self-consistency gate.

## Capabilities

### New Capabilities

- (none — extends existing capabilities)

### Modified Capabilities

- `heterogeneous-media`: a second density source (`MEDIUM_CLOUD`, analytic fBm) plugs into the
  `densityAt`/`mediumMajorant` seam alongside `MEDIUM_NANOVDB`; the "render at parity" requirement
  gains the `clouds` corpus scene.
- `pbrt-volume-import`: `MakeNamedMedium "cloud"` becomes supported (was a recorded skip), and
  `Material ""` joins `Material "interface"` as a null/boundary material.

## Impact

- **Importer**: `src/skinny/pbrt/media.py` (cloud params → overrides), `materials.py`/`api.py`
  (`Material ""` null boundary), `emit.py` (procedural volume emission if a Volume prim is used).
- **Renderer**: `src/skinny/renderer.py` (pack the cloud params + `mediumKind=MEDIUM_CLOUD`; no new
  descriptor binding — analytic density needs no texture).
- **Shaders**: new `materials/subsurface/cloud_noise.slang` (ported pbrt Perlin `Noise`/`DNoise` +
  `NoisePerm` table), `medium.slang` (`MEDIUM_CLOUD` `densityAt` case), `bindings.slang`
  (`MEDIUM_CLOUD` const); `main_pass.spv` recompile.
- **Parity**: `tests/pbrt/corpus/manifest.json` + a regenerated `clouds` reference EXR (pinned pbrt
  v4); `parity.py` if the combo table needs it.
- **Docs**: `docs/Architecture.md` medium-kind list, README/CLAUDE compatibility matrix, CHANGELOG.
- **Dependencies**: none — the fBm is self-contained Slang; the reference render uses the pinned
  pbrt v4 at `~/projects/pbrt-v4/build/pbrt`.
