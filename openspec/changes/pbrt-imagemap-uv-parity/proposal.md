## Why

The pbrt v4 importer already wires `imagemap` textures into `UsdUVTexture`
nodes and authors `primvars:st` for meshes that ship explicit UV data (verified
round-trip on synthetic scenes and 131/200 real bathroom plymeshes). But two
gaps remain before imagemap textures render faithfully against pbrt: the working
explicit-UV path has **no regression test** locking it, and shapes that carry no
UV in the source (sphere; trianglemesh/plymesh with no `uv`/`st`, ~69/200 real
meshes) get **no `st` primvar at all** — so a texture bound to them samples a
single constant point instead of the per-pbrt default parametrization. The
parity gate also never exercises a textured scene, so end-to-end GPU texture
sampling is unverified, and `docs/PbrtImport.md` still claims imagemap textures
"need mesh UVs to sample".

## What Changes

- Lock the existing explicit-UV path with regression tests: pbrt `trianglemesh`
  `uv`/`st` params and PLY `u/v`, `s/t`, `texture_u/texture_v` (ascii + binary
  little/big endian) → `primvars:st` (vertex interpolation) on the USD mesh, with
  the bound `UsdUVTexture` resolvable to a texture binding.
- Synthesize pbrt-faithful **default UVs** when a shape carries no source UV:
  - `sphere` (`tessellate_sphere`) gets parametric per-vertex UVs `u = j/segments`,
    `v = 1 − i/rings` (matches pbrt `u = φ/φmax`, `v = (θ−θmin)/(θmax−θmin)`).
  - `trianglemesh`/`plymesh` with no UV get per-triangle **faceVarying** UVs
    `{(0,0),(1,0),(1,1)}` (matches pbrt `Triangle` `shapes.h:897`).
  - Default UVs are only synthesized when the shape's bound material references a
    texture (untextured meshes stay UV-free to avoid bloating large scenes).
- Add a **textured parity scene** (trianglemesh quad + imagemap diffuse) and its
  pbrt-v4 reference EXR to the existing parity gate (`tests/pbrt/test_parity.py`),
  proving end-to-end GPU texture sampling within the relMSE/FLIP tolerance.
- Refresh `docs/PbrtImport.md`: correct the stale `imagemap` parity-matrix row and
  document UV passthrough + default-UV synthesis; update `CHANGELOG.md`.

## Capabilities

### New Capabilities
- `pbrt-texture-uv`: UV handling for imagemap textures in the pbrt v4 importer —
  explicit-UV passthrough to `primvars:st`, pbrt-faithful default-UV synthesis for
  UV-less textured shapes, and a textured-scene entry in the parity gate.

### Modified Capabilities
<!-- pbrt-scene-import / pbrt-parity-validation live only in the unarchived
     pbrt-v4-scene-import change, not in openspec/specs/, so there is no base spec
     to delta against; this work is captured as the new pbrt-texture-uv capability. -->

## Impact

- **Code:** `src/skinny/pbrt/emit.py` (`tessellate_sphere` UVs, default-UV helper,
  faceVarying `add_mesh` path), `src/skinny/pbrt/api.py` (`_shape_geometry` /
  `_emit_shape` default-UV wiring gated on textured material), no shader changes.
- **Tests:** new `tests/pbrt/test_uv.py` (unit), new corpus scene + reference EXR
  under `tests/pbrt/corpus/`, parity-gate row in `tests/pbrt/test_parity.py`.
- **Docs:** `docs/PbrtImport.md` parity matrix, `CHANGELOG.md`.
- **No** changes to the USD loader, descriptor bindings, or render path — the
  loader already consumes `primvars:st` and resolves `UsdUVTexture`.
