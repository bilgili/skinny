## Why

The pbrt v4 importer drops every `loopsubdiv` shape (`report.skipped(... "unsupported
shape type")`), so canonical pbrt scenes whose subject is a subdivision surface import
as near-empty. Converting `killeroos/killeroo-simple.pbrt` yields only the sphere light
and two backdrop quads — both killeroo bodies (each a `loopsubdiv` control cage in
`geometry/killeroo.pbrt`) vanish. `loopsubdiv` is a first-class pbrt shape, and the
importer's purpose is pbrt→USD parity, so silently discarding the main geometry is a
correctness gap.

## What Changes

- Add `loopsubdiv` to the importer's shape translation. The control cage (`point3 P`,
  `integer indices`) is tessellated to triangles **at import time**, mirroring how pbrt
  itself converts `loopsubdiv` to a triangle mesh before rendering.
- Tessellation is **pbrt-exact**: apply Loop subdivision `integer levels` times (interior
  + boundary vertex/edge rules with valence-weighted β), push the final vertices to their
  Loop **limit positions**, and compute **limit-surface normals** via the Loop tangent
  masks — matching `pbrt-v4`'s `loopsubdiv.cpp`.
- Route the resulting positions / indices / normals through the existing `trianglemesh`
  emit path (`bake_world_mesh` → `add_mesh`), so CTM baking, winding/reverse-orientation,
  default-UV synthesis for textured shapes, and area-light attachment all apply unchanged.
- The translation report counts `loopsubdiv` as `ok` (exact) instead of `skipped`.
- No renderer, USD-loader, or shader change: the renderer keeps consuming raw triangles;
  `SubdivisionScheme` stays `none`.

## Capabilities

### New Capabilities
- `pbrt-loopsubdiv-shape`: Translation of pbrt `loopsubdiv` shapes into triangle-mesh
  UsdGeom prims via pbrt-exact Loop subdivision (limit positions + limit normals),
  honoring `levels`, `P`, `indices`, the accumulated CTM, and reverse orientation.

### Modified Capabilities
<!-- none — pbrt-scene-import's "Shape translation" requirement lives in the still-in-flight
     pbrt-v4-scene-import change (not yet an archived spec), so this lands as its own
     capability, consistent with the pbrt-texture-uv follow-up. -->

## Impact

- **Code**: `src/skinny/pbrt/api.py` (`_shape_geometry` dispatch); new
  `src/skinny/pbrt/loopsubdiv.py` (subdivision + limit evaluation); `src/skinny/pbrt/emit.py`
  if a shared helper is needed. Pure-Python/NumPy, no new dependency.
- **Tests**: `tests/` — unit tests for the subdivision kernel (regular-valence limit
  positions/normals, boundary rule, `levels=0/1/2` triangle counts) and an importer
  round-trip; a killeroo import smoke test asserting the bodies are no longer skipped;
  optional pbrt parity-gate scene.
- **Docs**: pbrt importer shape-support list (README / importer docs) gains `loopsubdiv`.
- **Behavior**: scenes previously importing empty now carry their subdivision geometry;
  no change to scenes without `loopsubdiv`.
