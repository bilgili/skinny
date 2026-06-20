## Context

`src/skinny/pbrt/api.py::_shape_geometry` dispatches on `shp.type` and handles
`trianglemesh`, `plymesh`, and `sphere`; everything else hits the terminal
`report.skipped(f"shape:{t}", "unsupported shape type")` and contributes no
geometry. `loopsubdiv` — a pbrt Loop subdivision surface defined by `integer
levels`, `point3 P`, `integer indices` — therefore vanishes. In
`killeroos/killeroo-simple.pbrt`, both killeroo bodies are `loopsubdiv`
(`geometry/killeroo.pbrt`), so the import lands as a sphere light + two quads.

skinny's runtime has **no** subdivision-surface support: `emit.add_mesh` hardcodes
`CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)`, the USD loader bakes raw
triangles, and the renderer consumes triangle soup (the only "subdivision" path is
the skin head's displacement rebake, unrelated to USD subdiv schemes). Emitting a
USD mesh tagged `subdivisionScheme = "loop"` and hoping the renderer refines it is
therefore a dead end — nothing downstream would honor it.

pbrt resolves this the same way we must: `loopsubdiv.cpp` tessellates the control
cage to a concrete triangle mesh (with limit positions and limit normals) *before*
rendering. We replicate that conversion at import time and feed the existing
`trianglemesh` emit path.

## Goals / Non-Goals

**Goals:**
- Translate `loopsubdiv` shapes to triangle-mesh UsdGeom prims that match pbrt's
  tessellation: Loop subdivision applied `levels` times, vertices pushed to limit
  positions, per-vertex limit-surface normals.
- Reuse the existing emit pipeline so CTM bake, winding/`reverse_orientation`,
  default-UV synthesis for textured shapes, and area-light attachment are unchanged.
- Report `loopsubdiv` as exact (`ok`), not skipped.
- Keep it pure-Python/NumPy — no new runtime dependency, no shader/renderer change.

**Non-Goals:**
- Runtime/GPU subdivision, USD `subdivisionScheme = loop` honored by the renderer,
  or adaptive/feature-adaptive subdivision. (Import-time tessellation only.)
- Crease/corner tags, holes, or non-triangle control cages — pbrt `loopsubdiv` is
  triangles-only and uncreased; we match that scope.
- UV subdivision. pbrt `loopsubdiv` carries no per-vertex UVs; textured cases fall
  through to the importer's existing default-UV synthesis (faceVarying per-triangle).

## Decisions

### D1 — Tessellate at import, route through the `trianglemesh` path
`_shape_geometry` gains a `t == "loopsubdiv"` branch that reads `P`/`indices`/`levels`,
calls a new `loopsubdiv.subdivide(...)`, and returns the same
`(points_local, indices, normals, uvs)` tuple the `trianglemesh` branch returns
(`uvs=None`). All downstream baking (`bake_world_mesh`, `add_mesh`, default-UV
synthesis, area-light hookup in `_emit_shape`) is reused verbatim.
*Alternative considered:* emit `subdivisionScheme = "loop"` and subdivide in the
loader/renderer — rejected: no consumer exists and it would touch the GPU pipeline,
far exceeding scope and diverging from how the renderer ingests geometry.

### D2 — New module `src/skinny/pbrt/loopsubdiv.py`
The subdivision math is self-contained and unit-testable in isolation, so it lives in
its own module rather than bloating `api.py`. Public entry:
`subdivide(points, indices, levels) -> (points, indices, normals)`.
*Alternative:* inline in `emit.py` — rejected; emit.py is thin USD-authoring glue and
the algorithm warrants its own test surface.

### D3 — pbrt-exact Loop rules (limit positions + limit normals)
Implement the same scheme as `pbrt-v4`'s `loopsubdiv.cpp`:
- **Topology**: build the half-edge / face adjacency once, identify boundary edges
  (edges incident to a single face) and per-vertex valence and boundary flags.
- **Per-level refinement** (`levels` iterations):
  - *Even (existing) vertices* — interior: `(1 − n·β)·v + β·Σ neighbors`, with
    `β = 1/n·(5/8 − (3/8 + 1/4·cos(2π/n))²)` (pbrt's `beta(n)`; `n` = valence).
    boundary: `3/4·v + 1/8·(v_prev + v_next)` using the two boundary neighbors.
  - *Odd (new edge) vertices* — interior edge: `3/8·(a+b) + 1/8·(c+d)` (edge
    endpoints a,b; opposite vertices c,d of the two incident faces). boundary edge:
    `1/2·(a+b)`.
  - Each input triangle becomes 4 output triangles using the 3 new edge vertices.
- **Limit pass** (after the last refinement, matching pbrt's `weightOneRing`/
  `weightBoundary` final positions): push each vertex to its limit position
  (interior: `gamma`-weighted one-ring with `gamma = 1/(n + 3/(8β))`; boundary:
  `3/5·v + 1/5·(v_prev+v_next)`), and compute the **limit normal** from the two Loop
  tangent masks (`Σ cos(2πk/n)·e_k`, `Σ sin(2πk/n)·e_k` interior; the pbrt boundary
  tangent masks for valence 2..≥4 on the boundary) → `normalize(cross(t_u, t_v))`.
*Alternative — N-level subdivide only (no limit push), smooth normals from the final
mesh:* rejected by the chosen fidelity level; it drifts from pbrt on silhouettes and
shading and would not track a tight parity gate.

### D4 — Validation and degenerate input
If `P`/`indices` are absent/empty or `indices.size % 3 != 0`, emit
`report.skipped("shape:loopsubdiv", "<reason>")` and return `None` (no partial
geometry). `levels` defaults to pbrt's default; `levels = 0` returns the cage pushed
to its limit (still computes limit normals), with no refinement.

## Risks / Trade-offs

- **β / limit-weight formula drift from pbrt** → Mitigation: unit-test regular-valence
  (n=6) interior limit positions and normals against hand-derived values, and the
  boundary rule against a known strip; cross-check triangle counts (`4^levels · F`).
- **Non-manifold / open cages (boundaries) mishandled** → Mitigation: boundary
  detection from single-incidence edges with explicit boundary even/odd/limit rules;
  test an open patch (a disc) so boundary vertices stay on the boundary curve.
- **Performance on high `levels`** (4× triangles/level, Python/NumPy) → Mitigation:
  vectorize per-level neighbor sums with NumPy; real scenes use `levels` 1–3
  (killeroo = 1). Pathological `levels` is a scene-author choice, same as in pbrt.
- **Winding vs `reverse_orientation` and pbrt's B-matrix flip** → Mitigation: do not
  re-handle winding in the subdivider; emit cage-order triangles and let the shared
  `bake_world_mesh(reverse=…)` apply the importer's established convention, identical
  to the `trianglemesh` path (covered by existing winding tests + a loopsubdiv parity
  scene).
