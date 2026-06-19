## Context

The pbrt v4 importer (`src/skinny/pbrt/`) emits USD that the existing
`usd_loader.py` renders. For imagemap textures the importer already authors a
`UsdUVTexture` shader (file/wrap/sourceColorSpace) connected into
`UsdPreviewSurface`, and the loader resolves that to a texture binding and bakes
the mesh's `primvars:st` into the vertex buffer (flipping V to USD convention).
The explicit-UV emission path is in place:

- `ply.read_ply` → `PlyMesh.uvs` from `u/v`, `s/t`, or `texture_u/texture_v`
  (ascii + binary little/big endian).
- `api._shape_geometry` → `uv`/`st` params for `trianglemesh`, `mesh.uvs` for
  `plymesh`.
- `emit.add_mesh(..., uvs=)` → `primvars:st` (`TexCoord2fArray`, vertex interp).
- `emit.bake_world_mesh` flips winding by reversing per-face index order only;
  per-vertex `st` stays aligned (st is indexed via `faceVertexIndices`).

Two gaps remain: (1) no regression test guards this path; (2) shapes with no
source UV (`sphere` via `tessellate_sphere` returns `uvs=None`;
`trianglemesh`/`plymesh` lacking `uv`, ~69/200 real bathroom meshes) get no `st`,
so a bound texture samples one constant texel instead of pbrt's default
parametrization. The parity gate (`tests/pbrt/test_parity.py`) never renders a
textured scene, so end-to-end GPU sampling is unverified.

pbrt-v4's default UVs (confirmed in source): `Triangle` with no `uv` array uses
per-triangle `{(0,0),(1,0),(1,1)}` (`shapes.h:897`); `Sphere` uses
`u = φ/φmax, v = (θ−θmin)/(θmax−θmin)` (`shapes.cpp:54`).

## Goals / Non-Goals

**Goals:**
- Regression-test the working explicit-UV path (trianglemesh + ply ascii/binary).
- Synthesize pbrt-faithful default UVs for UV-less *textured* shapes so imagemap
  textures sample like pbrt instead of at a constant point.
- Prove end-to-end GPU texture sampling with a textured parity scene + pbrt
  reference EXR inside the existing gate.
- Correct the stale `docs/PbrtImport.md` imagemap row.

**Non-Goals:**
- No changes to `usd_loader.py`, descriptor bindings, or the render path — they
  already consume `primvars:st` + `UsdUVTexture`.
- No `UsdPrimvarReader_float2` st-reader chain on the `UsdUVTexture` (the loader
  binds the mesh's single `st` set directly; multi-UV-set is out of scope).
- No procedural/`scale`/`mix` texture UV work beyond the existing imagemap wiring.
- Not pixel-exact parity on default-UV meshes (triangulation order differs from
  pbrt); the gate scene uses explicit UVs for a deterministic comparison.

## Decisions

**D1 — Default UVs only for textured shapes.** pbrt assigns default UVs to every
primitive, but authoring `st` on every untextured mesh would bloat large scenes
(e.g. 800+ bathroom meshes, faceVarying = 3 vec2 per triangle). Decision:
synthesize default UVs only when the shape's bound material references a texture.
`_emit_shape` already knows the material; thread a `needs_uv` flag from a
material→texture lookup. *Alternative rejected:* always synthesize (simpler, but
wasteful memory on untextured geometry).

**D2 — Triangle default UVs are faceVarying.** pbrt's `{(0,0),(1,0),(1,1)}` is
per-triangle, not per-vertex, so a shared vertex gets different UVs in different
triangles. This requires `primvars:st` with `faceVarying` interpolation: `3·M`
values for `M` triangles, each triple `(0,0),(1,0),(1,1)`. `add_mesh` gains a UV
interpolation mode. *Alternative rejected:* per-vertex default `(0,0)` everywhere
(constant sample, defeats the purpose).

**D3 — Sphere default UVs are per-vertex parametric.** `tessellate_sphere`'s grid
maps cleanly to pbrt's sphere parametrization: vertex `(i,j)` over `rings×segments`
gets `u = j/segments`, `v = 1 − i/rings` (its `θ = π·i/rings` so `v = 1 − θ/π`
matches pbrt's `(θ−θmin)/(θmax−θmin)` with `θmin=π, θmax=0` for a full sphere).
Returned as vertex-interp `st`. Sphere always carries UVs (cheap, low vertex
count) — simpler than gating, and matches pbrt which always parametrizes spheres.

**D4 — faceVarying default UVs are authored on the *baked* (post-winding-flip)
triangles.** Winding flip reverses per-face vertex order; default UVs are assigned
to the final stored triangle order, so they are generated after `bake_world_mesh`
against `world_indices`. This is consistent (every final triangle gets the same
`(0,0),(1,0),(1,1)`) even though it won't be bit-identical to pbrt's pre-flip
assignment — acceptable for the non-gated default case (see Non-Goals).

**D5 — Parity scene uses explicit UVs + a high-contrast imagemap.** To make the
gate deterministic and sensitive to UV correctness, the new corpus scene is a
`trianglemesh` quad (or two) with explicit `uv` and a checker/gradient PNG diffuse
under an area or infinite light. Explicit UVs (not defaults) keep skinny↔pbrt
triangulation identical so relMSE/FLIP reflects sampling fidelity, not mesh
tessellation. Reference EXR rendered with the pinned pbrt-v4 build, committed via
`git add -f`, tolerance recorded in the corpus manifest.

## Risks / Trade-offs

- **faceVarying st + winding flip mismatch vs pbrt** → default-UV meshes are not
  pixel-exact; mitigated by keeping the parity gate on explicit-UV geometry and
  flagging default UVs as "approximated" in the parity matrix.
- **Texture PNG asset in the repo** (`.gitignore` is `*`) → commit the corpus PNG
  + EXR with `git add -f`; keep the image tiny (≤64²) to bound repo size.
- **GPU/thermal budget for the new gate scene** → reuse the existing harness;
  one extra low-res (≤256px) scene; runs under the same `test_parity.py` skip
  guards (no GPU/refs ⇒ skip cleanly).
- **Material→texture lookup at shape-emit time** → if a shape's material can't be
  resolved, fall back to no default UVs (status quo, no regression) and report it.
