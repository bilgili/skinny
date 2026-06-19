## 1. Regression tests — lock the working explicit-UV path

- [x] 1.1 `tests/pbrt/test_uv.py`: `trianglemesh` with `point2 uv` (and `st` alias) imports → mesh `primvars:st` (vertex interp) with matching per-vertex values
- [x] 1.2 PLY round-trip: ascii `u/v`, binary little-endian `s/t`, binary big-endian `texture_u/texture_v` → `read_ply().uvs` correct and emitted `primvars:st` matches
- [x] 1.3 Winding-flip test: negative-determinant CTM (`Scale -1 1 1`) on a UV'd mesh keeps `primvars:st` aligned to vertices (indices reversed, st unchanged)
- [x] 1.4 Run `tests/pbrt/test_uv.py` — these characterize current behavior and pass

## 2. Default-UV synthesis for UV-less shapes (TDD)

- [x] 2.1 Failing test: UV-less `trianglemesh` bound to an imagemap-textured material → `faceVarying` `primvars:st` of `{(0,0),(1,0),(1,1)}` per triangle; UV-less mesh with non-textured material → no `primvars:st`
- [x] 2.2 Failing test: `sphere` shape → vertex-interp `primvars:st` with `u = j/segments`, `v = 1 − i/rings`
- [x] 2.3 `emit.add_mesh`: add a UV-interpolation mode so `st` can be authored as `faceVarying` (3·M values) as well as `vertex`
- [x] 2.4 `emit.tessellate_sphere`: return parametric UVs; `api._shape_geometry` sphere branch threads them through
- [x] 2.5 Material→texture lookup helper (does the shape's bound material reference an imagemap?) + `api._emit_shape` gating: synthesize per-triangle faceVarying default UVs for textured UV-less `trianglemesh`/`plymesh`; report fallback when material unresolvable
- [x] 2.6 Unit tests green; `ruff check` clean (worktree: `--no-respect-gitignore`)

## 3. Textured parity scene + GPU gate (TDD)

- [x] 3.1 Author a textured corpus scene under `tests/pbrt/corpus/` — `trianglemesh` quad(s) with explicit `uv` + imagemap diffuse over a light; commit a tiny (≤64²) checker/gradient PNG asset (`git add -f`)
- [x] 3.2 Render the pbrt-v4 reference EXR with the pinned build (`~/projects/pbrt-v4`), commit it (`git add -f`), and record per-scene tolerance + content hash in the corpus manifest
- [x] 3.3 Add the textured scene to `tests/pbrt/test_parity.py`: import → offscreen GPU render → read linear-HDR accumulation → compare to reference within tolerance; fails naming scene+metric on regression; skips cleanly without GPU/refs
- [x] 3.4 Render labelled side-by-side (pbrt reference · skinny output) and surface it with `SendUserFile` (keep relMSE/FLIP numbers alongside)

## 4. Documentation and close-out

- [x] 4.1 `docs/PbrtImport.md`: correct the stale `imagemap` parity-matrix row; document explicit-UV passthrough + default-UV synthesis (sphere parametric, triangle faceVarying default) with the measured textured-scene error
- [x] 4.2 `CHANGELOG.md`: add an entry for imagemap-texture UV support + textured parity scene
- [x] 4.3 Final gate: `ruff` clean, full `tests/pbrt/` unit suite green, GPU parity gate green (incl. new textured scene), `openspec validate pbrt-imagemap-uv-parity --strict`, no doc drift
