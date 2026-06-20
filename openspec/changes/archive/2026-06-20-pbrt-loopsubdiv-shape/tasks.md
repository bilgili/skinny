## 1. Subdivision kernel

- [x] 1.1 Add `src/skinny/pbrt/loopsubdiv.py` with topology build from `(P, indices)`:
      per-edge incident-face lists, boundary-edge detection (single incidence),
      per-vertex valence, boundary flag, and ordered one-ring / boundary neighbors.
- [x] 1.2 Implement one refinement level: interior even rule (pbrt `β(n)`), boundary
      even rule (`3/4`,`1/8`,`1/8`), interior odd edge rule (`3/8`/`3/8`/`1/8`/`1/8`),
      boundary odd midpoint (`1/2`/`1/2`), and 1→4 triangle split. Vectorize neighbor
      sums with NumPy.
- [x] 1.3 Implement the limit pass: limit positions (interior one-ring `gamma` weights;
      boundary `3/5`,`1/5`,`1/5`) and limit normals from the Loop tangent masks
      (interior `cos`/`sin` one-ring masks; pbrt boundary tangent masks).
- [x] 1.4 Expose `subdivide(points, indices, levels) -> (points, indices, normals)`;
      handle `levels = 0` (limit push only) and raise/validate on malformed topology.

## 2. Importer wiring

- [x] 2.1 Add a `t == "loopsubdiv"` branch to `_shape_geometry` in
      `src/skinny/pbrt/api.py`: read `P`, `indices` (reshaped `(-1,3)`), `levels`
      (pbrt default), call `loopsubdiv.subdivide`, return
      `(points_local, indices, normals, None)` — same tuple shape as `trianglemesh`.
- [x] 2.2 Guard degenerate input (missing/empty `P`/`indices`, `indices.size % 3 != 0`):
      `report.skipped("shape:loopsubdiv", "<reason>")` and return `None`.
- [x] 2.3 Confirm the existing `_emit_shape` path (CTM bake via `bake_world_mesh` with
      `reverse=shp.reverse_orientation`, `add_mesh`, default-UV synthesis for textured
      shapes, area-light attachment) is reused unchanged — no edits needed there.

## 3. Tests

- [x] 3.1 Unit: triangle count is `4^levels · F` for `levels` 0/1/2 on a small cage.
- [x] 3.2 Unit: regular interior (valence 6) planar patch — limit point lies on the
      plane, limit normal equals the face normal (within tolerance).
- [x] 3.3 Unit: open boundary patch (disc) — boundary vertices depend only on boundary
      neighbors; boundary curve unchanged by interior vertices.
- [x] 3.4 Unit: malformed `indices` (not a multiple of 3 / empty) → skip entry, no raise.
- [x] 3.5 Importer: `killeroo-simple.pbrt` import reports the `loopsubdiv` bodies as
      exact (not skipped) and the body meshes exist in the output stage.

## 4. Parity & docs

- [ ] 4.1 (Optional) Add a small `loopsubdiv` parity scene to the pbrt corpus and verify
      it clears the importer parity gate (FLIP/relMSE within the existing threshold).
- [x] 4.2 Update the pbrt importer shape-support list (README / importer docs) to
      include `loopsubdiv`; note "tessellated to limit surface at import time".
- [x] 4.3 Run `.venv/bin/ruff check src/` and `.venv/bin/pytest` for the new tests;
      `openspec validate pbrt-loopsubdiv-shape --strict`.
