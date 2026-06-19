## ADDED Requirements

### Requirement: Explicit mesh UVs are emitted as `primvars:st`

The importer SHALL emit per-vertex texture coordinates as a `primvars:st`
`TexCoord2fArray` (vertex interpolation) whenever a shape carries explicit UV
data, so that imagemap textures bound to the shape sample correctly.

#### Scenario: trianglemesh `uv`/`st` parameter

- **WHEN** a `Shape "trianglemesh"` declares a `point2 uv` (or `st`) parameter
- **THEN** the emitted USD mesh has `primvars:st` (vertex interpolation) with one
  UV per vertex, in vertex order, matching the parameter values

#### Scenario: PLY vertex UVs (ascii and binary)

- **WHEN** a `Shape "plymesh"` references a PLY whose vertex element carries
  `u`/`v`, `s`/`t`, or `texture_u`/`texture_v` properties (ascii, binary
  little-endian, or binary big-endian)
- **THEN** `read_ply` returns those UVs and the emitted USD mesh has `primvars:st`
  (vertex interpolation) with the matching per-vertex values

#### Scenario: UVs survive winding flip

- **WHEN** a textured shape's bake matrix is orientation-reversing (so face
  winding is flipped)
- **THEN** the per-vertex `primvars:st` values remain aligned to their vertices
  (st is indexed via `faceVertexIndices`, unchanged by the index-order reversal)

### Requirement: Default UVs for UV-less textured triangle meshes

The importer SHALL synthesize pbrt-faithful default UVs for a `trianglemesh` or
`plymesh` that carries no source UV but is bound to a material referencing a
texture, so the texture samples like pbrt rather than at a single constant point.
Untextured UV-less meshes SHALL remain UV-free.

#### Scenario: textured mesh without source UV

- **WHEN** a UV-less `trianglemesh`/`plymesh` is bound to a material that
  references an imagemap texture
- **THEN** the emitted USD mesh has `primvars:st` with `faceVarying` interpolation,
  assigning `{(0,0),(1,0),(1,1)}` to the three vertices of each triangle
  (matching pbrt `Triangle` default UVs)

#### Scenario: untextured mesh without source UV

- **WHEN** a UV-less `trianglemesh`/`plymesh` is bound to a material with no
  texture
- **THEN** the emitted USD mesh has no `primvars:st` (status quo; no per-mesh UV
  bloat)

### Requirement: Parametric default UVs for spheres

The importer SHALL author parametric per-vertex UVs on tessellated `sphere`
shapes, matching pbrt's sphere parametrization, so textured spheres sample
consistently.

#### Scenario: sphere UV parametrization

- **WHEN** a `Shape "sphere"` is tessellated to a UV-sphere of `rings × segments`
- **THEN** vertex `(i, j)` receives `u = j / segments` and `v = 1 − i / rings`
  (equivalent to pbrt `u = φ/φmax`, `v = (θ−θmin)/(θmax−θmin)`), authored as a
  vertex-interpolation `primvars:st`

### Requirement: Textured scene in the parity gate

The pbrt-v4 parity gate SHALL include a textured scene that exercises an imagemap
diffuse over UV'd geometry, with a committed pbrt-v4 reference image, so that
end-to-end GPU texture sampling is validated against pbrt within tolerance.

#### Scenario: textured parity scene passes

- **WHEN** the textured corpus scene is imported, rendered offscreen on the GPU to
  a linear-HDR image, and compared to its pbrt-v4 reference EXR
- **THEN** the per-scene relMSE/FLIP is within the tolerance recorded in the corpus
  manifest, and the parity test fails (naming the scene + metric) when it is not

#### Scenario: gate skips cleanly without GPU or references

- **WHEN** the textured parity test runs with no GPU backend or no reference EXR
  present
- **THEN** the test is skipped (not failed), consistent with the existing parity
  gate behavior
