# pbrt-loopsubdiv-shape Specification

## Purpose
TBD - created by archiving change pbrt-loopsubdiv-shape. Update Purpose after archive.
## Requirements
### Requirement: Loop subdivision shape translation

The importer SHALL translate pbrt `loopsubdiv` shapes into triangle-mesh UsdGeom
prims by tessellating the control cage at import time. It SHALL read the `point3 P`
control vertices, the `integer indices` triangle list, and the `integer levels`
refinement count, and emit the tessellated positions, triangle indices, and
per-vertex normals through the same emit path as `trianglemesh` (CTM bake,
`reverse_orientation` winding, default-UV synthesis for textured shapes, and
area-light attachment all applying unchanged). A successfully translated
`loopsubdiv` SHALL be reported as exact, not skipped.

#### Scenario: loopsubdiv control cage is tessellated, not skipped
- **WHEN** a `Shape "loopsubdiv"` with `P`, `indices`, and `levels` is translated
- **THEN** a triangle-mesh prim with the subdivided geometry is authored under the
  accumulated CTM
- **AND** the translation report records the shape as exact (`ok`), with no
  "unsupported shape type" skip entry for it

#### Scenario: killeroo body imports its geometry
- **WHEN** `killeroos/killeroo-simple.pbrt` (whose bodies are `loopsubdiv`) is imported
- **THEN** the killeroo body meshes are present in the output stage
- **AND** no `loopsubdiv` shape is reported as skipped

### Requirement: pbrt-exact Loop tessellation

The tessellation SHALL match pbrt v4's `loopsubdiv` conversion: apply the Loop
subdivision rules `levels` times, then evaluate the limit surface. Interior even
vertices SHALL use the valence-weighted one-ring rule with pbrt's `β`; boundary even
vertices SHALL use the `3/4`, `1/8`, `1/8` boundary rule; interior odd (new edge)
vertices SHALL use the `3/8`/`3/8`/`1/8`/`1/8` mask and boundary odd vertices the
`1/2`/`1/2` midpoint; and each input triangle SHALL be split into four. After the
final refinement, vertices SHALL be pushed to their Loop limit positions and
per-vertex normals SHALL be computed from the Loop limit-surface tangent masks.

#### Scenario: triangle count quadruples per level
- **WHEN** a cage of `F` triangles is subdivided with `levels = L`
- **THEN** the emitted mesh has `4^L · F` triangles

#### Scenario: regular interior limit position and normal match Loop rules
- **WHEN** an interior vertex of valence 6 on a planar regular cage is evaluated
- **THEN** its limit position and limit normal match the analytic Loop limit values
  (planar patch → limit point on the plane, normal equal to the face normal) within
  numerical tolerance

#### Scenario: boundary vertices stay on the boundary curve
- **WHEN** a cage with an open boundary (e.g. a disc/patch) is subdivided
- **THEN** boundary vertices are positioned only from their boundary neighbors
  (boundary even/odd/limit rules), keeping the boundary curve independent of interior
  vertices

### Requirement: Degenerate loopsubdiv input is skipped cleanly

The importer SHALL skip a `loopsubdiv` shape that lacks usable geometry (missing or
empty `P` or `indices`, or an `indices` length not divisible by three), recording a
descriptive report entry and authoring no partial mesh. It MUST NOT raise on such
input; import of the rest of the scene SHALL continue.

#### Scenario: missing control data is skipped, not fatal
- **WHEN** a `loopsubdiv` shape has no `indices` (or an index count not a multiple of 3)
- **THEN** the importer records a skip entry describing the reason
- **AND** import of the rest of the scene continues without error

