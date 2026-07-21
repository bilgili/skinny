# mtlx-material-synthesis Specification (delta)

## ADDED Requirements

### Requirement: Material spec accepts exactly one of four forms

The synthesis layer SHALL accept a JSON material spec in exactly one of four
forms — `{preset}`, `{template, params}`, `{model: "preview", params}`, or
`{model: "standard_surface", params, graph?}` — and SHALL reject a spec that
mixes forms, names an unknown form, or supplies `graph` with any model other
than `standard_surface`. Rejection SHALL happen before any USD stage or
filesystem mutation.

#### Scenario: Preset form

- **WHEN** a spec `{preset: "marble_solid"}` is validated
- **THEN** it resolves to the curated `.mtlx` file for that preset and no
  document synthesis occurs

#### Scenario: Mixed forms rejected

- **WHEN** a spec supplies both `preset` and `model`
- **THEN** validation fails with an error naming the conflicting keys and
  nothing is authored

#### Scenario: Graph on wrong model rejected

- **WHEN** a spec supplies `model: "preview"` together with `graph`
- **THEN** validation fails stating that nodegraphs require
  `model: "standard_surface"`

### Requirement: Preset catalog is discovered server-side from the curated corpus

The preset catalog SHALL be derived at call time from the `.mtlx` files in
the curated materials directory (`assets/Usd-Mtlx-Example/materials/`), with
preset names formed by stripping the `standard_surface_` filename prefix.
Preset resolution SHALL be a dictionary lookup into the enumerated catalog —
the client-supplied name SHALL never be joined onto a filesystem path — and
the allowed-roots check is not applied to catalog resolution.

#### Scenario: Catalog lists curated materials

- **WHEN** the catalog is enumerated
- **THEN** it contains one entry per curated `.mtlx` file (including
  `marble_solid`, `wood_tiled`, `brass_tiled`, `chrome`, `glass`, `jade`)
  with a name and the file it resolves to

#### Scenario: Unknown preset rejected

- **WHEN** a spec names a preset absent from the catalog
- **THEN** validation fails with an error listing the available preset names

#### Scenario: Path-shaped preset name is not resolved as a path

- **WHEN** a spec supplies `preset: "../../../etc/foo"`
- **THEN** the lookup fails as an unknown preset; no filesystem path is
  constructed from the client string

### Requirement: Standard_surface synthesis builds a document gated by a generator dry-run

For `model: "standard_surface"` specs the synthesis layer SHALL build a
MaterialX document via the MaterialX Python API containing a
standard_surface shader node, a surfacematerial node, and — when `graph` is
given — a nodegraph wired per the spec's `nodes` and `connections`. Before
any file is written or prim authored, the document SHALL pass a GPU-free
Slang-generator dry-run (library generate + compute-fragment extraction);
a document the generator cannot process SHALL be rejected.

#### Scenario: Flat parameters authored

- **WHEN** a spec gives `params: {base_color: [0.7,0.2,0.1], specular_roughness: 0.3}`
- **THEN** the produced document's standard_surface node carries those input
  values

#### Scenario: Perlin-noise nodegraph authored

- **WHEN** a spec's graph contains a `fractal3d` node blended by a `mix`
  node into `base_color`
- **THEN** the produced document contains a nodegraph with those nodes and
  connections driving the shader's `base_color` input, and the dry-run
  extracts a compute fragment for it

#### Scenario: Invalid connection rejected

- **WHEN** a connection references a node or output that does not exist in
  the spec
- **THEN** synthesis fails with an error naming the dangling reference and
  no document is written

#### Scenario: Generator bailout rejected before authoring

- **WHEN** a structurally valid document trips a generator bailout in the
  dry-run
- **THEN** the spec is rejected as a tool error and no file or prim exists

### Requirement: Nodegraph node types are whitelisted and each is gen-proven by test

The synthesis layer SHALL maintain an explicit whitelist of MaterialX node
types as a data tuple: `fractal3d`, `noise2d`, `noise3d`, `position`,
`texcoord`, `mix`, `multiply`, `add`, `subtract`, `sin`, `power`,
`dotproduct`, `ramplr`, `ramptb`. (`checker` was evaluated and dropped by
the gen gate: this MaterialX build ships the node as `checkerboard`, so the
literal `checker` node has no nodedef.) A graph spec using any node
type outside the whitelist SHALL be rejected with an error listing the
supported set. Image/texture nodes (`image`, `tiledimage`) SHALL NOT be in
the v1 whitelist. Because the curated corpus proves only a subset of these
nodes, the hostless test suite SHALL run a per-node generator dry-run over
every whitelisted type; a node type failing that test SHALL be removed from
the tuple before merge, and the same test gates any future extension.

#### Scenario: Whitelisted graph accepted

- **WHEN** a graph uses only `position`, `fractal3d`, and `mix`
- **THEN** validation passes

#### Scenario: Unsupported node rejected with guidance

- **WHEN** a graph uses node type `voronoi` (not whitelisted)
- **THEN** validation fails and the error message contains the full list of
  supported node types

#### Scenario: Every whitelisted node compiles through the generator

- **WHEN** the hostless whitelist test runs
- **THEN** each whitelisted node type, synthesized into a minimal document,
  generates and extracts a compute fragment

### Requirement: Templates expand to standard_surface graph specs server-side

The synthesis layer SHALL provide named procedural templates — `noise` and
`marble_veins` (a `checker` template was dropped with its node per the gen
gate below) — each with a declared parameter
schema (colors, scale, octaves, …) expressible entirely in whitelisted node
types. A `{template, params}` spec SHALL expand server-side into a
`model: "standard_surface"` spec with a graph, then flow through the same
validation, dry-run, and synthesis path; templates SHALL introduce no
authoring mechanism of their own. A template whose required node type
fails the whitelist gen test SHALL be dropped rather than approximated.

#### Scenario: Noise template expansion

- **WHEN** `{template: "noise", params: {colorA: [0.9,0.9,0.85], colorB: [0.2,0.2,0.25], octaves: 4}}`
  is processed
- **THEN** the expanded spec contains a whitelisted noise nodegraph blending
  the two colors and synthesis proceeds as for a raw graph spec

#### Scenario: Template parameter bounds enforced

- **WHEN** a template parameter is outside its declared bounds (e.g.
  negative octaves)
- **THEN** validation fails before expansion produces a document

### Requirement: Element names are salted with the material prim name

Synthesis SHALL salt every generated document's surfacematerial, shader,
and nodegraph element names with the unique material prim name, and the naming
contract SHALL hold: holder prim name equals surfacematerial element name,
and the binding target is the holder prim path. Two materials synthesized
from identical specs SHALL therefore never share element names.

#### Scenario: Two same-template materials do not alias

- **WHEN** two materials are created from the same `noise` template spec
- **THEN** their documents carry distinct salted element names and each
  binds and renders independently

#### Scenario: Naming contract holds

- **WHEN** a material prim `/Materials/Speckle` is created from a synthesized
  document
- **THEN** the document's surfacematerial element is named `Speckle`

### Requirement: Promoted inputs carry a gen-uniform name mapping

Synthesis SHALL promote tunable values to nodegraph interface inputs — all
declared template parameters, and raw-graph node parameters explicitly
marked `expose: true` — and SHALL derive, from the dry-run's reflection,
a mapping from each promoted logical input name to the set of generated
uniform field names it controls (the generator names uniforms after
interior node inputs and may shatter one interface input into several
uniforms). The mapping is the material's editability contract: it SHALL be
returned with the synthesis result and persisted with the material, and
values not promoted are compile-time constants absent from the mapping.

#### Scenario: Template parameters mapped

- **WHEN** a `noise` template material is synthesized
- **THEN** the result maps `colorA`, `colorB`, and each declared noise
  control to one or more generated uniform names

#### Scenario: Shattered interface input maps to all its uniforms

- **WHEN** a promoted input feeds more than one node input in the graph
- **THEN** its mapping entry contains every generated uniform name derived
  from those node inputs

#### Scenario: Unexposed graph constant absent from the mapping

- **WHEN** a raw graph spec sets a node parameter without `expose: true`
- **THEN** the synthesis result's mapping has no entry for it

### Requirement: Synthesized documents are files in a server-owned session directory

Each synthesized document SHALL be written as a `.mtlx` file in a
server-owned session directory (tempdir-based), one file per created
material, named after the material prim, and flushed to disk before any
resync that re-reads material documents. The session directory is server
configuration in the same trust domain as the preset catalog and SHALL NOT
be required to fall inside the configured allowed roots; clients never
address it. When material creation fails after the file is written, the
file SHALL be deleted along with the rolled-back prims.

#### Scenario: One file per material

- **WHEN** two materials are synthesized in one session
- **THEN** two distinct `.mtlx` files exist in the session directory, each
  named after its material prim

#### Scenario: Rollback removes the session file

- **WHEN** stage-side authoring fails after the document was written
- **THEN** the session `.mtlx` file for that material no longer exists
