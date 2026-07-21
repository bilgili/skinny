# mcp-scene-control Specification (delta)

## ADDED Requirements

### Requirement: Material discovery tool

The server SHALL advertise a `material_list` tool returning, in one call:
the curated preset catalog (names plus per-preset editable inputs), the
parametric model parameter schemas for `preview` and `standard_surface`,
the whitelisted nodegraph node types, and the procedural template schemas.
The result SHALL be derived from server-side sources at call time (preset
directory listing, synthesis whitelist, template registry). Per-preset
editable inputs SHALL come from generator reflection over the preset
document (cacheable per file modification time), never from parsing `.mtlx`
input declarations directly — reflection names are the actual writable
keys; parsed names are not.

#### Scenario: One call arms the client

- **WHEN** a client calls `material_list`
- **THEN** the result contains presets, both model schemas, the supported
  graph node types, and template schemas, sufficient to construct any valid
  `scene_add_material` spec without further discovery

#### Scenario: Catalog reflects the directory

- **WHEN** a curated `.mtlx` file is present in the preset directory
- **THEN** `material_list` includes it without any code change

#### Scenario: Editable inputs are writable keys

- **WHEN** `material_list` reports an editable input for a graph preset
- **THEN** a `scene_set` using that name (once the material is bound and
  live) applies successfully

### Requirement: Material creation tool

The server SHALL advertise a `scene_add_material(spec, name?)` tool that
validates the spec through the synthesis layer (including the generator
dry-run for synthesized documents), authors a typed `UsdShade.Material`
holder prim under the `/Materials` scope in the session edit layer
(creating the scope if absent), and returns the new material prim path plus
version counters. Because material participation is binding-driven, the
result SHALL indicate that an unbound material is not yet live
(`live: false`) — it renders, is loaded into the material table, and
exposes editable properties only once bound. Validation failures SHALL be
reported as tool errors before any stage or filesystem mutation;
stage-side failures SHALL roll back all prims and session files the call
created. Preset specs SHALL be resolved server-side by name; the tool SHALL
NOT accept client filesystem paths for presets. Adding a preset that
already has a `/Materials` holder SHALL return the existing holder's path
rather than creating a duplicate (curated element names cannot coexist
twice); synthesized and template materials are never deduplicated.

#### Scenario: Preset material created

- **WHEN** a client calls `scene_add_material` with `{preset: "marble_solid"}`
- **THEN** a typed Material holder under `/Materials` referencing the
  curated marble `.mtlx` exists, the tool returns its path with
  `live: false`, and the structural version increments

#### Scenario: Preset dedup returns the existing holder

- **WHEN** a client adds `{preset: "marble_solid"}` twice
- **THEN** both calls return the same `/Materials` path and only one holder
  prim exists

#### Scenario: Procedural graph material created

- **WHEN** a client sends a `standard_surface` spec whose graph blends two
  colors by `fractal3d` noise
- **THEN** the synthesized `.mtlx` is referenced from a typed holder under
  `/Materials` and the material renders with the procedural pattern once
  bound

#### Scenario: Invalid spec leaves the stage untouched

- **WHEN** a spec fails validation (unknown preset, non-whitelisted node,
  dangling connection, out-of-bounds parameter, generator bailout)
- **THEN** the tool returns an explicit error, no prim or file is created,
  and the structural version does not change

### Requirement: Material binding tool

The server SHALL advertise a `scene_bind_material(prim_path, material_path)`
tool that authors the binding in the session edit layer with explicit
binding-relationship targets (set, not prepended — so the session binding
replaces rather than merges with a file-authored one under LIVRPS).
Binding an already-bound prim SHALL replace the binding (last-write-wins).
The tool SHALL error explicitly when either path does not exist, when the
target of `material_path` is neither a `Material`-typed prim nor a prim
carrying a `.mtlx` reference, or when `prim_path` is not a bindable
geometry prim.

#### Scenario: Bind material to primitive

- **WHEN** a client binds `/Materials/Marble` to `/World/Sphere`
- **THEN** the sphere renders with the marble material after resync, the
  material becomes live, and the structural version increments

#### Scenario: Session binding replaces a file-authored binding

- **WHEN** a client binds a material to a prim whose binding was authored
  in the loaded scene file
- **THEN** the session binding wins and the prim renders with the newly
  bound material

#### Scenario: Rebind replaces

- **WHEN** a client binds a second material to a prim already bound to a
  first
- **THEN** the second binding wins and the first material prim remains on
  the stage unbound

#### Scenario: Nonexistent material is an explicit error

- **WHEN** a client binds a `material_path` that has no prim
- **THEN** the tool returns an error naming the missing path and nothing
  changes

### Requirement: Primitive adds accept a material reference

`scene_add_primitive` SHALL accept an optional `material` argument carrying
either a preset/template name or an existing `/Materials/...` prim path.
When given a name, the material SHALL be created as if by
`scene_add_material` (including preset dedup) and bound to the new
primitive; when given a path, the existing material SHALL be bound and the
call SHALL error if it is absent. Supplying `material` together with any of
the inline `color`, `roughness`, or `metallic` arguments SHALL be rejected
as ambiguous. When `material` is omitted, behavior is unchanged (inline
preview material authored and bound).

#### Scenario: One call makes a marble sphere

- **WHEN** a client calls `scene_add_primitive(type: "Sphere", material: "marble_solid")`
- **THEN** the sphere prim, a marble material under `/Materials`, and the
  binding all exist after the single call and the sphere renders marble

#### Scenario: Existing material reused

- **WHEN** a client passes `material: "/Materials/Marble"` for a second
  primitive
- **THEN** the second primitive binds the same material prim and no new
  material is created

#### Scenario: Ambiguous composition rejected

- **WHEN** a client passes both `material` and `color`
- **THEN** the call fails with an error naming the conflicting arguments
  and nothing is authored

### Requirement: Live materials are editable through the property tools by logical input name

A created material SHALL, once live (bound and loaded), expose its promoted
logical inputs as editable properties on its scene-graph node, and
`scene_set` on such a property SHALL fan out through the persisted
gen-uniform mapping — applying the material override to every generated
uniform the logical input controls — and bump the material version.
Constant-shader `.mtlx` materials (no nodegraph) SHALL expose their
parameter-override keys the same way. A name absent from the promoted set
is absent from the node's properties, so `scene_set` on it SHALL fail with
the existing no-such-property error.

#### Scenario: Edit a synthesized material's color

- **WHEN** a client calls `scene_set` on the promoted `colorA` input of a
  bound noise-template material
- **THEN** every generated uniform mapped to `colorA` receives the
  override, the material version increments, and subsequent frames render
  with the new color

#### Scenario: Non-promoted constant yields no-such-property

- **WHEN** a client calls `scene_set` on a graph constant that was not
  promoted
- **THEN** the tool returns the existing error naming the property as
  absent from the node

### Requirement: Graph-material structural calls degrade to jobs

First binds of graph materials (including add-plus-bind composition via
`scene_add_primitive(material=…)`) SHALL be documented and treated as calls
that degrade to pollable jobs (`scene_job_status`): participation is
binding-driven, so an unbound add compiles nothing, and it is the first
bind that changes the scene's graph-set signature and triggers a full
pipeline rebuild on the render thread, exceeding the inline grace period. This is the existing structural-job mechanism, not a new one.

#### Scenario: First bind returns a job

- **WHEN** a client binds a procedural graph material (or adds-and-binds it
  in one `scene_add_primitive` call) on a loaded scene
- **THEN** the call may return a job id, and polling `scene_job_status`
  eventually reports the completed result
