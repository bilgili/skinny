# Delta: usd-scene-editing

## ADDED Requirements

### Requirement: Add an analytic primitive with a bound preview material

The renderer SHALL provide `add_primitive(prim_type, parent_prim_path, name,
transform, color, roughness, metallic)` that defines an analytic gprim of a
type the loader tessellates (Sphere, Cube, Cylinder, Cone, Capsule, Plane) in
the edit layer, authors a sibling `UsdShade` material with a
`UsdPreviewSurface` shader carrying the given (or default) color, roughness,
and metallic, binds it to the gprim, and re-syncs geometry from the stage. The
prim path SHALL be uniquified against existing prims and returned.

The primitive SHALL NOT be authored without a bound material: an unbound prim
resolves to the fallback material slot, which is protected from parameter
overrides, so its appearance could never be edited afterwards.

On failure after partial authoring, the edit SHALL be rolled back the same way
`add_light` rolls back — the authored prims (and any intermediate parents
created) removed from the edit layer.

#### Scenario: Primitive renders and re-syncs

- **WHEN** `add_primitive("Sphere", …)` is called on a stage with an active
  edit layer
- **THEN** the stage gains a sphere gprim with a bound preview-surface
  material, geometry re-syncs, and the returned prim path resolves on the stage

#### Scenario: Primitive material is override-editable

- **WHEN** a primitive is added and a parameter override is applied to its
  authored material
- **THEN** the override applies, because the material is a dedicated
  `scene.materials` entry rather than the protected fallback slot

#### Scenario: Rollback on failure

- **WHEN** authoring fails after the gprim or material was partially defined
- **THEN** the partially authored prims are removed from the edit layer and the
  stage composes as before the call

### Requirement: `add_model` rollback removes auto-created parents

`add_model`'s failure rollback SHALL remove parent prims it auto-created for the
requested parent path, not only the referenced prim itself, so a failed or
vetoed add whose parent chain was synthesized leaves the edit layer as it was
before the call. This SHALL match the parent-tracking rollback `add_light`
already performs.

#### Scenario: Vetoed add with a new parent leaves the edit layer clean

- **WHEN** an `add_model` under a not-yet-existing parent path is rolled back
  (by a validator veto or an authoring failure)
- **THEN** both the referenced prim and every parent Xform the call created are
  removed, and the edit layer composes exactly as before the call

### Requirement: `add_model` accepts a validator seam

`add_model` SHALL accept an optional validator callable, invoked as
`validate(stage, added_prim)` after the reference has recomposed but before
geometry re-sync, where `added_prim` is the prim `add_model` authored (its path
is chosen internally, so the caller cannot precompute it). A validator that
raises SHALL trigger the same rollback as an authoring failure — including
auto-created parents — and the exception SHALL propagate to the caller. When no
validator is supplied, behavior SHALL be unchanged.

The seam is on `add_model` alone: it is the only add that pulls external layers
and assets. `add_light` and `add_primitive` author no external files, so they
need no validator.

This exists so a policy layer (the MCP server's path allowlist) can inspect the
composed result — newly introduced layers and asset-valued attributes — and
veto the edit without a parallel authoring path and without paying a full
re-sync before rejecting.

#### Scenario: Validator veto rolls back

- **WHEN** `add_model` is called with a validator that raises on the composed
  result
- **THEN** the referenced prim and any auto-created parents are removed from the
  edit layer, no re-sync occurs, and the caller receives the validator's
  exception

#### Scenario: Validator receives the added prim

- **WHEN** `add_model` invokes the validator
- **THEN** it is called with the recomposed stage and the prim `add_model`
  authored, so the validator can walk that subtree

#### Scenario: No validator, no change

- **WHEN** `add_model` is called without a validator
- **THEN** authoring, re-sync, and rollback behavior are identical to before
  this capability existed

### Requirement: `add_light` accepts intensity and color at creation

`add_light` SHALL accept optional intensity and color parameters and, when
given, author them onto the light at define time rather than only its fixed
defaults. This is required so a client can create a light with those values in
one call — a post-creation property edit would not persist to a saved edit layer.

#### Scenario: Light authored with intensity and color

- **WHEN** `add_light` is called with intensity and color
- **THEN** the authored light's intensity and color attributes carry those
  values, readable from the stage and captured by a subsequent save

#### Scenario: Defaults unchanged when omitted

- **WHEN** `add_light` is called without intensity or color
- **THEN** it authors the same editor-friendly defaults as before this change
