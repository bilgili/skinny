# usd-scene-editing Specification (delta)

## ADDED Requirements

### Requirement: Add a material as an edit-layer structural edit

The renderer SHALL provide an `add_material` operation that authors a typed
`UsdShade.Material` holder prim under the `/Materials` scope in the session
edit layer — carrying either a reference to a `.mtlx` file (curated preset
or session-synthesized document, authored with an absolute asset path,
since the anonymous session layer offers no anchor for relative paths) or
an inline `UsdPreviewSurface` shader — and resyncs derived state. On any
failure the operation SHALL remove every prim it created (including an
auto-created `/Materials` scope) and any session `.mtlx` file it wrote,
then re-raise.

#### Scenario: Typed holder classifies as a material

- **WHEN** `add_material` authors a preset or synthesized material
- **THEN** the holder prim is `Material`-typed and the scene graph
  classifies it as a material node

#### Scenario: Failure rolls back prims and files

- **WHEN** authoring fails after `/Materials` was auto-created and a
  session document was written
- **THEN** neither the material prim, the auto-created scope, nor the
  session file remains

### Requirement: Session-layer `.mtlx` references are discovered by material intake

The `.mtlx` material intake (used on the default no-plugin path) SHALL
discover references authored in the stage's session layer in addition to
the root layer: asset-path collection and per-prim reference detection
SHALL scan the session layer's prim specs, resolving the absolute paths
authored there. Root-layer discovery behavior SHALL be unchanged.

#### Scenario: Session-authored reference is ingested

- **WHEN** a `/Materials` holder referencing a `.mtlx` file exists only in
  the session layer and a geometry prim binds it
- **THEN** after resync the material is loaded from that document exactly
  as a root-layer-referenced material would be

#### Scenario: Root-layer scenes unaffected

- **WHEN** a scene authored entirely in files (e.g. the three-materials
  demo) loads
- **THEN** intake results are identical to the pre-change behavior

### Requirement: Material participation is binding-driven

Material loading SHALL be binding-driven — loading, MaterialX graph
generation, and material-table entry occur through binding resolution
during geometry traversal: an
unbound `/Materials` holder SHALL compose on the stage but SHALL NOT be
loaded, generated, or listed in the material table until a geometry prim
binds it. `add_material` therefore reports created-but-not-live, and the
first bind is the moment the material participates in rendering.

#### Scenario: Unbound material does not render or generate

- **WHEN** `add_material` creates a holder and no prim binds it
- **THEN** after resync the material table and generated graph set are
  unchanged

#### Scenario: First bind makes it live

- **WHEN** a geometry prim binds the holder
- **THEN** after resync the material is loaded, its graph (if any) is
  generated, and its editable properties surface

### Requirement: Bind a material as an edit-layer structural edit

The renderer SHALL provide a `bind_material` operation that authors the
binding relationship on the target geometry prim in the session edit layer
with explicit targets (set, not prepended), replacing any existing binding,
then resyncs. Explicit targets are what make the session binding override a
file-authored binding rather than merge with it.

#### Scenario: Session binding overrides a file-authored binding

- **WHEN** `bind_material` targets a prim whose binding was authored in the
  loaded scene file
- **THEN** the session-layer binding wins and the prim renders with the
  newly bound material

#### Scenario: Rebinding resets accumulation

- **WHEN** a binding changes on a visible prim
- **THEN** derived state resyncs and progressive accumulation restarts

### Requirement: Live graph materials surface editable properties on their scene-graph nodes

The scene-graph build SHALL, for each live material carrying a persisted
logical-input mapping (from synthesis) or parameter-override keys
(constant-shader `.mtlx` materials), inject those inputs as editable properties on
the material's node, with values reflecting current override state, so the
path-addressed property tools can read and write them. Property writes
SHALL fan out through the mapping to every generated uniform the logical
input controls via the existing material-override path.

#### Scenario: Bound template material exposes its parameters

- **WHEN** a noise-template material is bound and the scene graph rebuilds
- **THEN** its node lists `colorA`, `colorB`, and the template's noise
  controls as editable properties

#### Scenario: Fan-out write reaches every mapped uniform

- **WHEN** a property whose mapping contains several generated uniforms is
  written
- **THEN** all of them receive the override in the same edit and the
  material version bumps once

### Requirement: Material edits persist through save per stage-origin branch

`save_edits` SHALL capture materials and bindings authored by
`add_material`/`bind_material`, handling each save branch:

- Anonymous-root stages (created scenes) export flattened, so the save
  SHALL post-process the exported layer: re-author each `/Materials`
  holder's `.mtlx` reference relative to the saved file, remove flatten
  residue under the holder, and copy session-synthesized documents into a
  `materials/` directory beside the stage — yielding a self-contained
  bundle.
- File-backed stages export the edit layer as an overlay; references SHALL
  be re-anchored relative to the export target and session documents copied
  the same way. Reload for this branch means re-attaching the exported
  overlay to the original scene.
- Curated presets whose documents reference texture files SHALL keep
  absolute references into the renderer's assets directory rather than
  being copied without their textures.

#### Scenario: Created-scene save is self-contained

- **WHEN** an anonymous-root scene with a bound synthesized material is
  saved to an allowed root and reloaded standalone from there
- **THEN** the material composes from the copied `.mtlx` and the render
  matches the pre-save scene

#### Scenario: Texture-bearing preset survives save

- **WHEN** a scene bound to `wood_tiled` is saved and reloaded
- **THEN** the wood material still resolves its textures (via the absolute
  assets reference), not a fallback
