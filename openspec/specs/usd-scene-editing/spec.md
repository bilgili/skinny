# usd-scene-editing Specification

## Purpose
TBD - created by archiving change runtime-scene-graph-editing. Update Purpose after archive.
## Requirements
### Requirement: Stage is the authoritative model with a non-destructive edit layer

On loading a USD scene the renderer SHALL retain the live `Usd.Stage` as the
authoritative scene model and author non-destructive edits into the stage's
**session layer** (`Usd.Stage.GetSessionLayer()`), set as the stage edit target.
The session layer is stronger than the entire root layer stack, so an edit
overrides any opinion authored in the root/file layer. The original root layer
SHALL NOT be written to disk by any editing operation. The flat scene and GPU
buffers SHALL be a derived cache that the renderer refreshes from the stage on
each edit.

A transform edit (`set_transform` / the shared local-transform author) SHALL be
able to override a prim's `xformOp:transform` even when that op is authored in
the loaded file, without raising a duplicate-op error. The author path SHALL
reuse an existing single `xformOp:transform` op by setting its value in the edit
target, rather than clearing and re-adding the op.

#### Scenario: Edit target is the session layer

- **WHEN** a USD scene is loaded for editing
- **THEN** the stage edit target is the session layer, and no editing sublayer is
  inserted into the root layer's sublayer stack

#### Scenario: Original file untouched by edits

- **WHEN** any editing operation authors to the stage
- **THEN** the opinion is written to the session (edit) layer and the original
  file on disk is unchanged

#### Scenario: File-authored transform can be overridden

- **WHEN** `set_transform` is called on a prim whose `xformOp:transform` is
  authored in the loaded root/file layer
- **THEN** no duplicate-op error is raised and the prim's composed local
  transform equals the newly authored matrix

### Requirement: Add a model by USD reference

The renderer SHALL provide `add_model(usd_path, parent_prim_path="/World", name=None, transform=None)`
that defines an `Xform` prim under the parent in the edit layer, adds a reference
to `usd_path`, authors the given transform (identity when omitted), bakes only
the newly referenced subtree, appends the resulting instances to the scene,
rebuilds the GPU geometry buffers, and returns the new prim's path. When the
parent prim does not exist it SHALL be created as an `Xform`.

#### Scenario: Added model appears in the scene

- **WHEN** `add_model` is called with a valid USD file path
- **THEN** the returned prim exists on the stage and the scene's instance count increases by the referenced model's instances

#### Scenario: Added model is visible

- **WHEN** a model is added and a frame is rendered
- **THEN** the rendered pixels change relative to before the add

#### Scenario: Added model honors its transform

- **WHEN** `add_model` is given a non-identity transform
- **THEN** the new instances are placed using that transform

### Requirement: Remove a node non-destructively

The renderer SHALL provide `remove_node(prim_path)` that deactivates the prim by
authoring `active = false` in the edit layer and re-reads the stage so the node
stops contributing to the render. The re-read SHALL cover meshes (instances),
distant/sphere lights, and the camera, so removing any of these prims drops it
from the scene. The original file SHALL remain unchanged.

#### Scenario: Removed mesh disappears

- **WHEN** `remove_node` is called on a present mesh prim path
- **THEN** the prim is inactive on the stage and its instances are absent from the scene

#### Scenario: Removed light disappears

- **WHEN** `remove_node` is called on a present light prim path
- **THEN** the prim is inactive and the light no longer contributes to the render

#### Scenario: Remove rebuilds surviving geometry from cache

- **WHEN** a node is removed
- **THEN** the surviving instances are re-uploaded without re-baking their meshes

### Requirement: Set a node transform with a fast resync

The renderer SHALL provide `set_transform(prim_path, matrix)` that authors
`xformOp:transform` on the prim in the edit layer and refreshes only the affected
instance transforms via the transform-only upload path, without re-baking or
re-concatenating geometry.

#### Scenario: Transform updates instance placement

- **WHEN** `set_transform` is called on a present prim path
- **THEN** the affected instances move to the new transform and no geometry re-bake occurs

### Requirement: Edits target prims by USD prim path

`MeshInstance` SHALL carry the originating `prim_path`, and the renderer SHALL
maintain an index from prim path to its instances, rebuilt whenever the instance
list is rebuilt. The existing `apply_instance_transform` and `apply_node_enabled`
operations SHALL identify their target by prim path rather than by integer
instance index, and all in-repo callers (the rotate gizmo, the ImGui scene-graph
panel, and the Qt scene-graph panel) SHALL pass the prim path. Resolution SHALL
succeed for mesh instances without a built scene graph (via the prim-path index)
and SHALL fall back to the scene-graph node for lights and cameras.

#### Scenario: Instance edits resolve by prim path

- **WHEN** `apply_instance_transform` or `apply_node_enabled` is called with a prim path
- **THEN** the matching instances are updated

#### Scenario: Index stays consistent after geometry edits

- **WHEN** a model is added and then a node removed
- **THEN** the prim-path-to-instance index matches the live instance list

### Requirement: Persist edits on request

The renderer SHALL provide `save_edits(path=None)` that writes the edit layer to
disk (default sibling `<scene>.edits.usda`). Saving SHALL produce a file that can
be reopened as a valid USD layer. No editing operation other than `save_edits`
SHALL write to disk.

#### Scenario: Saved edit layer is reopenable

- **WHEN** edits are made and `save_edits` is called
- **THEN** a USD layer file is written that reopens and contains the authored edits

### Requirement: Enumerate editable nodes

The renderer SHALL provide `list_nodes()` returning the editable prims of the
stage (at minimum each prim path and whether it is active), suitable for
scripting and as the data source for a future scene-graph UI.

#### Scenario: Listing reflects current edits

- **WHEN** a model is added and `list_nodes` is called
- **THEN** the new prim path is present in the result

### Requirement: Edits are atomic and reset accumulation

Editing operations SHALL validate their inputs before mutating the stage and
leave the scene and stage unchanged on failure. `add_model` SHALL raise
`ValueError` when `usd_path` is missing or unreadable, mutating nothing; if
baking the added subtree fails, the authored prim SHALL be removed.
`remove_node` and `set_transform` SHALL raise `ValueError` for an unknown prim
path. Every successful edit SHALL reset progressive accumulation.

#### Scenario: Invalid add leaves scene untouched

- **WHEN** `add_model` is called with a path that does not exist or is not valid USD
- **THEN** a `ValueError` is raised and the stage and scene are unchanged

#### Scenario: Unknown prim path is rejected

- **WHEN** `remove_node` or `set_transform` is called with a prim path that is not on the stage
- **THEN** a `ValueError` is raised and nothing is mutated

#### Scenario: Successful edit resets accumulation

- **WHEN** any edit completes successfully
- **THEN** the progressive accumulation frame counter resets to zero

### Requirement: Geometry edits refresh the derived scene graph

After `add_model` and `remove_node`, the renderer SHALL rebuild the derived scene
graph (`_scene_graph`, including the synthesized default-light injection) and bump
its version counter, so observers that poll the scene graph (the UI panels)
repaint to reflect the change.

#### Scenario: Scene graph reflects an add

- **WHEN** `add_model` completes
- **THEN** the rebuilt scene graph contains the new prim and the version counter has increased

#### Scenario: Scene graph reflects a remove

- **WHEN** `remove_node` completes
- **THEN** the rebuilt scene graph omits the removed prim and the version counter has increased

### Requirement: Lights carry prim-path identity preserved across resync

`LightDir` and `LightSphere` SHALL carry the originating `prim_path`. When a
geometry edit re-reads lights from the stage, the renderer SHALL preserve each
light's runtime `enabled` flag by matching prim path, so a user toggle is not
lost by an unrelated add/remove. Synthesized `/Skinny/*` default lights SHALL be
re-injected rather than dropped.

#### Scenario: Light enable survives an unrelated edit

- **WHEN** a light is toggled off and then an unrelated model is added
- **THEN** the light remains off after the resync

#### Scenario: Default lights survive a resync

- **WHEN** a scene relying on synthesized default lights undergoes a geometry edit
- **THEN** the default lights are still present in the rebuilt scene graph

### Requirement: Set a dome light's texture honoring lighting authority

The renderer SHALL provide `apply_dome_light_texture(env_index, path)` that loads
the HDR at `path`, mirrors it onto the source `UsdLuxDomeLight` prim's
`texture:file`, uploads it to the GPU, and makes the dome contribute light under
the **active lighting authority**, regardless of whether that dome had an
environment before the call.

When an authored USD scene is active (`uses_default_lights == False`), the loaded
environment SHALL be assigned to the authored scene's environment
(`_usd_scene.environment`) — constructing it when the dome previously had none —
so that the authority-selected environment is non-null and its contribution
intensity is non-zero. When the fallback default-lights authority is active, the
environment SHALL be written to the fallback environment library at `env_index`.
The env-upload cache SHALL be invalidated and accumulation SHALL reset so the new
texture is visible on the next rendered frame without any further edit.

#### Scenario: Texture set on a freshly added dome contributes immediately

- **WHEN** a dome light is added to an authored scene (so it starts with no
  environment) and `apply_dome_light_texture` is then called with a valid HDR
- **THEN** the authored scene's environment becomes non-null and the dome's
  environment contribution intensity is non-zero, with no enable toggle or
  reload required

#### Scenario: Texture swap on an already-textured dome still applies

- **WHEN** `apply_dome_light_texture` is called on a dome that already had an
  environment
- **THEN** the environment data is replaced in place and the rendered image
  reflects the new texture

#### Scenario: Missing file is reported and leaves the scene unchanged

- **WHEN** `apply_dome_light_texture` is called with a path that is not a file
- **THEN** it returns False, logs a load failure, and does not alter the active
  environment

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
- All curated presets SHALL keep absolute references into the renderer's
  assets directory and are never copied — texture-bearing docs would lose
  their textures if copied without walking filename-typed inputs, and the
  widened all-presets rule keeps save classification a single
  session-dir-vs-assets-dir test.

#### Scenario: Created-scene save is self-contained

- **WHEN** an anonymous-root scene with a bound synthesized material is
  saved to an allowed root and reloaded standalone from there
- **THEN** the material composes from the copied `.mtlx` and the render
  matches the pre-save scene

#### Scenario: Texture-bearing preset survives save

- **WHEN** a scene bound to `wood_tiled` is saved and reloaded
- **THEN** the wood material still resolves its textures (via the absolute
  assets reference), not a fallback

