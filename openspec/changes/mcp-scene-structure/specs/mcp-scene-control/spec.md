# Delta: mcp-scene-control

## REMOVED Requirements

### Requirement: Persistence, node authoring, and rendered output are excluded

**Reason**: Node authoring and persistence are now provided by the structural
tool set and save tool this change adds; only rendered-image return remains
excluded. The combined requirement is removed so a new image-only exclusion can
carry a title that matches its body.

**Migration**: Clients gain `scene_add_model` / `scene_add_primitive` /
`scene_add_light` / `scene_remove` / `scene_save`. The "Rendered image output is
excluded" requirement below preserves the one surviving exclusion.

## ADDED Requirements

### Requirement: Rendered image output is excluded

The server SHALL NOT expose a tool that returns rendered images, because any
edit resets progressive accumulation, so an immediate readback would return a
near-noise frame.

The save tool SHALL document that property edits made through the property-write
tool mutate in-memory render state without authoring to the USD edit layer, so a
save captures structural edits (adds, removals, transforms) but omits material
and light parameter edits — the same partial-save behavior the graphical
editor's save action has.

#### Scenario: No image tool is advertised

- **WHEN** a client lists available tools
- **THEN** no tool returning rendered image data is present

#### Scenario: Save tool documents the partial-save caveat

- **WHEN** a client reads the save tool's description
- **THEN** it states that property edits are not captured by a save, naming the
  distinction between structural edits and in-memory parameter edits

### Requirement: Structural scene tools

The server SHALL expose structural mutation as flat, single-purpose tools —
add a model by USD file reference, add an analytic primitive, add a light,
remove a node, and save the edit layer — each backed by the same renderer verb
the graphical editors call, not by a parallel authoring path. The server SHALL
NOT expose a single polymorphic add tool with union-typed parameters.

Each add tool SHALL accept an optional name and an optional parent prim path,
defaulting to `/World` (created if absent) when no parent is given — the
graphical editors derive a parent from the operator's selection, which an MCP
client does not have. Names SHALL be uniquified against existing prims rather
than failing on collision, and every successful result SHALL report the final
prim path chosen.

Removal SHALL apply the same deletability guard as the graphical editors —
the pseudo-root and synthesized `/Skinny/*` prims are refused — and SHALL be
non-destructive deactivation, with that semantic stated in the tool
description. Removal SHALL distinguish a path that resolves to no scene node
(reported as no such node) from a resolved node the guard refuses (reported as
not deletable), so the two are not conflated into one misleading message.

When no USD stage with an active edit layer is loaded, each structural tool
SHALL return an error stating that no editable USD stage is loaded.

#### Scenario: Add by file reference

- **WHEN** a client calls the add-model tool with the path of a USD file inside
  the allowed roots
- **THEN** the file is referenced under a new prim, the scene re-syncs, and the
  result reports the new prim's path

#### Scenario: Name collision is uniquified

- **WHEN** a client adds two objects with the same requested name
- **THEN** both succeed with distinct prim paths and each result reports the
  path actually used

#### Scenario: Synthesized node is not removable

- **WHEN** a client calls the remove tool on a `/Skinny/*` prim or the root
- **THEN** the tool returns an error and the scene is unchanged

#### Scenario: Removing an unknown path is reported as not found

- **WHEN** a client calls the remove tool on a path that resolves to no scene
  node
- **THEN** the error states no such node, distinct from the not-deletable error
  a guarded node produces

#### Scenario: No stage loaded

- **WHEN** a structural tool is called while no USD stage with an edit layer is
  active
- **THEN** the tool returns an error stating that no editable USD stage is
  loaded, not a transport error or attribute failure

#### Scenario: Save writes the edit layer

- **WHEN** a client calls the save tool with a destination path inside the
  allowed roots
- **THEN** the edit layer is written to that path and the result confirms the
  path written

#### Scenario: Save requires an explicit path

- **WHEN** a client calls the save tool with no path
- **THEN** the tool returns an error requiring an explicit in-root destination,
  rather than falling back to the renderer default beside the loaded scene,
  which is typically outside the roots and unchecked

### Requirement: Primitive adds carry an editable material

The add-primitive tool SHALL accept the analytic gprim types the loader can
tessellate (Sphere, Cube, Cylinder, Cone, Capsule, Plane) and SHALL author,
alongside the gprim, a dedicated preview-surface material bound to it, with
optional color, roughness, and metallic parameters applying sensible defaults
when omitted. A primitive SHALL NOT be authored bare, because an unbound prim
resolves to the protected fallback material slot and its appearance could
never be edited afterwards.

#### Scenario: Primitive is editable after creation

- **WHEN** a client adds a primitive and later writes a material parameter on
  the material node the add created
- **THEN** the write applies through the ordinary property-write dispatch, and
  the change is visible in the render

#### Scenario: Color at creation

- **WHEN** a client adds a primitive with a color argument
- **THEN** the authored material's diffuse color is that color, observable via
  a subsequent node read

#### Scenario: Unsupported primitive type

- **WHEN** a client requests a primitive type outside the supported set
- **THEN** the tool returns an error naming the supported types

### Requirement: Light adds accept common parameters

The add-light tool SHALL accept the same UsdLux schema set the graphical
editors offer (DistantLight, SphereLight, DomeLight, RectLight, DiskLight) and
optional intensity and color parameters valid across all of them. Type-specific
attributes such as a dome texture SHALL NOT be parameters of the add tool; they
remain post-add property writes.

#### Scenario: Light with intensity and color

- **WHEN** a client adds a light passing intensity and color
- **THEN** the authored light carries those values, observable via a subsequent
  node read

#### Scenario: Unsupported light type

- **WHEN** a client requests a light type outside the supported set
- **THEN** the tool returns an error naming the supported types

### Requirement: Transform arguments accept TRS or a matrix

Each add tool SHALL accept either a TRS form — translate, rotation as XYZ Euler
degrees, and scale as a scalar or three components — or a raw 16-element
row-major matrix, but SHALL reject a call providing both. The TRS form SHALL be
composed with the same convention the scene model's transform properties use,
so a subsequent node read returns the values the client wrote. The tool
documentation SHALL note that a sheared matrix does not round-trip through the
TRS properties a node read reports.

#### Scenario: TRS round-trips

- **WHEN** a client adds an object with translate, rotation, and scale values
- **THEN** a subsequent read of the created node's transform properties returns
  those values

#### Scenario: Both forms rejected

- **WHEN** a client provides both TRS components and a matrix
- **THEN** the tool returns an error and nothing is authored

### Requirement: Filesystem arguments are confined to allowed roots

Every filesystem path argument SHALL be resolved through symlinks to a real
path and required to lie under one of the configured root directories. This
covers the model file to reference, the save destination, and asset-typed
property writes such as texture and lens files. The roots SHALL be configurable
by a command-line option taking precedence over an environment variable, with a
default covering the platform temporary directories (both the per-user
temporary directory and `/tmp` resolved through symlinks, which differ on
macOS) and the process working directory.

For a model add, enforcement SHALL extend beyond the argument: after the
reference recomposes and the added subtree's payloads are loaded, every layer
newly introduced into the stage's used-layer set SHALL lie under the roots
(anonymous layers exempt, and layers already present before the add never
re-checked), and every asset-valued attribute in the added subtree — traversed
including instanced-prototype prims — SHALL have its resolved path lie under the
roots. The check SHALL use the resolved path, not the authored string (authored
asset paths are typically relative). An asset value that does not resolve to a
concrete path (asset-not-found, or a `<UDIM>`-style template) SHALL be rejected
rather than allowed to pass unchecked. A violation SHALL roll the added prim
back using the renderer's existing rollback and SHALL return an error naming the
offending path and the configured roots. The check is performed at add time; it
is a guardrail within one trust domain, not a sandbox — deferred payloads and
instanced textures are covered, but resolver plugins and edits made outside the
server are not — and the documentation SHALL say so.

#### Scenario: Argument outside roots

- **WHEN** a client calls a structural tool with a path outside every
  configured root
- **THEN** the tool returns an error naming the path and the roots, and nothing
  is authored

#### Scenario: Nested reference escapes the roots

- **WHEN** a referenced file inside the roots itself references a layer outside
  them
- **THEN** the add is rolled back after recompose and the error names the
  out-of-root layer

#### Scenario: Out-of-root texture in the added subtree

- **WHEN** a referenced file binds a texture whose resolved path lies outside
  the roots
- **THEN** the add is rolled back and the error names the texture path

#### Scenario: Operator's pre-existing layers are not policed

- **WHEN** the loaded scene already composes layers outside the roots and a
  client adds an unrelated in-root model
- **THEN** the add succeeds; only layers newly introduced by the add are
  checked

#### Scenario: Asset-typed property write outside roots

- **WHEN** a client writes a texture or lens file property with a path outside
  the roots
- **THEN** the write is rejected with an error naming the path and the roots

#### Scenario: Symlink does not bypass the roots

- **WHEN** a client passes an in-root symlink resolving outside the roots
- **THEN** the resolved real path is checked and the call is rejected

#### Scenario: Texture behind a deferred payload is checked

- **WHEN** a referenced file loads an out-of-root texture only through a payload
  or only inside an instanced prototype
- **THEN** the add loads payloads and traverses prototypes before the walk, so
  the out-of-root texture is caught and the add is rolled back

#### Scenario: Unresolvable asset is rejected

- **WHEN** an asset-valued attribute in the added subtree does not resolve to a
  concrete path
- **THEN** the add is rejected rather than passing the unresolved asset
  unchecked

### Requirement: Structural tools degrade to pollable jobs

Structural tools SHALL wait a short inline grace period for the render-thread
work to complete and return the result directly when it does. When the work
outlasts the grace period, the tool SHALL return a pending status with a job
identifier instead of cancelling, and a job-status tool SHALL report that job
as pending, completed with its result, or failed with its error. Completed job
results SHALL carry the final prim path. The store SHALL retain a bounded
number of recent jobs for a bounded time, and an unknown job identifier SHALL
be a tool error.

This replaces flat-timeout cancellation for structural work because a
cancelled-after-start add still completes on the render thread — the client
would be told the outcome is unknown while the scene silently changed, and a
retrying client would then duplicate the prim.

#### Scenario: Fast add returns synchronously

- **WHEN** a structural call's render-thread work completes within the grace
  period
- **THEN** the tool returns the completed result directly with no job to poll

#### Scenario: Slow add returns a job

- **WHEN** a structural call's work outlasts the grace period
- **THEN** the tool returns a pending status and a job identifier, and polling
  the job-status tool after completion returns the result including the final
  prim path

#### Scenario: Failed job reports its error

- **WHEN** a job's render-thread work raises
- **THEN** the job-status tool reports the failure with the error message, not
  a pending status forever

#### Scenario: Unknown job id

- **WHEN** the job-status tool is called with an identifier the store does not
  hold
- **THEN** the tool returns an error saying so
