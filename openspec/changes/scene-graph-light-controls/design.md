## Context

Runtime scene editing is already centralized on `Renderer`: edits are authored
to an anonymous strongest sublayer, then `add_model` and `remove_node` call
`_resync_geometry_from_stage()` to rebuild the derived scene, GPU resources, and
scene graph. Both scene-graph front-ends already share parent/deletability
helpers and expose the existing operations.

The loader supports five authored USD light schemas: DistantLight, SphereLight,
DomeLight, RectLight, and DiskLight. The recent `usd-light-authority` change
makes the presence of any such prim authoritative, so creating the first light
must naturally remove both renderer-owned fallback lights.

## Goals / Non-Goals

**Goals:**

- Author any renderer-supported USD light type without leaving the scene-graph
  GUI.
- Use the selected group-like prim as the parent, with `/World` as the safe
  fallback.
- Make light creation non-destructive, saveable, unique, immediately visible,
  and consistent across Qt and Panel.
- Preserve render-thread ownership in Qt and the Panel session lock.
- Reuse the existing light-authority and stage-resync paths.

**Non-Goals:**

- Do not add a custom light shader or a new USD light schema.
- Do not add a modal light-parameter wizard; properties continue to be edited
  through the selected scene-graph node.
- Do not change lighting equations, sampling, or fallback authority.
- Do not create lights when no USD stage/edit layer is active.

## Decisions

### 1. Add one schema-typed renderer operation

`Renderer.add_light(light_type, parent_prim_path="/World", name=None,
transform=None)` accepts the canonical schema names `DistantLight`,
`SphereLight`, `DomeLight`, `RectLight`, and `DiskLight`. It validates the type
before mutation, creates a missing parent as an Xform when necessary, chooses a
valid unique prim path, defines the selected `UsdLux` schema in the active edit
layer, authors defaults, optionally authors a transform, and then calls the
existing full scene resync.

If definition or resync fails, the newly created prim is removed from the edit
layer before the error is surfaced. The source USD file is never written
directly; "Save edits" remains the persistence boundary.

### 2. Author explicit, editable defaults

Every new light receives explicit white color, intensity `1`, and exposure `0`
attributes so the property editor exposes stable values. Type-specific defaults
are: DistantLight angle `0.53`, SphereLight radius `0.5`, RectLight width/height
`1`, and DiskLight radius `0.5`. A DomeLight starts without a texture asset; the
existing DomeLight texture property is the place to choose its HDR.

An omitted transform is identity. The operation accepts a transform for API
callers without requiring the GUI to guess a scene scale or camera-relative
placement.

### 3. One compact control presents all types

The Qt toolbar uses an "Add light" menu button and Panel uses the equivalent
menu control. Each menu contains one action per supported schema. Selecting an
action resolves the parent using the same selected-group helper as "Add model",
posts the operation to the render worker (Qt) or performs it under the session
lock (Panel), and reports the created path or a non-fatal error.

The control is disabled unless an editable stage is active. After resync, normal
scene-graph polling repopulates the tree; no special UI-side node insertion is
needed.

### 4. Existing authority transition is the integration seam

The resync recomputes `Scene.has_authored_lighting`. Creating the first light
therefore switches `uses_default_lights` to false, removes both synthesized
`/Skinny` fallback nodes, and uploads only authored lighting. Additional lights
remain authored authority. Deleting the last light through the existing action
restores the complete fallback pair.

## Risks / Trade-offs

- [A DomeLight without an HDR makes the authored environment black] → Keep the
  existing texture picker visible on the new node and document that choosing an
  HDR completes DomeLight setup.
- [Five actions crowd the toolbar] → Use one menu button rather than five
  permanent toolbar buttons.
- [An identity-positioned local light may start inside geometry] → Do not guess
  scene scale; expose the transform parameter to API callers and retain
  scene-graph transform editing as the positioning mechanism.
- [Creating the first light changes authority immediately] → This is the
  established all-or-nothing rule; cover the transition explicitly in tests.

## Migration Plan

1. Add the renderer operation and hostless USD authorship tests.
2. Add the shared light-type definitions and Qt/Panel controls.
3. Validate authority, UI wiring, docs, and OpenSpec alignment.

Rollback removes the new operation and controls. Existing edit layers and
authored USD lights remain valid standard USD data.

## Open Questions

None.
