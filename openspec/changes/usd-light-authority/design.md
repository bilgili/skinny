## Context

The renderer currently has several partially independent default-light paths:

- `Renderer._scene_authors_lights` decides whether to upload the slider-driven
  default DistantLight. It tests for *powered* lights, so zero-intensity or
  disabled authored lights are treated as absent.
- `_apply_usd_lights` keeps a built-in environment at intensity `0.5` when the
  stage has no DomeLight, even if the stage authors a DistantLight, SphereLight,
  RectLight, or DiskLight.
- `_inject_default_lights_into_scene_graph` fills missing light types
  independently, so a stage can receive only one member of the default pair.
- IBL and Direct Light controls are built as unconditional top-level sections.
  Persisted values and headless options can therefore affect lighting even
  when the USD should be authoritative.

This proposal replaces those independent rules with one source of truth used
by rendering, scene-graph projection, and UI visibility.

## Goals / Non-Goals

**Goals:**

- Treat USD-authored lighting as the sole lighting authority whenever the
  scene contains any supported authored light source.
- Add the built-in DistantLight and IBL together only when no authored lighting
  exists.
- Show the fallback controls and synthesized scene-graph nodes only while that
  fallback pair is active.
- Keep the policy correct after asynchronous load, stage resync, add/remove
  operations, and animation/live-light refreshes.
- Apply identical behavior in Qt, Panel/web, GLFW, and headless sessions on
  Vulkan and Metal.

**Non-Goals:**

- Do not change USD light extraction, shading equations, sampling, or shader
  layouts.
- Do not remove editing of USD-authored light properties from the scene graph.
- Do not change furnace mode; it remains an explicit diagnostic override that
  replaces ordinary scene lighting.
- Do not add a user option for mixing renderer defaults with authored lights.

## Decisions

### 1. Record authored-light presence separately from power

The loaded `Scene` will carry explicit authored-light presence derived during
USD traversal. Supported `UsdLux` sources (DistantLight, SphereLight,
DomeLight, RectLight, and DiskLight) count even when disabled or authored at
zero intensity. Rect/Disk lights count before they are converted to emissive
geometry. An explicitly emissive material instance also counts as authored
lighting, preserving the existing guarantee that Skinny does not add a phantom
sun to an emissive-only scene.

This metadata is preferable to inferring authority from packed renderer
records: packed records intentionally discard disabled and zero-power lights,
and Rect/Disk lights no longer look like `UsdLux` lights after conversion.

For the no-USD/default-head session, authored-light presence is false.

### 2. One predicate drives an all-or-nothing fallback pair

Expose a single renderer predicate/state such as
`uses_default_lights = not scene.has_authored_lighting`. Every fallback
decision consumes it:

- `true`: use the selected built-in environment at the fallback intensity and
  upload the built-in DistantLight; expose both default controls/nodes.
- `false`: use `scene.environment` only when the USD authors a DomeLight,
  upload only authored distant lights, and never fill a missing light type;
  expose no renderer-owned default controls/nodes.

Sphere, Rect, Disk, and emissive-mesh sources continue through their existing
buffers. A scene with only a SphereLight therefore has no environment fill; a
scene with only a DomeLight has no analytic sun. That is intentional: missing
light types are part of the USD author's lighting design.

The predicate is recomputed whenever the derived `Scene` is rebuilt rather
than cached permanently by object identity. Removing the last authored light
activates both fallbacks; adding the first authored light removes both.

### 3. Fallback settings cannot override authored lighting

`env_index`, `env_intensity`, `direct_light_index`, and the default distant
light's direction/color/intensity remain as fallback state so the default-head
and light-less-scene workflows stay adjustable. In authored-light mode:

- authored DomeLight texture, intensity, enabled state, and transform come from
  the USD-derived environment;
- authored DistantLight records are uploaded regardless of the hidden
  fallback `direct_light_index`;
- headless `--env-intensity` / `--no-direct` and equivalent `RenderOptions`
  values are retained for a later fallback scene but do not alter the current
  authored-light scene.

USD light changes continue through the scene-graph and `usd:` control paths,
which mutate the stage and rebuild/refresh the derived scene.

### 4. Default controls are conditional, authored controls remain

The shared UI tree will conditionally include the complete `IBL` and
`Direct Light` sections using the same `uses_default_lights` state. The Qt and
Panel backends must remove both the section heading and its body when the
condition becomes false, then restore them when it becomes true; an empty
accordion/group heading is not acceptable.

The synthesized `/Skinny/DefaultLight` and `/Skinny/DefaultDome` nodes follow
the same predicate. When USD lighting is authoritative, neither node is
injected. Real USD light nodes and their editable properties are unaffected.

The UI condition must react to the asynchronous metadata load and later scene
resyncs without rebuilding unrelated material or camera controls.

## Risks / Trade-offs

- [Scenes that relied on Skinny's fill become darker] → This is the intended
  breaking correction; document that a DomeLight must be authored when IBL is
  desired.
- [A zero-intensity authored light produces a black scene instead of defaults]
  → Presence represents author intent. The author can remove/deactivate the
  light prim through an edit if fallback lighting is desired.
- [Conditional top-level sections can leave stale headings in one UI backend]
  → Add parity tests for both Qt-tree and Panel-tree conditional section
  behavior, including false→true and true→false transitions.
- [Fallback state leaks through a hidden global toggle] → Test authored
  DistantLight and DomeLight scenes with adversarial persisted fallback values.
- [Rect/Disk conversion loses light identity] → Record authored-light presence
  during USD traversal, before conversion to emissive triangles.

## Migration Plan

1. Add authored-light metadata and the central authority predicate behind
   tests.
2. Route render and scene-graph fallback decisions through the predicate.
3. Make fallback controls conditional in both UI backends.
4. Update API/documentation to label fallback-only settings.

Rollback restores the previous powered-light predicate, per-light-type
scene-graph injection, and unconditional control sections. No persisted data
migration is required; existing fallback settings can remain stored and become
active again only for a light-less scene.

## Open Questions

None. The authority and visibility rules are explicit: any authored lighting
source disables both fallbacks and their controls.
