# Design: scene-intake-interface

## Context

`usd_loader.py` is 2,916 lines, 95 module-level defs, and owns almost no
state — it is nearly a pure transformer already. What stops it being a deep
module is its interface: 4 public loaders that the main consumer largely
bypasses, 9 privates reached by lazy import, and a back-reference into the
renderer from `resolve_control_binding`.

Three adoption paths exist because each grew for a different trigger
(initial load, streaming, post-edit) and each learned different lessons. The
post-edit path is the one that knows the most — it is the only one that
carries runtime state across a re-read.

## Goals / Non-Goals

**Goals**
- Intake returns a value; nothing flows back into intake from the renderer.
- One application path, so ordering is stated once.
- Time-indexed re-read as a call, not as per-frame reuse of extractor
  internals.
- Hostless assertion of intake results from synthetic stages.

**Non-Goals**
- Changing the `Scene` dataclass shape.
- Merging `Scene`, `SceneGraphNode` and the stage into one representation.
  Four representations exist and the redundancy is real, but collapsing them
  is a much larger change; this one only fixes who produces them.
- Touching the USD edit layer or MCP authoring.

## Decisions

### D1 — `SceneUpdate` is a value describing a change, not a new `Scene`

A full load is an update that replaces everything; a streamed batch is an
update that appends instances; a post-edit resync is an update that replaces
geometry while preserving runtime flags. One type, three fill patterns. This
is what lets one application path replace three.

### D2 — Runtime-state carry-over is part of applying an update

Instance-enabled flags, light-enabled flags and material overrides keyed by
`source_prim_path` (falling back to `name`) are renderer-side runtime state
never authored to USD. Today only `_resync_geometry_from_stage` preserves
them, in a block its own comment calls "finding #7". In the new shape,
"preserve runtime state across a geometry replacement" is a stated property of
applying an update, tested directly. Rejected: making intake aware of runtime
flags — that would reintroduce the back-reference.

### D3 — `id(_usd_scene)` as a UI change token must survive or be replaced

`ui/build_app_ui.py` compares `id(renderer._usd_scene)` at three sites to
detect a scene swap. `_resync_geometry_from_stage` hand-copies 8 fields
precisely to *avoid* changing that id. Two options: keep mutate-in-place and
preserve the token, or swap and give the UI an explicit version counter
alongside `_scene_graph_version` / `_material_version`, which already exist.
Prefer the explicit counter — `id()` as a change token is exactly the kind of
implicit interface this change exists to remove — but it makes the UI part of
the diff, so it is called out rather than assumed.

### D4 — `resolve_control_binding` inverts

Intake resolves a control binding to a description (target kind, index,
attribute path, value coercion). The renderer applies it. This deletes
`usd_loader`'s import of `skinny.params` and its three writes into renderer
state. The USD-driven control UI capability's observable behaviour is
unchanged.

### D5 — Promote, don't wrap

The 9 lazily-imported privates become part of the interface where the renderer
genuinely needs them per frame (camera and light re-read at time *t*, joint
matrices, LBS, smooth normals), and stay private where the need was incidental
(`_prim_has_mtlx_reference`, `_up_axis_rt` — folded into the update).
Wrapping all 9 in public aliases would keep the coupling and add a layer; the
deletion test says a pure alias module earns nothing.

### D6 — The pbrt front half is not in scope

`pbrt/` → `.usda` is already a clean, hostlessly tested stage (46 test files,
42 needing no device). The `customData["skinnyOverrides"]` channel between it
and the loader has **three separate readers** with their own merge orderings,
including one that must re-run `_derive_opacity_from_subsurface` because the
first derivation ran before customData was merged
(`usd_loader.py:1246-1253`). That is a real defect but it belongs with
`flat-material-field-table`, which owns the override key vocabulary. Noted
here, fixed there.

## Risks / Trade-offs

- **Risk: ordering differences between the three paths are load-bearing.**
  Two of them do things the third does not. Gate: before unifying, run each
  path on the same scene and diff the resulting `Scene` plus scene graph;
  every difference is either a bug to fix or a deliberate per-trigger step to
  keep in the update. Record the verdict per difference.
- **Risk: streaming is threaded.** `_bg_usd_stream` runs on a background
  thread and the graph is built before any instance exists, then back-filled
  by `populate_instance_refs`. The update type must be safe to build off-thread
  and apply on the render thread — this is the same discipline
  `renderer-command-interface` imposes on front-ends, and the two changes
  should agree on it.
- **Trade-off: `usd_loader.py` stays large.** This change fixes its interface,
  not its size. A later split by cluster (mesh, texture, light, camera,
  volume, skel) is possible once the interface is stable.

## Open Questions

- Should `SceneUpdate` carry the `SkeletalScene` (which holds a live
  `UsdSkel.Cache` and the stage, and must be kept alive)? A live pxr object
  inside a value type is uncomfortable; the alternative is a separate handle
  returned alongside. Leaning: separate handle.
- Does the streaming path keep its own queue, or become a sequence of
  `SceneUpdate`s? Leaning: a sequence, which makes it testable — but that
  changes the drain loop shape in `_poll_usd_streaming`.
