# Design: ui-spec-scene-properties

## Context

This change is not speculative: the pattern already exists and is already
tested. `ui/spec.py` + `ui/build_app_ui.py` + two backends is exactly the shape
proposed, applied to parameters. The docks were built before that seam existed,
or beside it.

The evidence that the duplication is costing something is the drift already
present: the Qt debug dock has no Escape binding, the Panel debug dock has
neither keyboard nor mouse, and the smaller Panel property switch silently omits
prop types the Qt switch handles (`lens_file`, `texture_file`, read-only colour
and vector, camera live-pull). The edit routing itself is already shared through
`scene_edit_actions.apply_scene_property`; only the property→control switch is
duplicated.

## Goals / Non-Goals

**Goals**
- One prop-type switch for scene properties and graph inputs.
- Shared edit semantics applied once, not re-inlined.
- The bind-exactly-once guard extended to docks.

**Non-Goals**
- Unifying the Python material editor. Qt-only, nothing to unify with; by the
  one-adapter rule that seam would be hypothetical.
- Rewriting the Camera Debug render path. Its geometry half is already shared;
  only the interaction surface is triplicated, and that is reconciled, not
  redesigned.
- Building a general widget framework. The spec covers the node types that
  exist, and grows when a new one appears.

## Decisions

### D1 — Extend the existing spec, do not add a second one

Scene properties and graph inputs become node types alongside parameters. A
separate "scene property spec" would be a second seam for the same job, and the
two would drift exactly as the two dock implementations have.

### D2 — Adapters keep widget construction only

The line is: deciding *what control a property needs* is shared; building the
widget is per-toolkit. Today Qt's 363 lines and Panel's 170 both do both. After
the split, the per-toolkit part is small and obviously parallel.

### D3 — Shared edit semantics stay shared

`ui/scene_edit_actions.py` already owns the fan-out-first guard and edit
semantics, and both front-ends already route every edit through it. The change
keeps that routing shared as the property→control switch moves into the node
layer: the adapter builds a node whose setter calls the shared dispatcher. No
Panel guard is re-inlined today, so none is deleted; the requirement is that the
seam MUST NOT introduce a re-inlined copy, and a test asserts that.

### D4 — Extend the existing test, do not invent a new one

`test_every_param_bound_exactly_once` is the model. The extended version
asserts that every scene-property type and graph-input type is bound exactly
once, so a type handled in one front-end and not the other fails the build —
which is precisely the failure mode that produced the current divergence.

### D5 — One dock at a time

Scene Graph first: it is the largest duplicate (363 vs 170), it carries the
Panel prop-type gaps to close, and it has real behaviour to preserve. Then
Material Graph, then BXDF, then Camera Debug. Each lands independently green.

### D6 — Key maps: reconcile or record

Four independent key tables exist (GLFW viewport, Qt viewport, web template, Qt
debug dock). Some divergence is legitimate — the web has no gizmo verb at all.
The genuine defects to reconcile are the Qt debug dock's missing Escape binding
and the web debug dock's lack of any keyboard or mouse. The rule: every
divergence is either fixed or recorded with its reason. `test_gizmo_mode_parity`
already pins one binding across two front-ends and is the pattern; it currently
excludes the web by construction.

### D7 — One node family, two source adapters (resolves the open question)

Scene properties and graph inputs share ONE node family: the existing
`ui/spec.py` leaf node types. Two thin adapters emit them —
`scene_property_to_node` and `graph_input_to_node`. They map onto the same
widget vocabulary (float→slider, colour→picker, vec→vector, bool→checkbox,
file→picker, read-only→label) and differ only in how a value is *sourced*: a
scene property carries constraint metadata (`min`/`max`/`growable`), while a
graph-input `PortView` carries no constraint metadata at all, so its ranges are
per-type. That difference changes how a node is built, not which node type it
is, so a second node family would add a seam without removing one. The spec
grew one read-only `Label` node (both front-ends lacked a shared read-only row)
and a `step` field on `Vector` (a transform span of ±1e6 needs a small
single-step); the other leaf types were reused unchanged.

## Risks / Trade-offs

- **Risk: the Qt docks' only tests are source-substring assertions.** They will
  not catch a behavioural regression from this change. Mitigation: the extended
  bind-exactly-once test plus per-node-type unit tests against a stub renderer
  — build the coverage first, then move the code.
- **Risk: Qt and Panel property behaviour differs in ways users depend on.**
  Before unifying each prop type, diff the two implementations and record every
  difference as intended or accidental. The Panel implementations are smaller
  because they do less; some of that "less" is a missing feature, not a
  simplification.
- **Trade-off: a large mechanical diff.** Bounded by D5's one-dock-at-a-time
  pacing.

## Open Questions

- ~~Do graph-input rows and scene-property rows want the same node types, or two
  families?~~ **Resolved (D7): one family, two source adapters.** The earlier
  premise that graph inputs "carry uniform metadata" was wrong — a `PortView`
  carries no constraint metadata, so ranges are per-type.
- Should the web's missing gizmo verb be added while reconciling key maps, or
  recorded as a deliberate gap? It is a command-path question as much as a UI
  one — coordinate with `renderer-command-interface`.
