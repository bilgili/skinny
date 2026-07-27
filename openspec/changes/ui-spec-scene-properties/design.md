# Design: ui-spec-scene-properties

## Context

This change is not speculative: the pattern already exists and is already
tested. `ui/spec.py` + `ui/build_app_ui.py` + two backends is exactly the shape
proposed, applied to parameters. The docks were built before that seam existed,
or beside it.

The evidence that the duplication is costing something is the drift already
present: a missing debug-camera key binding in Qt, a Panel dock with neither
keyboard nor mouse, and guards re-inlined in Panel with comments admitting it.

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

### D3 — Shared edit semantics, no re-inlining

`ui/scene_edit_actions.py` already owns the fan-out-first guard and edit
semantics; Panel re-inlines it at two sites and says so in comments. The node
carries the semantics; adapters call them. Deleting the re-inlined copies is
part of the change, not a follow-up.

### D4 — Extend the existing test, do not invent a new one

`test_every_param_bound_exactly_once` is the model. The extended version
asserts that every scene-property type and graph-input type is bound exactly
once, so a type handled in one front-end and not the other fails the build —
which is precisely the failure mode that produced the current divergence.

### D5 — One dock at a time

Scene Graph first: it is the largest duplicate (363 vs 170), it has the
re-inlined guards, and it has real behaviour to preserve. Then Material Graph,
then BXDF, then Camera Debug. Each lands independently green.

### D6 — Key maps: reconcile or record

Four independent key tables exist (GLFW viewport, Qt viewport, web template, Qt
debug dock). Some divergence is legitimate — the web has no gizmo verb at all,
and `Key_D` is genuinely taken by WASD in the Qt debug dock. The rule: every
divergence is either fixed or recorded with its reason. `test_gizmo_mode_parity`
already pins one binding across two front-ends and is the pattern; it currently
excludes the web by construction.

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

- Do graph-input rows and scene-property rows want the same node types, or two
  families? They overlap heavily (float, color, vec, file) but graph inputs
  carry uniform metadata. Leaning: one family with an optional metadata field.
- Should the web's missing gizmo verb be added while reconciling key maps, or
  recorded as a deliberate gap? It is a command-path question as much as a UI
  one — coordinate with `renderer-command-interface`.
