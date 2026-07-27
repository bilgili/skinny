# Change: ui-spec-scene-properties

## Why

The sidebar has a working seam: `ui/spec.py` declares toolkit-free nodes,
`ui/build_app_ui.py` builds one tree, and Qt and Panel each render it — with
`test_ui_spec.py::test_every_param_bound_exactly_once` guarding the parameter
registry against the tree.

The docks below it have no such seam. The same four docks are implemented
twice, independently:

| dock | Qt | Panel |
|---|---|---|
| Scene Graph | `qt/windows/scene_graph.py` 843 | `ui/panel/windows.py:48-493` (447) |
| Material Graph | `qt/windows/material_graph.py` 1,170 | `windows.py:681-929` (249) |
| BXDF | `qt/windows/bxdf.py` 547 | `windows.py:495-680` (186) |
| Camera Debug | `qt/windows/debug_viewport.py` 374 | `windows.py:930-1036` (107) |
| Python material editor | 660 | none |

~4.6k lines for four logical docks plus one Qt-only. Two backend tree-walkers
(`ui/qt/backend.py` 578, `ui/panel/backend.py` 602) render the same 11 node
types.

The largest single duplicate is the scene-property → widget mapping: Qt's
`_build_property_widget` plus eight `_add_*` helpers
(`qt/windows/scene_graph.py:397-760`, **363 lines**) against Panel's
`_build_scene_prop_widget` (`ui/panel/windows.py:243-412`, **170 lines**) —
the same prop-type switch (bool, float, color, vec3, vec2, int, lens file,
texture file), written independently. Panel additionally **re-inlines the
fan-out-first guard** from the shared `ui/scene_edit_actions.py` at two sites,
flagged in its own comments (`windows.py:261`, `:349`). Material-graph input
rows are the second largest: Qt `:658-830` (~173 lines) against Panel
`:782-857` (76).

Divergence has already happened. The Camera Debug interaction surface exists in
three forms — GLFW with a 12-key map, Qt with an 11-key map **missing
`Key_D` → `show_dof_planes`** (D is consumed by WASD) and no Esc binding, and
Panel with four buttons, no keyboard and no mouse. The Qt dock tests are
`inspect.getsource(...)` substring assertions — they pin that a call *appears
in the source*, not that it behaves.

## What Changes

- Extend the node spec to cover scene properties and material-graph inputs, as
  it already covers parameters: one prop-type switch, toolkit-free, with the
  shared edit semantics from `ui/scene_edit_actions.py` applied once.
- Qt and Panel keep only widget construction for each node type; the mapping
  from a scene property to a node moves into the shared layer.
- The re-inlined fan-out-first guards in the Panel windows are deleted in
  favour of the shared ones.
- Extend the existing bind-exactly-once test to scene properties and graph
  inputs, so a property type that renders in one front-end and not the other
  fails the build.
- Reconcile the Camera Debug key maps, or record deliberately which bindings
  are per-front-end.

## Capabilities

### Modified Capabilities

- `usd-scene-editing-ui`: scene-property editing is declared once as spec nodes
  with shared edit semantics, and rendered by per-toolkit widget adapters —
  replacing two independently written property-to-widget mappings and the
  re-inlined guards.

## Impact

- Modified: `src/skinny/ui/spec.py` (node types for scene properties and graph
  inputs), `src/skinny/ui/qt/windows/scene_graph.py`,
  `src/skinny/ui/qt/windows/material_graph.py`,
  `src/skinny/ui/panel/windows.py`, both backends.
- Expect ~530 lines of property mapping to become ~200.
- Unchanged: what the docks can do, the renderer, MCP.
- **Not** in scope: the Python material editor (Qt-only, no counterpart to
  unify with) and the Camera Debug render path (`debug_viewport.py`'s geometry
  half is already single-owner).
- Large mechanical diff — pace it one dock at a time, each independently green.
- Docs: `docs/Architecture.md` UI section.
