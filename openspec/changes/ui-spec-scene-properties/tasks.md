# Tasks: ui-spec-scene-properties

## 1. Coverage first (before moving code)

- [ ] 1.1 The Qt dock tests are `inspect.getsource` substring assertions and
      will not catch a behavioural regression. Add per-prop-type behavioural
      tests against a stub renderer for both front-ends, at current behaviour.
- [ ] 1.2 Diff Qt's `_build_property_widget` + 8 helpers
      (`qt/windows/scene_graph.py:397-760`) against Panel's
      `_build_scene_prop_widget` (`ui/panel/windows.py:243-412`) per prop type.
      Record every difference as intended or accidental — Panel is smaller
      partly because it does less.
- [ ] 1.3 Same for material-graph input rows (Qt `:658-830` vs Panel
      `:782-857`).
- [ ] 1.4 Tabulate the four key maps (GLFW viewport, Qt viewport, web template,
      Qt debug dock) and mark each divergence fix-or-record.

## 2. Spec node types

- [ ] 2.1 Add scene-property node types to `ui/spec.py` (bool, float, color,
      vec3, vec2, int, lens file, texture file), carrying the shared edit
      semantics.
- [ ] 2.2 Decide the open question: one node family for scene properties and
      graph inputs, or two. Record it.
- [ ] 2.3 Extend `test_every_param_bound_exactly_once` to the new types.

## 3. Migrate, one dock at a time

- [ ] 3.1 Scene Graph — largest duplicate, carries the re-inlined guards.
      Delete the two Panel re-inlinings (`windows.py:261`, `:349`).
- [ ] 3.2 Material Graph.
- [ ] 3.3 BXDF.
- [ ] 3.4 Camera Debug — including the key-map reconciliation from 1.4.

## 4. Gates

- [ ] 4.1 1.1's behavioural tests still green after each dock migrates.
- [ ] 4.2 `ruff check src/`; full hostless `pytest`.
- [ ] 4.3 Manual: open every dock in `skinny-gui` and in `skinny-web`; edit one
      property of each type in both.
- [ ] 4.4 Line-count check: the ~530 lines of property mapping are materially
      reduced, not merely relocated.
- [ ] 4.5 Docs: `docs/Architecture.md` UI section.
- [ ] 4.6 `openspec validate ui-spec-scene-properties --strict`.
