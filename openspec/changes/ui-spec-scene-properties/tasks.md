# Tasks: ui-spec-scene-properties

## 1. Coverage first (before moving code)

- [x] 1.1 The Qt dock tests are `inspect.getsource` substring assertions and
      will not catch a behavioural regression. Add per-prop-type behavioural
      tests against a stub renderer for both front-ends, at current behaviour.
      → `tests/test_scene_property_nodes.py`. Tests target the SHARED mapper
      (what control + what committed value), not the about-to-be-deleted
      per-toolkit code — the durable safety net both docks migrate onto.
- [x] 1.2 Diff Qt's `_build_property_widget` + 8 helpers
      (`qt/windows/scene_graph.py:397-760`) against Panel's
      `_build_scene_prop_widget` (`ui/panel/windows.py:243-412`) per prop type.
      → `notes/coverage.md`. **CORRECTION:** Panel does NOT re-inline the
      fan-out guard (`:256-261`/`:342-345` are docstrings acknowledging the
      shared guard). Real Panel gaps: lens_file, texture_file, readonly
      color/vec, camera live-pull.
- [x] 1.3 Same for material-graph input rows (Qt `:658-830` vs Panel
      `:782-857`). → `notes/coverage.md`. Panel missing vector2 + filename.
- [x] 1.4 Tabulate the four key maps (GLFW viewport, Qt viewport, web template,
      Qt debug dock) and mark each divergence fix-or-record. → `notes/coverage.md`.
      **CORRECTION:** Qt debug dock is NOT missing `Key_D → show_dof_planes`
      (bound at `debug_viewport.py:137`, guarded from WASD). Genuine
      reconcile-or-record items: Qt debug dock has no Escape; web debug has no
      keyboard/mouse. Reconciliation actions land with 3.4.

## 2. Spec node types

- [x] 2.1 Add scene-property node types to `ui/spec.py` (bool, float, color,
      vec3, vec2, int, lens file, texture file), carrying the shared edit
      semantics. → existing leaf nodes reused; added a read-only `Label` node
      (closes Panel readonly gaps) and `Vector.step` (transform spans need a
      small step). Both backends render both. Mapper:
      `ui/scene_property_nodes.py`.
- [x] 2.2 Decide the open question: one node family for scene properties and
      graph inputs, or two. Record it. → ONE family (existing spec leaf nodes),
      two thin source adapters. Recorded in `notes/coverage.md` §2.2.
- [x] 2.3 Extend `test_every_param_bound_exactly_once` to the new types.
      → `test_both_backends_dispatch_every_renderable_node_type` +
      `test_every_editable_{scene_prop,graph_input}_type_yields_an_editable_control`:
      a node type emitted by the mapper but not dispatched by both backends
      fails the build.

## 3. Migrate, one dock at a time  (3.1 done; 3.2-3.4 GUI-gated, see 4.3)

- [x] 3.1 Scene Graph — DONE both front-ends, hostless-green. Qt
      `_build_properties` and Panel `_build_scene_prop_widget` now build nodes
      via `scene_property_to_node` and render through the backend walker
      (`QtTreeBuilder` / `PanelTreeBuilder.render_leaf`). `commit` transport:
      Qt direct + `_status(reason)`; Panel `run_on_render_thread` + report. Qt
      `get_live` reads live camera scalars. Deleted Qt's 8 `_add_*` helpers +
      `_build_property_widget` (−325 lines) and Panel's switch (−116). The
      "delete two Panel re-inlinings" step was a NO-OP (1.2 correction).
      Behavioural dock tests rewritten (`test_qt_scene_graph_dock.py`,
      `test_panel_scene_graph_lights.py`). Side effects, recorded:
      (a) Panel gained `lens_file`/`texture_file` pickers + readonly rows by
      construction; (b) Panel backend `Vector`/`IntSpin` now render numeric
      inputs (was sliders) to match Qt and address unbounded scene values —
      also changes 3 bounded sidebar controls slider→input in Panel;
      (c) Qt lens/texture file loads now go through the shared `FilePicker` +
      `apply_scene_property` (proxy-marshalled) instead of the dock's async
      `_await`; the failure reason still surfaces via `_status`. Qt's growable
      live-range rescale nicety dropped for cross-front-end parity (fixed 1e9).
- [x] 3.2 Material Graph — DONE both front-ends. Qt `_refresh_side` and Panel
      `_build_graph_input_widget` build input rows via `graph_input_to_node`
      (which now also maps connected ports → read-only label) and render through
      the backend walker; commit stays each front-end's `_apply_value_edit` /
      `_apply_graph_edit` worker post. Deleted Qt's `_build_input_row` + 6
      `_add_*_row` helpers. Panel gained `vector2` + `filename` inputs by
      construction. New behavioural tests in `test_qt_material_graph_dock.py`
      (float edit routes; connected input is read-only).
- [x] 3.3 BXDF — N/A for this seam. The BXDF dock is a lobe **visualizer**
      (material combo + θ/φ sliders + canvas + mode radios); it has NO
      property-type switch to fold into `scene_property_to_node`. Its Qt/Panel
      duplication is the whole visualizer (canvas + eval), out of scope for the
      property-node seam. Recorded, not forced.
- [x] 3.4 Camera Debug — key-map reconciled. FIXED: Qt debug dock now binds
      Escape→close (matches GLFW). RECORDED: web debug is button-only
      (Top/Left/Back/reset), a deliberate gap (browser has no free-camera/gizmo
      verb). Both asserted in `test_gizmo_mode_parity.py`. (The proposal's
      "Qt missing Key_D" claim was false — 1.4 correction; Key_D is bound.)

## 4. Gates

- [x] 4.1 1.1's behavioural tests still green after each dock migrates. → green
      after Scene Graph + Material Graph; new behavioural dock tests added for
      both. (BXDF N/A; Camera Debug is key-map only.)
- [~] 4.2 `ruff check src/`; full hostless `pytest`. → ruff clean on
      `src/skinny/ui/`; 224 hostless UI/scene tests + 29 debug/gizmo tests pass,
      0 regressions. (16 `tests/pbrt` corpus failures are pre-existing
      missing-`assets/` in the worktree, unrelated.) A full-repo hostless sweep
      still owes a run on a checkout with `assets/` present.
- [ ] 4.3 Manual: open every dock in `skinny-gui` and in `skinny-web`; edit one
      property of each type in both.
- [x] 4.4 Line-count check: −562 net src lines across `src/skinny/ui/` (201
      insertions, 763 deletions). ~725 lines of duplicated per-toolkit property/
      input switches (Qt scene ~365 + Qt material ~170 + Panel scene ~120 +
      Panel material ~70) replaced by the single 232-line
      `scene_property_nodes.py`. Materially reduced + deduplicated, not
      relocated.
- [x] 4.5 Docs: added a "Shared property→control mapping" section to
      `docs/FrontEnds.md` (the front-end architecture doc; `Architecture.md`'s
      UI content moved there under `docs-split-large-docs`).
- [x] 4.6 `openspec validate ui-spec-scene-properties --strict`. → valid.
