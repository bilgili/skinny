# Tasks — restore render-thread tool docks

Implement in difficulty order (D6); each phase is independently landable and ends
with the dock opening + functioning without any GUI-thread live-renderer touch.

## 0. Shared infra (proxy plumbing)
NOTE (revised after Phase 1): the snapshot extension is only needed by docks
that *poll* renderer-owned state (Scene Graph). Docks whose reads are
event-driven (Python Material Editor) use the existing `proxy.request()` —
no `render_session.py` change. `render_session.py` stays Qt/GPU-free, so the
GUI-thread result marshalling lives in each dock as a `Signal(object)` hop, not
a shared render_session helper (0.3). These items move to the dock that first
needs them.
- [ ] 0.1 Extend `RendererStateSnapshot` with Scene-Graph-polled fields
      (`scene_graph` projection + `_scene_graph_version`, `usd_scene_id`/
      `usd_stage_id`, `materials`, camera params). → folded into Phase 2.
- [ ] 0.2 Populate + `apply_snapshot` those fields. → folded into Phase 2.
- [x] 0.3 GUI-thread result delivery = per-dock `Signal(object)` emitted from the
      worker's future done-callback (realized in Phase 1; reused by BXDF/Material
      Graph/Camera Debug). `render_session.py` unchanged.
- [x] 0.4 Hostless tests for the worker-side seam: `MaterialReloader` optional
      lock (`tests/test_material_reloader.py`) + dock render-thread-safety guard
      (`tests/test_qt_python_material_dock.py`).

## 1. Python Material Editor (easy) — DONE
- [x] 1.1 Construct `PythonMaterialEditorDock` with the proxy; drop the
      `render_lock` arg. Reload runs on the worker via
      `proxy.request(lambda r: MaterialReloader(r).reload(m, s))`;
      `MaterialReloader.render_lock` made optional (worker is single-owner →
      `nullcontext`).
- [x] 1.2 Module-list read routed through `proxy.request(lambda r:
      r.scene_python_modules())`; result marshalled to the GUI via `_modules_ready`
      → `_apply_modules`. Reload result marshalled via `_reload_ready` →
      `_apply_reload_result`. Poll + refresh both go through the worker.
- [x] 1.3 Unstub `app.py._open_python_material_editor` (proxy, no lock; import
      restored); `_open_python_material_in_editor` + session-restore reopen now
      functional.
- [x] 1.4 Verified: ruff + py_compile clean; 12 hostless tests green; offscreen-Qt
      runtime smoke — dock instantiates, `refresh_from_renderer` populates the
      combo via the worker, Compile posts one worker request. Interactive GPU
      click-through (edit a Python material, recompile, see viewport update)
      remains a manual step on a display host.

## 2. Scene Graph (medium)
- [ ] 2.1 Construct `SceneGraphDock` with the proxy; tree + property panel read the
      snapshot `scene_graph` projection + `_scene_graph_version` + camera params.
- [ ] 2.2 Route every write (`set_gizmo_target`, `add_model`, `remove_node`,
      `save_edits`, `apply_subtree_enabled`, `apply_node_enabled`,
      `apply_camera_param`, `apply_material_override`, `apply_light_override`,
      `apply_instance_transform`, `set_transform`, `apply_dome_light_texture`,
      `apply_camera_lens_file`) through `proxy.post()` with coalesce keys for
      drag-rate edits, uncoalesced for discrete structural edits.
- [ ] 2.3 One-shot selected-node detail reads via `proxy.request()`.
- [ ] 2.4 Unstub `app.py._open_scene_graph`; keep the `_open_python_material_in_editor`
      double-click hook; restore session-restore reopen.
- [ ] 2.5 Verify: open, select/toggle/transform nodes, edit material/light/camera
      params, save edits; changes reset accumulation and appear.

## 3. BXDF Visualizer (hard)
- [ ] 3.1 Construct `BXDFDock` with the proxy + viewport; combo from snapshot
      `materials`; material hash from `_material_version`/`mtlx_overrides`.
- [ ] 3.2 Route `request_bxdf_eval` / `request_bssrdf_eval` through a worker
      command; deliver the numpy grid to the GUI via the D4 `Signal` helper.
- [ ] 3.3 Scene pick via `viewport.arm_scene_pick(cb)`; result marshalled to GUI.
- [ ] 3.4 Unstub `app.py._open_bxdf`; restore session-restore reopen.
- [ ] 3.5 Verify: open, pick a material/surface, eval renders the lobe grid; show
      the produced image back.

## 4. Material Graph (hard)
- [ ] 4.1 Construct `MaterialGraphDock` with the proxy; material list + env from
      snapshot; node graph reads `_mtlx_scene_materials` projection.
- [ ] 4.2 Topology edits (add/delete/connect) → `proxy.post()` chain
      (`_gen_scene_materials` → `_upload_graph_param_buffers` → version bump) on the
      worker; flat-material edits via `apply_material_override`; env via mirrored
      `env_index`.
- [ ] 4.3 `render_material_preview` via `proxy.request()`; debounce on GUI, set the
      preview pixmap from the resolved `Future`.
- [ ] 4.4 Unstub `app.py._open_material_graph`; restore session-restore reopen.
- [ ] 4.5 Verify: open, edit graph topology + params, preview updates; show the
      preview image back.

## 5. Camera Debug viewport (hard)
- [ ] 5.1 Worker owns the `DebugViewport(ctx, …)`, constructed on first open via a
      `post()`/`request`; remove `main_lock`/`ctx` from the dock constructor.
- [ ] 5.2 Worker renders the embedded view each frame when open and emits its RGBA8
      in the snapshot (`debug_frame`); dock blits via `QImage`.
- [ ] 5.3 Camera/display input (`orbit`/`pan`/`zoom`/`toggle_cam_mode`/`show_grid`/
      wires) + open/resize/destroy lifecycle → `proxy.post()` commands.
- [ ] 5.4 Unstub `app.py._toggle_debug_viewport` + `_ensure_debug_dock`; restore
      session-restore reopen + close/destroy on the worker.
- [ ] 5.5 Verify: toggle debug view, orbit/pan/zoom, toggle overlays; frame updates
      live; GUI responsive; show a debug frame back.

## 6. Docs + close-out
- [ ] 6.1 Update `docs/Architecture.md` (Qt/render-thread + MetalContext sections)
      to state the tool docks are proxy-backed; update `AGENTS.md`/`CLAUDE.md` Qt
      notes.
- [ ] 6.2 `ruff check src/` clean; grep confirms zero GUI-thread live-renderer GPU
      calls in the five docks.
- [ ] 6.3 `openspec validate restore-render-thread-tool-docks --strict` passes;
      archive on completion.
