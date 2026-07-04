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
- [x] 0.1 Realized as a **separate** immutable `SceneStateSnapshot` (NOT folded
      into `RendererStateSnapshot`, whose `state_ready` fires once at startup):
      `scene_graph` projection + `scene_graph_version`, `usd_scene`/`scene`
      material projections, `has_usd_stage`/`has_usd_edit_layer`, `_CameraProj`.
      Built by `build_scene_state(renderer)` on the worker.
- [x] 0.2 `QtRendererProxy.apply_scene_state()` + `refresh_scene_state()` (request);
      proxy exposes `scene_graph`/`camera`/`_usd_stage`/`_usd_edit_layer`/`scene`
      from the cache so the dock's reads stay unchanged.
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

## 2. Scene Graph (medium) — DONE
- [x] 2.1 `SceneGraphDock` takes the proxy; `_tick` refreshes the proxy's scene
      cache via `refresh_scene_state()` (worker) then applies it + rebuilds the
      tree / runs camera pulls on the GUI thread. The dock's `self.renderer.*`
      reads (scene_graph, version, camera, `_usd_scene`, stage/edit-layer) hit
      the cache unchanged.
- [x] 2.2 The 8 fire-and-forget writes (`set_gizmo_target`, `apply_subtree_enabled`,
      `apply_node_enabled`, `apply_camera_param`, `apply_material_override`,
      `apply_light_override`, `apply_instance_transform`, `set_transform`) are
      `proxy.post()` wrappers with coalesce keys for drag-rate edits. The 5
      result-reporting edits (`add_model`, `save_edits`, `remove_node`,
      `apply_dome_light_texture`, `apply_camera_lens_file`) are `proxy.request()`
      → resolved off-thread via the dock's `_await` helper + `_run_on_gui`
      marshaller (no GUI-thread block).
- [x] 2.3 Result-reporting edits resolve via `request()`+`_await` (covers 2.3's
      round-trip need; no separate per-node detail request required).
- [x] 2.4 Unstub `app.py._open_scene_graph` (+ import); `_open_python_material_in_editor`
      double-click hook preserved; session-restore reopen functional.
- [x] 2.5 Verified: ruff + py_compile clean; 20 hostless tests green (6 proxy-surface
      TDD in `test_qt_scene_state_proxy.py` + 3 dock guards in
      `test_qt_scene_graph_dock.py`); offscreen-Qt smoke — tree populates from the
      worker refresh, camera proj reaches the proxy, node-select posts
      `set_gizmo_target` to the worker. Interactive GPU click-through remains manual.

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
