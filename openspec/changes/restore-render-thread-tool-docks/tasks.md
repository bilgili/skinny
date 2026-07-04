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

## 3. BXDF Visualizer (hard) — DONE
- [x] 3.1 `BXDFDock` takes the proxy + viewport. The material projection was
      extended (`_MaterialProj` += name/mtlx_target_name/parameter_overrides;
      `SceneStateSnapshot` += material_version/mtlx_overrides/usd_scene_id) so the
      combo + material hash read the worker-refreshed cache; poll drives
      `refresh_scene_state()` and scene-swap keys off the stable `_usd_scene_id`.
- [x] 3.2 `proxy.request_bxdf_eval`/`request_bssrdf_eval(req, cb, on_error=)` post
      the eval to the worker; the grid is delivered by a worker callback that
      emits `_eval_ready` → `_on_eval_ready` (GUI). The CPU fallback (pure-numpy
      `eval_grid`) runs in the worker's `on_error` hook and is delivered through
      the same signal (carrying its own dirs).
- [x] 3.3 `_arm_pick` wraps the pick callback so the worker-invoked
      `request_scene_pick` result is marshalled to the GUI via `_run_on_gui`
      before `_on_pick_result`/`_on_entrance_pick_result` touch widgets.
- [x] 3.4 Unstub `app.py._open_bxdf` (+ import); session-restore reopen functional.
- [x] 3.5 Verified: ruff + py_compile clean; 27 hostless tests (3 new proxy-surface
      TDD + 4 dock guards in `test_qt_bxdf_dock.py`); offscreen-Qt smoke — combo
      populates from the worker, pick marshals to GUI, eval dispatches to the
      worker and the lobe pixmap renders (`[GPU | log]`). Interactive GPU
      click-through remains manual.

## 4. Material Graph (hard) — DONE (edit path source-verified only)
- [x] 4.1 `MaterialGraphDock` takes the proxy; material combo reads the cached
      projection (`_MaterialProj` += `mtlx_scene_target`), env combo reads the
      mirrored `environments`/`env_index`; `_poll_scene_swap` drives
      `refresh_scene_state()` and keys off the stable `_usd_scene_id`.
- [x] 4.2 The whole edit+regen+rebuild sequence moved to the worker atomically:
      `_run_edit(mutate, …)` posts a `request()` whose worker closure runs the doc
      mutation, then `build_view`+`validate`+`_gen_scene_materials`+
      `_upload_graph_param_buffers`+version bump, and returns
      `(status, new_view, valid, msg)`. Doc helpers relocated to module-level
      worker functions (`_worker_doc`/`_worker_mtlx_node`/`_set_mtlx_input`) taking
      the real renderer. Flat edits → `apply_material_override`; env → mirrored
      `env_index` + `proxy.ensure_env_uploaded()`. `_on_material_picked`'s
      `build_view` runs on the worker (`_build_view_on_worker`).
- [x] 4.3 `proxy.render_material_preview(mid, prim, size)` → `request()`; debounced
      on the GUI, blitted from the resolved future via `_resolve_to_gui` (no
      GUI-thread `.result()`).
- [x] 4.4 Unstub `app.py._open_material_graph` (+ import); session-restore reopen
      functional.
- [x] 4.5 Verified: ruff + py_compile clean; 36 hostless tests (3 new proxy-surface
      TDD + 6 dock guards in `test_qt_material_graph_dock.py`); offscreen-Qt smoke —
      material + env combos populate from the worker, preview dispatches to the
      worker and blits. **Verification gap (recorded):** the MaterialX-doc topology
      edits (`_apply_connect`/`_disconnect`/`_delete_node`/`_add_node`/graph value
      edits) are **source + compile verified only** — `build_view`/`mx.Color3`/
      `doc.getNodeGraph` need a loaded MaterialX document (PyMaterialXGenSlang, the
      repo-root py3.13 venv) and interactive GPU to exercise end-to-end. A subtle
      relocation bug in the edit closures would not be caught by the offscreen
      smoke; interactive confirm required.

## 5. Camera Debug viewport (hard) — DONE (GPU render source-verified only)
- [x] 5.1 The `DebugViewport` GPU object lives on the render worker as
      `renderer.debug_viewport`, built + opened on first show via a worker command
      (`_worker_debug_create`, uses `renderer.ctx`). Dock constructor takes
      `(renderer=proxy, viewport)` — `ctx`/`main_lock` removed.
- [x] 5.2 The worker (`_RenderWorker._maybe_render_debug`) renders the embedded
      view each frame when active + open and emits a `DebugFrame` via
      `debug_frame_ready`; `RenderViewport` forwards it; the dock blits it. The
      idle-sleep ladder caps to ~30 Hz while the debug view is active.
- [x] 5.3 Camera/display input (drag→orbit/pan/look, wheel→zoom, WASD→free-cam
      move, view presets, display toggles) + show/hide(active)/resize/close(destroy)
      lifecycle → `proxy.post()` closures over module-level worker helpers
      (`_worker_debug_*`). Mode checks run on the worker.
- [x] 5.4 Unstub `app.py._ensure_debug_dock` (proxy+viewport, no ctx/lock) +
      `_toggle_debug_viewport` (+ import); session-restore reopen functional.
- [x] 5.5 Verified: ruff + py_compile clean; 42 hostless tests (6 dock guards in
      `test_qt_debug_viewport_dock.py` + a worker-loop guard); offscreen-Qt smoke —
      dock creates/opens the viewport on the worker, blits an emitted DebugFrame,
      and a mouse drag posts an orbit command to the worker viewport. **Verification
      gap (recorded):** the actual embedded GPU render (`render_embedded`) needs a
      real context + GPU; it is **source + compile verified only** — interactive
      confirm required (toggle the view, orbit/pan/zoom, overlays).

## 6. Docs + close-out
- [ ] 6.1 Update `docs/Architecture.md` (Qt/render-thread + MetalContext sections)
      to state the tool docks are proxy-backed; update `AGENTS.md`/`CLAUDE.md` Qt
      notes.
- [x] 6.2 `ruff check src/` clean; grep confirms zero GUI-thread live-renderer GPU
      calls in the five docks (all reads via cache/request, writes/GPU via post).
- [ ] 6.3 `openspec validate restore-render-thread-tool-docks --strict` passes;
      archive on completion.

## 6. Docs + close-out
- [ ] 6.1 Update `docs/Architecture.md` (Qt/render-thread + MetalContext sections)
      to state the tool docks are proxy-backed; update `AGENTS.md`/`CLAUDE.md` Qt
      notes.
- [ ] 6.2 `ruff check src/` clean; grep confirms zero GUI-thread live-renderer GPU
      calls in the five docks.
- [ ] 6.3 `openspec validate restore-render-thread-tool-docks --strict` passes;
      archive on completion.
