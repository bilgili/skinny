# Restore render-thread tool docks

## Why

`skinny-gui`'s **View** menu lists seven windows. Only **Render** and
**Controls** open — they are docks built at startup (`app.py:132`, `app.py:180`)
that the menu simply `show()`/`raise_()`. The other five —
**Scene Graph**, **Material Graph**, **Python Material Editor**,
**BXDF Visualizer**, **Camera Debug View** — do nothing but flash a status-bar
message:

> "… needs the snapshot-backed port for render-thread mode"

They were **stubbed out** by commit `382786f "Complete Qt render-thread
ownership"`. Before that commit each handler built a real dock (`SceneGraphDock`,
`BXDFDock`, `MaterialGraphDock`, `PythonMaterialEditorDock`, the embedded debug
viewport) from `ui/qt/windows/`. The render-thread migration moved `Renderer`
and GPU-context ownership onto the background render worker
(`qt-render-threading` spec); the GUI thread now talks to a `QtRendererProxy`
(`render_session.py`) via mirrored reads, `post()` commands, and `request()`
round-trips. The five dock classes still take the **live renderer** and touch it
directly on the GUI thread (`scene_graph.py` 35×, `material_graph.py` 21×,
`bxdf.py` 7×, `debug_viewport.py` 4×, `python_material_editor.py` 3×), so they
were disconnected rather than ported.

The dock code still exists, unused, in `ui/qt/windows/`. This is a **gap against
the existing `qt-render-threading` spec** — that spec already requires "dock
controls enqueue commands" and "inspector docks" to read snapshots. This change
closes the gap: it re-ports all five tool docks onto the proxy so they open and
function under render-thread ownership, restoring feature parity the GLFW app
never lost.

## What changes

- **All five docks receive `QtRendererProxy`, never the live renderer** — the
  same object the working Controls sidebar is built on (`app.py:167`). The
  `app.py` handlers (`_open_scene_graph`, `_open_material_graph`,
  `_open_python_material_editor`, `_open_bxdf`, `_toggle_debug_viewport`,
  `_ensure_debug_dock`) stop being stubs and build/show the real docks again,
  reusing the dock-holder fields already declared (`app.py:153-156`) and the
  session-restore reopen path (`app.py:442-450`).

- **Proxy reads** — params the proxy already mirrors work transparently. Non-param
  dock state that is cheap and polled (scene-graph object + `_scene_graph_version`,
  the material list, Python module names, USD scene/stage identity) is added to
  `RendererStateSnapshot` and refreshed via `apply_snapshot`; one-shot heavy reads
  use `proxy.request(cb) -> Future`.

- **Proxy writes** — every dock mutation (`renderer.apply_material_override`,
  `apply_instance_transform`, `set_gizmo_target`, `add_model`, `remove_node`,
  `save_edits`, `_gen_scene_materials`, env selection, …) routes through
  `proxy.post(lambda r: …, coalesce_key=…)`. GUI-held render locks
  (`render_lock`, `main_lock`) are removed — all GPU work now serializes on the
  worker.

- **GPU-producing docks** (Material Graph preview, BXDF/BSSRDF evaluation, the
  Camera Debug embedded viewport) gain a uniform "produce on the render thread,
  deliver to the GUI thread" path: the worker runs the GPU work and the result
  (preview pixels, eval grid, RGBA8 debug frame) is marshalled back to the dock
  via a Qt-signal / `Future` hop, never blitted from a GUI-thread GPU call. The
  Camera Debug viewport is treated like a second render surface: the worker owns
  the `DebugViewport` (it needs the GPU `ctx`) and emits frames; the dock is a
  passive `QImage` blit plus input→command forwarding.

- **Docs**: `docs/Architecture.md` (MetalContext/Qt section + the render-thread
  ownership description) and `AGENTS.md`/`CLAUDE.md` Qt notes updated to state the
  tool docks are proxy-backed.

Out of scope: no new dock features, no visual redesign — behavioural parity with
the pre-`382786f` docks under the new threading model only.

## Impact

- **Affected specs**: `qt-render-threading` (ADD explicit tool-dock requirements).
- **Affected code**: `src/skinny/ui/qt/app.py` (unstub handlers, snapshot wiring),
  `src/skinny/ui/qt/render_session.py` (extend `RendererStateSnapshot`, add a
  result-delivery helper for GPU-producing docks), and the five
  `src/skinny/ui/qt/windows/*.py` docks (accept the proxy; reads→snapshot/request,
  writes→post; drop GUI-held locks).
- **Risk**: the Camera Debug viewport is the highest-risk item (a second GPU
  surface under render-thread ownership); it is phased last and can ship
  independently. Each dock is independently landable in difficulty order.
