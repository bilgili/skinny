# Design — restore render-thread tool docks

## Context

Render-thread ownership (`qt-render-threading`) put `Renderer` + GPU context on
the worker. The GUI thread holds a `QtRendererProxy` (`render_session.py`):

- `__getattr__` serves mirrored `STATIC_PARAMS` values + choice lists.
- `__setattr__` / `set_path` write local mirror + `post()` the mutation.
- `post(cb, coalesce_key=)` — fire-and-forget worker command (last-write-wins
  coalescing).
- `request(cb) -> Future` — round-trip; worker runs `cb(renderer)`, completes the
  `Future`.
- `apply_snapshot(RendererStateSnapshot)` — GUI refreshes its mirror from an
  immutable worker snapshot each frame.

The Controls sidebar is the proven reference: `build_main_ui(self.renderer=proxy)`
— reads hit the mirror, writes go through `post()`. The five tool docks must be
brought onto the same three primitives.

## Two dock classes

The dependency audit splits the docks cleanly:

| Dock | Reads | Writes | Produces pixels/data on GPU? | Difficulty |
|------|-------|--------|------------------------------|------------|
| Python Material Editor | `scene_python_modules()` | recompile (delegated to `MaterialReloader`) | No | Easy |
| Scene Graph | `scene_graph`, `_scene_graph_version`, `_usd_scene`/`_usd_stage`, camera params | ~13 `apply_*`/`set_*`/`add`/`remove`/`save_edits` | No | Medium |
| BXDF Visualizer | material list, `mtlx_overrides`, `_material_version` | `request_bxdf_eval` / `request_bssrdf_eval` (async), scene pick | **Yes** (eval grid) | Hard |
| Material Graph | material list, `_mtlx_scene_materials`, `_mtlx_library` | topology edits → `_gen_scene_materials`+`_upload_graph_param_buffers`; env select | **Yes** (`render_material_preview`) | Hard |
| Camera Debug | owns a `DebugViewport(ctx, …)` | `render_embedded` under lock, camera/display state | **Yes** (2nd framebuffer) | Hard |

- **Data/command docks** (Python Material Editor, Scene Graph): reads →
  snapshot/`request()`, writes → `post()`. No pixel production. Standard proxy port.
- **GPU-producing docks** (BXDF, Material Graph, Camera Debug): the render thread
  must *produce* data (a numeric grid, a preview image, an RGBA8 frame) and hand
  it back to the GUI thread asynchronously.

## Decision D1 — every dock takes the proxy

All docks are constructed with `self.renderer` (the proxy), matching the sidebar.
No dock keeps a reference to the live `Renderer`. GUI-held locks (`render_lock`,
`main_lock`) that the old docks passed in are **removed** — mutual exclusion is now
provided structurally by the single-consumer command queue on the worker.

## Decision D2 — reads: snapshot for polled state, `request()` for one-shots

The docks currently poll on 200–500 ms timers. Rather than each dock reaching into
the live renderer on tick, the worker's per-frame `RendererStateSnapshot` is
extended with the fields the docks poll:

- `scene_graph` (immutable view/handle) + `scene_graph_version`
- `usd_scene_id` / `usd_stage_id` (identity, for swap detection)
- `materials` (id + display name + kind list) + `material_version`
- `python_modules` (names) + `active_python_module`
- camera param values the Scene Graph property panel shows

Docks read these from the cached snapshot on tick (no lock, no live access —
satisfies the "GUI reads immutable snapshots" requirement). Genuinely one-shot,
heavy, or interactive reads (e.g. fetching a full material's authored parameter
set when a node is selected) use `proxy.request(lambda r: …)` and fill the panel
when the `Future` resolves.

**Snapshot immutability caveat**: `scene_graph`/USD objects are renderer-owned and
mutable. The snapshot carries a *read-only projection* the dock renders from (names,
paths, versions, transforms) — not a live handle the dock writes through. Writes
always go via `post()` (D3). This preserves "SHALL NOT use the snapshot as a mutable
backdoor."

## Decision D3 — writes: `post()` with per-target coalesce keys

Every mutation becomes `proxy.post(lambda r: r.apply_x(...),
coalesce_key="…")`. Slider-rate edits (transform drags, material param scrubs)
coalesce on a `f"…:{path}:{name}"` key so a drag can't flood the queue; distinct
discrete actions (add/remove node, save edits) post uncoalesced to preserve
ordering. This mirrors how `QtRendererProxy.set_path` already coalesces
`param:{path}`.

## Decision D4 — GPU-producing docks: worker produces, GUI receives via signal

Introduce one shared pattern in `render_session.py`: a **result request** — the
GUI posts a callback that returns data; the worker runs it between frames and
delivers the result to the GUI thread. Two shapes, both already latent in the
queue:

1. **`Future`-based** (Material Graph preview, one-shot): `fut =
   proxy.request(lambda r: r.render_material_preview(mat_id, prim, size))`; the
   dock attaches a done-callback marshalled onto the GUI thread
   (`QTimer.singleShot(0, …)` or a `Signal(object)`), then sets the `QPixmap`.
   Debounced on the GUI side exactly as today.

2. **Callback/streaming** (BXDF `request_bxdf_eval(req, cb)`): the existing async
   API already takes a callback. Route it through the worker — `proxy.post(lambda
   r: r.request_bxdf_eval(req, worker_cb))` — where `worker_cb` emits a Qt
   `Signal(object)` the dock connects to, so the numpy grid lands on the GUI
   thread. Scene picks arm through `viewport.arm_scene_pick(cb)` (the viewport
   already lives on the GUI side and forwards a `post()` internally).

No dock ever calls a GPU method directly; every producer runs inside a worker
command.

## Decision D5 — Camera Debug viewport = a second render surface

The debug viewport is not an inspector; it owns a `DebugViewport(ctx, …)` with its
own framebuffer and ran a 33 ms render loop calling `render_embedded(renderer)`
under `main_lock`. Under render-thread ownership the clean model mirrors the main
viewport:

- The **worker** constructs and owns the `DebugViewport` (it needs the GPU `ctx`,
  which lives on the worker) when the dock is first opened, via a `post()`/`request`
  that returns a handle/first frame.
- Each worker frame, if the debug dock is open, the worker renders the embedded
  view and includes its RGBA8 buffer in the emitted snapshot (a `debug_frame`
  field), the same way `FrameSnapshot.pixels` carries the main frame.
- The **dock** is passive: it blits the latest `debug_frame` into a `QImage`,
  forwards camera/display input as `post()` commands (`orbit`, `pan`, `zoom`,
  `toggle_cam_mode`, `show_grid`, …), and posts open/resize/destroy lifecycle
  commands. No GUI-thread GPU call, no `main_lock`.

This is the largest single piece and is sequenced last so the other four can land
first.

## Decision D6 — sequencing (independently landable)

Implement and merge in ascending difficulty so each dock is a self-contained,
verifiable increment:

1. **Python Material Editor** (easy) — proves the proxy-construction + snapshot
   `python_modules` path with no GPU work.
2. **Scene Graph** (medium) — proves snapshot projection of a complex object +
   the full `post()` write set.
3. **BXDF Visualizer** (hard) — proves the D4 callback/`Signal` producer path.
4. **Material Graph** (hard) — proves D4 `Future` preview + topology-edit `post()`
   chain.
5. **Camera Debug** (hard) — proves D5 second-surface frame emission.

Each phase re-enables its `app.py` handler and its session-restore reopen; a phase
is done when the dock opens, functions, and does not touch the live renderer from
the GUI thread.

## Verification

- **Static**: `ruff check src/`; grep each ported dock for `self.renderer.` GPU
  calls on the GUI thread (must be zero — all through `post`/`request`).
- **Hostless**: unit-test the extended `RendererStateSnapshot`/`apply_snapshot`
  round-trip and the `request()` result-delivery helper without a GPU
  (`render_session.py` is deliberately Qt/GPU-free).
- **Interactive (manual, per phase)**: launch `skinny-gui --backend metal`
  (per machine convention), open each dock from View, exercise its controls while
  the renderer accumulates, confirm the GUI stays responsive and edits reset
  accumulation and take effect. Any produced image is shown back per the
  always-show-images rule.
- **Regression**: the parity-matrix and Metal-cleanup harnesses are unaffected
  (no shader/dispatch change); run `ruff` + the hostless `render_session` tests in
  CI.
