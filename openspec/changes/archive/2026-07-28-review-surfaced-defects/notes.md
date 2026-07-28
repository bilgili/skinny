# Implementation notes: review-surfaced-defects

Records the evidence and the decisions the tasks ask to be written down.

## 1.1 — Reproduction of the web render-thread kill

Hostless, against a stub renderer: the accumulation-state provider is pure
Python, so the failure needs no GPU. One thread runs the `mtlx_overrides`
provider in a loop (the render thread's `_current_state_hash`), another inserts
and removes `mtlx.*` keys (the Panel callback's `_set_nested` on the live
renderer, which a slider drag performs once per edited field plus its
`_GANGED_MTLX_FIELDS` siblings). It raises within a few thousand iterations:

```
Traceback (most recent call last):
  File "repro_mtlx.py", line 44, in <module>
    PROVIDER.extractor(r)
  File "src/skinny/params.py", line 247, in <lambda>
    lambda r: tuple(sorted(
                    ^^^^^^^
  File "src/skinny/params.py", line 248, in <genexpr>
    (k, _hashable_value(v)) for k, v in r.mtlx_overrides.items()
                                        ^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: dictionary keys changed during iteration
```

`_render_loop` had no `try` around its body, so this retired the render thread
while `_running` stayed `True` — the session's video stream froze for good with
nothing surfaced to the browser.

Regression cover: `tests/test_web_render_loop.py` runs the same two threads
through the fixed path (`set_param` posts; the loop applies between frames) and
asserts it does not raise.

## 1.2 — A second door to the same defect, found while wiring the first

Marshalling the sidebar's parameter path is not enough on its own.
`usd_loader.resolve_control_binding` — which builds the get/set closures for
USD-declared `skinny:ui:*` controls — called `_set_nested(renderer, path, v)`
directly. `_set_nested` resolves the intermediate objects itself, so given a
proxy it reaches *through* it:

- on the web pass-through proxy, `mtlx:<field>` inserts straight into the live
  `renderer.mtlx_overrides` from the Bokeh thread — defect 1 exactly, reached
  through a USD control instead of a slider;
- on `QtRendererProxy`, the same write lands in the proxy's own
  `mtlx_overrides` mirror and posts nothing — defect 3's shape.

Fixed at the owner rather than at the two call sites: `params.set_param_value`
is now the single seam for "write one renderer parameter by path" (prefer the
target's `set_path`, else `_set_nested`). `build_app_ui`, `presets.apply_preset`
and `resolve_control_binding` all call it. Pinned by
`test_usd_declared_controls_write_through_the_marshalling_seam` and
`test_preset_application_goes_through_the_marshalling_seam`.

## Codex pre-merge review — round 2

The required pre-merge review returned NEEDS-REVISION. Every blocking finding
was re-verified against the code before acting on it; all of them held.

**1. The fence made the render-loop guard useless (blocking, fixed).** Both
Vulkan paths reset the frame fence right after waiting on it, ~100–190 lines
before the submit that signals it. Anything raising in between — the uniform
pack, the HUD rasterise, the wavefront record — leaves the fence reset and
unsignaled, so the guard's *retry* blocks forever in `vkWaitForFences(…, 2**64-1)`
and `MAX_RENDER_FAILURES` never fires. The permanent freeze this change exists
to remove, reintroduced one layer down. The reset now sits immediately before
its `vkQueueSubmit` in both paths, so an exception in between leaves the fence
*signaled* and the next iteration proceeds. Pinned by
`test_fence_is_reset_only_immediately_before_its_submit`.

**2. Presets bypassed the new seam (blocking, fixed).** `presets.apply_preset`
looped raw `_set_nested` over every value, `mtlx.*` included — the third door to
defect 1, reached from the `preset_index` combo this change already touched. Now
routes through `params.set_param_value` like every other parameter write.

**3. The resize still ran on the caller's thread (blocking, fixed).** Task 2.1
prescribed routing the callback to the session's existing locked `resize`, and I
followed it — but the spec scenario says the resize is *applied on the owning
thread*, and a lock only serialises it. `SkinnySession.resize` now posts the
whole compound (renderer resize → encoder rebuild → frame drain → WebSocket
notify) with `post_with_reply` and waits; `_apply_resize` runs on the render
thread and must not re-take `_lock`, which the drain already holds.

**4. `MarshalledRenderer` passed `apply_*` straight through (blocking, fixed).**
The `material:` USD-declared control reached the live `apply_material_override`
off-thread while the class docstring claimed every write becomes a command. The
proxy now marshals that verb and **refuses** any other `apply_*` with an
`AttributeError` naming the fix, so the hazard is unrepresentable rather than
silent — the accept-then-drop lesson from `shader-variant-key-module`. The full
verb set is still `renderer-command-interface`; what changed is that the gap now
fails loudly instead of racing.

**6. Giving up was invisible (fixed).** `_running = False` with nothing reading
`_render_error` reads to the browser as a slow frame. A terminal failure now
sends a type=5 render-failed frame, closes the socket, and settles every pending
reply future so a caller blocked in `resize` does not wait out its timeout.

**7. The proxy lived in the wrong module (fixed).** `qt-render-threading`
requires the queue *and the proxy built on it* to live in a GUI-toolkit-free
module. `MarshalledRenderer` moved from `web_app.py` (which imports Panel and
Tornado) to `render_session.py`, taking `(queue, renderer_getter)` instead of a
session so it is front-end-neutral in fact and not just in placement.

**8. Docs pointed at the unsafe seam (fixed).** `docs/PythonAPI.md` recommended
`params._set_nested` for generic writes; it now recommends `set_param_value` and
says when `_set_nested` is still correct.

**5. Other web entry points bypass the queue (pre-existing, recorded).** Camera
input (`web_app.py`), model/HDR loading, the Panel scene-graph and Camera Debug
windows acting on `session.renderer` and `session.ctx` directly. All predate this
change and none is a *parameter* write; they belong to
`renderer-command-interface`, whose whole subject is giving the web front-end the
verb set. Also recorded there: `resolve_control_binding`'s `usd:` kind calls
`attr.Set(v)` on the stage from the caller's thread — a stage mutation, not
renderer state, and inert on Qt where `_usd_stage` is a sentinel.

**Gate sufficiency.** Codex's objection is fair: the loop test replaces
`_render_iteration`, so it could not have caught the fence hang, and nothing
exercised presets or the `material:` control through a proxy. Those three now
have tests.

### Two things I checked myself, because the re-review did not answer them

**The windowed-only risk in F1.** Moving the fence reset past the swapchain
acquire raises the question of an exception between acquire and submit: the
`image_available` semaphore would be signalled with nothing waiting on it, and
re-acquiring into it next iteration is undefined behaviour. It is **not
reachable**: nothing retries `render()`. The Qt worker's `except` is outside its
loop (`ui/qt/viewport.py:246` — it exits and tears down) and `app.py:678` has no
guard at all. The retry that makes the fence ordering matter exists only on the
web path, which calls `render_headless()` and never acquires a swapchain image.
The reset move in `render()` is therefore defensive, not load-bearing — and the
acquire-without-submit leak is a pre-existing hazard that would become live the
day someone adds a retry to a windowed loop.

**The refusal set in F4 is exactly right, and the hole was bigger than
reported.** `Renderer` has nine `apply_*` verbs; the shared tree reaches exactly
one — `apply_material_override`, from `_add_material_block`,
`_add_graph_uniform` and `_reset_graph_uniforms` (nine call sites in
`build_app_ui.py`) plus `resolve_control_binding`'s `material:` kind. So every
material colour-picker and graph-uniform slider in the **web sidebar** was
writing the live renderer off-thread, not just the USD-declared control codex
named. Marshalling that one verb covers all ten sites; refusing the other eight
breaks nothing, because they are reached only from the Qt/Panel docks, which act
on `session.renderer` or the Qt proxy directly and never through the tree.
`test_shared_tree_reaches_only_marshalled_renderer_verbs` pins the boundary: if
a second verb is ever wired into the tree, it fails instead of silently running
on the caller's thread.

### Review status, stated plainly

Round 1 completed and returned NEEDS-REVISION with eight findings; all are
addressed above. Round 2 is **partial**: the re-review carrying the targeted
questions stalled without producing a verdict, and a plain
`codex review --base main` over the same diff completed with *"No actionable
defects introduced by the diff were found."* The two questions codex never
answered — the windowed-only fence risk and the `apply_*` enumeration — are the
two I answered myself above. The resize deadlock trace remains reviewed only by
me: `resize` posts and blocks on the future, `run_pending` executes it inside
the render lock the drain already holds, and `_apply_resize` takes no lock, so
there is no second acquisition to deadlock on; a dead render thread settles the
future through `_notify_render_failed` rather than letting it time out.

## 1.5 — Slider-drag cost, direct write vs marshalled post

20 000 drag samples on `mtlx.layer_top_melanin`, stub renderer:

| path | cost |
|------|------|
| direct `_set_nested` (pre-fix, unsynchronised) | 0.145 µs / write |
| posted + coalesced (post-fix) | 0.363 µs / write |
| render-thread apply for the whole drag | 4.75 µs total |

Queue depth after all 20 000 samples is **1** — last-write-wins coalescing per
`param:<path>` means a drag never grows the queue, and the render thread applies
one write instead of 20 000. At a slider's real event rate (~10²/s) the extra
0.2 µs per event is not measurable. The posted path is no worse.

## 2.2 — `_add_resolution`'s direct-renderer fallback now raises

Call-site sweep: `build_main_ui` has exactly two production callers —
`ui/qt/app.py` (which already supplied
`resize_render_target=self.viewport.request_resize`) and `web_app.py` (which now
supplies `session.resize`). With both front-ends wired, the fallback has no
remaining legitimate user, and its only effect was to convert a missing wire
into an unsynchronised `renderer.resize` from the caller's thread. It now raises
`ValueError` at tree-build time, so the next front-end fails loudly at startup
instead of silently tearing frames.

`tests/test_ui_spec.py` was the one other caller: eleven tests built the tree
with no callbacks at all, and the change turned them red — which is the guard
working, not a regression. They now go through a `_callbacks()` helper that
supplies a stub resize, and a new test pins the refusal itself.

## 3.2 — Audit of the proxy's other locally-held renderer objects

| object | written from a control? | marshalled? |
|--------|------------------------|-------------|
| `clock` | yes — Play / Time / FPS | **no** (the defect); now via `set_clock_state` |
| `film` | yes — `film.iso`, `film.exposure_time` | yes: `set_path` updates the mirror under `_suppress_posts`, then posts the same `_set_nested` |
| `scene_graph` | no — replaced wholesale by `apply_scene_state` from a worker snapshot; edits route through the `apply_*` verbs | yes |
| `camera` | no — same, snapshot-replaced; edits go through `apply_camera_param` | yes |

The clock was the only unposted one.

## 4.1 — Panel dispatcher copy vs the shared `apply_scene_property`

Diffed per reference kind before deleting the copy.

| behaviour | Panel copy | shared | verdict |
|-----------|-----------|--------|---------|
| `light_env` (dome light) | **not handled** — falls through every branch, returns `None` | routes to `apply_light_override("env", …)` | **accidental** — the reported defect |
| failure reporting | always returns `None` | returns a reason string | **accidental** — the failure was not even reportable |
| `ref is None` | silently returns | resolves a material ancestor, else returns a reason | accidental |
| material ancestor lookup | `_find_material_ancestor` reads `renderer.scene_graph` | `find_material_ref(graph, node)`, `graph` defaulting to `renderer.scene_graph` | identical |
| fan-out guard (bool / vec3) | re-inlined at each call site, gated on `r.kind == "material"` | checked first inside the shared function, resolving the material ancestor when the node has no ref | shared is strictly more permissive — intended |
| `texture_file` / `lens_file` | no branch | routes to `apply_dome_light_texture` / `apply_camera_lens_file` | Panel builds no editable widget for these types today; gaining the route is harmless and correct |
| camera vec3 with an unknown axis | silently returns | returns a reason | intended |
| everything else (material override, fan-out overrides, `light_dir`/`light_sphere`, camera params, TRS recompose, stage-vs-runtime transform) | identical | identical | — |

No difference was found that is not an improvement, so the copy was deleted
rather than patched.

## 5.1 / 5.3 — Camera Debug key maps

Four maps tabulated. Two of them (GLFW `DebugViewport._on_key` and the Qt
**Camera Debug dock**) are the *same control surface* in two front-ends and so
must reconcile. The Qt **main viewport** and the GLFW **main app** are a
different surface — the render viewport — and are out of scope here. The web
front-end presents no Camera Debug key map at all.

| action | GLFW debug viewport | Qt debug dock | verdict |
|--------|--------------------|---------------|---------|
| camera mode | `C` | `C` | ✓ |
| reset camera | `F` | `F` | ✓ |
| mesh wires | `M` | `M` | ✓ |
| grid | `G` | `G` | ✓ |
| focus plane | `P` | `P` | ✓ |
| render area | `I` | `I` | ✓ |
| ortho | `O` | `O` | ✓ |
| **depth-of-field planes** | `D` | **absent** | **fixed** — `D` now bound |
| view top / back / left | `T` / `B` / `L` | `T` / `B` / `L` | ✓ |
| HUD | `Space` | `Space` | ✓ |
| movement (held) | `W A S D Q E` | `W A S D Q E` | ✓ |
| close | `Escape` | — | **recorded divergence** |

`D` was the only real drop. Both front-ends already had two independent
channels — a *held* set for free-camera movement and a *press* table for
toggles — and GLFW serves `D` from both. The Qt dock returned early for every
movement key, so the press channel never saw it. The fix serves both channels,
and filters `isAutoRepeat()` so holding `D` to strafe cannot flip the toggle
(GLFW filters `glfw.REPEAT` for the same reason).

`Escape` is recorded, not fixed: the GLFW viewport owns an OS window it can
close; the Qt surface is a `QDockWidget` closed by its own title bar and the
View menu, so a key binding would duplicate a chrome affordance.

The Qt map moved from a control-flow chain inside `keyPressEvent` to
module-level `MOVEMENT_KEYS` / `PRESS_ACTIONS` tables, so the reconciled set is
readable and assertable without a Qt application. Pinned by
`tests/test_qt_debug_viewport_dock.py`, which also asserts the transcribed GLFW
map against `DebugViewport._on_key`'s source — a change to either map without
the other now fails.

## 8 — Gates: what was actually run

| gate | result |
|------|--------|
| `ruff check src/` | clean (explicit target — the repo-root `.gitignore` is `*`, so a bare run passes vacuously) |
| hostless `pytest -m "not gpu"` | **2453 passed, 7 failed** (round 2) — byte-identical failure set to the `main` baseline (2432 passed, the same 7). All 7 pre-date this change: six `test_corpus_scene_imports_cleanly_mtlx[*]` and `test_mcp_tool_schemas::test_all_ten_tools_are_advertised`. +15 new tests, all passing. |
| parity matrix dual gate (Metal) | `pytest tests/pbrt/test_parity.py -k matrix` → **20 passed, 1 skipped, 1 xfailed**; re-run after the round-2 fixes with the identical result. pbrt-truth and self-consistency both unchanged; no baseline moved. |
| GPU smoke — Metal offscreen | HUD now composites and is current (was never filled on either Metal path); a render with no HUD text is byte-unchanged. |
| GPU smoke — Vulkan offscreen + windowed→screenshot | Also the two edited fence sites; re-run after the fence move, offscreen output **bit-identical** (sha `d51b6b00e7080b3c`) and a second windowed→screenshot round-trip succeeds, so the fence is left usable by both paths. The only validation output is the pre-existing `vkDestroyDevice` object-tracking leak at teardown — no submit/fence VUIDs. The case the deleted binding-1 rewrite claimed to protect. Screenshot after 24 windowed `render()` frames is a correct image (27 347/27 648 non-black); the offscreen/windowed difference is uniform MC speckle, max channel 36/255 — not a wrong-target binding, which would show as a torn or blank region. |

A note on the sweep command: `pyproject.toml` sets no `addopts`, so a bare
`pytest` **does** collect the gpu-marked tests. The hostless gate is
`pytest -m "not gpu"`; 505 tests are deselected by it.

### The three manual checks

Verified functionally against a **live `SkinnySession` with its render thread
running**, rather than by clicking in a browser or the Qt window. Each drives
the same function the widget setter calls:

- **2.3 (resolution mid-render)** — `ResolutionPicker.on_apply(320, 240)` through
  the tree's own callback returned `(320, 240)`; renderer *and* encoder both
  ended at 320×240 (no decoder-config mismatch), the render thread stayed alive,
  and the next frame off the queue was well-formed.
- **3.4 (animation transport)** — `_set_clock_value(proxy, "playing", True)` and
  `"playback_fps", 48.0`: both landed on the live clock **and** the time code
  advanced. Before the fix this wrote a proxy-local mirror and posted nothing.
- **1.x (slider drag)** — 400 posted `mtlx.layer_top_melanin` samples against the
  running loop: thread alive, no new render error, final value applied.
- **4.4 (dome-light edit)** — the Panel `intensity` widget on
  `/World/DomeLight` (`assets/first_mcp_scene.usda`), driven against a live
  session: `1.0 → 4.0` routed to `apply_light_override("env", 0, "intensity",
  4.0)`, returned no reason, moved `_usd_scene.environment.intensity` to 4.0,
  and the rendered frame brightened. Before the fix this fell through every
  branch of the Panel's copied dispatcher and returned `None`.

  **Scene selection matters here, and cost me a wrong conclusion first time.**
  `_add_light_props` only emits `intensity`/`exposure` when the attribute
  `HasAuthoredValue()`, so a dome that authors neither exposes no editable
  scalar at all — which is the case on `bunny.usda`, whose dome
  (`/World/light_infinite_edb0`) offers only the transform triple, `enabled`,
  and a `texture:file` that Panel renders read-only. That is a property of that
  scene, not of the route. Use a scene with an authored dome intensity.

## 7.3 — `prepare_usd_streaming`

Deleted, along with its `docs/PythonAPI.md` entry. It had zero call sites in
`src/`, `tests/` or `scripts/`: the streaming path reimplements the work inline,
so the published function was a second, untested implementation of it that no
caller could have been depending on.
