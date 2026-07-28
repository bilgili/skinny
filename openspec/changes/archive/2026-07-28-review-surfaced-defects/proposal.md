# Change: review-surfaced-defects

## Why

An adversarial review of the 2026-07-27 architecture proposals turned up defects
that have nothing to do with those refactors. They are recorded here so they get
fixed on their own schedule instead of riding along with a restructure. Every
claim below was re-verified against the tree at `86de263`.

**Three of them violate a requirement that already exists.**
`qt-render-threading` states: *"Any thread that is not the renderer's owning
thread — including a GUI thread and any in-process server thread — SHALL marshal
both reads and writes of renderer state through this queue."* The web front-end
does not.

1. **An `mtlx.*` slider can kill the web render thread permanently.**
   `params.py:245-250` builds the accumulation state hash by iterating
   `r.mtlx_overrides` in a generator; `renderer.py:10541` calls it every frame on
   the render thread. A slider insert adds a key — and 1–3 more through
   `_GANGED_MTLX_FIELDS` — so the iteration raises
   `RuntimeError: dictionary changed size during iteration`. `_render_loop`
   (`web_app.py:149`) has no `try` around its body, so the thread exits,
   `_running` stays `True`, and the session's video stream is frozen for good
   with no error surfaced to the browser.

2. **`renderer.resize` runs from the Tornado IOLoop thread with no lock.**
   `AppCallbacks.resize_render_target` exists (`build_app_ui.py:82`) and Qt
   supplies it (`ui/qt/app.py:179`), but the web front-end never sets it, so
   `_add_resolution` falls through to `renderer.resize(w, h)` directly
   (`build_app_ui.py:279-281`). That call destroys and recreates the offscreen
   image, readback buffer, accumulation image and HUD overlay
   (`renderer.py:10989-11009`) while the render thread may be inside
   `render_headless()` on those exact objects.

3. **The Qt animation transport is dead.** Play / Time / FPS write *through a
   sub-object* — `setattr(renderer.clock, "playing", …)` and
   `renderer.clock.set_normalized(…)` (`build_app_ui.py:196-207`). The proxy's
   `clock` is a local `PlaybackClock` installed with `object.__setattr__`
   (`render_session.py:290`), so the write mutates the mirror and posts nothing.
   The existing requirement is satisfied in the letter — no *top-level* attribute
   was written — while the control does nothing.

**Two more are front-end divergence:**

4. **Dome-light property edits are a silent no-op on web.** The shared dispatcher
   maps `light_env → "env"` (`ui/scene_edit_actions.py:91-95`), but
   `ui/panel/windows.py` does not call it — it reimplements the dispatcher, and
   its copy handles only `("light_dir", "light_sphere")` (`:434`). A `light_env`
   edit falls through every branch and returns `None`; the shared function would
   have returned a reason string, so the failure is not even reportable.

5. **The Qt debug dock is missing `Key_D` → depth-of-field planes.** The GLFW
   viewport has the identical WASD conflict and binds it anyway: `KEY_D` is
   polled for right-strafe (`debug_viewport.py:805`) *and* toggles the planes on
   press (`:2333-2335`). Qt already has the same two-channel structure but
   returns early for WASD keys (`ui/qt/windows/debug_viewport.py:344-347`).

**One is a rendered-output defect:**

6. **Headless renders composite a stale HUD.** `_build_hud_bytes` is uploaded
   only inside `render()` (`renderer.py:10609`), but `record_copy` runs in both
   `render()` (`:10648`) and `render_headless()` (`:10876`). A screenshot taken
   from a windowed session therefore composites whatever the *previous* frame
   uploaded.

**Three are stale claims — the code or docs assert something untrue:**

7. `render_headless` rewrites descriptor binding 1 every call
   (`renderer.py:10833-10846`) with a comment saying `render()` points that
   binding at the acquired swapchain image. It does not: all three `dstBinding=1`
   writes (`:4624`, `:10841`, `:11042`) target `_offscreen_output`, and `render()`
   blits instead (`:10611-10614`). The rewrite is vestigial and both comments are
   wrong.
8. `vk_wavefront.py:597` hard-codes `REC_VERTEX_STRIDE = 76`, a hand-typed copy
   of `wavefront_layout.py:107`'s value, which is *derived* from the Slang
   declaration under `reflection-owned-byte-layouts`.
9. `usd_loader.prepare_usd_streaming` (`:2853`) has zero call sites in `src/`,
   `tests/` or `scripts/` — the streaming path reimplements it inline — yet it is
   published in `docs/PythonAPI.md:541`. And `backend_select.py:16-19` still says
   `auto` resolves to Vulkan everywhere because "Metal shaded skin color is not
   yet at parity", contradicting `select_backend`'s own docstring at `:62-66`
   and CLAUDE.md.

Settings-file mutual erasure was found in the same pass and is **not** in scope
here — `session-settings-owner` owns it.

## What Changes

- Close the marshalling loopholes in `qt-render-threading` so defects 1–3 are
  ruled out by the requirement rather than by reviewer vigilance: writes through
  a sub-object count as renderer mutations; a front-end binds the shared control
  tree to a marshalling proxy, never to the live renderer; a front-end that
  offers a resolution control supplies the resize callback.
- Guard the render loop so a mutation-induced exception cannot silently retire a
  session's render thread.
- Make the Panel window call the shared scene-property dispatcher instead of its
  own copy, which fixes the `light_env` drop and restores failure reporting.
- Reconcile the Camera Debug key maps, or record each divergence with its reason.
- Upload the HUD on the same paths that copy it, so headless output composites
  the current frame.
- Delete the vestigial binding-1 rewrite and the two comments describing it.
- Source `REC_VERTEX_STRIDE` from the derived layout; delete or wire
  `prepare_usd_streaming`; correct the `backend_select` docstring.

## Capabilities

### Modified Capabilities

- `qt-render-threading`: the marshalling requirement is tightened to cover
  sub-object writes, control-tree binding, and the resize callback, and to
  require that a failed command cannot retire the owning thread.
- `usd-scene-editing-ui`: front-end consistency requires calling the shared
  dispatcher rather than reimplementing it, covering every light kind it routes,
  and reconciling or recording interaction bindings.

### New Capabilities

- `renderer-output-fidelity`: overlays composited into a frame are the current
  frame's, and no per-frame descriptor write exists without a target that
  differs from the one already bound.

## Impact

- Modified: `src/skinny/web_app.py` (resize callback, render-loop guard),
  `src/skinny/ui/panel/windows.py` (~90 lines deleted — the copied dispatcher),
  `src/skinny/render_session.py` (clock verbs on the proxy),
  `src/skinny/ui/build_app_ui.py` (animation setters), `src/skinny/renderer.py`
  (HUD upload, binding-1 removal), `src/skinny/vk_wavefront.py`,
  `src/skinny/usd_loader.py`, `src/skinny/backend_select.py`,
  `docs/PythonAPI.md`.
- **User-visible**: web parameter edits stop being able to freeze the stream;
  dome-light edits start working on web; the Qt animation transport starts
  working; headless screenshots stop showing a stale HUD.
- Each numbered defect is independently landable. 1–4 are the ones worth doing
  soon; 5–9 are hygiene.
- No relationship to the ten architecture proposals — this change may land
  before, after, or between any of them. It touches
  `ui/panel/windows.py` and `render_session.py`, which
  `renderer-command-interface` and `ui-spec-scene-properties` also touch; both of
  those are proposal-only, so sequencing is only a concern once either starts.
