## Context

The render-selection CLI flags live in one shared definition,
`cli_common.add_render_flags(parser, ...)`, which every front-end calls so the
flags cannot drift (see the `render-cli` spec). Render-area size is currently
**not** one of those flags:

- `skinny` (`app.py`) hardcodes `WINDOW_WIDTH = 1280`, `WINDOW_HEIGHT = 720`
  (`app.py:39-40`) and passes them to `glfw.create_window(...)` (`app.py:506`)
  and to `make_context(...)` (`app.py:~518`).
- `skinny-gui` (`ui/qt/app.py`) hardcodes the offscreen render size in
  `make_context(..., width=1280, height=720)` (`ui/qt/app.py:92`); the Qt
  outer window has its own `self.resize(1600, 900)` (`ui/qt/app.py:78`) and
  persists its frame via `qt_geometry` — that is the window chrome, separate
  from the render area.
- `skinny-render` (`headless.py`) **already** defines its own `--width` /
  `--height` (default 1024², `headless.py:338-339`) for offline output size.

Both backends already accept a size: `make_context(backend, window=None,
width=, height=, ...)` (`backend_select.py`) forwards to
`VulkanContext`/`MetalContext`, which use width/height as the swapchain extent
(windowed) or the compute target size (headless). So the GPU plumbing needs no
change — only the CLI surface and the two front-end call sites.

## Goals / Non-Goals

**Goals:**
- One shared definition of `--width` / `--height` (with `SKINNY_WIDTH` /
  `SKINNY_HEIGHT` env fallbacks) so `skinny` and `skinny-gui` parse them
  identically and cannot drift.
- Default render area **640×480** when neither flag nor env var is set.
- `skinny`: the requested size drives both the GLFW window and the render
  target.
- `skinny-gui`: the requested size drives the offscreen render area (the pixels
  the renderer computes).
- Clear startup error on a non-positive size, via the shared validation seam.

**Non-Goals:**
- Changing `skinny-render`'s existing `--width`/`--height` or its 1024² default.
- Runtime resizing of the render area after launch (the flags set the launch
  size only; live resize is out of scope).
- Persisting the chosen size to `~/.skinny/settings.json`. Render-area size is
  a fresh axis (current settings persist GLFW window *position* and the Qt
  *frame* geometry, not the render resolution), so there is no precedence
  conflict and nothing new to persist.
- Resizing the `skinny-gui` Qt outer window / dock layout — only the render
  area changes.

## Decisions

### Decision: Add the flags to the shared `add_render_flags`, gated by a `resolution=` parameter

Add `--width` and `--height` inside `cli_common.add_render_flags`, default
640×480, with `SKINNY_WIDTH` / `SKINNY_HEIGHT` env fallbacks resolved the same
way the other flags resolve their env fallbacks. Gate them behind a new
keyword parameter `resolution: bool = True` — mirroring the existing
`proposals=` opt-out that `skinny-gui` already uses — so a front-end that owns
its own size definition can opt out.

- `skinny`, `skinny-gui` (and `skinny-web`) call `add_render_flags(...)` with
  `resolution` left at its default `True` → they get the shared flags.
- `skinny-render` calls `add_render_flags(..., resolution=False)` and keeps its
  own `--width`/`--height` (1024² default, offline-output semantics) → **no
  argparse "conflicting option string" error**.

*Alternative considered — define the flags separately in `app.py` and
`ui/qt/app.py`:* rejected; it reintroduces exactly the drift the `render-cli`
shared surface exists to prevent (two defaults, two env-var spellings to keep
in sync).

*Alternative considered — add to the shared surface unconditionally and remove
headless's own flags:* rejected; headless deliberately defaults to a square
1024² offline size with different semantics (final image, not interactive
window). Folding them together would silently change the headless default and
couple two unrelated concerns.

### Decision: `--width`/`--height` mean "render-area pixel dimensions", not "outer window size"

For `skinny` the window *is* the render area, so the flag sizes the GLFW window
and the swapchain/compute target together. For `skinny-gui` the renderer draws
offscreen and Qt blits the result into a viewport widget; the flag sizes that
offscreen render target (`make_context` width/height — the pixels actually
computed). The Qt outer window keeps `self.resize(1600, 900)` and its persisted
`qt_geometry`; the viewport widget displays the rendered image within that
window. This matches the request ("set the size of the render area") and keeps
the Qt chrome/dock layout independent of the render resolution.

*Alternative considered — also resize the Qt window to fit the render area:*
rejected as scope creep and a worse default (a 640×480 outer window would clip
the dock panels). Decoupling render area from window chrome is the cleaner
model.

### Decision: Validate width/height in the shared validation seam

The front-ends already run a shared startup validation that rejects impossible
flag combinations (`render-cli` → "Reject impossible render-flag combinations
at startup"). Extend that seam to reject a non-positive `--width`/`--height`
(or env value) with a clear usage error naming the offending flag, exiting
before GPU init — consistent with the other startup rejections.

## Risks / Trade-offs

- **Default change 1280×720 → 640×480 surprises existing users** → Intentional
  per the request; documented in `README.md`/`CHANGELOG.md`. Anyone wanting the
  old size passes `--width 1280 --height 720` (or sets the env vars). Render
  area is not persisted, so no stored state silently overrides the new default.
- **Env-var fallback collides with a shell that exports `SKINNY_WIDTH`** →
  Same, intended precedence as the other flags (explicit flag > env > default);
  documented alongside `SKINNY_BACKEND` etc.
- **Non-square / tiny sizes stress the GPU path** (e.g. workgroup tiling,
  swapchain min extent) → width/height already flow through `make_context` for
  headless renders at arbitrary sizes, so the path is exercised; the validation
  only guards against non-positive values, leaving extreme-but-valid sizes to
  the existing context code.
- **Aspect-ratio / camera framing shifts with a non-16:9 default** → the camera
  projection already derives its aspect from the render dimensions; 640×480
  (4:3) simply reframes, no code change needed.

## Open Questions

- None blocking. (`skinny-web` inherits the shared flags for free; wiring its
  session render size to them is optional and can be a follow-up, since the
  request scopes the behavior to `skinny` and `skinny-gui`.)
