## Why

The interactive front-ends hardcode their render-area size (`skinny` opens a
1280×720 GLFW window; `skinny-gui` renders offscreen at 1280×720), so there is
no way to launch at a chosen resolution without editing source. A `--width` /
`--height` flag pair lets the user size the render area at launch — useful for
quick low-cost previews, fixed-aspect framing, and matching a target output
size — with a small, fast 640×480 default.

## What Changes

- Add `--width` and `--height` CLI flags to the shared render-flag surface so
  `skinny` and `skinny-gui` accept them from one definition (with `SKINNY_WIDTH`
  / `SKINNY_HEIGHT` environment fallbacks, matching the existing flag pattern).
- Default both to **640×480** when neither flag nor env var is set.
- `skinny` (windowed GLFW): size the GLFW window **and** the swapchain/compute
  render target to the requested width/height instead of the hardcoded
  `WINDOW_WIDTH`/`WINDOW_HEIGHT`.
- `skinny-gui` (Qt): size the offscreen render target (the `make_context`
  width/height — the pixels the renderer computes, i.e. the render area) to the
  requested values; the surrounding Qt window/dock layout keeps its own size.
- Validate the values at startup (positive integers) and exit with a clear
  usage error on a non-positive size, reusing the shared validation seam.
- `skinny-render` (headless) keeps its **own** existing `--width`/`--height`
  (default 1024²) — it already exposes them; the shared surface opts out there
  to avoid an argparse conflict, so its behavior is unchanged.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `render-cli`: add render-area resolution (`--width`/`--height`) to the shared
  render-flag surface, with a 640×480 default, exposed by the interactive
  front-ends and applied to the render area on `skinny` and `skinny-gui`.

## Impact

- **Code**: `src/skinny/cli_common.py` (`add_render_flags` — add the two flags +
  env fallbacks + validation); `src/skinny/app.py` (`skinny` — replace
  `WINDOW_WIDTH`/`WINDOW_HEIGHT` constants with parsed values for the GLFW window
  and `make_context`); `src/skinny/ui/qt/app.py` (`skinny-gui` — thread
  width/height into `make_context`). `src/skinny/headless.py` opts out of the
  shared flags to keep its own `--width`/`--height`.
- **Behavior**: default interactive render area changes from 1280×720 to
  640×480. No shader, descriptor-binding, or GPU-backend changes; both Vulkan
  and Metal `make_context` already accept width/height.
- **Docs**: `README.md` (CLI flag list), `docs/Architecture.md` if the flag
  surface is enumerated there.
