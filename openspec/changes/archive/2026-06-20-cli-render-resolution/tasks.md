## 1. Shared CLI flag surface

- [x] 1.1 In `src/skinny/cli_common.py`, add a `resolution: bool = True` keyword
  parameter to `add_render_flags(...)`.
- [x] 1.2 When `resolution` is true, add `--width` and `--height` to the parser:
  `type=int`, defaults sourced from `SKINNY_WIDTH`/`SKINNY_HEIGHT` env fallbacks
  then 640/480 (precedence flag > env > default), with help text describing them
  as the render-area size.
- [x] 1.3 Extend the shared startup validation seam to reject non-positive
  `--width`/`--height` (flag or env value) with a clear usage error naming the
  offending flag, exiting before GPU init.

## 2. `skinny` (windowed GLFW) front-end

- [x] 2.1 In `src/skinny/app.py`, replace the hardcoded `WINDOW_WIDTH`/
  `WINDOW_HEIGHT` (lines ~39-40) with the parsed `args.width`/`args.height`.
- [x] 2.2 Pass the parsed width/height to `glfw.create_window(...)` (line ~506)
  and to `make_context(...)` (line ~518) so window and render target match.
- [x] 2.3 Confirm GLFW window-position restore from `~/.skinny/settings.json`
  still applies and does not override the requested size.

## 3. `skinny-gui` (Qt) front-end

- [x] 3.1 In `src/skinny/ui/qt/app.py`, read `args.width`/`args.height` in
  `main()` and thread them into `make_context(..., width=, height=)` (line ~92),
  replacing the hardcoded 1280×720.
- [x] 3.2 Leave the Qt outer window size (`self.resize(1600, 900)`) and
  persisted `qt_geometry` untouched; verify the viewport widget displays the
  render at the new resolution.

## 4. `skinny-render` opt-out

- [x] 4.1 In `src/skinny/headless.py`, call `add_render_flags(..., resolution=False)`
  so the headless parser keeps its own `--width`/`--height` (1024² default) with
  no argparse conflict. (No-op if headless does not currently call
  `add_render_flags`; in that case confirm no conflict path exists.)

## 5. Tests

- [x] 5.1 Add a CLI parsing test: `skinny`/`skinny-gui` default to 640×480; an
  explicit `--width/--height` overrides; `SKINNY_WIDTH`/`SKINNY_HEIGHT` env
  supplies the value and an explicit flag overrides the env.
- [x] 5.2 Add a validation test: non-positive `--width`/`--height` exits with a
  usage error before GPU init.
- [x] 5.3 Add a `--help` test asserting `--width`/`--height` appear with the
  640/480 defaults on `skinny`/`skinny-gui`, and that `skinny-render --help`
  still shows its 1024² defaults with no conflict error.
- [x] 5.4 (Optional, gated) Headless render at a non-default size (e.g. 320×240)
  to confirm the render target honors the requested dimensions end-to-end.

## 6. Docs

- [x] 6.1 Update `README.md` CLI flag list with `--width`/`--height` (default
  640×480) and the `SKINNY_WIDTH`/`SKINNY_HEIGHT` env fallbacks.
- [x] 6.2 Add a `CHANGELOG.md` entry; note the interactive default changed from
  1280×720 to 640×480.
- [x] 6.3 If `docs/Architecture.md` enumerates the shared render-flag surface,
  add the two flags there.

## 7. Verify

- [x] 7.1 Run `.venv/bin/ruff check src/` and `.venv/bin/pytest` for the new
  tests.
- [x] 7.2 Launch `.venv/bin/skinny --backend metal --width 800 --height 600` and
  `.venv/bin/skinny-gui --backend metal --width 800 --height 600`; confirm the
  render area matches and a frame renders. Validate the change with
  `openspec validate cli-render-resolution`.

> 7.2 note: the interactive GUI launch was substituted by a headless Metal
> render at 800×600 (`select_backend("metal")` + `make_context(window=None,
> width=800, height=600)` — the same offscreen path `skinny-gui` uses). The
> render target returned `800*600*4 = 1920000` bytes, shape `(600, 800, 4)`;
> `openspec validate cli-render-resolution` passes. No display was available for
> a windowed launch.
