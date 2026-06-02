## 1. Shared CLI helper

- [x] 1.1 Add `src/skinny/cli_common.py` with `WALK_CHOICES = ("fused", "eye", "eye_light")`, `resolve_walk(value)` (normalizes case/whitespace, maps `megakernel`→`fused`, raises `ValueError` on unknown), and `add_render_flags(parser, *, integrator=True, execution=True, walk=True)` defining `--integrator {path,bdpt}` (default `path`), `--execution-mode {megakernel,wavefront}` (default env `SKINNY_EXECUTION_MODE` else `megakernel`), `--bdpt-walk` (free string, default env `SKINNY_BDPT_WALK` else `fused`, `metavar="{fused,eye,eye_light}"`).
- [x] 1.2 Carry over the existing help text (execution-mode session-fixed / Metal-pin note; bdpt-walk "wavefront+bdpt only, identical image, dispatch-overhead vs occupancy" note) onto the shared definitions.

## 2. Rename walk `megakernel` → `fused`

- [x] 2.1 `vk_wavefront.py`: `WALK_MODES = ("fused", "eye", "eye_light")`; default param `walk_mode="fused"`; the `walk_mode == "megakernel"` branches (entries selection + `_build_walk_dispatch`) → `"fused"`; update the explanatory comment (`fused — one wfBdptWalk kernel ...`).
- [x] 2.2 `renderer.py:~936`: accept `{fused,eye,eye_light}`, map `megakernel`→`fused` (reuse `cli_common.resolve_walk`); constructor default `bdpt_walk="fused"`. Confirm the renderer still passes the resolved string to `WavefrontBdptPass(walk_mode=...)`.
- [x] 2.3 Grep for remaining `"megakernel"` walk literals (tests, helpers) and repoint to `"fused"`; the alias keeps any missed external caller working.

## 3. Wire front-ends to the shared helper

- [x] 3.1 `app.py` (`skinny`): replace the inline `--execution-mode` / `--bdpt-walk` blocks with `add_render_flags(parser)`; normalize via `resolve_walk`; set the renderer's initial `integrator_index` from `--integrator` after construction (path=0, bdpt=1).
- [x] 3.2 `ui/qt/app.py` (`skinny-gui`): same — `add_render_flags`, drop the inline blocks, pass `--integrator` through `MainWindow` to the initial `integrator_index`.
- [x] 3.3 `web_app.py` (`skinny-web`): same — `add_render_flags`, set the per-session renderer's initial `integrator_index` from `--integrator` alongside `_EXECUTION_MODE` / `_BDPT_WALK`.
- [x] 3.4 `headless.py` (`skinny-render`): call `add_render_flags(parser)` (keeps the existing `--integrator`, now from the shared def); add `execution_mode` + `bdpt_walk` params to `HeadlessRenderer.__init__` and pass them into `Renderer(...)` at `headless.py:141`; thread `args.execution_mode` / `resolve_walk(args.bdpt_walk)` from `main()` through `RenderOptions`/the constructor.

## 4. Tests

- [x] 4.1 `cli_common` unit tests: default resolution (`fused`); `SKINNY_BDPT_WALK` / `SKINNY_EXECUTION_MODE` env fallback; `resolve_walk("megakernel") == "fused"`; `resolve_walk` raises on a bogus value; integrator/execution/walk choices + defaults.
- [x] 4.2 Front-end parity test: build each of the four parsers and assert all expose `--integrator`, `--execution-mode`, `--bdpt-walk` with identical choices/defaults (mirrors the cross-front-end consistency rule).
- [x] 4.3 Headless smoke: `skinny-render ... --execution-mode wavefront` renders a frame (reuse the headless harness), and `--bdpt-walk megakernel` resolves to `fused` without error.
- [x] 4.4 Repoint existing wavefront-bdpt tests that pass `walk_mode="megakernel"` to `"fused"`; add one asserting the `megakernel` alias still constructs the fused pass (same kernels as `fused`).

## 5. Docs + verification

- [x] 5.1 Update CLAUDE.md / AGENTS.md / README flag references + any `SKINNY_BDPT_WALK` mention to `fused` (note `megakernel` accepted as a deprecated alias).
- [x] 5.2 `ruff check src/` clean (no new errors vs `main`); `pytest` green.
- [x] 5.3 Confirm no shader recompile: `wfBdptWalk` SPIR-V + `main_pass.spv` byte-unchanged in git (rename is Python-side only).
