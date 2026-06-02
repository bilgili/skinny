## Why

The render-selection flags grew per-front-end and drifted apart. Three axes now
exist — **integrator** (`path` / `bdpt`), **execution mode** (`megakernel` /
`wavefront`), and the **wavefront-bdpt subpath-build walk** — but no front-end
exposes all three, and the walk reuses the word `megakernel`, which already
names a *different* concept on the execution axis.

1. **The flag set is inconsistent across front-ends.** The interactive trio
   (`skinny`, `skinny-gui`, `skinny-web`) takes `--execution-mode` +
   `--bdpt-walk` but **not** `--integrator` (integrator is runtime-cycle only).
   Headless (`skinny-render`) is the mirror image: it takes `--integrator` but
   **not** `--execution-mode` / `--bdpt-walk`. So you cannot script a headless
   wavefront render, and you cannot launch the GUI straight into `bdpt`.

2. **`megakernel` is overloaded.** The `--bdpt-walk megakernel` value names the
   *wavefront* single-kernel subpath build (one `wfBdptWalk` kernel — the "S1
   win", the fastest wavefront-bdpt build), which still runs the full wavefront
   compaction + connect counting-sort. It is **not** the execution-mode
   `megakernel` (the monolithic `main_pass.slang`, no wavefront at all). Reusing
   the word makes the two read as the same thing when they are distinct
   codepaths. The single-kernel-vs-staged distinction belongs to the *walk*
   axis; `megakernel` should belong only to the *execution* axis.

## What Changes

- **One shared flag definition.** Add `src/skinny/cli_common.py` with
  `add_render_flags(parser)` defining `--integrator {path,bdpt}` (default
  `path`), `--execution-mode {megakernel,wavefront}` (default `megakernel`, env
  `SKINNY_EXECUTION_MODE`), and `--bdpt-walk {fused,eye,eye_light}` (default
  `fused`, env `SKINNY_BDPT_WALK`), plus a `resolve_walk(value)` normalizer.
  Single source of truth so the four front-ends cannot drift again.
- **All four front-ends take all three flags.** `skinny`, `skinny-gui`,
  `skinny-web` **gain** `--integrator` (sets the initial `integrator_index`,
  still runtime-cycleable). `skinny-render` **gains** `--execution-mode` +
  `--bdpt-walk`, threaded `HeadlessRenderer.__init__` → `Renderer(...)`.
- **Rename the walk value `megakernel` → `fused`** so only the execution axis
  owns the word `megakernel`. `vk_wavefront.WALK_MODES` becomes
  `("fused", "eye", "eye_light")`; the internal `walk_mode == "megakernel"`
  branches and the `renderer.py` normalization switch to `fused`; help text
  updated. **No behavior or perf change** — the single-kernel `wfBdptWalk` build
  stays the default and is unmodified.
- **Backward-compatible alias.** `megakernel` is still accepted for
  `--bdpt-walk` (CLI, `SKINNY_BDPT_WALK` env, and any saved value) and silently
  resolves to `fused`. `--help` / `choices` advertise only `{fused,eye,eye_light}`.

## Capabilities

### Added Capabilities
- `render-cli`: a single shared command-line surface — `--integrator`,
  `--execution-mode`, `--bdpt-walk` — defined once and exposed identically by
  every front-end (windowed, Qt GUI, web, headless).

### Modified Capabilities
- `wavefront-execution`: the wavefront-bdpt single-kernel subpath-build mode is
  named `fused` (was `megakernel`), disambiguating it from the execution-mode
  `megakernel`; `megakernel` remains accepted as a deprecated alias. The
  execution-mode and bdpt-walk selections are additionally exposed by the
  headless front-end.

## Impact

- **Code:**
  - `src/skinny/cli_common.py` (new) — `add_render_flags` + `resolve_walk`.
  - `src/skinny/app.py`, `src/skinny/ui/qt/app.py`, `src/skinny/web_app.py` —
    replace the inline `--execution-mode` / `--bdpt-walk` definitions with
    `add_render_flags`; wire `--integrator` to the initial `integrator_index`.
  - `src/skinny/headless.py` — call `add_render_flags`; thread
    `execution_mode` + `bdpt_walk` through `HeadlessRenderer.__init__` into the
    `Renderer(...)` construction at `headless.py:141`.
  - `src/skinny/vk_wavefront.py` — `WALK_MODES` rename + the
    `walk_mode == "megakernel"` branches → `"fused"`; default arg `"fused"`.
  - `src/skinny/renderer.py` — the `bdpt_walk` normalization at `renderer.py:936`
    accepts `{fused,eye,eye_light}` and maps `megakernel` → `fused`; constructor
    default `bdpt_walk="fused"`.
- **Tests:** new `cli_common` unit tests (default resolution, env fallback,
  `megakernel`→`fused` alias, integrator/exec/walk choices); a front-end-parity
  test asserting the four parsers expose the same three flags; headless
  `--execution-mode wavefront` smoke render; update any test referencing
  `walk_mode="megakernel"` to `"fused"` (the alias keeps old callers working).
- **Behaviour:** no rendered-output change. `--bdpt-walk megakernel` still works
  (→ `fused`). GUI/web/windowed can now launch directly into `bdpt`; headless
  can now render in wavefront mode.
- **Docs:** CLAUDE.md / AGENTS.md / README flag references and any
  `SKINNY_BDPT_WALK` mention updated to `fused`.

## Notes

Pure CLI-surface + naming cleanup on top of the archived
`wavefront-execution-backend` and the completed `execution-mode-cli-flag`
changes. No shader recompile required (the `wfBdptWalk` kernel and
`main_pass.spv` are byte-unchanged; only the Python-side mode string is renamed).
