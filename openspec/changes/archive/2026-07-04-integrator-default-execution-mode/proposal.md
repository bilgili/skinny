## Why

Today the integrator and the execution mode are chosen independently, but they
are not independent: **SPPM has no megakernel path** and is refused at startup
unless the user *also* remembers to pass `--execution-mode wavefront`, while
`path` and `bdpt` want `megakernel`. Every SPPM launch is a two-flag ritual
(`--integrator sppm --execution-mode wavefront`), and a persisted `sppm`
integrator relaunches straight into the "SPPM requires wavefront" error. The
execution mode a given integrator needs is knowable — the tool should pick it.

## What Changes

- **`--execution-mode` gains an `auto` value, and `auto` becomes the default.**
  When the mode is `auto` (i.e. neither the flag nor `SKINNY_EXECUTION_MODE` was
  set to a concrete mode), the execution mode is **derived from the startup
  integrator**: `path` → `megakernel`, `bdpt` → `megakernel`, `sppm` →
  `wavefront`. This mirrors the existing `--backend {auto,metal,vulkan}` pattern.
- **The parameter still overrides.** An explicit `--execution-mode megakernel`
  or `--execution-mode wavefront` (flag or `SKINNY_EXECUTION_MODE` env) wins over
  the integrator-derived default and pins the mode for the session, exactly as
  today. Precedence: explicit mode > integrator-derived default.
- **`--integrator sppm` alone now just works** — it resolves to `wavefront` and
  no longer errors. The `sppm + megakernel` startup rejection is preserved but
  now fires **only** when the user explicitly forces `--execution-mode
  megakernel` (a genuine, deliberate impossible combo).
- **Execution mode stays fixed for the session** (constructor argument, not
  runtime-switchable or persisted) — unchanged. The derivation happens once, at
  startup, from the integrator that is active at launch (explicit `--integrator`,
  else the persisted integrator on interactive front-ends, else `path`).
- **Not in scope:** runtime-live switching of the execution mode when the
  integrator is cycled mid-session. Cycling to `sppm` inside a session launched
  in `megakernel` remains governed by the existing runtime gate; the fix is to
  launch with `--integrator sppm` (which now auto-selects `wavefront`).
- **Docs updated** to describe the new `auto` default and the derivation table.

## Capabilities

### New Capabilities
<!-- none — this modifies existing CLI/execution-mode selection behavior -->

### Modified Capabilities
- `render-cli`: the shared `--execution-mode` flag gains an `auto` choice that is
  the new default and derives the mode from the integrator; the startup
  rejection of `sppm + megakernel` is narrowed to fire only on an explicit
  megakernel override.
- `wavefront-execution`: the "no execution mode specified → `megakernel`" default
  is refined — with no explicit mode, `sppm` resolves to `wavefront` while
  `path`/`bdpt` still default to `megakernel`; the mode remains fixed for the
  session.

## Impact

- **Code:** `src/skinny/cli_common.py` (the `--execution-mode` flag definition +
  a new shared `resolve_execution_mode` mapper; `validate_render_flags`
  docstring/message); the four front-ends that construct the renderer —
  `src/skinny/app.py` (windowed), the Qt GUI, the web server, and
  `src/skinny/headless.py` — each resolves the mode from its startup integrator
  before constructing `Renderer`.
- **Behavior:** default execution mode for `path`/`bdpt` is unchanged
  (`megakernel`); `sppm` now defaults to `wavefront` instead of erroring.
  Explicit flag/env behavior is unchanged. No shader change, no descriptor
  binding change, no accumulation-hash change (execution mode is still fixed for
  the session).
- **Docs:** `README.md` (Compatibility matrix + CLI flag list), `CLAUDE.md`
  (Compatibility matrix + Commands wording), `docs/Wavefront.md`,
  `CHANGELOG.md`.
- **Tests:** `tests/` CLI-resolution unit coverage for the derivation + override
  precedence; the `render-cli` startup-validation tests updated so `--integrator
  sppm` alone passes and `sppm + explicit megakernel` still errors.
