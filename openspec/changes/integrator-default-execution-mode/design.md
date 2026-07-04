## Context

The render-selection flags are defined once in `src/skinny/cli_common.py`
(`add_render_flags`) and shared by all four front-ends (`skinny`, `skinny-gui`,
`skinny-web`, `skinny-render`). Two of them interact:

- `--integrator {path,bdpt,sppm}` — default `None` (a sentinel meaning "use the
  persisted value on interactive front-ends, else `path`"). On interactive
  front-ends it sets the **initial** integrator and stays runtime-cycleable; it
  is persisted in `settings.json`.
- `--execution-mode {megakernel,wavefront}` — default
  `os.environ.get("SKINNY_EXECUTION_MODE", "megakernel")`, a hardcoded
  `megakernel` fallback. It is **fixed for the session** (a `Renderer`
  constructor argument), never runtime-switchable, GUI-surfaced, or persisted.

The coupling that isn't modelled: **SPPM has no megakernel path.**
`validate_render_flags` (cli_common.py:96-106) rejects `--integrator sppm` unless
`--execution-mode wavefront` was also passed, so every SPPM launch needs two
flags, and a persisted `sppm` integrator relaunches into that error (because the
execution-mode default is a static `megakernel`, blind to the integrator).

Construction order matters. In `app.py` the `Renderer` is built with
`execution_mode=args.execution_mode` (app.py:533-546) **before** persisted params
(`_apply_saved_params`, app.py:585) and the CLI integrator override (app.py:589)
are applied — but the persisted `saved` dict is already loaded at that point, so
the startup integrator is *knowable* before construction even though it is
*applied* after.

Related machinery that does **not** change: `effective_execution_mode()`
(params.py:79-97) still clamps `bdpt + wavefront → megakernel` when unsupported;
`renderer.execution_modes` (renderer.py:1602-1611) still gates which modes a
backend exposes; `_current_state_hash` still includes `integrator_index` and
still omits `execution_mode_index` (fixed for the session).

## Goals / Non-Goals

**Goals:**

- Selecting an integrator picks its execution mode automatically at startup:
  `path` → `megakernel`, `bdpt` → `megakernel`, `sppm` → `wavefront`.
- `--integrator sppm` alone works — no second flag, no startup error.
- Keep the override: an explicit `--execution-mode` (flag or env) still wins and
  pins the mode for the session.
- One shared derivation used by every front-end, so they cannot drift.
- Execution mode stays fixed for the session (no new runtime state, no
  accumulation-hash change, no persistence).

**Non-Goals:**

- Runtime-live switching of the execution mode when the integrator is cycled
  mid-session. That would require making the execution mode runtime-switchable
  (building/rebuilding both pipelines) and is explicitly out of scope.
- Changing which integrator × execution-mode combinations are *valid*
  (`combo_is_valid` in `pbrt/parity.py` is unchanged) — only which mode is the
  *default*.
- Persisting the execution mode. It is re-derived from the (persisted)
  integrator on every launch.
- Any shader, descriptor-binding, or Metal/Vulkan backend change.

## Decisions

### Decision 1 — Add `auto` as a third `--execution-mode` choice, and make it the default

`--execution-mode` becomes `{auto,megakernel,wavefront}` with default
`os.environ.get("SKINNY_EXECUTION_MODE", "auto")`. `auto` means "derive from the
integrator"; `megakernel`/`wavefront` are explicit pins.

- **Why:** exactly mirrors the existing `--backend {auto,metal,vulkan}` (default
  `auto`) pattern, so the behavior is self-documenting in `--help` and familiar.
  `auto` cleanly distinguishes "user didn't choose" from "user chose
  megakernel," which a bare `megakernel` default cannot.
- **Alternatives considered:**
  - *`default=None` sentinel* (like `--integrator`/`--backend`): works, but is
    invisible in `--help` and less self-explanatory than a named `auto` value.
  - *Per-front-end hardcoded defaults*: reintroduces exactly the drift the shared
    `add_render_flags` source exists to prevent.

### Decision 2 — A single post-parse resolver in `cli_common.py`

argparse resolves the flag default eagerly, before `--integrator` is known, so
the derivation cannot live in the flag `default=`. Add a shared mapper:

```python
DEFAULT_EXECUTION_FOR_INTEGRATOR = {
    "path": "megakernel",
    "bdpt": "megakernel",
    "sppm": "wavefront",
}

def resolve_execution_mode(execution_mode: str | None, integrator: str | None) -> str:
    """auto → derive from the startup integrator; an explicit mode wins."""
    if execution_mode and execution_mode != "auto":
        return execution_mode                      # explicit flag/env pins it
    return DEFAULT_EXECUTION_FOR_INTEGRATOR.get(integrator or "path", "megakernel")
```

- **Why:** one function, one mapping, reused by all four front-ends — same
  no-drift guarantee as `add_render_flags`. Precedence falls out naturally:
  explicit mode > integrator-derived default.
- **Alternatives considered:** duplicating the `if auto:` logic at each call site
  (drift risk); putting the mapping on the `Renderer` (the renderer is
  constructed *with* the already-resolved mode — resolution belongs at the CLI
  layer, next to the other flag resolvers like `resolve_walk`).

### Decision 3 — Resolve before validation and before constructing the renderer, from the startup integrator

Each front-end computes its **resolved startup integrator** and calls
`resolve_execution_mode` before `validate_render_flags` and before building the
`Renderer`, writing the concrete mode back onto `args.execution_mode`:

- `skinny-render` (headless): integrator = `args.integrator or "path"` (no
  persistence).
- `skinny` / `skinny-gui` / `skinny-web` (interactive): integrator =
  `args.integrator` if given, else the persisted integrator name from `saved`,
  else `"path"`. `saved` is already loaded before renderer construction, so the
  persisted `sppm` case resolves to `wavefront` on the next launch with no flags
  — the key ergonomic win.

- **Why order this way:** `validate_render_flags` reads `args.execution_mode`;
  resolving first means the sppm guard sees the *derived* mode, so plain
  `--integrator sppm` (→ `wavefront`) passes while `--integrator sppm
  --execution-mode megakernel` (explicit) still fails. And the `Renderer` must be
  built with the final mode (it is fixed for the session).

### Decision 4 — Keep the `sppm + megakernel` startup guard; narrow it to explicit overrides

`validate_render_flags` keeps rejecting `sppm + megakernel`, but because the
default is now `auto` → `wavefront`, the guard only trips when the user
*explicitly* forced `--execution-mode megakernel`. Update its docstring and the
error message to say the mode was explicitly forced and that dropping the flag
(or using `--execution-mode wavefront`) is the fix.

- **Why:** an explicit impossible combo should still fail loudly and early
  (before GPU init) rather than silently do nothing — this is the existing
  `render-cli` contract, preserved.

## Risks / Trade-offs

- **Runtime cycling to `sppm` in a `megakernel`-launched session still can't run
  SPPM** → out of scope by design; documented. Mitigation: launching with
  `--integrator sppm` (now the only thing needed) picks `wavefront`, and the
  existing runtime gate continues to govern mid-session cycling. No regression —
  this was already the behavior.
- **`SKINNY_EXECUTION_MODE=auto` must be accepted** → include `auto` in the env
  fallback path and the flag `choices` so an explicit `auto` (flag or env)
  parses and derives, rather than raising an "invalid choice" error.
- **A persisted `sppm` integrator now silently selects `wavefront`** where it
  previously errored → intended improvement, but a behavior change for anyone
  who relied on the error. Mitigation: it is the documented, desired outcome and
  is called out in `CHANGELOG.md`.
- **Interactive resolver must read the persisted integrator the same way
  `_apply_saved_params` does** → mismatched key/format would derive from the
  wrong integrator. Mitigation: reuse the existing persisted-params accessor /
  `INTEGRATOR_INDEX` mapping rather than re-parsing `settings.json` by hand;
  cover with a unit test that a persisted `sppm` resolves to `wavefront`.
- **Backend availability of the derived mode** → if a backend does not expose
  `wavefront` for the integrator, the existing `execution_modes` /
  `effective_execution_mode` clamp still applies (and `sppm` there was already
  unusable). This change does not widen combo validity.

## Migration Plan

Additive and backward-compatible — no data migration:

1. Existing scripts passing an explicit `--execution-mode megakernel|wavefront`
   (or the env var) are unchanged — explicit still wins.
2. Scripts passing `--integrator sppm --execution-mode wavefront` keep working
   (explicit wavefront == derived wavefront); the second flag simply becomes
   optional.
3. Rollback: revert the `auto` choice + resolver; the hardcoded `megakernel`
   default returns and the two-flag SPPM ritual is required again.

## Open Questions

- None blocking. (Metal's `wavefront` availability for `sppm` is governed by the
  existing backend `execution_modes` gate and is orthogonal to this defaulting
  change; the stale "pinned to `megakernel` on Metal" wording in the
  `wavefront-execution` spec predates `metal-wavefront-parity` and is left
  untouched by this change.)
