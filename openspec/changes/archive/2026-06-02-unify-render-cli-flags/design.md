## Context

Three orthogonal render-selection axes, exposed unevenly across four front-ends,
with one overloaded term. This change unifies the surface and disambiguates the
term — no engine behavior changes.

```
   INTEGRATOR        EXECUTION           BDPT-WALK (wavefront + bdpt only)
   ┌──────┐          ┌───────────┐       ┌────────────────────────────────┐
   │ path │          │ megakernel│       │ fused      one wfBdptWalk kernel│  (was "megakernel")
   │ bdpt │          │ wavefront │       │ eye        staged eye build     │
   └──────┘          └───────────┘       │ eye_light  fully staged         │
                                         └────────────────────────────────┘
```

Coverage before this change:

| front-end          | `--integrator` | `--execution-mode` | `--bdpt-walk` |
|--------------------|:---:|:---:|:---:|
| `skinny` (app.py)  | ✗ runtime-only | ✓ | ✓ |
| `skinny-gui` (qt)  | ✗ | ✓ | ✓ |
| `skinny-web`       | ✗ | ✓ | ✓ |
| `skinny-render`    | ✓ | ✗ | ✗ |

Target: all four rows ✓ / ✓ / ✓, from one shared definition.

## Goals / Non-Goals

**Goals**
- Identical three-flag surface on every front-end, defined once.
- Remove the `megakernel` name collision between the execution axis and the
  wavefront-bdpt walk axis.
- Zero rendered-output change; keep the fast single-kernel walk as default.
- Don't break existing `SKINNY_BDPT_WALK=megakernel` env vars / saved values.

**Non-Goals**
- No change to integrator/execution/walk *semantics* or the kernels.
- No new walk mode; no removal of the single-kernel build.
- No reshuffle of the integrator from runtime-cycleable to fixed — `--integrator`
  only seeds the initial value on the interactive front-ends.

## Decisions

### 1. Rename, don't remove (`megakernel` walk → `fused`)
The walk `megakernel` value is the single-kernel `wfBdptWalk` build — the fastest
wavefront-bdpt build ("the S1 win"), and a *different* codepath from execution
`megakernel` (it still runs wavefront compaction + connect). Removing it would
delete the fast path; renaming it removes only the confusing name. Chosen:
rename to `fused`. The word `megakernel` then names exactly one thing (the
execution axis).

### 2. Shared helper module over per-front-end copies
`cli_common.add_render_flags(parser)` is the single definition. Four parsers call
it; drift becomes impossible. This matches the cross-front-end consistency rule
(camera/UI changes apply to every front-end + the debug viewport). The cost is a
small upfront refactor of three existing inline definitions.

### 3. Silent alias for `megakernel` walk
`resolve_walk("megakernel") == "fused"`, applied to CLI, `SKINNY_BDPT_WALK`, and
any persisted value. `argparse` `choices` advertise only `{fused,eye,eye_light}`,
so to accept the alias the flag is parsed as a free string and normalized by
`resolve_walk` (which raises on a genuinely unknown value). Keeps existing env
vars / muscle memory working with no deprecation warning noise.

### 4. `--integrator` seeds, does not pin
On `skinny` / `skinny-gui` / `skinny-web` the integrator stays a runtime,
GUI-cycleable parameter (`integrator_index` in `STATIC_PARAMS`). `--integrator`
only sets its **initial** value at construction — consistent with how
`--bdpt-walk` is session-fixed but `--integrator` is not. Headless already
treats `--integrator` as fixed; unchanged.

## `cli_common` shape

```python
# src/skinny/cli_common.py
WALK_CHOICES = ("fused", "eye", "eye_light")
_WALK_ALIASES = {"megakernel": "fused"}

def resolve_walk(value: str) -> str:
    v = str(value).strip().lower()
    v = _WALK_ALIASES.get(v, v)
    if v not in WALK_CHOICES:
        raise ValueError(f"unknown bdpt walk {value!r} (expected {WALK_CHOICES} or 'megakernel')")
    return v

def add_render_flags(parser, *, integrator=True, execution=True, walk=True):
    if integrator:
        parser.add_argument("--integrator", choices=("path", "bdpt"), default="path", ...)
    if execution:
        parser.add_argument("--execution-mode", choices=("megakernel", "wavefront"),
                            default=os.environ.get("SKINNY_EXECUTION_MODE", "megakernel"), ...)
    if walk:
        # free string (not choices=) so the megakernel alias is accepted; normalized later.
        parser.add_argument("--bdpt-walk", default=os.environ.get("SKINNY_BDPT_WALK", "fused"),
                            metavar="{fused,eye,eye_light}", ...)
```

Each front-end then calls `args.bdpt_walk = resolve_walk(args.bdpt_walk)` (or
the renderer normalizes — `renderer.py:936` already does, so it just needs the
alias added there too for the direct-construction path).

The `integrator`/`execution`/`walk` keyword switches let a front-end omit a flag
if it must (none currently needs to — all four take all three), without copying
the definition.

## Risks / Trade-offs

- **Free-string `--bdpt-walk` loses argparse's built-in choices error** for the
  advertised values. Mitigated: `resolve_walk` raises a clear `ValueError`
  listing the valid set + the alias; a front-end-parity test covers it.
- **Refactor touches three working parsers.** Mitigated: the parity test asserts
  all four expose identical flags + defaults; behavior tests assert unchanged
  output.

## Migration

`SKINNY_BDPT_WALK=megakernel` and `--bdpt-walk megakernel` keep working (→
`fused`). Saved settings never persisted the walk (it's CLI/session-fixed), so
no settings migration. `main_pass.spv` and the `wfBdptWalk` SPIR-V are
byte-unchanged.
