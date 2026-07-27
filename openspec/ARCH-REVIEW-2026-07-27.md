# Architecture review — 2026-07-27

Ten deepening proposals from a second architecture pass (deep-module lens) over
the surface the 2026-07-23 round left behind. That round produced 8 changes, 7
of which have landed; these are what those left.

Proposal-only. Nothing implemented. All pass `openspec validate --strict`.

## The architectural view

Every candidate here makes the same move the previous round made: **a fact
mirrored across N call sites gets one owner, derived from an authoritative
source, with a hostless gate that fails on drift.**

Four of the ten extend capabilities the previous round created:

| previous change | capability it created | extended by |
|---|---|---|
| `renderer-module-carveout` | `renderer-module-structure` | `renderer-gpu-resource-set`, `renderer-pure-core-extraction`, `frame-plan-split` |
| `reflection-owned-byte-layouts` | `shader-byte-layouts` | `flat-material-field-table` |
| `unified-render-envelope-predicate` | `render-envelope` | `choice-table-owners` defers to it |

The other six mint new capabilities at seams that do not have an owner yet:
the backend adapter, scene intake, session settings, the renderer command
interface, and the UI node spec.

## The ten changes

| change | capability | strength |
|---|---|---|
| `renderer-gpu-resource-set` | `renderer-module-structure` (ADD) | **Strong** |
| `gpu-backend-adapter` | `gpu-backend-adapter` (NEW) + `metal-backend` (MOD) | **Strong** |
| `scene-intake-interface` | `scene-intake` (NEW) | **Strong** |
| `flat-material-field-table` | `shader-byte-layouts` (ADD) | **Strong** |
| `session-settings-owner` | `session-settings` (NEW) | **Strong** — live data loss |
| `renderer-command-interface` | `renderer-command-interface` (NEW) + `qt-render-threading` (MOD) | **Strong** — unsynchronised writes |
| `renderer-pure-core-extraction` | `renderer-module-structure` (ADD) | **Strong** |
| `frame-plan-split` | `renderer-module-structure` (ADD) | Worth exploring |
| `ui-spec-scene-properties` | `usd-scene-editing-ui` (MOD) | Worth exploring |
| `choice-table-owners` | `choice-table-ownership` (NEW) | Worth exploring |

## Where the mass sits

`renderer.py` is 11,604 lines — 17% of the Python host and 6× the next module.
Every renderer-cluster candidate is a piece of it.

- **40** live `is_metal` branches in `renderer.py`, plus 15 Metal-only methods
- **15** function-local imports of `usd_loader` privates, invisible in the
  import graph
- **273** byte-packing sites; 5 route through `slang_layout`
- **1** module-scope `import vulkan` gating every hostless renderer test

## Top recommendation

**`renderer-gpu-resource-set`.** 1,342 lines with no external module importing
them — the cluster in `renderer.py` where deepening has the least chance of
breaking a caller. It pairs allocation with destruction (today 7,800 lines
apart), absorbs 10 of the 40 backend branches, and gives `gpu-backend-adapter`
its first honest consumer.

Then `gpu-backend-adapter` — the interface the resource set reveals.

`session-settings-owner` and `renderer-pure-core-extraction` are independent
and small; land them whenever. `session-settings-owner` fixes live data loss,
so prefer sooner.

## Dependency order

```
renderer-pure-core-extraction  →  flat-material-field-table
renderer-gpu-resource-set      →  gpu-backend-adapter  →  frame-plan-split
scene-intake-interface         →  frame-plan-split

independent, any time:  session-settings-owner · renderer-command-interface
                        ui-spec-scene-properties · choice-table-owners
```

Three changes ADD to `renderer-module-structure` — archive them in the order
`renderer-pure-core-extraction` → `renderer-gpu-resource-set` →
`frame-plan-split`, or the deltas apply against the wrong base. This is the
same hazard the spectral trio hit.

## Report

The HTML candidate report this set came from was written to the session
scratchpad and is ephemeral: `architecture-review-20260727-0830.html`.
