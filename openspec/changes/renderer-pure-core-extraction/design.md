# Design: renderer-pure-core-extraction

## Context

The carve-out capability already has three landed stages (`mlt_chain`,
`frame_derive`, the wavefront pass factories) and a stated five-step pattern.
This is the same pattern applied to the region *above* the class, which is the
easiest region in the file: no `self`, no device handles, no ordering
constraints.

The reason it matters more than its difficulty suggests is the silent-skip
failure mode. A test that skips because an SDK is missing looks identical in CI
output to a test that ran — and this project has already been bitten by it.

## Goals / Non-Goals

**Goals**
- Device-free code importable with no GPU package.
- Enforced, not incidental: a gate that fails on a GPU import.
- Zero signature or value change.

**Non-Goals**
- Making `Renderer` itself importable without `vulkan`. That is the backend
  adapter's job, if it ever happens.
- Redesigning the packers. `flat-material-field-table` does that, afterwards.
- Moving anything that touches `self` or a device handle.

## Decisions

### D1 — Split by cluster, not into `renderer_pure.py`

One catch-all module would be a shallow container: nothing binds camera math to
EXR writing. Separate modules by subject — material packing, camera, film IO,
SPPM budget, texture pool, helpers — each of which has a real consumer already
(`debug_viewport` wants camera; `bxdf` wants `_hashable_value`; the parity
harness wants film IO).

### D2 — Re-export from `renderer` for one release, then repoint

35 test references and several source imports point at `skinny.renderer`.
Re-exporting keeps the move mechanical and reviewable. But the tests must be
repointed in the same change, otherwise their hostlessness is not actually
enforced — a test importing `skinny.renderer` still drags in `vulkan` even if
the symbol it wants now lives elsewhere. Source consumers can keep the
re-export; tests may not.

### D3 — The gate is a subprocess import check

`tests/test_render_session_module.py` already does this for "no Qt import".
Same shape: import each new module in a subprocess with `vulkan` made
unimportable, and fail if it raises. This is what turns "device-free" from a
comment into a property.

### D4 — `TexturePool` moves even though it holds GPU objects

It holds them; it does not import them. Its constructor takes a resource
module. It moves, and its hostless test uses a fake — which is the same
recording-fake shape `renderer-gpu-resource-set` needs, so the two changes
should share it.

### D5 — Nothing else is touched

No renaming, no reordering, no "while I'm here". The value of this change is
that its diff is reviewable as a pure move; anything else dilutes that.

## Risks / Trade-offs

- **Risk: an import cycle.** `debug_viewport` imports camera from `renderer`
  today; after the move it imports the camera module directly, and `renderer`
  imports it too. No cycle, but check each new module's imports explicitly.
- **Risk: a constant is read at import time somewhere that now resolves
  differently.** All of these are plain values; capture them before and after
  and compare.
- **Risk: `from skinny.renderer import *`-style reliance.** Grep for it; there
  should be none, but the re-export covers it either way.
- **Trade-off: more modules.** Six small modules instead of one giant one. That
  is the point; each has a consumer that wants exactly it.

## Open Questions

- Does `SkinParameters` belong with material packing or on its own? It is the
  skin path's own record with a documented std140 layout and a single consumer;
  leaning toward its own module beside the skin material code.
- Should the EXR/Radiance writers move to `pbrt/` beside the other image
  format code (`hdr.py`, `envmap.py`)? Leaning no — those are readers for
  intake, these are writers for output; different direction, different
  consumer.
