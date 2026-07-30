# Design: frame-plan-split

## Context

The per-frame path is the last big undivided region of `renderer.py`. It is
also the riskiest to touch: every rendered image goes through it, and the
duplication between windowed and headless is exactly the kind that hides a
divergence.

`frame_derive.py` already exists — the pure frame-constant derivation carved
out earlier. This change is its natural continuation one level up: from "derive
the constants" to "derive the whole frame's decisions".

## Goals / Non-Goals

**Goals**
- A frame plan that is a value, inspectable without a device.
- One dispatch body for windowed and headless.
- Ordering constraints stated rather than implied by line order.

**Non-Goals**
- Changing what any frame does. Same dispatches, same order, same images.
- Merging the Metal and Vulkan execute paths — that is
  `gpu-backend-adapter`'s job; this change makes them consume the same plan.
- Reordering `update()`'s scene-sync steps. They are unpicked here, not
  rearranged.

## Decisions

### D1 — The plan is derived, not executed, and holds no device handles

The plan names passes, counts, flags and decisions. It holds no buffers, no
command buffers, no pipelines. That is what makes it assertable and what keeps
this change from becoming a rewrite of the execute path.

### D2 — Windowed and headless differ only in target

Today they differ in target *and* in a duplicated middle. After the split, the
target supplies: where the output goes, whether a swapchain is acquired and
presented, and whether a readback follows. Everything between the barrier and
the submit is one body.

The per-call binding rewrite that `render_headless` performs
(`renderer.py:10820`-ish, binding 1) must be understood before it is moved —
either it is a genuine target difference, or it is a latent bug that windowed
rendering avoids by luck. Decide and record.

### D3 — Scene sync keeps its order; only its home changes

`update()`'s 18 steps have accumulated ordering dependencies (light upload
gated on authority after the scene rebuild; rebake before instance upload;
accumulation hash last). This change moves them, groups them, and states the
dependencies — it does not reorder them. Any reorder is a separate change with
its own gate.

### D4 — The accumulation reset stays where it is owned

`_current_state_hash` is derived from the `params.py` registry under
`accumulation-reset-registry`. The plan *consumes* the reset decision; it does
not re-derive it. Two owners for one decision is what that capability exists to
prevent.

### D5 — Ordering constraints become assertions

`poll_pick_result` is called from both `render` and `render_headless` and is
ordering-critical relative to `_pack_uniforms`. In the split, "pick drain
precedes uniform pack" is a property of the plan's step order and is asserted,
rather than being a fact about two line numbers in two functions.

### D6 — Land last

`renderer-gpu-resource-set` gives execute a stable resource interface;
`gpu-backend-adapter` stage 3 gives the plan a recording target to execute
against. Doing this change first would mean writing the execute path twice.

## Risks / Trade-offs

- **Risk: an image changes.** This is the highest-risk change in the set. Gate:
  the full parity matrix, both gates, before and after — identical, not close —
  plus per-integrator smoke on both backends and both execution modes.
- **Risk: the headless binding rewrite (D2) is load-bearing.** Investigate
  before moving; if it is a bug, fix it as a separate, announced change rather
  than folding it into a refactor.
- **Risk: the Metal band/tile decisions leak into the plan.** They should —
  banding is a frame decision, not a dispatch detail — but they must be
  expressed as capability-driven (`needs_watchdog_tiling`) rather than
  `is_metal`, or this change re-imports the branch problem.
- **Trade-off: one more indirection per frame.** The plan is derived once per
  frame; the cost is nil next to a dispatch.

## Open Questions

- Should scene sync produce a dirty set that the plan consumes, or mutate and
  let the plan read renderer state? A dirty set is cleaner and makes "what
  changed this frame" inspectable — but it is a bigger change. Leaning: start
  with the plan reading state, add the dirty set only if the plan's inputs turn
  out to be hard to enumerate.
- Do the per-frame sync objects (fences, semaphores, command buffers) belong to
  the target or to the resource set? Coordinate with
  `renderer-gpu-resource-set`'s open question rather than deciding twice.
