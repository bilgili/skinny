# Change: frame-plan-split

## Why

One `update()` + `render()` pair touches roughly 34 distinct responsibilities,
and the headless path holds a near-verbatim copy of the middle of it.

`update(dt)` (`renderer.py:10461-10558`, 98 lines) does: time and frame
counters, config-matrix emit, FPS smoothing, USD streaming poll, playback clock
advance, animation frame apply, USD live-edit refresh, light recompute, scene
snapshot rebuild, default scene-graph ensure, default-light injection,
distant-light upload gated on authority, aux light authority sync, environment
upload, mesh rebake, tattoo upload, material-type re-upload on scatter change,
accumulation state hash and reset with BDPT splat zero, and gizmo segment
rebuild.

`render()` (`:10559-10795`, 237 lines) does: backend branch with a Metal early
return, fence wait and reset, pick drain, swapchain acquire, uniform pack and
upload, MaterialX skin repack and upload, HUD build and upload, command-buffer
record, cross-frame accumulation barrier, a three-arm execution-mode gate, four
image-layout transitions, offscreen-to-swapchain blit, submit, present, neural
frame-end swap, and frame-index rotation.

`render_headless()` (`:10796-10965`, 170 lines) duplicates the barrier,
execution-mode gate and dispatch block near-verbatim — `:10877-10899` against
`:10648-10670` — plus a per-call binding rewrite. The Metal side has its own
twins at `:9820-9995`.

Consequences: "which passes will this frame run" is not a value anything can
inspect; a change to the execution-mode gate must be made in two or four
places; and ordering constraints (the pick drain must precede the uniform pack)
are invisible. The golden test for `_pack_uniforms` is GPU-marked precisely
because it must construct a renderer to get there.

## What Changes

- Split the per-frame path into three stages:
  1. **scene sync** — the state-advancing half of `update()`: streaming,
     animation, uploads, rebake, live-edit refresh.
  2. **frame plan** — a pure derivation producing a value: execution mode, the
     pass sequence, accumulation state and reset decision, tiling/banding
     decisions, and which optional work (HUD, splat zero, neural swap) this
     frame performs.
  3. **execute** — records and submits the plan against a target (windowed
     swapchain or offscreen), so windowed and headless differ only in target.
- The duplicated barrier / execution-mode / dispatch block is written once.
- The frame plan is assertable without a device, and — once
  `gpu-backend-adapter`'s recording adapter exists — executable against it, so
  pass ordering is testable on any host.
- Pure refactor: same dispatches, same order, same images.

## Capabilities

### Modified Capabilities

- `renderer-module-structure`: adds a carve-out stage — the per-frame path
  becomes sync / plan / execute, with the plan a pure value and one dispatch
  body shared by the windowed and headless targets, under the existing
  bit-identity requirement for carve-out stages.

## Impact

- New: a frame-plan module (pure) and an execute path; hostless tests over the
  plan.
- Modified: `src/skinny/renderer.py` — `update`, `render`, `render_headless`,
  and the Metal render twins (`_render_scene_metal`, `_render_wavefront_metal`,
  `_render_headless_metal`, `_render_windowed_metal`).
- Unchanged: dispatch sequence, image output, accumulation semantics, the
  `params`-registry-derived state hash (owned by `accumulation-reset-registry`,
  consumed here).
- **Depends on** `renderer-gpu-resource-set` (execute needs a stable resource
  interface) and benefits from `gpu-backend-adapter` stage 3 (the recording
  adapter is what makes the plan's execution testable). Land last of the
  renderer-cluster changes.
- Docs: `docs/Architecture.md` per-frame section; `docs/Megakernel.md` and
  `docs/Wavefront.md` where they describe the frame path.
