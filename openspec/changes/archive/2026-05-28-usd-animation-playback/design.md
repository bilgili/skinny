## Context

`usd_loader.py` already threads a `time: Usd.TimeCode` argument through every
read function (`_read_open_stage`, `_extract_camera`, `_extract_lights`,
`_world_transform`, mesh/gprim readers), but `Renderer._load_usd_model` only
ever reads at the default time code, so authored animation is frozen. The
loader retains the open stage (`self._usd_stage`, renderer.py:1553), which lets
us re-evaluate prims at runtime.

Geometry and BVH are baked once in **object space**; per-instance 4×4 world
transforms live in the TLAS `instance_buffer` (`_upload_instances`,
renderer.py:2255–2266). The distant-light SSBO is already re-uploaded every
frame inside `update(dt)` (renderer.py:5563). The renderer is a *progressive*
path tracer: `accum_frame` resets whenever `_current_state_hash()` changes
(renderer.py:5507, 5587). The viewport camera is selected by `camera_mode`
(`"orbit"`/`"free"`), with `self.camera` returning the active one
(renderer.py:1145) and `toggle_camera_mode` transferring viewpoint
(renderer.py:1165). UI is built once from a shared spec tree
(`ui/params.py` → `ui/spec.py` → `ui/build_app_ui.py`) consumed by both the Qt
and Panel backends.

## Goals / Non-Goals

**Goals:**
- Time-driven playback of *cheap* time-sampled prims: xform tracks, camera
  tracks, light attribute animation — no mesh rebake.
- A reusable playback clock + animated-prim index that the deforming-mesh and
  skeletal changes build on.
- Built-in transport controls present in Qt, Panel, and debug viewport via the
  shared spec tree.
- A `usd` camera mode following an animated USD camera, with the user able to
  switch back to Orbit/Free.
- Correct progressive-accumulation behavior: noisy while playing, clean while
  paused.

**Non-Goals:**
- Time-sampled deforming-mesh points/normals (requires per-frame rebake + BVH) —
  separate change `usd-deforming-mesh-anim`.
- UsdSkel skeletal animation — separate change `usd-skeletal-anim`.
- USD-attribute-declared control UIs — separate change `usd-driven-control-ui`.
- Sub-frame motion blur / temporal AA.

## Decisions

### D1: Per-frame stage re-eval with a load-time animated-prim index (Approach C)
Each playback frame re-evaluates the retained `_usd_stage` at
`current_time_code`, but only for prims recorded at load as having authored time
samples (`HasAuthoredTimeSamples` on xformOps / light attrs / camera). Per-frame
cost scales with the animated set, not total scene size.

- **Alternative A (re-traverse whole stage/frame)**: simplest, but O(scene) pxr
  work every frame even when one prim moves. Rejected for large static scenes.
- **Alternative B (pre-sampled keyframe arrays + custom interpolation)**:
  decouples the render loop from pxr but reinvents USD value resolution and
  interpolation (matrix decompose/slerp), adding state and correctness risk.
  Rejected — pxr already resolves values correctly at any time code.

### D2: Playback clock as a small dedicated unit
A `PlaybackClock` holds `start_time_code`, `end_time_code`,
`time_codes_per_second`, `playing`, `current_time_code`, `playback_fps`, `loop`,
and `has_animation`. `advance(dt)` moves `current_time_code` by
`dt * playback_fps` (in time-code units) when playing and wraps within
`[start, end]`. Pure logic, no GPU/pxr deps → unit-testable in isolation. The
renderer owns one instance, populated from stage metadata
(`startTimeCode`/`endTimeCode`/`timeCodesPerSecond`) at load.

Rationale: keeps time math out of the already-large `renderer.py` update path
and gives a clean seam for the headless `--animate` path to reuse.

### D3: Xform animation = TLAS instance re-upload, no rebake
For each animated xform prim, recompute its world matrix at `current_time_code`
via the existing `_world_transform(prim, time)` and overwrite that instance's
record in `instance_buffer` (re-upload via the existing `_upload_instances`
path). Geometry/BVH are untouched. This is what keeps #1 cheap and is the reason
deforming meshes are deferred (they would require re-baking object-space data +
rebuilding the BVH).

### D4: `usd` camera mode via a follower camera
Add `"usd"` to `camera_mode`. Introduce a lightweight follower whose view/proj
matrices are set each frame from the USD camera evaluated at `current_time_code`
(reusing `_extract_camera`). `self.camera` returns it when
`camera_mode == "usd"`. The mode is only offered when the loaded stage has a USD
camera; switching to Orbit/Free hands control back to the user. Accumulation
sees camera motion because the follower's `state_signature()` reflects the
current matrices.

### D5: Normalized 0–1 time scrubber
The transport scrubber is a normalized 0–1 slider; its getter/setter map to
`[start_time_code, end_time_code]`. This avoids rebuilding the UI after the
async USD load discovers the real time range (slider `lo`/`hi` are fixed at
build time, but the range is unknown until the stage opens). A real-time-code /
frame-number readout is a later polish item, not part of #1.

### D6: Accumulation reset via the state hash
Add `current_time_code` to `_current_state_hash()`. While playing, time changes
every frame → accumulation resets → 1 spp per displayed frame (inherent to a
progressive path tracer in motion). While paused, the hash is stable → normal
progressive accumulation. `playing` itself need not be hashed; the time value
subsumes it.

### D7: Transport UI through the shared spec tree
Add to `ui/params.py`/`ui/spec.py`/`ui/build_app_ui.py`: a play/pause Checkbox
bound to `clock.playing`, the normalized time Slider, and an fps IntSpin; extend
the camera-mode Combo's choices with `"USD"`. The animation Section is gated on
`clock.has_animation` so static scenes show no transport. Both Qt and Panel
backends pick these up automatically since they walk the same tree. (Honors the
project rule that camera/UI changes apply across all front-ends + debug
viewport.)

## Risks / Trade-offs

- **Per-frame pxr eval cost on the main thread** → Mitigated by D1's animated
  set (only animated prims re-evaluated) and by D3 (transform-only re-upload, no
  rebake). If eval still stalls on huge animated sets, the load-time index can
  later cache sampled keyframes (escape hatch toward Approach B) without changing
  the spec.
- **Noisy frames during playback** (accumulation resets each frame) → Inherent
  and expected for a progressive PT; paused frames converge. Documented as
  intended behavior, not a bug.
- **USD camera vs. user camera conflict** → Resolved by an explicit mode (D4);
  USD camera only drives the viewport when selected.
- **Async load race** (time range unknown until the stage opens) → D5's
  normalized scrubber sidesteps post-load UI rebuilds; transport stays inert
  until `has_animation` flips true.
- **Headless divergence** → The existing `--animate/--frames/--fps` path should
  reuse `PlaybackClock` to avoid two time models; if reconciliation is
  non-trivial it is deferred and noted, with the windowed clock authoritative.

## Open Questions (resolved during implementation)

- **fps default**: follows the stage's `timeCodesPerSecond`, falling back to 24
  when unauthored (`build_playback_clock`).
- **fps persistence / start state**: playback always starts paused
  (`PlaybackClock` default `playing=False`); fps is re-derived from the stage on
  each load. Separate settings.json persistence of `playback_fps` was dropped —
  it would conflict with the per-load stage-fps reset, and the stage value is the
  natural default.
- **Headless `--animate`**: left as a standalone batch exporter (explicit
  per-frame TimeCodes), not reconciled onto `PlaybackClock`; the two serve
  different purposes (deterministic export vs. wall-clock interactive playback).
