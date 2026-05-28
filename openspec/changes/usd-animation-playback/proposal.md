## Why

skinny loads USD scenes but only ever evaluates the default time code — any
authored animation (moving objects, camera tracks, animated lights) is frozen.
This change adds time-driven playback for the *cheap* time-sampled prim types
(transforms, camera, lights) that require no mesh rebake, establishing the
playback clock and accumulation handling that the deforming-mesh and skeletal
animation work will later build on.

## What Changes

- Add a **playback clock** that maps wall-clock `dt` to USD `timeCode`, loops
  over the stage's authored time range at a configurable fps, and reports
  whether the loaded stage has any animation.
- At load, **index which prims actually animate** (xform tracks, lights, camera)
  via `HasAuthoredTimeSamples`, so per-frame cost scales with the animated set,
  not total scene size.
- **Re-evaluate animated transforms/lights/camera per frame** during playback:
  recompute world matrices for animated instances and re-upload only their TLAS
  `instance_buffer` records; re-extract animated light params; no mesh rebake or
  BVH rebuild.
- Add a **`usd` camera mode** that drives the viewport from an animated USD
  camera track during playback; the user can switch back to Orbit/Free at any
  time.
- Add **built-in transport controls** (play/pause, normalized time scrubber,
  fps) to the shared param/spec tree so they appear in the Qt panel, web panel,
  and debug viewport. The animation section is shown only when the stage has
  animation.
- Fold `current_time_code` into the accumulation state hash so playback resets
  progressive accumulation each frame (1 spp while playing) and accumulates
  normally when paused.

Out of scope (separate later changes): time-sampled deforming-mesh points,
UsdSkel skeletal animation, and USD-attribute-declared control UIs.

## Capabilities

### New Capabilities
- `usd-animation-playback`: time-driven playback of cheap time-sampled USD prims
  (transforms, camera, lights), the playback clock + animated-prim index, the
  `usd` camera mode, the built-in transport UI, and the accumulation-reset
  contract during playback.

### Modified Capabilities
<!-- None: loader-directory-memory is unrelated; no existing spec's requirements change. -->

## Impact

- **Code**: `src/skinny/renderer.py` (playback clock state, `update(dt)`
  re-eval, `_current_state_hash`, `camera_mode`/`self.camera`,
  `_upload_instances` re-upload path), `src/skinny/usd_loader.py` (animated-prim
  index, per-time re-eval helpers), `src/skinny/ui/params.py` +
  `src/skinny/ui/spec.py` + `src/skinny/ui/build_app_ui.py` (transport widgets,
  camera-mode "USD" choice), Qt/Panel backends pick up the new widgets via the
  shared spec tree.
- **Dependencies**: none new — pxr USD already a dependency; uses existing
  TimeCode-aware read functions.
- **Shaders**: none — geometry/BVH stay object-space; only instance transforms,
  light buffers, and camera matrices change at runtime. No `main_pass.spv`
  recompile.
- **Settings/headless**: transport state (fps) may persist in settings.json;
  reconcile with the existing headless `--animate/--frames/--fps` path to reuse
  the playback clock where cheap.
