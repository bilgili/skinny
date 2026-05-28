## 1. Playback clock

- [x] 1.1 Add a `PlaybackClock` unit (start/end time code, time_codes_per_second, playing, current_time_code, playback_fps, loop, has_animation) with `advance(dt)` that moves time by `dt * playback_fps` and wraps within range
- [x] 1.2 Default fps from stage `timeCodesPerSecond`, fall back to fixed default when unauthored; populate range from `startTimeCode`/`endTimeCode`
- [x] 1.3 Unit tests: advance while playing, loop wrap at end, paused no-op, no-animation inert

## 2. Animated-prim index (load time)

- [x] 2.1 In `usd_loader.py`, scan the stage for prims with authored time samples on transform stack, light attrs, and camera; return animated-xform paths, animated-light paths, camera-animated flag
- [x] 2.2 Wire the index + clock range into `Renderer._load_usd_model`; set `has_animation` from the index
- [x] 2.3 Unit test on a synthetic in-memory Sdf layer with a time-sampled xform: detected as animated; a static prim excluded (uses pxr in the 3.13 venv)

## 3. Per-frame re-evaluation

- [x] 3.1 In `Renderer.update(dt)`, advance the clock; when playing and time changed, re-evaluate only indexed animated prims at `current_time_code`
- [x] 3.2 Animated transforms: recompute world matrix via `_world_transform(prim, time)` and re-upload only that instance's `instance_buffer` record (no rebake/BVH)
- [x] 3.3 Animated lights: re-extract at time and update the corresponding light buffers
- [x] 3.4 Verify no mesh rebake or BVH rebuild is triggered by transform/light animation (headless: single buffer-grow at load only)

## 4. USD camera mode

- [x] 4.1 Add a follower camera fed by `_extract_camera` at `current_time_code`; add `"usd"` to `camera_mode` and return it from `self.camera` when active
- [x] 4.2 Offer `usd` mode only when the stage has a USD camera; switching to Orbit/Free returns user control
- [x] 4.3 Ensure the follower's `state_signature()` reflects current matrices so accumulation resets on camera motion

## 5. Accumulation

- [x] 5.1 Add `current_time_code` to `_current_state_hash()`
- [x] 5.2 Verify: playing resets `accum_frame` each advanced frame (1 spp); paused converges normally (paused accumulation confirmed in headless A/B; reset follows from time code being in the state hash)

## 6. Transport UI (shared spec tree)

- [x] 6.1 Add play/pause Checkbox, normalized 0–1 time Slider (maps to `[start,end]`), and fps IntSpin to `ui/params.py` + `ui/spec.py` + `ui/build_app_ui.py`
- [x] 6.2 Extend camera-mode Combo choices with `"USD"` (only when a USD camera exists)
- [x] 6.3 Gate the Animation section on `has_animation`
- [x] 6.4 Confirm the controls render in Qt panel, web panel, and debug viewport from the shared tree (Qt confirmed by user; shared spec tree drives Panel + debug)

## 7. Headless + settings reconciliation

- [x] 7.1 Headless `--animate/--frames/--fps` left standalone by decision: it is a deterministic batch exporter iterating explicit TimeCodes, orthogonal to the wall-clock interactive `PlaybackClock`; the windowed clock is authoritative for interactive playback
- [x] 7.2 Playback starts paused (PlaybackClock default); `playback_fps` follows the stage's `timeCodesPerSecond` on each load. Separate fps persistence intentionally dropped — it would conflict with the per-load stage-fps reset

## 8. Verification

- [x] 8.1 New files ruff-clean; no new lint errors in changed files; `pytest -m "not gpu"` green (165 passed) + headless animation test green (built 3.13 venv)
- [x] 8.2 Headless A/B render at two time codes asserting an animated instance moved (`tests/test_headless_animation.py`; centroid shifts x=15→x=113)
- [x] 8.3 Manual: load an animated USD in the windowed app, scrub + play/pause confirmed by user (anim_ball.usda)
