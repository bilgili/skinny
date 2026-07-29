# Tasks

## 1. Confirm the defect

- [x] 1.1 Write a hostless test that authors a DistantLight `inputs:intensity`
      with two time samples and no default value.
- [x] 1.2 Assert the extracted radiance differs between the two time codes, and
      that each value matches its authored sample.
- [x] 1.3 Run the test and confirm it fails against the current code.

## 2. Fix the owner

- [x] 2.1 Add a required `time: Usd.TimeCode` parameter to
      `_light_color_radiance`.
- [x] 2.2 Read `inputs:color`, `inputs:intensity` and `inputs:exposure` at that
      time code.

## 3. Pass the time code at every call site

- [x] 3.1 `_extract_distant_light` passes its own `time`.
- [x] 3.2 `_extract_dome_light` passes its own `time`.
- [x] 3.3 `_extract_sphere_light` passes its own `time`.
- [x] 3.4 `_rect_light_to_instance` passes its own `time`.
- [x] 3.5 `_disk_light_to_instance` passes its own `time`.
- [x] 3.6 `renderer.py` environment-HDR path passes
      `Usd.TimeCode(float(self.clock.current_time_code))`.

## 4. Fix the second read in the sphere light

- [x] 4.1 Read the stashed colour, intensity and exposure in
      `_extract_sphere_light` at the same time code.

## 5. Verify

- [x] 5.1 Confirm the new test passes.
- [x] 5.2 Confirm no `_light_color_radiance` call site remains without a time
      code.
- [x] 5.3 Run the hostless test suite and compare failures against the
      pre-change baseline.
- [x] 5.4 Run `ruff check src/skinny/usd_loader.py src/skinny/renderer.py`.
- [x] 5.5 Run `openspec validate light-emission-time-sampling --strict`.

## 6. Codex pre-merge review findings

- [x] 6.1 P2 — the spec delta required per-frame re-evaluation for all five
      light types, but the renderer refreshes only DistantLight and SphereLight.
      Corrected the specification: ADD the read-at-time-code requirement for all
      five types, MODIFY the per-frame one to name the types it refreshes and
      record the three exclusions.
- [x] 6.2 P2 — `_read_open_stage` defaulted its evaluation time to
      `Usd.TimeCode.Default()`, so a normal load still read every light at the
      fallback. Dome/rect/disk never re-extract, so the fix was unreachable for
      them. Default to the stage's start time code.
- [x] 6.3 P2 — `headless._to_timecode(None)` substituted `Default()` for an
      absent time, skipping the start-time branch in every headless render.
      Return `None` and let the loader decide.
- [x] 6.4 Same shape found while checking 6.3:
      `Renderer._refresh_usd_live_state` re-extracted lights at `Default()`.
      Use the clock's time code.
- [x] 6.5 Re-run the review until clean.

## 7. Documentation

- [x] 6.1 Update `CHANGELOG.md`.
- [x] 6.2 Record the `tests/fixtures/anim_reread.json` re-capture as a follow-up
      for `scene-intake-interface`. Do not edit the fixture in this change.
