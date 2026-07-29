## Why

`_light_color_radiance` in `src/skinny/usd_loader.py` reads `inputs:color`,
`inputs:intensity` and `inputs:exposure` with `attr.Get()`. That call carries no
time code. USD resolves a time-code-free `Get()` at the default time code. An
attribute that holds only time samples has no value at the default time code.
USD then returns the schema fallback.

The schema fallback for `UsdLuxDistantLight` intensity is 50000. A stage that
authors `float inputs:intensity.timeSamples = {0: 3, 24: 7}` therefore renders at
50000 at every time code. The animation is invisible. A `UsdLuxSphereLight` shows
the same defect with fallback 1.0.

The callers already compute the correct time code and pass it to the extraction
functions. `_extract_distant_light`, `_extract_sphere_light`,
`_extract_dome_light`, `_rect_light_to_instance` and `_disk_light_to_instance`
all accept a `time` argument. Each one then drops it at the
`_light_color_radiance` call. The per-frame animation path
`Renderer._reextract_animated_lights` runs through the same two functions, so
playback re-extracts an animated light and still gets the fallback value.

`_extract_sphere_light` repeats the same defect a second time. It re-reads
colour, intensity and exposure separately for its stashed `color` and `intensity`
fields, again with no time code.

The `usd-animation-playback` capability already requires time-correct light
parameters. The current code does not meet that requirement. The requirement text
does not name the emission attributes, so the defect stayed invisible.

## What Changes

- Give `_light_color_radiance` a required `time: Usd.TimeCode` parameter. Read
  all three attributes at that time code.
- Pass the caller's own `time` at all five `usd_loader.py` call sites.
- Read the stashed colour, intensity and exposure in `_extract_sphere_light` at
  the same time code.
- Pass the renderer's current time code at the `renderer.py` dome call site.
- Add a hostless test. The test authors a time-sampled light intensity and
  asserts that the extracted radiance differs between two time codes.

The parameter is required, not optional. An optional parameter keeps the defect
reachable at every site that forgets to pass it.

Behaviour on a stage with no time samples does not change. `Get(time)` and
`Get()` return the same value when the attribute holds a default value or only a
schema fallback.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `usd-animation-playback`: name the light emission attributes that per-frame
  re-evaluation must read at the current time code, and require the read to use
  that time code.

## Impact

- `src/skinny/usd_loader.py`: `_light_color_radiance` signature and five call
  sites; the three extra reads in `_extract_sphere_light`.
- `src/skinny/renderer.py`: one call site in the dome-light environment path.
- `tests/`: one new hostless test for time-sampled light intensity.
- `openspec/changes/scene-intake-interface`: that change captured
  `tests/fixtures/anim_reread.json` from the pre-fix extraction. Re-capture the
  fixture with `tests/fixtures/_capture_anim_reread.py` after this change lands.
- No shader, descriptor, byte-layout, or dependency changes.

## Ordering

`openspec/changes/scene-intake-interface` preserves this defect on purpose. Its
specification requires per-frame values identical to the pre-change extraction.
This change must land separately from that one. See that change's `baseline.md`,
section "Defect found while capturing".
