# Design

## Ownership

`_light_color_radiance` is the single owner of the light emission read. It
combines `inputs:color`, `inputs:intensity` and `inputs:exposure` into one linear
radiance vector. Every light extraction path routes through it.

The defect is one missing argument at that owner. The fix stays at the owner. No
caller gains a guard, and no caller re-reads an attribute to work around the
owner.

`_extract_sphere_light` is the one exception. It reads the same three attributes
a second time for its stashed `color` and `intensity` fields. Those reads carry
the same defect. This change gives them the same time code. It does not merge
them into the owner, because they keep colour and intensity separate while the
owner returns only the product.

## D1: The time parameter is required, not optional

`_light_color_radiance(light_api, time)` takes `time` as a positional parameter
with no default.

A default of `Usd.TimeCode.Default()` would compile at every existing call site
and would keep the exact defect the change removes. The failure is silent: the
render shows a plausible number, not an error. A required parameter converts
every missed site into an immediate `TypeError`.

Discarded alternative: an optional parameter plus a warning when the caller omits
it. That adds a runtime branch and a log line to hide a mistake the type
signature can prevent.

## D2: The renderer dome site passes the playback clock's time code

`renderer.py` calls `_light_color_radiance` in the environment-HDR path. That
path runs when a user loads an HDR onto an authored dome light. It has no `time`
argument of its own.

The renderer holds `self.clock`, a `PlaybackClock` constructed in `__init__`. The
call site therefore passes `Usd.TimeCode(float(self.clock.current_time_code))`.

Discarded alternative: pass `Usd.TimeCode.Default()`. That keeps the fallback
value for an animated dome and reintroduces the defect at the one site that this
change touches outside the loader.

The clock reads 0.0 for a stage with no animation. `Get(0.0)` and `Get()` agree
on an attribute that holds a default value, so a static scene keeps its current
result.

## D3: The read is fixed, the re-extraction cadence is not

Two different things could each make an animated light wrong: a read that
ignores the time code, and a light the renderer never re-reads during playback.
This change fixes the first only.

`Renderer._reextract_animated_lights` re-extracts DistantLight and SphereLight
each frame. It does not re-extract DomeLight, RectLight or DiskLight. That
exclusion is deliberate and predates this change: the per-frame path promises no
mesh rebake and no BVH rebuild, but a RectLight is carried as emissive geometry,
so refreshing its radiance means rebaking that geometry and re-uploading its
BLAS. A DomeLight would re-decode its HDR texture. Both break the promise the
per-frame path is built on.

So all five light types now read at the correct time code, and three of them
still read it only once, at stage extraction. The specification states both
facts rather than implying a per-frame refresh the renderer does not perform.

Discarded alternative: extend `_reextract_animated_lights` to all five types.
That is a much larger change — it needs a rebake-and-reupload path for animated
emissive geometry and a texture-decode cache for the dome — and it belongs to
whichever change adds animated area-light support, not to a fix for a
time-blind read.

## D4: The load evaluates at the stage start time code

A time code threaded correctly is still wrong if its value is wrong.
`_read_open_stage` defaulted `eval_time` to `Usd.TimeCode.Default()`, so a normal
load read every light at the default time code and got the schema fallback for a
time-samples-only attribute — the exact defect, one level up.

Distant and sphere lights recovered on the first playback frame. Dome, rect and
disk lights never re-extract, so for them the fallback was permanent: the fix was
unreachable through the normal load path.

`eval_time` now defaults to `Usd.TimeCode(stage.GetStartTimeCode())`.
`build_playback_clock` sets `current_time_code=start`, so the load now agrees
with the first rendered frame instead of disagreeing with it.

This changes the evaluation time for every attribute the load reads, not only
lights. That is the point: one evaluation time for the whole read is what makes
the loaded scene self-consistent. A stage with no time samples resolves
identically either way, so every static scene — the whole parity corpus and the
confirming suite — is unaffected. The full hostless suite shows the same failure
set before and after.

Discarded alternative: special-case the light reads to use the start time code
while the rest of the load stays at `Default()`. That puts a second evaluation
time inside one read, so a light and the geometry it illuminates could resolve at
different times.

## D5: No behaviour change without time samples

`Attribute.Get(time)` equals `Attribute.Get()` when the attribute holds an
authored default value or only a schema fallback. USD resolves a time code
against time samples first, then the default value, then the fallback. An
attribute with no time samples ignores the time code.

Every existing scene without animated light attributes therefore renders
identically. The parity matrix and the confirming-scene suite need no baseline
change.

## D6: The test asserts variation, not a fixed value

The hostless test authors an intensity with two time samples. It extracts the
light at both time codes and asserts that the two radiance values differ, and
that each matches its authored sample.

A test that asserts one absolute number passes against the schema fallback if the
sample count is one. Asserting variation across time codes fails on the defect
directly. The test also pins each value, so a fix that returns a wrong-but-
varying number still fails.

## Fixture re-capture

`tests/fixtures/anim_reread.json` records the pre-fix extraction for the
`scene-intake-interface` change. That change requires per-frame values identical
to the pre-change extraction, so it must keep the defect. After this change lands
on `main`, re-capture the fixture:

```bash
PYTHONPATH=src .venv/bin/python tests/fixtures/_capture_anim_reread.py
```

The re-capture belongs to `scene-intake-interface`, not to this change. This
change does not edit that fixture.
