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

## D3: No behaviour change without time samples

`Attribute.Get(time)` equals `Attribute.Get()` when the attribute holds an
authored default value or only a schema fallback. USD resolves a time code
against time samples first, then the default value, then the fallback. An
attribute with no time samples ignores the time code.

Every existing scene without animated light attributes therefore renders
identically. The parity matrix and the confirming-scene suite need no baseline
change.

## D4: The test asserts variation, not a fixed value

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
