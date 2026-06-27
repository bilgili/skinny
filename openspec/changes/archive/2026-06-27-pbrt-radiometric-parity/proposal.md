## Why

Two findings drive this change.

**1. pbrt film params are baked at import, so they cannot be changed on the fly.**
The importer collapses pbrt's film exposure into one scalar
`exposure_scale = exposureTime · ISO / 100` (`api.py:_film_exposure_scale`) and
**bakes it into emitter and environment radiance** at import time
(`api.py:75/233`, `lights.py:58`). The `UsdGeom.Camera` carries geometry only
(focalLength, apertures, fStop, focusDistance, transform, `skinny:lens:*`,
provenance) — **no `iso`/`exposure`/`shutter`**. The renderer's `self.exposure`
is a *display* EV applied before tonemap, not the linear film gain. So ISO and
exposure are frozen into the scene at import and there is no scenegraph knob to
retune them live.

**2. Absolute radiance does not match pbrt, and the parity gate never noticed.**
The gate is exposure-blind (relMSE/FLIP aligned), so absolute brightness drifted.
Measured vs pbrt: `diffuse_arealight` 1.63×, `glass_arealight` 1.57× (area-lit, no
ISO), `conductor_infinite` 0.95× (env-lit, close), sssdragon ~2× (env-lit
subsurface). The earlier hypothesis ("imaging ratio missing on the env") was
**wrong** — the ratio is already baked into emitters and into *converted*
`.exr`/`.pfm` env maps. The real gaps are narrower:

- **`.hdr`-direct env drops the scale** (`lights.py:98-100`): a pbrt infinite
  light already pointing at a `.hdr` is referenced verbatim, so its `scale` (pbrt
  `scale` × imaging ratio) is silently lost.
- A **light-type-dependent absolute offset** independent of exposure: area-lit
  scenes ~1.6× bright, env-lit metal ~1.0×. This points at the emissive-triangle
  radiance / NEE normalization, not the film gain.

## What Changes

- **Film params become live scenegraph camera attributes.** Author `skinny:film:iso`
  + `skinny:film:exposureTime` (and the standard `UsdGeom.Camera` `exposure`) on
  the camera prim. The renderer reads them, computes the imaging ratio
  `exposureTime · ISO / 100`, and applies it as a **live linear output scale**
  (re-read each frame, like the display exposure) — **instead of baking** it into
  emitters/env. For a linear path tracer, baking a scalar into every emitter is
  algebraically identical to scaling the output, so already-correct renders are
  unchanged; the difference is (a) it is now editable on the fly and (b) the
  `.hdr`-direct env no longer loses its `scale`.
- **Expose ISO + exposure in the UI.** Add them to `ALL_PARAMS` and the
  USD-driven control surface so they retune live and round-trip to the scene.
- **Fix the `.hdr`-direct env scale drop** — the pbrt `scale` rides the live film
  scale (imaging ratio) or the DomeLight intensity, never dropped.
- **Diagnose + fix the light-type absolute offset** (area-lit ~1.6×): cross-render
  a diffuse-white patch under a known area light vs pbrt, find the emissive /
  NEE radiance factor, fix at its source.
- **Absolute-radiance gate.** Add an un-exposure-aligned mean-ratio + relMSE gate
  to the parity matrix beside the exposure-blind one, and re-record `measured`
  baselines once the calibration lands.

## Non-Goals

- The subsurface walk itself (already structurally correct).
- Tonemapping / display-sRGB output and the existing display-EV slider.
- A per-scene fudge scalar — the fix is principled (imaging ratio + emissive
  radiance), not a tuned constant.
- Spectral-vs-RGB residuals (a known, smaller inherent difference vs pbrt).
- Animating film params over time (static authored values; live edit only).
