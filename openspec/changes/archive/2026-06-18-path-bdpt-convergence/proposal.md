## Why

The unidirectional path tracer and the bidirectional path tracer (BDPT) are both
unbiased Monte-Carlo estimators of the same rendering equation, so for any scene
they MUST converge to the **same** image (differing only in noise). They do not.
On `glass_arealight` (a smooth dielectric sphere lit by an area light), the path
tracer permanently misses the **caustic** under the sphere and the **reflection
of the area light** on the sphere — features BDPT (and pbrt's reference) render.
Measured mean brightness diverges (path 82.5 vs BDPT 110): the path tracer is
**biased**, not merely noisier.

Root cause is in the path integrator's emitter accounting
(`shaders/integrators/path.slang:171`):

```slang
if (bounce == 0u || fc.numEmissiveTriangles == 0u)
    radiance += throughput * br.bsdfSample.emission;
```

Emissive-triangle (area-light) emission hit by a BSDF-sampled ray is added only
at the primary bounce; at all later bounces it is left to next-event estimation
(NEE). But NEE **cannot sample a delta (perfectly specular) lobe** — a smooth
dielectric's reflect/refract directions. So when a camera ray specularly
reflects or refracts off the glass and lands on the area light, NEE contributes
nothing *and* the BSDF-hit emission is skipped: the contribution is dropped
entirely. The same gap kills the caustic path (camera → floor → BSDF ray →
glass refract → light). BDPT captures these via light-subpath connections and
already flags delta bounces correctly (`bdpt.slang` `deltaBounce`), so the two
estimators diverge. Notably the **sphere-light** branch a few lines below already
handles the delta case (`w_bsdf = 1` for `pdf == 0`/transmitted) — only the
emissive-triangle path is missing it.

## What Changes

- **Fix the specular→emitter bias** in the path integrator: a BSDF-sampled ray
  that hits an emissive triangle after a **delta/specular** bounce SHALL add that
  emission at full weight (the delta lobe has no NEE partner), mirroring the
  existing sphere-light delta handling and BDPT's `deltaBounce` accounting.
  Non-delta bounces keep their current NEE-based accounting (unbiased already).
- **Apply to every path-integrator surface**: the megakernel path
  (`path.slang`), the record-dump variant used for neural training
  (`path_record.slang`, same line), and the wavefront path (which shows the
  identical mean and bias). Recompile `main_pass.spv`.
- **Regression test**: a headless A/B test asserting path and BDPT **converge to
  the same image** on `glass_arealight` within a tolerance (relMSE / mean-energy)
  at a fixed high spp — guarding against future divergence.
- **Doc note** on the path tracer's specular-emitter MIS rule.

This is a renderer correctness fix; it does not change BDPT (the reference) and
leaves non-specular transport bit-unchanged.

## Capabilities

### New Capabilities
- `integrator-convergence`: The path tracer and BDPT SHALL be energy-consistent —
  for the same scene and convergence they produce the same expected image. Covers
  the specular→area-light emission rule (delta bounces hitting emissive triangles
  contribute at full weight, no double-count with NEE) and the A/B convergence
  regression.

### Modified Capabilities
<!-- None. The fix is additive correctness within the path integrator; no
     existing spec governs the megakernel path tracer's MIS/emitter rule. -->

## Impact

- **Shaders:** `shaders/integrators/path.slang`, `shaders/integrators/path_record.slang`,
  and the wavefront path integrator's emission accumulation; recompile
  `main_pass.spv` via `slangc` (runtime-compiled on the headless/Vulkan path).
- **Reuses unchanged:** `bdpt.slang` (already correct — the reference), `nee.slang`
  (delta lobes correctly produce no NEE partner).
- **Tests:** new headless A/B convergence test (path vs BDPT on `glass_arealight`,
  and a furnace/energy check that non-specular scenes are unchanged).
- **Docs:** `docs/Megakernel.md` / `docs/Wavefront.md` (path-tracer specular-emitter
  MIS rule); `CHANGELOG.md`.
- **Out of scope:** caustic *variance* (the fix makes the path tracer unbiased but
  it stays noisier than BDPT/SPPM on caustics — converging slowly is expected);
  adding SPPM; the pbrt importer.
