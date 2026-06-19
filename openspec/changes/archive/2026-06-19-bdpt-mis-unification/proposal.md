## Why

After `bdpt-energy-convergence` (the t=0 emissive eye-hit gate), BDPT's **accumulation**
matches the path tracer, but its **display** (what the app shows) does not: BDPT
reads brighter than the path tracer — measured on `bathroom.usda` (×1.19) and on
the pure-diffuse corpus scene (display ×1.117, no caustics ⇒ pure double-count).

Two independent causes, both from BDPT's MIS being a *hybrid* rather than one
partition:

1. **The s=1 light-tracer splat** (`splatLightVertex`) was added at full weight,
   double-counting every diffuse path the eye subpath already captures. It is
   composited only on the display (`main_pass.slang`), never into `accumBuffer`,
   so all accum-based gates were blind to it. *(Fixed first — see Tasks §1.)*

2. **`connectT1` (the t=1 NEE) uses a standalone 2-strategy power heuristic**
   (`powerHeuristic(pdfLightSA, bsdfPdf)`), while `connectGeneric` (t≥2) uses the
   full `misWeight`. For an indirect path the t=1 and t≥2 strategies then do not
   partition: t=1 is over-weighted (it does not divide by the t≥2 alternatives),
   so BDPT's *eye-side accumulation itself* runs ~2% bright on indirect transport.

Net: the BDPT estimator is not a single MIS partition, so it does not converge to
the (pbrt-matching) path tracer in absolute energy.

## What Changes

- **MIS-weight the s=1 splat** (`splatMisWeight` = `misWeight` specialised to s=1,
  light-side ratios only, with the camera as the eye endpoint). Splat keeps full
  weight only where the eye side is zero (specular caustics on a directly-seen
  diffuse surface), ≈0 where the eye side already has the path. *(Pinhole `We`;
  thick-lens `We` abstraction is a tracked follow-up.)*
- **Rewrite `connectT1` (t=1 NEE) onto `misWeight`** for emissive-triangle area
  lights: build the sampled light point as `lit[0]` (pdfFwd = light area pdf),
  compute the endpoint reverse pdfs (eye BSDF → light = `pdf_ytm1_rev`; light
  emission dir → z = `pdf_zsm1_rev`; z BSDF → z₋₂ = `pdf_zsm2_rev`), and weight
  with `misWeight(eye, s, {lit0}, 1, …)`. This puts t=1 in the same partition as
  t=0 (gated), t≥2 (`connectGeneric`), and the s=1 splat, so the weights sum to 1
  and the indirect over-count cancels. The light endpoint stays an **emitter**
  (radiance Le, geometry cos — no 1/π reflector term).
- **t=0 emissive eye hit** keeps its gate (NEE-only for non-primary, non-delta) —
  already consistent with the path tracer; the `s+t==2` primary and `isDelta`
  cases are exactly what `misWeight` would return, so no change needed there.
- **Display-based convergence gate**: a new test asserting BDPT's *display* (incl.
  splat) mean energy ≈ the path tracer's on the diffuse corpus scene (the accum
  gates cannot see the splat).

Scope: `shaders/integrators/bdpt.slang` (+ wavefront via the shared helpers).
Sphere / directional / env NEE keep their current 2-strategy MIS (the area-light
corpus + bathroom use emissive triangles); unifying those is out of scope here.

## Impact

- Specs: `integrator-convergence` (ADDED: BDPT display converges to the path
  tracer; single MIS partition across t=0/t=1/t≥2/s=1).
- Code: `bdpt.slang` (splat + connectT1), `tests/pbrt/test_bdpt_energy.py` (or a
  new display gate). No path-tracer change. Accum gates must stay green.
- Risk: MIS bias if the endpoint pdfs are wrong — gated by the display
  convergence test + the existing relMSE/FLIP parity gates.
