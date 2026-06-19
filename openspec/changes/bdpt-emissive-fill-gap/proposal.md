## Why

On `bathroom.usda` (pbrt-imported, ~1.5k emissive triangles forming the area
lights) the BDPT integrator rendered **~0.8× the path tracer's brightness** in
default mode, with the broad interior *fill* almost entirely missing (linear-HDR
median ratio bdpt/path ≈ 0.005 — the interior reads near-black while the
directly-visible emissive panels matched). The gap was originally filed as an
"env-fill" gap, but runtime instrumentation proved otherwise:

- **Environment transport is already matched** — decomposing BDPT's env
  contribution (direct z1 NEE / deeper-vertex NEE / escape ray) against the path
  tracer's per-bounce env NEE gives a total ratio of **0.994**. Env is not the
  cause.
- **The s=1 light-tracer splat is negligible** here (0.2 % of BDPT's total), so
  the accumulation-vs-display measurement difference does not explain the gap.
- The gap is in **emissive area-light transport**, and it is a genuine **bias**
  (it does not shrink as samples grow: still ~0.8× at 512 spp).

Root cause: the `emissive-mesh-nee` change made emissive-triangle selection
**power-weighted** (`p_i = w_i / Σw`, with `pdfArea = p_i / triArea`), and updated
the path tracer's `nee.slang` to draw through the matching `sampleEmissiveTriangle`
CDF. BDPT's eye-side NEE (`connectT1` in `bdpt.slang`, shared by the megakernel
and the wavefront backend) was **left selecting uniformly by index**
(`rng.next() * numEmissiveTriangles`) while still dividing by the power-weighted
`pdfArea`. Selecting with one distribution and dividing by another's pdf is
biased: bright panels (high `p_i`, rarely drawn at `1/N`) are under-counted, so
the indirect emissive fill comes out far too dark. On the single-quad
`diffuse_arealight` corpus scene `N` is tiny and uniform ≈ power-weighted, so the
existing gates never caught it.

## What Changes

- **BDPT `connectT1` selects emissive triangles power-weighted**, through the
  same `sampleEmissiveTriangle` cumulative-power CDF the path tracer uses, so the
  draw matches the `pSel`-based `pdfArea` that `samplePoint` reports. This makes
  the eye-side emissive NEE an unbiased estimator identical to `path.slang`'s, and
  fixes both the megakernel and the wavefront BDPT (they share `connectT1`).
- No change to the path tracer, to env transport, to the s=1 splat, or to the
  emissive-MIS partition (`misWeight`) — the fix is purely the selection
  distribution at the `connectT1` emissive draw.

Result on `bathroom.usda` (Metal, default mode): bdpt/path mean **0.79 → ~0.97**,
median **0.005 → 0.66**, and the rendered image is visually indistinguishable
from the path tracer (was near-black interior). All BDPT/emissive gates stay
green (`test_bdpt_energy`, `test_emissive_nee`, `test_convergence`,
`test_metal_wavefront_bdpt_ab`).

## Known residual (out of scope here)

A smaller residual remains in the *median* (bdpt/path ≈ 0.66): BDPT's `misWeight`
partition assigns part of the emissive-NEE weight to the t≥2 light-walk
connections, but `sampleLightOrigin` still selects emissive triangles uniformly
and its light-origin area pdf omits the per-triangle selection probability, so
those connections under-deliver their share and the redistributed energy is
partially lost. Making the light subpath power-weighted **and** MIS-consistent
(across `sampleLightOrigin`, `connectGeneric`, and `splatMisWeight`) is a larger,
corpus-risking change; it is deliberately left as a follow-up. The mean is at
parity and the image matches; the residual is not visually significant.
