## Why

`assets/sss_dragon_small.usda` (and its sibling `dragon_sss.usda`, the parity
corpus `dragon` scene) render the subsurface dragon **solid black** under the
wavefront **BDPT** and **SPPM** integrators, while the megakernel (path + BDPT)
and wavefront **path** all render it correctly. The corpus already parked this as
`"known_divergent": true` with empty baselines ("subsurface residual… follow-up")
— it was never actually shading in wavefront BDPT/SPPM.

The dragon's `shape_0` is `MATERIAL_TYPE_SUBSURFACE` (a non-flat material — the
volumetric interior random walk). The **megakernel** handles non-flat materials
under BDPT by routing those pixels to the **path tracer**, never to BDPT:

```slang
// main_pass.slang
bool useBdpt = (fc.integratorType == INTEGRATOR_BDPT) && hit.hit
            && ((materialTypes[hit.materialId] & 0xFFu) == MATERIAL_TYPE_FLAT);
if (useBdpt) { /* BDPT */ } else if (hit.hit) { PathTracer().estimateRadiance(...); }
```

So the megakernel's `BDPTIntegrator.estimateRadiance` non-flat "bail to black"
branch is **dead code** — it is never reached, because `main_pass` diverts
non-flat hits to the path tracer first.

The wavefront BDPT/SPPM eye passes copied the *bail* without the *fallback*:

- `wavefront_bdpt.slang` (`wfBdptGenEye`): a non-flat first hit returns early with
  `aux.escaped` left at 0 → **black**. Its comment claims "Matches
  estimateRadiance" — but the true megakernel behaviour is the path fallback in
  `main_pass`, not the (unreachable) bail.
- `wavefront_sppm.slang` (eye pass): a non-flat eye hit stores **no visible
  point**, so the pixel receives only its (near-zero) direct term → **black**.

Wavefront **path** is unaffected because it already routes non-flat lanes through
the `WF_SLOT_OTHER` catch-all shade (`evaluateBounce` → `subsurfaceRadiance3D`).

This is a wavefront↔megakernel parity gap, not a subsurface-transport bug. (It is
distinct from the recorded BDPT/SPPM **volume** exclusion: free-standing
`MATERIAL_TYPE_VOLUME` media legitimately have no BDPT/SPPM transport. Subsurface
is different — the megakernel resolves it by path fallback, and wavefront must
too.)

## What Changes

- **Wavefront BDPT** (`wfBdptGenEye`): replace the non-flat-first-hit bail with a
  path fallback — `aux.escaped = PathTracer().estimateRadiance(ray, firstHit,
  rng)` — mirroring `main_pass.slang`. The lane builds no eye subpath and no light
  subpath of its own (exactly as a megakernel non-flat pixel runs no BDPT); s=1
  light-tracer splats from other lanes still composite onto it via
  `lightSplatBuffer`. Flat lanes are byte-unchanged.
- **Wavefront SPPM** (eye pass): path-integrate a non-flat eye hit
  (`PathTracer().estimateRadiance`) and add its radiance to the pixel, in addition
  to the existing direct term, since no photon visible point can be stored on the
  subsurface walk. Flat eye hits are unchanged.
- Both fallbacks are the wavefront-compiled path tracer (`SKINNY_WAVEFRONT`), so
  subsurface uses the same `subsurfaceRadiance3D` per-segment interior walk that
  wavefront path already ships — no new transport code, same watchdog envelope as
  wavefront path on this dragon.
- **Scoped to the terminal non-flat types (`SUBSURFACE`, `SKIN`).** Their
  `evaluateBounce` returns a complete radiance with no continuation, so
  `estimateRadiance` evaluates one vertex and stops — a single terminal eval, the
  same bounded dispatch shape as the wavefront path catch-all shade. The
  non-terminal types (`VOLUME`, `PYTHON`) keep bouncing inside `estimateRadiance`;
  running that multi-bounce loop in the single once-per-frame wavefront command
  buffer (no band-tiling, unlike the megakernel) would be an **unbounded command
  buffer** under `SKINNY_METAL` (metal-dispatch-hygiene). They therefore keep the
  black bail under wavefront BDPT/SPPM (unchanged — still rendered by the
  megakernel path); lifting that is a follow-up gated on a bounded fallback
  dispatch. VOLUME under BDPT/SPPM is already a recorded exclusion; PYTHON under
  wavefront BDPT/SPPM was already black, so neither regresses.
- **Parity corpus**: the `dragon` scene stops being all-black under wavefront
  BDPT/SPPM. Re-measure and record the per-combo baselines; the scene's
  `known_divergent` disposition is updated to reflect that it now shades (the
  remaining subsurface-vs-pbrt residual, if any, stays a separate follow-up — this
  change only removes the black-out).
- **Docs**: `docs/Wavefront.md` (BDPT/SPPM staging: non-flat lanes fall back to the
  path tracer) and the `CLAUDE.md` / `README.md` compatibility matrix note.

Metal/Vulkan: no backend-specific code; any adaptation stays behind
`#if defined(SKINNY_METAL)`. Vulkan SPIR-V for flat-only scenes is byte-stable.
