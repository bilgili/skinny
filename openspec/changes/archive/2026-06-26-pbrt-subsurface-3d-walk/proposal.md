## Why

After the unit-scale, env-orientation, and walk-energy fixes the pbrt sssdragon
renders translucent, correctly lit, and energy-conserving — but still dimmer and
redder than pbrt's dipole BSSRDF. The remaining error is the **1D-slab geometry
approximation**: the interior walk models the medium as a flat slab of
perpendicular thickness (`hit.backT − hit.t`), which mis-estimates path length on
a curved/complex mesh, so paths travel too far through the blue-absorbing Skin1
medium → redder and darker.

A throwaway prototype (committed on the WIP branch — `subsurfaceRadiance3D` in
`subsurface_walk.slang`) replaced the slab with a **true 3D interior walk**:
import `scene_trace`, and at each scattered vertex `traceScene(fc, ray)` to the
real mesh boundary, delta-track that segment, Fresnel-split at the boundary
(transmit→env / reflect→continue). Measured results:

- **Furnace energy** (non-absorbing sphere, white env, target ~1.0), τ≈20:
  η = 1.0 / 1.2 / 1.5 → **0.94 / 0.91 / 0.83** vs the slab's 0.80 / 0.72 / 0.59 —
  near-conserving and η-independent (the slab's residual loss was the geometry).
- **sssdragon**: no GPU-watchdog hang on the 28.8M-tri mesh in **wavefront** even
  with per-segment BVH traces (cap 16, ~83 s — same wall-time as the slab; the
  trace cost amortizes). Mean 0.097 (slab, cap 64) → **0.108** (3D, cap 16); a
  higher cap closes more of the gap toward pbrt's 0.211.

The prototype dropped distant-light NEE and ran only on wavefront; this change
hardens it into a correct, gated, backend-aware estimator.

## What Changes

- **Wavefront subsurface uses the 3D interior walk.** Promote
  `subsurfaceRadiance3D` to the production estimator for the wavefront path:
  per-segment `traceScene` to the real boundary instead of the slab.
- **Megakernel keeps the 1D-slab.** A per-segment trace inside the megakernel
  **single-dispatch** is the real watchdog/reboot risk (the slab was chosen to
  avoid exactly this). Gate by execution mode: wavefront → 3D walk, megakernel →
  existing slab. (The dispatch in `path.slang evaluateBounce` selects which; pass
  a flag or branch on an execution-mode define.)
- **Restore distant-light NEE in 3D.** The slab did per-scatter NEE to the first
  distant light through the front face. The 3D version must shadow-trace from the
  scatter vertex toward `lightDir` to the boundary, attenuate by the medium
  transmittance over that distance × exit Fresnel, so distant-lit subsurface does
  not regress. (sssdragon is env-only, so NEE is exercised by a distant-lit gate.)
- **Exit refraction + cap tuning.** Refract the escape direction (Snell) for the
  env lookup instead of using the raw interior direction; pick a final
  `SSS_MAX_BOUNCES` that maximizes energy within the wavefront watchdog (16 was
  safe and already beat slab-cap-64; test 32/48 for headroom).

## Non-Goals

- A dipole/diffusion BSSRDF (pbrt's analytic approach) — out of scope; this keeps
  the unbiased volumetric walk.
- Area/emissive lights *inside* the medium (pre-existing follow-up).
- Megakernel 3D walk (kept on the slab by design for watchdog safety).
