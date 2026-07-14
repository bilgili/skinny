# Fix SPPM env-direct under-count (terminal-VP MIS companion)

## Why

SPPM (`--integrator sppm`, wavefront) renders ~15–25% dimmer than path/bdpt on
env-lit scenes (path≈bdpt = ground truth). Root-caused by decomposition on an
**env-only** scene (a diffuse ground + box under just the dome; 256², same
spp=1024, firefly-robust median), where a flat plane's value is almost pure env
**direct** (analytic ≈ albedo·L):

| region | sppm/path (before) |
|--------|--------------------|
| flat ground (open, ~pure env direct) | **0.735** |
| whole | 0.749 |

**Confirmed root cause:** the SPPM eye **terminates at the first diffuse visible
point** (`wfSppmEye` returns after `sppmStoreVisiblePoint`). At that vertex it
computes direct lighting with the shared, **MIS-weighted** `allLightsNEE` — the
env NEE is weighted by `powerHeuristic(envPdf, bsdfPdf)`, expecting a
BSDF-sampled env-miss companion. A *continuing* path adds that companion at its
next bounce (the loop-top env-miss term); **SPPM stops here** (photons carry the
indirect), so the companion never fires and the env NEE is down-weighted **with
no counterpart** → env direct is under-counted.

The deficit is env-**direct**, not indirect:
- A probe forcing the env NEE to **full weight** lifts env-only sppm/path
  0.735 → **0.998**.
- Turning the env off leaves sppm/path ≈ 1.0 (small lights: `bsdfPdf≈0` ⇒
  `powerHeuristic≈1` ⇒ NEE already full weight — which is why sphere-lit scenes
  were fine and the bug only shows under a **broad** env).

This is distinct from the merged `sppm-photon-dispatch-tiling`, and not a
normalization/convergence issue.

## What Changes

- **Add the env-miss MIS companion at the terminal visible point** in
  `wfSppmEye`. Right before storing the VP, trace one BSDF sample; if it escapes
  to the environment, add `throughput · bsdfWeight · L_env · powerHeuristic(
  bsdfPdf, envPdf)` — mirroring the loop-top env-miss term that a continuing path
  would contribute. Env direct then = NEE + BSDF-miss, exactly as `path.slang`
  computes it. Gated `bs.pdf > 0 && !bs.transmitted && furnace == 0`; the eye's
  MIS-weighted env NEE and the glossy-continue caustic walk are unchanged.
- Spectral mirrored under `#if defined(SKINNY_SPECTRAL)` (spectral bounce weight
  `flatResponseS/bs.pdf`, `upsampleIlluminantBound`).

## Impact

- Affected specs: `photon-mapping` (SPPM direct-lighting completeness).
- Affected code: `shaders/integrators/wavefront_sppm.slang` (`wfSppmEye` store
  branch only); wavefront `.spv` recompiles from source; one extra scene trace
  per terminal VP.
- Result (GPU-verified, matched spp=1024, firefly-robust median):
  env-only flat ground **0.735 → 0.998**; `glass_caustics_test` all regions
  **0.75–0.87 → 0.95–1.04**. Independent of, stacks on,
  `sppm-photon-dispatch-tiling`.
