## Why

BDPT under-shades directly-visible and one-bounce area-light emission by ~3%
versus the unidirectional path tracer — a bias, not noise (stable across 256 vs
1024 spp). On the `mat_emissive` suite scene BDPT reads mean/pbrt 0.9451 and
pbrt-truth relMSE 0.1292 where the path tracer reads 0.9743 and 0.0522. This is
the BDPT follow-up to the landed path-tracer fix (change
`emissive-triangle-bsdf-mis`, living spec `emissive-light-sampling`), whose scope
was explicitly "Path + Metal only; BDPT/SPPM = follow-up stories (kept old
emission handling)."

Root cause: BDPT's eye random walk terminating on an emissive triangle (the
`t = 0` strategy — a path built entirely from the camera) is currently **dropped
entirely** whenever a next-event-estimation partner exists. All three transports
carry the identical binary gate
`if (noNeePartner) L += throughput·emission;` (`bdpt.slang`,
`bdpt_spectral.slang`, `wavefront/wavefront_bdpt.slang`): weight 1 when no NEE
samples the light, weight 0 otherwise. The weight-0 case discards the
BSDF-sampling strategy's share of the direct-lighting estimate, leaving only
`connectT1`'s power-heuristic-weighted NEE (`< 1`), so area lights read dim.

## What Changes

- Replace the `t = 0` binary weight-{0,1} gate in all three BDPT transports with
  a proper MIS weight for the emitter-hit strategy, computed through the **same
  `misWeight()` partition** that `connectT1`'s `t = 1` NEE already uses — so
  `t=0 + t=1 + t≥2 + s=1` sum to one across every strategy that can generate the
  path. The full-weight branches (no NEE partner: `s == 2` primary hit, a delta
  bounce into the emitter, or a scene with no emissive triangles) are preserved.
- Deliberately **not** a `path.slang`-style 2-strategy `powerHeuristic(bsdfPdf,
  pdfLightSA)`: the codebase already rejected that for `connectT1` (it ignored
  the `t≥2` and `s=1` alternatives and ran ~2% bright). BDPT weights every
  strategy through `misWeight`; the `t = 0` term must too, or the partition will
  not sum to 1.
- Add one shared helper `emitterHitMisWeightT0(eye, s)` in `bdpt.slang` (beside
  `misWeight`) that reconstructs the two reverse pdfs the partition needs at the
  emitter vertex `z = eye[s-1]` and its predecessor `prev = eye[s-2]`:
  `z.pdfRev = Rec709-lum(z.emission) / emissiveTotalPower` (the per-triangle
  area cancels under the area·luminance power-weighted CDF, mirroring the path
  tracer's index-free reconstruction) and
  `prev.pdfRev = convertSAtoArea(cosOut/π, z, prev)` (the diffuse-emitter
  directional pdf). Reused by the megakernel-RGB, megakernel-spectral (via its
  existing `mirrorRgb` RGB view), and staged-wavefront transports so all three
  stay bit-identical to each other.
- Re-measure and **lower** the recorded pbrt-truth `baselines` for
  `bdpt|megakernel`, `bdpt|wavefront` and their `|spectral` variants on
  `mat_emissive` and `mat_emissive_mtlx` toward the path anchor. Harness-first,
  tighten-only: a baseline may only move down. `sppm|*` is untouched by this
  change (separate transport) and stays.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `emissive-light-sampling`: adds a requirement that BDPT's eye-subpath
  emitter-hit (`t = 0`) emission is MIS-weighted through the full `misWeight`
  partition (not dropped, not full-weight), the BDPT analogue of the existing
  path-tracer "BSDF-sampling MIS complement for emissive-triangle hits"
  requirement, across the megakernel, wavefront, and spectral transports.

## Impact

- Shaders: `src/skinny/shaders/integrators/bdpt.slang` (new helper + megakernel
  `t=0` branch), `src/skinny/shaders/integrators/bdpt_spectral.slang`
  (megakernel-spectral `t=0` branch), `src/skinny/shaders/wavefront/wavefront_bdpt.slang`
  (staged `t=0` branch). No new bindings, no `FrameConstants` change
  (`emissiveTotalPower` already exists from the path-tracer fix).
- Behaviour: BDPT megakernel ≡ wavefront preserved (RGB); spectral BDPT tracks
  its RGB sibling. BDPT direct/one-bounce area-light energy rises to match the
  path tracer (removes the ~3% dim bias); the estimator stays unbiased.
- Tests / baselines: `tests/pbrt/corpus/manifest.json` — lowered pbrt-truth
  baselines for the BDPT combos on `mat_emissive` / `mat_emissive_mtlx`. Vulkan
  megakernel `main_pass.spv` recompile is the Vulkan-side follow-up boundary
  (Metal compiles Slang in-process); validation runs on the Metal backend.
- Out of scope: SPPM and MLT (not on this branch) keep their emission handling;
  their baselines re-measure in their own changes when they rebase onto this.
