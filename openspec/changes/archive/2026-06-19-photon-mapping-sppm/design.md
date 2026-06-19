## Context

skinny renders caustics — specular→diffuse transport such as a highlight
focused through a glass object onto a diffuse floor — only via its path tracer
and BDPT, both of which converge slowly there (the archived path↔BDPT
convergence change notes caustic variance as expected and defers "Adding SPPM"
to a separate change; this is that change). pbrt v4, skinny's parity reference,
provides `Integrator "sppm"` for exactly this regime. skinny's pbrt→USD
importer parses the `sppm` directive into `customLayerData["pbrt"]` but maps it
to no renderer behavior, so imported sppm scenes render on the path tracer.

The renderer already has the scaffolding SPPM needs: a wavefront execution mode
with a staged kernel driver, per-stage queues, a GPU suballocator, progressive
accumulation with a state-hash reset, power-weighted emissive-mesh / light
importance CDFs (reusable as the photon emission distribution), and a native
Metal backend at wavefront parity with Vulkan. SPPM is added as a third
integrator that reuses this machinery rather than a parallel pipeline.

This design covers **PM-1** only: surface SPPM on flat materials, both backends
(Metal-first), the importer mapping, and the caustic parity gate.

## Goals / Non-Goals

**Goals:**
- A GPU Stochastic Progressive Photon Mapping integrator selectable as
  `INTEGRATOR_SPPM`, wavefront-only, reusing the existing staged driver.
- Correct, **consistent** (asymptotically unbiased as radius→0) caustic
  rendering on flat materials, verified against pbrt v4 `sppm` by the parity
  harness on both Metal and Vulkan.
- pbrt importer recognizes `sppm`, carries its parameters to USD metadata, and
  auto-selects the skinny SPPM integrator on load.
- Native **Metal** is the primary development/verification backend; **Vulkan**
  reaches parity within the same change.
- A `photon-mapping` capability spec written so PM-2 (skin/BSSRDF) and PM-3
  (volumetric) extend it without rewriting the surface contract.

**Non-Goals:**
- Skin / BSSRDF photon deposition (PM-2) and volumetric / media photon
  transport — volume photons, beam-radiance estimate (PM-3).
- A megakernel SPPM path. SPPM's global visible-point/photon structure does not
  fit the per-pixel megakernel; sppm is wavefront-only like ReSTIR / neural.
- Final-gather, classic kd-tree photon maps, photon-mapping-based GI for the
  whole image, or replacing NEE direct lighting. Direct lighting stays on NEE;
  SPPM contributes the indirect/caustic term.
- Bidirectional or MIS-weighted photon/eye connection beyond the SPPM estimator.

## Decisions

### D1 — SPPM (stochastic), not PPM or classic 2-pass photon mapping
Each accumulation pass re-traces a fresh stochastic camera path per pixel to
produce that pass's visible point, so a single per-pixel visible-point slot
suffices and the whole image — not a fixed hit-point set — converges. This
matches pbrt v4 `sppm` 1:1 (so importer mapping is exact) and folds into
skinny's existing per-frame progressive accumulation: one SPPM pass == one
accumulation frame. *Alternatives:* PPM (fixed camera hit points + a separate
hit-point buffer that never re-traces — wastes memory, drifts from pbrt, worse
fit to per-frame accumulation); classic 2-pass with a full kd-tree photon map +
final gather (biased, large storage, GPU kd-tree build is painful, not what
pbrt v4 ships).

### D2 — Per-pass pipeline as four wavefront stages
1. **Eye stage** — trace the camera ray through specular/perfectly-glossy
   bounces until the first non-specular (diffuse/rough) flat-material hit; write
   one **visible point** per pixel: `pos, ns, beta (path throughput to the
   point), packed BSDF params, materialId, radius r, photon count N,
   accumulated flux tau, per-pass flux accumulator Phi, per-pass count M`. Pixels
   whose camera path escapes or dies before a non-specular hit store an inactive
   visible point and contribute only direct/escape radiance this pass.
2. **Grid-build stage** — a uniform spatial hash over visible-point positions,
   cell size = current max search radius. Built by **counting sort** (count
   per cell → exclusive prefix sum → scatter indices), not an atomic linked
   list: deterministic, and the only atomics are integer adds that both targets
   support cheaply.
3. **Photon stage** — emit photons from lights using the existing
   power-weighted emissive-mesh / light CDFs (binding 18 et al.) for emission
   importance; trace through the scene with Russian roulette; at each
   non-specular interaction, look up the grid cell(s) the photon falls in and,
   for each visible point within its radius, atomically add the photon's flux
   contribution into that point's `Phi` and increment `M`.
4. **Update stage** — per visible point apply the SPPM radius/flux reduction
   (`N' = N + γM`, `r' = r·sqrt(N'/(N+M))`, `tau' = (tau+Phi)·(r'/r)²`, γ=2/3),
   then resolve the pixel's radiance estimate `tau / (Nₑₘᵢₜₜₑₐ·π·r²)` and
   composite with the NEE direct term into the accumulation image.

*Alternatives:* a single fused mega-stage (can't share the global grid across
pixels); atomic-linked-list grid (nondeterministic ordering, more divergent
Metal atomics).

### D3 — Reuse the staged wavefront driver, suballocator, and light CDFs
New buffers (visible points, grid cell offsets + sorted indices, photon
records/queues) are allocated through the existing suballocator and dispatched
through the existing wavefront stage driver. Photon emission reuses the
emissive-mesh / light importance CDFs already built for NEE rather than a new
distribution. This keeps SPPM a set of additional stages, not a parallel engine.

### D4 — Metal-first, with buffer fold + readback compaction up front
The dev host resolves `auto`→Metal, so Metal is built and verified first;
Vulkan parity follows in the same change. Two Metal constraints are designed in
from the start, not retrofitted:
- **31-slot argument-table cap.** The new buffers are **folded** into a small
  number of combined buffers (the graph-param-fold lesson: the Metal buffer
  index is parameter order, not `vk::binding`, so the fix is reducing buffer
  *count*, e.g. one combined SPPM-state buffer + one grid buffer + one
  photon-record buffer), and a binding-count budget is tracked in the binding
  map before any kernel is wired.
- **Per-pass queue compaction** uses the Metal CPU-readback fallback the
  wavefront driver already uses for indirect dispatch slot counts (Metal
  slang-rhi indirect dispatch is a no-op today). This adds readbacks per pass on
  Metal; accepted for PM-1 and documented.
Vulkan uses GPU indirect dispatch and `VK_KHR` device atomics directly.

### D5 — Integrator selection mirrors the existing path/bdpt seam exactly
`INTEGRATOR_SPPM = 2u` in `common.slang`; `INTEGRATOR_INDEX["sppm"] = 2` and
`--integrator sppm` in `cli_common.py`; GUI mode list gains "SPPM";
`integratorType` packed into `FrameConstants` drives dispatch. `sppm` is
wavefront-only: a megakernel/`--execution-mode megakernel` selection refuses
with a clear usage error (the render-cli gating pattern), matching how the
neural proposal is wavefront-gated. The accumulation state-hash includes the
integrator index, so switching to/from sppm resets accumulation cleanly.

### D6 — Importer maps sppm to metadata + integrator selection
`state.py` already stores the integrator tuple; `emit.py`/`metadata.py` are
extended so that when the integrator is `sppm`, its params (`numiterations`,
`maxdepth`, `photonsperiteration`, `radius`, `seed`) are written to USD metadata
**and** the stage records the skinny SPPM integrator as the selected integrator
(the same mechanism RenderOptions/parity uses). `report.py` reports sppm as
*mapped* (surface case) rather than *skipped*. The initial search radius uses
the pbrt `radius` parameter when present, else a scene-bbox-derived heuristic.

## Risks / Trade-offs

- **Metal 31-slot argument cap blown by new buffers** → Fold visible-point /
  grid / photon buffers into combined buffers and budget the binding count in
  `docs/Architecture.md` before wiring any kernel (D4). Verified by a
  Metal-compile smoke test with SPPM active alongside ≥2 graph materials.
- **Per-pass CPU-readback compaction slows Metal** → Accepted for PM-1 and
  documented in the compatibility matrix; the equal-time comparison is reported,
  not gated. Vulkan keeps GPU indirect dispatch.
- **Atomic flux deposit contention / precision** on hot visible points →
  Counting-sort grid bounds per-cell work; deposit accumulates in fp32 with the
  SPPM `(r'/r)²` rescale applied once per pass in the update stage, not per
  photon, to limit error growth.
- **SPPM is consistent, not unbiased at finite radius** → The parity gate uses
  relMSE / FLIP thresholds at a fixed pass budget (matching pbrt's own finite
  radius), not exact equality; an energy-ratio check guards gross bias. The spec
  requires the estimate to *converge* as radius shrinks, tested by a
  radius-sweep trend rather than a single-frame equality.
- **Caustic emission importance** — uniform light selection makes caustic
  photons noisy → Reuse the power-weighted emissive/light CDFs already proven by
  the emissive-mesh NEE change for photon emission.
- **Scope creep into skin/volume** → The capability spec fences PM-1 to surface
  flat materials; skin/volume requirements are written as explicitly deferred
  phases so reviewers can reject any flat-path change that reaches into them.

## Migration Plan

Additive and default-off. The default integrator stays `path`; `sppm` is opt-in
via `--integrator sppm` / GUI. No existing scene output changes (the
accumulation byte-identity baseline from scene-sampling still holds for
path/bdpt). Rollback = revert the change; no persisted-settings migration beyond
a new allowed `integrator` value. Ships Metal-then-Vulkan but both land in this
change, so there is no intermediate release with only one backend.

## Open Questions

- **Photons per pass vs pixels.** Start with `photonsperiteration` from pbrt
  when importing, else a fixed default scaled to resolution. Whether to decouple
  photon count from pixel count for interactive cadence is a tuning question for
  implementation, not a spec requirement.
- **Glossy (non-perfect-specular) visible-point cutoff.** PM-1 stores the
  visible point at the first surface whose roughness exceeds a threshold;
  the exact threshold (and whether very-rough specular is treated as diffuse for
  storage) is calibrated against the parity scene during implementation.
- **Shared grid cell size strategy** — single global cell size from the max
  radius vs per-level grid. PM-1 uses a single global cell size; a multi-res
  grid is a possible PM-1 follow-up if hot cells dominate, not required by the
  spec.
