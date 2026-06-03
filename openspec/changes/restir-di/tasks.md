> RESUME (at commit 34d14bb). M2 DONE: restir/light_ris.slang reservoir-resamples
> M=8 env candidates (MIS-weighted target p̂=luminance(f·NdotL·Le·wNEE), sourcePdf=
> envPdf/T, deferred visibility) + directional NEE; restir_primary calls it.
> Converges to NEE (unbiased, option B — composes with shade's BSDF half, no shade
> gate). The reservoir is currently EPHEMERAL (built+resolved in one pass).
>
> NEXT = the REUSE (the defining ReSTIR feature; needs persistent per-pixel state):
>   1. Store the reservoir in a per-pixel buffer (descriptor set 2 on RestirDiPass)
>      + a G-buffer (pos/normal/matId/wo) — alloc renderer-side, double-buffer the
>      reservoir for temporal. Split restir_primary into fill-reservoir (write) +
>      resolve (read+shade), OR add separate passes (initial→spatial→resolve).
>   2. SPATIAL reuse (restir/spatial.slang): merge k neighbor reservoirs via
>      reservoirMerge (done+tested) with pHatInDst = the neighbor's sample
>      re-evaluated at THIS shading point (needs G-buffer) + the unbiased m_i/1-Z
>      + reconnection Jacobian (cosθ/d² ratio) + horizon/visibility domain check.
>      THE hard math. Test: lower variance (use a HIGH-CONTRAST env, gray demo env
>      understates it) + still converges.
>   3. TEMPORAL reuse (restir/temporal.slang): merge prev-frame reservoir (same
>      pixel, progressive) + M-cap.
> Also pending: M2b sphere/emissive-tri + BSDF candidates (canonical, same pattern)
> · config UBO (M_light/k/radius/M_cap/biased) · 7.x biased toggle. reservoir +
> merge cores done+tested; M1 plumbing + M2 RIS done.
>
> (older resume blocks below kept for the build map.)
>
> --- prior:
> RESUME (at commit 2bd05b8). DONE + verified:
> - Pure-math: reuse-seam (parity) · reservoir core (RIS) · reservoirMerge. 12
>   slangpy unit tests.
> - M1 PLUMBING (the hard coupled trunk): renderable ReSTIR DI wavefront vertical
>   WORKS + converges to NEE. restir/restir_primary.slang (primary-direct pass) +
>   vk_wavefront.RestirDiPass + the bounce-0 hook in WavefrontPathPass + renderer
>   wiring (reuse_modes+="ReSTIR DI", build-on-reuse=ReSTIR, capability gate,
>   reuseMode-gate) + sampling.RestirDiReuse plugin. depth-0 gate live.
>   tests/test_restir_render.py: converge-to-NEE + megakernel identity fallback.
>   reuse=none still bit-identical both backends.
>
> NEXT = M2 (reservoir RIS — the CANONICAL integration; correctness-critical,
> build with focus). NOT a one-line swap: M1 relocated the FULL allLightsNEE
> (carries the per-light MIS weight vs BSDF), so a light-only RIS would have no
> clean pdf to MIS against shade's BSDF-hits-light → BIAS. The chosen fix
> (design Decision 5) = put BSDF candidates IN the RIS AND gate shade's depth-0
> BSDF-hits-light. Build in restir_primary.slang:
>   - Candidate gen (M_light light + M_bsdf BSDF), SOLID-ANGLE measure:
>     * light candidate: pick a non-delta emitter (sphere/tri/env) ~ uniform (or
>       power pmf); u=rng.next2(); ls=light.samplePoint(pos,u) [ILight, returns
>       point/normal/radiance/pdfArea]; wi=dir(ls.point-pos); pdfSA = pdfArea·d²/
>       cosLight (area→Ω) or es.pdf for env; sourcePdf=(1/E)·pdfSA;
>       p̂=luminance(f(wo,wi)·NdotL·ls.radiance) UNSHADOWED; w=p̂/sourcePdf;
>       store LightSampleRef=(type,id, u-packed) so resolve REPRODUCES the point.
>     * bsdf candidate: mat.sample→wi; trace; if it hits a light, p̂ likewise;
>       sourcePdf = bsdfPdf(wi). (Reuses sampleBounceDirection? or mat.sample.)
>   - reservoirUpdate per candidate → reservoirFinalize → W.
>   - resolve: decode survivor (type,id,u), re-sample ls, wi; ONE shadow ray
>     V=visibleSegment; integrand_RGB=f·NdotL·radiance; direct=integrand_RGB·V·W.
>   - DIRECTIONAL lights are delta (pdfArea=0): NOT in RIS — add via plain NEE
>     (loop distantLights, shadow ray) like allLightsNEE's directional branch.
>   - GATE: also gate shade's depth-0 BSDF-hits-light (the sphere-light + env-miss
>     terms in wfFinishShade / the bounce loop) when reuseMode==RESTIR_DI, since
>     ReSTIR now owns the BSDF technique too. (M1 left it on; M2 must gate it.)
>   - Light APIs: lights/{sphere,emissive_triangle,directional}_light.slang
>     (ILight.samplePoint/pdfSolidAngle, load*Impl); environment.slang
>     sampleEnvDir/envPdf; nee.slang for the patterns. reservoir.slang done+tested.
>   Test: converges to NEE (canonical = same integral) + lower variance at low
>   spp. THEN 3.x spatial (G-buffer + descriptor set 2 + Jacobian/MIS), 6.x
>   temporal, 7.x biased.
>
> (old NEXT, now done — kept for the build map:)
> NEXT GOAL = renderable initial-RIS ReSTIR (the coupled trunk, one vertical):
>   1. STUDY vk_wavefront.py pass machinery (how a pass compiles/binds/dispatches
>      + the per-frame schedule) — prerequisite for all plumbing.
>   2. FREEZE contracts: G-buffer struct (pos/normal/matId/wo), descriptor
>      bindings, ReSTIR config UBO/push-constant, pass-schedule. Parity-safe.
>   3. BUILD: restir/initial.slang (light+BSDF candidates over the unified light
>      set, unshadowed p̂=f·Le·G) → RestirDiReuse(ReusePlugin) + wavefront pass +
>      reservoir buffer → restir/resolve.slang (shadow ray + f·V·W) → activate the
>      depth-0 gate in reuse.slang (set reuseMode=1, register RestirDiReuse,
>      reuse_modes += "ReSTIR DI", wavefront capability gate).
>   4. TEST: converge-to-reference (ReSTIR vs reuse=none, high-spp image-mean
>      match) via test_sampling_parity::_accumulate. THE unbiased gate.
>   Then 3.x spatial reuse (Jacobian/MIS — hard math), 6.x temporal, 7.x biased.
>
> PLUMBING DESIGN (derived from vk_wavefront.py study — the build spec):
>   - Wavefront schedule (WavefrontPathPass.record_dispatch ~L603-668): per tile
>     generate → for each bounce { intersect → buildargs → scatter → shade(flat
>     slot0) → shade(catchall slot1) } → resolve. Bounce-0 `intersect` (~L651) =
>     PRIMARY HIT (HitInfo in the hit buffer). `shade` runs reuseDirect (my
>     depth-0 gate returns 0 when reuseMode=1). Set 0 = scene bindings (fc/lights/
>     env/materials/accum); set 1 = path-state(0)+hit(1)+6 queues(2-7); 12B push.
>   - ReSTIR buffers: per-PIXEL reservoir ×2 (prev/curr) + G-buffer, in a NEW
>     descriptor set 2, owned by RestirDiReuse. Persist across the bounce loop +
>     frames (temporal). Sized num_pixels, alloc/resize in renderer.
>   - HOOK: WavefrontPathPass gets an optional `restir` ref; in record_dispatch,
>     AT bounce 0 only, AFTER the primary intersect + BEFORE shade, call
>     restir.record_primary_direct(cmd, scene_set, hit_buf, state_buf): records
>     G-buffer fill → initial RIS → resolve (shadow ray + f·V·W, ADD into the
>     path-state radiance at pixelIndex). Shade's gated reuseDirect=0 ⇒ no double
>     count; wfPathResolve flushes radiance→accum as today.
>   - Build order: restir/initial.slang (M_light light + M_bsdf BSDF candidates
>     over scene_lights/env, unshadowed p̂=f·Le·G via mat.evaluate) → restir/
>     resolve.slang (one shadow ray for survivor, f·V·W) → RestirDiReuse in
>     sampling/reuse.py (buffers + set-2 + pipelines + record_primary_direct) →
>     WavefrontPathPass hook → renderer.py (register, reuse_modes+="ReSTIR DI",
>     wavefront-only capability gate, buffer alloc/resize, state-hash) → converge
>     test. Reservoir+merge cores (restir/reservoir.slang) already done+tested.
>   - Spatial pass set 2 also needs the canonical-point p̂ re-eval (mat.evaluate
>     at the G-buffer point) + the Jacobian — that's phase 3.x, after this lands.
>
> Worktree ../skinny-restir-di has the venv + mtlx symlink + parity goldens —
> no re-setup. export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
> DYLD_LIBRARY_PATH=$VULKAN_SDK/lib ; bin/python3.13 -m pytest tests/test_restir.py
>
> Phasing: P1 (spatial + unbiased + shade), P2 (progressive temporal), P4
> (biased toggle + tuning). P3 (reprojected temporal — motion vectors + prev
> G-buffer) is a separate follow-on change, reserved in the config enum.

## 1. Reuse-seam Slang module (realize the scene-sampling reservation)

- [x] 1.1 `shaders/sampling/reuse.slang` realized: `reuseDirect<TM:IMaterial>` routes primary-direct through the seam in both backends (path.slang FLAT/PYTHON + wavefront flat_bounce.slang); identity ⇒ stock `allLightsNEE`, with the inert depth-0 ReSTIR gate wired. Parity bit-identical both backends (goldens c7f77c5a / 803d9f9c). Commit 35be502.
- [ ] 1.2 `common.slang`: `RESTIR_DI` reuse-mode constant; the per-pixel G-buffer record layout (world pos, normal, materialId, wo) — scalar layout, Python-mirrored.

## 2. Reservoir core + initial RIS (P1)

- [x] 2.1 `shaders/restir/reservoir.slang`: `LightSampleRef` (type:2|id:30 + uv) + `Reservoir` (y, wSum, W, M, pHat) + pure RIS ops (reset/update/finalize, `W = wSum/(M·p̂)`). RIS unbiasedness unit-tested in isolation (test_restir_harness.slang + test_restir.py, slangpy): unbiased M=1/4/16, selection ∝ weight. Commit a4a4ce1. (The light-domain `p̂ = luminance(f·Le·G)` lives in 2.2, needs scene bindings.)
- [ ] 2.2 `shaders/restir/initial.slang`: initial RIS pass — draw `M_light` light candidates (unified sampler over directional/sphere/emissive-tri/env) + `M_bsdf` BSDF candidates, weighted-reservoir-sample to 1. No shadow rays (unshadowed p̂). Reads the G-buffer.
- [ ] 2.3 G-buffer fill: write pos/normal/materialId/wo per pixel from the primary hit (augment the wavefront primary intersect/shade or a small fill pass).

## 3. Spatial reuse + unbiased combination (P1)

- [ ] 3.1 `shaders/restir/spatial.slang`: gather `spatial_k` neighbors in `spatial_radius`; reject on G-buffer dissimilarity (normal/depth thresholds); merge reservoirs.
- [ ] 3.2 Unbiased combination: per-neighbor MIS weight `m_i` + the domain-aware `1/Z` normalization (count neighbors whose domain could produce `y`) + the reconnection Jacobian (`cosθ/d²` ratio between shading points) + horizon/visibility domain check.

## 4. Resolve + integration gate (P1)

- [ ] 4.1 `shaders/restir/resolve.slang`: one shadow ray for the surviving `y`; direct = `f(y)·V(y)·W`; write into the path's radiance / accumulation at the pixel.
- [ ] 4.2 Integration gate: at `depth == 0` with `reuseMode == RESTIR_DI`, skip `allLightsNEE` + the sphere-light/env-miss direct term in the bounce; still sample the bounce direction via the proposal mixture + spawn indirect. `depth ≥ 1` unchanged. Guard against env double-count vs the proposal mixture.

## 5. Host: plugin + passes + buffers (P1)

- [ ] 5.1 `sampling/reuse.py`: `RestirDiReuse(ReusePlugin)` (`reuse_mode = RESTIR_DI`) — owns the reservoir (×2) + G-buffer buffers, the ReSTIR config UBO, `build/destroy/resize/reset`, `passes()`, `bindings()`.
- [ ] 5.2 `sampling/registry.py`: register `RestirDiReuse`; `REUSE_PLUGINS` += it; renderer `reuse_modes` → `["None", "ReSTIR DI"]`.
- [ ] 5.3 `vk_wavefront.py` (or new `vk_restir.py`): build + schedule the ReSTIR pass set in wavefront mode when active; reuse-mode switch triggers the pass rebuild.
- [ ] 5.4 `renderer.py`: allocate/resize per-pixel buffers on framebuffer resize; wavefront capability gate (megakernel/Metal → identity); fold the ReSTIR config + reuse mode into `_current_state_hash`.
- [ ] 5.5 `params.py`: ReSTIR sub-config params (`M_light`, `M_bsdf`, `spatial_k`, `spatial_radius`, `M_cap`, `biased`, regime toggles) — gated visible when ReSTIR active.

## 6. Progressive temporal (P2)

- [ ] 6.1 `shaders/restir/temporal.slang`: merge the prev-frame reservoir at the same pixel (progressive = identity reprojection, static camera); cap `M` at `M_cap`; unbiased temporal combination.
- [ ] 6.2 Wire the double-buffer swap (prev/curr) per accumulation iteration; reset prev on accumulation reset.

## 7. Biased toggle + tuning (P4)

- [ ] 7.1 `biased` path in spatial/temporal merge: sum reservoirs, normalize by `ΣM`, skip `m_i`/Jacobian. Toggle via the ReSTIR UBO.
- [ ] 7.2 Sensible default tuning (`M_light`/`M_bsdf`/`spatial_k`/`radius`/`M_cap`) + document the cost/quality trade.

## 8. Tests + verification

- [ ] 8.1 Converge-to-reference (unbiased gate): ReSTIR DI vs stock NEE, both high-spp on an emissive/area-light scene — integrated radiance within tolerance. Reuse `test_sampling_parity::_accumulate`.
- [ ] 8.2 Variance reduction: ReSTIR error < stock-NEE error at low spp vs a converged reference.
- [ ] 8.3 Temporal beats spatial: progressive-temporal error < spatial-only error.
- [ ] 8.4 Biased bounded: biased darkening within a stated bound, no divergence.
- [ ] 8.5 Capability gate: `reuse=ReSTIR` + megakernel → identity (no reservoir passes); wavefront builds them.
- [ ] 8.6 Furnace still passes; pinned-seed runs reproducible.
- [ ] 8.7 `ruff` + `slangc` recompile (`main_pass` + wavefront variants) + `py_compile`.

## Out of scope (this change)

- [ ] P3 reprojected temporal (motion vectors + prev-frame G-buffer + disocclusion) — follow-on change.
- World-space / secondary-vertex ReSTIR; ReSTIR GI / PT; denoising.
