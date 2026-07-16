# Skinny — Photon Mapping (GPU SPPM)

This document is the implementation reference for **SPPM** (Stochastic
Progressive Photon Mapping) — skinny's third integrator and its caustic-efficient
estimator for specular→diffuse light transport (a focused highlight cast through
a glass object onto a diffuse floor, the regime where the unidirectional path
tracer and BDPT converge slowly). It covers the per-pass pipeline, the governing
equations and the exact shader symbols that realize them, the GPU buffer layout,
the pbrt importer mapping, the wavefront-only + flat-only constraints, and the
deferred phases.

This change is **PM-1**, the first change of a phased `photon-mapping`
capability. It delivers the core *surface* SPPM estimator on **flat materials
only**; the layered skin/BSSRDF photon path (PM-2) and volumetric/media photon
transport (PM-3) are deferred to follow-up changes against the same capability
spec (see [Deferred phases](#deferred-phases)).

> Equations are shipped as **SVG images** (the repo's GitLab does not render
> KaTeX/`$$` math reliably). The LaTeX sources live in
> `docs/diagrams/sppm/equations.json`; regenerate the SVGs with
> `docs/diagrams/restir/render.cjs` (MathJax 3, publication quality — needs Node
> + `mathjax-full`) or the dependency-free
> `docs/diagrams/sppm/gen_svg_equations.cjs` fallback (`node
> docs/diagrams/sppm/gen_svg_equations.cjs`). Inline symbols (r_i, N_i, τ, Φ, π)
> are plain Unicode.

SPPM rides the **wavefront execution backend** documented in
[Wavefront.md](Wavefront.md) (the SPPM stage list lives there beside the
path/BDPT tables); its set-1 descriptor bindings are in
[Architecture.md](Architecture.md) (descriptor binding map); the generic
path/BDPT integrators are in [README.md](../README.md); the pbrt `sppm` importer
mapping is in [PbrtImport.md](PbrtImport.md).

## What SPPM is

Photon mapping splits light transport into two walks that meet in the middle. An
**eye pass** walks a camera ray to a diffuse *visible point* on a surface; a
**photon pass** shoots energy-carrying particles from the lights and deposits
their flux onto every visible point within a search radius. The radiance at a
visible point is then a **density estimate** of the deposited flux over the disc
of that radius. This decoupling is what makes caustics cheap: the specular chain
is traced once from each side, and the two ends are joined by a spatial lookup
rather than by a low-probability random connection.

*Progressive* photon mapping (Hachisuka et al.) removes the memory cost and the
bias of a fixed radius: instead of storing all photons, it runs **many passes of
a fresh photon batch** and, after each pass, **shrinks every visible point's
radius** while accumulating flux, so the estimate converges to the true radiance
as the radius → 0. *Stochastic* PPM (Hachisuka & Jensen 2009) further re-samples
**one** stochastic visible point per pixel per pass, so the eye pass is a single
cheap trace and depth-of-field / glossy eye paths fall out for free.

In skinny one **SPPM pass == one progressive-accumulation frame**: the existing
accumulation loop drives the progression, and the per-pixel estimator state
(radius r_i, photon count N_i, accumulated flux τ_i) **persists across frames**
while the eye pass rewrites only the per-pass geometry. Direct lighting reuses
the existing NEE path (computed once in the eye stage); the photon term carries
only the *indirect / caustic* complement, so the two are disjoint with no
double-count.

Because the eye **terminates** at the visible point (photons carry the indirect,
so it never continues), it must add the **env-miss MIS companion** of that
vertex's environment NEE itself: `allLightsNEE` MIS-weights the env NEE by
`powerHeuristic(envPdf, bsdfPdf)`, expecting a BSDF-sampled env-miss from the next
bounce — which a continuing path would supply but SPPM never traces. So the eye
takes one BSDF sample at the visible point and, if it escapes to the environment,
adds `throughput · bsdfWeight · L_env · powerHeuristic(bsdfPdf, envPdf)`, giving
env direct = NEE + BSDF-miss exactly as `path.slang`. Without it the env NEE is
down-weighted with no counterpart and env direct is under-counted ~25 % under a
broad environment (change `sppm-caustic-dimness`); small analytic lights are
unaffected (their BSDF pdf ≈ 0 ⇒ the NEE is already effectively full weight).

### Scope and limits

| Property | Value |
| --- | --- |
| Integrator | **`INTEGRATOR_SPPM = 2`** (`common.slang`). The third integrator after `path` (0) and `bdpt` (1). |
| Backend | **Wavefront only**, on **both Vulkan and native Metal**. Vulkan uses `WavefrontSppmPass` (`vk_wavefront.py`); Metal uses `MetalWavefrontSppmPass` (`metal_wavefront.py`) — the kernels are Metal-portable (typed buffers, ~15/31 slots, no Metal-only gate) and the caustic parity matches across backends. The megakernel has no global photon map, so `--integrator sppm` under megakernel is **refused** with a clear message (`cli_common._validate_integrator`). |
| Materials | **Flat only** — `UsdPreviewSurface` / `standard_surface` / `OpenPBR` / Python flat materials, same gating as ReSTIR DI and neural guiding. Glass / mirror **are** flat materials in skinny (a delta lobe), so the eye and photon walks pass through them as caustic carriers; skin / MaterialX-graph / python-graph / debug receivers terminate the walk (PM-1 is the surface case). |
| Selection | `--integrator sppm` (requires `--execution-mode wavefront`; `cli_common.INTEGRATOR_INDEX["sppm"] = 2`) across the front-ends, or the pbrt importer when the scene used `Integrator "sppm"`/`"photonmap"`. |
| Direct light | Reuses stock **NEE** (the eye stage); the photon term is the indirect/caustic complement only. |

## Stages of rendering

One SPPM pass is a **four-logical-stage** pipeline over **eight kernels**, all in
`shaders/integrators/wavefront_sppm.slang` (entries `wfSppm*` — Slang's Metal
target reserves `main`). Every dispatch has a host-known count (num_pixels,
numCells, photonsEmitted) → plain dispatch, **no indirect dispatch / no Metal
CPU-readback stall**.

![GPU SPPM per-pass pipeline: eye → grid build (count/scan/scatter) → photon → update, with the persistent per-pixel estimator feeding back across accumulation frames](diagrams/sppm_pipeline.svg)

| Stage | Kernel(s) | Work |
|-------|-----------|------|
| **EYE** | `wfSppmEye` (over num_pixels, tiled) | Trace the camera path through the flat **delta lobe** (glass/mirror are flat delta materials) to the first non-specular flat hit. Store **one** stochastic `VisiblePoint`/pixel `{pos, ns, wo, beta, the evaluated flat BSDF}`, plus the full per-pass **direct** term `ld` (NEE at the VP + any emitter/env reached through the specular chain). Persistent `radius`/`n`/`tau` are preserved; on first activation (`n == 0`) the radius inits to `fc.sppmInitialRadius`. A pixel that escapes leaves its VP **inactive** so a stale point can't sit in the grid. |
| **GRID BUILD** | `wfSppmGridCount` → `wfSppmGridScanBlock` → `wfSppmGridScanBlockSums` → `wfSppmGridScanAdd` → `wfSppmGridScatter` | A uniform spatial **hash grid** built by **counting sort** over the active visible points: per-cell atomic count → blocked **exclusive prefix sum** (count → offset) → scatter pixel indices into their cell buckets. `numCells = next_pow2(2·num_pixels)`; `cellSize = sppmInitialRadius` (a valid upper bound on every active VP's radius, since radii only shrink — so the 3×3×3 neighbour scan never misses a photon). |
| **PHOTON** | `wfSppmPhotonTrace` (over `fc.sppmPhotonsEmitted`; **breadth-tiled on Metal**, change `sppm-photon-dispatch-tiling`) | **Emit** power-weighted photons (emissive triangles via `sampleEmissiveTriangle` pSel, sphere lights, distant beam, env beam via `sampleEnvDir` — `beta = Le·π / p_sel`, distant/env `beta = L·πR²/p_sel` from the bbox disc), **trace** with Russian roulette, and **deposit** only at non-specular vertices with `depth ≥ 1` (so the photon term is disjoint from the NEE direct: `depth == 0` is light→VP direct that NEE owns; the SDS caustic carrier light→specular→VP still deposits). Deposit via a de-duplicated **3×3×3 neighbour-cell scan**, adding the **bare** f_r `= evaluate(woVP,wiVP).response / max(wiVP.z, 1e-4)` (no cosine — the photon-map density estimate) with portable **uint fixed-point** `InterlockedAdd` into `SppmAccum`. |
| **UPDATE** | `wfSppmUpdate` (over num_pixels, tiled) | The per-pass indirect estimate `L_indirect = (vp.beta ⊗ Φ) / (N_emitted · π · r²)` — **this pass's** deposited flux Φ, scaled by the eye throughput `vp.beta` (change `sppm-vp-beta-resolve`), at the gather radius (the radius before this pass's reduction); `sample = vp.ld + L_indirect`; **running-mean composite** into the accumulation film (exactly like `wfPathResolve`), which is what averages the per-pass estimates into the progressive result. The SPPM reduction `N' = N + γM`, `r' = r·√(N'/(N+M))` (γ = 2/3) then **advances the radius** for the next pass so the per-pass bias shrinks; display + clear the accumulator. |

### Split dispatch order

The grid + photon stages are **global** over every visible point, so they must
run **after all eye tiles** and **before any update tile** — never interleaved
per tile. `record_sppm_loop` (`wavefront_driver.py`) encodes exactly this
mandated split order (an adversarial-review requirement); `tiles == 1`
(num_pixels ≤ stream_size) is just the degenerate case:

```text
[frame 0 only]  clear the persistent visible-point buffer
phase 1   all eye tiles            (write every pixel's visible point)   → barrier
phase 2   grid build               clear → count → scan → scatter        → barrier
phase 3   photon pass              clear accum → [emit / trace / deposit]×⌈N/batch⌉  → barrier
                                   (Metal: each sub-batch of SKINNY_SPPM_METAL_PHOTON_BATCH
                                    photons is a flushed command buffer — bounds the watchdog
                                    by breadth, bit-identical to one dispatch)
phase 4   all update tiles         reduce + resolve + composite          → barrier
```

The backend recorder (`_VkSppmRecorder` on Vulkan) supplies the path-loop
primitives (`push_tile`, `dispatch_full`, `dispatch_one`, `barrier`) plus
`dispatch_count(entry, count, group)` for the host-counted grid/photon stages and
the `clear_visible_points()` / `clear_grid()` / `clear_accum()` buffer clears
(`vkCmdFillBuffer`).

## Glossy / near-specular continuation

PM-1 stored a visible point at the **first non-delta hit** (any flat surface
with `bs.pdf > 0`). That is correct for diffuse and rough surfaces, but it breaks
on a **glossy metal**: its narrow specular lobe makes the photon gather too
sparse to rebuild the sharp reflection (few photons land inside the search disc
along the mirror-like direction), and because a pure metal carries **no diffuse
term**, the stored VP's bare-f_r evaluation washes the reflection out entirely.
Concretely, the brass sphere in `three_materials_demo` failed to reflect the
neighbouring wood/marble spheres — it rendered as a near-flat highlight instead
of a mirror-with-roughness.

The fix is a **roughness-thresholded, metallic-gated eye-walk continuation**.
When the BSDF-sampled lobe at a hit is metallic (`metalness ≥ 0.5`) and its
roughness is below `fc.sppmGlossyContinueRoughness`, the eye walk does **not**
store a visible point there. Instead it follows the BSDF-sampled direction one
more bounce — exactly the way the existing **delta caustic carrier** (glass /
mirror) is continued — so the VP lands on the **next non-glossy surface**. The
sharp reflection is then reconstructed at *that* surface and averaged across the
progressive passes like any other VP. The predicate (identical in the eye and
photon stages) is:

```text
bs.pdf > 0.0  &&  m.roughness < fc.sppmGlossyContinueRoughness  &&  m.metallic >= 0.5
```

The **photon stage** treats a glossy-continued vertex the same way it treats a
delta vertex — **as specular, with no deposit** — so the disjoint
direct(NEE)/indirect(photon) split is preserved and the SPPM energy ratio is
unchanged: the photon that would otherwise have deposited onto the glossy VP
instead continues to the surface the eye VP now lives on.

**Threshold semantics.** A threshold of `0` reproduces PM-1 exactly
(`m.roughness < 0` is never true, so nothing is continued and every glossy metal
stores a VP at the first hit — delta-only behaviour). The default is `≈ 0.6`,
set with `--sppm-glossy-roughness` (env `SKINNY_SPPM_GLOSSY_ROUGHNESS`); it is
folded into `_current_state_hash` so changing it resets accumulation cleanly. The
threshold is in **perceptual (USD) roughness**, so it must also reach
pbrt-imported polished metals: pbrt's perceptual roughness `r` imports through
`alpha = sqrt(r)` then `usd = sqrt(alpha) = r**0.25`, so a polished
pbrt-roughness-`0.1` conductor lands at usd `≈ 0.562` (GGX alpha `≈ 0.316`). `0.6`
is an alpha `≲ 0.36` "polished metal" cutoff that catches it while leaving a
pbrt-roughness-`0.3` metal (usd `≈ 0.740`) on the gather side.

**Reached-light MIS.** A glossy-continued vertex runs env *and* emissive-triangle
NEE like any flat vertex, so any light its carrier ray reaches must be MIS-weighted
(not added at full weight, which would double-count it):

- **Env escape** — a metal reflecting only a distant/IBL light (e.g. the
  `conductor_infinite` gate scene, where no photon can ever deposit on the sole
  metal surface) weights the escaped env by `powerHeuristic(bsdfPdf, envPdf(dir))`.
- **Emissive-triangle hit** — the emission is weighted by
  `powerHeuristic(bsdfPdf, pdfLightSA)`, `pdfLightSA` reconstructed without the tri
  index (`lum·d² / (emissiveTotalPower·cosLight)`).

Both mirror `integrators/path.slang` exactly. `spawnedBySpecular` is therefore
**delta-only** (`bs.pdf ≤ 0`) — the earlier code treated *every* continued vertex
as delta for the emissive gate, a latent full-weight double-count that the raised
threshold would expose. Only perfectly-specular (delta) carriers keep full weight;
transmitted lobes and furnace mode suppress the env companion. With this, SPPM
converges to the path reference on glossy-metal-under-light scenes instead of
diverging.

**Why the metallic guard.** `BSDFSample` carries **no sampled-lobe id**, and the
shared flat BSDF (`interfaces.slang` / `flat_material.slang`) is kept
**byte-frozen** — so path / BDPT keep byte-identical BSDF sampling and
evaluation. (Appending `sppmGlossyContinueRoughness` to the shared
`FrameConstants` *does* perturb that struct's declaration in every consumer's
SPIR-V — the same trailing-field effect PM-1's four SPPM tail fields had — but
the field is **appended**, so no existing field offset moves, and path / BDPT
never read it: their render output is unchanged.) The `metalness ≥ 0.5`
guard is therefore an implementation proxy for "this sample is the sharp
specular lobe": it keeps **dielectric diffuse samples** (e.g. the marble's
`specular_roughness = 0.1`, the wood) on the **gather** side — they still store a
VP and accumulate photons — while only **metals** (brass `metalness = 1`,
`roughness ≈ 0.15`) continue. Without the guard, a low-roughness dielectric's
*diffuse* sample would be wrongly continued and its surface would stop gathering.

**Deferred — full final gather.** A lower-variance variant for **mid-roughness**
glossy is a true *final gather*: store the glossy VP and, in a follow-up pass,
shoot **one BSDF-sampled gather ray** per glossy VP that reads the photon
estimate at *its* hit point (rather than walking the eye ray through). This trades
the single-bounce continuation's reuse of the next surface's VP for an explicit
importance-sampled gather, which is better-behaved across the full roughness
range — but it needs an **extra wavefront ray pass** (gather-ray trace + photon
lookup) and the attendant record buffers. It is recorded here as the natural next
change against the `photon-mapping` capability and is **not built in this
change**.

## Equations

Notation: Le is a light's emitted radiance; f_r is the bare BSDF (response
divided by the cosine, **no cosine** — the photon-map density estimate); β is the
per-photon flux; Φ is a visible point's accumulated per-pass flux; τ its
persistent flux; r its search radius; N its accumulated photon count; M its
per-pass photon count; γ the SPPM reduction parameter (2/3). lum(·) is luminance.

### 1. Photon emission

Each photon is sampled from one scene light. Group selection is
**power-proportional** over the present light groups (emissive triangle / sphere
/ distant / environment; change `sppm-power-proportional-photon-groups`): the
host computes each group's emitted power —

| group | emitted power Φ_g |
| --- | --- |
| emissive triangles | `π · Σ(area · lum)` (the existing `emissiveTotalPower` sum) |
| sphere lights | `4π² · Σ(lum · r²)` (full-sphere diffuse emitters) |
| distant lights | `πR² · Σ lum` (parallel beam through the bbox disc) |
| environment | `πR² · envIntensity · ∫L dω` (pbrt `ImageInfiniteLight::Phi`; the sin θ-weighted luminance integral is computed alongside the env CDF) |

— normalizes them into a selection pmf (`renderer._sppm_photon_group_pmf`;
uniform-over-present fallback when the total power is 0/non-finite), and uploads
it in `FrameConstants` (`sppmGroupPmfE/S/D/Env`). `sppmEmitPhoton` walks the pmf
as a CDF and divides each branch's flux by its **actual** selection probability,
so the estimator stays unbiased while per-photon flux equalises across groups
(`Φ_g / p_g ≈ Φ_total`). Under the previous uniform `1/G` selection a dominant
environment (β = L·πR²/(gsel·p_dir), bbox disc πR² ≫ a small local emitter) rode
on an equal photon share as sparse, enormous splats — firefly speckle on any
scene mixing a weak local light with an environment. Within the emissive group,
triangles remain **power-weighted** (pSel, folded into the area-measure
selection pdf p_sel). For a diffuse area emitter the emission cosine cancels the
cosine sampling pdf, so the carried flux is

![beta = Le · pi / p_sel](diagrams/sppm/photon-beta.svg)

The `1/N_emitted` normalization (over the photons emitted this pass) is **not**
folded into β here — it is applied once in the update stage's radiance estimate.
A distant light emits a parallel beam from a bbox-covering disc, accounting the
`A_disk` factor in β (`pdfDir` is a delta, no cosine).

The **environment** (change `sppm-env-indirect-transport`) is a fourth group,
present iff the env is live (`furnaceMode == 0 && envIntensity > 0` — the same
standing preconditions env NEE relies on). An env photon importance-samples a
sky direction via `sampleEnvDir` (the identical distribution env NEE uses), is
emitted inward from the same bbox-covering disc as the distant beam, and carries
the pbrt `ImageInfiniteLight::SampleLe` flux — the disc position pdf `1/(πR²)`
cancels against the disc area, leaving the non-delta direction pdf in the
denominator:

`beta = L_env(ω) · πR² / (p_sel · p_dir(ω))`

Samples with `p_dir ≤ 0` (equirect poles / degenerate distribution) are rejected
before the divide — an unguarded pole sample yields an infinite β that poisons
Russian roulette and the whole walk. Under `SKINNY_SPECTRAL` the env radiance is
upsampled with `upsampleIlluminantBound` at the shared per-pass wavelengths,
mirroring the sphere-light branch. Env DIRECT stays owned by the eye stage (env
NEE at the VP + the terminal env-miss companion); env photons contribute only
`depth ≥ 1` deposits, so the partition is disjoint by construction.

> **Implements:** `sppmEmitPhoton` in `wavefront_sppm.slang`
> (`beta = ls.radiance * PI / max(selA, 1e-20)` for area emitters;
> `beta = dl.rad * (PI·R²) / selPdf` for the distant beam;
> `beta = es.radiance * (PI·R²) / max(pSel · es.pdf, 1e-20)` for env photons).

| symbol | code | meaning |
| --- | --- | --- |
| Le | `ls.radiance` | sampled emitter radiance |
| p_sel | `selA` | area-measure light-selection pdf (group × within-group pSel × pdfArea) |
| β | `beta` | per-photon flux (out of `sppmEmitPhoton`) |

### 2. Photon deposit (the density estimate)

At a non-specular photon vertex (after ≥ 1 prior bounce), the photon's flux is
deposited onto every active visible point `j` within `||Δ|| ≤ r_j` and on the
same hemisphere, weighted by **that visible point's own** BSDF — the **bare**
f_r, no cosine, which is the photon-map surface density estimator:

![phi_j += beta · f_r,  f_r = response(wo, wi) / max(wi_z, 1e-4)](diagrams/sppm/deposit.svg)

The deposit is a portable **uint fixed-point** `InterlockedAdd` (neither
Vulkan-core nor Metal guarantees fp atomics): φ is scaled by
`SPPM_FLUX_FIXED_SCALE = 2²⁰` before the add and decoded back in the update
stage. NaN/inf/negative φ are dropped (mirroring `atomicSplatRadiance`). The
3×3×3 neighbour cells are **de-duplicated** (a `seen[27]` set) so a hash
collision can't double-deposit.

> **Implements:** `sppmDepositPhoton` in `wavefront_sppm.slang`
> (`f = mat.evaluate(woVP, wiVP).response / max(wiVP.z, 1e-4)`; `phi = beta * f`;
> `InterlockedAdd(sppmAccum[j].phi{R,G,B}, uint(phi · SPPM_FLUX_FIXED_SCALE))`).

| symbol | code | meaning |
| --- | --- | --- |
| f_r | `f` | bare BSDF response ÷ cosine at the visible point |
| β | `beta` | incoming photon flux |
| φ_j | `sppmAccum[j].phi{R,G,B}` | per-pass deposited flux (fixed-point) |
| M_j | `sppmAccum[j].m` | per-pass photon count at this point |

### 3. The SPPM reduction (per pass)

After the photon pass, each active visible point reduces its persistent estimator
(Hachisuka & Jensen 2009): the radius shrinks by the photon-count ratio, the
accumulated flux is rescaled by the same ratio so the density estimate stays
consistent, and the count advances by γ·M. With γ = 2/3:

![N' = N + gamma · M](diagrams/sppm/update-N.svg)

![r' = r · sqrt(N' / (N + M))](diagrams/sppm/update-r.svg)

![tau' = (tau + Phi)(r'/r)^2 = (tau + Phi) · N'/(N + M)](diagrams/sppm/update-tau.svg)

`(r'/r)² = N'/(N+M)`, so τ is rescaled with a single multiply. A pass with no
photons (`M == 0`) leaves the estimator untouched.

> **Implements:** `sppmUpdate` in `integrators/sppm_state.slang`
> (`nNew = nOld + gamma·mf`; `ratio = nNew / (nOld + mf)`;
> `vp.tau = (vp.tau + phi) * ratio`; `vp.radius *= sqrt(ratio)`; `vp.n = nNew`).

| symbol | code | meaning |
| --- | --- | --- |
| N / N' | `vp.n` / `nNew` | accumulated photon count (float, for γ) |
| M | `m` (`acc.m`) | per-pass photon count at this point |
| r / r' | `vp.radius` | search radius (monotonically non-increasing) |
| τ / τ' | `vp.tau` | persistent accumulated flux |
| Φ | `phi` (`sppmDecodeFlux(acc)`) | per-pass decoded flux |
| γ | `gamma` (= 2/3) | SPPM reduction parameter |

### 4. The radiance estimate and composite

The per-pass indirect radiance at a visible point is the photon-map density
estimate — **this pass's** deposited flux Φ over the disc, scaled by the
**eye throughput** β_eye = `vp.beta` (camera→VP, excluding the VP BSDF; pbrt
folds it into τ at pass end), normalized by the photons emitted this pass, at
the radius the photons gathered within:

![L_indirect = (beta_eye ⊗ Phi) / (N_emitted · pi · r^2)](diagrams/sppm/radiance.svg)

β_eye ≡ 1 for a directly-viewed VP and for clear Fresnel-sampled delta glass
(the sample weight cancels to exactly 1); it attenuates/tints the photon term
through lossy or tinted eye chains (change `sppm-vp-beta-resolve` — without it
the photon term through tinted glass was over-bright and un-tinted). In
spectral mode the β⊗Φ product is per-λ at the shared pass wavelengths, applied
before the λ→sRGB resolve. The radius/N advance is flux-independent, so the
`sppmUpdate` fold stays β-free.

> **Why per-pass Φ, not accumulated τ.** skinny composites the per-pass estimate
> into the accumulation film with a **running mean** (below), so the film is what
> averages the passes into the progressive result. Using the *accumulated* τ here
> *and* film-averaging would count the flux twice (an early bug that rendered
> caustics ~14× too bright). So the displayed estimate is the **per-pass** Φ at
> the **pre-reduction** gather radius; the SPPM `N`/`r` reduction still runs to
> **advance the radius** each pass, which shrinks the per-pass bias (bias → 0 as
> r → 0). This is consistent: each pass is an independent density estimate at a
> shrinking radius, and the running mean both averages them and reduces variance.

The composited per-pixel sample adds the per-pass direct (computed once in the
eye stage — NEE at the VP plus any emitter/env through the specular chain) to the
indirect estimate:

![L = L_d + L_indirect](diagrams/sppm/sample.svg)

`sample` is sanitized (NaN/inf/negative → 0) and folded into the accumulation
film with the same **running mean** every wavefront resolve kernel uses
(`(prev·n + sample)/(n+1)`), then written to the display via `wfWriteDisplay`.

> **Implements:** `wfSppmUpdate` in `wavefront_sppm.slang` (`phi =
> sppmDecodeFlux(acc)`; `lIndirect = (vp.beta · phi) / (max(fc.sppmPhotonsEmitted,1)·π·rGather²)`
> with `rGather = vp.radius` before the reduction; `sppmUpdate(vp, phi, …)`
> advances the radius; `sample = vp.ld + lIndirect`; running-mean composite into
> `accumBuffer`; `wfWriteDisplay`).

| symbol | code | meaning |
| --- | --- | --- |
| Φ | `sppmDecodeFlux(acc)` | this pass's deposited flux (decoded from the fixed-point accumulator) |
| β_eye | `vp.beta` | eye throughput camera→VP, excluding the VP BSDF |
| N_emitted | `fc.sppmPhotonsEmitted` | photons emitted this pass (the estimator divisor) |
| r | `vp.radius` | gather radius (before this pass's reduction) |
| L_d | `vp.ld` | per-pass direct (NEE + specular-chain emitters) |
| L_indirect | `lIndirect` | photon-map indirect estimate |

### Spatial-hash grid

The grid is a uniform spatial hash. Its cell count is the next power of two ≥
2·W·H (so the hash masks with `& (numCells − 1)`):

![n_cells = 2^ceil(log2(2·W·H))](diagrams/sppm/numcells.svg)

The integer cell coordinate is `floor((pos − boundsMin) / cellSize)` with
`cellSize = sppmInitialRadius`; the hash mixes the coordinate with the standard
`(73856093, 19349663, 83492791)` primes through `pcgHash`. The host mirror is
`wavefront_layout.sppm_grid_cell_count`, locked against the Slang
`sppmGridCellCount` by `tests/test_sppm_state.py`.

> **Implements:** `sppmGridCellCount` / `sppmCellCoord` / `sppmCellHash` in
> `integrators/sppm_state.slang`, shared by the eye / grid / photon stages.

## Per-pixel state

Two GPU-internal scratch structs, one element **per pixel** (sized by num_pixels,
the *persistent* per-pixel estimator — **not** `stream_size`), allocated but never
packed by the host (like the wavefront records). `tests/test_sppm_state.py` locks
both struct layouts.

```slang
// integrators/sppm_state.slang  (scalar = 180 B Vulkan, MSL = 240 B)
struct VisiblePoint {
    float3 pos;            // world hit position           (rewritten each pass)
    float3 ns;             // world shading normal, post normal-map (each pass)
    float3 wo;             // world outgoing dir toward the camera (this pass)
    float3 beta;           // eye throughput EXCLUDING the VP BSDF (this pass)
    float3 ld;             // per-pass direct (NEE + specular-chain emitters)
    float3 albedo;         // ── evaluated flat BSDF: FlatHitMat (minus emission) + F0
    float3 F0;             //    so the photon deposit rebuilds FlatMaterial and
    float3 coatColor;      //    evaluates f_r with NO texture refetch / graph
    float  roughness;      //    re-run / double normal-map (pbrt's store-the-BSDF
    float  metallic;       //    approach)
    float  specular;
    float  ior;
    float  opacity;
    float  coat;
    float  coatRoughness;
    float  coatIOR;
    float3 transmissionColor;  // Stage-2 rich inputs (flat-lobes-rich-inputs) —
    float3 specularColor;      // stored like every FlatHitMat field the lobe
    float  diffuseRoughness;   // model reads (fix-sppm-bathroom-black-walls)
    float3 tau;            // accumulated reflected flux   (PERSISTS)
    uint   flags;          // bit0 = VP_ACTIVE
    float  radius;         // current search radius r_i    (PERSISTS)
    float  n;              // accumulated photon count N_i (PERSISTS; float for γ)
};

// Per-pass photon-deposit accumulator (separate buffer, atomically written).
struct SppmAccum {        // scalar = 16 B, MSL = 16 B
    uint phiR, phiG, phiB; // fixed-point per-pass flux (÷ SPPM_FLUX_FIXED_SCALE = 2²⁰)
    uint m;                // photons deposited into this point this pass (M_i)
};
```

The visible point embeds the **evaluated** flat BSDF (the
`albedo..diffuseRoughness` block mirrors `FlatHitMat` minus emission, plus
`F0`), not reconstruction inputs:
`fetchFlatHitData` re-applies the normal map and re-runs the MaterialX graph, so
storing the shading normal alone would double-apply the normal map and re-running
the graph per deposit would be expensive. `sppmStoreVisiblePoint` (eye stage)
stores the evaluated material; `sppmLoadMaterial` (photon stage) rebuilds the
`FlatMaterial` for the bare-f_r evaluation with no refetch. Splitting the atomic
flux into a separate `SppmAccum` keeps the deposit portable (uint fixed-point —
neither backend guarantees fp atomics) and keeps the hot `VisiblePoint` free of
atomically-mutated fields.

**The mirror is a correctness contract, enforced hostlessly.** Every
`FlatHitMat` field the lobe model reads must have a `VisiblePoint` slot, be
written by `sppmStoreVisiblePoint`, and be rebuilt by `sppmLoadMaterial`
(`emission` is the one documented exemption — direct, not BRDF). When
`flat-lobes-rich-inputs` grew `FlatHitMat` by three fields after PM-1 shipped,
the un-slotted rebuild fed **undefined values** into `evaluate()` at deposit
time and zeroed every photon's flux (`τ == 0` scene-wide while `m` kept
shrinking the radius): SPPM silently degraded to its eye-pass direct term, and
indirect-lit surfaces rendered black (bathroom walls). Fixed in change
`fix-sppm-bathroom-black-walls`; two parse-lock tests in
`tests/test_sppm_state.py` now fail the build if `FlatHitMat` ever grows a
field without the full VP slot + store + rebuild chain.

## GPU buffers and bindings

SPPM is a distinct wavefront pass (`WavefrontSppmPass`, `vk_wavefront.py`) with
its **own set 1** (it does not share the path pass's stream state). The four
buffers are **typed structured buffers** — not the `ByteAddressBuffer` fold the
blueprint first proposed: a SPPM kernel references only ~15/31 Metal buffer slots
(it never compiles the neural weights), so the typed buffers fit with headroom,
stay type-safe, and need **no `SKINNY_METAL_SPPM` gate**.

| Set 1 binding | Buffer | Type | Content |
|---|---|---|---|
| 0 | `sppmVisiblePoints` | `RWStructuredBuffer<VisiblePoint>` | `[num_pixels]` — the **persistent** per-pixel estimator (geometry + evaluated BSDF + per-pass direct + τ/r/N) |
| 1 | `sppmAccum` | `RWStructuredBuffer<SppmAccum>` | `[num_pixels]` — per-pass fixed-point atomic flux accumulator, cleared each pass |
| 2 | `sppmGrid` | `RWStructuredBuffer<uint>` | four contiguous sub-ranges over `numCells`: `gridCount` \| `gridOffset` \| `gridCursor` \| `sortedIdx` |
| 3 | `sppmScanScratch` | `RWStructuredBuffer<uint>` | `ceil(numCells / 256)` block sums for the parallel prefix scan |

Set 0 is the renderer's shared megakernel scene set (it provides `fc` with the
SPPM `FrameConstants` tail plus scene geometry / materials / lights). The
12-byte push constant is the same `{streamBase, shadeSlot(unused), streamSize}`
tile layout the path pass uses.

### FrameConstants tail

Five fields are appended to `FrameConstants` (`common.slang`), packed by both
`_pack_uniforms` and `_pack_uniforms_msl` via `_FC_SCALAR_FIELDS`, read only by
the SPPM kernels (`integratorType == 2`); +28 B, within the 768 B UBO (the
import-time `_VK_UNIFORM_BUFFER_BYTES` assert self-guards). The scalar blob grows
540 → 544 B; on Metal the new 4-byte float tips the reflected `fc` struct past its
prior trailing padding, so the MSL `fc` size grows **592 → 640 B** (verified live
under guarded Metal; `_pack_uniforms_msl` sizes from the reflection, so the Metal
megakernel self-adapts — only the `_MSL_FC_BYTES` test pin is hand-tracked):

| Field | Type | Meaning |
|---|---|---|
| `sppmInitialRadius` | `float` | initial per-pixel search radius r₀ — the pbrt `radius` param if imported, else ≈ **0.1 % of the scene bbox diagonal** (`max(diag·0.001, 1e-4)`) |
| `sppmCellSize` | `float` | spatial-hash cell size (== `sppmInitialRadius`, a valid upper bound as radii shrink) |
| `sppmGridRes` | `uint3` | per-axis grid resolution — packed but **unused** by the kernels (they hash from `width*height`) |
| `sppmPhotonsEmitted` | `uint` | photons emitted per pass (the `1/N_emitted` estimator divisor); default `num_pixels` (one photon/pixel), or the `_sppm_photons_override` renderer attribute (set by the pbrt `photonsperiteration` import) |
| `sppmGlossyContinueRoughness` | `float` | glossy / near-specular eye-walk continuation threshold (see [Glossy / near-specular continuation](#glossy--near-specular-continuation)). A metallic sample whose roughness is below this value is continued one bounce instead of storing a VP. **0** reproduces PM-1 (delta-only); default ≈ **0.6** (perceptual/USD roughness; reaches pbrt-imported polished metals, `usd = r**0.25`), set via `--sppm-glossy-roughness` / `SKINNY_SPPM_GLOSSY_ROUGHNESS`. |

All three SPPM tuning overrides (`_sppm_radius_override`,
`_sppm_photons_override`, `_sppm_glossy_roughness_override`) are added to
`_current_state_hash`, so changing any of them resets accumulation cleanly (an
A/B that varies only the glossy threshold on one reused renderer converges from
scratch rather than accumulating across configurations). Only the glossy
threshold has a CLI flag — `--sppm-glossy-roughness` (env
`SKINNY_SPPM_GLOSSY_ROUGHNESS`); the radius / photon overrides are renderer
attributes set by the pbrt `sppm` importer (`api.sppm_selection`).

## Host wiring

| Site | File | Role |
|---|---|---|
| `WavefrontSppmPass` + `_VkSppmRecorder` | `vk_wavefront.py` | compile the 8 entries, the 4-buffer set-1 layout, real `vkCmdDispatch`, `vkCmdFillBuffer` clears; allocate the state buffer by num_pixels |
| `record_sppm_loop` | `wavefront_driver.py` | the mandated split order (all-eye → one-grid → one-photon → all-update); `tiles == 1` degenerate case |
| `_record_wavefront_dispatch` (`integrator_index == 2` branch) | `renderer.py` | dispatch the SPPM pass; compute `sppmInitialRadius` from the pbrt `radius` else the bbox heuristic; default photons/pass = num_pixels |
| `sppm_buffer_sizes` / `sppm_grid_buffer_sizes` / `sppm_grid_cell_count` | `wavefront_layout.py` | host buffer sizing + the cell-count mirror |
| `INTEGRATOR_INDEX["sppm"] = 2`, `--integrator sppm`, `--sppm-radius`, `--sppm-photons-per-pass`, the wavefront gate | `cli_common.py` | selection + incompatibility gating (refuses megakernel) |

## pbrt importer mapping

skinny's pbrt v4 importer already recorded `Integrator "sppm"` in metadata but
mapped it to nothing — sppm scenes silently rendered on the path tracer. PM-1
closes that: `Integrator "sppm"` / `"photonmap"` is now recognized and **mapped**
(reported `exact`, not `skipped`).

- `metadata.scene_metadata` records a normalized skinny selection under
  `customLayerData["pbrt"]["skinny"] = {integrator: "sppm", radius?, photons?}`
  alongside the exact pbrt integrator spec — `radius` seeds r₀,
  `photonsperiteration` the photons-per-pass override.
- `api.sppm_selection(stage)` reads that selection back (or `None` for any other
  integrator), so a loader or the parity harness can pick the SPPM integrator and
  seed its initial radius / photon count without re-parsing pbrt param names.
- `api.import_pbrt` reports `integrator:sppm → "mapped to skinny SPPM (wavefront,
  flat materials)"`.

See [PbrtImport.md § pbrt metadata carry](PbrtImport.md#pbrt-metadata-carry).

## Post-mortem: the "SPPM caustic speckle" was the phantom sun (2026-07)

An investigation of "SPPM's caustics don't match bdpt" on the glass-caustics
scene ended with SPPM being the **most correct** integrator. The scattered
bright speckle in the shadow/penumbra ground was the glass caustic of the
renderer's **synthesized default DistantLight** (injected into any scene
authoring no directional light — even one lit by its own SphereLight). A delta
light through delta glass is an SDS path the path tracer fundamentally cannot
sample, and BDPT's light subpath historically skipped distant lights — so only
SPPM's photon pass rendered that real transport, as needle-sharp filaments read
as fireflies. Estimator form (the per-pass film-mean is algebraically identical
to τ-accumulation), initial radius, photon count, RR floor, and mesh
tessellation were all experimentally ruled out. The fix lives in change
`distant-light-caustic-parity`: the default light is injected only into
scenes that author no powered light, and BDPT gained the distant-light
subpath walk so authored suns over glass render their caustics on every
integrator. Methodology keepers: mask on a reference's own bright pixels (a
geometric box mixes caustic and shadow), split regions by brightness tercile,
and never compare SPPM via `direct_light_index=1` (it strips SPPM's eye direct
differently from the megakernel integrators).

## Verification

On Vulkan (Apple M5 Pro), a Cornell-box scene renders an **SPPM / path energy
ratio of 1.008** — neither double-counting the direct term nor missing the
indirect/caustic complement, confirming the NEE-owns-direct / photon-owns-indirect
split. The caustic parity gate (a glass object over a diffuse plane, rendered
under pbrt v4 `sppm` as the reference EXR and compared via `parity.py`) exercises
the specular→diffuse regime SPPM is built for.

## Deferred phases

PM-1 is the **surface (flat-material)** SPPM estimator. Two follow-up changes
extend the same `photon-mapping` capability spec; each is its own reviewable
implementation plan and its own research problem:

- **PM-2 — skin / BSSRDF photon path.** The layered skin estimator chain
  (subsurface scattering, the §1–§6 BSSRDF estimator) is untouched in PM-1; a
  diffusion / photon-beam deposit onto the skin layers is deferred.
- **PM-3 — volumetric / media photon transport.** Photon transport through
  participating media (the delta-tracked volume path, Henyey-Greenstein phase) is
  deferred.

Until then, a non-flat first hit (skin / python / debug) terminates the eye and
photon walks inactive, and media are not part of the photon transport.

## Papers and references

| Area | Reference |
|------|-----------|
| Photon mapping | Jensen, "Global Illumination using Photon Maps", EGWR 1996 |
| Progressive photon mapping | Hachisuka, Ogaki, Jensen, "Progressive Photon Mapping", SIGGRAPH Asia 2008 |
| Stochastic progressive photon mapping | Hachisuka, Jensen, "Stochastic Progressive Photon Mapping", SIGGRAPH Asia 2009 |
