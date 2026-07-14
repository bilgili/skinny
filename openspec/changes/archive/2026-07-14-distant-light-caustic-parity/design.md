## Context

Diagnosis (this session, RGB + Metal, `assets/glass_caustics_spectral.usda`):

- The scene authors one SphereLight. The renderer's per-frame distant-light
  mirror (`renderer.py` ~9764) falls back to the **synthesized default
  DistantLight** whenever `_usd_scene.lights_dir` is empty — regardless of other
  authored lights. Headless CLI runs therefore add a phantom sun; the parity
  harness works around it by force-setting `direct_light_index` per scene.
- The phantom sun casts a real glass caustic. Transport support today:
  SPPM photon pass **walks** distant photons (disk emission,
  `wavefront_sppm.slang:615-637`); the path tracer **cannot** sample the path
  (delta light + delta glass); BDPT **skips** it — `sampleLightOrigin` builds a
  `BDPT_VK_LIGHT_DIR` pseudo-vertex usable only for eye-side connections, and
  the light-subpath walk is explicitly not launched
  (`bdpt.slang:1089 if (lo.vertex.kind != BDPT_VK_LIGHT_DIR)`).
- Result: SPPM alone renders the phantom caustic, as needle-sharp filaments →
  read as fireflies. Suppressing the phantom (authoring a zero-intensity
  DistantLight) removes every firefly at the **default** SPPM radius and gives
  `bdpt/path = 1.002`. Estimator form, initial radius, photon count, RR floor,
  tessellation, and `vp.beta` were all experimentally ruled out. pbrt has no
  mismatch because it never injects unauthored lights.

## Goals / Non-Goals

**Goals:**
- Authored-light scenes render with exactly their authored lights on every
  front-end (kill the phantom sun there); truly unlit scenes keep the default
  light + sliders.
- BDPT carries distant-light indirect transport (including specular caustics)
  so `bdpt ≡ sppm` on scenes that *do* author a sun over glass; MIS stays a
  single non-double-counting partition.
- RGB + spectral, Metal + Vulkan, megakernel BDPT (BDPT has no wavefront-only
  variant of this seam beyond the shared shader).

**Non-Goals:**
- SPPM env-indirect dimness (no env photon emission; null-sun scene reads
  sppm/path ≈ 0.78 in sphere-shadow, caustic-mask ≈ 0.94) — recorded follow-up.
- SPPM deposit `vp.beta` omission (through-glass visible points) — follow-up.
- Path-tracer support for delta-delta SDS paths — fundamentally impossible;
  recorded as a per-component exclusion, exactly as pbrt documents it.
- No SPPM changes at all: SPPM was correct here.

## Decisions

**D1 — Synthesis predicate: inject the default light only into light-less scenes.**
One helper on the renderer, `_scene_authors_lights(scene) -> bool`, true when the
loaded USD scene has any of: a **powered** `lights_dir` entry, a **powered**
`lights_sphere` entry, emissive-material triangles (the count
`_upload_emissive_triangles` derives), or an authored dome —
**pinned to `_usd_scene.environment is not None`** (`usd_loader` sets it only for
an authored `UsdLux.DomeLight`; the renderer's built-in HDRI backdrop must NOT
count, or the no-USD/unlit path dies). "Powered" reuses `_has_power`
(`_upload_distant_lights`): a scene authoring only zero-intensity lights counts
as unlit (keeps the default sun) rather than rendering black. The per-frame
mirror becomes: authored powered `lights_dir` → upload those; else if
`_scene_authors_lights` → upload **zero** distant records; else → slider default
light (unchanged for unlit scenes / the no-USD default session).
`direct_light_index` keeps its existing global-off semantics on top.
**Authority is the load-time `_usd_scene`**: scene-graph-editor live edits that
add/remove lights do not re-derive the predicate in v1 (the default-light prim
stays in the editor, mirroring to zero records on authored-light scenes) —
documented limitation. Corollary to document: a geometry-only USD under the
built-in HDRI still gets sun + HDRI (no authored light → default injected), as
the head session expects.
*Alternative rejected:* per-front-end flags (`--no-default-sun`) — leaves every
non-CLI surface wrong by default and diverges from the parity harness's intent.
*Alternative rejected:* only suppress when another *DistantLight* exists — that
is today's behavior, and precisely the bug (a SphereLight scene still gets the
phantom sun).

**D2 — BDPT distant emission mirrors SPPM's photon emitter and pbrt-v3 `SampleLe`.**
The distant light-origin branch samples a real emission ray: origin on a
scene-bounds-covering disk (center `boundsMin + extent/2`, radius
`R = max(|extent|/2, ε)`, offset in the disk basis);
`beta = L·πR² / lightPickPdf`, `pdfPos = lightPickPdf/(πR²)`, `pdfDir = 1` with
`isDelta = true` (delta *direction*). Concrete deltas the reviewer pinned as
load-bearing (mirror `sppmEmitPhoton`'s distant branch,
`wavefront_sppm.slang:615-637`):
- `dirOut = -L.direction` — today's `dirOut = dir` points **toward** the light
  (`connectT1`/`visibleDirectional` convention); the emission ray travels the
  other way.
- `vertex.position` MUST be set to the sampled disk point (today it stays the
  `float3(0)` init); `vertex.N = -dir` stays as-is; the walk ray origin offsets
  `+N·0.002` with `N` now pointing into the scene.
- `fc.sceneBoundsMin/Extent` are packed unconditionally (`_pack_uniforms`), so
  the BDPT disk ≡ the SPPM disk on both backends.
`R` is over-conservative (bounding sphere): that is a pure **variance** cost,
never bias — an optional follow-up may shrink it toward the specular geometry's
projected extent.
*Alternative rejected:* keeping the skip and adding a special "distant splat"
pass — duplicates the whole walk machinery for one light type.

**D3 — MIS bookkeeping: the double-count is analytically certain and the fix is
specified up front (not deferred to the gates).**
Enabling the walk makes the path `camera → A(diffuse) → DIR` reachable by TWO
finite-pdf strategies: `connectT1`'s distant NEE — which today adds
`z.throughput·resp·L.radiance` at **unconditional full weight**
(`bdpt.slang:693-707`, no `powerHeuristic`, no `misWeight`, unlike the sphere
and emissive branches below it) — and the `s = 1` camera splat of the first
walked vertex, whose `splatMisWeight` resolves to 1 (every ratio term crosses a
delta). Left unchanged, that is a hard **2× over-count of all distant direct
lighting**. Required fixes, in the design not the gate:
1. **Bring `connectT1`'s distant branch into the `misWeight` partition**, exactly
   like the emissive branch at `bdpt.slang:776-817`: build a `lit[0]` distant
   pseudo-vertex (`kind = BDPT_VK_LIGHT_DIR`, `pdfFwd = lightPickPdf`,
   `isDelta = true`), fill the endpoint reverse pdfs, call
   `misWeight(eye, s, lit, 1, …)`. The weight MUST degrade to exactly 1.0 when
   no walked partner exists (distant subpath escaped before any surface hit), so
   pure-direct distant scenes (bathroom/dragon) stay byte-consistent.
2. **Extend `convertSAtoArea` with a DIR-as-source case** (`bdpt.slang:98-108`
   currently special-cases only DIR-as-target): the first walked vertex's
   `pdfFwd` must be the parallel-projection area density
   `pdfPos_disk / |N_A·dir|` — **no `cos/d²` falloff**, because the disk-to-hit
   distance is a placement artifact of `center + dir·R`. This is pbrt's
   `InfiniteLightDensity`/`Pdf_Le` analog, wired, not just named. The caustic
   chain is insensitive to this (delta neighbours zero those terms) but every
   non-caustic distant strategy weight — precisely the double-count-prone ones —
   depends on it.
For the delta-glass caustic chain the splat remains the only non-zero strategy
(confirmed against `splatMisWeight`: weight 1). Gate B's no-double-count check
then *verifies* the derivation instead of substituting for it.

**D3b — Scope: three shader files, ~eight walk-skip sites, three origin seeds.**
The DIR walk-skip and origin seed are not one site. Enumerated:
- `integrators/bdpt.slang` — seed `~:247`, skip `~:1089` (megakernel RGB);
- `integrators/bdpt_spectral.slang` — its own `sampleLightOriginS` seed `~:190`
  ("Distant: no light walk spawned … β unused") and skip `~:883`;
- `wavefront/wavefront_bdpt.slang` — its own `sampleLightOriginS` seed `~:227`
  and **six** skip sites (`~:937, 1096, 1151, 1180, 1931, 1949` — light tail,
  staged eye_light, spectral variants).
All three seeds get the D2 emission; all skip sites are removed; the B1 MIS fix
is replicated in `connectT1S` / the spectral and wavefront connects. Wavefront
consequence to handle: the `WF_BDPT_SLOT_NEE`/`SLOT_FULL` lane gate keys off
`lightLen` — distant subpaths that now walk flip those lanes into `SLOT_FULL`,
changing stream-size/heavy-eye budgeting; verify the tiling bounds still hold.
Megakernel ≡ wavefront BDPT self-consistency (task 3.4) is the enforcement.

**D4 — Spectral: reuse the scalar MIS; recolor the distant emission per-λ.**
`bdpt_spectral` already reuses the RGB MIS verbatim; the distant branch recolors
`beta` per-λ via the authored SPD slot exactly as SPPM's emitter does
(`distantLightSpd(slot, sw)`, else `upsampleIlluminantBound`). RGB `.spv` stays
byte-identical under the spectral split (the standing guard).

**D5 — Gates record today's honest baselines; follow-ups must lower them.**
Gate A (phantom policy, the original scene): no SPPM local fireflies at default
radius, `bdpt/path` ≈ 1.00, and SPPM within a *recorded* baseline vs bdpt
(caustic-mask ≈ 0.94, shadow-box ≈ 0.78 — the env-indirect follow-up gap; the
gate pins it so the follow-up must improve it, never hides it). Gate B (authored
sun over glass, new suite scene) is **variance-realistic**: BDPT reaches the
distant caustic only via single-vertex splats from a scene-bounds disk (no
spatial reuse), so at matched spp it is far noisier than SPPM's density
estimate. The gate therefore compares at **matched wall-clock (equal-time) or an
explicitly recorded high-spp budget with firefly-robust region medians**
(median-of-region, same-budget only — the SPPM-dimness methodology lessons),
records the honest number, and never hides a magnitude error behind a loosened
tolerance. `path` is recorded as the excluded integrator for the caustic
component; distant *direct* regions must be unchanged before/after (the
no-double-count check). Gate B's glass is **non-dispersive** in v1 (m9): the
light-side specular chain under `--spectral` otherwise needs a hero-λ collapse
consistent with the eye side — deferred with the dispersion follow-ups.

## Risks / Trade-offs

- [BDPT MIS regression on ordinary distant-lit scenes] → Gate B explicitly
  checks distant *direct* lighting is unchanged (no double-count); the full
  parity matrix re-runs (bathroom/dragon author distant lights).
- [Behavior change from removing the phantom sun] → **BREAKING** for anyone who
  liked the extra sun on authored-light scenes; flagged in proposal + CHANGELOG.
  The parity harness's own `direct_light_index` workaround becomes redundant but
  stays valid (idempotent).
- [Splat MIS weight ≠ 1 on the delta chain] → caustic dim or missing in Gate B;
  the sppm cross-check catches magnitude errors (both integrators must agree
  without either being the reference).
- [Metal watchdog] → no new unbounded loops (walk depth stays `BDPT_MAX_VERTS`;
  megakernel row-band tiling unchanged).
- [Scene-graph editor UX] → default-light prim present but inert on
  authored-light scenes; documented, and the light list still shows authored
  lights as the active set.

## Migration Plan

Renderer policy is a pure host change; BDPT is a shader + `.spv` recompile.
Rollback = revert the commit. No persisted state involved (`settings.json`
sliders keep their meaning for unlit scenes).

## Open Questions

- Should the suite scene for Gate B live in `tests/assets/suite/` (confirming
  suite, with `_gen` builder) or as a plain corpus scene? Default: confirming
  suite, since it is a per-axis discriminating scene by construction.

(The connectT1-distant-MIS question is no longer open: the design review proved
the double-count analytically and D3 now specifies the fix up front.)
