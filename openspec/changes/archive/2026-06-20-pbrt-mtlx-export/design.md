# Design — pbrt MaterialX export (+ Stage-2 fidelity roadmap)

This change ships only the **exporter** (Stage 1). The fidelity payoff it enables
lives in a separate, parity-gated rendering track (Stage 2, Changes 2–5 below).
This document records the full exploration so the gradient and its hard
constraints are not lost.

## Context — how materials flow today

```
pbrt scene ─▶ materials.map_material ─▶ UsdPreviewSurface ─▶ .usda
                                                              │
usd_loader._extract_material / _load_mtlx_materials ◀─────────┘
   │
   ├─ renderer.pack_flat_material      ─▶ binding 14  FlatMaterialParams  ─┐
   └─ renderer.pack_std_surface_params ─▶ binding 19  StdSurfaceParams ────┤
                                                                           │
   path / bdpt / wavefront_sppm / main_pass  ─ loadFlatMaterial ─▶ FlatMaterial.sample()
        (ALL production integrators)            (reads binding 14 subset)  │
                                                                           │
   preview_pass.slang (Vulkan-only debug viewport) ─ evalStdSurfaceBSDF ◀──┘
        (ONLY consumer of binding 19 / the rich closure)
```

Two load-bearing facts:

1. **`evalStdSurfaceBSDF` is preview-only by design**, not omission.
   `flat-bsdf-lobes` (spec.md:36) mandates the unified `FlatMaterial` lobe model
   as the single BSDF for PT and BDPT in both execution modes, and states
   `evalStdSurfaceBSDF` SHALL NOT appear in the estimator path. The unified model
   guarantees `sample().pdf == evaluate().pdf`, bounded per-lobe weight
   (firefly-free *by construction*, no clamp), and an unbiased proposal mixture
   (ReSTIR / neural guiding / BDPT all consume `evaluate()`). The std_surface
   closure tree has none of those guaranteed.

2. **Binding 19 (`StdSurfaceParams`) is already packed for every material** and
   already in every pipeline's descriptor set — `pack_std_surface_params` reads
   both standard_surface *and* UsdPreviewSurface names. The rich slots
   (`transmission_color`, `subsurface_radius`, `specular_anisotropy`,
   `thin_walled`, …) are simply **default-filled** today because UsdPreviewSurface
   can't express them. The data path exists; both ends are empty.

⇒ Fidelity is gated on two independent things: (a) the **exporter** filling the
rich slots with real pbrt values (this change), and (b) the **lobe model**
growing to read them (Stage 2). Neither alone changes pixels.

## Exporter design decisions (this change)

- **Sidecar `.mtlx`, not inline UsdShade** (per explore decision). Produces a
  portable standalone MaterialX document — the interop deliverable — referenced
  from the `.usda`. Loader path: `_collect_mtlx_asset_paths` finds the
  reference, `_load_mtlx_materials` parses it, matching `surfacematerial` element
  names to bound USD material leaf names.
- **Shadowing hazard.** `_load_mtlx_materials` is a *fallback* used "when
  `ComputeBoundMaterial` fails (missing usdMtlx plugin)". If the `.usda` also
  binds a competing UsdPreviewSurface material, `ComputeBoundMaterial` succeeds
  and the `.mtlx` is ignored — the richer params never load. Decision: under
  `-mtlx` the exported stage SHALL make the MaterialX document authoritative
  (bind to it / not author a shadowing UsdPreviewSurface for the same material),
  and SHALL yield the **same** `Material` whether the host has the usdMtlx plugin
  (→ `_extract_material` reads `outputs:mtlx:surface`) or not (→
  `_load_mtlx_materials` fallback). Both intake paths must be exercised in tests.
- **OpenPBR vs standard_surface.** Target `standard_surface` (Autodesk) input
  names — `pack_std_surface_params` and `_STD_SURFACE_TO_FLAT` read those
  directly; the `_OPENPBR_TO_STD_SURFACE` alias table exists if OpenPBR is chosen
  later, but standard_surface is the lower-friction target.
- **Roughness calibration parity.** Reuse the exact chain in `materials.py`
  (`pbrt_roughness_to_alpha`, `alpha_to_usd_roughness`) so `-mtlx` and the
  UsdPreviewSurface export agree on roughness; anisotropic `uroughness`/
  `vroughness` map to `specular_roughness` + `specular_anisotropy` instead of
  collapsing to the isotropic geometric mean.

## Stage-2 roadmap — growing the unified lobe set (separate changes)

Goal: make the production integrators consume the rich params this exporter
writes, **without** calling `evalStdSurfaceBSDF` (forbidden by `flat-bsdf-lobes`)
— i.e. by adding lobes/terms to the unified model while preserving its
invariants. Every Stage-2 change MODIFIES `flat-bsdf-lobes` and must clear:
four-way convergence PT-BSDF / PT-(BSDF+Env) / PT-Env / BDPT on `three_materials`
(megakernel + wavefront); Metal↔Vulkan shaded parity; the pbrt parity harness
(relMSE+FLIP vs pbrt v4); proposal-mixture unbiasedness (brass-no-darkening);
firefly-free with no clamps.

| Change | Scope | pbrt features unlocked | Cost | Risk |
|--------|-------|------------------------|------|------|
| **2 — flat-lobes-tierA** | fill existing lobes from binding-19 inputs; **+ smooth colored glass (4a)** | `transmission_color` tint, `specular_color`, conductor complex-IOR Fresnel, Oren-Nayar `diffuse_roughness`, colored *smooth* dielectric | M | low — tints + a **delta** transmission (pdf=0), rides the working delta path |
| **3 — flat-lobes-aniso-translucent** | new lobes | anisotropic GGX (`uroughness`/`vroughness`), `thin_walled` translucent (`thindielectric`, `diffusetransmission`) | M–L | med — new non-delta lobes folded into the unified pdf + per-lobe sampler seam |
| **4b — flat-lobes-rough-dielectric** | first **non-delta transmission** | rough glass (`dielectric` roughness>0) | L | **high** — see below |
| **5 — flat-subsurface-bssrdf** | BSSRDF | pbrt `subsurface` radius/anisotropy | XL | high — BSSRDF ≠ BSDF; integrator surgery, likely reuse skin path |

### The 4a / 4b fault line (most important finding)

The "glass" gap is mostly cheap. pbrt `dielectric` defaults to roughness 0 →
**delta**. Today `FlatMaterial.sample()` already does delta refract/reflect, but
tints by `albedo` and gates on `opacity` (refracts *white*). **4a** = carry
`transmission`/`transmission_color` in `FlatHitMat`, tint the delta branch,
gate on `transmission`. Stays delta (pdf=0) → **no BDPT risk**, covers most pbrt
glass. It belongs in Change 2.

**4b (rough glass) is the research tail, and its real cost is BDPT, not the
lobe.** It would be the first **non-delta transmissive lobe** in the system:

- New GGX **refraction** VNDF sampler (none exists; `samplers/ggx.slang` is
  reflection-only). Importance weight reduces to `(1-F)·G₁` → **bounded by
  construction, so the firefly-free invariant does extend** — the unified-model
  premise survives a transmission lobe.
- `flatBsdfPdf` / `flatBsdfResponse` must branch into the lower hemisphere with
  the refraction Jacobian and keep `sample().pdf == evaluate().pdf`; `evaluate()`
  gains eta/side bookkeeping it lacks today.
- **Killer constraint:** BDPT today treats *all* transmission as delta
  (`deltaBounce = bs.pdf <= 0.0`, bdpt.slang:445) and excludes transmitted
  directions from MIS (`!bs.transmitted`); there is **no η²/adjoint refraction
  correction anywhere** because delta-only transmission never needs it. A
  non-delta BTDF forces de-delta-ing the BDPT path (the `pdfFwdOmega` sentinel,
  the transmitted-MIS exclusions) **and** adding the **η² refraction adjoint**
  (Veach non-symmetry, light- vs eye-subpath). `flat-bsdf-lobes` requires PT≡BDPT
  convergence, and refraction non-symmetry is exactly where naive BDPT diverges.
  So 4b is a **BDPT estimator change**, not a lobe-file edit. Consider a carve-out
  (rough glass PT-only, or an explicit BDPT-refraction extension) when scoping it.

### Net feasibility verdict

- **Interop half:** clean — this change, days.
- **Fidelity half:** a gradient. Change 2 (incl. smooth colored glass) is
  achievable and high-value at low risk; Changes 3–5 are progressively heavier,
  with rough glass (4b) and BSSRDF (5) genuine research isolatable into their own
  gated changes. ~80% of the "glass fidelity" goal de-risks into Change 2.
