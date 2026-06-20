# Design — flat-lobes-rich-inputs (Stage-2 Tier A)

## Stage-2 scope (where this sits)

Stage-1 (`pbrt-mtlx-export`, shipped) made the exporter carry rich
`standard_surface` params and packs them into binding 19, but the production
integrators ignore them (they shade the `FlatMaterial` subset). Stage-2 = grow
the **unified lobe model** to read those params, **without** calling
`evalStdSurfaceBSDF` (preview-only per `flat-bsdf-lobes`). The roadmap is a
gradient, decomposed into independent changes:

| Change | Adds | Cost / risk |
|--------|------|-------------|
| **Ch2 — this change** | tints/response only: `transmission_color`, `specular_color`, `diffuse_roughness` (Oren-Nayar) + smooth colored glass | M / **low** (no new lobes, pdf untouched) |
| Ch3 | anisotropic GGX, thin-walled translucent (new lobes) | M–L / med |
| Ch4b | rough dielectric BTDF (rough glass) | L / **high** — BDPT η² adjoint |
| Ch5 | subsurface BSSRDF | XL / research |

This change is deliberately the part that needs **no pdf change**, so it can land
fast and de-risk ~the colored-glass / tinted-spec fidelity goal before the harder
lobe work.

## Current state (what's there to extend)

`FlatHitMat` (flat_shading.slang:71) = `{albedo, roughness, metallic, specular,
ior, opacity, emission, coat, coatRoughness, coatIOR, coatColor}`. The unified
lobe set (flat_lobes.slang) is `{coat, spec, diffuse}` GGX/GGX/Lambert plus a
**delta** transmission branch in `FlatMaterial.sample()` (flat_material.slang:32)
that refracts with `weight = albedo`, gated on `opacity < 1`.
`pack_flat_material` (renderer.py:377) packs `FlatMaterialParams` (binding 14)
from `parameter_overrides`; `pack_std_surface_params` already packs
`transmission_color`/`specular_color`/`diffuse_roughness` into binding 19 (unread
by the estimator).

## Ch2 mechanics

### 1. Smooth colored glass (delta — zero invariant risk)
The delta transmission branch already exists and carries pdf = 0 (MIS weight 1,
no NEE partner, no hemisphere bookkeeping). Change only the **weight**:
`weight = transmissionColor` instead of `albedo`. Because it stays delta, none of
the single-pdf / MIS / bounded-weight machinery is touched. Back-compat:
`transmissionColor` defaults to `albedo` when `transmission_color` is absent (so
UsdPreviewSurface glass — `diffuseColor = (1,1,1)`, `opacity = 0` — stays white,
byte-identical). This is where the `-mtlx` export starts to matter: it supplies a
non-white `transmission_color` UsdPreviewSurface cannot express.

### 2. specular_color tint (response-only)
The spec lobe weight (`F · G₁` in both `sample()` and `flatBsdfResponse`)
multiplies by `specularColor`. It is a constant per-hit factor on the lobe
response — the **pdf is unchanged** (the GGX VNDF draw/density don't depend on
it), so `sample().pdf == evaluate().pdf` holds. Bounded: `specularColor ∈ [0,1]`
keeps the weight bounded by construction. Default white ⇒ no change.

### 3. Oren-Nayar diffuse (response-only)
Replace the Lambert diffuse **response** with Oren-Nayar driven by
`diffuseRoughness`, keeping **cosine sampling** (so the diffuse lobe's pdf is
unchanged — still `NdotL/π`). `response/pdf` stays bounded (Oren-Nayar ≤ ~1.05×
Lambert at grazing; still firefly-free). `diffuseRoughness = 0` ⇒ exact Lambert ⇒
no change. Must update **both** `sample()`'s diffuse weight and
`flatBsdfResponse`'s diffuse term identically (the single-source-of-truth rule).

### Plumbing
- `FlatHitMat` += `transmissionColor` (float3), `specularColor` (float3),
  `diffuseRoughness` (float).
- `FlatMaterialParams` struct (binding 14) + `pack_flat_material` += the three
  fields; `pack_flat_material` reads `transmission_color` (fallback `diffuseColor`),
  `specular_color` (fallback white), `diffuse_roughness` (fallback 0).
- `fetchFlatHitData` populates them.
- **MSL note:** adding fields shifts the `FlatMaterialParams` scalar layout —
  re-check the Metal stride / offsets (`pack_*_msl` if one exists) and the
  `_current_state_hash` is unaffected. Keep the Vulkan SPIR-V byte-identical
  except the cbuffer growth (behaviorally unchanged for default inputs).

## Invariants & gates (the tax)

`flat-bsdf-lobes` requires: single pdf, bounded weight (no clamp), unbiased
mixture, canonical BSDF for PT+BDPT in both modes. Ch2 touches only **weights /
responses** (and a delta tint), never the **pdf** — so the invariants hold by
construction, but must be re-verified:

- four-way convergence PT-BSDF / PT-(BSDF+Env) / PT-Env / BDPT on
  `three_materials`, megakernel **and** wavefront (existing gate; values for
  default inputs must be unchanged).
- Metal ↔ Vulkan shaded parity.
- a new **colored-glass** parity scene (a tinted `dielectric`) proving the tint
  reaches pixels and matches a pbrt v4 reference; and a tinted-`specular_color` /
  `diffuse_roughness` check.
- pbrt parity corpus unchanged for the existing (untinted) scenes — back-compat
  proof.
- firefly-free, no clamp added.

## Risks

- **Oren-Nayar bounded-weight:** confirm `response/pdf` stays bounded under the
  proposal mixture (the firefly-free-by-construction property). If Oren-Nayar's
  grazing term needs care, keep the qualitative Oren-Nayar (bounded) rather than a
  full energy-preserving variant.
- **MSL stride drift:** the binding-14 layout grows; verify offsets on Metal
  (history shows fc-blob/stride pins are alignment-sensitive).
- **Back-compat:** the corpus parity gate is the guard — any non-zero delta on an
  untinted scene means a default leaked.
