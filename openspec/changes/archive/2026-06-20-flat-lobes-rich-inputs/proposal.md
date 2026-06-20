## Why

The pbrt `-mtlx` exporter (`pbrt-mtlx-export`) and the MaterialX loaders now feed
rich `standard_surface` parameters into the renderer — `transmission_color`,
`specular_color`, `diffuse_roughness` — and `pack_std_surface_params` already
packs them into binding 19. But the **production** integrators (path / BDPT /
wavefront SPPM / megakernel) shade flat materials through the unified
`FlatMaterial` lobe model (`materials/flat/flat_lobes.slang`), whose `FlatHitMat`
carries only the UsdPreviewSurface subset (`albedo, roughness, metallic,
specular, ior, opacity, coat*, emission`). So those richer parameters are
**dead** — colored glass renders white, metals carry no `specular_color` tint,
diffuse is always Lambert. This is the documented "Stage-2" gap: the export
carries the data; nothing reads it.

`evalStdSurfaceBSDF` (the full closure that *does* read them) is preview-only **by
design** — `flat-bsdf-lobes` forbids it in the PT/BDPT estimator because it lacks
the single-pdf / bounded-weight / unbiased-mixture guarantees the integrators,
ReSTIR, and the neural proposal depend on. So the fix is **not** to call it; it is
to make the unified lobe model itself consume the extra inputs.

This change is **Tier A** of the Stage-2 roadmap (see the archived
`pbrt-mtlx-export/design.md`): fill the **existing** `{coat, spec, diffuse,
delta-transmission}` lobe set from the richer params, adding **no new lobes** and
**no transmissive BTDF**. It is the high-value, low-risk entry point — every item
is a tint/response change that leaves the pdf structure untouched, so the
`flat-bsdf-lobes` invariants hold by construction. Paired with the `-mtlx`
exporter it delivers the first real fidelity win (colored glass, tinted speculars)
without the BDPT-adjoint surgery (Ch4b) or BSSRDF work (Ch5).

## What Changes

- `FlatHitMat` (and `FlatMaterialParams` binding 14 + `pack_flat_material` +
  `fetchFlatHitData`) SHALL gain `transmissionColor` (float3), `specularColor`
  (float3), and `diffuseRoughness` (float), packed from the
  `transmission_color` / `specular_color` / `diffuse_roughness` overrides.
- **Smooth colored glass:** the unified BSDF's **delta** transmission branch
  SHALL tint by `transmissionColor` instead of `albedo`. It stays a delta event
  (pdf = 0) — no MIS / hemisphere / Jacobian change — so no invariant is touched.
- **Specular tint:** the GGX spec lobe's reflectance SHALL multiply by
  `specularColor` (a response-only factor; pdf unchanged).
- **Oren-Nayar diffuse:** the diffuse lobe SHALL use the Oren-Nayar response
  driven by `diffuseRoughness` (a response-only change; the lobe keeps cosine
  sampling, so its pdf — and thus `sample().pdf == evaluate().pdf` — is
  unchanged).
- All three SHALL be **back-compatible**: absent inputs default to the prior
  behavior (`transmissionColor` falls back to `albedo`; `specularColor` = white;
  `diffuseRoughness` = 0 ⇒ Lambert), so existing UsdPreviewSurface renders and the
  pbrt parity corpus are byte-unchanged.
- The `flat-bsdf-lobes` invariants SHALL be preserved and re-verified: single pdf
  (`sample().pdf == evaluate().pdf`), bounded per-lobe weight (firefly-free, no
  clamp), unbiased proposal mixture, and four-way PT-BSDF / PT-(BSDF+Env) / PT-Env
  / BDPT convergence on `three_materials`, in both execution modes and on both
  Metal and Vulkan.

## Non-Goals (deferred to later Stage-2 changes)

- **Anisotropic GGX** (`specular_anisotropy`, `uroughness`/`vroughness`) and
  **thin-walled translucent** (`thindielectric`, `diffusetransmission`) — new
  lobes in the unified pdf. → Ch3.
- **Rough dielectric BTDF** (rough glass, `dielectric` roughness > 0) — the first
  non-delta transmissive lobe; requires de-delta-ing the BDPT path and the η²
  refraction adjoint. → Ch4b (research).
- **Subsurface BSSRDF** (`subsurface_radius`) — not a BSDF; integrator surgery,
  likely reusing the skin path. → Ch5 (research).
- **Conductor angular complex-IOR Fresnel** — conductors keep the import-time
  artistic `base_color` (RGB normal-incidence) approximation for now.
- The `-mtlx` subsurface/coated round-trip equivalence regression is a separate
  exporter fix (tracked follow-up), independent of this lobe work.
