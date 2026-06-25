## Why

`assets/dragon_removed.usda` (a pbrt `coateddiffuse` floor, exported as a
`UsdPreviewSurface` with `clearcoat = 1` / `clearcoatRoughness = 0.669`) renders
with a large dark region under a near-uniform environment. It should read as an
almost single-colour, diffuse-dominated surface with only a faint glossy coat.

The regression surfaced with commit `eb1a0f2` (pbrt-mtlx-roundtrip-fix). That
commit's `_canonicalize_coat` folds UsdPreviewSurface `clearcoat`/
`clearcoatRoughness` onto the canonical `coat`/`coat_roughness` slots so a coated
material's coat lobe survives the flat-material pack. Before it, the coat was
silently dropped (`coat = 0`) — the surface was pure diffuse and looked uniform.
The commit correctly turns the coat lobe **on**, which exposed a **latent
energy-loss bug in the flat coat lobe**:

The coat lobe's selection probability is `pCoat = coat · fresnelDielectric(NdotV,
coatIOR)`. `fresnelDielectric(cosI, eta)` (flat_shading.slang) uses the Snell
convention `sin²θt = eta²·(1−cos²θi)`, i.e. **`eta = η_incident / η_transmitted`**.
The view ray *enters* the coat from air, so the entering ratio is `1/coatIOR ≈
0.667`. The coat lobe instead passes `coatIOR = 1.5` **raw** — the *exiting*
(coat→air) direction — which triggers spurious total internal reflection for any
view angle beyond ~42° from the local normal (`sinT2 = 2.25·sin²θ > 1`). `pCoat`
then saturates to `1.0` over most of the visible surface, so the base
diffuse/spec lobes (attenuated by `1 − pCoat`) are zeroed, while the coat
reflection lobe correctly uses Schlick `F0 = 0.04`. The base energy is thrown
away and only a faint coat reflection survives → the dark region.

Measured on Metal/Path (same env): the coated render's mean drops to **101.9 vs
240.6** uncoated — a 2.4× energy loss. The glass-refraction branch
(`flat_material.slang:39`, `entering ? 1/ior : ior`) and the subsurface boundary
(`subsurface_walk.slang:135`, `fresnelDielectric(NdotV, 1.0/ior)` "reflectance
entering") already use the correct entering convention; only the coat lobe was
wrong.

## What Changes

- **Flat coat lobe Fresnel uses the entering eta.** At the three coat
  lobe-selection sites — `FlatMaterial.sample()` and `evaluate()`
  (`flat_material.slang`) and `flatBsdfResponse()` (`flat_lobes.slang`) — replace
  `fresnelDielectric(NdotV, m.coatIOR)` with `fresnelDielectric(NdotV, 1.0 /
  m.coatIOR)`. `coatIOR` is clamped `≥ 1.0`, so `1/coatIOR ∈ (0, 1]` and no
  spurious TIR occurs (TIR is physically impossible entering a denser medium).
  This makes `pCoat` track the coat's true reflectance (`F0 = 0.04` at normal,
  → 1 only at true grazing), matching the Schlick `F0` already used by the coat
  reflection weight — energy is conserved.
- **Both backends, both execution modes.** `flat_material.slang` /
  `flat_lobes.slang` are shared by the megakernel and wavefront path/BDPT/ReSTIR
  integrators. Recompile the checked-in Vulkan SPIR-V (`main_pass.spv` and the
  wavefront kernels that include the flat material); Metal compiles the Slang
  in-process. The change is response/selection-only — no struct/layout/stride
  change, so packing is byte-unchanged.
- **Regression test.** Add a coat energy-conservation check: a white-coat over a
  mid-grey diffuse under a uniform (furnace-like) environment must render within
  a few percent of the same material with `coat = 0` — it must not crater. Pins
  the entering-eta convention against re-introduction.

## Impact

- Affected specs: `flat-bsdf-lobes` (coat lobe energy behaviour).
- Affected code: `src/skinny/shaders/materials/flat/flat_material.slang`,
  `src/skinny/shaders/materials/flat/flat_lobes.slang`, regenerated Vulkan
  `.spv`, new regression test under `tests/`.
- Parity matrix: no coated/clearcoat scene is in the parity corpus, so the
  dual-gate is unperturbed. Non-coated flat materials (`coat = 0`) are
  byte-unchanged — the coat branch is guarded `m.coat > 0.0`.
- Docs: `docs/SkinRendering.md` / flat-material section note on the coat Fresnel
  convention if one exists; `CHANGELOG.md`.
