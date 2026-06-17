## Context

skinny loads scenes exclusively through USD (`usd_loader._read_usd_stage` →
`Scene` of `MeshInstance`/`Material`/`LightDir`/`LightSphere`/`LightEnvHDR`/
`CameraOverride`). The loader already understands UsdGeom meshes, UsdLux lights,
UsdPreviewSurface + MaterialX `standard_surface`/OpenPBR shaders, a PBRT-style
thick-lens camera (`LensSystem`/`LensElement`, front-to-rear, signed radius,
aperture-stop marking), and per-material `customData` extensions
(`skinnyMaterialX`, `skinnyOverrides`). Headless rendering exposes a linear-HDR
accumulation readback distinct from the tonemapped sRGB display path
(`render_headless`).

pbrt v4 is spectral by default, left-handed, unitless, and has its own
graphics-state model (CTM stack, named materials/textures, area-light state,
`Object`/`ObjectInstance`). The job is to translate a pbrt v4 text scene into a
USD stage the existing loader consumes, then prove the rendered images nearly
match pbrt's own, gated by an automated metric. No skinny code outside the new
`src/skinny/pbrt/` package changes.

This design references `proposal.md` for motivation and the two spec files
(`pbrt-scene-import`, `pbrt-parity-validation`) for the normative requirements.

## Goals / Non-Goals

**Goals:**
- Parse the common pbrt v4 text surface (directives, params, graphics state,
  `Include`/`Import`, instancing, `.ply`/`.pbrt` assets) in pure Python.
- Translate shapes, materials, lights, camera, textures, spectra, and
  homogeneous media into a USD stage loadable unchanged by `usd_loader`.
- Achieve **near-matching images** between skinny and pbrt v4 on a curated
  corpus, gated by FLIP + relMSE against checked-in reference EXRs.
- Emit a translation report and maintain a parity matrix (matched / approximated
  / unsupported) so divergence is always explicit, never silent.

**Non-Goals:**
- skinny → pbrt export (direction is import-only).
- Full spectral rendering — skinny is RGB; spectra are reduced to RGB.
- Heterogeneous (grid/VDB) media, photon/BDPT pbrt integrators, motion blur /
  animation parity (single static frame only), and exact sampler-sequence match.
- Modifying `usd_loader`/renderer spec behavior — translation targets only what
  the loader + renderer already support; the rest is best-effort + flagged.

## Decisions

### D1 — USD as the bridge (not a direct `Scene` build)
Emit a `.usda`/`.usd` stage and feed it to the existing loader. Reuses the
mature extraction, MaterialX runtime, thick-lens camera, scene-graph/edit-layer,
and animation paths for free; the importer only has to *author* USD, never
render. Constructs skinny supports but USD/UsdPreviewSurface can't cleanly carry
(exact lobe weights, media params) ride along as `customData.skinnyOverrides` /
`skinnyMaterialX`, which the loader already merges into `Material`.
*Alternatives:* build `Scene` dataclasses directly (rejected — duplicates the
loader, bypasses scene-graph/animation/MaterialX); hybrid USD+side-channel
(folded into this decision via `customData`, no separate format).

### D2 — Pure-Python pbrt v4 parser (own tokenizer + state machine)
The pbrt format is small: a token stream of directives, quoted strings, numbers,
bracketed arrays, and `"type name" [values]` typed parameters. Implement a
tokenizer + recursive `Include`/`Import` resolver + a graphics-state machine that
mirrors pbrt exactly: CTM stack (`AttributeBegin/End`, `TransformBegin/End`),
`CoordinateSystem`/`CoordSysTransform`, current material/named materials, current
named textures, current area-light, reverse-orientation, and
`ObjectBegin/ObjectInstance` for instancing. Split into an **options block**
(camera, film, sampler, integrator, accelerator, color space) before
`WorldBegin` and a **world block** after.
*Alternatives:* vendor the C++ `pbrt-parser` (build burden, v3-leaning, FFI);
shell out to a pbrt binary's converter (requires pbrt installed, partial export).
Pure Python keeps it dependency-free and CI-friendly.

### D3 — Coordinate, winding, and unit bridge
pbrt is left-handed with `LookAt` producing a world-to-camera transform and the
camera looking down +z; USD/skinny is right-handed. Apply one fixed
change-of-basis `B` (z-flip) to every CTM so geometry, normals, and camera land
in skinny's right-handed world with consistent triangle winding; invert
`LookAt` to a camera-to-world and compose with `B`. Carry pbrt's unitless scale
1:1 into USD and set the scene scale (`metersPerUnit` / skinny `mm_per_unit`) so
thick-lens geometry and inverse-square light fall-off stay self-consistent.
Camera-ray and point-projection golden tests pin the transform.

### D4 — Material mapping table (parity-critical)
Map each pbrt material onto the closest skinny flat lobe set / MaterialX
`standard_surface`, with the two largest parity levers — **roughness remap** and
**conductor IOR→RGB** — replicated exactly and unit-tested:

| pbrt material | skinny target | notes |
|---|---|---|
| `diffuse` | Lambert (`diffuseColor` = `reflectance`) | pbrt v4 diffuse is Lambert; direct |
| `conductor` | metallic GGX, `metallic=1` | `baseColor` = Fresnel-conductor reflectance at normal incidence from (η,k) or named metal; roughness via pbrt remap |
| `dielectric` | dielectric (`ior`, transmission→opacity gate) | smooth/rough glass; honor `eta` |
| `thindielectric` | thin-glass approximation | flagged approx |
| `coateddiffuse` / `coatedconductor` | coat lobe + base | `coat`/`coatRoughness`/`coatIOR` |
| `diffusetransmission` | diffuse + transmission | two-sided |
| `subsurface` | dielectric boundary + homogeneous interior medium | maps to `volume_render` HG where renderable; else flag |
| `mix`, `interface`, unknown | best-effort / passthrough | reported |

Roughness: pbrt remaps `roughness`→α unless `remaproughness false`; replicate
pbrt v4's mapping precisely (and the `uroughness`/`vroughness` anisotropy path)
rather than skinny's perceptual default.

### D5 — Spectrum → RGB reduction
Reduce every spectrum to linear RGB by integrating against CIE XYZ under the
scene's rendering color space (default sRGB/Rec709 unless a `ColorSpace`
directive says otherwise): named spectra (`metal-Au-eta`, glass IORs), `blackbody
[T]`, sampled `.spd`, and RGB-as-reflectance/-illuminant. RGB inputs pass through
(pbrt upsamples-then-reintegrates ≈ identity for in-gamut reflectances; the
residual is documented). Conductor (η,k) spectra collapse to an RGB F0 via
Fresnel. This is an inherent skinny-RGB vs pbrt-spectral divergence absorbed by
the parity tolerance and recorded per-scene; the corpus favors near-neutral
spectra to keep the gap small.

### D6 — Light mapping
`distant`→`DistantLight`; `point`→`SphereLight` with a small radius;
`infinite`→`DomeLight` (importer writes/links an equirect `.hdr` since dome
extraction is `.hdr`-only and collapses color×intensity to a scalar — energy
verified); area (`diffuse` on a shape)→ skinny's synthesized emissive mesh,
matching pbrt one-sided default vs `twosided`. `spot` has no skinny equivalent →
best-effort (sphere light) + flag in the report and parity matrix. Light
radiance follows pbrt's `power`/`scale`/`L`/`illuminance` semantics reduced to
linear-HDR RGB.

### D7 — Camera mapping
`perspective`: pbrt `fov` addresses the **shorter** image axis; convert to
skinny's vertical FOV + vertical aperture accounting for aspect (and
`screenwindow`). `realistic` (lens description) maps almost directly to
`LensSystem`/`LensElement` (same front-to-rear / signed-radius / aperture-stop
convention) — strong reuse. `focaldistance`/`lensradius`/`aperturediameter` →
`focus_distance`/`fstop`.

### D8 — Linear-HDR parity comparison, converged
Compare skinny's **linear-HDR accumulation** (`read_accumulation_hdr`) against
pbrt's linear EXR, bypassing tonemap/sRGB entirely, so the gate measures
transport+shading, not display. Render both to a fixed high spp so Monte-Carlo
noise sits below tolerance; map pbrt `integrator "path"`/`maxdepth` → skinny path
tracer + max bounces. Any pbrt sensor/imaging scalar (`iso`, `exposuretime`) is
replicated as a single global multiplier. Metrics: **relMSE**
`mean((a-b)²/(b²+ε))` (robust to a residual global scale + noise) and **FLIP**
(perceptual, computed on identically-tonemapped copies). Per-scene tolerances in
a corpus manifest.

### D9 — Corpus + reference EXRs checked in
A handful of tiny (≤256px) scenes, each isolating one axis: diffuse sphere under
an area light; conductor under an env map; glass under an area light; coated
material; homogeneous-medium/subsurface; instanced grid; thick-lens DOF.
pbrt-rendered reference EXRs are committed (force-added — repo `.gitignore` is
`*`) with a manifest pinning the pbrt version + per-scene tolerance + scene hash.
References are golden, regenerated deliberately (documented), never live in CI.

### D10 — Assets, entry points, report
Minimal own `.ply` reader (ascii + binary little/big-endian) to avoid a runtime
dep; texture images referenced as-is (skinny loads png/jpg/exr/hdr). CLI
`skinny-import-pbrt scene.pbrt -o scene.usda` + `python -m skinny.pbrt` +
`skinny.pbrt.import_pbrt(path) -> Stage`. Every run emits a translation report
listing exact maps, approximations, and skips, keyed to the parity matrix.

## Risks / Trade-offs

- **Spectral→RGB divergence is fundamental** → curate corpus toward near-neutral
  spectra; tolerance absorbs the residual; per-scene error logged in the parity
  matrix; assume sRGB rendering color space by default and honor `ColorSpace`.
- **Roughness-remap mismatch silently shifts highlights** → replicate pbrt v4's
  exact remap (and `remaproughness false`) and unit-test α against known pbrt
  values before any image compare.
- **Conductor complex-IOR reduced to a normal-incidence RGB F0** loses angular
  Fresnel tint vs pbrt's spectral Fresnel → accept for v1, document, prefer
  low-saturation metals in the corpus.
- **FOV-axis / handedness bugs reframe the image and fail parity opaquely** →
  golden transform + camera-ray unit tests projecting known world points before
  rendering.
- **Dome-light scalar collapse + `.hdr`-only loader** → importer emits an
  equirect `.hdr` and verifies integrated energy; EXR env maps converted, not
  referenced raw.
- **spot / heterogeneous media / generic subsurface unsupported in skinny** →
  best-effort translate + explicit report entry + parity matrix "approx/
  unsupported"; never emit a confidently-wrong render.
- **Reference EXRs drift across pbrt versions** → manifest pins pbrt version +
  scene content hash; refs are golden artifacts regenerated on purpose.
- **Monte-Carlo noise above tolerance** → fixed high spp for refs and skinny,
  relMSE chosen for noise/scale robustness, no denoiser in the loop.
- **`customData` extensions could mask loader gaps** → translation only emits
  extensions the loader already reads; anything beyond loader/renderer support is
  flagged, not faked.

## Migration Plan

Purely additive offline tooling — no runtime feature flag, no rollback surface.
Suggested implementation order (drives `tasks.md`): (1) tokenizer + param parser
+ graphics-state machine; (2) geometry + transforms + instancing → USD with
golden transform tests; (3) camera + lights; (4) materials + roughness remap +
spectrum→RGB; (5) homogeneous media + subsurface best-effort; (6) CLI/Python
entry + report; (7) parity harness (FLIP/relMSE) + corpus + reference EXRs + the
parity-matrix doc. Each stage is independently testable; image parity is gated
only after (1)–(4) land for a given corpus scene.

## Open Questions

- Default rendering color space when a scene omits `ColorSpace` — assume
  sRGB/Rec709; confirm against pbrt v4's actual default.
- Test-only EXR read: vendor a minimal RGBA-scanline EXR reader vs a dev
  dependency (`OpenEXR`/`imageio`) — lean minimal unless half-float/tiled EXRs in
  the corpus force a dep.
- How `subsurface`'s dielectric-interface + interior-medium model best maps onto
  skinny's HG `volume_render` without the skin BSSRDF — may need a small
  homogeneous-medium material path or stay best-effort + flagged for v1.
