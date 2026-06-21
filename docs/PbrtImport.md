# pbrt v4 scene import

skinny can read a [pbrt v4](https://pbrt.org) text scene and convert it into a
USD stage that the existing `usd_loader` pipeline loads unchanged. The goal is
**near-matching images** between skinny and pbrt v4 for the same scene, validated
by an automated error gate. Direction is **import-only** (pbrt → skinny); the
"exporter" is the pbrt → skinny converter itself.

The implementation lives in `src/skinny/pbrt/` and emits USD via `pxr`
(UsdGeom / UsdLux / UsdPreviewSurface), carrying skinny-specific extensions in
`customData.skinnyOverrides`.

## Usage

```bash
# CLI
.venv/bin/skinny-import-pbrt scene.pbrt -o scene.usda
# module form
.venv/bin/python -m skinny.pbrt scene.pbrt -o scene.usda
```

```python
from skinny.pbrt import import_pbrt

stage, report = import_pbrt("scene.pbrt", out="scene.usda")
print(report)        # exact / approx / skipped per construct
# scene.usda now loads via Renderer(usd_scene_path="scene.usda")
```

The CLI exits non-zero when any construct was **unsupported** (skipped), so
scripts can gate on a clean import. The translation report classifies every
construct as `exact`, `approx`, or `skipped` with a reason — nothing is silently
dropped.

## How it works

| Stage | Module | Notes |
|-------|--------|-------|
| Tokenize | `tokenizer.py` | comments, quoted strings, numbers, `[..]` arrays |
| Parse directives | `parser.py` | typed `"type name" [values]` params, `Include`/`Import`, options/world split |
| Graphics state | `state.py` | CTM stack, named materials/textures, area-light state, reverse-orientation, `ObjectInstance` instancing → IR |
| Transforms | `transform.py` | column-vector 4×4; left→right-handed bridge |
| Spectra → RGB | `spectra.py` | CIE (analytic) + blackbody + named metals |
| Materials | `materials.py` | pbrt material → UsdPreviewSurface |
| Lights | `lights.py` | pbrt light → UsdLux / emissive mesh |
| Camera | `camera.py` | `perspective` fov-axis conversion; `realistic` lens |
| Media/SSS | `media.py` | homogeneous + subsurface best-effort via `customData` |
| Emit USD | `emit.py`, `api.py` | baked world-space meshes, UsdShade networks |

### Coordinate bridge

pbrt is left-handed; skinny/USD is right-handed. Every CTM is mapped by the fixed
change-of-basis `B = diag(1, 1, -1, 1)` (`M → B·M·B`, point `p → B·p`). Because
`B` is a reflection it reverses triangle winding, so geometry baked through the
bake matrix `B·CTM` has its winding flipped when that matrix is
orientation-reversing (XOR the per-shape `ReverseOrientation`). Geometry is baked
to **world-space points with an identity transform**, which sidesteps USD xform
handedness entirely; the camera and lights use authored xforms verified against
the loader.

### Roughness calibration (parity-critical)

- pbrt v4: `alpha = sqrt(roughness)` when `remaproughness` (default), else
  `alpha = roughness`.
- skinny GGX: `alpha = usd_roughness²` (UsdPreviewSurface roughness passes
  straight through).
- therefore the emitted `usd_roughness = sqrt(alpha)`.

Anisotropic `uroughness`/`vroughness` are reduced to an isotropic alpha
(geometric mean) and flagged.

### Spectrum → RGB

skinny is an RGB renderer, so every spectrum is reduced to linear RGB: `rgb`
params pass through; `blackbody [T]` integrates a Planckian SPD against the CIE
XYZ colour-matching functions (Wyman–Sloan–Shirley analytic fit) then XYZ → linear
sRGB; named conductor spectra (`metal-Au-eta`/`-k`) map to a tabulated RGB IOR;
inline sampled spectra integrate directly. The residual RGB-vs-spectral
divergence is inherent and is absorbed by the per-scene parity tolerance.

### Texture-valued parameters

pbrt lets any `FloatTexture`/`SpectrumTexture` material parameter (e.g.
`roughness`, `reflectance`) be a constant **or** a named texture. The importer
mirrors pbrt's `GetFloatTexture`/`GetSpectrumTexture`: a single promoting accessor
(`get_float_texture`/`get_spectrum_texture`) resolves every textureable parameter
to either a constant (pbrt wraps these in a `*ConstantTexture`) or a resolved
texture, so material mapping never branches on float-vs-texture and never calls
`float()` on a texture name. Each textured parameter maps to its **own** USD input
via the `_TEXTURABLE` table (`reflectance → diffuseColor` as `.rgb`/Color3f,
`roughness → roughness` as `.r`/Float) — nothing is assumed to be `diffuseColor`.
`_TEXTURABLE`'s `value_type` is the one source of truth for the connected channel
and is read by `_author_texture`.

A `scale`-wrapped `imagemap` resolves to the inner image (the `scale` factor and
the perceptual roughness remap are not applied to the connection — flagged
`approx`). A texture class the importer does not support (`checkerboard`/`mix`/
`constant`), an unresolvable name, or a textured parameter with no USD texture
input (`eta`, `interface.roughness`) degrades to the parameter's scalar/rgb
default with an `approx` note — pbrt would `ErrorExit`; skinny stays best-effort so
a partly-unsupported scene still imports. This is what lets the `crown` scene
(texture-valued roughness over hundreds of materials, including nested `scale`/
`imagemap`) import.

### Texture UVs

`imagemap` textures are wired to a `UsdUVTexture` connected into the
`UsdPreviewSurface`; the loader samples it through the mesh's `primvars:st`. UVs
are emitted as follows:

- **Explicit UVs pass through** to per-vertex `primvars:st`: a `trianglemesh`
  `uv`/`st` parameter, or a PLY vertex `u/v`, `s/t`, or `texture_u/texture_v`
  (ascii + binary little/big-endian). The orientation-reversing handedness bake
  only reverses per-face index order, so per-vertex `st` stays aligned.
- **Default UVs are synthesized** when a shape carries no source UV but its bound
  material references a texture, matching pbrt's built-in parametrization so the
  texture still samples (rather than reading one constant texel): a `sphere` gets
  parametric per-vertex UVs (`u = φ/φmax`, `v = (θ−θmin)/(θmax−θmin)`); a
  `trianglemesh`/`plymesh` gets pbrt's per-triangle `faceVarying`
  `{(0,0),(1,0),(1,1)}`. Untextured UV-less meshes stay UV-free.

## MaterialX sidecar export (`-mtlx`)

`skinny-import-pbrt … -mtlx` writes, alongside the `.usda`, a portable MaterialX
sidecar (`<out>.mtlx`) of `standard_surface` materials referenced from the stage
(`materials.map_material_mtlx` + `mtlx_emit`). It targets the richer
`standard_surface` slots UsdPreviewSurface cannot express —
`transmission`/`transmission_color`, separate `coat`/`coat_color`/`coat_IOR`,
`subsurface_radius`, `specular_anisotropy` from `uroughness`/`vroughness`,
`thin_walled` — and the loader recovers them via `_load_mtlx_materials` (the
usdMtlx-plugin-absent fallback) or `_extract_material('mtlx')` (plugin present).

For **diffuse / conductor / dielectric** materials `-mtlx` and the
UsdPreviewSurface output still render pixel-identically (verified on
`glass_arealight`: relMSE 0.0215 / FLIP 0.0270 vs pbrt for both, and 0.000000
between them) — the production integrators consume the `FlatMaterial` subset of
either export.

**Stage-2 Tier A (change `flat-lobes-rich-inputs`) closes the first slice of the
gap:** `transmission_color`, `specular_color`, and `diffuse_roughness` are no
longer inert. The unified flat / `standard_surface` lobe BSDF now reads them
directly (the export's richer slots feed the existing `{coat, spec, diffuse,
delta-transmission}` lobe set — still **without** the preview-only
`evalStdSurfaceBSDF`), so **colored glass** (tinted transmission) and **tinted
speculars** now actually render through the flat path, and `diffuse_roughness`
drives an Oren-Nayar diffuse. See [Architecture.md → Flat Material
BSDF](Architecture.md#flat-material-bsdf-materialsflatflat_materialslang--flat_lobesslang).
**Stage-2 Ch5 (change `pbrt-subsurface-volumetric`) imports `subsurface` as a
volumetric interior medium** rather than the old `opacity = 0` clear-glass
lowering. A `subsurface` material becomes a smooth dielectric boundary (`eta`)
wrapping a homogeneous interior medium (σ_a, σ_s, Henyey-Greenstein `g`),
transported by a delta-tracked (Woodcock / null-collision) volumetric random
walk — refract in, march the interior, refract out. Coefficients follow pbrt's
precedence: explicit `sigma_a`/`sigma_s` (× `scale`) → named preset (`Skin1`, …,
the pbrt measured table) → `reflectance` + `mfp` via the Jensen diffuse-albedo
inversion. The `-mtlx` `standard_surface` inputs (`subsurface_color`,
`subsurface_radius`, `subsurface_scale`, `subsurface_anisotropy`) map to the
**same** (σ_a, σ_s, `g`), so native-USD and `-mtlx` imports agree. The
coefficients ride on `skinnyOverrides` `customData`
(`subsurface_sigma_a`/`subsurface_sigma_s`/`subsurface_g`, plus `ior` for the
boundary eta) → merged into `Material.parameter_overrides`; the renderer packs
them inline into `FlatMaterialParams` (binding 13) and tags
`MATERIAL_TYPE_SUBSURFACE` when `subsurface_sigma_*` is non-zero. The σ are
stored **per world unit** — the pbrt mm⁻¹ values divided by the stage's
`mm_per_unit` (1000, from `emit.PBRT_STAGE_METERS_PER_UNIT`) — so the walk's
`σ·L·mmPerUnit` reproduces pbrt's per-scene-unit optical depth instead of a
1000×-inflated, opaque one (change `pbrt-subsurface-unit-scale`; see
`docs/Subsurface.md`). pbrt's
tabulated dipole BSSRDF and skinny's 3D random walk agree qualitatively (both
milky), not bit-for-bit — corpus scene `subsurface_infinite` gates at relMSE
0.079. **Limitations:** the walk transports the environment (IBL) plus a single
distant light; area/emissive lights *inside* the medium are a follow-up.
Heterogeneous / NanoVDB grids and free-standing `MediumInterface` media stay out
of scope (the transport is majorant / null-collision and the medium is
handle-referenced, so they slot in additively). The remainder of Stage-2 —
`specular_anisotropy` and the rough-glass BTDF (rough transmission with MIS) —
stays future work; see the change's `design.md` for the Stage-2 roadmap and the
`flat-bsdf-lobes` (preview-only `evalStdSurfaceBSDF`) constraint.

> **`subsurface` and `coated*` `-mtlx` round-trip** (change
> `pbrt-mtlx-roundtrip-fix`; the interior is now consumed as a volumetric
> random walk per Stage-2 Ch5 above, not an `opacity = 0` clear boundary).
> UsdPreviewSurface maps `subsurface` to a dielectric boundary plus a
> `skinnyOverrides` homogeneous interior, and
> `coated*` to `clearcoat`. The `-mtlx` `.mtlx` fallback once bridged only
> `transmission → opacity`/`emission`, built the `Material` from the document
> alone (so the `over`-prim `skinnyOverrides` interior was never read), and emitted
> `coat` (not `clearcoat`) — so a `-mtlx` subsurface dragon rendered **opaque**
> where the UsdPreviewSurface export rendered translucent (sssdragon `-mtlx`
> relMSE 0.88 / FLIP 0.46, UsdPreviewSurface 0.49 / 0.13). Now the fallback
> (`_load_mtlx_materials`) bridges `subsurface → opacity = 0`,
> `_resolve_material_binding` merges the bound prim's `skinnyOverrides` interior,
> the loader canonicalizes `clearcoat`/`clearcoatRoughness` →
> `coat`/`coat_roughness` (the slots the `FlatMaterial` packer reads — this also
> fixed a silent coat drop on the UsdPreviewSurface path), and
> `map_material_mtlx` `coateddiffuse` reads the coat roughness from pbrt
> `roughness`. The two exports of `dragon_10` now render equivalently (relMSE
> 0.0001 / FLIP 0.0002 between them on Metal wavefront). **Both** export paths'
> `coated*` renders change as a result — the coat lobe now contributes.

> **Area-light emission gotcha.** `standard_surface` emission is
> `emission`(weight) × `emission_color`, and the round-trip recovers
> `emissiveColor` **only when `emission` > 0**. The exporter authors the unit
> weight, not `emission_color` alone — omitting it drops every area light and the
> scene renders black.

End-to-end GPU sampling is gated by the `texture_quad` corpus scene (a UV-mapped
quad with a gradient imagemap diffuse), which catches any UV flip/mirror/swap.

## pbrt metadata carry

The UsdPreviewSurface / UsdLux output is a *portable approximation*. Alongside it
the importer records the **exact pbrt source spec** as USD metadata, so skinny
(or a pbrt round-trip) can recover information the USD schema loses — spectra,
exact BxDF parameters, integrator, sensor — without affecting other USD
consumers (the `pbrt` namespace is ignored by tools that don't know it).

- **Stage-wide** `customLayerData["pbrt"]` — `integrator` (type + params, e.g.
  `sppm` / `maxdepth` / `radius`), `sampler`, `film` (`iso`, resolution,
  `maxcomponentvalue`), and `colorSpace`.
- **Per Material / Light / Camera** `customData["pbrt"]` —
  `{type, [name], params, paramTypes}`. `params` holds native USD-friendly
  values; `paramTypes` records each pbrt parameter type (`float` / `rgb` /
  `spectrum` / `blackbody` / `bool` / `string` / `texture` / …) so the value is
  self-describing and reconstructable.
- **Per emissive mesh** `customData["pbrtAreaLight"]` — the `AreaLightSource`
  emission spec (`L` / `scale` / `twosided`), preserving e.g. a `blackbody`
  temperature that the baked `emissiveColor` would otherwise discard.

Example (a conductor whose UsdPreviewSurface is an RGB approximation, but whose
exact named-spectrum IOR survives in metadata):

```
customData["pbrt"] = {
    "type": "conductor",
    "params":     { "eta": "metal-Ag-eta", "k": "metal-Ag-k", "roughness": 0.001 },
    "paramTypes": { "eta": "spectrum",     "k": "spectrum",   "roughness": "float" },
}
```

The encoder lives in `metadata.py`. This is **carriage only** — rendering the
carried data faithfully (spectral IOR, measured BRDFs) requires the
matching machinery in skinny's renderer; the loader can grow into the metadata
incrementally.

## Integrator mapping

The pbrt integrator is carried in `customLayerData["pbrt"]["integrator"]`
(type + params). Two integrators are *actively mapped* onto a skinny renderer
counterpart (the rest are carriage-only):

| pbrt integrator | skinny | notes |
|-----------------|--------|-------|
| `path` / `volpath` | path tracer | the default renderer |
| `sppm` / `photonmap` | **SPPM** (`INTEGRATOR_SPPM`) | wavefront-only, flat materials — see [PhotonMapping.md](PhotonMapping.md) |

For an `sppm` / `photonmap` scene the importer additionally records a normalized
skinny selection under `customLayerData["pbrt"]["skinny"]` and reports the
integrator as **mapped** (`exact`, no longer `skipped`):

```python
customLayerData["pbrt"]["skinny"] = {
    "integrator": "sppm",
    "radius":  0.05,    # optional — pbrt `radius`, seeds the SPPM initial search radius r₀
    "photons": 400000,  # optional — pbrt `photonsperiteration`, the photons-per-pass override
}
```

`api.sppm_selection(stage)` reads this dict back (or returns `None` for any other
integrator), so a loader or the parity harness can select skinny's SPPM
integrator and seed its initial radius / photons-per-pass directly from the pbrt
parameters — without re-parsing pbrt param names. Encoders live in
`metadata.scene_metadata`; the reader is `api.sppm_selection`.

## Parity validation

The corpus under `tests/pbrt/corpus/` holds small (≤128 px) scenes, each
isolating one parity axis, plus a `manifest.json` of per-scene tolerances. The
harness (`parity.py`) imports a scene, renders it in skinny via the headless
path, reads the **linear-HDR accumulation** (not the tonemapped sRGB display),
and compares it to a checked-in pbrt v4 reference EXR with two metrics:

- **relMSE** — `mean((a-b)² / (b² + ε))` on linear RGB (robust to a residual
  global scale and to Monte-Carlo noise; a least-squares exposure scalar is
  applied first).
- **FLIP** — a FLIP-style perceptual difference on identically-tonemapped copies.

The gate runs with **no pbrt binary** present — it relies solely on the committed
reference EXRs (`tests/pbrt/test_parity.py`, marked `gpu`). Until the references
are generated it skips, and only the import-smoke checks run.

> **Reference generation (offline).** Render each corpus `.pbrt` with a pbrt v4
> binary at the manifest resolution/spp into `tests/pbrt/corpus/refs/<name>.exr`,
> force-add them (`git add -f` — the repo `.gitignore` is `*`), then pin the pbrt
> version in the manifest and tighten the tolerances to the measured error.

## Measured parity

Skinny rendered against pbrt v4 (`5f7a606`) reference EXRs at 128×128, linear-HDR
accumulation, least-squares exposure aligned. The gate passes at these tolerances
(regression guard with headroom over measured):

| corpus scene | relMSE | FLIP | what it isolates |
|--------------|--------|------|------------------|
| `diffuse_arealight` | 0.087 | 0.041 | Lambert + area-light transport |
| `conductor_infinite` | 0.133 | 0.068 | gold Fresnel + GGX remap + constant env |
| `glass_arealight` | 0.128 | 0.071 | dielectric refraction |
| `texture_quad` | 0.0024 | 0.0119 | imagemap diffuse over explicit mesh UVs |
| `subsurface_infinite` | 0.079 | 0.089 | volumetric random-walk interior (dipole-vs-walk qualitative parity) |

FLIP ≤ 0.07 across the corpus — perceptually near-matching. The residual relMSE
is dominated by the inherent RGB-vs-spectral reduction, the conductor's
normal-incidence Fresnel approximation, area-light radiance scale, and pbrt's
brighter glass caustics. Two render-config notes that mattered for parity: the
harness zeroes skinny's *default ambient environment* for scenes with no pbrt
`infinite` light (so the background is black like pbrt), and a textureless
constant `infinite` light is baked to a uniform `.hdr` so skinny's dome path
reproduces it.

## Parity matrix

Support status per pbrt feature.

| pbrt feature | status | notes |
|--------------|--------|-------|
| `trianglemesh`, `sphere` shapes | matched | baked world-space meshes |
| `plymesh` (ascii / binary PLY, incl. `.ply.gz`) | matched | minimal in-repo PLY reader; gzip-compressed PLY is gunzipped transparently (magic-byte sniff) |
| `loopsubdiv` shape | matched | tessellated to the Loop limit surface at import (pbrt-exact: `levels` refinements + limit positions + limit normals), then baked as a triangle mesh |
| `ObjectInstance` instancing | matched | duplicated baked meshes |
| `diffuse` material | matched | Lambert reflectance |
| `conductor` material | approx | complex IOR → RGB normal-incidence Fresnel |
| `dielectric` material | matched | IOR + transmission/opacity gate |
| `coateddiffuse` / `coatedconductor` | approx | coat lobe + base |
| `thindielectric` | approx | thin-glass approximation |
| `diffusetransmission` | approx | diffuse + partial opacity |
| `subsurface` material | approx | smooth dielectric boundary (`eta`) + homogeneous interior medium (σ_a, σ_s, Henyey-Greenstein `g`) via `skinnyOverrides` `customData`; rendered as a delta-tracked volumetric random walk (`MATERIAL_TYPE_SUBSURFACE`), not clear glass. pbrt's tabulated dipole BSSRDF vs skinny's 3D random walk agree qualitatively (both milky), not bit-for-bit |
| `perspective` camera | matched | shorter-axis fov conversion |
| `realistic` camera | matched | lens file → `skinny:lens:*` (thick-lens `LensSystem`) |
| negative-scale camera (`Scale -1`) | matched | improper basis flagged `pbrt:mirrored`; renderer mirrors at ray-gen (`FrameConstants.cameraMirror`) → relMSE 0.009 / FLIP 0.021 vs pbrt |
| non-Y-up camera up/roll (pbrt Z-up) | matched | the authored up (camera local +Y → world) rides `CameraOverride.up`; the renderer builds the view basis from `(position, forward, up)` via `_look_at(..., world_up)` instead of assuming `(0,1,0)`, so Z-up scenes (e.g. `sssdragon`) orient correctly. Y-up cameras default to `(0,1,0)` ⇒ byte-identical. Composes with the mirror flag |
| `distant` / `point` lights | matched / approx | point emitted as a small sphere |
| `spot` light | unsupported | no skinny spotlight; flagged |
| area (`diffuse`) light | approx | emissive mesh; sidedness may differ |
| `infinite` light | matched | constant baked to `.hdr`; **square** `.exr`/`.pfm` image maps are pbrt's equal-area octahedral layout and are **reprojected** to skinny's equirectangular `.hdr` (`equiarea.py`, `Rx(+90)` axis map: pbrt Z-up sky → skinny +y-up); non-square maps assumed already lat-long and passed through |
| film `iso` / exposure | matched | `imagingRatio = exposureTime·ISO/100` baked into emitters |
| `imagemap` texture (reflectance/roughness) | matched | `UsdUVTexture` over `primvars:st`; explicit mesh UVs passed through, pbrt default UVs synthesized for UV-less textured shapes (see Texture UVs) |
| homogeneous medium | approx | coefficients via `customData` |
| heterogeneous (grid/VDB) medium | unsupported | flagged, not emitted |
| spectral rendering | unsupported | RGB reduction only (documented divergence) |
| `path` integrator | matched | renders with skinny's path tracer |
| `sppm` / `photonmap` integrator | matched | mapped to skinny's SPPM integrator (wavefront, flat materials); `radius` → initial search radius, `photonsperiteration` → photons/pass (see [Integrator mapping](#integrator-mapping)) |
