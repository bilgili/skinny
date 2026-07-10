# Skinny — Spectral Rendering

This document is the implementation reference for skinny's **hero-wavelength
spectral** path (change `spectral-rendering`, activated with `--spectral`). It
covers the estimator, the RGB→spectrum upsampling model, the exact spectral
sources (blackbody, illuminant SPD, conductor Fresnel, glass dispersion), and the
film resolve back to linear sRGB, with the governing equations and the exact
shader/numpy symbols that realize them.

> **Work in progress.** The megakernel transport lives on the `spectral-megakernel`
> branch; on `main` the single capability flag
> `skinny.spectral_capability.SPECTRAL_IMPLEMENTED` is `False`, so `--spectral` is
> refused at startup ("not yet implemented") on every front-end and the parity
> matrix records the spectral axis as a "not yet wired" skip — it is **never**
> silently rendered as RGB. Flip that one flag with the transport to enable both
> the CLI gate and `parity.combo_is_valid` at once. See the
> [README compatibility matrix](../README.md) and
> [CLAUDE.md](../CLAUDE.md) for the current state and scope guards.

> Equations are shipped as **SVG images** (the repo's GitLab does not render
> KaTeX/`$$` math reliably). The LaTeX sources live in
> `docs/diagrams/spectral/equations.json`; regenerate the SVGs with
> `node docs/diagrams/spectral/render.cjs docs/diagrams/spectral/equations.json
> docs/diagrams/spectral` (MathJax 3, publication quality — needs Node +
> `mathjax-full`). Inline symbols (λ, p̂, ȳ, σ_t) are plain Unicode.

Everything here mirrors the checked-in **numpy reference** — the GPU kernel is
validated GPU≡numpy in `tests/kernels/test_spectrum_kernels.py`:

| Layer | File |
| --- | --- |
| GPU spectral core | `src/skinny/shaders/spectrum.slang` |
| GPU spectral integrator | `src/skinny/shaders/integrators/path_spectral.slang` |
| numpy estimator mirror | `src/skinny/pbrt/spectral.py` |
| vendored pbrt tables + upsampling | `src/skinny/pbrt/data/spectral_tables.py` |
| CMF / Planck / sRGB primitives | `src/skinny/pbrt/spectra.py` |

The descriptor bindings (45–47 `spectralScale`/`spectralData`/`spectralD65`, plus
48 `spectralMetals`), the spectral compile variant, and the shader module map are
documented in [Architecture.md § Descriptor Binding Map](Architecture.md#descriptor-binding-map)
and [Architecture.md § Modules](Architecture.md#modules) — this document does not
duplicate them.

## What hero-wavelength spectral is

A path tracer transports **radiance**. In an RGB renderer that radiance is a
3-vector and every product (BSDF × light, throughput × environment) is a
component-wise RGB multiply — which is only an approximation of the true spectral
product ∫ f(λ)·L(λ) dλ. It cannot express **metamerism** (two spectra that match
under one light and diverge under another), **dispersion** (a wavelength-dependent
index of refraction), or the exact **complex-index Fresnel** of a real metal.

skinny's spectral path replaces the RGB 3-vector with a small bundle of
**monochromatic samples**. At the start of each camera path it draws **N = 4**
wavelengths (a *hero* wavelength plus 3 rotated companions), transports each one
monochromatically through the whole path, and Monte-Carlo integrates the result
against the CIE colour-matching functions at the film. RGB inputs (albedos,
emitters, the environment map) are **upsampled** to plausible reflectance /
illuminant spectra on the fly, and named metals / glasses use their real spectral
data. The result is energy-consistent with RGB on neutral scenes but reproduces
metameric colour bleed, dispersion, and exact conductor tint.

This is the *hero-wavelength* method of Wilkie et al. (2014) over pbrt's
wavelength machinery (Pharr, Jakob & Humphreys, PBR 4th ed., ch. 4): one sample
`u` seeds all N wavelengths, so a single stratified draw covers the visible band
and the companions share the same path geometry.

### Scope and limits (v1)

| Property | Value |
| --- | --- |
| Integrator | **Path only.** BDPT / SPPM are excluded at compile time (the spectral branch never calls them). |
| Execution mode | **Megakernel only.** Wavefront is a designated follow-up; `--spectral` refuses an explicit `--execution-mode wavefront`. |
| Materials | **Flat only** (`UsdPreviewSurface` / `standard_surface` / `OpenPBR` / Python-material flats). Skin / subsurface / heterogeneous-volume scenes are refused at startup; a non-flat hit inside the loop terminates the path rather than mis-shade. |
| Reuse / guiding | **None.** ReSTIR reuse and the neural proposal are refused. |
| Wavelengths | **4** hero-rotated, drawn from pbrt's visible-λ importance pdf. |
| Film | CIE resolve (Wyman CMF) into the **existing RGBA32F accumulation** — exposure / tonemap / readback are untouched. |
| Backends | Vulkan and native Metal, at parity (compile-time `-DSKINNY_SPECTRAL` on both). |

The spectral transport is a **deliberately separate integrator**
(`SpectralPathTracer`), not a widening of the shared RGB carriers: turning
`BSDFSample` / `LightSample` / `BounceResult` into `float4` would break every
`float3` assignment across the flat / skin / python / BDPT / record tree. Instead
`SpectralPathTracer` carries a `float4` `Spectrum` throughput/radiance itself and
**reuses the RGB flat `sample()`** for the wavelength-independent geometry (the
sampled direction `wi`, its solid-angle pdf, delta-ness), recolouring per
wavelength. The RGB build never imports `spectrum.slang` or
`path_spectral.slang`, so its SPIR-V is **byte-unchanged**.

![Hero-wavelength megakernel: draw 4 λ, per-λ transport, CIE film resolve, shared accumulation](diagrams/spectral/pipeline.svg)

## The hero-wavelength estimator

### 1. Sampling the wavelengths

A single uniform sample `u ∈ [0,1)` is warped to a wavelength by pbrt's
`SampleVisibleWavelengths`, whose target density concentrates samples around the
luminance peak (~538 nm) where the eye is most sensitive:

![lambda(u) = 538 − 138.888889·artanh(0.85691062 − 1.82750197·u)](diagrams/spectral/sample-lambda.svg)

with the matching importance pdf (zero outside the visible band):

![p(lambda) = 0.0039398042 / cosh^2(0.0072(lambda − 538)), lambda in [360,830] nm](diagrams/spectral/lambda-pdf.svg)

The **hero rotation** draws all N wavelengths from the *same* `u` by offsetting it
by `i/N` and wrapping into `[0,1)` — one stratified draw covers the whole band and
every companion shares the path's geometry:

![lambda_i = lambda({u + i/N}), i = 0..N−1, N = 4](diagrams/spectral/hero-rotate.svg)

> **Implements:** `sampleVisibleWavelength` / `visibleWavelengthPdf` /
> `sampleWavelengths` in `spectrum.slang`; `sample_visible_wavelength` /
> `visible_wavelength_pdf` / `sample_wavelengths` in `spectral.py`. The hero draw is
> the **first** consumption of the pixel RNG in `SpectralPathTracer.estimateRadiance`
> (`sw = sampleWavelengths(rng.next())`) — the RGB build has no such draw, so its
> RNG stream and SPIR-V are unchanged.

| symbol | code | meaning |
| --- | --- | --- |
| u | `rng.next()` | one uniform sample per path |
| λᵢ | `sw.lambda[i]` | the i-th hero-rotated wavelength (nm) |
| p(λᵢ) | `sw.pdf[i]` | its sampling pdf (0 ⇒ terminated) |
| N | `4` (`SPECTRUM_N`) | wavelengths per path |

### 2. Secondary termination (for dispersion)

A **dispersive** interaction (a refraction whose direction depends on λ) is
incoherent with a shared bundle: the companions would each bend a different way.
pbrt's `TerminateSecondary` collapses the bundle to the hero wavelength — it zeros
the companion pdfs and divides the hero pdf by N so the estimator stays unbiased
(the hero now stands in for the whole bundle):

![p_0 <- p_0/N, p_1 = p_2 = p_3 = 0](diagrams/spectral/terminate.svg)

Because a companion contributes `value·(1/pdf)` and its `pdf` is now 0, the film
resolve (§6) simply drops it via its `pdf > 0 ? 1/pdf : 0` guard. A
**non-dispersive** path (constant-IOR glass, no glass at all) never terminates and
keeps all 4 wavelengths — this is the unbiasedness guarantee tested in the mirror.

> **Implements:** `terminateSecondary` / `secondaryTerminated` in `spectrum.slang`;
> `terminate_secondary` / `SampledWavelengths.secondary_terminated` in `spectral.py`.

## The upsampling model (RGB → spectrum)

An RGB scene input has no spectrum; skinny reconstructs a plausible one on the fly
with the **Jakob & Hanika (2019)** sigmoid model, using **pbrt's exact
`sRGBToSpectrumTable`** (vendored verbatim as `rgb2spec_srgb.npz`, RES = 64). A
reflectance is a smooth bounded spectrum

![S(lambda) = s(c0·lambda^2 + c1·lambda + c2), s(x) = 1/2 + x/(2 sqrt(1+x^2))](diagrams/spectral/sigmoid.svg)

whose 3 coefficients `(c₀,c₁,c₂)` come from a **trilinear lookup** into the table,
keyed on the RGB triple (pbrt's `RGBToSpectrumTable::operator()`). A uniform RGB
takes pbrt's closed-form branch, and the r ∈ {0,1} endpoints ride the IEEE ±∞
limit into the sigmoid's {0,1} saturation (a finite ±1e9 reproduces it) — this is
what lets a white furnace close (a naïve `c₂ = 0` would reflect 50 %).

### Reflectance vs. illuminant

A **surface reflectance** (albedo, specular / coat colour, glass tint) is the raw
sigmoid, evaluated at each hero wavelength:

![R(lambda) = S_c(rgb)(lambda)](diagrams/spectral/reflectance.svg)

An **emitter** (light radiance, environment, flat emission) is pbrt's
`RGBIlluminantSpectrum` — the sigmoid **shape** multiplied by the CIE **D65**
reference illuminant and a scale that keeps HDR emitters inside the sigmoid gamut:

![L(lambda) = scale·S_c(rgb/scale)(lambda)·D65(lambda), scale = 2·max(r,g,b)](diagrams/spectral/illuminant.svg)

The `scale = 2·max(rgb)` normalisation matters: a plain clamp of a bright light
into `[0,1]` would collapse it to white and lose its intensity. D65 is uploaded
**pre-normalized to unit luminance** (`spectral.d65_normalized()`), so a unit RGB
illuminant resolves to unit radiance and the shader matches the mirror bit-for-bit.
(An earlier un-normalized D65 made a white light resolve ≈ 49× too bright.)

> **Implements:** `sigmoidPoly` / `rgbSigmoidCoeffs` / `upsampleReflectance` /
> `upsampleIlluminant` in `spectrum.slang`; `sigmoid_poly` /
> `rgb_to_sigmoid_coeffs` in `spectral_tables.py` and `upsample_reflectance` /
> `upsample_illuminant` / `d65_normalized` in `spectral.py`. On the GPU the coefficient
> table (`spectralData`), its RES node array (`spectralScale`), and D65 (`spectralD65`)
> are storage buffers at bindings 46 / 45 / 47 (spectral-build-only). Inside the
> integrator the fetches are shared per hit via `SpectralFlatColors` /
> `upsampleFlatColors` so the sigmoid lookup runs once per hit, not once per light.

| symbol | code | meaning |
| --- | --- | --- |
| S(λ) | `sigmoidPoly(co, lam)` | sigmoid-polynomial reflectance at λ |
| c | `rgbSigmoidCoeffs(rgb, …)` | the 3 sigmoid coefficients from the table |
| σ (scale) | `sc = 2·max(rgb)` | HDR-preserving illuminant scale |
| D65(λ) | `sampleCurve5nm(lam, d65, …)` | unit-luminance D65 SPD, 5 nm grid |

## Per-wavelength transport

The bounce loop is a standard unidirectional path tracer, but throughput and
radiance are `Spectrum` (a `float4` bundle) and every material/light product is
formed **per wavelength**. The BSDF *geometry* (`wi`, its pdf, delta-ness) comes
from the RGB `mat.sample()` — it is wavelength-independent — and only the
**response** is recoloured, so the non-delta throughput weight per wavelength
reproduces the RGB sample weight exactly:

![beta ∗= f(lambda, wo→wi)/p(wi); L += beta ⊙ L_NEE(lambda)](diagrams/spectral/bounce-weight.svg)

- **NEE** (`spectralAllLightsNEE`) mirrors the RGB `allLightsNEE`, but forms the
  `response(λ)·L(λ)` product per wavelength (each RGB light / env radiance
  upsampled as an illuminant). Every **scalar** factor — the area→solid-angle
  pdf, the MIS power heuristic, the mixture proposal pdf — is the **same** value
  the RGB NEE uses (`mixtureProposalPdf` on the RGB material), so direct and
  indirect stay MIS-consistent.
- **Russian roulette** and **MIS** stay scalar: they weight the whole bundle at
  once (RR keys off `max(β)` across the 4 lanes).
- **Emission** at a flat hit is MIS-gated exactly as the RGB path
  (`emissive-triangle-bsdf-hit-mis`), using the RGB emission's luminance for the
  light-sampling pdf.

`materials/flat/flat_lobes.slang::flatBsdfResponseSpectral` is the per-λ mirror of
`flatBsdfResponse`: colored reflectances are passed pre-upsampled, the Schlick
term evaluates per-λ on `F0`, and every scalar lobe weight is identical to the RGB
sample, so `response/pdf` reproduces the RGB weight.

## Exact spectral sources

RGB upsampling is a fallback; where the scene carries a *real* spectral identity,
the importer preserves it on `skinnyOverrides` and the GPU consumes it directly.

### Blackbody emitters (Planck)

A pbrt `blackbody` emitter preserves its temperature `T`. The emission at each
hero wavelength is the (relative) Planck spectral radiance:

![L_e(lambda, T) = (2hc^2 / lambda^5) · 1/(exp(hc/lambda·kB·T) − 1)](diagrams/spectral/planck.svg)

The Planck SPD carries the blackbody **chromaticity** on its own; a separate
scalar fixes only the **luminance** so the hero-λ resolve reproduces the material's
authored linear-sRGB emission `c_RGB`:

![scale = Y_target·∫ȳ dλ / Y_Planck, Y_target = c_RGB·(0.2126,0.7152,0.0722)](diagrams/spectral/planck-scale.svg)

`Y_target` is the Rec.709 luminance of the emission, which for a linear-sRGB colour
equals its CIE Y under D65 (the Y row of the sRGB→XYZ matrix); the `∫ȳ dλ` factor
puts the raw Planck luminance into the film resolve's normalization (§6). *(Numpy
prep landed as `spectral.blackbody_emission` / `blackbody_scale`; the GPU consumer
re-hooks to the emissive-triangle path — pbrt blackbody area lights import as
emissive triangles, task 6.1.)*

### Conductor Fresnel (complex index)

A named metal (`au`/`ag`/`al`/`cu`) preserves its identity on
`skinnyOverrides["conductor_metal"]`; its complex index η̃ = η + iκ is looked up per
hero wavelength from the vendored eta/k curves (`spectralMetals`, binding 48, 5 nm
grid) and the specular lobe uses pbrt's exact unpolarized `FrComplex` instead of a
scalar Schlick approximation:

![Complex-index Fresnel: eta-tilde = eta + i·kappa; r_parallel, r_perp; R = 1/2(|r_par|^2 + |r_perp|^2)](diagrams/spectral/fresnel.svg)

This gives a real metal its correct wavelength-dependent tint (validated: gold
renders warmer / more saturated than its RGB Schlick approximation).

> **Implements:** `namedMetalEtaK` + `fresnelConductor` in `flat_shading.slang`
> (used by `flatBsdfResponseSpectral` when `FlatMaterialParams.conductorMetalId ≠ 0`);
> `fresnel_conductor` / `named_metal_eta_k` in `spectral.py`.

### Glass dispersion (Cauchy)

A named glass preserves a Cauchy fit; the index falls with wavelength (normal
dispersion, blue index > red):

![n(lambda) = A + B / lambda_um^2](diagrams/spectral/cauchy.svg)

The Cauchy `A` rides the material's `ior` lane and `B` the spare
`FlatMaterialParams._normalBiasPad.w` (`glassCauchyB`) — **no new buffer**. After
the RGB `sample()` produces a **delta refraction**, the integrator re-refracts the
hero wavelength at `n(λ₀)` and terminates the secondaries. The gate that decides
whether to disperse is:

![disperse iff (p_bsdf = 0) and (B > 0) and (wo_z·wi_z < 0)](diagrams/spectral/dispersion-gate.svg)

The `ω_o^z·ω_i^z < 0` test detects a refraction geometrically (wi and wo on
opposite sides of the interface) — this catches **both** the entering and the
exiting refraction (the exit has `wi.z > 0`, so a `transmitted` flag alone would
miss it). Constant-IOR glass (`B = 0`) fails the gate, keeps all 4 wavelengths, and
stays unbiased; a hero-λ total-internal-reflection falls back to an (achromatic)
reflection.

> **Implements:** the dispersion block in `path_spectral.slang`
> (`SpectralPathTracer.estimateRadiance`, gated on `glassCauchyB`); `cauchy_ior` /
> `should_terminate_secondary` in `spectral.py`; `named_glass_cauchy` /
> `named_glass_ior` in `spectral_tables.py`.

## The film resolve (spectrum → linear sRGB)

At path termination the N monochromatic radiances are Monte-Carlo integrated
against the CIE 1931 colour-matching functions — skinny uses the **Wyman-Sloan-
Shirley (2013)** analytic multi-Gaussian fit `cie_xyz_bar`, shared by the importer,
this mirror, and the shader so all three agree on one CMF:

![XYZ = 1/(N∫ȳ dλ) · sum c(lambda_i)/p(lambda_i) · (xbar,ybar,zbar)(lambda_i)](diagrams/spectral/resolve-xyz.svg)

The `∫ȳ dλ` divisor (`CIE_Y_INTEGRAL = 106.9229674725`, baked to match the numpy
mirror) is the CMF normalization; the `1/N Σ (·)/p(λᵢ)` is the wavelength
Monte-Carlo estimator, with a terminated companion (`p = 0`) contributing nothing.
The XYZ triple is then the standard linear sRGB (D65) matrix, clamped ≥ 0:

![sRGB = M_XYZ->sRGB · XYZ](diagrams/spectral/xyz-srgb.svg)

`SpectralPathTracer.estimateRadiance` resolves to a `float3` **internally** at
return (`spectrumResolveToLinearSRGB(sw, radiance)`), so the `IIntegrator`
interface and the `main_pass.slang` tail (NaN sanitise / `clampSampleRadiance` /
accumulation) stay `float3` and byte-identical to the RGB path — the spectral
bundle never escapes the integrator.

> **Implements:** `cieXYZBar` / `xyzToLinearSRGB` / `sampledSpectrumToXYZ` /
> `spectrumResolveToLinearSRGB` in `spectrum.slang`; `cie_xyz_bar` /
> `xyz_to_linear_srgb` in `spectra.py` and `spectrum_to_xyz` /
> `resolve_to_linear_srgb` in `spectral.py`.

| symbol | code | meaning |
| --- | --- | --- |
| c(λᵢ) | `values[i]` | radiance at hero wavelength λᵢ |
| (x̄,ȳ,z̄) | `cieXYZBar(lam)` | Wyman CMF fit |
| ∫ȳ dλ | `CIE_Y_INTEGRAL` | CMF normalization (106.9229674725) |
| M | `xyzToLinearSRGB` | XYZ → linear-sRGB (D65) matrix |

## Equation → implementation map

| Equation | Symbol | File |
| --- | --- | --- |
| Visible-λ warp `λ(u)` | `sampleVisibleWavelength` | `spectrum.slang` |
| Visible-λ pdf `p(λ)` | `visibleWavelengthPdf` | `spectrum.slang` |
| Hero rotation `λᵢ = λ({u+i/N})` | `sampleWavelengths` | `spectrum.slang` |
| Secondary termination | `terminateSecondary` | `spectrum.slang` |
| Sigmoid reflectance `S(λ)` | `sigmoidPoly` / `rgbSigmoidCoeffs` | `spectrum.slang` |
| Reflectance upsample | `upsampleReflectance` | `spectrum.slang` |
| Illuminant upsample (× D65) | `upsampleIlluminant` | `spectrum.slang` |
| Per-λ flat response | `flatBsdfResponseSpectral` | `materials/flat/flat_lobes.slang` |
| Spectral NEE | `spectralAllLightsNEE` | `integrators/path_spectral.slang` |
| Bounce loop / throughput | `SpectralPathTracer.estimateRadiance` | `integrators/path_spectral.slang` |
| Blackbody Planck + scale | `blackbody_emission` / `blackbody_scale` | `pbrt/spectral.py` |
| Conductor Fresnel `R(λ)` | `fresnelConductor` / `namedMetalEtaK` | `materials/flat/flat_shading.slang` |
| Cauchy IOR + dispersion gate | dispersion block | `integrators/path_spectral.slang` |
| CIE resolve `XYZ` | `sampledSpectrumToXYZ` | `spectrum.slang` |
| XYZ → sRGB | `xyzToLinearSRGB` | `spectrum.slang` |
| Host buffers + `--spectral` wiring | `renderer._spectral` block | `renderer.py` |
| Capability gate | `SPECTRAL_IMPLEMENTED` | `spectral_capability.py` |
| Parity axis | `spectral_envelope` / `combo_is_valid` | `pbrt/parity.py` |

## Verification

- **GPU ≡ numpy** — `tests/kernels/test_spectrum_kernels.py` (gpu-marked) compares
  the `spectrum.slang` harness wrappers to `spectral.py` for wavelength sampling,
  secondary termination, reflectance/illuminant upsampling (incl. HDR + chromatic),
  and the film resolve; max rel err ≈ 1e-6.
- **Compile gate** — `tests/test_spectrum_compile.py` proves both the RGB and
  `-DSKINNY_SPECTRAL` variants typecheck to SPIR-V.
- **Data round-trips** — `tests/pbrt/test_spectral_data.py` (upsample→CIE
  round-trip, D65 chromaticity, metal/glass sanity) and
  `tests/pbrt/test_spectral_payload.py` (import payload preservation; RGB-only
  scenes author no new payload).
- **Byte-identity** — the default (no-define) build's RGB render of `int_bleed` is
  unchanged by the spectral `#if` blocks (identical mean 0.31306 across A/B runs).
- **Energy / furnace** — `int_bleed` renders energy-consistent under `--spectral`
  (spectral mean 0.31343 vs RGB 0.31306) with plausible metameric colour bleed;
  `furnace_lambert` closes under the uniformity gate.

## References

1. **A. Wilkie, S. Nawaz, M. Droske, A. Weidlich, J. Hanika.** *Hero Wavelength
   Spectral Sampling.* Computer Graphics Forum (EGSR) 33(4), 2014. — the hero /
   rotated-companion wavelength method (§1–§2).
2. **W. Jakob, J. Hanika.** *A Low-Dimensional Function Space for Efficient
   Spectral Upsampling.* Computer Graphics Forum (Eurographics) 38(2), 2019. — the
   sigmoid RGB→spectrum reflectance model and its coefficient table (the upsampling
   model).
3. **M. Pharr, W. Jakob, G. Humphreys.** *Physically Based Rendering: From Theory
   to Implementation*, 4th ed., 2023 (ch. 4 *Radiometry, Spectra, and Color*). —
   `SampleVisibleWavelengths` / `VisibleWavelengthsPDF`, `RGBAlbedoSpectrum` /
   `RGBIlluminantSpectrum`, `RGBToSpectrumTable`, `FrComplex`, and the film XYZ
   resolve; skinny's tables are vendored verbatim from pbrt-v4.
4. **C. Wyman, P.-P. Sloan, P. Shirley.** *Simple Analytic Approximations to the
   CIE XYZ Color Matching Functions.* Journal of Computer Graphics Techniques
   2(2), 2013. — the multi-Gaussian CMF fit used in the film resolve (§6).
5. **CIE.** *Colorimetry*, 15:2004; **IEC 61966-2-1** (sRGB). — the D65 whitepoint,
   the CIE 1931 observer, and the XYZ→sRGB primaries.
