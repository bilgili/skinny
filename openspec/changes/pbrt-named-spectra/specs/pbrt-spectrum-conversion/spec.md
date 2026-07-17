## MODIFIED Requirements

### Requirement: Colored sampled spectra keep the CIE projection

A sampled spectrum whose authored values are not all exactly equal SHALL be reduced through
linear interpolation onto the 360–830 nm grid, integration against the CIE colour-matching
functions, division by the CMF integral, XYZ → linear sRGB, and a clamp to ≥ 0. The division
SHALL apply to **both** the reflectance and the illuminant branch, so that a colored spectrum
is continuous with the achromatic shortcut as its values approach equality, matching pbrt
(which divides by `CIE_Y_integral` for all spectra).

Previously the illuminant branch omitted the division, making a colored illuminant ~107× a
constant one of the same magnitude (`[400 10 700 10]` → `[10, 10, 10]` but
`[400 10 700 10.000001]` → `[1283, 1015, 971]`). Reflectance results are unchanged.

#### Scenario: illuminant projection no longer jumps two orders of magnitude

- **WHEN** an illuminant spectrum `[400 10 700 10.000001]` is imported
- **THEN** the result is the same order of magnitude as the constant
  `[400 10 700 10]` result `[10, 10, 10]` — differing only by the inherent
  equal-energy-whitepoint tint (~20%, the same tint the reflectance branch has
  always had), not by the previous ~107× factor

#### Scenario: the illuminant and reflectance branches are the same projection

- **WHEN** the same spectrum shape is imported as an illuminant and as a
  reflectance
- **THEN** the two results agree up to the authored magnitude — both divide by the
  CMF integral, so neither carries a scale convention the other lacks

#### Scenario: colored reflectance unchanged

- **WHEN** a colored reflectance spectrum is imported
- **THEN** the resulting RGB is bit-identical to the pre-change projection

#### Scenario: near-constant spectrum is still colored

- **WHEN** a spectrum's values differ by any nonzero amount
- **THEN** it takes the projection path, not the achromatic shortcut

### Requirement: Vendored spectral data tables

The importer package SHALL vendor the spectral data the renderer's spectral mode requires,
under `src/skinny/pbrt/data/`, copied verbatim from pbrt-v4 so spectral mode matches pbrt:
full spectral eta/k curves for **every** pbrt named metal (`Ag`, `Al`, `Au`, `Cu`, `CuZn`,
`MgO`, `TiO2`), the SPDs for **every** scene-addressable pbrt illuminant (`stdillum-A`,
`stdillum-D50`, `stdillum-D65`, `stdillum-F1` … `stdillum-F12`, and `illum-acesD60`;
the `canon_*` / `ilford_*` camera-response curves are film sensor responses, excluded with
the film-sensor work), wavelength-dependent IOR
fits for **every** pbrt named glass (`glass-BK7`, `glass-BAF10`, `glass-FK51A`,
`glass-LASF9`, `glass-F5`, `glass-F10`, `glass-F11`), and the RGB→spectrum
sigmoid-coefficient table with its checked-in generation script. Each named glass SHALL
carry its **own** Cauchy `(A, B)` coefficients, least-squares fit to pbrt's tabulated eta
over the 360–830 nm grid, and its own scalar IOR at the sodium d-line (589.3 nm). A
hostless test SHALL validate the tables against pbrt-published values, including a recorded
per-glass maximum fit residual.

#### Scenario: table validation is hostless

- **WHEN** the data-validation test runs on a host with no GPU
- **THEN** upsampled round-trip error and spot-checked coefficients match pbrt-published
  values within recorded tolerances

#### Scenario: every named glass has distinct dispersion

- **WHEN** the Cauchy coefficients for `glass-LASF9` and `glass-BK7` are looked up
- **THEN** they differ, and each reproduces its own pbrt tabulated eta curve across
  360–830 nm within the recorded per-glass residual tolerance (≤ 8e-3)

#### Scenario: named glass d-line IOR matches pbrt

- **WHEN** the scalar IOR for `glass-LASF9` is looked up
- **THEN** it is pbrt's eta at 589.3 nm (≈ 1.850), not the generic dielectric default

#### Scenario: extended metals are vendored

- **WHEN** the eta/k curves for `metal-CuZn-eta` / `metal-MgO-eta` / `metal-TiO2-eta` are
  looked up
- **THEN** each returns its own vendored curve pair on the 360–830 nm grid

### Requirement: Spectral payloads are preserved alongside the RGB reduction

For every spectrum-valued parameter it reduces to RGB, the importer SHALL additionally
preserve the raw spectral payload **that something consumes** for renderer consumption in
spectral mode, riding the established `skinnyOverrides` side-channel: the authored
temperature for `blackbody` parameters, the sample pairs (densely resampled onto the
360–830 nm / 5 nm grid) for `spectrum` parameters **on lights**, the named-spectrum identity
for named spectra (including conductor eta/k, glass IOR, and standard-illuminant names), and
wavelength-dependent IOR fits for named glasses.

A payload SHALL NOT be authored where no consumer exists: an inline `spectrum` on a
**material** is deliberately not preserved, because only light prims have an SPD path
(`_extract_light_spd`) — writing it would serialize a dead override that reads as a working
feature. Material reflectance spectrally upsamples from its RGB reduction. The payload SHALL
be written identically on
the plain-USD and the `-mtlx` authoring paths, so the authored identity survives
import → `.usda` → render on both. The existing RGB reduction SHALL remain unchanged and
remain the value consumed by the RGB pipeline. Scenes that author no spectra SHALL carry no
new payload.

#### Scenario: blackbody temperature preserved

- **WHEN** a light authors `"blackbody L" [3000]`
- **THEN** the imported USD carries both the existing RGB reduction and the temperature
  `3000` in the light's spectral payload

#### Scenario: authored illuminant SPD preserved

- **WHEN** a light authors a non-constant inline `spectrum L`
- **THEN** the imported USD carries the SPD resampled onto the 5 nm grid alongside the RGB
  reduction

#### Scenario: a spectrum file reference is reported, not mistaken for a name

- **WHEN** a parameter references a spectrum **file** (pbrt falls back to reading a file when
  a name does not match a built-in spectrum) and skinny has no spectrum-file reader
- **THEN** the import records an APPROX note identifying it as an unread spectrum file, rather
  than reporting it as an unknown glass/metal name substituted by a fallback material

#### Scenario: named conductor identity preserved

- **WHEN** a material references a named metal spectrum (e.g. `metal-Au-eta`)
- **THEN** the imported USD records the spectrum name so the renderer can bind the vendored
  spectral eta/k curve

#### Scenario: inline material spectrum authors no dead override

- **WHEN** a material authors an inline non-constant `"spectrum reflectance"
  [400 0.1 700 0.9]`
- **THEN** the material's RGB reduction is unchanged and **no** spectral payload is authored
  on it — nothing consumes a material SPD, so the override would be dead data that looks
  like a working feature

#### Scenario: identity survives the -mtlx path

- **WHEN** a scene with `"spectrum eta" "glass-BK7"` is imported with `-mtlx`
- **THEN** the material's `skinnyOverrides` records the same glass identity the plain-USD
  path records

#### Scenario: RGB-authored scenes unchanged

- **WHEN** a scene authors only `rgb`/`float` values
- **THEN** the imported USD is byte-identical to the pre-change import (no empty payloads
  authored)

## ADDED Requirements

### Requirement: Named glasses resolve to their own dispersion and IOR

A dielectric whose `eta` is a named glass spectrum SHALL resolve to that glass's own
identity: the `glass_dispersion` override SHALL carry a key distinct per recognised glass,
and the material's scalar IOR SHALL be that glass's d-line index rather than the generic
dielectric default. A named glass SHALL NOT silently adopt another glass's dispersion.

#### Scenario: LASF9 does not render as BK7

- **WHEN** a dielectric authors `"spectrum eta" "glass-LASF9"`
- **THEN** the imported material's scalar IOR is ≈ 1.850 and its `glass_dispersion` key
  resolves to LASF9's own Cauchy coefficients — not BK7's

#### Scenario: BK7 keeps its key

- **WHEN** a dielectric authors `"spectrum eta" "glass-BK7"`
- **THEN** the `glass_dispersion` override key is unchanged from before this change

#### Scenario: scalar-IOR dielectric authors nothing

- **WHEN** a dielectric authors a plain `"float eta" [1.5]`
- **THEN** no `glass_dispersion` override is written and the import is unchanged

### Requirement: Named illuminants resolve to chromaticity and SPD

A `spectrum`-valued light parameter naming a pbrt standard illuminant SHALL reduce to that
illuminant's linear-RGB chromaticity, normalised to unit luminance so that only chromaticity
derives from the name and magnitude continues to derive from the light's `L`/`scale`. This
SHALL apply to every light type that accepts a spectrum-valued parameter (distant, point,
spot, infinite, area). The importer SHALL additionally preserve the identity such that the
loader resolves it to a vendored 95-sample SPD wherever an SPD is consumed; a
named-illuminant payload SHALL NOT be discarded where an equivalent inline sampled payload
would be honoured.

Spectral SPD **consumption** is bounded by the existing envelope and this change does not
widen it: only distant lights bind an SPD (binding 50, per-distant-light, capacity 16), and
area lights carry only a blackbody `(temperature, scale)` pair (binding 49). For point,
spot, infinite, and area lights a named illuminant therefore yields correct RGB chromaticity
but no spectral SPD — identical to how an inline `spectrum L` behaves on those lights today.
Widening SPD consumption beyond distant lights is a separate change.

#### Scenario: stdillum-A is warm, not white

- **WHEN** a light of any type authors `"spectrum L" "stdillum-A"`
- **THEN** the imported RGB is the tungsten chromaticity of CIE illuminant A (red > blue),
  at unit luminance — not the neutral default

#### Scenario: named illuminant reaches the SPD binding on a distant light

- **WHEN** a scene with a `stdillum-A` **distant** light is loaded in spectral mode
- **THEN** the loader resolves the named payload to the vendored 95-sample SPD and the
  light carries it, exactly as an inline sampled SPD would

#### Scenario: non-distant lights get chromaticity without an SPD

- **WHEN** a `stdillum-A` **area** light is loaded in spectral mode
- **THEN** its emission carries the correct RGB chromaticity and is spectrally upsampled
  from that RGB — no SPD is bound, matching the pre-existing behaviour of an inline
  `spectrum L` on an area light, and no payload-shape error is raised

#### Scenario: light magnitude is unaffected by the name

- **WHEN** the same light authors `"spectrum L" "stdillum-A"` with a given `scale`
- **THEN** the imported radiance luminance matches what the pre-change import produced for
  that `scale`, differing only in chromaticity

### Requirement: Unrecognised named spectra are reported, not silently substituted

When a named glass, metal, or illuminant spectrum is not recognised, the importer SHALL
substitute a documented fallback (remaining a best-effort translator that does not hard-fail
the scene) and SHALL record an APPROX note in the import report that names both the
unrecognised spectrum and the substituted fallback.

#### Scenario: unknown glass reports its fallback

- **WHEN** a dielectric authors `"spectrum eta" "glass-NOSUCH"`
- **THEN** the import succeeds with a generic crown-glass fallback and the report carries an
  APPROX note naming `glass-NOSUCH` and the substitution

#### Scenario: unknown metal reports its fallback

- **WHEN** a conductor authors an unrecognised named metal eta
- **THEN** the import succeeds and the report carries an APPROX note naming the
  unrecognised spectrum and the substituted metal

#### Scenario: recognised names report nothing

- **WHEN** a scene authors only recognised named spectra
- **THEN** no APPROX note about named-spectrum substitution is recorded

### Requirement: Named-glass baselines track the refit in both render modes

The named-glass Cauchy coefficients SHALL be refit from pbrt's own tabulated eta for all
glasses including BK7. Named-glass scenes therefore shift in **both** modes: spectral by the
refit (|Δn| ≈ 3e-4), and RGB by the per-glass d-line IOR replacing the generic `eta` default
(BK7 `1.5 → 1.51673`; LASF9 `1.5 → 1.850`). The recorded pbrt-truth metrics for every
affected scene MUST be re-measured **in both RGB and spectral modes** with the standard
parity harness against the unchanged pbrt reference EXRs, and the re-measured values MUST
NOT be worse than the previously recorded ones. Megakernel ≡ wavefront self-consistency MUST
be re-verified.

The committed `.usda`/`.mtlx` scene assets are importer **output** and hard-code the old
generic IOR, so they SHALL be regenerated before any baseline is measured, and the
regenerated files SHALL be diffed to confirm the per-glass IOR actually appears. A baseline
measured over stale assets does not count as evidence.

#### Scenario: regenerated assets carry the per-glass IOR

- **WHEN** the named-glass scene assets are regenerated after the importer fix
- **THEN** their authored IOR changes from the generic `1.5` to the glass's d-line value
  (e.g. BK7 `1.51673`), and this diff is confirmed **before** any baseline is re-measured

#### Scenario: baselines improve, never loosen

- **WHEN** the affected scenes are re-rendered after the refit
- **THEN** the recorded relMSE/FLIP values are updated to the new (lower or equal)
  measurements, and no tolerance or baseline is raised

#### Scenario: RGB-mode named-glass scenes are re-measured too

- **WHEN** a named-glass scene is re-rendered in RGB mode after the importer supplies the
  d-line IOR
- **THEN** its RGB pbrt-truth baseline is re-measured and holds or improves — the change is
  not treated as spectral-only

#### Scenario: a worse metric blocks the change

- **WHEN** a re-measured pbrt-truth metric comes out worse than the recorded one
- **THEN** the fit is treated as incorrect and fixed, rather than the baseline being raised

### Requirement: Named-metal ids stay aligned with the spectral upload order

The named-conductor id packed for the GPU SHALL equal the metal's index in the
`spectralMetals` upload order plus one, because the shader resolves a metal's curves by
`(metalId - 1) * stride` into that buffer. Extending the metal set SHALL append to both the
id map and the upload order, leaving every pre-existing metal's id and buffer offset
unchanged. A hostless test SHALL assert the two agree.

#### Scenario: existing metals keep their ids

- **WHEN** the metal set is extended with CuZn/MgO/TiO2
- **THEN** Au/Ag/Al/Cu keep ids 1–4 and their existing `spectralMetals` byte offsets, so
  scenes using them render bit-identically

#### Scenario: id map and upload order cannot drift

- **WHEN** the id map and the upload order disagree on any metal's position
- **THEN** the hostless alignment test fails

### Requirement: The shader's named-conductor gate covers every uploaded metal

The shader's compile-time bound on named-conductor Fresnel ids SHALL equal the number of
metals the host uploads, so that no uploaded metal silently falls through to the RGB Schlick
approximation. A hostless test SHALL assert the shader constant matches the host's upload
length.

#### Scenario: a newly vendored metal actually uses its curves

- **WHEN** a conductor names `metal-CuZn-eta` and renders in spectral mode
- **THEN** it shades with the vendored CuZn eta/k, not the RGB Schlick fallback

#### Scenario: a lagging shader bound fails the build

- **WHEN** the shader's metal-count constant is lower than the host upload length
- **THEN** the hostless test fails rather than the extra metals silently mis-rendering
