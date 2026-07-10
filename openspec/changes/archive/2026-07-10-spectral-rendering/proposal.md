# Spectral Rendering Support

## Why

skinny is an RGB renderer end-to-end: every spectrum the pbrt importer sees (blackbody
emitters, sampled SPDs, named conductor IORs) is collapsed to linear sRGB at import time
(`pbrt/spectra.py param_to_rgb`), and all GPU transport carries `float3` radiance. This is a
recorded, systematic divergence from pbrt v4's spectral ground truth in the parity matrix —
several corpus scenes gate on relaxed baselines purely because of RGB reduction — and it makes
wavelength-dependent effects (dispersion, accurate blackbody/illuminant color, conductor
Fresnel) impossible to render correctly. Adding hero-wavelength spectral transport closes the
largest remaining physical-fidelity gap against pbrt and unlocks dispersion.

## What Changes

- New opt-in **spectral render mode** (`--spectral` CLI flag, session-fixed like
  `--execution-mode`, never runtime-switchable): kernels are compiled **either spectral or RGB
  at startup**; in spectral mode paths carry radiance/throughput as 4 wavelength samples
  (hero wavelength + 3 stratified rotations over [360, 830] nm) instead of RGB.
- **Spectral film resolve**: per-sample spectral radiance is converted through tabulated CIE 1931
  CMFs to XYZ, then to linear sRGB, *before* the existing RGBA32F accumulation buffer — the
  progressive-accumulation, exposure, tonemap, and display pipeline is byte-unchanged downstream.
- **RGB→spectrum upsampling** on the GPU (Jakob–Hanika 2019 sigmoid-polynomial coefficient
  table, new sampled-texture binding) so existing RGB assets — material parameters, textures,
  HDR environments — render under spectral mode without re-authoring.
- **Spectral emitters**: blackbody SPDs evaluated analytically at sampled wavelengths;
  RGB-specified lights and env maps upsampled via the illuminant (D65-anchored) variant of the
  upsampling table.
- **Spectral conductor Fresnel**: named metal eta/k tabulated over wavelength instead of the
  current 3-primary RGB approximation.
- **Dispersion**: dielectrics with wavelength-dependent IOR (Cauchy/Sellmeier from pbrt's named
  glasses) collapse the path to the hero wavelength on refraction (pbrt's secondary-terminate
  strategy) and produce physically correct dispersion.
- **Importer preserves spectra**: pbrt import carries raw SPD payloads (blackbody temperature,
  sampled spectra, named spectra) through to the renderer alongside the existing RGB reduction,
  instead of discarding them.
- **Scope guards** (recorded, mirrored in the parity matrix): spectral mode covers the **path
  integrator in the megakernel execution mode** (Vulkan + Metal) over **flat materials**,
  analytic lights, emissive triangles, and the environment. The wavefront execution mode, BDPT,
  SPPM, ReSTIR-DI spatial/temporal reuse, heterogeneous volumes, and the skin BSSRDF remain
  RGB-only for now (wavefront spectral is the designated first follow-up) — spectral mode with
  an unsupported combo is refused at startup exactly as `combo_is_valid` records.

## Capabilities

### New Capabilities

- `spectral-rendering`: hero-wavelength sampling and spectral throughput in the path integrator;
  CIE film resolve to the existing accumulation buffer; RGB→spectrum upsampling for materials,
  textures, and environment; spectral blackbody/conductor evaluation; dispersion for
  wavelength-dependent dielectrics; mode selection and combo validity.

### Modified Capabilities

- `pbrt-spectrum-conversion`: importer requirement changes from "reduce every spectrum to RGB"
  to "reduce to RGB **and** preserve the raw spectral payload (blackbody temperature, SPD
  samples, named-spectrum identity) for renderer consumption in spectral mode".
- `render-parity-matrix`: matrix gains a spectral-mode axis with validity rules matching the
  scope guards above; coverage meta-tests enforce that every integrator × spectral combination
  has a recorded validity entry; spectral combos gate against pbrt spectral ground truth with
  the expectation that baselines **tighten** (pbrt refs are spectral-rendered).

## Impact

- **Shaders** (`src/skinny/shaders/`): `common.slang` (SampledSpectrum/SampledWavelengths
  types), `interfaces.slang` (`Spectrum` carrier in `BSDFSample`/`LightSample`),
  `integrators/path.slang`, `materials/flat/*`, `lights/*`, `environment.slang`, film resolve
  in `main_pass.slang`. New CMF + upsampling-table data. Spectral megakernel is a second
  compiled variant; the default `main_pass.spv` stays byte-identical.
- **Renderer** (`renderer.py`): upsampling-table upload (new descriptor binding → update the
  `docs/Architecture.md` binding map), `_pack_uniforms` / `_FRAME_FIELDS` tail append, spectral
  mode state in `_current_state_hash()` (accumulation reset).
- **CLI** (`cli_common.py`, front-ends): `--spectral` flag, env var, startup validation against
  integrator/execution-mode (mirrors `resolve_execution_mode` precedent).
- **Importer** (`pbrt/spectra.py`, `materials.py`): spectral payload preservation; vendored CIE
  CMF table and upsampling coefficient tables under `pbrt/data/` (generated by a checked-in
  script; no new runtime dependency).
- **Parity harness** (`pbrt/parity.py`, `tests/pbrt/`): spectral axis, validity rules, coverage
  meta-test, at least one dispersion/spectral discriminating scene in the confirming suite.
- **Docs**: `docs/Architecture.md` (binding map, module map), README (flag + compatibility
  matrix), CLAUDE.md compatibility matrix, CHANGELOG.
- **Metal budget note**: one new sampled texture must fit the 128-texture compute argument
  limit; the bindless pool trim precedent (`nanovdb`, 120→119) applies.
