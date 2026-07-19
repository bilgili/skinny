# spectral-rendering — delta (spectral-bdpt-megakernel)

## MODIFIED Requirements

### Requirement: Session-fixed spectral mode selection

The renderer SHALL provide an opt-in spectral render mode selected at startup via a
`--spectral` CLI flag on all front-ends (env var `SKINNY_SPECTRAL`), fixed for the session
like the execution mode, and not persisted. The spectral-vs-RGB decision SHALL be made once at
startup — kernels compile as the spectral variant or the RGB variant, never switchable
mid-session. Startup SHALL refuse the flag with a clear error when combined with an
unsupported configuration: an integrator other than `path` or `bdpt` (the SPPM refusal SHALL
name the wavefront follow-up), an explicit `wavefront` execution mode (flag or env; `auto`
with `path`/`bdpt` already resolves to `megakernel`), ReSTIR DI reuse, the neural directional
proposal, or a scene that requires subsurface/skin or heterogeneous-volume transport. When the
flag is absent, behavior SHALL be identical to today's RGB pipeline.

#### Scenario: spectral path session starts

- **WHEN** the app starts with `--spectral --integrator path` (either backend) on a
  flat-material scene
- **THEN** the session runs the megakernel with spectrally compiled kernels and reports the
  mode in startup logs

#### Scenario: spectral BDPT session starts

- **WHEN** the app starts with `--spectral --integrator bdpt` (either backend) on a
  flat-material scene
- **THEN** the session runs the megakernel with spectrally compiled kernels and the
  bidirectional integrator, and reports the mode in startup logs

#### Scenario: unsupported integrator refused

- **WHEN** the app starts with `--spectral --integrator sppm`
- **THEN** startup fails with an error naming the unsupported combination — including that
  SPPM spectral awaits the spectral wavefront follow-up — before any GPU work

#### Scenario: wavefront execution mode refused

- **WHEN** the app starts with `--spectral --execution-mode wavefront` (or the env equivalent)
- **THEN** startup fails with an error naming the unsupported combination

#### Scenario: unsupported transport refused

- **WHEN** `--spectral` is combined with ReSTIR DI reuse, the neural proposal, or a scene
  containing skin/subsurface or heterogeneous-volume materials
- **THEN** startup fails with an error naming the unsupported combination

#### Scenario: default sessions unaffected

- **WHEN** the app starts without `--spectral`
- **THEN** the RGB pipeline runs unchanged and no spectral resources are created

## ADDED Requirements

### Requirement: Hero-wavelength spectral transport in the bidirectional path integrator

In spectral mode with `--integrator bdpt` the megakernel SHALL run a bidirectional integrator
that carries radiance and throughput as 4 hero-rotated spectral samples per path, drawing the
wavelengths once per pixel path and sharing them across the eye subpath, the light subpath,
every connection strategy, and the light-tracer splat. All strategy families SHALL transport
spectrally: the eye random walk, the light random walk (whose origin throughput upsamples the
emitter's radiance as an illuminant, with the blackbody Planck override applied at
walk-capable origins — emissive triangles; directional lights spawn no light walk and the
environment has no light-subpath origin, so authored-SPD distant lights and environment
metamerism reach the image via t=1 NEE and eye-walk hits, which carry the existing v1
overrides), s≥2/t=0 emissive-vertex hits, t=1 NEE connections, t≥2 generic connections, and
the s=1 camera splat. The renderer SHALL dispatch the integrator selected at runtime within
the spectral envelope — `path` and `bdpt` switch live with the standard accumulation reset —
while any other integrator selection in a spectral session SHALL be pinned to `path` and the
pin SHALL be reported in the resolved-configuration output. Sampled directions, pdfs, and MIS weights
SHALL remain the scalar wavelength-independent quantities of the RGB machinery; per-wavelength
recoloring SHALL use the same upsampling, conductor-Fresnel, and exact-source facilities as
the spectral path integrator. Hero-wavelength glass dispersion SHALL apply on either subpath
with the same secondary-wavelength termination semantics as the spectral path integrator. The
RGB build SHALL remain byte-identical (all additions compiled only under the spectral
variant), and the spectral path integrator's output SHALL be unchanged by any code shared
with the bidirectional integrator.

#### Scenario: spectral BDPT matches the spectral path anchor

- **WHEN** a flat-material suite scene reachable by both integrators is rendered under
  `--spectral` with `bdpt` and with `path` at matched sampling effort
- **THEN** the two images agree within the recorded self-consistency tolerance
  (`compute_metrics`, no hand-rolled error formula)

#### Scenario: light-tracer splat is spectrally resolved

- **WHEN** a spectral BDPT sample produces an s=1 light-tracer contribution
- **THEN** the contribution is resolved through the CIE film resolve to linear sRGB before
  entering the existing splat buffer, and the splat compositing pipeline is unchanged

#### Scenario: exact spectral sources reach light subpaths

- **WHEN** a light subpath originates on a blackbody emissive triangle
- **THEN** its throughput carries the exact Planck SPD at the hero wavelengths (not an RGB
  upsample), observable through s=1 splats and t≥2 connections, consistent with the
  NEE-side override for the same emitter

#### Scenario: authored-SPD distant lights stay exact via NEE

- **WHEN** a spectral BDPT render includes an authored-SPD distant light
- **THEN** t=1 NEE connections carry the exact authored SPD at the hero wavelengths (the
  existing v1 override), and no light subpath is spawned from the directional light

#### Scenario: runtime integrator switching in a spectral session

- **WHEN** the user switches the integrator between `path` and `bdpt` at runtime in a
  spectral session on an interactive front-end
- **THEN** the dispatched integrator actually changes (accumulation and the splat buffer
  reset), and selecting any other integrator pins the dispatch to `path` with the pin
  reported in the resolved-configuration rows

#### Scenario: backends agree spectrally under BDPT

- **WHEN** the same flat-material suite scene is rendered under `--spectral --integrator
  bdpt` on the Vulkan and native Metal backends at matched sampling effort
- **THEN** the two images agree within the recorded backend-parity tolerance
  (`compute_metrics`)

#### Scenario: dispersion collapses the path on either subpath

- **WHEN** an eye or light subpath crosses a dispersive dielectric (Cauchy B > 0) via a delta
  refraction
- **THEN** transport continues at the hero wavelength with secondary wavelengths terminated,
  and subsequent connections and the film resolve honor the terminated wavelengths

#### Scenario: RGB build unaffected

- **WHEN** the megakernel is compiled without the spectral define after the change lands
- **THEN** the produced `main_pass.spv` is byte-identical to the pre-change binary

#### Scenario: spectral path output preserved across the helper hoist

- **WHEN** a spectral `path` render of a suite scene is compared before and after this change
  at a fixed seed
- **THEN** the images are identical (shared-helper extraction is pure code motion)
