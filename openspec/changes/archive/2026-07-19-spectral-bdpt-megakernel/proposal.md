# Spectral BDPT (megakernel)

## Why

Spectral rendering v1 (change `spectral-rendering`) landed hero-wavelength transport for the
path integrator only; `--spectral --integrator bdpt` is refused at startup, so scenes that
need bidirectional transport (caustics through dispersive glass, strong indirect from small
emitters) cannot render spectrally. The BDPT megakernel already restricts itself to flat
materials — exactly the v1 spectral envelope — so widening the envelope to BDPT closes the
gap without touching the wavefront, skin, or volume follow-ups.

SPPM stays excluded: it has no megakernel path (photon pass is wavefront-only, and
`sppm` + explicit `megakernel` is refused at startup), and spectral has no wavefront
transport yet. SPPM spectral remains a recorded parity-matrix exclusion until a spectral
wavefront foundation exists.

## What Changes

- New spectral BDPT integrator module (`integrators/bdpt_spectral.slang`), compiled only
  under `-DSKINNY_SPECTRAL`, mirroring the `path_spectral.slang` pattern: a separate
  integrator carrying `Spectrum` (float4) radiance/throughput that reuses the RGB flat
  machinery for wavelength-independent geometry (directions, pdfs, MIS weights stay scalar)
  and recolors per wavelength via `flatBsdfResponseSpectral` + the existing upsample/SPD
  helpers (conductor Fresnel, blackbody Planck, authored illuminant SPDs, D65 upsampling).
- The spectral megakernel branch in `main_pass.slang` dispatches BDPT when
  `fc.integratorType == INTEGRATOR_BDPT` (today the spectral branch is path-only by
  construction); RGB SPIR-V stays byte-identical (all changes live behind `SKINNY_SPECTRAL`).
- The renderer's spectral integrator **pin** widens: `_active_integrator_index` returns 0
  (path) for every spectral session today — without widening it to `{path, bdpt}` the new
  shader branch is dead code. SPPM stays pinned → path (mirroring the RGB megakernel's
  silent path fallback), and the config-matrix row (`_collect_config_rows`' inlined copy of
  the pin) reports the pin the same way. Runtime path↔bdpt switching works in a spectral
  session (accumulation + splat buffer already reset via `_current_state_hash`).
- Light-tracer (s = 1) splats resolve λ → linear sRGB via the CIE film resolve **before**
  the atomic add into `lightSplatBuffer`, so the existing RGB splat compositing is untouched.
- Hero-λ glass dispersion (Cauchy) applies on both subpath walks with the same
  `terminateSecondary` collapse as the spectral path tracer.
- Startup gate widened: `reject_spectral_unsupported` accepts `--integrator bdpt`
  (megakernel, BSDF proposal, no reuse — same residual envelope); refusal wording for SPPM
  updated to name the wavefront follow-up.
- Parity matrix: `spectral_envelope` admits `(bdpt, megakernel, spectral)` into the rendered
  set; `(sppm, spectral)` stays a recorded exclusion. Spectral BDPT is gated by the same
  dual gates (pbrt-truth + self-consistency vs the spectral path anchor) as every combo.
- Docs: `docs/Spectral.md`, compatibility matrices in `CLAUDE.md` + `README.md`,
  `docs/Megakernel.md` scope wording.

## Capabilities

### New Capabilities

(none — this widens an existing capability's envelope)

### Modified Capabilities

- `spectral-rendering`: the session-fixed envelope requirement changes (an integrator other
  than `path` **or `bdpt`** is refused; BDPT is admitted), and a new requirement covers
  hero-wavelength spectral transport in the bidirectional path integrator (subpath walks,
  connections, light-tracer splat resolve, dispersion, exact spectral sources).
- `render-parity-matrix`: the spectral envelope requirement changes — `(BDPT, megakernel,
  spectral)` becomes a valid rendered combo instead of a recorded rejection; SPPM/wavefront
  spectral rejections remain.

## Impact

- **Shaders**: new `src/skinny/shaders/integrators/bdpt_spectral.slang`; spectral dispatch
  branch in `main_pass.slang`; possibly small shared helpers hoisted from
  `path_spectral.slang` (spectral NEE reuse). All behind `SKINNY_SPECTRAL` — the RGB
  `main_pass.spv` must remain byte-identical (existing hostless guard test).
- **Python**: `cli_common.reject_spectral_unsupported` (all four front-ends inherit),
  `pbrt/parity.py` (`spectral_envelope`, combo enumeration), `renderer.py` integrator-pin
  sites (`_active_integrator_index`, `_collect_config_rows`) and possibly the Metal
  megakernel band budget (`_METAL_MEGAKERNEL_BAND_PIXELS` is keyed by integrator only — the
  spectral BDPT kernel is strictly longer, re-measure). No renderer buffer changes
  (bindings 45–51 already uploaded for every spectral session).
- **Tests**: hostless envelope/CLI-gate tests, parity-matrix validity tests, GPU spectral
  BDPT vs spectral path A/B (self-consistency anchor), dispersion demo scene under BDPT,
  Metal watchdog: BDPT spectral runs under the existing megakernel row-band tiling
  (`metal-megakernel-watchdog-tiling`) — kill-harness pass required (kernel got longer).
- **Docs**: `docs/Spectral.md`, `docs/Megakernel.md`, `CLAUDE.md`/`README.md` compatibility
  matrices, `CHANGELOG.md`.
