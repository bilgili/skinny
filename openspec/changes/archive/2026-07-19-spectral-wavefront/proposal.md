# Spectral wavefront integrators

## Why

Spectral rendering is wired for the **megakernel** only: `--spectral` renders
`(path, megakernel, flat)` (change `spectral-rendering`) and `(bdpt, megakernel, flat)`
(change `spectral-bdpt-megakernel`). Every wavefront combo is refused at startup —
`--execution-mode wavefront` under `--spectral` raises, and SPPM (which has **no**
megakernel path — its photon pass is wavefront-only) is refused outright. So dispersive
caustics and photon-mapped spectral transport cannot render at all, and the wavefront's
short staged kernels — the natural home for heavy transport under the Metal watchdog —
never see a spectrum.

The two megakernel spectral integrators already prove the transport math (hero-wavelength
NEE, sigmoid/D65 upsampling, exact conductor Fresnel, blackbody/authored SPD sources,
hero-λ dispersion) over flat materials. Porting that to the wavefront widens the envelope
to **all three wavefront integrators** (path, BDPT, SPPM) and finally unblocks spectral
SPPM, which has been a recorded parity-matrix exclusion since v1.

## What Changes

- **Wavefront path/stream records carry a spectrum.** The per-path record structs
  (Slang + the Python-side stream buffers) that today carry `float3` throughput/radiance
  gain a hero-wavelength `Spectrum` (float4) payload and the sampled wavelengths, so
  transport between staged kernels (raygen → intersect → shade → connect → …) is
  per-wavelength. All widening is gated `#if defined(SKINNY_SPECTRAL)` so the RGB record
  layout, stream sizing, and SPIR-V stay byte-identical.
- **Wavefront path integrator renders spectrally.** The staged shade/connect kernels
  recolor per wavelength via the already-hoisted `spectral_flat_common.slang` helpers
  (`flatBsdfResponseSpectral`, upsample/NEE, exact sources) — directions, pdfs, and MIS
  weights stay scalar, exactly the v1 split.
- **Wavefront BDPT renders spectrally**, reusing the spectral strategy families proven in
  `bdpt_spectral.slang` across the staged eye/light subpath kernels and the camera-splat
  resolve (λ → linear sRGB before the atomic add, as the megakernel does).
- **SPPM renders spectrally** — the first spectral integrator with a photon pass: photons
  carry a `Spectrum` flux through the wavefront photon kernels; the eye pass gathers
  per-wavelength and resolves at the visible-point measurement. This is the combo that
  required a spectral wavefront foundation to exist.
- **Startup gate widens.** `reject_spectral_unsupported` accepts `--execution-mode
  wavefront` and `--integrator sppm` under `--spectral` (still flat materials, BSDF
  proposal, no ReSTIR reuse, no neural — those remain follow-ups). Refusal wording for the
  still-excluded combos (neural proposal, reuse, skin/volume) is updated.
- **Renderer spectral pin widens** to admit the wavefront execution mode and the SPPM
  integrator for a spectral session (`_active_integrator_index` / execution-mode
  resolution / `_collect_config_rows`), so the new kernels are reachable, not dead code.
- **Parity matrix** admits `(path|bdpt|sppm, wavefront, spectral)` into the rendered set
  under the standard dual gates (pbrt-truth + self-consistency vs the spectral megakernel
  path anchor); the remaining spectral rejections (neural, reuse, skin/subsurface/volume)
  stay recorded exclusions.
- **Capability flag / docs.** `spectral_capability` gains a wavefront-scope note; compat
  matrices in `CLAUDE.md` + `README.md`, `docs/Spectral.md`, `docs/Wavefront.md`, and
  `CHANGELOG.md` update.

## Capabilities

### New Capabilities

(none — this widens the envelope of existing capabilities)

### Modified Capabilities

- `spectral-rendering`: the session-fixed envelope requirement changes — the wavefront
  execution mode and the SPPM integrator are admitted (no longer refused), and new
  requirements cover hero-wavelength transport through the staged wavefront kernels
  (spectral path records, spectral BDPT subpaths + splat, spectral SPPM photon flux +
  gather).
- `render-parity-matrix`: the spectral envelope requirement changes — the three
  `(integrator, wavefront, spectral)` combos become valid rendered combos instead of
  recorded rejections; neural/reuse/skin spectral rejections remain.
- `wavefront-execution`: the wavefront record/stream requirement changes — the per-path
  transport payload carries a hero-wavelength spectrum under `SKINNY_SPECTRAL` (RGB layout
  unchanged).

## Impact

- **Shaders**: spectral branches (all `#if defined(SKINNY_SPECTRAL)`) in the wavefront
  record structs and the staged path/BDPT/SPPM kernels; reuse of
  `integrators/spectral_flat_common.slang` and `spectrum.slang`. RGB SPIR-V for every
  wavefront kernel must stay byte-identical (hostless guard).
- **Python**: `cli_common.reject_spectral_unsupported` (all four front-ends inherit);
  `wavefront_driver.py` + `metal_wavefront.py` stream-record sizing/upload for the widened
  spectral payload; `renderer.py` spectral pin + execution-mode resolution; `pbrt/parity.py`
  (`spectral_envelope`, combo enumeration + validity). Metal indirect-dispatch CPU-readback
  fallback and record-drain path must handle the wider records.
- **Tests**: hostless envelope/CLI-gate tests; parity-matrix validity + coverage meta-tests
  (the SPPM×spectral row); GPU self-consistency A/B (spectral wavefront path/bdpt/sppm vs
  the spectral megakernel path anchor); dispersion demo under wavefront BDPT; Metal
  dispatch-hygiene kill harness (new/longer kernels).
- **Docs**: `docs/Spectral.md`, `docs/Wavefront.md`, `CLAUDE.md` + `README.md` compatibility
  matrices, `CHANGELOG.md`.
