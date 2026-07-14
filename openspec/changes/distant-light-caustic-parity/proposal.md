## Why

On glass-caustic scenes SPPM visibly disagrees with `bdpt`/`path` (bright speckle
across the shadow/penumbra ground). Root cause (verified by phantom-light A/B,
pbrt cross-check, and shader bisection): the renderer **synthesizes a default
DistantLight** for any USD scene that authors no directional light — even when
the scene authors other lights (the caustic scene has a SphereLight). That
phantom sun casts a real caustic through the glass spheres which **only SPPM can
render**: a delta light through delta glass is an SDS path that the path tracer
fundamentally cannot sample, and BDPT structurally skips it — `sampleLightOrigin`
produces a `BDPT_VK_LIGHT_DIR` pseudo-vertex but the light-subpath random walk is
never launched for it (`bdpt.slang:1089`), so distant-light radiance can never
refract through a specular chain. SPPM's photon pass emits distant photons from a
scene-bounds disk and renders the caustic correctly — as concentrated filaments
that read as "fireflies". Suppressing the phantom light (authoring a
zero-intensity DistantLight) removes every firefly at the default SPPM radius and
brings `bdpt/path` to 1.002. pbrt has no such mismatch because it never
synthesizes lights the scene did not author.

Both halves are real defects: the phantom sun changes authored scenes' lighting
(and the parity harness already has to disable it specially), and BDPT silently
drops a whole transport family (distant-light indirect/caustics) that both the
path tracer's NEE (direct only) and SPPM (full) do carry.

## What Changes

- **Default-light synthesis policy**: the synthesized default DistantLight is
  injected only when the loaded USD scene authors **no light at all** (no
  DistantLight, no SphereLight, no emissive-material mesh, no DomeLight). A scene
  with any authored light renders with exactly its authored lights, on every
  front-end (interactive, headless CLI, web, Qt), matching the parity harness's
  existing intent. Sliders keep driving the default light for truly unlit scenes
  and the default head/no-USD session. **BREAKING** for workflows that relied on
  the phantom sun lighting a scene that authored only non-directional lights.
- **BDPT distant-light subpath walk**: `sampleLightOrigin`'s distant branch gains
  a real emission ray — origin sampled on a scene-bounds-covering disk, direction
  along the light (mirroring the SPPM photon emitter and pbrt's
  `DistantLight::SampleLe`), with the disk-area factor in `beta` and matching
  `pdfPos` — and the light-subpath `randomWalk` is launched for it. BDPT then
  renders distant-light indirect transport and specular caustics (s ≥ 2 and the
  s = 1 camera splat), with the MIS partition extended so no strategy
  double-counts against the existing distant-light NEE.
- Both fixes apply to RGB and `SKINNY_SPECTRAL` builds (spectral BDPT reuses the
  scalar MIS; the distant branch recolors per-λ via the authored SPD exactly like
  SPPM's photon emitter) and to both backends.
- Regression gates: (a) the caustic scene with only its authored SphereLight —
  `path ≡ bdpt ≡ sppm` with no speckle; (b) the same scene plus an authored
  DistantLight over the glass — `bdpt ≡ sppm` on the distant-light caustic that
  `path` cannot sample (path documented as the odd one out for that component).

## Capabilities

### New Capabilities
- `default-light-synthesis`: when the renderer may inject its built-in default
  DistantLight into a loaded scene, and the guarantee that authored-light scenes
  render with exactly their authored lights.

### Modified Capabilities
- `integrator-convergence`: BDPT gains the distant-light emission strategy — the
  light subpath SHALL walk from distant lights (disk emission), the MIS partition
  SHALL cover the new strategy without double-counting distant NEE, and BDPT
  SHALL agree with SPPM on distant-light specular caustics that unidirectional
  path tracing cannot sample.

## Impact

- `src/skinny/renderer.py` — the per-frame distant-light mirror fallback
  (`update`, ~line 9764: `lights_dir` empty → slider light) becomes
  any-authored-light aware; `_apply_usd_lights` / default-light scene-graph
  wiring keeps the default prim for unlit scenes.
- **Three** shader files (design D3b enumerates ~8 walk-skip sites + 3 origin
  seeds): `integrators/bdpt.slang` (seed, skip, `convertSAtoArea` DIR-as-source
  case, `connectT1` distant-NEE MIS partition), `integrators/bdpt_spectral.slang`
  (own seed + skip + per-λ recolor), `wavefront/wavefront_bdpt.slang` (own seed +
  six skips + lane budgeting).
- Recompiled `.spv` set checked in (megakernel + wavefront BDPT, RGB + spectral).
- `tests/` — default-light policy unit tests (hostless); parity additions for the
  two regression gates; existing parity manifests unaffected (the harness already
  disabled the phantom light explicitly).
- Docs: `docs/Architecture.md` (default-light behavior), `docs/PhotonMapping.md`
  (SPPM speckle post-mortem note), `CHANGELOG.md`.
- Follow-ups explicitly out of scope: SPPM env-indirect dimness (no env photon
  emission; fair null-sun scene reads sppm/path ≈ 0.78 in shadow), and the
  latent `vp.beta` omission at the SPPM photon deposit.
