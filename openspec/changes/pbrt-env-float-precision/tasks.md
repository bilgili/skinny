# Tasks — pbrt-env-float-precision (FOLLOW-UP, not started)

Split out of `pbrt-radiometric-parity` (user decision 2026-06-26). Do **after** that
change lands. Headless GPU rules from CLAUDE.md apply (Metal backend, SDK env vars).

## 1. Float-direct env loading

- [ ] 1.1 `usd_loader.py` dome loader (`_dome_to_env_hdr` near line 1124): read
  `.exr`/`.pfm` directly as float32 (reuse `pbrt/envmap.load_env_image` or an EXR
  reader; memory: imageio misreads pbrt `.exr` → use OpenEXR), reproject equal-area →
  equirect when square, upload to the `RGBA32F` env texture. Keep `.hdr` path.
- [ ] 1.2 `pbrt/lights.py` infinite branch: for `.exr`/`.pfm`, reference the float map
  (reprojected) without writing an RGBE `.hdr`; carry `scale` on DomeLight intensity
  (consistent with the `.hdr`-direct fix in pbrt-radiometric-parity). Constant infinite
  keeps RGBE/intensity.
- [ ] 1.3 Pure-python test: a `.exr` infinite map imports to a float dome with no
  RGBE intermediate written; HDR pixel > 1.0 survives to ≥ float precision.

## 2. Bilinear, adequate-resolution resample

- [ ] 2.1 `environment.py::_resize_equirect`: bilinear (area-weighted) resample;
  raise/remove the fixed `1024×512` cap (or make it adaptive to source res). Keep the
  importance-sampling CDF build consistent with the resampled map.
- [ ] 2.2 Verify env NEE / MIS CDFs still align with the resampled texture (no
  direction drift); unit test on a synthetic gradient.

## 3. Parity guard

- [ ] 3.1 Add an env-fidelity scene (HDR `.exr` infinite map with a small bright
  feature) to `tests/pbrt/corpus/manifest.json`; regen its reference with pbrt v4.
- [ ] 3.2 Gate absolute + exposure-blind error vs the reference; record `measured`.

## 4. Docs + OpenSpec

- [ ] 4.1 `docs/PbrtImport.md` §Lights: env maps imported/sampled as float, bilinear
  resample; remove the RGBE/nearest residual note.
- [ ] 4.2 `openspec validate pbrt-env-float-precision --strict`; ruff clean.
- [ ] 4.3 Commit → archive → merge → push.
