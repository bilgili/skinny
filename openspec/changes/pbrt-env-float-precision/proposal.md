## Why

pbrt samples environment / infinite-light maps as floating-point radiance: `.exr`
is kept as half/float, `.hdr` (Radiance RGBE) is decoded to float32 at load, and
LDR formats are decoded to float on lookup. Filtering (the equal-area lookup) runs
on floats. skinny's GPU env texture is already `RGBA32F` and `.hdr` decodes to
float32, so the *sampling* side matches — but the **importer** does not:

- **8-bit requantization.** A pbrt `infinite` light pointing at a `.exr`/`.pfm`
  map is reprojected (equal-area → equirect) and then written back out as a
  Radiance **RGBE `.hdr`** (`pbrt/hdr.py::_floats_to_rgbe_array`, an 8-bit mantissa
  + shared exponent). The dome loader (`usd_loader.py`) only reads `.hdr`, so a
  `.exr` cannot be loaded directly and is forced through the RGBE hop. RGBE keeps
  full dynamic range (no clamp), but adds ≤~0.4% per-pixel mantissa quantization
  that pbrt never incurs.
- **Fixed low-res nearest resize.** Every env map is resampled to `1024×512` with
  nearest-neighbour (`environment.py::_resize_equirect`). pbrt samples the full-res
  image bilinearly, so a high-frequency env (small bright sun) aliases / loses
  energy in skinny — a larger absolute-radiance divergence than the RGBE step for
  env-lit scenes.

This was split out of `pbrt-radiometric-parity` (which handles the film imaging
ratio, the `.hdr`-direct `scale` drop, the light-type absolute offset, and the
absolute-radiance gate) because it is a distinct ingestion-fidelity axis with its
own GPU-verification cost. It is **not** a fix for the 1.6×/2× absolute offsets that
`pbrt-radiometric-parity` targets — it is float-precision/filtering hygiene.

## What Changes

- **Float-preserving env import.** The dome loader reads `.exr`/`.pfm` infinite-light
  maps directly as float32 (alongside `.hdr`), and the importer stops round-tripping
  env maps through RGBE — the reprojected float radiance reaches the GPU `RGBA32F`
  texture without an 8-bit hop. Constant infinite lights keep their (near-exact)
  RGBE/intensity path.
- **Higher-fidelity resample.** Replace the fixed `1024×512` nearest-neighbour env
  resize with a bilinear (area-weighted) resample and raise/remove the internal
  cap so a high-res env map keeps its energy, matching pbrt's float bilinear lookup
  within tolerance.
- **Parity guard.** Add an env-fidelity parity scene (a high-dynamic-range `.exr`
  infinite map with a small bright feature) to the matrix and gate its absolute /
  exposure-blind error against the pbrt v4 reference.

## Non-Goals

- The film imaging ratio, `.hdr`-direct `scale`, light-type absolute offset, and the
  absolute-radiance gate — those are `pbrt-radiometric-parity`.
- Equal-area ↔ equirect orientation (already `pbrt-env-equiarea`).
- Importance-sampling CDF construction precision (already float; unchanged).
- GPU texture format (already `RGBA32F`).
