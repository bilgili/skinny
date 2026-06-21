## Why

The subsurface walk lost ~50% of its energy at the optical depths real scenes
hit, so the sssdragon (and any thick/high-η subsurface object) rendered far too
dark even after the unit-scale and env-orientation fixes. A furnace sweep (a
non-absorbing sphere in a white environment, which must return ~unity) exposed
two energy bugs:

1. **The Metal bounce cap was `8`.** A high-albedo random walk needs ~`τ²`
   scatter events to diffuse out of a medium of optical depth `τ`; at `τ ≈ 20`
   that is ~400 bounces. Capping at 8 truncated almost every path *without
   adding its radiance*, so energy just vanished — and because every `τ ≥ 20`
   slammed the same wall, the loss plateaued (`τ=20` and `τ=200` both ≈ 0.476).

2. **The boundary escape discarded internal reflection.** On reaching a face the
   walk added only the transmitted `Ft·env` and `break`-ed, dropping the
   `(1−Ft)` internally-reflected fraction. At a dielectric boundary (η = 1.5 for
   skin) the diffuse internal reflectance is large, so this discarded roughly
   half the interior multiple-scattering — worse at higher η.

Furnace evidence (cap 8 → 64, with the boundary fix):
`τ≈1` 0.98→0.997, `τ≈20` 0.476→0.802, `τ≈200` 0.476→0.799 (plateau eliminated),
and at `τ≈20` the η-sweep 1.0/1.2/1.5 goes 0.474/0.307/0.214 →
0.801/0.724/0.593 (η=1.5 ≈ 2.8×). The sssdragon brightens 0.081→0.097 with no
GPU-watchdog hang.

## What Changes

- **Raise the Metal subsurface bounce cap from 8 to 64.** Enough to conserve
  energy across the `τ` range skin/dragon scenes hit, while staying within the
  macOS GPU watchdog on the wavefront path (the production path for large scenes;
  large megakernel scenes OOM first). Vulkan was already 64.
- **Fresnel-split the boundary *escape*, not just the entry.** On reaching a
  face, transmit with probability `Ft` (carry the env out) else internally
  reflect — flip the boundary-normal component of the direction and keep walking.
  Importance-sampling the split keeps throughput unbiased (no `Ft`/`1−Ft`
  weights). This realizes the "Fresnel-split internal-reflection vs refraction on
  boundary hits" the spec already required but the escape path did not implement.
- **Strengthen the energy-conservation gate** to assert ~unity across optical
  depth (not just a thin slab), so a future cap/boundary regression is caught.

## Non-Goals (remaining follow-up)

- **Full pixel parity with pbrt's dipole BSSRDF.** Residual dragon dimness/
  redness is the **1D-slab geometry approximation**: the walk models the interior
  as a flat slab of perpendicular thickness, which mis-estimates path length on a
  curved/complex mesh (longer effective paths → more blue absorption → redder/
  darker than pbrt's milky look). A true 3D interior walk (or a diffusion/dipole
  tail for the deep regime) is a separate, larger change.
- **Higher caps / megakernel watchdog.** `64` is a watchdog-conservative balance;
  pushing toward unity at very high `τ` needs the 3D-walk work above, not just a
  larger cap.
