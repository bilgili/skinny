## 1. Walk energy fixes (subsurface_walk.slang)

- [x] 1.1 Raise the Metal `SSS_MAX_BOUNCES` from 8 to 64 (Vulkan already 64); document the τ²-diffusion rationale and the watchdog tradeoff.
- [x] 1.2 Fresnel-split the boundary escape: with probability `Ft` transmit + carry env (no `Ft` weight); else reflect the direction across the face normal and `continue` the walk. Document why the old transmit-only `break` discarded ~half the interior light.

## 2. Verification (furnace energy — brightness-independent)

- [x] 2.1 Furnace τ-sweep (non-absorbing sphere, white env): τ≈1 0.98→0.997, τ≈20 0.476→0.802, τ≈200 0.476→0.799 — the high-τ plateau-collapse is eliminated.
- [x] 2.2 Furnace η-sweep at τ≈20: 1.0/1.2/1.5 → 0.801/0.724/0.593 (η=1.5 ≈ 2.8× the pre-fix 0.214).
- [x] 2.3 Watchdog: sssdragon (28.8M tris, wavefront, cap 64) renders without GPU-watchdog hang (~83 s).
- [x] 2.4 Subsurface source/unit tests stay green (routing, coeffs, unit-scale, forward-compat).

## 3. Render verification (manual A/B)

- [x] 3.1 sssdragon mean 0.081→0.097; capture pbrt | units+env | +walk-energy side-by-side. Note residual dimness/redness is the 1D-slab geometry (deferred).

## 4. Validate

- [x] 4.1 `openspec validate pbrt-subsurface-walk-energy --strict`; ruff clean.
