# Spectral Dispersion Showcase Asset

## Why

Spectral rendering (`--spectral`) ships hero-wavelength glass dispersion, but no
first-class demo scene exists that makes the effect unmistakable. The suite scene
`tests/assets/suite/spec_prism/` is a 128² gate scene tuned for metrics, and
`assets/glass_caustics_spectral.usda` shows only faint BK7 fringing on sphere
caustics. The only Cauchy fit in `_GLASS_CAUCHY` is BK7 (B = 0.0042 µm²), whose
angular spread is too small to read as a rainbow at demo scale. Users need a
`skinny-render --spectral assets/<scene>.usda` one-liner that produces the classic
prism rainbow.

## What Changes

- New demo asset `assets/dispersion_prism.usda`: dark-room prism scene — a
  triangular high-dispersion flint-glass prism, a narrow white slit light aimed
  through it, a matte white screen/floor catching the refracted fan, and a camera
  framed so both the prism and the spectral fan are visible. Authored directly as
  USD (same `skinnyOverrides.glass_dispersion` seam the importer emits), no
  `.pbrt` source required.
- Extend the named-glass Cauchy table (`_GLASS_CAUCHY` in
  `src/skinny/pbrt/data/spectral_tables.py`) with at least one real
  high-dispersion flint glass (SF11-family, B ≈ 3× BK7) so the asset's rainbow is
  clearly separated; numpy mirror (`pbrt/spectral.py` consumers) and the GPU
  binding path (`renderer.py` → `glassCauchyB`) pick it up through the existing
  name lookup — no shader change.
- Hostless test: asset loads, the prism material carries the flint
  `glass_dispersion` override, and `named_glass_cauchy` resolves it to a fit with
  B strictly greater than BK7's.
- README/docs note: the demo asset and how to render it (`--spectral`, path and
  bdpt; SPPM accepts `--spectral` but shows no rainbow — v1 has no SPPM
  dispersion).

## Capabilities

### New Capabilities

(none — this rides the existing spectral-rendering capability)

### Modified Capabilities

- `spectral-rendering`: the named-glass dispersion requirement gains a
  high-dispersion flint entry in the recognized Cauchy set, and a new requirement
  that the repo ships a demo asset whose spectral render visibly separates the
  spectrum (rainbow) versus a monochrome RGB render of the same scene.

## Impact

- `assets/dispersion_prism.usda` (new)
- `src/skinny/pbrt/data/spectral_tables.py` (`_GLASS_CAUCHY` + docstrings)
- `tests/` — one hostless test module (asset integrity + table entry)
- `README.md` / `docs/Spectral.md` — demo asset mention
- No shader, binding, or pipeline changes; `glassCauchyB` plumbing already
  generic over the table.
