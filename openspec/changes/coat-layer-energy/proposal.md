# Change: coat-layer-energy

## Why

A coated metal imported from pbrt renders **1.60x too bright** and too
desaturated. The coat lobe removes too little energy from the layer below it.

Measured on `tests/assets/suite/mat_coated_metal/` (gold `coatedconductor`,
Metal backend, path|wavefront, 128², 256 spp), sphere-region mean luminance:

| | pbrt | skinny | ratio |
|---|---|---|---|
| coated gold sphere | 0.1818 | 0.2917 | **1.604** |
| grey floor | 0.1602 | 0.1390 | 0.868 |
| environment background | 0.3982 | 0.3979 | **0.999** |

The defect is in the **coat**, and the measurements say so three ways.

**It is not the lighting.** The environment background matches to 0.999.

**It is not the metal.** The same gold under the same lighting, uncoated
(`tests/assets/suite/mat_conductor/`), matches pbrt closely — and errs *dark*,
which is the expected single-scatter GGX loss:

| uncoated gold sphere | ratio skinny/pbrt |
|---|---|
| roughness 0 | 0.923 |
| roughness 0.3 | 0.978 |

**It looks like under-attenuation rather than addition** — both renderers darken
a metal when it is coated, skinny just less. Coated sphere ÷ bare sphere within
each renderer:

| | coat's effect on the metal |
|---|---|
| pbrt | **0.298x** |
| skinny | **0.489x** |

0.489 / 0.298 = 1.64, which is suspiciously close to the 1.604 disc ratio.

**This last comparison is UNCONTROLLED and is not load-bearing.** The two scenes
differ in base roughness (0.3 vs 0.45) and in geometry (two spheres vs one), so
the ratio-of-ratios only cancels the nuisance variables if both renderers
respond to roughness identically — which is assumed, not shown. The matching
background rules out *miss* rays, not sphere illumination, area-light
visibility or secondary bounces, and the floor's 0.868x proves those paths
already disagree. Establishing the coat as the cause therefore needs an exact
counterfactual — one scene, one sphere, one base roughness, `coat` the only
difference, rendered in both systems. That is the first task, not a premise.

What survives without it: the environment matches (0.999x), the uncoated metal
matches (0.923–0.978x), and the coated one does not (1.604x). The coat is where
to look.

Two hypotheses were tested and refuted; they are recorded so nobody re-runs them:

- **The metal's RGB response.** skinny's named-metal F0 table reproduces the
  spectrally-integrated normal-incidence reflectance to within **0.3% luminance**
  for Au, Ag, Al and Cu. `--spectral` does not close the gap either (relMSE
  0.04112 vs RGB 0.03792).
- **"Zero the clearcoat" probe.** Zeroing skinny's coat makes the sphere
  *brighter* (0.930x) and still 1.73x pbrt, which reads as an acquittal of the
  coat but is not: it removes skinny's coat while the pbrt reference keeps its
  own, so it measures the wrong difference.

## What Changes

- **Quantify the loss channels first.** A per-material white-furnace probe on a
  lossless coated material measures how much energy skinny's coat keeps, with no
  pbrt involved. Read pbrt's `CoatedConductorBxDF` alongside it and attribute the
  1.64x to interface transmission, TIR re-hits, and layered-walk truncation.
- **Attenuate the base by the coat's transmission** in the flat BSDF, in whatever
  form the attribution supports. The current `coatAttenuation = coatColor` is the
  identity for every pbrt-imported coated material, so it is a free slot for the
  real factor rather than a value to fight.
- **Gate it.** `mat_coated_metal` becomes a real energy gate rather than a
  spelling gate, and a coated per-material furnace probe joins the furnace class.
- **BREAKING (renders):** every coated material changes appearance —
  `coateddiffuse` and `coatedconductor`, both flavours, both backends. Suite
  scenes `mat_plastic` and `samp_many_lights` use `coateddiffuse` and will move.

## Capabilities

### Modified Capabilities

- `flat-bsdf-lobes`: the coat lobe's energy contract. The existing requirement
  covers only the coat's Fresnel *selection* term, and its scenario asserts the
  coat "barely darkens a diffuse base" — true for a Lambert base, false for a
  metal, where the measurement says the coat must remove ~70% of it.
- `furnace-closure`: a coated material joins the per-material furnace probe, so
  the coat's energy is gated by an absolute invariant and not only against pbrt.

## Impact

- **Code:** `src/skinny/shaders/materials/flat/flat_material.slang` (the coat
  branch of `sample`, and the matching `evaluate`), possibly
  `src/skinny/pbrt/materials.py` if the coat needs a resolved transmission value
  it does not carry today.
- **Assets / gates:** `tests/pbrt/corpus/manifest.json` (re-measured baselines
  for every coated scene), a new coated furnace scene under
  `tests/assets/suite/`, and `tests/assets/suite/_gen/build.py`.
- **Docs:** `docs/SkinRendering.md` or `docs/Architecture.md` wherever the flat
  lobe set is described, `KnownBugs.md` item 4 (resolved), `CHANGELOG.md`.
- **Depends on** `coatedconductor-roughness-spelling`, which is implemented on
  branch `coatedconductor-roughness-spelling` (commit `ed09cb2`) and **not yet
  merged**. It contributes the `mat_coated_metal` scene this change gates
  against, and it fixed the coat's own roughness calibration. Do not start this
  change from a tree without it, or the measurements will not reproduce.

## Non-Goals

- **Multiple-scattering GGX energy compensation** for the base lobe. The
  uncoated conductor measures 0.923x at roughness 0 — a real but separate ~8%
  single-scatter loss, and fixing it here would confound the coat measurement.
- **pbrt's layered-BxDF controls** `thickness`, `albedo`, `g`, `maxdepth` and
  `nsamples`, which skinny reads on neither coated type. Matching pbrt's
  *stochastic layered walk* is a much larger change than making the existing
  single-scattering coat conserve energy.
- The floor's 0.868x. It is a different sign and probably the base lobe's
  single-scatter loss plus a different bounce distribution; it needs its own
  localization.
