# Tasks: coat-layer-energy

Group 1 produces numbers, not code. Three mechanisms for this gap have already
been proposed and all three were wrong or overstated (proposal.md). The failure
mode is not a missing fix, it is **reasoning from uncontrolled comparisons**, so
group 1 builds the controlled one first.

**Start from branch `coatedconductor-roughness-spelling` (`ed09cb2`), not main.**
It supplies `mat_coated_metal` and fixed the coat's roughness calibration;
without it none of the recorded numbers reproduce.

## 1. Control the comparison, then attribute it

- [ ] 1.1 Build the **exact counterfactual**: one scene, one sphere, one
      placement, one lighting rig, authored twice — `Material "conductor"` and
      `Material "coatedconductor"` with the SAME metal eta/k and the SAME base
      roughness. Render all four cells (2 materials x pbrt/skinny). The coat's
      effect is then coated ÷ bare over identical scenes, and the ratio of those
      two ratios is the number this change must move. Do NOT reuse the
      `mat_conductor` vs `mat_coated_metal` comparison recorded below: those
      scenes differ in base roughness (0.3 vs 0.45) and geometry, so it is
      suggestive only.
- [ ] 1.2 Sweep pbrt over the counterfactual to find which channel carries the
      loss: `thickness`, `albedo`, `g`, `maxdepth`, `nsamples`, interface
      roughness. Note before starting that interface transmission and TIR are
      **redistribution, not net loss** (the returned energy re-hits the base and
      some escapes later — what it does is multiply by the base reflectance
      again), and that pbrt's real absorption channel is the layer medium
      (`thickness = 0.01`, `albedo = 0` by default, throughput x medium
      transmittance at every crossing).
- [ ] 1.3 Separate the share that is pbrt **estimator truncation** (`maxdepth`,
      `nsamples`) from the share that is physical. Truncation is a property of
      pbrt's estimator, NOT a quantity to reproduce — if it is material, say so
      and do not match it.
- [ ] 1.4 Build a lossless coated furnace probe: a white base with `coat = 1`,
      default `coatIOR` 1.5, beside the uncoated control. "Lossless" means the
      base reflects 1.0 in every channel and the coat has white `coat_color`, so
      a correct model returns the furnace constant exactly. Use the
      **per-material** furnace path (`furnace.render_furnace` with
      `per_material` + `furnace_material`): plain furnace mode overrides EVERY
      material, so coated and uncoated scenes render bit-identical — verified, it
      returns 0.8789 for white-diffuse, coated-white and coated-metal alike.
- [ ] 1.5 Decide the transfer function's FORM from 1.2–1.4 and write it into
      design D2 before any shader code. `(1-F(NdotV))·(1-F(NdotL))` is only ~0.92
      near normal incidence at IOR 1.5, so it alone cannot account for 1.64x — a
      fix that stops there closes a small part of the gap while looking
      principled.

## 2. One coat-transfer function, consumed by every estimator

- [ ] 2.1 Add the coat-transfer function to the flat material shader. Its
      signature admits `wo`, `wi`, the coat's weight / IOR / roughness and the
      base lobe's reflectance — an internal-reflection term depends on what the
      base sends back up, so a per-material `float3` constant cannot express it.
- [ ] 2.2 `sample()` consumes it (the `coatAttenuation` slot, which is
      `coatColor` = identity for every pbrt-imported coated material).
- [ ] 2.3 `flatBsdfResponse` / `evaluate()` consumes the SAME function — this is
      the path NEE, BDPT, ReSTIR and the env/neural proposals read.
- [ ] 2.4 `flatBsdfResponseSpectral` consumes it too. Omitting this makes an RGB
      and a `--spectral` render of one coated material disagree, and it is the
      copy most likely to be forgotten.
- [ ] 2.5 Response-only: no pdf changes, so lobe selection and MIS weights are
      untouched. `weight = response / pdf`; the pdf stays a valid sampling
      density.
- [ ] 2.6 `coat = 0` stays byte-identical — every coat term is already gated on
      `m.coat > 0.0`.

## 3. Gates, each with a number

- [ ] 3.1 **The counterfactual gate.** skinny's coated ÷ bare ratio from 1.1
      matches pbrt's to within a recorded tolerance. This is the primary gate;
      record the pre-fix value as the baseline it must beat, and state the
      minimum improvement that counts — "moves toward pbrt" passes on a 1%
      darkening and is not a gate.
- [ ] 3.2 **Estimator agreement.** `sample()`'s expected throughput and the
      integral of `evaluate().response` agree for a coated material, and
      `sample().pdf == evaluate().pdf` for the same direction pair. This is what
      catches the transfer landing in one consumer and not another.
- [ ] 3.3 **RGB ≡ spectral.** A coated material renders consistently under
      `--spectral` and RGB (the existing spectral self-consistency tolerance).
      Fails loudly if 2.4 was skipped.
- [ ] 3.4 **The furnace probe from 1.4** is committed as a suite scene with a
      `furnace_per_material` disposition, with its shortfall recorded as the
      coat's energy error rather than absorbed into a non-uniformity baseline.
- [ ] 3.5 **Shape, not one point.** Sweep the fix over coat IOR, base
      reflectance, base roughness, coat roughness, fractional `coat`, and an
      authored non-white `coat_color`. A function calibrated at one point that
      diverges elsewhere is a curve fit.
- [ ] 3.6 Re-measure EVERY coated scene: `mat_plastic` and `samp_many_lights`
      (`coateddiffuse`), `mat_coated_metal` and its `_mtlx` sibling. Each must
      improve by the 3.1 threshold or better. A coated scene whose error grows is
      a failed fix, not a baseline to update. `mat_coated_metal` starts at relMSE
      **0.03792** / FLIP 0.03630, sphere ratio **1.604** — the ratio is the
      number that has to move, since relMSE is dominated by the background, which
      already matches at 0.999x.
- [ ] 3.7 Uncoated controls do NOT move: `mat_conductor` (0.923x / 0.978x),
      `furnace_conductor`, `furnace_lambert`, `furnace_rough_conductor`. A change
      reaching them means the coat gate leaked.
- [ ] 3.8 Full parity matrix on the Metal backend, one guarded process at a time,
      across path / BDPT / SPPM and both execution modes. Re-record the manifest
      `baselines` for scenes that legitimately moved; never loosen a
      self-consistency tolerance to hide a divergence.
- [ ] 3.9 Hostless sweep against the recorded pre-existing failures;
      `ruff check src/skinny`.

## 4. Documentation

- [ ] 4.1 `KnownBugs.md` item 4 resolved, with the final attribution replacing
      the "not yet resolved" note it carries now.
- [ ] 4.2 The flat lobe-set documentation records that the coat attenuates the
      layer below it, by what factor, and which estimators consume it.
- [ ] 4.3 `CHANGELOG.md` — every coated material changes appearance, so this is a
      user-visible render change, not an internal fix.
- [ ] 4.4 `openspec validate --strict`; codex pre-merge review.

## Recorded measurements

All from the Metal backend, path|wavefront, 128², 256 spp, on `ed09cb2`.

**Controlled** (same scene, same lighting, region means):

| Quantity | Value |
|---|---|
| `mat_coated_metal` sphere ratio skinny/pbrt | **1.604** |
| ... floor ratio | 0.868 |
| ... environment background ratio | **0.999** |
| `mat_conductor` sphere ratio, roughness 0 / 0.3 (uncoated) | 0.923 / 0.978 |
| `mat_coated_metal` pbrt-truth relMSE / FLIP | 0.03792 / 0.03630 |

**Uncontrolled, suggestive only** — the two scenes differ in base roughness and
geometry, so this does not establish the coat as the cause. Task 1.1 replaces it:

| Quantity | Value |
|---|---|
| coat's effect on the metal — pbrt | 0.298x |
| coat's effect on the metal — skinny | 0.489x |
| implied under-attenuation | 1.64 |

Refuted, do not re-run: the metal's RGB response (the named-metal F0 table
reproduces the spectrally-integrated normal-incidence reflectance to within 0.3%
luminance for Au/Ag/Al/Cu); `--spectral` (0.04112 vs RGB 0.03792); and the "zero
the clearcoat" probe, which removes skinny's coat while pbrt keeps its own and so
measures the wrong difference.
