# Design: coat-layer-energy

## Context

`flat_material.slang` models a coated material as **three lobes over one
parameter set** — `{coat, spec, diffuse}` — selected stochastically. The coat's
only effect on the layer beneath it is the selection probability:

```slang
float coatFresnel = (m.coat > 0.0) ? fresnelDielectric(NdotV, 1.0 / m.coatIOR) : 0.0;
float pCoat       = m.coat * coatFresnel;
float3 coatAttenuation = (m.coat > 0.0) ? m.coatColor : float3(1.0);
...
if (m.coat > 0.0 && rng.next() < pCoat) { /* coat lobe */ }
...
s.weight = bounceWeight * coatAttenuation;   // base lobes
```

`coatAttenuation` is `coatColor`, which the pbrt importer sets to `[1,1,1]` for
**every** coated material (`materials.py`, both coated branches). So the base is
attenuated by exactly one factor: the probability that the coat lobe was not
chosen. That is one Fresnel reflection's worth.

pbrt's `CoatedConductorBxDF` is a stochastic **layered** BxDF. A path refracts
through the interface, scatters off the conductor, and may reflect back down off
the *underside* of the interface — re-hitting the metal, multiplying by
`R_metal < 1` again — before it escapes. Energy leaves at every crossing.

The gap is measured, not inferred (numbers in `proposal.md`): the coat's effect
on the metal is **0.298x in pbrt and 0.489x in skinny**, a ratio of **1.64**,
and the sphere-region ratio is **1.604**. The environment matches to 0.999x and
the uncoated metal to 0.923–0.978x, so neither lighting nor the metal is
implicated.

## Goals / Non-Goals

**Goals:**

- The coat removes the right amount of energy from the layer below it, for a
  metal base and a diffuse base alike.
- The loss is **attributed** before it is fixed: how much is interface
  transmission, how much TIR re-hits, how much pbrt's own walk truncation.
- An absolute gate (white furnace) as well as a comparative one (pbrt-truth), so
  the coat's energy is pinned even where no pbrt reference exists.

**Non-Goals:**

- Reproducing pbrt's stochastic layered walk. This change makes the existing
  single-scattering coat conserve energy correctly; it does not add a layer
  simulation, and it does not read `thickness` / `albedo` / `g` / `maxdepth` /
  `nsamples`.
- The base lobe's own single-scatter GGX loss (0.923x uncoated at roughness 0).
- The floor's 0.868x.

## Decisions

### D1 — Control the comparison before attributing it

Three mechanisms have been proposed for this gap and all three were wrong or
overstated (see `proposal.md`). The pattern is not "we lack a fix", it is **we
keep reasoning from uncontrolled comparisons**, so the first artifact is a
controlled one.

The evidence that stands is bracketing: the environment matches to 0.999x, the
uncoated metal to 0.923–0.978x, the coated one to 1.604x. The evidence that does
NOT stand is the 0.298x-vs-0.489x ratio-of-ratios: `mat_conductor` and
`mat_coated_metal` differ in base roughness (0.3 vs 0.45) and in geometry, and
that cancels only if both renderers respond to roughness identically. The
matching background rules out miss rays, not sphere illumination or area-light
visibility, and the floor's 0.868x shows those paths already disagree.

So the first task is an **exact counterfactual**: one scene, one sphere, one
placement, one lighting rig, `conductor.roughness` equal to the plain
`conductor`'s `roughness`, rendered four ways —

| | pbrt | skinny |
|---|---|---|
| `Material "conductor"` | ✓ | ✓ |
| `Material "coatedconductor"`, same metal params | ✓ | ✓ |

The coat's effect is then coated ÷ bare within each renderer over *identical*
scenes, and the ratio of those two is the number this change has to move. Until
that exists, no percentage attributed to any mechanism means anything.

### D1b — Attribute against pbrt's real loss channel, not a guessed one

Task 1's second half reads pbrt's `LayeredBxDF`. Two corrections to the naive
framing, both of which would have sent the fix the wrong way:

- **Interface transmission and TIR are redistribution, not net loss.** Energy
  reflected back down off the interface underside is not gone; it re-hits the
  base and some of it escapes later. What it *does* do is multiply by the base's
  reflectance again, which darkens a metal and tints it — but that is a
  consequence of the base, not a coat absorption term.
- **pbrt's actual absorption channel is the layer medium.** The default coated
  material carries `thickness = 0.01`, `albedo = 0`, `g = 0`, and the walk
  multiplies throughput by the medium transmittance at every crossing. That is a
  real loss and skinny has no equivalent.
- **`maxdepth` / `nsamples` are estimator truncation.** Whatever energy pbrt
  loses to them is a property of pbrt's estimator, NOT a physical quantity to
  reproduce. If a measurable share of the 1.64x turns out to be truncation, the
  correct response is to say so and not match it.

The attribution is therefore a **sweep**, not a single ratio: vary `thickness`,
`albedo`, `maxdepth`, `nsamples` and interface roughness in pbrt over the
counterfactual scene and watch which one moves the gap.

### D2 — The seam is ONE coat-transfer function, not the `coatAttenuation` slot

`coatAttenuation` is where the factor gets *applied* in `sample`. It is not the
seam, because `sample` is not the only consumer of the material's response:

| consumer | reads |
|---|---|
| BSDF-only transport | `sample().weight` |
| NEE, BDPT, ReSTIR, env/neural proposals | `evaluate().response` |
| spectral transport | `flatBsdfResponseSpectral` |

A factor written into `sample` and the RGB `evaluate` gives **three copies and
one omission** — the spectral response would keep the old energy, so an RGB and
a `--spectral` render of the same coated material would disagree, and every
non-BSDF proposal would disagree with the BSDF one. That is the same defect class
as the hand-rolled roughness chain that `coatedconductor-roughness-spelling` just
deleted.

So the change introduces **one coat-transfer function** that all three consume.
Its signature has to admit what the transfer actually depends on — `wo`, `wi`,
the coat's weight / IOR / roughness, and the base lobe's reflectance — because an
internal-reflection term is not a material-level scalar: it depends on what the
base sends back up. A `float3` constant per material cannot express it.

This is a **response-only** change: it does not alter any pdf, so it does not
alter MIS weights or lobe selection. `weight = response / pdf`, so changing the
response changes the weight and leaves the pdf a valid sampling density.

The obvious candidate for the transfer is the classic entering-and-exiting factor
`(1 - F(NdotV)) * (1 - F(NdotL))`, times `coatColor`. At coat IOR 1.5 that is
only ~0.92 near normal incidence, so **it cannot account for 1.64x on its own**,
and a fix that stops there would close a small part of the gap while looking
principled. D1's sweep decides the real form.

**Discarded alternative:** tune `coat_color` at import to a value that makes this
scene match. That moves a rendering defect into the importer, breaks every
authored `coat_color`, and cannot generalize across coat IOR or base reflectance.

### D3 — The gate is absolute first, comparative second

`mat_coated_metal` gates against pbrt, which is the goal, but a pbrt-truth number
alone cannot tell an energy fix from a coincidence at one roughness and one
metal. The white furnace can: a lossless coated material has an exactly known
answer. So the furnace probe from D1 becomes a committed gate, and the pbrt-truth
baselines are re-measured on top of it.

**The furnace harness has a trap worth writing down.** Plain furnace mode
overrides *every* material in the scene, so a furnace render of a coated scene
and of an uncoated one come out **bit-identical** and the probe silently proves
nothing. The per-material path (`furnace.render_furnace` with `per_material` and
`furnace_material` set, the `furnace_per_material` disposition) is the one that
arms the bit on a single flagged material. Use it.

### D4 — Every coated scene is expected to move

`coateddiffuse` and `coatedconductor` share the coat branch, so `mat_plastic` and
`samp_many_lights` move too. That is correct — a Lambert base under a coat is
under-attenuated by the same mechanism, just less visibly, because it has far
less energy to keep. Their baselines are **re-measured**, and every one of them
must move toward pbrt. A coated scene whose error grows is a failed fix, not a
baseline to update.

## Risks / Trade-offs

- **The living spec's weight contract has to change with it.** `flat-bsdf-lobes`
  requires that `evaluate().response / evaluate().pdf` reduce to the native lobe
  importance weight (`F·G₁` for the GGX lobes, the albedo term for diffuse). A
  coat transfer multiplies that, so the requirement becomes "coat-transfer ×
  native weight" — with the boundedness it exists to guarantee retained, since
  the transfer is ≤ 1. Leaving it unmodified would ship two contradictory
  normative statements.
- **The existing diffuse scenario reads as a licence.** `flat-bsdf-lobes` says a
  coated diffuse "is within a few percent of the same material with `coat = 0`".
  That stays true for a Lambert base, but as written it would also excuse the
  metal case that is 1.64x wrong. The delta has to state the reason — the
  attenuation scales with what the base reflects — so the diffuse number is a
  consequence, not a rule.
- **A single scene can be matched by the wrong function.** A fix calibrated at
  one coat IOR (1.5), one base (gold), one roughness pair can be a curve fit that
  happens to land. The counterfactual sweep, the furnace probe and the
  `coateddiffuse` scenes are the defence; a gate phrased as "moves toward pbrt"
  is not, because a 1% improvement satisfies it. Every gate here carries a
  number.
- **Depends on unmerged work.** `coatedconductor-roughness-spelling` (`ed09cb2`)
  supplies `mat_coated_metal` and fixed the coat's own roughness calibration.
  Starting from a tree without it will not reproduce any number in `proposal.md`.
- **GPU-bound.** Every measurement here needs the Metal backend and the pinned
  pbrt v4; none of it runs in a hostless sweep. One guarded Metal process at a
  time.

## Design review (codex, pre-implementation)

Five findings, one CRITICAL, all folded in:

- **[CRITICAL] The localization was not controlled.** The 0.298x-vs-0.489x
  ratio-of-ratios compares scenes differing in base roughness AND geometry, so
  the cancellation was assumed rather than shown; a matching background rules out
  miss rays, not sphere illumination or area-light visibility, and the floor's
  0.868x proves those paths already disagree. D1 now demands an exact
  counterfactual, and `proposal.md` / `tasks.md` label that ratio suggestive-only.
- **[HIGH] The attribution was partly misconceived.** Interface transmission and
  TIR are redistribution, not net loss; `maxdepth`/`nsamples` are estimator
  truncation, not physics to reproduce; and the framing omitted pbrt's actual
  absorption channel, the layer medium (`thickness`, `albedo`, throughput ×
  transmittance per crossing). D1b replaces the single-ratio attribution with a
  parameter sweep.
- **[HIGH] `coatAttenuation` is a slot, not a seam.** `sample()`,
  `evaluate()` and `flatBsdfResponseSpectral` are three consumers; a factor
  applied in the first two leaves spectral transport and every non-BSDF proposal
  on the old energy. D2 is rewritten around one shared transfer function, and its
  parameters must admit the base reflectance — an internal-reflection term is not
  a per-material constant.
- **[HIGH] The spec delta left contradictory contracts**, and "a weight that
  disagrees with its pdf is bias" was simply wrong — the relation is
  `weight = response / pdf`, and a response may change while the pdf stays a
  valid density. The delta now MODIFIES the living bounded-weight requirement to
  "coat transfer × native weight" (bounded, since the transfer is ≤ 1) and states
  the change is response-only.
- **[HIGH] The gates could pass a curve fit.** "Approaches" and "moves toward"
  admit a 1% improvement. Every gate now carries a number, plus estimator-agreement,
  RGB≡spectral and a shape sweep over IOR / base reflectance / roughness /
  fractional coat / authored `coat_color`.
