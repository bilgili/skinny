## Context

PM-1 (`photon-mapping-sppm`, merged) stores a visible point at the first hit
whose sampled lobe is non-delta (`bs.pdf > 0`) — see
`shaders/integrators/wavefront_sppm.slang` `wfSppmEye`. Photon density estimation
then reconstructs that point's radiance. For a glossy reflector the lobe is a
narrow peak, so the gather cannot rebuild the sharp reflection; pure metals,
having no diffuse term, lose almost all signal. The A/B on `three_materials_demo`
(brass vs path) confirmed the loss is localized to the glossy spheres (scene-wide
mean 145 path / 138 SPPM).

## Goals / Non-Goals

- **Goal:** make near-specular flat reflectors reconstruct their reflections
  under SPPM, cheaply, without regressing PM-1's caustic parity or energy split.
- **Non-Goal:** low-variance mid-roughness glossy gather (that needs final
  gather — see below); skin/volumetric (PM-2/PM-3); changing photon emission.

## Decision: roughness-thresholded eye-walk continuation

Extend the eye-walk carrier branch. Today:

```
if (bs.pdf > 0.0) { store VP; return; }   // gather here
else { continue; }                         // delta carrier
```

Becomes (conceptually):

```
let glossyContinue = (bs.pdf > 0.0) && (lobeRoughness < fc.sppmGlossyContinueRoughness);
if (bs.pdf > 0.0 && !glossyContinue) { store VP; return; }
// delta OR glossy-continue ⇒ carry the throughput one bounce
```

`lobeRoughness` is the roughness of the lobe actually sampled (the metal/specular
lobe for a metal; not the coat/diffuse mix). The walk already accumulates
`throughput *= bs.weight` and re-traces, so the visible point lands on the next
non-glossy-continued surface and the reflection is reconstructed there. Over the
progressive passes the single sampled glossy direction per pass averages out,
matching how SPPM already averages the diffuse VP placement.

### Why this over alternatives

| Option | Reflection quality | Cost | Verdict |
|--------|--------------------|------|---------|
| Roughness-thresholded continue (this) | sharp for near-specular; noisy→converging for mid-rough | none beyond the existing carrier branch | **chosen** — high impact, fits the pipeline |
| Final gather at glossy VPs | low variance across the whole glossy range | +1 wavefront ray pass per VP + a gather kernel | deferred follow-up |
| Split BSDF (specular ray + diffuse gather) | correct everywhere | largest rewrite | out of scope |

## Deferred: final gather

For mid-roughness glossy (where one sampled direction per pass is noisy), the
fuller fix is a final-gather pass: at a glossy VP, BSDF-sample one gather ray,
look up the photon density estimate at its hit, and weight by the lobe. This
needs an extra ray/lookup stage in `record_sppm_loop` and a second VP role; it is
recorded here as the natural next change once the cheap continuation lands.

## Risks

- **Threshold choice.** Too high ⇒ matte surfaces wrongly continue (noise, lost
  diffuse VPs); too low ⇒ metals still gather. Default tuned on the demo;
  exposed via `--sppm-glossy-roughness` for scenes that need it.
- **fc tail growth.** One float added to the FrameConstants tail ⇒ bump
  `_FC_SCALAR_FIELDS` (Vulkan scalar blob 540→544 B; `_VK_UNIFORM_BUFFER_BYTES`
  768 still covers it) and re-pin the MSL fc size test, in the same change (the
  same lesson as PM-1's 516→540 bump).
- **Metal slot budget.** No new buffers ⇒ the ~15/31 SPPM slot budget is
  unaffected.

## Migration

Default threshold reproduces useful behavior; `0` reproduces PM-1 exactly. No
scene or API breakage.
