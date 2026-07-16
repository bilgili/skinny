# Design — sppm-power-proportional-photon-groups

## Context

`sppmEmitPhoton` (wavefront_sppm.slang ~545) picks one of up to four photon-emission
groups (emissive triangles, sphere lights, distant lights, environment) with
`gsel = 1/G` and divides each branch's flux by `gsel`. Unbiased, but per-photon flux
is `≈ Φ_g·G` — proportional to the group's own power — so the env group's huge power
(`πR²·∫L dω`, bbox disc πR² ≈ 85+ on glass_caustics_test) rides on 1/G of the photon
budget as sparse, enormous splats → firefly speckle (~1.7× path-tracer noise).

pbrt solves this with a light *power distribution*: sample lights ∝ emitted power Φ,
divide by the actual pmf. Per-photon flux then ≈ Φ_total for every group.

Host already owns every quantity needed:
- `_emissive_total_power = Σ(area·lum)` (renderer.py ~5867, feeds
  `fc.emissiveTotalPower`) → Φ_E = π·Σ(area·lum).
- Sphere lights packed in `_upload_sphere_lights` (radiance, radius) →
  Φ_S = Σ lum·π·(4πr²) = 4π²·Σ(lum·r²).
- Distant lights packed in `_upload_distant_lights` (radiance) → Φ_D = πR²·Σlum
  (parallel beam through the bbox disc; matches the shader's emission geometry).
- Env CDF built in `environment.build_env_distribution` from sinθ-weighted luminance;
  its grid total gives ∫L dω = total·(π/H)·(2π/W) → Φ_env = πR²·envIntensity·∫L dω
  (= pbrt `ImageInfiniteLight::Phi` = 4π²R²·L_avg).
- R = `max(0.5·‖sceneBoundsExtent‖, 1e-4)` — same formula the shader uses.

## Goals / Non-Goals

**Goals:**
- Equalise per-photon flux across emission groups; kill the env-photon fireflies on
  mixed weak-local-light + environment scenes.
- Keep the estimator unbiased: divide by the actual selection probability.
- Zero behavior change when only one group is present (pmf = 1 for it), and graceful
  uniform fallback when total power is zero/non-finite.

**Non-Goals:**
- No per-light power CDF *within* a group (emissive triangles already power-weight
  internally; sphere/distant stay uniform within-group — within-group flux spread is
  not the observed failure).
- No change to emission geometry, flux formulas, validity guards, the depth ≥ 1
  env-indirect partition, or SPPM eye/update stages.
- No minimum-probability floor (pure pbrt-style power proportionality; see Risks).

## Decisions

**D1 — pmf computed on host, uploaded via FrameConstants scalar tail.**
Summing light powers per photon on GPU is wasteful and the data (env integral,
triangle areas) is host-resident. Four floats `sppmGroupPmfE/S/D/Env` append after
`filmMaxComponent` and before the `#if defined(SKINNY_METAL)` `tileOriginY`, with
`_pack_uniforms` packing them in exactly that order. The load-bearing contract is
**packer order == field order** (the struct is scalar-packed — `uint3 sppmGridRes`
and `float4 proposalAlpha` already sit at tight offsets with no std140 padding, so a
`float4` would have been equally correct; individual floats are just the established
scalar-tail idiom). Zero when the integrator isn't SPPM. Alternatives — packing raw
powers and normalising in-shader, or 1 new fc float (env integral) + reusing
`fc.emissiveTotalPower` + in-shader sphere/distant sums — rejected: normalisation +
fallback logic is host-testable pure Python, and the 3-float saving isn't worth
splitting the pmf across host and shader.

**D2 — pmf assembly is a pure module-level helper.**
`_sppm_photon_group_pmf(emissive_power, sphere_power, distant_power, env_power)`
in renderer.py: clamp negatives/non-finites to 0, normalise; if the sum is ≤ 0 or
non-finite, return uniform over groups with *presence* (power arguments are passed
already presence-gated: 0 for an absent group, so "uniform over positive-presence
groups" needs the presence bits too — signature takes `(powers, present)` tuples).
Hostless-unit-testable without a GPU.

**D3 — presence predicates mirror the shader exactly.**
Host gates each group's power by the same predicate the shader uses for `hasX`:
`numEmissiveTriangles > 0`, `numSphereLights > 0`, `numDistantLights > 0`,
`furnaceMode == 0 && envIntensity > 0`. A present group with zero measured power
(can't happen today — zero-power lights are dropped at upload) would get pmf 0 and
contribute nothing; unbiased either way.

**D4 — shader selection = CDF walk over the four fc pmf entries.**
Replace the uniform `pick` mapping with: draw `u`, walk `pE, pS, pD, pEnv`
accumulating; choose the first group with `u < acc`; the **last positive-pmf group
absorbs the residual** (float cumsum < 1). Each branch divides by its own `p_g`
(replacing `gsel`). The existing `G == 0` early-out stays as a belt; additionally
return false if all four pmf entries are ≤ 0 (stale/foreign fc). Selection is
λ-independent, so the spectral build shares the identical scalar pmf.

**D5 — power sums cached where the data is packed.**
`_upload_sphere_lights` stores `Σ lum·r²` over the *enabled, capacity-clamped* set;
`_upload_distant_lights` stores `Σ lum` likewise; `_ensure_env_uploaded` stores the
env luminance integral (`build_env_distribution` grows a third return value — its
grid total; both call sites updated). `πR²`/`4π²` constants and `envIntensity` are
applied at pack time so light edits and intensity slides stay live without re-upload.
Accumulation reset needs no new hash field: every input (light sets, env, intensity,
furnace) already resets via `_current_state_hash`.

**D6 — probe/override hook.** A host attribute `_sppm_group_pmf_override`
(None default; a 4-tuple forces the packed pmf verbatim) keeps the existing spec's
"forced-env probe" scenario expressible under the new selection path (pmf
`[0,0,0,1]`) without touching the shader.

**D7 — spec delta is one ADDED requirement** ("Photon emission group selection is
power-proportional") on `photon-mapping`. The existing env-emission requirement
words flux as `1/p_sel` generically and needs no text change.

## Risks / Trade-offs

- **[Caustic photon share drops]** On glass_caustics the sphere light's power share
  is small (Φ_S ≈ 4π²·lum·0.04 vs Φ_env with πR² ≈ 85 ⇒ p_S ≈ 1–2%), so the
  sphere-driven caustic term gets ~25–50× fewer photons than under uniform selection
  — its variance can rise while overall speckle collapses. Note the truth image's
  caustic is largely env-driven on this scene (the env dominates total power), so
  power-proportional allocation plausibly *improves* the caustic region too — but
  whole-image noise_sigma is background-dominated and **blind** to a caustic-region
  regression (design-review MAJOR). → The A/B gate MUST measure **caustic-region**
  noise_sigma/relMSE (masked to the caustic footprint) in addition to whole-image,
  before/after at matched spp. Pre-agreed remedy if the region regresses vs the
  uniform baseline: a p_min floor (e.g. 0.05 per present group) in the host pmf
  helper — a one-line amendment, applied and re-measured within this change.
- **[FrameConstants layout drift Metal vs Vulkan]** New fields must sit before the
  Metal-gated `tileOriginY` and the packer order must match field order (obs: Metal
  reflects offsets from the struct; Vulkan reads the byte blob). → Wiring guard test
  asserts field order in common.slang and pack order in `_pack_uniforms`.
- **[Env integral double-scaling]** `sampleEnvDir` folds `envIntensity` into radiance;
  the CDF/luminance grid is unscaled. Host multiplies the integral by `envIntensity`
  exactly once at pack time. → Unit test pins the formula.
- **[Distant-vs-env power comparability]** Φ_D uses the same πR² disc as the shader's
  emission geometry, so distant/env relative weights are consistent by construction.

## Migration Plan

Shader + host land together (fc struct change forces a full recompile of all wavefront
kernels and the megakernel). RGB `.spv` byte-diff is expected and confined to fc
layout. Rollback = revert the commit; no persisted state or scene-format change.

## Open Questions

None — measurement outcomes (A/B deltas, gate baselines) are recorded during apply.
