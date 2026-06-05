## Context

`unify-flat-bsdf-on-lobes` (merged, archived as spec `flat-bsdf-lobes`) put
`FlatMaterial.sample()` and `FlatMaterial.evaluate()` on one `{coat, spec,
diffuse}` lobe set and added a per-lobe `samplerId` parameter to
`flatSampleLobe` / `flatLobePdf` in `shaders/materials/flat/flat_lobes.slang`.
The parameter is currently **ignored** — every call site passes
`FLAT_SAMPLER_NATIVE` and the helpers hard-wire the 2023 spherical-cap / bounded
VNDF (coat/spec — `samplers/ggx.slang` already uses the Heitz & d'Eon 2023
spherical-cap parameterisation) and cosine (diffuse). The dispatch indirection is
in place but inert.

The proposal seam (`sampling/proposals.py`, `ProposalPlugin`, the
`PROPOSAL_PLUGINS` registry, `proposal_mask_and_alpha()` →
`FrameConstants.proposalMask` → `fc` read in `shaders/sampling/proposal.slang` →
`proposal_preset_index` GUI param + `--proposals` CLI) is the established pattern
for "host registry folds a runtime selection into one `FrameConstants` field that
the shader reads, no new bindings for analytic plugins." This change applies that
same pattern to per-lobe sampler selection.

Constraint that dominates the design: `unify-flat-bsdf-on-lobes` is unbiased and
firefly-free **because `sample()` and `evaluate()` share one pdf and the per-lobe
weight is bounded by construction**. Any sampler the seam injects must preserve
both properties without a clamp or MIS net (the locked "native-bounded + trust"
decision).

## Goals / Non-Goals

**Goals:**

- Populate the `samplerId` seam so each lobe's draw strategy is selectable at
  runtime and from the GUI.
- Register the first alternates: the Heitz-2018 basis-form VNDF (GGX coat/spec —
  a different warp of the same distribution the native 2023 spherical-cap
  sampler draws from), uniform-hemisphere (Lambert diffuse).
- Keep `sample()`/`evaluate()` pdf agreement — hence unbiasedness and bounded
  weight — for **every** registered strategy, structurally rather than by test.
- Reuse the proposal seam's host-registry / `FrameConstants` transport shape so
  the two seams look and behave consistently.

**Non-Goals:**

- No new lobes (the set stays `{coat, spec, diffuse}`).
- No neural / online-trained / learned samplers.
- No per-material selection — the selection is a global GUI/CLI knob shared by
  all flat materials this iteration.
- No automatic MIS safety net and no weight clamp; an injected sampler whose pdf
  does not track its f is a defect, not an accepted approximation.

## Decisions

### D1 — Selection lives in one `FrameConstants` field, not per-material

Add a single `uint flatLobeSamplers` to `FrameConstants` (8 bits per lobe:
`coat | spec<<8 | diff<<16`). The renderer holds three selection indices and
folds them to this uint in `_pack_uniforms`, in std140 lockstep with the struct
declaration in `common.slang`.

*Why:* the selection is an interactive A/B knob, identical for every flat
material in the frame — exactly the role `proposalMask` plays. A per-material
`samplerId` would add SSBO state and a material-authoring surface for no benefit
this iteration. *Alternative rejected:* per-material field in `FlatHitMat`
(more GPU state, no GUI story, premature).

### D2 — Response is sampler-invariant; only the {draw, pdf, weight} triple plugs

`flatBsdfResponse` returns `f·cos`, a physical quantity independent of how `wi`
was drawn. It stays **byte-for-byte unchanged** (it internally spells `f·cos`
via the native pdf purely as algebra). Pluggability is confined to: the draw
(`flatSampleLobe`), the density (`flatLobePdf`), and `sample()`'s per-lobe
bounded weight.

*Why:* this is the minimal surface that keeps NEE / BDPT / ReSTIR — all of which
consume `evaluate().response` — untouched. *Alternative rejected:* recomputing
the per-lobe weight generically as `f·cos / pdf` (reintroduces the unbounded
division at grazing microfacets that Change 1 removed for the spec lobe).

### D3 — `flatLobePdf` branches on samplerId only where densities differ

The native 2023 spherical-cap VNDF and the alternate Heitz-2018 basis-form VNDF
sample the **same** visible-normal distribution `D_vis`; their solid-angle pdf is
analytically identical (only the unit-square→`D_vis` warp differs). So for the GGX
coat/spec lobes `flatLobePdf` stays samplerId-agnostic — one analytic VNDF pdf
(the existing `GGXSampler.pdf`) — and only `flatSampleLobe` branches to pick the
warp. `flatLobePdf` branches on samplerId **only** for the diffuse lobe (cosine
pdf `NdotL/π` vs uniform-hemisphere pdf `1/2π`).

*Why this matters:* because the GGX pdf is one shared code path, `sample().pdf`
and `evaluate().pdf` are equal by construction for the basis-form strategy — not
a numeric coincidence to be verified, but a structural identity. The basis-form
strategy therefore **cannot** introduce bias; it can only change the noise
realization (and exhibit the basis singularity near `V ∥ N` that the 2023
spherical-cap form was designed to remove). That is precisely why it is a safe
first alternate: same distribution, same pdf, same bounded weight, different
warp.

### D4 — Per-lobe bounded weight per strategy

- **GGX basis-form (and native spherical-cap):** the VNDF estimator
  `BRDF·cos / pdf` reduces to `F·G₁` for any warp of the same `D_vis`, so
  `sample()`'s coat/spec weight code is unchanged — only the draw call changes.
  Bounded ≤ 1, as today.
- **Diffuse uniform-hemisphere:** weight `= f·cos / pdf = (albedo/π)·cos /
  (1/2π) = 2·diffHue·cos`, bounded ≤ `2·albedo`. `sample()`'s diffuse branch
  selects this bounded form when the diffuse samplerId is uniform.

*Why:* each strategy carries its own closed-form bounded weight (decision #3 of
the brainstorm). No strategy divides two mismatched models.

### D5 — Host registry mirrors `ProposalPlugin`; dispatch falls back to native

New `sampling/lobe_samplers.py`: a `LobeSamplerStrategy` descriptor (`name`,
`valid_lobes` mask, `shader_id`, `cli_token`) and a registry list, shaped like
`proposals.py` + `PROPOSAL_PLUGINS`. The GUI builds each lobe's dropdown from the
strategies whose `valid_lobes` includes that lobe (data-driven, not three
hand-maintained lists). The shader `switch` falls back to the native branch for
any `(lobeKind, samplerId)` pair it does not recognize, so a stale persisted
selection or an out-of-range id degrades safely to native rather than producing
garbage.

*Why:* consistency with the proposal seam (one mental model for "pluggable
sampling in skinny") and graceful degradation on bad input.

### D6 — CLI / GUI / persistence parallel to `--proposals`

`--lobe-samplers coat=sphcap,spec=sphcap,diff=uniform` (with an env fallback),
three `ALL_PARAMS` dropdowns (`flat.coat_sampler`, `flat.spec_sampler`,
`flat.diffuse_sampler`) resolved via the existing `_get_nested`/`_set_nested`,
and the three indices added to `_current_state_hash` so a change resets
progressive accumulation — exactly as the proposal/reuse selections behave.

## Risks / Trade-offs

- **std140 packer drift** (adding a `FrameConstants` field) → the
  `common.slang` struct and `renderer._pack_uniforms` must change together; guard
  with a byte-size assertion / existing uniform-pack test so a mismatch fails
  loudly instead of corrupting later fields (`proposalAlpha`).
- **Both backends must pack the field** → the Metal and Vulkan paths share
  `_pack_uniforms`, but verify the Metal `ShaderCursor` binding picks up the
  enlarged uniform; add the all-native pixel-identity check on both backends.
- **A wrong basis-form warp** would silently raise variance or — if its implied
  density diverged from the shared VNDF pdf — bias, defeating D3 → mitigation: the
  parity gate (brass ≈ 0.219 to 5 dp with coat/spec = basis-form VNDF, mega ≡
  wave, PT ≡ BDPT) plus a variance-non-regression bound vs native.
- **Diffuse uniform is intentionally higher variance** (it is the worse sampler)
  → it ships as a framework demonstrator / A/B reference, not a default; the
  variance test asserts cosine < uniform, confirming the seam actually swaps the
  strategy.
- **Stale persisted selection** referencing a removed strategy id → host
  validates indices on load and the shader falls back to native (D5).

## Migration Plan

Additive and backward-compatible. With the default selection (all lobes native)
the output is pixel-identical to current `main` on both backends — the gate.
No data migration; new settings keys are absent in old snapshots (treated as
native) and ignored by older builds. Rollback is a plain revert of the branch;
the `FrameConstants` field is the only wire-format change and it is append-only.

## Open Questions

- None blocking. The exact bit layout of `flatLobeSamplers` (packed uint vs three
  scalar uints) is an implementation detail settled in tasks; packed uint is the
  default to minimize std140 disturbance.
