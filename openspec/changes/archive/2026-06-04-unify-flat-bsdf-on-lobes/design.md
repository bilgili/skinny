## Context

`FlatMaterial` (the flat / UsdPreviewSurface / `std_surface` BSDF used for every
non-skin surface) exposes two methods through `IMaterial`:

- `sample(wo, rng)` — draws a bounce direction from a **3-lobe analytic model**:
  coat GGX (VNDF) + base-spec GGX (VNDF) + Lambert diffuse, over `FlatHitMat`
  (`data.mat`, binding 13). Its per-lobe Monte-Carlo weight is the VNDF identity
  `F·G₁(L)`, **bounded above by 1 by design** (the firefly fix documented in
  `samplers/ggx.slang` and `tools/firefly_debug.py`).
- `evaluate(wo, wi)` — returns the **full MaterialX `std_surface` closure**
  (`evalStdSurfaceBSDF`, `mtlx_std_surface.slang`) over `StdSurfaceParams`
  (`sp`, binding 19): `mx_dielectric_bsdf` coat, `mx_conductor_bsdf` with complex
  `mx_artistic_ior`, `mx_layer_bsdf` Fresnel layering, Oren-Nayar diffuse, energy
  compensation. Its `response` is glued onto the **3-lobe** `flatBsdfPdf`.

These are two unrelated BSDFs. They diverge three ways: different param source
(`data.mat` vs `sp`, with different packed defaults), `evaluate()` clamps
roughness `max(·, 0.04)` and multiplies pdf by opacity while `sample()` does
neither, and the diffuse term is Lambert in `sample()` but Oren-Nayar in
`evaluate()`. So `sample().pdf ≠ evaluate().pdf`, and `evaluate().response/pdf`
is two models divided — unbounded.

`evaluate()` is the renderer's **canonical BSDF**, called by NEE (`nee.slang`),
BDPT connections and reverse pdfs (`integrators/bdpt.slang`,
`wavefront/wavefront_bdpt.slang`), ReSTIR (`restir/*`), and the directional
proposal seam (`sampling/proposal.slang`). The seam **draws** with `sample()`
but **weights** with `evaluate()` (`weight = ev.response / mixPdf`,
`mixPdf = α_b·ev.pdf + α_e·envPdf`), and NEE's `mixtureProposalPdf` also uses
`evaluate().pdf`. When the env proposal is mixed in (BSDF+Env / Env presets), the
estimator therefore relies entirely on `evaluate()` agreeing with `sample()`.
For layered brass it doesn't → the mixture is biased (brass ~3.7% dark). The
BSDF-only preset is immune only because its fast path uses `sample()`'s own
self-consistent bounded weight.

## Goals / Non-Goals

**Goals:**
- One lobe-structured BSDF as the single source of truth, consumed identically by
  `sample()` and `evaluate()`, so `sample().pdf == evaluate().pdf` structurally
  and `evaluate().response/pdf` reduces to the bounded native per-lobe weight.
- Eliminate the proposal-mixture bias on layered/coated materials while staying
  firefly-free **by construction** (no clamp, no bias).
- One canonical BSDF for Path Tracer **and** BDPT, megakernel **and** wavefront.
- Establish a per-lobe, runtime-pluggable sampler **seam** — shipped unpopulated
  (native strategies only) — so Change 2 can inject alternative samplers without
  re-touching this code.
- Preserve the BSDF-only default exactly (pixel-identical), per `scene-sampling`'s
  baseline requirement.

**Non-Goals (→ future `per-lobe-sampler-registry`, Change 2):**
- Host sampler registry, GUI per-lobe selector, any non-native sampler.
- Whole-lobe composition (adding/removing/reordering lobes at runtime).
- Adding sheen / subsurface / rough-transmission lobes to the path-traced model.
- Energy-compensation / multiscatter parity with the MaterialX closure.

## Decisions

### D1 — Unify *down* onto the 3-lobe sampler model (not up to the closure)

`evaluate()` is rebuilt from the same lobe primitives `sample()` uses; the
MaterialX closure leaves the path-traced/BDPT estimator path.

*Rationale:* the bounce **already** samples only 3 lobes + delta transmission, so
the closure's extra richness was energy NEE/BDPT could see but the bounce could
never produce — the divergence, not a feature. The bounded `F·G₁` weight is a
deliberate, load-bearing firefly fix; the unify must preserve it, which requires
`evaluate()` to share the sampler's GGX so `response/pdf` reduces to `F·G₁`.
BDPT and PT-BSDF already converge to the 3-lobe-consistent value (brass ≈ 0.219),
so "down" lands on the already-correct answer.

*Alternatives considered:*
- **Up to the closure (B):** keep `evaluate()=evalStdSurfaceBSDF`, rewrite
  `sample()`+pdf to importance-sample the layered closure with a provably bounded
  weight. Rejected: research-grade (VNDF + exact solid-angle pdf for a
  Fresnel-layered coat-over-conductor closure), high firefly risk, and it churns
  the hot bounce path.
- **Param-sync only (C):** keep both models, equalize defaults/clamp/op.
  Rejected by prior evidence — the *models* differ, not just params; a partial
  sync produced an unexplained −7.5% shift.

### D2 — Single param source = `FlatHitMat` (`data.mat`)

`evaluate()` reads `data.mat`, the same struct `sample()` already uses. Drop the
`FlatMaterial.sp` member and the `loadStdSurfaceParams` call from
`loadFlatMaterial`.

*Rationale:* removes the `StdSurfaceParams`-vs-`FlatHitMat` default divergence at
the source, and a per-hit `std_surface` param load. `loadFlatMaterial`'s graph
base-color dispatch already writes `data.mat.albedo` independently of `sp`, so
nothing else needs `sp`; grep confirms the replaced `evaluate()` was its only
consumer. `evalStdSurfaceBSDF` + binding 19 stay for `preview_pass` (raster).

### D3 — `flatBsdfResponse` as the exact inverse of `sample()`'s weights

Reuse `flatBsdfPdf` verbatim (it is already `Σ selectProb · sampler.pdf`). Add
`flatBsdfResponse(wo, wi, mat) = Σ_lobe f_lobe·cos`, built per lobe as
`pdf_lobe · weight_lobe` with `weight_lobe` the **same bounded weight `sample()`
returns**:

- spec lobe: `f·cos = pdf_spec · (F·G₁)` (VNDF identity), so `response/pdf → F·G₁`.
- diffuse lobe: `f·cos = diffHue · cos/π`, with `pdf = cos/π` → `response/pdf =
  diffHue` (bounded).
- coat layered over the attenuated base, matching `sample()`'s `coatAttenuation`.

This yields `response = f·cos` (MaterialX convention, what NEE/BDPT expect) **and**
`response/pdf` ≤ the native per-lobe bound — consistency and firefly-freedom from
the same construction. The math is the algebraic inverse of weights `sample()`
already computes, so it is a mechanical extraction, not new BSDF theory.

### D4 — Runtime-pluggable sampler seam, shipped unpopulated

Each lobe carries a `samplerId`; a dispatch function maps it to an `ISampler`.
This change registers only native strategies (GGX-VNDF, Lambert) and the dispatch
has only the native case. The seam (the `samplerId` field + dispatch indirection)
exists so Change 2 adds cases + a host registry without re-refactoring.

*Rationale:* mirrors the shipped proposal seam (`proposals.py` ↔ mask bits) and
the `scene-sampling` plugin philosophy; runtime selection is what the project's
live-A/B verification depends on. *Alternatives:* compile-time generic
composition (zero dispatch cost, but no runtime switch, combinatorial variants) —
deferred; hybrid — YAGNI until a second sampler per lobe exists.

### D5 — Native-bounded weight, "trust" model for future injection

The native default samplers stay analytically bounded; the framework adds **no**
automatic MIS-vs-native net and **no** weight clamp. Keeping an injected
sampler's weight bounded (its pdf tracking its target `f`) is the IS author's
responsibility; `tools/firefly_debug.py` is the debug meter.

*Rationale:* zero overhead on the native (only) path that ships here; an
auto-net doubles per-lobe pdf evals; a clamp injects bias against the project's
unbiased-estimator goal. (Injection itself is Change 2; this only records the
contract.)

### D6 — Diffuse lobe is Lambert in both paths

`evaluate()`'s diffuse term becomes Lambert (cosine), matching `sample()`'s
`LambertSampler`, replacing the closure's Oren-Nayar.

*Rationale:* consistency requires the eval diffuse to match the sampled diffuse.
Cost: loses Oren-Nayar retroreflection on rough-diffuse surfaces. Impact is
negligible for the `three_materials` set (diffuse_roughness ≈ 0); Oren-Nayar can
return as a future native sampler+matching-eval pair if needed.

## Risks / Trade-offs

- **Changing `evaluate()` shifts BDPT MIS weights and absolute radiance for every
  flat material** → gate on *internal* convergence (PT-BSDF / BSDF+Env / Env /
  BDPT → one value **per column**, mega + wavefront) rather than matching the old
  absolute; regenerate goldens. The shift is *toward* the already-correct
  BSDF-only value.
- **The `sample()` refactor could break bit-exact BSDF-only parity** → preserve
  the RNG draw order and lobe-selection branch structure exactly; the
  `scene-sampling` "default is byte-identical" requirement + a pixel-identical
  check on the default preset is the guard.
- **Dropping `loadStdSurfaceParams` could regress textured/graph albedo** → the
  graph dispatch in `loadFlatMaterial` writes `data.mat.albedo` independently of
  `sp`; verify brass/marble/wood colors are unchanged on `three_materials`.
- **3-lobe model omits sheen/SSS/rough-transmission lobes some `std_surface`
  materials author** → those were never importance-sampled by the bounce, so
  NEE/BDPT were already inconsistent there; documented limitation, deferred to a
  future whole-lobe change.
- **Wavefront compiles per-material kernels (`per-material-pipeline`)** → the new
  `flat_lobes.slang` is included via `flat_material.slang`, which both mega and
  wavefront already import; wavefront parity is in the gate.
- **Injected samplers (Change 2) could reintroduce fireflies** → out of scope
  here; the seam ships native-only and D5 records the bounded-weight contract.

## Migration / Verification

Pure refactor — no flag, no data migration. Default (BSDF-only) behavior is
preserved, so rollout is just merge; rollback is reverting the branch (nothing
depends on it).

Verification order:
1. Default preset pixel-identical to pre-change (mega + wavefront).
2. Per-column convergence on `three_materials` IBL-only: PT-BSDF / BSDF+Env / Env
   / BDPT → one value per column (brass ≈ 0.219, plus marble, wood).
3. `tests/test_sampling_parity.py::test_env_proposal_unbiased_and_reduces_variance`
   passes; regenerate gitignored goldens
   `tests/_sampling_parity_golden_{megakernel,wavefront}.txt`.
4. ReSTIR suite passes.
5. Throwaway readback asserting `sample().pdf == evaluate().pdf` on the brass
   primary hit (removed before merge) to retire the prior −7.5% anomaly class.

## Open Questions

- Home for the lobe-kind enum + `samplerId` type: `flat_lobes.slang` vs a shared
  `interfaces.slang` — resolve during implementation by what keeps modules clean.
- Whether the coat attenuation / base layering in `flatBsdfResponse` matches the
  old closure closely enough that marble/wood don't *visibly* shift (gate is
  convergence, not absolute match, but worth an eyeball pass).
