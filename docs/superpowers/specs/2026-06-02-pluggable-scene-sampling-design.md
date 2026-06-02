# Pluggable scene-sampling seam (ReSTIR / neural-guiding ready)

**Date:** 2026-06-02
**Status:** Approved design

## Problem

We want to experiment with whole-scene sampling algorithms — **ReSTIR**
(spatiotemporal reservoir reuse) and a **neural spline-flow guided sampler**
(learned directional proposal, see `neural_spline_guiding_docs/`). Neither fits
the current renderer:

- Importance sampling today lives **inside materials** (`IMaterial.sample` picks
  its own BSDF lobe and returns `weight = f·cos/bsdfPdf`), plus NEE
  (`nee.slang::allLightsNEE`) and the env-importance CDFs (bindings 31–32). MIS
  is power-heuristic, hard-wired to the BSDF pdf.
- The `ISampler` seam (`interfaces.slang`) is low-level (tangent-space BSDF
  lobes, compile-time monomorphized). There is no seam at the level ReSTIR /
  neural guiding operate: *which direction or light to pick at each path vertex*,
  via reuse (ReSTIR) or a learned proposal mixed into MIS (neural).

This change adds **the seam those algorithms plug into** — and nothing more. It
ships a baseline (refactor of today's behaviour) plus one tiny second proposal to
prove the mixture path works end-to-end. ReSTIR and the neural sampler are each
their own follow-up spec (see *Follow-up specs*).

## Decisions (locked during brainstorm)

1. **Scope = the seam only.** Ship the pluggable abstraction + a baseline plugin
   + one small second proposal. ReSTIR and neural guiding are separate specs.
2. **Wavefront-primary.** The seam targets the wavefront backend (persistent
   buffers, multi-pass, compute inference are first-class). Megakernel supports
   only the subset that fits a single dispatch (the inline proposal mixture).
3. **Two orthogonal attach points, one host abstraction.** A *directional-proposal*
   hook at the BSDF bounce, and a *reuse/resampling* hook around NEE + the
   indirect spawn. They compose freely (neural proposal + ReSTIR reuse "just
   works"), mirroring the already-separated `nee.slang`.
4. **Reference plugins that ship with the seam:** `BsdfProposal` (baseline,
   end-to-end wired, proves pixel-parity) + `EnvImportanceProposal` (tiny second
   proposal, exercises 2-proposal MIS for real). Reuse side ships
   `IdentityReuse` only (interface + passthrough).
5. **Proposal dispatch = runtime-uniform mixture.** A small proposal registry
   selected by `fc.proposalMask` + `fc.proposalAlpha`, one-sample MIS, divide
   by the full mixture pdf. Matches the `fc.integratorType` precedent; instant
   A/B switching, no recompile. The reuse hook is pass-structural (rebuilt on
   switch, like `execution_mode`).

## Architecture

```
                       bounce vertex
   Ray ─► hit ─► [PROPOSAL seam] ─► wi, mixPdf ─► throughput
                      │                              │
                 [REUSE seam] ◄── NEE + indirect ────┘
```

Both seams share one host abstraction and differ only in attach point:

- **`SamplingPlugin`** (Python ABC) owns lifecycle + *optional* GPU passes +
  buffers + descriptor bindings + `FrameConstants` uniform bits + UI/CLI/settings
  wiring. Subtypes `ProposalPlugin` / `ReusePlugin`. Cheap analytic plugins own
  zero passes (pure inline callable); heavy plugins (neural pre-pass, ReSTIR
  reservoir passes — later) own passes through the same socket.
- **Slang side** provides a per-vertex callable + an exact-pdf callable. No
  existential types — monomorphized like materials / `ISampler`.

### Proposal seam (bounce direction)

All directions tangent-space (N=+Z); all pdfs **solid-angle (sr⁻¹)**.

```hlsl
struct ProposalContext { float3 woT, N, T, B, position; uint materialId;
                         float roughness, metallic; float3 baseColor; };
struct ProposalSample  { float3 wiT; float pdf; bool delta, valid; uint version; };

interface IProposal {
    ProposalSample sample(ProposalContext c, float2 u);   // draw a direction
    float          pdf(ProposalContext c, float3 wiT);    // exact SA pdf; 0 if unproducible
}
```

`MixtureProposal`, generic over the material `<TM : IMaterial>`, performs
**one-sample MIS**: pick proposal *j* ∝ αⱼ (from `fc.proposalMask` +
`fc.proposalAlpha`), draw `wi`, return `mixPdf = Σ_k α_k · pdf_k(wi)`. The
bounce in `integrators/path.slang::evaluateBounce` (and the wavefront shade)
changes from `mat.sample(...)` to:

```hlsl
MixtureProposalResult mp = sampleMixture(mat, ctx, rng);
if (mp.delta) {
    // singular lobe: take the material's own exact delta weight (today's path)
} else {
    BSDFSample ev = mat.evaluate(wo, mp.wiT);      // response = f·cos, ev.pdf = bsdfPdf
    br.bsdfSample.wi     = mp.wiT;
    br.bsdfSample.weight = ev.response / max(mp.mixPdf, EPS);
    br.bsdfSample.pdf    = mp.mixPdf;              // feeds downstream sphere/env-miss MIS
}
```

**Parity lever.** Baseline set `{BSDF}` ⇒ `mixPdf == bsdfPdf` ⇒
`response/bsdfPdf` equals today's `weight` ⇒ bit-identical output.

**Three correctness rules, interface-enforced:**

1. **Delta pass-through.** `delta=true` (pdf==0 lobe: mirror, smooth dielectric)
   skips the mixture and takes the material's own weight. A singular lobe cannot
   MIS-mix with a continuous proposal.
2. **NEE coupling.** `allLightsNEE`'s MIS companion pdf must switch from
   `bsdf.pdf` to the **same** `mixturePdf(mat, ctx, wiToLight)`. Otherwise the
   estimator is biased whenever a non-BSDF proposal is active. (Baseline:
   identical, since the mixture is BSDF-only.)
3. **Same pdf as sampled.** Downstream sphere-light and env-miss MIS use
   `mixPdf` (the actual sampling density of the chosen `wi`), not `bsdfPdf`.

Concrete proposals shipped:

- **`BsdfProposal`** — `sample` delegates to `mat.sample`, `pdf` to
  `mat.evaluate(...).pdf`. Zero buffers, zero passes. Mask bit 0, always on.
- **`EnvImportanceProposal`** — reuses the **existing** env-CDF bindings 31–32;
  no new GPU state. Samples a world direction from the env distribution and
  converts world↔tangent inside `sample`/`pdf` using `c.T/B/N`. Mask bit 1.

### Reuse seam (NEE + indirect)

This spec defines the **interface + identity baseline only**. The integrator
calls the reuse seam instead of inline `allLightsNEE`; `IdentityReuse` forwards
verbatim to stock NEE + the stock indirect spawn (parity). On the Python side a
`ReusePlugin` can later inject reservoir passes + per-pixel buffers + temporal
reprojection before/after the shade pass — which is why reuse is pass-structural
(rebuilt on switch), not a hot uniform branch. Reservoirs / RIS / spatio-temporal
passes are **out of scope** here; the socket is shaped to receive them.

## Changes

### 1. Slang — proposal seam (new + edits)

- **New `src/skinny/shaders/sampling/proposal.slang`**: `ProposalContext`,
  `ProposalSample`, `IProposal`, `MixtureProposalResult`, and the generic
  `sampleMixture<TM:IMaterial>` / `mixturePdf<TM:IMaterial>`. Reads
  `fc.proposalMask` / `fc.proposalAlpha`. Tag-switch over the ≤3 proposal kinds
  (no existential), same pattern as `evaluateBounce`.
- **New `src/skinny/shaders/sampling/proposals/bsdf.slang`,
  `env_importance.slang`** — the two concrete `IProposal`s. `env_importance`
  imports `environment.slang` and reuses `sampleEnvDir` / `envPdf`.
- **`integrators/path.slang::evaluateBounce`** (FLAT + PYTHON cases, ~L45–97):
  replace the direct `mat.sample(wo, rng)` with the `sampleMixture` block above.
  Build `ProposalContext` from the already-available `wo/N/T/B/h`.
- **`wavefront/wf_shade_common.slang` + `wavefront/wavefront_path.slang`**: the
  per-material shade kernels build the same `ProposalContext` and call
  `sampleMixture`; `wfFinishShade` already consumes `br.bsdfSample.{wi,weight,pdf}`
  unchanged.
- **`nee.slang::allLightsNEE` / `neeLightEstimator`**: thread a mixture-pdf
  callback so the NEE MIS companion weight uses `mixturePdf` (rule §2). Signature
  gains the `ProposalContext` (+ the generic material it already has). Env-NEE
  branch (L92–106) likewise switches its companion pdf.

### 2. Slang — reuse seam (new + edits)

- **New `src/skinny/shaders/sampling/reuse.slang`**: a thin interface +
  `identityReuseDirect<TM:IMaterial>(...)` that calls stock `allLightsNEE`, and
  the indirect-spawn passthrough. The integrator routes direct lighting through
  this seam. Baseline = identity ⇒ parity.

### 3. Slang — `FrameConstants` (`common.slang`)

Add, at the documented std140 tail (kept in sync with `pack()`):

```hlsl
uint   proposalMask;      // bit k = proposal k active (bit0 BSDF, bit1 env, …)
uint   reuseMode;         // 0 = identity (ReSTIR later)
float4 proposalAlpha;     // normalized selection weights (one per slot, ≤4), Σ = 1
                          // float4 not float[4]: dodges the std140 16-byte array stride
```

### 4. Host — new `src/skinny/sampling/` package

- `plugin.py` — `SamplingPlugin` ABC (`name`, `attach_point`, `build/destroy/
  resize/reset`, `bindings()`, `passes()`, `write_uniforms(fc)`, `ui_controls()`,
  `cli_token`, `settings_keys()`) + `ProposalPlugin` / `ReusePlugin`.
- `registry.py` — `PROPOSAL_PLUGINS`, `REUSE_PLUGINS` (name → class).
- `proposals.py` — `BsdfProposal`, `EnvImportanceProposal` (no passes/bindings;
  set mask bit + alpha only).
- `reuse.py` — `IdentityReuse` (no-op).

### 5. Host — `renderer.py` wiring

- `self.active_proposals: list[ProposalPlugin]` (default `[BsdfProposal()]`);
  `self.active_reuse: ReusePlugin` (default `IdentityReuse()`).
- `_pack_uniforms` / `FrameConstants` packer: write `proposalMask`,
  `proposalAlpha` (float4), `reuseMode`. **`SkinParameters.pack()`-style std140 rule
  applies — the byte layout must match the Slang struct exactly.**
- Wavefront build (`vk_wavefront.py` pass assembly): active plugins contribute
  `passes()` / `bindings()`. **Reuse switch → pass rebuild** (mirror
  `execution_mode`); proposal toggle → uniform change only, no rebuild.
- `_current_state_hash()`: add `proposalMask`, the alpha tuple, and the reuse
  plugin id ⇒ accumulation auto-resets on any sampling change.

### 6. Host — front-end consistency (all front-ends + debug viewport)

- **CLI** (`cli_common.py`): `--proposals bsdf,env` and `--reuse none`, wired into
  `skinny`, `skinny-gui`, `skinny-web` exactly like `--integrator` /
  `--execution-mode` / `--bdpt-walk`.
- **UI**: proposal checkboxes + alpha + reuse selector in `app.py` (ImGui),
  `web_app.py`, `debug_viewport.py`.
- **Settings** (`settings.py`): persist the proposal set + alphas + reuse id in
  `settings.json` (snapshot like the other render-selection fields).

## Unbiasedness / PDF contract

- All proposal pdfs **solid-angle**; mixture `Σ α_k pdf_k`; estimator
  `response / mixPdf`; αₖ normalized to Σ = 1.
- Delta lobe ⇒ skip mixture, material's exact weight.
- NEE companion pdf == bounce mixture pdf (§2).
- **Neural-forward reservation (no impl now):** `ProposalSample.version` is
  carried (baseline 0); the contract documents *"a sample's pdf must come from
  the network version that drew it."* Lets the neural spec drop in without an
  interface break.
- Debug build `-D SKINNY_DEBUG_SAMPLING`: assert pdf finite & > 0 for non-delta;
  a unit test integrates each proposal's pdf over the hemisphere ≈ 1.

## Testing

`tests/test_headless.py` harness (repo-root 3.13 venv + `VULKAN_SDK` /
`DYLD_LIBRARY_PATH`, per CLAUDE.md).

1. **Parity** — baseline (`{bsdf}`, `reuse=none`) pixel-identical to current
   `main` on a fixed scene/seed/frame-count, **both** megakernel and wavefront.
2. **Mixture-PDF sanity** — `{bsdf, bsdf}` α=.5/.5 equals the `{bsdf}` image
   (proves the one-sample-MIS / mixture-pdf plumbing, not a dormant path).
3. **Second-proposal correctness** — `{bsdf, env}` on an IBL scene: unbiased
   (converges to the same reference as bsdf-only at high spp) **and** lower
   variance at low spp on a glossy/diffuse case.
4. **NEE-coupling regression** — toggling the env proposal must not bias NEE
   (guards §2).
5. `ruff check src/`, `slangc` recompile (`main_pass.spv` + wavefront variants),
   `py_compile`.

## Out of scope

ReSTIR reservoirs / RIS / spatio-temporal passes · neural inference / training /
weights / replay buffer / double-buffering / CUDA-Vulkan interop / MPS · guided
NEE · path-suffix (GI) reuse. The seam only has to make these **droppable-in**.

## Follow-up specs

- **ReSTIR DI** — first `ReusePlugin`: reservoir buffers, temporal reprojection,
  spatial reuse passes, RIS shading. Validates the reuse seam.
- **Neural spline-flow guiding** — `ProposalPlugin` with an inference pre-pass +
  double-buffered weights + external trainer + replay buffer (the 7-stage plan in
  `neural_spline_guiding_docs/`). Validates the proposal seam against a learned,
  pass-owning proposal.

## Implementation-planning transition

Per the project workflow, implementation planning goes through an **OpenSpec
change proposal** (`openspec/changes/`) rather than a generic plan doc.
