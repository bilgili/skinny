> Progress: PROPOSAL SEAM COMPLETE — both backends.
> BSDF + env proposals, one-sample MIS, NEE coupling. Baseline bit-identical in
> megakernel (c7f77c5a) AND wavefront (803d9f9c) — cross-checkout proofs vs main
> pre-seam. Env unbiased + variance-reducing. `--proposals`/`--reuse` CLI +
> data-driven GUI/web/debug selectors + settings + accumulation-reset all wired.
> Remaining is REUSE-seam only, and gated on ReSTIR (identity reuse = current
> behavior, needs no code): 2.1 reuse.slang module, 5.4 wavefront reuse passes.
> Those land with the ReSTIR follow-up change, not here.

## 1. Slang — proposal seam

- [x] 1.1 `shaders/sampling/proposal.slang`: `ProposalContext`, generic `sampleBounceDirection<TM:IMaterial>` (one-sample MIS mixture) + `mixtureProposalPdf<TM:IMaterial>`, reading `fc.proposalMask` + `fc.proposalAlpha`. (Implemented as concrete generic functions over the fixed proposal set rather than an existential `IProposal` + separate `ProposalSample`/`MixtureProposalResult` — folded for the 2-proposal case; split into an interface when a 3rd proposal lands.)
- [x] 1.2 BSDF proposal — delegates to `mat.sample` / `mat.evaluate(...).pdf`; delta lobes (pdf==0) pass through unmixed. (Folded into `proposal.slang`, not a separate `bsdf.slang`.) Mask bit 0.
- [x] 1.3 Env-importance proposal — samples a world dir via the existing env CDFs (`sampleEnvDir`/`envPdf`), world↔tangent via `c.T/B/N`, exact solid-angle pdf, no new bindings. (Folded into `proposal.slang`.) Mask bit 1.

## 2. Slang — reuse seam

- [~] 2.1 `shaders/sampling/reuse.slang` (reuse interface + `identityReuseDirect`). **Deferred to the ReSTIR change.** Identity reuse IS the current behavior (the integrator calls stock `allLightsNEE` + spawns the indirect ray as before), so wrapping it in a Slang seam module now would be a no-op abstraction with no consumer. The host socket (`ReusePlugin`, `reuseMode` uniform, `IdentityReuse`, state-hash) is reserved; the Slang module lands when ReSTIR gives it a second implementation to switch against.

## 3. Slang — integrator + NEE + FrameConstants edits

- [x] 3.1 `integrators/path.slang::evaluateBounce` (FLAT + PYTHON): build `ProposalContext`, route `mat.sample` through `sampleBounceDirection`. Single-proposal fast path returns the native sample (bit-identical); mixture path sets `bsdfSample.{wi, weight=f·cos/mixPdf, pdf=mixPdf}`.
- [x] 3.2 `nee.slang` (`neeLightEstimator` + env-NEE branch): MIS companion pdf switched to `mixtureProposalPdf(mat, ctx, wi)`. Context built internally from existing args — no signature change, so wavefront callers are untouched.
- [x] 3.3 Wavefront shade kernels routed through `sampleBounceDirection`. The catch-all `wfPathShade` already calls the seam-routed `evaluateBounce`; the flat fast-path `wavefront/flat_bounce.slang::evaluateFlatBounce` now does too. (BDPT subpath `mat.sample` calls are out of scope — proposals are a path-tracer feature.)
- [x] 3.4 `common.slang` `FrameConstants`: added `uint proposalMask`, `uint reuseMode`, `float4 proposalAlpha`. Scalar layout (`-fvk-use-scalar-layout`) → tight append, byte-matched to the Python packer.
- [x] 3.5 `main_pass.spv` + the wavefront `_wfpath_shade*.spv` variants recompile from the content-hashed source tree. `flat_bounce.slang`'s new `import sampling.proposal` (→ `environment`) compiles within the MoltenVK big-kernel limit (verified by a wavefront render).

## 4. Host — sampling package

- [x] 4.1 `src/skinny/sampling/plugin.py`: `SamplingPlugin` ABC + `ProposalPlugin` / `ReusePlugin` (lifecycle hooks; analytic plugins no-op them).
- [x] 4.2 `src/skinny/sampling/{proposals.py,reuse.py}`: `BsdfProposal`, `EnvImportanceProposal`, `IdentityReuse`.
- [x] 4.3 `src/skinny/sampling/registry.py`: `PROPOSAL_PLUGINS` / `REUSE_PLUGINS`, `parse_proposals`/`parse_reuse`, `proposal_mask_and_alpha`.

## 5. Host — renderer wiring

- [x] 5.1 `renderer.py`: proposal mixture + reuse modeled as discrete presets (`proposal_preset_index`/`reuse_index` + `*_modes`), resolved to plugin instances by `_active_proposals()`/`_active_reuse()`. CLI override applied post-construction like the integrator (`proposal_preset_from_token`).
- [x] 5.2 Uniform packer writes `proposalMask`, `reuseMode`, `proposalAlpha` (float4, Σ=1) — exact scalar-layout byte order.
- [x] 5.3 `_current_state_hash()`: includes `proposal_preset_index` + `reuse_index` so accumulation resets on any sampling change.
- [~] 5.4 Wavefront reuse-pass assembly (`vk_wavefront.py`). **Deferred to the ReSTIR change** — identity reuse owns no passes/bindings, so there is nothing to assemble until ReSTIR adds reservoir passes. The `ProposalPlugin`/`ReusePlugin.passes()`/`bindings()` hooks exist for it to fill.

## 6. Host — front-end consistency

- [x] 6.1 `cli_common.py`: `--proposals {bsdf,bsdf,env,env}` + `--reuse none` (+ `SKINNY_PROPOSALS`/`SKINNY_REUSE` env), applied in `app.py`, `web_app.py`, `headless.py` (skinny-render) like `--integrator`.
- [x] 6.2 UI selectors: added `_disc("Proposals", …)` + `_disc("Reuse", …)` to `STATIC_PARAMS` — the data-driven UI surfaces them in ImGui (`app.py`), web (`web_app.py`), and the debug viewport for free.
- [x] 6.3 Settings: the new `_disc` params are in the `STATIC_PARAMS` snapshot, so `settings.json` persists/restores `proposal_preset_index` + `reuse_index` automatically.

## 7. Tests + verification

- [x] 7.1 Parity: baseline (`{bsdf}`, `none`) pixel-identical to pre-seam in **both backends** — `test_baseline_parity[megakernel]` (golden `c7f77c5a`) + `[wavefront]` (golden `803d9f9c`), each proven by a cross-checkout render of main pre-seam vs the seam worktree.
- [~] 7.2 Mixture-pdf sanity: the `proposalMask` bitset makes a literal `{bsdf,bsdf}` degenerate (same bit ⇒ collapses to `{bsdf}`), so the identical-double-proposal form is N/A. The mixture-pdf path (`α_b·bsdfPdf + α_e·envPdf`) is instead exercised + proven correct by the `{bsdf,env}` unbiasedness test (7.3).
- [x] 7.3 Second-proposal correctness: `{bsdf, env}` unbiased at high spp (image-mean within 3%) + lower RMSE at 16 spp with IBL isolated (`test_env_proposal_unbiased_and_reduces_variance`).
- [x] 7.4 NEE-coupling regression: env unbiasedness test exercises the coupled NEE pdf (energy match ⇒ no NEE bias).
- [x] 7.5 Verification: ruff clean on all touched files (0 new errors vs main baseline); `py_compile` OK; `slangc` recompiles `main_pass` + wavefront `_wfpath_shade*` variants; sampling parity suite green both backends; wavefront A/B + render regression (`test_wavefront_path_ab`, `test_wavefront_render`) 11 passed; `ui_spec`/`cli_common` green. (Pre-existing `test_web` GpuInfo/session failures are environment-only — identical on main.)
