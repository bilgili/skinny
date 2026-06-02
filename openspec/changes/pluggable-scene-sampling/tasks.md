> Progress: megakernel proposal seam complete (commits f9e9f12, 344ac4d).
> BSDF + env proposals with one-sample MIS + NEE coupling; baseline bit-identical
> (cross-checkout proof vs main pre-seam, hash c7f77c5a); env unbiased + variance-
> reducing. Remaining: wavefront shade kernels, reuse-seam Slang module, CLI/UI/
> settings selection, state-hash reset, {bsdf,bsdf} sanity test.

## 1. Slang — proposal seam

- [x] 1.1 `shaders/sampling/proposal.slang`: `ProposalContext`, generic `sampleBounceDirection<TM:IMaterial>` (one-sample MIS mixture) + `mixtureProposalPdf<TM:IMaterial>`, reading `fc.proposalMask` + `fc.proposalAlpha`. (Implemented as concrete generic functions over the fixed proposal set rather than an existential `IProposal` + separate `ProposalSample`/`MixtureProposalResult` — folded for the 2-proposal case; split into an interface when a 3rd proposal lands.)
- [x] 1.2 BSDF proposal — delegates to `mat.sample` / `mat.evaluate(...).pdf`; delta lobes (pdf==0) pass through unmixed. (Folded into `proposal.slang`, not a separate `bsdf.slang`.) Mask bit 0.
- [x] 1.3 Env-importance proposal — samples a world dir via the existing env CDFs (`sampleEnvDir`/`envPdf`), world↔tangent via `c.T/B/N`, exact solid-angle pdf, no new bindings. (Folded into `proposal.slang`.) Mask bit 1.

## 2. Slang — reuse seam

- [ ] 2.1 New `shaders/sampling/reuse.slang`: reuse interface + `identityReuseDirect<TM:IMaterial>(...)` calling stock `allLightsNEE`, plus the indirect-spawn passthrough. Baseline = identity. (Not yet — the integrator still calls `allLightsNEE` directly; identity reuse is a no-op so behavior is correct, but the Slang seam module isn't created.)

## 3. Slang — integrator + NEE + FrameConstants edits

- [x] 3.1 `integrators/path.slang::evaluateBounce` (FLAT + PYTHON): build `ProposalContext`, route `mat.sample` through `sampleBounceDirection`. Single-proposal fast path returns the native sample (bit-identical); mixture path sets `bsdfSample.{wi, weight=f·cos/mixPdf, pdf=mixPdf}`.
- [x] 3.2 `nee.slang` (`neeLightEstimator` + env-NEE branch): MIS companion pdf switched to `mixtureProposalPdf(mat, ctx, wi)`. Context built internally from existing args — no signature change, so wavefront callers are untouched.
- [ ] 3.3 `wavefront/wf_shade_common.slang` + `wavefront/wavefront_path.slang`: same `sampleBounceDirection` call in the shade kernels; `wfFinishShade` consumes `bsdfSample.{wi,weight,pdf}` unchanged. **(Pending — wavefront still calls `mat.sample` directly.)**
- [x] 3.4 `common.slang` `FrameConstants`: added `uint proposalMask`, `uint reuseMode`, `float4 proposalAlpha`. Scalar layout (`-fvk-use-scalar-layout`) → tight append, byte-matched to the Python packer.
- [x] 3.5 `main_pass.spv` recompiles from the changed source tree (content-hashed cache key picks up the new `sampling/proposal.slang`). Wavefront variants pending with 3.3.

## 4. Host — sampling package

- [x] 4.1 `src/skinny/sampling/plugin.py`: `SamplingPlugin` ABC + `ProposalPlugin` / `ReusePlugin` (lifecycle hooks; analytic plugins no-op them).
- [x] 4.2 `src/skinny/sampling/{proposals.py,reuse.py}`: `BsdfProposal`, `EnvImportanceProposal`, `IdentityReuse`.
- [x] 4.3 `src/skinny/sampling/registry.py`: `PROPOSAL_PLUGINS` / `REUSE_PLUGINS`, `parse_proposals`/`parse_reuse`, `proposal_mask_and_alpha`.

## 5. Host — renderer wiring

- [ ] 5.1 `renderer.py`: `active_proposals` (default `[BsdfProposal()]`) + `active_reuse` (default `IdentityReuse()`) **done**; CLI constructor args **pending** (6.1).
- [x] 5.2 Uniform packer writes `proposalMask`, `reuseMode`, `proposalAlpha` (float4, Σ=1) — exact scalar-layout byte order.
- [ ] 5.3 `_current_state_hash()`: include `proposalMask`, the alpha tuple, and the reuse-plugin id so accumulation resets on any sampling change. **(Pending — needed once selection is user-changeable.)**
- [ ] 5.4 Wavefront build (`vk_wavefront.py`): active plugins contribute `passes()`/`bindings()`; reuse-mode switch triggers a pass rebuild. **(Pending — no reuse passes yet.)**

## 6. Host — front-end consistency

- [ ] 6.1 `cli_common.py`: `--proposals bsdf,env` + `--reuse none` (+ env fallbacks), wired into `skinny`/`skinny-gui`/`skinny-web`.
- [ ] 6.2 UI selectors in `app.py` (ImGui), `web_app.py`, `debug_viewport.py`.
- [ ] 6.3 `settings.py`: persist proposal set + alphas + reuse id.

## 7. Tests + verification

- [ ] 7.1 Parity: baseline (`{bsdf}`, `none`) pixel-identical to pre-seam. **Megakernel ✓** (cross-checkout hash `c7f77c5a`, `tests/test_sampling_parity.py::test_baseline_parity`). **Wavefront pending** (with 3.3).
- [ ] 7.2 Mixture-pdf sanity: `{bsdf, bsdf}` α=.5/.5 equals `{bsdf}`. **(Pending — `{bsdf,env}` covered instead; the identical-double-proposal sanity is still worth adding.)**
- [x] 7.3 Second-proposal correctness: `{bsdf, env}` unbiased at high spp (image-mean within 3%) + lower RMSE at 16 spp with IBL isolated (`test_env_proposal_unbiased_and_reduces_variance`).
- [x] 7.4 NEE-coupling regression: env unbiasedness test exercises the coupled NEE pdf (energy match ⇒ no NEE bias).
- [ ] 7.5 `ruff check src/` + full `pytest` + `slangc` recompile of wavefront variants + `py_compile`. **(Partial — ruff clean on touched files, megakernel sampling tests green; full-suite + wavefront recompile pending.)**
