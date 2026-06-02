## 1. Slang — proposal seam

- [ ] 1.1 New `shaders/sampling/proposal.slang`: `ProposalContext`, `ProposalSample` (`wiT`, `pdf`, `delta`, `valid`, `version`), `IProposal { sample; pdf }`, `MixtureProposalResult`, and generic `sampleMixture<TM:IMaterial>` / `mixturePdf<TM:IMaterial>` reading `fc.proposalMask` + `fc.proposalAlpha`. Tag-switch over the proposal kinds (no existential), mirroring `evaluateBounce`.
- [ ] 1.2 New `shaders/sampling/proposals/bsdf.slang`: `BsdfProposal` — `sample` delegates to `mat.sample`, `pdf` to `mat.evaluate(...).pdf`; flags delta lobes (pdf==0). Mask bit 0.
- [ ] 1.3 New `shaders/sampling/proposals/env_importance.slang`: `EnvImportanceProposal` — imports `environment.slang`, samples a world dir via the existing env CDFs and converts world↔tangent with `c.T/B/N`; exact solid-angle `pdf`. Mask bit 1. No new bindings.

## 2. Slang — reuse seam

- [ ] 2.1 New `shaders/sampling/reuse.slang`: reuse interface + `identityReuseDirect<TM:IMaterial>(...)` calling stock `allLightsNEE`, plus the indirect-spawn passthrough. Baseline = identity.

## 3. Slang — integrator + NEE + FrameConstants edits

- [ ] 3.1 `integrators/path.slang::evaluateBounce` (FLAT + PYTHON cases): build `ProposalContext` from `wo/N/T/B/h`; replace `mat.sample(wo,rng)` with `sampleMixture` → on non-delta, `mat.evaluate(wo, wiT)` and set `bsdfSample.{wi, weight=response/mixPdf, pdf=mixPdf}`; on delta, keep the material's own sample/weight.
- [ ] 3.2 `nee.slang` (`neeLightEstimator` + `allLightsNEE`, incl. the env-NEE branch): switch the BSDF-technique MIS companion pdf to `mixturePdf(mat, ctx, wiToLight)`. Thread the `ProposalContext` through.
- [ ] 3.3 `wavefront/wf_shade_common.slang` + `wavefront/wavefront_path.slang`: same `sampleMixture` call in the shade kernels; `wfFinishShade` consumes `bsdfSample.{wi,weight,pdf}` unchanged.
- [ ] 3.4 `common.slang` `FrameConstants`: add `uint proposalMask`, `uint reuseMode`, `float4 proposalAlpha` at the documented std140 tail; keep byte layout in lockstep with the Python packer.
- [ ] 3.5 Recompile `main_pass.spv` and the affected wavefront `.spv` variants with `slangc`.

## 4. Host — sampling package

- [ ] 4.1 New `src/skinny/sampling/plugin.py`: `SamplingPlugin` ABC (`name`, `attach_point`, `build/destroy/resize/reset`, `bindings()`, `passes()`, `write_uniforms(fc)`, `ui_controls()`, `cli_token`, `settings_keys()`) + `ProposalPlugin` / `ReusePlugin`.
- [ ] 4.2 New `src/skinny/sampling/{proposals.py,reuse.py}`: `BsdfProposal`, `EnvImportanceProposal` (set mask bit + alpha only), `IdentityReuse` (no-op).
- [ ] 4.3 New `src/skinny/sampling/registry.py`: `PROPOSAL_PLUGINS`, `REUSE_PLUGINS` (name → class) + a parser for the CLI token lists.

## 5. Host — renderer wiring

- [ ] 5.1 `renderer.py`: `active_proposals: list[ProposalPlugin]` (default `[BsdfProposal()]`), `active_reuse: ReusePlugin` (default `IdentityReuse()`); constructor args from the CLI.
- [ ] 5.2 Uniform packer: write `proposalMask`, `proposalAlpha` (float4, Σ=1), `reuseMode` into the `FrameConstants` UBO with the exact std140 byte layout.
- [ ] 5.3 `_current_state_hash()`: include `proposalMask`, the alpha tuple, and the reuse-plugin id so accumulation resets on any sampling change.
- [ ] 5.4 Wavefront build (`vk_wavefront.py`): active plugins contribute `passes()`/`bindings()`; a reuse-mode switch triggers a pass rebuild (mirror `execution_mode`); a proposal toggle is a uniform-only change.

## 6. Host — front-end consistency

- [ ] 6.1 `cli_common.py`: `--proposals bsdf,env` + `--reuse none` (+ env fallbacks), wired into `skinny`, `skinny-gui`, `skinny-web` like `--integrator`/`--execution-mode`/`--bdpt-walk`.
- [ ] 6.2 UI selectors: proposal checkboxes + alpha + reuse selector in `app.py` (ImGui), `web_app.py`, and `debug_viewport.py`.
- [ ] 6.3 `settings.py`: persist the proposal set + alphas + reuse id in `settings.json`.

## 7. Tests + verification

- [ ] 7.1 Parity: baseline (`{bsdf}`, `reuse=none`) pixel-identical to current `main` on a fixed scene/seed/frames, both megakernel and wavefront (`tests/test_headless.py` harness).
- [ ] 7.2 Mixture-pdf sanity: `{bsdf, bsdf}` α=.5/.5 equals the `{bsdf}` image (proves the one-sample-MIS / mixture-pdf plumbing).
- [ ] 7.3 Second-proposal correctness: `{bsdf, env}` on an IBL scene — unbiased at high spp vs the bsdf-only reference, lower variance at low spp.
- [ ] 7.4 NEE-coupling regression: toggling the env proposal does not bias NEE.
- [ ] 7.5 `ruff check src/` + `pytest`; `slangc` recompile of `main_pass.spv` + wavefront variants; `py_compile` the new modules.
