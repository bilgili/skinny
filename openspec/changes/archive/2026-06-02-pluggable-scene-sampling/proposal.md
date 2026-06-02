## Why

We want to experiment with whole-scene sampling algorithms — **ReSTIR**
(spatiotemporal reservoir reuse) and a **neural spline-flow guided sampler**
(learned directional proposal, see `neural_spline_guiding_docs/`). Neither fits
the renderer today:

- Importance sampling lives **inside materials** (`IMaterial.sample` picks its
  own BSDF lobe and returns `weight = f·cos/bsdfPdf`), plus NEE
  (`nee.slang::allLightsNEE`) and the env-importance CDFs (bindings 31–32). MIS
  is power-heuristic, hard-wired to the BSDF pdf.
- The `ISampler` seam is low-level (tangent-space BSDF lobes). There is no seam
  at the level ReSTIR / neural guiding operate: *which direction or light to
  pick at each path vertex*, via reuse (ReSTIR) or a learned proposal MIS-mixed
  at the bounce (neural).

This change adds **only the seam those algorithms plug into**, plus a baseline
refactor of today's behaviour and one tiny second proposal to prove the mixture
path end-to-end. ReSTIR and the neural sampler are each their own follow-up
change. Full design: `docs/superpowers/specs/2026-06-02-pluggable-scene-sampling-design.md`.

## What Changes

- **Two orthogonal, composable seams.** A *directional-proposal* hook at the
  BSDF bounce, and a *reuse/resampling* hook around NEE + the indirect spawn.
  One Python `SamplingPlugin` host abstraction (`ProposalPlugin` / `ReusePlugin`)
  owns lifecycle + optional GPU passes/buffers/bindings + `FrameConstants`
  uniform bits + UI/CLI/settings wiring.
- **Proposal mixture (runtime-uniform, one-sample MIS).** A small proposal
  registry selected by `fc.proposalMask` + `fc.proposalAlpha`; pick one proposal
  ∝ α per bounce, divide by the full mixture pdf `Σ_k α_k·pdf_k(wi)`. The bounce
  switches from `mat.sample` to `sampleMixture` + `mat.evaluate`. Matches the
  `fc.integratorType` precedent — instant switching, no recompile.
- **Three correctness rules baked into the interface:** delta-lobe pass-through
  (pdf==0 → material's own weight); **NEE-coupling** (NEE's MIS companion pdf
  becomes the same mixture pdf); same-pdf-as-sampled for downstream sphere/env
  MIS.
- **Ships two proposals + identity reuse:** `BsdfProposal` (baseline, proves
  pixel-parity), `EnvImportanceProposal` (second proposal, reuses the existing
  env CDFs — no new GPU state), `IdentityReuse` (forwards to stock NEE).
- **Reuse seam = interface + identity baseline only.** The `ReusePlugin` socket
  is shaped so ReSTIR can later inject reservoir passes/buffers, but no
  reservoirs are built here. Reuse selection is pass-structural (rebuilt on
  switch, like `--execution-mode`).
- **Selection across all front-ends.** `--proposals bsdf,env` / `--reuse none`
  (+ env fallbacks) mirroring `--integrator`/`--execution-mode`/`--bdpt-walk`,
  wired into `skinny`/`skinny-gui`/`skinny-web`, the ImGui + web UI + debug
  viewport, and `settings.json`. Any change resets progressive accumulation.

## Capabilities

### Added Capabilities

- `scene-sampling`: a pluggable, composable sampling-strategy layer — a
  directional-proposal mixture at the bounce and a reuse hook around direct +
  indirect lighting — with a baseline that preserves current output exactly, a
  second (environment-importance) proposal, and command-line/GUI/persisted
  selection. Shaped so ReSTIR and neural guiding drop in as later changes
  without an interface break.

## Impact

- **Shaders (new):** `shaders/sampling/proposal.slang` (`IProposal`,
  `ProposalContext`/`ProposalSample`, generic `sampleMixture`/`mixturePdf`),
  `shaders/sampling/proposals/{bsdf,env_importance}.slang`,
  `shaders/sampling/reuse.slang` (interface + identity).
- **Shaders (edit):** `integrators/path.slang::evaluateBounce` (FLAT/PYTHON
  cases: `mat.sample` → `sampleMixture` + `mat.evaluate`); `nee.slang`
  (companion pdf via `mixturePdf`); `wavefront/wf_shade_common.slang` +
  `wavefront/wavefront_path.slang` (same proposal call in the shade kernels);
  `common.slang` `FrameConstants` (+`proposalMask`, `reuseMode`, `proposalAlpha`
  float4). Recompile `main_pass.spv` + wavefront variants with `slangc`.
- **Host (new):** `src/skinny/sampling/` (`plugin.py`, `registry.py`,
  `proposals.py`, `reuse.py`).
- **Host (edit):** `renderer.py` (active plugin lists, uniform packing,
  `_current_state_hash`, wavefront pass/binding assembly, reuse-switch rebuild);
  `cli_common.py` (`--proposals`/`--reuse`); `app.py` + `web_app.py` +
  `debug_viewport.py` (selectors); `params.py`/`ui/*` (UI controls);
  `settings.py` (persistence).
- **Tests:** parity (baseline == current `main`, both backends);
  mixture-pdf sanity (`{bsdf,bsdf}` == `{bsdf}`); env-proposal unbiasedness +
  variance reduction; NEE-coupling regression. `ruff` + `slangc` recompile +
  `py_compile`.
- **Behaviour:** with defaults (`bsdf`, `reuse=none`) output is byte-identical
  to today; new proposals/reuse modes are opt-in.
- **Risk:** the NEE-coupling rule (a non-BSDF proposal MUST feed the same
  mixture pdf into the NEE MIS weight or the estimator biases) and the std140
  layout of the new `FrameConstants` fields are the two correctness hot spots.

## Notes

Authored as the seam-only foundation. ReSTIR DI (first `ReusePlugin`) and neural
spline-flow guiding (a `ProposalPlugin` with an inference pre-pass +
double-buffered weights + external trainer) are separate later changes, each
validating one seam.
