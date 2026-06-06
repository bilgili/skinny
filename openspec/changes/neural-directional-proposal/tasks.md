## 1. Slang inference + parity

- [x] 1.1 New `shaders/sampling/neural_flow.slang`: port the spline-flow forward (`u → wi`) — coupling layers, MLP, rational-quadratic spline forward, logdet
- [x] 1.2 Port the inverse (`wi → pdf`) — RQ spline inverse (analytic quadratic solve) + solid-angle Jacobian; both forward and inverse share the weight buffers
- [x] 1.3 Define the flat weight layout (`weights[]`, `biases[]`, `LayerHeader[]`) and a `spline_flow` exporter that bakes a trained net to that file format
- [x] 1.4 Slang↔PyTorch pdf-parity test: identical (condition, base sample) → direction + solid-angle pdf agree within tolerance (the 1a gate, before any render wiring). `tests/test_neural_parity.py` re-implements `neural_flow.slang` in numpy off the same flat weight layout and locks it against committed PyTorch goldens (`tests/data/neural_parity/`, baked by `generate_goldens.py`); CI-runnable with no torch/GPU. Forward `max |Δwi|=4.3e-6`, `rel Δpdf=2.5e-5`; inverse `rel Δpdf=5.2e-5`; fwd→inv round-trip `2e-14` (bars `1e-4` / `1e-3`). Optional GPU-runtime check (`sampleNeural`/`pdfNeural` over slangpy) skips cleanly where the typed `StructuredBuffer<NfLayerHeader>` cannot bind — GPU parity stays validated via the headless bring-up (4.3).

## 2. Bindings + GPU state

- [x] 2.1 Declare bindings 33/34/35 (weights / biases / layer-headers) in `sampling/neural_proposal.slang` + the hand-declared scene-set layout (`vk_compute._create_descriptor_set_layout`); the scope doc's 25/26/27 collide with `GRAPH_BINDING_BASE`, so 33/34/35 sit above the graph range + env CDFs. (`Architecture.md` map: in 6.4.)
- [x] 2.2 Host weight buffers + upload helper (`sampling/neural_weights.py` NFW1 loader + dummy baker; renderer `_sync_neural_weights`); load the baked weights file
- [x] 2.3 Add the `networkVersion` field (baseline 0) to the proposal sample (`WfNeuralSample`/`ProposalContext` + `FrameConstants.neuralNetworkVersion`) in `common.slang` / `interfaces.slang` / `proposal.slang`

## 3. Seam wiring (proposal bit2)

- [x] 3.1 `PROPOSAL_NEURAL = 0x4` in `proposal.slang`; extended `sampleBounceDirection` (3-way one-sample MIS, precomputed neural candidate) + `mixtureProposalPdf` (inline inverse) via `proposalWeights` (per-lane renormalisation, gated by `neuralValid`); `neuralActive` threaded through `nee`/`reuse` for unbiased NEE coupling
- [x] 3.2 `NeuralProposal(ProposalPlugin)` (`mask_bit=0x4`, `default_weight`) registered in `proposals.py`/`registry`. GPU state is renderer-owned (mirrors `RestirDiReuse` → `RestirDiPass`): weight buffers + `WavefrontNeuralProposalPass`, not the throwaway plugin instances
- [x] 3.3 `--proposals bsdf,neural` CLI + `BSDF + Neural` preset (data-driven `_disc` selector + settings persistence via `proposal_preset_index`, auto-surfaced); accumulation resets on proposal change (existing state-hash)

## 4. Wavefront neural pass (1a plumbing)

- [x] 4.1 `WavefrontNeuralProposalPass` (`wavefront/neural_proposal_pass.slang::wfNeuralProposal`): consumes per-lane `HitInfo[]` + state → builds condition `c` → forward Slang inference → writes per-lane `(wi, pdf)` (set-1 binding 8, owned by `WavefrontPathPass`)
- [x] 4.2 Renderer lazy build/destroy (`_ensure`/`_destroy_wavefront_path_pass`, `set_neural`, rebuild key includes `_neural_active()`); dispatched every bounce between scatter and shade; megakernel rejects neural (`_pack_uniforms` strips bit2 + warns; `_neural_active` gates on wavefront)
- [x] 4.3 `main_pass.spv` recompiled (headless run regenerates it; UBO=508). DUMMY-net bring-up PROVEN on a loaded Cornell box (`tests/test_neural_headless.py`): the neural pre-pass builds + hooks + renders nonzero; `{bsdf,neural}` converges to the `{bsdf}` reference (unbiased); clean teardown (0 validation errors after adding the weight-buffer cleanup)

## 5. Offline data pipe (1b)

- [ ] 5.1 Record-dump mode: renderer emits per-vertex `(position, wi, contribution)` records to a file
- [ ] 5.2 `spline_flow` trainer reads the records and trains a per-scene flow using the EXACT renderer condition encoding, then bakes the weights via the 1.3 exporter

## 6. Verification + docs (1c + gates)

- [x] 6.1 Gate: default `{bsdf}` megakernel ≡ wavefront on the Cornell box (`test_default_bsdf_megakernel_wavefront_parity`, rel-mean-diff = 0.0000) — the seam changes regressed neither backend ({bsdf} fast path untouched). `tests/test_neural_headless.py` is the loaded-scene parity harness.
- [x] 6.2 Gate: `{bsdf, neural}` (dummy net) converges to the `{bsdf}` reference (`test_neural_unbiased_matches_bsdf`, rel-mean-diff = 0.0020 < 0.05) — the mixture-MIS estimator is UNBIASED. (== BDPT follows from `{bsdf}` ≡ BDPT, already maintained in the repo.)
- [ ] 6.3 Gate: equal-time efficiency + firefly tail of `{bsdf, neural}` vs `{bsdf, env}+ReSTIR` — GATED on a TRAINED net (5.2); the dummy net adds MLP cost with no variance reduction, so equal-time is only meaningful once a scene-trained net is baked. Harness extends `tests/test_neural_headless.py`.
- [x] 6.4 Docs: `Architecture.md` (binding map 33/34/35 + set-1 binding 8 + module tree + UBO 508), `Wavefront.md` (neural-proposal seam section), `README.md` (`--proposals bsdf,neural`), `CHANGELOG.md`, `PythonAPI.md` (`NeuralProposal` export)

> **Remaining (need a loaded-scene headless GPU run / external training; not landed this pass):**
> - 4.3 functional prove, 6.1/6.2/6.3 gates: the wavefront path pass + neural pre-pass only build once scene bindings exist (a real USD scene), so the dummy-net unbiased/bit-identical + ==BDPT + equal-time A/Bs need a headless USD render harness (pump `update()` until instances stream in), not the empty-scene smoke. Plumbing is proven (UBO 508, both backends construct + render with no validation errors, bindings 33/34/35 + set-1 8 bound, neural pass build/dispatch wired).
> - 5.1 record-dump + 5.2 `spline_flow` training: the offline data pipe (1b). Condition encoding to mirror is canonical in `sampling/neural_proposal.slang::neuralCondition` (pos→[-1,1]³ via scene AABB, N, wo).
