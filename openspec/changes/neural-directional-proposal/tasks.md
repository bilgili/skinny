## 1. Slang inference + parity

- [x] 1.1 New `shaders/sampling/neural_flow.slang`: port the spline-flow forward (`u → wi`) — coupling layers, MLP, rational-quadratic spline forward, logdet
- [x] 1.2 Port the inverse (`wi → pdf`) — RQ spline inverse (analytic quadratic solve) + solid-angle Jacobian; both forward and inverse share the weight buffers
- [x] 1.3 Define the flat weight layout (`weights[]`, `biases[]`, `LayerHeader[]`) and a `spline_flow` exporter that bakes a trained net to that file format
- [ ] 1.4 Slang↔PyTorch pdf-parity test: identical (condition, base sample) → direction + solid-angle pdf agree within tolerance (the 1a gate, before any render wiring)

## 2. Bindings + GPU state

- [ ] 2.1 Declare bindings 25/26/27 (weights / biases / layer-headers) in `bindings.slang`; update the `Architecture.md` binding map
- [ ] 2.2 Host weight buffers + upload helper; load the baked weights file
- [ ] 2.3 Add the `networkVersion` field (baseline 0) to the proposal sample type in `common.slang` / `proposal.slang`

## 3. Seam wiring (proposal bit2)

- [ ] 3.1 `PROPOSAL_NEURAL = 0x4` in `proposal.slang`; extend `sampleBounceDirection` + `mixtureProposalPdf` with the neural term (weight `proposalAlpha.z`), reading the precomputed per-lane `(wi, pdf)`
- [ ] 3.2 `NeuralProposalPlugin(ProposalPlugin)` (`mask_bit=0x4`, `default_weight`) with `build/destroy/resize` owning the weight buffers + binding and loading weights; register in `proposals.py` / registry
- [ ] 3.3 `--proposals bsdf,neural` CLI + data-driven GUI `_disc` selector + settings persistence; reset accumulation on change

## 4. Wavefront neural pass (1a plumbing)

- [ ] 4.1 `WavefrontNeuralProposalPass`: consume per-lane `HitInfo[]` → build condition `c` → Slang inference → write per-lane `(wi, pdf)` buffer
- [ ] 4.2 Renderer lazy build/destroy of the pass (mirror `RestirDiPass`); wire into the wavefront gate between shade and bounce; reject neural on the megakernel backend
- [ ] 4.3 Recompile `main_pass.spv`; bring up with a DUMMY net (cosine-equivalent weights) → prove `{bsdf, neural}` unbiased and default `{bsdf}` bit-identical (the 1a milestone)

## 5. Offline data pipe (1b)

- [ ] 5.1 Record-dump mode: renderer emits per-vertex `(position, wi, contribution)` records to a file
- [ ] 5.2 `spline_flow` trainer reads the records and trains a per-scene flow using the EXACT renderer condition encoding, then bakes the weights via the 1.3 exporter

## 6. Verification + docs (1c + gates)

- [ ] 6.1 Gate: default `{bsdf}` pixel-identical (megakernel + wavefront); extend the sampling-parity goldens
- [ ] 6.2 Gate: `{bsdf, neural}` converges == BDPT (unbiased) on the test scene (headless A/B)
- [ ] 6.3 Gate: equal-time efficiency + firefly tail of `{bsdf, neural}` vs `{bsdf, env}+ReSTIR` on the scene; record the numbers
- [ ] 6.4 Docs: `Architecture.md` (bindings + module map), scene-sampling docs, `README.md` CLI flag, `CHANGELOG.md`
