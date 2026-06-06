## Why

The scene-sampling seam guides the BSDF bounce with analytic proposals (BSDF,
environment) and direct light via NEE / ReSTIR-DI, but **indirect / multi-bounce
incident radiance (depth ≥ 1) is unguided** — BSDF sampling alone is poor where
indirect light is concentrated (caustics, colour-bleed, light through gaps), so those
regions are noisy. A learned proposal that targets the full integrand `f·Li·cos` can
sample those directions. A standalone prototype (`spline_flow`) proved a
position-conditioned neural spline flow beats cosine **2.5–13× equal-time** on
concentrated indirect, robust and unbiased, with a small deployable net. This change
lands that proposal in the renderer (frozen, static-scene) as the foundation for later
online/dynamic guiding.

## What Changes

- A new **`neural`** directional proposal occupying scene-sampling **proposal bit2**
  (`proposalMask`) + `proposalAlpha.z` weight — composes in the existing one-sample-MIS
  mixture. The default `{bsdf}` selection stays **bit-identical**.
- A **wavefront-only `WavefrontNeuralProposalPass`**: consumes per-lane `HitInfo[]`, runs
  Slang spline-flow inference (forward `u→wi` sample + inverse `wi→pdf`), writes per-lane
  `(wi, pdf)` that the bounce stage MIS-mixes. Megakernel keeps the inline `{bsdf,env}`
  subset (neural inline infeasible — MoltenVK big-kernel limit).
- **Frozen offline-trained weights** uploaded to **free descriptor bindings 25/26/27**
  (weights / biases / layer-headers).
- A **`NeuralProposalPlugin(ProposalPlugin)`** that owns the weight buffer + binding via
  the existing `build/destroy/resize` hooks and loads a weights file; selectable via
  `--proposals bsdf,neural`, the data-driven GUI selector, and persisted settings.
- A **path-record dump** mode (renderer emits per-vertex `(x, wi, contribution)` to a
  file) to train a per-scene net offline in `spline_flow`, then bake weights.
- A **`networkVersion`** field (baseline 0) on the proposal sample, so a sample uses the
  pdf of the version that produced it (foundation for online training).
- A **Slang↔PyTorch pdf-parity** test for the flow inference.
- **Not in scope (Stage 2, later change):** online/dynamic training (replay buffer, async
  trainer, double-buffer swap); cross-scene generalisation; megakernel neural; CUDA perf.

## Capabilities

### New Capabilities
- `neural-directional-proposal`: a learned neural spline-flow directional proposal for the
  BSDF bounce that samples toward the incident-light-aware integrand and reports an exact
  solid-angle pdf, plugged into the scene-sampling MIS mixture; frozen offline-trained
  weights, wavefront-only, with an offline training-record dump path and Slang/PyTorch
  pdf parity.

### Modified Capabilities
- `scene-sampling`: the proposal set gains a third (neural) proposal that requires real
  **GPU state** (weight buffers + descriptor bindings) and carries a per-sample
  **`networkVersion`** in the proposal-density contract; the mixture/MIS unbiasedness
  requirement must hold with a learned proposal whose density comes from its stamped
  version, and the wavefront seam must support a proposal pre-pass producing per-lane
  `(wi, pdf)`.

## Impact

- **Code:** `src/skinny/sampling/` (new `NeuralProposalPlugin` + registry/CLI/GUI wiring);
  `src/skinny/renderer.py` (lazy pass build/destroy, weight upload, record-dump);
  `src/skinny/shaders/sampling/proposal.slang` (+ a new flow-inference Slang module);
  `shaders/bindings.slang` / `common.slang` (bindings 25–27, `networkVersion`);
  `vk_compute.py` / `metal_backend.py` (new wavefront pass dispatch).
- **Descriptor bindings:** NEW 25/26/27 (neural weights/biases/layer-headers) → update the
  `Architecture.md` binding map. New per-lane `(wi,pdf)` wavefront buffer.
- **Backends:** wavefront-primary (Vulkan + Metal wavefront); megakernel unchanged (keeps
  `{bsdf,env}`).
- **Docs:** `Architecture.md` (binding + module map), scene-sampling/ReSTIR docs as
  relevant; new CLI flag in `README.md`.
- **Tests:** Slang-vs-PyTorch pdf parity; default-`{bsdf}` bit-identical; neural == BDPT
  (unbiased); equal-time efficiency vs `{bsdf,env}+ReSTIR`.
- **Dependencies:** offline training stays in the standalone `spline_flow` repo (PyTorch);
  no new runtime Python dependency (inference is in-shader Slang).
