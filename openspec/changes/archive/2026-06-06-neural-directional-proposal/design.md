## Context

The scene-sampling seam (shipped) exposes a runtime directional-proposal mixture
(`proposalMask` / `proposalAlpha`, one-sample MIS) plus a reuse seam. Proposals so far
are analytic — BSDF (bit0), environment (bit1) — and own no GPU state. ReSTIR DI is the
precedent for a wavefront-only, renderer-built, GPU-stateful sampling mode. A standalone
prototype (`spline_flow`) validated neural path-guiding through Stage 0/0b/0c: a
position-conditioned rational-quadratic neural spline flow with exact solid-angle pdf
beats cosine 2.5–13× equal-time on concentrated indirect, stays unbiased and
firefly-free, and a small (real-time-sized) net suffices. This change integrates that
flow as proposal **bit2**, frozen and static-scene, in the renderer.

## Goals / Non-Goals

**Goals:**
- Neural proposal bit2 in the seam; default `{bsdf}` stays bit-identical.
- Wavefront-only `WavefrontNeuralProposalPass`; unbiased; equal-time win on one static scene.
- Offline pipeline: renderer record-dump → train in `spline_flow` → bake weights → load.
- Slang inference (forward + inverse) with PyTorch pdf parity.
- `networkVersion` field (baseline 0) as the online-training foundation.

**Non-Goals:**
- Online / dynamic training (replay, async trainer, double-buffer swap) — Stage 2.
- Cross-scene generalisation; megakernel neural; CUDA performance tuning; hashgrid
  spatial encoding (raw position suffices per Stage 0c).

## Decisions

- **Wavefront pre-pass, not inline in the bounce kernel.** Keeps the MLP out of the
  bounce kernel (MoltenVK big-kernel limit kills inline on Metal), amortises inference
  over live lanes, mirrors `RestirDiPass`. → neural is wavefront-only; megakernel keeps
  the inline `{bsdf,env}` subset. *Alt:* inline — rejected (kernel size, no amortisation).
- **Slang in-shader inference, not host.** Per-bounce per-lane inference can't round-trip
  to the host. Port forward + inverse + logdet to Slang over flat weight buffers. Cost is
  the CUDA-stage concern; correctness first.
- **Weights on free descriptor bindings 25/26/27** (weights / biases / LayerHeader, flat
  layout). Frozen, uploaded once on plugin `build`.
- **Condition = (position normalised to scene bbox, shading normal, wo); raw concat → MLP,
  no hashgrid.** Stage 0c showed capacity is not the bottleneck to ~6 lights/3D. The
  **offline trainer MUST use the identical encoding** — mismatch raises variance silently.
- **Frozen, offline-trained, per scene.** De-risks inference independent of a moving
  target; the record-dump is reused verbatim by Stage 2; the offline net is the online
  warm start. Online is a separate later change.
- **De-risk order 1a → 1b → 1c.** 1a = plumbing with a dummy net (prove unbiased + default
  bit-identical), isolating the Slang-parity risk from training quality; 1b = record-dump
  + offline train; 1c = scene-trained net beats `{bsdf,env}+ReSTIR` equal-time.
- **`networkVersion` baseline 0 now.** The scene-sampling pdf contract already reserves
  the field; populating it avoids interface churn at Stage 2.
- **Plugin owns GPU state via the existing `build/destroy/resize` hooks.**
  `NeuralProposalPlugin(ProposalPlugin)`; the data-driven GUI selector gives all
  front-ends + persisted settings for free.

## Risks / Trade-offs

- **Slang↔PyTorch pdf parity** (spline inverse + logdet) → run a fixed-input parity test
  (1a) before any render wiring; bit-close tolerance.
- **Inference cost** (~6 ms/est on MPS) may not be equal-time net-positive on Mac → the
  correctness gate is convergence, not speed (perf is the CUDA stage); mixture-MIS keeps
  it unbiased regardless of cost.
- **Condition-encoding mismatch** offline↔renderer → silent variance (not bias) → single
  shared encoding definition + a check feeding identical `c` to both.
- **Per-scene weights only** (no generalisation) → acceptable at Stage 1; one weights file
  per scene.
- **Hard shadow edges undercovered** (Stage 0b) → the mixture-MIS `α_b·bsdfPdf` term
  bounds the variance; documented, not a blocker.

## Migration Plan

Additive and gated: the default proposal set is unchanged → output bit-identical; neural
is opt-in via `--proposals` / GUI. Bindings 25–27 are allocated only when neural is
active. Rollback = deselect neural (frees buffers) or revert the plugin. Shader change
requires recompiling `main_pass.spv` with `slangc`.

## Open Questions

- Weights-file format + versioning (flat binary matching the `LayerHeader` layout vs npz)
  — lean flat binary.
- Scene-bbox source for position normalisation (USD stage bounds vs renderer scene bounds)
  — must equal the offline trainer's.
- Whether the per-lane `(wi, pdf)` lives in a new wavefront descriptor-set slot or reuses
  an existing scratch buffer.
- 1b record-dump format shared verbatim with the Stage 2 trainer (intended yes).
