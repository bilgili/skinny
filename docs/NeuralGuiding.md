# Skinny — Neural Directional Proposal (SplineFlow)

This document is the implementation reference for **SplineFlow**, skinny's
learned **neural directional proposal** for path guiding. It samples a bounce
direction from a position- and material-conditioned **rational-quadratic neural
spline flow** (RQ-NSF) with an *exact* solid-angle pdf, and composes it
unbiasedly with the analytic BSDF/environment proposals through the
scene-sampling seam's one-sample MIS mixture. It covers the rendering stages, the
governing equations and the exact shader symbols that realize them, the network
architecture, the size×precision study, the design choices, the controls, and the
source papers.

> Equations are shipped as **SVG images** (the repo's GitLab does not render
> KaTeX/`$$` math reliably). The LaTeX sources live in
> `docs/diagrams/neural/equations.json`; regenerate the SVGs with
> `../restir/render.cjs` (MathJax 3, publication quality — needs Node +
> `mathjax-full`) or the dependency-free `gen_svg_equations.cjs` fallback
> (`node docs/diagrams/neural/gen_svg_equations.cjs`). Inline symbols (ω, q_ω,
> α, β, Σ) are plain Unicode.

SplineFlow plugs into the **scene-sampling proposal seam** (the `ProposalPlugin`
socket reserved by the sampling change), the sibling of the `ReusePlugin` socket
ReSTIR rides. The seam and the wavefront execution backend it runs on are
documented in [Architecture.md](Architecture.md) (descriptor binding map) and
[Wavefront.md](Wavefront.md) (the bounce-stage proposal hook); the generic
path/BDPT integrators live in [README.md](../README.md). The pre-implementation
brainstorm and decision history are archived under
`openspec/changes/archive/2026-06-06-neural-directional-proposal/` (Stage 1) and
`openspec/changes/archive/2026-06-07-neural-precision-size-study/` (Stage 2) —
**this document describes the shipped code**.

## What SplineFlow is

A path tracer's variance is dominated by how well its bounce sampler matches the
true integrand `f · L_i · cosθ`. The material's own BSDF sampler matches `f·cosθ`
but is blind to where light *actually comes from*: in a scene with concentrated
indirect illumination, most BSDF draws point at darkness and the throughput
estimate is noisy.

**Path guiding** learns the missing factor — the incident radiance `L_i` — and
biases the bounce toward it. SplineFlow does this with a **conditional normalizing
flow**: an invertible map `T_θ : [0,1]² → [0,1]²` that warps a uniform base
sample into a hemisphere direction whose density `q_ω` concentrates where
`f · L_i · cosθ` is large. Two properties make it usable in an *unbiased*
renderer:

- **Exact density, both ways.** Because the flow is invertible in closed form,
  the same weights give a forward draw (`u → ω`) *and* the exact inverse density
  (`ω → q_ω`) of any direction — the latter is what the MIS mixture needs to
  weight a direction another proposal drew.
- **Unbiased composition.** SplineFlow is never used alone. It is one technique
  in a one-sample-MIS *mixture* with the always-on BSDF proposal (and optionally
  the environment proposal); the estimator divides by the full mixture pdf, so a
  learned lobe that is wrong on some lanes raises variance but never bias.

![A learned hemisphere lobe conditioned on the shading state steers the bounce toward incident radiance, where the broad BSDF lobe mostly misses](diagrams/neural/fig_path_guiding.svg)

The network is **frozen and trained offline** (per scene, in the standalone
`spline_flow` PyTorch repo) from path records the renderer itself dumps; at render
time skinny only runs GPU *inference*. With the default proposal set `{bsdf}` the
mixture collapses to the material's native sampler and the image is
**byte-identical** to the pre-seam renderer.

### Scope and limits

| Property | Value |
| --- | --- |
| Backend | **Wavefront only.** The megakernel keeps the `{bsdf, env}` subset; selecting neural on the megakernel/Metal is reported unsupported (capability gate). |
| Vertices | **All flat/python bounces** (`depth ≥ 0`), upper hemisphere only — the flow's domain. |
| Materials | **Flat / standard_surface / OpenPBR / python only.** Skin / MaterialX-graph lanes set `neuralValid = false` and keep the `{bsdf, env}` subset (weights renormalise). |
| Training | **Frozen, offline, per scene** in the `spline_flow` repo; the renderer dumps `.nrec` records and loads `.nfw1` weights. No online/adaptive training (a reserved Stage 3). |
| Precision/size | Network size (`NF_LAYERS/BINS/HIDDEN`) and MLP precision (fp32 / fp16-storage / fp16-compute) are **build-time configurable**; the spline core + pdf stay fp32 always. |

## Stages of rendering

SplineFlow runs as a **pre-pass + seam** pair on the wavefront backend, mirroring
how the ReSTIR DI pass slots into the bounce.

![Online per-bounce inference plus the offline record→train→export→load loop](diagrams/neural/neural_pipeline.svg)

**Online (per bounce, the only on-device cost):**

1. **`wfNeuralProposal`** (`shaders/wavefront/neural_proposal_pass.slang`) — the
   forward pre-pass. Scheduled between scatter and the flat shade by
   `WavefrontPathPass.record` (after the intersect fills `npHits[]`). For each
   live lane it builds the condition from the lane's `HitInfo` + ray, draws a
   **decorrelated** base sample `u` from `(pixelIndex, frameIndex·8 + depth)` —
   *not* the shade RNG, so enabling neural never perturbs the `{bsdf}` draw path —
   calls `neuralSampleWorld`, stamps the network version, and writes one
   `WfNeuralSample {wi, pdf, version, valid}` (32 B) to set-1 binding 8.
2. **Proposal seam** (`shaders/sampling/proposal.slang`,
   `sampleBounceDirection`) — the shade kernel's bounce stage. It draws the BSDF
   candidate, then runs one-sample MIS over `{bsdf | env | neural}`: pick a
   technique ∝ α, reuse the pre-pass's neural `wi` when neural is picked, and for
   *any* chosen direction divide throughput by the **full** mixture pdf. The
   arbitrary-direction neural density is the only inline MLP eval (`neuralPdfWorld`
   — the inverse), evaluated when another technique drew the direction; the
   forward candidate's pdf is reused otherwise.

**Offline (per scene, no render-time cost):**

3. **`mainImageRecord`** (`shaders/integrators/path_record.slang`) — a second
   megakernel entry, an RR-free path tracer that traces the same `{bsdf, env}`
   paths and, for every guideable bounce, appends a `PathRecord` to a GPU buffer
   the host reads back into a `.nrec` file. Megakernel because one thread owns the
   whole path, so the tail radiance is known at loop end and attributed back from
   a local register stack.
4. **`spline_flow`** (PyTorch, standalone repo) fits `q_θ` to those records, and
   **`export_weights.py`** writes the frozen `.nfw1` weights that the renderer
   loads (`neural_weights.load_neural_weights`) and uploads to bindings 33/34/35.

### Per-lane / per-record state

```hlsl
// shaders/interfaces.slang — the pre-pass output the seam consumes
struct WfNeuralSample {
    float3 wi;        // world-space neural-drawn direction
    float  pdf;       // solid-angle pdf at wi (forward draw)
    uint   version;   // producing network version (baseline 0)
    uint   valid;     // 1 only on flat/python live lanes with pdf > 0
};

// shaders/integrators/path_record.slang — one offline training sample (64 B)
struct PathRecord {
    float3 pos;       // world hit position (trainer normalises via header AABB)
    float3 normal;    // world shading normal (= neuralCondition N)
    float3 wo;        // world outgoing dir toward the previous vertex
    float3 wiLocal;   // sampled bounce dir in flow-local (x=T, y=N, z=B), y-up
    float3 contrib;   // (L_final − L_k)/β_in,k — the RGB training weight
    uint   depth;     // bounce index (0 = primary hit)
};
```

The condition `(pos, normal, wo)` recorded offline is **byte-for-byte** the input
to the inference-time `neuralCondition`; the scene AABB rides in the `.nrec`
header so the trainer normalises position identically. `wiLocal` is projected onto
the **same** shading frame the wavefront pass builds at inference (`N` from the
hit, `T,B` from the hit tangent or `buildBasis`), so train and infer frames match
exactly — a mismatch raises variance silently rather than biasing.

## Equations

Notation: `c` is the condition; `u ∈ [0,1]²` a uniform base sample; `z` the flow
output on the unit square; `ω` the hemisphere direction; `q_□` the density on the
unit square; `q_ω` the solid-angle density (sr⁻¹); `f` the BSDF response including
the cosine; `L_i` incident radiance; `β` the path throughput. The flow hemisphere
is y-up.

### 1. Condition encoding

The 9-float condition handed to the flow — position normalised to the scene AABB,
the world shading normal, the outgoing world direction:

![Condition c = (2(p − b_min)/e − 1, N, ω_o) ∈ ℝ⁹](diagrams/neural/cond.svg)

### 2. Forward draw

A draw maps a uniform base sample through the conditional flow:

![z = T_θ(u; c), u ~ U([0,1]²)](diagrams/neural/flow-fwd.svg)

`T_θ` is a stack of `NF_LAYERS` coupling layers. Each layer conditions on one of
the two coordinates (plus `c`) and transforms the other with a monotone
rational-quadratic spline; consecutive layers alternate which coordinate is
transformed:

![z'_t = RQS(z_t; θ_L(z_c, c)), z'_c = z_c](diagrams/neural/coupling.svg)

### 3. Rational-quadratic spline

Inside a bin `k` (knot `x_k`, width `w_k`, height `h_k`, boundary derivatives
`d_k, d_{k+1}`), the monotone RQ transform is

![y = y_k + h_k(s θ² + d_k θ(1−θ)) / (s + (d_k + d_{k+1} − 2s) θ(1−θ))](diagrams/neural/rqs.svg)

![θ = (x − x_k)/w_k, s = h_k/w_k](diagrams/neural/rqs-where.svg)

The bin widths/heights come from softmax-normalized MLP outputs (with a `1e-4`
floor) and the derivatives from softplus; the inverse is the analytic solution of
the underlying quadratic. The spline math is evaluated in **fp32 in every
precision mode**.

### 4. Log-det Jacobian

Because each spline is monotone and coordinates alternate, the flow's log
Jacobian determinant is the sum of the per-layer 1-D spline log-derivatives:

![log|det ∂z/∂u| = Σ_L log|∂y_L/∂x_L|](diagrams/neural/logdet.svg)

### 5. Density on the unit square

The base sample is uniform on `[0,1]²` (density 1), so by change of variables the
forward draw's density on the square is the inverse Jacobian:

![q_□(z) = exp(−log|det ∂z/∂u|)](diagrams/neural/pdf-square.svg)

### 6. Solid-angle density

The y-up square→hemisphere map (`φ = 2πu`, `cosθ = v`) has constant Jacobian
`2π`, so the solid-angle density is

![q_ω(ω) = q_□(z) / 2π](diagrams/neural/pdf-omega.svg)

### 7. Inverse density of an arbitrary direction

For a direction the flow did *not* draw (a BSDF- or env-sampled `ω`), the mixture
needs `q_ω(ω)`. Map `ω` back to the square (`z = M⁻¹(ω)`), run the flow in
reverse, and accumulate the inverse log-det:

![q_ω(ω) = (1/2π) exp(log|det ∂u/∂z|), z = M⁻¹(ω)](diagrams/neural/pdf-inv.svg)

### 8. Mixture pdf

The active proposals form a one-sample-MIS mixture with weights that renormalise
to Σ = 1 over the active, valid techniques:

![p_mix(ω) = α_b p_bsdf(ω) + α_e p_env(ω) + α_n q_ω(ω)](diagrams/neural/mix-pdf.svg)

### 9. Unbiased throughput update

Whatever technique is chosen, throughput is divided by the **full** mixture pdf —
this is what keeps the estimator unbiased:

![β ← β · f(ω)cosθ / p_mix(ω)](diagrams/neural/estimator.svg)

### 10. Training weight (offline)

Each recorded vertex stores the per-unit-throughput tail radiance, so a
contribution-weighted fit learns `q ∝ f · L_i · cosθ`:

![w_k = (L_final − L_k) / β_in,k](diagrams/neural/contrib.svg)

### 11. Training objective (offline)

The flow is fit by minimizing the contribution-weighted negative log-likelihood
(forward KL / mass-covering) over the recorded dataset `D`:

![L(θ) = −E_{(c,ω,w)~D}[ w · log q_θ(ω | c) ]](diagrams/neural/loss.svg)

## Equation → implementation map

| Equation | Symbol | File |
| --- | --- | --- |
| Condition encoding (§1) | `neuralCondition` | `sampling/neural_proposal.slang` |
| Forward draw `u → ω` (§2, §6) | `sampleNeural` / `neuralSampleWorld` | `sampling/neural_flow.slang` / `neural_proposal.slang` |
| Coupling-layer stack (§2) | `nf_flow_forward` / `nf_flow_inverse` | `sampling/neural_flow.slang` |
| Conditioner MLP (§3) | `nf_mlp` / `nf_linear` | `sampling/neural_flow.slang` |
| Knot decode (softmax/softplus) | `nf_decode` | `sampling/neural_flow.slang` |
| RQ spline forward (§3, §4) | `nf_rqs_fwd` | `sampling/neural_flow.slang` |
| RQ spline inverse (§7) | `nf_rqs_inv` | `sampling/neural_flow.slang` |
| Square ↔ hemisphere map (§6) | `nf_square_to_hemi` / `nf_hemi_to_square` | `sampling/neural_flow.slang` |
| Inverse density `ω → q_ω` (§7) | `pdfNeural` / `neuralPdfWorld` | `sampling/neural_flow.slang` / `neural_proposal.slang` |
| Mixture weights α (§8) | `proposalWeights` | `sampling/proposal.slang` |
| Mixture pdf + update (§8, §9) | `sampleBounceDirection` / `mixtureProposalPdf` | `sampling/proposal.slang` |
| Forward pre-pass (per lane) | `wfNeuralProposal` | `shaders/wavefront/neural_proposal_pass.slang` |
| Training weight (§10) | backward attribution | `shaders/integrators/path_record.slang` |
| Record dump | `estimateRadianceRecord` / `emitRecord` | `shaders/integrators/path_record.slang` |
| Weight buffers (33/34/35) | `neuralWeights` / `neuralBiases` / `neuralLayers` | `sampling/neural_proposal.slang` |
| GPU pass + buffers | `WavefrontNeuralProposalPass` | `vk_wavefront.py` |
| Host config + load/bake | `neural_config` / `_sync_neural_weights` | `renderer.py` |
| Proposal selector plugin | `NeuralProposal` | `sampling/proposals.py` |
| NFW1 weight format / loss (§11) | `NeuralWeights` / `export_weights.py` | `sampling/neural_weights.py` / `spline_flow` |
| NREC record format | `read_records` / `RECORD_DTYPE` | `sampling/path_records.py` |

## Network architecture

![NF_LAYERS coupling layers alternate the masked coordinate; each layer's conditioner MLP emits the RQ-spline knots](diagrams/neural/fig_flow_arch.svg)

The flow is a faithful Slang port of the verified PyTorch prototype
(`spline_flow/train.py`: `ConditionalSplineFlow2D`). Per coupling layer there are
**three** `Linear` layers — `NF_MLP_IN (=1+9) → NF_HIDDEN → NF_HIDDEN →
3·NF_BINS+1` — with SiLU on the hidden layers; the final layer's `3K+1` outputs
decode to the `K` bin widths, `K` heights, and `K+1` boundary derivatives. The
default net is **6 layers · 24 bins · 96 hidden** (~103 k weights), reproduced
byte-for-byte when no `-D` override is given.

`NF_LAYERS`, `NF_BINS`, and `NF_HIDDEN` are `#ifndef`-guarded macros: `slangc -D
NF_HIDDEN=48` (etc.) selects an off-default size. The host threads the same `-D`
into every module that imports `neural_flow.slang` (the pre-pass, the inline
inverse, the record entry), folds the dims into the pipeline cache key, and the
NFW1 loader asserts the baked `(layers, bins, hidden, cond)` matches the built
dims.

## Precision & size study

Stage 2 makes the MLP's floating-point precision selectable while keeping the
spline core and the reported pdf at full precision in every mode. Two type
aliases, threaded through `-D`:

| Mode | `-D` flags | Weight storage | GEMM accumulate | Notes |
| --- | --- | --- | --- | --- |
| **fp32** (default) | *(none)* | `float` | `float` | Byte-identical to ship. |
| **fp16-storage** | `-D NF_WT=half` | `half` (½ bytes) | `float` | Halves the weight buffer + bandwidth. |
| **fp16-compute** | `-D NF_WT=half -D NF_CT=half` | `half` | `half` | + half ALU throughput (Apple Silicon). |

The RQ-spline math (softmax/cumsum/exp/log + the analytic inverse-quadratic solve)
and the returned solid-angle pdf stay `float` in **every** mode — fp16 there is
catastrophic-cancellation prone. NFW1 stays **fp32 on disk**; the host casts
fp32 → half at upload, and `vk_context` capability-gates fp16: a device lacking
`shaderFloat16` / `storageBuffer16BitAccess` silently falls back to fp32 and
reports the fallback.

### Quality-vs-cost results

A study harness sweeps a bounded size × precision grid on the Metal/MoltenVK
backend, measuring quality (size axis = held-out NLL; precision axis = fp16
pdf-parity drift) against cost (ms/frame + weight-buffer bytes), with an
in-renderer unbiased + firefly check per cell. Full results +
methodology notes: [`docs/diagrams/neural_study/RESULTS.md`](diagrams/neural_study/RESULTS.md)
(raw grid in `size_precision.csv`). Scene: flat Cornell box, 96×96, MoltenVK,
`{bsdf}` reference mean 0.00808. **21/21** cells ran (7 sizes × 3 precisions).

| L · B · H | precision | ms/frame | weight bytes | NLL | unbiased rel | firefly p99.9 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 6·24·48 | **fp16-compute** | **13.0** | **75 456** | −0.276 | 0.0029 | 0.0240 |
| 4·24·96 | fp16-compute | 17.3 | 137 472 | −0.281 | 0.0025 | 0.0238 |
| 6·16·96 | fp16-storage | 60.4 | 178 560 | −0.282 | 0.0024 | 0.0238 |
| 6·24·96 | fp32 | 121.6 | 412 416 | −0.279 | 0.0026 | 0.0239 |
| 6·24·144 | fp16-compute | 266.0 | 392 256 | −0.281 | 0.0019 | 0.0238 |

Findings (see the RESULTS notes for the full reasoning):

- **Every cell is unbiased** (rel-mean < 0.003) and **firefly-bounded** (p99.9 ≈
  0.024); fp16 costs no measurable quality — pdf-parity drift is ~4e-4 (storage) /
  ~1e-3 (compute) vs fp32.
- **Quality is flat across size** on this broad-indirect scene (NLL spread ~2%);
  a concentrated-indirect scene would spread NLL and raise the knee.
- **fp16 weights are exactly ½** the fp32 bytes. **Within a size**
  (the 3 precisions measured adjacently) **fp16-compute < fp16-storage < fp32** in
  6/7 sizes — the real Apple-Silicon win. *Cross-size* ms is thermally noisy (cells
  run sequentially; the GPU heats over the sweep) and is indicative only.
- **Recommended ship config: L6 · B24 · H48 @ fp16-compute** — the Pareto knee:
  the smallest footprint (75 456 B, **18%** of the fp32 baseline) within 2% of the
  best NLL, unbiased and firefly-bounded.

## Controls

SplineFlow is selected through the **proposal set**, consistent with the other
proposals (`sampling/proposals.py`):

| Control | Where | Options | Maps to | Effect |
| --- | --- | --- | --- | --- |
| **Proposal set** | `--proposals` CLI / UI selector / settings | e.g. `bsdf` · `bsdf,env` · `bsdf,neural` | `fc.proposalMask` (bits 0x1/0x2/0x4) | Which proposals mix. Triggers a wavefront pass rebuild + accumulation reset. |
| **Mixture weights** | host `proposalAlpha` | per-proposal `default_weight` | `fc.proposalAlpha.{x,y,z}` | Pre-normalisation MIS weights (α_b/α_e/α_n). |
| **Weights file** | `NeuralProposal(weights_path=…)` | `.nfw1` path / `None` | renderer `_neural_weights_path` | Per-scene net; `None` → renderer resolves per scene or bakes a dummy. |
| **Network size** | build-time `-D` | `NF_LAYERS/BINS/HIDDEN` | pipeline cache key | Off-default net dims (must match the baked weights). |
| **Precision** | `NeuralBuildConfig.precision` | fp32 · fp16-storage · fp16-compute | `-D NF_WT/NF_CT` | MLP storage/compute precision; spline + pdf stay fp32. |

The mask bits (`PROPOSAL_BSDF=0x1`, `PROPOSAL_ENV=0x2`, `PROPOSAL_NEURAL=0x4`)
mirror the Slang `PROPOSAL_*` constants in `sampling/proposal.slang`. The neural
proposal participates **only** when its bit is set *and* the lane has a valid
precomputed forward sample; otherwise its weight folds away and `{bsdf, env}`
renormalise to Σ = 1, keeping each lane unbiased. Changing the proposal set resets
progressive accumulation (folded into `_current_state_hash`).

When neural is **inactive**, the renderer binds 1-element dummy buffers at
33/34/35 (`make_dummy_weights`), and `make_dummy_weights`/`bake_dummy_weights`
produce a valid all-zero net for plumbing bring-up — an all-zero flow is the
identity-ish map and stays unbiased.

## Caveats and limits

- **Wavefront-only.** The megakernel/Metal backends keep the `{bsdf, env}` subset;
  requesting neural there is reported unsupported (capability gate, like
  wavefront-BDPT), not silently ignored.
- **Flat/python materials only.** Skin / MaterialX-graph lanes set
  `neuralValid = false` and pass through with `{bsdf, env}`.
- **Frozen / offline.** Weights are trained per scene in `spline_flow` and loaded
  read-only; there is no online/adaptive update at render time. Per-sample
  `networkVersion` stamping (baseline 0) is the foundation for a reserved Stage 3
  online-training replay buffer.
- **Condition must match the trainer byte-for-byte.** `neuralCondition` and the
  record dump's `(pos, normal, wo)` + AABB header are a shared contract; a
  mismatch raises variance silently rather than biasing.
- **Forward pre-pass amortises one draw per lane.** The arbitrary-direction
  inverse pdf is still evaluated inline in the shade kernel — the only inline MLP
  eval on the hot path.

## Verification

SplineFlow is validated against the BSDF-only / BDPT reference as ground truth:

- `tests/test_neural_parity.py` — locks the Slang port against the PyTorch
  reference: a numpy re-implementation of `neural_flow.slang` is checked against
  baked goldens (`tests/data/neural_parity/`, generated once with the `spline_flow`
  torch venv). Runs in CI with **no torch and no GPU**; the on-device variant
  drives the real `sampleNeural`/`pdfNeural` and skips where unavailable. Covers
  the §7 fp16 pdf-parity drift used as the study's precision axis.
- `tests/test_neural_headless.py` — end-to-end headless render with
  `{bsdf, neural}` active: the dummy (all-zero) net stays unbiased, and a trained
  net converges to the BSDF-only reference within noise.
- **Density integrates to one.** The flow's normalization
  (∫ q_ω dω ≈ 1 for a fixed condition) is the authoritative unbiasedness gate,
  carried over from the `spline_flow` prototype's PDF-normalization check.
- **Default parity.** `{bsdf}` is byte-identical to the pre-seam renderer (the
  pre-pass uses a decorrelated RNG and the mixture fast-path returns the material's
  native sample verbatim).

## References

1. **C. Durkan, A. Bekasov, I. Murray, G. Papamakarios.** *Neural Spline Flows.*
   NeurIPS, 2019. — the monotone rational-quadratic coupling transform (§3) and
   its closed-form inverse that this flow ports.
2. **L. Dinh, J. Sohl-Dickstein, S. Bengio.** *Density Estimation Using Real NVP.*
   ICLR, 2017. — affine/spline **coupling layers** with a tractable Jacobian (§2,
   §4); the alternating-mask architecture.
3. **T. Müller, B. McWilliams, F. Rousselle, M. Gross, J. Novák.** *Neural
   Importance Sampling.* ACM TOG 38(5), 2019. — learned importance sampling for
   light transport via normalizing flows; the path-guiding target `q ∝ f·L_i·cos`
   (§10, §11) and the unbiased MIS composition.
4. **E. Veach.** *Robust Monte Carlo Methods for Light Transport Simulation.* PhD
   thesis, Stanford University, 1997. — multiple importance sampling and the
   one-sample estimator backing the proposal mixture (§8, §9).
5. **T. Müller, M. Gross, J. Novák.** *Practical Path Guiding for Efficient
   Light-Transport Simulation.* EGSR (CGF 36(4)), 2017. — the per-scene,
   contribution-weighted path-guiding setup that the offline record→train loop
   follows.
