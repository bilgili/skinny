# Tasks — neural-nis-baseline

> Depends on `spline_flow` change `nis-baseline-comparison` (piecewise-quadratic
> coupling, one-blob encoding, trained NIS net). Land that first.

## 0. Prerequisite bugfix — online-training scene-bounds (online-bounds-fix)

Found while validating the existing `neural` (RQ) proposal renders before adding
NIS: the online trainer conditioned on RAW world position while the shader's
`neuralCondition` conditions on AABB-normalised position, so the position channel
was off-distribution at inference and the online-trained net collapsed to a
generic, scene-agnostic (near no-op) proposal. Fixed here because BOTH `neural`
and `nis` share this online path; any renderer-scene number is meaningless until
the online net actually conditions on the scene.

- [x] 0.1 `enable_online_training` passes `bounds=tuple(self._neural_scene_bounds())` into `TrainerConfig` (was `None` → `_bounds()` fell back to `(0,1)` raw position); same AABB as `_pack_uniforms` inference + `dump_path_records`
- [x] 0.2 Driver `scripts/cornell_online_train.py`: `--trainer` flag, default `auto` (→ MLX Apple GPU) instead of the hardcoded `cpu` numpy oracle
- [x] 0.3 Diagnostics: `scripts/nis_bounds_ablation.py` (controlled A/B isolating the bounds bug: BROKEN test-NLL@infer +4.35 / no lobe tracking vs FIXED −3.68 / tracks), `nis_eval_net_density.py`, `nis_train_curve.py`
- [x] 0.4 Confirmed on real nets: cornell_fixed vs glass_fixed now scene-specific (density L1≈0.8; pre-fix nets were ≈identical), concentrated (C≤0.68)
- [x] 0.5 Apple-native stack (Metal render + MLX trainer) validated end-to-end: online train (record drain + MLX + Metal neural inference) works; `--backend metal` driver flag + `scripts/nis_sweep_*_metal.json`. Metal sweep matches Vulkan — bsdf budget curves BIT-IDENTICAL across backends (cross-backend render parity), same equal-sample conclusion. Metal render + MLX contend for the Apple GPU → ~8× slower training cycles (5.5 vs 0.66 s/cyc). (Vulkan+CUDA on the 4090 deferred to the user.)
- [x] 0.6 Harness double-normalisation bug FOUND + FIXED: `read_accumulation_hdr` already returns the running MEAN (verified: raw image spp-invariant, raw.mean~0.40 for N=1..128) but `_accumulate_to`/`_sweep_budgets` divided by `samples` again → mean/spp, so every MSE was dominated by a deterministic `mean²(1/b−1/ref)²` bias and the earlier bsdf/neural ratio≈1.0 was an ARTIFACT, not a measurement. Fixed (use the mean directly). Added `--no-direct` (zero analytic distant light on cells+ref) and `--dump-images`.
- [ ] 0.7 OPEN — CORRECTED result (fixed harness, --no-direct, Metal res48 seeds3): neural is a NET LOSS, not a no-op. ratio bsdf/neural <1: cornell 0.79/0.78/0.81, glass 0.84/0.99/0.85 (16/64/256spp); neural firefly p99.9 higher (cornell 7.3e-2 vs 6.1e-2; glass 4.6e-1 vs 3.3e-1). Cause: online MLE `w=luminance(contrib)` has NO firefly/contribution clamp → guide concentrates toward firefly dirs → adds variance. FIX: clamp/robustify the contribution weight in `build_dataset_np` (cap w at a percentile) + retrain, then re-run the --no-direct corrected sweep.

## 1. Shader piecewise-quadratic coupling (nf-coupling-shader)

- [ ] 1.1 Add `nis_decode(params[], out masses[], out heights[])` to `neural_flow.slang` (softmax bin masses + `K+1` vertex heights; min-mass floor)
- [ ] 1.2 Add `nis_quad_fwd(x, masses, heights, out logdet)` and `nis_quad_inv(y, masses, heights, out logdet)` (monotone piecewise-quadratic CDF; per-bin quadratic inverse; closed-form `log|dy/dx|`)
- [ ] 1.3 Add `NF_COUPLING` build define (`0=rqs` default | `1=nis-pq`); gate the per-coupling call in `nf_flow_forward`/`nf_flow_inverse`
- [ ] 1.4 Parity: `NF_COUPLING=rqs` byte-identical to the shipped kernel; `nis-pq` forward/inverse match the `spline_flow` trainer's reference samples/pdfs (extend `test_neural_parity.py`)
- [ ] 1.5 Confirm chart sandwich + `NF_LOG2PI` (V1 equal-area, constant `2π`) untouched

## 2. One-blob conditioner encoding (oneblob-encoding)

- [ ] 2.1 Add a one-blob encoder to the `NF_ENCODING` define path (shader `nf_encode` + the `neural_weights.py` `Encoding` enum / `encoded_cond_dim`)
- [ ] 2.2 Host parity mirror for the one-blob encode; confirm Jacobian-free (coupling log-det unchanged)

## 3. NFW1 coupling tag (nfw1-coupling-tag)

- [ ] 3.1 Add a coupling-type field to the NFW1 header (`neural_weights.py`); bump version or use a reserved slot
- [ ] 3.2 `load_neural_weights` validates coupling tag vs compiled `NF_COUPLING`; mismatch errors clearly (mirror the encoding-dim check)
- [ ] 3.3 Update `NeuralBuildConfig` to carry coupling; thread into `slang_defines()`

## 4. NIS proposal plugin (nis-proposal)

- [ ] 4.1 Add `NisProposal(ProposalPlugin)` to `proposals.py` (`cli_token="nis"`, `mask_bit=0x8`)
- [ ] 4.2 Add the shader proposal bit slot in `proposal.slang`; confirm the mixture-MIS pdf division and BSDF-only bit-identity default are unchanged
- [ ] 4.3 Wire its weight buffers/bindings (distinct from the `neural` bindings)

## 5. Trainer handoff (nis-handoff)

- [ ] 5.1 Torch backend builds `ConditionalSplineFlow2D(coupling="nis-pq", encoding=one-blob)` from `spline_flow`; export to NFW1 with the coupling tag
- [ ] 5.2 `networkVersion`-stamped upload through the existing frozen-render-side path; parity test net round-trips trainer → NFW1 → shader

## 6. Variance harness + scenes (nis-eval)

- [ ] 6.1 Add `nis` to `guiding_variance_sweep.py` proposals; add renderer-scene cells at chart `V1`
- [ ] 6.2 Add a single comparison-scene catalog (`scripts/comparison_scenes.py` or a JSON registry): per scene name, source/provenance, license, static/animated, resolution, transport regime; sweep cells reference scenes by catalog key
- [ ] 6.3 Populate the catalog: Cornell, three-materials, Veach Ajar (static), NVIDIA Emerald Square (animated), the ReSTIR scene — downloaded from Bitterli Rendering Resources / ORCA, converted via the USD loader, licenses recorded
- [ ] 6.4 Emit equal-sample-count + equal-time variance, equal-variance time, `1/(var·t)` efficiency for {baseline, flow, nis, restir-di, flow+restir}; both operating-point renders retained
- [ ] 6.5 NIS analysis protocol: add the image-error metric (relative-L2 + MAPE vs high-spp reference) and convergence curves (error vs spp **and** error vs wall-clock time) per method; export the convergence figure for the paper
- [ ] 6.6 Training-cost amortization: report one-time training cost vs per-frame inference cost; state the frame budget at which the flow amortizes its inference overhead
- [ ] 6.7 Equal-parameter companion run (matched layers/bins/width); reuse the `spline_flow`-winning config (ablations run there, not in the render matrix)
- [ ] 6.8 Export the catalog as the single source for the paper's benchmark-scene section + data-availability statement
- [ ] 6.9 **PBR materials (author intent)**: sweep the OpenPBR/MaterialX reflective subset (`assets/materialxusd/tests/physically_based`, ~72 materials — the `pbr-material-brdf-tables` library) under the renderer, not one GGX family
- [ ] 6.10 **NIS-paper scenes & budgets (author intent)**: add NIS's own 16-scene corpus (`paper/sources/research_benchmark_scenes.md`: Bathroom, Bedroom, Bookshelf, Copper Hairball, Country Kitchen, Salle de Bain, Staircase, Veach Door, …) to the catalog and compare at NIS's equal-sample + equal-time budgets with image-error convergence

## 7. Gates (nis-gates)

- [ ] 7.1 Renderer unbiasedness gate: converged NIS render matches the reference before any scene number is reported
- [ ] 7.2 No silent caps: log any dropped cell / non-converged net

## 8. Validation

- [ ] 8.1 `openspec validate neural-nis-baseline --strict` passes
- [ ] 8.2 All parity + gate tests green; `NF_COUPLING=rqs` byte-identity regression green
