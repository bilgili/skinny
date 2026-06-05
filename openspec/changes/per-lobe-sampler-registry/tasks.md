## 1. Worktree headless environment + baseline capture

- [x] 1.1 Stand up headless rendering in this worktree: copy the repo-root Python
  3.13 venv (`./bin/python3.13`) and rewrite its `.pth` to point at this
  worktree's `src/`, then rsync the main checkout's `main_pass.spv` in as a
  starting point (the renderer recompiles it at runtime via the vk_compute
  source-hash cache, so later shader edits just work). Export `VULKAN_SDK` +
  `DYLD_LIBRARY_PATH` per CLAUDE.md. Confirm `from MaterialX import
  PyMaterialXGenSlang` imports.
- [x] 1.2 Capture the all-native baseline: render `three_materials_demo.usda`
  IBL-only (`direct_light_index=1`) under PT-BSDF and BDPT, megakernel and
  wavefront, and store the per-column means (brass ≈ 0.219, marble, wood) plus a
  full default-preset frame. This is the pixel-identity + convergence reference
  the gate compares against. (Note: headless USD loads the fallback
  UsdPreviewSurface — brass is still coat+metal and reproduces the seam.)

## 2. Shader seam — `shaders/materials/flat/flat_lobes.slang`

- [x] 2.1 Add sampler-id constants alongside `FLAT_SAMPLER_NATIVE`:
  `FLAT_SAMPLER_SPHCAP` (GGX coat/spec) and `FLAT_SAMPLER_UNIFORM` (diffuse).
- [x] 2.2 Implement the Heitz-2018 basis-form VNDF warp (Heitz, JCGT 7(4) 2018 —
  the orthonormal-basis construction) as a new sample routine over the same GGX
  roughness, returning a direction drawn from the same `D_vis` as the native
  sampler so they share `GGXSampler.pdf`. NOTE: `samplers/ggx.slang` is ALREADY
  the 2023 spherical-cap / bounded-VNDF form (native); the 2018 basis-form is the
  genuinely-different warp (and exhibits the basis singularity near `V ∥ N` that
  the 2023 form removed).
- [x] 2.3 Implement uniform-hemisphere sampling for the diffuse lobe (restrict the
  existing uniform-sphere sampler to `z > 0`), with pdf `1/2π`.
- [x] 2.4 Make `flatSampleLobe` dispatch on `samplerId`: coat/spec → Heitz VNDF or
  spherical-cap; diffuse → cosine or uniform-hemisphere; any unrecognized
  `(lobeKind, samplerId)` pair falls back to the native branch.
- [x] 2.5 Make `flatLobePdf` branch on `samplerId` **only for the diffuse lobe**
  (cosine `NdotL/π` vs uniform `1/2π`); GGX lobes keep one shared analytic VNDF
  pdf (the basis-form and the native spherical-cap share `D_vis`), so the GGX pdf
  stays samplerId-agnostic — this is what makes the basis-form parity structural.
- [x] 2.6 Leave `flatBsdfResponse` (= `f·cos`) byte-for-byte unchanged.

## 3. Per-lobe bounded weight in `sample()` — `shaders/materials/flat/flat_material.slang`

- [x] 3.1 Unpack the per-lobe sampler ids from the new `FrameConstants` field and
  pass each lobe's id into `flatSampleLobe` / `flatLobePdf` (replacing the
  hard-coded `FLAT_SAMPLER_NATIVE` at every call site) in both `sample()` and
  `evaluate()`.
- [x] 3.2 Keep the coat/spec weight as `F·G₁` (unchanged — the basis-form shares
  the VNDF estimator). Add the diffuse-uniform bounded weight branch
  (`2·diffHue·cos`) selected when the diffuse sampler id is uniform; cosine keeps
  the existing weight.
- [x] 3.3 Throwaway verification (pin the structural claim before host wiring):
  force coat/spec = basis-form VNDF in-shader, render the gate scene, and confirm
  `sample().pdf == evaluate().pdf` and brass still ≈ 0.219. Revert the force.

## 4. GPU transport — `shaders/common.slang` + `renderer.py`

- [x] 4.1 Append `uint flatLobeSamplers` to `FrameConstants` in `common.slang`
  (8 bits/lobe: `coat | spec<<8 | diff<<16`), documenting the bit layout in the
  struct comment next to `proposalMask`.
- [x] 4.2 Pack `flatLobeSamplers` in `renderer._pack_uniforms` in std140 lockstep
  with the struct, in the exact field order. Add/extend a uniform-pack byte-size
  assertion so a layout mismatch fails loudly instead of corrupting
  `proposalAlpha`.

## 5. Host registry — `src/skinny/sampling/lobe_samplers.py`

- [x] 5.1 Add a `LobeSamplerStrategy` descriptor (name, `valid_lobes` mask,
  `shader_id`, `cli_token`) and a registry list, mirroring `sampling/proposals.py`
  + `PROPOSAL_PLUGINS`. Register: native (all lobes), spherical-cap (coat+spec),
  uniform-hemisphere (diffuse). Export a parse helper (`--lobe-samplers` tokens →
  per-lobe ids) and a fold helper (three indices → packed `flatLobeSamplers`
  uint), parallel to `parse_proposals` / `proposal_mask_and_alpha`.

## 6. Renderer wiring — `renderer.py`

- [x] 6.1 Hold three selection indices (`coat/spec/diffuse_sampler_index`) with
  per-lobe option lists derived from the registry; fold them to the packed uint
  consumed in 4.2.
- [x] 6.2 Add the three indices to `_current_state_hash()` so changing any per-lobe
  sampler resets progressive accumulation.

## 7. CLI, GUI, persistence — `app.py`

- [x] 7.1 Add `--lobe-samplers coat=…,spec=…,diff=…` parsing (env fallback),
  parallel to `--proposals`, overriding the persisted selection.
- [x] 7.2 Add three per-lobe dropdown params to `ALL_PARAMS`
  (`flat.coat_sampler`, `flat.spec_sampler`, `flat.diffuse_sampler`), resolved via
  `_get_nested`/`_set_nested`, options data-driven from the registry's
  `valid_lobes`. Ensure they persist in the settings snapshot and appear in the
  interactive UI (and debug viewport, per the GUI-consistency rule).

## 8. Gate / tests — `tests/`

- [x] 8.1 Per-sampler parity: extend `tests/test_sampling_parity.py` so that with
  coat/spec = basis-form VNDF the gate scene converges to the SAME per-column
  values as native (brass ≈ 0.219, marble, wood), megakernel ≡ wavefront and
  PT ≡ BDPT.
  Regenerate the gitignored goldens
  (`tests/_sampling_parity_golden_{megakernel,wavefront}.txt`).
- [x] 8.2 Diffuse variance A/B: add a test asserting the diffuse uniform-hemisphere
  strategy converges to the same mean as cosine but has higher low-sample variance
  (proves the swap is live and unbiased, weight bounded).
- [x] 8.3 Default pixel-identity: assert the all-native default selection is
  pixel-identical to the 1.2 baseline on both backends.
- [x] 8.4 Run the ReSTIR suite and confirm it stays green (the unified BSDF /
  evaluate path is unchanged for native).

## 9. Docs

- [x] 9.1 `docs/Architecture.md`: add `flatLobeSamplers` to the FrameConstants /
  uniform documentation (note: no new descriptor binding).
- [x] 9.2 Update the flat-BSDF section of `docs/SkinRendering.md` (or the relevant
  doc) describing the per-lobe sampler registry and the registered strategies;
  use an SVG (not ASCII) for any diagram, authored under `docs/diagrams/`.
- [x] 9.3 `README.md`: document the `--lobe-samplers` CLI flag alongside
  `--proposals`. Add a `CHANGELOG.md` entry.

## 10. Validate + finalize

- [x] 10.1 `openspec validate per-lobe-sampler-registry --strict` passes.
- [x] 10.2 `ruff check src/` clean; full `pytest` green (parity, variance, ReSTIR,
  default-identity).
- [x] 10.3 Update the project memory and prepare the PR / merge from the worktree.
