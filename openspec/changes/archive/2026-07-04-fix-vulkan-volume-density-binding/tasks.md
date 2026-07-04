## 1. Reproduce & confirm root cause

- [x] 1.1 Repro on MoltenVK: `tests/test_sppm_gpu.py::test_sppm_builds_and_renders_finite`
  fails at the megakernel driver-pipeline compile with the VUID-07988
  validation error naming `[Set 0, Binding 26, "volumeDensity"]` and the
  `SPIR-V to MSL conversion error: nullptr`.
- [x] 1.2 Confirmed `main_pass.spv` references binding 26 (`spirv-dis`: `OpDecorate
  %volumeDensity Binding 26`, `DescriptorSet 0`), all 8 `wfSppm*` kernels convert
  to MSL standalone (`spirv-cross --msl`, empty stderr — the reported big-kernel
  SPIRV-Cross suspicion was wrong), and `_create_descriptor_set_layout` had no
  binding-26 entry (git: no commit ever added it; `aa9d70e` added the shader decl,
  the write, and the pool size only).

## 2. Fix

- [x] 2.1 Added the binding-26 `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` entry
  (`volumeDensity`) to `ComputePipeline._create_descriptor_set_layout`
  (`vk_compute.py`), in numeric order after binding 24, with a comment mirroring
  the bindings.slang design note + the MoltenVK failure mode.

## 3. Tests

- [x] 3.1 New hostless `tests/test_vk_binding_layout.py`: mini-preprocessor over
  `bindings.slang` (Vulkan branch, `SKINNY_METAL` undefined) collects single-arg
  `[[vk::binding(N)]]`; source scan of `_create_descriptor_set_layout` collects
  `binding=N` (+ GRAPH_BINDING_BASE=25 when conditional). Asserts shader ⊆ layout.
  Verified RED before 2.1 (`AssertionError: … bindings [26] …`), GREEN after
  (3 passed); includes self-guards so a stale regex can't pass vacuously.
- [x] 3.2 GPU gate on MoltenVK: `test_sppm_builds_and_renders_finite` **passes**
  (was a hard build failure). `test_sppm_energy_matches_path_tracer` now runs and
  **xfails** — see 3.4. Caustic parity (`test_sppm_caustic_parity_vs_pbrt_reference`,
  Metal) still passes.
- [x] 3.3 Hostless: `test_vk_binding_layout.py` 3 passed; `ruff` clean; `py_compile`
  clean.
- [x] 3.4 The re-armed energy gate exposed a **separate, pre-existing** SPPM
  diffuse-indirect energy deficit (NOT introduced here): cornell-box-sphere
  SPPM/path ratio ≈0.71 (sRGB) / ≈0.48 (linear), stable across 16→64 frames and
  IDENTICAL on both backends (MoltenVK 0.727, native Metal 0.712 — so not a
  MoltenVK artifact and not fixable by porting to Metal), while SPPM caustic
  transport still matches pbrt. This is the photon-flux regression the gate was
  meant to catch; its fix is the sibling change `fix-sppm-bathroom-black-walls`
  (not in this tree). Marked `xfail(strict=True)` with the measured evidence —
  band NOT widened (would hide a real divergence); strict flips it to a failing
  XPASS when the flux fix lands, forcing removal.

## 4. Docs

- [x] 4.1 `docs/Architecture.md` binding map: added the Vulkan-set-0 declaration
  requirement + MoltenVK failure mode + the hostless-audit note under the
  binding-26 discussion.
- [x] 4.2 `CHANGELOG.md` Unreleased → Fixed entry.

## 5. Validate

- [x] 5.1 `openspec validate fix-vulkan-volume-density-binding --strict`.
