# Tasks: shader-variant-key-module

## 1. Baseline capture (before any code moves)

- [x] 1.1 Record, for the current tree, a golden snapshot per compile site:
      Vulkan flag tuples (megakernel RGB+spectral, preview, wavefront
      foundation, all 28 RGB full-tree kernels, MLT RGB+spectral,
      representative neural configs), Metal defines dicts (megakernel,
      foundation, trivial, wavefront base + neural/records/spectral/mlt
      states, SPPM with default and non-default neural config), and all
      tagged `.spv` out-filenames. Store as a **permanent** hostless test
      fixture (kept after migration — it anchors the agreement test to
      recorded reality, not the module's own table). Capture flag tuples
      with exact positions of `-fvk-use-scalar-layout` per site (three
      distinct orders exist).
- [x] 1.2 Record blake2b `_cache_key` values for the Vulkan megakernel and
      preview pipelines over a pinned source tree (cache-hit check for 4.2).

## 2. Module + tests (no consumers yet)

- [x] 2.1 Add `src/skinny/shader_variants.py`: `Target`/`Family` enums
      (MEGAKERNEL, WAVEFRONT, WAVEFRONT_FOUNDATION, PREVIEW, DEBUG_RASTER),
      frozen `ShaderVariantKey` (spectral, mlt, metal_neural, metal_records,
      optional `NeuralBuildConfig`), `__post_init__` enforcing the
      `(target, family)` validity table (foundation Vulkan-only, debug-raster
      Metal-only) + axis rules, shared internal define table,
      `slangc_defines()` returning named ordered segments
      (`base`/`spectral`/`neural`/`mlt`), `session_defines()`,
      `cache_token()`, `METAL_ONLY_DEFINES`, and the recorded-asymmetry
      table (initial entry: Metal-SPPM neural `NF_*` defines).
- [x] 2.2 Add `tests/test_shader_variants.py`: cross-backend agreement sweep
      over families valid on both targets (Metal defines minus
      `METAL_ONLY_DEFINES` == Vulkan defines per key, modulo
      recorded-asymmetry entries; SPPM non-default-neural case must match the
      recorded entry exactly); permanent golden define tuples/dicts from 1.1;
      `cache_token()` goldens incl. empty default; invalid-combination and
      invalid-(target,family) raises; default neural config emits zero
      `NF_*` flags and empty slug.
- [x] 2.3 Verify the design D5 asymmetry record against the 1.1 goldens
      (Metal SPPM `NF_*` defines vs Vulkan SPPM none; MLT confirmed
      symmetric) and pin each Vulkan site's segment splice positions
      relative to `-fvk-use-scalar-layout` from the golden tuples.

## 3. Vulkan consumers

- [x] 3.1 Migrate `vk_compute.ComputePipeline._compile_slang` and
      `PreviewPipeline._compile_slang` to the module's define segments,
      splicing at the recorded positions (megakernel: `base` before
      `-fvk-use-scalar-layout`, `spectral` **after** it); assert emitted flag
      tuples equal the 1.1 goldens (order and spelling).
- [x] 3.2 Verify cache-hit incl. the spectral megakernel: `_cache_key` values
      equal 1.2 and a rebuild over an unchanged tree reuses the cached `.spv`
      without invoking `slangc`.
- [x] 3.3 Migrate `vk_wavefront._slang_flags` (`SKINNY_WAVEFRONT` spliced
      **after** the scalar-layout flag), `_compile_full_spv` (all segments
      **before** it + `cache_token()` for the filename tag), and the MLT
      compile site; assert all 28 RGB kernel flag tuples and every tagged
      out-filename (`_mlt`/`_spectral`/neural slugs) match 1.1 byte-for-byte.

## 4. Metal consumers

- [x] 4.1 Migrate the three `metal_compute` define-literal sites to
      `session_defines()` (full-dict single assignment — SlangPy
      `opts.defines` copy-on-read gotcha); assert dicts equal the 1.1 goldens.
- [x] 4.2 Migrate `metal_wavefront._metal_slang_session` and the five
      per-pass define assemblies; delete the local `_defines_dict`; assert
      per-pass dicts equal 1.1 for representative
      (neural, records, spectral, mlt) states.

## 5. Renderer / layout seam

- [x] 5.1 Build one `ShaderVariantKey` per compile request at the renderer
      call path; source `wavefront_layout` sizer `spectral`/`msl` arguments
      from the key's axes. Sizer signatures unchanged.
- [x] 5.2 Hostless check: sizer outputs unchanged for RGB and spectral,
      scalar and MSL (existing `wavefront_layout` tests stay green).

## 6. Verification and docs

- [x] 6.1 Hostless sweep green: `tests/test_shader_variants.py` + existing
      hostless suites (`tests/pbrt/test_matrix.py`, layout tests,
      `tests/test_metal_cleanup.py` non-gpu subset — compile plumbing was
      touched).
- [x] 6.2 GPU smoke under dispatch-hygiene rules (one guarded Metal process):
      one megakernel + one wavefront frame per backend, RGB and spectral;
      confirm rendered output unchanged and no recompile churn in
      `build/spv_cache`.
- [x] 6.3 Byte-identity confirmation: recompile the megakernel and the 28 RGB
      wavefront kernels pre/post on the same slangc — `.spv` bytes identical
      (guarantees 1–4 of the spec requirement).
- [x] 6.4 Update `docs/Architecture.md` with a short "Shader variant key"
      section naming the module as the matrix owner; run
      `openspec validate shader-variant-key-module`.
