# Change: shader-variant-key-module

## Why

The shader build-variant matrix has no owning module. A compiled kernel's
identity is the product of several axes — compile target (Vulkan SPIR-V via
`slangc` vs native Metal via an in-process SlangPy session), pipeline family
(megakernel / wavefront full-tree / wavefront foundation / preview-debug),
spectral (`SKINNY_SPECTRAL`), MLT (`SKINNY_MLT`), the neural build config
(size × precision × encoding × coupling × chart, via
`NeuralBuildConfig.slang_defines()`), and the Metal-only argument-table gates
(`SKINNY_METAL_NEURAL`, `SKINNY_METAL_RECORDS`) — but every backend re-derives
its own define list at its own compile sites:

- `vk_compute.py` — `ComputePipeline._compile_slang` (megakernel:
  `SKINNY_COMPUTE_PIPELINE` + conditional `SKINNY_SPECTRAL`) and
  `PreviewPipeline._compile_slang` (`SKINNY_COMPUTE_PIPELINE` only); each has
  its own copy of `_cache_key` (blake2b over entry + src + flags + source tree
  → `build/spv_cache/<hash>.spv`).
- `vk_wavefront.py` — `_slang_flags` (foundation kernels,
  `SKINNY_WAVEFRONT=1`), `_compile_full_spv` (full-tree kernels:
  `SKINNY_COMPUTE_PIPELINE` + `SKINNY_WAVEFRONT` + conditional
  `SKINNY_SPECTRAL` + caller-supplied neural `-D` tokens + a filename `tag`),
  and the MLT pass site (`defines=("-D", "SKINNY_MLT=1"), tag="_mlt"`).
- `metal_compute.py` — three separate `opts.defines = {...}` dict literals:
  the megakernel (`ComputePipeline._build`, :676), the Metal preview pipeline
  (`PreviewPipelineMetal._build`, :995), and the Metal-only compute
  rasteriser (`DebugRasterMetal`, :1139, bare `SKINNY_METAL`).
- `metal_wavefront.py` — `_metal_slang_session`'s base dict
  (`SKINNY_COMPUTE_PIPELINE`/`SKINNY_METAL`/`SKINNY_WAVEFRONT`) plus five
  call-site dict assemblies layering `SKINNY_METAL_NEURAL`,
  `SKINNY_METAL_RECORDS`, `SKINNY_SPECTRAL`, `SKINNY_MLT`, and the neural
  defines (converted from `-D` tuples via a local `_defines_dict`).
- `wavefront_layout.py` — host-side sizers take `spectral=`/`msl=` booleans
  that must agree with the very same defines, with nothing tying them together.

Vulkan and Metal wavefront are two adapters that must agree on define
semantics per variant with nothing enforcing agreement — the recorded
guarantees (RGB wavefront `.spv` byte-identical across the spectral change for
all 28 kernels; megakernel `.spv` byte-identical under the `SKINNY_MLT` RNG
override; Vulkan SPIR-V byte-unchanged by any Metal-gated define; the default
`NeuralBuildConfig` emitting zero flags) are upheld only by convention.

The `.spv` filename tokens already on disk under `src/skinny/shaders/` are an
ad-hoc encoding of exactly this missing key: `_{cache_tag}` slugs like
`L6B24H96_E1_fp16-compute` (`L6B24H96` appears in 35 kernel filenames;
`fp16-compute` in 14), `_mlt`, `_spectral` — assembled by string concatenation
at `_compile_full_spv` (`f"{out_name}{tag}{spectral_suffix}.spv"`) with the
neural part coming from `NeuralBuildConfig.cache_tag` three files away.

## What Changes

- Add one hostless variant-key module: a frozen dataclass `ShaderVariantKey`
  capturing the axes above (with a `(target, family)` validity table —
  wavefront-foundation is Vulkan-only, the debug compute rasteriser is
  Metal-only), with three derivations — `slangc_defines()` (Vulkan `-D`
  tokens as named ordered segments, spliced by each site at its recorded
  position relative to `-fvk-use-scalar-layout` so the hashed flag tuples
  stay byte-identical), `session_defines()` (dict for the SlangPy Metal
  session), and `cache_token()` (the `.spv` filename tag) — plus an explicit
  `METAL_ONLY_DEFINES` set and a recorded-asymmetry table (one entry found:
  Metal SPPM compiles with the active neural `NF_*` defines, Vulkan SPPM
  passes none).
- Migrate all existing compile sites (`vk_compute.ComputePipeline` /
  `PreviewPipeline`, `vk_wavefront` foundation + full-tree + MLT,
  `metal_compute` megakernel/foundation/trivial, `metal_wavefront` session +
  per-pass sites) to consume the module instead of hand-assembling defines.
  `wavefront_layout.py` sizers accept the key (or its axes read off the key)
  so host sizing and shader defines cannot drift.
- Reuse, not replace, `NeuralBuildConfig`: the neural size/precision/encoding
  axes stay owned by `sampling/neural_weights.py`; the key composes it.
- Add a hostless test asserting that for every valid key the Vulkan and Metal
  define sets are identical modulo `METAL_ONLY_DEFINES` and target-form
  (`-D` tokens vs dict), and that `cache_token()` reproduces today's filename
  tags byte-for-byte (including the empty default).
- Pure refactor: no new variants, no dispatch changes, no behavior change.
  Compiled artifacts (SPIR-V bytes, `build/spv_cache` keys, tagged `.spv`
  filenames) are byte-identical before and after.

## Capabilities

### New Capabilities

- `shader-variant-key`: single owning module for the build-variant matrix —
  variant key → define list + cache token — consumed by both backends'
  megakernel and wavefront compile paths and the wavefront host sizers, with
  the cross-backend agreement and byte-identity guarantees stated as testable
  requirements.

### Modified Capabilities

None — this is a pure refactor. `metal-backend`, `wavefront-execution`,
`spectral-rendering`, `metropolis-light-transport`, and
`neural-directional-proposal` requirements are unchanged; their compiled
artifacts must be byte-identical, which the new capability's requirements
enforce.

## Impact

- New: `src/skinny/shader_variants.py` (hostless, no GPU imports), one
  hostless test module.
- Modified (consumers only, no semantic change): `src/skinny/vk_compute.py`,
  `src/skinny/vk_wavefront.py`, `src/skinny/metal_compute.py`,
  `src/skinny/metal_wavefront.py`, `src/skinny/wavefront_layout.py`, and the
  renderer call sites that currently pass raw define booleans/tuples.
- Unchanged: shaders, `NeuralBuildConfig` internals, `build/spv_cache` hash
  scheme, on-disk `.spv` names, CLI surface, docs-visible behavior.
- Docs: `docs/Architecture.md` gains a short section naming the module as the
  owner of the variant matrix.
