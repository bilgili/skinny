# shader-variant-key (delta)

## ADDED Requirements

### Requirement: Single owning variant-key module

The system SHALL provide one hostless module (`src/skinny/shader_variants.py`,
importable with no GPU/window dependencies) that owns the shader build-variant
matrix: a frozen `ShaderVariantKey` value over the axes target (Vulkan |
Metal), pipeline family (megakernel | wavefront full-tree | wavefront
foundation | preview/trivial), spectral, MLT, the Metal-only gates
(`SKINNY_METAL_NEURAL`, `SKINNY_METAL_RECORDS`), and an optional composed
`NeuralBuildConfig`. The module SHALL derive, from a key alone: the Vulkan
`slangc` define tokens as **named ordered segments** (`base`, `spectral`,
`neural`, `mlt`) that each compile site splices at its recorded historical
position relative to its own flag scaffolding (`-fvk-use-scalar-layout`
placement differs per site and is hashed positionally into the `spv_cache`
key, so a single contiguous block cannot reproduce the existing flag tuples),
the Metal SlangPy defines dict, and the `.spv` cache filename token. Both
derivational forms MUST be produced from one shared internal define table so
they cannot diverge. The module SHALL enforce a `(target, family)` validity
table — `WAVEFRONT_FOUNDATION` is Vulkan-only (Metal has no foundation
compile), `DEBUG_RASTER` (`DebugRasterMetal`) is Metal-only (the Vulkan debug
viewport is a graphics rasteriser), megakernel/wavefront/preview are valid on
both. The module SHALL NOT re-derive the neural `NF_*` defines or the neural
cache slug — it MUST compose `NeuralBuildConfig.slang_defines()` /
`NeuralBuildConfig.cache_tag` unchanged.

#### Scenario: One key yields both backend forms

- **WHEN** a valid `ShaderVariantKey` is constructed and both
  `slangc_defines()` and `session_defines()` are read
- **THEN** the two forms encode the same define name→value set, differing
  only in representation (`-D name=value` tokens vs dict) and in the members
  of the module's declared Metal-only define set

#### Scenario: Invalid axis combinations are unrepresentable

- **WHEN** a key is constructed with a Metal-only gate on a Vulkan-target key,
  with `mlt=True` outside the wavefront family, or with a `(target, family)`
  pair outside the validity table (e.g. Metal `WAVEFRONT_FOUNDATION`, Vulkan
  `DEBUG_RASTER`)
- **THEN** construction raises an error naming the invalid combination,
  rather than silently dropping or emitting the define

### Requirement: All compile sites consume the module

Every shader compile site that today hand-assembles variant defines SHALL
consume the variant-key module instead: `vk_compute.ComputePipeline` and
`vk_compute.PreviewPipeline`, the `vk_wavefront` foundation
(`_slang_flags`), full-tree (`_compile_full_spv`), and MLT compile sites, the
three `metal_compute` define-literal sites, and the `metal_wavefront` session
plus per-pass define assemblies (whose local `_defines_dict` helper is
removed). The `wavefront_layout` host sizers' `spectral`/`msl` inputs SHALL be
sourced from the active key's axes at the renderer call path, so host sizing
and shader defines cannot disagree.

#### Scenario: No hand-assembled variant defines remain

- **WHEN** the source tree is searched for variant define assembly
  (`SKINNY_SPECTRAL` / `SKINNY_MLT` / `SKINNY_METAL_NEURAL` /
  `SKINNY_METAL_RECORDS` emission into compile flags or defines dicts)
  outside `shader_variants.py`
- **THEN** no compile site outside the module emits them; all listed sites
  obtain their defines and cache token from a `ShaderVariantKey`

#### Scenario: Sizers agree with defines by construction

- **WHEN** the renderer sizes wavefront queue buffers for a spectral (or MSL)
  compile
- **THEN** the `spectral`/`msl` values passed to the `wavefront_layout`
  sizers are read from the same `ShaderVariantKey` that produced that
  compile's defines

### Requirement: Cross-backend define agreement is tested

A hostless test SHALL assert, for every family valid on both targets under
the `(target, family)` validity table and for every key pair in that sweep
differing only in target, that the Metal define set minus the module's
explicit `METAL_ONLY_DEFINES` constant equals the Vulkan define set (names
and values), modulo an explicit recorded-asymmetry table in the module. The
Metal-only defines (`SKINNY_METAL`, `SKINNY_METAL_NEURAL`,
`SKINNY_METAL_RECORDS`) MUST be declared in that named constant, and each
recorded asymmetry MUST be a named table entry — initially exactly one: the
Metal SPPM pass compiles with the active `NeuralBuildConfig` `NF_*` defines
while the Vulkan SPPM compile passes none (asymmetric today; vacuous at the
default config, which emits zero `NF_*` flags). Deliberate asymmetries are
explicit, not implicit in call sites. The golden define/token fixtures
captured from the pre-refactor tree SHALL be retained as **permanent** test
fixtures (not migration scaffolding), so the agreement test is anchored to
recorded reality rather than being purely self-referential against the
module's own table.

#### Scenario: Backend adapters see identical semantics per key

- **WHEN** the hostless agreement test runs over the sweep of keys in
  families valid on both targets
- **THEN** for each key, `session_defines()` restricted to non-Metal-only
  names equals the parsed `slangc_defines()` segments except for
  recorded-asymmetry table entries, and any unrecorded mismatch fails the
  test naming the key and the differing define

#### Scenario: SPPM neural asymmetry is recorded, not invented ad hoc

- **WHEN** the agreement sweep evaluates an SPPM wavefront key with a
  non-default `NeuralBuildConfig`
- **THEN** the Metal/Vulkan `NF_*` divergence matches the module's named
  recorded-asymmetry entry exactly, and removing that entry (or any drift
  from the recorded shape) fails the test

### Requirement: Cache tokens are byte-identical to the pre-change encoding

`cache_token()` SHALL reproduce the existing `.spv` filename tag derivation
byte-for-byte: the neural slug (`_` + `NeuralBuildConfig.cache_tag`, e.g.
`_L6B24H96_E1_fp16-compute`) only when the neural config emits defines, then
`_mlt` when MLT, then `_spectral` when spectral, and the empty string for the
default key. Existing tagged `.spv` filenames and the `build/spv_cache`
content-hash keys SHALL remain valid without a cache flush — including for
the spectral megakernel. Because the cache key hashes the full flag tuple
positionally and the existing sites interleave defines with
`-fvk-use-scalar-layout` in different orders (the megakernel appends
`SKINNY_SPECTRAL` after it, the wavefront full-tree compile puts all defines
before it, the foundation flags put `SKINNY_WAVEFRONT` after it), each
migrated site MUST splice the module's define segments at its recorded
historical position so the resulting flag tuples are byte-identical in order
and spelling to the pre-refactor tuples.

#### Scenario: Tagged filenames unchanged

- **WHEN** `cache_token()` is evaluated for the default key, an MLT key, a
  spectral key, and a non-default neural key
- **THEN** the tokens equal `""`, `"_mlt"`, `"_spectral"`, and
  `"_<cache_tag>"` respectively — matching the filenames already on disk
  (e.g. `_wfneural_L6B24H96_E1_fp16-compute.spv`) with no renamed artifacts

#### Scenario: spv cache hits survive the migration

- **WHEN** a pipeline (including the spectral megakernel, whose
  `SKINNY_SPECTRAL` define follows `-fvk-use-scalar-layout` in the hashed
  flag tuple) is rebuilt over an unchanged shader tree after the migration of
  its compile site
- **THEN** the blake2b cache key equals the pre-migration key and the cached
  `.spv` is reused without invoking `slangc`

### Requirement: Recorded byte-identity guarantees hold

The module and its consumers MUST preserve the recorded compiled-artifact
guarantees: (1) with `spectral=False`, all 28 RGB wavefront kernels and the
megakernel compile with define sets byte-identical to today's, so their
SPIR-V is byte-identical; (2) `SKINNY_MLT` is emitted only for wavefront MLT
kernel keys — the megakernel SPIR-V is byte-unchanged by the MLT axis; (3) no
Vulkan-target key ever emits a Metal-gated define, so Vulkan SPIR-V is
byte-unchanged by the Metal axes; (4) the default `NeuralBuildConfig` yields
zero `NF_*` defines and an empty neural slug, keeping the shipped kernels
byte-identical. This change SHALL introduce no new variant, no new define,
and no dispatch-behavior change.

#### Scenario: RGB wavefront kernels byte-identical

- **WHEN** the 28 RGB wavefront kernels are compiled through the module with
  `spectral=False`
- **THEN** each kernel's define tuple equals its pre-refactor tuple and the
  produced `.spv` bytes are identical

#### Scenario: Megakernel unaffected by the MLT axis

- **WHEN** the megakernel is compiled through the module in a session where
  MLT wavefront kernels are also built
- **THEN** the megakernel's define set contains no `SKINNY_MLT` and its
  SPIR-V is byte-identical to the pre-refactor artifact

#### Scenario: Vulkan SPIR-V unchanged by Metal-gated defines

- **WHEN** any Vulkan-target key's `slangc_defines()` is inspected across the
  full valid sweep
- **THEN** no member of `METAL_ONLY_DEFINES` appears, and the Vulkan SPIR-V
  for every key is byte-identical to its pre-refactor artifact
