# shader-byte-layouts Specification

## Purpose
TBD - created by archiving change reflection-owned-byte-layouts. Update Purpose after archive.
## Requirements
### Requirement: Single derived authority for host-mirrored byte layouts

One layout module SHALL own every byte layout the host mirrors from a Slang
struct — the `FrameConstants` uniform block (base and `SKINNY_MLT` variants,
including the `tileOriginY` patch offset and the Vulkan uniform-buffer size
bound), `StdSurfaceParams`, `FlatMaterialParams`, and the wavefront-family
records (`WavefrontPathState`, `RecVertex`, `VisiblePoint`, `SppmAccum`,
`BDPTVertex`, `WfBdptAux`, and the MLT chain structs, in RGB and
`SKINNY_SPECTRAL` variants) — deriving each struct's ordered field list from
the authoritative `.slang` declaration and computing scalar (Vulkan
`-fvk-use-scalar-layout`) and MSL (Metal) offsets and strides from it.
Packers and buffer allocators SHALL consume the module; the codebase SHALL NOT
carry a hand-maintained field/offset table that re-authors a layout the module
owns. The module SHALL resolve exactly the preprocessor gates the mirrored
structs use — `SKINNY_SPECTRAL`, `SKINNY_MLT`, and `SKINNY_METAL` — per
queried variant, and MUST fail loudly (raise) on any other gate or any struct
declaration it cannot fully classify; it MUST NOT emit a guessed offset. For
the `FrameConstants` host scalar blob the module SHALL apply the registered
blob rule — the variant's declared fields in order with `tileOriginY` always
present and relocated to the tail (after the MLT tail when present) — because
declaration order is not blob order: under an MLT pack `mltSigma` MUST land
at the offset the Vulkan MLT SPIR-V expects (immediately after
`sppmGroupPmfEnv`), and on Vulkan variants whose compiled struct lacks
`tileOriginY` the trailing word is benign filler within the oversized uniform
buffer.

#### Scenario: FrameConstants tables are derived, not hand-authored

- **WHEN** the renderer packs the `fc` uniform block for either backend after
  this change lands
- **THEN** the ordered field names and scalar sizes it walks (base and MLT
  variants), the `tileOriginY` patch offset, and the Vulkan uniform-buffer
  size bound are all queried from the layout module's parse of the
  `FrameConstants` declaration, and no hand-written `_FC_*` field table
  remains in the renderer

#### Scenario: Material param packers consume derived layouts

- **WHEN** a `StdSurfaceParams` or `FlatMaterialParams` record is packed
  (scalar) or MSL-relocated
- **THEN** the field order, per-field offsets, and record strides come from
  the layout module, and the packed record lengths equal the module's derived
  scalar strides

#### Scenario: Wavefront allocators size from derived field lists

- **WHEN** a wavefront, SPPM, BDPT, or MLT buffer size is computed for any
  (backend layout, spectral) combination
- **THEN** the stride is computed from the field list the module parsed out of
  the owning `.slang` source — resolving `#if defined(SKINNY_SPECTRAL)` per
  variant — through the existing public sizer API, with no hand-listed field
  mirror remaining in the sizing module

#### Scenario: MLT blob order preserved by the blob rule

- **WHEN** the module derives the `FrameConstants` scalar-blob field table
  for the MLT variant
- **THEN** the table ends with the MLT tail fields followed by `tileOriginY`
  last, `mltSigma`'s running offset equals the offset pinned by the MLT host
  blob-order lock, and the MLT blob is exactly 32 bytes longer than the base
  blob

#### Scenario: Unparseable struct fails instead of guessing

- **WHEN** a mirrored Slang struct gains a declaration form or field type the
  layout module does not recognize
- **THEN** querying that struct's layout raises an error naming the struct and
  the unrecognized text, and the hostless layout test sweep fails

### Requirement: Layout drift fails hostless tests

The system SHALL surface any divergence between a mirrored Slang struct and a
host packer or allocator as a failing test that runs without a GPU (plain
`pytest`, no device, no gpu marker). The hostless gates SHALL check, per struct and
variant: (a) the derived field list covers the packed record with no gaps or
overlaps, (b) for every hostlessly constructible packer (the material param
packers, which take a plain object) the packer's output length equals the
derived scalar stride, and (c) each derived stride equals a pinned golden
value recorded in the tests, so that a simultaneous parser-and-shader drift
still trips a human-visible failure. For `FrameConstants` — whose packer is
bound to a constructed renderer and whose module import requires the Vulkan
SDK, so packer-invoking checks cannot be hostless — the hostless gates SHALL
instead pin the derived blobs directly (golden base and MLT blob lengths and
a golden field-order lock, including the re-pointed MLT blob-order
assertions), and the blob↔packer coverage equality SHALL be enforced as a
runtime guard at every uniform pack site (extending the existing Metal
relocation drift-guard to the Vulkan upload path). Existing hostless layout
tests (the `wavefront_layout` stride/field locks, `test_struct_layout`,
`test_sppm_state`, the `test_mlt_host` blob-order locks)
SHALL be preserved at full strength — pointed at the
module where they read offsets, never deleted or loosened — and the existing
gpu-marked Metal reflection lock tests SHALL be retained as the ground truth
for the MSL layout rules and extended to the uniform block and material
param structs.

#### Scenario: Field reorder in a shader struct fails hostlessly

- **WHEN** two fields of a mirrored Slang struct are swapped in the `.slang`
  source and the hostless test sweep runs with no GPU present
- **THEN** at least one layout gate fails (pinned-stride, pinned field-order,
  coverage, or packer-length equality) before any GPU dispatch could read the
  drifted layout

#### Scenario: Golden strides pin the derivation

- **WHEN** the layout module's derivation changes such that any mirrored
  struct's scalar or MSL stride differs from its recorded golden value
- **THEN** a hostless test fails and the golden value must be consciously
  re-measured and updated in the same change

#### Scenario: MSL ground truth stays gpu-locked

- **WHEN** the gpu-marked layout lock tests run on a Metal host
- **THEN** the module's computed MSL offsets and strides for the locked
  structs (including the `fc` uniform block and `StdSurfaceParams`) equal the
  offsets and strides reported by the compiled program's live reflection

### Requirement: Layout-authority adoption is byte-invariant

Adopting the derived layout module SHALL NOT change any byte the GPU sees:
the Vulkan SPIR-V SHALL remain byte-unchanged (no `.slang` source is
modified), every packer SHALL produce byte-identical output to its
pre-change form, and rendered output SHALL be bit-identical at equal
settings. On the Metal path the runtime uniform packer SHALL continue to
take its offsets from the compiled module's live reflection (per the
`metal-backend` capability) and SHALL cross-check that live reflection
against the derived layout so a mismatch in either direction fails loudly
rather than mis-packing.

#### Scenario: Vulkan SPIR-V byte-unchanged

- **WHEN** the megakernel and wavefront shaders are compiled after this
  change lands
- **THEN** every produced `.spv` is byte-identical to its pre-change form

#### Scenario: Packed bytes and rendered output identical

- **WHEN** the same scene and settings are packed and rendered before and
  after a migration stage lands
- **THEN** the stage's packer outputs are byte-identical and the accumulated
  image is bit-identical on the verified backend/integrator combinations

#### Scenario: Live reflection cross-checked on Metal

- **WHEN** the Metal uniform packer relocates the scalar blob using the
  compiled pipeline's reflected offsets
- **THEN** those reflected offsets and the reflected struct size are asserted
  equal to the layout module's derived MSL layout, and a disagreement raises
  instead of uploading a mis-packed blob

