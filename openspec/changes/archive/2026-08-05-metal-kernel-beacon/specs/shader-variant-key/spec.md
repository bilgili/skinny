# Shader variant key

## ADDED Requirements

### Requirement: A Metal-only trace axis carries the beacon session define

`ShaderVariantKey` SHALL gain a boolean `metal_trace` axis that emits the
`SKINNY_METAL_TRACE` define into the Metal session defines. The axis SHALL ride
the Metal `base` define segment and reach `session_defines()` ONLY, exactly like
`metal_neural` and `metal_records`. It SHALL NOT touch `cache_token()` or
`spv_cache_key()`, because those name and hash the Vulkan `.spv` disk artifact and
a traced build is always a Metal in-process compile. There is no Metal `.spv`
disk cache, so no on-disk collision is possible.

The axis SHALL be valid only on Metal-target keys. A Vulkan-target key with
`metal_trace` set SHALL be refused, matching the `__post_init__` rule for
`metal_neural` and `metal_records`. The axis SHALL default off, so the default
variant is the current production variant. `shader_variants.py` SHALL own the
`SKINNY_METAL_TRACE` define; no compile site SHALL hand-assemble it.

Unlike the wavefront-only `metal_neural` and `metal_records` gates, `metal_trace`
SHALL be valid on ALL Metal families — megakernel, wavefront, preview, and
debug-raster — because the beacon binds to every compute kernel. A `metal_trace`
key SHALL NOT be refused on the megakernel, preview, or debug-raster family.

#### Scenario: a traced Metal session defines the beacon
- **WHEN** the backend builds a Metal kernel with `metal_trace` on and again with
  `metal_trace` off
- **THEN** the traced session's `session_defines()` contains `SKINNY_METAL_TRACE`
  and the production session's `session_defines()` does not

#### Scenario: metal_trace is valid on every Metal family
- **WHEN** a `ShaderVariantKey` sets the Metal target with `metal_trace` on for the
  megakernel, wavefront, preview, or debug-raster family
- **THEN** the key is accepted, because the beacon binds to every compute kernel

#### Scenario: a Vulkan key rejects the trace axis
- **WHEN** a `ShaderVariantKey` sets the Vulkan target together with `metal_trace`
- **THEN** the key is refused as invalid, matching the existing Metal-only-axis
  rule for `SKINNY_METAL_NEURAL` and `SKINNY_METAL_RECORDS`
