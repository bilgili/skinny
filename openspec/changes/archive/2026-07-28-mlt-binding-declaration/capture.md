# Baseline capture — mlt-binding-declaration (tasks 1.1, 1.2)

Recorded from the pre-change sources at `e0dac42`. This is the state the move
must preserve exactly.

## 1.1 The pairing, from every source that states it

Byte sizes at a fixed budget: `num_chains=16384`, `bootstrap_samples=100000`
(scalar and `msl=True` are identical — every field is a 4-byte scalar).

| `size_key` | Vulkan binding | Metal global name | bytes |
|------------|----------------|-------------------|-------|
| `mlt_primary_samples`   | 52 | `mltPrimarySamples`   | 50 331 648 |
| `mlt_chain_meta`        | 53 | `mltChainMeta`        |    524 288 |
| `mlt_current_records`   | 54 | `mltCurrentRecords`   |  2 097 152 |
| `mlt_bootstrap_weights` | 55 | `mltBootstrapWeights` |    400 000 |
| `mlt_chain_seeds`       | 56 | `mltChainSeeds`       |     65 536 |
| `mlt_proposal_records`  | 57 | `mltProposalRecords`  |  2 097 152 |

Sources cross-checked, all in agreement today:

| Source | States | Agrees |
|--------|--------|--------|
| `shaders/common.slang:546` | `[[vk::binding(52)]] … mltPrimarySamples` | ✅ |
| `shaders/wavefront/wavefront_mlt.slang:72–84` | bindings 53–57 + names | ✅ |
| `gpu_resources.MLT_BINDINGS:433` | `(52…57)` | ✅ |
| `vk_wavefront.WavefrontMltPass._BINDINGS:1128` | `(binding, size_key)` | ✅ |
| `metal_wavefront.MetalWavefrontMltPass._BINDINGS:1155` | `(name, size_key)` | ✅ |
| `wavefront_layout.mlt_buffer_sizes:296` | the six keys + sizes | ✅ |
| **`vk_compute.py:688`** (**not in the proposal's table**) | `(52…57)`, for the set-0 descriptor-set **layout** | ✅ |

**Scope note — a sixth declaration site.** The proposal enumerated five. A
seventh source exists: `vk_compute._create_descriptor_set_layout` hardcodes
`(52, 53, 54, 55, 56, 57)` for the wavefront set-0 layout. It states binding
numbers only, but it is the same fact, and it is the site whose drift produces
the `fix-vulkan-volume-density-binding` failure (a binding the shader
references but the pipeline layout omits → MoltenVK `SPIR-V to MSL conversion
error: nullptr`). Folded into this change: it becomes a consumer of the same
table. The spec requirement's "none of them may carry its own binding table"
covers it.

Note `wavefront_mlt.slang` declares 53–56 contiguously at :72–75 and 57
separately at :84 (behind its own comment block); the parse must not assume a
contiguous run.

## 1.2 Emission order, both backends

**Vulkan — descriptor writes.** `SceneResourceSet.bind_vulkan`
(`gpu_resources.py:769`) emits, per descriptor set, in this order:

1. every non-spectral declaration in `VULKAN_WRITE_SEQUENCE`
2. **then** the six MLT dummy writes, iterating `MLT_BINDINGS` → 52, 53, 54,
   55, 56, 57, all pointing at the `record_counter` buffer
   (`MLT_DUMMY_ATTR`), only when `mlt_bindings=True`
3. then the spectral tail (`_SPECTRAL_TAIL`, bindings 45–51)

`WavefrontMltPass.descriptor_bindings` (`vk_wavefront.py:1191`) then hands the
renderer the real buffers as `(binding, StorageBuffer)` in `_BINDINGS` order —
52, 53, 54, 55, 56, 57.

`vk_compute._create_descriptor_set_layout` appends the six
`VkDescriptorSetLayoutBinding` entries in ascending order 52…57, after the
spectral block (45–51) and before `GRAPH_BINDING_BASE`.

**Metal — bind-by-name.** `MetalWavefrontMltPass.__init__`
(`metal_wavefront.py:1225–1229`) builds `self.buffers` keyed by `size_key` and
`self._bind_map` keyed by Slang global name, both iterating `_BINDINGS` in the
same order. `_bind_map` is a dict, so only its *content* is load-bearing, not
its order; `self.buffers` likewise. Order is preserved anyway by iterating the
one table.

**Invariant for the move:** one table, iterated in declaration order 52→57,
reproduces all four emission sites byte for byte.
