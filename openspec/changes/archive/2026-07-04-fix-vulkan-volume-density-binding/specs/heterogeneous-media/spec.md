## ADDED Requirements

### Requirement: the density-field binding is declared on every Vulkan pipeline layout

The `volumeDensity` density-field texture (set 0, binding 26) SHALL be declared
in the Vulkan descriptor-set layout produced by
`ComputePipeline._create_descriptor_set_layout` — the layout shared by the
megakernel driver pipeline and every wavefront stage pipeline (via
`scene_bindings_only`) — as a combined image sampler, matching the
`[[vk::binding(26)]] Sampler3D<float>` declaration in `bindings.slang`. More
generally, every `[[vk::binding(N)]]` declaration active in the Vulkan branch
of `bindings.slang` SHALL have a corresponding entry in that layout: a shader
that references a binding absent from the pipeline layout is undefined
behaviour on Vulkan and a hard `SPIR-V to MSL conversion error: nullptr`
pipeline-build failure on MoltenVK.

#### Scenario: Vulkan pipelines build on MoltenVK

- **WHEN** a `VulkanContext`-backed `Renderer` compiles the megakernel driver
  pipeline (and subsequently the wavefront SPPM/path/BDPT stage pipelines) on
  macOS/MoltenVK
- **THEN** pipeline creation succeeds with no
  `VUID-VkComputePipelineCreateInfo-layout-07988` validation error and no
  MoltenVK SPIR-V→MSL conversion failure, and the SPPM energy-vs-path GPU gate
  (`tests/test_sppm_gpu.py`) runs to a verdict

#### Scenario: shared scene-set bindings are audited hostlessly

- **WHEN** the hostless binding audit compares the Vulkan-branch
  `[[vk::binding(N)]]` declarations of `bindings.slang` against the
  `_create_descriptor_set_layout` binding list
- **THEN** every shader-declared binding is present in the layout (the
  conditional MaterialX graph-param binding 25 counts as declared), so a new
  shared scene binding cannot ship without its Vulkan layout entry
