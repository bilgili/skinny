## Why

Every Vulkan pipeline build is broken at HEAD on MoltenVK: `VulkanContext`
renders die at the megakernel driver-pipeline compile with

```
[mvk-error] SPIR-V to MSL conversion error: nullptr
[mvk-error] VK_ERROR_INITIALIZATION_FAILED: Compute shader function could not be compiled into pipeline.
```

and the validation layer names the root cause exactly:

```
VUID-VkComputePipelineCreateInfo-layout-07988: SPIR-V uses descriptor
[Set 0, Binding 26, variable "volumeDensity"] but the binding was not
declared in the VkPipelineLayoutCreateInfo::pSetLayouts[0].
```

Commit `aa9d70e` (nanovdb-volume-rendering) added the heterogeneous-medium
density field `volumeDensity` — `[[vk::binding(26)]] Sampler3D<float>` in
`bindings.slang` — to the shaders, to the renderer's descriptor *writes*
(`renderer.py` binds the 1×1×1 zero fallback at binding 26, "always bound"),
and to the descriptor-*pool* sizing ("volume density grid (6)"). But it never
added binding 26 to `ComputePipeline._create_descriptor_set_layout`
(`vk_compute.py`) — the hand-maintained set-0 layout list still enumerates
0–24, optional 25, 30/31, 33–37. The megakernel SPIR-V references binding 26
unconditionally (the medium walk is compiled in), so on MoltenVK — which
builds its SPIR-V→MSL resource map from the pipeline layout — the missing
binding makes SPIRV-Cross return a null mapping and the pipeline create fails
hard. On a driver that tolerates the mismatch it is still undefined behaviour
(`vkUpdateDescriptorSets` also writes binding 26 into a set whose layout does
not declare it).

Nobody noticed because the standing GPU gates run on the native Metal backend
(which binds by name, no Vulkan layout). The only gpu tests that stand up a
raw `VulkanContext` are `tests/test_sppm_gpu.py::test_sppm_builds_and_renders_finite`
and `::test_sppm_energy_matches_path_tracer` — both have been failing since
Jul 2, which is how a zero-photon-flux SPPM regression shipped unnoticed
(fixed separately in fix-sppm-bathroom-black-walls). The reported "wavefront
SPPM pipeline" suspicion was a red herring: the tests die at `Renderer`
construction, before `WavefrontSppmPass` is ever built, and all eight
`wfSppm*` kernels convert to MSL cleanly in isolation.

## What Changes

- **Declare binding 26 in the Vulkan set-0 layout.** Add a
  `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER` entry for binding 26
  (`volumeDensity`, `Sampler3D<float>`) to
  `ComputePipeline._create_descriptor_set_layout`. This is the megakernel
  layout AND the shared wavefront set-0 layout (`scene_bindings_only`), so one
  entry fixes every pipeline. Pool sizing already counts it; descriptor writes
  already exist; no shader change, no Metal change, no layout/stride change.
- **Hostless layout-audit regression test.** New test parses the Vulkan-branch
  `[[vk::binding(N)]]` declarations out of `bindings.slang` (mini-preprocessor
  with `SKINNY_METAL` undefined) and the `binding=N` entries out of
  `_create_descriptor_set_layout`, and asserts every shared scene-set shader
  binding is declared in the layout. Fails at HEAD (26 missing); pins the
  "shader declares it ⇒ Vulkan layout declares it" contract so the next
  binding added to `bindings.slang` cannot repeat this failure mode.
- **Re-arm the Vulkan SPPM gate.** With the layout fixed, the two failing
  `test_sppm_gpu.py` tests run again on macOS/MoltenVK — no port to the Metal
  path needed; they stay the raw-Vulkan gate they were designed to be.

## Impact

- Affected specs: `heterogeneous-media` (density-field binding contract).
- Affected code: `src/skinny/vk_compute.py` (one layout entry), new
  `tests/test_vk_binding_layout.py` (hostless audit).
- Vulkan behaviour: pipelines that failed to build now build; on drivers that
  tolerated the mismatch the render output is unchanged (the descriptor write
  and shader were already in place — only the layout declaration was missing).
- Metal: untouched (binds by name; `volumeDensity` already in the argument
  table per design D8).
- Gates: `tests/test_sppm_gpu.py -m gpu` green again on this host; SPPM
  energy-vs-path regression protection restored.
