# Skinny — Documentation Index

Every reference document at the top level of `docs/` is listed here. Each one
owns a subject: a change updates the document that owns what it touched. The
nested directories are generated artifacts rather than reference documents —
`docs/diagrams/` holds the SVG and equation generators with their result
reports, `docs/superpowers/` records history — so they are not enumerated
below. Start at
[Architecture.md](Architecture.md) for the renderer overview, or at
[../README.md](../README.md) for installation and the command lines.

A reference document stays at or below 700 lines. When one grows past that, it
is split at a subject boundary and the new document is registered here.

## Renderer internals

| Document | Owns |
|---|---|
| [Architecture.md](Architecture.md) | The hub: high-level pipeline, GPU execution flow, key invariants, and the map of these documents |
| [ShaderPipeline.md](ShaderPipeline.md) | Pluggable Slang interfaces, the material and integrator pipeline, MaterialX nodegraph codegen, environment importance sampling, SlangPile |
| [SceneSystem.md](SceneSystem.md) | USD intake, scene graph, instancing, lights, textures, skinning, camera / lens / debug viewport |
| [GpuResources.md](GpuResources.md) | Descriptor binding map, GPU resource inventory, host-mirrored byte layouts, shader variant key, `FrameConstants` layout |
| [HostModules.md](HostModules.md) | Python module map, front-end bring-up, the renderer carve-out pattern, the device-free pure core |
| [Backends.md](Backends.md) | Backend selection, `MetalContext`, the Vulkan path, the `gfx/` abstraction |
| [FrontEnds.md](FrontEnds.md) | Headless render API, web application, display tail (exposure, tone map, tool readback) |
| [ImplementationMap.md](ImplementationMap.md) | The per-file map of the Python package, the shader tree, and the tests |

## Execution modes and integrators

| Document | Owns |
|---|---|
| [RenderingModes.md](RenderingModes.md) | What the renderer can be told to do: backend, resolution, the compatibility matrix, the sampling modes, furnace mode |
| [Megakernel.md](Megakernel.md) | The one-dispatch execution mode |
| [Wavefront.md](Wavefront.md) | The staged execution mode: queues, material bucketing, per-stage kernels |
| [ReSTIR.md](ReSTIR.md) | ReSTIR direct-lighting reuse: reservoirs, RIS, GRIS |
| [PhotonMapping.md](PhotonMapping.md) | The GPU SPPM integrator |
| [MetropolisLightTransport.md](MetropolisLightTransport.md) | PSSMLT over BDPT: chains, bootstrap, film splats |
| [Spectral.md](Spectral.md) | Hero-wavelength spectral rendering |

## Materials, skin, and volumes

| Document | Owns |
|---|---|
| [SkinRendering.md](SkinRendering.md) | The three-layer skin model and the §1–§6 estimator chain |
| [Subsurface.md](Subsurface.md) | Subsurface transport and the interior random walk |

## Neural guiding

| Document | Owns |
|---|---|
| [NeuralGuiding.md](NeuralGuiding.md) | The SplineFlow directional proposal: equations, network, precision, verification |
| [SplineFlows.md](SplineFlows.md) | The first-principles theory companion to the guiding model |
| [OnlineTraining.md](OnlineTraining.md) | The online-training loop and the weight handoff |

## Tooling and interoperability

| Document | Owns |
|---|---|
| [PythonAPI.md](PythonAPI.md) | The public Python surface |
| [PbrtImport.md](PbrtImport.md) | The pbrt v4 scene importer and its feature parity |
| [ParityHarness.md](ParityHarness.md) | The parity matrix, the dual gate, the confirming-scene suite |
