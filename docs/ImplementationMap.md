# Skinny — Implementation Map

This document is the per-file map of the source tree: the Python package, the
Slang shader tree, and the tests.

It has two sections. *Implementation Map* annotates the modules a reader meets
most, one line of purpose each. *File Listing* names the rest of the tree, where
the file name is the answer. The two lists were separate documents until change
`docs-split-large-docs` merged them; the duplicate bare listings of
`src/skinny/` and `src/skinny/ui/` were dropped, because the annotated tables
above cover every file they named. `__init__.py` was the one file those
listings named that had no annotated row, so it gained one rather than being
lost.

For the renderer overview see [Architecture.md](Architecture.md). For the host
module ownership seams see [HostModules.md](HostModules.md).

---

## Implementation Map

### Python entry points

| File | Purpose |
|------|---------|
| `__init__.py` | Package root — declares `__version__` |
| `app.py` | GLFW shader-debug entry — keyboard + window only |
| `ui/qt/app.py` | `skinny-gui` Qt entry — `MainWindow`, viewport + docks |
| `web_app.py` | Panel web app, per-session renderer, Tornado video WebSocket |

### Renderer + scene

| File | Purpose |
|------|---------|
| `renderer.py` | Backend-neutral render orchestration, uniforms, environment/mesh/texture upload, frame loop |
| `scene.py` | Scene graph data classes (`MeshInstance`, `Material`, `Light*`, `Scene`) |
| `scene_intake.py` | USD stage to a `SceneUpdate` value — the one interface the renderer consumes |
| `usd_controls.py` | Applies a resolved USD control binding to a renderer (no GPU dependency) |
| `usd_loader.py` | USD stage to `Scene` conversion (with MaterialX API fallback) |
| `materialx_runtime.py` | MaterialX document loading, Slang code generation, uniform packing |
| `mesh.py` | OBJ loading, normalization, subdivision, displacement, BVH construction |
| `mesh_cache.py` | On-disk BVH cache (zstd-compressed vertex/index/BVH blobs) |
| `environment.py` | Built-in and HDR environment loading |
| `head_textures.py` | Detail map loading (normal, roughness, displacement) at 2048² |
| `presets.py` | Fitzpatrick I--VI presets; user presets are read-only |
| `tattoos.py` | Tattoo image loading |
| `params.py` | Shared parameter definitions (`ParamSpec`), get/set helpers |
| `settings.py` | User settings persistence (merge-on-write) |
| `session_snapshot.py` | The persisted session schema: shared + contributed keys, capture/restore |
| `fetch_hdrs.py` | Poly Haven HDRI download helper |
| `lens_optics.py` | PBRT-v3 thick-lens helpers (CPU exit-pupil bounding) |
| `bxdf_math.py` | CPU BSDF eval + lobe rasterisation for the BXDF visualiser |
| `gizmo.py` | Transform gizmo math (rotate/translate × world/local) + line-list buffer building |
| `debug_viewport.py` | Second-window camera/lens/wireframe debug renderer |
| `mtlx_graph_view.py` | View-model for MaterialX nodegraph editor |
| `scene_graph.py` | USD prim hierarchy tree model with typed editable properties |

### Backend abstractions

| File | Purpose |
|------|---------|
| `gfx/backend.py` | `Backend` ABC — shader target, caps, device, presenter |
| `gfx/device.py` | Device abstraction over queues / allocators |
| `gfx/presenter.py` | Surface / swapchain abstraction (None for headless) |
| `gfx/vulkan/` | Vulkan implementation of `Backend` / `Device` / `Presenter` |
| `gfx/metal/` | Legacy `Backend` abstraction stub; the production Metal path uses `metal_*.py` |
| `gpu_backend.py` | **Owns the seam**: `BackendCapabilities`, the one-sided/divergent tables, the adapter surface reader |
| `recording_compute.py` | Third adapter — records allocations/bindings/dispatches, executes nothing (hostless tests) |
| `backend_select.py` | Shared `auto` / Vulkan / Metal backend selection; resolves the adapter module by `backend_name` |
| `vk_context.py` | Vulkan instance, device, queue setup (windowed + headless) |
| `vk_compute.py` | Compute pipeline, descriptor layout, GPU buffer/image helpers |
| `vk_wavefront.py` | Vulkan wavefront execution passes |
| `metal_context.py` | Native Metal device, queue, and presentation setup |
| `metal_compute.py` | Native Metal compute pipeline and resource helpers |
| `metal_wavefront.py` | Native Metal wavefront execution passes |
| `hardware.py` | GPU enumeration, vendor detection, encoder selection |
| `video_encoder.py` | H264/JPEG encoding with hardware-aware fallback chain |

### UI

| File | Purpose |
|------|---------|
| `ui/spec.py` | Pure dataclass widget tree — no Qt / Panel imports |
| `ui/build_app_ui.py` | Builds the shared sidebar tree (used by Qt + Panel) |
| `ui/direction_math.py` | Light-direction picker math (shared math, no UI deps) |
| `ui/qt/backend.py` | Walks the spec tree, instantiates Qt widgets |
| `ui/qt/viewport.py` | `RenderViewport` Qt widget — embeds the renderer's offscreen image |
| `ui/qt/render_session.py` | Qt render-session config, GUI proxy, immutable snapshots, and command queue |
| `ui/qt/camera_input.py` | Mouse → camera mapping for the viewport |
| `ui/qt/direction_picker.py` | Hemisphere widget |
| `ui/qt/windows/scene_graph.py` | Scene graph inspector dock (tree above, properties below) |
| `ui/qt/windows/material_graph.py` | MaterialX nodegraph editor dock |
| `ui/qt/windows/bxdf.py` | BXDF visualiser dock with material picker |
| `ui/qt/windows/debug_viewport.py` | Camera debug viewport dock |
| `ui/panel/backend.py` | Walks the spec tree, instantiates Panel widgets |
| `ui/panel/windows.py` | Panel ports of scene graph / BXDF / material graph / debug viewport |

### Shaders (Slang)

| File | Purpose |
|------|---------|
| `main_pass.slang` | Primary camera path, progressive accumulation, tone mapping |
| `preview_pass.slang` | Material preview tile renderer |
| `common.slang` | Shared types, `FrameConstants`, `MtlxSkinParams` UBO layout |
| `bindings.slang` | Descriptor set bindings |
| `interfaces.slang` | `ISampler`, `IMaterial`, `ILight`, `IIntegrator` |
| `scene_trace.slang` | TLAS/BLAS ray traversal |
| `scene_lights.slang` | Light sampling (distant, sphere, rect, emissive tri) |
| `environment.slang` | Environment lookup and furnace fallback |
| `mtlx_std_surface.slang` | MaterialX `standard_surface` approximation |
| `mtlx_closures.slang` | MaterialX closure helpers |
| `mtlx_noise.slang` | MaterialX noise functions |
| `mtlx_gen_shim.slang` | Bindless `SamplerTexture2D` shim for generated MaterialX modules |
| `debug_line.slang` | Vertex/fragment pipeline for the debug viewport line list |
| `cameras/pinhole.slang` | Pinhole camera ray gen |
| `cameras/thick_lens.slang` | PBRT-v3 thick-lens ray gen |
| `materials/flat/flat_material.slang` | Flat (non-skin) BSDF: sample/evaluate via `IMaterial` |
| `materials/flat/flat_shading.slang` | Flat-material data loading, GGX helpers, procedural color |
| `materials/debug_normal_material.slang` | Normal visualisation `IMaterial` |
| `samplers/{ggx,lambert,uniform_sphere,henyey_greenstein,mis_combine}.slang` | Sampler library + MIS power heuristic |
| `sampling/{proposal,reuse}.slang` | Scene-sampling seam — directional-proposal mixture + reuse hook |
| `sampling/{neural_flow,neural_proposal}.slang` | Neural directional proposal — spline-flow inference (`neural_flow`) + renderer adapter (`neural_proposal`) |
| `lights/{sphere,emissive_triangle,directional}_light.slang` | `ILight` implementations |
| `integrators/{path,bdpt}.slang` | `IIntegrator` implementations |
| `integrators/{wavefront_sppm,sppm_state}.slang` | GPU SPPM — the 8 per-pass kernels (`wavefront_sppm`) + `VisiblePoint`/`SppmAccum` state and the spatial-hash + reduction helpers (`sppm_state`) |

## File Listing

### Backend abstraction (`src/skinny/gfx/`)

```
backend.py               device.py              presenter.py
pipeline.py              command.py             resources.py
shader_compiler.py       types.py
vulkan/{backend.py, device.py, presenter.py, command.py,
        resources.py, sync.py, _helpers.py}
metal/                   (placeholder)
```

### Web Templates (`src/skinny/web_templates/`)

```
video_player.html        WebCodecs decoder + camera controls (JS)
scene_tree.html          USD scene graph tree + property editor (web)
```

### SlangPile (`src/skinny/slangpile/`)

```
__init__.py              api.py                 types.py
registry.py              runtime.py             diagnostics.py
verification.py          cli.py                 compiler/__init__.py
compiler/module.py
README.md
```

### Shaders (`src/skinny/shaders/`)

```
common.slang             interfaces.slang        bindings.slang
main_pass.slang          preview_pass.slang
spectrum.slang           # hero-wavelength core (-DSKINNY_SPECTRAL variant only): SampledWavelengths,
                         # visible-λ importance sampling, Wilkie hero rotation, Jakob-Hanika RGB→spectrum
                         # upsampling (upsampleReflectance/upsampleIlluminant), Wyman CMF + XYZ→sRGB film resolve
scene_trace.slang        scene_lights.slang      nee.slang
mesh_head.slang          sdf_head.slang
environment.slang        volume_render.slang
mtlx_closures.slang      mtlx_std_surface.slang  mtlx_noise.slang
mtlx_gen_shim.slang      generated_materials.slang
debug_line.slang
cameras/{pinhole.slang, thick_lens.slang}
materials/debug_normal_material.slang
materials/flat/{flat_material.slang, flat_lobes.slang, flat_shading.slang}
                # flat_lobes: flatBsdfResponseSpectral = per-λ mirror of flatBsdfResponse (-DSKINNY_SPECTRAL)
materials/subsurface/{subsurface_walk.slang, medium.slang, volume_walk.slang,
                      cloud_noise.slang}  // pbrt volumetric SSS + free-standing NanoVDB media
                                          // + procedural cloud (classic Perlin fBm, MEDIUM_CLOUD)
materials/skin/{skin_material.slang, skin_bssrdf.slang, skin_shading.slang,
                skin_direct.slang, skin_ibl_specular.slang, skin_ibl_diffuse.slang,
                skin_volume.slang, skin_transmission.slang, skin_hair_sheen.slang,
                detail.slang}
samplers/{ggx.slang, lambert.slang, uniform_sphere.slang,
          henyey_greenstein.slang, mis_combine.slang}
sampling/{proposal.slang, reuse.slang,               # scene-sampling seam
          neural_flow.slang,                          # pure spline flow (coupling/RQ/MLP, fwd/inverse)
          neural_proposal.slang}                      # renderer adapter (weight buffers 33/34/35, world map)
lights/{sphere_light.slang, emissive_triangle_light.slang, directional_light.slang}
integrators/{path.slang, bdpt.slang,
             path_record.slang,                       # mainImageRecord training-record dump (5.1)
             path_spectral.slang}                     # SpectralPathTracer (-DSKINNY_SPECTRAL variant, float4 Spectrum carriers)
wavefront/{wavefront_path.slang, wavefront_bdpt.slang, wf_shade_common.slang,
           flat_bounce.slang, wavefront_state.slang,
           neural_proposal_pass.slang,                # WavefrontNeuralProposalPass pre-pass
           build_args.slang, scatter.slang, compaction.slang, indirect_paint.slang}
restir/{reservoir.slang, light_ris.slang, restir_primary.slang}
generated/                          # MaterialXGenSlang output, gitignored
lib/mx_closure_type.glsl
```

### MaterialX (`src/skinny/mtlx/`)

```
skinny_defs.mtlx                    skinny_skin_default.mtlx
genslang/skinny_genslang_impl.mtlx
genslang/skinny_skin_epidermis_genslang.slang
genslang/skinny_skin_dermis_genslang.slang
genslang/skinny_skin_subcut_genslang.slang
genslang/skinny_scattering_layer_genslang.slang
genslang/skinny_skin_layered_bsdf_genslang.slang
genslang/skinny_skin_layered_vdf_genslang.slang
genslang/python_materials/*.py     # SlangPile material drafts
genslang/slangpile_manifest.json
```

### Tests (`tests/`)

```
conftest.py              helpers.py               __init__.py
test_environment.py      test_headless.py         test_integration.py
test_intersections.py    test_lights.py           test_math.py
test_materialx_graph.py  test_mis.py              test_mtlx_closures.py
test_sampling.py         test_skin_optics.py      test_slangpile.py
test_slangpile_execution.py  test_struct_layout.py
test_ui_spec.py          test_volume.py           test_web.py
harnesses/test_common_harness.slang   harnesses/test_environment_harness.slang
harnesses/test_light_harness.slang    harnesses/test_sampler_harness.slang
harnesses/test_skin_harness.slang     harnesses/test_volume_harness.slang
kernels/energy_ref.py    kernels/sampling_ref.py
```
