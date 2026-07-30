# Skinny — Papers and References

This document collects the core transport and shading references: MIS, BDPT,
SPPM, GGX, Fresnel, and the realistic camera.

It is not a complete bibliography. Each subject document carries its own
references for the technique it owns — [ReSTIR.md](ReSTIR.md),
[MetropolisLightTransport.md](MetropolisLightTransport.md),
[Spectral.md](Spectral.md), [NeuralGuiding.md](NeuralGuiding.md),
[SplineFlows.md](SplineFlows.md), and
[SkinRendering.md](SkinRendering.md) for the skin estimator chain.

---

## Papers and References

| Area | Files | Reference |
|------|-------|-----------|
| MIS | `samplers/mis_combine.slang`, `integrators/bdpt.slang` | Veach, "Robust Monte Carlo Methods for Light Transport Simulation", PhD thesis, 1997 |
| Bidirectional path tracing | `integrators/bdpt.slang` | Veach and Guibas, "Bidirectional Estimators for Light Transport", 1995 |
| Bidirectional path tracing | `integrators/bdpt.slang` | Lafortune and Willems, "Bi-Directional Path Tracing", 1993 |
| Stochastic progressive photon mapping | `integrators/wavefront_sppm.slang`, `integrators/sppm_state.slang` | Hachisuka and Jensen, "Stochastic Progressive Photon Mapping", SIGGRAPH Asia 2009 |
| GGX microfacet | `materials/flat/flat_shading.slang` | Walter, Marschner, Li, Torrance, "Microfacet Models for Refraction through Rough Surfaces", EGSR 2007 |
| Fresnel approximation | `materials/flat/flat_shading.slang` | Schlick, "An Inexpensive BRDF Model for Physically-Based Rendering", 1994 |
| Realistic camera | `lens_optics.py`, `shaders/cameras/thick_lens.slang` | Pharr, Jakob, Humphreys, *Physically Based Rendering 3e*, Ch. 6 |

Skin-, subsurface-, volume-, and head-geometry references live in
[SkinRendering.md](SkinRendering.md). Supporting techniques (ACES tone mapping,
PCG hashing, median-split BVH, Worley noise, Box-Muller sampling) are standard
implementation building blocks.
