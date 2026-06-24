## ADDED Requirements

### Requirement: Subsurface opacity refraction gate requires an interior medium

The loader bridge that lowers a material's `opacity` to `0` on account of a `subsurface` input SHALL fire ONLY for a material that carries a non-zero interior medium (`subsurface_sigma_a` or `subsurface_sigma_s`).

This opacity bridge exists so the flat path's delta-dielectric refraction branch
(`flat_material.slang: if (m.opacity < 1.0)`) fires for a pbrt subsurface
boundary; it MUST gate on exactly the materials that
`renderer._material_is_subsurface` classifies as `MATERIAL_TYPE_SUBSURFACE` and
routes through the volumetric interior walk.

A plain Autodesk `standard_surface` (or OpenPBR) `subsurface` *weight* with no
interior medium (e.g. the `three_materials_demo` marble: `subsurface = 0.4`,
`subsurface_color`, no σ_a/σ_s) is a diffuse subsurface-scattering shading term,
not a transmissive boundary. The bridge SHALL leave its `opacity` untouched so it
remains an opaque flat material; forcing `opacity = 0` there would turn it into a
clear dielectric that refracts the environment (rendering as a dark, near-black,
speckled ball that ignores the lights). The gate SHALL remain a no-op when
`subsurface` is absent/zero, when σ_a and σ_s are both zero, or when an explicit
`opacity` was already authored. This applies identically on both intake paths (the
native-USD parse and the `.mtlx` API fallback), every backend, execution mode and
integrator, because it is a host-side material-loading invariant.

#### Scenario: a standard_surface subsurface weight stays opaque

- **WHEN** a `standard_surface` material with `subsurface = 0.4` and no
  `subsurface_sigma_a`/`subsurface_sigma_s` is loaded
- **THEN** no `opacity` override is derived (the material stays opaque, routes to
  `MATERIAL_TYPE_FLAT`, and renders as a lit diffuse surface — the marble sphere
  shows its grey base with blue veining and responds to the environment)

#### Scenario: a genuine subsurface medium still opens the gate

- **WHEN** a material carrying a non-zero interior medium
  (`subsurface_sigma_a`/`subsurface_sigma_s`, e.g. an imported pbrt
  `Material "subsurface"`) with a `subsurface` weight is loaded and no explicit
  `opacity` was authored
- **THEN** `opacity` is set to `0` so the refraction boundary of the volumetric
  subsurface walk fires, unchanged from before this gate
