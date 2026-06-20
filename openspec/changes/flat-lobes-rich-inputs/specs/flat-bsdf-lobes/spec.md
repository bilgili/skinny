## ADDED Requirements

### Requirement: Unified lobe BSDF consumes standard_surface tint/roughness inputs

The unified flat / `std_surface` lobe BSDF SHALL consume the `transmission_color`,
`specular_color`, and `diffuse_roughness` material inputs when present, filling
the **existing** `{coat, spec, diffuse, delta-transmission}` lobe set without
adding a new lobe and without invoking `evalStdSurfaceBSDF` (which remains
preview-only). Specifically: the delta transmission branch SHALL tint by
`transmission_color`; the specular GGX lobe response SHALL multiply by
`specular_color`; and the diffuse lobe SHALL use the Oren-Nayar response driven by
`diffuse_roughness`. Each SHALL be a **weight/response-only** change that leaves
the solid-angle pdf of every lobe unchanged, so `sample().pdf == evaluate().pdf`
continues to hold and `response/pdf` stays the bounded native per-lobe weight (no
clamp).

These inputs SHALL be back-compatible: when an input is absent the BSDF SHALL
reproduce the prior behavior exactly — `transmission_color` defaults to the
material `albedo`, `specular_color` defaults to white, and `diffuse_roughness = 0`
yields exact Lambert — so existing UsdPreviewSurface renders and the pbrt parity
corpus are byte-unchanged.

#### Scenario: colored dielectric tints transmitted radiance

- **WHEN** a `dielectric` material carries a non-white `transmission_color` (e.g.
  via the `-mtlx` export, which UsdPreviewSurface cannot represent) and is hit by
  a path/BDPT ray
- **THEN** the delta-refracted throughput is tinted by `transmission_color`
  (not the achromatic/`albedo` weight), and the converged image matches a pbrt v4
  colored-glass reference within the scene's relMSE/FLIP tolerance

#### Scenario: absent rich inputs leave existing renders unchanged

- **WHEN** a material carries no `transmission_color` / `specular_color` /
  `diffuse_roughness` (the UsdPreviewSurface case)
- **THEN** the rendered result is byte/behaviour-identical to the pre-change
  unified BSDF, and the pbrt parity corpus scenes show ≈ 0 relMSE delta

#### Scenario: tint and roughness inputs do not break pdf symmetry

- **WHEN** a material with a non-white `specular_color` and non-zero
  `diffuse_roughness` (Oren-Nayar) is hit and a non-delta direction `wi` is drawn
- **THEN** `sample().pdf` and `evaluate().pdf` for that `(wo, wi)` are equal to
  floating-point tolerance, and `evaluate().response / evaluate().pdf` stays
  bounded (no firefly, no clamp introduced)

#### Scenario: four-way convergence preserved

- **WHEN** `three_materials` renders IBL-only at high sample count under PT-BSDF,
  PT-(BSDF+Env), PT-Env, and BDPT, in both megakernel and wavefront modes
- **THEN** all four still converge to one per-column radiance (matching the
  pre-change reference for the unchanged-input materials), on both Metal and
  Vulkan
