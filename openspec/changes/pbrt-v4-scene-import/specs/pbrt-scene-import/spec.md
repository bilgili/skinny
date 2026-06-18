## ADDED Requirements

### Requirement: pbrt v4 text parsing

The importer SHALL parse the pbrt v4 text scene format into a structured
directive stream: directives, quoted strings, numeric and boolean values,
bracketed arrays, `#` comments, and typed parameters of the form
`"type name" [values]` (`float`, `integer`, `point2`/`point3`, `vector`,
`normal`, `rgb`, `spectrum`, `blackbody`, `bool`, `string`, `texture`). It SHALL
resolve `Include` and `Import` recursively with paths relative to the including
file, and SHALL separate the pre-`WorldBegin` options block (camera, film,
sampler, integrator, accelerator, color space) from the post-`WorldBegin` world
block.

#### Scenario: Typed parameters and arrays parse
- **WHEN** a directive carries `"float roughness" [0.1]` and `"rgb reflectance" [0.2 0.3 0.4]`
- **THEN** the importer exposes `roughness=0.1` and `reflectance=(0.2,0.3,0.4)` with their declared types

#### Scenario: Includes resolve relative to the scene
- **WHEN** a scene file contains `Include "geometry/mesh.pbrt"`
- **THEN** the importer reads that file relative to the scene directory and inlines its directives in place

#### Scenario: Malformed input fails with location
- **WHEN** the file contains an unterminated array or unknown parameter type
- **THEN** the importer raises an error naming the file and line/token rather than silently producing a partial scene

### Requirement: Graphics-state semantics

The importer SHALL reproduce pbrt's graphics-state model: a current-transform
(CTM) stack pushed/popped by `AttributeBegin`/`AttributeEnd` and
`TransformBegin`/`TransformEnd`; named coordinate systems
(`CoordinateSystem`/`CoordSysTransform`); the current material, named materials
(`MakeNamedMaterial`/`NamedMaterial`), current named textures, the current area
light, and reverse-orientation state; and object instancing via
`ObjectBegin`/`ObjectEnd`/`ObjectInstance`.

#### Scenario: Transform scope is restored
- **WHEN** geometry is emitted inside an `AttributeBegin`/`AttributeEnd` block that applies a `Translate`
- **THEN** that translation affects only geometry inside the block and the CTM is restored afterward

#### Scenario: Object instances share geometry under distinct transforms
- **WHEN** an `ObjectBegin "o"` … `ObjectEnd` is referenced by two `ObjectInstance "o"` directives with different CTMs
- **THEN** the output contains two instances of the same geometry, each under its own world transform

### Requirement: Coordinate, winding, and unit conversion

The importer SHALL convert pbrt's left-handed coordinate space into skinny's
right-handed USD world via a fixed change-of-basis applied to every CTM, keeping
triangle winding and shading normals consistent, and SHALL convert `LookAt` into
a camera-to-world transform in that space. It SHALL carry pbrt's unitless scale
into the USD stage and set the scene scale so light fall-off and thick-lens
geometry remain self-consistent.

#### Scenario: Known point projects to the expected pixel
- **WHEN** a world point of known pbrt coordinates is transformed through the converted camera
- **THEN** it projects to the same normalized image location pbrt would produce (within a golden tolerance)

#### Scenario: Winding is preserved
- **WHEN** a front-facing pbrt triangle is converted
- **THEN** its USD winding/normal still faces the camera (no inside-out geometry from the handedness flip)

### Requirement: Shape translation

The importer SHALL translate `trianglemesh`, `plymesh` (reading the referenced
`.ply`), and `sphere` shapes into UsdGeom prims with positions, indices, and —
when present — normals and UVs, under the accumulated CTM. Instanced shapes SHALL
be emitted as multiple loadable instances of shared geometry.

#### Scenario: Triangle mesh round-trips to a loadable mesh
- **WHEN** a `trianglemesh` with `P`, `indices`, and `uv` is translated
- **THEN** the emitted USD mesh carries matching points, face indices, and UVs and loads through `usd_loader` as a `MeshInstance`

#### Scenario: ply asset is read
- **WHEN** a `plymesh "filename" "mesh.ply"` references an ascii or binary PLY
- **THEN** the importer reads its vertices/faces and emits an equivalent USD mesh

### Requirement: Material translation

The importer SHALL translate pbrt materials onto the closest skinny target
(flat lobe set or MaterialX `standard_surface`), per the design mapping table:
`diffuse`, `conductor`, `dielectric`, `thindielectric`, `coateddiffuse`,
`coatedconductor`, and `diffusetransmission`. It SHALL replicate pbrt v4's
roughness→alpha remap (including `remaproughness false` and `uroughness`/
`vroughness`), reduce conductor complex IOR (η,k) to an RGB normal-incidence
reflectance, and route transmission to skinny's opacity/IOR refraction gate.
Unrecognized materials SHALL be translated best-effort and recorded in the report.

#### Scenario: Diffuse maps to Lambert albedo
- **WHEN** a `"diffuse"` material has `"rgb reflectance" [0.8 0.2 0.2]`
- **THEN** the emitted material has diffuse base color (0.8,0.2,0.2) and metallic 0

#### Scenario: Conductor roughness remap is applied
- **WHEN** a `"conductor"` material specifies `"float roughness" [r]` with default `remaproughness`
- **THEN** the emitted GGX alpha equals pbrt v4's remapped value for `r` (unit-tested against known pbrt outputs), and `remaproughness false` passes `r` through as alpha

#### Scenario: Dielectric drives the refraction path
- **WHEN** a `"dielectric"` material has `"float eta" [1.5]`
- **THEN** the emitted material sets IOR 1.5 and opens skinny's transmission/refraction gate

### Requirement: Spectrum to RGB reduction

The importer SHALL reduce every pbrt spectrum — named spectra, `blackbody [T]`,
sampled `.spd`, and RGB-as-reflectance/-illuminant — to linear RGB by integrating
against CIE XYZ under the scene's rendering color space, defaulting to sRGB/
Rec709 when no `ColorSpace` directive is present. The residual RGB-vs-spectral
divergence SHALL be documented, not hidden.

#### Scenario: Blackbody illuminant becomes RGB radiance
- **WHEN** a light specifies `"blackbody L" [6500]`
- **THEN** the importer emits a linear-RGB radiance consistent with a 6500 K white under the default color space

#### Scenario: Named conductor spectrum reduces to RGB reflectance
- **WHEN** a conductor references a named metal spectrum (e.g. `metal-Au-eta`/`metal-Au-k`)
- **THEN** the importer emits an RGB base color from the Fresnel reflectance at normal incidence

### Requirement: Light translation

The importer SHALL translate `distant`→`DistantLight`, `point`→`SphereLight`
(small radius), `infinite`→`DomeLight` (emitting/linking an equirect `.hdr` and
verifying integrated energy), and area (`diffuse`) emitters on shapes into
skinny's emissive-mesh light, honoring one-sided vs `twosided`. Light radiance
SHALL follow pbrt's `L`/`scale`/`power`/`illuminance` semantics reduced to
linear-HDR RGB. Lights skinny cannot represent (e.g. `spot`) SHALL be translated
best-effort and flagged in the report and parity matrix.

#### Scenario: Infinite light becomes a usable dome
- **WHEN** an `infinite` light references an environment map
- **THEN** the importer emits a `DomeLight` skinny loads as `LightEnvHDR` with energy matching the source map

#### Scenario: Area light sidedness is preserved
- **WHEN** an `AreaLightSource "diffuse"` without `twosided` is attached to a shape
- **THEN** the emitted emissive mesh radiates from the front face only

#### Scenario: Unsupported light is flagged, not faked
- **WHEN** a `spot` light is encountered
- **THEN** the importer emits a best-effort approximation and records it as "approx" in the report rather than silently emitting a matching light

### Requirement: Camera translation

The importer SHALL translate `perspective` cameras, converting pbrt's `fov`
(defined on the shorter image axis) into skinny's vertical FOV + vertical
aperture accounting for aspect ratio and `screenwindow`, and SHALL translate
`realistic` lens descriptions into `LensSystem`/`LensElement`. Depth-of-field
parameters (`lensradius`/`aperturediameter`, `focaldistance`) SHALL map to
`fstop`/`focus_distance`.

#### Scenario: fov converts on the correct axis
- **WHEN** a `perspective` camera has `"float fov" [40]` on a non-square film
- **THEN** the emitted camera's vertical FOV matches pbrt's framing for that aspect ratio (shorter-axis fov honored)

#### Scenario: Realistic lens maps to the thick-lens system
- **WHEN** a `realistic` camera references a lens description
- **THEN** the importer emits ordered `LensElement`s (signed radius, thickness, IOR, aperture, aperture-stop) loaded as a `LensSystem`

### Requirement: Participating media and subsurface (best-effort)

The importer SHALL translate homogeneous participating media and `subsurface`
materials best-effort onto skinny's homogeneous volume / material path, carrying
parameters via `customData` where UsdPreviewSurface cannot. Heterogeneous
(grid/VDB) media SHALL be detected and flagged unsupported rather than emitted
incorrectly.

#### Scenario: Homogeneous medium carries its coefficients
- **WHEN** a `MakeNamedMedium "m" "type" "homogeneous"` with absorption/scattering is bound to geometry
- **THEN** the emitted scene carries those coefficients to skinny's volume path (directly or via `customData`)

#### Scenario: Heterogeneous medium is flagged
- **WHEN** a grid/VDB medium is encountered
- **THEN** the importer records it as "unsupported" in the report and does not emit a wrong homogeneous stand-in

### Requirement: CLI and Python entry point

The importer SHALL expose a console command `skinny-import-pbrt <scene.pbrt> -o
<out.usda>`, a `python -m skinny.pbrt` module entry, and a `skinny.pbrt`
public function that returns/writes a USD stage. The emitted USD SHALL load
through the existing `usd_loader` with no loader changes.

#### Scenario: CLI emits a loadable stage
- **WHEN** `skinny-import-pbrt scene.pbrt -o scene.usda` is run
- **THEN** `scene.usda` is written and `Renderer(usd_scene_path="scene.usda")` loads the expected instances, lights, and camera

### Requirement: Translation report

Each import SHALL produce a human-readable report classifying every translated
construct as exact, approximated, or skipped, with a reason for anything not
exact, keyed to the parity matrix.

#### Scenario: Approximations and skips are listed
- **WHEN** a scene contains a `spot` light and a heterogeneous medium
- **THEN** the report lists the spot light as "approx" and the medium as "skipped/unsupported" with reasons
