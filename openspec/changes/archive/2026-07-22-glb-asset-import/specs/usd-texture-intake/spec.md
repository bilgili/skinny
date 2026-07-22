# usd-texture-intake

## ADDED Requirements

### Requirement: Interface-connected texture file inputs resolve
The USD loader SHALL resolve a `UsdUVTexture` `file` input whose value is authored as a connection to a Material-prim interface input (rather than a locally authored asset value), by following the connection chain to the attribute that produces the asset value. The walk SHALL be bounded and SHALL yield the same `TextureBinding` (path, channel selector, colorspace, wrap modes, scale/bias) as an equivalent locally authored `file` value. When no asset value materializes at the end of the chain, the binding SHALL be skipped exactly as an unconnected, unauthored `file` input is today.

#### Scenario: Apple usdextract-converted GLB resolves all textures
- **WHEN** a stage authored by Apple's glTF→USD conversion is loaded, where every `UsdUVTexture.inputs:file` is connected to a Material interface input (e.g. `file <- Material0.baseColorTexture`)
- **THEN** the loader produces texture bindings for `diffuseColor`, `roughness`, and `metallic` with the interface-supplied file paths, and the packed-channel selectors (`roughness ← .g`, `metallic ← .b`) and colorspaces (`sRGB` diffuse, `raw` packed) are preserved

#### Scenario: Dangling interface connection degrades to constant fallback
- **WHEN** a `file` input connects to an interface input that has no authored asset value
- **THEN** the loader skips the texture binding without error and the shader input falls back to its authored constant or default, identical to today's unconnected-file behavior

### Requirement: UsdTransform2d st transforms are honored
The USD loader SHALL detect a `UsdTransform2d` node between the primvar reader and a `UsdUVTexture` in a binding's st chain and SHALL bake its transform into the mesh UV data at load time, applied **in raw USD st-space before the loader's USD→skinny V-convention flip**: `skinny_uv = flip( translation + R(rotation) · (scale ⊙ raw_st) )`, where `flip(v) = (u, 1 − v)` is the existing unconditional convention flip. An identity or absent transform SHALL leave the UV path bit-identical to current behavior. The mesh content hash that keys the persistent mesh cache SHALL be computed over the post-bake UVs, so differing transforms on shared geometry produce distinct cache entries. When multiple bindings on one material author differing transforms, the loader SHALL warn and apply the transform shared by the majority of that material's bindings.

#### Scenario: glTF V-flip renders upright
- **WHEN** a GLB-derived material authors `UsdTransform2d` with `scale (1, -1)`, `translation (0, 1)`, `rotation 0` on every texture's st chain
- **THEN** the transform and the convention flip cancel, the final uploaded UVs equal the raw authored `primvars:st`, and a vertically asymmetric texture renders with the same orientation as the source glTF

#### Scenario: Shared geometry under differing transforms does not collide in the mesh cache
- **WHEN** the same mesh geometry is loaded under two materials whose st chains author different `UsdTransform2d` values
- **THEN** the two loads produce distinct mesh cache keys and each uploaded UV set reflects its own material's transform

#### Scenario: Scenes without st transforms are unchanged
- **WHEN** a stage authors `UsdUVTexture` nodes fed directly by the primvar reader with no `UsdTransform2d`
- **THEN** the uploaded UV data is bit-identical to the loader's output before this change

#### Scenario: Conflicting transforms on one material
- **WHEN** two texture bindings of the same material author different `UsdTransform2d` values
- **THEN** the loader emits a warning naming the material and applies the majority transform, and the load completes without error
