## ADDED Requirements

### Requirement: Deterministic light prim names

The pbrt importer SHALL derive each imported light's USD prim name from its
position in the scene's light list, not from any process-varying value (such as
a Python object identity / memory address). Re-importing the same `.pbrt` file
SHALL produce byte-identical light prim names.

#### Scenario: Same scene imported twice

- **WHEN** the same `.pbrt` scene is imported twice in separate processes
- **THEN** every light prim has the identical name in both imports
- **AND** the emitted `.usda` bytes for the light prims are identical

#### Scenario: Distinct lights get distinct names

- **WHEN** a scene declares two or more lights
- **THEN** each light prim receives a distinct name

### Requirement: Deterministic synthesized asset filenames

Synthesized environment `.hdr` filenames SHALL be deterministic across
re-imports. Filenames derived from a light's prim name (the constant-env
`_const.hdr` and converted-env `_env.hdr` files) inherit the prim-name
determinism, so a regenerated `.usda` references an `.hdr` whose name is stable
and can be tracked once.

#### Scenario: Constant env baked twice

- **WHEN** a scene with a constant (no-map) infinite light is imported twice
- **THEN** the baked `.hdr` filename is identical in both imports
- **AND** the `.usda` `textureFile` reference matches the baked filename
