## ADDED Requirements

### Requirement: Set a dome light's texture honoring lighting authority

The renderer SHALL provide `apply_dome_light_texture(env_index, path)` that loads
the HDR at `path`, mirrors it onto the source `UsdLuxDomeLight` prim's
`texture:file`, uploads it to the GPU, and makes the dome contribute light under
the **active lighting authority**, regardless of whether that dome had an
environment before the call.

When an authored USD scene is active (`uses_default_lights == False`), the loaded
environment SHALL be assigned to the authored scene's environment
(`_usd_scene.environment`) — constructing it when the dome previously had none —
so that the authority-selected environment is non-null and its contribution
intensity is non-zero. When the fallback default-lights authority is active, the
environment SHALL be written to the fallback environment library at `env_index`.
The env-upload cache SHALL be invalidated and accumulation SHALL reset so the new
texture is visible on the next rendered frame without any further edit.

#### Scenario: Texture set on a freshly added dome contributes immediately

- **WHEN** a dome light is added to an authored scene (so it starts with no
  environment) and `apply_dome_light_texture` is then called with a valid HDR
- **THEN** the authored scene's environment becomes non-null and the dome's
  environment contribution intensity is non-zero, with no enable toggle or
  reload required

#### Scenario: Texture swap on an already-textured dome still applies

- **WHEN** `apply_dome_light_texture` is called on a dome that already had an
  environment
- **THEN** the environment data is replaced in place and the rendered image
  reflects the new texture

#### Scenario: Missing file is reported and leaves the scene unchanged

- **WHEN** `apply_dome_light_texture` is called with a path that is not a file
- **THEN** it returns False, logs a load failure, and does not alter the active
  environment
