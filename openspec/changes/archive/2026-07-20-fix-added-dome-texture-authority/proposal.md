# Fix: dome-light texture edit ignored when the dome had no environment

## Why

Editing a `UsdLuxDomeLight`'s `texture:file` from the scene-graph panel does not
change the lighting when the dome had **no** environment at the moment the
texture is set — the classic case being a dome the user just created with
"Add light". The texture is written to the prim and uploaded to the GPU, but the
dome contributes **zero** light. The user experiences this as a refresh lag:
toggling the light's enable off/on (which happens to force a full stage resync)
makes it appear.

Root cause, verified headlessly on Metal (`cornell_box_emissive.usda` +
`add_light("DomeLight")` + set texture):

`Renderer.apply_dome_light_texture` (`renderer.py:8033`) branches on whether an
environment *already exists*:

```python
authored_dome = (self._is_usd_active() and self._usd_scene is not None
                 and self._usd_scene.environment is not None)
```

A freshly added dome has no `texture:file`, and `_extract_dome_light` returns
`None` for a textureless dome, so `_usd_scene.environment is None`. Setting the
texture therefore takes the **fallback** branch, which writes the default-lights
HDR library (`self.environments[idx]` / `self.env_index`). But on an authored
scene `uses_default_lights` is False, so `scene_environment_for_authority`
returns `usd_scene.environment` — still `None` — and
`environment_contribution_intensity(None)` is `0.0`. The dome is dark forever.

Measured: `_usd_scene.environment` stays `None` and contribution stays `0.0`
before add, after add, after texture-set, and after the enable toggle. A dome
authored *with* a texture works correctly (swap shifts the image immediately),
confirming the bug is specific to the no-prior-environment case.

## What Changes

- `apply_dome_light_texture` branches on the active **lighting authority**
  (`uses_default_lights`), not on whether an environment already exists.
- On an authored scene (`uses_default_lights == False`) with no
  `_usd_scene.environment`, it constructs a `LightEnvHDR` from the loaded env
  and assigns it to `_usd_scene.environment` so the authority-selected
  environment becomes non-`None` and contributes.
- Existing cache invalidation (`_last_env_index`) and accumulation reset
  (`_material_version`) are unchanged.

Out of scope (filed separately): Panel/web front-end has no `texture_file`
dispatch route; `_extract_dome_light` rejects `.exr`/`.pfm`; added
`RectLight`/`DiskLight` reach no runtime scene.

## Impact

- Affected spec: `usd-scene-editing` (new requirement for dome texture editing).
- Affected code: `src/skinny/renderer.py` (`apply_dome_light_texture`).
- No descriptor/binding/shader change; behavior-only fix on the editing path.
