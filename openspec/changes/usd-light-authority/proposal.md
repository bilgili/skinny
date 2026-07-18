## Why

Skinny currently mixes two lighting authorities: USD-authored lights and
renderer-owned default lighting. In particular, it can retain a built-in IBL
when a scene authors another light type, and it exposes default-light controls
even when those controls should not affect an authored scene. This changes the
lighting intended by the USD author and makes it unclear which state owns the
render.

## What Changes

- **BREAKING**: make lighting authority all-or-nothing. If a loaded USD scene
  contains any authored lighting source, Skinny uses only the lighting from the
  USD and synthesizes neither its default DistantLight nor its default IBL.
- If the scene contains no authored lighting source, Skinny adds both fallback
  lights together: the built-in DistantLight and the built-in IBL.
- Use authored-light presence, not current power, as the decision. A disabled
  or zero-intensity authored USD light still expresses author intent and
  suppresses the fallback pair.
- Show the renderer-owned IBL and Direct Light controls only while the fallback
  pair is active. Hide both control groups and both synthesized scene-graph
  nodes when USD lighting is authoritative; USD-authored light properties
  remain editable in the scene graph.
- Apply the same policy to every front-end, the headless API, runtime scene
  resyncs, and both GPU backends. Fallback-only persisted/API/CLI values do not
  override authored USD lighting.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `default-light-synthesis`: replace per-light/powered-light fallback behavior
  with one all-or-nothing USD-light authority rule and make default-light
  control visibility follow that rule.

## Impact

- `src/skinny/scene.py`, `src/skinny/usd_loader.py`: retain whether the stage
  contains a supported authored light independently of its intensity or
  enabled state.
- `src/skinny/renderer.py`: centralize the authority predicate; gate distant
  upload, environment selection, fallback state, and scene-graph injection
  from it; re-evaluate after scene edits/resyncs.
- `src/skinny/params.py`, `src/skinny/ui/build_app_ui.py`,
  `src/skinny/ui/{qt,panel}/backend.py`: conditionally expose the IBL and Direct
  Light control groups without leaving empty sections.
- Headless/API behavior: `--env-intensity`, `--no-direct`, and their
  `RenderOptions` equivalents become fallback-only settings.
- Tests and docs: replace the default-light decision table, add conditional-UI
  and resync coverage, and update `Architecture.md`, `PythonAPI.md`, README,
  and the changelog.
- No shader, descriptor, or dependency changes are expected.
