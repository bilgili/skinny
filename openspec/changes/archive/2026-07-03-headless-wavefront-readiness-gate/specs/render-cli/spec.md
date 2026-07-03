# render-cli delta

## ADDED Requirements

### Requirement: Headless readiness gate is execution-mode-aware

The headless render entry points SHALL gate frame dispatch on the renderer's
backend-aware readiness signal (`_backend_render_ready`), not on the presence
of the megakernel pipeline. This covers `HeadlessRenderer.render_to_array`,
`HeadlessRenderer.render_scene`, and therefore every `skinny-render`
invocation. In wavefront execution mode the megakernel pipeline is intentionally
not built (`scene_bindings_only`), so a pipeline-presence check would reject
every wavefront render of a valid scene.

#### Scenario: Wavefront headless render is not rejected by the gate

- **WHEN** a headless render is invoked in wavefront execution mode on a scene
  that built successfully (scene bindings present, megakernel pipeline absent
  by design)
- **THEN** `render_to_array` / `render_scene` proceed to accumulate and return
  the frame instead of raising
  `render pipeline failed to build — scene has no usable materials`

#### Scenario: Unready renderer still raises the materials error

- **WHEN** a headless render is invoked and the backend is not ready to
  dispatch (e.g. the scene produced no usable materials)
- **THEN** `render_to_array` / `render_scene` raise the
  `render pipeline failed to build — scene has no usable materials`
  `RuntimeError` before any accumulation
