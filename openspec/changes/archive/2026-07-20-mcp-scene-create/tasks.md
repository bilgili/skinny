## 1. Design review

- [x] 1.1 Spin up an xhigh-effort design-review subagent over proposal.md +
  design.md + the delta spec; fold findings back in before coding. (Verdict
  NEEDS-REVISION → 2 blockers C1/M1 + minors folded in: allow_empty loader,
  TLAS reset, state reset, models[] label, dropped __signature__ task.)

## 2. Loader + upload fixes (root-cause, shared paths)

- [x] 2.1 Add `allow_empty: bool = False` to `load_scene_from_stage` /
  `_read_open_stage`: when set, skip the `ValueError` empty-geometry raise
  (usd_loader.py:1990) and return a well-formed empty `Scene`
  (`materials=[default]`, empty instance/light lists,
  `has_authored_lighting=False`, `mm_per_unit` from the stage). Disk load keeps
  the guard (default false). Verify the post-guard code (bbox/camera framing)
  tolerates empty `prim_data`; if not, build the empty `Scene` inline instead.
- [x] 2.2 Pass `allow_empty=True` from `_resync_geometry_from_stage` so
  remove-last-instance no longer raises there (latent bug) and empty-scene
  resync works.
- [x] 2.3 In `_upload_usd_scene`'s `if not scene.instances:` branch, reset the
  TLAS via `self._upload_instances([])` so `_num_instances` → 0 (fixes ghost
  geometry on force-replace AND remove-last-instance).

## 3. Renderer: create_empty_scene()

- [x] 3.1 Add `create_empty_scene()` to `renderer.py`: build an in-memory stage
  (`Usd.Stage.CreateInMemory()`), author `/World` Xform as default prim, set
  up-axis Y and metersPerUnit 1.
- [x] 3.2 Adopt the stage: `_usd_scene = load_scene_from_stage(stage,
  allow_empty=True)`; enter the USD-active state mirroring `set_usd_scene` —
  **append a `"USD:"` label to `self.models`** and set `_usd_model_index` /
  `model_index` to it (label append required, not just index); `_usd_stage =
  stage`; reset `_usd_edit_layer = None`; call `_attach_edit_layer()`.
- [x] 3.3 Reset stale per-scene state for force-replace: `_usd_up_axis_rt`,
  `_anim_index`, `_skeletal`, `_usd_controls`, `clock`, `_usd_bake_done` to
  their defaults.
- [x] 3.4 Finalize via `_resync_geometry_from_stage()` (scene-graph build +
  synthetic light/dome/camera injection + upload + version bump).

## 4. MCP tool: scene_create

- [x] 4.1 Add `scene_create(self, force: bool = False)` to `mcp_server.py` with a
  `write(renderer)` callback: refuse via `SceneToolError` when
  `has_editable_stage(renderer)` and not `force`; else call
  `renderer.create_empty_scene()` and return `{**_versions(renderer)}`. Route
  through `self._structural(write)`.
- [x] 4.2 Register the tool in the build_app tool tuple (no `__signature__`
  special-casing — `_wrap` sets it generically for every tool).

## 5. Tests

- [x] 5.1 Hostless unit for `create_empty_scene()`: after the call, `_usd_stage`
  and `_usd_edit_layer` are set, the scene graph contains `/World`, and an empty
  `Scene` (no instances) results without raising.
- [x] 5.2 Extend the MCP tool-schema test to assert a `scene_create` tool with a
  `force` boolean parameter.
- [x] 5.3 Behavior test: `scene_create` refuses when an editable stage is present,
  and `force=true` replaces it.
- [x] 5.4 Regression test: `load_scene_from_stage(..., allow_empty=True)` on a
  `/World`-only stage returns an empty `Scene`; and a created scene does not fall
  back to the analytic head (`_rebake_if_needed`).

## 6. Docs

- [x] 6.1 Add `scene_create` to the README MCP tool list.

## 7. Validate & land

- [x] 7.1 `openspec validate mcp-scene-create --strict`.
- [x] 7.2 Run hostless tests + `ruff check src/` on changed files.
- [x] 7.3 Codex (or review-subagent fallback) pre-merge review; fold findings.
- [x] 7.4 Merge the worktree to main; archive the change.
