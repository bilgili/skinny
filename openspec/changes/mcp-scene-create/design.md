## Context

The in-process MCP scene server (`src/skinny/mcp_server.py`) exposes structural
tools that author into a USD edit layer. `has_editable_stage(renderer)` gates
them on **both** `_usd_stage` and `_usd_edit_layer` being set, and those are set
only by `_load_usd_model(path)` — the async disk-load path. With no scene loaded,
every structural tool returns *"no editable USD stage is loaded"*.

The renderer already has the pieces to build an editable stage in memory:
- `_attach_edit_layer()` creates an anonymous strongest sublayer and makes it the
  stage edit target (sets `_usd_edit_layer`).
- `_resync_geometry_from_stage()` re-reads the stage into the flat scene + GPU
  buffers, rebuilds the derived `scene_graph`, re-injects the synthetic
  `/Skinny/DefaultLight` + `DefaultDome` + `MainCamera`, uploads, and bumps
  `_material_version` / `_scene_graph_version`.
- `set_usd_scene(scene, stage)` shows the synchronous "adopt this stage" pattern
  (sets `_usd_scene`, `_usd_model_index`, attaches edit layer, uploads) but does
  **not** build the scene graph — so it alone would leave `scene_list` dead.

MCP structural tools run on the render thread via `_structural` →
`self._proxy.request(callback)`; that path returns the result within a grace
window or parks a pollable job. Synthesizing an empty stage is cheap, so it
settles inline in practice.

## Goals / Non-Goals

**Goals:**
- One MCP tool, `scene_create`, that yields an editable empty stage so
  `scene_add_*` / `scene_save` work with no prior disk load.
- Reuse the existing edit-layer + resync finalize path verbatim; no duplicated
  slices of the load pipeline.
- Protect unsaved structural edits: refuse-unless-`force`.

**Non-Goals:**
- No `empty_scene.usda` asset shipped (synthesize, don't ship a file).
- No seeded geometry/lights/camera in the authored stage (synthetic defaults
  cover lighting + framing).
- No path-allowlist change — nothing is written until `scene_save`.
- No new async/job machinery — reuse `_structural`.

## Decisions

**Synthesize in-memory, not a shipped asset.** `create_empty_scene()` builds the
stage with `Usd.Stage.CreateInMemory()`, authors `/World` as an `UsdGeom.Xform`
default prim, sets `UsdGeom.SetStageUpAxis(stage, y)` and
`SetStageMetersPerUnit(stage, 1.0)`. No file, no packaging concern, no asset
resolution.

**`/World` is load-bearing.** `ui/scene_edit_actions.add_parent_for_node`
defaults an add to `/World`; without that Xform the first `scene_add_primitive`
would author under a non-existent parent. Lights/dome/camera are deliberately
omitted — `_resync_geometry_from_stage` re-injects the synthetic ones, so
authoring real ones would create duplicates the client must then `scene_remove`.

**The loader raises on empty geometry — thread `allow_empty` so an empty stage
is a valid scene, not an error (fixes a latent resync bug too).**
`load_scene_from_stage` → `_read_open_stage` raises
`ValueError("...contains no usable mesh or gprim geometry")` when `prim_data` is
empty (usd_loader.py:1990). A bare `/World` Xform hits it — so the naive
`create_empty_scene` throws on its happy path, and `_resync_geometry_from_stage`
(which re-calls `load_scene_from_stage`, renderer.py:5934) *already* raises today
when `scene_remove` deletes the last instance. Root-cause fix: add
`allow_empty: bool = False` to `load_scene_from_stage` / `_read_open_stage` that
skips the empty-geometry raise and returns a well-formed empty `Scene`
(`materials=[default]`, empty instance/light lists, `has_authored_lighting=False`,
`mm_per_unit` from the stage). Disk load keeps the guard (default false);
`_resync_geometry_from_stage` passes `allow_empty=True`, which unblocks both
`create_empty_scene` and remove-last-prim. **Implementation fork:** confirm the
loader's post-guard code (bbox/camera framing) tolerates empty `prim_data`; if it
assumes non-empty, build the empty `Scene` inline in `create_empty_scene` and add
the same tolerance to the resync path.

**Reuse the resync finalize path for the graph + defaults + version bump.**
`create_empty_scene()`:
1. builds `stage` as above;
2. `self._usd_scene = load_scene_from_stage(stage, allow_empty=True)` and enters
   the USD-active state, mirroring `set_usd_scene` (renderer.py:5876-5879):
   **append a `"USD:"` label to `self.models`** and set `_usd_model_index` /
   `model_index` to it — sites index `self.models[self._usd_model_index]`
   (renderer.py:5425), so the label append is required, not just the index;
3. `self._usd_stage = stage`; `self._usd_edit_layer = None`;
   `self._attach_edit_layer()`;
4. `self._resync_geometry_from_stage()` → builds `scene_graph`, injects synthetic
   default light/dome/camera, uploads, bumps versions.

This keeps `scene_list` / `scene_get` / `scene_remove` (which need `scene_graph`)
working immediately after creation.

**Reset the TLAS on the zero-instance path.** `_upload_usd_scene`'s
`if not scene.instances:` branch (renderer.py:5500) uploads lights only and never
touches `_num_instances` — which is set only by `_upload_instances`
(renderer.py:8826) and packed into `FrameConstants` to bound TLAS traversal
(renderer.py:10167). So a fresh renderer keeps the analytic head's
`_num_instances=1`, and a force-replace of an N-instance scene keeps N ghost
records. Fix in the shared branch: call `self._upload_instances([])` (resets
`_num_instances=0`, uploads an empty TLAS) — this also fixes remove-last-instance,
so it belongs in the shared path, not in `create_empty_scene`.

**Reset stale per-scene state on force-replace.** `_load_usd_model` sets
`_usd_up_axis_rt`, `_anim_index`, `_skeletal`, `_usd_controls`, `clock`,
`_usd_bake_done` (renderer.py:3479-3493); neither resync nor set_usd_scene touch
them. For a fresh Y-up empty stage the defaults are inert, but force-replacing a
Z-up / animated scene would carry a stale rotation + clock. `create_empty_scene`
resets them to defaults.

**Refuse-unless-`force`.** `scene_create`'s `write(renderer)` checks
`has_editable_stage(renderer)`; if true and `force` is false it raises
`SceneToolError("a scene is already loaded; pass force=true to replace it and "
"discard unsaved structural edits")`. `force=true` proceeds and
`create_empty_scene()` overwrites `_usd_stage` / `_usd_edit_layer` with the fresh
stage. The guard lives inside the render-thread `write` callback — same-thread
with the other structural tools. (No stronger atomicity claim than they have:
`_load_usd_model` sets `_usd_stage` from a daemon thread, so this shares the same
pre-existing, operator-initiated load race the server docstring already
acknowledges, mcp_server.py:24-28.)

**Versions-only result.** `write` returns `{**_versions(renderer)}`; `_structural`
wraps it as `{"status": "done", ...}`. No `path` — matches the anonymous-stage
reality and signals "re-enumerate" via the bumped counters. `_resync_geometry_from_stage`
bumps both `_material_version` (5957) and `_scene_graph_version` (5969), so the
change is observable.

**FastMCP registration.** `scene_create` takes `force: bool = False`. No special
handling: `_wrap` already sets `call.__signature__ = inspect.signature(fn)` for
every tool generically (mcp_server.py:732), so the bool default surfaces in the
schema automatically — registration is just adding the callable to the tuple.

## Risks / Trade-offs

- **Synthetic light/dome/camera injection is effectively guaranteed on the MCP
  hosts.** `_init_default_light_stage()` is called *unconditionally* in
  `Renderer.__init__` (renderer.py:1642), so every renderer has
  `_default_light_stage`; the injection early-returns only if the `pxr` import
  fails at construction. The MCP server attaches to *both* the GLFW front-end
  (app.py:683) and the Qt front-end (ui/qt/app.py:111), both of which construct a
  full renderer — so a created scene is lit + framed on both. (Earlier "only the
  Qt front-end / headless gap" framing was wrong.)
- **Empty-stage loader tolerance** — the `allow_empty` fork above must yield a
  well-formed empty `Scene`; if the loader's post-guard code assumes non-empty
  geometry, fall back to inline `Scene` construction. Covered by the hostless unit
  test.
- **`force` is a blunt instrument** — it discards *all* unsaved edits, not a
  diff. Acceptable: the alternative (inspecting edit-layer prim specs to detect
  "real" edits) is fiddly for marginal gain, and the default refusal already
  makes loss non-silent.
- **Empty USD scene is an unexercised renderer state.** `_rebake_if_needed`'s
  USD-active early-return is gated on `scene.instances` (renderer.py:8936), so a
  zero-instance created scene falls through to the mesh-source branch — benign
  only while `_usd_model_index >= len(_mesh_sources)`. Test that a created scene
  renders without silently switching back to the analytic head.
- **No megakernel/shader impact, no descriptor-binding change** — `main_pass.spv`
  is untouched.
