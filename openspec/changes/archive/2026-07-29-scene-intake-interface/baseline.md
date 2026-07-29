# Baseline: what the three adoption paths do (tasks 1.1 / 1.2)

Recorded against `main` at `248a7a8`. Line numbers are pre-change
`src/skinny/renderer.py`.

Method: each path was read end to end and its steps transcribed in order. The
table is the source for `SceneUpdate`'s per-trigger fields. The empirical
identity check (same scene, same image) is gate 6.2.

## The three paths

- **A** — `set_usd_scene(scene, stage=None)` (`:3617`), synchronous, headless.
- **B** — `_poll_usd_streaming()` metadata phase (`:3123`), async, interactive.
- **C** — `_resync_geometry_from_stage()` (`:3743`), post-edit.

## Step table

| # | Step | A | B | C |
|---|------|---|---|---|
| 1 | Enter USD-active (`models` label + `_usd_model_index`) | yes | done earlier in `_load_usd_model` | no |
| 2 | `_usd_scene` | swap | swap | **mutate 8 fields** |
| 3 | `film_max_component` | yes | yes | **no** |
| 4 | `mm_per_unit` adopt | **no** | yes, guarded `!= 120.0` | **no** |
| 5 | Attach edit layer | if `stage` given | in the bg thread | already attached |
| 6 | Carry runtime state (instance/light enabled, material overrides) | **no** | n/a (fresh) | yes |
| 7 | `_sync_volume_grid` | yes, **before** materials | yes, **after** `_gen_scene_materials` | yes, before materials |
| 8 | `_gen_scene_materials` | yes | yes | yes |
| 9 | `_frame_camera_to_scene` | if first, or if `camera_override` | yes | **no** |
| 10 | `_override_to_orbit(usd_camera, …)` | **no** | if `camera_override` | **no** |
| 11 | `_apply_control_defaults` | **no** | if `_usd_controls` | **no** |
| 12 | `_refresh_camera_node` | **no** | yes | yes |
| 13 | `_inject_default_lights_into_scene_graph` | **no** | yes | yes |
| 14 | Build scene graph | **no** (documented limitation) | in the bg thread | yes |
| 15 | `_upload_usd_scene` | yes | if `scene.instances` | yes |
| 16 | `_material_version += 1` | **no** | **no** | yes |
| 17 | `_scene_graph_version += 1` | **no** | only on instance-ref back-fill | yes |

## Verdict per difference

| # | Verdict | Disposition |
|---|---------|-------------|
| 1 | Deliberate — B enters the state in `_load_usd_model` before the read starts; C never leaves it. | `SceneUpdate.enter_usd_active` |
| 2 | **Bug class.** C hand-copies 8 fields only because `id(_usd_scene)` is a UI change token (D3). `film_max_component` was already forgotten from that list — exactly the failure the hand-copy invites. | Swap always; replace the id token with an explicit counter. |
| 3 | Bug (latent). C re-reads the stage but never adopts the re-read film clamp. Harmless today because add/remove edits do not change the film, so the value is unchanged. | Always adopt from the update's scene. |
| 4 | Deliberate. `120.0` is `Scene`'s sentinel default; adopting it would clobber the renderer's skin scale. A's headless callers set `mm_per_unit` themselves. | `SceneUpdate.adopt_mm_per_unit`, guard kept. |
| 5 | Deliberate — trigger-specific ownership of the stage. | `SceneUpdate.stage` (attach when the update carries one). |
| 6 | Deliberate, and D2 promotes it. Carry-over must NOT become unconditional: `parameter_overrides` mixes authored loader values with live edits, so on a full load the old authored value would win over the newly authored one. | `SceneUpdate.carry_runtime_state`, true for resync only. |
| 7 | **Not a real conflict.** The stated constraint (`_sync_volume_grid` docstring) is "before the scene's *material upload*". That upload is `_upload_flat_materials` inside `_upload_usd_scene`, which follows the grid sync in all three. B's order is therefore also correct. | Unified order uses A/C's stricter order. |
| 8 | Same in all three. | Unified. |
| 9 | Deliberate. C must not yank the user's camera on an unrelated edit. A's "first or authored camera" rule is the same intent expressed differently. | `SceneUpdate.frame_camera` (`"always"` / `"if_first_or_authored"` / `"never"`). |
| 10 | Deliberate — seeds the USD camera follower so the user can switch to `usd` mode before pressing play. Only meaningful where a follower exists. | Folded into the camera step; no-op when `camera_override is None`. |
| 11 | Deliberate. Defaults apply once at load; re-applying on every resync would clobber the user's later edits. A never extracts controls at all. | `SceneUpdate.apply_control_defaults`. |
| 12–14 | Deliberate consequence of #14: A builds no scene graph, so the two graph-dependent steps cannot run. In the unified path they are conditional on `scene_graph is not None`, not on the trigger. | Conditional, not per-trigger. |
| 15 | B's `if scene.instances` guard is redundant — `_upload_usd_scene` already has a zero-instance branch that resets the TLAS. | Unconditional. |
| 16 | Bug (latent) in A and B: neither bumps `_material_version`, so a raw USD edit is not in the accumulation state hash. Both are followed by an accumulation reset from another field of the hash (a fresh scene changes several), so it is invisible today. | Always bump. |
| 17 | Follows from #2: with `id()` as the token, a *swap* was itself the signal, so A and B did not need a bump. With an explicit counter, every apply must bump. | Always bump. |

### One behaviour change, deliberate: force-replace no longer carries state

`create_empty_scene` used to finish by calling `_resync_geometry_from_stage`,
so it inherited that path's runtime-state carry-over: the *previous* scene's
material overrides were merged onto the new empty stage's materials by name.
That is the finding-#7 machinery firing where it should not — the method's own
docstring says any previously loaded stage is *replaced*. A force-replace now
uses `SceneUpdate.replacing`, which carries no runtime state. Pinned by
`test_a_replacing_update_keeps_no_runtime_state`.

## `id(_usd_scene)` change-token sites

The proposal names three sites in `ui/build_app_ui.py`. There are **six**, in
three roles:

| Site | Role |
|------|------|
| `ui/build_app_ui.py:570` | Animation section rebuild token |
| `ui/build_app_ui.py:583` | Scene Controls section rebuild token |
| `ui/panel/windows.py:482` | Material-list repopulate poll |
| `ui/panel/windows.py:664` | Material-inputs repopulate poll |
| `renderer.py:4688` | `_sync_auxiliary_light_authority` cache token |
| `renderer.py:8437`, `:8439` | Environment-upload cache key |

All six move to the explicit `scene_version` counter. The two renderer-internal
caches are the reason the counter is strictly better than `id()`: under C's
mutate-in-place the id never changed, so both caches depended on either a
`force=True` argument or an unrelated `id(env_hdr.data)` term to avoid going
stale.

## Per-frame re-read (task 1.3)

`_apply_animation_frame` (`:3366`) re-derives, per time code:

- instance world transforms — `_world_transform(prim, time) @ rt4`
- distant + sphere lights — `_extract_distant_light` / `_extract_sphere_light`,
  each rotated by `rt`
- camera override — `_extract_camera(stage, time)`, position and forward
  rotated by `rt`
- skeletal joint matrices — `compute_joint_matrices(binding, time)`

`_refresh_usd_live_state` (`:3543`) performs the same three non-skeletal
re-reads at `TimeCode.Default()`. That duplication is the identity target: one
`read_at_time` call serves both.

The fixture is `tests/fixtures/anim_reread.json`, captured from a real headless
Metal renderer by `tests/fixtures/_capture_anim_reread.py` over a five-point
sweep of a **Z-up** stage — Y-up would leave the up-axis rotation `None` and
test none of the composition math.

### Defect found while capturing: animated light intensity does not animate

The fixture records a constant `radiance` of `(50000, 45000, 40000)` for a
`DistantLight` whose `inputs:intensity` is time-sampled `3 → 7`.

`_light_color_radiance` (`usd_loader.py:1426`) calls `intensity_attr.Get()`
with **no time code**. On an attribute that carries only time samples and no
default, that returns the *schema fallback* — 50000 for `UsdLuxDistantLight` —
so neither the sampled values nor the animation are ever seen. Colour and
exposure have the same hole. Only the light's transform animates today.

This is pre-change behaviour and `read_at_time` preserves it verbatim: the
requirement is that per-frame values are identical to what the pre-change
extraction produced, so fixing it here would break its own identity gate. It is
recorded as a follow-up.
