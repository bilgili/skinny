# Tasks: mcp-scene-structure

## 1. Path policy module (hostless, no renderer)

- [x] 1.1 Create `src/skinny/mcp_paths.py`: `resolve_roots(cli_value, env)` with
      precedence flag > `SKINNY_MCP_ROOTS` > default
      `[realpath(gettempdir()), realpath("/tmp"), realpath(cwd)]` (deduped);
      `check_path(path, roots)` realpath + prefix check returning None or a
      retry-grade reason naming the path and roots.
- [x] 1.2 Add `validate_added_subtree(stage, prim, pre_layers, roots)`: first
      `prim.Load(Usd.LoadWithDescendants)` so deferred payloads compose (F3);
      diff `stage.GetUsedLayers()` against `pre_layers` (anonymous layers
      exempt); then walk the subtree with
      `Usd.PrimRange(prim, Usd.TraverseInstanceProxies())` (F3) collecting
      asset-valued attributes; check each attribute's **resolved** path
      (`Sdf.AssetPath.resolvedPath`, not the authored string), rejecting empty
      resolution / udim templates (F4); raise with offending path on violation.
- [x] 1.3 Hostless tests: precedence, macOS `/tmp` vs `gettempdir()` realpath
      equivalence, symlink escape rejected, pre-existing out-of-root layers not
      policed, nested-reference and texture escapes caught, **payload-hidden and
      instanced-prototype texture caught (F3)**, **relative asset path resolves
      correctly + unresolvable/udim rejected (F4)** (tiny fixture `.usda` files
      under a tmp root).

## 2. Renderer verbs

- [x] 2.0 **Fix `add_model` rollback first (F2):** port `add_light`'s
      `missing_parent_paths` tracking (:6125-6133) + reversed removal
      (:6162-6164) into `add_model`'s `except` (:6079-6082), which currently
      removes only the referenced prim. Test: forced-failure add under a novel
      parent leaves the edit layer byte-clean.
- [x] 2.1 Add optional `validate=` callable to **`add_model` only** (F12),
      invoked as `validate(stage, added_prim)` post-recompose/pre-resync with
      the prim `add_model` authored; raise ⇒ the fixed rollback (2.0) ⇒
      exception propagates; default None byte-identical.
- [x] 2.1b Extend `add_light(intensity=None, color=None)` (F5): `Set` the
      intensity/color attrs at define time when given, beside the existing
      `Create*Attr().Set` (:6140-6141); omitted ⇒ current defaults. Test:
      authored values readable from stage; defaults unchanged when omitted.
- [x] 2.2 Implement `Renderer.add_primitive(prim_type, parent_prim_path, name,
      transform, color, roughness, metallic)`: schema-table like `add_light`;
      define gprim + sibling `UsdShade.Material` + `UsdPreviewSurface`
      (diffuseColor/roughness/metallic, defaults) + bind; uniquified path
      returned; rollback removes gprim, material, and created parents (mirror
      `add_light`). No `validate=` (F12 — authors no external files).
- [x] 2.3 Tests in `tests/test_scene_editing.py` style: primitive renders after
      resync, material is a dedicated `scene.materials` entry accepting
      `apply_material_override` (not slot 0), rollback on forced failure,
      validator veto on `add_model` rolls back (incl. auto-created parents) with
      no resync, **TRS round-trip through the real scene-graph builder
      (compose→author→decompose), Euler order confirmed (F13)**, **remove then
      re-add same name → distinct path, old subtree not resurrected (F11)**.

## 3. MCP tools

- [x] 3.1 Job store in `SceneTools`: dict id→future+meta, uuid ids, last-50 /
      10-min pruning, **`threading.Lock`-guarded** — the loop thread reads while
      the render thread settles futures (F6).
- [x] 3.2 `_structural(callback)` helper: post_with_reply, inline grace wait
      (2 s) → done result, else park future → `{status:"pending", job_id}`;
      never cancel after grace. Note (F6): the grace is a deliberate loop-wide
      stall (FastMCP runs sync bodies on the event loop), so keep it ≤2 s.
- [x] 3.3 `scene_job_status(job_id)`: **strictly non-blocking (F6)** —
      `future.done()` then immediate `result()`, never `result(timeout=…)`;
      pending / done+result / failed+error; unknown id ⇒ tool error.
- [x] 3.4 Transform helper: TRS (`translate`, `rotate_euler_deg`, `scale`
      scalar|vec3) XOR `matrix[16]` row-major ⇒ 4×4 via `compose_trs_matrix`;
      both given ⇒ error; shear caveat in tool docs.
- [x] 3.5 `scene_add_model(usd_path, name?, parent?, …transform)`: root-check
      argument, then **inside the posted closure** (render thread, F1) capture
      `pre_layers = stage.GetUsedLayers()` and call `renderer.add_model(...,
      validate=lambda stage, prim: validate_added_subtree(stage, prim,
      pre_layers, roots))`.
- [x] 3.6 `scene_add_primitive(type, color?, roughness?, metallic?, name?,
      parent?, …transform)` over `add_primitive`.
- [x] 3.7 `scene_add_light(light_type, intensity?, color?, name?, parent?,
      …transform)` over the extended `add_light` (2.1b) — intensity/color
      authored at define time, not post-add.
- [x] 3.8 `scene_remove(path)`: distinguish "no such node" (unresolved) from
      "not deletable" (`is_deletable` refused) (F10); SetActive(False)
      semantics in description.
- [x] 3.9 `scene_save(path)`: **path required** (F9 — `save_edits()`'s default
      lands beside the scene, outside roots), root-check, `renderer.save_edits`;
      partial-save caveat (property edits not captured) prominent in the
      description.
- [x] 3.10 Asset-valued `scene_set` writes get the same root check, gated on
      the generic asset-type predicate `_coerce` uses — not a hardcoded name
      trio (F8).
- [x] 3.11 All six registered through `_wrap` in `build_app`; no-stage ⇒ "no
      editable USD stage is loaded" error, **including mapping `save_edits`'s
      `RuntimeError("no edit layer attached")` to the same friendly text (F9)**.

## 4. CLI plumbing

- [x] 4.1 `--mcp-roots` (comma-separated) + `SKINNY_MCP_ROOTS` in the shared
      MCP argument group (`cli_common.py`), threaded to `SceneTools` on all
      four front-ends.

## 5. Tests (MCP layer, hostless fake renderer)

- [x] 5.1 Extend the existing fake-renderer MCP suite: each tool happy path,
      name uniquification reported, `/Skinny/*` remove refused vs unknown path
      "no such node" (F10), both-transform error, unsupported primitive/light
      type errors, save-without-path refused (F9).
- [x] 5.2 Job pattern: fast op returns done inline; slow op (stalled fake
      queue) returns pending then done with final prim path; failed job
      reports error; unknown id errors; pruning; `scene_job_status` never
      blocks (F6).
- [x] 5.3 Path policy end-to-end through the tools: out-of-root argument,
      nested-reference escape rollback (edit layer clean incl. parents, F2),
      out-of-root `scene_set` texture write via the generic asset predicate
      (F8).

## 6. Docs + gate

- [x] 6.1 `docs/PythonAPI.md` MCP section: six new tools, job pattern, roots
      config, partial-save caveat. `docs/Architecture.md` MCP paragraph
      updated.
- [x] 6.2 `README.md` if MCP flags are listed there; CHANGELOG entry.
- [x] 6.3 `ruff check src/`, full hostless pytest, `openspec validate
      mcp-scene-structure`.
- [x] 6.4 Codex pre-merge review — **skipped by explicit user request**
      (`skip the codex review`); not merged, worktree left in place.
