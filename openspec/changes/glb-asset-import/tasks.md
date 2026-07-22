# Tasks: glb-asset-import

## 1. Loader — interface-connected file resolution (D1)

- [x] 1.1 Extend `_resolve_texture_binding` in `src/skinny/usd_loader.py`: when `file_input.Get()` is None, resolve the asset via the existing `GetValueProducingAttributes` machinery (`_resolve_connected_value` pattern, already in use at ~line 357) — no hand-rolled walk; covers Material interface inputs and NodeGraph indirection alike
- [x] 1.2 Hostless test: minimal `.usda` with `file <- Material.<name>Texture` interface connections (Apple-converter shape) resolves bindings for `diffuseColor`/`roughness`/`metallic` with correct channels (`rgb`/`g`/`b`) and colorspaces (`sRGB`/`raw`/`raw`)
- [x] 1.3 Hostless test: dangling interface connection (no authored asset) skips the binding without error, constant fallback intact

## 2. Loader — UsdTransform2d UV bake (D2)

- [x] 2.1 Walk each binding's st chain in `usd_loader.py`, capturing `UsdTransform2d` `scale`/`rotation`/`translation` into a new `TextureBinding.uv_transform` field (UsdTransform2d order: scale, rotate, translate)
- [x] 2.2 Bake the per-material transform in **raw USD st-space, before the existing convention flip at ~line 184** (`skinny_uv = flip(t + R·(s⊙raw_st))`); identity/absent transform short-circuits to the current bit-identical path
- [x] 2.3 Reorder the per-prim build loop: resolve the material before finalizing the mesh source, and compute `content_hash` after the UV mutation so the persistent mesh cache keys on post-bake UVs; hostless test asserts differing transforms on shared geometry yield distinct cache keys
- [x] 2.4 Conflicting-transform fallback: warn naming the material, apply the majority transform among that material's bindings
- [x] 2.5 Hostless test through the full load path (not `_read_mesh_attrs` directly): glTF V-flip fixture — final uploaded UVs equal the raw authored `primvars:st` (transform and convention flip cancel); asymmetric UV values asserted exactly
- [x] 2.6 Hostless test: no-transform scene produces bit-identical UV output vs pre-change loader (regression guard for the short-circuit)

## 3. Converter + MCP — glb_import.py and scene_import_glb (D3, D5)

- [x] 3.1 New `src/skinny/glb_import.py`: pure-Python GLB→USD converter (pygltflib parse + pxr authoring) — meshes (POSITION/NORMAL/TEXCOORD_0), embedded PNG/JPEG/WebP image extraction, pbrMetallicRoughness → UsdPreviewSurface with direct `file` values, `metallic ← .b` / `roughness ← .g`, UVs pre-flipped to USD convention, no `UsdTransform2d`; out-of-scope glTF features (Draco, sparse accessors, skinning, animation, vendor extensions) raise a feature-naming error; add `pygltflib` to project dependencies
- [x] 3.2 Hostless converter tests: Trellis-shape GLB fixture round-trips through the converter and the USD loader — bindings resolve with correct channels/colorspaces, UVs land in USD convention, unsupported-feature GLB refused by name, malformed GLB raises cleanly
- [x] 3.3 Add `scene_import_glb` to `src/skinny/mcp_server.py`: signature mirroring `scene_add_model` plus `glb_path`/`out_dir`/`overwrite`; `check_path` both paths before conversion; `mkdir -p` the out_dir; refuse an existing `.usd*` conversion unless `overwrite=true`; conversion on the tool thread before any renderer post; document the synchronous conversion latency in the tool docstring
- [x] 3.4 Delegate the produced `.usdc` to the existing `scene_add_model` implementation (same `validate_added_subtree`, same job degradation) — no duplicated add logic
- [x] 3.5 MCP tests: path rejection (glb outside roots, out_dir outside roots), overwrite refusal, unsupported-feature error, conversion-failure leaves renderer untouched, success returns prim path + version counters

## 4. Regression assets and GPU gate (D4)

- [x] 4.1 Author the hostless fixture assets: minimal `.usda` + 4×4 PNGs for tasks 1.2/2.4 under `tests/assets/`
- [x] 4.2 Produce and commit the small GLB-derived render asset (≲50 k faces, textures ≤256²) converted via `usdextract` at authoring time, under `tests/assets/`
- [x] 4.3 GPU-marked render gate in the `TestMaterialXGraphDemoRender` style: textured region non-white, plus an orientation-discriminating patch assertion that fails on V-flip
- [x] 4.4 Run the gate on Metal (`PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest ... -m gpu`) and the hostless suites with `.venv`

## 5. Docs and close-out

- [x] 5.1 `docs/PythonAPI.md`: `scene_import_glb` tool contract (params, errors, job degradation)
- [x] 5.2 `docs/Architecture.md`: loader texture-resolution notes (interface-connected file, UsdTransform2d bake) in the USD loader section
- [x] 5.3 `README.md`: asset-import workflow (`scene_import_glb` one-call path on all platforms; Apple `usdextract` as the manual macOS alternative), including the local TRELLIS.2 pipeline as the worked example
- [x] 5.4 `openspec validate glb-asset-import` clean; fold design-review findings; codex pre-merge review before merge
