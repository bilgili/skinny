# Design: glb-asset-import

## Context

TRELLIS.2 (local MLX server, `~/projects/trellis2-apple`, port 8082) produces GLB assets with PBR textures: a diffuse sRGB map plus a glTF-style packed metallicRoughness map (roughness = G, metallic = B). Apple's system USD tools (`/usr/bin/usdextract`, Apple USD 0.25.2, ships with macOS) convert GLB → `.usdc` + extracted PNGs with a standard UsdPreviewSurface network:

```
texCoordReader (UsdPrimvarReader_float2, varname=st)
  → <input>_stTransform (UsdTransform2d, scale=(1,-1), translation=(0,1))
  → <input> (UsdUVTexture, file <- Material0.<name>Texture [interface input],
             sourceColorSpace, wrapS/T)
  → UsdPreviewSurface (diffuseColor <- .rgb, metallic <- .b, roughness <- .g)
```

`usd_loader.py` already handles channel selectors, colorspace, wrap modes, and scale/bias. Two shapes in this network defeat it:

1. `file` is authored as a **connection to a Material interface input**, not a local value. `_resolve_texture_binding` calls `file_input.Get()` → None → returns None → the binding is silently dropped and the input falls back to its constant default. Verified: the crown renders geometry perfectly but flat white.
2. **`UsdTransform2d` is ignored** everywhere in the loader. Once fix 1 lands, every glTF-derived texture would sample V-flipped.

MCP agents drive scene assembly through `mcp_server.py` structural tools (`scene_add_model` takes a `usd_path` inside allowed roots). There is no path from "GLB on disk" to "referenced in the scene" without manual conversion.

## Goals / Non-Goals

**Goals:**
- GLB-derived UsdPreviewSurface assets render with correct textures in every front-end that loads USD scenes (the fixes live in the shared loader, not a front-end).
- Both fixes are generic USD-correctness fixes, not Apple-converter special cases: interface-connected inputs and `UsdTransform2d` are standard UsdShade/UsdPreviewSurface constructs any DCC may author.
- One MCP call imports a GLB into the live scene.
- Hostless regression coverage for both loader fixes; a GPU render gate proving a packed-texture GLB-derived asset renders non-white with correct orientation.

**Non-Goals:**
- Running or managing TRELLIS.2 itself (server, weights, Modly extension) — separate concern, nothing in-repo depends on it.
- glTF opacity/transmission, normal maps, emissive, occlusion intake — only the channels the flat-material binder already consumes (`diffuseColor`, `roughness`, `metallic`) are in scope; others remain future work.
- Full glTF 2.0 coverage in the converter — scope is the asset shape local generators emit (single/multi mesh, POSITION/NORMAL/TEXCOORD_0, embedded PNG/JPEG/WebP images, pbrMetallicRoughness). Out-of-scope features that would silently corrupt or misplace an import are **refused by name**: Draco/meshopt/quantization, sparse accessors, skinning, animation, morph targets, node transforms, mesh instancing, and external image URIs. Features that merely degrade fidelity are **not imported** and left as a documented gap rather than refused: normal/occlusion/emissive textures, secondary UV sets, alpha modes. The accessor decoder honors `byteStride` (interleaved buffers) and `normalized` integers and bounds-checks every read.

  Deliberately-accepted residual gaps (post-review, change scoped to generator output): a `UsdTransform2d` reached only through a `NodeGraph` output on a texture's `st` (no known producer authors this — usdextract and this converter author it directly); and the `check_path`→convert TOCTOU window, which is identical to the existing `scene_add_model` contract and inside the same "guardrail against agent mistakes, not a sandbox against an adversary" threat model, not a new exposure.
- Per-texture arbitrary UV transforms in the shader (see D2).

## Decisions

### D1: Resolve `file` through the existing value-producing-attribute walk

In `_resolve_texture_binding`, when `file_input.Get()` is None, resolve the asset through the same machinery `_resolve_connected_value` already uses (`GetValueProducingAttributes`, in use at `usd_loader.py:357` — availability is proven, not an open question). This canonically follows Material interface inputs *and* NodeGraph/nested-NodeGraph indirection (guc, Blender, Houdini authoring), which a hand-rolled Material-only walk would miss. The channel selector is untouched — it comes from the first-hop `src_output_name`, so packed `.b`/`.g` reads survive.

*Alternative rejected:* a manual "walk to the Material interface input" — Apple's converter authors exactly that shape, but the design's generic-correctness goal requires the canonical walk, and the canonical walk is *less* code (review finding #3).

### D2: Bake `UsdTransform2d` into mesh UVs at load time — no shader change

The loader today applies an unconditional USD→skinny V-convention flip to every mesh UV (`uvs[:,1] = 1 − uvs[:,1]`, `usd_loader.py:184-186`). The `UsdTransform2d` must compose **in raw USD st-space, before that flip**:

```
skinny_uv = flip( translation + R(rotation)·(scale ⊙ raw_st) )
```

(UsdTransform2d op order: scale, then rotate, then translate.) For the glTF case (`scale (1,−1)`, `translation (0,1)`, `rotation 0`) the transform and the convention flip cancel, so the baked UV equals `raw_st` — the correct top-origin result. An identity/absent transform reduces exactly to today's path, bit-unchanged. Applying the transform after the flip would negate offsets and reverse rotations for any non-flip transform — this ordering is normative (review finding #1).

Mechanically this is new loader flow, not a drop-in: `_resolve_texture_binding` never walks the `st` input today, `TextureBinding` grows a `uv_transform` field, and the per-prim build loop must **resolve the material before finalizing the mesh source** so two things hold: (a) the UV mutation happens before upload, and (b) `content_hash` is computed (or recomputed) *after* the mutation, so the persistent mesh disk cache (`make_cache_key(source.content_hash, …)`) keys on the transformed UVs. Without the hash reorder, a shared prototype referenced under materials with different transforms collides on one cache entry and serves stale UVs (review finding #2).

*Rationale:* baking touches only the loader — no `FlatMaterialParams` growth, no Slang change, no `.spv` recompile, no Metal argument-table pressure. Baking an affine map per-vertex is exact versus a per-fragment transform (affine commutes with barycentric interpolation; wrap modes still apply per-sample). *Alternative rejected:* per-binding UV transform threaded into the shader — grows the material struct and touches every sampling site in both backends for a case (mixed transforms within one material) with no known producer. If mixed transforms appear: warn, use the majority transform, record the limitation.

*Consequence:* UVs become per-(mesh, material) when the same geometry is shared under materials with different st transforms — accepted, and made safe by the hash reorder above.

### D3: `scene_import_glb` = built-in pure-Python conversion + existing `scene_add_model` write path

The converter is an in-repo module, `src/skinny/glb_import.py` (pygltflib for GLB parsing, pxr for USD authoring — both platform-independent), so the tool works identically on macOS, Linux, and Windows. It authors the *canonical simple* UsdPreviewSurface shape: `file` values set directly on `UsdUVTexture` (no interface indirection), UVs pre-flipped to USD convention at conversion time (no `UsdTransform2d`), embedded images (PNG/JPEG/WebP — Pillow decodes all three) extracted beside the `.usdc`, packed metallicRoughness wired as `metallic ← .b` / `roughness ← .g`. The empirically verified Trellis output (plain PNG images, no glTF extensions, POSITION/NORMAL/TEXCOORD_0) sits well inside this scope; out-of-scope glTF features are refused with an error naming the feature (see Non-Goals). The loader fixes (D1/D2) remain fully justified independently: they make *externally* converted assets (Apple `usdextract`, DCC exports) render correctly, and `usdextract` output is exactly the GPU-gate asset shape.

Tool signature mirrors `scene_add_model`: `glb_path`, optional `name`, `parent`, `translate` / `rotate_euler_deg` / `scale` / `matrix`, plus optional `out_dir` and `overwrite`. Flow:

1. `check_path(glb_path)` against allowed roots (reject early, same guardrail contract).
2. `out_dir` default: `<glb_dir>/<glb_stem>_usd/`; also `check_path`-validated; created with `mkdir -p` before conversion. If the directory already contains a prior conversion (`.usd*` present), the tool refuses unless `overwrite=true` — overwriting an asset that a live reference may still compose is undefined behavior we decline rather than reason about (review finding #8).
3. Convert **on the tool thread, before any renderer post** — the same pattern `scene_add_primitive` uses for `.mtlx` synthesis. Conversion is synchronous and un-pollable: the job machinery covers the *add*, not the conversion, so a very large GLB means one long blocking tool call. Accepted for v1 and documented in the tool contract (review finding #7).
4. Hand the produced `.usdc` to the exact `scene_add_model` implementation (same `validate_added_subtree` roots re-check on the composed result, same job degradation for slow adds). No second copy of the add logic.

*Alternatives rejected:* a `usdextract` subprocess (macOS-only — fails the Windows/Linux requirement; stays available as a manual path on macOS); Blender headless (heavyweight external dependency for a task pygltflib + pxr cover in a few hundred lines); converting inside the renderer-thread write closure (multi-second conversion would stall frames).

### D4: Regression assets — tiny hand-authored USDA for hostless, small GLB-derived asset for the GPU gate

- Hostless loader tests use a minimal committed `.usda` authoring both defeating shapes (interface-connected `file`, `UsdTransform2d` V-flip) against tiny (4×4) PNGs — asserts `TextureBinding` paths, channels, and the baked UV values directly, no GPU.
- The GPU gate commits a small GLB-derived asset (decimated to ≲50 k faces, textures downscaled ≤256²) under `tests/assets/`, converted via `usdextract` at authoring time (output committed, converter not needed at test time). Render + patch assertions in the `TestMaterialXGraphDemoRender` style: non-white color in a known textured region, plus an orientation-discriminating asymmetric texture patch so a V-flip regression fails.

*Alternative rejected:* committing the full 50 MB crown — repo bloat for no extra coverage.

### D5: Converter scope errors instead of platform guards

With the converter in-repo there is no platform gate: `scene_import_glb` is available wherever the MCP server runs (macOS, Linux, Windows). The failure surface is converter *scope*: a GLB using an unsupported glTF feature (Draco, sparse accessors, skinning, animation, vendor extensions) gets a call-time tool error naming the feature, consistent with mcp-scene-control's explicit-error ethos. `pygltflib` is a small pure-Python dependency added to the project requirements; its absence (a broken install) surfaces as an import-time tool error naming the package.

## Risks / Trade-offs

- [Apple USD converter output drifts across macOS versions] → Both loader fixes target standard UsdShade constructs, not converter quirks; the committed GPU-gate asset freezes one known-good conversion, so drift shows up as a new-import failure, not a silent gate rot.
- [UV bake diverges when one mesh's materials disagree on st transform] → Warn + majority transform + recorded limitation; no known producer authors this today.
- [Converter meets a GLB outside its scope (Draco, sparse accessors, extensions)] → Refused with a feature-naming error before any renderer interaction; scope matches what local generators emit, verified against the Trellis output (plain PNG, no extensions).
- [Converter drifts from what the loader consumes] → The hostless suite round-trips converter output through the loader (bindings, channels, UV convention) so a divergence fails tests, not renders.
- [Baked UVs change accumulation hash] → Baking happens before upload; a re-load of the same scene produces identical UVs, so accumulation behavior is unchanged for unaffected scenes (identity transform short-circuits to the exact current code path).
- [Persistent mesh disk cache serves stale UVs] → Structural, not incidental: the build loop resolves the material before finalizing `content_hash`, so the cache key covers the baked UVs (D2); the hostless suite asserts a transform change produces a different key.
- [`binding.source_color_space` is captured but the flat binder chooses sRGB-vs-linear by input name] → Pre-existing behavior, correct for glTF's conventions (diffuse sRGB, packed raw); honoring the authored token is recorded as out of scope, not silently claimed.
- [Packed metallicRoughness texture counts against the bindless pool] → No new pressure: two textures per material, same as any textured flat material today; the 119-slot Metal pool budget is untouched.

## Migration Plan

Loader fixes are additive (previously-dropped bindings now resolve; identity-transform scenes bit-unchanged). No settings, format, or binding-map changes. Rollback = revert; no data migration. The MCP tool is new surface on all platforms; out-of-scope GLBs degrade to a clear feature-naming error, never a crash.

## Open Questions

- Trellis GLBs can carry opacity; `usdextract` emitted no opacity texture for the crown. Whether glTF alpha → UsdPreviewSurface `opacity` intake is worth scoping later stays open (non-goal here).
