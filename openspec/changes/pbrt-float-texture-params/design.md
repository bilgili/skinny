## Context

The pbrt importer (`src/skinny/pbrt/`) translates pbrt-v4 scenes to USD with no
renderer-side knowledge of pbrt. `materials.py:map_material` builds
UsdPreviewSurface inputs in two passes:

1. **Scalar pass** — per material type, reads constant params (e.g.
   `_resolve_roughness` → `ParamSet.float("roughness")` → `Param.float` →
   `float(values[0])`).
2. **Texture pass** — iterates `_TEXTURABLE = {reflectance, roughness}` and, for
   any param with `type == "texture"`, calls `resolve_texture` and records a
   `tex_inputs` connection.

`resolve_texture` already handles `imagemap` and recursively unwraps a `scale`
texture to its inner image. `references_texture` already guards `pp.type ==
"texture"` correctly.

The bug: the **scalar pass runs first and is not texture-aware**. When
`roughness` is a `FloatTexture` reference (pbrt allows this for any FloatTexture
param), `Param.float` does `float("mitra_right_back-roughness")` and raises
`ValueError`, aborting the whole import before the texture pass can run.
`crown.pbrt` triggers this (`"texture roughness" ["mitra_right_back-roughness"]`,
itself a `scale`→`imagemap` graph). The color path (`reflectance` →
`spectra.param_to_rgb`) already tolerates texture-typed params by returning the
default, which is why only the scalar path crashes.

## Goals / Non-Goals

**Goals:**
- A texture-typed scalar parameter never reaches `float()`; importer never raises
  on it.
- Texture-valued `roughness`/`uroughness`/`vroughness` route through the existing
  `tex_inputs` connection path (reusing `resolve_texture`'s imagemap + scale
  unwrap).
- Unsupported/unresolvable texture bindings degrade to the scalar default with an
  APPROX note, not a hard failure.
- `crown.pbrt` imports cleanly.

**Non-Goals:**
- Full pbrt texture-graph fidelity (mix/checkerboard/constant FloatTextures,
  `scale` factor preservation, UV transforms). Out of scope — record APPROX.
- Any renderer/shader/`.spv` change. Translation layer only.
- Texturing parameters that skinny's UsdPreviewSurface target cannot connect
  (e.g. `eta` has no USD texture input) — the accessor still resolves them, but a
  `Tex` tag on such a param falls back to `Const(default)` + APPROX (no connection
  authored).

## Decisions

**D1 — Mirror pbrt's `GetFloatTexture`: one promoting accessor, no float-vs-texture
branch in material mapping.** pbrt-v4 resolves *every* textureable parameter
through a single accessor that promotes a constant to a constant-texture, so a
material never branches on "float or texture" — it always holds a texture handle
and evaluates it at the surface UV at shading time:

```cpp
// pbrt-v4 src/pbrt/paramdict.cpp — GetFloatTextureOrNull
if (p->type == "texture")     return floatTextures[p->strings[0]];          // named texture handle
else if (p->type == "float")  return new FloatConstantTexture(value);       // constant PROMOTED
// GetFloatTexture(name, default) = OrNull(name) ?? FloatConstantTexture(default)
// materials.cpp: uRoughness = parameters.GetFloatTexture("roughness", 0.f, alloc);
```

Adopt the same shape in the importer. Add `get_float_texture(params, name,
default, textures, base_dir)` and `get_spectrum_texture(...)` returning a tagged
value — `Const(scalar|rgb)` or `Tex(image_path, color_space)` — that promotes
exactly like pbrt: absent → `Const(default)`; `type=="float"/"spectrum"/"rgb"` →
`Const(value)`; `type=="texture"` → resolve the named texture (imagemap,
`scale`-unwrapped) → `Tex(...)`.

Material mapping calls this uniformly per parameter; there is no naive
`float(values[0])` path left to crash on a texture name — the crash is removed
*structurally*, not patched per call site. USD authoring branches once on the
tag: `Const` → constant `UsdPreviewSurface` input; `Tex` → `UsdUVTexture`
connection (output channel + input type from the `_TEXTURABLE` `value_type`, D5).
The mesh's `primvars:st` is skinny's analog of pbrt's `TexCoord2f uv` evaluation
context. This collapses the current two-pass (scalar-then-texture) flow into one
uniform pass, matching pbrt.

Rejected alternative: a `ParamSet.float_const` guard that returns the default for
texture-typed params. It patches each scalar reader instead of unifying — the next
FloatTexture param re-introduces the risk, and it keeps the float-vs-texture split
pbrt deliberately avoids.

Deliberate deviation from pbrt: pbrt `ErrorExit`s on an unknown texture name or
unsupported class; skinny instead returns `Const(default)` + an APPROX note so a
partly-unsupported scene still imports (best-effort translator, not a renderer).

**D2 — Don't special-case roughness; the accessor + map handle all params.**
`roughness` is already in `_TEXTURABLE`, so once mapping goes through the promoting
accessor the `Tex` tag drives a `tex_inputs["roughness"]` connection with no
roughness-specific code. `uroughness`/`vroughness` resolve via the same accessor
and collapse to the single USD `roughness` connection (anisotropic → isotropic
APPROX note).

**D3 — Scalar fallback value when textured.** When `roughness` is texture-bound,
the scalar default feeds the constant `roughness` input that the connection then
overrides. Use the same default the material type already sets (conductor/coated
paths), so a target that ignores the connection still renders plausibly. Record
the APPROX note already produced by the texture pass on unresolved textures.

**D4 — Spec home.** Extend `pbrt-texture-uv` (the living pbrt-texture capability)
rather than a new capability — this is texture-binding resolution, adjacent to the
existing imagemap/UV requirements.

**D5 — General pbrt-texture → USD-parameter mapping; never assume diffuse.**
Each textured pbrt parameter maps to its *own* USD input via a single mapping
table. The importer must not assume a texture targets `diffuseColor`; roughness
maps roughness→roughness because it has its own map entry, not because it rides a
diffuse path.

Make the map the single source of truth and drive authoring from it:
- `_TEXTURABLE` carries `pbrt_param → (usd_input, value_type)` where `value_type`
  is the connection kind (`color3f` vs `float`/scalar). This replaces today's two
  drifting sources of truth — the **unused** `_scalar` flag in `_TEXTURABLE` and
  the separate hardcoded `_SCALAR_TEX_INPUTS = {"roughness","metallic","opacity"}`
  set keyed on USD name. Remove `_SCALAR_TEX_INPUTS`.
- The texture pass iterates the map: for each pbrt param with `type == "texture"`,
  `tex_inputs[usd_input] = resolve_texture(...)`.
- `_author_texture` selects the `UsdUVTexture` output channel (`.rgb` for color,
  `.r` for scalar) and the `UsdPreviewSurface` input type from the map's
  `value_type` — not from a hardcoded name set. Generic over `usd_input`.
- Extensible by data: adding `uroughness`/`vroughness`/`displacement`/conductor
  `reflectance`/etc. is one new map row, no new authoring code.

The UV set is orthogonal and shared: a shape has one `primvars:st` (synthesized
when `references_texture` — which already iterates `_TEXTURABLE`, so roughness
counts), and every texture node samples it, matching pbrt's
one-parametrization-per-shape model. Which *parameter* each node drives is set
explicitly by the map, not assumed. No explicit `UsdPrimvarReader`/`st` is
authored — same convention as the GPU-parity-proven `texture_quad` diffuse scene.

Consequence: this change adds no UV-coordinate code; it generalizes the
texture→parameter mapping so the connection target is data-driven. The only
downstream unknown is GPU *consumption* (below), not the mapping.

## Risks / Trade-offs

- **Silent under-fidelity** (scale factor dropped, mix/checkerboard ignored) →
  Mitigation: every lossy resolution appends an APPROX note surfaced in the import
  report; spec scenario asserts the note.
- **`Param.float` left strict could still crash a future scalar reader that
  forgets the guard** → Mitigation: route all texturable scalar reads through
  `float_const`; add a regression test per texturable param.
- **Connection authored but skinny's USD loader / GPU path samples only diffuse
  textures, not the roughness connection** → the diffuse texture path is
  parity-proven (`texture_quad`); roughness-texture *consumption* is unverified.
  If unsupported, import + USD are correct but the render falls back to constant
  roughness. Mitigation: a verification task confirms the loader reads the
  roughness `UsdUVTexture` (headless: textured-roughness vs constant must differ);
  if it does not, that GPU-loader work is a scoped follow-up, not a blocker for the
  crash fix.
- **Imagemap UV transform / `scale` factor dropped** — `resolve_texture` returns
  only `(path, colorspace)`, discarding `uscale`/`vscale`/`udelta`/`vdelta` and the
  `scale` multiplier → Mitigation: APPROX note; faithful UV-transform support is a
  separate concern.

## Migration Plan

Pure importer change, no persisted data or runtime format. Land behind tests; no
rollback concerns beyond reverting the commit. `.spv` artifacts unchanged.

## Open Questions

- Should `crown.pbrt` join the pbrt-v4 parity corpus (with a committed reference
  EXR), or is a crash-free smoke-import sufficient for this change? Default:
  smoke-import here; corpus parity as a follow-up if roughness-texture GPU
  sampling needs validation.
