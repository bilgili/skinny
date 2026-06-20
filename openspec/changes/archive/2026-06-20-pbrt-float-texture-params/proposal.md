## Why

Importing a pbrt-v4 scene whose material binds a scalar parameter to a **named
texture** (a `FloatTexture`, e.g. `"texture roughness" ["mitra_right_back-roughness"]`)
crashes the importer. `crown.pbrt` does exactly this, so the scene cannot be
imported at all:

```
File "src/skinny/pbrt/parser.py", line 74, in float
    return float(self.values[0])
ValueError: could not convert string to float: 'mitra_right_back-roughness'
```

pbrt allows any `FloatTexture`-typed material parameter to be either a constant
*or* a texture reference; the importer's scalar resolvers assume a constant and
call `float()` on the texture name. The texture-connection plumbing
(`_TEXTURABLE`, `resolve_texture`, `tex_inputs`) already exists and already
handles `imagemap` and nested `scale` textures — the scalar path just crashes
before it runs.

## What Changes

- The importer SHALL resolve every textureable material parameter through one
  promoting accessor that mirrors pbrt-v4's `GetFloatTexture`/`GetSpectrumTexture`:
  it returns a constant *or* a texture handle, promoting constants uniformly (pbrt
  wraps them in `FloatConstantTexture`). Material mapping no longer calls `float()`
  on a parameter that may be a texture name, so the crash on texture-valued
  FloatTexture params is removed structurally and the constant-vs-texture decision
  lives in one place.
- A textured pbrt parameter SHALL be mapped to its **own** USD parameter via a
  single texture→parameter map (never assumed to be `diffuseColor`). The map
  becomes the single source of truth for `(usd_input, value_type)`, replacing the
  unused `_scalar` flag and the separate hardcoded `_SCALAR_TEX_INPUTS` set;
  connection authoring (`UsdUVTexture` output channel + input type) is driven from
  it. Texture-valued `roughness`/`uroughness`/`vroughness` therefore connect to the
  USD `roughness` input (with the existing `scale`→`imagemap` unwrap), the same
  generic path that maps `reflectance`→`diffuseColor`.
- When a bound texture is unresolvable/unsupported (not `imagemap` or a
  `scale`-wrapped imagemap), the importer SHALL record an APPROX note and use the
  scalar default rather than failing the whole import.
- Parity coverage: `crown.pbrt` (texture-valued roughness, nested `scale`/
  `imagemap`) imports without error.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `pbrt-texture-uv`: add a requirement that texture-valued (`FloatTexture`)
  scalar material parameters resolve to a texture connection or a scalar default,
  and never crash the importer.

## Impact

- Code: `src/skinny/pbrt/materials.py` (promoting `get_float_texture`/
  `get_spectrum_texture` accessor + unified single-pass mapping), `src/skinny/pbrt/
  api.py` (`_author_texture` driven by the map `value_type`, drop
  `_SCALAR_TEX_INPUTS`). No renderer, shader, or `.spv` changes — importer-only.
- Tests: new unit tests for texture-valued roughness/eta resolution; `crown.pbrt`
  smoke-import; optional addition to the pbrt-v4 parity corpus.
- No new dependencies. Vulkan and Metal unaffected (translation layer only).
