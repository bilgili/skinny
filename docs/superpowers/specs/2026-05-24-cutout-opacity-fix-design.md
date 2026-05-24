# Cutout-Opacity Fix Design

Restore binary alpha-cutout transparency for UsdPreviewSurface materials whose `opacity` input is connected to a texture's alpha channel and `opacityThreshold > 0`. Concrete failing case: `assets/assets-main/full_assets/McUsd/McUsd.usda`'s sunflower quads (sunflower_back / sunflower_front / sunflower_bottom / sunflower_top), each authored with:

```
float inputs:opacity.connect = .../diffuse_texture.outputs:a
float inputs:opacityThreshold = 0.5
```

## Problem

Sunflower quads render with the alpha-test silhouette only weakly visible: the transparent-edge regions are correctly culled (the path tracer's `isCutoutTransparent` skip loop fires for `alpha < threshold`), but the **opaque** regions appear semi-transparent rather than fully opaque, and the `opacityThreshold` parameter has no visible effect when adjusted.

Root cause is in `src/skinny/shaders/materials/flat/flat_shading.slang::fetchFlatHitData`. After sampling the opacity texture it writes the raw alpha value into `out_.mat.opacity` (e.g. 0.9 for an "opaque" sunflower texel) and only snaps to zero when `alpha < threshold`. The opaque-side value stays sub-unity, which causes `FlatMaterial.sample` (in `flat_material.slang`) to enter its `if (m.opacity < 1.0)` dielectric-refraction branch at `ior = 1.5`. Rays bend and partially transmit through the sunflower, producing the "transparent but not totally" appearance. The threshold parameter genuinely does nothing on opaque pixels because the snap-to-zero branch never fires for them.

This regressed during the recent flat-material overhaul (the same uncommitted diff that introduces `opacityTextureIdx`, `opacityThreshold`, `channelMask`, and per-input channel selectors). UsdPreviewSurface spec semantics for `opacityThreshold > 0` are binary: any alpha below the threshold is fully transparent, any alpha at or above is fully opaque.

## Approach

Switch `fetchFlatHitData` to two explicit opacity modes determined per material:

- **Cutout mode** — `opacityTextureIdx != 0xFFFFFFFFu && opacityThreshold > 0`. The path/bdpt integrators already drop `alpha < threshold` hits via `isCutoutTransparent`. For every hit that reaches `fetchFlatHitData` we therefore know the alpha was `>= threshold`, so the BSDF must see `opacity = 1.0`. Skip the opacity texture fetch in this mode — it cannot influence the result.
- **Alpha-blend mode** — opacity texture present, `opacityThreshold == 0`, OR constant `p.opacity < 1`. Existing behavior is preserved: sample the opacity texture (or fall through to the constant) and let `FlatMaterial.sample` route `(1 - opacity)` of evaluations through the delta-transmission branch.

The `isCutoutTransparent` helper itself does not need to change. Its behavior is already correct: returns true exactly when the textured alpha is below the threshold. The path tracer skip loop and shadow-ray pass-through stay intact.

The current `if (p.opacityThreshold > 0.0 && out_.mat.opacity < p.opacityThreshold) out_.mat.opacity = 0.0` block becomes dead code under the new structure (the cutout path bypasses the texture fetch entirely; the alpha-blend path has `threshold == 0` and skips the condition). Remove it.

## Component changes

**`src/skinny/shaders/materials/flat/flat_shading.slang`** — single function edit:

```slang
out_.mat.opacity = clamp(p.opacity, 0.0, 1.0);
bool cutoutMode = (p.opacityTextureIdx != 0xFFFFFFFFu) && (p.opacityThreshold > 0.0);
if (p.opacityTextureIdx != 0xFFFFFFFFu && !cutoutMode)
{
    float4 s = flatMaterialTextures[NonUniformResourceIndex(p.opacityTextureIdx)]
                   .SampleLevel(h.uv, 0.0);
    out_.mat.opacity = pickChannel(s, channelFor(p.channelMask, 12u));
}
// Removed: dead `if (threshold > 0 && opacity < threshold) opacity = 0` block.
```

No other files need to change. `FlatMaterialParams` layout, the renderer's pack path, the USD loader's threshold-and-channel plumbing, and the integrators' skip loops are all already correct for this fix.

## Data flow

1. USD loader (`_extract_material`) walks the surface shader inputs; the connected `opacity` input becomes a `TextureBinding` with `channel = "a"`, and the constant `opacityThreshold = 0.5` lands in `parameter_overrides`.
2. `pack_flat_material` packs `opacityTextureIdx`, `opacityThreshold = 0.5`, and `channelMask` with the `a` selector at the opacity slot.
3. Path tracer traces a primary ray, lands on a sunflower hit. `isCutoutTransparent` samples the texture's alpha; if below 0.5 the integrator advances the ray past the hit and retries (up to 16 hops per bounce). This already works.
4. For an opaque-side hit (alpha `>= 0.5`), `fetchFlatHitData` now detects cutout mode and forces `m.opacity = 1.0`. `FlatMaterial.sample` skips the refraction branch entirely and runs the standard coat / spec / diffuse lobes against the textured albedo.

## Testing

- Verify visually with the failing asset: load `McUsd.usda`, confirm each sunflower quad shows the expected silhouette with fully opaque petal interiors and no refractive bend.
- Smoke-test the alpha-blend path: render a material that uses `opacity < 1` for refraction (the glass material from commit `ad98a9f`) and confirm refraction still works.
- Run `.venv/bin/ruff check src/` and `.venv/bin/pytest`. No Python-side struct-layout changes, so `tests/test_struct_layout.py` should pass unchanged.

## Out of scope

Bug B (replacing dielectric refraction with straight-through delta transmission for non-cutout alpha) is deferred. The current refraction path is preserved for glass-like materials.
