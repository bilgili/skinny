## Why

`skinny --integrator sppm assets/bathroom.usda --execution-mode wavefront`
aborts at shader compile with:

```
shaders/generated/shape_15_mat_graph.slang:169:36
    float2 geomprop_UV0_out1 = vd.texcoord_0.xy;
                               ^^ undefined identifier 'vd'.
```

The per-material graph emitter (`materialx_runtime.MaterialLibrary.
_emit_graph_fragment`) rewrites the MaterialXGenSlang vertex-data inputs
(`vd.*`) into the fragment function's parameters (`P_in` / `N_in` / `T_in` /
`UV_in`). It handles the `<geompropvalue geomprop="UVMap">` UV form
(`vd.i_geomprop_UVMap` → `UV_in`) but **not** the default `<texcoord>` UV form
that MaterialXGenSlang emits as `vd.texcoord_0` when an `<image>` node has no
explicit texture-coordinate input. bathroom.mtlx drives every `base_color`
image off the default UV set, so the generated graph keeps a raw `vd.texcoord_0`
that resolves to nothing in the fragment scope — the module fails to compile and
takes down `scene_trace` → `wavefront_sppm` (and any other pipeline that imports
the material).

This was latent until now: the existing graph scenes (marble / brass / wood)
use solid noise or `<geompropvalue UVMap>` tiled images, so none exercised the
default-`<texcoord>` path. It is not SPPM-specific — any execution mode or
integrator that compiles this scene's material modules hits the same undefined
identifier; SPPM-wavefront is simply the path the user ran first.

## What Changes

- **Rewrite the default-UV vertex input.** `_emit_graph_fragment` adds
  `vd.texcoord_0` → `UV_in` to its `vd.*` substitution set, alongside the
  existing `vd.i_geomprop_UVMap` → `UV_in`. The trailing `.xy` swizzle stays
  valid on the `float2 UV_in` parameter, so a bare `<image>` on the default UV
  set shades from the mesh UVs like the explicit-UVMap form already does.
- **Fall back instead of emitting uncompilable Slang.** After all known `vd.*`
  rewrites, if any `vd.` token still remains in the body the emitter returns
  `None` (and logs the leftover), routing the material through the flat /
  std_surface SSBO path rather than producing a module that fails to compile.
  This converts every future unhandled vertex input (secondary UV sets, vertex
  colors, …) from a hard scene-load crash into a graceful, rendered fallback.

## Impact

- Affected spec: `per-material-pipeline` (adds a "graph vertex inputs are
  rewritten or the graph falls back" requirement).
- Affected code: `src/skinny/materialx_runtime.py` (`_emit_graph_fragment`).
- No shader-source, descriptor-binding, or param-layout change; megakernel and
  wavefront forms are emitted from the same fixed fragment, so both backends and
  all integrators benefit identically.
