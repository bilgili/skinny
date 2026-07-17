## Context

`add_light` (`src/skinny/pbrt/lights.py:73`) names the prim:

```python
name = sanitize(f"light_{ltype}_{id(light) & 0xFFFF:x}")
```

`id(light)` is the object's memory address — non-deterministic across processes.
The `.hdr` filenames (`{name}_const.hdr` at `:150`, `{name}_env.hdr` at `:215`)
inherit that non-determinism. The single call site is
`src/skinny/pbrt/api.py:103`:

```python
for light in scene.lights:
    add_light(stage, world, light, report, asset_dir=..., ...)
```

Shapes already use a positional convention: `f"{world}/shape_{i}"`
(`api.py:91`, enumerate index). Lights should follow the same pattern.

## Goals / Non-Goals

Goals:
- Same `.pbrt` in → byte-identical light prim names + synthesized `.hdr` names.
- Distinct lights keep distinct names.
- Match the existing `shape_{i}` positional convention (one obvious pattern).

Non-Goals:
- No change to light radiometry, transforms, or spectral payloads.
- No change to `.hdr` pixel content — only filenames.
- No name-stability guarantee across *content* edits (reordering/adding a light
  legitimately churns names; that reflects a real change, not address noise).

## Decision

Use the light's index in `scene.lights` as the suffix.

- `api.py`: `for i, light in enumerate(scene.lights): add_light(..., index=i)`.
- `lights.py`: add an `index: int` parameter; name becomes
  `sanitize(f"light_{ltype}_{index}")`.

Index over content-hash: it is the laziest option that satisfies both
requirements, mirrors `shape_{i}`, and yields human-readable names. A content
hash would also be deterministic but adds code, is opaque in the `.usda`, and
buys nothing here — two identical lights in one scene still need distinct names,
which the index gives for free and a pure content hash does not.

`sanitize` is retained (integer index is already safe, but keeping the call
costs nothing and preserves the one code path).

## Risks / Trade-offs

- **One-time asset churn.** Existing suite/corpus `.usda` + synthesized `.hdr`
  names change once. Mitigation: regenerate the affected assets, `git add -f`
  the renamed `.hdr`, and confirm the old orphaned `.hdr` files are removed.
- **Parity baselines.** Pixel content is unchanged (identical `.hdr` bytes), so
  baselines must not move. Mitigation: run the parity gate on an affected scene
  and confirm no baseline shift — verify, do not assume.

## Migration Plan

1. Land the code change.
2. Regenerate affected suite assets (`tests/assets/suite/_gen/build.py` /
   `build_pbr.py` re-import; `regen_refs.py` unaffected — no pixel change).
3. `git add -f` the renamed `.hdr`; delete the orphaned old-named `.hdr`.
4. Run the parity gate on an affected scene; confirm baselines unchanged.

## Open Questions

None.
