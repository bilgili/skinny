## Why

The pbrt importer names light prims with `id(light) & 0xFFFF` — the Python
object's memory address — so importing the same `.pbrt` twice emits different
prim names and, for synthesized constant environment lights, a different baked
`.hdr` filename. Because the repo `.gitignore` is deny-all (`*`), a freshly baked
`.hdr` is untracked unless someone remembers `git add -f`; a regenerated `.usda`
then points at an `.hdr` a fresh clone lacks, and the loader silently falls back
to a gray environment. That exact failure (a dropped `light_infinite_f620_const.hdr`
turning a blue sky gray) already cost real debugging time and blew a parity
baseline from 0.075 to 0.584 (`parity-scene-asset-integrity`). Address noise also
makes asset regeneration non-reproducible — every regen churns light names and
orphans previously-tracked `.hdr` files, drowning real diffs in noise.

## What Changes

- Replace the `id(light) & 0xFFFF` suffix in `src/skinny/pbrt/lights.py` with the
  light's index in `scene.lights`, so the prim name is `light_<ltype>_<i>`.
- Thread the enumerated index from the `add_light` call site in
  `src/skinny/pbrt/api.py` into `add_light`.
- Synthesized `.hdr` filenames (`<name>_const.hdr`, `<name>_env.hdr`) become
  deterministic as a consequence (they derive from the prim name).
- Regenerate the existing suite/corpus assets that carry synthesized-env `.hdr`
  files so their names stabilize once, and re-track the renamed `.hdr` with
  `git add -f`. Pixel content is unchanged (identical `.hdr` bytes) — parity
  baselines must NOT move; verify rather than assume.

## Capabilities

### New Capabilities
- `pbrt-light-naming`: deterministic, content-derived prim naming for imported
  pbrt lights — same scene in yields byte-identical light prim names (and
  synthesized `.hdr` filenames) out; distinct lights keep distinct names.

### Modified Capabilities

## Impact

- `src/skinny/pbrt/lights.py` (`add_light` signature + name derivation),
  `src/skinny/pbrt/api.py` (call site passes the index).
- Existing `tests/assets/suite/*` (and any corpus) assets with synthesized-env
  `.hdr` files: one-time regeneration + re-track. Names in the checked-in `.usda`
  change once; pixel content does not.
