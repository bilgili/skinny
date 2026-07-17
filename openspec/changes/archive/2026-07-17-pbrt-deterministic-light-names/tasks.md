## 1. Code change

- [x] 1.1 Add an `index: int` parameter to `add_light` in `src/skinny/pbrt/lights.py`; change the name to `sanitize(f"light_{ltype}_{index}")`.
- [x] 1.2 Pass the enumerated index from the call site in `src/skinny/pbrt/api.py` (`for i, light in enumerate(scene.lights): add_light(..., index=i)`).

## 2. Regression test

- [x] 2.1 Add a hostless test: import the same `.pbrt` (e.g. `tests/assets/suite/spec_prism/spec_prism.pbrt`) twice and assert identical light prim names + identical synthesized `.hdr` filenames.
- [x] 2.2 Assert a two-light scene yields two distinct light prim names.

## 3. Asset regeneration

- [x] 3.1 Regenerate affected suite assets via `tests/assets/suite/_gen/build.py` (the 5 env-bearing scenes — mat_dielectric, mat_conductor, spec_prism, mat_textured, samp_env_glossy — all come from `build.py`, which emits plain + `_mtlx`; `build_pbr.py` prunes infinite-light hdrs and produces none). The `.usda` light prim + `textureFile` names update once.
- [x] 3.2 Per env folder the two old hex `.hdr` (plain + `_mtlx` import) collapse to one shared `light_infinite_0_const.hdr` (identical bytes). Delete BOTH old hex files (10 total across 5 folders); `git add -f` the one new index-named `.hdr` per folder.

## 4. Verify

- [x] 4.1 `ruff check src/` clean; run the new + existing pbrt light tests.
- [x] 4.2 Run the parity gate on an affected scene; confirm baselines unchanged (pixel content identical, only names changed).
- [x] 4.3 `openspec validate pbrt-deterministic-light-names`.
- [x] 4.4 codex pre-merge review; fold findings back in.
