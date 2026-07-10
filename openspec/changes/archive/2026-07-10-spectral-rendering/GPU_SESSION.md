# Spectral rendering — GPU session kickoff

The hostless foundation is merged to `main`. This note is the pickup point for
the GPU-only remainder (megakernel spectral transport + verification). Read this,
then `tasks.md` (11/33 done), `design.md`, and the three `specs/` deltas.

## Where things stand (main)

Merged & hostless-tested: pbrt-exact data + numpy mirror (`pbrt/spectral.py`,
`pbrt/data/spectral_tables.py`), importer payload preservation
(`param_spectral_payload`), `--spectral` CLI flag + **not-implemented gate**,
`shaders/spectrum.slang` core (slangc-gated), parity `spectral` axis, WIP docs,
and a hostless dispersion example.

**The single unlock:** `src/skinny/spectral_capability.py` →
`SPECTRAL_IMPLEMENTED = False`. It is referenced **live** by both
`cli_common.reject_spectral_unsupported` (CLI refusal) and
`pbrt.parity.combo_is_valid` (matrix axis). Flip it to `True` in the SAME change
that lands the megakernel transport — that turns the CLI flag and the parity
sweep on together. Do NOT flip it before the transport renders correctly.

## Setup (each GPU session)

```bash
# isolated worktree off main (convention)
git worktree add ../skinny-spectral-gpu -b spectral-megakernel main
cd ../skinny-spectral-gpu
ln -s ../skinny/.venv .venv && ln -s ../skinny/bin bin   # 3.12 venv + 3.13 headless env

# headless/GPU env (needed even on Metal — renderer.py imports vulkan at load)
export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
# worktree src wins over the editable install:
#   PYTHONPATH=src ./bin/python3.13 ...   (3.13 = has MaterialX Slang gen + matplotlib)
```

Thermal rule (CLAUDE.md): **one guarded Metal process at a time**; gpu-marked
tests never run in the default sweep. Always `--backend metal` on this host.

`main_pass.slang` needs the runtime-generated `shaders/generated_materials.slang`
to compile — it exists in the main checkout; copy it into the worktree
(`cp ../skinny/src/skinny/shaders/generated_materials.slang src/skinny/shaders/`)
or generate it by constructing a Renderer once. Recompiling does NOT reproduce
the checked-in `main_pass.spv` byte-for-byte (slangc drift), so frame 4.3 as
"no-define build unchanged by the spectral `#if` blocks", not "== checked-in".

## Task order (dependency-first)

1. **3.2 descriptors** — bind the vendored buffers the shader already expects
   (`spectrum.slang` fns take `scale`/`data`/`d65` StructuredBuffers + res/count).
   Upload `pbrt/data/rgb2spec_srgb.npz` (res, scale, data) and `spectral_curves.npz`
   (d65 + Ag/Al/Au/Cu eta/k) as StorageBuffers. New bindings AFTER 32; make them
   **spectral-build-only** so the RGB descriptor layout is byte-unchanged. Extend
   the Vulkan layout + `tests/pbrt/test_vk_binding_layout.py`; Metal bind-by-name.
   Update the `docs/Architecture.md` binding map (task 9.1).
2. **3.3** compile plumbing — `-DSKINNY_SPECTRAL` variant (decide checked-in
   `main_pass_spectral.spv` vs on-demand; Metal folds the define in-process).
3. **5.1–5.3 transport** — under `#if defined(SKINNY_SPECTRAL)` only:
   convert `BSDFSample`/`LightSample`/`BounceResult` + path `throughput`/`radiance`
   to `Spectrum` (float4); draw hero λ once at path start (extra `rng.next()`);
   upsample material/light/env RGB via `spectrum.slang`; film-resolve to sRGB
   **before** `clampSampleRadiance` + accumulation. Keep RR/MIS scalar (luminance
   of the Spectrum). Exclude BDPT/SPPM/wavefront from the spectral build.
4. **4.2 / 4.3 / 5.4 / 5.5** verify — flip the GPU≡numpy harness
   (`tests/harnesses/test_spectrum_harness.slang` vs `pbrt/spectral.py`); RGB
   byte-identity; flat-scene + furnace smoke on Metal; Vulkan↔Metal A/B.
5. **6.1–6.5** exact sources + dispersion — blackbody Planck, spectral conductor
   Fresnel, authored illuminant SPD lookup, hero-λ dispersion +
   `terminateSecondary` (mirror `examples/dispersion_demo.py` physics); add the
   discriminating suite scene + pbrt refs.
6. **flip `SPECTRAL_IMPLEMENTED = True`** — now `--spectral` runs and the matrix
   enumerates `(path, megakernel, spectral)`.
7. **7.3 / 7.4 / 8** — GPU parity sweep (assert spectral relMSE/FLIP ≤ RGB on
   spectrum-authored scenes), confirm no RGB baseline drift, Metal kill harness,
   equal-time perf.
8. **9.1 / 9.3 / 9.4** — Architecture binding map, spectral doc (LaTeX→SVG per
   repo convention), PythonAPI, full `ruff` + hostless `pytest`.

## Verify commands

```bash
# hostless matrix + metric battery (any host)
PYTHONPATH=src .venv/bin/python -m pytest tests/pbrt/test_matrix.py \
  tests/pbrt/test_metrics.py tests/test_cli_common.py -q
# full GPU parity sweep
PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest \
  tests/pbrt/test_parity.py -k matrix
# Metal kill harness (guarded, one at a time)
PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest \
  tests/test_metal_cleanup.py -m gpu -q
```

Pre-merge: codex review (global rule), fold findings, `--no-ff` merge, push,
remove worktree. Then `openspec archive spectral-rendering` once all tasks pass.
