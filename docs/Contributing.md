# Skinny — Testing and Development

This document covers how to run the test suite and the development
conventions for changing the renderer.

For the documentation rules — one subject per document, the 700-line
ceiling, the index — see `CLAUDE.md` § Documentation upkeep. For the parity
harness that gates renderer changes see [ParityHarness.md](ParityHarness.md).

---

## Testing

The test suite covers shader math, sampling, lighting, volume rendering,
struct layout, MaterialX closures, MaterialX nodegraph compilation, skin
optics, headless rendering, SlangPile transpilation, the shared widget-tree
spec, and the web application. Tests are organized by subsystem with Slang
harness shaders in `tests/harnesses/` and reference kernels in
`tests/kernels/`.

```powershell
.\Scripts\python -m pytest
```

GPU-dependent tests are marked `@pytest.mark.gpu`; statistical Monte Carlo
tests are marked `@pytest.mark.slow`; SlangPile-specific tests are marked
`@pytest.mark.slangpile`.

### Adding a compute pass

A new compute entry point in the shader tree must be **registered or excluded**,
or the build fails (change `recording-adapter-live-bindings`).

- Register it in `recording_compute.RECORDABLE_PASSES` when the host builds its
  bind map through `SceneResourceSet.metal_binds()`. Give the entry module, the
  entry point, its `ShaderVariantKey`, and a bind-map provider — usually
  `lambda ctx: scene_binds(ctx)`. Then regenerate the declared-globals golden so
  the gate knows what the pass declares:

  ```bash
  PYTHONPATH=src .venv/bin/python -m tests.fixtures.gen_recording_pass_globals
  ```

  The gate then asserts the pass binds every global `slangc` reflects for it.
- Otherwise add it to `recording_compute.RECORDABLE_EXCLUSIONS` **with a
  reason**. An exclusion for a pass that no longer exists also fails, because a
  stale one silently re-admits the gap it was meant to bound.

Regenerate the golden too after any shader edit that changes a registered pass's
declared globals — the `gpu`-marked freshness test re-runs the compiler and
diffs, so a stale golden is caught. `tests/test_recording_pass_coverage.py` is
the gate; the design is in
[Backends.md § Live bindings on the recording adapter](Backends.md#live-bindings-on-the-recording-adapter-change-recording-adapter-live-bindings).
Never hand a registered pass a literal set of globals — the compiler and the
host, not the test, supply the two sides.

## Development

Compile Python:

```powershell
.\Scripts\python -m py_compile src\skinny\app.py src\skinny\renderer.py
```

Compile main shader:

```powershell
slangc src\skinny\shaders\main_pass.slang -target spirv -entry mainImage -stage compute -o src\skinny\shaders\main_pass.spv -I src\skinny\shaders
```
