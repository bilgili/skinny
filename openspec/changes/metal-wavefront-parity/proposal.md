## Why

The native Metal backend reached full megakernel parity with Vulkan, and `auto`
now resolves to Metal on Apple Silicon. But the entire wavefront execution
axis — staged path/BDPT, ReSTIR DI reuse, and the neural directional proposal —
is pinned to Vulkan, so on the default Apple-Silicon path those features silently
degrade (wavefront falls back to megakernel; ReSTIR/neural fall back to their
analytic subsets). Bringing wavefront and its plugins to Metal closes the last
parity gap and makes every integrator and reuse mode available on the backend
most users actually run.

## What Changes

- A **backend-neutral wavefront driver**: lift the Vulkan-specific staged
  dispatch loop in `vk_wavefront.py` (command-buffer recording +
  `vkCmdDispatchIndirect`) onto the duck-typed compute wrappers so the same
  generate → intersect → build_args → scatter → per-material shade → resolve
  bounce loop runs on `metal_compute` as well as `vk_compute`.
- **Indirect compute dispatch on Metal**: add `dispatch_indirect` to
  `metal_compute` (GPU-written dispatch args via slang-rhi), with a CPU
  count-readback fallback if slang-rhi cannot dispatch indirect, so the
  per-material shade and tiled bounce loop run without a host round-trip.
- **ReSTIR DI on Metal**: build the screen-space reservoir + light-RIS + spatial
  reuse passes on the Metal wavefront path through the existing scene-sampling
  seam, with persistent reservoir buffers across frames.
- **Neural directional proposal on Metal**: dispatch `neural_proposal_pass`
  every bounce on Metal and load frozen offline weights as Metal GPU state.
- **Metal capability enablement**: flip `supports_fp16_storage` /
  `supports_fp16_compute` to reflect the real device, add a weight-handoff path
  (buffer upload, or file fallback), and fall back to fp32 where fp16 is absent.
- **`auto` keeps resolving to Metal** with the full feature set; `--backend
  vulkan` stays byte-identical.
- **Headless A/B parity**: Metal-vs-Vulkan image equivalence for wavefront path,
  wavefront BDPT (every walk mode), ReSTIR DI, and neural proposal.

No user-facing CLI/flag changes and no **BREAKING** changes — behavior only
*improves* on Metal; Vulkan is untouched.

## Capabilities

### New Capabilities

<!-- None — this change modifies the requirements of existing specs only. -->

### Modified Capabilities

- `metal-backend`: lift the deferral of wavefront/ReSTIR/neural on Metal; the
  capability flags `supports_fp16_storage`/`supports_fp16_compute` report the
  real device instead of an unconditional `false`; add indirect-dispatch and
  multi-pass-loop support to the Metal context/compute surface.
- `wavefront-execution`: the "Wavefront is a Vulkan-backend feature" requirement
  (and its "Metal pins to megakernel" scenario) changes — wavefront renders via
  staged dispatches on **both** Metal and Vulkan, sharing the same tiled bounce
  loop and per-bounce memory bound.
- `restir-di`: the reuse-mode requirement changes from "wavefront-only" on
  Vulkan to "builds on the wavefront path on either backend"; the megakernel
  identity fallback is unchanged.
- `neural-directional-proposal`: the "Wavefront-only neural inference pass" and
  "Frozen offline-trained weights as loadable GPU state" requirements change to
  run the inference pass and hold weights on Metal as well as Vulkan.

## Impact

- **Code**: `src/skinny/vk_wavefront.py` (extract a backend-neutral driver, e.g.
  `wavefront_passes.py`), `src/skinny/metal_compute.py` (indirect dispatch,
  multi-pass loop, fp16 + weight upload), `src/skinny/metal_context.py`
  (capability flags), `src/skinny/renderer.py` (construct the wavefront/ReSTIR/
  neural passes on Metal instead of gating to Vulkan), `backend_select.py`
  (capability gating unchanged but re-verified).
- **Shaders**: no logic changes expected; the wavefront/restir/neural `.slang`
  modules must compile cleanly under the Metal in-process Slang→Metal path
  (scalar layout parity with the SPIR-V `-fvk-use-scalar-layout` build).
- **Tests**: new headless A/B parity tests (Metal vs Vulkan) per integrator and
  reuse mode; GPU-free wavefront-layout tests stay backend-agnostic.
- **Docs**: `docs/Wavefront.md`, `docs/ReSTIR.md`, `docs/NeuralGuiding.md`,
  `docs/Architecture.md` (binding map / backend notes), `README.md`/`CLAUDE.md`
  backend tables.
- **Dependencies**: none new — relies on the already-vendored SlangPy/slang-rhi
  and MaterialX-from-source toolchain.
